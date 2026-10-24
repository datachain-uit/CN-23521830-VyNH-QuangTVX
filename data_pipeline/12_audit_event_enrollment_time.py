"""Audit mapped events rejected by enrollment/end time; never edit production data."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import duckdb

AS_OF = '2026-10-19'
SOURCES = {
    'video': ('final_quality_tables/_stage/video_events.parquet', 'video_id'),
    'problem': ('eda_engagement_features_comparison/_stage/problem_rows/**/*.parquet', 'problem_id'),
    'comment': ('final_quality_tables/_stage/comment_prepared.parquet', 'comment_id'),
    'reply': ('final_quality_tables/_stage/reply_prepared.parquet', 'reply_id'),
}


def qs(x):
    return "'" + str(x).replace("'", "''") + "'"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str) + '\n')


def records(con, sql):
    cur = con.execute(sql)
    keys = [x[0] for x in cur.description]
    return [dict(zip(keys, row)) for row in cur.fetchall()]


def mdtable(headers, rows):
    def cell(x):
        if x is None:
            return 'NULL'
        if isinstance(x, float):
            return f'{x:.4f}'
        if isinstance(x, int):
            return f'{x:,}'.replace(',', '.')
        return str(x).replace('|', '\\|').replace('\n', ' ')
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                      '| ' + ' | '.join('---' for _ in headers) + ' |']
                     + ['| ' + ' | '.join(cell(x) for x in row) + ' |' for row in rows])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root')
    args = parser.parse_args()
    base = Path(args.root or (Path.cwd()/'working_13-10'
                             if (Path.cwd()/'working_13-10').is_dir() else Path.cwd())).resolve()
    out = base/'event_enrollment_time_audit'
    for sub in ['tables/pair_source', 'tables/filtered_events', '_scratch']:
        (out/sub).mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    con = duckdb.connect()
    con.execute("SET memory_limit='2GB'")
    con.execute('SET threads=2')
    con.execute("SET TimeZone='UTC'")
    con.execute(f"SET temp_directory={qs(out/'_scratch')}")
    con.execute("SET max_temp_directory_size='2GB'")
    source_paths = [
        base/'05_build_review_time_phases.py', base/'06_build_final_quality_tables.py',
        base/'review_time_phases/run_manifest.json', base/'final_quality_tables/run_manifest.json',
        base/'review_time_phases/tables/user_course_phase_times.parquet',
        base/'final_quality_tables/_stage/active_candidates.parquet',
        base/'weighted_loqv_alpha_0p1/tables/user_course_loqv_weighted.parquet',
    ]
    for pattern, _ in SOURCES.values():
        source_paths.extend(sorted(base.glob(pattern)))
    for pattern in ['input/moocubex_parquet/tables/entities/reply/*.parquet',
                    'input/moocubex_parquet/tables/relations/comment_reply/*.parquet']:
        source_paths.extend(sorted(base.glob(pattern)))
    hashes = {str(p.relative_to(base)):sha(p) for p in source_paths}

    def view(name, pattern):
        con.execute(f'CREATE OR REPLACE VIEW {name} AS SELECT * FROM read_parquet({qs(base/pattern)},union_by_name=true)')

    def temp(name, sql):
        con.execute(f'CREATE OR REPLACE TEMP TABLE {name} AS {sql}')

    def save(relative, sql):
        path = out/'tables'/relative
        partial = path.with_suffix('.partial')
        con.execute(f'COPY ({sql}) TO {qs(partial)} (FORMAT PARQUET,COMPRESSION ZSTD)')
        partial.replace(path)

    view('time_pairs', 'review_time_phases/tables/user_course_phase_times.parquet')
    view('candidates', 'final_quality_tables/_stage/active_candidates.parquet')
    temp('relevant_pairs', '''SELECT t.user_id,t.course_id,t.enroll_time,t.course_end,t.time_status,t.time_eligible
        FROM time_pairs t JOIN candidates c USING(user_id,course_id)''')
    eligibility = records(con,'SELECT time_status,count(*) n_pairs FROM time_pairs GROUP BY 1 ORDER BY n_pairs DESC')
    source_cases, delta_stats, source_totals = [], [], []
    for source, (pattern, reference) in SOURCES.items():
        view('source_events', pattern)
        con.execute(f'''CREATE OR REPLACE TEMP VIEW joined_events AS
          SELECT {qs(source)} event_source,e.user_id,e.course_id,e.lo_key,
            cast(e.{reference} AS VARCHAR) event_reference,e.event_time,p.enroll_time,p.course_end,
            coalesce(p.time_status,'NO_ENROLLMENT_RECORD') pair_time_status,
            coalesce(p.time_eligible,false) time_eligible,
            epoch(e.event_time)-epoch(p.enroll_time) delta_seconds,
            CASE WHEN p.user_id IS NULL THEN 'NO_ENROLLMENT_RECORD'
                 WHEN NOT p.time_eligible THEN 'PAIR_TIME_INELIGIBLE'
                 WHEN e.event_time IS NULL THEN 'INVALID_EVENT_TIME'
                 WHEN e.event_time<p.enroll_time THEN 'BEFORE_ENROLLMENT'
                 WHEN e.event_time>p.course_end THEN 'AFTER_COURSE_END'
                 ELSE 'WITHIN_FINAL_WINDOW' END event_time_case
          FROM source_events e LEFT JOIN relevant_pairs p USING(user_id,course_id)''')
        source_totals.append(dict(source=source, n_events=con.execute('SELECT count(*) FROM source_events').fetchone()[0]))
        source_cases.extend(records(con,'''SELECT event_source,pair_time_status,event_time_case,
            count(*) n_events,count(DISTINCT(user_id,course_id)) n_pairs
            FROM joined_events GROUP BY 1,2,3 ORDER BY 1,2,3'''))
        delta_stats.extend(records(con,'''SELECT event_source,event_time_case,count(*) n_events,
            min(delta_seconds) min_seconds,quantile_cont(delta_seconds,0.01) p01_seconds,
            quantile_cont(delta_seconds,0.05) p05_seconds,quantile_cont(delta_seconds,0.25) p25_seconds,
            median(delta_seconds) median_seconds,avg(delta_seconds) mean_seconds,
            quantile_cont(delta_seconds,0.75) p75_seconds,quantile_cont(delta_seconds,0.95) p95_seconds,
            quantile_cont(delta_seconds,0.99) p99_seconds,max(delta_seconds) max_seconds
            FROM joined_events WHERE time_eligible AND event_time IS NOT NULL
            GROUP BY 1,2 ORDER BY 1,2'''))
        save(f'pair_source/{source}.parquet', '''SELECT event_source,user_id,course_id,enroll_time,course_end,
            pair_time_status,time_eligible,count(*) n_events,
            count(*) FILTER(WHERE event_time<enroll_time) n_before,
            count(*) FILTER(WHERE event_time BETWEEN enroll_time AND course_end) n_within,
            count(*) FILTER(WHERE event_time>course_end) n_after,
            count(*) FILTER(WHERE event_time IS NULL) n_invalid,
            min(event_time) first_event,max(event_time) last_event,
            min(delta_seconds) first_delta_seconds,max(delta_seconds) last_delta_seconds,
            min(delta_seconds) FILTER(WHERE event_time BETWEEN enroll_time AND course_end) first_valid_delta_seconds,
            max(delta_seconds) FILTER(WHERE event_time BETWEEN enroll_time AND course_end) last_valid_delta_seconds,
            max(delta_seconds) FILTER(WHERE delta_seconds<0) nearest_negative_delta_seconds
          FROM joined_events GROUP BY 1,2,3,4,5,6,7''')
        save(f'filtered_events/{source}.parquet', '''SELECT * FROM joined_events
            WHERE time_eligible AND event_time_case<>'WITHIN_FINAL_WINDOW' ''')
        print('Audited',source,source_totals[-1]['n_events'],'events',flush=True)

    con.execute(f"CREATE VIEW pair_source AS SELECT * FROM read_parquet({qs(out/'tables/pair_source/*.parquet')})")
    con.execute(f"CREATE VIEW filtered_events AS SELECT * FROM read_parquet({qs(out/'tables/filtered_events/*.parquet')})")
    first_event_profiles = records(con,'''SELECT event_source,count(*) n_pairs_with_event,
        count(first_valid_delta_seconds) n_pairs_with_valid_event,
        median(first_delta_seconds)/3600 median_first_event_delta_hours,
        min(first_valid_delta_seconds)/3600 min_first_valid_delta_hours,
        quantile_cont(first_valid_delta_seconds,0.05)/3600 p05_first_valid_delta_hours,
        median(first_valid_delta_seconds)/3600 median_first_valid_delta_hours,
        quantile_cont(first_valid_delta_seconds,0.95)/3600 p95_first_valid_delta_hours,
        max(first_valid_delta_seconds)/3600 max_first_valid_delta_hours
        FROM pair_source WHERE time_eligible GROUP BY 1 ORDER BY 1''')
    temp('pair_audit', '''WITH counts AS (
        SELECT user_id,course_id,min(enroll_time) enroll_time,min(course_end) course_end,
          count(*) n_sources,sum(n_events) n_events,sum(n_before) n_before,
          sum(n_within) n_within,sum(n_after) n_after,sum(n_invalid) n_invalid,
          min(first_event) first_event,max(last_event) last_event,
          min(first_delta_seconds) first_delta_seconds,max(last_delta_seconds) last_delta_seconds,
          max(nearest_negative_delta_seconds) nearest_negative_delta_seconds
        FROM pair_source WHERE time_eligible GROUP BY 1,2)
      SELECT *,CASE WHEN n_before=n_events THEN 'ALL_BEFORE_ENROLLMENT'
        WHEN n_within=n_events THEN 'ALL_WITHIN_WINDOW'
        WHEN n_before>0 AND n_within>0 THEN 'BEFORE_AND_WITHIN'
        WHEN n_after=n_events THEN 'ALL_AFTER_COURSE_END'
        ELSE 'OTHER_MIXED_OR_INVALID' END pair_event_case FROM counts''')
    save('pair_time_audit.parquet','SELECT * FROM pair_audit ORDER BY user_id,course_id')
    pair_cases = records(con,'''SELECT pair_event_case,count(*) n_pairs,sum(n_events) n_events,
        sum(n_before) n_before,sum(n_within) n_within,sum(n_after) n_after,sum(n_invalid) n_invalid
        FROM pair_audit GROUP BY 1 ORDER BY n_pairs DESC''')
    all_before_stats = records(con,'''SELECT count(*) n_pairs,sum(n_events) n_events,
        min(first_delta_seconds)/3600 min_first_delta_hours,
        median(first_delta_seconds)/3600 median_first_delta_hours,
        quantile_cont(first_delta_seconds,0.95)/3600 p95_first_delta_hours,
        median(last_delta_seconds)/3600 median_last_delta_hours,
        min(last_delta_seconds)/3600 min_last_delta_hours,
        max(last_delta_seconds)/3600 max_last_delta_hours,
        count(*) FILTER(WHERE abs(last_delta_seconds+28800)<=60) pairs_nearest_8h_within_1m,
        count(*) FILTER(WHERE abs(last_delta_seconds+28800)<=300) pairs_nearest_8h_within_5m,
        count(*) FILTER(WHERE abs(last_delta_seconds+28800)<=900) pairs_nearest_8h_within_15m
        FROM pair_audit WHERE pair_event_case='ALL_BEFORE_ENROLLMENT' ''')[0]

    temp('bins', '''SELECT * FROM (VALUES
      (1,'<1 phút',0.0,60.0),(2,'1–5 phút',60.0,300.0),(3,'5–30 phút',300.0,1800.0),
      (4,'30–60 phút',1800.0,3600.0),(5,'1–6 giờ',3600.0,21600.0),
      (6,'6–7 giờ',21600.0,25200.0),(7,'7–8 giờ',25200.0,28800.0),
      (8,'8–9 giờ',28800.0,32400.0),(9,'9–12 giờ',32400.0,43200.0),
      (10,'12–24 giờ',43200.0,86400.0),(11,'1–7 ngày',86400.0,604800.0),
      (12,'7–30 ngày',604800.0,2592000.0),(13,'>=30 ngày',2592000.0,NULL::DOUBLE)
      ) t(bin_order,bin_name,lower_seconds,upper_seconds)''')
    negative_bins = records(con,'''SELECT b.bin_order,b.bin_name,count(e.event_time) n_events,
        count(DISTINCT(e.user_id,e.course_id)) FILTER(WHERE e.event_time IS NOT NULL) n_pairs
      FROM bins b LEFT JOIN filtered_events e ON e.event_time_case='BEFORE_ENROLLMENT'
        AND -e.delta_seconds>=b.lower_seconds AND (b.upper_seconds IS NULL OR -e.delta_seconds<b.upper_seconds)
      GROUP BY 1,2 ORDER BY 1''')
    pair_bins = records(con,'''SELECT b.bin_order,b.bin_name,count(p.user_id) n_pairs
      FROM bins b LEFT JOIN pair_audit p ON p.pair_event_case='ALL_BEFORE_ENROLLMENT'
        AND -p.last_delta_seconds>=b.lower_seconds
        AND (b.upper_seconds IS NULL OR -p.last_delta_seconds<b.upper_seconds)
      GROUP BY 1,2 ORDER BY 1''')
    near_8h = records(con,'''SELECT count(*) n_before_events,count(DISTINCT(user_id,course_id)) n_before_pairs,
        count(*) FILTER(WHERE abs(delta_seconds+28800)<=60) events_within_1m_of_minus8h,
        count(*) FILTER(WHERE abs(delta_seconds+28800)<=300) events_within_5m_of_minus8h,
        count(*) FILTER(WHERE abs(delta_seconds+28800)<=900) events_within_15m_of_minus8h
      FROM filtered_events WHERE event_time_case='BEFORE_ENROLLMENT' ''')[0]
    view('video_events',SOURCES['video'][0])
    temp('eligible_video', '''SELECT v.*,p.enroll_time,p.course_end,
        epoch(v.event_time)-epoch(p.enroll_time) delta_seconds
      FROM video_events v JOIN relevant_pairs p USING(user_id,course_id) WHERE p.time_eligible''')
    sensitivity = records(con,'''WITH shifts AS (SELECT * FROM (VALUES (-8),(0),(7),(8),(9)) t(shift_hours))
      SELECT shift_hours,count(*) n_events,
        count(*) FILTER(WHERE event_time+shift_hours*INTERVAL '1 hour'<enroll_time) n_before,
        count(*) FILTER(WHERE event_time+shift_hours*INTERVAL '1 hour' BETWEEN enroll_time AND course_end) n_within,
        count(*) FILTER(WHERE event_time+shift_hours*INTERVAL '1 hour'>course_end) n_after
      FROM eligible_video CROSS JOIN shifts GROUP BY shift_hours ORDER BY shift_hours''')
    temp('all_before_shift', '''SELECT p.*,
        count(v.event_time) FILTER(WHERE v.event_time+INTERVAL '8 hours' BETWEEN p.enroll_time AND p.course_end) n_within_if_video_plus8h,
        count(v.event_time) FILTER(WHERE v.event_time+INTERVAL '8 hours'<p.enroll_time) n_still_before_if_video_plus8h,
        count(v.event_time) FILTER(WHERE v.event_time+INTERVAL '8 hours'>p.course_end) n_after_if_video_plus8h
      FROM pair_audit p JOIN eligible_video v USING(user_id,course_id)
      WHERE p.pair_event_case='ALL_BEFORE_ENROLLMENT' GROUP BY ALL''')
    recovery = records(con,'''SELECT count(*) n_pairs,
        count(*) FILTER(WHERE n_within_if_video_plus8h>0) pairs_with_any_in_window_if_plus8h,
        count(*) FILTER(WHERE n_within_if_video_plus8h=n_events) pairs_all_events_in_window_if_plus8h,
        count(*) FILTER(WHERE n_within_if_video_plus8h=0) pairs_still_without_in_window_event,
        sum(n_within_if_video_plus8h) events_in_window_if_plus8h,
        sum(n_still_before_if_video_plus8h) events_still_before_if_plus8h,
        sum(n_after_if_video_plus8h) events_after_end_if_plus8h FROM all_before_shift''')[0]
    view('reply_raw','input/moocubex_parquet/tables/entities/reply/*.parquet')
    view('reply_parent_raw','input/moocubex_parquet/tables/relations/comment_reply/*.parquet')
    view('comment_parent',SOURCES['comment'][0])
    reply_parent_audit = records(con,'''WITH parent AS (
        SELECT reply_id,min(comment_id) parent_comment_id FROM reply_parent_raw
        GROUP BY 1 HAVING count(DISTINCT comment_id)=1),
      typed AS (
        SELECT r.id reply_id,try_cast(r.create_time AS TIMESTAMP) event_time,c.event_time parent_time,
          row_number() OVER(PARTITION BY r.id ORDER BY r.create_time,r.user_id,r.text) id_rank
        FROM reply_raw r JOIN parent p ON r.id=p.reply_id JOIN comment_parent c ON p.parent_comment_id=c.comment_id)
      SELECT count(*) n_joinable_rows,
        count(*) FILTER(WHERE id_rank=1) n_unique_candidates,
        count(*) FILTER(WHERE id_rank=1 AND event_time IS NULL) n_invalid_event_time,
        count(*) FILTER(WHERE id_rank=1 AND event_time<parent_time) n_reply_before_parent,
        count(*) FILTER(WHERE id_rank=1 AND event_time>=parent_time) n_retained_in_reply_prepared FROM typed''')[0]
    save('all_before_enrollment_pairs.parquet','SELECT * FROM all_before_shift ORDER BY user_id,course_id')
    view('master','weighted_loqv_alpha_0p1/tables/user_course_loqv_weighted.parquet')
    status_crosscheck = records(con,'''SELECT label_status_weighted,count(*) n_pairs
      FROM pair_audit p JOIN master m USING(user_id,course_id)
      WHERE p.pair_event_case='ALL_BEFORE_ENROLLMENT' GROUP BY 1''')
    examples = records(con,'''SELECT p.user_id,p.course_id,p.enroll_time,p.course_end,p.first_event,p.last_event,
        p.n_events,p.first_delta_seconds/3600 first_delta_hours,p.last_delta_seconds/3600 last_delta_hours,
        p.n_within_if_video_plus8h
      FROM all_before_shift p ORDER BY abs(p.last_delta_seconds+28800),p.user_id,p.course_id LIMIT 5''')
    tail_examples = records(con,'''SELECT user_id,course_id,enroll_time,first_event,last_event,n_events,
        first_delta_seconds/86400 first_delta_days,last_delta_seconds/86400 last_delta_days
      FROM all_before_shift ORDER BY first_delta_seconds,user_id,course_id LIMIT 5''')
    checks = {
        'time_pairs_unique':con.execute('SELECT count(*)=count(DISTINCT(user_id,course_id)) FROM relevant_pairs').fetchone()[0],
        'source_case_counts_conserve_rows':all(sum(r['n_events'] for r in source_cases if r['event_source']==s['source'])==s['n_events'] for s in source_totals),
        'eligible_case_counts_match_pair_audit':sum(r['n_events'] for r in source_cases if r['pair_time_status']=='ELIGIBLE_TIME')==sum(r['n_events'] for r in pair_cases),
        'event_bucket_partition':con.execute('SELECT sum(n_before+n_within+n_after+n_invalid)=sum(n_events) FROM pair_audit').fetchone()[0],
        'before_bins_conserve_events':sum(r['n_events'] for r in negative_bins)==near_8h['n_before_events'],
        'all_before_pair_bins_conserve_pairs':sum(r['n_pairs'] for r in pair_bins)==all_before_stats['n_pairs'],
        'prior_5130_pairs_reproduced':all_before_stats['n_pairs']==5130,
        'prior_29414_video_events_reproduced':all_before_stats['n_events']==29414,
        'all_before_pairs_have_no_final_label':len(status_crosscheck)==1 and status_crosscheck[0]['label_status_weighted']=='NO_OBSERVED_CHAPTER',
        'all_before_group_is_video_only':con.execute('''SELECT count(*)=0 FROM pair_source s JOIN pair_audit p USING(user_id,course_id)
            WHERE p.pair_event_case='ALL_BEFORE_ENROLLMENT' AND s.event_source<>'video' ''').fetchone()[0],
        'delta_sign_matches_event_case':con.execute('''SELECT count(*)=0 FROM filtered_events
            WHERE (event_time_case='BEFORE_ENROLLMENT' AND delta_seconds>=0)
               OR (event_time_case='AFTER_COURSE_END' AND event_time<=course_end)''').fetchone()[0],
        'all_before_plus8h_counts_conserve_events':recovery['events_in_window_if_plus8h']+recovery['events_still_before_if_plus8h']+recovery['events_after_end_if_plus8h']==all_before_stats['n_events'],
        'video_sensitivity_counts_conserve_events':all(r['n_before']+r['n_within']+r['n_after']==r['n_events'] for r in sensitivity),
        'first_valid_deltas_are_nonnegative':con.execute('''SELECT count(*)=0 FROM pair_source WHERE time_eligible
            AND (first_valid_delta_seconds<0 OR (n_within>0 AND first_valid_delta_seconds IS NULL))''').fetchone()[0],
        'reply_parent_filter_reproduces_prepared_source':reply_parent_audit['n_retained_in_reply_prepared']==next(r['n_events'] for r in source_totals if r['source']=='reply'),
        'source_files_unchanged':all(sha(base/p)==h for p,h in hashes.items()),
    }
    dump(out/'validation.json',checks)
    assert all(checks.values()),checks
    source_context = json.loads((base/'review_time_phases/summary.json').read_text()).get('video_time_audit')
    summary = dict(as_of=AS_OF,delta_definition='event_time minus enroll_time, seconds',
        eligibility=eligibility,source_totals=source_totals,source_event_cases=source_cases,
        event_delta_statistics=delta_stats,first_event_pair_profiles=first_event_profiles,
        reply_parent_time_filter=reply_parent_audit,pair_cases=pair_cases,all_before_pair_statistics=all_before_stats,
        negative_event_bins=negative_bins,all_before_pair_nearest_event_bins=pair_bins,near_minus8h=near_8h,
        video_shift_sensitivity=sensitivity,all_before_plus8h_sensitivity=recovery,
        all_before_label_status=status_crosscheck,examples_near_8h=examples,tail_examples=tail_examples,
        mapped_video_source_filter_context=source_context,validation=checks,elapsed_seconds=round(time.monotonic()-started,3))
    dump(out/'summary.json',summary)
    report = render(summary)
    report_path=out/'bao_cao_do_lech_thoi_gian.md'
    report_path.write_text(report)
    manifest=dict(as_of=AS_OF,engine_source_sha256=sha(__file__),source_files_sha256=hashes,
        report_sha256=sha(report_path),production_mutation='none',
        scope='Prepared mapped source events; all registration statuses audited; delta profiles on time-eligible pairs.',
        timezone_correction_applied=False,shift_sensitivity_uses_frozen_course_end=True)
    dump(out/'run_manifest.json',manifest)
    dump(out/'_SUCCESS.json',dict(manifest_sha256=sha(out/'run_manifest.json'),report_sha256=sha(report_path)))
    print('AUDIT_COMPLETE',json.dumps(dict(pair_cases=pair_cases,near_minus8h=near_8h,recovery=recovery,validation=checks),default=str),flush=True)


def render(s):
    source_eligible_counts=[]
    for source in SOURCES:
        def count(case):
            return sum(r['n_events'] for r in s['source_event_cases']
                       if r['event_source']==source and r['pair_time_status']=='ELIGIBLE_TIME' and r['event_time_case']==case)
        source_eligible_counts.append([source,count('BEFORE_ENROLLMENT'),count('WITHIN_FINAL_WINDOW'),
                                       count('AFTER_COURSE_END'),count('INVALID_EVENT_TIME')])
    parts=[
        '# Các trường hợp bị lọc thời gian và độ lệch event–enrollment',
        '**Ngày:** '+s['as_of']+'. Tất cả thống kê quét đầy đủ nguồn đã mapping, không lấy mẫu. Không sửa dataset.',
        '## 1. Phạm vi và quy ước',
        'Delta = event_time − enroll_time. Delta âm nghĩa event đang được đặt trước đăng ký; delta dương nghĩa sau đăng ký. '
        'Đơn vị lưu trong thống kê là giây, bảng trình bày quy đổi giờ hoặc ngày. '
        'Timestamp dùng đúng cách parse hiện tại: video epoch qua DuckDB UTC, enrollment là chuỗi không kèm timezone. '
        'Chưa có xác nhận hai nguồn có cùng múi giờ gốc.',
        'Nguồn là video_events, mapped problem rows, comment_prepared và reply_prepared của pipeline 06, trước lọc enrollment/end. '
        'Đây là timestamp hợp lệ về parse/mapping, không phải khẳng định điểm/thời lượng hoặc timezone đã đúng. '
        'Problem row là trạng thái user–problem, không phải từng attempt.',
        '## 2. Trạng thái cặp ghi danh trước khi lọc event',
        mdtable(['Trạng thái','Số cặp'],[[r['time_status'],r['n_pairs']] for r in s['eligibility']]),
        '## 3. Các trường hợp event theo từng nguồn',
        'Tóm tắt trên cặp đủ điều kiện thời gian:',
        mdtable(['Nguồn','Trước enrollment','Trong cửa sổ cuối kỳ','Sau course_end','Timestamp NULL'],source_eligible_counts),
        'Bảng đầy đủ gồm cả event không có ghi danh và event thuộc cặp không đủ điều kiện:',
        mdtable(['Nguồn','Trạng thái cặp','Trường hợp event','Số event','Số cặp'],
                [[r[k] for k in ['event_source','pair_time_status','event_time_case','n_events','n_pairs']] for r in s['source_event_cases']]),
        'NO_ENROLLMENT_RECORD: không tìm thấy đúng cặp trong bảng ghi danh. PAIR_TIME_INELIGIBLE: có ghi danh nhưng cặp không đủ điều kiện thời gian. '
        'Hai nhóm này tách khỏi event-before-enrollment của cặp hợp lệ. BEFORE_ENROLLMENT, AFTER_COURSE_END và INVALID_EVENT_TIME là event bị loại ở cặp time-eligible. '
        'WITHIN_FINAL_WINDOW là event được giữ theo cửa sổ cuối kỳ. Số cặp giữa các nguồn có thể trùng; không cộng trực tiếp để ra tổng cặp duy nhất.',
        '## 4. Độ lệch của event trước đăng ký và event được giữ',
        'Đơn vị bảng: giờ. Phân vị là theo event, không theo user–course. Bảng chỉ xét cặp đủ điều kiện thời gian.',
        mdtable(['Nguồn','Trường hợp','Số event','Min Δ(h)','P5 Δ(h)','Median Δ(h)','Mean Δ(h)','P95 Δ(h)','Max Δ(h)'],
                [[r['event_source'],r['event_time_case'],r['n_events']]
                  +[r[k]/3600 for k in ['min_seconds','p05_seconds','median_seconds','mean_seconds','p95_seconds','max_seconds']]
                  for r in s['event_delta_statistics']]),
        'Đối chiếu ở cấp cặp: mỗi user–course chỉ có một event đầu tiên cho mỗi nguồn. '
        'Event hợp lệ đầu tiên ở đây là event đầu tiên còn lại sau khi áp dụng enrollment/end; '
        'trung vị của event này không phải trung vị của tất cả event:',
        mdtable(['Nguồn','Cặp có event','Cặp có event trong cửa sổ','Median Δ event đầu (h)',
                 'P5 Δ event hợp lệ đầu (h)','Median Δ event hợp lệ đầu (h)','P95 Δ event hợp lệ đầu (h)'],
                [[r[k] for k in ['event_source','n_pairs_with_event','n_pairs_with_valid_event',
                  'median_first_event_delta_hours','p05_first_valid_delta_hours','median_first_valid_delta_hours',
                  'p95_first_valid_delta_hours']] for r in s['first_event_pair_profiles']]),
        'Không có event sau course_end ở cặp đủ điều kiện trong nguồn prepared hiện tại. '
        'course_end đã được suy ra từ hoạt động cuối, nên kết quả này không tự xác nhận timezone là đúng.',
        '## 5. Cặp bị mất toàn bộ event và cặp chỉ mất một phần',
        mdtable(['Trường hợp cặp','Số cặp','Số event','Trước enroll','Trong cửa sổ','Sau course_end','Thiếu timestamp'],
                [[r[k] for k in ['pair_event_case','n_pairs','n_events','n_before','n_within','n_after','n_invalid']] for r in s['pair_cases']]),
        'ALL_BEFORE_ENROLLMENT là nhóm 5.130 cặp đã phát hiện. BEFORE_AND_WITHIN vẫn có hoạt động hợp lệ, chỉ bị mất một phần event; '
        'không gộp hai nhóm này thành nhóm hoàn toàn chưa quan sát.',
        '## 6. Phân bố khoảng thời gian event sớm hơn enroll',
        'Các bin dùng đầu trái đóng, đầu phải mở. Số cặp trong bảng event có thể xuất hiện ở nhiều bin.',
        mdtable(['Sớm hơn enroll','Số event','Số cặp có event trong bin'],
                [[r['bin_name'],r['n_events'],r['n_pairs']] for r in s['negative_event_bins']]),
        'Với riêng cặp ALL_BEFORE_ENROLLMENT, mỗi cặp chỉ đếm một lần theo event cuối cùng, tức event gần enrollment nhất:',
        mdtable(['Event cuối sớm hơn enroll','Số cặp'],
                [[r['bin_name'],r['n_pairs']] for r in s['all_before_pair_nearest_event_bins']]),
        '## 7. Dấu hiệu lệch khoảng 8 giờ',
        mdtable(['Thống kê','Giá trị'],list(s['near_minus8h'].items())+list(s['all_before_pair_statistics'].items())),
        'Sự tập trung quanh −8 giờ là dấu hiệu cần kiểm tra timezone, không tự chứng minh UTC+8 hoặc cho phép sửa có chọn lọc chỉ các event âm.',
        '## 8. Phân tích giả định dịch toàn bộ video',
        'Chỉ là sensitivity: cộng cùng một offset vào toàn bộ video time-eligible rồi so với enrollment/course_end hiện tại. '
        'Course_end được giữ cố định; chưa tái dựng course bounds, cutoff, nhãn hoặc kiểm tra lại cửa sổ nguồn sau dịch. '
        'Đây không phải dataset đã sửa và không phải đánh giá triển khai.',
        mdtable(['Dịch video (giờ)','Tổng event','Trước enroll','Trong cửa sổ','Sau course_end'],
                [[r[k] for k in ['shift_hours','n_events','n_before','n_within','n_after']] for r in s['video_shift_sensitivity']]),
        'Riêng nhóm mất toàn bộ event:',
        mdtable(['Thống kê nếu video +8h','Giá trị'],list(s['all_before_plus8h_sensitivity'].items())),
        '## 9. Ví dụ và đuôi phân bố',
        'Ví dụ cặp có event cuối gần −8 giờ:',
        mdtable(['User','Course','Enroll','Event đầu','Event cuối','Số event','Δ đầu (h)','Δ cuối (h)','Event trong cửa sổ nếu +8h'],
                [[r[k] for k in ['user_id','course_id','enroll_time','first_event','last_event','n_events',
                                 'first_delta_hours','last_delta_hours','n_within_if_video_plus8h']] for r in s['examples_near_8h']]),
        'Các cặp có event sớm nhất cách enrollment xa nhất; đơn vị ngày:',
        mdtable(['User','Course','Enroll','Event đầu','Event cuối','Số event','Δ đầu (ngày)','Δ cuối (ngày)'],
                [[r[k] for k in ['user_id','course_id','enroll_time','first_event','last_event','n_events',
                                 'first_delta_days','last_delta_days']] for r in s['tail_examples']]),
        '## 10. Kiểm tra, giới hạn và dữ liệu chi tiết',
        'Kiểm tra bảo toàn số event, dấu delta, phân nhóm cặp, đối chiếu nhóm 5.130/29.414 với master và bảo toàn nguồn. '
        'Event sau cutoff của phase nhưng trong cửa sổ cuối kỳ là sự kiện tương lai đối với phase đó; '
        'audit này không gọi việc loại chúng ở phase sớm là lỗi.',
        'Các bộ lọc trước nguồn prepared không được gộp vào số BEFORE_ENROLLMENT: video ngoài cửa sổ nguồn, '
        'reply trước parent, resource không mapping được hoặc liên kết không nhất quán. '
        'Kiểm tra nguồn video ở bước 05: '+str(s['mapped_video_source_filter_context'])+'.',
        'Đã tái lập riêng bộ lọc thời gian reply so với parent comment, trước khi xét enrollment:',
        mdtable(['Thống kê reply-parent','Giá trị'],list(s['reply_parent_time_filter'].items())),
        '- [Thống kê đầy đủ và phân vị P1/P5/P25/P50/P75/P95/P99](summary.json)\n'
        '- [Kiểm tra](validation.json)\n'
        '- [Manifest và checksum nguồn](run_manifest.json)\n'
        '- [Cặp và các trường hợp thời gian](tables/pair_time_audit.parquet)\n'
        '- [Nhóm mất toàn bộ event, kèm sensitivity +8h](tables/all_before_enrollment_pairs.parquet)\n'
        '- [Event bị lọc: video](tables/filtered_events/video.parquet)\n'
        '- [Event bị lọc: problem](tables/filtered_events/problem.parquet)\n'
        '- [Event bị lọc: comment](tables/filtered_events/comment.parquet)\n'
        '- [Event bị lọc: reply](tables/filtered_events/reply.parquet)\n'
        '- [Mã tái lập](../12_audit_event_enrollment_time.py)',
        'Chạy từ working_13-10 bằng Python có DuckDB 1.3.2: python 12_audit_event_enrollment_time.py.',
    ]
    return '\n\n'.join(parts)+'\n'


if __name__=='__main__':
    main()
