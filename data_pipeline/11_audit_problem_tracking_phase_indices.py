"""Recompute chapter COELO/ACELO at pre-problem-tracking phase cutoffs.

This is an isolated diagnostic, not a replacement of phase features or targets.
It uses the current final-index recipe and frozen train P95, with video clipped
at the phase cutoff. Unobserved chapters remain absent, and empty pairs null.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

import duckdb

TRACKING_START = '2020-05-01 00:00:00'


def qs(value):
    return "'" + str(value).replace("'", "''") + "'"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + '\n')


def records(con, sql):
    cur = con.execute(sql)
    names = [x[0] for x in cur.description]
    return [dict(zip(names, row)) for row in cur.fetchall()]


def table(headers, data):
    def cell(x):
        if x is None:
            return 'NULL'
        if isinstance(x, float):
            return f'{x:.6f}'
        if isinstance(x, dt.datetime):
            return x.isoformat(sep=' ')
        return str(x).replace('|', '\\|').replace('\n', ' ')
    return '\n'.join(
        ['| ' + ' | '.join(headers) + ' |',
         '| ' + ' | '.join('---' for _ in headers) + ' |']
        + ['| ' + ' | '.join(cell(x) for x in row) + ' |' for row in data])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root')
    args = parser.parse_args()
    base = Path(args.root or (Path.cwd() / 'working_13-10'
                             if (Path.cwd() / 'working_13-10').is_dir() else Path.cwd())).resolve()
    out = base / 'problem_tracking_phase_audit'
    (out / 'tables').mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("SET memory_limit='2GB'")
    con.execute('SET threads=2')
    con.execute("SET TimeZone='UTC'")
    constants = json.loads((base / 'final_quality_tables/protocol_constants.json').read_text())
    p95 = float(constants['P95_IF_raw_training'])
    source_paths = [
        base / '06_build_final_quality_tables.py',
        base / '09_build_phase_features.py',
        base / 'final_quality_tables/protocol_constants.json',
        base / 'final_quality_tables/_stage/video_window.parquet',
        base / 'final_quality_tables/_stage/comment_window.parquet',
        base / 'final_quality_tables/_stage/reply_prepared.parquet',
        base / 'final_quality_tables/tables/course_splits.parquet',
    ]
    for pattern in [
        'phase_features/supervised_raw/*.parquet',
        'input/moocubex_model/processed/dim_lo_catalog/*.parquet',
        'eda_engagement_features_comparison/_stage/problem_rows/**/*.parquet',
    ]:
        source_paths.extend(sorted(base.glob(pattern)))
    hashes = {str(p.relative_to(base)): sha(p) for p in source_paths}

    def view(name, pattern):
        con.execute(f'CREATE VIEW {name} AS SELECT * FROM read_parquet({qs(base / pattern)},union_by_name=true)')

    def temp(name, sql):
        con.execute(f'CREATE OR REPLACE TEMP TABLE {name} AS {sql}')

    def save(name, sql):
        target = out / 'tables' / f'{name}.parquet'
        con.execute(f'COPY ({sql}) TO {qs(target)} (FORMAT PARQUET,COMPRESSION ZSTD)')

    view('phase_raw', 'phase_features/supervised_raw/*.parquet')
    view('catalog', 'input/moocubex_model/processed/dim_lo_catalog/*.parquet')
    view('video_source', 'final_quality_tables/_stage/video_window.parquet')
    view('comment_source', 'final_quality_tables/_stage/comment_window.parquet')
    view('reply_source', 'final_quality_tables/_stage/reply_prepared.parquet')
    view('problem_source', 'eda_engagement_features_comparison/_stage/problem_rows/**/*.parquet')
    view('course_bounds', 'final_quality_tables/tables/course_splits.parquet')
    start = f'TIMESTAMP {qs(TRACKING_START)}'
    raw_problem = records(con, '''SELECT min(event_time) first_mapped_problem_time,
        max(event_time) last_mapped_problem_time,count(*) n_mapped_problem_rows,
        count(*) FILTER(WHERE event_time<TIMESTAMP '2020-05-01') n_before_tracking
        FROM problem_source''')[0]
    effective_video = records(con, '''SELECT min(event_time) first_effective_video_time,
        max(event_time) last_effective_video_time,count(*) n_segments,
        count(*) FILTER(WHERE event_time<TIMESTAMP '2020-05-01') n_before_problem_start
        FROM video_source''')[0]
    print('Verified mapped problem range', raw_problem, flush=True)

    # Six deterministic, high-social-activity pairs from different courses.
    # Selected for readable before/after illustrations, not representativeness.
    temp('chosen_pairs', f'''WITH ranked AS (
        SELECT a.user_id,a.course_id,a.n_events p1_events,
          z.n_problem_submissions p4_problem_rows,
          row_number() OVER(PARTITION BY a.course_id ORDER BY a.n_events DESC,a.user_id) rn
        FROM phase_raw a JOIN phase_raw z USING(user_id,course_id)
        WHERE a.phase='P1' AND z.phase='P4' AND a.cutoff_time<{start}
          AND z.cutoff_time>={start} AND a.n_events>0 AND z.n_problem_submissions>0)
      SELECT * EXCLUDE(rn) FROM ranked WHERE rn=1
      ORDER BY p1_events DESC,user_id,course_id LIMIT 6''')
    temp('snapshots', f'''SELECT f.user_id,f.course_id,f.phase,f.split,f.enroll_time,
        f.cutoff_time,b.course_end,f.cutoff_time<{start} before_problem_tracking,
        s.user_id IS NOT NULL selected_example,
        CASE WHEN f.cutoff_time<{start} THEN 'BEFORE_PROBLEM_SOURCE'
             WHEN f.enroll_time<{start} THEN 'PARTIALLY_IN_PROBLEM_SOURCE'
             ELSE 'WITHIN_PROBLEM_SOURCE_DATES' END problem_source_scope
      FROM phase_raw f JOIN course_bounds b USING(course_id)
      LEFT JOIN chosen_pairs s USING(user_id,course_id)
      WHERE f.cutoff_time<{start} OR s.user_id IS NOT NULL''')
    print('Diagnostic snapshots', con.execute('SELECT count(*) FROM snapshots').fetchone()[0], flush=True)

    # Distinct observed keys come from events at this cutoff, never final keys.
    temp('video_w', '''WITH selected AS (
        SELECT v.*,s.phase,s.enroll_time,s.cutoff_time,
          least(v.end_point-v.start_point,
            greatest(0.0,epoch(s.cutoff_time)-epoch(v.event_time))*v.speed) observed_media_seconds
        FROM video_source v JOIN snapshots s USING(user_id,course_id)
        WHERE v.valid_measurement AND v.event_time BETWEEN s.enroll_time AND s.cutoff_time)
      SELECT *,start_point+observed_media_seconds observed_end_point,
        event_time+(observed_media_seconds/speed)*INTERVAL '1 second' observed_end_time
      FROM selected WHERE observed_media_seconds>0''')
    temp('problem_w', '''SELECT p.*,s.phase,s.enroll_time,s.cutoff_time
      FROM problem_source p JOIN snapshots s USING(user_id,course_id)
      WHERE p.event_time BETWEEN s.enroll_time AND s.cutoff_time''')
    temp('comment_w', '''SELECT c.user_id,c.course_id,c.lo_key,c.event_time,s.phase,s.enroll_time,s.cutoff_time
      FROM comment_source c JOIN snapshots s USING(user_id,course_id)
      WHERE c.event_time BETWEEN s.enroll_time AND s.cutoff_time''')
    temp('reply_w', '''SELECT r.user_id,r.course_id,r.lo_key,r.event_time,s.phase,s.enroll_time,s.cutoff_time
      FROM reply_source r JOIN snapshots s USING(user_id,course_id)
      WHERE r.event_time BETWEEN s.enroll_time AND s.cutoff_time''')
    temp('video_agg', '''WITH previous AS (
        SELECT *,max(observed_end_point) OVER(
          PARTITION BY user_id,course_id,phase,lo_key,video_id ORDER BY start_point,observed_end_point
          ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) previous_end FROM video_w),
      islands AS (SELECT *,sum(CASE WHEN previous_end IS NULL OR start_point>previous_end THEN 1 ELSE 0 END)
        OVER(PARTITION BY user_id,course_id,phase,lo_key,video_id ORDER BY start_point,observed_end_point) island
        FROM previous),
      merged AS (SELECT user_id,course_id,phase,lo_key,video_id,island,
        max(observed_end_point)-min(start_point) union_seconds FROM islands GROUP BY 1,2,3,4,5,6),
      unique_time AS (SELECT user_id,course_id,phase,lo_key,sum(union_seconds) unique_watch_seconds
        FROM merged GROUP BY 1,2,3,4),
      counts AS (SELECT user_id,course_id,phase,lo_key,count(*) n_video_segments,
        count(DISTINCT video_id) n_videos_watched FROM video_w GROUP BY 1,2,3,4)
      SELECT c.*,u.unique_watch_seconds FROM counts c JOIN unique_time u USING(user_id,course_id,phase,lo_key)''')
    temp('problem_agg', '''SELECT user_id,course_id,phase,lo_key,count(*) n_problems_attempted,
        count(DISTINCT exercise_id) n_exercises_attempted,sum(attempts) total_problem_attempts,
        count(*) FILTER(WHERE attempts IS NULL OR attempts<=0) invalid_attempts,
        coalesce(sum(CASE WHEN score_valid THEN score WHEN fallback_possible THEN is_correct*problem_max_score END),0) earned_score_sum,
        coalesce(sum(CASE WHEN efficiency_valid THEN is_correct::DOUBLE/attempts END),0) efficiency_sum,
        count(*) FILTER(WHERE NOT score_valid AND NOT fallback_possible) n_scores_unresolved,
        count(*) FILTER(WHERE NOT efficiency_valid) n_efficiency_unresolved
      FROM problem_w GROUP BY 1,2,3,4''')
    counts = [
        ('v','n_video_segments'),('v','n_videos_watched'),('v','unique_watch_seconds'),
        ('p','n_problems_attempted'),('p','n_exercises_attempted'),('p','total_problem_attempts'),
        ('p','invalid_attempts'),('p','earned_score_sum'),('p','efficiency_sum'),
        ('p','n_scores_unresolved'),('p','n_efficiency_unresolved'),
    ]
    count_sql = ','.join(f'coalesce({a}.{name},0) {name}' for a,name in counts)
    temp('chapter_inputs', f'''WITH keys AS (
        SELECT user_id,course_id,phase,lo_key FROM video_agg
        UNION SELECT user_id,course_id,phase,lo_key FROM problem_agg
        UNION SELECT user_id,course_id,phase,lo_key FROM comment_w
        UNION SELECT user_id,course_id,phase,lo_key FROM reply_w)
      SELECT k.*,c.chapter_major,c.n_videos_available,c.available_video_seconds_proxy,
        c.n_videos_missing_duration,c.n_exercises_available,c.n_problems_available,
        c.available_max_score_sum,c.n_problems_missing_score,
        s.enroll_time,s.cutoff_time,s.course_end,s.split,s.before_problem_tracking,
        s.selected_example,s.problem_source_scope,{count_sql}
      FROM keys k JOIN snapshots s USING(user_id,course_id,phase)
      JOIN catalog c USING(course_id,lo_key)
      LEFT JOIN video_agg v USING(user_id,course_id,phase,lo_key)
      LEFT JOIN problem_agg p USING(user_id,course_id,phase,lo_key)''')
    temp('chapter_numeric', '''SELECT *,
        CASE WHEN n_videos_missing_duration=0 AND available_video_seconds_proxy>0
          THEN least(1.0,greatest(0.0,unique_watch_seconds/available_video_seconds_proxy)) END TS,
        n_videos_watched::DOUBLE/nullif(n_videos_available,0) RV,
        n_exercises_attempted::DOUBLE/nullif(n_exercises_available,0) AV,
        CASE WHEN invalid_attempts=0 THEN total_problem_attempts::DOUBLE/nullif(n_problems_available,0) END IF_raw,
        CASE WHEN n_problems_missing_score=0 AND n_scores_unresolved=0
          THEN earned_score_sum/nullif(available_max_score_sum,0) END KL,
        CASE WHEN n_efficiency_unresolved=0 THEN efficiency_sum/nullif(n_problems_available,0) END CFA
      FROM chapter_inputs''')
    temp('chapter_indices', f'''WITH normalized AS (
        SELECT *,CASE WHEN IF_raw IS NOT NULL
          THEN least(1.0,greatest(0.0,ln(1+IF_raw)/ln(1+{p95!r}))) END IF_value FROM chapter_numeric)
      SELECT *,list_count([TS,RV,AV,IF_value]) coelo_n_components,
        CASE WHEN list_count([TS,RV,AV,IF_value])>=2 THEN list_avg([TS,RV,AV,IF_value]) END COELO,
        list_count([KL,CFA]) acelo_n_components,list_avg([KL,CFA]) ACELO
      FROM normalized''')
    temp('pair_indices', '''WITH agg AS (
        SELECT user_id,course_id,phase,count(*) n_chapters_observed,
          count(COELO) n_chapters_with_coelo,count(ACELO) n_chapters_with_acelo,
          sum(n_video_segments) n_video_segments,sum(n_problems_attempted) n_problem_rows,
          sum(total_problem_attempts) problem_attempts,
          sum(unique_watch_seconds) video_unique_seconds,
          avg(COELO) COELO_mean_observed_chapters,avg(ACELO) ACELO_mean_observed_chapters
        FROM chapter_indices GROUP BY 1,2,3)
      SELECT s.*,coalesce(a.n_chapters_observed,0) n_chapters_observed,
        coalesce(a.n_chapters_with_coelo,0) n_chapters_with_coelo,
        coalesce(a.n_chapters_with_acelo,0) n_chapters_with_acelo,
        coalesce(a.n_video_segments,0) n_video_segments,coalesce(a.n_problem_rows,0) n_problem_rows,
        coalesce(a.problem_attempts,0) problem_attempts,coalesce(a.video_unique_seconds,0) video_unique_seconds,
        a.COELO_mean_observed_chapters,a.ACELO_mean_observed_chapters
      FROM snapshots s LEFT JOIN agg a USING(user_id,course_id,phase)''')
    summary_rows = records(con, '''SELECT phase,count(*) n_snapshots,
        count(*) FILTER(WHERE n_chapters_observed>0) with_observed_chapter,
        count(*) FILTER(WHERE n_chapters_observed=0) without_observed_chapter,
        count(COELO_mean_observed_chapters) with_coelo,
        count(ACELO_mean_observed_chapters) with_acelo,
        count(*) FILTER(WHERE ACELO_mean_observed_chapters=0) acelo_zero,
        max(n_problem_rows) max_problem_rows,
        avg(COELO_mean_observed_chapters) mean_coelo,
        avg(ACELO_mean_observed_chapters) mean_acelo
      FROM pair_indices WHERE before_problem_tracking GROUP BY phase ORDER BY phase''')
    save('chapter_phase_indices','SELECT * FROM chapter_indices ORDER BY user_id,course_id,phase,chapter_major')
    save('user_course_phase_indices','SELECT * FROM pair_indices ORDER BY user_id,course_id,phase')
    save('selected_user_courses','SELECT * FROM chosen_pairs')

    # Match independent existing phase aggregates and manually check identities.
    checks = {
        'mapped_problem_source_has_no_event_before_2020_05_01': raw_problem['n_before_tracking']==0,
        'one_row_per_pair_phase': con.execute('SELECT count(*)=count(DISTINCT(user_id,course_id,phase)) FROM pair_indices').fetchone()[0],
        'one_row_per_observed_chapter_phase': con.execute('SELECT count(*)=count(DISTINCT(user_id,course_id,phase,lo_key)) FROM chapter_indices').fetchone()[0],
        'pretracking_counts_match_full_supervised_phase_tables': con.execute(f'''SELECT count(*)=0 FROM (
            (SELECT user_id,course_id,phase FROM phase_raw WHERE cutoff_time<{start}
             EXCEPT SELECT user_id,course_id,phase FROM pair_indices WHERE before_problem_tracking)
            UNION ALL
            (SELECT user_id,course_id,phase FROM pair_indices WHERE before_problem_tracking
             EXCEPT SELECT user_id,course_id,phase FROM phase_raw WHERE cutoff_time<{start}))''').fetchone()[0],
        'no_future_source_event': all(con.execute(f'SELECT count(*)=0 FROM {name} WHERE event_time<enroll_time OR event_time>cutoff_time').fetchone()[0]
                                      for name in ['video_w','problem_w','comment_w','reply_w']),
        'video_clipped_at_cutoff': con.execute("SELECT count(*)=0 FROM video_w WHERE observed_end_time>cutoff_time+INTERVAL '1 microsecond'").fetchone()[0],
        'no_pretracking_problem_rows': con.execute('SELECT count(*)=0 FROM pair_indices WHERE before_problem_tracking AND n_problem_rows<>0').fetchone()[0],
        'no_pretracking_mapped_video_in_current_source': effective_video['n_before_problem_start']==0,
        'no_pretracking_video_in_audited_snapshots': con.execute('SELECT count(*)=0 FROM pair_indices WHERE before_problem_tracking AND n_video_segments<>0').fetchone()[0],
        'pretracking_acelo_zero_when_problem_catalog_nonempty': con.execute('''SELECT count(*)=0 FROM chapter_indices
            WHERE before_problem_tracking AND n_problems_available>0 AND (ACELO IS NULL OR ACELO<>0 OR CFA<>0)''').fetchone()[0],
        'pretracking_av_if_zero_with_positive_denominators': con.execute('''SELECT count(*)=0 FROM chapter_indices
            WHERE before_problem_tracking AND n_problems_available>0 AND n_exercises_available>0
              AND (AV IS NULL OR AV<>0 OR IF_value IS NULL OR IF_value<>0)''').fetchone()[0],
        'pretracking_coelo_four_component_identity': con.execute('''SELECT count(*)=0 FROM chapter_indices
            WHERE before_problem_tracking AND coelo_n_components=4 AND abs(COELO-(TS+RV)/4)>1e-12''').fetchone()[0],
        'unobserved_pairs_keep_null_indices': con.execute('''SELECT count(*)=0 FROM pair_indices WHERE n_chapters_observed=0
            AND (COELO_mean_observed_chapters IS NOT NULL OR ACELO_mean_observed_chapters IS NOT NULL)''').fetchone()[0],
        'event_counts_match_existing_phase_features': con.execute('''SELECT count(*)=0 FROM pair_indices a
            JOIN phase_raw r USING(user_id,course_id,phase) WHERE a.n_video_segments<>r.n_video_segments
              OR a.n_problem_rows<>r.n_problem_submissions OR a.problem_attempts<>r.problem_attempts
              OR abs(a.video_unique_seconds-r.video_unique_seconds)>1e-6''').fetchone()[0],
        'indices_in_unit_interval': con.execute('''SELECT count(*)=0 FROM chapter_indices
            WHERE COELO < -1e-12 OR COELO>1+1e-12 OR ACELO < -1e-12 OR ACELO>1+1e-12''').fetchone()[0],
        'source_files_unchanged': all(sha(base/p)==h for p,h in hashes.items()),
    }
    dump(out/'validation.json',checks)
    assert all(checks.values()), checks
    selected = records(con,'SELECT * FROM chosen_pairs ORDER BY p1_events DESC,user_id,course_id')
    examples = records(con, '''SELECT p.user_id,p.course_id,p.phase,p.cutoff_time,p.before_problem_tracking,
        p.n_chapters_observed,p.n_video_segments,p.n_problem_rows,
        p.COELO_mean_observed_chapters,p.ACELO_mean_observed_chapters
      FROM pair_indices p JOIN chosen_pairs c USING(user_id,course_id)
      ORDER BY c.p1_events DESC,p.user_id,p.course_id,p.phase''')
    chapter_examples = records(con, '''WITH ranked AS (
        SELECT *,row_number() OVER(PARTITION BY user_id,course_id ORDER BY COELO DESC,chapter_major) rn
        FROM chapter_indices WHERE before_problem_tracking AND phase='P1'
          AND selected_example AND coelo_n_components=4)
      SELECT user_id,course_id,lo_key,phase,cutoff_time,n_videos_watched,n_problems_available,
        n_problems_attempted,TS,RV,AV,IF_value,KL,CFA,COELO,ACELO
      FROM ranked WHERE rn=1 ORDER BY COELO DESC,user_id,course_id''')
    same_chapter_examples = records(con, '''WITH chosen_chapters AS (
        SELECT user_id,course_id,lo_key FROM chapter_indices
        WHERE selected_example GROUP BY 1,2,3
        HAVING count(*)=4 AND count(*) FILTER(WHERE before_problem_tracking AND n_problems_attempted=0)>0
          AND max(n_problems_attempted)>0),
      ranked AS (SELECT c.*,row_number() OVER(PARTITION BY c.user_id,c.course_id
        ORDER BY max(p.n_problems_attempted) DESC,c.lo_key) rn
        FROM chosen_chapters c JOIN chapter_indices p USING(user_id,course_id,lo_key)
        GROUP BY c.user_id,c.course_id,c.lo_key)
      SELECT p.user_id,p.course_id,p.lo_key,p.phase,p.cutoff_time,p.n_problems_available,
        p.n_problems_attempted,p.AV,p.IF_value,p.KL,p.CFA,p.COELO,p.ACELO
      FROM chapter_indices p JOIN ranked r USING(user_id,course_id,lo_key)
      WHERE r.rn=1 ORDER BY p.user_id,p.course_id,p.phase''')
    unobserved_examples = records(con, '''SELECT user_id,course_id,phase,cutoff_time,n_chapters_observed,
        n_video_segments,n_problem_rows,COELO_mean_observed_chapters,ACELO_mean_observed_chapters
      FROM pair_indices WHERE before_problem_tracking AND n_chapters_observed=0
      ORDER BY phase,user_id,course_id LIMIT 3''')
    one_example_per_phase = records(con, '''WITH ranked AS (
        SELECT *,row_number() OVER(PARTITION BY phase ORDER BY user_id,course_id) rn
        FROM pair_indices WHERE before_problem_tracking AND n_chapters_observed>0)
      SELECT user_id,course_id,phase,cutoff_time,n_chapters_observed,n_video_segments,
        n_problem_rows,COELO_mean_observed_chapters,ACELO_mean_observed_chapters
      FROM ranked WHERE rn=1 ORDER BY phase''')
    summary = {
        'as_of':'2026-10-17','problem_tracking_start':TRACKING_START,
        'scope':'All labeled phase snapshots before the problem-source start, plus four phases of six selected pairs.',
        'not_a_production_label_update':True,'formula_recipe':constants,
        'mapped_problem_source':raw_problem,'effective_video_source':effective_video,
        'before_tracking_by_phase':summary_rows,
        'selected_pairs':selected,'example_pair_indices':examples,'example_chapter_indices':chapter_examples,
        'same_chapter_examples':same_chapter_examples,'unobserved_examples':unobserved_examples,
        'one_observed_example_per_pretracking_phase':one_example_per_phase,
        'n_audit_snapshots':con.execute('SELECT count(*) FROM pair_indices').fetchone()[0],
        'n_observed_chapter_snapshots':con.execute('SELECT count(*) FROM chapter_indices').fetchone()[0],
        'validation':checks,
    }
    dump(out/'summary.json',summary)
    doc = [
        '# Kiểm chứng COELO và ACELO tại phase trước cửa sổ log problem',
        '**Ngày:** 17/10/2026. **Phạm vi:** tập supervised 164.429 user–course.',
        'Đây là bảng chẩn đoán mới: tính lại chỉ số tại cutoff bằng công thức đang dùng, '
        'không phải ba chỉ số đã được lưu sẵn cho từng phase. Nhãn cuối kỳ, feature và split hiện tại không thay đổi.',
        '## 1. Phương pháp và phạm vi kết luận',
        'Đọc dòng sự kiện đã mapping; lọc trong [enroll_time, cutoff]. Video vượt cutoff được clip theo playback speed, '
        'sau đó hợp các khoảng nội dung để tính unique watch time. Chỉ tạo chapter đã có sự kiện trước cutoff. '
        'Comment/reply chỉ xác định chapter quan sát, không tham gia trực tiếp công thức COELO/ACELO.',
        'Mẫu số là catalog chapter hiện tại như bước 06. P95 IF giữ nguyên giá trị train cuối kỳ '
        f'{p95:g}; không fit lại trên phase và không lấy chỉ số cuối kỳ làm chỉ số phase. '
        'Cửa sổ không có chapter quan sát giữ COELO/ACELO NULL.',
        'Nguồn problem đã mapping có thời gian sớm nhất '
        f"**{raw_problem['first_mapped_problem_time']}**, không có dòng trước 01/05/2020. "
        'Điều này chứng minh không có bằng chứng làm bài trong nguồn ở giai đoạn đó; '
        'không chứng minh người học ngoài thực tế chưa làm bài hoặc nguyên nhân upstream thiếu log.',
        'Điểm bổ sung từ kiểm chứng: nguồn video đã mapping và lọc cuối kỳ đang dùng có segment sớm nhất '
        f"**{effective_video['first_effective_video_time']}**. Không có video đã nối được trước 01/05 trong nguồn này. "
        'Vì vậy các chapter quan sát trước mốc trong tập kiểm chứng đến từ comment/reply; '
        'không có ví dụ thực tế vừa có video trước tháng 5 vừa thiếu problem ở tập này. '
        'Cửa sổ video được phép dùng 19/01–03/11 là khác với khoảng thực tế còn lại sau mapping/lọc; '
        'không được suy rộng mốc 11/06 thành ngày tracking bắt đầu của toàn bộ raw video.',
        '## 2. Kết quả trên toàn bộ snapshot có nhãn trước mốc problem',
        table(['Phase','Snapshot trước mốc','Có chapter quan sát','Không có chapter','Có COELO','Có ACELO',
               'ACELO=0','Max dòng problem','Mean COELO','Mean ACELO'],
              [[r[k] for k in ['phase','n_snapshots','with_observed_chapter','without_observed_chapter',
                              'with_coelo','with_acelo','acelo_zero','max_problem_rows','mean_coelo','mean_acelo']]
               for r in summary_rows]),
        'Mean chỉ tính trên chỉ số không NULL, không thay các cặp chưa quan sát bằng 0. '
        'COELO/ACELO user–course là trung bình các chỉ số chapter có giá trị; '
        'n_chapters_with_coelo/acelo được giữ trong Parquet để kiểm tra độ phủ.',
        'Một ví dụ có chapter quan sát ở mỗi phase trước mốc, chọn theo thứ tự user_id/course_id:',
        table(['User','Course','Phase','Cutoff','Chapter','Video segment','Problem row','COELO','ACELO'],
              [[r[k] for k in ['user_id','course_id','phase','cutoff_time','n_chapters_observed',
                              'n_video_segments','n_problem_rows','COELO_mean_observed_chapters',
                              'ACELO_mean_observed_chapters']] for r in one_example_per_phase]),
        '## 3. Cùng user–course qua các phase',
        'Chọn tối đa sáu course khác nhau có sự kiện ở P1 trước 01/05 và có problem ở P4 sau mốc; '
        'trong mỗi course chọn người có nhiều sự kiện P1 nhất, sau đó lấy sáu cặp đứng đầu. '
        'Đây là các ví dụ có chủ đích để đối chiếu, không phải mẫu đại diện và không chứng minh tác động nhân quả.',
        table(['User','Course','Phase','Cutoff','Trước mốc?','Chapter','Video segment','Problem row','COELO mean','ACELO mean'],
              [[r[k] for k in ['user_id','course_id','phase','cutoff_time','before_problem_tracking',
                              'n_chapters_observed','n_video_segments','n_problem_rows',
                              'COELO_mean_observed_chapters','ACELO_mean_observed_chapters']] for r in examples]),
        'Tập chapter quan sát có thể mở rộng giữa các phase. Vì vậy thay đổi mean qua phase có thể đến từ '
        'cả sự kiện mới và thay đổi tập chapter; không diễn giải mọi mức tăng/giảm là tác động riêng của tracking.',
        '### Đối chiếu giữ nguyên user–course–chapter',
        'Bảng sau giữ nguyên chapter để loại ảnh hưởng thay đổi tập chapter trong phép lấy trung bình. '
        'Số problem catalog là mẫu số tĩnh theo chính sách hiện tại; problem đã làm chỉ đếm dòng trước cutoff.',
        table(['User','Course','Chapter','Phase','Cutoff','Problem catalog','Problem đã làm',
               'AV','IF','KL','CFA','COELO','ACELO'],
              [[r[k] for k in ['user_id','course_id','lo_key','phase','cutoff_time',
                              'n_problems_available','n_problems_attempted','AV','IF_value','KL',
                              'CFA','COELO','ACELO']] for r in same_chapter_examples]),
        '## 4. Tách thành phần ở từng chapter trước mốc',
        table(['User','Course','Chapter','Phase','Video đã xem','Problem catalog','Problem đã làm',
               'TS','RV','AV','IF','KL','CFA','COELO','ACELO'],
              [[r[k] for k in ['user_id','course_id','lo_key','phase','n_videos_watched','n_problems_available',
                              'n_problems_attempted','TS','RV','AV','IF_value','KL','CFA','COELO','ACELO']]
               for r in chapter_examples]),
        'Các dòng trên có đủ bốn thành phần COELO. Do không có dòng problem, code hiện tại điền các tổng làm bài bằng 0: '
        'AV=0, IF=0; khi mẫu số hợp lệ, KL/CFA cũng bằng 0. Khi đó '
        'COELO=(TS+RV+0+0)/4 và ACELO=0. COELO không tự trở thành NULL chỉ vì nguồn problem chưa có log. '
        'Kết quả bằng 0 là hệ quả của công thức và quy tắc điền số đếm, chưa phải bằng chứng năng lực bằng 0.',
        '## 5. Giới hạn và hướng xử lý',
        'Tính đúng theo code không xác nhận tính hợp lệ của diễn giải. Ở phase hoàn toàn trước nguồn problem, '
        'ACELO cần được xem là chưa đo được thay vì kết luận học yếu. Trong tập kiểm chứng này, '
        'tất cả cặp có chapter quan sát trước mốc đều có COELO=ACELO=0 theo recipe hiện tại. '
        'Các cặp chưa có chapter quan sát vẫn giữ NULL, không bị xóa để làm đẹp ví dụ:',
        table(['User','Course','Phase','Cutoff','Chapter quan sát','Video segment','Problem row','COELO','ACELO'],
              [[r[k] for k in ['user_id','course_id','phase','cutoff_time','n_chapters_observed',
                              'n_video_segments','n_problem_rows','COELO_mean_observed_chapters',
                              'ACELO_mean_observed_chapters']] for r in unobserved_examples]),
        'Bước tiếp theo cần chốt chính sách độ phủ: giữ master, gắn trạng thái ngoài/giao một phần cửa sổ nguồn, '
        'và điều kiện dùng chỉ số/nhãn cho nghiên cứu. Việc đổi chính sách đó chưa được áp dụng trong kiểm chứng này.',
        '## 6. Dữ liệu và tái lập',
        '- [Bảng user–course–phase](tables/user_course_phase_indices.parquet)\n'
        '- [Bảng thành phần user–course–chapter–phase](tables/chapter_phase_indices.parquet)\n'
        '- [Các cặp ví dụ được chọn](tables/selected_user_courses.parquet)\n'
        '- [Thống kê và ví dụ JSON](summary.json)\n'
        '- [Kiểm tra tính nhất quán](validation.json)\n'
        '- [Mã tính kiểm chứng](../11_audit_problem_tracking_phase_indices.py)\n'
        '- [Công thức và hằng số nguồn](../final_quality_tables/protocol_constants.json)\n'
        '- [Từ điển feature hiện tại](../phase_features/tu_dien_features.md)',
        'Chạy từ thư mục working_13-10 bằng Python có DuckDB 1.3.2: '
        'python 11_audit_problem_tracking_phase_indices.py. '
        'Tất cả phép ghi chỉ nằm trong problem_tracking_phase_audit.',
    ]
    report = out/'bao_cao_kiem_chung.md'
    report.write_text('\n\n'.join(doc)+'\n')
    manifest = {
        'as_of':'2026-10-17','source_engine_sha256':sha(__file__),'source_files_sha256':hashes,
        'report_sha256':sha(report),'formula_source':'06_build_final_quality_tables.py',
        'phase_video_clip_source':'09_build_phase_features.py',
        'production_data_mutation':'none','source_problem_min_is_not_proof_of_platform_tracking_policy':True,
    }
    dump(out/'run_manifest.json',manifest)
    dump(out/'_SUCCESS.json',{'manifest_sha256':sha(out/'run_manifest.json'),'report_sha256':sha(report)})
    print('AUDIT_COMPLETE',len(checks),'checks',json.dumps(summary_rows,default=str),flush=True)


if __name__=='__main__':
    main()
