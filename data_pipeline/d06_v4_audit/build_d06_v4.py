"""Prepare D06 source-event evidence for 20 train/dev pairs, not human sign-off.

Raw IDs/event rows are written ONLY to a private directory outside this repo.
The aggregate report contains counts and hashes, never learner IDs/event rows.
"""
from __future__ import annotations

import argparse
from collections import Counter
import datetime as dt
import hashlib
import html
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
PHASES = ['P1', 'P2', 'P3', 'P4']
SEED = 20261023
DATE = '2026-10-23'


def qs(value):
    return "'" + str(value).replace("'", "''") + "'"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + '\n')


def records(con, sql):
    result = con.execute(sql)
    return [dict(zip([x[0] for x in result.description], row)) for row in result.fetchall()]


def sql_list(paths):
    return '[' + ','.join(qs(p) for p in paths) + ']'


def preserve_human_work(private):
    """Refuse to overwrite a dossier after someone edits it or returns a review."""
    if list(private.rglob('D06_HUMAN_REVIEW*.json')):
        raise ValueError('A returned human review exists; use a new private output directory')
    checksums = private/'SHA256SUMS.txt'
    if checksums.exists():
        for line in checksums.read_text().splitlines():
            expected, relative = line.split('  ',1)
            path = private/relative
            if not path.is_file() or digest(path)!=expected:
                raise ValueError('Private dossier was edited; preserve it and use a new output directory')


def make_html(private, pairs, comparisons, ledger):
    """Self-contained private viewer; local review export is a human action."""
    def display(value):
        return 'NULL' if value is None else str(value)

    by_pair = {p['enrollment_id']: [] for p in pairs}
    for item in comparisons:
        by_pair[item['enrollment_id']].append(item)
    by_row = {}
    for event in ledger:
        by_row.setdefault(event['row_id'], []).append(event)
    sections = []
    manual = ['# Phiếu kiểm tra thủ công D06 CQ V4', '',
              'Người kiểm tra: ____________________', 'Ngày kiểm tra: ____________________', '',
              'Đối chiếu event nguồn trong index.html/events.jsonl trước khi điền kết luận.',
              'MACHINE_PASS không thay kết luận của con người. Ô trống là CHƯA KIỂM.', '',
              '| Ca | Phase | Cutoff | Max event bắt đầu đã dùng | Max thời gian quan sát | Máy | Người kiểm tra | Ngày | Kết luận/mismatch | Hash hồ sơ ca |',
              '|---|---|---|---|---|---|---|---|---|---|']
    for pair in pairs:
        case_id = pair['case_id']
        subset = by_pair[pair['enrollment_id']]
        payload = {'pair': pair, 'snapshots': subset,
                   'events': [e for s in subset for e in by_row.get(s['row_id'], [])]}
        case_path = private / 'cases' / (case_id + '.json')
        dump(case_path, payload)
        case_hash = digest(case_path)
        cards = []
        for snapshot in subset:
            row_events = by_row.get(snapshot['row_id'], [])
            # Included events first; each group ordered from latest to earliest.
            row_events.sort(key=lambda e: (e['included'], str(e['event_time']), e['event_kind']), reverse=True)
            rows = []
            for e in row_events:
                fields = [e['event_kind'], e['event_id'], e['raw_timestamp'], e['event_time'],
                          e['observed_end_time'], e['disposition'], e['resource_id'], e['source_file']]
                rows.append('<tr class="' + ('included' if e['included'] else 'excluded') + '">' +
                            ''.join('<td>' + html.escape(str(v) if v is not None else 'NULL') + '</td>' for v in fields) + '</tr>')
            stats = (f"Cutoff: {snapshot['cutoff_time']} | Max start: {display(snapshot['expected_max_start'])} | "
                     f"Max observed: {display(snapshot['expected_max'])} | n_events: {snapshot['expected_n_events']} | "
                     f"Máy: {snapshot['machine_status']}")
            rid = snapshot['row_id']
            cards.append(f'''<article data-row="{rid}" data-case="{case_id}" data-phase="{snapshot['phase']}"
              data-evidence-hash="{case_hash}" data-cutoff="{snapshot['cutoff_time']}"
              data-max-start="{snapshot['expected_max_start'] or ''}" data-max-observed="{snapshot['expected_max'] or ''}"><h3>{snapshot['phase']}</h3>
              <p>{html.escape(stats)}</p><p>Cutoff tái tính: {snapshot['recomputed_cutoff']};
              margin tới cutoff (giây): {display(snapshot['margin_seconds'])};
              event dùng: {snapshot['included_source_rows']}; event nguồn bị loại: {snapshot['excluded_source_rows']}.</p>
              <details><summary>Xem toàn bộ event nguồn của phase ({len(row_events)} dòng)</summary>
              <p>Xanh: đã dùng; nền xám: bị loại. Video có thể cắt đuôi tại cutoff; xem max observed thay vì coi toàn segment được dùng.</p>
              <div class="scroll"><table><thead><tr><th>Loại</th><th>Event ID</th><th>Timestamp gốc</th><th>Event time</th>
              <th>Observed end</th><th>Phân loại</th><th>Resource</th><th>Source file</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div></details>
              <p><label><input type="checkbox" class="inspected"> Tôi đã đối chiếu event nguồn và cutoff của phase này</label></p>
              <label>Kết luận thủ công <select class="decision"><option value="">CHƯA KIỂM</option><option>PASS</option><option>FAIL</option><option>HOLD</option></select></label>
              <label>Ghi chú/mismatch <input class="notes" placeholder="Ghi event/điểm cần giải thích"></label></article>''')
            manual.append('| ' + ' | '.join(str(x) if x is not None else 'NULL' for x in [
                case_id, snapshot['phase'], snapshot['cutoff_time'], snapshot['expected_max_start'],
                snapshot['expected_max'], snapshot['machine_status'], '', '', '', case_hash]) + ' |')
        sections.append(f'''<section><h2>{case_id} · {pair['split']} · {pair['sample_origin']}</h2>
          <p>user_id: {html.escape(pair['user_id'])}; course_id: {html.escape(pair['course_id'])};
          enrollment_id: {pair['enrollment_id']}</p><p>Hash ca: <code>{case_hash}</code> ·
          <a href="cases/{case_id}.json">Nguồn JSON của ca</a></p>{''.join(cards)}</section>''')
    script = r'''
function exportReview() {
 const reviewer=document.querySelector('#reviewer').value.trim();
 const reviewDate=document.querySelector('#review-date').value;
 if(!reviewer || !reviewDate) { alert('Điền tên và ngày kiểm tra trước khi xuất biên bản.'); return; }
 const reviews=[...document.querySelectorAll('article')].map(a=>({case_id:a.dataset.case,phase:a.dataset.phase,row_id:a.dataset.row,
   evidence_sha256:a.dataset.evidenceHash,cutoff_time:a.dataset.cutoff,
   max_event_start:a.dataset.maxStart || null,max_observed_time:a.dataset.maxObserved || null,
   source_events_inspected:a.querySelector('.inspected').checked,
   decision:a.querySelector('.decision').value || 'NOT_REVIEWED',notes:a.querySelector('.notes').value}));
 if(reviews.some(r=>r.decision==='PASS' && !r.source_events_inspected)) {
   alert('Một phase được chọn PASS nhưng chưa xác nhận đã đối chiếu event nguồn.'); return;
 }
 const completed=reviews.every(r=>r.source_events_inspected && r.decision==='PASS');
 const body={dataset_version:'CQ V4',scope:'D06 human source-event review',manifest_sha256:DOSSIER_HASH,reviewer,review_date:reviewDate,
   status:completed?'HUMAN_PASS_80_SNAPSHOTS':'DRAFT_OR_UNRESOLVED',reviews};
 const blob=new Blob([JSON.stringify(body,null,2)],{type:'application/json'});
 const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='D06_HUMAN_REVIEW.json';a.click();
 setTimeout(()=>URL.revokeObjectURL(a.href),1000);
}
document.querySelector('#save').addEventListener('click',exportReview);
'''
    page = '''<!doctype html><html lang="vi"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
    <title>D06 CQ V4 · Hồ sơ kiểm tra nguồn riêng tư</title><style>
    body{font:15px/1.6 system-ui,sans-serif;max-width:1400px;margin:32px auto;padding:0 20px;color:#17243a;background:#f5f7fa}
    h1,h2,h3{line-height:1.3}section,header{background:white;padding:24px;margin:24px 0;border-radius:10px;border:1px solid #d9e1eb}
    article{padding:16px 0;border-top:1px solid #d9e1eb}table{border-collapse:collapse;width:100%;font-size:12px}
    td,th{border:1px solid #d9e1eb;padding:8px;text-align:left;white-space:nowrap}th{background:#e7edf5;position:sticky;top:0}
    .included{background:#eaf7f1}.excluded{background:#f3f4f6;color:#586475}.scroll{overflow:auto;max-height:420px}
    input,select,button{font:inherit;padding:7px;margin:5px}input.notes{width:min(600px,70%)}code{overflow-wrap:anywhere}
    .notice{border-left:4px solid #b87816;padding:12px;background:#fff4da}button{cursor:pointer;background:#163f70;color:white;border:0;border-radius:6px}
    @media print{.scroll{max-height:none;overflow:visible}button{display:none}section{break-before:page}td,th{white-space:normal}}
    </style><header><h1>D06 CQ V4: 20 ca train/dev × 4 phase</h1>
    <p class="notice">Hồ sơ có định danh và event thật: chỉ chia sẻ riêng. Kiểm tra máy không phải kiểm tra thủ công.
    Tên, ngày, checkbox và kết luận dưới đây đều chưa được điền tự động. Không có feature/nhãn test.</p>
    <p>Giữ 18 ca train/dev của mẫu cũ; thêm 2 ca bằng Python random.Random(20261023).sample trên danh sách enrollment_id đã sort.</p>
    <p>UTC theo builder. Max start là timestamp bắt đầu event; max observed gồm phần video đã clip tại cutoff.
    n_events không cộng thêm reply received, nhưng min/max bằng chứng có xét received như audit gốc.</p>
    <p><a href="MANIFEST.json">Manifest/hash nguồn</a> · <a href="comparison.json">80 snapshot đối chiếu</a> ·
    <a href="events.jsonl">Toàn bộ ledger nguồn</a> · <a href="PHIEU_KIEM_TRA_THU_CONG.md">Phiếu Markdown</a></p>
    <label>Người kiểm tra <input id="reviewer"></label><label>Ngày kiểm tra <input id="review-date" type="date"></label>
    <button id="save">Xuất biên bản đã điền / bản nháp</button><p>Lựa chọn chỉ ở trang hiện tại; xuất JSON trước khi đóng trang.</p></header>'''
    page += ''.join(sections) + '<script>const DOSSIER_HASH=' + json.dumps(digest(private/'MANIFEST.json')) + ';\n' + script + '</script></html>'
    (private/'index.html').write_text(page)
    (private/'PHIEU_KIEM_TRA_THU_CONG.md').write_text('\n'.join(manual) + '\n')


def build(private_dir):
    import duckdb
    private = Path(private_dir).resolve()
    if private == ROOT or ROOT in private.parents:
        raise ValueError('Private event evidence must be outside the code repository')
    preserve_human_work(private)
    private.mkdir(parents=True, exist_ok=True)
    private.chmod(0o700)
    (private/'cases').mkdir(exist_ok=True)
    con = duckdb.connect()
    con.execute("SET threads=2")
    con.execute("SET memory_limit='1200MB'")
    con.execute("SET TimeZone='UTC'")
    checks = {}

    def check(name, ok):
        checks[name] = bool(ok)

    def view(name, relative):
        con.execute(f'CREATE TEMP VIEW {name} AS SELECT * FROM read_parquet({qs(ROOT/relative)},filename=true)')

    old_path = ROOT/'phase_features_v4/audit/cutoff_sample_20_pairs.parquet'
    old = records(con, f'''SELECT DISTINCT enrollment_id,user_id,course_id,split
        FROM read_parquet({qs(old_path)}) WHERE split IN ('train','dev') ORDER BY enrollment_id''')
    check('old_train_dev_pairs_18', len(old) == 18)
    train_dev_p1 = [ROOT/'phase_features_v4/supervised_raw/P1'/f'{split}.parquet' for split in ['train','dev']]
    con.execute(f'''CREATE TEMP TABLE eligible AS SELECT enrollment_id,user_id,course_id,split FROM
        read_parquet({sql_list(train_dev_p1)})''')
    eligible = records(con, 'SELECT * FROM eligible ORDER BY enrollment_id')
    old_ids = {r['enrollment_id'] for r in old}
    candidates = [r for r in eligible if r['enrollment_id'] not in old_ids]
    added = random.Random(SEED).sample(candidates, 2)
    chosen = [dict(r, sample_origin='retained_old_train_dev') for r in old]
    chosen += [dict(r, sample_origin='added_seeded_sample') for r in added]
    chosen.sort(key=lambda r: r['enrollment_id'])
    for i, pair in enumerate(chosen, 1):
        pair['case_id'] = f'D06-{i:02d}'
    check('selected_pairs_20_unique', len({r['enrollment_id'] for r in chosen}) == 20)
    check('only_train_dev_selected', all(r['split'] in ['train','dev'] for r in chosen))
    check('all_old_18_retained', old_ids <= {r['enrollment_id'] for r in chosen})
    con.execute('''CREATE TEMP TABLE selected(enrollment_id VARCHAR,user_id VARCHAR,course_id VARCHAR,
        split VARCHAR,sample_origin VARCHAR,case_id VARCHAR)''')
    con.executemany('INSERT INTO selected VALUES (?,?,?,?,?,?)', [tuple(r[k] for k in
        ['enrollment_id','user_id','course_id','split','sample_origin','case_id']) for r in chosen])
    snapshot_paths = [ROOT/'phase_features_v4/supervised_raw'/phase/(split+'.parquet')
                      for phase in PHASES for split in ['train','dev']]
    con.execute(f'''CREATE TEMP TABLE snapshots AS SELECT f.row_id,f.enrollment_id,f.user_id,f.course_id,
        f.course_family_id,f.phase,f.split,f.enroll_time,f.cutoff_time,f.course_end,f.n_events,
        s.sample_origin,s.case_id FROM read_parquet({sql_list(snapshot_paths)}) f
        JOIN selected s USING(enrollment_id)''')
    check('snapshots_80', con.execute('SELECT count(*)=80 AND count(DISTINCT row_id)=80 FROM snapshots').fetchone()[0])
    check('four_phases_each', con.execute('SELECT count(*)=0 FROM (SELECT enrollment_id FROM snapshots GROUP BY 1 HAVING count(*)<>4 OR count(DISTINCT phase)<>4 OR count(DISTINCT split)<>1)').fetchone()[0])
    print('Selected 18 retained + 2 added train/dev pairs; 80 snapshots', flush=True)

    for name, relative in {
        'resources':'input/moocubex_model/processed/dim_lo_resource/*.parquet',
        'lo_problems':'input/moocubex_model/processed/bridge_lo_problem/*.parquet',
        'course_comments':'input/moocubex_model/processed/bridge_course_comment/*.parquet',
        'raw_video':'input/moocubex_parquet/tables/relations/user_video/*.parquet',
        'raw_problem':'input/moocubex_parquet/tables/relations/user_problem/*.parquet',
        'raw_comment':'input/moocubex_parquet/tables/entities/translated_comment/*.parquet',
        'raw_reply':'input/moocubex_parquet/tables/entities/reply/*.parquet',
        'comment_reply':'input/moocubex_parquet/tables/relations/comment_reply/*.parquet',
    }.items():
        view(name, relative)
    con.execute('''CREATE TEMP TABLE video_map AS SELECT resource_id video_id,min(course_id) course_id,min(lo_key) lo_key
        FROM resources WHERE resource_type='video' GROUP BY 1 HAVING count(DISTINCT lo_key)=1''')
    con.execute('''CREATE TEMP TABLE problem_map AS SELECT problem_id,min(course_id) course_id,min(lo_key) lo_key,
        min(exercise_id) exercise_id FROM lo_problems GROUP BY 1 HAVING count(DISTINCT lo_key)=1''')
    con.execute('''CREATE TEMP TABLE social_map AS SELECT resource_id,min(course_id) course_id,
        CASE WHEN count(DISTINCT lo_key)=1 THEN min(lo_key) END lo_key
        FROM resources GROUP BY 1 HAVING count(DISTINCT course_id)=1''')
    con.execute('''CREATE TEMP TABLE prepared_comments AS WITH typed AS (
        SELECT c.id comment_id,'U_'||cast(c.user_id AS VARCHAR) user_id,c.resource_id,c.text,
        c.create_time raw_epoch_ms,c.filename source_file,
        cast(to_timestamp(c.create_time/1000.0) AS TIMESTAMP) event_time,
        coalesce(r.course_id,cc.course_id) course_id,
        coalesce(r.lo_key,coalesce(r.course_id,cc.course_id)||':CH-1') lo_key,
        nullif(regexp_replace(trim(c.text),'\\s+',' ','g'),'') normalized_text
        FROM raw_comment c LEFT JOIN social_map r USING(resource_id)
        LEFT JOIN course_comments cc ON c.id=cc.comment_id
        WHERE c.user_id IS NOT NULL AND c.create_time BETWEEN 0 AND 4102444800000
        AND coalesce(r.course_id,cc.course_id) IS NOT NULL
        AND (r.course_id IS NULL OR cc.course_id IS NULL OR r.course_id=cc.course_id)),
        ids AS(SELECT *,row_number() OVER(PARTITION BY comment_id ORDER BY event_time,user_id,resource_id,text) rn FROM typed)
        SELECT * EXCLUDE(rn),row_number() OVER(PARTITION BY user_id,resource_id,event_time,normalized_text ORDER BY comment_id) exact_duplicate_rank
        FROM ids WHERE rn=1''')
    con.execute('''CREATE TEMP TABLE parent_map AS SELECT reply_id,min(comment_id) parent_comment_id
        FROM comment_reply GROUP BY 1 HAVING count(DISTINCT comment_id)=1''')
    con.execute('''CREATE TEMP TABLE prepared_replies AS WITH typed AS (
        SELECT r.id reply_id,r.user_id,r.create_time raw_time,r.filename source_file,
        try_cast(r.create_time AS TIMESTAMP) event_time,p.parent_comment_id,c.event_time parent_time,
        c.course_id,c.lo_key,c.resource_id,
        row_number() OVER(PARTITION BY r.id ORDER BY r.create_time,r.user_id,r.text) rn
        FROM raw_reply r JOIN parent_map p ON r.id=p.reply_id JOIN prepared_comments c ON p.parent_comment_id=c.comment_id)
        SELECT * EXCLUDE(rn) FROM typed WHERE rn=1 AND user_id IS NOT NULL AND event_time>=parent_time''')
    con.execute('''CREATE TEMP TABLE source_events(user_id VARCHAR,course_id VARCHAR,lo_key VARCHAR,event_kind VARCHAR,
        event_id VARCHAR,resource_id VARCHAR,event_time TIMESTAMP,raw_timestamp VARCHAR,source_file VARCHAR,
        valid_video BOOLEAN,start_point DOUBLE,end_point DOUBLE,speed DOUBLE,source_window_ok BOOLEAN,
        parent_comment_id VARCHAR,parent_time TIMESTAMP,parent_comment_valid BOOLEAN)''')
    con.execute('''INSERT INTO source_events WITH selected_users AS (SELECT DISTINCT user_id FROM selected),
        items AS(SELECT u.user_id,u.filename,unnest(u.seq) item,generate_subscripts(u.seq,1) item_index
        FROM raw_video u JOIN selected_users s USING(user_id)),
        mapped AS(SELECT i.*,m.course_id,m.lo_key FROM items i JOIN video_map m ON i.item.video_id=m.video_id
        JOIN selected s ON i.user_id=s.user_id AND m.course_id=s.course_id),
        segments AS(SELECT user_id,filename,course_id,lo_key,item.video_id video_id,item_index,
        unnest(item.segment) seg,generate_subscripts(item.segment,1) segment_index FROM mapped)
        SELECT user_id,course_id,lo_key,'video',
        sha256(filename||':'||user_id||':'||item_index::VARCHAR||':'||segment_index::VARCHAR),video_id,
        CASE WHEN seg.local_start_time BETWEEN 0 AND 4102444800 THEN cast(to_timestamp(seg.local_start_time) AS TIMESTAMP) END,
        cast(seg.local_start_time AS VARCHAR),filename,
        coalesce(seg.start_point>=0 AND seg.end_point>seg.start_point AND isfinite(seg.start_point)
        AND isfinite(seg.end_point) AND seg.speed>0 AND isfinite(seg.speed),false),
        seg.start_point,seg.end_point,seg.speed,
        seg.local_start_time>=epoch(TIMESTAMP '2020-01-19') AND seg.local_start_time<epoch(TIMESTAMP '2020-11-04'),
        NULL,NULL,NULL FROM segments''')
    con.execute('''INSERT INTO source_events SELECT p.user_id,m.course_id,m.lo_key,'problem',p.log_id,m.exercise_id,
        try_cast(p.submit_time AS TIMESTAMP),p.submit_time,p.filename,NULL,NULL,NULL,NULL,true,NULL,NULL,NULL
        FROM raw_problem p JOIN (SELECT DISTINCT user_id FROM selected) u USING(user_id)
        JOIN problem_map m USING(problem_id) JOIN selected s ON p.user_id=s.user_id AND m.course_id=s.course_id''')
    con.execute('''INSERT INTO source_events SELECT c.user_id,c.course_id,c.lo_key,'comment',c.comment_id,c.resource_id,
        c.event_time,c.raw_epoch_ms::VARCHAR,c.source_file,NULL,NULL,NULL,NULL,true,NULL,NULL,
        c.normalized_text IS NOT NULL AND c.exact_duplicate_rank=1
        FROM prepared_comments c JOIN selected s USING(user_id,course_id)''')
    con.execute('''INSERT INTO source_events SELECT r.user_id,r.course_id,r.lo_key,'reply',r.reply_id,r.resource_id,
        r.event_time,r.raw_time,r.source_file,NULL,NULL,NULL,NULL,true,r.parent_comment_id,r.parent_time,true
        FROM prepared_replies r JOIN selected s USING(user_id,course_id)''')
    con.execute('''INSERT INTO source_events SELECT c.user_id,c.course_id,c.lo_key,'received',r.reply_id,c.resource_id,
        r.event_time,r.raw_time,r.source_file,NULL,NULL,NULL,NULL,true,c.comment_id,c.event_time,
        c.normalized_text IS NOT NULL AND c.exact_duplicate_rank=1
        FROM prepared_comments c JOIN selected s USING(user_id,course_id)
        JOIN prepared_replies r ON r.parent_comment_id=c.comment_id''')
    print('Reconstructed selected-pair video/problem/comment/reply events from input', flush=True)
    con.execute('''CREATE TEMP TABLE event_ledger AS WITH classified AS (
        SELECT s.case_id,s.enrollment_id,s.row_id,s.phase,s.split,s.enroll_time,s.cutoff_time,e.*,
        CASE WHEN e.event_time IS NULL THEN 'INVALID_TIME'
        WHEN NOT coalesce(e.source_window_ok,false) THEN 'OUTSIDE_VIDEO_SOURCE_WINDOW'
        WHEN e.event_time<s.enroll_time THEN 'BEFORE_ENROLLMENT'
        WHEN e.event_time>s.cutoff_time THEN 'AFTER_CUTOFF'
        WHEN e.event_kind='received' AND NOT coalesce(e.parent_comment_valid,false) THEN 'PARENT_COMMENT_FILTERED'
        WHEN e.event_kind='received' AND e.parent_time<s.enroll_time THEN 'PARENT_BEFORE_ENROLLMENT'
        WHEN e.event_kind='received' AND e.parent_time>s.cutoff_time THEN 'PARENT_AFTER_CUTOFF'
        ELSE 'INCLUDED' END disposition
        FROM snapshots s JOIN source_events e USING(user_id,course_id))
        SELECT *,disposition='INCLUDED' included,
        CASE WHEN disposition='INCLUDED' AND event_kind='video' AND valid_video
        THEN event_time+(least(end_point-start_point,greatest(0.0,epoch(cutoff_time)-epoch(event_time))*speed)/speed)*INTERVAL '1 second'
        WHEN disposition='INCLUDED' THEN event_time END observed_end_time,
        CASE WHEN disposition='INCLUDED' AND event_kind='video' AND valid_video
        THEN least(end_point-start_point,greatest(0.0,epoch(cutoff_time)-epoch(event_time))*speed)
        WHEN disposition='INCLUDED' AND event_kind='video' THEN 0 END observed_media_seconds
        FROM classified''')
    audit_paths = [ROOT/'phase_features_v4/audit'/phase/'part-000.parquet' for phase in PHASES]
    con.execute(f'''CREATE TEMP VIEW saved_audit AS SELECT a.* FROM read_parquet({sql_list(audit_paths)}) a
        JOIN snapshots s USING(row_id)''')
    con.execute('''CREATE TEMP TABLE comparison AS WITH expected AS (
        SELECT row_id,min(event_time) FILTER(WHERE included) expected_min,
        max(event_time) FILTER(WHERE included) expected_max_start,
        max(observed_end_time) expected_max,
        count(*) FILTER(WHERE included AND event_kind<>'received') expected_n_events,
        count(*) FILTER(WHERE included) included_source_rows,count(*) FILTER(WHERE NOT included) excluded_source_rows,
        count(*) FILTER(WHERE included AND observed_end_time>cutoff_time+INTERVAL '1 microsecond') out_of_window
        FROM event_ledger GROUP BY 1), recomputed AS (
        SELECT s.*,make_timestamp(epoch_us(enroll_time)+(date_diff('microsecond',enroll_time,course_end)*
        CASE phase WHEN 'P1' THEN 1 WHEN 'P2' THEN 1 WHEN 'P3' THEN 3 WHEN 'P4' THEN 9 END)//
        CASE phase WHEN 'P1' THEN 4 WHEN 'P2' THEN 2 WHEN 'P3' THEN 4 WHEN 'P4' THEN 10 END) recomputed_cutoff
        FROM snapshots s)
        SELECT s.*,a.min_feature_event_time saved_min,a.max_feature_event_time saved_max,
        a.n_out_of_window_events saved_n_out_of_window,a.row_id IS NOT NULL saved_audit_present,
        e.expected_min,e.expected_max_start,e.expected_max,coalesce(e.expected_n_events,0) expected_n_events,
        coalesce(e.included_source_rows,0) included_source_rows,coalesce(e.excluded_source_rows,0) excluded_source_rows,
        coalesce(e.out_of_window,0) out_of_window,
        epoch(s.cutoff_time)-epoch(e.expected_max) margin_seconds,
        s.cutoff_time=s.recomputed_cutoff AND s.n_events=coalesce(e.expected_n_events,0)
        AND a.min_feature_event_time IS NOT DISTINCT FROM e.expected_min
        AND a.max_feature_event_time IS NOT DISTINCT FROM e.expected_max
        AND coalesce(a.n_out_of_window_events,0)=0 AND coalesce(e.out_of_window,0)=0
        AND (coalesce(e.included_source_rows,0)=0 OR a.row_id IS NOT NULL) machine_pass
        FROM recomputed s LEFT JOIN saved_audit a USING(row_id) LEFT JOIN expected e USING(row_id)''')
    comparisons = records(con, 'SELECT *,CASE WHEN machine_pass THEN \'MACHINE_PASS\' ELSE \'MACHINE_FAIL\' END machine_status FROM comparison ORDER BY case_id,phase')
    check('cutoff_formula_80', all(s['cutoff_time'] == s['recomputed_cutoff'] for s in comparisons))
    check('source_vs_saved_all_80', all(s['machine_pass'] for s in comparisons))
    check('included_events_within_cutoff', all(s['out_of_window'] == 0 for s in comparisons))
    source_counts = records(con, 'SELECT event_kind,count(*) n_source_rows FROM source_events GROUP BY 1 ORDER BY 1')
    dispositions = records(con, 'SELECT disposition,count(*) n_snapshot_event_rows FROM event_ledger GROUP BY 1 ORDER BY 1')
    coverage = records(con, '''SELECT
        count(*) FILTER(WHERE included AND event_kind='video' AND valid_video AND
            observed_media_seconds<end_point-start_point-1e-9) clipped_video_snapshot_events,
        count(*) FILTER(WHERE included AND event_kind='received') received_snapshot_events,
        count(*) FILTER(WHERE included AND event_kind='reply') authored_reply_snapshot_events,
        count(*) FILTER(WHERE included AND event_kind='video' AND NOT valid_video) invalid_video_snapshot_events,
        count(*) FILTER(WHERE included AND lo_key LIKE '%:CH-1') unknown_chapter_snapshot_events
        FROM event_ledger''')[0]
    coverage['empty_snapshots'] = sum(s['included_source_rows']==0 for s in comparisons)
    check('has_negative_control_future_events', any(r['disposition'] == 'AFTER_CUTOFF' and r['n_snapshot_event_rows']>0 for r in dispositions))
    ledger = records(con, 'SELECT * FROM event_ledger ORDER BY case_id,phase,event_time,event_kind,event_id')
    for e in ledger:
        e['source_file'] = str(Path(e['source_file']).relative_to(ROOT))
    source_rows = records(con, 'SELECT * FROM source_events ORDER BY user_id,course_id,event_time,event_kind,event_id')
    for e in source_rows:
        e['source_file'] = str(Path(e['source_file']).relative_to(ROOT))
    dump(private/'selection.json', {'seed': SEED,'algorithm': 'random.Random(seed).sample(sorted eligible train/dev excluding retained old 18, 2)',
         'eligible_train_dev_pairs':len(eligible),'candidate_pairs_after_exclusion':len(candidates),'pairs':chosen})
    dump(private/'comparison.json', comparisons)
    for filename, items in [('events.jsonl',ledger), ('source_events.jsonl',source_rows)]:
        with (private/filename).open('w') as stream:
            for item in items:
                stream.write(json.dumps(item, ensure_ascii=False, default=str) + '\n')
    source_files = set()
    for relative in [
        'input/moocubex_model/processed/dim_lo_resource/*.parquet',
        'input/moocubex_model/processed/bridge_lo_problem/*.parquet',
        'input/moocubex_model/processed/bridge_course_comment/*.parquet',
        'input/moocubex_parquet/tables/relations/user_video/*.parquet',
        'input/moocubex_parquet/tables/relations/user_problem/*.parquet',
        'input/moocubex_parquet/tables/entities/translated_comment/*.parquet',
        'input/moocubex_parquet/tables/entities/reply/*.parquet',
        'input/moocubex_parquet/tables/relations/comment_reply/*.parquet',
    ]:
        source_files.update(ROOT.glob(relative))
    source_files.update(snapshot_paths+audit_paths+[old_path,Path(__file__),ROOT/'final_quality_v4/splits.json',
        ROOT/'phase_features_v4/run_manifest.json',ROOT/'22_build_phase_features_v4.py',ROOT/'16_build_phase_features_v2.py'])
    print('Hashing input/source files and creating the private reviewer dossier', flush=True)
    source_hashes = [{'path':str(p.relative_to(ROOT)), 'bytes':p.stat().st_size,'sha256':digest(p)} for p in sorted(source_files)]
    dump(private/'MANIFEST.json', {
        'dataset_version':'CQ V4','date':DATE,'status':'MACHINE_CHECKED_AWAITING_HUMAN_REVIEW',
        'reviewer':None,'review_date':None,'human_reviewed_pairs':0,
        'seed':SEED,'python':sys.version.split()[0],'duckdb':duckdb.__version__,
        'selected_pairs':20,'snapshots':80,'test_feature_or_label_rows_used':0,
        'sampling_note':'18 original train/dev pairs retained; 2 added from train/dev only. Old test rows not used.',
        'time_policy':'UTC per original builders; cutoff inclusive; video observed tail clipped; no timezone offset correction',
        'raw_input_provenance':'Raw input event tables mapped through dim/bridge, not the old aggregate comparison.',
        'scope_limit':'Machine audit supports the manual source-event check. It is not a human signature or full feature/label validation.',
        'checks':checks,'source_counts':source_counts,'dispositions':dispositions,
        'sample_coverage':coverage,'source_files':source_hashes})
    make_html(private, chosen, comparisons, ledger)
    files = sorted(p for p in private.rglob('*') if p.is_file() and p.name!='SHA256SUMS.txt')
    (private/'SHA256SUMS.txt').write_text(''.join(f'{digest(p)}  {p.relative_to(private)}\n' for p in files))
    for path in files+[private/'SHA256SUMS.txt']:
        path.chmod(0o600)
    aggregate = {
        'dataset_version':'CQ V4','date':DATE,'status':'MACHINE_CHECKED_AWAITING_HUMAN_REVIEW',
        'selected_pairs':20,'retained_old_train_dev_pairs':18,'added_train_dev_pairs':2,'snapshots':80,
        'pair_split_counts':dict(Counter(p['split'] for p in chosen)),
        'seed':SEED,'sampling_algorithm':'Python random.Random(seed).sample over sorted enrollment_id candidates',
        'checks':checks,'machine_pass_snapshots':sum(s['machine_pass'] for s in comparisons),
        'machine_fail_snapshots':sum(not s['machine_pass'] for s in comparisons),
        'human_reviewed_pairs':0,'human_reviewed_snapshots':0,'reviewer':None,'review_date':None,
        'test_feature_or_label_rows_used':0,'source_counts':source_counts,'dispositions':dispositions,
        'sample_coverage':coverage,
        'private_manifest_sha256':digest(private/'MANIFEST.json'),
        'private_checksums_sha256':digest(private/'SHA256SUMS.txt'),
        'script_sha256':digest(__file__),
        'python':sys.version.split()[0],'duckdb':duckdb.__version__,
        'source_files_hashed':len(source_hashes),
        'source_row_data_in_aggregate':False,
    }
    dump(Path(__file__).parent/'summary.json',aggregate)
    con.close()
    print(json.dumps(aggregate, ensure_ascii=False))
    if not all(checks.values()):
        raise RuntimeError('D06 machine checks have unresolved failures; inspect private comparison.json')
    return aggregate


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--private-output', default='/private/tmp/d06_cq_v4_2026_10_23')
    args = parser.parse_args()
    build(args.private_output)
