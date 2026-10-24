"""Build Review section 2.5 cutoffs from notebook 01/02 inputs.

One observation is (user_id, course_id), with no synthetic enrollment/root ID.
Only time eligibility is decided here; quality labels and features are not built.
The companion notebook embeds this module and needs only DuckDB.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import time
from pathlib import Path

import duckdb

PHASES = {'P1': (1, 4), 'P2': (1, 2), 'P3': (3, 4), 'P4': (9, 10)}
VIDEO_START = '2020-01-19 00:00:00'
VIDEO_END_EXCLUSIVE = '2020-11-04 00:00:00'


def qs(value):
    return "'" + str(value).replace("'", "''") + "'"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def discover_inputs():
    for parent in [Path.cwd(), Path.cwd() / 'working_13-10']:
        root = parent / 'input'
        if (root / 'moocubex_parquet/catalog.json').exists() and (root / 'moocubex_model/catalog.json').exists():
            return root / 'moocubex_parquet', root / 'moocubex_model'
    raw, model = [], []
    for parent in [Path('/kaggle/working'), Path('/kaggle/input')]:
        if parent.is_dir():
            for catalog in parent.rglob('catalog.json'):
                root = catalog.parent
                if (root / 'tables/relations/user_video').is_dir():
                    raw.append(root)
                if (root / 'processed/bridge_user_course_enrollment').is_dir():
                    model.append(root)
    if len(raw) == len(model) == 1:
        return raw[0], model[0]
    raise RuntimeError('Set raw_root and model_root explicitly; inputs are not unique.')


def cutoff_sql(enroll_expr, end_expr, phase):
    """Exact rational arithmetic; truncate sub-microsecond fractions downward."""
    numerator, denominator = PHASES[phase]
    return (f'make_timestamp(epoch_us({enroll_expr}) + '
            f"(date_diff('microsecond', {enroll_expr}, {end_expr}) * {numerator}) // {denominator})")


def time_status_sql():
    return """CASE
        WHEN user_id IS NULL OR trim(user_id)='' OR course_id IS NULL OR trim(course_id)=''
            THEN 'INVALID_KEY'
        WHEN NOT course_exists_in_catalog THEN 'COURSE_NOT_IN_CATALOG'
        WHEN enroll_time IS NULL THEN 'INVALID_ENROLL_TIME'
        WHEN course_end IS NULL THEN 'NO_COURSE_ACTIVITY'
        WHEN course_start IS NULL OR course_end<course_start THEN 'INVALID_COURSE_BOUNDS'
        WHEN enroll_time>course_end THEN 'ENROLLMENT_AFTER_COURSE_END'
        WHEN enroll_time=course_end THEN 'ENROLLMENT_AT_COURSE_END'
        ELSE 'ELIGIBLE_TIME' END"""


def validate_examples(con):
    """Test different enrollments sharing an end, early registration and edge spans."""
    con.execute("""CREATE TEMP TABLE example_inputs AS
        SELECT * FROM (VALUES
          ('A','C',TIMESTAMP '2020-05-01',TIMESTAMP '2020-05-02',TIMESTAMP '2020-05-21',true),
          ('B','C',TIMESTAMP '2020-05-11',TIMESTAMP '2020-05-02',TIMESTAMP '2020-05-21',true),
          ('AT_END','C',TIMESTAMP '2020-05-21',TIMESTAMP '2020-05-02',TIMESTAMP '2020-05-21',true),
          ('AFTER_END','C',TIMESTAMP '2020-05-22',TIMESTAMP '2020-05-02',TIMESTAMP '2020-05-21',true),
          ('NO_END','C',TIMESTAMP '2020-05-01',NULL,NULL,true),
          ('NO_CATALOG','X',TIMESTAMP '2020-05-01',NULL,NULL,false),
          ('NO_ENROLL','C',NULL,TIMESTAMP '2020-05-02',TIMESTAMP '2020-05-21',true),
          ('SHORT','S',TIMESTAMP '2020-05-01',TIMESTAMP '2020-05-01',TIMESTAMP '2020-05-01 00:00:01',true)
        ) t(user_id,course_id,enroll_time,course_start,course_end,course_exists_in_catalog)""")
    fields = ','.join(f"CASE WHEN time_status='ELIGIBLE_TIME' THEN {cutoff_sql('enroll_time', 'course_end', p)} END cutoff_{p}"
                      for p in PHASES)
    cur = con.execute(f'SELECT *, {fields} FROM (SELECT *, {time_status_sql()} time_status FROM example_inputs) ORDER BY user_id')
    rows = [dict(zip([col[0] for col in cur.description], row)) for row in cur.fetchall()]
    by_user = {r['user_id']: r for r in rows}
    expected = {
        'A': ['2020-05-06 00:00:00', '2020-05-11 00:00:00', '2020-05-16 00:00:00', '2020-05-19 00:00:00'],
        'B': ['2020-05-13 12:00:00', '2020-05-16 00:00:00', '2020-05-18 12:00:00', '2020-05-20 00:00:00'],
        'SHORT': ['2020-05-01 00:00:00.250000', '2020-05-01 00:00:00.500000',
                  '2020-05-01 00:00:00.750000', '2020-05-01 00:00:00.900000'],
    }
    for user, values in expected.items():
        assert [by_user[user][f'cutoff_{p}'] for p in PHASES] == [dt.datetime.fromisoformat(v) for v in values]
    for user, status in [('AT_END', 'ENROLLMENT_AT_COURSE_END'), ('AFTER_END', 'ENROLLMENT_AFTER_COURSE_END'),
                         ('NO_END', 'NO_COURSE_ACTIVITY'), ('NO_CATALOG', 'COURSE_NOT_IN_CATALOG'),
                         ('NO_ENROLL', 'INVALID_ENROLL_TIME')]:
        assert by_user[user]['time_status'] == status
        assert all(by_user[user][f'cutoff_{p}'] is None for p in PHASES)
    # Independent Python implementation, including a non-divisible microsecond span.
    for micros in [1, 3, 10, 999999, 86400000001]:
        start = dt.datetime(2020, 5, 1)
        end = start + dt.timedelta(microseconds=micros)
        for phase, (num, den) in PHASES.items():
            value = con.execute(f"SELECT {cutoff_sql('CAST(? AS TIMESTAMP)', 'CAST(? AS TIMESTAMP)', phase)}",
                                [start, start, end]).fetchone()[0]
            assert value == start + dt.timedelta(microseconds=micros * num // den)
    return {'passed': True, 'cases': rows, 'python_cross_checks': 20,
            'sub_microsecond_policy': 'floor to microsecond; do not round to a day'}


def run_build(raw_root=None, model_root=None, output_root=None, memory_limit='2GB', threads=2):
    if raw_root is None or model_root is None:
        default_raw, default_model = discover_inputs()
        raw_root, model_root = raw_root or default_raw, model_root or default_model
    raw, model = Path(raw_root).resolve(), Path(model_root).resolve()
    out = Path(output_root or ('/kaggle/working/review_time_phases' if Path('/kaggle').is_dir()
                              else raw.parent.parent / 'review_time_phases')).resolve()
    for root in [raw, model]:
        if out == root or root in out.parents or out in root.parents:
            raise ValueError('Output must be separate from raw/model input directories.')
    out.mkdir(parents=True, exist_ok=True)
    tables = out / 'tables'; tables.mkdir(exist_ok=True)
    phase_dir = out / 'phase_cutoffs'; phase_dir.mkdir(exist_ok=True)
    scratch = out / '_scratch'; scratch.mkdir(exist_ok=True)
    success = out / '_SUCCESS.json'
    if success.exists():
        success.unlink()
    con = duckdb.connect()
    con.execute(f'SET threads={int(threads)}')
    con.execute(f'SET memory_limit={qs(memory_limit)}')
    con.execute(f'SET temp_directory={qs(scratch)}')
    con.execute("SET max_temp_directory_size='4GB'")
    con.execute("SET TimeZone='UTC'")
    con.execute('SET preserve_insertion_order=false')
    started = time.monotonic()
    queries, inventory, outputs = [], [], []

    def view(name, root, relative):
        files = sorted((root / relative).glob('*.parquet'))
        if not files:
            raise FileNotFoundError(root / relative)
        inventory.extend({'root': 'raw' if root == raw else 'model', 'path': str(f.relative_to(root)),
                          'bytes': f.stat().st_size, 'mtime_ns': f.stat().st_mtime_ns} for f in files)
        con.execute(f'CREATE VIEW {name} AS SELECT * FROM read_parquet({qs(root / relative / "*.parquet")})')

    def query(name, sql):
        tick = time.monotonic(); con.execute(sql)
        queries.append({'name': name, 'elapsed_seconds': round(time.monotonic() - tick, 3), 'sql': sql})
        print(name, f'{queries[-1]["elapsed_seconds"]:.2f}s', flush=True)

    def export(name, sql, directory=tables):
        target = directory / f'{name}.parquet'
        partial = target.with_suffix('.parquet.partial')
        query(name, f'COPY ({sql}) TO {qs(partial)} (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 122880)')
        partial.replace(target)
        outputs.append(target)
        return target

    def rows(sql):
        cursor = con.execute(sql)
        return [dict(zip([column[0] for column in cursor.description], row)) for row in cursor.fetchall()]

    view('courses', model, 'processed/dim_course')
    view('enroll', model, 'processed/bridge_user_course_enrollment')
    view('resources', model, 'processed/dim_lo_resource')
    view('lo_problem', model, 'processed/bridge_lo_problem')
    view('course_comment', model, 'processed/bridge_course_comment')
    view('raw_video', raw, 'tables/relations/user_video')
    view('raw_problem', raw, 'tables/relations/user_problem')
    view('comments', raw, 'tables/entities/translated_comment')
    view('replies', raw, 'tables/entities/reply')
    view('comment_reply', raw, 'tables/relations/comment_reply')
    fixture_results = validate_examples(con)

    query('map_video', """CREATE TEMP TABLE map_video AS
        SELECT resource_id video_id,min(course_id) course_id FROM resources
        WHERE resource_type='video' GROUP BY 1 HAVING count(DISTINCT course_id)=1""")
    query('map_problem', """CREATE TEMP TABLE map_problem AS
        SELECT problem_id,min(course_id) course_id FROM lo_problem
        GROUP BY 1 HAVING count(DISTINCT course_id)=1""")
    query('map_comment', """CREATE TEMP TABLE map_comment AS
        SELECT comment_id,min(course_id) course_id,count(DISTINCT course_id) n_courses
        FROM course_comment GROUP BY 1""")
    query('map_reply', """CREATE TEMP TABLE map_reply AS
        SELECT r.reply_id,min(c.course_id) course_id
        FROM comment_reply r LEFT JOIN map_comment c USING(comment_id)
        GROUP BY 1 HAVING count(DISTINCT c.course_id)=1
        AND count(*) FILTER(WHERE c.comment_id IS NULL OR c.n_courses<>1)=0""")
    # Projection UNNEST streams items without repeating the original user's whole
    # nested seq. Reduce epoch lists in-place; never expand all video segments.
    video_time_filter = (f"x>=epoch(TIMESTAMP {qs(VIDEO_START)}) "
                         f"AND x<epoch(TIMESTAMP {qs(VIDEO_END_EXCLUSIVE)})")
    query('mapped_video_times', f"""CREATE TEMP TABLE mapped_video_times AS
        WITH items AS (SELECT unnest(seq) item FROM raw_video),
        mapped AS (
          SELECT m.course_id,list_transform(item.segment,s->s.local_start_time) epochs
          FROM items JOIN map_video m ON item.video_id=m.video_id
        ), filtered AS (
          SELECT course_id,coalesce(len(epochs),0) n_all,
            list_filter(epochs,x->{video_time_filter}) valid_epochs FROM mapped
        ) SELECT course_id,n_all,coalesce(len(valid_epochs),0) n_valid,
          list_min(valid_epochs) min_epoch,list_max(valid_epochs) max_epoch FROM filtered""")
    source_queries = {
        'video': """SELECT course_id,cast(to_timestamp(min(min_epoch)) AS TIMESTAMP) first_time,
            cast(to_timestamp(max(max_epoch)) AS TIMESTAMP) last_time,sum(n_valid)::BIGINT n_events
          FROM mapped_video_times WHERE n_valid>0 GROUP BY 1""",
        'problem': """SELECT m.course_id,min(try_cast(p.submit_time AS TIMESTAMP)) first_time,
            max(try_cast(p.submit_time AS TIMESTAMP)) last_time,count(*)::BIGINT n_events
          FROM raw_problem p JOIN map_problem m USING(problem_id)
          WHERE try_cast(p.submit_time AS TIMESTAMP) IS NOT NULL GROUP BY 1""",
        'comment': """SELECT m.course_id,min(cast(to_timestamp(c.create_time/1000.0) AS TIMESTAMP)) first_time,
            max(cast(to_timestamp(c.create_time/1000.0) AS TIMESTAMP)) last_time,count(*)::BIGINT n_events
          FROM comments c JOIN map_comment m ON c.id=m.comment_id
          WHERE m.n_courses=1 AND c.create_time BETWEEN 0 AND 4102444800000 GROUP BY 1""",
        'reply': """SELECT m.course_id,min(try_cast(r.create_time AS TIMESTAMP)) first_time,
            max(try_cast(r.create_time AS TIMESTAMP)) last_time,count(*)::BIGINT n_events
          FROM replies r JOIN map_reply m ON r.id=m.reply_id
          WHERE try_cast(r.create_time AS TIMESTAMP) IS NOT NULL GROUP BY 1""",
    }
    for name, sql in source_queries.items():
        query(f'{name}_bounds', f'CREATE TEMP TABLE {name}_bounds AS {sql}')
    union_sql = ' UNION ALL '.join(f'SELECT {qs(name)} source,* FROM {name}_bounds' for name in source_queries)
    source_path = export('course_source_time_bounds', union_sql)
    con.execute(f'CREATE VIEW source_bounds AS SELECT * FROM read_parquet({qs(source_path)})')
    bound_path = export('course_time_bounds', """WITH aggregate_bounds AS (
        SELECT course_id,min(first_time) course_start,max(last_time) course_end,
          count(*) n_activity_sources,sum(n_events)::BIGINT n_activity_events,
          string_agg(source,',' ORDER BY source) activity_sources,
          list(source ORDER BY source) FILTER(WHERE last_time=latest) course_end_sources
        FROM (SELECT *,max(last_time) OVER(PARTITION BY course_id) latest FROM source_bounds)
        GROUP BY 1)
        SELECT c.course_id,c.course_name,b.* EXCLUDE(course_id)
        FROM courses c LEFT JOIN aggregate_bounds b USING(course_id)""")
    con.execute(f'CREATE VIEW course_bounds AS SELECT * FROM read_parquet({qs(bound_path)})')
    video_audit = rows("""SELECT sum(n_all)::BIGINT mapped_segments,
        sum(n_valid)::BIGINT in_source_window,sum(n_all-n_valid)::BIGINT outside_source_window
        FROM mapped_video_times""")[0]
    for table in ['map_video', 'map_problem', 'map_comment', 'map_reply', 'mapped_video_times']:
        con.execute(f'DROP TABLE {table}')

    # Preserve every existing pair for audit; only positive intervals receive cutoffs.
    cutoff_fields = ',\n'.join(f"CASE WHEN time_status='ELIGIBLE_TIME' THEN {cutoff_sql('enroll_time', 'course_end', p)} END cutoff_{p}"
                               for p in PHASES)
    wide_path = export('user_course_phase_times', f"""WITH base AS (
        SELECT e.user_id,e.course_id,e.enroll_time,b.course_start,b.course_end,
          (b.course_id IS NOT NULL) course_exists_in_catalog
        FROM enroll e LEFT JOIN course_bounds b USING(course_id)
      ), classified AS (SELECT *,{time_status_sql()} time_status FROM base)
      SELECT *,time_status='ELIGIBLE_TIME' time_eligible,
        enroll_time<course_start registered_before_first_course_activity,
        date_diff('microsecond',enroll_time,course_end) observation_span_us,
        {cutoff_fields}
      FROM classified""")
    con.execute(f'CREATE VIEW times AS SELECT * FROM read_parquet({qs(wide_path)})')
    for phase, (num, den) in PHASES.items():
        export(phase, f"""SELECT user_id,course_id,{qs(phase)} phase,
            {num}::DOUBLE/{den} phase_fraction,enroll_time,course_end,
            cutoff_{phase} cutoff_time FROM times WHERE time_eligible""", phase_dir)

    eligibility_path = export('time_eligibility_summary', """SELECT time_status,count(*) n_pairs,
        count(DISTINCT course_id) n_courses FROM times GROUP BY 1 ORDER BY n_pairs DESC""")
    eligibility = rows(f'SELECT * FROM read_parquet({qs(eligibility_path)})')
    stats = rows("""SELECT count(*) n_input_pairs,count(*) FILTER(WHERE time_eligible) n_time_eligible_pairs,
        count(DISTINCT course_id) n_input_courses,
        count(DISTINCT course_id) FILTER(WHERE time_eligible) n_courses_with_time_eligible_pairs,
        count(*) FILTER(WHERE time_eligible AND registered_before_first_course_activity) n_eligible_early_registrations,
        min(enroll_time) first_enroll_time,max(enroll_time) last_enroll_time,
        min(course_end) first_course_end,max(course_end) last_course_end
        FROM times""")[0]
    export('course_time_eligibility', """SELECT course_id,count(*) n_pairs,
        count(*) FILTER(WHERE time_eligible) n_time_eligible_pairs,
        count(*) FILTER(WHERE time_status='ENROLLMENT_AFTER_COURSE_END') n_after_end,
        count(*) FILTER(WHERE time_status='ENROLLMENT_AT_COURSE_END') n_at_end,
        min(enroll_time) first_registration,max(enroll_time) last_registration
        FROM times GROUP BY 1""")
    phase_counts = []
    for phase in PHASES:
        path = phase_dir / f'{phase}.parquet'
        record = rows(f"SELECT {qs(phase)} phase,count(*) n_rows,min(cutoff_time) first_cutoff,max(cutoff_time) last_cutoff FROM read_parquet({qs(path)})")[0]
        phase_counts.append(record)

    # Independent cohort checks verify that joins/exports preserve the input keys.
    input_duplicates = con.execute('SELECT count(*) FROM (SELECT user_id,course_id FROM enroll GROUP BY 1,2 HAVING count(*)>1)').fetchone()[0]
    checks = {
        'input_user_course_unique': input_duplicates == 0,
        'all_input_pairs_preserved': stats['n_input_pairs'] == con.execute('SELECT count(*) FROM enroll').fetchone()[0],
        'one_bounds_row_per_catalog_course': con.execute('SELECT count(*)=count(DISTINCT course_id) AND count(*)=(SELECT count(*) FROM courses) FROM course_bounds').fetchone()[0],
        'eligibility_reconciles': sum(r['n_pairs'] for r in eligibility) == stats['n_input_pairs'],
        'four_exports_have_expected_count': all(r['n_rows'] == stats['n_time_eligible_pairs'] for r in phase_counts),
        'eligible_rows_have_all_cutoffs': con.execute("SELECT count(*)=0 FROM times WHERE time_eligible AND (cutoff_P1 IS NULL OR cutoff_P2 IS NULL OR cutoff_P3 IS NULL OR cutoff_P4 IS NULL)").fetchone()[0],
        'eligible_cutoffs_ordered_inside_window': con.execute("""SELECT count(*)=0 FROM times WHERE time_eligible AND
            NOT(enroll_time<=cutoff_P1 AND cutoff_P1<=cutoff_P2 AND cutoff_P2<=cutoff_P3
                AND cutoff_P3<=cutoff_P4 AND cutoff_P4<course_end)""").fetchone()[0],
        'ineligible_rows_have_no_cutoffs': con.execute("SELECT count(*)=0 FROM times WHERE NOT time_eligible AND (cutoff_P1 IS NOT NULL OR cutoff_P2 IS NOT NULL OR cutoff_P3 IS NOT NULL OR cutoff_P4 IS NOT NULL)").fetchone()[0],
        'formula_cases_passed': fixture_results['passed'],
    }
    for phase in PHASES:
        checks[f'{phase}_exact_formula_all_pairs'] = con.execute(f"SELECT count(*)=0 FROM times WHERE time_eligible AND cutoff_{phase} IS DISTINCT FROM {cutoff_sql('enroll_time', 'course_end', phase)}").fetchone()[0]
    # Earlier EDA uses the same mapping/time definition, providing an independent check.
    previous = raw.parent.parent / 'eda_review_course_enrollment/tables/course_bounds_candidates.parquet'
    if previous.exists():
        checks['matches_previous_cleaned_course_bounds'] = con.execute(f"""SELECT count(*)=0 FROM course_bounds b FULL JOIN
            (SELECT * FROM read_parquet({qs(previous)}) WHERE variant='VIDEO_STUDY_2020_JAN_NOV') p USING(course_id)
            WHERE b.course_id IS NULL OR p.course_id IS NULL OR b.course_start IS DISTINCT FROM p.course_start
              OR b.course_end IS DISTINCT FROM p.course_end""").fetchone()[0]
    invalid_checks = [name for name, ok in checks.items() if not ok]
    validation = {'checks': checks, 'fixtures': fixture_results, 'failed_checks': invalid_checks}
    (out / 'validation.json').write_text(json.dumps(validation, ensure_ascii=False, indent=2, default=str))
    if invalid_checks:
        raise AssertionError('Validation failed: ' + ', '.join(invalid_checks))

    # Reproducible real examples: two registrations from each of five courses.
    examples = export('phase_examples', """WITH chosen_courses AS (
        SELECT course_id FROM times WHERE time_eligible GROUP BY 1
        HAVING min(enroll_time)<max(enroll_time) ORDER BY course_id LIMIT 5
      ), ranked AS (
        SELECT t.*,row_number() OVER(PARTITION BY t.course_id ORDER BY enroll_time,user_id) pos,
          row_number() OVER(PARTITION BY t.course_id ORDER BY enroll_time DESC,user_id) reverse_pos
        FROM times t JOIN chosen_courses c USING(course_id) WHERE time_eligible
      ) SELECT * EXCLUDE(pos,reverse_pos) FROM ranked WHERE pos=1 OR reverse_pos=1 ORDER BY course_id,enroll_time""")
    summary = {'statistics': stats, 'eligibility': eligibility, 'phases': phase_counts,
               'n_snapshot_rows': sum(r['n_rows'] for r in phase_counts),
               'course_bounds': rows("SELECT count(*) n_courses,count(*) FILTER(WHERE course_end IS NOT NULL) with_activity,count(*) FILTER(WHERE course_end IS NULL) without_activity FROM course_bounds")[0],
               'video_time_audit': video_audit}
    summary['examples'] = rows(f'SELECT * FROM read_parquet({qs(examples)})')
    (out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    source_digest = globals().get('ENGINE_SOURCE_SHA256') or sha256(Path(__file__))
    manifest = {
        'analysis': 'Review section 2.5 time phases; no labels/features/splits',
        'created_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
        'duckdb_version': duckdb.__version__, 'engine_source_sha256': source_digest,
        'raw_root': str(raw), 'model_root': str(model), 'sampling': 'none',
        'elapsed_seconds': round(time.monotonic() - started, 3), 'inputs': inventory,
        'input_fingerprint': hashlib.sha256(json.dumps(inventory, sort_keys=True).encode()).hexdigest(),
        'policies': {
            'observation_key': ['user_id', 'course_id'],
            'start': 'existing enroll_time, including registrations before first course activity',
            'course_end': 'latest uniquely course-mapped timestamp across video/problem/comment/reply; excludes registration',
            'course_start': 'earliest same-source activity; metadata, not used in the phase formula',
            'valid_activity': 'timestamped action; invalid score or watch interval does not erase the existence of the action',
            'video_source_window': [VIDEO_START, VIDEO_END_EXCLUSIVE],
            'other_source_windows': 'all supplied parseable timestamps; no additional global study-window filter',
            'mapping': 'unique course; reply requires all parent links to resolve to the same course',
            'timezone': 'UTC computational convention; naive strings treated consistently as UTC; platform timezone not independently verified',
            'phase_fractions': {p: n/d for p,(n,d) in PHASES.items()},
            'rounding': 'integer microseconds; floor fractional microsecond, never round to calendar days',
            'eligibility': 'catalog + valid start/end + enroll_time < course_end; no activity-support filter on the user',
            'event_selection_for_next_step': 'same user/course and enroll_time <= event_time <= cutoff_time',
            'scope': 'time eligibility only; no final label eligibility or assertion of complete event coverage',
        },
        'outputs': [{'path': str(p.relative_to(out)), 'bytes': p.stat().st_size, 'sha256': sha256(p)} for p in outputs],
        'checks': checks, 'queries': queries,
    }
    (out / 'run_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str))
    write_report(out)
    success.write_text(json.dumps({'created_at_utc': manifest['created_at_utc'], 'checks_passed': len(checks),
                                  'summary_sha256': sha256(out / 'summary.json'),
                                  'manifest_sha256': sha256(out / 'run_manifest.json')}, indent=2))
    con.close()
    print('COMPLETE', json.dumps(stats, default=str), 'snapshots=', summary['n_snapshot_rows'], flush=True)
    return summary


def write_report(out):
    out = Path(out)
    s = json.loads((out / 'summary.json').read_text())
    checks = json.loads((out / 'validation.json').read_text())['checks']
    fmt = lambda value: f'{int(value):,}'.replace(',', '.')
    lines = [
        '# Mốc thời gian P1–P4 theo mục 2.5 của Review', '',
        '`e_kc = enroll_time` của user trong course; `T_c` là hoạt động hợp lệ cuối cùng của toàn course. '
        'Mỗi cặp `(user_id, course_id)` là một đối tượng; không tạo enrollment_id hay course_root_id.', '',
        '`cutoff_p = e_kc + lambda_p * (T_c - e_kc)`, với lambda = 0,25; 0,50; 0,75; 0,90. '
        'Tính bằng microsecond và làm tròn xuống phần nhỏ hơn một microsecond. Không làm tròn ngày.', '',
        '## Kết quả', '',
        f'- Cặp user–course đầu vào: {fmt(s["statistics"]["n_input_pairs"])}.',
        f'- Đủ điều kiện thời gian: {fmt(s["statistics"]["n_time_eligible_pairs"])} cặp, '
        f'thuộc {fmt(s["statistics"]["n_courses_with_time_eligible_pairs"])} khóa.',
        f'- Tổng snapshot thời gian: {fmt(s["n_snapshot_rows"])}.', '',
        '| Trạng thái | Số cặp | Số course_id trong trạng thái |', '|---|---:|---:|',
    ]
    lines.extend(f'| {r["time_status"]} | {fmt(r["n_pairs"])} | {fmt(r["n_courses"])} |' for r in s['eligibility'])
    lines += [
        '', 'Một course có thể xuất hiện ở nhiều trạng thái; không cộng số course giữa các dòng.', '',
        '## Quy tắc nguồn và thời gian', '',
        '- Tính course bounds từ log video, bài tập, bình luận và trả lời nối duy nhất về course. '
        'Đăng ký không kéo dài mốc cuối khóa.',
        '- Video dùng `[2020-01-19, 2020-11-04)` theo thời gian thu thập công bố. '
        'Các nguồn khác dùng toàn bộ timestamp parse được. Không áp cửa sổ toàn cục 01/05–03/11 hoặc 01/05–14/12.',
        '- Tính timestamp epoch theo UTC; chuỗi không có múi giờ giữ quy ước UTC nhất quán của EDA. '
        'Múi giờ thực tế của nền tảng chưa được xác minh độc lập.',
        '- Một hoạt động có timestamp hợp lệ vẫn là một hành động dù score hoặc interval xem chưa hợp lệ. '
        'Quy tắc tính feature xem/điểm cần kiểm tra riêng ở bước sau.',
        '- Đăng ký sớm hơn hoạt động đầu tiên của course vẫn được giữ; không đổi ngày đăng ký thành ngày xem đầu tiên.',
        '- Không tự loại người học ít/không có log. Không suy ra nhãn kém hay điền thiếu bằng 0.',
        '- Trường hợp `enroll_time >= course_end`, thiếu mốc cuối, ngoài catalog hoặc timestamp không hợp lệ '
        'giữ trong bảng rộng và không sinh snapshot.',
        '', '## Đầu ra', '',
        '- `tables/course_time_bounds.parquet`: mốc đầu/cuối của từng course và nguồn tạo mốc cuối.',
        '- `tables/course_source_time_bounds.parquet`: mốc theo nguồn để truy nguyên.',
        '- `tables/user_course_phase_times.parquet`: mọi cặp đầu vào, trạng thái thời gian và bốn cutoff '
        '(null ở các cặp chưa đủ điều kiện). Đây là kết quả tính thời gian trên bridge có sẵn.',
        '- `phase_cutoffs/P1.parquet` đến `P4.parquet`: bảng dài chỉ gồm các cặp đủ điều kiện thời gian.',
        '- `tables/phase_examples.parquet`: ví dụ các user trong cùng course có ngày ghi danh khác nhau.',
        '- `summary.json`, `run_manifest.json`, `validation.json`: thống kê, nguồn dữ liệu, hash, SQL và kiểm tra.',
        '', '## Cách dùng ở bước tạo đặc trưng', '',
        'Mỗi snapshot lấy sự kiện đúng user–course trong `[enroll_time, cutoff_time]`; các phase tích lũy. '
        'Nhãn cuối kỳ tính trên `[enroll_time, course_end]` và giữ nguyên qua bốn phase. '
        'Phải chia train/dev/test theo course trước khi fit tiền xử lý hoặc huấn luyện.', '',
        'Các trường cutoff, course_end, observation_span_us và trạng thái thời gian là metadata giao thức; '
        'không tự đưa chúng vào danh sách feature. Tập feature sẽ được chốt theo TEMPO-DQ.', '',
        '## Kiểm tra và giới hạn', '',
        f'Đã đạt {sum(bool(ok) for ok in checks.values())}/{len(checks)} kiểm tra trên dữ liệu và fixtures. '
        'Có kiểm tra bằng phép tính Python độc lập, hai ngày ghi danh cùng một course, đăng ký sớm, '
        'span dưới một ngày, thời lượng không dương và thiếu mốc.', '',
        '`course_end` là mốc quan sát hồi cứu, không phải ngày đóng khóa chính thức. '
        'Các nguồn có độ phủ khác nhau; đủ điều kiện thời gian chưa đồng nghĩa đủ điều kiện tính COELO/AFELO/ACELO. '
        'Bước này chưa tạo feature, q/y, split hay nhãn chất lượng.', '',
    ]
    (out / 'bao_cao_time_phases.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-root')
    parser.add_argument('--model-root')
    parser.add_argument('--output-root')
    args = parser.parse_args()
    run_build(args.raw_root, args.model_root, args.output_root)
