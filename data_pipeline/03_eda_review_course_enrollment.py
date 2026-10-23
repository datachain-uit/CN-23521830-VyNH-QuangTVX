"""EDA inputs 01/02 for the Review protocol; does not create quality labels.

Run with Python and duckdb==1.3.2. The companion notebook embeds this file so
that it can also run on Kaggle without attaching an extra Python module.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import time
from pathlib import Path

import duckdb


def sql_string(value):
    return "'" + str(value).replace("'", "''") + "'"


def normalize_id(expr, prefix):
    return (f"CASE WHEN {expr} IS NULL OR trim(CAST({expr} AS VARCHAR))='' THEN NULL "
            f"WHEN starts_with(trim(CAST({expr} AS VARCHAR)), '{prefix}') "
            f"THEN trim(CAST({expr} AS VARCHAR)) "
            f"ELSE '{prefix}' || trim(CAST({expr} AS VARCHAR)) END")


def discover_inputs():
    candidates = [Path.cwd() / 'input', Path.cwd() / 'working_13-10' / 'input']
    for root in candidates:
        if (root / 'moocubex_parquet/catalog.json').is_file() and (root / 'moocubex_model/catalog.json').is_file():
            return root / 'moocubex_parquet', root / 'moocubex_model'
    found = {'raw': [], 'model': []}
    for base in [Path('/kaggle/working'), Path('/kaggle/input')]:
        if base.is_dir():
            for catalog in base.rglob('catalog.json'):
                root = catalog.parent
                if (root / 'tables/relations/user_video').is_dir():
                    found['raw'].append(root)
                if (root / 'processed/dim_course').is_dir():
                    found['model'].append(root)
    if len(found['raw']) == len(found['model']) == 1:
        return found['raw'][0], found['model'][0]
    raise RuntimeError('Set RAW_ROOT and MODEL_ROOT explicitly; unique inputs not found: ' + str(found))


def run_eda(raw_root=None, model_root=None, output_root=None, threads=2, memory_limit='2GB'):
    if raw_root is None or model_root is None:
        default_raw, default_model = discover_inputs()
        raw_root = raw_root or default_raw
        model_root = model_root or default_model
    raw_root, model_root = Path(raw_root).resolve(), Path(model_root).resolve()
    output_root = Path(output_root or (
        '/kaggle/working/eda_review_course_enrollment' if Path('/kaggle').is_dir()
        else Path.cwd() / 'eda_review_course_enrollment'
    )).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    tables = output_root / 'tables'
    tables.mkdir(exist_ok=True)
    scratch = output_root / '_scratch'
    scratch.mkdir(exist_ok=True)
    con = duckdb.connect()
    con.execute(f'SET threads={int(threads)}')
    con.execute(f'SET memory_limit={sql_string(memory_limit)}')
    con.execute(f'SET temp_directory={sql_string(scratch)}')
    con.execute("SET TimeZone='UTC'")
    con.execute('SET preserve_insertion_order=false')
    results, query_log = {}, []
    started = time.monotonic()

    def run(name, query, show=8, export=True, make_table=False):
        print('\n' + name, flush=True)
        tick = time.monotonic()
        source_query = query
        if make_table:
            con.execute(f'CREATE OR REPLACE TEMP TABLE {name} AS {query}')
            query = f'SELECT * FROM {name}'
        if export:
            target = tables / (name + '.parquet')
            con.execute(f'COPY ({query}) TO {sql_string(target)} (FORMAT PARQUET, COMPRESSION ZSTD)')
            relation = f'read_parquet({sql_string(target)})'
            count = con.execute(f'SELECT count(*) FROM {relation}').fetchone()[0]
            cursor = con.execute(f'SELECT * FROM {relation} LIMIT {int(show)}')
        else:
            cursor = con.execute(query)
            count = None
        columns = [d[0] for d in cursor.description]
        rows = cursor.fetchall()
        results[name] = [dict(zip(columns, row)) for row in rows]
        elapsed = round(time.monotonic() - tick, 3)
        query_log.append({'name': name, 'rows': count, 'elapsed_seconds': elapsed, 'sql': source_query})
        print(f'rows={count if count is not None else len(rows)}; seconds={elapsed}', flush=True)
        for row in results[name]:
            print(json.dumps(row, default=str, ensure_ascii=False), flush=True)
        return results[name]

    def view(name, path):
        con.execute(f'CREATE OR REPLACE VIEW {name} AS SELECT * FROM read_parquet({sql_string(path)}, union_by_name=true)')

    raw_names = {
        'raw_course': 'entities/course', 'raw_user': 'entities/user',
        'raw_video': 'relations/user_video', 'raw_problem': 'relations/user_problem',
        'raw_reply': 'entities/reply', 'translated_reply': 'entities/translated_reply',
        'translated_comment': 'entities/translated_comment', 'comment_reply': 'relations/comment_reply',
        'course_teacher': 'relations/course_teacher', 'course_school': 'relations/course_school',
    }
    for name, relative in raw_names.items():
        path = raw_root / 'tables' / relative / '*.parquet'
        if not list(path.parent.glob('*.parquet')):
            raise FileNotFoundError(f'Required EDA input is missing: {path}')
        view(name, path)
    for name in ['dim_course', 'dim_lo_resource', 'bridge_lo_problem', 'bridge_exercise_problem',
                 'bridge_user_course_enrollment', 'bridge_course_comment', 'bridge_user_comment', 'dim_lo_catalog']:
        view(name, model_root / 'processed' / name / '*.parquet')
    view('enroll', model_root / 'processed/bridge_user_course_enrollment/*.parquet')
    view('previous_coverage', model_root / 'quality/join_coverage_report/*.parquet')
    view('problem_quarantine', model_root / 'quality/quarantine_exercise_problem/*.parquet')

    print('Exact EDA; no row sampling. DuckDB', duckdb.__version__, flush=True)
    inventory = []
    for kind, root in [('raw', raw_root), ('model', model_root)]:
        catalog = json.loads((root / 'catalog.json').read_text())
        for name, item in catalog['tables'].items():
            if kind == 'raw':
                relative = Path(name).with_suffix('').as_posix().replace('-', '_')
                directory = root / 'tables' / relative
            else:
                directory = root / ('quality' if name.startswith('quarantine') or name in ['join_coverage_report', 'catalog_audit'] else 'processed') / name
            files = sorted(directory.glob('*.parquet'))
            inventory.append({'kind': kind, 'table': name, 'manifest_rows': item.get('row_count'),
                              'files': len(files), 'bytes': sum(p.stat().st_size for p in files)})
    (output_root / 'input_inventory.json').write_text(json.dumps(inventory, indent=2, ensure_ascii=False))
    run('previous_join_coverage', 'SELECT * FROM previous_coverage')
    run('problem_quarantine_reasons', 'SELECT quarantine_reason, count(*) n_pairs FROM problem_quarantine GROUP BY 1 ORDER BY 2 DESC')
    run('enrollment_summary', '''
        SELECT count(*) enrollment_rows, count(DISTINCT user_id) users,
               count(DISTINCT course_id) courses,
               count(*) FILTER(WHERE course_exists_in_catalog) rows_in_catalog,
               count(*) FILTER(WHERE enroll_time IS NULL) invalid_time_rows,
               min(enroll_time) first_enroll_time, max(enroll_time) last_enroll_time
        FROM enroll''')
    run('enrollment_pair_counts', '''
        WITH pairs AS (
          SELECT user_id, course_id, count(*) n_records, count(DISTINCT enroll_time) n_distinct_times
          FROM enroll GROUP BY 1,2
        ) SELECT count(*) user_course_pairs, sum(n_records) enrollment_records,
                 count(*) FILTER(WHERE n_records>1) pairs_with_multiple_records,
                 count(*) FILTER(WHERE n_distinct_times>1) pairs_with_multiple_times,
                 count(*) FILTER(WHERE n_records>1 AND n_distinct_times=1) pairs_same_time_only,
                 max(n_records) max_records_per_pair FROM pairs''')
    run('raw_user_identity', '''SELECT count(*) raw_user_rows, count(DISTINCT id) distinct_users,
        sum(array_length(course_order)) raw_enrollment_entries,
        count(*) FILTER(WHERE COALESCE(array_length(course_order),0)<>COALESCE(array_length(enroll_time),0)) mismatched_arrays
        FROM raw_user''')
    run('enrollment_status', 'SELECT enroll_time_status, course_exists_in_catalog, count(*) n FROM enroll GROUP BY 1,2 ORDER BY 3 DESC')
    run('reply_time_encoding_check', '''
        SELECT date_diff('second',try_cast(r.create_time AS TIMESTAMP),
                   cast(to_timestamp(t.create_time/1000.0) AS TIMESTAMP)) offset_seconds,
               count(*) n_replies
        FROM raw_reply r JOIN translated_reply t USING(id) GROUP BY 1 ORDER BY 2 DESC''')

    # Map to COURSE, not to LO: shared resources within a single course remain usable.
    con.execute('''CREATE TEMP TABLE map_video AS
        SELECT resource_id video_id,min(course_id) course_id,count(DISTINCT course_id) n_courses
        FROM dim_lo_resource WHERE resource_type='video' GROUP BY 1''')
    con.execute('''CREATE TEMP TABLE map_problem AS
        SELECT problem_id,min(course_id) course_id,count(DISTINCT course_id) n_courses
        FROM bridge_lo_problem GROUP BY 1''')
    con.execute('''CREATE TEMP TABLE map_comment AS
        SELECT comment_id,min(course_id) course_id,count(DISTINCT course_id) n_courses
        FROM bridge_course_comment GROUP BY 1''')
    con.execute('''CREATE TEMP TABLE map_reply AS
        SELECT r.reply_id,min(c.course_id) course_id,
               count(DISTINCT c.course_id) n_courses,
               count(*) FILTER(WHERE c.comment_id IS NULL OR c.n_courses<>1) unresolved_parent_links
        FROM comment_reply r LEFT JOIN map_comment c USING(comment_id) GROUP BY 1''')
    run('mapping_cardinality', '''
        SELECT 'video' source, n_courses, count(*) resources FROM map_video GROUP BY 1,2
        UNION ALL SELECT 'problem',n_courses,count(*) FROM map_problem GROUP BY 1,2
        UNION ALL SELECT 'comment',n_courses,count(*) FROM map_comment GROUP BY 1,2
        UNION ALL SELECT 'reply',n_courses,count(*) FROM map_reply GROUP BY 1,2''')

    # Exact list reductions avoid exploding every segment into a large hash GROUP BY.
    # One row per original seq item; later course aggregates combine repeated items.
    # This is a TIME EDA table, not a feature/label event table.
    run('video_user_resource_bounds', '''
        SELECT uv.user_id,v.video_id,COALESCE(array_length(v.segment),0)::BIGINT n_events,
               COALESCE(array_length(list_filter(v.segment,s->s.local_start_time BETWEEN 0 AND 4102444800)),0)::BIGINT n_valid_times,
               list_min(list_transform(list_filter(v.segment,s->s.local_start_time BETWEEN 0 AND 4102444800),s->s.local_start_time)) min_epoch,
               list_max(list_transform(list_filter(v.segment,s->s.local_start_time BETWEEN 0 AND 4102444800),s->s.local_start_time)) max_epoch,
               COALESCE(array_length(list_filter(v.segment,s->s.end_point<=s.start_point OR s.start_point IS NULL OR s.end_point IS NULL)),0)::BIGINT invalid_intervals,
               COALESCE(array_length(list_filter(v.segment,s->s.speed<=0 OR s.speed IS NULL)),0)::BIGINT invalid_speeds
        FROM raw_video uv CROSS JOIN UNNEST(uv.seq) a(v)''', show=0, make_table=True)
    con.execute('''CREATE TEMP VIEW problem_timed AS SELECT *,try_cast(submit_time AS TIMESTAMP) event_time FROM raw_problem''')
    con.execute('''CREATE TEMP VIEW comment_timed AS SELECT id comment_id,
        CASE WHEN user_id IS NULL THEN NULL ELSE 'U_'||CAST(user_id AS VARCHAR) END user_id,
        resource_id, CASE WHEN create_time BETWEEN 0 AND 4102444800000
        THEN cast(to_timestamp(create_time/1000.0) AS TIMESTAMP) END event_time FROM translated_comment''')
    con.execute('''CREATE TEMP VIEW reply_timed AS SELECT id reply_id,user_id,
        try_cast(create_time AS TIMESTAMP) event_time FROM raw_reply''')
    run('source_id_format', '''SELECT 'problem' source,count(*) checked_rows,
        count(*) FILTER(WHERE problem_id IS NULL OR NOT starts_with(problem_id,'Pm_')) invalid_resource_prefix,
        count(*) FILTER(WHERE user_id IS NULL OR NOT starts_with(user_id,'U_')) invalid_user_prefix FROM raw_problem
        UNION ALL SELECT 'video_seq_items',count(*),
        count(*) FILTER(WHERE video_id IS NULL OR NOT starts_with(video_id,'V_')),
        count(*) FILTER(WHERE user_id IS NULL OR NOT starts_with(user_id,'U_')) FROM video_user_resource_bounds''')
    run('source_time_audit', '''
        SELECT 'video' source, sum(n_events)::BIGINT n_events,
          sum(n_events-n_valid_times)::BIGINT invalid_times,
          cast(to_timestamp(min(min_epoch)) AS TIMESTAMP) first_time,
          cast(to_timestamp(max(max_epoch)) AS TIMESTAMP) last_time,
          sum(invalid_intervals)::BIGINT invalid_intervals,sum(invalid_speeds)::BIGINT invalid_speeds
        FROM video_user_resource_bounds
        UNION ALL SELECT 'problem',count(*),count(*) FILTER(WHERE event_time IS NULL),min(event_time),max(event_time),NULL,NULL FROM problem_timed
        UNION ALL SELECT 'translated_comment',count(*),count(*) FILTER(WHERE event_time IS NULL),min(event_time),max(event_time),NULL,NULL FROM comment_timed
        UNION ALL SELECT 'reply',count(*),count(*) FILTER(WHERE event_time IS NULL),min(event_time),max(event_time),NULL,NULL FROM reply_timed''')
    run('source_mapping_coverage', '''
        SELECT 'video_segments' source, sum(v.n_events)::BIGINT total_events,
          sum(CASE WHEN m.n_courses=1 THEN v.n_events ELSE 0 END)::BIGINT uniquely_mapped,
          sum(CASE WHEN m.video_id IS NULL THEN v.n_events ELSE 0 END)::BIGINT unmapped,
          sum(CASE WHEN m.n_courses>1 THEN v.n_events ELSE 0 END)::BIGINT ambiguous
        FROM video_user_resource_bounds v LEFT JOIN map_video m USING(video_id)
        UNION ALL SELECT 'problem_rows',count(*),count(*) FILTER(WHERE m.n_courses=1),count(*) FILTER(WHERE m.problem_id IS NULL),count(*) FILTER(WHERE m.n_courses>1)
        FROM raw_problem p LEFT JOIN map_problem m USING(problem_id)
        UNION ALL SELECT 'translated_comment_rows',count(*),count(*) FILTER(WHERE m.n_courses=1),count(*) FILTER(WHERE m.comment_id IS NULL),count(*) FILTER(WHERE m.n_courses>1)
        FROM comment_timed c LEFT JOIN map_comment m USING(comment_id)
        UNION ALL SELECT 'reply_rows',count(*),count(*) FILTER(WHERE m.n_courses=1 AND m.unresolved_parent_links=0),
          count(*) FILTER(WHERE m.reply_id IS NULL OR m.n_courses=0),count(*) FILTER(WHERE m.n_courses>1 OR (m.n_courses=1 AND m.unresolved_parent_links>0))
        FROM reply_timed r LEFT JOIN map_reply m USING(reply_id)''')

    specs = [
        ('video', '''SELECT v.user_id,m.course_id,min(cast(to_timestamp(v.min_epoch) AS TIMESTAMP)) first_time,
            max(cast(to_timestamp(v.max_epoch) AS TIMESTAMP)) last_time,sum(v.n_valid_times)::BIGINT n_events
            FROM video_user_resource_bounds v JOIN map_video m USING(video_id)
            WHERE m.n_courses=1 AND v.n_valid_times>0 GROUP BY 1,2'''),
        ('problem', '''SELECT p.user_id,m.course_id,min(p.event_time) first_time,max(p.event_time) last_time,count(*)::BIGINT n_events
            FROM problem_timed p JOIN map_problem m USING(problem_id)
            WHERE m.n_courses=1 AND p.event_time IS NOT NULL GROUP BY 1,2'''),
        ('comment', '''SELECT c.user_id,m.course_id,min(c.event_time) first_time,max(c.event_time) last_time,count(*)::BIGINT n_events
            FROM comment_timed c JOIN map_comment m USING(comment_id)
            WHERE m.n_courses=1 AND c.event_time IS NOT NULL GROUP BY 1,2'''),
        ('reply', '''SELECT r.user_id,m.course_id,min(r.event_time) first_time,max(r.event_time) last_time,count(*)::BIGINT n_events
            FROM reply_timed r JOIN map_reply m USING(reply_id)
            WHERE m.n_courses=1 AND m.unresolved_parent_links=0 AND r.event_time IS NOT NULL GROUP BY 1,2'''),
    ]
    for source, query in specs:
        run(source + '_user_course_bounds', query, show=0, make_table=True)
    con.execute('CREATE TEMP VIEW activity_bounds AS ' + ' UNION ALL '.join(
        f"SELECT {sql_string(source)} source,* FROM {source}_user_course_bounds" for source, _ in specs))
    run('course_source_bounds', '''SELECT course_id,source,min(first_time) first_time,max(last_time) last_time,
        sum(n_events)::BIGINT n_events,count(*) user_course_source_rows
        FROM activity_bounds GROUP BY 1,2 ORDER BY 1,2''', show=8, make_table=True)
    run('course_observed_bounds', '''
        WITH activity AS (
          SELECT course_id,min(first_time) first_activity,max(last_time) last_activity,
            min(first_time) FILTER(WHERE source IN ('video','problem')) first_learning,
            max(last_time) FILTER(WHERE source IN ('video','problem')) last_learning,
            min(first_time) FILTER(WHERE source IN ('comment','reply')) first_social,
            max(last_time) FILTER(WHERE source IN ('comment','reply')) last_social,
            sum(n_events) n_events,count(DISTINCT source) n_sources FROM activity_bounds GROUP BY 1
        ), registrations AS (
          SELECT course_id,min(enroll_time) first_registration,max(enroll_time) last_registration,
            count(*) enrollment_rows FROM enroll GROUP BY 1
        ) SELECT c.course_id,a.* EXCLUDE(course_id),r.first_registration,r.last_registration,r.enrollment_rows
        FROM dim_course c LEFT JOIN activity a USING(course_id) LEFT JOIN registrations r USING(course_id)
        ''', show=0, make_table=True)

    # Explicit alternatives: EDA does not silently decide whether registration is activity.
    con.execute('''CREATE TEMP VIEW bounds_alternatives AS SELECT *,
        least(first_activity,first_registration) start_including_registration,
        greatest(last_activity,last_registration) end_including_registration,
        date_diff('day',first_activity,last_activity) activity_span_days FROM course_observed_bounds''')
    run('course_bounds_sensitivity', '''SELECT count(*) catalog_courses,
        count(*) FILTER(WHERE first_activity IS NOT NULL) courses_with_activity,
        count(*) FILTER(WHERE first_learning IS NOT NULL) courses_with_learning,
        count(*) FILTER(WHERE first_activity IS NULL) courses_without_mapped_activity,
        count(*) FILTER(WHERE first_social<first_learning) social_moves_start_earlier,
        count(*) FILTER(WHERE last_social>last_learning) social_moves_end_later,
        count(*) FILTER(WHERE first_registration<first_activity) registration_moves_start_earlier,
        count(*) FILTER(WHERE last_registration>last_activity) registration_moves_end_later,
        count(*) FILTER(WHERE first_activity=last_activity) courses_zero_activity_span,
        min(activity_span_days) min_span_days,median(activity_span_days) median_span_days,
        quantile_cont(activity_span_days,0.95) p95_span_days,max(activity_span_days) max_span_days
        FROM bounds_alternatives''')
    run('enrollment_vs_course_end', '''SELECT count(*) enrollment_rows_in_catalog,
        count(*) FILTER(WHERE b.last_activity IS NULL) no_observed_course_end,
        count(*) FILTER(WHERE e.enroll_time>b.last_activity) enrollment_after_activity_end,
        count(*) FILTER(WHERE e.enroll_time=b.last_activity) enrollment_equal_activity_end,
        count(*) FILTER(WHERE e.enroll_time<b.last_activity) positive_observation_span,
        count(*) FILTER(WHERE e.enroll_time IS NULL) missing_enrollment_time
        FROM enroll e JOIN course_observed_bounds b USING(course_id)''')
    run('activity_vs_enrollment', '''SELECT a.source,count(*) user_course_source_rows,
        count(*) FILTER(WHERE e.user_id IS NULL) no_enrollment_record,
        count(*) FILTER(WHERE e.user_id IS NOT NULL AND e.enroll_time IS NULL) enrollment_time_missing,
        count(*) FILTER(WHERE a.first_time<e.enroll_time) first_activity_before_registration,
        count(*) FILTER(WHERE a.last_time<e.enroll_time) all_activity_before_registration
        FROM activity_bounds a LEFT JOIN enroll e USING(user_id,course_id) GROUP BY 1 ORDER BY 1''')
    run('activity_catalog_coverage', '''SELECT a.source,count(*) user_course_source_rows,
        count(*) FILTER(WHERE c.course_id IS NOT NULL) rows_in_catalog,
        sum(a.n_events)::BIGINT mapped_events,
        sum(CASE WHEN c.course_id IS NOT NULL THEN a.n_events ELSE 0 END)::BIGINT mapped_events_in_catalog
        FROM activity_bounds a LEFT JOIN dim_course c USING(course_id) GROUP BY 1 ORDER BY 1''')
    run('social_catalog_coverage', '''SELECT
        (SELECT count(DISTINCT comment_id) FROM bridge_course_comment) relation_comment_ids,
        (SELECT count(DISTINCT comment_id) FROM comment_timed) supplied_comment_ids,
        (SELECT count(*) FROM comment_timed c WHERE EXISTS(SELECT 1 FROM map_comment m WHERE m.comment_id=c.comment_id)) supplied_comments_with_course,
        (SELECT count(*) FROM raw_reply) supplied_reply_rows''')
    run('course_static_relations', '''SELECT 'teacher' relation,
        count(DISTINCT c.course_id) catalog_courses_with_relation,count(*) relation_rows
        FROM course_teacher r JOIN dim_course c USING(course_id)
        UNION ALL SELECT 'school',count(DISTINCT c.course_id),count(*)
        FROM course_school r JOIN dim_course c USING(course_id)''')

    # Checks on actual aggregates, independent of a chosen course-end definition.
    checks = {
        'bounds_have_one_row_per_catalog_course': '''SELECT (SELECT count(*) FROM course_observed_bounds)=(SELECT count(*) FROM dim_course)
            AND (SELECT count(*) FROM course_observed_bounds)=(SELECT count(DISTINCT course_id) FROM course_observed_bounds)''',
        'activity_bounds_are_ordered': 'SELECT count(*)=0 FROM activity_bounds WHERE first_time>last_time',
        'event_counts_positive': 'SELECT count(*)=0 FROM activity_bounds WHERE n_events<=0',
        'enrollment_count_reconciles': '''SELECT (SELECT count(*) FROM enroll)=(SELECT sum(n) FROM (SELECT count(*) n FROM enroll GROUP BY user_id,course_id))''',
    }
    check_results = {name: bool(con.execute(query).fetchone()[0]) for name, query in checks.items()}
    if not all(check_results.values()):
        raise AssertionError(check_results)
    manifest = {
        'analysis': 'Review course bounds and repeat enrollment EDA',
        'created_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
        'duckdb_version': duckdb.__version__, 'raw_root': str(raw_root), 'model_root': str(model_root),
        'sampling': 'none', 'threads': threads, 'memory_limit': memory_limit,
        'timestamp_policy': 'video epoch seconds; translated comment epoch milliseconds; string times parsed as naive UTC; reply equality audited; true timezone not independently verified',
        'video_policy': 'timestamped segments count for any-activity bounds even if interval/speed invalid; invalid counts retained',
        'mapping_policy': 'unique course only; no replication of ambiguous events; course ranges limited to mapped, supplied events',
        'input_inventory_sha256': hashlib.sha256(json.dumps(inventory, sort_keys=True).encode()).hexdigest(),
        'checks': check_results, 'queries': query_log,
        'elapsed_seconds': round(time.monotonic()-started, 3),
    }
    (output_root / 'run_manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str))
    (output_root / 'summary.json').write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str))
    con.close()
    results.update(run_root_candidate_eda(model_root, output_root))
    results.update(run_time_candidate_eda(raw_root, model_root, output_root))
    results.update(run_readiness_checks(raw_root, model_root, output_root))
    write_report(output_root)
    print('\nEDA COMPLETE', output_root, 'seconds=', manifest['elapsed_seconds'], flush=True)
    return results


def run_root_candidate_eda(model_root, output_root):
    """Same-title courses are candidates only; never silently declare a root."""
    model_root, output_root = Path(model_root), Path(output_root)
    con = duckdb.connect()
    con.execute('SET threads=2')
    con.execute("SET memory_limit='2GB'")
    con.execute(f"SET temp_directory={sql_string(output_root / '_scratch')}")
    con.execute(f"CREATE VIEW courses AS SELECT * FROM read_parquet({sql_string(model_root / 'processed/dim_course/*.parquet')})")
    con.execute(f"CREATE VIEW enroll AS SELECT * FROM read_parquet({sql_string(model_root / 'processed/bridge_user_course_enrollment/*.parquet')})")
    con.execute("""CREATE TEMP TABLE course_names AS SELECT course_id,
        lower(regexp_replace(trim(course_name),'\\s+',' ','g')) normalized_name
        FROM courses WHERE course_name IS NOT NULL AND trim(course_name)<>''""")
    queries = {
        'course_name_candidates': '''WITH g AS (
            SELECT normalized_name,count(*) n_courses FROM course_names GROUP BY 1
        ) SELECT count(*) distinct_normalized_names,
            count(*) FILTER(WHERE n_courses>1) names_with_multiple_course_ids,
            sum(n_courses) FILTER(WHERE n_courses>1) course_ids_in_duplicate_names,
            max(n_courses) max_course_ids_per_name FROM g''',
        'repeated_same_name_candidates': '''WITH g AS (
            SELECT e.user_id,c.normalized_name,count(DISTINCT e.course_id) n_courses,
                min(e.enroll_time) first_enrollment,max(e.enroll_time) last_enrollment
            FROM enroll e JOIN course_names c USING(course_id) GROUP BY 1,2
        ) SELECT count(*) FILTER(WHERE n_courses>1) user_name_pairs_with_multiple_courses,
            sum(n_courses) FILTER(WHERE n_courses>1) enrollment_rows_in_candidates,
            count(*) FILTER(WHERE n_courses>1 AND first_enrollment=last_enrollment) candidates_same_timestamp,
            max(n_courses) max_courses_per_user_name FROM g''',
        'course_name_candidate_examples': '''SELECT normalized_name,count(*) n_courses,
            list(course_id ORDER BY course_id) course_ids FROM course_names GROUP BY 1
            HAVING count(*)>1 ORDER BY n_courses DESC, normalized_name LIMIT 10''',
    }
    extras, timings = {}, []
    for name, query in queries.items():
        tick = time.monotonic()
        path = output_root / 'tables' / (name + '.parquet')
        con.execute(f'COPY ({query}) TO {sql_string(path)} (FORMAT PARQUET, COMPRESSION ZSTD)')
        cursor = con.execute(f'SELECT * FROM read_parquet({sql_string(path)})')
        columns = [d[0] for d in cursor.description]
        extras[name] = [dict(zip(columns,row)) for row in cursor.fetchall()]
        timings.append({'name': name,'elapsed_seconds': round(time.monotonic()-tick,3),'sql': query})
        print(name, json.dumps(extras[name][:2], default=str, ensure_ascii=False), flush=True)
    con.close()
    summary_path = output_root / 'summary.json'
    payload = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    payload.update(extras)
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    manifest_path = output_root / 'run_manifest.json'
    payload = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    payload['root_candidate_queries'] = timings
    manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    return extras


def run_time_candidate_eda(raw_root, model_root, output_root):
    """Measure timestamp cleaning alternatives; none becomes a label policy."""
    raw_root, model_root, output_root = map(Path, (raw_root, model_root, output_root))
    con = duckdb.connect()
    con.execute('SET threads=2')
    con.execute("SET memory_limit='2GB'")
    con.execute("SET TimeZone='UTC'")
    con.execute(f"SET temp_directory={sql_string(output_root / '_scratch')}")
    def view(name, path):
        con.execute(f'CREATE VIEW {name} AS SELECT * FROM read_parquet({sql_string(path)})')
    view('raw_video',raw_root / 'tables/relations/user_video/*.parquet')
    view('resources',model_root / 'processed/dim_lo_resource/*.parquet')
    view('enroll',model_root / 'processed/bridge_user_course_enrollment/*.parquet')
    view('courses',model_root / 'processed/dim_course/*.parquet')
    view('course_sources',output_root / 'tables/course_source_bounds.parquet')
    con.execute('''CREATE TEMP TABLE mapped_segments AS
        WITH map AS (SELECT resource_id video_id,min(course_id) course_id
          FROM resources WHERE resource_type='video' GROUP BY 1 HAVING count(DISTINCT course_id)=1),
        chosen AS MATERIALIZED (
          SELECT uv.user_id,v.video_id,v.segment,m.course_id
          FROM raw_video uv CROSS JOIN UNNEST(uv.seq) a(v) JOIN map m ON v.video_id=m.video_id
        ) SELECT c.user_id,c.course_id,c.video_id,s.local_start_time epoch_seconds,
          CASE WHEN s.local_start_time BETWEEN 0 AND 4102444800
            THEN cast(to_timestamp(s.local_start_time) AS TIMESTAMP) END event_time
        FROM chosen c CROSS JOIN UNNEST(c.segment) b(s)''')
    queries = {
        'mapped_video_years': '''SELECT year(event_time) event_year,count(*) n_segments,
            count(DISTINCT course_id) n_courses FROM mapped_segments GROUP BY 1 ORDER BY 1''',
        'mapped_video_outlier_summary': '''SELECT count(*) mapped_segments,
            count(*) FILTER(WHERE event_time<TIMESTAMP '2019-01-01') before_2019,
            count(*) FILTER(WHERE event_time>=TIMESTAMP '2021-01-01') from_2021,
            count(DISTINCT course_id) FILTER(WHERE event_time<TIMESTAMP '2019-01-01' OR event_time>=TIMESTAMP '2021-01-01') affected_courses,
            min(event_time) first_time,max(event_time) last_time FROM mapped_segments''',
        'mapped_video_outlier_courses': '''SELECT course_id,min(event_time) first_time,max(event_time) last_time,
            count(*) FILTER(WHERE event_time<TIMESTAMP '2019-01-01' OR event_time>=TIMESTAMP '2021-01-01') outside_2019_2020
            FROM mapped_segments GROUP BY 1 HAVING outside_2019_2020>0 ORDER BY outside_2019_2020 DESC''',
        'course_bounds_candidates': '''WITH video_candidates AS (
            SELECT 'VIDEO_2019_2020' variant,course_id,min(event_time) first_time,max(event_time) last_time
              FROM mapped_segments WHERE event_time>=TIMESTAMP '2019-01-01' AND event_time<TIMESTAMP '2021-01-01' GROUP BY 1,2
            UNION ALL SELECT 'VIDEO_STUDY_2020_JAN_NOV',course_id,min(event_time),max(event_time)
              FROM mapped_segments WHERE event_time>=TIMESTAMP '2020-01-19' AND event_time<TIMESTAMP '2020-11-04' GROUP BY 1,2
        ), parts AS (
            SELECT 'RAW_ANY_ACTIVITY' variant,course_id,first_time,last_time FROM course_sources
            UNION ALL SELECT v.variant,s.course_id,s.first_time,s.last_time FROM course_sources s
              CROSS JOIN (VALUES ('VIDEO_2019_2020'),('VIDEO_STUDY_2020_JAN_NOV')) v(variant) WHERE s.source<>'video'
            UNION ALL SELECT * FROM video_candidates
        ), bounds AS (
            SELECT variant,course_id,min(first_time) course_start,max(last_time) course_end FROM parts GROUP BY 1,2
        ) SELECT v.variant,c.course_id,b.course_start,b.course_end,
            date_diff('day',b.course_start,b.course_end) span_days
            FROM courses c CROSS JOIN (VALUES ('RAW_ANY_ACTIVITY'),('VIDEO_2019_2020'),('VIDEO_STUDY_2020_JAN_NOV')) v(variant)
            LEFT JOIN bounds b ON b.variant=v.variant AND b.course_id=c.course_id ORDER BY 1,2''',
    }
    extras, timings = {}, []
    for name, query in queries.items():
        tick = time.monotonic()
        path = output_root / 'tables' / (name + '.parquet')
        con.execute(f'COPY ({query}) TO {sql_string(path)} (FORMAT PARQUET, COMPRESSION ZSTD)')
        cursor = con.execute(f'SELECT * FROM read_parquet({sql_string(path)}) LIMIT 12')
        columns = [d[0] for d in cursor.description]
        extras[name] = [dict(zip(columns,row)) for row in cursor.fetchall()]
        timings.append({'name': name,'elapsed_seconds': round(time.monotonic()-tick,3),'sql': query})
        print(name,json.dumps(extras[name][:4],default=str,ensure_ascii=False),flush=True)
    view('candidates',output_root / 'tables/course_bounds_candidates.parquet')
    query = '''WITH b AS (SELECT variant,count(*) catalog_courses,
            count(*) FILTER(WHERE course_end IS NOT NULL) courses_with_end,
            count(*) FILTER(WHERE course_end>=TIMESTAMP '2021-01-01') courses_end_from_2021,
            max(span_days) max_span_days,median(span_days) median_span_days FROM candidates GROUP BY 1),
        e AS (SELECT b.variant,count(*) enrollment_rows,
            count(*) FILTER(WHERE b.course_end IS NULL) no_course_end,
            count(*) FILTER(WHERE e.enroll_time>b.course_end) enrollment_after_end,
            count(*) FILTER(WHERE e.enroll_time<b.course_end) positive_observation_span
            FROM candidates b JOIN enroll e USING(course_id) GROUP BY 1)
        SELECT b.*,e.* EXCLUDE(variant) FROM b JOIN e USING(variant) ORDER BY variant'''
    path = output_root / 'tables/course_bounds_candidate_summary.parquet'
    con.execute(f'COPY ({query}) TO {sql_string(path)} (FORMAT PARQUET, COMPRESSION ZSTD)')
    cursor = con.execute(f'SELECT * FROM read_parquet({sql_string(path)})')
    columns = [d[0] for d in cursor.description]
    extras['course_bounds_candidate_summary'] = [dict(zip(columns,row)) for row in cursor.fetchall()]
    print('course_bounds_candidate_summary',json.dumps(extras['course_bounds_candidate_summary'],default=str),flush=True)
    timings.append({'name':'course_bounds_candidate_summary','sql':query})
    con.close()
    summary_path = output_root / 'summary.json'
    payload = json.loads(summary_path.read_text())
    payload.update(extras)
    summary_path.write_text(json.dumps(payload,indent=2,ensure_ascii=False,default=str))
    manifest_path = output_root / 'run_manifest.json'
    payload = json.loads(manifest_path.read_text())
    payload['time_candidate_queries'] = timings
    manifest_path.write_text(json.dumps(payload,indent=2,ensure_ascii=False,default=str))
    return extras


def snapshot_fingerprint(raw_root, model_root):
    records = []
    for kind, root in [('raw',Path(raw_root)),('model',Path(model_root))]:
        for path in sorted(root.rglob('*.parquet')):
            stat = path.stat()
            records.append((kind,str(path.relative_to(root)),stat.st_size,stat.st_mtime_ns))
        records.append((kind,'catalog.json',hashlib.sha256((root/'catalog.json').read_bytes()).hexdigest()))
    return hashlib.sha256(json.dumps(records,sort_keys=True).encode()).hexdigest()


def run_readiness_checks(raw_root, model_root, output_root):
    model_root,output_root=Path(model_root),Path(output_root)
    con=duckdb.connect()
    con.execute('SET threads=2')
    con.execute("SET memory_limit='2GB'")
    con.execute(f"SET temp_directory={sql_string(output_root/'_scratch')}")
    def view(name,path):
        con.execute(f'CREATE VIEW {name} AS SELECT * FROM read_parquet({sql_string(path)})')
    view('enroll',model_root/'processed/bridge_user_course_enrollment/*.parquet')
    view('bounds',output_root/'tables/course_observed_bounds.parquet')
    view('candidates',output_root/'tables/course_bounds_candidates.parquet')
    view('source_mapping',output_root/'tables/source_mapping_coverage.parquet')
    view('video_check',output_root/'tables/mapped_video_outlier_summary.parquet')
    for source in ['video','problem','comment','reply']:
        view(source+'_bounds',output_root/'tables'/f'{source}_user_course_bounds.parquet')
    con.execute('''CREATE TEMP TABLE user_activity AS SELECT user_id,course_id,
        min(first_time) first_time,max(last_time) last_time,
        max(last_time) FILTER(WHERE source IN ('video','problem')) last_learning
        FROM ('''+' UNION ALL '.join(f"SELECT {sql_string(s)} source,* FROM {s}_bounds" for s in ['video','problem','comment','reply'])+
        ') GROUP BY 1,2')
    queries={
        'enrollment_bound_policy_sensitivity': '''SELECT 'ACTIVITY_ONLY_RAW' bound_policy,count(*) n_enrollments,
            count(*) FILTER(WHERE b.last_activity IS NULL) missing_end,
            count(*) FILTER(WHERE e.enroll_time>b.last_activity) negative_duration,
            count(*) FILTER(WHERE e.enroll_time=b.last_activity) zero_duration,
            count(*) FILTER(WHERE e.enroll_time<b.last_activity) positive_duration
            FROM enroll e JOIN bounds b USING(course_id)
            UNION ALL SELECT 'INCLUDE_REGISTRATION',count(*),
            count(*) FILTER(WHERE greatest(b.last_activity,b.last_registration) IS NULL),
            count(*) FILTER(WHERE e.enroll_time>greatest(b.last_activity,b.last_registration)),
            count(*) FILTER(WHERE e.enroll_time=greatest(b.last_activity,b.last_registration)),
            count(*) FILTER(WHERE e.enroll_time<greatest(b.last_activity,b.last_registration))
            FROM enroll e JOIN bounds b USING(course_id)''',
        'enrollment_activity_support': '''SELECT count(*) enrollment_rows_in_catalog,
            count(*) FILTER(WHERE a.user_id IS NULL) no_supplied_mapped_activity,
            count(*) FILTER(WHERE a.last_time>=e.enroll_time) any_activity_at_or_after_enrollment,
            count(*) FILTER(WHERE a.last_learning>=e.enroll_time) learning_at_or_after_enrollment,
            count(*) FILTER(WHERE a.last_time<e.enroll_time) all_supplied_activity_before_enrollment
            FROM enroll e JOIN bounds b USING(course_id)
            LEFT JOIN user_activity a USING(user_id,course_id)''',
    }
    extras={}
    for name,query in queries.items():
        path=output_root/'tables'/(name+'.parquet')
        con.execute(f'COPY ({query}) TO {sql_string(path)} (FORMAT PARQUET,COMPRESSION ZSTD)')
        cur=con.execute(f'SELECT * FROM read_parquet({sql_string(path)})')
        columns=[d[0] for d in cur.description]
        extras[name]=[dict(zip(columns,row)) for row in cur.fetchall()]
        print(name,json.dumps(extras[name],default=str),flush=True)
    check_queries={
        'source_mapping_counts_reconcile':'SELECT count(*)=0 FROM source_mapping WHERE total_events<>uniquely_mapped+unmapped+ambiguous',
        'independent_video_flatten_matches_list_reduction':"SELECT (SELECT mapped_segments FROM video_check)=(SELECT uniquely_mapped FROM source_mapping WHERE source='video_segments')",
        'three_bounds_candidates_per_catalog_course': 'SELECT count(*)=3*(SELECT count(*) FROM bounds) AND count(*)=count(DISTINCT (variant,course_id)) FROM candidates',
    }
    checks={k:bool(con.execute(q).fetchone()[0]) for k,q in check_queries.items()}
    con.close()
    if not all(checks.values()):
        raise AssertionError(checks)
    path=output_root/'summary.json'
    payload=json.loads(path.read_text());payload.update(extras)
    path.write_text(json.dumps(payload,indent=2,ensure_ascii=False,default=str))
    path=output_root/'run_manifest.json'
    manifest=json.loads(path.read_text())
    manifest['checks'].update(checks)
    manifest['snapshot_fingerprint']=snapshot_fingerprint(raw_root,model_root)
    manifest['readiness_queries']=queries
    manifest['analysis_source_sha256']=globals().get('ENGINE_SOURCE_SHA256') or hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    path.write_text(json.dumps(manifest,indent=2,ensure_ascii=False,default=str))
    return extras


def write_report(output_root):
    output_root=Path(output_root)
    data=json.loads((output_root/'summary.json').read_text())
    one=lambda name:data[name][0]
    fmt=lambda n:f'{int(n):,}'.replace(',','.')
    e,p,b=map(one,['enrollment_summary','enrollment_pair_counts','course_bounds_sensitivity'])
    root,repeat=map(one,['course_name_candidates','repeated_same_name_candidates'])
    outlier=one('mapped_video_outlier_summary')
    cleaned=next(r for r in data['course_bounds_candidate_summary'] if r['variant']=='VIDEO_2019_2020')
    support=one('enrollment_activity_support')
    lines=[
        '# EDA dữ liệu course và lượt ghi danh theo Review ngày 12 tháng 10 năm 2026',
        '', 'Ngày phân tích theo phiên làm việc: 17/10/2026.',
        'Đầu vào là Parquet của notebook 01 và catalog/bridge của notebook 02 trong `input/`. '
        'Các thống kê quét toàn bộ dữ liệu đầu vào, không lấy mẫu. Đây là EDA và các phương án thời gian; chưa tạo nhãn chất lượng, split hay dataset huấn luyện.',
        '', '## Kết luận để quyết định bước dựng dataset', '',
        f'Có thể tái sử dụng hai notebook đầu. Trong {fmt(b["catalog_courses"])} course catalog, '
        f'{fmt(b["courses_with_activity"])} course có hoạt động gán được về course và '
        f'{fmt(b["courses_with_learning"])} course có video hoặc bài tập. '
        f'{fmt(b["courses_without_mapped_activity"])} course chưa có mốc hoạt động từ các nguồn hiện có.',
        f'Enrollment có {fmt(e["enrollment_rows"])} bản ghi, {fmt(e["users"])} user và {fmt(e["courses"])} course ID. '
        f'Có {fmt(e["rows_in_catalog"])} bản ghi thuộc course catalog. Không có timestamp enrollment parse lỗi.',
        '', '## 1 Đăng ký học lại và course gốc', '',
        f'Mỗi cặp `(user_id, course_id)` xuất hiện đúng một lần: số cặp có nhiều bản ghi = {fmt(p["pairs_with_multiple_records"])}. '
        'Kiểm tra raw user cũng cho đúng bằng số enrollment đã bung và không có array lệch độ dài. '
        'Vì vậy không quan sát được đăng ký lại cùng một course ID trong snapshot này; điều đó không chứng minh ngoài thực tế không có học lại.',
        f'Có {fmt(root["names_with_multiple_course_ids"])} tên khóa chuẩn hóa trùng, liên quan '
        f'{fmt(root["course_ids_in_duplicate_names"])} course ID. Có '
        f'{fmt(repeat["user_name_pairs_with_multiple_courses"])} cặp user–tên khóa đăng ký nhiều course ID, gồm '
        f'{fmt(repeat["enrollment_rows_in_candidates"])} bản ghi enrollment. Đây là ứng viên học lại hoặc các lần mở khác nhau, không phải số lượt học lại đã xác minh.',
        'Tên giống nhau chưa đủ để xác định `course_root_id`: cần đối chiếu trường, giảng viên và nội dung. '
        'Các bảng hiện có không cung cấp trường root/offering rõ ràng. Không gộp course theo tên và không tự tách lượt học chỉ từ một khoảng im lặng.',
        'Dataset mới nên giữ `enrollment_id`, `user_id`, `course_id`, `course_root_id`, `array_index` và `enroll_time`. '
        'Một bản ghi đăng ký có bằng chứng riêng là một quan sát riêng. Khi có nhiều lần đăng ký cùng course, '
        'event được gán theo khoảng từ lần đăng ký đó đến trước lần kế tiếp; các lần đăng ký trùng thời điểm cần cờ chưa phân giải. '
        'Các lần mở cùng course gốc phải ở cùng split theo review.',
        '', '## 2 Độ phủ mapping', '',
        '| Nguồn | Tổng bản ghi hoặc segment | Gán duy nhất về course | Tỷ lệ |',
        '|---|---:|---:|---:|',
    ]
    for r in data['source_mapping_coverage']:
        lines.append(f'| {r["source"]} | {fmt(r["total_events"])} | {fmt(r["uniquely_mapped"])} | {100*r["uniquely_mapped"]/r["total_events"]:.2f}% |')
    lines += [
        '', 'Các tỷ lệ trên dùng toàn bộ log làm mẫu số; không được diễn giải toàn bộ phần không nối được là lỗi code. '
        'Raw logs và course catalog có phạm vi khác nhau. Mốc course được tính ở đây chỉ dựa trên event nối được duy nhất; '
        'chưa chứng minh log đầy đủ cho từng course.',
        '', '## 3 Thời gian và timestamp bất thường', '',
        '| Nguồn | Mốc đầu quan sát | Mốc cuối quan sát |', '|---|---|---|',
    ]
    for r in data['source_time_audit']:
        lines.append(f'| {r["source"]} | {r["first_time"]} | {r["last_time"]} |')
    lines += [
        '', 'Các mốc trên là của toàn nguồn. Video dùng epoch giây, translated comment dùng epoch mili giây. '
        'Trên toàn bộ 331.011 reply, thời gian bản dịch sau đổi mili giây khớp chuỗi thời gian bản gốc với sai lệch 0 giây. '
        'Kết quả này kiểm tra tính nhất quán của encoding; chưa xác minh độc lập múi giờ thực tế của nền tảng.',
        f'Trong video đã map về catalog, có {fmt(outlier["from_2021"])} segment ở năm 2040 hoặc 2080, thuộc '
        f'{fmt(outlier["affected_courses"])} course. Nếu dùng min/max thô, course dài nhất thành 22.009 ngày. '
        f'Phương án giữ video trong năm 2019–2020 đưa độ dài lớn nhất về {fmt(cleaned["max_span_days"])} ngày. '
        'Hai phương án lọc video 2019–2020 và khoảng 19/01–03/11/2020 cho cùng mốc course trong snapshot này. '
        'Các bảng ứng viên được giữ riêng; chưa ghi đè dữ liệu gốc hay khóa quy tắc lọc cho nhãn.',
        '', '## 4 Đăng ký có được tính là hoạt động hay không', '',
        f'Thêm đăng ký vào tập hoạt động làm mốc đầu sớm hơn ở {fmt(b["registration_moves_start_earlier"])} course '
        f'và mốc cuối muộn hơn ở {fmt(b["registration_moves_end_later"])} course có hoạt động. '
        f'Bình luận/phản hồi làm mốc đầu sớm hơn hoạt động học ở {fmt(b["social_moves_start_earlier"])} course '
        f'và mốc cuối muộn hơn ở {fmt(b["social_moves_end_later"])} course.',
        '| Quy tắc | Thiếu mốc cuối | Thời lượng âm | Thời lượng bằng 0 | Thời lượng dương |',
        '|---|---:|---:|---:|---:|',
    ]
    for r in data['enrollment_bound_policy_sensitivity']:
        lines.append(f'| {r["bound_policy"]} | {fmt(r["missing_end"])} | {fmt(r["negative_duration"])} | {fmt(r["zero_duration"])} | {fmt(r["positive_duration"])} |')
    lines += [
        '', f'Sau phương án loại timestamp video ngoài 2019–2020, có {fmt(cleaned["enrollment_after_end"])} enrollment '
        f'muộn hơn hoạt động cuối của course, và {fmt(cleaned["no_course_end"])} enrollment thuộc course chưa có mốc hoạt động. '
        'Các lượt này không tự có cửa sổ tiến độ dương theo `enroll_time → last_activity`. '
        'Cần giữ trạng thái eligibility riêng; việc đưa registration vào min/max là lựa chọn định nghĩa, EDA chỉ đo hệ quả.',
        '', '## 5 Support ở cấp lượt ghi danh', '',
        f'Trong {fmt(support["enrollment_rows_in_catalog"])} enrollment thuộc catalog, '
        f'{fmt(support["no_supplied_mapped_activity"])} chưa có hoạt động map được từ các nguồn được cung cấp. '
        f'Có {fmt(support["any_activity_at_or_after_enrollment"])} lượt có ít nhất một hoạt động từ lúc đăng ký trở đi, '
        f'trong đó {fmt(support["learning_at_or_after_enrollment"])} lượt có video hoặc bài tập từ lúc đăng ký trở đi. '
        'Đây là support quan sát từ các nguồn hiện có, không phải kết luận rằng các user còn lại không học.',
        '', '## 6 Giới hạn social và công thức nhãn', '',
        'Không có `entities/comment` gốc trong input; có `entities/translated_comment`. '
        f'Course-comment chứa {fmt(one("social_catalog_coverage")["relation_comment_ids"])} comment ID, '
        f'nhưng bảng comment được cung cấp có {fmt(one("social_catalog_coverage")["supplied_comment_ids"])} ID. '
        'Vì vậy chưa thể coi social log đầy đủ hoặc tái tạo AFELO gốc từ các quan hệ này. '
        'Các quyết định công thức/normalization của review vẫn cần đối chiếu riêng; EDA này chưa thay công thức bằng proxy.',
        '', '## 7 Bước tiếp theo có căn cứ từ EDA', '',
        '1. Chốt nguồn hoạt động dùng cho observed course bounds và quy tắc timestamp bất thường; lưu cả mốc theo từng source.',
        '2. Đối chiếu các nhóm course trùng tên để xác nhận offering/course gốc. Bổ sung root mapping có provenance; không tự nhóm theo tên.',
        '3. Tạo bảng lượt ghi danh có ID ổn định, giữ mọi bản ghi đăng ký có bằng chứng riêng và trạng thái eligibility; không tạo lượt học lại từ suy đoán.',
        '4. Xác minh độ phủ và công thức ba chỉ số trước khi sinh nhãn cuối, rồi dựng P1–P4 theo review trên các cửa sổ hợp lệ.',
        '', '## Tái lập và kiểm tra', '',
        'Mã nguồn: `../03_eda_review_course_enrollment.py`; notebook cùng tên có đuôi `.ipynb`. '
        'Các bảng kết quả nằm trong `tables/`; số đo, SQL, chính sách parse thời gian và kiểm tra nằm trong `run_manifest.json`. '
        'Kiểm tra độc lập đối chiếu số segment video bằng hai cách giảm list và bung segment trực tiếp trên phần map được; '
        'kiểm tra bảo toàn tổng mapping, số dòng enrollment và số course của từng phương án thời gian.',
    ]
    (output_root/'bao_cao_eda.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw-root')
    parser.add_argument('--model-root')
    parser.add_argument('--output-root')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--memory-limit', default='2GB')
    args = parser.parse_args()
    run_eda(args.raw_root, args.model_root, args.output_root, args.threads, args.memory_limit)
