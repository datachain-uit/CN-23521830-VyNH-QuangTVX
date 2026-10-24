"""Build one full-window user-course feature table using the v3 cohort.

Every activity feature is recomputed from events through course_end (100%).
No public phase/phase_fraction columns. Preserve both v3 target variants and
split assignments; refit full-window numeric preprocessing for each train.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import shutil
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


h = load_module('quality_fullA_helpers', '18_build_final_quality_v3.py')
events = load_module('event_fullA_helpers', '16_build_phase_features_v2.py')
events.old.PHASES['fullA'] = (1, 1)
VERSION = 'fullA-full-window-v3-cohort-2026-10-21'
NUMERIC = [x for x in events.NUMERIC if x != 'phase_fraction']
CATEGORICAL = events.CATEGORICAL
META = ['user_id', 'course_id', 'split', 'enroll_time', 'course_end']
TARGETS, SPLITS = h.TARGETS, h.SPLITS
qs, view, write, records, dump = h.qs, h.view, h.write, h.records, h.dump


def build_windows(con, shard):
    events.table(con, 'pairs', f'''SELECT p.user_id,p.course_id,p.enroll_time,p.course_end,p.course_end cutoff_time,s.split,
        c.course_n_videos,c.course_n_exercises,c.course_n_problems,c.course_n_chapters,
        CASE WHEN year(p.enroll_time)-u.year_of_birth BETWEEN 10 AND 100 AND u.year_of_birth>=1900
          THEN (year(p.enroll_time)-u.year_of_birth)::DOUBLE END learner_age_at_enrollment,
        coalesce(e.code,0) learner_code,c.teacher_set_code,c.school_set_code,c.field_set_code,
        coalesce(u.gender_code,-1) gender_code
        FROM pair_slice p LEFT JOIN pair_splits s USING(user_id,course_id)
        LEFT JOIN course_encoded c USING(course_id)
        LEFT JOIN (SELECT * FROM static_users WHERE user_shard={shard}) u USING(user_id)
        LEFT JOIN enc_learner e ON p.user_id=e.token''')
    events.table(con, 'video_activity_w', f'''WITH selected AS (
        SELECT v.*,p.enroll_time,p.cutoff_time,
          CASE WHEN valid_measurement THEN least(v.end_point-v.start_point,
            greatest(0.0,epoch(p.cutoff_time)-epoch(v.event_time))*v.speed) ELSE 0 END observed_media_seconds
        FROM video_events v JOIN pairs p USING(user_id,course_id)
        WHERE hash(v.user_id)%32={shard} AND v.event_time BETWEEN p.enroll_time AND p.cutoff_time)
        SELECT *,start_point+observed_media_seconds observed_end_point,
          CASE WHEN valid_measurement THEN event_time+(observed_media_seconds/speed)*INTERVAL '1 second'
            ELSE event_time END observed_end_time FROM selected''')
    events.table(con, 'video_w', 'SELECT * FROM video_activity_w WHERE valid_measurement AND observed_media_seconds>0')
    events.table(con, 'problem_w', '''SELECT e.*,p.enroll_time,p.cutoff_time FROM problem_slice e JOIN pairs p USING(user_id,course_id)
        WHERE e.event_time BETWEEN p.enroll_time AND p.cutoff_time''')
    events.table(con, 'comment_all_w', f'''SELECT c.*,p.enroll_time,p.cutoff_time FROM comments c JOIN pairs p USING(user_id,course_id)
        WHERE hash(c.user_id)%32={shard} AND c.event_time BETWEEN p.enroll_time AND p.cutoff_time''')
    events.table(con, 'comment_w', 'SELECT * FROM comment_all_w WHERE normalized_text IS NOT NULL AND exact_duplicate_rank=1')
    events.table(con, 'reply_w', f'''SELECT r.*,p.enroll_time,p.cutoff_time FROM replies r JOIN pairs p USING(user_id,course_id)
        WHERE hash(r.user_id)%32={shard} AND r.event_time BETWEEN p.enroll_time AND p.cutoff_time''')
    events.table(con, 'received_w', '''SELECT c.user_id,c.course_id,c.lo_key,c.comment_id,r.reply_id,r.event_time,
        c.enroll_time,c.cutoff_time FROM comment_w c JOIN replies r ON c.comment_id=r.parent_comment_id
        WHERE r.event_time BETWEEN c.event_time AND c.cutoff_time''')


LOADER = '''"""Read a full-window labeled split, selecting a single alpha target."""
from pathlib import Path
import json
_CONNECTION = None

def feature_columns(ready=False, root=None):
    root = Path(root).resolve() if root is not None else Path(__file__).resolve().parent
    config = json.loads((root / 'feature_columns.json').read_text())
    numeric = list(json.loads((root / 'preprocessing.json').read_text())['numeric']) if ready else config['raw_numeric_features']
    return numeric + config['categorical_features']

def load_fullA(split='train', alpha=0.1, ready=False, root=None, connection=None, metadata=True):
    import duckdb
    global _CONNECTION
    if split not in {'train', 'dev', 'test'}:
        raise ValueError('split must be train, dev or test')
    if alpha not in {0.1, 1, 1.0}:
        raise ValueError('alpha must be 0.1 or 1')
    root = Path(root).resolve() if root is not None else Path(__file__).resolve().parent
    columns = feature_columns(ready, root)
    if metadata:
        columns = ['user_id','course_id','split','enroll_time','course_end'] + columns
    select = ','.join('"' + x.replace('"','""') + '"' for x in columns)
    suffix = '0p1' if alpha == 0.1 else '1'
    path = root / ('model_ready' if ready else 'supervised_raw') / (split + '.parquet')
    if not path.is_file():
        raise FileNotFoundError(path)
    if connection is None:
        if _CONNECTION is None:
            _CONNECTION = duckdb.connect()
        connection = _CONNECTION
    connection.execute("SET TimeZone='UTC'")
    return connection.sql(f"SELECT {select},q_alpha_{suffix} q,y_alpha_{suffix} y FROM read_parquet('" + str(path).replace("'","''") + "')")
'''


def deliver_variant(con, out, base, phase_root, variant):
    out.mkdir(parents=True, exist_ok=True)
    for name in ['learner', 'teacher_set', 'school_set', 'field_set']:
        target = out / 'encoders' / (name + '.parquet')
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(phase_root / 'encoders' / target.name, target)
    view(con, 'variant_learner', out / 'encoders/learner.parquet')
    view(con, 'variant_course', phase_root / '_stage/course_encoded.parquet')
    split_column = 'split' if variant == 'user_course_stratified' else 'split_course_random'
    # Category vocabularies are valid for this variant's unchanged v3 train
    # population. Reuse their exact codes; numeric parameters must be refitted.
    for split in SPLITS:
        write(con, out / 'supervised_raw' / (split + '.parquet'), f'''SELECT f.* EXCLUDE(split_course_random) REPLACE(
            f.{split_column} AS split,coalesce(e.code,0) AS learner_code,
            coalesce(c.teacher_set_code,0) AS teacher_set_code,coalesce(c.school_set_code,0) AS school_set_code,
            coalesce(c.field_set_code,0) AS field_set_code)
            FROM full_table f LEFT JOIN variant_learner e ON f.user_id=e.token
            LEFT JOIN variant_course c USING(course_id)
            WHERE f.y_alpha_0p1 IS NOT NULL AND f.{split_column}={qs(split)}''')
        write(con, out / 'sequences' / (split + '.parquet'), f'''SELECT f.user_id,f.course_id,f.{split_column} split,
            s.behavior_sequence FROM full_table f LEFT JOIN full_sequences s USING(user_id,course_id)
            WHERE f.y_alpha_0p1 IS NOT NULL AND f.{split_column}={qs(split)}''')
    view(con, 'fit_rows', out / 'supervised_raw/train.parquet')
    medians = records(con, 'SELECT ' + ','.join(f'median({x}) {x}' for x in NUMERIC) + ' FROM fit_rows')[0]
    active = [x for x in NUMERIC if medians[x] is not None]
    moments = records(con, 'SELECT ' + ','.join(
        f'avg(coalesce({x}::DOUBLE,{float(medians[x])!r})) {x}_mean,'
        f'stddev_pop(coalesce({x}::DOUBLE,{float(medians[x])!r})) {x}_std' for x in active) + ' FROM fit_rows')[0]
    params = {}
    for x in active:
        std = float(moments[x + '_std'])
        params[x] = {'median': float(medians[x]), 'mean': float(moments[x + '_mean']),
                     'scale': std if math.isfinite(std) and std > 1e-12 else 1.0,
                     'constant_after_imputation': std <= 1e-12}
    preprocessing = {'version': VERSION, 'window': '[enroll_time, course_end]', 'split_variant': variant,
        'fit_split': 'train', 'fit_rows': con.execute('SELECT count(*) FROM fit_rows').fetchone()[0],
        'numeric': params, 'all_missing_train_excluded': [x for x in NUMERIC if x not in active],
        'add_missing_indicators': False}
    dump(out / 'preprocessing.json', preprocessing)
    for split in SPLITS:
        expressions = [f'((coalesce({x}::DOUBLE,{p["median"]!r})-{p["mean"]!r})/{p["scale"]!r}) {x}' for x, p in params.items()]
        write(con, out / 'model_ready' / (split + '.parquet'), f'''SELECT {','.join(META)},
            {','.join(expressions + CATEGORICAL + TARGETS)}
            FROM read_parquet({qs(out / 'supervised_raw' / (split + '.parquet'))})''')
    config = {'version': VERSION, 'grain': ['user_id', 'course_id'], 'window': '[enroll_time, course_end]',
        'split_variant': variant, 'metadata_columns': META, 'raw_numeric_features': NUMERIC,
        'numeric_features': active, 'categorical_features': CATEGORICAL, 'model_feature_columns': active + CATEGORICAL,
        'distance_feature_columns': active, 'target_columns': TARGETS,
        'feature_groups': {k: [x for x in v if x != 'phase_fraction'] for k, v in events.old.GROUPS.items()},
        'preprocessing_scope': 'Full-window labeled train of this split variant; shared across alpha targets',
        'canonical_table_metadata_columns': ['user_id','course_id','split','split_course_random','enroll_time','course_end']}
    dump(out / 'feature_columns.json', config)
    dictionary = json.loads((phase_root / 'feature_dictionary.json').read_text())
    dictionary = [dict(item, time_scope='inclusive enrollment through course_end; 100% window')
                  for item in dictionary if item['name'] != 'phase_fraction']
    dump(out / 'feature_dictionary.json', dictionary)
    (out / 'load_data.py').write_text(LOADER, encoding='utf-8')

    checks = {}

    def check(name, sql):
        checks[name] = bool(con.execute(sql).fetchone()[0])
        if not checks[name]:
            raise AssertionError(variant + ': ' + name)

    view(con, 'variant_raw', out / 'supervised_raw/*.parquet')
    view(con, 'variant_ready', out / 'model_ready/*.parquet')
    view(con, 'variant_sequences', out / 'sequences/*.parquet')
    check('one_row_per_labeled_pair', 'SELECT count(*)=178598 AND count(*)=count(DISTINCT(user_id,course_id)) FROM variant_raw')
    check('scores_labels_and_assignment_preserved', f'''SELECT count(*)=0 FROM variant_raw r
        FULL JOIN (SELECT * FROM full_table WHERE y_alpha_0p1 IS NOT NULL) f USING(user_id,course_id)
        WHERE r.user_id IS NULL OR f.user_id IS NULL OR r.split IS DISTINCT FROM f.{split_column}
        OR {' OR '.join('r.' + x + ' IS DISTINCT FROM f.' + x for x in TARGETS + NUMERIC + ['enroll_time','course_end','gender_code'])}''')
    for name in ['learner', 'teacher_set', 'school_set', 'field_set']:
        checks[name + '_same_v3_train_vocabulary'] = h.sha256(out / 'encoders' / (name + '.parquet')) == h.sha256(phase_root / 'encoders' / (name + '.parquet'))
    for split in SPLITS:
        view(con, 'one_raw', out / 'supervised_raw' / (split + '.parquet'))
        view(con, 'one_ready', out / 'model_ready' / (split + '.parquet'))
        view(con, 'one_seq', out / 'sequences' / (split + '.parquet'))
        expected = con.execute(f'SELECT count(*) FROM full_table WHERE y_alpha_0p1 IS NOT NULL AND {split_column}={qs(split)}').fetchone()[0]
        check(split + '_counts', f'''SELECT (SELECT count(*)={expected} FROM one_raw)
            AND (SELECT count(*)={expected} FROM one_ready) AND (SELECT count(*)={expected} FROM one_seq)''')
        check(split + '_correct_split', f'SELECT count(*)=0 FROM one_raw WHERE split IS DISTINCT FROM {qs(split)}')
        same = META + CATEGORICAL + TARGETS
        check(split + '_ready_metadata_and_targets', f'''SELECT count(*)=0 FROM one_raw r FULL JOIN one_ready m USING(user_id,course_id)
            WHERE r.user_id IS NULL OR m.user_id IS NULL OR {' OR '.join('r.' + x + ' IS DISTINCT FROM m.' + x for x in same)}''')
        check(split + '_ready_finite', 'SELECT count(*)=0 FROM one_ready WHERE ' + ' OR '.join(f'{x} IS NULL OR NOT isfinite({x})' for x in active))
        check(split + '_sequences_complete', '''SELECT count(*)=0 FROM one_raw r FULL JOIN one_seq s USING(user_id,course_id)
            WHERE r.user_id IS NULL OR s.user_id IS NULL OR s.behavior_sequence IS NULL OR r.split IS DISTINCT FROM s.split''')
    view(con, 'ready_train', out / 'model_ready/train.parquet')
    nonconstant = [x for x, p in params.items() if not p['constant_after_imputation']]
    check('train_standardization', 'SELECT ' + ' AND '.join(f'abs(avg({x}))<1e-8 AND abs(stddev_pop({x})-1)<1e-8' for x in nonconstant) + ' FROM ready_train')
    if variant == 'course_random':
        check('no_course_id_overlap', 'SELECT count(*)=0 FROM (SELECT course_id FROM variant_raw GROUP BY 1 HAVING count(DISTINCT split)>1)')
    summary = {'version': VERSION, 'split_variant': variant, 'numeric_features': len(NUMERIC),
        'categorical_features': len(CATEGORICAL), 'labeled_pairs': 178598,
        'split_counts': records(con, 'SELECT split,count(*) n,count(DISTINCT course_id) courses FROM variant_raw GROUP BY 1 ORDER BY 1'),
        'class_counts': {a: records(con, f'SELECT split,y_alpha_{a} y,count(*) n FROM variant_raw GROUP BY 1,2 ORDER BY 1,2') for a in ['0p1','1']},
        'fit_rows': preprocessing['fit_rows'], 'checks_passed': sum(checks.values()), 'checks_total': len(checks)}
    if not all(checks.values()):
        raise AssertionError('Variant validation failed')
    dump(out / 'validation_delivery.json', checks)
    dump(out / 'summary_delivery.json', summary)
    print(f'{variant}: full-window train preprocessing fitted; {sum(checks.values())}/{len(checks)} checks passed', flush=True)
    return summary


def build(output=None, threads=2, memory='1500MB'):
    started = time.monotonic()
    out = Path(output or ROOT / 'fullA').resolve()
    quality, phase = ROOT / 'final_quality_v3', ROOT / 'phase_features_v3'
    for source in [quality, phase, ROOT / 'final_quality_v2', ROOT / 'phase_features_v2']:
        if out == source or source in out.parents or out in source.parents:
            raise ValueError('Output must be separate from source datasets')
        if not (source / '_SUCCESS.json').is_file():
            raise RuntimeError(str(source) + ' is incomplete')
    protected = [quality, phase, ROOT / 'final_quality_v2', ROOT / 'phase_features_v2']
    before = {str(p): h.inventory(p) for p in protected}
    out.mkdir(parents=True, exist_ok=True)
    (out / '_SUCCESS.json').unlink(missing_ok=True)
    con = h.connection(out, threads, memory)
    view(con, 'time_pairs', quality / 'tables/user_course_phase_times.parquet')
    view(con, 'quality_final', quality / 'tables/user_course_final/*.parquet')
    view(con, 'pair_splits', quality / 'tables/user_course_splits.parquet')
    view(con, 'course_splits', quality / 'tables/user_course_splits_course_random.parquet')
    con.execute('''CREATE TEMP TABLE labeled_targets AS SELECT user_id,course_id,
        q_alpha_0p1,y_alpha_0p1,q_alpha_1,y_alpha_1,n_chapters_observed,
        n_video_events_sum_observed,n_problem_events_sum_observed,total_problem_attempts_sum_observed,
        n_problems_correct_sum_observed,n_exercises_attempted_sum_observed,n_comments_for_sentiment_sum_observed,
        n_replies_authored_sum_observed,n_visits_sum_observed
        FROM quality_final WHERE y_alpha_0p1 IS NOT NULL''')
    for name, path in {
        'course_encoded': phase / '_stage/course_encoded.parquet',
        'enc_learner': phase / 'encoders/learner.parquet',
        'static_users': ROOT / 'phase_features_v2/_stage/static_users.parquet',
        'video_events': ROOT / 'final_quality_v2/_stage/video_window.parquet',
        'comments': ROOT / 'final_quality_v2/_stage/comment_window.parquet',
        'replies': ROOT / 'final_quality_v2/_stage/replies_prepared.parquet',
        'sentiment': quality / 'tables/comment_sentiment.parquet',
    }.items():
        view(con, name, path)
    raw_problem = ROOT / 'eda_engagement_features_comparison/_stage/problem_rows'
    raw_inventory = h.inventory(raw_problem)
    signature = hashlib.sha256(json.dumps({'version': VERSION, 'source': before,
        'problems': raw_inventory, 'builder_sha256': h.sha256(__file__),
        'helper_sha256': h.sha256(ROOT / '16_build_phase_features_v2.py'),
        'aggregation_sha256': h.sha256(ROOT / '09_build_phase_features.py')}, sort_keys=True).encode()).hexdigest()
    checkpoint_file = out / 'checkpoints.json'
    cache = json.loads(checkpoint_file.read_text()) if checkpoint_file.exists() else {}
    if cache.get('signature') != signature:
        cache = {'signature': signature, 'shards': {}}
    monotone = ['n_video_segments','n_video_events','n_videos_watched','video_media_seconds','video_unique_seconds',
        'n_problem_submissions','n_exercises_attempted','problem_attempts','n_correct_problems','n_first_try_correct',
        'n_comments','n_replies_authored','n_replies_received','n_events','n_active_days','n_chapters_touched',
        'n_unknown_chapters_touched','n_chapters_touched_all','n_visits_30m']
    reference = {'n_video_events': 'n_video_events_sum_observed',
        'n_problem_submissions': 'n_problem_events_sum_observed', 'problem_attempts': 'total_problem_attempts_sum_observed',
        'n_correct_problems': 'n_problems_correct_sum_observed', 'n_exercises_attempted': 'n_exercises_attempted_sum_observed',
        'n_comments': 'n_comments_for_sentiment_sum_observed', 'n_replies_authored': 'n_replies_authored_sum_observed',
        'n_visits_30m': 'n_visits_sum_observed', 'n_chapters_touched_all': 'n_chapters_observed'}
    for shard in range(32):
        token = str(shard)
        paths = {kind: out / '_stage' / kind / f'part-{shard:03d}.parquet' for kind in ['features','audit','sequences']}
        if token in cache['shards'] and all(p.is_file() and h.sha256(p) == cache['shards'][token]['hashes'][kind] for kind, p in paths.items()):
            print('REUSE fullA shard', shard, flush=True)
            continue
        tick = time.monotonic()
        events.table(con, 'pair_slice', f'''SELECT user_id,course_id,enroll_time,course_end FROM time_pairs
            WHERE time_eligible AND hash(user_id)%32={shard}''')
        view(con, 'problem_slice', raw_problem / f'user_shard={shard}/*.parquet')
        build_windows(con, shard)
        events.aggregate_windows(con)
        write(con, paths['features'], f'''SELECT * EXCLUDE(phase,phase_fraction,cutoff_time),cutoff_time course_end
            FROM ({events.old.feature_sql('fullA')})''')
        write(con, paths['audit'], '''SELECT p.user_id,p.course_id,p.enroll_time,p.cutoff_time course_end,
            a.min_feature_event_time,a.max_feature_event_time,a.n_out_of_window_events
            FROM activity_features a JOIN pairs p USING(user_id,course_id)''')
        write(con, paths['sequences'], f'''SELECT user_id,course_id,
            list(struct_pack(week_index:=week_index,{','.join(f'{x}:={x}::DOUBLE' for x in events.old.SEQ_FIELDS)}) ORDER BY week_index) behavior_sequence
            FROM weekly_features GROUP BY 1,2''')
        view(con, 'written', paths['features'])
        view(con, 'p4_reference', phase / 'features/P4' / f'part-{shard:03d}.parquet')
        stats = records(con, f'''SELECT count(*) n,count(DISTINCT(f.user_id,f.course_id)) n_unique,
            count(*) FILTER(WHERE f.user_id IS NULL OR p.user_id IS NULL OR {' OR '.join(f'f.{x}+1e-7<p.{x}' for x in monotone)}) n_p4_fail,
            count(*) FILTER(WHERE f.n_events>p.n_events) n_more_events_than_p4
            FROM written f FULL JOIN p4_reference p USING(user_id,course_id)''')[0]
        stats['n_bad_final_counts'] = con.execute(f'''SELECT count(*) FROM written f LEFT JOIN labeled_targets t USING(user_id,course_id)
            WHERE {' OR '.join(f'f.{x} IS DISTINCT FROM coalesce(t.{y},0)' for x, y in reference.items())}''').fetchone()[0]
        stats['n_out_of_window'] = con.execute('SELECT coalesce(sum(n_out_of_window_events),0) FROM activity_features').fetchone()[0]
        expected = con.execute('SELECT count(*) FROM pairs').fetchone()[0]
        if stats['n'] != stats['n_unique'] or stats['n'] != expected or any(stats[k] for k in ['n_p4_fail','n_bad_final_counts','n_out_of_window']):
            raise AssertionError((shard, stats))
        stats['hashes'] = {kind: h.sha256(path) for kind, path in paths.items()}
        cache['shards'][token] = stats
        dump(checkpoint_file, cache)
        print(f'BUILT fullA shard {shard}: {stats["n"]:,} rows, {time.monotonic()-tick:.1f}s', flush=True)
    view(con, 'raw_features', out / '_stage/features/*.parquet')
    view(con, 'audits', out / '_stage/audit/*.parquet')
    view(con, 'full_sequences', out / '_stage/sequences/*.parquet')
    # The one canonical fullA table. Both split assignments are metadata; the
    # canonical category codes use the default user-course train vocabulary.
    write(con, out / 'fullA.parquet', f'''SELECT f.user_id,f.course_id,f.split,g.split split_course_random,
        f.enroll_time,f.course_end,{','.join('f.' + x for x in NUMERIC + CATEGORICAL)},
        {','.join('t.' + x for x in TARGETS)} FROM raw_features f
        LEFT JOIN course_splits g USING(user_id,course_id) LEFT JOIN labeled_targets t USING(user_id,course_id)''')
    view(con, 'full_table', out / 'fullA.parquet')
    summaries = {}
    summaries['user_course_stratified'] = deliver_variant(con, out, out, phase, 'user_course_stratified')
    alternate = out / 'course_random_split'
    summaries['course_random'] = deliver_variant(con, alternate, out, phase / 'course_random_split', 'course_random')
    checks = {f'shard_{shard:02d}_complete_full_window':
              cache['shards'][str(shard)]['n'] == cache['shards'][str(shard)]['n_unique']
              and all(cache['shards'][str(shard)][k] == 0 for k in ['n_p4_fail','n_bad_final_counts','n_out_of_window'])
              for shard in range(32)}

    def check(name, sql):
        checks[name] = bool(con.execute(sql).fetchone()[0])
        if not checks[name]:
            raise AssertionError(name)

    check('full_population_preserved', 'SELECT count(*)=10455017 FROM full_table')
    check('labeled_population_preserved', 'SELECT count(*)=178598 FROM full_table WHERE y_alpha_0p1 IS NOT NULL')
    check('same_q_and_y_as_v3', f'''SELECT count(*)=0 FROM labeled_targets t FULL JOIN
        (SELECT * FROM full_table WHERE y_alpha_0p1 IS NOT NULL) f USING(user_id,course_id)
        WHERE t.user_id IS NULL OR f.user_id IS NULL OR {' OR '.join('t.' + x + ' IS DISTINCT FROM f.' + x for x in TARGETS)}''')
    check('same_split_assignments', '''SELECT count(*)=0 FROM
        (SELECT * FROM full_table WHERE y_alpha_0p1 IS NOT NULL) f
        JOIN pair_splits p USING(user_id,course_id) JOIN course_splits g USING(user_id,course_id)
        WHERE f.split IS DISTINCT FROM p.split OR f.split_course_random IS DISTINCT FROM g.split''')
    check('unlabeled_not_fabricated', '''SELECT count(*)=0 FROM full_table WHERE y_alpha_0p1 IS NULL
        AND (q_alpha_0p1 IS NOT NULL OR q_alpha_1 IS NOT NULL OR y_alpha_1 IS NOT NULL OR split IS NOT NULL OR split_course_random IS NOT NULL)''')
    check('finite_raw_observed_values', 'SELECT count(*)=0 FROM full_table WHERE ' + ' OR '.join(f'({x} IS NOT NULL AND NOT isfinite({x}))' for x in NUMERIC))
    check('no_event_outside_full_window', '''SELECT count(*)=0 FROM audits WHERE n_out_of_window_events<>0
        OR min_feature_event_time<enroll_time OR max_feature_event_time>course_end+INTERVAL '1 microsecond' ''')
    checks['public_table_has_no_phase_columns'] = not ({'phase','phase_fraction','cutoff_time'} & {x[0] for x in con.execute('DESCRIBE full_table').fetchall()})
    checks['full_window_observes_additional_activity_after_p4'] = sum(v['n_more_events_than_p4'] for v in cache['shards'].values()) > 0
    checks['sources_unchanged'] = all(before[str(p)] == h.inventory(p) for p in protected) and raw_inventory == h.inventory(raw_problem)
    if not all(checks.values()):
        raise AssertionError('FullA validation failed')
    summary = {'version': VERSION, 'table': 'fullA.parquet', 'grain': ['user_id','course_id'],
        'window': '[enroll_time, course_end]', 'window_fraction': 1.0,
        'rows': con.execute('SELECT count(*) FROM full_table').fetchone()[0],
        'labeled_pairs': 178598, 'unlabeled_pairs': 10276419, 'numeric_features': len(NUMERIC),
        'categorical_features': len(CATEGORICAL), 'columns': len(con.execute('DESCRIBE full_table').fetchall()),
        'no_activity_pairs': con.execute('SELECT count(*) FROM full_table WHERE n_events=0').fetchone()[0],
        'pairs_with_more_events_than_p4': sum(v['n_more_events_than_p4'] for v in cache['shards'].values()),
        'raw_missing': records(con, 'SELECT ' + ','.join(f'count(*) FILTER(WHERE {x} IS NULL) {x}' for x in NUMERIC) + ' FROM full_table')[0],
        'deliveries': summaries, 'checks_passed': sum(checks.values()), 'checks_total': len(checks)}
    manifest = {'version': VERSION, 'client_date': '2026-10-21', 'source_sha256': h.sha256(__file__),
        'source_signature': signature, 'quality_root': str(quality), 'phase_root': str(phase),
        'source_inventory': before, 'problem_inventory': raw_inventory,
        'window': '100% inclusive enrollment through existing v3 course_end; no phase partition',
        'features': 'Recomputed from raw mapped events using v2 feature definitions; phase and phase_fraction omitted',
        'targets': 'Frozen v3 alpha 0.1 and alpha 1 targets',
        'categories': 'Canonical table uses default v3 train encoders; course delivery remaps to its own v3 train encoders',
        'preprocessing': 'Fresh numeric medians/means/std fitted on full-window train, separately for both unchanged splits',
        'intended_use': 'Full-window analysis or end-of-window baseline; not early prediction',
        'elapsed_seconds': round(time.monotonic() - started, 3)}
    dump(out / 'summary.json', summary)
    dump(out / 'validation.json', checks)
    dump(out / 'run_manifest.json', manifest)
    lines = ['# fullA — Feature trên toàn bộ cửa sổ', '',
        'Một dòng cho mỗi user-course của cohort v3 sau lọc. Mọi feature hành vi được tính lại từ log trong [enroll_time, course_end], bao gồm 100% cửa sổ. Enrollment/course_end kế thừa cách chuẩn hóa của v3.',
        '', f'Bảng chính: **fullA.parquet**, {summary["rows"]:,} dòng, {summary["columns"]} cột; {summary["labeled_pairs"]:,} cặp có nhãn và {summary["unlabeled_pairs"]:,} cặp chưa có nhãn.',
        '', '**X:** 56 feature số + 5 feature danh mục. Không có phase, phase_fraction hoặc bốn snapshot 25/50/75/90%. Giá trị raw giữ NULL và số 0 theo định nghĩa cũ.',
        '', '**Metadata:** user_id, course_id, split (stratified user-course), split_course_random, enroll_time, course_end. **Target:** q_alpha_0p1, y_alpha_0p1, q_alpha_1, y_alpha_1, nguyên giá trị v3.',
        '', 'Các mã danh mục trong bảng chính dùng encoder train của bản user-course. Để dùng bản course random, đọc dữ liệu supervised/model_ready tại course_random_split/; các mã đã được ánh xạ đúng encoder của bản đó.',
        '', '## Các bản sử dụng', '', '| Bản | Train | Dev | Test |', '|---|---:|---:|---:|']
    for variant, s in summaries.items():
        counts = {x['split']: x['n'] for x in s['split_counts']}
        lines.append(f'| {variant} | {counts["train"]:,} | {counts["dev"]:,} | {counts["test"]:,} |')
    lines += ['', 'Mỗi root có supervised_raw/{train,dev,test}.parquet, model_ready/{train,dev,test}.parquet, sequences/{train,dev,test}.parquet, encoders/, preprocessing.json và load_data.py. Không có thư mục P1–P4.',
        '', 'Numeric preprocessing được fit lại trên full-window train của riêng bản split. Vocabulary category tái sử dụng đúng train v3 vì cohort và assignment không đổi. Hai alpha dùng chung X và split. Course random không giao course_id giữa train/dev/test; user vẫn có thể giao.',
        '', '## Đọc bảng', '', '```python', 'import duckdb',
        "fullA = duckdb.sql(\"SELECT * FROM read_parquet('working_13-10/fullA/fullA.parquet')\")",
        'print(fullA.shape)', '```', '',
        'Loader tại từng root: load_fullA(split="train", alpha=0.1, ready=True); dùng feature_columns(ready=True) để chọn X. Không đưa q/y hoặc các cột split vào X.',
        '', 'Chuỗi tuần được tính trên toàn bộ cửa sổ và lưu riêng trong sequences/; không phải các phase thời gian.',
        '', f'Có {summary["pairs_with_more_events_than_p4"]:,} cặp có thêm event so với snapshot P4; counts cuối cửa sổ đã đối chiếu với bảng chất lượng v3. Chi tiết kiểm tra và NULL trong summary.json/validation.json.',
        '', '**Diễn giải:** fullA dùng toàn bộ cửa sổ tạo nhãn nên là dữ liệu phân tích cuối cửa sổ/baseline cuối kỳ; không còn giao thức dự đoán sớm. q vẫn là nhãn proxy từ hành vi cùng cửa sổ, P95 kế thừa cohort v2; course_end là mốc hoạt động cuối hậu nghiệm.',
        '', 'Tái tạo: python working_13-10/20_build_fullA.py. Cần các bảng event/mapping v2 và cohort/assignment v3; không inference sentiment lại.']
    (out / 'README.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    all_passed = sum(checks.values()) + sum(s['checks_passed'] for s in summaries.values())
    for folder, variant in [(out, 'user_course_stratified'), (alternate, 'course_random')]:
        dump(folder / '_SUCCESS.json', {'version': VERSION, 'split_variant': variant,
            'labeled_pairs': 178598, 'checks_passed_delivery': summaries[variant]['checks_passed'],
            'all_checks_passed': all_passed, 'manifest_sha256': h.sha256(out / 'run_manifest.json')})
    con.close()
    print(f'COMPLETE fullA: {summary["rows"]:,} rows; {all_passed} checks passed', flush=True)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--memory', default='1500MB')
    args = parser.parse_args()
    build(args.output, args.threads, args.memory)
