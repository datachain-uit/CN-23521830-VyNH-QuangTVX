"""Filter verified v2 snapshots and refit preprocessing for both v3 split designs.

Default: stratified random user-course split. Additional course_random_split/:
random disjoint course split. Numerical raw features and cutoffs are unchanged;
category vocabularies and per-phase preprocessing are fitted on each new train.
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
spec = importlib.util.spec_from_file_location('quality_v3', ROOT / '18_build_final_quality_v3.py')
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
dump, sha256, inventory = helper.dump, helper.sha256, helper.inventory
view, write, records, qs = helper.view, helper.write, helper.records, helper.qs
PHASES, SPLITS, TARGETS = helper.PHASES, helper.SPLITS, helper.TARGETS
VERSION = 'phase-features-v3-filtered-pair-and-course-random-2026-10-21'


def build_variant(out, source, quality, assignment_file, variant, threads, memory):
    started = time.monotonic()
    out.mkdir(parents=True, exist_ok=True)
    (out / '_SUCCESS.json').unlink(missing_ok=True)
    source_inventory = inventory(source)
    signature = hashlib.sha256(json.dumps({
        'version': VERSION, 'source': source_inventory, 'builder_sha256': sha256(__file__),
        'helper_sha256': sha256(helper.__file__), 'quality_manifest': sha256(quality / 'run_manifest.json'),
        'assignment_sha256': sha256(assignment_file), 'variant': variant}, sort_keys=True).encode()).hexdigest()
    con = helper.connection(out, threads, memory)
    view(con, 'assignment', assignment_file)
    view(con, 'excluded', quality / 'tables/excluded_user_courses.parquet')
    view(con, 'quality_final', quality / 'tables/user_course_final/*.parquet')
    write(con, out / 'tables/user_course_targets.parquet', f'''SELECT f.user_id,f.course_id,s.split,
        {','.join('f.' + x for x in TARGETS)} FROM quality_final f LEFT JOIN assignment s USING(user_id,course_id)''')
    view(con, 'targets', out / 'tables/user_course_targets.parquet')
    con.execute('CREATE TEMP TABLE labeled AS SELECT * FROM targets WHERE y_alpha_0p1 IS NOT NULL')
    write(con, out / 'tables/labeled_user_course_splits.parquet', 'SELECT * FROM labeled ORDER BY user_id,course_id')
    if variant == 'course_random':
        shutil.copyfile(quality / 'tables/course_random_splits.parquet', out / 'tables/course_splits.parquet')

    # Rebuild every category vocabulary from this variant's labeled train pool.
    view(con, 'course_static', source / 'tables/course_static_metadata.parquet')
    shutil.copyfile(source / 'tables/course_static_metadata.parquet', out / 'tables/course_static_metadata.parquet')
    encoder_sources = {'learner': "SELECT DISTINCT user_id token FROM labeled WHERE split='train'"}
    for name, field in [('teacher_set', 'teacher_signature'), ('school_set', 'school_signature'), ('field_set', 'field_signature')]:
        encoder_sources[name] = f'''SELECT DISTINCT c.{field} token FROM course_static c
            JOIN labeled t USING(course_id) WHERE t.split='train' AND c.{field} IS NOT NULL'''
    for name, sql in encoder_sources.items():
        con.execute(f'CREATE TEMP TABLE enc_{name} AS SELECT token,row_number() OVER(ORDER BY token)::INTEGER code FROM ({sql})')
        write(con, out / 'encoders' / (name + '.parquet'), 'SELECT * FROM enc_' + name)
    con.execute('''CREATE TEMP TABLE course_encoded AS SELECT c.*,
        coalesce(t.code,0) teacher_set_code,coalesce(s.code,0) school_set_code,coalesce(f.code,0) field_set_code
        FROM course_static c LEFT JOIN enc_teacher_set t ON c.teacher_signature=t.token
        LEFT JOIN enc_school_set s ON c.school_signature=s.token LEFT JOIN enc_field_set f ON c.field_signature=f.token''')
    write(con, out / '_stage/course_encoded.parquet', 'SELECT * FROM course_encoded')
    config = json.loads((source / 'feature_columns.json').read_text())
    numeric, categorical, meta = config['raw_numeric_features'], config['categorical_features'], config['metadata_columns']
    config.update({'version': VERSION, 'split_variant': variant,
                   'preprocessing_scope': 'Per-phase labeled train of this split variant; shared by both alpha targets'})
    dump(out / 'feature_columns.json', config)
    shutil.copyfile(source / 'feature_dictionary.json', out / 'feature_dictionary.json')
    shutil.copyfile(source / 'load_data.py', out / 'load_data.py')

    checkpoint_file = out / 'checkpoints.json'
    cache = json.loads(checkpoint_file.read_text()) if checkpoint_file.is_file() else {}
    if cache.get('signature') != signature:
        cache = {'signature': signature, 'parts': {}}
    for phase in PHASES:
        for source_part in sorted((source / 'features' / phase).glob('*.parquet')):
            token = phase + '/' + source_part.name
            target_part = out / 'features' / phase / source_part.name
            if token in cache['parts'] and target_part.is_file() and sha256(target_part) == cache['parts'][token]['sha256']:
                continue
            view(con, 'source_part', source_part)
            write(con, target_part, '''SELECT f.* REPLACE(s.split AS split,
                coalesce(e.code,0) AS learner_code,coalesce(c.teacher_set_code,0) AS teacher_set_code,
                coalesce(c.school_set_code,0) AS school_set_code,coalesce(c.field_set_code,0) AS field_set_code)
                FROM source_part f ANTI JOIN excluded x USING(user_id,course_id)
                LEFT JOIN assignment s USING(user_id,course_id)
                LEFT JOIN enc_learner e ON f.user_id=e.token LEFT JOIN course_encoded c USING(course_id)''')
            view(con, 'written_part', target_part)
            unchanged = numeric + ['phase', 'enroll_time', 'cutoff_time', 'gender_code']
            bad = ' OR '.join('f.' + x + ' IS DISTINCT FROM v.' + x for x in unchanged)
            result = records(con, f'''SELECT count(*) n,
                count(DISTINCT(f.user_id,f.course_id)) n_unique,
                count(*) FILTER(WHERE f.user_id IS NULL OR v.user_id IS NULL OR {bad}) n_bad_values,
                count(*) FILTER(WHERE f.n_events=0) n_no_activity
                FROM written_part f FULL JOIN
                (SELECT v.* FROM source_part v ANTI JOIN excluded e USING(user_id,course_id)) v USING(user_id,course_id)''')[0]
            if result['n'] != result['n_unique'] or result['n_bad_values']:
                raise AssertionError((token, result))
            result['sha256'] = sha256(target_part)
            cache['parts'][token] = result
            dump(checkpoint_file, cache)
        # Keep the already verified event-window audit, filtered by the same final
        # exclusion rather than removing early snapshots that have only -1.
        write(con, out / 'audit' / phase / 'part-000.parquet', f'''SELECT a.*
            FROM read_parquet({qs(source / 'audit' / phase / '*.parquet')}) a
            ANTI JOIN excluded e USING(user_id,course_id)''')
        print(f'{variant}: {phase} full snapshots filtered and raw values verified', flush=True)

    preprocessing = {}
    for phase in PHASES:
        view(con, 'phase_x', out / 'features' / phase / '*.parquet')
        view(con, 'old_sequences', source / 'sequences' / phase / '*.parquet')
        for split in SPLITS:
            write(con, out / 'supervised_raw' / phase / (split + '.parquet'), f'''SELECT f.*,
                {','.join('t.' + x for x in TARGETS)} FROM phase_x f JOIN labeled t USING(user_id,course_id)
                WHERE t.split={qs(split)}''')
            write(con, out / 'sequences' / phase / (split + '.parquet'), f'''SELECT f.* REPLACE(t.split AS split)
                FROM old_sequences f JOIN labeled t USING(user_id,course_id) WHERE t.split={qs(split)}''')
        view(con, 'fit_rows', out / 'supervised_raw' / phase / 'train.parquet')
        medians = records(con, 'SELECT ' + ','.join(f'median({x}) {x}' for x in numeric) + ' FROM fit_rows')[0]
        active = [x for x in numeric if medians[x] is not None]
        moments = records(con, 'SELECT ' + ','.join(
            f'avg(coalesce({x}::DOUBLE,{float(medians[x])!r})) {x}_mean,'
            f'stddev_pop(coalesce({x}::DOUBLE,{float(medians[x])!r})) {x}_std'
            for x in active) + ' FROM fit_rows')[0]
        params = {}
        for name in active:
            std = float(moments[name + '_std'])
            params[name] = {'median': float(medians[name]), 'mean': float(moments[name + '_mean']),
                            'scale': std if math.isfinite(std) and std > 1e-12 else 1.0,
                            'constant_after_imputation': std <= 1e-12}
        preprocessing[phase] = {'fit_split': 'train', 'split_variant': variant,
            'fit_rows': con.execute('SELECT count(*) FROM fit_rows').fetchone()[0],
            'numeric': params, 'all_missing_train_excluded': [x for x in numeric if x not in active],
            'add_missing_indicators': False}
        for split in SPLITS:
            expressions = [f'((coalesce({x}::DOUBLE,{p["median"]!r})-{p["mean"]!r})/{p["scale"]!r}) {x}' for x, p in params.items()]
            write(con, out / 'model_ready' / phase / (split + '.parquet'), f'''SELECT {','.join(meta)},
                {','.join(expressions + categorical + TARGETS)}
                FROM read_parquet({qs(out / 'supervised_raw' / phase / (split + '.parquet'))})''')
        print(f'{variant}: {phase} train-only preprocessing fit on {preprocessing[phase]["fit_rows"]:,} rows', flush=True)
    dump(out / 'preprocessing.json', preprocessing)

    checks = {}

    def check(name, sql):
        checks[name] = bool(con.execute(sql).fetchone()[0])
        if not checks[name]:
            raise AssertionError(name)

    eligible = con.execute('SELECT count(*) FROM targets').fetchone()[0]
    labeled_count = con.execute('SELECT count(*) FROM labeled').fetchone()[0]
    view(con, 'supervised', out / 'supervised_raw/*/*.parquet')
    view(con, 'ready', out / 'model_ready/*/*.parquet')
    view(con, 'sequences', out / 'sequences/*/*.parquet')
    view(con, 'audits', out / 'audit/*/*.parquet')
    check('expected_full_population', f'SELECT {eligible}=10455017 AND {labeled_count}=178598')
    check('no_excluded_supervised_pair', 'SELECT count(*)=0 FROM supervised JOIN excluded USING(user_id,course_id)')
    check('four_rows_and_one_split_per_pair', '''SELECT count(*)=0 FROM
        (SELECT user_id,course_id FROM supervised GROUP BY 1,2 HAVING count(*)<>4 OR count(DISTINCT split)<>1 OR count(DISTINCT phase)<>4)''')
    check('no_fabricated_unlabeled_split', 'SELECT count(*)=0 FROM targets WHERE y_alpha_0p1 IS NULL AND split IS NOT NULL')
    check('all_supervised_targets_and_splits_match', f'''SELECT count(*)=0 FROM supervised f FULL JOIN labeled t USING(user_id,course_id)
        WHERE f.user_id IS NULL OR t.user_id IS NULL OR f.split IS DISTINCT FROM t.split OR
        {' OR '.join('f.' + x + ' IS DISTINCT FROM t.' + x for x in TARGETS)}''')
    check('no_out_of_window_events', '''SELECT count(*)=0 FROM audits WHERE n_out_of_window_events<>0
        OR min_feature_event_time<enroll_time OR max_feature_event_time>cutoff_time+INTERVAL '1 microsecond' ''')
    check('model_feature_allowlist_has_no_targets', 'SELECT ' + str(not (set(numeric + categorical) & set(TARGETS + ['COELO', 'AFELO', 'ACELO', 'q', 'y', 'course_end', 'split']))).lower())
    for name, sql in encoder_sources.items():
        check(name + '_exact_train_vocabulary', f'''SELECT count(*)=0 FROM enc_{name} e
            FULL JOIN ({sql}) t USING(token) WHERE e.token IS NULL OR t.token IS NULL''')
        check(name + '_positive_unique_codes', f'SELECT min(code)>0 AND count(*)=count(DISTINCT code) AND count(*)=count(DISTINCT token) FROM enc_{name}')
    if variant == 'course_random':
        check('course_ids_disjoint_between_splits', '''SELECT count(*)=0 FROM
            (SELECT course_id FROM labeled GROUP BY 1 HAVING count(DISTINCT split)>1)''')

    for phase in PHASES:
        parts = [v for k, v in cache['parts'].items() if k.startswith(phase + '/')]
        checks[phase + '_all_keys_and_raw_values_match_filtered_v2'] = (
            len(parts) == 32 and sum(x['n'] for x in parts) == eligible
            and all(x['n'] == x['n_unique'] and x['n_bad_values'] == 0 for x in parts))
        for split in SPLITS:
            view(con, 'one_raw', out / 'supervised_raw' / phase / (split + '.parquet'))
            view(con, 'one_ready', out / 'model_ready' / phase / (split + '.parquet'))
            view(con, 'one_seq', out / 'sequences' / phase / (split + '.parquet'))
            expected = con.execute(f'SELECT count(*) FROM labeled WHERE split={qs(split)}').fetchone()[0]
            check(phase + '_' + split + '_counts_and_uniqueness', f'''SELECT
                (SELECT count(*)={expected} AND count(*)=count(DISTINCT(user_id,course_id)) FROM one_raw)
                AND (SELECT count(*)={expected} AND count(*)=count(DISTINCT(user_id,course_id)) FROM one_ready)
                AND (SELECT count(*)={expected} AND count(*)=count(DISTINCT(user_id,course_id)) FROM one_seq)''')
            same = ['split', 'phase', 'enroll_time', 'cutoff_time'] + categorical + TARGETS
            check(phase + '_' + split + '_ready_and_sequences_keys', f'''SELECT
                (SELECT count(*)=0 FROM one_raw r FULL JOIN one_ready m USING(user_id,course_id)
                 WHERE r.user_id IS NULL OR m.user_id IS NULL OR {' OR '.join('r.' + x + ' IS DISTINCT FROM m.' + x for x in same)})
                AND (SELECT count(*)=0 FROM one_raw r FULL JOIN one_seq s USING(user_id,course_id)
                 WHERE r.user_id IS NULL OR s.user_id IS NULL OR r.split IS DISTINCT FROM s.split OR r.phase IS DISTINCT FROM s.phase)''')
            view(con, 'original_sequences', source / 'sequences' / phase / '*.parquet')
            check(phase + '_' + split + '_sequence_values_unchanged', '''SELECT count(*)=0 FROM one_seq s
                LEFT JOIN original_sequences o USING(user_id,course_id) WHERE o.user_id IS NULL
                OR s.behavior_sequence IS DISTINCT FROM o.behavior_sequence''')
            active = list(preprocessing[phase]['numeric'])
            check(phase + '_' + split + '_ready_finite', 'SELECT count(*)=0 FROM one_ready WHERE ' +
                  ' OR '.join(x + ' IS NULL OR NOT isfinite(' + x + ')' for x in active))
        view(con, 'train_ready', out / 'model_ready' / phase / 'train.parquet')
        nonconstant = [x for x, p in preprocessing[phase]['numeric'].items() if not p['constant_after_imputation']]
        check(phase + '_ready_train_mean_zero_std_one', 'SELECT ' + ' AND '.join(
            f'abs(avg({x}))<1e-8 AND abs(stddev_pop({x})-1)<1e-8' for x in nonconstant) + ' FROM train_ready')
        check(phase + '_fit_train_count', f'''SELECT count(*)={preprocessing[phase]['fit_rows']}
            FROM labeled WHERE split='train' ''')
    checks['source_v2_unchanged'] = source_inventory == inventory(source)
    if not all(checks.values()):
        raise AssertionError([name for name, passed in checks.items() if not passed])
    summary = {'version': VERSION, 'split_variant': variant, 'eligible_pairs': eligible, 'labeled_pairs': labeled_count,
        'numeric_features': len(numeric), 'categorical_features': len(categorical),
        'labeled_splits': records(con, 'SELECT split,count(*) n,count(DISTINCT course_id) n_courses,count(DISTINCT user_id) n_users FROM labeled GROUP BY 1 ORDER BY 1'),
        'class_counts': {s: records(con, f'SELECT split,y_alpha_{s} y,count(*) n FROM labeled GROUP BY 1,2 ORDER BY 1,2') for s in ['0p1', '1']},
        'full_phase_counts': [{'phase': p, 'n': sum(v['n'] for k, v in cache['parts'].items() if k.startswith(p + '/')),
            'n_no_activity': sum(v['n_no_activity'] for k, v in cache['parts'].items() if k.startswith(p + '/'))} for p in PHASES],
        'supervised_phase_counts': records(con, 'SELECT phase,split,count(*) n,count(*) FILTER(WHERE n_events=0) n_no_activity FROM supervised GROUP BY 1,2 ORDER BY 1,2'),
        'courses_in_multiple_splits': con.execute('SELECT count(*) FROM (SELECT course_id FROM labeled GROUP BY 1 HAVING count(DISTINCT split)>1)').fetchone()[0],
        'users_in_multiple_splits': con.execute('SELECT count(*) FROM (SELECT user_id FROM labeled GROUP BY 1 HAVING count(DISTINCT split)>1)').fetchone()[0],
        'checks_passed': sum(checks.values()), 'checks_total': len(checks)}
    manifest = {'version': VERSION, 'client_date': '2026-10-21', 'split_variant': variant,
        'quality_root': str(quality), 'source_phase_root': str(source), 'source_inventory': source_inventory,
        'assignment_file': str(assignment_file), 'assignment_sha256': sha256(assignment_file),
        'source_sha256': sha256(__file__), 'source_signature': signature,
        'cutoffs': 'Stored v2 cutoffs; raw numerical features and sequences exactly preserved after final-pair exclusion',
        'preprocessing': 'Per-phase medians, means/std and categorical vocabularies fitted only on this variant labeled train',
        'unlabeled': 'Retained in features/ and targets; split=NULL, excluded from supervised files',
        'limitations': 'Frozen v2 label P95 references use the original cohort; course_end is retrospective; user overlap is allowed',
        'elapsed_seconds': round(time.monotonic() - started, 3)}
    dump(out / 'summary.json', summary)
    dump(out / 'validation.json', checks)
    dump(out / 'run_manifest.json', manifest)
    lines = ['# Phase features v3: ' + variant, '',
        'Cùng bộ lọc user-course với final_quality_v3; giữ snapshot sớm chưa có hoạt động hoặc chỉ có -1 nếu cuối cùng user có hoạt động ở chapter xác định được.',
        '', 'Mỗi phase có 10.455.017 cặp đầy đủ và 178.598 cặp supervised. Raw NULL được giữ; model_ready dùng median imputation và standardization fit trên train của riêng phase và riêng bản split.',
        '', '## Split', '', '| Tập | User-course | Course |', '|---|---:|---:|']
    for row in summary['labeled_splits']:
        lines.append(f'| {row["split"]} | {row["n"]:,} | {row["n_courses"]:,} |')
    lines += ['', 'Split theo ' + ('course, không trùng course_id giữa các tập; không stratify nhãn.' if variant == 'course_random' else 'user-course, stratify tổ hợp hai nhãn alpha; course và user có thể giao giữa các tập.'),
        'Hai alpha dùng cùng X và cùng split. Seed=42, tỷ lệ 70/10/20 áp dụng cho đơn vị chia. Một user có thể thuộc nhiều tập ở cả hai bản.',
        '', '## Sử dụng', '',
        '- `supervised_raw/P1/train.parquet` (tương tự P2–P4 và dev/test): raw X và cả hai target.',
        '- `model_ready/`: X đã xử lý bằng tham số train; `sequences/`: chuỗi hành vi theo tuần.',
        '- `features/`: mọi cặp sau lọc; cặp chưa có nhãn có split=NULL.',
        '- `encoders/`, `preprocessing.json`: fit riêng cho bản split này; không dùng chéo giữa hai bản.',
        '- `load_data.py`: `load_phase(phase="P1", split="train", alpha=0.1, ready=True)`.',
        '', 'Giới hạn kế thừa: P95 tạo nhãn dùng quần thể hiệu chuẩn v2; cutoff dựa course_end hậu nghiệm. Course random kiểm tra tổng quát hóa sang course mới trong mẫu này, vẫn cho phép user xuất hiện ở nhiều tập.',
        '', f'Kiểm tra: {sum(checks.values())}/{len(checks)} đạt.']
    (out / 'BAO_CAO_PHASE_FEATURES_V3.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    dump(out / '_SPLIT_SUCCESS.json', {'version': VERSION, 'split_variant': variant,
        'labeled_pairs': labeled_count, 'checks_passed': sum(checks.values()), 'manifest_sha256': sha256(out / 'run_manifest.json')})
    con.close()
    print(f'{variant}: {sum(checks.values())}/{len(checks)} checks passed', flush=True)
    return summary


def build(output=None, threads=2, memory='1200MB'):
    quality = ROOT / 'final_quality_v3'
    source = ROOT / 'phase_features_v2'
    out = Path(output or ROOT / 'phase_features_v3').resolve()
    for origin in [quality, source, ROOT / 'phase_features']:
        if out == origin or origin in out.parents or out in origin.parents:
            raise ValueError('Output must be separate from sources')
    for origin in [quality, source]:
        if not (origin / '_SUCCESS.json').is_file():
            raise RuntimeError(str(origin) + ' is incomplete')
    out.mkdir(parents=True, exist_ok=True)
    (out / '_SUCCESS.json').unlink(missing_ok=True)
    main = build_variant(out, source, quality, quality / 'tables/user_course_splits.parquet',
                         'user_course_stratified', threads, memory)
    alternate_out = out / 'course_random_split'
    alternate = build_variant(alternate_out, source, quality, quality / 'tables/user_course_splits_course_random.parquet',
                              'course_random', threads, memory)
    # Compare the complete labeled populations and their targets across designs.
    con = helper.connection(out, threads, memory)
    view(con, 'main_targets', out / 'tables/labeled_user_course_splits.parquet')
    view(con, 'alternate_targets', alternate_out / 'tables/labeled_user_course_splits.parquet')
    comparison = bool(con.execute(f'''SELECT count(*)=0 FROM main_targets a FULL JOIN alternate_targets b USING(user_id,course_id)
        WHERE a.user_id IS NULL OR b.user_id IS NULL OR {' OR '.join('a.' + x + ' IS DISTINCT FROM b.' + x for x in TARGETS)}''').fetchone()[0])
    con.close()
    if not comparison:
        raise AssertionError('Split variants must use identical labeled populations and targets')
    dump(out / 'split_variants.json', {'version': VERSION, 'same_pairs_and_targets': comparison,
        'default': {'root': '.', 'unit': 'user-course', 'stratification': 'joint alpha labels'},
        'additional': {'root': 'course_random_split', 'unit': 'course_id', 'stratification': None},
        'summary': {'user_course_stratified': main['labeled_splits'], 'course_random': alternate['labeled_splits']}})
    (out / 'BAO_CAO_PHASE_FEATURES_V3.md').open('a', encoding='utf-8').write(
        '\nBản bổ sung: `course_random_split/`, cùng mẫu và target; course_id không giao giữa train/dev/test. Xem split_variants.json.\n')
    for folder, summary in [(alternate_out, alternate), (out, main)]:
        dump(folder / '_SUCCESS.json', {'version': VERSION, 'split_variant': summary['split_variant'],
            'labeled_pairs': summary['labeled_pairs'], 'checks_passed': summary['checks_passed'],
            'same_pairs_and_targets_across_variants': comparison, 'manifest_sha256': sha256(folder / 'run_manifest.json')})
    return main, alternate


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--memory', default='1200MB')
    args = parser.parse_args()
    build(args.output, args.threads, args.memory)
