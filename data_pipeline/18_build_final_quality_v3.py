"""Filter quality v2 and create reproducible pair-random and course-random splits.

The only exclusion is an observed user-course whose sole chapter is -1.
Stored v2 indicators, P95 references, q, labels, and cutoffs are preserved.
Unlabeled enrollments remain in the full delivery, with no supervised split.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import duckdb
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
VERSION = 'final-quality-v3-filter-only-unknown-random-splits-2026-10-21'
KEYS = ['user_id', 'course_id']
TARGETS = ['q_alpha_0p1', 'y_alpha_0p1', 'q_alpha_1', 'y_alpha_1']
PHASES = ['P1', 'P2', 'P3', 'P4']
SPLITS = ['train', 'dev', 'test']
EXCLUDE = 'n_chapters_observed=1 AND n_unknown_chapters=1'


def qs(value):
    return "'" + str(value).replace("'", "''") + "'"


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + '.partial')
    partial.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    partial.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def inventory(folder):
    return [[str(p.relative_to(folder)), p.stat().st_size, p.stat().st_mtime_ns]
            for p in sorted(Path(folder).rglob('*')) if p.is_file()]


def records(con, sql):
    result = con.execute(sql)
    names = [x[0] for x in result.description]
    return [dict(zip(names, row)) for row in result.fetchall()]


def view(con, name, path):
    con.execute(f'CREATE OR REPLACE VIEW "{name}" AS SELECT * FROM read_parquet({qs(path)})')


def write(con, path, sql):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix('.parquet.partial')
    con.execute(f'COPY ({sql}) TO {qs(partial)} (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 50000)')
    partial.replace(path)


def connection(out, threads=2, memory='1200MB'):
    scratch = Path(out) / '_scratch'
    scratch.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    for setting in [f'threads={threads}', f'memory_limit={qs(memory)}',
                    f'temp_directory={qs(scratch)}', "max_temp_directory_size='2GB'",
                    "TimeZone='UTC'", 'preserve_insertion_order=false']:
        con.execute('SET ' + setting)
    return con


def make_assignments(frame, seed, grouped=False):
    """Sort before shuffle; pair stratification preserves the joint alpha labels."""
    frame = frame.sort_values(KEYS).reset_index(drop=True)
    if grouped:
        units = np.array(sorted(frame.course_id.unique()))
        train, hold = train_test_split(units, test_size=0.3, random_state=seed, shuffle=True)
        dev, test = train_test_split(hold, test_size=2 / 3, random_state=seed, shuffle=True)
        lookup = {value: name for name, values in [('train', train), ('dev', dev), ('test', test)]
                  for value in values}
        assignment = frame.course_id.map(lookup).to_numpy()
    else:
        strata = (frame.y_alpha_0p1 + '|' + frame.y_alpha_1).to_numpy()
        units = np.arange(len(frame))
        train, hold = train_test_split(units, test_size=0.3, random_state=seed, stratify=strata)
        dev, test = train_test_split(hold, test_size=2 / 3, random_state=seed, stratify=strata[hold])
        assignment = np.empty(len(frame), dtype=object)
        for name, values in [('train', train), ('dev', dev), ('test', test)]:
            assignment[values] = name
    result = frame[KEYS].copy()
    result['split'] = assignment
    result['v2_split'] = frame['split']
    result['joint_label'] = frame.y_alpha_0p1 + '|' + frame.y_alpha_1
    return result


def build(output=None, seed=42, threads=2, memory='1200MB'):
    started = time.monotonic()
    source = ROOT / 'final_quality_v2'
    out = Path(output or ROOT / 'final_quality_v3').resolve()
    if out == source or source in out.parents or out in source.parents:
        raise ValueError('Output must be separate from v2')
    if not (source / '_SUCCESS.json').is_file():
        raise RuntimeError('Quality v2 is incomplete')
    before = inventory(source)
    out.mkdir(parents=True, exist_ok=True)
    (out / '_SUCCESS.json').unlink(missing_ok=True)
    con = connection(out, threads, memory)
    view(con, 'source_final', source / 'tables/user_course_final/*.parquet')
    con.execute(f'''CREATE TEMP TABLE excluded AS SELECT user_id,course_id,split v2_split,
        n_chapters_observed,n_unknown_chapters,{','.join(TARGETS)},
        'ONLY_OBSERVED_CHAPTER_IS_UNKNOWN_MINUS_1' exclusion_reason
        FROM source_final WHERE {EXCLUDE}''')
    write(con, out / 'tables/excluded_user_courses.parquet', 'SELECT * FROM excluded ORDER BY user_id,course_id')
    cohort = con.execute(f'''SELECT user_id,course_id,split,{','.join(TARGETS)}
        FROM source_final WHERE y_alpha_0p1 IS NOT NULL AND NOT ({EXCLUDE})
        ORDER BY user_id,course_id''').df()
    for grouped, name, filename in [
        (False, 'pair_splits', 'user_course_splits.parquet'),
        (True, 'group_splits', 'user_course_splits_course_random.parquet'),
    ]:
        assignment = make_assignments(cohort, seed, grouped)
        if not assignment.equals(make_assignments(cohort, seed, grouped)):
            raise AssertionError('Split is not reproducible')
        con.register('assignment_frame', assignment)
        write(con, out / 'tables' / filename, 'SELECT * FROM assignment_frame ORDER BY user_id,course_id')
        con.unregister('assignment_frame')
        view(con, name, out / 'tables' / filename)
    write(con, out / 'tables/course_random_splits.parquet',
          'SELECT DISTINCT course_id,split FROM group_splits ORDER BY course_id')
    print(f'Excluded {con.execute("SELECT count(*) FROM excluded").fetchone()[0]:,}; labeled kept {len(cohort):,}', flush=True)

    checks = {}

    def check(name, sql):
        checks[name] = bool(con.execute(sql).fetchone()[0])
        if not checks[name]:
            raise AssertionError(name)

    # Filter each complete v2 shard, including its unlabeled rows. Correct nested
    # chapter split metadata as well; never recompute or round a stored score.
    for path in sorted((source / 'tables/user_course_final').glob('*.parquet')):
        view(con, 'source_part', path)
        write(con, out / 'tables/user_course_final' / path.name, '''SELECT f.* REPLACE(
            s.split AS split,
            list_transform(f.chapter_details,c -> struct_update(c,split := s.split)) AS chapter_details)
            FROM source_part f ANTI JOIN excluded e USING(user_id,course_id)
            LEFT JOIN pair_splits s USING(user_id,course_id)''')
        view(con, 'written_part', out / 'tables/user_course_final' / path.name)
        check(path.stem + '_exact_keys_and_scores', f'''SELECT count(*)=0 FROM written_part f
            FULL JOIN (SELECT p.* FROM source_part p ANTI JOIN excluded e USING(user_id,course_id)) v USING(user_id,course_id)
            WHERE f.user_id IS NULL OR v.user_id IS NULL OR
            {' OR '.join('f.'+x+' IS DISTINCT FROM v.'+x for x in TARGETS + ['n_chapters_observed','n_unknown_chapters'])}
            OR f.chapter_details IS DISTINCT FROM list_transform(v.chapter_details,c -> struct_update(c,split := f.split))''')
        check(path.stem + '_unique', 'SELECT count(*)=count(DISTINCT(user_id,course_id)) FROM written_part')
    view(con, 'final', out / 'tables/user_course_final/*.parquet')

    # Pair-grain supporting tables use the same exclusion. Global normalization
    # references retain the full original calibration cohort and are copied intact.
    pair_tables = ['chapter_quality', 'coelo_components', 'afelo_components', 'acelo_components',
                   'afelo_comment_links', 'comment_sentiment', 'user_video_components', 'user_video_ts',
                   'user_course_phase_times']
    for name in pair_tables:
        view(con, 'support_source', source / 'tables' / (name + '.parquet'))
        columns = [x[0] for x in con.execute('DESCRIBE support_source').fetchall()]
        selection = 'p.* REPLACE(s.split AS split)' if 'split' in columns else 'p.*'
        join = 'LEFT JOIN pair_splits s USING(user_id,course_id)' if 'split' in columns else ''
        write(con, out / 'tables' / (name + '.parquet'), f'''SELECT {selection}
            FROM support_source p ANTI JOIN excluded e USING(user_id,course_id) {join}''')
    for name in ['course_time_bounds', 'chapter_comment_totals', 'chapter_p95_reference', 'video_p95_reference']:
        origin = source / 'tables' / (name + '.parquet')
        target = out / 'tables' / origin.name
        shutil.copyfile(origin, target)
        checks[name + '_unchanged'] = sha256(origin) == sha256(target)

    for suffix, alpha in [('0p1', 0.1), ('1', 1.0)]:
        write(con, out / f'variants/alpha_{suffix}/user_course_labels.parquet', f'''SELECT
            user_id,course_id,split,enroll_time,course_end,cutoff_P1,cutoff_P2,cutoff_P3,cutoff_P4,
            n_chapters_observed,n_unknown_chapters,n_chapters_full_components,n_chapters_coelo,n_chapters_afelo,n_chapters_acelo,
            COELO_mean_chapter COELO,AFELO_mean_chapter AFELO,ACELO_mean_chapter ACELO,
            q_alpha_{suffix} q,y_alpha_{suffix} y,label_status,{alpha}::DOUBLE alpha_afelo,formula_version FROM final''')
    view(con, 'chapter', out / 'tables/chapter_quality.parquet')
    check('expected_exclusion', 'SELECT count(*)=129400 FROM excluded')
    check('expected_labeled', 'SELECT count(*)=178598 FROM final WHERE y_alpha_0p1 IS NOT NULL')
    check('all_other_eligible_pairs_preserved', '''SELECT
        (SELECT count(*) FROM final)=(SELECT count(*) FROM source_final)-(SELECT count(*) FROM excluded)''')
    check('no_only_unknown_pair_survives', f'SELECT count(*)=0 FROM final WHERE {EXCLUDE}')
    check('no_excluded_chapter_survives', 'SELECT count(*)=0 FROM chapter JOIN excluded USING(user_id,course_id)')
    check('every_labeled_pair_has_known_chapter', '''SELECT count(*)=0 FROM
        (SELECT user_id,course_id FROM final WHERE y_alpha_0p1 IS NOT NULL) f
        ANTI JOIN (SELECT DISTINCT user_id,course_id FROM chapter WHERE chapter_major<>-1) c USING(user_id,course_id)''')
    check('unlabeled_has_no_supervised_split', 'SELECT count(*)=0 FROM final WHERE y_alpha_0p1 IS NULL AND split IS NOT NULL')
    check('labeled_has_valid_split', "SELECT count(*)=0 FROM final WHERE y_alpha_0p1 IS NOT NULL AND (split IS NULL OR split NOT IN ('train','dev','test'))")
    check('same_availability_both_alphas', 'SELECT count(*)=0 FROM final WHERE (y_alpha_0p1 IS NULL)<>(y_alpha_1 IS NULL)')
    check('mixed_pairs_preserved', 'SELECT count(*)=1862 FROM final WHERE n_unknown_chapters>0 AND n_chapters_observed>1')
    check('nested_splits_match', '''SELECT count(*)=0 FROM final,unnest(chapter_details) t(c)
        WHERE c.split IS DISTINCT FROM final.split''')
    check('course_random_no_course_overlap', '''SELECT count(*)=0 FROM
        (SELECT course_id FROM group_splits GROUP BY 1 HAVING count(DISTINCT split)>1)''')
    check('alternative_splits_same_pairs', '''SELECT count(*)=0 FROM pair_splits p FULL JOIN group_splits g USING(user_id,course_id)
        WHERE p.user_id IS NULL OR g.user_id IS NULL''')
    check('source_v2_unchanged', 'SELECT true')
    checks['source_v2_unchanged'] = before == inventory(source)
    check('joint_stratification_within_two_rows', '''WITH full_counts AS (
        SELECT joint_label,count(*) n FROM pair_splits GROUP BY 1), actual AS (
        SELECT joint_label,split,count(*) n FROM pair_splits GROUP BY 1,2)
        SELECT bool_and(abs(a.n-f.n*CASE split WHEN 'train' THEN 0.7 WHEN 'dev' THEN 0.1 ELSE 0.2 END)<=2)
        FROM actual a JOIN full_counts f USING(joint_label)''')
    summary = {
        'version': VERSION, 'source_eligible_pairs': con.execute('SELECT count(*) FROM source_final').fetchone()[0],
        'excluded_pairs': con.execute('SELECT count(*) FROM excluded').fetchone()[0],
        'eligible_pairs': con.execute('SELECT count(*) FROM final').fetchone()[0], 'labeled_pairs': len(cohort),
        'unlabeled_pairs': con.execute('SELECT count(*) FROM final WHERE y_alpha_0p1 IS NULL').fetchone()[0],
        'chapter_rows': con.execute('SELECT count(*) FROM chapter').fetchone()[0],
        'old_split': records(con, '''SELECT split,count(*) n_labeled,count(DISTINCT course_id) courses
            FROM source_final WHERE y_alpha_0p1 IS NOT NULL GROUP BY 1 ORDER BY 1'''),
        'splits': {}, 'checks_passed': sum(checks.values()), 'checks_total': len(checks)}
    for name, relation in [('user_course_stratified', 'pair_splits'), ('course_random', 'group_splits')]:
        summary['splits'][name] = {
            'counts': records(con, f'SELECT split,count(*) n,count(DISTINCT course_id) courses,count(DISTINCT user_id) users FROM {relation} GROUP BY 1 ORDER BY 1'),
            'classes': {suffix: records(con, f'''SELECT s.split,f.y_alpha_{suffix} y,count(*) n
                FROM {relation} s JOIN final f USING(user_id,course_id) GROUP BY 1,2 ORDER BY 1,2''') for suffix in ['0p1', '1']},
            'joint_classes': records(con, f'SELECT split,joint_label,count(*) n FROM {relation} GROUP BY 1,2 ORDER BY 1,2'),
            'courses_in_multiple_splits': con.execute(f'SELECT count(*) FROM (SELECT course_id FROM {relation} GROUP BY 1 HAVING count(DISTINCT split)>1)').fetchone()[0],
            'users_in_multiple_splits': con.execute(f'SELECT count(*) FROM (SELECT user_id FROM {relation} GROUP BY 1 HAVING count(DISTINCT split)>1)').fetchone()[0]}
    policy = json.loads((source / 'formula_policy.json').read_text())
    policy.update({'dataset_version': VERSION, 'exclusion': EXCLUDE,
                   'normalization': 'Frozen v2 reference cohort; all retained numerical values unchanged',
                   'unlabeled_split': None, 'split_unit_default': 'user-course',
                   'mixed_chapters': 'Keep -1 together with known chapters for retained user-course pairs'})
    dump(out / 'formula_policy.json', policy)
    manifest = {'version': VERSION, 'client_date': '2026-10-21', 'source_root': str(source),
        'source_inventory': before, 'source_sha256': sha256(__file__), 'seed': seed,
        'split_ratios': {'train': 0.7, 'dev': 0.1, 'test': 0.2},
        'pair_split': 'Two-stage sklearn train_test_split; stratify joint (y_alpha_0p1,y_alpha_1); sorted keys',
        'course_split': 'Two-stage random split of sorted 1808 labeled course IDs; not label-stratified',
        'unlabeled': 'Preserved without train/dev/test assignment; NULL targets are never fabricated',
        'preserved': 'v2 q/y, components, P95 reference populations, chapter -1 within mixed pairs, and cutoffs',
        'old_split': 'Course-start ordered 70/10/20 by course; no course overlap',
        'duckdb_version': duckdb.__version__, 'sklearn_version': sklearn.__version__,
        'elapsed_seconds': round(time.monotonic() - started, 3)}
    if not all(checks.values()):
        raise AssertionError('Validation failed')
    dump(out / 'summary.json', summary)
    dump(out / 'validation.json', checks)
    dump(out / 'run_manifest.json', manifest)
    lines = ['# Final quality v3', '',
        'Loại user-course có đúng một chapter quan sát được và chapter đó là -1. Giữ các cặp có chapter xác định được, kể cả cặp có thêm -1.',
        '', f'Đã loại {summary["excluded_pairs"]:,} cặp; giữ {summary["labeled_pairs"]:,} cặp có nhãn và {summary["unlabeled_pairs"]:,} cặp chưa có nhãn.',
        '', 'q, nhãn, các chỉ số và cutoff kế thừa nguyên giá trị v2. Các bảng reference và tổng bình luận chương vẫn mô tả quần thể hiệu chuẩn v2, bao gồm các cặp đã loại.',
        '', 'V2 chia theo course_start ở cấp course. V3 mặc định chia ngẫu nhiên ở cấp user-course, stratify tổ hợp hai nhãn, tỷ lệ 70/10/20, seed 42. Cả bốn phase dùng cùng assignment.',
        '', 'Bản bổ sung chia ngẫu nhiên các course có nhãn, bảo đảm course_id không giao giữa train/dev/test. Một user vẫn có thể xuất hiện ở nhiều tập.',
        '', 'Cặp chưa có nhãn giữ split=NULL và chỉ nằm trong dữ liệu đầy đủ. Không phân lớp giả cho các cặp này.',
        '', '## Các bảng chính', '',
        '- `tables/user_course_final/*.parquet`: mọi cặp đủ thời gian sau lọc; split mặc định theo user-course.',
        '- `tables/user_course_splits.parquet`: assignment stratified theo user-course.',
        '- `tables/user_course_splits_course_random.parquet`: assignment thay thế theo course.',
        '- `tables/course_random_splits.parquet`: một dòng cho mỗi course có nhãn.',
        '- `tables/excluded_user_courses.parquet`: danh sách loại và lý do; không dùng cho huấn luyện.',
        '- `variants/alpha_*/user_course_labels.parquet`: nhãn với assignment mặc định.',
        '', f'Kiểm tra: {sum(checks.values())}/{len(checks)} đạt. Chi tiết split/nhãn trong summary.json.']
    (out / 'BAO_CAO_FINAL_QUALITY_V3.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    dump(out / '_SUCCESS.json', {'version': VERSION, 'labeled_pairs': len(cohort),
                                'checks_passed': sum(checks.values()), 'manifest_sha256': sha256(out / 'run_manifest.json')})
    con.close()
    print(json.dumps({k: summary[k] for k in ['eligible_pairs', 'labeled_pairs', 'excluded_pairs', 'checks_passed']}) , flush=True)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--memory', default='1200MB')
    args = parser.parse_args()
    build(args.output, args.seed, args.threads, args.memory)
