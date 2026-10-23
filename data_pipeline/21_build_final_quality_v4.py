"""Build final_quality_v4 from frozen V3 labels and a global content-family split."""
from __future__ import annotations

import argparse
import json
import shutil
import time
from collections import Counter

import numpy as np
import pandas as pd

from cq_v4_common import *


def family_tables(con, threshold, cohort):
    sets = {course: set() for course in cohort.course_id}
    for course, ccid in con.execute('SELECT course_id,ccid FROM course_ccid ORDER BY 1,2').fetchall():
        sets[course].add(ccid)
    mapping, edges = course_families(sets, threshold)
    members = cohort.copy()
    members['course_family_id'] = members.course_id.map(mapping)
    members['n_ccid'] = members.course_id.map(lambda x: len(sets[x]))
    sizes = Counter(mapping.values())
    members['family_size_courses'] = members.course_family_id.map(sizes)
    members['mapping_status'] = members.apply(lambda r: 'singleton_no_ccid' if r.n_ccid == 0 else
        'singleton_lt5_ccid' if r.n_ccid < 5 else 'jaccard_component' if r.family_size_courses > 1 else 'singleton_no_match', axis=1)
    summary = members.groupby('course_family_id', sort=True).agg(
        family_start=('course_start', 'min'), family_end=('course_end', 'max'),
        n_courses=('course_id', 'size'), n_enroll_labeled=('n_enroll_labeled', 'sum'),
        n_E=('n_E', 'sum'), n_G=('n_G', 'sum'), n_A=('n_A', 'sum')).reset_index()
    summary['has_E'] = summary.n_E > 0
    members = members.merge(summary[['course_family_id', 'family_start', 'family_end']], on='course_family_id', validate='many_to_one')
    members = members.rename(columns={'family_start': 'family_min_course_start', 'family_end': 'family_max_course_end'})
    return members.sort_values('course_id'), summary, pd.DataFrame(edges, columns=['course_id_a', 'course_id_b', 'intersection', 'union_size', 'jaccard'])


def build(threads=2, memory='1200MB', seed=42):
    started = time.monotonic()
    source, out = ROOT / 'final_quality_v3', ROOT / 'final_quality_v4'
    out.mkdir(exist_ok=True)
    (out / '_SUCCESS.json').unlink(missing_ok=True)
    before = source_inventory(source)
    con = connect(out, threads, memory)
    checks = Checks()
    view(con, 'source_final', source / 'tables/user_course_final/*.parquet')
    con.execute('''CREATE TEMP TABLE labeled AS SELECT user_id,course_id,enroll_time,course_start,course_end,
        cutoff_P1,cutoff_P2,cutoff_P3,cutoff_P4,q_alpha_0p1,y_alpha_0p1,q_alpha_1,y_alpha_1,
        sha256(to_json(struct_pack(user_id:=user_id,course_id:=course_id))) enrollment_id
        FROM source_final WHERE y_alpha_1 IS NOT NULL''')
    cohort = con.execute('''SELECT course_id,min(course_start) course_start,max(course_end) course_end,
        count(*) n_enroll_labeled,count(*) FILTER(WHERE y_alpha_1='E') n_E,
        count(*) FILTER(WHERE y_alpha_1='G') n_G,count(*) FILTER(WHERE y_alpha_1='A') n_A
        FROM labeled GROUP BY 1 ORDER BY 1''').df()
    checks.add('v3_labeled_cohort_unchanged_expected', len(cohort) == 1808 and cohort.n_enroll_labeled.sum() == 178598)
    view(con, 'resources', ROOT / 'input/moocubex_model/processed/dim_lo_resource/*.parquet')
    view(con, 'ccid_map', ROOT / 'input/moocubex_parquet/tables/relations/video_id_ccid/*.parquet')
    con.execute('''CREATE TEMP TABLE course_videos AS SELECT DISTINCT r.course_id,r.resource_id video_id
        FROM resources r JOIN (SELECT DISTINCT course_id FROM labeled) c USING(course_id)
        WHERE starts_with(r.resource_id,'V_')''')
    con.execute('''CREATE TEMP TABLE course_ccid AS SELECT DISTINCT c.course_id,m.ccid
        FROM course_videos c JOIN ccid_map m USING(video_id)
        WHERE m.ccid IS NOT NULL AND trim(m.ccid)<>'' ''')
    members, families, edges = family_tables(con, 0.9, cohort)
    other_members, other_families, other_edges = family_tables(con, 0.8, cohort)
    assignment = family_split(families, seed)
    checks.add('split_reproducible_from_same_input', assignment.equals(family_split(families, seed)))
    other_assignment = family_split(other_families, seed)
    for name, frame in [('course_family_members', members), ('course_family_summary', families),
                        ('course_family_edges', edges), ('family_splits', assignment),
                        ('sensitivity_0p8_family_members', other_members), ('sensitivity_0p8_family_splits', other_assignment)]:
        con.register('frame', frame)
        write(con, out / f'{name}.parquet', 'SELECT * FROM frame')
        con.unregister('frame')
    view(con, 'members', out / 'course_family_members.parquet')
    view(con, 'families', out / 'course_family_summary.parquet')
    view(con, 'family_splits', out / 'family_splits.parquet')
    write(con, out / 'course_splits.parquet', '''SELECT m.course_id,m.course_family_id,s.split
        FROM members m JOIN family_splits s USING(course_family_id) ORDER BY course_id''')
    view(con, 'course_splits', out / 'course_splits.parquet')
    write(con, out / 'tables/user_course_splits.parquet', '''SELECT l.*,s.course_family_id,s.split
        FROM labeled l JOIN course_splits s USING(course_id) ORDER BY user_id,course_id''')
    view(con, 'assignment', out / 'tables/user_course_splits.parquet')
    split_policy = {
        'dataset_version': VERSION, 'created_date': DATE, 'seed': seed,
        'family': {'minimum_ccid': 5, 'jaccard': 0.9, 'graph': 'connected components',
                   'candidate_search': 'global shared ccid; no title filter', 'ambiguous_video_mapping': 'all distinct non-empty ccid relations retained'},
        'test': 'last ceil(0.2*N_families) by family_start, family_id',
        'train_dev': 'review reference: qcut(rank(method=first),4) x has_E; round(0.125*n_stratum), numpy default_rng seed42',
        'small_strata': 'round per stratum including Python ties-to-even; no manual movement; fail if any global split empty',
        'primary_target': 'y_alpha_1',
        'source_hashes': {'source_manifest': sha256(source / 'run_manifest.json'),
                          'builder': sha256(__file__), 'common': sha256(ROOT / 'cq_v4_common.py')},
        'assignment': dict(zip(assignment.course_family_id, assignment.split)),
        'course_assignment': {r.course_id: {'course_family_id': r.course_family_id, 'split': r.split}
                              for r in members.merge(assignment[['course_family_id','split']], on='course_family_id').itertuples()},
    }
    split_policy['assignment_sha256'] = canonical_hash(split_policy['assignment'])
    dump(out / 'splits.json', split_policy)
    print(f'Families: {len(families)}; edges: {len(edges)}; split locked before features', flush=True)

    for path in sorted((source / 'tables/user_course_final').glob('*.parquet')):
        view(con, 'source_part', path)
        write(con, out / 'tables/user_course_final' / path.name, '''SELECT f.* REPLACE(
            a.split AS split,list_transform(f.chapter_details,c -> struct_update(c,split:=a.split)) AS chapter_details),
            m.course_family_id,a.enrollment_id
            FROM source_part f LEFT JOIN members m USING(course_id) LEFT JOIN assignment a USING(user_id,course_id)''')
        view(con, 'part', out / 'tables/user_course_final' / path.name)
        original_columns = [x[0] for x in con.execute('DESCRIBE source_part').fetchall() if x[0] not in ['split', 'chapter_details']]
        checks.sql(con, path.stem + '_all_original_values_preserved', f'''SELECT count(*)=0 FROM part f FULL JOIN source_part v USING(user_id,course_id)
            WHERE f.user_id IS NULL OR v.user_id IS NULL OR {' OR '.join('f.'+x+' IS DISTINCT FROM v.'+x for x in original_columns)}
            OR f.chapter_details IS DISTINCT FROM list_transform(v.chapter_details,c -> struct_update(c,split:=f.split))''')
        checks.sql(con, path.stem + '_keys_unique', 'SELECT count(*)=count(DISTINCT(user_id,course_id)) FROM part')
    view(con, 'final', out / 'tables/user_course_final/*.parquet')
    pair_tables = ['chapter_quality', 'coelo_components', 'afelo_components', 'acelo_components',
                   'afelo_comment_links', 'comment_sentiment', 'user_video_components', 'user_video_ts', 'user_course_phase_times']
    for name in pair_tables:
        view(con, 'support', source / 'tables' / (name + '.parquet'))
        columns = [x[0] for x in con.execute('DESCRIBE support').fetchall()]
        selection = 'p.* REPLACE(a.split AS split)' if 'split' in columns else 'p.*'
        write(con, out / 'tables' / (name + '.parquet'), f'''SELECT {selection},m.course_family_id
            FROM support p LEFT JOIN members m USING(course_id) LEFT JOIN assignment a USING(user_id,course_id)''')
    for name in ['course_time_bounds', 'chapter_comment_totals', 'chapter_p95_reference', 'video_p95_reference', 'excluded_user_courses']:
        shutil.copyfile(source / 'tables' / (name + '.parquet'), out / 'tables' / (name + '.parquet'))
        checks.add(name + '_byte_identical_v3', sha256(source / 'tables' / (name + '.parquet')) == sha256(out / 'tables' / (name + '.parquet')))
    for suffix in ['0p1', '1']:
        write(con, out / f'variants/alpha_{suffix}/user_course_labels.parquet', f'''SELECT user_id,course_id,course_family_id,enrollment_id,split,
            enroll_time,course_start,course_end,cutoff_P1,cutoff_P2,cutoff_P3,cutoff_P4,
            q_alpha_{suffix} q,y_alpha_{suffix} y,formula_version FROM final''')
        checks.sql(con, 'thresholds_' + suffix, f'''SELECT count(*)=0 FROM final WHERE y_alpha_{suffix} IS DISTINCT FROM
            CASE WHEN q_alpha_{suffix}<0.2 THEN 'E' WHEN q_alpha_{suffix}<0.4 THEN 'G' WHEN q_alpha_{suffix}>=0.4 THEN 'A' END''')
    checks.sql(con, 'full_population_preserved', 'SELECT count(*)=10455017 AND count(y_alpha_1)=178598 FROM final')
    checks.sql(con, 'labeled_has_family_split', 'SELECT count(*)=0 FROM final WHERE y_alpha_1 IS NOT NULL AND (course_family_id IS NULL OR split IS NULL OR enrollment_id IS NULL)')
    checks.sql(con, 'unlabeled_not_fabricated', 'SELECT count(*)=0 FROM final WHERE y_alpha_1 IS NULL AND (split IS NOT NULL OR enrollment_id IS NOT NULL)')
    checks.sql(con, 'no_family_leakage', 'SELECT count(*)=0 FROM (SELECT course_family_id FROM assignment GROUP BY 1 HAVING count(DISTINCT split)>1)')
    checks.sql(con, 'no_course_leakage', 'SELECT count(*)=0 FROM (SELECT course_id FROM assignment GROUP BY 1 HAVING count(DISTINCT split)>1)')
    checks.sql(con, 'enrollment_hashes_unique', 'SELECT count(*)=count(DISTINCT enrollment_id) FROM assignment')
    checks.sql(con, 'all_courses_once', 'SELECT count(*)=1808 AND count(DISTINCT course_id)=1808 FROM members')
    checks.sql(con, 'test_size_correct', 'SELECT count(*) FILTER(WHERE split=\'test\')=ceil(0.2*count(*)) FROM family_splits')
    checks.sql(con, 'test_chronological', "SELECT (SELECT min(family_start) FROM family_splits WHERE split='test') >= (SELECT max(family_start) FROM family_splits WHERE split<>'test')")
    first_test = assignment.loc[assignment.split == 'test', 'family_start'].min()
    overlap = assignment[(assignment.split == 'train') & (assignment.family_end > first_test)]
    overlap_info = {'first_test_family_start': str(first_test), 'numerator': len(overlap),
                    'denominator': int((assignment.split == 'train').sum()),
                    'fraction': len(overlap) / int((assignment.split == 'train').sum()),
                    'overlap_families': overlap.course_family_id.tolist()}
    overlap_info['requires_nolfo'] = overlap_info['fraction'] > 0.05
    dump(out / 'temporal_overlap.json', overlap_info)
    main_groups = {k: frozenset(g.course_id) for k,g in members.groupby('course_family_id')}
    other_groups = {k: frozenset(g.course_id) for k,g in other_members.groupby('course_family_id')}
    changed = sum(main_groups[r.course_family_id] != other_groups[other_members.set_index('course_id').loc[r.course_id,'course_family_id']] for r in members.itertuples())
    sensitivity = {}
    for threshold, mm, ff, ee, ss in [(0.9,members,families,edges,assignment), (0.8,other_members,other_families,other_edges,other_assignment)]:
        joined = mm.merge(ss[['course_family_id','split']], on='course_family_id')
        sensitivity[str(threshold)] = {'families': len(ff), 'multi_course_families': int((ff.n_courses>1).sum()),
            'max_family_courses': int(ff.n_courses.max()), 'edges': len(ee),
            'family_size_distribution': {str(k): int(v) for k,v in ff.n_courses.value_counts().sort_index().items()},
            'split_counts': joined.groupby('split').agg(courses=('course_id','size'),enrollments=('n_enroll_labeled','sum'),E=('n_E','sum'),G=('n_G','sum'),A=('n_A','sum')).reset_index().to_dict('records')}
    sensitivity['courses_changed_component_membership'] = changed
    dump(out / 'family_sensitivity.json', sensitivity)
    (out / 'COURSE_FAMILY_SENSITIVITY.md').write_text('# Độ nhạy family V4\n\nNgưỡng chính 0,9; 0,8 chỉ dùng đối chiếu, không thay split chính.\n\n' +
        '| Jaccard | Số họ | Họ nhiều course | Số cạnh | Họ lớn nhất |\n|---|---:|---:|---:|---:|\n' +
        ''.join(f'| {t} | {sensitivity[t]["families"]} | {sensitivity[t]["multi_course_families"]} | {sensitivity[t]["edges"]} | {sensitivity[t]["max_family_courses"]} |\n' for t in ['0.9','0.8']) +
        f'\nCó {changed} course đổi tập thành viên cùng họ. Thống kê split/class ở `family_sensitivity.json`.\n')
    summary = {'dataset_version': VERSION, 'date': DATE, 'eligible_pairs': 10455017, 'labeled_pairs': 178598,
        'unlabeled_pairs': 10276419, 'families': len(families), 'labeled_courses': len(members),
        'family_edges': len(edges), 'mapping_status': members.mapping_status.value_counts().to_dict(),
        'split_counts': records(con, '''SELECT split,count(*) n_enrollments,count(DISTINCT course_id) n_courses,
            count(DISTINCT course_family_id) n_families,count(DISTINCT user_id) n_users FROM assignment GROUP BY 1 ORDER BY 1'''),
        'class_counts': records(con, 'SELECT split,y_alpha_1 y,count(*) n FROM assignment GROUP BY 1,2 ORDER BY 1,2'),
        'strata': records(con, "SELECT stratum,count(*) n_families,count(*) FILTER(WHERE split='dev') n_dev FROM family_splits WHERE split<>'test' GROUP BY 1 ORDER BY 1"),
        'temporal_overlap': overlap_info, 'sensitivity': sensitivity, 'checks_passed': len(checks.items),
        'ccid_mapping': records(con, '''SELECT (SELECT count(*) FROM course_videos) course_video_pairs,
            (SELECT count(*) FROM course_ccid) course_ccid_pairs,
            (SELECT count(*) FROM (SELECT video_id FROM ccid_map GROUP BY 1 HAVING count(DISTINCT ccid)>1)) ambiguous_video_ids_in_source''')[0]}
    policy = json.loads((source / 'formula_policy.json').read_text())
    policy.update({'dataset_version': VERSION, 'split_unit_default': 'course_family_id', 'primary_target': 'y_alpha_1',
                   'feature_policy': 'user supplied features; TEMPO-DQ comparison not required by user',
                   'course_quality_population': 'retained labeled enrollments only'})
    dump(out / 'formula_policy.json', policy)
    checks.add('source_v3_files_unchanged', before == source_inventory(source))
    summary['checks_passed'] = len(checks.items)
    dump(out / 'summary.json', summary)
    checks.save(out / 'validation.json')
    manifest = {'dataset_version': VERSION, 'created_date': DATE, 'stage': 'final_quality', 'seed': seed,
        'source': str(source), 'source_inventory_sha256': canonical_hash(before),
        'source_manifest_sha256': sha256(source / 'run_manifest.json'),
        'builder_sha256': sha256(__file__), 'common_sha256': sha256(ROOT / 'cq_v4_common.py'),
        'split_sha256': sha256(out / 'splits.json'), 'assignment_sha256': split_policy['assignment_sha256'],
        'preserved': 'All V3 scalar values, targets, chapters and cutoffs; only split and new metadata differ',
        'duckdb_version': duckdb.__version__, 'numpy_version': np.__version__, 'pandas_version': pd.__version__,
        'elapsed_seconds': round(time.monotonic()-started,3)}
    dump(out / 'run_manifest.json', manifest)
    dump(out / '_SUCCESS.json', {'dataset_version': VERSION, 'scope': 'final-quality data build only',
        'checks_passed': len(checks.items), 'manifest_sha256': sha256(out / 'run_manifest.json')})
    con.close()
    checksums(out)
    print(json.dumps({k:summary[k] for k in ['families','labeled_pairs','split_counts','checks_passed']}, default=str), flush=True)
    return summary


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--threads',type=int,default=2);p.add_argument('--memory',default='1200MB');p.add_argument('--seed',type=int,default=42)
    a=p.parse_args();build(a.threads,a.memory,a.seed)
