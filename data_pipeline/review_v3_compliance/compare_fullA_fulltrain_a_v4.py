"""Read-only data audit of legacy fullA and CQ V4 FullTrain-A.

Writes aggregate evidence beside this script; does not rebuild either dataset.
Run with the Python environment containing DuckDB used for CQ V4.
"""
from __future__ import annotations

import hashlib
import json
import math
import tempfile
import time
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
OLD = ROOT / 'fullA'
V4 = ROOT / 'phase_features_v4'
NEW = V4 / 'fulltrain_a'
TARGETS = ['q_alpha_0p1', 'y_alpha_0p1', 'q_alpha_1', 'y_alpha_1']


def q(value):
    return "'" + str(value).replace("'", "''") + "'"


def col(value):
    return '"' + value.replace('"', '""') + '"'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def inventory(roots):
    return {str(p.relative_to(ROOT)): [p.stat().st_size, p.stat().st_mtime_ns]
            for root in roots for p in sorted(root.rglob('*')) if p.is_file()}


def main():
    started = time.monotonic()
    before = inventory([OLD, NEW])
    old_config, new_config = load(OLD/'feature_columns.json'), load(NEW/'feature_columns.json')
    old_prep, new_prep = load(OLD/'preprocessing.json'), load(NEW/'preprocessing.json')
    numeric = old_config['raw_numeric_features']
    categorical = new_config['categorical_features']
    result = {'date': '2026-10-22', 'duckdb_version': duckdb.__version__,
              'scope': 'Exact Parquet comparison of existing artifacts; no training or event-log recomputation.',
              'checks': {}, 'schemas': {}, 'populations': {}}

    def check(name, passed, evidence=None):
        result['checks'][name] = {'status': 'PASS' if passed else 'FAIL', 'evidence': evidence}

    with tempfile.TemporaryDirectory(prefix='fullA-v4-audit-') as scratch:
        con = duckdb.connect()
        for setting in ['threads=2', "memory_limit='1200MB'", 'preserve_insertion_order=false',
                        "TimeZone='UTC'", 'temp_directory='+q(scratch)]:
            con.execute('SET '+setting)

        def records(sql):
            cursor = con.execute(sql)
            return [dict(zip([x[0] for x in cursor.description], row)) for row in cursor.fetchall()]

        def scalar(sql):
            return con.execute(sql).fetchone()[0]

        def view(name, path):
            con.execute(f'CREATE TEMP VIEW {name} AS SELECT * FROM read_parquet({q(path)})')

        def mismatch(left, right, columns, keys='user_id,course_id'):
            expressions = ['count(*) joined_rows',
                           'count(*) FILTER(WHERE a.user_id IS NULL OR b.user_id IS NULL) missing_key_rows']
            expressions += [f'count(*) FILTER(WHERE a.{col(x)} IS DISTINCT FROM b.{col(x)}) {col(x)}'
                            for x in columns]
            return records('SELECT '+','.join(expressions)+f' FROM {left} a FULL JOIN {right} b USING({keys})')[0]

        for name, path in [('canonical', OLD/'fullA.parquet'), ('old_raw', OLD/'supervised_raw/*.parquet'),
                           ('old_ready', OLD/'model_ready/*.parquet'),
                           ('course_random', OLD/'course_random_split/supervised_raw/*.parquet'),
                           ('new_raw', NEW/'train_raw.parquet'), ('new_ready', NEW/'train_ready.parquet'),
                           ('assignment', V4/'tables/labeled_user_course_splits.parquet'),
                           ('pool_train', V4/'supervised_raw/*/train.parquet'),
                           ('static', V4/'tables/course_static_metadata.parquet'),
                           ('members', V4/'course_family_members.parquet')]:
            view(name, path)
        for name in ['canonical', 'old_raw', 'old_ready', 'new_raw', 'new_ready']:
            result['schemas'][name] = {r[0]: r[1] for r in con.execute('DESCRIBE '+name).fetchall()}
        for name in ['canonical', 'old_raw', 'course_random', 'new_raw', 'assignment']:
            result['populations'][name] = records(f'''SELECT count(*) n_rows,
                count(DISTINCT(user_id,course_id)) distinct_pairs,
                count(DISTINCT user_id) n_users,count(DISTINCT course_id) n_courses,
                count(y_alpha_1) n_labeled,
                count(*) FILTER(WHERE user_id IS NULL OR course_id IS NULL) null_keys FROM {name}''')[0]
        for name, pop in result['populations'].items():
            check(name+'_unique_nonnull_pairs', pop['n_rows']==pop['distinct_pairs'] and pop['null_keys']==0, pop)

        con.execute('CREATE TEMP TABLE canonical_labeled AS SELECT * FROM canonical WHERE y_alpha_1 IS NOT NULL')
        canonical_check = mismatch('old_raw', 'canonical_labeled', list(result['schemas']['old_raw']))
        result['legacy_supervised_vs_canonical'] = canonical_check
        check('legacy_supervised_exactly_matches_canonical_labeled',
              all(v==0 for k,v in canonical_check.items() if k!='joined_rows'), canonical_check)
        con.execute('CREATE TEMP TABLE source_for_A AS SELECT o.* FROM old_raw o SEMI JOIN new_raw n USING(user_id,course_id)')
        compared = numeric+TARGETS+['enroll_time','course_end','gender_code']
        result['raw_vs_source'] = mismatch('new_raw', 'source_for_A', compared)
        check('all_56_raw_numeric_targets_times_gender_preserved',
              all(v==0 for k,v in result['raw_vs_source'].items() if k!='joined_rows'), result['raw_vs_source'])
        con.execute("CREATE TEMP VIEW train_assignment AS SELECT * FROM assignment WHERE split='train'")
        assignment_check = mismatch('new_raw', 'train_assignment',
                                    ['enrollment_id','course_family_id','split','enroll_time','course_end']+TARGETS)
        result['new_vs_v4_assignment'] = assignment_check
        check('A_exactly_V4_train_cohort_and_targets',
              all(v==0 for k,v in assignment_check.items() if k!='joined_rows'), assignment_check)
        bad_window = scalar("""SELECT count(*) FROM new_raw WHERE split IS DISTINCT FROM 'train'
            OR phase IS DISTINCT FROM 'FULL_A' OR phase_fraction IS DISTINCT FROM 1.0
            OR cutoff_time IS DISTINCT FROM course_end OR enroll_time>course_end
            OR enrollment_id IS NULL OR row_id IS NULL OR course_family_id IS NULL""")
        check('A_full_window_metadata',bad_window==0,{'bad_rows':bad_window})
        ids = records('SELECT count(*) n,count(DISTINCT enrollment_id) enrollments,count(DISTINCT row_id) row_ids FROM new_raw')[0]
        check('A_unique_enrollment_and_row_ids',len(set(ids.values()))==1,ids)
        bad_row_id = scalar("""SELECT count(*) FROM new_raw WHERE row_id IS DISTINCT FROM
            sha256(to_json(struct_pack(enrollment_id:=enrollment_id,phase:='FULL_A')))""")
        check('A_row_id_formula',bad_row_id==0,{'bad_rows':bad_row_id})
        result['source_split_of_A'] = records('SELECT o.split old_split,count(*) n FROM old_raw o JOIN new_raw n USING(user_id,course_id) GROUP BY 1 ORDER BY 1')
        result['split_transitions'] = {}
        result['split_counts'] = {}
        result['course_and_family_overlap'] = {}
        for name in ['old_raw','course_random','assignment']:
            result['split_counts'][name] = records(f'''SELECT split,count(*) n,count(DISTINCT course_id) courses,
                count(DISTINCT user_id) users FROM {name} GROUP BY 1 ORDER BY 1''')
            result['course_and_family_overlap'][name] = {
                'courses_in_multiple_splits': scalar(f'SELECT count(*) FROM (SELECT course_id FROM {name} GROUP BY 1 HAVING count(DISTINCT split)>1)'),
                'families_in_multiple_splits': scalar(f'''SELECT count(*) FROM (SELECT m.course_family_id
                    FROM {name} r JOIN members m USING(course_id) GROUP BY 1 HAVING count(DISTINCT r.split)>1)''')}
            if name!='assignment':
                result['split_transitions'][name] = records(f'''SELECT o.split old_split,a.split v4_split,count(*) n
                    FROM {name} o JOIN assignment a USING(user_id,course_id) GROUP BY 1,2 ORDER BY 1,2''')
        overlap = records('''SELECT split,
            count(*) FILTER(WHERE enrollment_id IN(SELECT enrollment_id FROM new_raw)) enrollment_overlap,
            count(DISTINCT course_id) FILTER(WHERE course_id IN(SELECT course_id FROM new_raw)) course_overlap,
            count(DISTINCT course_family_id) FILTER(WHERE course_family_id IN(SELECT course_family_id FROM new_raw)) family_overlap
            FROM assignment WHERE split<>'train' GROUP BY 1 ORDER BY 1''')
        result['A_overlap_with_V4_heldout_by_enrollment_course_family'] = overlap
        check('A_no_V4_heldout_enrollment_course_family_overlap',
              all(row[k]==0 for row in overlap for k in ['enrollment_overlap','course_overlap','family_overlap']),overlap)
        result['V4_heldout_user_overlap'] = records('''SELECT split,count(*) n,
            count(*) FILTER(WHERE user_id IN(SELECT user_id FROM new_raw)) known_user_enrollments
            FROM assignment WHERE split<>'train' GROUP BY 1 ORDER BY 1''')
        check('V4_assignment_no_course_or_family_overlap',
              all(x==0 for x in result['course_and_family_overlap']['assignment'].values()),
              result['course_and_family_overlap']['assignment'])
        pool = records('''SELECT count(*) n_rows,count(DISTINCT enrollment_id) enrollments,
            count(DISTINCT phase) phases FROM pool_train''')[0]
        pool_bad = scalar('''SELECT count(*) FROM (SELECT enrollment_id FROM pool_train GROUP BY 1
            HAVING count(*)<>4 OR count(DISTINCT phase)<>4)''')
        cohort_bad = scalar('''SELECT count(*) FROM (SELECT DISTINCT enrollment_id FROM pool_train) p
            FULL JOIN new_raw n USING(enrollment_id) WHERE p.enrollment_id IS NULL OR n.enrollment_id IS NULL''')
        check('A_same_enrollments_as_four_phase_Pool_B_train',pool_bad==0 and cohort_bad==0,
              {'pool':pool,'bad_phase_groups':pool_bad,'missing_cohort_keys':cohort_bad})
        print('Cohort, raw features, targets and split comparisons complete',flush=True)

        result['schema_diff_supervised'] = {
            'added': sorted(result['schemas']['new_raw'].keys()-result['schemas']['old_raw'].keys()),
            'removed': sorted(result['schemas']['old_raw'].keys()-result['schemas']['new_raw'].keys())}
        check('numeric_inventory_is_56_original_plus_phase_fraction',
              [x for x in new_config['raw_numeric_features'] if x!='phase_fraction']==numeric
              and new_config['raw_numeric_features'].count('phase_fraction')==1)
        check('model_allowlist_excludes_metadata_and_targets',
              not(set(new_config['model_feature_columns']) & set(new_config['metadata_columns']+TARGETS)))
        check('selector_preserves_56_numeric_order',new_config['selector_feature_columns']==numeric)
        result['feature_counts'] = {'legacy_numeric':len(numeric),'legacy_categorical':len(categorical),
            'legacy_model':len(old_config['model_feature_columns']),
            'v4_numeric':len(new_config['raw_numeric_features']),
            'v4_model':len(new_config['model_feature_columns']),
            'v4_selector':len(new_config['selector_feature_columns'])}
        result['categorical_code_changes'] = records('SELECT '+','.join(
            f'count(*) FILTER(WHERE n.{x} IS DISTINCT FROM o.{x}) {x}' for x in categorical)+
            ' FROM new_raw n JOIN old_raw o USING(user_id,course_id)')[0]
        result['encoders'] = {}
        for name, signature in [('learner','user_id'),('teacher_set','teacher_signature'),
                                ('school_set','school_signature'),('field_set','field_signature')]:
            view('enc_old_'+name,OLD/'encoders'/f'{name}.parquet')
            view('enc_new_'+name,V4/'encoders'/f'{name}.parquet')
            code = name+'_code'
            stats = records(f'''SELECT count(*) n,
                count(*) FILTER(WHERE o.{code}=0 AND n.{code}<>0) newly_known,
                count(*) FILTER(WHERE o.{code}<>0 AND n.{code}=0) newly_unknown,
                count(*) FILTER(WHERE o.{code}<>0 AND n.{code}<>0 AND eo.token IS DISTINCT FROM en.token) decoded_token_mismatch,
                count(*) FILTER(WHERE n.{code}<>0 AND en.token IS NULL) invalid_new_codes
                FROM new_raw n JOIN old_raw o USING(user_id,course_id)
                LEFT JOIN enc_old_{name} eo ON o.{code}=eo.code
                LEFT JOIN enc_new_{name} en ON n.{code}=en.code''')[0]
            if name=='learner':
                bad = scalar(f'''SELECT count(*) FROM new_raw n LEFT JOIN enc_new_{name} e ON n.user_id=e.token
                    WHERE n.{code} IS DISTINCT FROM coalesce(e.code,0)''')
                token_source = "SELECT DISTINCT user_id token FROM assignment WHERE split='train'"
            else:
                bad = scalar(f'''SELECT count(*) FROM new_raw n LEFT JOIN static s USING(course_id)
                    LEFT JOIN enc_new_{name} e ON s.{signature}=e.token
                    WHERE n.{code} IS DISTINCT FROM coalesce(e.code,0)''')
                token_source = f"SELECT DISTINCT {signature} token FROM static JOIN assignment USING(course_id) WHERE split='train' AND {signature} IS NOT NULL"
            vocab_bad = scalar(f'''SELECT count(*) FROM enc_new_{name} e FULL JOIN ({token_source}) t USING(token)
                WHERE e.token IS NULL OR t.token IS NULL''')
            stats.update({'expected_V4_code_mismatches':bad,'V4_train_vocab_missing_or_extra':vocab_bad,
                          'old_vocab_size':scalar('SELECT count(*) FROM enc_old_'+name),
                          'new_vocab_size':scalar('SELECT count(*) FROM enc_new_'+name)})
            result['encoders'][name] = stats
            check(name+'_remapping_and_train_only_vocabulary',
                  stats['decoded_token_mismatch']==stats['invalid_new_codes']==bad==vocab_bad==0,stats)

        result['preprocessing'] = {'old_fit_rows':old_prep['fit_rows'],'new_fit_rows':new_prep['fit_rows'],
            'new_pool':new_prep['pool'],'changed_parameters':{},'numeric_ready_changes':{}}
        # Recompute the fitted parameters independently from the declared train rows.
        # Parallel aggregate order may differ in the last floating-point bits.
        result['preprocessing']['independent_refit'] = {}
        for name, train_view, prep in [('v4','pool_train',new_prep),
                                        ('legacy','old_raw',old_prep)]:
            where = '' if name=='v4' else " WHERE split='train'"
            params = prep['numeric']
            medians = records('SELECT '+','.join(f'median({x}) {x}' for x in params)+
                              ' FROM '+train_view+where)[0]
            moment_sql = []
            for x in params:
                imputed = f'coalesce({x}::DOUBLE,{medians[x]!r})'
                moment_sql.extend([f'avg({imputed}) {x}_mean',f'stddev_pop({imputed}) {x}_scale'])
            moments = records('SELECT '+','.join(moment_sql)+' FROM '+train_view+where)[0]
            errors = {}
            for x,p in params.items():
                scale = moments[x+'_scale'] if moments[x+'_scale']>1e-12 else 1.0
                expected = {'median':medians[x],'mean':moments[x+'_mean'],'scale':scale}
                errors[x] = {k:abs(expected[k]-p[k]) for k in expected}
            bad = [x for x,p in params.items() if medians[x]!=p['median'] or
                   not math.isclose(moments[x+'_mean'],p['mean'],rel_tol=1e-10,abs_tol=1e-10) or
                   not math.isclose(moments[x+'_scale'] if moments[x+'_scale']>1e-12 else 1.0,
                                    p['scale'],rel_tol=1e-10,abs_tol=1e-10)]
            evidence = {'numeric_columns':len(params),'mismatched_columns':bad,
                        'absolute_errors':errors,'relative_and_absolute_tolerance':1e-10}
            result['preprocessing']['independent_refit'][name] = evidence
            check(name+'_preprocessor_recomputed_from_declared_train_only',not bad,evidence)
        for x in numeric:
            if old_prep['numeric'][x]!=new_prep['numeric'][x]:
                result['preprocessing']['changed_parameters'][x] = {
                    'old':old_prep['numeric'][x], 'new':new_prep['numeric'][x]}
        for raw, ready, params in [('new_raw','new_ready',new_prep['numeric']),
                                   ('old_raw','old_ready',old_prep['numeric'])]:
            preserved = categorical+TARGETS+['enroll_time','course_end','split']
            if raw=='new_raw': preserved += ['row_id','enrollment_id','course_family_id','phase','cutoff_time']
            aligned = mismatch(raw,ready,preserved)
            check(ready+'_metadata_targets_categories_aligned',
                  all(v==0 for k,v in aligned.items() if k!='joined_rows'),aligned)
            expressions = []
            finite = []
            for x,p in params.items():
                expected=f'((coalesce(r.{x}::DOUBLE,{p["median"]!r})-{p["mean"]!r})/{p["scale"]!r})'
                expressions.append(f'count(*) FILTER(WHERE m.{x} IS DISTINCT FROM {expected}) {x}')
                finite.append(f'count(*) FILTER(WHERE m.{x} IS NULL OR NOT isfinite(m.{x})) {x}')
            transform=records('SELECT '+','.join(expressions)+f' FROM {raw} r JOIN {ready} m USING(user_id,course_id)')[0]
            nonfinite=records('SELECT '+','.join(finite)+f' FROM {ready} m')[0]
            check(ready+'_numeric_transformation_exact',not any(transform.values()),transform)
            check(ready+'_numeric_finite',not any(nonfinite.values()),nonfinite)
        result['preprocessing']['numeric_ready_changes'] = records('SELECT '+','.join(
            f'count(*) FILTER(WHERE n.{x} IS DISTINCT FROM o.{x}) {x}' for x in numeric)+
            ' FROM new_ready n JOIN old_ready o USING(user_id,course_id)')[0]
        result['phase_fraction'] = records('''SELECT min(r.phase_fraction) raw_min,max(r.phase_fraction) raw_max,
            min(m.phase_fraction) ready_min,max(m.phase_fraction) ready_max
            FROM new_raw r JOIN new_ready m USING(user_id,course_id)''')[0]
        check('shared_preprocessor_is_byte_identical',digest(NEW/'preprocessing.json')==digest(V4/'preprocessing.json'))
        check('shared_feature_config_is_byte_identical',digest(NEW/'feature_columns.json')==digest(V4/'feature_columns.json'))
        check('train_alias_is_byte_identical',digest(NEW/'train.parquet')==digest(NEW/'train_ready.parquet'))
        check('runtime_A_is_byte_identical',digest(NEW/'train_ready.parquet')==digest(V4/'runtime_dev_only/fulltrain_a/train.parquet'))
        manifest=load(NEW/'run_manifest.json')
        check('source_manifest_hash_matches_current_fullA',manifest['source_manifest_sha256']==digest(OLD/'run_manifest.json'))
        check('preprocessor_fit_rows_match_four_phase_train',new_prep['fit_rows']==pool['n_rows'])
        result['class_counts'] = {}
        for name in ['old_raw','new_raw']:
            result['class_counts'][name] = {alpha:records(f'''SELECT split,y_alpha_{alpha} y,count(*) n,
                count(*)::DOUBLE/sum(count(*)) OVER(PARTITION BY split) fraction
                FROM {name} GROUP BY 1,2 ORDER BY 1,2''') for alpha in ['0p1','1']}
        result['raw_missingness'] = records('SELECT '+','.join(f'count(*) FILTER(WHERE {x} IS NULL) {x}'
                                                              for x in new_config['raw_numeric_features'])+' FROM new_raw')[0]
        result['A_raw_zero_activity_rows'] = scalar('SELECT count(*) FROM new_raw WHERE n_events=0')
        result['hashes'] = {str(p.relative_to(ROOT)):digest(p) for p in sorted(NEW.glob('*')) if p.is_file()}
        check('source_directories_unchanged',before==inventory([OLD,NEW]))
        con.close()

    result['checks_passed']=sum(v['status']=='PASS' for v in result['checks'].values())
    result['checks_total']=len(result['checks'])
    result['status']='PASS' if result['checks_passed']==result['checks_total'] else 'FAIL'
    result['elapsed_seconds']=round(time.monotonic()-started,2)
    output=OUT/'fullA_fulltrain_a_v4_comparison.json'
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2,default=str)+'\n')
    print(json.dumps({k:result[k] for k in ['status','checks_passed','checks_total','elapsed_seconds']},ensure_ascii=False))
    print('Evidence: '+str(output))


if __name__=='__main__':
    main()
