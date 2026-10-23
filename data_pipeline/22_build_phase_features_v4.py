"""Build V4 raw/ready phases, shared train preprocessing, FullTrain-A and dev-only delivery."""
from __future__ import annotations

import argparse
import json
import math
import shutil
import time

from cq_v4_common import *

META = ['user_id','course_id','course_family_id','enrollment_id','row_id','phase','split','enroll_time','cutoff_time','course_end']


def row_id(phase):
    return f"sha256(to_json(struct_pack(enrollment_id:=a.enrollment_id,phase:={phase})))"


def stats(con, out, quality):
    """Course endpoints use one row per retained labeled enrollment, not snapshots."""
    view(con, 'assigned', quality / 'tables/user_course_splits.parquet')
    view(con, 'full_final', quality / 'tables/user_course_final/*.parquet')
    write(con, out / 'tables/course_level_targets.parquet', '''WITH coverage AS (
        SELECT course_id,count(*) n_eligible FROM full_final GROUP BY 1)
        SELECT a.split,a.course_id,a.course_family_id,count(*) n_c,max(c.n_eligible) n_eligible,
        count(*)::DOUBLE/max(c.n_eligible) labeled_fraction,avg(q_alpha_1) Q_c,
        CASE WHEN avg(q_alpha_1)<0.2 THEN 'E' WHEN avg(q_alpha_1)<0.4 THEN 'G' ELSE 'A' END Y_c,
        avg((y_alpha_1='E')::DOUBLE) pi_E,avg((y_alpha_1='G')::DOUBLE) pi_G,avg((y_alpha_1='A')::DOUBLE) pi_A,
        avg(q_alpha_1) FILTER(WHERE y_alpha_1='E') mu_E_course,
        avg(q_alpha_1) FILTER(WHERE y_alpha_1='G') mu_G_course,
        avg(q_alpha_1) FILTER(WHERE y_alpha_1='A') mu_A_course
        FROM assigned a JOIN coverage c USING(course_id) GROUP BY 1,2,3 ORDER BY 1,2''')
    view(con, 'course_targets', out / 'tables/course_level_targets.parquet')
    mu = {r['y']: r['mu'] for r in records(con, "SELECT y_alpha_1 y,avg(q_alpha_1) mu FROM assigned WHERE split='train' GROUP BY 1")}
    if not (0<=mu['E']<0.2<=mu['G']<0.4<=mu['A']<=1):
        raise AssertionError('Invalid train class means')
    constants = {'dataset_version': VERSION, 'target': 'y_alpha_1', 'score': 'q_alpha_1', 'mu': mu,
        'mu_fit_population': 'one row per retained labeled train enrollment', 'nmin': 30, 'delta_C': 0.01,
        'delta': 0.005, 'class_weight_policy': 'natural', 'course_population': 'retained labeled enrollments',
        'estimator': 'PCC: Q_hat=sum(mean_probability_y * mu_y)',
        'course_endpoint': 'EC and signed error; course-class Macro-F1 not a primary endpoint'}
    dump(out / 'protocol_constants.json', constants)
    write(con, out / 'tables/course_oracle_diagnostics.parquet', f'''SELECT *,n_c>=30 eligible_ec,
        pi_E*{mu['E']!r}+pi_G*{mu['G']!r}+pi_A*{mu['A']!r} oracle_Q_hat,
        pi_E*{mu['E']!r}+pi_G*{mu['G']!r}+pi_A*{mu['A']!r}-Q_c oracle_signed_error,
        abs(pi_E*{mu['E']!r}+pi_G*{mu['G']!r}+pi_A*{mu['A']!r}-Q_c) oracle_abs_error
        FROM course_targets''')
    view(con, 'oracle', out / 'tables/course_oracle_diagnostics.parquet')
    result = {'course_sizes': records(con, '''SELECT split,count(*) n_courses,min(n_c) min_n,
        quantile_cont(n_c,0.25) p25_n,median(n_c) median_n,quantile_cont(n_c,0.75) p75_n,max(n_c) max_n,
        count(*) FILTER(WHERE n_c>=30) n_courses_ec,count(*) FILTER(WHERE n_c<30) n_courses_below_30,
        count(DISTINCT course_family_id) FILTER(WHERE n_c>=30) n_families_with_ec_course,
        stddev_pop(pi_E) sd_pi_E,stddev_pop(pi_G) sd_pi_G,stddev_pop(pi_A) sd_pi_A
        FROM course_targets GROUP BY 1 ORDER BY 1'''),
        'course_classes': records(con, 'SELECT split,Y_c,count(*) n,count(*) FILTER(WHERE n_c>=30) n_ec FROM course_targets GROUP BY 1,2 ORDER BY 1,2'),
        'oracle_floor': records(con, 'SELECT split,count(*) n_courses,avg(oracle_abs_error) floor,avg(oracle_signed_error) signed FROM oracle WHERE eligible_ec GROUP BY 1 ORDER BY 1'),
        'concentration_course': records(con, '''SELECT split,count(*) n_courses,
            max(n_c)::DOUBLE/sum(n_c) largest_share,pow(sum(n_c)::DOUBLE,2)/sum(pow(n_c::DOUBLE,2)) effective_courses
            FROM course_targets GROUP BY 1 ORDER BY 1'''),
        'concentration_family': records(con, '''SELECT split,count(*) n_families,max(n)::DOUBLE/sum(n) largest_share,
            pow(sum(n)::DOUBLE,2)/sum(pow(n::DOUBLE,2)) effective_families
            FROM (SELECT split,course_family_id,sum(n_c) n FROM course_targets GROUP BY 1,2) GROUP BY 1 ORDER BY 1'''),
        'test_user_old_new': records(con, '''SELECT count(*) n_test_enrollments,
            count(*) FILTER(WHERE user_id IN(SELECT user_id FROM assigned WHERE split='train')) old_user_enrollments,
            count(*) FILTER(WHERE user_id NOT IN(SELECT user_id FROM assigned WHERE split='train')) new_user_enrollments
            FROM assigned WHERE split='test' ''')[0],
        'icc1': []}
    for split in SPLITS + ['all']:
        condition = '' if split == 'all' else ' WHERE split=' + qs(split)
        con.execute('CREATE OR REPLACE TEMP VIEW icc_rows AS SELECT * FROM assigned' + condition)
        con.execute('''CREATE OR REPLACE TEMP VIEW icc_groups AS SELECT course_id,count(*) n,avg(q_alpha_1) mean_q
            FROM icc_rows GROUP BY 1''')
        v = records(con, '''SELECT (SELECT count(*) FROM icc_rows) N,(SELECT count(*) FROM icc_groups) K,
            (SELECT sum(n*pow(mean_q-(SELECT avg(q_alpha_1) FROM icc_rows),2)) FROM icc_groups) ssb,
            (SELECT sum(pow(q_alpha_1-mean_q,2)) FROM icc_rows JOIN icc_groups USING(course_id)) ssw,
            (SELECT sum(pow(n::DOUBLE,2)) FROM icc_groups) n_squared''')[0]
        msb, msw = v['ssb']/(v['K']-1), v['ssw']/(v['N']-v['K'])
        n0=(v['N']-v['n_squared']/v['N'])/(v['K']-1)
        result['icc1'].append({'split':split,'N':v['N'],'K':v['K'],'icc1':(msb-msw)/(msb+(n0-1)*msw)})
    view(con, 'teachers', ROOT / 'input/moocubex_parquet/tables/relations/course_teacher/*.parquet')
    result['test_teacher_old_new'] = records(con, '''WITH train_teachers AS (
        SELECT DISTINCT teacher_id FROM teachers JOIN (SELECT DISTINCT course_id FROM assigned WHERE split='train') t USING(course_id)),
        test_courses AS (SELECT course_id,count(*) n_enroll FROM assigned WHERE split='test' GROUP BY 1),
        course_teacher AS (SELECT t.course_id,max(t.n_enroll) n_enroll,count(DISTINCT teacher_id) n_teachers,
            count(DISTINCT teacher_id) FILTER(WHERE teacher_id IN(SELECT teacher_id FROM train_teachers)) n_old
            FROM test_courses t LEFT JOIN teachers USING(course_id) GROUP BY 1)
        SELECT sum(n_enroll) n_test_enrollments,
            coalesce(sum(n_enroll) FILTER(WHERE n_teachers=0),0) no_teacher_metadata,
            coalesce(sum(n_enroll) FILTER(WHERE n_teachers>0 AND n_old=0),0) all_teachers_new,
            coalesce(sum(n_enroll) FILTER(WHERE n_teachers>0 AND n_old=n_teachers),0) all_teachers_old,
            coalesce(sum(n_enroll) FILTER(WHERE n_old>0 AND n_old<n_teachers),0) mixed_old_new
        FROM course_teacher''')[0]
    dump(out / 'course_statistics.json', result)
    return result


LOADER = '''"""Read CQ V4 using fixed feature order; never fit preprocessing."""
from pathlib import Path
import json
_CONNECTION = None

def _root(root):
    return Path(root).resolve() if root is not None else Path(__file__).resolve().parent

def feature_columns(view="learner", root=None):
    if view not in {"learner", "selector"}:
        raise ValueError("view must be learner or selector")
    return json.loads((_root(root) / (view + "_feature_columns.json")).read_text())["columns"]

def load_phase(phase="P1", split="train", ready=True, view="learner", root=None, connection=None):
    global _CONNECTION
    import duckdb
    root = _root(root)
    if phase not in {"P1", "P2", "P3", "P4", None} or split not in {"train", "dev", "test"}:
        raise ValueError("invalid phase/split")
    runtime = (root / "bundle_manifest.json").exists()
    if runtime:
        if split == "test":
            raise PermissionError("This dev-only bundle has no test access")
        if not ready:
            raise ValueError("The runtime bundle provides ready data; raw is in the private phase root")
        if split == "dev" and phase is None:
            raise ValueError("Select a dev phase")
        path = root / "pool_b" / ("train.parquet" if split == "train" else "dev_phase" + phase[-1] + ".parquet")
    else:
        base = "model_ready" if ready else "supervised_raw"
        path = root / base / (phase or "*") / (split + ".parquet")
    if connection is None:
        if _CONNECTION is None:
            _CONNECTION = duckdb.connect()
        connection = _CONNECTION
    columns = feature_columns(view, root)
    metadata = ["row_id","enrollment_id","user_id","course_id","course_family_id","phase","split","enroll_time","cutoff_time","course_end"]
    select = ",".join('"' + x + '"' for x in metadata + columns)
    relation = connection.sql("SELECT " + select + ",q_alpha_1 q,y_alpha_1 y FROM read_parquet('" + str(path).replace("'","''") + "')")
    return relation.filter("phase='" + phase + "'") if phase else relation

def load_fullA(ready=True, view="learner", root=None, connection=None):
    global _CONNECTION
    import duckdb
    root = _root(root)
    runtime = (root / "bundle_manifest.json").exists()
    if runtime and not ready:
        raise ValueError("Runtime bundle has ready FullTrain-A only")
    path = root / "fulltrain_a" / ("train.parquet" if runtime else "train_" + ("ready" if ready else "raw") + ".parquet")
    if connection is None:
        if _CONNECTION is None:
            _CONNECTION = duckdb.connect()
        connection = _CONNECTION
    columns = feature_columns(view, root)
    meta = ["row_id","enrollment_id","user_id","course_id","course_family_id","phase","split","enroll_time","cutoff_time","course_end"]
    return connection.sql("SELECT " + ",".join('"'+x+'"' for x in meta+columns) + ",q_alpha_1 q,y_alpha_1 y FROM read_parquet('" + str(path).replace("'","''") + "')")
'''


def build(threads=2, memory='1200MB'):
    started=time.monotonic()
    source, quality, out = ROOT/'phase_features_v3', ROOT/'final_quality_v4', ROOT/'phase_features_v4'
    if not (quality/'_SUCCESS.json').exists():
        raise RuntimeError('Build final_quality_v4 first')
    out.mkdir(exist_ok=True);(out/'tables').mkdir(exist_ok=True);(out/'_SUCCESS.json').unlink(missing_ok=True)
    before=source_inventory(source)
    con=connect(out,threads,memory);checks=Checks()
    view(con,'assignment',quality/'tables/user_course_splits.parquet')
    view(con,'members',quality/'course_family_members.parquet')
    view(con,'course_times',quality/'tables/course_time_bounds.parquet')
    view(con,'final',quality/'tables/user_course_final/*.parquet')
    view(con,'course_static',source/'tables/course_static_metadata.parquet')
    shutil.copyfile(source/'tables/course_static_metadata.parquet',out/'tables/course_static_metadata.parquet')
    write(con,out/'tables/user_course_targets.parquet',f'''SELECT user_id,course_id,course_family_id,enrollment_id,split,{','.join(TARGETS)} FROM final''')
    shutil.copyfile(quality/'tables/user_course_splits.parquet',out/'tables/labeled_user_course_splits.parquet')
    for name in ['course_family_members.parquet','course_family_summary.parquet','family_splits.parquet','course_splits.parquet','splits.json','temporal_overlap.json','COURSE_FAMILY_SENSITIVITY.md']:
        shutil.copyfile(quality/name,out/name)
    encoder_sources={'learner':"SELECT DISTINCT user_id token FROM assignment WHERE split='train'"}
    for name,field in [('teacher_set','teacher_signature'),('school_set','school_signature'),('field_set','field_signature')]:
        encoder_sources[name]=f"SELECT DISTINCT {field} token FROM course_static JOIN assignment USING(course_id) WHERE split='train' AND {field} IS NOT NULL"
    for name,sql in encoder_sources.items():
        con.execute(f'CREATE TEMP TABLE enc_{name} AS SELECT token,row_number() OVER(ORDER BY token)::INTEGER code FROM ({sql})')
        write(con,out/'encoders'/f'{name}.parquet','SELECT * FROM enc_'+name)
        checks.sql(con,name+'_exact_train_vocabulary',f'''SELECT count(*)=0 FROM enc_{name} e FULL JOIN ({sql}) t USING(token) WHERE e.token IS NULL OR t.token IS NULL''')
    con.execute('''CREATE TEMP TABLE course_encoded AS SELECT c.*,m.course_family_id,b.course_end,
        coalesce(t.code,0) teacher_set_code,coalesce(s.code,0) school_set_code,coalesce(f.code,0) field_set_code
        FROM course_static c LEFT JOIN members m USING(course_id) LEFT JOIN course_times b USING(course_id)
        LEFT JOIN enc_teacher_set t ON c.teacher_signature=t.token
        LEFT JOIN enc_school_set s ON c.school_signature=s.token LEFT JOIN enc_field_set f ON c.field_signature=f.token''')
    config=json.loads((source/'feature_columns.json').read_text())
    numeric,categorical=config['raw_numeric_features'],config['categorical_features']
    selector=[x for x in numeric if x!='phase_fraction'];learner=numeric+categorical
    checks.add('same_user_feature_inventory',len(numeric)==57 and len(categorical)==5 and len(selector)==56)
    forbidden=set(TARGETS+['q','y','COELO','AFELO','ACELO','course_id','course_family_id','user_id','row_id','split','course_end','cutoff_time'])
    checks.add('feature_allowlists_exclude_target_and_metadata',not(forbidden & set(learner)))
    for name,columns in [('selector',selector),('learner',learner)]:
        dump(out/f'{name}_feature_columns.json',{'dataset_version':VERSION,'columns':columns,'columns_sha256':canonical_hash(columns),
            'phase_fraction': 'excluded from selector geometry' if name=='selector' else 'retained and scaled by shared train pool',
            'categorical_columns':[] if name=='selector' else categorical})
    config.update({'version':VERSION,'dataset_version':VERSION,'metadata_columns':META,'distance_feature_columns':selector,
        'selector_feature_columns':selector,'model_feature_columns':learner,'preprocessing_scope':'one pooled labeled V4 train across P1-P4',
        'split_variant':'family_temporal_test_stratified_train_dev','primary_target':'y_alpha_1','tempo_dq_comparison':'not required by user'})
    dump(out/'feature_columns.json',config)
    dictionary=json.loads((source/'feature_dictionary.json').read_text())
    for item in dictionary:
        if item['name']=='phase_fraction':
            item['definition']='Window fraction: P1=.25,P2=.50,P3=.75,P4=.90; FullTrain-A=1.0; shared train scaling.'
    dump(out/'feature_dictionary.json',dictionary)
    for phase in PHASES:
        for part in sorted((source/'features'/phase).glob('*.parquet')):
            view(con,'old_part',part)
            write(con,out/'features'/phase/part.name,f'''SELECT f.* REPLACE(a.split AS split,
                coalesce(e.code,0) AS learner_code,coalesce(c.teacher_set_code,0) AS teacher_set_code,
                coalesce(c.school_set_code,0) AS school_set_code,coalesce(c.field_set_code,0) AS field_set_code),
                c.course_family_id,a.enrollment_id,CASE WHEN a.enrollment_id IS NOT NULL THEN {row_id('f.phase')} END row_id,c.course_end
                FROM old_part f LEFT JOIN assignment a USING(user_id,course_id)
                LEFT JOIN enc_learner e ON f.user_id=e.token LEFT JOIN course_encoded c USING(course_id)''')
            view(con,'new_part',out/'features'/phase/part.name)
            unchanged=numeric+['phase','enroll_time','cutoff_time','gender_code']
            checks.sql(con,f'{phase}_{part.stem}_raw_preserved',f'''SELECT count(*)=0 FROM new_part f FULL JOIN old_part v USING(user_id,course_id)
                WHERE f.user_id IS NULL OR v.user_id IS NULL OR {' OR '.join('f.'+x+' IS DISTINCT FROM v.'+x for x in unchanged)}''')
        view(con,'phase_x',out/'features'/phase/'*.parquet')
        checks.sql(con,phase+'_all_eligible_pairs_preserved','SELECT count(*)=10455017 AND count(*)=count(DISTINCT(user_id,course_id)) FROM phase_x')
        view(con,'old_seq',source/'sequences'/phase/'*.parquet')
        for split in SPLITS:
            write(con,out/'supervised_raw'/phase/f'{split}.parquet',f'''SELECT {','.join('f.'+x for x in META+numeric+categorical)},
                {','.join('a.'+x for x in TARGETS)} FROM phase_x f JOIN assignment a USING(user_id,course_id)
                WHERE a.split={qs(split)} ORDER BY f.row_id''')
            write(con,out/'sequences'/phase/f'{split}.parquet',f'''SELECT s.* REPLACE(a.split AS split),a.course_family_id,a.enrollment_id,
                {row_id('s.phase')} row_id FROM old_seq s JOIN assignment a USING(user_id,course_id)
                WHERE a.split={qs(split)} ORDER BY row_id''')
        write(con,out/'audit'/phase/'part-000.parquet',f'''SELECT s.*,a.course_family_id,a.enrollment_id,{row_id('s.phase')} row_id
            FROM read_parquet({qs(source/'audit'/phase/'*.parquet')}) s JOIN assignment a USING(user_id,course_id)''')
        print(f'{phase}: full raw snapshots and supervised splits complete',flush=True)
    view(con,'fit_rows',out/'supervised_raw/*/train.parquet')
    med=records(con,'SELECT '+','.join(f'median({x}) {x}' for x in numeric)+' FROM fit_rows')[0]
    checks.add('no_train_all_missing_feature',all(med[x] is not None for x in numeric))
    moments=records(con,'SELECT '+','.join(f'avg(coalesce({x}::DOUBLE,{float(med[x])!r})) {x}_mean,stddev_pop(coalesce({x}::DOUBLE,{float(med[x])!r})) {x}_std' for x in numeric)+' FROM fit_rows')[0]
    params={}
    for x in numeric:
        std=float(moments[x+'_std']);params[x]={'median':float(med[x]),'mean':float(moments[x+'_mean']),
            'scale':std if math.isfinite(std) and std>1e-12 else 1.0,'constant_after_imputation':std<=1e-12}
    fit_rows=con.execute('SELECT count(*) FROM fit_rows').fetchone()[0]
    preprocessing={'dataset_version':VERSION,'fit_split':'train','fit_rows':fit_rows,'pool':'all four labeled train phases',
        'numeric':params,'all_missing_train_excluded':[],'add_missing_indicators':False,'input_order':learner,'output_order':learner,
        'categorical_columns':categorical,'categorical_policy':'train-only vocabularies; unknown/missing=0; gender retains source codes',
        'software':{'duckdb':duckdb.__version__,'numpy':np.__version__,'pandas':pd.__version__}}
    dump(out/'preprocessing.json',preprocessing)

    def make_ready(raw,ready):
        expressions=[f'((coalesce({x}::DOUBLE,{p["median"]!r})-{p["mean"]!r})/{p["scale"]!r}) {x}' for x,p in params.items()]
        write(con,ready,f'''SELECT {','.join(META)},{','.join(expressions+categorical+TARGETS)}
            FROM read_parquet({qs(raw)}) ORDER BY row_id''')

    for phase in PHASES:
        for split in SPLITS:
            make_ready(out/'supervised_raw'/phase/f'{split}.parquet',out/'model_ready'/phase/f'{split}.parquet')
    view(con,'supervised',out/'supervised_raw/*/*.parquet')
    view(con,'ready',out/'model_ready/*/*.parquet')
    view(con,'seqs',out/'sequences/*/*.parquet')
    view(con,'audits',out/'audit/*/*.parquet')
    checks.sql(con,'all_rows_have_exactly_one_family','SELECT count(*)=0 FROM supervised WHERE course_family_id IS NULL OR enrollment_id IS NULL OR row_id IS NULL')
    checks.sql(con,'four_phases_one_assignment_and_target', '''SELECT count(*)=0 FROM (SELECT enrollment_id FROM supervised GROUP BY 1
        HAVING count(*)<>4 OR count(DISTINCT phase)<>4 OR count(DISTINCT split)<>1 OR count(DISTINCT y_alpha_1)<>1 OR count(DISTINCT q_alpha_1)<>1)''')
    checks.sql(con,'all_snapshot_row_ids_unique','SELECT count(*)=714392 AND count(DISTINCT row_id)=714392 FROM supervised')
    checks.sql(con,'target_values_match_quality',f'''SELECT count(*)=0 FROM supervised f JOIN assignment a USING(user_id,course_id)
        WHERE f.split IS DISTINCT FROM a.split OR f.course_family_id IS DISTINCT FROM a.course_family_id OR {' OR '.join('f.'+x+' IS DISTINCT FROM a.'+x for x in TARGETS)}''')
    checks.sql(con,'no_family_leakage','SELECT count(*)=0 FROM (SELECT course_family_id FROM supervised GROUP BY 1 HAVING count(DISTINCT split)>1)')
    checks.sql(con,'no_recorded_out_of_window_event',"SELECT count(*)=0 FROM audits WHERE n_out_of_window_events<>0 OR min_feature_event_time<enroll_time OR max_feature_event_time>cutoff_time+INTERVAL '1 microsecond'")
    checks.sql(con,'fit_uses_all_and_only_train_snapshots',f"SELECT count(*)={fit_rows} AND count(DISTINCT phase)=4 FROM supervised WHERE split='train'")
    checks.sql(con,'shared_ready_train_mean_zero_std_one','SELECT '+ ' AND '.join(f'abs(avg({x}))<1e-8 AND abs(stddev_pop({x})-1)<1e-8' for x,p in params.items() if not p['constant_after_imputation'])+" FROM ready WHERE split='train'")
    checks.sql(con,'phase_fraction_retains_four_values','SELECT count(DISTINCT phase_fraction)=4 FROM ready')
    checks.sql(con,'no_null_nan_inf_in_ready_numeric','SELECT count(*)=0 FROM ready WHERE '+' OR '.join(f'{x} IS NULL OR NOT isfinite({x})' for x in numeric))
    for phase in PHASES:
        numerator,denominator={'P1':(1,4),'P2':(1,2),'P3':(3,4),'P4':(9,10)}[phase]
        checks.sql(con,phase+'_cutoff_formula_exact',f'''SELECT count(*)=0 FROM supervised WHERE phase={qs(phase)} AND epoch_us(cutoff_time)<>
            epoch_us(enroll_time)+(epoch_us(course_end)-epoch_us(enroll_time))*{numerator}//{denominator}''')
        for split in SPLITS:
            expected=con.execute(f'SELECT count(*) FROM assignment WHERE split={qs(split)}').fetchone()[0]
            checks.sql(con,f'{phase}_{split}_counts_schema_alignment',f'''SELECT
                (SELECT count(*)={expected} FROM supervised WHERE phase={qs(phase)} AND split={qs(split)})
                AND (SELECT count(*)={expected} FROM ready WHERE phase={qs(phase)} AND split={qs(split)})
                AND (SELECT count(*)={expected} FROM seqs WHERE phase={qs(phase)} AND split={qs(split)})''')
    checks.sql(con,'raw_ready_metadata_targets_aligned',f'''SELECT count(*)=0 FROM supervised r FULL JOIN ready m USING(row_id)
        WHERE r.row_id IS NULL OR m.row_id IS NULL OR {' OR '.join('r.'+x+' IS DISTINCT FROM m.'+x for x in META+categorical+TARGETS)}''')
    checks.sql(con,'raw_ready_numeric_transformation_exact','SELECT count(*)=0 FROM supervised r JOIN ready m USING(row_id) WHERE '+
        ' OR '.join(f'm.{x} IS DISTINCT FROM ((coalesce(r.{x}::DOUBLE,{p["median"]!r})-{p["mean"]!r})/{p["scale"]!r})' for x,p in params.items()))
    checks.sql(con,'sequence_keys_match_snapshot_keys','SELECT count(*)=0 FROM supervised r FULL JOIN seqs s USING(row_id) WHERE r.row_id IS NULL OR s.row_id IS NULL OR r.course_family_id IS DISTINCT FROM s.course_family_id OR r.split IS DISTINCT FROM s.split')

    # Reuse full-window raw features, never the old fullA numeric scaler/codes.
    view(con,'old_fullA',ROOT/'fullA/supervised_raw/*.parquet')
    expression=[('1.0::DOUBLE phase_fraction' if x=='phase_fraction' else 'f.'+x) for x in numeric]
    codes=['coalesce(e.code,0)::INTEGER learner_code','c.teacher_set_code','c.school_set_code','c.field_set_code','f.gender_code']
    write(con,out/'fulltrain_a/train_raw.parquet',f'''SELECT f.user_id,f.course_id,a.course_family_id,a.enrollment_id,
        {row_id(qs('FULL_A'))} row_id,'FULL_A' phase,'train' split,f.enroll_time,f.course_end cutoff_time,f.course_end,
        {','.join(expression+codes+['a.'+x for x in TARGETS])}
        FROM old_fullA f JOIN assignment a USING(user_id,course_id) LEFT JOIN enc_learner e ON f.user_id=e.token
        LEFT JOIN course_encoded c USING(course_id) WHERE a.split='train' ORDER BY row_id''')
    make_ready(out/'fulltrain_a/train_raw.parquet',out/'fulltrain_a/train_ready.parquet')
    shutil.copyfile(out/'fulltrain_a/train_ready.parquet',out/'fulltrain_a/train.parquet')
    shutil.copyfile(out/'preprocessing.json',out/'fulltrain_a/preprocessing.json')
    dump(out/'fulltrain_a/feature_columns.json',config)
    view(con,'fullA_raw',out/'fulltrain_a/train_raw.parquet')
    view(con,'fullA_ready',out/'fulltrain_a/train_ready.parquet')
    checks.sql(con,'fullA_one_row_exact_train_cohort', '''SELECT
        (SELECT count(*)=count(DISTINCT enrollment_id) FROM fullA_raw) AND
        (SELECT count(*)=0 FROM fullA_raw f FULL JOIN (SELECT * FROM assignment WHERE split='train') a USING(enrollment_id)
            WHERE f.enrollment_id IS NULL OR a.enrollment_id IS NULL)''')
    checks.sql(con,'fullA_lambda1_train_only',"SELECT count(*)=0 FROM fullA_raw WHERE split<>'train' OR cutoff_time<>course_end OR phase_fraction<>1.0")
    checks.sql(con,'fullA_raw_features_preserved', 'SELECT count(*)=0 FROM fullA_raw n JOIN old_fullA o USING(user_id,course_id) WHERE '+
        ' OR '.join('n.'+x+' IS DISTINCT FROM o.'+x for x in numeric if x!='phase_fraction'))
    checks.sql(con,'fullA_ready_finite','SELECT count(*)=0 FROM fullA_ready WHERE '+' OR '.join(f'{x} IS NULL OR NOT isfinite({x})' for x in numeric))
    checks.add('fullA_uses_identical_pooled_preprocessing',sha256(out/'preprocessing.json')==sha256(out/'fulltrain_a/preprocessing.json'))
    dump(out/'fulltrain_a/run_manifest.json',{'dataset_version':VERSION,'source':str(ROOT/'fullA'),'source_manifest_sha256':sha256(ROOT/'fullA/run_manifest.json'),
        'window_fraction':1.0,'split':'train','preprocessing':'same artifact as Pool B train','rows':fit_rows//4,'phase_fraction':1.0})
    dump(out/'fulltrain_a/validation.json',{k:v for k,v in checks.items.items() if k.startswith('fullA')})
    course_stats=stats(con,out,quality)
    checks.sql(con,'course_means_recomputed_exactly','''SELECT count(*)=0 FROM course_targets c JOIN
        (SELECT course_id,avg(q_alpha_1) q,count(*) n FROM assignment GROUP BY 1) a USING(course_id)
        WHERE abs(c.Q_c-a.q)>1e-9 OR c.n_c<>a.n''')
    checks.add('train_mu_order_and_class_ranges',True, json.loads((out/'protocol_constants.json').read_text())['mu'])
    checks.add('source_v3_features_unchanged',before==source_inventory(source))
    missing=records(con,'SELECT phase,split,count(*) n,'+','.join(f'count(*) FILTER(WHERE {x} IS NULL) {x}' for x in numeric)+' FROM supervised GROUP BY 1,2 ORDER BY 1,2')
    dump(out/'missingness.json',{'numeric_null_counts':missing,'denominator':'n in each phase/split record'})
    write(con,out/'audit/cutoff_sample_20_pairs.parquet','''SELECT r.*,a.min_feature_event_time,a.max_feature_event_time,a.n_out_of_window_events
        FROM (SELECT row_id,enrollment_id,user_id,course_id,course_family_id,phase,split,enroll_time,cutoff_time,course_end,n_events FROM supervised) r
        LEFT JOIN audits a USING(row_id) WHERE r.enrollment_id IN(SELECT enrollment_id FROM assignment ORDER BY sha256(enrollment_id) LIMIT 20)
        ORDER BY r.enrollment_id,r.phase''')
    checks.sql(con,'cutoff_sample_20_pairs_80_snapshots',f"SELECT count(*)=80 AND count(DISTINCT enrollment_id)=20 FROM read_parquet({qs(out/'audit/cutoff_sample_20_pairs.parquet')})")
    summary={'dataset_version':VERSION,'eligible_pairs_per_phase':10455017,'labeled_pairs_per_phase':178598,
        'total_supervised_snapshots':714392,'numeric_features':len(numeric),'categorical_features':len(categorical),
        'selector_features':len(selector),'learner_features':len(learner),'fit_rows':fit_rows,
        'class_counts':records(con,'SELECT phase,split,y_alpha_1 y,count(*) n FROM supervised GROUP BY 1,2,3 ORDER BY 1,2,3'),
        'phase_counts':records(con,'SELECT phase,split,count(*) n,count(*) FILTER(WHERE n_events=0) n_no_activity FROM supervised GROUP BY 1,2 ORDER BY 1,2'),
        'encoder_sizes':{name:con.execute('SELECT count(*) FROM enc_'+name).fetchone()[0] for name in encoder_sources},
        'course_statistics':course_stats,'checks_passed':len(checks.items)}
    dump(out/'summary.json',summary)
    (out/'load_cq_v4.py').write_text(LOADER)
    manifest={'dataset_version':VERSION,'date':DATE,'stage':'phase_features','source_phase_manifest_sha256':sha256(source/'run_manifest.json'),
        'source_fullA_manifest_sha256':sha256(ROOT/'fullA/run_manifest.json'),'quality_manifest_sha256':sha256(quality/'run_manifest.json'),
        'builder_sha256':sha256(__file__),'common_sha256':sha256(ROOT/'cq_v4_common.py'),'split_sha256':sha256(out/'splits.json'),
        'preprocessing_sha256':sha256(out/'preprocessing.json'),'feature_policy':'user supplied V3 features; no TEMPO-DQ comparison',
        'raw_preserved':True,'software':preprocessing['software'],'elapsed_seconds':round(time.monotonic()-started,3)}
    dump(out/'run_manifest.json',manifest)

    runtime=out/'runtime_dev_only'
    runtime.mkdir(exist_ok=True)
    write(con,runtime/'pool_b/train.parquet',"SELECT * FROM ready WHERE split='train' ORDER BY row_id")
    for phase in PHASES:
        shutil.copyfile(out/'model_ready'/phase/'dev.parquet',runtime/'pool_b'/f'dev_phase{phase[-1]}.parquet')
    (runtime/'fulltrain_a').mkdir(exist_ok=True)
    shutil.copyfile(out/'fulltrain_a/train_ready.parquet',runtime/'fulltrain_a/train.parquet')
    for name in ['selector_feature_columns.json','learner_feature_columns.json','feature_dictionary.json','preprocessing.json','protocol_constants.json','splits.json','course_splits.parquet','family_splits.parquet']:
        shutil.copyfile(out/name,runtime/name)
    (runtime/'load_cq_v4.py').write_text(LOADER)
    dump(runtime/'bundle_manifest.json',{'dataset_version':VERSION,'scope':'train/dev only','test_excluded':True,
        'split_sha256':sha256(out/'splits.json'),'preprocessing_sha256':sha256(out/'preprocessing.json'),
        'tables':['pool_b/train.parquet']+[f'pool_b/dev_phase{i}.parquet' for i in range(1,5)]+['fulltrain_a/train.parquet'],
        'test_note':'Split metadata may list test families/courses; no test feature or row label is distributed'})
    view(con,'runtime_rows',runtime/'pool_b/*.parquet')
    checks.sql(con,'runtime_has_no_test_row',"SELECT count(*)=0 FROM runtime_rows WHERE split NOT IN ('train','dev')")
    checks.add('runtime_has_no_test_named_file',not any('test' in p.name.lower() for p in runtime.rglob('*') if p.is_file()))
    checks.sql(con,'runtime_raw_ready_rows_align','''SELECT count(*)=0 FROM runtime_rows r FULL JOIN
        (SELECT row_id FROM ready WHERE split IN('train','dev')) x USING(row_id) WHERE r.row_id IS NULL OR x.row_id IS NULL''')
    import importlib.util
    spec=importlib.util.spec_from_file_location('v4_loader_smoke',out/'load_cq_v4.py');loader=importlib.util.module_from_spec(spec);spec.loader.exec_module(loader)
    train=loader.load_phase(phase=None,root=runtime,connection=con)
    checks.add('loader_reads_pooled_train',train.count('*').fetchone()[0]==fit_rows)
    checks.add('loader_selector_count',len(loader.feature_columns('selector',runtime))==56)
    checks.add('loader_reads_fullA',loader.load_fullA(root=runtime,connection=con).count('*').fetchone()[0]==fit_rows//4)
    denied=False
    try:loader.load_phase(split='test',root=runtime,connection=con)
    except PermissionError:denied=True
    checks.add('runtime_loader_denies_test',denied)
    for phase in PHASES:
        n=loader.load_phase(phase=phase,split='dev',root=runtime,connection=con).count('*').fetchone()[0]
        checks.add('loader_dev_'+phase,n==con.execute("SELECT count(*) FROM assignment WHERE split='dev'").fetchone()[0])
    checks.save(out/'validation.json')
    checks.save(runtime/'validation.json')
    summary['checks_passed']=len(checks.items);dump(out/'summary.json',summary)
    dump(out/'_SUCCESS.json',{'dataset_version':VERSION,'scope':'data build and loader checks only','checks_passed':len(checks.items),
        'manifest_sha256':sha256(out/'run_manifest.json'),'ready_for_m0_v4':False,'pending':'model/probe checks and independent review; see handoff_status.json'})
    con.close()
    checksums(out)
    print(json.dumps({k:summary[k] for k in ['fit_rows','total_supervised_snapshots','checks_passed']}) ,flush=True)
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--threads',type=int,default=2);p.add_argument('--memory',default='1200MB')
    a=p.parse_args();build(a.threads,a.memory)
