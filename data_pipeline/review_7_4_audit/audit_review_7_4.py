"""Read persisted outputs and audit Review 7.4; write only beside this script.

This audit does not rebuild data, assign labels, fit models, or change splits.
Run with an environment containing DuckDB. No version-dependent hash sharding.
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile
import xml.etree.ElementTree as ET

import duckdb

OUT = Path(__file__).resolve().parent
BASE = OUT.parent
OUT.mkdir(parents=True, exist_ok=True)
con = duckdb.connect()
con.execute("SET threads=2")
con.execute("SET memory_limit='2GB'")
con.execute("SET TimeZone='UTC'")
con.execute("SET temp_directory='" + str(OUT / '_scratch').replace("'", "''") + "'")
result = {
    'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
    'audit_duckdb_version': duckdb.__version__,
    'scope': 'Persisted Parquet and local protocol files; no source mutations or model training.',
    'note': 'Build recorded DuckDB 1.3.2. Audit reads with its own version; does not reproduce hash sharding.',
}
read_paths = set()


def quote(value):
    return "'" + str(value).replace("'", "''") + "'"


def query(sql):
    cur = con.execute(sql)
    return [dict(zip([d[0] for d in cur.description], row)) for row in cur.fetchall()]


def collect(name, sql):
    result[name] = query(sql)
    print(name, json.dumps(result[name], ensure_ascii=False, default=str), flush=True)


def view(name, pattern):
    files = list(BASE.glob(pattern))
    if not files:
        raise FileNotFoundError(pattern)
    read_paths.update(files)
    con.execute(f'CREATE VIEW {name} AS SELECT * FROM read_parquet({quote(BASE / pattern)})')


view('targets', 'phase_features/tables/user_course_targets.parquet')
view('raw', 'phase_features/supervised_raw/P*.parquet')
view('ready', 'phase_features/model_ready/P*.parquet')
view('splits', 'final_quality_tables/tables/course_splits.parquet')
view('bounds', 'review_time_phases/tables/course_time_bounds.parquet')
view('audit_events', 'phase_features/audit/P*/*.parquet')
view('original', 'final_quality_tables/tables/user_course_final/*.parquet')
view('static', 'phase_features/tables/course_static_metadata.parquet')
con.execute('CREATE TABLE labeled AS SELECT * FROM targets WHERE y IS NOT NULL')

collect('master', '''SELECT count(*) n_rows, count(DISTINCT(user_id,course_id)) n_unique_pairs,
    count(*) FILTER(WHERE user_id IS NULL OR course_id IS NULL) n_null_keys,
    count(*) FILTER(WHERE y IS NOT NULL) n_labeled,
    count(*) FILTER(WHERE (y IS NULL)<>(q IS NULL)) n_q_y_null_disagreement,
    count(*) FILTER(WHERE q NOT BETWEEN 0 AND 1) n_q_out_of_range,
    count(*) FILTER(WHERE y IS DISTINCT FROM CASE WHEN q IS NULL THEN NULL
        WHEN q<0.2 THEN 'E' WHEN q<0.4 THEN 'G' ELSE 'A' END) n_bad_tau FROM targets''')
collect('labeled_splits', '''SELECT split,count(*) n_pairs,count(DISTINCT user_id) n_users,
    count(DISTINCT course_id) n_labeled_courses,count(*) FILTER(WHERE y='E') E,
    count(*) FILTER(WHERE y='G') G,count(*) FILTER(WHERE y='A') A
    FROM labeled GROUP BY 1 ORDER BY 1''')
collect('course_splits', '''SELECT split,count(*) n_courses,min(course_start) min_start,
    max(course_start) max_start,min(course_end) min_end,max(course_end) max_end
    FROM splits GROUP BY 1 ORDER BY 1''')
collect('split_integrity', '''SELECT
    (SELECT count(*) FROM (SELECT course_id FROM splits GROUP BY 1 HAVING count(*)<>1)) duplicate_course_keys,
    (SELECT count(*) FROM (SELECT course_id FROM targets GROUP BY 1 HAVING count(DISTINCT split)<>1)) multi_split_courses,
    (SELECT count(*) FROM targets t LEFT JOIN splits s USING(course_id)
        WHERE t.split IS DISTINCT FROM s.split) inconsistent_master_split,
    (SELECT count(*) FROM raw r JOIN labeled t USING(user_id,course_id)
        WHERE r.split IS DISTINCT FROM t.split) inconsistent_snapshot_split''')

split_json = json.loads((BASE/'final_quality_tables/splits.json').read_text())
persisted = {r['course_id']: r['split'] for r in query('SELECT course_id,split FROM splits')}
ordered = query('SELECT course_id FROM splits ORDER BY course_start,course_id')
n = len(ordered)
ntr, ndv = round(0.7*n), round(0.1*n)
ref = {r['course_id']: ('train' if i<ntr else 'dev' if i<ntr+ndv else 'test')
       for i,r in enumerate(ordered)}
result['splits_json_matches_parquet'] = split_json == persisted
result['reference_rounding'] = {
    'n_courses': n, 'reference_counts': {'train': ntr, 'dev': ndv, 'test': n-ntr-ndv},
    'different_course_assignments': [
        {'course_id': k,'persisted': v,'reference': ref[k]}
        for k,v in persisted.items() if v != ref[k]
    ],
    'reference': 'Review Appendix E temporal_course_split: round(0.7*n), round(0.1*n).',
}
collect('temporal_overlap', '''WITH boundary AS (
    SELECT min(course_start) first_test_start FROM splits WHERE split='test')
    SELECT first_test_start,count(*) n_train_courses,
    count(*) FILTER(WHERE course_end>first_test_start) n_overlapping,
    count(*) FILTER(WHERE course_end<=first_test_start) n_retained_strict,
    avg((course_end>first_test_start)::INTEGER) overlap_fraction
    FROM splits CROSS JOIN boundary WHERE split='train' GROUP BY 1''')
collect('strict_train_diagnostic_only', '''SELECT count(*) n_labeled_pairs,
    count(DISTINCT t.course_id) n_labeled_courses,count(*) FILTER(WHERE y='E') E,
    count(*) FILTER(WHERE y='G') G,count(*) FILTER(WHERE y='A') A
    FROM labeled t JOIN splits s USING(course_id)
    WHERE t.split='train' AND s.course_end<=(SELECT min(course_start) FROM splits WHERE split='test')''')
collect('phase_snapshots', '''SELECT phase,split,count(*) n_rows,count(DISTINCT(user_id,course_id)) n_pairs,
    count(*) FILTER(WHERE y='E') E,count(*) FILTER(WHERE y='G') G,count(*) FILTER(WHERE y='A') A,
    count(*) FILTER(WHERE n_events=0) n_empty FROM raw GROUP BY 1,2 ORDER BY 1,2''')
collect('phase_consistency', '''SELECT
    (SELECT count(*) FROM (SELECT user_id,course_id FROM raw GROUP BY 1,2 HAVING count(*)<>4
        OR count(DISTINCT phase)<>4 OR count(DISTINCT y)<>1 OR count(DISTINCT q)<>1
        OR count(DISTINCT split)<>1)) n_inconsistent_pairs,
    (SELECT count(*) FROM raw r JOIN labeled t USING(user_id,course_id)
        WHERE r.q IS DISTINCT FROM t.q OR r.y IS DISTINCT FROM t.y) n_mismatched_targets,
    (SELECT count(*) FROM raw r JOIN splits s USING(course_id) WHERE abs(
        epoch_us(r.cutoff_time)-epoch_us(r.enroll_time)-floor((epoch_us(s.course_end)-epoch_us(r.enroll_time))
        * CASE r.phase WHEN 'P1' THEN 0.25 WHEN 'P2' THEN 0.5 WHEN 'P3' THEN 0.75 WHEN 'P4' THEN 0.9 END))>1)
        n_bad_cutoffs_tolerance_1_microsecond,
    (SELECT count(*) FROM audit_events WHERE max_feature_event_time>cutoff_time
        OR min_feature_event_time<enroll_time OR n_out_of_window_events<>0) n_out_of_window_audit_rows''')
collect('pool_b_strata', "SELECT phase,y,count(*) n_rows FROM raw WHERE split='train' GROUP BY 1,2 ORDER BY 1,2")
collect('mu_train', "SELECT y,count(*) n_pairs,avg(q) mu,min(q) min_q,max(q) max_q FROM labeled WHERE split='train' GROUP BY 1 ORDER BY 1")
con.execute('''CREATE TABLE course_diagnostic AS SELECT course_id,split,count(*) n_labeled,
    count(DISTINCT y) n_classes,avg(q) Q_labeled,
    CASE WHEN avg(q)<0.2 THEN 'E' WHEN avg(q)<0.4 THEN 'G' ELSE 'A' END Y_labeled
    FROM labeled GROUP BY 1,2''')
collect('course_diagnostic_counts', '''SELECT split,Y_labeled,count(*) n_courses,
    count(*) FILTER(WHERE n_labeled>=30) n_courses_ge30,
    count(*) FILTER(WHERE n_classes>1) n_courses_multiple_learner_labels
    FROM course_diagnostic GROUP BY 1,2 ORDER BY 1,2''')
collect('course_diagnostic_sizes', '''SELECT split,count(*) n_courses,
    count(*) FILTER(WHERE n_labeled<30) n_courses_lt30,min(n_labeled) min_n,
    quantile_cont(n_labeled,0.25) q25,median(n_labeled) median_n,
    quantile_cont(n_labeled,0.75) q75,max(n_labeled) max_n FROM course_diagnostic GROUP BY 1 ORDER BY 1''')
collect('label_variants', '''SELECT o.y original_y,t.y alpha01_y,count(*) n_pairs
    FROM labeled t JOIN original o USING(user_id,course_id) GROUP BY 1,2 ORDER BY 1,2''')
collect('test_learners_old_new', '''SELECT count(*) n_test_pairs,
    count(*) FILTER(WHERE tr.user_id IS NOT NULL) n_pairs_seen_learner,
    count(*) FILTER(WHERE tr.user_id IS NULL) n_pairs_unseen_learner,
    count(DISTINCT t.user_id) n_test_users,
    count(DISTINCT t.user_id) FILTER(WHERE tr.user_id IS NOT NULL) n_seen_users
    FROM labeled t LEFT JOIN (SELECT DISTINCT user_id FROM labeled WHERE split='train') tr USING(user_id)
    WHERE t.split='test' ''')
con.execute('''CREATE TABLE atomic_teachers AS SELECT DISTINCT s.course_id,s.split,
    unnest(string_split(m.teacher_signature,'|')) teacher_id
    FROM splits s JOIN static m USING(course_id) WHERE teacher_signature IS NOT NULL''')
collect('test_teachers_old_new', '''SELECT count(DISTINCT te.teacher_id) n_test_teachers,
    count(DISTINCT te.teacher_id) FILTER(WHERE tr.teacher_id IS NOT NULL) n_seen,
    count(DISTINCT te.teacher_id) FILTER(WHERE tr.teacher_id IS NULL) n_unseen
    FROM atomic_teachers te LEFT JOIN (SELECT DISTINCT teacher_id FROM atomic_teachers
        WHERE split='train') tr USING(teacher_id) WHERE te.split='test' ''')
collect('test_code_unknown', '''SELECT count(*) n_test_pairs,
    count(*) FILTER(WHERE learner_code=0) learner_unknown,
    count(*) FILTER(WHERE teacher_set_code=0) teacher_set_unknown,
    count(*) FILTER(WHERE school_set_code=0) school_set_unknown,
    count(*) FILTER(WHERE field_set_code=0) field_set_unknown
    FROM raw WHERE phase='P1' AND split='test' ''')

cfg = json.loads((BASE/'phase_features/feature_columns.json').read_text())
pp = json.loads((BASE/'phase_features/preprocessing.json').read_text())
names = cfg['model_feature_columns']
forbidden = {'q','y','course_id','course_end','cutoff_time','enroll_time','split','target_alpha','target_formula_version'}
result['feature_contract'] = {
    'n_X': len(names),'n_numeric': len(cfg['numeric_features']),
    'n_categorical': len(cfg['categorical_features']),
    'n_distance':len(cfg['distance_feature_columns']),
    'forbidden_X': [x for x in names if x in forbidden or any(t in x.lower() for t in ['coelo','afelo','acelo','loqv','missing','mask','label_status'])],
    'categorical_in_geometry': sorted(set(cfg['categorical_features']) & set(cfg['distance_feature_columns'])),
    'add_missing_indicators': pp['add_missing_indicators'],
    'fit_rows_persisted': pp['fit_rows'],
    'fit_rows_recomputed':con.execute("SELECT count(*) FROM raw WHERE split='train'").fetchone()[0],
}
numeric = cfg['numeric_features']
med = query('SELECT '+','.join(f'median({x}) {x}' for x in numeric)+" FROM raw WHERE split='train'")[0]
expressions=[]
for x in numeric:
    expr=f'coalesce({x}::DOUBLE,{float(med[x])!r})'
    expressions.extend([f'avg({expr}) {x}_mean',f'stddev_pop({expr}) {x}_scale'])
moments=query('SELECT '+','.join(expressions)+" FROM raw WHERE split='train'")[0]
errors={'median':0.0,'mean':0.0,'scale':0.0}
for x in numeric:
    vals={'median':float(med[x]),'mean':float(moments[x+'_mean']),
          'scale':float(moments[x+'_scale']) if moments[x+'_scale']>1e-12 else 1.0}
    for k,v in vals.items(): errors[k]=max(errors[k],abs(v-pp['numeric'][x][k]))
result['preprocessing_recomputed_max_abs_errors']=errors
bad_ready=' OR '.join(f'{x} IS NULL OR NOT isfinite({x})' for x in numeric)
collect('ready_integrity',f'''SELECT count(*) n_rows,count(DISTINCT(user_id,course_id,phase)) n_keys,
    count(*) FILTER(WHERE {bad_ready}) n_nonfinite_rows FROM ready''')
expressions=[f'abs(r.{x}-(coalesce(f.{x}::DOUBLE,{pp["numeric"][x]["median"]!r})-{pp["numeric"][x]["mean"]!r})/{pp["numeric"][x]["scale"]!r})' for x in numeric]
collect('ready_transform_error',f'''SELECT max(greatest({','.join(expressions)})) max_abs_error
    FROM raw f JOIN ready r USING(user_id,course_id,phase)''')

con.execute("CREATE TABLE sample_pairs AS SELECT user_id,course_id FROM labeled ORDER BY md5(user_id||'|'||course_id||'|review74-seed20261017') LIMIT 20")
sample=query('''SELECT r.user_id,r.course_id,r.phase,r.enroll_time,r.cutoff_time,r.n_events,
    a.min_feature_event_time,a.max_feature_event_time,a.n_out_of_window_events,
    CASE WHEN r.n_events=0 THEN a.user_id IS NULL ELSE a.user_id IS NOT NULL
        AND a.min_feature_event_time>=r.enroll_time AND a.max_feature_event_time<=r.cutoff_time
        AND a.n_out_of_window_events=0 END audit_consistent
    FROM raw r JOIN sample_pairs USING(user_id,course_id)
    LEFT JOIN audit_events a USING(user_id,course_id,phase) ORDER BY r.user_id,r.course_id,r.phase''')
with (OUT/'cutoff_sample_20_pairs.csv').open('w',newline='',encoding='utf-8-sig') as f:
    writer=csv.DictWriter(f,fieldnames=list(sample[0]));writer.writeheader();writer.writerows(sample)
result['cutoff_sample']={'n_pairs':20,'n_snapshots':len(sample),
    'all_consistent':all(r['audit_consistent'] for r in sample),
    'selection':'Deterministic MD5 ordering with seed string review74-seed20261017',
    'limitation':'Checks persisted event-audit aggregates; not an independent manual reconstruction from raw logs.'}

with ZipFile(BASE/'Review-2026-10-12.docx') as z:
    root=ET.fromstring(z.read('word/document.xml'))
paragraphs=[''.join(x.text or '' for x in p.iter() if x.tag.endswith('}t'))
    for p in root.findall('.//{http://schemas.openxmlformats.org/wordprocessingml/2006/main}body//{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p')]
result['review_sections']={}
for start,stop in [('7.4.','7.5.'),('7.11.','7.12.')]:
    i=next(i for i,p in enumerate(paragraphs) if p.startswith(start))
    j=next(i+k+1 for k,p in enumerate(paragraphs[i+1:]) if p.startswith(stop))
    result['review_sections'][start]=paragraphs[i:j]
sources=['Review-2026-10-12.docx','DATA_CARD.md','BAO_CAO_XAY_DUNG_DATASET.md',
    'final_quality_tables/splits.json','final_quality_tables/run_manifest.json',
    'final_quality_tables/protocol_constants.json','phase_features/run_manifest.json',
    'phase_features/feature_columns.json','phase_features/preprocessing.json',
    'phase_features/validation.json','phase_features/temporal_invariance_validation.json',
    'weighted_loqv_alpha_0p1/run_manifest.json','09_build_phase_features.py']
result['source_sha256']={p:hashlib.sha256((BASE/p).read_bytes()).hexdigest() for p in sources}
result['audit_script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
result['parquet_read_inventory']=[{'path':str(p.relative_to(BASE)),
    'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns} for p in sorted(read_paths)]
(OUT/'evidence.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,default=str)+'\n',encoding='utf-8')
print('AUDIT_COMPLETE',OUT/'evidence.json')
