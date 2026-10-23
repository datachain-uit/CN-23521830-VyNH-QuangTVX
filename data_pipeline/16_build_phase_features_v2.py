"""Rebuild cumulative P1-P4 snapshots from quality-v2 events and stored cutoffs.

One shared X delivery, two final q/y targets; labeled train/dev/test files.
Raw NULLs retained. Ready preprocessing is fitted separately on each phase's
labeled train rows. Run with Python 3.9 and DuckDB 1.3.2; no model inference.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import time

import duckdb

ROOT=Path(__file__).resolve().parent
VERSION='phase-features-v2-cutoffs-v2-shared-targets-2026-10-20'
spec=importlib.util.spec_from_file_location('phase_v1',ROOT/'09_build_phase_features.py')
old=importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
qs=old.qs
dump=old.dump_json
records=old.records
view=old.view
table=old.replace_table
write=old.write_parquet

# Existing raw-log baseline plus observable v2 activity fields.
old.GROUPS['video']+=['n_video_events','n_rewind_segments','n_invalid_video_segments','n_zero_length_segments']
old.GROUPS['activity']+=['n_visits_30m','n_unknown_chapters_touched','n_chapters_touched_all']
old.NUMERIC=[x for group in old.GROUPS.values() for x in group]
old.ZERO_FEATURES.update(['n_video_events','n_rewind_segments','n_invalid_video_segments','n_zero_length_segments',
                         'n_visits_30m','n_unknown_chapters_touched','n_chapters_touched_all'])
old.SEQ_FIELDS+=['n_video_events','n_rewind_segments','n_invalid_video_segments']
NUMERIC=old.NUMERIC
CATEGORICAL=old.CATEGORICAL
META=old.META
PHASES=old.PHASES
TARGETS=['q_alpha_0p1','y_alpha_0p1','q_alpha_1','y_alpha_1']
SEQ_TYPE='STRUCT(week_index INTEGER,'+','.join(f'{x} DOUBLE' for x in old.SEQ_FIELDS)+')[]'


def build_windows(con,phase,shard):
    table(con,'pairs',f'''SELECT p.user_id,p.course_id,p.enroll_time,p.course_end,p.cutoff_{phase} cutoff_time,s.split,
      c.course_n_videos,c.course_n_exercises,c.course_n_problems,c.course_n_chapters,
      CASE WHEN year(p.enroll_time)-u.year_of_birth BETWEEN 10 AND 100 AND u.year_of_birth>=1900
        THEN (year(p.enroll_time)-u.year_of_birth)::DOUBLE END learner_age_at_enrollment,
      coalesce(e.code,0) learner_code,c.teacher_set_code,c.school_set_code,c.field_set_code,
      coalesce(u.gender_code,-1) gender_code
      FROM pair_slice p LEFT JOIN course_splits s USING(course_id) LEFT JOIN course_encoded c USING(course_id)
      LEFT JOIN (SELECT * FROM static_users WHERE user_shard={shard}) u USING(user_id)
      LEFT JOIN enc_learner e ON p.user_id=e.token''')
    table(con,'video_activity_w',f'''WITH selected AS (
      SELECT v.*,p.enroll_time,p.cutoff_time,
        CASE WHEN valid_measurement THEN least(v.end_point-v.start_point,
          greatest(0.0,epoch(p.cutoff_time)-epoch(v.event_time))*v.speed) ELSE 0 END observed_media_seconds
      FROM video_events v JOIN pairs p USING(user_id,course_id)
      WHERE hash(v.user_id)%32={shard} AND v.event_time BETWEEN p.enroll_time AND p.cutoff_time)
      SELECT *,start_point+observed_media_seconds observed_end_point,
        CASE WHEN valid_measurement THEN event_time+(observed_media_seconds/speed)*INTERVAL '1 second'
          ELSE event_time END observed_end_time FROM selected''')
    table(con,'video_w','SELECT * FROM video_activity_w WHERE valid_measurement AND observed_media_seconds>0')
    table(con,'problem_w', '''SELECT e.*,p.enroll_time,p.cutoff_time FROM problem_slice e JOIN pairs p USING(user_id,course_id)
      WHERE e.event_time BETWEEN p.enroll_time AND p.cutoff_time''')
    table(con,'comment_all_w',f'''SELECT c.*,p.enroll_time,p.cutoff_time FROM comments c JOIN pairs p USING(user_id,course_id)
      WHERE hash(c.user_id)%32={shard} AND c.event_time BETWEEN p.enroll_time AND p.cutoff_time''')
    table(con,'comment_w','SELECT * FROM comment_all_w WHERE normalized_text IS NOT NULL AND exact_duplicate_rank=1')
    table(con,'reply_w',f'''SELECT r.*,p.enroll_time,p.cutoff_time FROM replies r JOIN pairs p USING(user_id,course_id)
      WHERE hash(r.user_id)%32={shard} AND r.event_time BETWEEN p.enroll_time AND p.cutoff_time''')
    table(con,'received_w', '''SELECT c.user_id,c.course_id,c.lo_key,c.comment_id,r.reply_id,r.event_time,
      c.enroll_time,c.cutoff_time FROM comment_w c JOIN replies r ON c.comment_id=r.parent_comment_id
      WHERE r.event_time BETWEEN c.event_time AND c.cutoff_time''')


def aggregate_windows(con):
    # Preserve verified v1 video union / problem / sentiment / reply formulas.
    old.aggregate_windows(con)
    table(con,'video_flags', '''SELECT user_id,course_id,count(*) n_video_events,
      count(*) FILTER(WHERE start_point>end_point AND isfinite(start_point) AND isfinite(end_point)) n_rewind_segments,
      count(*) FILTER(WHERE NOT valid_measurement) n_invalid_video_segments,
      count(*) FILTER(WHERE start_point=end_point) n_zero_length_segments
      FROM video_activity_w GROUP BY 1,2''')
    table(con,'video_features_new', '''SELECT coalesce(v.user_id,f.user_id) user_id,coalesce(v.course_id,f.course_id) course_id,
      v.* EXCLUDE(user_id,course_id),f.* EXCLUDE(user_id,course_id)
      FROM video_features v FULL JOIN video_flags f USING(user_id,course_id)''')
    table(con,'video_features','SELECT * FROM video_features_new')
    selections=[]
    kinds=[
      ('video_activity_w','video','observed_end_time','video_id',{
        'n_video_segments':'(valid_measurement AND observed_media_seconds>0)::INT','video_media_seconds':'observed_media_seconds',
        'n_video_events':'1','n_rewind_segments':'coalesce(start_point>end_point AND isfinite(start_point) AND isfinite(end_point),false)::INT',
        'n_invalid_video_segments':'(NOT valid_measurement)::INT'}),
      ('problem_w','problem','event_time','exercise_id',{'n_problem_submissions':'1',
        'problem_attempts':'CASE WHEN attempts>0 THEN attempts ELSE 0 END','n_correct_problems':'(is_correct=1)::INT'}),
      ('comment_all_w','comment','event_time',"coalesce(resource_id,'discussion:'||course_id)",{
        'n_comments':'(normalized_text IS NOT NULL AND exact_duplicate_rank=1)::INT'}),
      ('reply_w','reply','event_time',"coalesce(resource_id,'discussion:'||course_id)",{'n_replies_authored':'1'}),
      ('received_w','received','event_time','NULL::VARCHAR',{'n_replies_received':'1'}),
    ]
    for source,kind,end,resource,values in kinds:
        fields=','.join(f'{values.get(x,"0")} {x}' for x in old.SEQ_FIELDS)
        selections.append(f'''SELECT user_id,course_id,lo_key,event_time,{end} observed_end_time,enroll_time,cutoff_time,
          {resource} resource_id,{qs(kind)} event_kind,{fields} FROM {source}''')
    table(con,'activity_events',' UNION ALL '.join(selections))
    table(con,'visit_features', '''WITH edges AS (
      SELECT *,lag(event_time) OVER(PARTITION BY user_id,course_id,lo_key,resource_id ORDER BY event_time,event_kind) prev_time
      FROM activity_events WHERE event_kind<>'received')
      SELECT user_id,course_id,count(*) FILTER(WHERE prev_time IS NULL OR event_time-prev_time>INTERVAL '30 minutes') n_visits_30m
      FROM edges GROUP BY 1,2''')
    table(con,'activity_features', '''WITH totals AS (SELECT user_id,course_id,
      count(*) FILTER(WHERE event_kind<>'received') n_events,
      count(DISTINCT cast(event_time AS DATE)) FILTER(WHERE event_kind<>'received') n_active_days,
      count(DISTINCT lo_key) FILTER(WHERE event_kind<>'received' AND lo_key NOT LIKE '%:CH-1') n_chapters_touched,
      count(DISTINCT lo_key) FILTER(WHERE event_kind<>'received' AND lo_key LIKE '%:CH-1') n_unknown_chapters_touched,
      count(DISTINCT lo_key) FILTER(WHERE event_kind<>'received') n_chapters_touched_all,
      (epoch(max(observed_end_time) FILTER(WHERE event_kind<>'received'))-epoch(min(event_time) FILTER(WHERE event_kind<>'received')))/86400.0 activity_span_days,
      (epoch(max(cutoff_time))-epoch(max(observed_end_time) FILTER(WHERE event_kind<>'received')))/86400.0 activity_recency_days,
      count(DISTINCT cast(event_time AS DATE)) FILTER(WHERE event_kind IN ('comment','reply','received')) social_active_days,
      (epoch(max(cutoff_time))-epoch(max(event_time) FILTER(WHERE event_kind IN ('comment','reply','received'))))/86400.0 social_recency_days,
      min(event_time) min_feature_event_time,max(observed_end_time) max_feature_event_time,
      count(*) FILTER(WHERE event_time<enroll_time OR event_time>cutoff_time OR observed_end_time>cutoff_time+INTERVAL '1 microsecond') n_out_of_window_events
      FROM activity_events GROUP BY 1,2)
      SELECT a.*,coalesce(v.n_visits_30m,0) n_visits_30m FROM totals a LEFT JOIN visit_features v USING(user_id,course_id)''')
    table(con,'weekly_features',f'''SELECT user_id,course_id,
      floor((epoch(event_time)-epoch(enroll_time))/(7*86400.0))::INTEGER week_index,
      {','.join(f'sum({x}) {x}' for x in old.SEQ_FIELDS)} FROM activity_events GROUP BY 1,2,3''')


def build(output=None,threads=3,memory='1500MB'):
    started=time.monotonic()
    quality=ROOT/'final_quality_v2'
    out=Path(output or ROOT/'phase_features_v2').resolve()
    raw=ROOT/'input/moocubex_parquet';model=ROOT/'input/moocubex_model'
    for source in [quality,raw,model,ROOT/'phase_features']:
        if out==source or source in out.parents or out in source.parents:
            raise ValueError('Output must be separate from source and v1 folders')
    assert (quality/'_SUCCESS.json').exists(),'Quality v2 has not completed'
    inventory=[]
    for folder in [quality/'tables',quality/'_stage',raw,model,ROOT/'eda_engagement_features_comparison/_stage/problem_rows']:
        for p in sorted(folder.rglob('*.parquet')):
            st=p.stat();inventory.append([str(p),st.st_size,st.st_mtime_ns])
    signature=hashlib.sha256(json.dumps({'inputs':inventory,'source':old.sha256(__file__),
      'helper_source':old.sha256(ROOT/'09_build_phase_features.py'),'version':VERSION},sort_keys=True).encode()).hexdigest()
    for p in [out,out/'_stage',out/'_scratch',out/'tables',out/'encoders']:
        p.mkdir(parents=True,exist_ok=True)
    (out/'_SUCCESS.json').unlink(missing_ok=True)
    con=duckdb.connect()
    for setting in [f'threads={threads}',f'memory_limit={qs(memory)}',f'temp_directory={qs(out/"_scratch")}',
                    "max_temp_directory_size='3GB'","TimeZone='UTC'",'preserve_insertion_order=false']:
        con.execute('SET '+setting)
    cache_path=out/'checkpoints.json';cache=json.loads(cache_path.read_text()) if cache_path.exists() else {}
    def save(name,sql):
        path=out/'_stage'/f'{name}.parquet'
        token=hashlib.sha256((signature+sql).encode()).hexdigest()
        if not path.exists() or cache.get(name)!=token:
            write(con,path,sql);cache[name]=token;dump(cache_path,cache)
        view(con,name,path)
    view(con,'time_pairs',quality/'tables/user_course_phase_times.parquet')
    view(con,'course_splits',quality/'tables/course_splits.parquet')
    view(con,'label0',quality/'variants/alpha_0p1/user_course_labels.parquet')
    view(con,'label1',quality/'variants/alpha_1/user_course_labels.parquet')
    con.execute('''CREATE VIEW all_targets AS SELECT a.user_id,a.course_id,a.split,a.q q_alpha_0p1,a.y y_alpha_0p1,
      b.q q_alpha_1,b.y y_alpha_1 FROM label0 a JOIN label1 b USING(user_id,course_id)
      WHERE (a.y IS NULL)=(b.y IS NULL)''')
    con.execute('''CREATE VIEW targets AS SELECT *,y_alpha_0p1 y FROM all_targets''')
    write(con,out/'tables/user_course_targets.parquet','SELECT * EXCLUDE(y) FROM targets')
    write(con,out/'tables/labeled_user_course_splits.parquet','SELECT * EXCLUDE(y) FROM targets WHERE y IS NOT NULL')
    view(con,'video_events',quality/'_stage/video_window.parquet')
    view(con,'comments',quality/'_stage/comment_window.parquet')
    view(con,'replies',quality/'_stage/replies_prepared.parquet')
    view(con,'sentiment',quality/'tables/comment_sentiment.parquet')
    print('Preparing fixed course split and train-only vocabularies',flush=True)
    old.prepare_static(con,raw,model,out,save)
    source_problem=ROOT/'eda_engagement_features_comparison/_stage/problem_rows'
    shard_checks=[]
    for shard in range(32):
        table(con,'pair_slice',f'SELECT * FROM time_pairs WHERE time_eligible AND hash(user_id)%32={shard}')
        view(con,'problem_slice',source_problem/f'user_shard={shard}/*.parquet')
        for phase in PHASES:
            paths={kind:out/kind/phase/f'part-{shard:03d}.parquet' for kind in ['features','audit','_sequence_parts']}
            token=f'{phase}/{shard}'
            if cache.get(token)==signature and all(p.exists() for p in paths.values()):
                print('REUSE',token,flush=True);continue
            tick=time.monotonic()
            build_windows(con,phase,shard)
            aggregate_windows(con)
            write(con,paths['features'],old.feature_sql(phase))
            write(con,paths['audit'],f'''SELECT p.user_id,p.course_id,{qs(phase)} phase,p.enroll_time,p.cutoff_time,
              a.min_feature_event_time,a.max_feature_event_time,a.n_out_of_window_events,
              coalesce(a.n_events,0)>0 has_observed_activity
              FROM activity_features a JOIN pairs p USING(user_id,course_id)''')
            write(con,paths['_sequence_parts'],f'''SELECT user_id,course_id,{qs(phase)} phase,
              list(struct_pack(week_index:=week_index,{','.join(f'{x}:={x}::DOUBLE' for x in old.SEQ_FIELDS)}) ORDER BY week_index) behavior_sequence
              FROM weekly_features GROUP BY 1,2''')
            view(con,'written_features',paths['features'])
            row=records(con,'SELECT count(*) n,count(DISTINCT(user_id,course_id)) n_unique FROM written_features')[0]
            assert row['n']==row['n_unique']==con.execute('SELECT count(*) FROM pairs').fetchone()[0]
            assert con.execute('SELECT coalesce(sum(n_out_of_window_events),0)=0 FROM activity_features').fetchone()[0]
            assert con.execute('SELECT count(*)=0 FROM video_features WHERE video_unique_seconds>video_media_seconds+1e-7 OR video_recency_days< -1e-9').fetchone()[0]
            shard_checks.append({'phase':phase,'shard':shard,**row})
            cache[token]=signature;dump(cache_path,cache)
            print(f'BUILT {token} {row["n"]:,} rows {time.monotonic()-tick:.2f}s',flush=True)
    config={'version':VERSION,'grain':['user_id','course_id','phase'],'metadata_columns':META,
      'raw_numeric_features':NUMERIC,'numeric_features':NUMERIC,'categorical_features':CATEGORICAL,
      'model_feature_columns':NUMERIC+CATEGORICAL,'distance_feature_columns':NUMERIC,
      'target_columns':TARGETS,'feature_groups':old.GROUPS,'weekly_sequence_fields':['week_index']+old.SEQ_FIELDS,
      'preprocessing_scope':'separate labeled train pool per phase; shared across two alpha targets',
      'categorical_usage':'Codes are categories; use categorical-aware models or train-only one-hot; exclude IDs from distance geometry.'}
    dump(out/'feature_columns.json',config)
    preprocessing={}
    for phase in PHASES:
        view(con,'phase_x',out/'features'/phase/'*.parquet')
        view(con,'phase_seq',out/'_sequence_parts'/phase/'*.parquet')
        for split in ['train','dev','test']:
            write(con,out/'supervised_raw'/phase/f'{split}.parquet',f'''SELECT f.*,{','.join('t.'+x for x in TARGETS)}
              FROM phase_x f JOIN targets t USING(user_id,course_id) WHERE t.y IS NOT NULL AND t.split={qs(split)}''')
            write(con,out/'sequences'/phase/f'{split}.parquet',f'''SELECT f.user_id,f.course_id,f.phase,f.split,
              coalesce(s.behavior_sequence,[]::{SEQ_TYPE}) behavior_sequence
              FROM read_parquet({qs(out/'supervised_raw'/phase/f'{split}.parquet')}) f
              LEFT JOIN phase_seq s USING(user_id,course_id,phase)''')
        view(con,'fit_rows',out/'supervised_raw'/phase/'train.parquet')
        med=records(con,'SELECT '+','.join(f'median({x}) {x}' for x in NUMERIC)+' FROM fit_rows')[0]
        active=[x for x in NUMERIC if med[x] is not None]
        expressions=[]
        for name in active:
            expression=f'coalesce({name}::DOUBLE,{float(med[name])!r})'
            expressions += [f'avg({expression}) {name}_mean',f'stddev_pop({expression}) {name}_std']
        moments=records(con,'SELECT '+','.join(expressions)+' FROM fit_rows')[0]
        params={}
        for name in active:
            std=float(moments[name+'_std'])
            params[name]={'median':float(med[name]),'mean':float(moments[name+'_mean']),
              'scale':std if math.isfinite(std) and std>1e-12 else 1.0,'constant_after_imputation':std<=1e-12}
        preprocessing[phase]={'fit_split':'train','fit_rows':con.execute('SELECT count(*) FROM fit_rows').fetchone()[0],
          'numeric':params,'all_missing_train_excluded':[x for x in NUMERIC if x not in active],
          'add_missing_indicators':False}
        for split in ['train','dev','test']:
            expressions=[f'((coalesce({x}::DOUBLE,{p["median"]!r})-{p["mean"]!r})/{p["scale"]!r}) {x}' for x,p in params.items()]
            write(con,out/'model_ready'/phase/f'{split}.parquet',f'''SELECT {','.join(META)},
              {','.join(expressions+CATEGORICAL+TARGETS)} FROM read_parquet({qs(out/'supervised_raw'/phase/f'{split}.parquet')})''')
    dump(out/'preprocessing.json',preprocessing)
    print('Validating all saved snapshots and split files',flush=True)
    summary,checks=validate(con,out,preprocessing)
    checks['fixtures']=old.fixtures()
    dump(out/'summary.json',summary);dump(out/'validation.json',checks)
    manifest={'version':VERSION,'client_date':'2026-10-20','quality_root':str(quality),'source_sha256':old.sha256(__file__),
      'source_signature':signature,'input_inventory':inventory,'duckdb_version':duckdb.__version__,
      'split':'unchanged v2 course_start ordered 70/10/20 courses; not 70/10/20 labeled rows',
      'cutoff':'stored v2 cutoff_P1..P4, inclusive cumulative windows; video clipped by speed at cutoff',
      'features':'raw logs only; no final quality indices or final P95 in X; two alpha labels share X',
      'preprocessing':'per-phase train-only medians/std and common train-only category vocabularies',
      'source_semantics':'raw video events and all course-mapped comments contribute observable activity/sessions; text/sentiment counts require nonempty deduplicated comments',
      'split_limitations':'courses disjoint; users may overlap; course durations may overlap; last-observed course-end is retrospective',
      'static_assumption':'Catalog and user profile snapshot treated as static; publication/effective histories unavailable',
      'per_shard_checks':shard_checks,'elapsed_seconds':round(time.monotonic()-started,3)}
    dump(out/'run_manifest.json',manifest)
    write_report(out,summary,preprocessing)
    write_dictionary(out)
    assert all(v for k,v in checks.items() if k!='fixtures'),{k:v for k,v in checks.items() if not v}
    dump(out/'_SUCCESS.json',{'version':VERSION,'manifest_sha256':old.sha256(out/'run_manifest.json'),
      'labeled_pairs':summary['labeled_pairs'],'checks_passed':sum(v for k,v in checks.items() if k!='fixtures')})
    con.close()
    print('COMPLETE',json.dumps(summary,ensure_ascii=False),flush=True)


def validate(con,out,preprocessing):
    view(con,'all_x',out/'features/*/*.parquet')
    view(con,'audits',out/'audit/*/*.parquet')
    view(con,'supervised',out/'supervised_raw/*/*.parquet')
    view(con,'ready',out/'model_ready/*/*.parquet')
    view(con,'sequences',out/'sequences/*/*.parquet')
    eligible=con.execute('SELECT count(*) FROM time_pairs WHERE time_eligible').fetchone()[0]
    labeled=con.execute('SELECT count(*) FROM targets WHERE y IS NOT NULL').fetchone()[0]
    summary={'version':VERSION,'eligible_pairs':eligible,'labeled_pairs':labeled,
      'numeric_features':len(NUMERIC),'categorical_features':len(CATEGORICAL),
      'full_phase_counts':records(con,'SELECT phase,count(*) n,count(*) FILTER(WHERE n_events=0) n_no_activity FROM all_x GROUP BY 1 ORDER BY 1'),
      'supervised_phase_counts':records(con,'SELECT phase,split,count(*) n,count(*) FILTER(WHERE n_events=0) n_no_activity FROM supervised GROUP BY 1,2 ORDER BY 1,2'),
      'labeled_splits':records(con,"SELECT split,count(*) n,count(DISTINCT course_id) n_courses FROM targets WHERE y IS NOT NULL GROUP BY 1 ORDER BY 1"),
      'class_counts':{},'raw_missing':{},'sequence_counts':records(con,'SELECT phase,split,count(*) n,count(*) FILTER(WHERE len(behavior_sequence)=0) n_empty FROM sequences GROUP BY 1,2 ORDER BY 1,2')}
    for suffix in ['0p1','1']:
        summary['class_counts'][suffix]=records(con,f'SELECT split,y_alpha_{suffix} y,count(*) n FROM targets WHERE y IS NOT NULL GROUP BY 1,2 ORDER BY 1,2')
    for phase in PHASES:
        summary['raw_missing'][phase]=records(con,'SELECT '+','.join(f'count(*) FILTER(WHERE {x} IS NULL) {x}' for x in NUMERIC)+f' FROM supervised WHERE phase={qs(phase)}')[0]
    checks={}
    def ck(name,sql):
        checks[name]=bool(con.execute(sql).fetchone()[0])
    ck('targets_cover_all_eligible_pairs',f'SELECT count(*)={eligible} FROM targets')
    ck('same_label_availability_for_both_alphas','SELECT count(*)=0 FROM label0 a FULL JOIN label1 b USING(user_id,course_id) WHERE a.user_id IS NULL OR b.user_id IS NULL OR (a.y IS NULL)<>(b.y IS NULL) OR a.split IS DISTINCT FROM b.split')
    ck('four_supervised_rows_per_pair',f'SELECT count(*)={labeled*4} AND count(*)=count(DISTINCT(user_id,course_id,phase)) FROM supervised')
    ck('no_null_targets_in_supervised','SELECT count(*)=0 FROM supervised WHERE y_alpha_0p1 IS NULL OR y_alpha_1 IS NULL')
    ck('same_split_across_phases','SELECT count(*)=0 FROM (SELECT user_id,course_id FROM supervised GROUP BY 1,2 HAVING count(*)<>4 OR count(DISTINCT split)<>1)')
    ck('courses_disjoint_between_splits','SELECT count(*)=0 FROM (SELECT course_id FROM supervised GROUP BY 1 HAVING count(DISTINCT split)>1)')
    ck('features_use_v2_split','SELECT count(*)=0 FROM all_x x JOIN course_splits s USING(course_id) WHERE x.split IS DISTINCT FROM s.split')
    ck('no_future_or_pre_enrollment_events',"SELECT count(*)=0 FROM audits WHERE n_out_of_window_events<>0 OR min_feature_event_time<enroll_time OR max_feature_event_time>cutoff_time+INTERVAL '1 microsecond'")
    ck('finite_raw_when_observed','SELECT count(*)=0 FROM all_x WHERE '+' OR '.join(f'({x} IS NOT NULL AND NOT isfinite({x}))' for x in NUMERIC))
    ck('nonnegative_counts','SELECT count(*)=0 FROM all_x WHERE '+' OR '.join(f'{x} < -1e-9' for x in sorted(old.ZERO_FEATURES)))
    ck('catalog_coverage_bounded','SELECT count(*)=0 FROM all_x WHERE '+' OR '.join(f'{x}< -1e-9 OR {x}>1+1e-9' for x in ['video_coverage_ratio','exercise_coverage_ratio','problem_coverage_ratio','chapter_coverage_ratio']))
    ck('unknown_chapter_separate_from_catalog_coverage','SELECT count(*)=0 FROM all_x WHERE n_chapters_touched_all<>n_chapters_touched+n_unknown_chapters_touched OR n_unknown_chapters_touched>1')
    ck('rewinds_are_subset_of_invalid_segments','SELECT count(*)=0 FROM all_x WHERE n_rewind_segments>n_invalid_video_segments OR n_invalid_video_segments>n_video_events')
    ck('video_union_not_greater_than_total','SELECT count(*)=0 FROM all_x WHERE video_unique_seconds>video_media_seconds+1e-7')
    ck('sentiment_null_without_comment','SELECT count(*)=0 FROM all_x WHERE n_comments=0 AND (sentiment_p_positive_mean IS NOT NULL OR sentiment_p_neutral_mean IS NOT NULL OR sentiment_p_negative_mean IS NOT NULL)')
    ck('sentiment_complete_for_observed_comments','SELECT count(*)=0 FROM all_x WHERE n_comments>0 AND (sentiment_p_positive_mean IS NULL OR abs(sentiment_p_positive_mean+sentiment_p_neutral_mean+sentiment_p_negative_mean-1)>1e-6)')
    ck('reply_rate_bounded','SELECT count(*)=0 FROM all_x WHERE n_comments_with_reply>n_comments OR comment_reply_rate<0 OR comment_reply_rate>1')
    ck('ready_preserves_supervised_keys','SELECT count(*)=0 FROM supervised ANTI JOIN ready USING(user_id,course_id,phase)')
    ck('sequences_preserve_supervised_keys','SELECT count(*)=0 FROM supervised ANTI JOIN sequences USING(user_id,course_id,phase)')
    ck('ready_and_sequences_row_counts',f'SELECT (SELECT count(*) FROM ready)={labeled*4} AND (SELECT count(*) FROM sequences)={labeled*4}')
    ck('no_final_target_scalers_in_x','SELECT true')
    checks['no_final_target_scalers_in_x']=not(set(NUMERIC+CATEGORICAL)&set(TARGETS+['COELO','AFELO','ACELO','FA','FV','TS','IF','q','y','course_end','cutoff_time']))
    ck('train_vocabulary_only','SELECT count(*)=0 FROM enc_learner e ANTI JOIN (SELECT DISTINCT user_id token FROM targets WHERE y IS NOT NULL AND split=\'train\') t USING(token)')
    for phase in PHASES:
        xp=out/'features'/phase/'*.parquet'
        ck(phase+'_all_eligible_rows',f'SELECT count(*)={eligible} AND count(*)=count(DISTINCT(user_id,course_id)) FROM read_parquet({qs(xp)})')
        ck(phase+'_exact_stored_cutoff',f'''SELECT count(*)=0 FROM read_parquet({qs(xp)}) f
          FULL JOIN (SELECT * FROM time_pairs WHERE time_eligible) t USING(user_id,course_id)
          WHERE f.user_id IS NULL OR t.user_id IS NULL OR f.cutoff_time IS DISTINCT FROM t.cutoff_{phase}''')
        ck(phase+'_final_labels_unchanged',f'''SELECT count(*)=0 FROM supervised x JOIN targets t USING(user_id,course_id)
          WHERE x.phase={qs(phase)} AND ({' OR '.join('x.'+v+' IS DISTINCT FROM t.'+v for v in TARGETS)})''')
        active=list(preprocessing[phase]['numeric'])
        ck(phase+'_ready_numeric_complete',f'''SELECT count(*)=0 FROM ready WHERE phase={qs(phase)} AND
          ({' OR '.join(x+' IS NULL OR NOT isfinite('+x+')' for x in active)})''')
        ck(phase+'_training_fit_count',f"SELECT count(*)={preprocessing[phase]['fit_rows']} FROM supervised WHERE phase={qs(phase)} AND split='train'")
        for split in ['train','dev','test']:
            ck(phase+'_'+split+'_only_correct_split',f'SELECT count(*)=0 FROM read_parquet({qs(out/"supervised_raw"/phase/(split+".parquet"))}) WHERE split<>{qs(split)}')
    summary['checks_passed']=sum(checks.values());summary['checks_total']=len(checks)
    return summary,checks


def write_report(out,s,preprocessing):
    fmt=lambda x:f'{int(x):,}'.replace(',','.')
    lines=['# Feature P1–P4 và train/dev/test theo cutoff v2','', 'Ngày xây dựng: 20/10/2026. Đầu ra riêng `phase_features_v2/`; không ghi đè v1 hoặc nhãn quality v2.',
      '', '## Quy mô và split', '',f'Mỗi phase giữ **{fmt(s["eligible_pairs"])}** cặp đủ điều kiện. Tập supervised giữ **{fmt(s["labeled_pairs"])}** cặp mỗi phase, kể cả snapshot chưa có hoạt động. Hai alpha 0.1 và 1 dùng chung X và cùng split.',
      '', '| Split | User-course có nhãn mỗi phase | Course có nhãn |','|---|---:|---:|']
    for r in s['labeled_splits']:
        lines.append(f'| {r["split"]} | {fmt(r["n"])} | {fmt(r["n_courses"])} |')
    lines += ['', 'Kế thừa split v2: xếp course theo course_start và chia 70/10/20 course. Tỷ lệ số user-course có nhãn không bắt buộc bằng 70/10/20. Một course thuộc đúng một split; một user có thể có course ở nhiều split. Cặp user-course giữ cùng split ở cả bốn phase.',
      '', '## Cửa sổ và nguồn dữ liệu', '',
      '- Dùng trực tiếp cutoff_P1..P4 đã lưu ở final_quality_v2, không làm tròn thêm. Cửa sổ tích lũy [enroll_time,cutoff]. Epoch chuyển UTC giống pipeline v2; không tự cộng 7/8 giờ.',
      '- Video bắt đầu trước cutoff nhưng kết thúc sau cutoff được cắt theo thời gian còn lại và speed. Hợp interval nội dung sau khi cắt để tính unique/rewatch; không chia tỷ lệ từ tổng cuối kỳ.',
      '- Start > end: đếm tua ngược và event/visit, không cộng thời lượng xem. Start = end được đếm riêng. Mean tốc độ/thời lượng/recency video hợp lệ giữ NULL khi không tính được.',
      '- Problem là trạng thái tổng hợp, chỉ dùng khi submit_time<=cutoff; không dựng thời điểm giả cho từng attempt. Các feature score/efficiency baseline vẫn là feature hành vi trước cutoff; chúng không thay thế công thức ACELO v2 dùng làm nhãn.',
      '- Comment/reply khôi phục chapter -1 và sentiment v2 được dùng theo timestamp. Phản hồi nhận được chỉ có hiệu lực sau khi reply xuất hiện, dù reply được tạo bởi user ngoài cohort supervised.',
      '- Đếm n_visits_30m bằng phiên user-resource, ngắt khi gap>30 phút, gom video/problem/comment/reply giống v2. N_events và session dùng cả comment đã mapping/thời gian hợp lệ trước lọc text; n_comments và sentiment chỉ dùng comment có text hợp lệ, khử trùng chính xác.',
      '- n_chapters_touched chỉ đếm chapter catalog để tính coverage; n_unknown_chapters_touched và n_chapters_touched_all lưu riêng chapter -1, không làm coverage vượt 1.',
      '', '## Feature, nhãn và xử lý thiếu', '',
      f'Có **{s["numeric_features"]} feature số và {s["categorical_features"]} mã danh mục**. Kế thừa baseline log hiện có, thêm đếm event video, tua ngược/invalid/zero-length, lượt ghé và chapter chưa mapping. Không tuyên bố tái lập danh sách gốc TEMPO-DQ.',
      '', 'Feature thô ở `features/` và `supervised_raw/` giữ NULL của rate/mean/recency không xác định; số đếm vắng log là 0 trong phạm vi nguồn. Không bỏ mẫu vì feature thiếu hoặc phase chưa có hoạt động.',
      '', 'Hai target q_alpha_0p1/y_alpha_0p1 và q_alpha_1/y_alpha_1 được ghép sau aggregate feature, cố định ở mọi phase. Không đưa target, các chỉ số cuối kỳ, P95 cuối kỳ, course_end hay cutoff_time vào allowlist X.',
      '', 'Model-ready: median imputation và standard scaling fit riêng trên train của từng phase, dùng đúng tham số đó cho dev/test của phase; không dùng q/y để fit. Code danh mục fit một lần từ các cặp train có nhãn, giá trị chưa thấy/missing ->0; gender giữ mã nguồn và -1 cho thiếu/không hợp lệ. Mã danh mục không dùng như tọa độ khoảng cách.',
      '', '## File bàn giao', '',
      '- `features/P1/part-*.parquet` đến P4: feature thô mọi cặp đủ thời gian, chưa ghép nhãn.',
      '- `supervised_raw/P1/train.parquet`, `dev.parquet`, `test.parquet` đến P4: dữ liệu có nhãn, chưa impute, chứa cả hai target.',
      '- `model_ready/P1/train.parquet`, `dev.parquet`, `test.parquet` đến P4: dữ liệu đã xử lý bằng tham số train cùng phase.',
      '- `sequences/P1/train.parquet`, `dev.parquet`, `test.parquet` đến P4: chuỗi tuần thưa tính từ enrollment; snapshot trống giữ mảng rỗng. Lượng của segment được gán tuần bắt đầu sự kiện sau khi clip cutoff.',
      '- `tables/labeled_user_course_splits.parquet`: assignment user-course có nhãn và cả hai target. `tables/user_course_targets.parquet`: mọi cặp đủ thời gian và target NULL khi chưa có nhãn.',
      '- `feature_columns.json`, `feature_dictionary.json`, `preprocessing.json`, `load_data.py`: hợp đồng cột, công thức và hàm đọc theo phase/split/alpha.',
      '- `audit/`, `validation.json`, `summary.json`, `run_manifest.json`: bằng chứng cửa sổ, độ phủ, nhãn và lineage.',
      '', '## Nhãn theo split', '']
    for suffix in ['0p1','1']:
        lines += [f'Alpha {suffix.replace("p",".")}:','','| Split | E | G | A |','|---|---:|---:|---:|']
        for split in ['train','dev','test']:
            d={r['y']:r['n'] for r in s['class_counts'][suffix] if r['split']==split}
            lines.append(f'| {split} | {fmt(d.get("E",0))} | {fmt(d.get("G",0))} | {fmt(d.get("A",0))} |')
        lines.append('')
    lines += ['## Kiểm tra và giới hạn','',f'Đạt **{s["checks_passed"]}/{s["checks_total"]}** kiểm tra dữ liệu; thêm fixture cắt video, reply tương lai, problem chưa submit và snapshot trống.',
      '', '- Course-end lấy từ hoạt động cuối quan sát được: mốc chia phase là hồi cứu, không phải ngày kết thúc đã biết lúc triển khai. Có thể đổi sang lịch end biết trước hoặc horizon cố định nếu đánh giá online.',
      '- Course không giao giữa split nhưng thời gian course có thể chồng nhau; user có thể giao giữa split. Chưa tuyên bố đánh giá không có mọi dạng phụ thuộc giữa người học hoặc thời gian.',
      '- Tracking video/problem/social có khoảng khác nhau. Không có log không chứng minh không hoạt động. Catalog và profile được coi là snapshot tĩnh vì thiếu lịch sử hiệu lực.',
      '- Giữ raw để có thể chọn phương án impute/scaler khác; nếu gộp nhiều phase trong một mô hình, fit lại preprocessing trên train của pool đó, không trộn các scaler per-phase một cách ngầm định.','']
    (out/'BAO_CAO_PHASE_FEATURES_V2.md').write_text('\n'.join(lines),encoding='utf-8')


def write_dictionary(out):
    # Reuse exact baseline definitions and append v2 definitions/semantic changes.
    definitions={
      'n_video_events':'Số event video có timestamp trong cửa sổ, gồm hợp lệ, tua ngược, zero-length và valid start đúng cutoff chưa có thời lượng.',
      'n_rewind_segments':'Số segment start_point>end_point với vị trí hữu hạn; proxy tua ngược, không cộng duration.',
      'n_invalid_video_segments':'Số video event không hợp lệ về interval/speed, vẫn là hoạt động timestamped.',
      'n_zero_length_segments':'Số video event start_point=end_point; duration=0, không gộp với tua ngược.',
      'n_visits_30m':'Số phiên cùng user-resource (giữ course/chapter), ngắt gap>30 phút, từ video/problem/comment/reply; không đếm reply nhận thêm lần nữa.',
      'n_unknown_chapters_touched':'Số chapter -1 có hoạt động riêng user trong course; tối đa 1, không cộng vào mẫu số/chỉ số coverage catalog.',
      'n_chapters_touched_all':'Số chapter catalog và chapter -1 có hoạt động riêng user trước cutoff.',
      'n_chapters_touched':'Số chapter catalog có hoạt động riêng user trước cutoff, không gồm chapter -1.',
      'n_events':'Số raw event video, problem, comment đã mapping/thời gian hợp lệ và reply authored; gồm invalid video và comment trước lọc text, không đếm thêm reply received.',
    }
    additions={'n_video_events','n_rewind_segments','n_invalid_video_segments','n_zero_length_segments',
      'n_visits_30m','n_unknown_chapters_touched','n_chapters_touched_all'}
    extended=old.GROUPS
    old.GROUPS={group:[x for x in cols if x not in additions] for group,cols in extended.items()}
    try:
        old.write_dictionary(out)
    finally:
        old.GROUPS=extended
    rows=json.loads((out/'feature_dictionary.json').read_text())
    for group,cols in extended.items():
        for name in cols:
            if name in additions:
                rows.append({'name':name,'group':group,'kind':'numeric','definition':definitions[name],
                  'no_event_rule':'zero','time_scope':'inclusive enrollment through stored v2 cutoff'})
    for row in rows:
        if row['name'] in definitions:
            row['definition']=definitions[row['name']]
    dump(out/'feature_dictionary.json',rows)
    text=['# Từ điển feature v2','', '| Feature | Nhóm | Định nghĩa |','|---|---|---|']
    text += [f'| `{r["name"]}` | {r["group"]} | {r["definition"]} |' for r in rows]
    text += ['', 'Các trường cuối kỳ và P95 label không tham gia X. Median/scaler fit riêng trên train từng phase; raw giữ NULL. Tập feature là baseline từ log có sẵn, không phải danh sách TEMPO-DQ gốc.','']
    (out/'tu_dien_features.md').write_text('\n'.join(text),encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output')
    parser.add_argument('--threads',type=int,default=3)
    parser.add_argument('--memory',default='1500MB')
    args=parser.parse_args()
    build(args.output,args.threads,args.memory)
