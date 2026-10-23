"""Build cutoff-safe user/course/phase features from existing event rows.

All time-eligible pairs are retained; final alpha-0.1 labels are joined only
after features. This is an explicitly defined raw-log baseline, not a claim
of reproducing the unavailable full TEMPO-DQ feature list.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import time
from pathlib import Path

import duckdb

VERSION = 'raw_log_phase_features_v1'
PHASES = {'P1': (1,4), 'P2': (1,2), 'P3': (3,4), 'P4': (9,10)}
GROUPS = {
 'video': ['n_video_segments','n_videos_watched','video_media_seconds','video_wall_seconds',
    'video_unique_seconds','video_rewatch_seconds','video_sessions_30m','n_revisited_videos',
    'video_mean_speed','video_mean_segment_seconds','video_active_days','video_recency_days'],
 'problem': ['n_problem_submissions','n_exercises_attempted','problem_attempts','n_correct_problems',
    'n_first_try_correct','problem_correct_rate','problem_first_try_correct_rate','problem_mean_attempts',
    'problem_attempted_score_ratio','problem_efficiency_mean','problem_active_days','problem_recency_days'],
 'social': ['n_comments','comment_mean_characters','n_replies_authored','n_replies_received',
    'n_comments_with_reply','comment_reply_rate','sentiment_p_positive_mean','sentiment_p_neutral_mean',
    'sentiment_p_negative_mean','social_active_days','social_recency_days'],
 'activity': ['n_events','n_active_days','n_chapters_touched','activity_span_days','activity_recency_days'],
 'context': ['course_n_videos','course_n_exercises','course_n_problems','course_n_chapters',
    'learner_age_at_enrollment','phase_fraction','video_coverage_ratio','exercise_coverage_ratio',
    'problem_coverage_ratio','chapter_coverage_ratio'],
}
NUMERIC = [x for fields in GROUPS.values() for x in fields]
CATEGORICAL = ['learner_code','teacher_set_code','school_set_code','field_set_code','gender_code']
META = ['user_id','course_id','phase','split','enroll_time','cutoff_time']
ZERO_FEATURES = set(['n_video_segments','n_videos_watched','video_media_seconds','video_wall_seconds',
    'video_unique_seconds','video_rewatch_seconds','video_sessions_30m','n_revisited_videos','video_active_days',
    'n_problem_submissions','n_exercises_attempted','problem_attempts','n_correct_problems','n_first_try_correct',
    'problem_active_days','n_comments','n_replies_authored','n_replies_received','n_comments_with_reply',
    'social_active_days','n_events','n_active_days','n_chapters_touched'])
SEQ_FIELDS = ['n_video_segments','video_media_seconds','n_problem_submissions','problem_attempts',
    'n_correct_problems','n_comments','n_replies_authored','n_replies_received']


def qs(value):
    return "'"+str(value).replace("'", "''")+"'"


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def dump_json(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    temporary=p.with_suffix(p.suffix+'.partial')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
    temporary.replace(p)


def fingerprint(raw,model):
    rows=[]
    for kind,root in [('raw',Path(raw)),('model',Path(model))]:
        for p in sorted(root.rglob('*.parquet')):
            s=p.stat();rows.append([kind,str(p.relative_to(root)),s.st_size,s.st_mtime_ns])
    return hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()


def records(con,sql):
    cursor=con.execute(sql);names=[x[0] for x in cursor.description]
    return [dict(zip(names,r)) for r in cursor.fetchall()]


def view(con,name,path):
    con.execute(f'CREATE OR REPLACE VIEW "{name}" AS SELECT * FROM read_parquet({qs(path)},union_by_name=true)')


def replace_table(con,name,sql):
    con.execute(f'CREATE OR REPLACE TEMP TABLE {name} AS {sql}')


def write_parquet(con,path,sql):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    partial=path.with_suffix('.parquet.partial')
    con.execute(f'COPY ({sql}) TO {qs(partial)} (FORMAT PARQUET,COMPRESSION ZSTD,ROW_GROUP_SIZE 50000)')
    partial.replace(path)


def cutoff_sql(start,end,phase):
    num,den=PHASES[phase]
    return f"make_timestamp(epoch_us({start})+(date_diff('microsecond',{start},{end})*{num})//{den})"


def fixtures():
    con=duckdb.connect()
    try:
        # One clip starts before cutoff and ends after it: only 15 media seconds
        # are observable, not the full 60; speed=1.5, wall duration=10 seconds.
        value=con.execute('SELECT least(60.0,greatest(0.0,10.0)*1.5)').fetchone()[0]
        assert value==15
        # A future reply must not alter an earlier feature snapshot.
        assert con.execute("SELECT count(*) FROM (VALUES (TIMESTAMP '2020-01-03')) r(t) WHERE t<=TIMESTAMP '2020-01-02'").fetchone()[0]==0
        # A final aggregate problem row is unavailable before its submit_time.
        assert con.execute("SELECT count(*) FROM (VALUES (TIMESTAMP '2020-01-03',5)) p(t,attempts) WHERE t<=TIMESTAMP '2020-01-02'").fetchone()[0]==0
        # Preserve a pair with no early log even when the final label is known.
        assert con.execute("WITH p(id,y) AS (VALUES (1,'G')),e(id,n) AS (SELECT 1,1 WHERE false) SELECT p.id,coalesce(n,0),y FROM p LEFT JOIN e USING(id)").fetchone()==(1,0,'G')
        return {'video_clip_stops_at_cutoff':True,'future_reply_excluded':True,
            'future_problem_state_excluded':True,'empty_early_snapshot_retained':True}
    finally:con.close()


def prepare_static(con,raw,model,out,save):
    view(con,'users_raw',raw/'tables/entities/user/*.parquet')
    view(con,'course_raw',model/'processed/dim_course/*.parquet')
    view(con,'resources',model/'processed/dim_lo_resource/*.parquet')
    view(con,'lo_catalog',model/'processed/dim_lo_catalog/*.parquet')
    view(con,'lo_problem',model/'processed/bridge_lo_problem/*.parquet')
    view(con,'teachers_raw',raw/'tables/relations/course_teacher/*.parquet')
    view(con,'schools_raw',raw/'tables/relations/course_school/*.parquet')
    save('static_users', '''SELECT id user_id,year_of_birth,
        CASE WHEN gender IN (0,1,2) THEN gender ELSE -1 END gender_code,
        hash(id)%32 user_shard FROM users_raw''')
    replace_table(con,'course_static', '''WITH teachers AS (
        SELECT course_id,string_agg(DISTINCT teacher_id,'|' ORDER BY teacher_id) teacher_signature
        FROM teachers_raw WHERE teacher_id IS NOT NULL GROUP BY 1), schools AS (
        SELECT course_id,string_agg(DISTINCT school_id,'|' ORDER BY school_id) school_signature
        FROM schools_raw WHERE school_id IS NOT NULL GROUP BY 1), rc AS (
        SELECT course_id,count(DISTINCT resource_id) FILTER(WHERE resource_type='video') course_n_videos,
          count(DISTINCT resource_id) FILTER(WHERE resource_type='exercise') course_n_exercises
        FROM resources GROUP BY 1), pc AS (
        SELECT course_id,count(DISTINCT problem_id) course_n_problems FROM lo_problem GROUP BY 1), lc AS (
        SELECT course_id,count(*) course_n_chapters FROM lo_catalog GROUP BY 1)
      SELECT c.course_id,t.teacher_signature,s.school_signature,
        nullif(array_to_string(list_sort(c.field),'|'),'') field_signature,
        coalesce(r.course_n_videos,0) course_n_videos,coalesce(r.course_n_exercises,0) course_n_exercises,
        coalesce(p.course_n_problems,0) course_n_problems,coalesce(l.course_n_chapters,0) course_n_chapters
      FROM course_raw c LEFT JOIN teachers t USING(course_id) LEFT JOIN schools s USING(course_id)
        LEFT JOIN rc r USING(course_id) LEFT JOIN pc p USING(course_id) LEFT JOIN lc l USING(course_id)''')
    write_parquet(con,out/'tables/course_static_metadata.parquet','SELECT * FROM course_static')
    for name,source_sql in [
        ('learner',"SELECT DISTINCT user_id token FROM targets WHERE split='train' AND y IS NOT NULL"),
        ('teacher_set',"SELECT DISTINCT c.teacher_signature token FROM course_static c JOIN targets t USING(course_id) WHERE t.split='train' AND t.y IS NOT NULL AND c.teacher_signature IS NOT NULL"),
        ('school_set',"SELECT DISTINCT c.school_signature token FROM course_static c JOIN targets t USING(course_id) WHERE t.split='train' AND t.y IS NOT NULL AND c.school_signature IS NOT NULL"),
        ('field_set',"SELECT DISTINCT c.field_signature token FROM course_static c JOIN targets t USING(course_id) WHERE t.split='train' AND t.y IS NOT NULL AND c.field_signature IS NOT NULL")]:
        replace_table(con,f'enc_{name}',f'SELECT token,row_number() OVER(ORDER BY token)::INTEGER code FROM ({source_sql})')
        write_parquet(con,out/f'encoders/{name}.parquet',f'SELECT * FROM enc_{name}')
    save('course_encoded', '''SELECT c.*,
        coalesce(t.code,0) teacher_set_code,coalesce(s.code,0) school_set_code,coalesce(f.code,0) field_set_code
        FROM course_static c LEFT JOIN enc_teacher_set t ON c.teacher_signature=t.token
        LEFT JOIN enc_school_set s ON c.school_signature=s.token LEFT JOIN enc_field_set f ON c.field_signature=f.token''')


def build_windows(con,phase,shard):
    cut=cutoff_sql('p.enroll_time','p.course_end',phase)
    replace_table(con,'pairs',f'''SELECT p.user_id,p.course_id,p.enroll_time,p.course_end,{cut} cutoff_time,s.split,
        c.course_n_videos,c.course_n_exercises,c.course_n_problems,c.course_n_chapters,
        CASE WHEN year(p.enroll_time)-u.year_of_birth BETWEEN 10 AND 100 AND u.year_of_birth>=1900
             THEN (year(p.enroll_time)-u.year_of_birth)::DOUBLE END learner_age_at_enrollment,
        coalesce(e.code,0) learner_code,c.teacher_set_code,c.school_set_code,c.field_set_code,
        coalesce(u.gender_code,-1) gender_code
      FROM pair_slice p LEFT JOIN course_splits s USING(course_id) LEFT JOIN course_encoded c USING(course_id)
      LEFT JOIN (SELECT * FROM static_users WHERE user_shard={shard}) u USING(user_id)
      LEFT JOIN enc_learner e ON p.user_id=e.token''')
    replace_table(con,'video_w',f'''WITH selected AS (
        SELECT v.*,p.enroll_time,p.cutoff_time,
          least(v.end_point-v.start_point,greatest(0.0,epoch(p.cutoff_time)-epoch(v.event_time))*v.speed) observed_media_seconds
        FROM video_events v JOIN pairs p USING(user_id,course_id)
        WHERE hash(v.user_id)%32={shard} AND v.valid_measurement
          AND v.event_time BETWEEN p.enroll_time AND p.cutoff_time)
      SELECT *,start_point+observed_media_seconds observed_end_point,
        event_time+(observed_media_seconds/speed)*INTERVAL '1 second' observed_end_time
      FROM selected WHERE observed_media_seconds>0''')
    replace_table(con,'problem_w', '''SELECT p.*,x.enroll_time,x.cutoff_time FROM problem_slice p
        JOIN pairs x USING(user_id,course_id) WHERE p.event_time BETWEEN x.enroll_time AND x.cutoff_time''')
    replace_table(con,'comment_w',f'''SELECT c.*,x.cutoff_time FROM comments c JOIN pairs x USING(user_id,course_id)
        WHERE hash(c.user_id)%32={shard} AND c.normalized_text IS NOT NULL AND c.exact_duplicate_rank=1
          AND c.event_time BETWEEN x.enroll_time AND x.cutoff_time''')
    replace_table(con,'reply_w',f'''SELECT r.*,p.enroll_time,p.cutoff_time FROM replies r JOIN pairs p USING(user_id,course_id)
        WHERE hash(r.user_id)%32={shard} AND r.event_time BETWEEN p.enroll_time AND p.cutoff_time''')
    replace_table(con,'received_w', '''SELECT c.user_id,c.course_id,c.lo_key,c.comment_id,r.reply_id,r.event_time,
        c.enroll_time,c.cutoff_time FROM comment_w c JOIN replies r ON c.comment_id=r.parent_comment_id
        WHERE r.event_time BETWEEN c.event_time AND c.cutoff_time''')


def aggregate_windows(con):
    replace_table(con,'video_features', '''WITH previous AS (
        SELECT *,max(observed_end_point) OVER(PARTITION BY user_id,course_id,video_id ORDER BY start_point,observed_end_point
          ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) previous_end FROM video_w), islands AS (
        SELECT *,sum(CASE WHEN previous_end IS NULL OR start_point>previous_end THEN 1 ELSE 0 END)
          OVER(PARTITION BY user_id,course_id,video_id ORDER BY start_point,observed_end_point) island FROM previous),
      merged AS (SELECT user_id,course_id,video_id,island,max(observed_end_point)-min(start_point) union_seconds FROM islands GROUP BY 1,2,3,4),
      unique_time AS (SELECT user_id,course_id,sum(union_seconds) video_unique_seconds FROM merged GROUP BY 1,2),
      edges AS (SELECT *,lag(event_time) OVER(PARTITION BY user_id,course_id ORDER BY event_time,video_id,start_point) prev_course,
        lag(event_time) OVER(PARTITION BY user_id,course_id,video_id ORDER BY event_time,start_point) prev_video FROM video_w),
      revisits AS (SELECT user_id,course_id,video_id,count(*) FILTER(WHERE prev_video IS NULL OR event_time-prev_video>INTERVAL '30 minutes') n_sessions FROM edges GROUP BY 1,2,3),
      revisit_counts AS (SELECT user_id,course_id,count(*) FILTER(WHERE n_sessions>1) n_revisited_videos FROM revisits GROUP BY 1,2),
      totals AS (SELECT user_id,course_id,count(*) n_video_segments,count(DISTINCT video_id) n_videos_watched,
        sum(observed_media_seconds) video_media_seconds,sum(observed_media_seconds/speed) video_wall_seconds,
        count(*) FILTER(WHERE prev_course IS NULL OR event_time-prev_course>INTERVAL '30 minutes') video_sessions_30m,
        avg(speed) video_mean_speed,avg(observed_media_seconds) video_mean_segment_seconds,
        count(DISTINCT cast(event_time AS DATE)) video_active_days,
        (epoch(max(cutoff_time))-epoch(max(observed_end_time)))/86400.0 video_recency_days
        FROM edges GROUP BY 1,2)
      SELECT t.*,u.video_unique_seconds,greatest(0.0,t.video_media_seconds-u.video_unique_seconds) video_rewatch_seconds,
        r.n_revisited_videos FROM totals t LEFT JOIN unique_time u USING(user_id,course_id)
        LEFT JOIN revisit_counts r USING(user_id,course_id)''')
    replace_table(con,'problem_features', '''SELECT user_id,course_id,count(*) n_problem_submissions,
        count(DISTINCT exercise_id) n_exercises_attempted,
        sum(CASE WHEN attempts>0 THEN attempts ELSE 0 END) problem_attempts,
        count(*) FILTER(WHERE is_correct=1) n_correct_problems,
        count(*) FILTER(WHERE is_correct=1 AND attempts=1) n_first_try_correct,
        avg(is_correct::DOUBLE) problem_correct_rate,
        count(*) FILTER(WHERE is_correct=1 AND attempts=1)::DOUBLE/count(*) problem_first_try_correct_rate,
        avg(CASE WHEN attempts>0 THEN attempts END) problem_mean_attempts,
        CASE WHEN count(*) FILTER(WHERE NOT score_valid AND NOT fallback_possible)=0
          THEN sum(CASE WHEN score_valid THEN score WHEN fallback_possible THEN is_correct*problem_max_score END)
            /nullif(sum(problem_max_score),0) END problem_attempted_score_ratio,
        avg(CASE WHEN efficiency_valid THEN is_correct::DOUBLE/attempts END) problem_efficiency_mean,
        count(DISTINCT cast(event_time AS DATE)) problem_active_days,
        (epoch(max(cutoff_time))-epoch(max(event_time)))/86400.0 problem_recency_days
        FROM problem_w GROUP BY 1,2''')
    replace_table(con,'comment_features', '''SELECT c.user_id,c.course_id,count(*) n_comments,
        avg(length(c.text)) comment_mean_characters,avg(s.p_pos) sentiment_p_positive_mean,
        avg(s.p_neu) sentiment_p_neutral_mean,avg(s.p_neg) sentiment_p_negative_mean
        FROM comment_w c LEFT JOIN sentiment s USING(comment_id) GROUP BY 1,2''')
    replace_table(con,'reply_features', 'SELECT user_id,course_id,count(*) n_replies_authored FROM reply_w GROUP BY 1,2')
    replace_table(con,'received_features', '''SELECT user_id,course_id,count(*) n_replies_received,
        count(DISTINCT comment_id) n_comments_with_reply FROM received_w GROUP BY 1,2''')
    # Weekly sequence event values are assigned to the event's starting week;
    # no segment is longer than 44 minutes in the validated current inputs.
    selections=[]
    types=[('video_w','video','observed_end_time',{'n_video_segments':'1','video_media_seconds':'observed_media_seconds'}),
      ('problem_w','problem','event_time',{'n_problem_submissions':'1','problem_attempts':'CASE WHEN attempts>0 THEN attempts ELSE 0 END','n_correct_problems':'(is_correct=1)::INTEGER'}),
      ('comment_w','comment','event_time',{'n_comments':'1'}),
      ('reply_w','reply','event_time',{'n_replies_authored':'1'}),
      ('received_w','received','event_time',{'n_replies_received':'1'})]
    for table,kind,end,values in types:
        fields=','.join(f'{values.get(name,"0")} {name}' for name in SEQ_FIELDS)
        selections.append(f"SELECT user_id,course_id,lo_key,event_time,{end} observed_end_time,enroll_time,cutoff_time,{qs(kind)} event_kind,{fields} FROM {table}")
    replace_table(con,'activity_events',' UNION ALL '.join(selections))
    replace_table(con,'activity_features', '''SELECT user_id,course_id,
        count(*) FILTER(WHERE event_kind<>'received') n_events,
        count(DISTINCT cast(event_time AS DATE)) FILTER(WHERE event_kind<>'received') n_active_days,
        count(DISTINCT lo_key) FILTER(WHERE event_kind<>'received') n_chapters_touched,
        (epoch(max(observed_end_time) FILTER(WHERE event_kind<>'received'))-epoch(min(event_time) FILTER(WHERE event_kind<>'received')))/86400.0 activity_span_days,
        (epoch(max(cutoff_time))-epoch(max(observed_end_time) FILTER(WHERE event_kind<>'received')))/86400.0 activity_recency_days,
        count(DISTINCT cast(event_time AS DATE)) FILTER(WHERE event_kind IN ('comment','reply','received')) social_active_days,
        (epoch(max(cutoff_time))-epoch(max(event_time) FILTER(WHERE event_kind IN ('comment','reply','received'))))/86400.0 social_recency_days,
        min(event_time) min_feature_event_time,max(observed_end_time) max_feature_event_time,
        count(*) FILTER(WHERE event_time<enroll_time OR event_time>cutoff_time OR observed_end_time>cutoff_time+INTERVAL '1 microsecond') n_out_of_window_events
        FROM activity_events GROUP BY 1,2''')
    sums=','.join(f'sum({field}) {field}' for field in SEQ_FIELDS)
    replace_table(con,'weekly_features',f'''SELECT user_id,course_id,
        floor((epoch(event_time)-epoch(enroll_time))/(7*86400.0))::INTEGER week_index,
        {sums} FROM activity_events GROUP BY 1,2,3''')


def feature_sql(phase):
    feature_alias={name:'v' for name in GROUPS['video']}
    feature_alias.update({name:'p' for name in GROUPS['problem']})
    feature_alias.update({name:'a' for name in GROUPS['activity']})
    feature_alias.update({name:'c' for name in ['n_comments','comment_mean_characters','sentiment_p_positive_mean','sentiment_p_neutral_mean','sentiment_p_negative_mean']})
    feature_alias.update({'n_replies_authored':'r','n_replies_received':'z','n_comments_with_reply':'z',
        'social_active_days':'a','social_recency_days':'a'})
    fields=[]
    for name in NUMERIC:
        if name in feature_alias:
            expr=f'{feature_alias[name]}.{name}'
            if name in ZERO_FEATURES:expr=f'coalesce({expr},0)'
        elif name=='comment_reply_rate':expr='coalesce(z.n_comments_with_reply,0)::DOUBLE/nullif(c.n_comments,0)'
        elif name=='phase_fraction':
            n,d=PHASES[phase];expr=f'{n}::DOUBLE/{d}'
        elif name=='video_coverage_ratio':expr='coalesce(v.n_videos_watched,0)::DOUBLE/nullif(b.course_n_videos,0)'
        elif name=='exercise_coverage_ratio':expr='coalesce(p.n_exercises_attempted,0)::DOUBLE/nullif(b.course_n_exercises,0)'
        elif name=='problem_coverage_ratio':expr='coalesce(p.n_problem_submissions,0)::DOUBLE/nullif(b.course_n_problems,0)'
        elif name=='chapter_coverage_ratio':expr='coalesce(a.n_chapters_touched,0)::DOUBLE/nullif(b.course_n_chapters,0)'
        else:expr=f'b.{name}'
        fields.append(f'{expr} {name}')
    return f'''SELECT b.user_id,b.course_id,{qs(phase)} phase,b.split,b.enroll_time,b.cutoff_time,
        {','.join(fields)},{','.join('b.'+x for x in CATEGORICAL)}
      FROM pairs b LEFT JOIN video_features v USING(user_id,course_id)
      LEFT JOIN problem_features p USING(user_id,course_id) LEFT JOIN comment_features c USING(user_id,course_id)
      LEFT JOIN reply_features r USING(user_id,course_id) LEFT JOIN received_features z USING(user_id,course_id)
      LEFT JOIN activity_features a USING(user_id,course_id)'''


def run_build(quality_root=None,label_root=None,output_root=None,raw_root=None,model_root=None,
              time_root=None,memory_limit='2GB',threads=2):
    started=time.monotonic()
    base=Path.cwd()/'working_13-10' if (Path.cwd()/'working_13-10').is_dir() else Path.cwd()
    quality=Path(quality_root or base/'final_quality_tables').resolve()
    labels=Path(label_root or base/'weighted_loqv_alpha_0p1').resolve()
    out=Path(output_root or base/'phase_features').resolve()
    qm=json.loads((quality/'run_manifest.json').read_text())
    lm=json.loads((labels/'run_manifest.json').read_text())
    assert qm['finalization']['afelo_mode']=='sentiment'
    assert lm['alpha']==0.1,'This feature/target delivery uses the approved alpha=0.1 labels.'
    for root,manifest in [(quality,qm),(labels,lm)]:
        done=json.loads((root/'_SUCCESS.json').read_text())
        assert done['manifest_sha256']==sha256(root/'run_manifest.json'),f'Stale source completion marker: {root}'
    raw=Path(raw_root or (base/'input/moocubex_parquet' if (base/'input/moocubex_parquet').exists() else qm['raw_root'])).resolve()
    model=Path(model_root or (base/'input/moocubex_model' if (base/'input/moocubex_model').exists() else qm['model_root'])).resolve()
    times=Path(time_root or (base/'review_time_phases' if (base/'review_time_phases').exists() else qm['time_root'])).resolve()
    for source in [quality,labels,raw,model,times]:
        assert source.is_dir(),source
        if out==source or source in out.parents or out in source.parents:
            raise ValueError('Phase feature output must be separate from input roots.')
    fp=fingerprint(raw,model)
    assert fp==qm['input_fingerprint'],'Raw/model snapshot differs from validated event stages.'
    problem_stage=raw.parent.parent/'eda_engagement_features_comparison/_stage/problem_rows'
    pm_path=problem_stage.parent.parent/'run_manifest.json'
    assert problem_stage.is_dir() and json.loads(pm_path.read_text())['input_fingerprint']==fp, 'Verified problem-row stage unavailable.'
    pair_dir=quality/'_stage/eligible_pair_shards'
    assert json.loads((pair_dir/'_SUCCESS.json').read_text())['time_sha256']==sha256(times/'run_manifest.json')
    source_hash=globals().get('FEATURE_SOURCE_SHA256') or sha256(__file__)
    con=duckdb.connect()
    con.execute(f'SET threads={int(threads)}');con.execute(f'SET memory_limit={qs(memory_limit)}')
    for folder in [out,out/'_stage',out/'_scratch',out/'tables',out/'encoders']:
        folder.mkdir(parents=True,exist_ok=True)
    con.execute(f'SET temp_directory={qs(out/"_scratch")}')
    con.execute("SET max_temp_directory_size='3GB'");con.execute("SET TimeZone='UTC'")
    con.execute('SET preserve_insertion_order=false')
    if (out/'_SUCCESS.json').exists():(out/'_SUCCESS.json').unlink()
    view(con,'source_targets',labels/'tables/user_course_loqv_weighted.parquet')
    con.execute('''CREATE VIEW targets AS SELECT user_id,course_id,split,q_weighted q,y_weighted y,
        label_status_weighted label_status,afelo_alpha target_alpha,weighted_formula_version target_formula_version
        FROM source_targets''')
    cohort=records(con,"SELECT count(*) n,sum(hash(user_id,course_id,split)) key_hash_sum FROM targets WHERE y IS NOT NULL")[0]
    stage_files=[quality/'_stage/video_window.parquet',quality/'_stage/comment_window.parquet',
        quality/'_stage/reply_prepared.parquet',quality/'tables/afelo_comment_links.parquet']
    sentiment_path=Path(qm['finalization']['sentiment_path'])
    if not sentiment_path.is_file():sentiment_path=base/'afelo_sentiment_cardiff/tables/comment_sentiment.parquet'
    assert sha256(sentiment_path)==qm['finalization']['sentiment_sha256']
    stage_info=[[str(p.relative_to(problem_stage)),p.stat().st_size,p.stat().st_mtime_ns]
        for p in sorted(problem_stage.rglob('*.parquet'))]
    feature_lineage={'input_fingerprint':fp,'time_manifest_sha256':sha256(times/'run_manifest.json'),
        'source_engine_sha256':source_hash,'cohort_keys':cohort,
        'event_stage_sha256':{p.name:sha256(p) for p in stage_files},
        'problem_stage_files':stage_info,'sentiment_sha256':sha256(sentiment_path)}
    signature=hashlib.sha256(json.dumps(feature_lineage,sort_keys=True,default=str).encode()).hexdigest()
    checkpoint=out/'checkpoints.json'
    cache=json.loads(checkpoint.read_text()) if checkpoint.exists() else {}
    def save(name,sql):
        target=out/'_stage'/f'{name}.parquet'
        step=hashlib.sha256((signature+sql).encode()).hexdigest()
        if not target.exists() or cache.get(str(target.relative_to(out)))!=step:
            write_parquet(con,target,sql);cache[str(target.relative_to(out))]=step;dump_json(checkpoint,cache)
        view(con,name,target)
    write_parquet(con,out/'tables/user_course_targets.parquet','SELECT * FROM targets')
    view(con,'course_splits',quality/'tables/course_splits.parquet')
    view(con,'time_pairs',times/'tables/user_course_phase_times.parquet')
    view(con,'video_events',quality/'_stage/video_window.parquet')
    view(con,'comments',quality/'_stage/comment_window.parquet')
    view(con,'replies',quality/'_stage/reply_prepared.parquet')
    view(con,'sentiment',sentiment_path)
    print('Preparing static attributes and train-only categorical vocabularies',flush=True)
    prepare_static(con,raw,model,out,save)
    per_file_checks=[]
    sequence_fields=','.join(SEQ_FIELDS)
    sequence_type='STRUCT(week_index INTEGER,'+','.join(f'{x} DOUBLE' for x in SEQ_FIELDS)+')[]'
    for shard in range(32):
        pair_path=pair_dir/f'user_shard={shard}'/'*.parquet'
        view(con,'pair_slice',pair_path)
        view(con,'problem_slice',problem_stage/f'user_shard={shard}'/'*.parquet')
        for phase in PHASES:
            tick=time.monotonic()
            paths={kind:out/kind/phase/f'part-{shard:03d}.parquet' for kind in ['features','audit','sequences']}
            token=f'{phase}/{shard:03d}'
            if cache.get(token)==signature and all(p.exists() for p in paths.values()):
                print('REUSE',token,flush=True)
                continue
            build_windows(con,phase,shard)
            aggregate_windows(con)
            write_parquet(con,paths['features'],feature_sql(phase))
            write_parquet(con,paths['audit'],f'''SELECT p.user_id,p.course_id,{qs(phase)} phase,p.enroll_time,p.cutoff_time,
                a.min_feature_event_time,a.max_feature_event_time,a.n_out_of_window_events,
                coalesce(a.n_events,0)>0 has_observed_activity
                FROM activity_features a JOIN pairs p USING(user_id,course_id)''')
            write_parquet(con,paths['sequences'],f'''SELECT user_id,course_id,{qs(phase)} phase,
                list(struct_pack(week_index:=week_index,{','.join(f'{x}:={x}::DOUBLE' for x in SEQ_FIELDS)}) ORDER BY week_index) behavior_sequence
                FROM weekly_features GROUP BY 1,2''')
            view(con,'written_features',paths['features'])
            check=records(con,f'''SELECT count(*) n_rows,count(DISTINCT(user_id,course_id)) n_unique,
                count(*) FILTER(WHERE split IS NULL OR hash(user_id)%32<>{shard}) n_bad_meta
                FROM written_features''')[0]
            assert check['n_rows']==con.execute('SELECT count(*) FROM pairs').fetchone()[0]
            assert check['n_rows']==check['n_unique'] and check['n_bad_meta']==0,check
            assert con.execute('SELECT coalesce(sum(n_out_of_window_events),0) FROM activity_features').fetchone()[0]==0
            assert con.execute('SELECT count(*) FROM video_features WHERE video_unique_seconds>video_media_seconds+1e-7 OR video_recency_days< -1e-9').fetchone()[0]==0
            per_file_checks.append(dict(phase=phase,shard=shard,**check))
            cache[token]=signature;dump_json(checkpoint,cache)
            print(f'BUILT {token}: {check["n_rows"]:,} pairs, {time.monotonic()-tick:.2f}s',flush=True)
    for phase in PHASES:
        view(con,'phase_x',out/'features'/phase/'*.parquet')
        view(con,'phase_sequence',out/'sequences'/phase/'*.parquet')
        # Label availability selects supervised rows; feature nulls never select rows.
        write_parquet(con,out/'supervised_raw'/f'{phase}.parquet', '''SELECT f.*,t.q,t.y,
            t.target_alpha,t.target_formula_version FROM phase_x f JOIN targets t USING(user_id,course_id)
            WHERE t.y IS NOT NULL''')
        write_parquet(con,out/'supervised_sequences'/f'{phase}.parquet',f'''SELECT f.user_id,f.course_id,f.phase,f.split,
            coalesce(s.behavior_sequence,[]::{sequence_type}) behavior_sequence
            FROM read_parquet({qs(out/'supervised_raw'/f'{phase}.parquet')}) f
            LEFT JOIN phase_sequence s USING(user_id,course_id,phase)''')
    view(con,'supervised_raw',out/'supervised_raw/*.parquet')
    # Fit on Pool B train snapshots only. Actual q/y values are never selected.
    replace_table(con,'fit_rows',f"SELECT {','.join(NUMERIC)} FROM supervised_raw WHERE split='train'")
    medians=records(con,'SELECT '+','.join(f'median({x}) {x}' for x in NUMERIC)+' FROM fit_rows')[0]
    active_numeric=[x for x in NUMERIC if medians[x] is not None]
    all_missing=[x for x in NUMERIC if medians[x] is None]
    expressions=[]
    for name in active_numeric:
        med=float(medians[name]);expr=f'coalesce({name}::DOUBLE,{med!r})'
        expressions += [f'avg({expr}) {name}_mean',f'stddev_pop({expr}) {name}_std']
    moments=records(con,'SELECT '+','.join(expressions)+' FROM fit_rows')[0]
    params={}
    for name in active_numeric:
        standard= float(moments[name+'_std'])
        params[name]={'median':float(medians[name]),'mean':float(moments[name+'_mean']),
            'scale':standard if math.isfinite(standard) and standard>1e-12 else 1.0,
            'constant_after_imputation':standard<=1e-12}
    preprocessing={'fit_scope':'Pool B train snapshots with valid final label; no q/y values used',
        'fit_rows':con.execute('SELECT count(*) FROM fit_rows').fetchone()[0],
        'numeric':params,'all_missing_train_excluded':all_missing,
        'categorical':'Sorted vocabularies fit on labeled train pairs; unseen/missing -> 0; gender source codes retained with -1 for invalid/missing.',
        'geometry':'Use scaled numeric allowlist only; identity/category codes are excluded.',
        'add_missing_indicators':False}
    dump_json(out/'preprocessing.json',preprocessing)
    for phase in PHASES:
        expressions=[f'((coalesce({x}::DOUBLE,{p["median"]!r})-{p["mean"]!r})/{p["scale"]!r}) {x}' for x,p in params.items()]
        view(con,'phase_supervised',out/'supervised_raw'/f'{phase}.parquet')
        write_parquet(con,out/'model_ready'/f'{phase}.parquet',f'''SELECT {','.join(META)},
            {','.join(expressions+CATEGORICAL)},q,y,target_alpha,target_formula_version FROM phase_supervised''')
    feature_config={'version':VERSION,'grain':['user_id','course_id','phase'],'metadata_columns':META,
        'raw_numeric_features':NUMERIC,'numeric_features':active_numeric,'categorical_features':CATEGORICAL,
        'model_feature_columns':active_numeric+CATEGORICAL,'distance_feature_columns':active_numeric,
        'target_columns':['q','y'],'excluded_target_metadata':['target_alpha','target_formula_version'],
        'weekly_sequence_fields':['week_index']+SEQ_FIELDS,'feature_groups':GROUPS,
        'all_missing_train_excluded':all_missing,
        'categorical_usage':'Treat codes as categories, not ordered numeric geometry; use a categorical-aware learner or train-only one-hot encoding.'}
    dump_json(out/'feature_columns.json',feature_config)
    print('Validating saved datasets',flush=True)
    summary,checks=validate(con,out,times,cohort,feature_config)
    checks['fixtures']=fixtures()
    assert all(v for k,v in checks.items() if k!='fixtures'),checks
    summary['validation']=checks
    summary['elapsed_seconds']=round(time.monotonic()-started,3)
    summary['version']=VERSION;summary['numeric_features']=len(active_numeric);summary['categorical_features']=len(CATEGORICAL)
    manifest={'created_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'version':VERSION,
        'engine_source_sha256':source_hash,'feature_input_signature':signature,'feature_lineage':feature_lineage,
        'quality_root':str(quality),'raw_root':str(raw),'model_root':str(model),'time_root':str(times),
        'label_root':str(labels),'label_manifest_sha256':sha256(labels/'run_manifest.json'),
        'target_alpha':0.1,'sampling':'none','duckdb_version':duckdb.__version__,
        'feature_contract':'User-approved raw-log baseline; full original TEMPO-DQ feature list not supplied.',
        'encoding':'Train-only vocabularies for learner and sorted teacher/school/field signatures; metadata stores original course signatures.',
        'sequence':'Sparse weeks since enrollment, sorted by week index; empty sequence retained. Segment amounts assigned to starting week.',
        'time_policy':'Inclusive enrollment to per-user phase cutoff; partial video segments clipped to cutoff using playback speed.',
        'problem_limitation':'user-problem aggregate becomes available at submit_time; per-attempt history cannot be reconstructed.',
        'static_assumption':'Catalog resources, teacher/school associations and user profile snapshot treated as static; validity histories unavailable.',
        'split_limitation':'Existing course split is provisional and contains temporal overlap; see notebook 06 report.',
        'no_feature_label_values':'No final indices, q/y, label status or missingness masks in model feature allowlists.',
        'per_file_checks':per_file_checks}
    dump_json(out/'summary.json',summary);dump_json(out/'validation.json',checks);dump_json(out/'run_manifest.json',manifest)
    write_report(out,summary,feature_config,preprocessing)
    write_dictionary(out)
    dump_json(out/'_SUCCESS.json',{'manifest_sha256':sha256(out/'run_manifest.json'),
        'feature_input_signature':signature,'target_alpha':0.1})
    con.close();print('PHASE_FEATURES_COMPLETE',json.dumps(summary,ensure_ascii=False,default=str),flush=True)
    return summary


def validate(con,out,times,cohort,config):
    view(con,'all_x',out/'features/*/*.parquet');view(con,'all_audit',out/'audit/*/*.parquet')
    view(con,'all_sequences',out/'sequences/*/*.parquet');view(con,'ready',out/'model_ready/*.parquet')
    view(con,'supervised_seq',out/'supervised_sequences/*.parquet')
    n_pairs=con.execute('SELECT count(*) FROM targets').fetchone()[0]
    summary={'all_time_eligible_pairs':n_pairs,'labeled_user_course_pairs':cohort['n'],
        'phase_feature_rows':records(con,'SELECT phase,count(*) n_rows,count(*) FILTER(WHERE n_events=0) n_no_observed_activity FROM all_x GROUP BY 1 ORDER BY 1'),
        'supervised_phase_rows':records(con,'SELECT phase,split,count(*) n_rows,count(*) FILTER(WHERE n_events=0) n_no_observed_activity FROM supervised_raw GROUP BY 1,2 ORDER BY 1,2'),
        'supervised_class_distribution':records(con,'SELECT phase,split,y,count(*) n_rows FROM supervised_raw GROUP BY 1,2,3 ORDER BY 1,2,3'),
        'sequence_rows':records(con,'SELECT phase,count(*) n_rows FROM all_sequences GROUP BY 1 ORDER BY 1'),
        'supervised_sequence_rows':records(con,'SELECT phase,count(*) n_rows,count(*) FILTER(WHERE len(behavior_sequence)=0) n_empty_sequences FROM supervised_seq GROUP BY 1 ORDER BY 1'),
        'raw_numeric_nulls':records(con,'SELECT '+','.join(f'count(*) FILTER(WHERE {x} IS NULL) {x}' for x in NUMERIC)+' FROM supervised_raw')[0]}
    tests={
        'all_time_eligible_rows_retained_at_each_phase':all(x['n_rows']==n_pairs for x in summary['phase_feature_rows']) and len(summary['phase_feature_rows'])==4,
        'four_supervised_snapshots_per_labeled_pair':con.execute('SELECT count(*) FROM supervised_raw').fetchone()[0]==4*cohort['n'],
        'supervised_keys_unique':con.execute('SELECT count(*)=count(DISTINCT(user_id,course_id,phase)) FROM supervised_raw').fetchone()[0],
        'fixed_final_label_across_phases':con.execute('SELECT count(*)=0 FROM (SELECT user_id,course_id FROM supervised_raw GROUP BY 1,2 HAVING count(*)<>4 OR count(DISTINCT y)<>1 OR count(DISTINCT q)<>1)').fetchone()[0],
        'features_and_targets_share_split':con.execute('SELECT count(*)=0 FROM all_x f JOIN targets t USING(user_id,course_id) WHERE f.split IS DISTINCT FROM t.split').fetchone()[0],
        'all_observed_events_cut_at_phase':con.execute("SELECT count(*)=0 FROM all_audit WHERE n_out_of_window_events<>0 OR min_feature_event_time<enroll_time OR max_feature_event_time>cutoff_time+INTERVAL '1 microsecond'").fetchone()[0],
        'all_features_have_valid_metadata':con.execute('SELECT count(*)=0 FROM all_x WHERE user_id IS NULL OR course_id IS NULL OR phase IS NULL OR split IS NULL OR cutoff_time<enroll_time').fetchone()[0],
        'no_target_or_missingness_columns_in_model_allowlist':all(not any(token in x.lower() for token in ['coelo','afelo','acelo','loqv','missing','mask','label_status']) and x not in ['q','y','course_id','cutoff_time','enroll_time','split'] for x in config['model_feature_columns']),
        'identity_codes_excluded_from_geometry':not(set(CATEGORICAL)&set(config['distance_feature_columns'])),
        'no_missing_or_infinite_ready_numeric':con.execute('SELECT count(*)=0 FROM ready WHERE '+ ' OR '.join(f'{x} IS NULL OR NOT isfinite({x})' for x in config['numeric_features'])).fetchone()[0],
        'raw_feature_values_finite_when_observed':con.execute('SELECT count(*)=0 FROM all_x WHERE '+ ' OR '.join(f'({x} IS NOT NULL AND NOT isfinite({x}))' for x in NUMERIC)).fetchone()[0],
        'counts_and_coverages_nonnegative':con.execute('SELECT count(*)=0 FROM all_x WHERE '+ ' OR '.join(f'{x}< -1e-9' for x in ZERO_FEATURES|{'video_coverage_ratio','exercise_coverage_ratio','problem_coverage_ratio','chapter_coverage_ratio'})).fetchone()[0],
        'coverage_ratios_not_above_one':con.execute('SELECT count(*)=0 FROM all_x WHERE '+ ' OR '.join(f'{x}>1+1e-9' for x in ['video_coverage_ratio','exercise_coverage_ratio','problem_coverage_ratio','chapter_coverage_ratio'])).fetchone()[0],
        'sentiment_probability_means_sum_to_one':con.execute('SELECT count(*)=0 FROM all_x WHERE n_comments>0 AND (sentiment_p_positive_mean IS NULL OR abs(sentiment_p_positive_mean+sentiment_p_neutral_mean+sentiment_p_negative_mean-1)>1e-6)').fetchone()[0],
        'reply_rate_bounded':con.execute('SELECT count(*)=0 FROM all_x WHERE n_comments_with_reply>n_comments OR comment_reply_rate NOT BETWEEN 0 AND 1').fetchone()[0],
        'no_comment_has_no_imputed_raw_sentiment':con.execute('SELECT count(*)=0 FROM all_x WHERE n_comments=0 AND (sentiment_p_positive_mean IS NOT NULL OR sentiment_p_neutral_mean IS NOT NULL OR sentiment_p_negative_mean IS NOT NULL)').fetchone()[0],
        'supervised_sequences_preserve_all_rows':con.execute('SELECT count(*)=0 FROM supervised_raw ANTI JOIN supervised_seq USING(user_id,course_id,phase)').fetchone()[0] and con.execute('SELECT count(*) FROM supervised_seq').fetchone()[0]==4*cohort['n'],
        'ready_rows_preserve_supervised_rows':con.execute('SELECT count(*)=0 FROM supervised_raw ANTI JOIN ready USING(user_id,course_id,phase)').fetchone()[0] and con.execute('SELECT count(*) FROM ready').fetchone()[0]==4*cohort['n'],
    }
    for phase in PHASES:
        cut=cutoff_sql('enroll_time','course_end',phase)
        expected_path=times/f'phase_cutoffs/{phase}.parquet'
        view(con,'expected_phase',expected_path)
        tests[f'{phase}_cutoffs_match_notebook05']=con.execute(f'''SELECT count(*)=0 FROM
            read_parquet({qs(out/'features'/phase/'*.parquet')}) f JOIN expected_phase e USING(user_id,course_id)
            WHERE f.cutoff_time IS DISTINCT FROM e.cutoff_time''').fetchone()[0]
        tests[f'{phase}_feature_keys_preserve_all_eligible_pairs']=con.execute(f'''SELECT count(*)=0 FROM expected_phase
            ANTI JOIN read_parquet({qs(out/'features'/phase/'*.parquet')}) f USING(user_id,course_id)''').fetchone()[0]
    # Canonical vocabularies contain only tokens observed in labeled train pairs.
    tests['learner_vocabulary_fit_only_on_train']=con.execute("SELECT count(*)=0 FROM enc_learner e ANTI JOIN (SELECT DISTINCT user_id token FROM targets WHERE split='train' AND y IS NOT NULL) t USING(token)").fetchone()[0]
    tests['preprocessing_fit_row_count_matches_train_pool']=con.execute('SELECT count(*) FROM fit_rows').fetchone()[0]==con.execute("SELECT count(*)*4 FROM targets WHERE split='train' AND y IS NOT NULL").fetchone()[0]
    return summary,tests


def write_dictionary(out):
    definitions={
      'n_video_segments':'Số segment video hợp lệ có thời lượng quan sát >0 trước cutoff.',
      'n_videos_watched':'Số video khác nhau có segment quan sát được.',
      'video_media_seconds':'Tổng giây nội dung video đã quan sát; clip đoạn cuối theo cutoff và playback speed.',
      'video_wall_seconds':'Tổng observed_media_seconds/speed; proxy thời gian xem, chưa khử chồng đồng thời.',
      'video_unique_seconds':'Tổng độ dài hợp các khoảng nội dung đã xem theo từng video, chỉ dùng segment trước cutoff.',
      'video_rewatch_seconds':'video_media_seconds - video_unique_seconds.',
      'video_sessions_30m':'Số phiên video trong user-course; phiên mới nếu khoảng cách giữa hai thời điểm bắt đầu >30 phút.',
      'n_revisited_videos':'Số video có hơn một phiên; phiên trong từng video tách bằng gap bắt đầu >30 phút.',
      'video_mean_speed':'Trung bình playback speed của segment có thời lượng quan sát >0.',
      'video_mean_segment_seconds':'Trung bình giây nội dung của các segment sau clip cutoff.',
      'video_active_days':'Số ngày UTC khác nhau có segment bắt đầu được quan sát.',
      'video_recency_days':'Số ngày từ kết thúc segment quan sát cuối đến cutoff.',
      'n_problem_submissions':'Số dòng user-problem có submit_time trong cửa sổ; không phải chuỗi từng attempt.',
      'n_exercises_attempted':'Số exercise_id khác nhau được nối từ problem đã xuất hiện trước cutoff.',
      'problem_attempts':'Tổng attempts dương của các dòng trạng thái user-problem có sẵn trước cutoff.',
      'n_correct_problems':'Số dòng problem có is_correct=1 trước cutoff.',
      'n_first_try_correct':'Số dòng is_correct=1 và attempts=1 trước cutoff.',
      'problem_correct_rate':'Trung bình is_correct trên các problem quan sát được.',
      'problem_first_try_correct_rate':'n_first_try_correct/n_problem_submissions.',
      'problem_mean_attempts':'Trung bình attempts dương trên các problem quan sát được.',
      'problem_attempted_score_ratio':'Tổng điểm thu được/tổng max score của problem đã làm; giữ điểm từng phần, fallback binary như notebook 06; null nếu có điểm không giải được.',
      'problem_efficiency_mean':'Trung bình is_correct/attempts trên các dòng attempts dương hợp lệ đã quan sát.',
      'problem_active_days':'Số ngày UTC có submit_time trước cutoff.',
      'problem_recency_days':'Số ngày từ submit_time cuối quan sát đến cutoff.',
      'n_comments':'Số comment không rỗng, loại trùng user/resource/time/normalized-text theo quy ước notebook 06.',
      'comment_mean_characters':'Trung bình số ký tự text của comment hợp lệ trước cutoff.',
      'n_replies_authored':'Số reply do người học viết, nối được về resource/course và thời gian hợp lệ.',
      'n_replies_received':'Số reply tới comment hợp lệ của người học, reply_time không sau cutoff của người nhận.',
      'n_comments_with_reply':'Số comment khác nhau đã nhận ít nhất một reply trước cutoff.',
      'comment_reply_rate':'n_comments_with_reply/n_comments; có comment chưa nhận reply ->0; không có comment ->null.',
      'sentiment_p_positive_mean':'Trung bình p_pos đã suy luận ở notebook 07, chỉ trên comment trước cutoff; không có comment ->null.',
      'sentiment_p_neutral_mean':'Trung bình p_neu của comment trước cutoff; không có comment ->null.',
      'sentiment_p_negative_mean':'Trung bình p_neg của comment trước cutoff; không có comment ->null.',
      'social_active_days':'Số ngày UTC có comment, reply viết hoặc reply nhận trong cửa sổ.',
      'social_recency_days':'Số ngày từ tương tác xã hội cuối quan sát đến cutoff.',
      'n_events':'Số video segment + problem row + comment + reply do người học viết; không đếm reply nhận thêm lần nữa.',
      'n_active_days':'Số ngày UTC có ít nhất một sự kiện của người học; không tính riêng reply nhận.',
      'n_chapters_touched':'Số lo_key khác nhau của các sự kiện của người học trước cutoff.',
      'activity_span_days':'Khoảng ngày giữa sự kiện đầu tiên và kết thúc quan sát cuối, không gồm riêng reply nhận.',
      'activity_recency_days':'Số ngày từ kết thúc quan sát cuối của sự kiện người học đến cutoff.',
      'course_n_videos':'Số video_id khác nhau trong catalog course; giả định catalog tĩnh.',
      'course_n_exercises':'Số exercise_id khác nhau trong catalog course.',
      'course_n_problems':'Số problem_id khác nhau trong bridge_lo_problem của course.',
      'course_n_chapters':'Số big chapter trong dim_lo_catalog của course.',
      'learner_age_at_enrollment':'Năm ghi danh trừ year_of_birth; chỉ giữ birth>=1900 và tuổi trong [10,100]; ngoài miền ->null.',
      'phase_fraction':'Tỷ lệ phase theo giao thức: 0,25 / 0,50 / 0,75 / 0,90; không phải tỷ lệ hoàn thành suy ra từ log tương lai.',
      'video_coverage_ratio':'n_videos_watched/course_n_videos; mẫu số 0 ->null.',
      'exercise_coverage_ratio':'n_exercises_attempted/course_n_exercises; mẫu số 0 ->null.',
      'problem_coverage_ratio':'n_problem_submissions/course_n_problems; mẫu số 0 ->null.',
      'chapter_coverage_ratio':'n_chapters_touched/course_n_chapters; mẫu số 0 ->null.',
      'learner_code':'Mã danh mục user_id; vocabulary fit trên user của các cặp train có nhãn; chưa thấy ->0. Không dùng làm tọa độ khoảng cách.',
      'teacher_set_code':'Mã danh mục cho tập teacher_id đã sort và ghép của course; vocabulary chỉ fit train. Giữ chữ ký gốc ở course_static_metadata.',
      'school_set_code':'Mã danh mục cho tập school_id của đơn vị cung cấp course; vocabulary chỉ fit train.',
      'field_set_code':'Mã danh mục cho tập lĩnh vực field của course; vocabulary chỉ fit train.',
      'gender_code':'Mã nguồn gender 0/1/2; giá trị khác hoặc thiếu ->-1; xử lý như danh mục, không suy diễn ý nghĩa chưa được tài liệu xác nhận.',
    }
    rows=[]
    for group,fields in GROUPS.items():
        for name in fields:
            rows.append({'name':name,'group':group,'kind':'numeric','definition':definitions[name],
                'no_event_rule':'zero' if name in ZERO_FEATURES else 'null for undefined ratios/means/recency; static fields follow catalog',
                'time_scope':'static snapshot/enrollment' if group=='context' else 'inclusive enrollment through per-pair phase cutoff'})
    rows += [{'name':name,'group':'static_identity','kind':'categorical','definition':definitions[name],
        'distance_feature':False} for name in CATEGORICAL]
    dump_json(Path(out)/'feature_dictionary.json',rows)
    lines=['# Từ điển feature theo phase','',
        'Đơn vị bảng: user_id × course_id × phase. Đây là baseline từ log hiện có theo lựa chọn người dùng; '
        'không tuyên bố khớp toàn bộ danh sách gốc TEMPO-DQ.', '',
        '| Feature | Nhóm | Định nghĩa |','|---|---|---|']
    lines += [f'| `{r["name"]}` | {r["group"]} | {r["definition"]} |' for r in rows]
    lines += ['', 'Số đếm/thời lượng vắng log là 0 trong phạm vi nguồn quan sát; không chứng minh người học thực tế không hoạt động. '
        'Mean/rate/recency không xác định giữ null ở bảng thô. Chỉ bản model_ready dùng median/scaler fit trên train, không thêm missing indicators.', '',
        'Chuỗi tuần: week_index=floor((event_time-enroll_time)/7 ngày), chỉ giữ tuần có sự kiện. '
        'Mỗi phần tử gồm week_index và '+', '.join(SEQ_FIELDS)+'. '
        'Các lượng được gán theo tuần bắt đầu sự kiện; segment video vẫn clip phần sau cutoff. '
        'Tuần trống không tạo phần tử; dùng week_index để giữ khoảng cách thời gian. Cặp chưa có log dùng chuỗi rỗng.', '']
    (Path(out)/'tu_dien_features.md').write_text('\n'.join(lines),encoding='utf-8')


def write_report(out,s,config,preprocessing):
    fmt=lambda n:f'{int(n):,}'.replace(',','.')
    lines=['# Feature user course theo bốn phase','',
        'Đã tạo feature từ log trước cutoff theo Mục 2 của Review, theo lựa chọn xây dựng baseline từ dữ liệu hiện có. '
        'Danh sách gốc TEMPO-DQ không được cung cấp đầy đủ nên không tuyên bố tái lập đúng tập 65 feature của paper.', '',
        f'- Giữ {fmt(s["all_time_eligible_pairs"])} user–course × 4 phase ở kho feature đầy đủ.',
        f'- Tập supervised có {fmt(s["labeled_user_course_pairs"])} user–course × 4 phase, nhãn alpha=0,1 cố định cuối kỳ.',
        f'- X gồm {s["numeric_features"]} feature số và {s["categorical_features"]} mã danh mục; thêm khối chuỗi tuần riêng.',
        f'- Imputer/scaler fit trên {fmt(preprocessing["fit_rows"])} snapshot train (Pool B).', '',
        '| Phase | Tổng snapshot | Chưa có sự kiện quan sát |','|---|---:|---:|']
    lines += [f'| {r["phase"]} | {fmt(r["n_rows"])} | {fmt(r["n_no_observed_activity"])} |' for r in s['phase_feature_rows']]
    lines += ['', '## Tập supervised', '', '| Phase | Split | Snapshot | Chưa có sự kiện |','|---|---|---:|---:|']
    lines += [f'| {r["phase"]} | {r["split"]} | {fmt(r["n_rows"])} | {fmt(r["n_no_observed_activity"])} |' for r in s['supervised_phase_rows']]
    lines += ['', '## Các bảng và cách dùng', '',
        '- `features/P1/*.parquet` đến P4: toàn bộ cặp đủ điều kiện thời gian; 32 file mỗi phase, một bảng logic. '
        'Chứa metadata khóa/thời gian và feature thô; không có q/y hoặc chỉ số cuối kỳ.',
        '- `tables/user_course_targets.parquet`: master nhãn riêng, đủ mọi cặp, giữ q/y null khi chưa gán được nhãn.',
        '- `supervised_raw/P1.parquet` đến P4: feature thô ghép nhãn cho các cặp y hợp lệ; không loại dòng vì X thiếu hay phase chưa hoạt động.',
        '- `model_ready/P1.parquet` đến P4: số đã median-impute và standard-scale bằng thống kê train; mã danh mục giữ dạng mã. '
        'Learner cần hỗ trợ categorical hoặc thêm one-hot encoder fit train. Dùng allowlist trong feature_columns.json; '
        'không lấy mọi cột số tự động làm X.',
        '- `sequences/P1/*.parquet` đến P4: chuỗi tuần thưa của các cặp đã có sự kiện; '
        '`supervised_sequences/P1.parquet` đến P4 giữ đủ cohort huấn luyện, chuỗi rỗng nếu chưa quan sát.',
        '- `audit/P1/*.parquet` đến P4: thời điểm đầu/cuối của sự kiện sử dụng và kiểm tra cửa sổ. '
        'has_observed_activity chỉ dành kiểm toán, không nằm trong X.',
        '- `encoders/`: vocabulary chỉ từ cặp train có nhãn. Đối tượng chưa thấy hoặc thiếu ->0. '
        'Giảng viên/trường/lĩnh vực được mã hóa theo tập ID đã sort; đây là lựa chọn baseline được ghi rõ, chưa đối chiếu encoding gốc TEMPO-DQ.',
        '- `feature_columns.json`, `feature_dictionary.json`, `tu_dien_features.md`, `preprocessing.json`: hợp đồng cột, công thức và tham số train.', '',
        '## Kiểm soát thời gian và nhãn', '',
        'Feature tích lũy trong [enroll_time,cutoff]. Video được clip tại cutoff bằng playback speed; '
        'unique watch time phải hợp lại các khoảng nội dung từ segment đã clip, không lấy tỷ lệ của tổng cuối kỳ. '
        'Comment/reply và phản hồi nhận được đều lọc timestamp trước khi aggregate. '
        'Sentiment dùng xác suất frozen model notebook 07 của comment trước cutoff; không chạy lại model.', '',
        'Nhãn alpha=0,1 chỉ được ghép sau bước feature. q/y, COELO/AFELO/ACELO, trạng thái nhãn, '
        'cờ thiếu, course_end và course_id không thuộc allowlist X. phase_fraction là feature giao thức cho Pool B. '
        'Các mã danh mục định danh không nằm trong distance_feature_columns. '
        'Thay đổi giá trị nhãn mà giữ cùng cohort không yêu cầu tính lại log feature; vẫn cần ghép lại y và các bước học dùng y.', '',
        '## Giới hạn', '',
        '- user_problem là trạng thái tổng hợp tại submit_time. Không thể biết attempts/is_correct/score tại các lần nộp trước đó; '
        'chỉ đưa toàn dòng vào khi submit_time không sau cutoff.',
        '- Catalog, liên kết giảng viên/trường và profile người học được coi là tĩnh vì không có lịch sử hiệu lực. '
        'Chưa xác nhận mọi thuộc tính snapshot tồn tại ở thời điểm vận hành thực tế.',
        '- course_end lấy hồi cứu từ hoạt động cuối; đây là giao thức phase hồi cứu đã chọn, không phải lịch đóng khóa được biết trước.',
        '- Nguồn video chỉ trong khoảng thu thập công bố; nguồn xã hội có độ phủ thấp. Vắng log không chứng minh không hoạt động.',
        '- Split course kế thừa 06/07 và còn chồng thời gian. Cần phương án NOLFO khi đánh giá vận hành.',
        '- Nhãn hiện có chỉ A/G, không có E; tạo feature không làm thay đổi hạn chế phân bố nhãn.', '',
        '## Kiểm tra', '',
        f'Đạt {sum(bool(v) for k,v in s["validation"].items() if k!="fixtures")}/'
        f'{len(s["validation"])-1} kiểm tra dữ liệu và {len(s["validation"]["fixtures"])} fixture thời gian. '
        'Bao gồm đủ cặp/snapshot, cutoff khớp notebook 05, không có event tương lai, nhãn cố định, '
        'raw sentiment null đúng khi không có comment, giữ snapshot trống và fit tiền xử lý chỉ trên train.', '']
    (Path(out)/'bao_cao_phase_features.md').write_text('\n'.join(lines),encoding='utf-8')


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['quality-root','label-root','output-root','raw-root','model-root','time-root']:
        parser.add_argument('--'+name)
    args=parser.parse_args()
    run_build(args.quality_root,args.label_root,args.output_root,args.raw_root,args.model_root,args.time_root)
