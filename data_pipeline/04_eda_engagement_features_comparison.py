"""Compare engagement component inputs in all data vs 2020-05-01..2020-12-14.

Counts/quantiles use all supplied rows. No sentiment model, normalization fit,
composite quality label, or train/test split is created by this EDA.
"""
from __future__ import annotations

import datetime as dt
import glob as glob_module
import hashlib
import json
import shutil
import time
from pathlib import Path

import duckdb

WINDOW_START = '2020-05-01 00:00:00'
WINDOW_END_EXCLUSIVE = '2020-12-15 00:00:00'


def qs(value):
    return "'" + str(value).replace("'", "''") + "'"


def input_fingerprint(raw, model):
    values=[]
    for kind,root in [('raw',Path(raw)),('model',Path(model))]:
        for p in sorted(root.rglob('*.parquet')):
            s=p.stat();values.append([kind,str(p.relative_to(root)),s.st_size,s.st_mtime_ns])
    return hashlib.sha256(json.dumps(values,sort_keys=True).encode()).hexdigest()


def discover():
    for parent in [Path.cwd(),Path.cwd()/'working_13-10']:
        root=parent/'input'
        if (root/'moocubex_parquet/catalog.json').exists() and (root/'moocubex_model/catalog.json').exists():
            return root/'moocubex_parquet',root/'moocubex_model'
    raw=[];model=[]
    for parent in [Path('/kaggle/working'),Path('/kaggle/input')]:
        if parent.is_dir():
            for p in parent.rglob('catalog.json'):
                if (p.parent/'tables/relations/user_video').is_dir():raw.append(p.parent)
                if (p.parent/'processed/dim_lo_catalog').is_dir():model.append(p.parent)
    if len(raw)==len(model)==1:return raw[0],model[0]
    raise RuntimeError('Set raw_root and model_root explicitly; unique inputs not found.')


def run_eda(raw_root=None,model_root=None,output_root=None,memory_limit='2GB',threads=1):
    if raw_root is None or model_root is None:
        raw_default,model_default=discover()
        raw_root=raw_root or raw_default;model_root=model_root or model_default
    raw,model=Path(raw_root).resolve(),Path(model_root).resolve()
    out=Path(output_root or ('/kaggle/working/eda_engagement_features_comparison' if Path('/kaggle').is_dir()
              else raw.parent.parent/'eda_engagement_features_comparison')).resolve()
    tables=out/'tables';tables.mkdir(parents=True,exist_ok=True)
    scratch=out/'_scratch';scratch.mkdir(exist_ok=True)
    con=duckdb.connect();con.execute(f'SET threads={int(threads)}')
    con.execute(f'SET memory_limit={qs(memory_limit)}');con.execute(f'SET temp_directory={qs(scratch)}')
    con.execute("SET TimeZone='UTC'");con.execute('SET preserve_insertion_order=false')
    con.execute("SET partitioned_write_max_open_files=8")
    con.execute("SET max_temp_directory_size='4GB'")
    log=[];summary={};tick_all=time.monotonic()
    base_fingerprint=input_fingerprint(raw,model)
    cache_path=out/'step_checkpoints.json'
    cache=json.loads(cache_path.read_text()) if cache_path.exists() else {}
    if cache.get('input_fingerprint')!=base_fingerprint:
        cache={'input_fingerprint':base_fingerprint,'policy_version':'feature-input-eda-v1','steps':{}}
    lineage=[]
    def cache_write():
        temporary=cache_path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(cache,ensure_ascii=False,indent=2));temporary.replace(cache_path)
    def signature(query):
        return hashlib.sha256(json.dumps([base_fingerprint,'feature-input-eda-v1',lineage,query],sort_keys=True).encode()).hexdigest()

    def view(name,path):
        if not glob_module.glob(str(path)):raise FileNotFoundError(path)
        con.execute(f'CREATE OR REPLACE VIEW {name} AS SELECT * FROM read_parquet({qs(path)},union_by_name=true)')

    def save(name,query,preview=5):
        print('\n'+name,flush=True);tick=time.monotonic()
        p=tables/(name+'.parquet')
        digest=signature(query);previous=cache['steps'].get(name,{})
        reused=p.exists() and previous.get('signature')==digest and previous.get('bytes')==p.stat().st_size
        if not reused:
            con.execute(f'COPY ({query}) TO {qs(p)} (FORMAT PARQUET,COMPRESSION ZSTD)')
        view(name,p)
        n=con.execute(f'SELECT count(*) FROM {name}').fetchone()[0]
        cur=con.execute(f'SELECT * FROM {name} LIMIT {int(preview)}')
        cols=[x[0] for x in cur.description]
        summary[name]=[dict(zip(cols,r)) for r in cur.fetchall()]
        elapsed=round(time.monotonic()-tick,3)
        log.append({'name':name,'rows':n,'seconds':elapsed,'sql':query,'checkpoint_reused':reused})
        cache['steps'][name]={'signature':digest,'bytes':p.stat().st_size,'rows':n};lineage.append(digest);cache_write()
        print('rows=',n,'seconds=',elapsed,'checkpoint=',reused,flush=True)
        if preview and n<=10:
            for r in summary[name]:print(json.dumps(r,default=str,ensure_ascii=False),flush=True)
        return n

    for name,rel in {'uv':'relations/user_video','up':'relations/user_problem',
                     'comments':'entities/translated_comment','replies':'entities/reply',
                     'comment_reply':'relations/comment_reply'}.items():
        view(name,raw/'tables'/rel/'*.parquet')
    for name in ['dim_lo_resource','dim_lo_catalog','bridge_lo_problem','bridge_video_ccid',
                 'dim_video','bridge_course_comment','dim_problem']:
        view(name,model/'processed'/name/'*.parquet')
    con.execute(f"CREATE TEMP TABLE scopes AS SELECT * FROM (VALUES ('ALL_AVAILABLE',NULL::TIMESTAMP,NULL::TIMESTAMP),"
                f"('WINDOW_20200501_20201214',TIMESTAMP {qs(WINDOW_START)},TIMESTAMP {qs(WINDOW_END_EXCLUSIVE)})) s(scope,t0,t1)")
    inside=lambda t:f"({t}>=TIMESTAMP {qs(WINDOW_START)} AND {t}<TIMESTAMP {qs(WINDOW_END_EXCLUSIVE)})"

    # Exactly one assignment to a chapter is required for indicator components.
    con.execute('''CREATE TEMP TABLE resource_map AS SELECT resource_id,min(course_id) course_id,
        min(lo_key) lo_key,min(resource_type) resource_type,count(DISTINCT lo_key) n_los
        FROM dim_lo_resource GROUP BY 1''')
    con.execute('''CREATE TEMP TABLE problem_map AS SELECT problem_id,min(course_id) course_id,min(lo_key) lo_key,
        min(exercise_id) exercise_id,max(problem_max_score) problem_max_score,
        min(score_catalog_status) score_catalog_status,count(DISTINCT lo_key) n_los,
        count(DISTINCT exercise_id) n_exercises FROM bridge_lo_problem GROUP BY 1''')
    con.execute('''CREATE TEMP TABLE comment_course AS SELECT comment_id,min(course_id) course_id,
        count(DISTINCT course_id) n_courses FROM bridge_course_comment GROUP BY 1''')
    save('catalog_quality','''SELECT count(*) n_los,count(DISTINCT course_id) n_courses,
        count(*) FILTER(WHERE n_videos_available>0) los_with_video,
        count(*) FILTER(WHERE n_exercises_available>0) los_with_exercise,
        count(*) FILTER(WHERE n_problems_available>0) los_with_problem,
        count(*) FILTER(WHERE n_videos_available>0 AND n_videos_missing_duration=0) los_complete_duration_proxy,
        count(*) FILTER(WHERE n_problems_available>0 AND n_problems_missing_score=0) los_complete_problem_max_score,
        count(*) FILTER(WHERE n_videos_missing_duration>0) los_missing_some_duration,
        count(*) FILTER(WHERE n_problems_missing_score>0) los_missing_some_problem_score
        FROM dim_lo_catalog''')
    save('catalog_distributions','''SELECT * FROM (
        SELECT 'n_videos_available' feature,count(*) n, min(n_videos_available)::DOUBLE min_value,
          median(n_videos_available)::DOUBLE median_value,quantile_cont(n_videos_available,0.95)::DOUBLE p95,
          max(n_videos_available)::DOUBLE max_value FROM dim_lo_catalog
        UNION ALL SELECT 'n_exercises_available',count(*),min(n_exercises_available),median(n_exercises_available),quantile_cont(n_exercises_available,0.95),max(n_exercises_available) FROM dim_lo_catalog
        UNION ALL SELECT 'n_problems_available',count(*),min(n_problems_available),median(n_problems_available),quantile_cont(n_problems_available,0.95),max(n_problems_available) FROM dim_lo_catalog)''')

    # Source-level field audit. All data means no date filter; anomalies stay visible.
    video_parts=[]
    for scope in ['ALL_AVAILABLE','WINDOW_20200501_20201214']:
        condition='TRUE' if scope=='ALL_AVAILABLE' else f's.local_start_time>=epoch(TIMESTAMP {qs(WINDOW_START)}) AND s.local_start_time<epoch(TIMESTAMP {qs(WINDOW_END_EXCLUSIVE)})'
        # Reduce nested lists within each user row; a global UNNEST carries very
        # large parent lists and caused unnecessary spill on the raw video table.
        predicates={
            'n_segments':'TRUE',
            'invalid_intervals':'s.start_point IS NULL OR s.end_point IS NULL OR s.start_point<0 OR s.end_point<=s.start_point OR NOT isfinite(s.start_point) OR NOT isfinite(s.end_point)',
            'invalid_speeds':'s.speed IS NULL OR s.speed<=0 OR NOT isfinite(s.speed)',
            'invalid_epoch_range':'s.local_start_time IS NULL OR s.local_start_time<0 OR s.local_start_time>4102444800',
            'outside_paper_video_window':"s.local_start_time<epoch(TIMESTAMP '2020-01-19') OR s.local_start_time>=epoch(TIMESTAMP '2020-11-04')",
        }
        fields=[]
        for metric,predicate in predicates.items():
            nested=f'list_sum(list_transform(u.seq,v->COALESCE(array_length(list_filter(v.segment,s->({condition}) AND ({predicate}))),0)))'
            fields.append(f'sum(COALESCE({nested},0))::BIGINT {metric}')
        video_parts.append(f'SELECT {qs(scope)} AS scope,'+','.join(fields)+' FROM uv u')
    save('video_source_quality',' UNION ALL '.join(video_parts))
    problem_parts=[]
    for scope in ['ALL_AVAILABLE','WINDOW_20200501_20201214']:
        where='TRUE' if scope=='ALL_AVAILABLE' else inside('try_cast(submit_time AS TIMESTAMP)')
        problem_parts.append(f'''SELECT {qs(scope)} AS scope,count(*) n_rows,
          count(*) FILTER(WHERE try_cast(submit_time AS TIMESTAMP) IS NULL) invalid_time,
          count(*) FILTER(WHERE attempts IS NULL OR attempts<=0) invalid_attempts,
          count(*) FILTER(WHERE is_correct IS NULL OR is_correct NOT IN (0,1)) invalid_correctness,
          count(*) FILTER(WHERE score IS NULL) score_null,
          count(*) FILTER(WHERE score<0) score_negative,
          min(attempts) attempts_min,max(attempts) attempts_max,avg(attempts) attempts_mean,
          min(score) score_min,max(score) score_max,
          min(try_cast(submit_time AS TIMESTAMP)) first_time,max(try_cast(submit_time AS TIMESTAMP)) last_time
          FROM up WHERE {where}''')
    save('problem_source_quality',' UNION ALL '.join(problem_parts))
    # Partition the filtered problem source once. High-cardinality distinct/group
    # operations then read small user shards instead of retaining 16M keys at once.
    problem_stage=out/'_stage'/'problem_rows'
    problem_stage.parent.mkdir(exist_ok=True)
    print('\nPartition mapped problem rows into 32 user shards',flush=True)
    tick=time.monotonic()
    stage_query='''SELECT u.*,m.course_id,m.lo_key,m.exercise_id,m.problem_max_score,m.n_exercises,
        try_cast(u.submit_time AS TIMESTAMP) event_time,
        COALESCE(isfinite(u.score) AND u.score>=0 AND m.problem_max_score>0 AND u.score<=m.problem_max_score+1e-9,FALSE) score_valid,
        COALESCE(u.is_correct IN (0,1) AND m.problem_max_score>0 AND isfinite(m.problem_max_score),FALSE) fallback_possible,
        COALESCE(u.attempts>0 AND u.is_correct IN (0,1),FALSE) efficiency_valid,
        hash(u.user_id)%32 user_shard
        FROM up u JOIN problem_map m USING(problem_id) WHERE m.n_los=1'''
    stage_digest=signature(stage_query)
    stage_done=problem_stage/'_SUCCESS.json'
    stage_reused=stage_done.exists() and json.loads(stage_done.read_text()).get('signature')==stage_digest
    if not stage_reused:
        if problem_stage.exists():shutil.rmtree(problem_stage)
        con.execute(f'COPY ({stage_query}) TO {qs(problem_stage)} (FORMAT PARQUET,COMPRESSION ZSTD,PARTITION_BY(user_shard),ROW_GROUP_SIZE 50000)')
        stage_done.write_text(json.dumps({'signature':stage_digest}))
    lineage.append(stage_digest)
    log.append({'name':'problem_stage','seconds':round(time.monotonic()-tick,3),'sql':stage_query,'shards':32,'checkpoint_reused':stage_reused})
    view('problem_joined',problem_stage/'*'/'*.parquet')
    identity_rows=[]
    for shard in range(32):
        glob=problem_stage/f'user_shard={shard}'/'*.parquet'
        row=con.execute(f'''SELECT count(*) n_rows,count(DISTINCT log_id) distinct_log_ids,
            count(DISTINCT (user_id,problem_id)) distinct_user_problem,
            count(*) FILTER(WHERE event_time IS NULL) invalid_times,
            count(*) FILTER(WHERE n_exercises<>1) ambiguous_exercise_rows,
            count(*) FILTER(WHERE log_id<>substr(user_id,3)||'_'||substr(problem_id,4)) log_id_format_mismatches
            FROM read_parquet({qs(glob)})''').fetchone()
        identity_rows.append((shard,)+row)
    con.execute('''CREATE TEMP TABLE identity_shards(shard INTEGER,n_rows BIGINT,distinct_log_ids BIGINT,
        distinct_user_problem BIGINT,invalid_times BIGINT,ambiguous_exercise_rows BIGINT,log_id_format_mismatches BIGINT)''')
    con.executemany('INSERT INTO identity_shards VALUES (?,?,?,?,?,?,?)',identity_rows)
    save('mapped_problem_identity','''SELECT sum(n_rows)::BIGINT n_rows,sum(distinct_log_ids)::BIGINT distinct_log_ids,
        sum(distinct_user_problem)::BIGINT distinct_user_problem,sum(invalid_times)::BIGINT invalid_times,
        sum(ambiguous_exercise_rows)::BIGINT ambiguous_exercise_rows,sum(log_id_format_mismatches)::BIGINT log_id_format_mismatches
        FROM identity_shards''')
    identity=summary['mapped_problem_identity'][0]
    if identity['n_rows']!=identity['distinct_user_problem']:
        raise RuntimeError('Repeated user-problem rows need an audited as-of state rule before this EDA aggregates attempts.')

    # Materialize only uniquely mapped video rows before exploding segments.
    save('video_segments','''WITH chosen AS MATERIALIZED (
        SELECT u.user_id,v.video_id,v.segment,m.course_id,m.lo_key
        FROM uv u CROSS JOIN UNNEST(u.seq) a(v)
        JOIN resource_map m ON m.resource_id=v.video_id AND m.n_los=1 AND m.resource_type='video'
        ) SELECT c.user_id,c.course_id,c.lo_key,c.video_id,
          s.start_point,s.end_point,s.speed,s.local_start_time epoch_seconds,
          CASE WHEN s.local_start_time BETWEEN 0 AND 4102444800 THEN cast(to_timestamp(s.local_start_time) AS TIMESTAMP) END event_time,
          COALESCE(s.start_point>=0 AND s.end_point>s.start_point AND isfinite(s.start_point) AND isfinite(s.end_point)
            AND s.speed>0 AND isfinite(s.speed),FALSE) valid_measurement
        FROM chosen c CROSS JOIN UNNEST(c.segment) b(s)''',preview=0)
    save('video_mapped_quality',f'''SELECT s.scope,count(*) n_segments,
        count(*) FILTER(WHERE v.valid_measurement AND v.event_time IS NOT NULL) usable_segments,
        count(*) FILTER(WHERE NOT v.valid_measurement) invalid_measurements,
        count(*) FILTER(WHERE v.event_time<TIMESTAMP '2019-01-01' OR v.event_time>=TIMESTAMP '2021-01-01') outlier_calendar_segments,
        count(DISTINCT v.user_id) n_users,count(DISTINCT v.lo_key) n_los,count(DISTINCT v.course_id) n_courses
        FROM video_segments v CROSS JOIN scopes s WHERE s.t0 IS NULL OR (v.event_time>=s.t0 AND v.event_time<s.t1) GROUP BY 1''')
    save('video_scoped','''SELECT s.scope,v.* FROM video_segments v CROSS JOIN scopes s
        WHERE v.valid_measurement AND v.event_time IS NOT NULL AND (s.t0 IS NULL OR (v.event_time>=s.t0 AND v.event_time<s.t1))''',preview=0)
    save('video_union_intervals','''WITH distinct_intervals AS (
        SELECT DISTINCT scope,user_id,course_id,lo_key,video_id,start_point,end_point FROM video_scoped
        ), previous AS (
        SELECT *,max(end_point) OVER(PARTITION BY scope,user_id,lo_key,video_id
          ORDER BY start_point,end_point ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) previous_end FROM distinct_intervals
        ), numbered AS (
        SELECT *,sum(CASE WHEN previous_end IS NULL OR start_point>previous_end THEN 1 ELSE 0 END)
          OVER(PARTITION BY scope,user_id,lo_key,video_id ORDER BY start_point,end_point ROWS UNBOUNDED PRECEDING) island FROM previous
        ) SELECT scope,user_id,course_id,lo_key,video_id,island,min(start_point) interval_start,max(end_point) interval_end
          FROM numbered GROUP BY 1,2,3,4,5,6''',preview=0)
    save('video_features','''WITH raw_stats AS (
        SELECT scope,user_id,course_id,lo_key,count(*) n_video_segments,count(DISTINCT video_id) n_videos_watched,
          sum(end_point-start_point) raw_watch_seconds,sum((end_point-start_point)/speed) wall_watch_seconds,
          min(event_time) first_video_time,max(event_time) last_video_time FROM video_scoped GROUP BY 1,2,3,4
        ), unique_stats AS (
        SELECT scope,user_id,course_id,lo_key,sum(interval_end-interval_start) unique_watch_seconds
          FROM video_union_intervals GROUP BY 1,2,3,4
        ), session_edges AS (
        SELECT *,lag(event_time) OVER(PARTITION BY scope,user_id,lo_key,video_id ORDER BY event_time,start_point,end_point) previous_time FROM video_scoped
        ), sessions AS (
        SELECT scope,user_id,course_id,lo_key,video_id,
          count(*) FILTER(WHERE previous_time IS NULL OR event_time-previous_time>INTERVAL '30 minutes') n_sessions
          FROM session_edges GROUP BY 1,2,3,4,5
        ), session_stats AS (
        SELECT scope,user_id,course_id,lo_key,sum(n_sessions) n_video_sessions_30m,
          count(*) FILTER(WHERE n_sessions>1) n_revisited_videos FROM sessions GROUP BY 1,2,3,4
        ) SELECT r.*,u.unique_watch_seconds,r.raw_watch_seconds-u.unique_watch_seconds rewatch_seconds,
          s.n_video_sessions_30m,s.n_revisited_videos FROM raw_stats r JOIN unique_stats u USING(scope,user_id,course_id,lo_key)
          JOIN session_stats s USING(scope,user_id,course_id,lo_key)''',preview=0)

    # Problem windows are often identical; calculate once when the full source fits.
    problem_same=(summary['problem_source_quality'][0]['n_rows']==summary['problem_source_quality'][1]['n_rows'])
    save('problem_mapped_quality',f'''SELECT s.scope,count(*) n_rows,
        count(*) FILTER(WHERE p.score_valid) score_direct_valid,
        count(*) FILTER(WHERE NOT p.score_valid AND p.fallback_possible) score_fallback_possible,
        count(*) FILTER(WHERE NOT p.score_valid AND NOT p.fallback_possible) score_unresolved,
        count(*) FILTER(WHERE p.problem_max_score IS NULL OR p.problem_max_score<=0) max_score_missing_or_zero,
        count(*) FILTER(WHERE p.score>p.problem_max_score+1e-9 AND p.problem_max_score>0) score_exceeds_max,
        count(*) FILTER(WHERE NOT p.efficiency_valid) efficiency_invalid,
        count(DISTINCT p.user_id) n_users,count(DISTINCT p.course_id) n_courses,count(DISTINCT p.lo_key) n_los
        FROM problem_joined p CROSS JOIN scopes s WHERE s.t0 IS NULL OR (p.event_time>=s.t0 AND p.event_time<s.t1) GROUP BY 1''')
    problem_agg='''SELECT user_id,course_id,lo_key,count(*) n_problems_attempted,count(DISTINCT exercise_id) n_exercises_attempted,
        sum(attempts) total_problem_attempts,count(*) FILTER(WHERE attempts IS NULL OR attempts<=0) invalid_attempts,
        count(*) FILTER(WHERE is_correct=1) n_problems_correct,
        count(*) FILTER(WHERE is_correct=1 AND attempts=1) n_first_try_correct,
        count(*) FILTER(WHERE is_correct IS NULL OR is_correct NOT IN (0,1)) invalid_correctness,
        sum(CASE WHEN score_valid THEN score WHEN fallback_possible THEN is_correct*problem_max_score END) earned_score_sum,
        sum(problem_max_score) FILTER(WHERE problem_max_score>0) attempted_max_score_sum,
        sum(CASE WHEN efficiency_valid THEN is_correct::DOUBLE/attempts END) efficiency_sum,
        count(*) FILTER(WHERE score_valid) n_scores_valid,
        count(*) FILTER(WHERE NOT score_valid AND fallback_possible) n_scores_fallback,
        count(*) FILTER(WHERE NOT score_valid AND NOT fallback_possible) n_scores_unresolved,
        count(*) FILTER(WHERE NOT efficiency_valid) n_efficiency_unresolved
        FROM {source} WHERE event_time IS NOT NULL {window_filter} GROUP BY 1,2,3'''
    if problem_same:
        agg_stage=out/'_stage'/'problem_features'
        print('\nAggregate problem features by 32 user shards',flush=True)
        tick=time.monotonic()
        agg_digest=signature(problem_agg);agg_done=agg_stage/'_SUCCESS.json'
        agg_reused=agg_done.exists() and json.loads(agg_done.read_text()).get('signature')==agg_digest
        if not agg_reused:
            if agg_stage.exists():shutil.rmtree(agg_stage)
            agg_stage.mkdir()
            for shard in range(32):
                source=f'read_parquet({qs(problem_stage/f"user_shard={shard}"/"*.parquet")})'
                query=problem_agg.format(source=source,window_filter='')
                con.execute(f'COPY ({query}) TO {qs(agg_stage/f"part-{shard:03d}.parquet")} (FORMAT PARQUET,COMPRESSION ZSTD)')
            agg_done.write_text(json.dumps({'signature':agg_digest}))
        lineage.append(agg_digest)
        log.append({'name':'problem_feature_shards','seconds':round(time.monotonic()-tick,3),'sql_template':problem_agg,'shards':32,'checkpoint_reused':agg_reused})
        save('problem_features_base',f'SELECT * FROM read_parquet({qs(agg_stage/"*.parquet")})',preview=0)
        save('problem_features','SELECT s.scope,p.* FROM problem_features_base p CROSS JOIN scopes s',preview=0)
    else:
        save('problem_features',"SELECT 'ALL_AVAILABLE' AS scope,p.* FROM ("+problem_agg.format(source='problem_joined',window_filter='')+
             ") p UNION ALL SELECT 'WINDOW_20200501_20201214',p.* FROM ("+problem_agg.format(source='problem_joined',window_filter='AND '+inside('event_time'))+") p",preview=0)

    # Social fields: preserve course-only comments and require a resource to assign LO.
    save('comment_prepared','''WITH typed AS (
        SELECT c.id comment_id,CASE WHEN c.user_id IS NOT NULL THEN 'U_'||CAST(c.user_id AS VARCHAR) END user_id,
          c.resource_id,c.text,
          CASE WHEN c.create_time BETWEEN 0 AND 4102444800000 THEN cast(to_timestamp(c.create_time/1000.0) AS TIMESTAMP) END event_time,
          r.course_id resource_course,r.lo_key resource_lo,r.n_los,cc.course_id relation_course,cc.n_courses
        FROM comments c LEFT JOIN resource_map r ON c.resource_id=r.resource_id LEFT JOIN comment_course cc ON c.id=cc.comment_id
        ) SELECT *,CASE WHEN n_los=1 AND (relation_course IS NULL OR (n_courses=1 AND relation_course=resource_course)) THEN resource_lo END lo_key,
          CASE WHEN n_los=1 AND (relation_course IS NULL OR (n_courses=1 AND relation_course=resource_course)) THEN resource_course
               WHEN n_courses=1 THEN relation_course END course_id,
          CASE WHEN text IS NULL OR trim(text)='' THEN 'EMPTY'
               WHEN length(trim(text))<=5 THEN 'SHORT_LE5_CHARS' ELSE 'LONGER' END text_status,
          row_number() OVER(PARTITION BY comment_id ORDER BY event_time,user_id,resource_id,text) duplicate_id_rank,
          count(*) OVER(PARTITION BY comment_id) duplicate_id_count
        FROM typed''',preview=0)
    save('social_source_quality','''SELECT s.scope,count(*) comment_rows,count(DISTINCT c.comment_id) distinct_comment_ids,
        count(*) FILTER(WHERE c.event_time IS NULL) invalid_times,
        count(*) FILTER(WHERE c.course_id IS NOT NULL) course_mapped,
        count(*) FILTER(WHERE c.lo_key IS NOT NULL) lo_mapped,
        count(*) FILTER(WHERE c.text_status='EMPTY') empty_text,
        count(*) FILTER(WHERE c.text_status='SHORT_LE5_CHARS') short_text,
        count(*) FILTER(WHERE c.duplicate_id_rank>1) duplicate_id_extra_rows,
        count(*) FILTER(WHERE c.text_status='LONGER' AND c.lo_key IS NOT NULL) lo_mapped_longer_text
        FROM comment_prepared c CROSS JOIN scopes s WHERE s.t0 IS NULL OR (c.event_time>=s.t0 AND c.event_time<s.t1) GROUP BY 1''')
    con.execute('''CREATE TEMP TABLE reply_parent AS SELECT r.reply_id,count(DISTINCT r.comment_id) n_parents,
        min(r.comment_id) parent_comment_id FROM comment_reply r GROUP BY 1''')
    save('reply_prepared','''SELECT r.id reply_id,r.user_id,try_cast(r.create_time AS TIMESTAMP) event_time,
        p.parent_comment_id,p.n_parents,c.event_time parent_time,c.course_id,c.lo_key,
        row_number() OVER(PARTITION BY r.id ORDER BY r.create_time,r.user_id,r.text) duplicate_id_rank
        FROM replies r LEFT JOIN reply_parent p ON r.id=p.reply_id
        LEFT JOIN comment_prepared c ON p.parent_comment_id=c.comment_id AND c.duplicate_id_rank=1''',preview=0)
    save('reply_source_quality','''SELECT s.scope,count(*) reply_rows,
        count(*) FILTER(WHERE r.n_parents=1) unique_parent,
        count(*) FILTER(WHERE r.n_parents>1) multiple_parents,
        count(*) FILTER(WHERE r.lo_key IS NOT NULL AND r.n_parents=1) lo_mapped,
        count(*) FILTER(WHERE r.event_time<r.parent_time) before_parent_comment,
        count(*) FILTER(WHERE r.duplicate_id_rank>1) duplicate_id_extra_rows
        FROM reply_prepared r CROSS JOIN scopes s WHERE s.t0 IS NULL OR (r.event_time>=s.t0 AND r.event_time<s.t1) GROUP BY 1''')
    save('comment_scoped','''SELECT s.scope,c.* FROM comment_prepared c CROSS JOIN scopes s
        WHERE c.duplicate_id_rank=1 AND c.lo_key IS NOT NULL AND c.user_id IS NOT NULL AND c.event_time IS NOT NULL
        AND (s.t0 IS NULL OR (c.event_time>=s.t0 AND c.event_time<s.t1))''',preview=0)
    save('reply_scoped','''SELECT s.scope,r.* FROM reply_prepared r CROSS JOIN scopes s
        WHERE r.duplicate_id_rank=1 AND r.n_parents=1 AND r.lo_key IS NOT NULL AND r.user_id IS NOT NULL
        AND r.event_time IS NOT NULL AND r.event_time>=r.parent_time
        AND (s.t0 IS NULL OR (r.event_time>=s.t0 AND r.event_time<s.t1))''',preview=0)
    save('social_features','''WITH received AS (
        SELECT DISTINCT c.scope,c.comment_id FROM comment_scoped c JOIN reply_scoped r
          ON c.scope=r.scope AND c.comment_id=r.parent_comment_id AND r.event_time>=c.event_time
        ), comment_counts AS (
        SELECT c.scope,c.user_id,c.course_id,c.lo_key,count(*) n_comments,
          count(*) FILTER(WHERE c.text_status='LONGER') n_comments_longer_text,
          count(*) FILTER(WHERE x.comment_id IS NOT NULL) n_comments_with_reply,
          count(*) FILTER(WHERE x.comment_id IS NOT NULL AND c.text_status='LONGER') n_longer_comments_with_reply
          FROM comment_scoped c LEFT JOIN received x USING(scope,comment_id) GROUP BY 1,2,3,4
        ), reply_counts AS (
        SELECT scope,user_id,course_id,lo_key,count(*) n_replies_authored FROM reply_scoped GROUP BY 1,2,3,4
        ) SELECT COALESCE(c.scope,r.scope) AS scope,COALESCE(c.user_id,r.user_id) user_id,
          COALESCE(c.course_id,r.course_id) course_id,COALESCE(c.lo_key,r.lo_key) lo_key,
          COALESCE(c.n_comments,0) n_comments,COALESCE(c.n_comments_longer_text,0) n_comments_longer_text,
          COALESCE(c.n_comments_with_reply,0) n_comments_with_reply,COALESCE(c.n_longer_comments_with_reply,0) n_longer_comments_with_reply,
          COALESCE(r.n_replies_authored,0) n_replies_authored
          FROM comment_counts c FULL JOIN reply_counts r USING(scope,user_id,course_id,lo_key)''',preview=0)

    save('user_lo_features','''WITH keys AS (
        SELECT scope,user_id,course_id,lo_key FROM video_features UNION
        SELECT scope,user_id,course_id,lo_key FROM problem_features UNION
        SELECT scope,user_id,course_id,lo_key FROM social_features
        ), joined AS (
        SELECT k.*,d.n_videos_available,d.n_exercises_available,d.n_problems_available,
          d.available_video_seconds_proxy,d.n_videos_missing_duration,d.available_max_score_sum,d.n_problems_missing_score,
          (v.user_id IS NOT NULL) has_video,(p.user_id IS NOT NULL) has_problem,(s.user_id IS NOT NULL) has_social,
          COALESCE(v.n_videos_watched,0) n_videos_watched,COALESCE(v.n_video_segments,0) n_video_segments,
          COALESCE(v.raw_watch_seconds,0) raw_watch_seconds,COALESCE(v.unique_watch_seconds,0) unique_watch_seconds,
          COALESCE(v.rewatch_seconds,0) rewatch_seconds,COALESCE(v.n_video_sessions_30m,0) n_video_sessions_30m,
          COALESCE(v.n_revisited_videos,0) n_revisited_videos,
          COALESCE(p.n_problems_attempted,0) n_problems_attempted,COALESCE(p.n_exercises_attempted,0) n_exercises_attempted,
          COALESCE(p.total_problem_attempts,0) total_problem_attempts,
          COALESCE(p.n_problems_correct,0) n_problems_correct,COALESCE(p.n_first_try_correct,0) n_first_try_correct,
          COALESCE(p.earned_score_sum,0) earned_score_sum,COALESCE(p.attempted_max_score_sum,0) attempted_max_score_sum,
          COALESCE(p.n_scores_valid,0) n_scores_valid,COALESCE(p.n_scores_fallback,0) n_scores_fallback,
          COALESCE(p.n_scores_unresolved,0) n_scores_unresolved,COALESCE(p.n_efficiency_unresolved,0) n_efficiency_unresolved,
          COALESCE(p.efficiency_sum,0) efficiency_sum,COALESCE(p.invalid_attempts,0) invalid_attempts,
          COALESCE(p.invalid_correctness,0) invalid_correctness,
          COALESCE(s.n_comments,0) n_comments,COALESCE(s.n_comments_longer_text,0) n_comments_longer_text,
          COALESCE(s.n_comments_with_reply,0) n_comments_with_reply,
          COALESCE(s.n_longer_comments_with_reply,0) n_longer_comments_with_reply,
          COALESCE(s.n_replies_authored,0) n_replies_authored
        FROM keys k JOIN dim_lo_catalog d USING(course_id,lo_key)
        LEFT JOIN video_features v USING(scope,user_id,course_id,lo_key)
        LEFT JOIN problem_features p USING(scope,user_id,course_id,lo_key)
        LEFT JOIN social_features s USING(scope,user_id,course_id,lo_key)
        ) SELECT *,
          unique_watch_seconds/NULLIF(available_video_seconds_proxy,0) TS_proxy_partial_catalog,
          CASE WHEN n_videos_missing_duration=0 THEN unique_watch_seconds/NULLIF(available_video_seconds_proxy,0) END TS_proxy_complete_catalog,
          n_videos_watched::DOUBLE/NULLIF(n_videos_available,0) RV,
          n_exercises_attempted::DOUBLE/NULLIF(n_exercises_available,0) AV,
          CASE WHEN invalid_attempts=0 THEN total_problem_attempts::DOUBLE/NULLIF(n_problems_available,0) END IF_raw,
          earned_score_sum/NULLIF(available_max_score_sum,0) KL_partial_catalog,
          CASE WHEN n_problems_missing_score=0 AND n_scores_unresolved=0 THEN earned_score_sum/NULLIF(available_max_score_sum,0) END KL_complete_catalog,
          CASE WHEN n_efficiency_unresolved=0 THEN efficiency_sum/NULLIF(n_problems_available,0) END CFA_inverse_attempts_proxy,
          earned_score_sum/NULLIF(attempted_max_score_sum,0) mastery_attempted_partial,
          n_problems_attempted::DOUBLE/NULLIF(n_problems_available,0) problem_participation,
          (n_comments>0)::INTEGER comment_participation,(n_replies_authored>0)::INTEGER reply_participation,
          n_comments_with_reply::DOUBLE/NULLIF(n_comments,0) response_received_rate,
          n_longer_comments_with_reply::DOUBLE/NULLIF(n_comments_longer_text,0) response_received_rate_longer_text,
          n_revisited_videos::DOUBLE/NULLIF(n_videos_watched,0) return_visit_rate_diagnostic
        FROM joined''',preview=0)
    save('scope_support','''SELECT scope,count(*) user_lo_rows,count(DISTINCT user_id) users,
        count(DISTINCT (user_id,course_id)) user_course_pairs,count(DISTINCT course_id) courses,count(DISTINCT lo_key) los,
        count(*) FILTER(WHERE has_video) with_video,count(*) FILTER(WHERE has_problem) with_problem,
        count(*) FILTER(WHERE has_social) with_social,count(*) FILTER(WHERE has_video AND has_problem) with_video_and_problem,
        count(*) FILTER(WHERE (has_video OR has_problem) AND has_social) with_learning_and_social,
        count(*) FILTER(WHERE has_video AND has_problem AND has_social) with_all_three_sources,
        count(*) FILTER(WHERE TS_proxy_complete_catalog IS NOT NULL AND RV IS NOT NULL AND AV IS NOT NULL AND IF_raw IS NOT NULL) complete_coelo_component_denominators,
        count(*) FILTER(WHERE KL_complete_catalog IS NOT NULL AND CFA_inverse_attempts_proxy IS NOT NULL) complete_academic_component_denominators
        FROM user_lo_features GROUP BY 1''')

    features=['raw_watch_seconds','unique_watch_seconds','rewatch_seconds','n_videos_watched','n_video_sessions_30m',
              'n_problems_attempted','total_problem_attempts','n_problems_correct','n_scores_valid','n_scores_fallback',
              'n_comments','n_comments_longer_text','n_replies_authored','n_comments_with_reply',
              'TS_proxy_partial_catalog','TS_proxy_complete_catalog','RV','AV','IF_raw',
              'KL_partial_catalog','KL_complete_catalog','CFA_inverse_attempts_proxy','problem_participation',
              'comment_participation','reply_participation','response_received_rate','return_visit_rate_diagnostic']
    con.execute('''CREATE TEMP TABLE common_keys AS SELECT user_id,course_id,lo_key FROM user_lo_features
        GROUP BY 1,2,3 HAVING count(DISTINCT scope)=2''')
    cohort_sql='''CREATE TEMP VIEW feature_cohorts AS
        SELECT 'ALL_OBSERVED_USER_LO' cohort,* FROM user_lo_features
        UNION ALL SELECT 'LEARNING_USER_LO',* FROM user_lo_features WHERE has_video OR has_problem
        UNION ALL SELECT 'VIDEO_USER_LO',* FROM user_lo_features WHERE has_video
        UNION ALL SELECT 'PROBLEM_USER_LO',* FROM user_lo_features WHERE has_problem
        UNION ALL SELECT 'SOCIAL_USER_LO',* FROM user_lo_features WHERE has_social
        UNION ALL SELECT 'COMMON_USER_LO',f.* FROM user_lo_features f JOIN common_keys USING(user_id,course_id,lo_key)'''
    con.execute(cohort_sql)
    lineage.append(hashlib.sha256(cohort_sql.encode()).hexdigest())
    distribution_parts=[]
    for feature in features:
        distribution_parts.append(f'''SELECT cohort,scope,{qs(feature)} feature,count(*) n_rows,
          count({feature}) n_non_null,count(*) FILTER(WHERE {feature}=0) n_zero,
          count(*) FILTER(WHERE {feature}>1+1e-9) n_above_one,
          count(*) FILTER(WHERE {feature}>1) n_above_one_strict,min({feature})::DOUBLE min_value,
          quantile_cont({feature},0.25)::DOUBLE p25,median({feature})::DOUBLE median_value,
          quantile_cont({feature},0.75)::DOUBLE p75,quantile_cont({feature},0.95)::DOUBLE p95,
          quantile_cont({feature},0.99)::DOUBLE p99,max({feature})::DOUBLE max_value,
          avg({feature}) mean_value,stddev_pop({feature}) std_value FROM feature_cohorts GROUP BY 1,2''')
    save('feature_distributions',' UNION ALL '.join(distribution_parts),preview=0)
    change_parts=[]
    for feature in features:
        change_parts.append(f'''SELECT {qs(feature)} feature,count(*) common_rows,
          count(*) FILTER(WHERE (a.{feature} IS NULL AND w.{feature} IS NULL)
            OR abs(a.{feature}-w.{feature})<=1e-9*greatest(1.0,abs(a.{feature}),abs(w.{feature}))) unchanged_rows,
          count(*) FILTER(WHERE a.{feature} IS NOT NULL) all_non_null,
          count(*) FILTER(WHERE w.{feature} IS NOT NULL) window_non_null,
          count(*) FILTER(WHERE a.{feature} IS NOT NULL AND w.{feature} IS NOT NULL) paired_non_null_rows,
          avg(a.{feature}) all_mean,avg(w.{feature}) window_mean,
          avg(w.{feature}-a.{feature}) mean_change
          FROM user_lo_features a JOIN user_lo_features w USING(user_id,course_id,lo_key)
          WHERE a.scope='ALL_AVAILABLE' AND w.scope='WINDOW_20200501_20201214' ''')
    save('common_cohort_changes',' UNION ALL '.join(change_parts),preview=0)
    save('support_overlap','''WITH a AS (SELECT * FROM user_lo_features WHERE scope='ALL_AVAILABLE'),
        w AS (SELECT * FROM user_lo_features WHERE scope='WINDOW_20200501_20201214')
        SELECT (SELECT count(*) FROM a) all_user_lo,(SELECT count(*) FROM w) window_user_lo,
          (SELECT count(*) FROM common_keys) common_user_lo,
          (SELECT count(*) FROM a WHERE NOT EXISTS(SELECT 1 FROM w WHERE w.user_id=a.user_id AND w.course_id=a.course_id AND w.lo_key=a.lo_key)) all_only_user_lo''')
    # Component availability is explicit: no guessed likes, ratings or sentiment.
    availability=[
        ['COELO','TS','AVAILABLE_WITH_PROXY','union video intervals; caption-end denominator; incomplete-duration flag'],
        ['COELO','RV','AVAILABLE','distinct watched videos / fixed catalog video count'],
        ['COELO','AV','AVAILABLE','distinct attempted exercises / fixed catalog exercise count'],
        ['COELO','IF','RAW_AVAILABLE','attempts / available problems; no normalization fitted'],
        ['AFELO','FA likes/dislikes','UNAVAILABLE','no like/dislike columns in supplied tables'],
        ['AFELO','RS ratings','UNAVAILABLE','no rating columns in supplied tables'],
        ['AFELO proxy','comment/reply participation','AVAILABLE_PARTIAL_COVERAGE','resource-linked comments and parent-linked replies only'],
        ['AFELO proxy','response_received_rate','AVAILABLE_PARTIAL_COVERAGE','comment and received reply both inside the same scope'],
        ['AFELO proxy','sentiment_valence','NOT_COMPUTED','text exists; no sentiment model/labels supplied'],
        ['AFELO diagnostic','return visits','DIAGNOSTIC_ONLY','30-minute session gap; not interpreted as positive affect'],
        ['ACELO','KL','AVAILABLE_WITH_MISSINGNESS','direct/fallback score; incomplete catalog produces a separate diagnostic'],
        ['ACELO','CFA','PROXY_ONLY','is_correct / attempts; published final formula not locked by this EDA'],
    ]
    (out/'component_availability.json').write_text(json.dumps(availability,ensure_ascii=False,indent=2))
    checks={
        'user_lo_unique':con.execute('SELECT count(*)=count(DISTINCT (scope,user_id,course_id,lo_key)) FROM user_lo_features').fetchone()[0],
        'window_is_subset':con.execute("SELECT count(*)=0 FROM video_scoped WHERE scope='WINDOW_20200501_20201214' AND NOT "+inside('event_time')).fetchone()[0],
        'interval_union_not_greater_than_raw':con.execute('SELECT count(*)=0 FROM video_features WHERE unique_watch_seconds>raw_watch_seconds+1e-6 OR rewatch_seconds< -1e-6').fetchone()[0],
        'received_reply_numerator_bounded':con.execute('SELECT count(*)=0 FROM social_features WHERE n_comments_with_reply>n_comments OR n_longer_comments_with_reply>n_comments_longer_text').fetchone()[0],
        'no_future_reply_credit':con.execute('SELECT count(*)=0 FROM reply_scoped WHERE event_time<parent_time').fetchone()[0],
        'catalog_counts_not_exceeded':con.execute('SELECT count(*)=0 FROM user_lo_features WHERE n_videos_watched>n_videos_available OR n_problems_attempted>n_problems_available OR n_exercises_attempted>n_exercises_available').fetchone()[0],
        'bounded_non_duration_components':con.execute('''SELECT count(*)=0 FROM user_lo_features
            WHERE RV< -1e-9 OR RV>1+1e-9 OR AV< -1e-9 OR AV>1+1e-9
            OR KL_complete_catalog< -1e-9 OR KL_complete_catalog>1+1e-9
            OR CFA_inverse_attempts_proxy< -1e-9 OR CFA_inverse_attempts_proxy>1+1e-9''').fetchone()[0],
    }
    if problem_same:
        checks['problem_features_identical_on_common_cohort']=con.execute("SELECT count(*)=0 FROM common_cohort_changes WHERE feature IN ('n_problems_attempted','total_problem_attempts','n_problems_correct','n_scores_valid','n_scores_fallback') AND unchanged_rows<>common_rows").fetchone()[0]
    if not all(checks.values()):raise AssertionError(checks)
    con.close()
    engine_hash=globals().get('ENGINE_SOURCE_SHA256') or hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    manifest={'created_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'window_start':WINDOW_START,
        'window_end_exclusive':WINDOW_END_EXCLUSIVE,'raw_root':str(raw),'model_root':str(model),
        'duckdb_version':duckdb.__version__,'threads':threads,'memory_limit':memory_limit,
        'sampling':'none; exact counts and quantiles','input_fingerprint':input_fingerprint(raw,model),
        'analysis_source_sha256':engine_hash,'checks':checks,'problem_windows_identical':problem_same,
        'elapsed_seconds':round(time.monotonic()-tick_all,3),'queries':log,
        'policies':{'full_data':'ALL_AVAILABLE has no calendar filter; calendar outliers remain visible',
          'video_usable':'parseable timestamp and finite, positive interval/speed; union intervals after scope filter',
          'denominators':'same complete static catalog in both scopes; no resource release schedule assumed',
          'social':'raw/longer-text variants; <=5 chars is a diagnostic, not a learning-noise classifier',
          'received_reply':'both comment and reply inside scope; reply not earlier than comment',
          'sentiment':'not computed; absent model output is not neutral sentiment',
          'time_zone':'epoch interpreted in UTC; naive strings interpreted consistently as UTC; platform timezone unverified',
          'cohorts':'sparse observed user-course-LO rows; no full Cartesian population and no dropout labels'}}
    (out/'run_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2,default=str))
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,default=str))
    write_report(out)
    print('\nCOMPLETE',out,'seconds=',manifest['elapsed_seconds'],flush=True)
    return summary


def write_report(out):
    out=Path(out);s=json.loads((out/'summary.json').read_text())
    fmt=lambda n:f'{int(n):,}'.replace(',','.')
    db=duckdb.connect()
    cur=db.execute(f"SELECT * FROM read_parquet({qs(out/'tables/feature_distributions.parquet')})")
    cols=[c[0] for c in cur.description]
    distributions={}
    for row in cur.fetchall():
        r=dict(zip(cols,row));distributions[(r['cohort'],r['scope'],r['feature'])]=r
    db.close()
    number=lambda v:'null' if v is None else f'{float(v):.5g}'
    def distribution_table(features,cohort='COMMON_USER_LO'):
        lines=['',f'Phân phối tại cohort `{cohort}`. Các cột lần lượt là mean, median và p95; bỏ qua null khi tính thống kê.',
            '', '| Feature | Toàn mean | Toàn median | Toàn p95 | Khoảng mean | Khoảng median | Khoảng p95 |',
            '|---|---:|---:|---:|---:|---:|---:|']
        for feature in features:
            a=distributions[(cohort,'ALL_AVAILABLE',feature)];w=distributions[(cohort,'WINDOW_20200501_20201214',feature)]
            values=[number(r[k]) for r in [a,w] for k in ['mean_value','median_value','p95']]
            lines.append('| '+feature+' | '+' | '.join(values)+' |')
        return lines
    lines=['# EDA các thành phần COELO AFELO ACELO theo hai phạm vi thời gian','',
        'Khoảng so sánh: từ 01/05/2020 đến hết 14/12/2020, hiện thực bằng `[2020-05-01,2020-12-15)`. '
        'Toàn data không lọc ngày; timestamp bất thường được giữ trong audit và gắn cờ. '
        'Thống kê quét toàn bộ dữ liệu, không lấy mẫu. Chưa tạo nhãn tổng hợp, fit normalization hoặc chạy sentiment.',
        '', '## 1 Số lượng đầu vào theo nguồn','',
        '| Nguồn | Toàn data | Trong khoảng | Giữ lại |','|---|---:|---:|---:|']
    for name,count,title in [('video_source_quality','n_segments','Video segment'),('problem_source_quality','n_rows','Problem row'),
                              ('social_source_quality','comment_rows','Translated comment'),('reply_source_quality','reply_rows','Reply')]:
        rows={r['scope']:r for r in s[name]};a=rows['ALL_AVAILABLE'][count];w=rows['WINDOW_20200501_20201214'][count]
        lines.append(f'| {title} | {fmt(a)} | {fmt(w)} | {100*w/a:.2f}% |')
    lines += ['', 'Bảng này mô tả toàn nguồn. Các component ở những phần sau chỉ dùng event gán được duy nhất tới chapter trong catalog. '
              'Không diễn giải mọi event ngoài catalog là lỗi dữ liệu.',
        '', 'Các event dùng được ở cấp chapter:', '',
        '| Nguồn | Toàn data | Trong khoảng |', '|---|---:|---:|',
    ]
    for name,key,title in [('video_mapped_quality','usable_segments','Video đủ interval/speed'),
                           ('problem_mapped_quality','n_rows','Problem nối được chapter'),
                           ('social_source_quality','lo_mapped','Comment nối được chapter'),
                           ('reply_source_quality','lo_mapped','Reply nối được chapter')]:
        rows={r['scope']:r for r in s[name]}
        lines.append(f'| {title} | {fmt(rows["ALL_AVAILABLE"][key])} | {fmt(rows["WINDOW_20200501_20201214"][key])} |')
    lines += [
        '', '## 2 Population quan sát để tính component','',
        '| Phạm vi | User–course–chapter | Có video | Có problem | Có social | Có cả ba nguồn |',
        '|---|---:|---:|---:|---:|---:|']
    for r in s['scope_support']:
        lines.append(f'| {r["scope"]} | {fmt(r["user_lo_rows"])} | {fmt(r["with_video"])} | {fmt(r["with_problem"])} | {fmt(r["with_social"])} | {fmt(r["with_all_three_sources"])} |')
    overlap=s['support_overlap'][0]
    lines += ['',f'Có {fmt(overlap["common_user_lo"])} user–course–chapter xuất hiện trong cả hai phạm vi. '
              'Bảng `common_cohort_changes` so sánh đúng cùng các đối tượng này để tránh nhầm biến động population với biến động feature.',
        '', '## 3 COELO','',
        '`TS`: hợp nhất interval theo user–course–chapter–video sau khi lọc thời gian. '
        'Giữ riêng raw seconds, unique seconds, rewatch và session với gap 30 phút. '
        'Duration mẫu số là caption-end proxy; `TS_proxy_partial_catalog` và `TS_proxy_complete_catalog` được tách riêng.',
        '`RV`: số video đã xem / video catalog. `AV`: exercise đã thử / exercise catalog. '
        '`IF_raw`: tổng attempts / số problem catalog; chưa fit P95 hoặc min–max để tạo IF chuẩn hóa.',
        'Mẫu số catalog giống nhau giữa hai phạm vi. Việc tài nguyên có thực sự được mở ở từng ngày chưa được xác minh. '
        'Ngay cả khi mọi video đều có caption proxy, proxy vẫn có thể ngắn hơn duration thực: TS chưa clip có thể vượt 1. '
        'Không dùng việc clip để che hiện tượng này trong EDA.',
    ]
    lines += distribution_table(['raw_watch_seconds','unique_watch_seconds','rewatch_seconds','n_videos_watched',
        'n_video_sessions_30m','TS_proxy_complete_catalog','RV','AV','IF_raw'])
    lines += distribution_table(['unique_watch_seconds','rewatch_seconds','n_video_sessions_30m',
        'TS_proxy_complete_catalog','RV'],cohort='VIDEO_USER_LO')
    lines += ['', '## 4 ACELO','',
        'Audit riêng score null/âm/vượt điểm tối đa, attempts không hợp lệ và correctness không hợp lệ. '
        'Điểm trực tiếp chỉ hợp lệ khi có thang điểm tối đa dương và nằm trong thang. '
        'Fallback dùng `is_correct × problem_max_score` khi đủ trường. Phần không đủ dữ liệu được giữ là unresolved.',
        'Dòng bài tập được lọc bằng `submit_time`; `attempts` là giá trị tổng hợp đang lưu trong dòng. '
        'Chưa thể khẳng định thời điểm riêng của từng attempt chỉ từ trường này.',
        'Giữ các thành phần `KL_partial_catalog`, `KL_complete_catalog`, `CFA_inverse_attempts_proxy`, '
        'problem participation và mastery attempted. KL với catalog thiếu điểm không được gộp chung với KL đủ catalog. '
        'Không chốt công thức ACELO tổng hợp vì các tài liệu hiện có còn khác nhau ở phép gộp.',
        '', '| Phạm vi | Problem row map được | Score trực tiếp hợp lệ | Có thể fallback | Chưa giải được score |',
        '|---|---:|---:|---:|---:|',
    ]
    for r in s['problem_mapped_quality']:
        lines.append(f'| {r["scope"]} | {fmt(r["n_rows"])} | {fmt(r["score_direct_valid"])} | {fmt(r["score_fallback_possible"])} | {fmt(r["score_unresolved"])} |')
    lines += distribution_table(['n_problems_attempted','total_problem_attempts','n_problems_correct','n_scores_valid','n_scores_fallback',
        'KL_complete_catalog','CFA_inverse_attempts_proxy','problem_participation'])
    lines += distribution_table(['n_problems_attempted','total_problem_attempts','KL_complete_catalog',
        'CFA_inverse_attempts_proxy'],cohort='PROBLEM_USER_LO')
    lines += ['', '## 5 AFELO và social proxy','',
        'Không có like/dislike/rating trong dữ liệu được cung cấp để tái tạo AFELO gốc. '
        'EDA đo số comment, reply do user viết, comment nhận phản hồi và tỷ lệ participation/response.',
        'Comment chỉ xuống chapter khi resource nối duy nhất và không mâu thuẫn với course-comment. '
        'Reply cần một parent comment đã có resource để xác định chapter. '
        'Tỷ lệ nhận phản hồi chỉ ghi nhận reply nằm trong cùng phạm vi thời gian với comment, tránh lấy phản hồi tương lai.',
        'Bản `longer_text` chỉ loại text rỗng hoặc dài tối đa 5 ký tự để làm sensitivity. '
        'Đây không phải bộ phân loại noise học tập. Sentiment chưa được chạy; không điền sentiment thiếu bằng 0 hay trung tính.',
    ]
    lines += distribution_table(['n_comments','n_comments_longer_text','n_replies_authored','n_comments_with_reply',
        'comment_participation','reply_participation','response_received_rate'])
    lines += distribution_table(['n_comments','n_replies_authored','response_received_rate'],cohort='SOCIAL_USER_LO')
    lines += ['', '## 6 Độ đầy đủ catalog','',
        '| Mục | Số chapter |','|---|---:|']
    cat=s['catalog_quality'][0]
    for k,title in [('n_los','Tất cả'),('los_with_video','Có video'),('los_complete_duration_proxy','Đủ caption-duration proxy'),
                    ('los_with_problem','Có problem'),('los_complete_problem_max_score','Đủ điểm tối đa problem')]:
        lines.append(f'| {title} | {fmt(cat[k])} |')
    lines += ['', 'Độ đầy đủ của component trên cùng cohort giữa hai phạm vi:', '',
        '| Component | Số user–LO chung | Non-null toàn | Non-null khoảng |', '|---|---:|---:|---:|']
    for feature in ['TS_proxy_complete_catalog','RV','AV','IF_raw','KL_complete_catalog','CFA_inverse_attempts_proxy','response_received_rate']:
        a=distributions[('COMMON_USER_LO','ALL_AVAILABLE',feature)];w=distributions[('COMMON_USER_LO','WINDOW_20200501_20201214',feature)]
        lines.append(f'| {feature} | {fmt(a["n_rows"])} | {fmt(a["n_non_null"])} | {fmt(w["n_non_null"])} |')
    lines += ['', '## 7 Cách đọc các bảng phân phối','',
        '`feature_distributions.parquet` có count, non-null, zero, mean, p25, median, p75, p95, p99, min/max và độ lệch chuẩn. '
        'Có sáu cohort: tất cả user–LO quan sát, user–LO có learning activity, có video, có problem, có social activity và cùng user–LO giữa hai phạm vi. '
        'Cột `n_above_one` chỉ có nghĩa kiểm tra biên với các tỷ lệ; count/seconds lớn hơn 1 là bình thường. '
        'Dùng tolerance 1e-9 cho kiểm tra biên và so sánh số thực; giữ thêm `n_above_one_strict` để thấy sai số làm tròn.',
        '`common_cohort_changes.mean_change` là trung bình chênh lệch trên các cặp đều non-null; '
        'hai cột mean riêng có thể dùng số dòng non-null khác nhau. Các cột đếm paired/non-null cho phép đối chiếu.',
        'Zero ở các count là không quan sát thấy hoạt động trong nguồn hiện có của một user–LO đã xuất hiện; '
        'không phải kết luận user không học hoặc có chất lượng kém. Mẫu số bằng 0 sinh null.',
        '', '## 8 Tái lập','',
        'Mã nguồn và notebook: `../04_eda_engagement_features_comparison.py` và `.ipynb`. '
        'SQL, policies, hash đầu vào, thời gian chạy và invariant checks nằm trong `run_manifest.json`. '
        'Các tỷ lệ ở đây mô tả input/component, chưa phải nhãn COELO/AFELO/ACELO dùng cho review.']
    (out/'bao_cao_eda_features.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--raw-root');parser.add_argument('--model-root');parser.add_argument('--output-root')
    args=parser.parse_args();run_eda(args.raw_root,args.model_root,args.output_root)
