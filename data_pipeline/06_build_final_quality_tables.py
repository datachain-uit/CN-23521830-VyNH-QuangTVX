"""Final-window user/chapter component tables, with AFELO deferred by default.

Uses notebook 01/02 inputs and notebook 05 time bounds. No phase features or
training model is created. Missing chapters stay unobserved. The companion
notebook embeds this source and needs only DuckDB.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import time
from pathlib import Path

import duckdb

FORMULA_VERSION = 'schema12_observed_chapters_v1'
POLICY_VERSION = 'final-window-v1'
KEYS = 'user_id,course_id,lo_key'
META = 'user_id,course_id,chapter_major,lo_key,enroll_time,course_end,split'
COE_INPUTS = ['n_videos_available','available_video_seconds_proxy','n_videos_missing_duration',
    'n_exercises_available','n_problems_available','n_videos_watched','n_video_segments',
    'n_invalid_video_segments','raw_watch_seconds','unique_watch_seconds','rewatch_seconds',
    'n_video_sessions_30m','n_revisited_videos','n_exercises_attempted','n_problems_attempted',
    'total_problem_attempts','invalid_attempts']
AFE_INPUTS = ['n_comments_raw','n_comments_filtered','n_comments_empty','n_comments_duplicate',
    'n_replies_authored','n_user_comments_with_reply']
ACE_INPUTS = ['n_problems_available','available_max_score_sum','n_problems_missing_score',
    'n_problems_attempted','n_problems_correct','n_first_try_correct','total_problem_attempts',
    'earned_score_sum','attempted_max_score_sum','efficiency_sum','n_scores_valid',
    'n_scores_fallback','n_scores_unresolved','n_efficiency_unresolved','invalid_attempts']
COUNT_FIELDS = ['n_videos_watched','n_video_segments','n_invalid_video_segments',
    'raw_watch_seconds','unique_watch_seconds','rewatch_seconds','n_video_sessions_30m',
    'n_revisited_videos','n_exercises_attempted','n_problems_attempted','total_problem_attempts',
    'n_problems_correct','n_first_try_correct','earned_score_sum','attempted_max_score_sum',
    'efficiency_sum','n_scores_valid','n_scores_fallback','n_scores_unresolved',
    'n_efficiency_unresolved','invalid_attempts','n_comments_raw','n_comments_filtered',
    'n_comments_empty','n_comments_duplicate','n_replies_authored','n_user_comments_with_reply']


def qs(value):
    return "'" + str(value).replace("'", "''") + "'"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(raw, model):
    values = []
    for kind, root in [('raw',Path(raw)),('model',Path(model))]:
        for p in sorted(root.rglob('*.parquet')):
            s=p.stat(); values.append([kind,str(p.relative_to(root)),s.st_size,s.st_mtime_ns])
    return hashlib.sha256(json.dumps(values,sort_keys=True).encode()).hexdigest()


def discover():
    for parent in [Path.cwd(),Path.cwd()/'working_13-10']:
        r=parent/'input'
        if (r/'moocubex_parquet/catalog.json').exists() and (r/'moocubex_model/catalog.json').exists():
            return r/'moocubex_parquet',r/'moocubex_model',parent/'review_time_phases'
    found_raw,found_model,found_times=[],[],[]
    for parent in [Path('/kaggle/input'),Path('/kaggle/working')]:
        if parent.is_dir():
            for p in parent.rglob('catalog.json'):
                if (p.parent/'tables/relations/user_video').is_dir(): found_raw.append(p.parent)
                if (p.parent/'processed/dim_lo_catalog').is_dir(): found_model.append(p.parent)
            for p in parent.rglob('user_course_phase_times.parquet'):
                found_times.append(p.parent.parent)
    if len(found_raw)==len(found_model)==len(found_times)==1:
        return found_raw[0],found_model[0],found_times[0]
    raise RuntimeError('Set raw_root, model_root and time_root explicitly.')


def open_db(out,memory_limit='2GB',threads=2):
    scratch=Path(out)/'_scratch';scratch.mkdir(parents=True,exist_ok=True)
    con=duckdb.connect();con.execute(f'SET threads={int(threads)}')
    con.execute(f'SET memory_limit={qs(memory_limit)}')
    con.execute(f'SET temp_directory={qs(scratch)}')
    con.execute("SET max_temp_directory_size='4GB'")
    con.execute("SET TimeZone='UTC'")
    con.execute('SET preserve_insertion_order=false')
    return con


def dictionaries(con,sql):
    cursor=con.execute(sql);names=[c[0] for c in cursor.description]
    return [dict(zip(names,row)) for row in cursor.fetchall()]


def fixture_checks():
    con=duckdb.connect()
    try:
        # Retain a real partial score; use binary fallback only when score is invalid.
        rows=con.execute("""WITH p(score,max_score,is_correct,attempts) AS (VALUES
            (0.5,1.0,0,2),(NULL,2.0,1,2),(-1.0,2.0,0,3),(NULL,NULL,1,1))
            SELECT CASE WHEN score>=0 AND score<=max_score AND max_score>0 THEN score
                        WHEN max_score>0 THEN is_correct*max_score END earned,
                   is_correct::DOUBLE/attempts efficiency FROM p""").fetchall()
        assert rows==[(0.5,0.0),(2.0,0.5),(0.0,0.0),(None,1.0)]
        assert con.execute('SELECT list_avg([NULL::DOUBLE,0.5]),list_count([NULL::DOUBLE,0.5])').fetchone()==(0.5,1)
        # Average distances after chapter-level calculation, not distance of averages.
        q=(0.0+1.0)/2
        vec_distance=math.sqrt(3*(1-0.5)**2)/math.sqrt(3)
        assert abs(q-vec_distance)<1e-12
        # Non-collinear chapter vectors show the aggregation difference.
        q=(1.0/math.sqrt(3)+math.sqrt(2)/math.sqrt(3))/2
        vec_distance=math.sqrt(3*(1-0.5)**2)/math.sqrt(3)
        assert abs(q-vec_distance)>0.1
        assert con.execute("SELECT count(*) FROM (VALUES (TIMESTAMP '2020-05-01'),(TIMESTAMP '2020-05-03'),(TIMESTAMP '2020-05-05')) t(event_time) WHERE event_time BETWEEN TIMESTAMP '2020-05-02' AND TIMESTAMP '2020-05-04'").fetchone()[0]==1
        return {'partial_score_preserved':True,'binary_fallback_only_when_needed':True,
                'missing_max_score_stays_null':True,'mean_non_null_semantics':True,
                'chapter_distance_before_aggregation':True,'final_window_filter':True}
    finally:
        con.close()


def run_build(raw_root=None,model_root=None,time_root=None,output_root=None,memory_limit='2GB',threads=2):
    if raw_root is None or model_root is None or time_root is None:
        a,b,c=discover();raw_root=raw_root or a;model_root=model_root or b;time_root=time_root or c
    raw,model,times=map(lambda p:Path(p).resolve(),(raw_root,model_root,time_root))
    out=Path(output_root or ('/kaggle/working/final_quality_tables' if Path('/kaggle').is_dir()
                            else raw.parent.parent/'final_quality_tables')).resolve()
    for r in [raw,model,times]:
        if r==out or r in out.parents or out in r.parents: raise ValueError('Output must be separate from inputs.')
    out.mkdir(parents=True,exist_ok=True)
    tables=out/'tables';tables.mkdir(exist_ok=True)
    stage=out/'_stage';stage.mkdir(exist_ok=True)
    success=out/'_SUCCESS.json'
    if success.exists():success.unlink()
    con=open_db(out,memory_limit,threads)
    base_fp=fingerprint(raw,model)
    time_hash=sha256(times/'run_manifest.json')
    run_signature=hashlib.sha256((base_fp+time_hash+POLICY_VERSION).encode()).hexdigest()
    started=time.monotonic();log=[];lineage=[]
    cache_path=out/'step_checkpoints.json'
    cache=json.loads(cache_path.read_text()) if cache_path.exists() else {}

    def view(name,path):
        identifier='"'+name.replace('"','""')+'"'
        con.execute(f'CREATE OR REPLACE VIEW {identifier} AS SELECT * FROM read_parquet({qs(path)},union_by_name=true)')

    def save(name,sql,folder=stage):
        p=folder/f'{name}.parquet';tick=time.monotonic()
        signature=hashlib.sha256((run_signature+''.join(lineage)+sql).encode()).hexdigest()
        reused=p.exists() and cache.get(str(p.relative_to(out)))==signature
        if not reused:
            partial=p.with_suffix('.parquet.partial')
            con.execute(f'COPY ({sql}) TO {qs(partial)} (FORMAT PARQUET,COMPRESSION ZSTD,ROW_GROUP_SIZE 122880)')
            partial.replace(p);cache[str(p.relative_to(out))]=signature
            cache_path.write_text(json.dumps(cache,indent=2))
        view(name,p)
        lineage.append(signature)
        log.append({'name':name,'sql':sql,'seconds':round(time.monotonic()-tick,3),'cached':reused})
        print(name,'cached' if reused else 'built',log[-1]['seconds'],'seconds',flush=True)
        return p

    for name,relative in [('lo_catalog','dim_lo_catalog'),('resources','dim_lo_resource'),('lo_problem','bridge_lo_problem'),
                           ('course_comment','bridge_course_comment')]:
        view(name,model/'processed'/relative/'*.parquet')
    view('time_pairs',times/'tables/user_course_phase_times.parquet')
    view('course_bounds',times/'tables/course_time_bounds.parquet')
    for name,relative in [('uv','relations/user_video'),('up','relations/user_problem'),
                           ('comments','entities/translated_comment'),('replies','entities/reply'),('comment_reply','relations/comment_reply')]:
        view(name,raw/'tables'/relative/'*.parquet')
    con.execute('''CREATE TEMP TABLE resource_map AS SELECT resource_id,min(course_id) course_id,min(lo_key) lo_key,
        min(resource_type) resource_type FROM resources GROUP BY 1 HAVING count(DISTINCT lo_key)=1''')
    con.execute('''CREATE TEMP TABLE problem_map AS SELECT problem_id,min(course_id) course_id,min(lo_key) lo_key,
        min(exercise_id) exercise_id,max(problem_max_score) problem_max_score,count(DISTINCT exercise_id) n_exercises
        FROM lo_problem GROUP BY 1 HAVING count(DISTINCT lo_key)=1''')
    con.execute('''CREATE TEMP TABLE comment_course AS SELECT comment_id,min(course_id) course_id,
        count(DISTINCT course_id) n_courses FROM course_comment GROUP BY 1''')
    save('course_splits', '''WITH eligible_courses AS (
        SELECT DISTINCT course_id FROM time_pairs WHERE time_eligible
      ), ranked AS (SELECT b.*,row_number() OVER(ORDER BY course_start,course_id) pos,
          count(*) OVER() n_courses FROM course_bounds b JOIN eligible_courses e USING(course_id))
      SELECT course_id,course_start,course_end,
        CASE WHEN pos<=floor(n_courses*0.7) THEN 'train'
             WHEN pos<=floor(n_courses*0.8) THEN 'dev' ELSE 'test' END split
      FROM ranked''',tables)
    split_rows=dictionaries(con,'SELECT course_id,split FROM course_splits ORDER BY course_id')
    (out/'splits.json').write_text(json.dumps({r['course_id']:r['split'] for r in split_rows},indent=2))

    # Reuse the already validated mapped row stage, not its full-window aggregates.
    prior_dir=raw.parent.parent/'eda_engagement_features_comparison'
    prior_stage=prior_dir/'_stage/problem_rows'
    prior_manifest=prior_dir/'run_manifest.json'
    use_prior=prior_manifest.exists() and prior_stage.exists() and json.loads(prior_manifest.read_text()).get('input_fingerprint')==base_fp
    if use_prior:
        problem_stage=prior_stage
        print('Reuse verified mapped problem rows; re-filter each row by enrollment/end.',flush=True)
    else:
        problem_stage=stage/'problem_rows';problem_stage.mkdir(exist_ok=True)
        done=problem_stage/'_SUCCESS.json'
        if not done.exists() or json.loads(done.read_text()).get('signature')!=run_signature:
            # Only our own stage is replaced on a changed input snapshot.
            import shutil
            shutil.rmtree(problem_stage);problem_stage.mkdir()
            con.execute(f'''COPY (SELECT u.*,m.course_id,m.lo_key,m.exercise_id,m.problem_max_score,m.n_exercises,
                try_cast(u.submit_time AS TIMESTAMP) event_time,
                coalesce(isfinite(u.score) AND u.score>=0 AND m.problem_max_score>0 AND u.score<=m.problem_max_score+1e-9,false) score_valid,
                coalesce(u.is_correct IN (0,1) AND m.problem_max_score>0 AND isfinite(m.problem_max_score),false) fallback_possible,
                coalesce(u.attempts>0 AND u.is_correct IN (0,1),false) efficiency_valid,hash(u.user_id)%32 user_shard
                FROM up u JOIN problem_map m USING(problem_id)) TO {qs(problem_stage)}
                (FORMAT PARQUET,COMPRESSION ZSTD,PARTITION_BY(user_shard),ROW_GROUP_SIZE 50000)''')
            done.write_text(json.dumps({'signature':run_signature}))
    view('mapped_problems',problem_stage/'**/*.parquet')

    save('video_events', '''WITH items AS (SELECT user_id,unnest(seq) item FROM uv),
        segments AS (SELECT user_id,m.course_id,m.lo_key,item.video_id,unnest(item.segment) s
          FROM items JOIN resource_map m ON item.video_id=m.resource_id AND m.resource_type='video')
      SELECT user_id,course_id,lo_key,video_id,s.start_point,s.end_point,s.speed,
        cast(to_timestamp(s.local_start_time) AS TIMESTAMP) event_time,
        coalesce(s.start_point>=0 AND s.end_point>s.start_point AND isfinite(s.start_point)
          AND isfinite(s.end_point) AND s.speed>0 AND isfinite(s.speed),false) valid_measurement
      FROM segments WHERE s.local_start_time>=epoch(TIMESTAMP '2020-01-19')
        AND s.local_start_time<epoch(TIMESTAMP '2020-11-04')''')
    save('comment_prepared', '''WITH typed AS (
        SELECT c.id comment_id,'U_'||cast(c.user_id AS VARCHAR) user_id,c.resource_id,c.text,
          cast(to_timestamp(c.create_time/1000.0) AS TIMESTAMP) event_time,r.course_id,r.lo_key,
          nullif(regexp_replace(trim(c.text),'\\s+',' ','g'),'') normalized_text
        FROM comments c JOIN resource_map r ON c.resource_id=r.resource_id
        LEFT JOIN comment_course cc ON c.id=cc.comment_id
        WHERE c.create_time BETWEEN 0 AND 4102444800000 AND c.user_id IS NOT NULL
          AND (cc.comment_id IS NULL OR (cc.n_courses=1 AND cc.course_id=r.course_id))
      ), ids AS (SELECT *,row_number() OVER(PARTITION BY comment_id ORDER BY event_time,user_id,resource_id,text) id_rank FROM typed)
      SELECT * EXCLUDE(id_rank),row_number() OVER(PARTITION BY user_id,resource_id,event_time,normalized_text ORDER BY comment_id) exact_duplicate_rank
      FROM ids WHERE id_rank=1''')
    con.execute('''CREATE TEMP TABLE reply_parent AS SELECT reply_id,min(comment_id) parent_comment_id
        FROM comment_reply GROUP BY 1 HAVING count(DISTINCT comment_id)=1''')
    save('reply_prepared', '''WITH typed AS (
        SELECT r.id reply_id,r.user_id,try_cast(r.create_time AS TIMESTAMP) event_time,
          p.parent_comment_id,c.event_time parent_time,c.course_id,c.lo_key,
          row_number() OVER(PARTITION BY r.id ORDER BY r.create_time,r.user_id,r.text) id_rank
        FROM replies r JOIN reply_parent p ON r.id=p.reply_id
        JOIN comment_prepared c ON p.parent_comment_id=c.comment_id)
      SELECT * EXCLUDE(id_rank) FROM typed WHERE id_rank=1 AND event_time>=parent_time''')
    save('active_candidates', f'''SELECT {KEYS.split(',')[0]},course_id FROM mapped_problems GROUP BY 1,2
        UNION SELECT user_id,course_id FROM video_events
        UNION SELECT user_id,course_id FROM comment_prepared
        UNION SELECT user_id,course_id FROM reply_prepared''')
    save('active_pairs', '''SELECT t.user_id,t.course_id,t.enroll_time,t.course_end,s.split
        FROM time_pairs t JOIN active_candidates a USING(user_id,course_id)
        JOIN course_splits s USING(course_id) WHERE t.time_eligible''')
    save('video_window', '''SELECT v.* FROM video_events v JOIN active_pairs p USING(user_id,course_id)
        WHERE v.event_time BETWEEN p.enroll_time AND p.course_end''')
    save('video_union', '''WITH intervals AS (
        SELECT DISTINCT user_id,course_id,lo_key,video_id,start_point,end_point FROM video_window WHERE valid_measurement
      ), previous AS (SELECT *,max(end_point) OVER(PARTITION BY user_id,course_id,lo_key,video_id
          ORDER BY start_point,end_point ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) prev_end FROM intervals),
      numbered AS (SELECT *,sum(CASE WHEN prev_end IS NULL OR start_point>prev_end THEN 1 ELSE 0 END)
          OVER(PARTITION BY user_id,course_id,lo_key,video_id ORDER BY start_point,end_point) island FROM previous)
      SELECT user_id,course_id,lo_key,video_id,island,min(start_point) interval_start,max(end_point) interval_end
      FROM numbered GROUP BY 1,2,3,4,5''')
    save('video_features', '''WITH counts AS (
        SELECT user_id,course_id,lo_key,count(*) FILTER(WHERE valid_measurement) n_video_segments,
          count(*) FILTER(WHERE NOT valid_measurement) n_invalid_video_segments,
          count(DISTINCT video_id) FILTER(WHERE valid_measurement) n_videos_watched,
          coalesce(sum(end_point-start_point) FILTER(WHERE valid_measurement),0) raw_watch_seconds,
          min(event_time) first_video_time,max(event_time) last_video_time FROM video_window GROUP BY 1,2,3
      ), unique_seconds AS (SELECT user_id,course_id,lo_key,sum(interval_end-interval_start) unique_watch_seconds
        FROM video_union GROUP BY 1,2,3), edges AS (
        SELECT *,lag(event_time) OVER(PARTITION BY user_id,course_id,lo_key,video_id ORDER BY event_time,start_point,end_point) prev_time
        FROM video_window WHERE valid_measurement), sessions AS (
        SELECT user_id,course_id,lo_key,video_id,count(*) FILTER(WHERE prev_time IS NULL OR event_time-prev_time>INTERVAL '30 minutes') n_sessions
        FROM edges GROUP BY 1,2,3,4), session_stats AS (
        SELECT user_id,course_id,lo_key,sum(n_sessions) n_video_sessions_30m,count(*) FILTER(WHERE n_sessions>1) n_revisited_videos
        FROM sessions GROUP BY 1,2,3)
      SELECT c.*,coalesce(u.unique_watch_seconds,0) unique_watch_seconds,
        c.raw_watch_seconds-coalesce(u.unique_watch_seconds,0) rewatch_seconds,
        coalesce(s.n_video_sessions_30m,0) n_video_sessions_30m,coalesce(s.n_revisited_videos,0) n_revisited_videos
      FROM counts c LEFT JOIN unique_seconds u USING(user_id,course_id,lo_key)
      LEFT JOIN session_stats s USING(user_id,course_id,lo_key)''')

    # Aggregate by user-hash shards to bound the distinct user/problem memory use.
    problem_feature_dir=stage/'problem_feature_shards';problem_feature_dir.mkdir(exist_ok=True)
    problem_agg='''SELECT p.user_id,p.course_id,p.lo_key,count(*) n_problems_attempted,
        count(DISTINCT p.exercise_id) n_exercises_attempted,sum(p.attempts) total_problem_attempts,
        count(*) FILTER(WHERE p.attempts IS NULL OR p.attempts<=0) invalid_attempts,
        count(*) FILTER(WHERE p.is_correct=1) n_problems_correct,
        count(*) FILTER(WHERE p.is_correct=1 AND p.attempts=1) n_first_try_correct,
        coalesce(sum(CASE WHEN p.score_valid THEN p.score WHEN p.fallback_possible THEN p.is_correct*p.problem_max_score END),0) earned_score_sum,
        coalesce(sum(p.problem_max_score) FILTER(WHERE p.problem_max_score>0),0) attempted_max_score_sum,
        coalesce(sum(CASE WHEN p.efficiency_valid THEN p.is_correct::DOUBLE/p.attempts END),0) efficiency_sum,
        count(*) FILTER(WHERE p.score_valid) n_scores_valid,
        count(*) FILTER(WHERE NOT p.score_valid AND p.fallback_possible) n_scores_fallback,
        count(*) FILTER(WHERE NOT p.score_valid AND NOT p.fallback_possible) n_scores_unresolved,
        count(*) FILTER(WHERE NOT p.efficiency_valid) n_efficiency_unresolved,
        min(p.event_time) first_problem_time,max(p.event_time) last_problem_time
        FROM {source} p JOIN active_pairs a USING(user_id,course_id)
        WHERE p.event_time BETWEEN a.enroll_time AND a.course_end GROUP BY 1,2,3'''
    for shard in range(32):
        source=f'read_parquet({qs(problem_stage/f"user_shard={shard}"/"*.parquet")})'
        save(f'part-{shard:03d}',problem_agg.format(source=source),problem_feature_dir)
    view('problem_features',problem_feature_dir/'*.parquet')
    save('comment_window', '''SELECT c.*,p.enroll_time,p.course_end FROM comment_prepared c
        JOIN active_pairs p USING(user_id,course_id) WHERE c.event_time BETWEEN p.enroll_time AND p.course_end''')
    save('afelo_comment_links', '''SELECT * FROM comment_window
        WHERE normalized_text IS NOT NULL AND exact_duplicate_rank=1''',tables)
    save('social_features', '''WITH received AS (
        SELECT DISTINCT c.comment_id FROM afelo_comment_links c JOIN reply_prepared r
          ON c.comment_id=r.parent_comment_id AND r.event_time>=c.event_time AND r.event_time<=c.course_end
      ), counts AS (SELECT c.user_id,c.course_id,c.lo_key,count(*) n_comments_raw,
          count(*) FILTER(WHERE c.normalized_text IS NOT NULL AND c.exact_duplicate_rank=1) n_comments_filtered,
          count(*) FILTER(WHERE c.normalized_text IS NULL) n_comments_empty,
          count(*) FILTER(WHERE c.exact_duplicate_rank>1) n_comments_duplicate,
          count(*) FILTER(WHERE x.comment_id IS NOT NULL AND c.normalized_text IS NOT NULL AND c.exact_duplicate_rank=1) n_user_comments_with_reply,
          min(c.event_time) first_comment_time,max(c.event_time) last_comment_time
        FROM comment_window c LEFT JOIN received x USING(comment_id) GROUP BY 1,2,3),
      reply_counts AS (SELECT r.user_id,r.course_id,r.lo_key,count(*) n_replies_authored,
          min(r.event_time) first_reply_time,max(r.event_time) last_reply_time
        FROM reply_prepared r JOIN active_pairs p USING(user_id,course_id)
        WHERE r.event_time BETWEEN p.enroll_time AND p.course_end GROUP BY 1,2,3)
      SELECT coalesce(c.user_id,r.user_id) user_id,coalesce(c.course_id,r.course_id) course_id,
        coalesce(c.lo_key,r.lo_key) lo_key,c.* EXCLUDE(user_id,course_id,lo_key),
        r.* EXCLUDE(user_id,course_id,lo_key) FROM counts c FULL JOIN reply_counts r USING(user_id,course_id,lo_key)''')
    video_fields=['n_videos_watched','n_video_segments','n_invalid_video_segments','raw_watch_seconds',
                  'unique_watch_seconds','rewatch_seconds','n_video_sessions_30m','n_revisited_videos']
    problem_fields=[x for x in COUNT_FIELDS if x not in video_fields+AFE_INPUTS]
    select_counts=','.join(f'coalesce({alias}.{field},0) {field}' for alias,fields in [('v',video_fields),('p',problem_fields),('s',AFE_INPUTS)] for field in fields)
    save('chapter_inputs',f'''WITH keys AS (
        SELECT {KEYS} FROM video_features UNION SELECT {KEYS} FROM problem_features UNION SELECT {KEYS} FROM social_features)
      SELECT k.*,d.* EXCLUDE(course_id,lo_key),a.enroll_time,a.course_end,a.split,
        v.user_id IS NOT NULL has_video,p.user_id IS NOT NULL has_problem,s.user_id IS NOT NULL has_social,
        {select_counts},least(v.first_video_time,p.first_problem_time,s.first_comment_time,s.first_reply_time) first_observed_event,
        greatest(v.last_video_time,p.last_problem_time,s.last_comment_time,s.last_reply_time) last_observed_event
      FROM keys k JOIN lo_catalog d USING(course_id,lo_key) JOIN active_pairs a USING(user_id,course_id)
      LEFT JOIN video_features v USING(user_id,course_id,lo_key) LEFT JOIN problem_features p USING(user_id,course_id,lo_key)
      LEFT JOIN social_features s USING(user_id,course_id,lo_key)''')
    save('chapter_numeric', '''SELECT *,
        unique_watch_seconds/nullif(available_video_seconds_proxy,0) TS_partial_catalog,
        CASE WHEN n_videos_missing_duration=0 AND available_video_seconds_proxy>0
          THEN least(1.0,greatest(0.0,unique_watch_seconds/available_video_seconds_proxy)) END TS,
        n_videos_watched::DOUBLE/nullif(n_videos_available,0) RV,
        n_exercises_attempted::DOUBLE/nullif(n_exercises_available,0) AV,
        CASE WHEN invalid_attempts=0 THEN total_problem_attempts::DOUBLE/nullif(n_problems_available,0) END IF_raw,
        CASE WHEN n_problems_missing_score=0 AND n_scores_unresolved=0 THEN earned_score_sum/nullif(available_max_score_sum,0) END KL,
        CASE WHEN n_efficiency_unresolved=0 THEN efficiency_sum/nullif(n_problems_available,0) END CFA,
        earned_score_sum/nullif(attempted_max_score_sum,0) mastery_attempted_partial,
        n_problems_attempted::DOUBLE/nullif(n_problems_available,0) problem_participation
      FROM chapter_inputs''')
    p95=con.execute("SELECT quantile_cont(IF_raw,0.95) FROM chapter_numeric WHERE split='train' AND IF_raw IS NOT NULL").fetchone()[0]
    if p95 is None or not math.isfinite(p95) or p95<=0:
        raise RuntimeError('Train-only P95_IF_raw must be finite and positive; inspect training population.')
    constants={'formula_version':FORMULA_VERSION,'P95_IF_raw_training':p95,
        'reference':'one final-window row per observed user-course-chapter in train only',
        'coelo_min_components':2,'acelo_min_components':1,'unobserved_chapters':'exclude; do not assign LOQV=1',
        'q_requires_all_observed_chapters_scored':True,'split_method':'70/10/20 courses ordered by observed course_start, then course_id'}
    (out/'protocol_constants.json').write_text(json.dumps(constants,ensure_ascii=False,indent=2))
    save('coelo_components',f'''WITH normalized AS (
        SELECT *,CASE WHEN IF_raw IS NOT NULL THEN least(1.0,greatest(0.0,ln(1+IF_raw)/ln(1+{repr(p95)}))) END IF_value FROM chapter_numeric),
      counted AS (SELECT *,list_count([TS,RV,AV,IF_value]) coelo_n_components FROM normalized)
      SELECT {META},{','.join(COE_INPUTS)},TS_partial_catalog,TS,RV,AV,IF_raw,IF_value AS "IF",
        coelo_n_components,CASE WHEN coelo_n_components>=2 THEN list_avg([TS,RV,AV,IF_value]) END COELO,
        'caption_proxy' duration_source,{qs(FORMULA_VERSION)} formula_version FROM counted''',tables)
    save('acelo_components',f'''SELECT {META},{','.join(ACE_INPUTS)},KL,CFA,
        list_count([KL,CFA]) acelo_n_components,list_avg([KL,CFA]) ACELO,
        mastery_attempted_partial,problem_participation,
        CASE WHEN KL IS NOT NULL AND CFA IS NOT NULL THEN 'KL_AND_CFA'
             WHEN CFA IS NOT NULL THEN 'CFA_ONLY' WHEN KL IS NOT NULL THEN 'KL_ONLY' ELSE 'UNAVAILABLE' END acelo_status,
        {qs(FORMULA_VERSION)} formula_version FROM chapter_numeric''',tables)
    # Base provenance is written before finalization, so AFELO can be filled later.
    checks={'fixtures':fixture_checks(),
        'chapter_inputs_unique':con.execute(f'SELECT count(*)=count(DISTINCT ({KEYS})) FROM chapter_inputs').fetchone()[0],
        'chapter_events_inside_final_window':con.execute('SELECT count(*)=0 FROM chapter_inputs WHERE first_observed_event<enroll_time OR last_observed_event>course_end').fetchone()[0],
        'source_tables_same_count':con.execute('SELECT (SELECT count(*) FROM chapter_inputs)=(SELECT count(*) FROM coelo_components) AND (SELECT count(*) FROM chapter_inputs)=(SELECT count(*) FROM acelo_components)').fetchone()[0],
        'all_observed_chapter_keys_preserved':con.execute(f'''SELECT count(*)=0 FROM (
            SELECT {KEYS} FROM video_features UNION SELECT {KEYS} FROM problem_features
            UNION SELECT {KEYS} FROM social_features) k ANTI JOIN chapter_inputs b USING(user_id,course_id,lo_key)''').fetchone()[0],
        'video_union_not_greater_than_raw':con.execute('SELECT count(*)=0 FROM chapter_inputs WHERE unique_watch_seconds>raw_watch_seconds+1e-8').fetchone()[0],
        'received_reply_bounded':con.execute('SELECT count(*)=0 FROM chapter_inputs WHERE n_user_comments_with_reply>n_comments_filtered').fetchone()[0],
        'component_values_bounded':con.execute('SELECT count(*)=0 FROM chapter_numeric WHERE RV>1+1e-9 OR AV>1+1e-9 OR KL>1+1e-9 OR CFA>1+1e-9 OR KL<0 OR CFA<0').fetchone()[0],
    }
    assert all(v for k,v in checks.items() if k!='fixtures'),checks
    manifest={'created_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'duckdb_version':duckdb.__version__,
        'raw_root':str(raw),'model_root':str(model),'time_root':str(times),'input_fingerprint':base_fp,
        'time_manifest_sha256':time_hash,'reused_validated_problem_stage':use_prior,'sampling':'none',
        'elapsed_seconds':round(time.monotonic()-started,3),'queries':log,'base_checks':checks,
        'policies':{'window':'inclusive enroll_time to course_end; all sources filtered before aggregation',
          'missing_chapters':'not materialized; unobserved, not LOQV=1',
          'video':'valid timestamp in paper window; invalid intervals counted, excluded from watch metrics; union intervals',
          'problem':'user-problem aggregate assigned at submit_time; no invented individual attempt times',
          'social_filter':'remove empty text and exact same user/resource/time/normalized-text duplicates; no short-text length threshold',
          'labels':'AFELO deferred by user; no fabricated q/y; all observed chapters must be scorable before final q',
          'roles':'final-window label/audit tables only; not P1-P4 model features'}}
    (out/'run_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2,default=str))
    con.close()
    return finalize_quality(out,memory_limit=memory_limit,threads=threads)


def finalize_quality(output_root,sentiment_path=None,afelo_mode='deferred',memory_limit='2GB',threads=2):
    """Fill AFELO and labels from persisted base tables, without raw data scans.

    sentiment input parquet: comment_id,p_pos,p_neu,p_neg. The optional
    social_only mode is explicit; default deferred keeps AFELO/LOQV/q/y null.
    """
    if afelo_mode not in ['deferred','sentiment','social_only']:
        raise ValueError('afelo_mode must be deferred, sentiment or social_only')
    if afelo_mode=='sentiment' and sentiment_path is None:
        raise ValueError('Provide sentiment_path with comment_id,p_pos,p_neu,p_neg.')
    out=Path(output_root).resolve();tables=out/'tables';stage=out/'_stage'
    con=open_db(out,memory_limit,threads)
    manifest=json.loads((out/'run_manifest.json').read_text());times=Path(manifest['time_root'])
    def view(name,path):
        identifier='"'+name.replace('"','""')+'"'
        con.execute(f'CREATE OR REPLACE VIEW {identifier} AS SELECT * FROM read_parquet({qs(path)})')
    def save(name,sql,folder=tables):
        p=folder/f'{name}.parquet';partial=p.with_suffix('.parquet.partial')
        con.execute(f'COPY ({sql}) TO {qs(partial)} (FORMAT PARQUET,COMPRESSION ZSTD,ROW_GROUP_SIZE 122880)')
        partial.replace(p);view(name,p);print(name,'built',flush=True)
        return p
    for name in ['chapter_numeric','chapter_inputs']:view(name,stage/f'{name}.parquet')
    for name in ['coelo_components','acelo_components','afelo_comment_links','course_splits']:view(name,tables/f'{name}.parquet')
    view('time_pairs',times/'tables/user_course_phase_times.parquet')
    view('lo_catalog',Path(manifest['model_root'])/'processed/dim_lo_catalog/*.parquet')
    if afelo_mode=='sentiment':
        view('sentiment',Path(sentiment_path).resolve())
        invalid=con.execute('''SELECT count(*) FROM sentiment WHERE comment_id IS NULL OR p_pos IS NULL OR p_neu IS NULL OR p_neg IS NULL
            OR NOT isfinite(p_pos) OR NOT isfinite(p_neu) OR NOT isfinite(p_neg)
            OR p_pos<0 OR p_neu<0 OR p_neg<0 OR p_pos>1 OR p_neu>1 OR p_neg>1 OR abs(p_pos+p_neu+p_neg-1)>1e-6''').fetchone()[0]
        assert invalid==0,'Invalid sentiment probability rows'
        assert con.execute('SELECT count(*)=count(DISTINCT comment_id) FROM sentiment').fetchone()[0],'Duplicate sentiment comment IDs'
        con.execute('''CREATE TEMP VIEW sentiment_summary AS SELECT c.user_id,c.course_id,c.lo_key,
            count(*) n_sentiment_comments,sum(s.p_pos) sentiment_pos_sum,sum(s.p_neu) sentiment_neu_sum,sum(s.p_neg) sentiment_neg_sum
            FROM afelo_comment_links c JOIN sentiment s USING(comment_id) GROUP BY 1,2,3''')
    else:
        con.execute('''CREATE TEMP VIEW sentiment_summary AS SELECT NULL::VARCHAR user_id,NULL::VARCHAR course_id,NULL::VARCHAR lo_key,
            0::BIGINT n_sentiment_comments,NULL::DOUBLE sentiment_pos_sum,NULL::DOUBLE sentiment_neu_sum,NULL::DOUBLE sentiment_neg_sum WHERE false''')
    success=out/'_SUCCESS.json'
    if success.exists():success.unlink()
    score_expr='NULL::DOUBLE' if afelo_mode=='deferred' else 'list_avg([comment_participation,reply_participation,response_received_rate,sentiment_valence])'
    save('afelo_components',f'''WITH inputs AS (
        SELECT b.*,coalesce(s.n_sentiment_comments,0) n_sentiment_comments,
          s.sentiment_pos_sum,s.sentiment_neu_sum,s.sentiment_neg_sum,
          (b.n_comments_filtered>0)::DOUBLE comment_participation,(b.n_replies_authored>0)::DOUBLE reply_participation,
          b.n_user_comments_with_reply::DOUBLE/nullif(b.n_comments_filtered,0) response_received_rate,
          CASE WHEN s.n_sentiment_comments>0 THEN ((s.sentiment_pos_sum-s.sentiment_neg_sum)/s.n_sentiment_comments+1)/2 END sentiment_valence
        FROM chapter_numeric b LEFT JOIN sentiment_summary s USING(user_id,course_id,lo_key))
      SELECT {META},{','.join(AFE_INPUTS)},n_sentiment_comments,sentiment_pos_sum,sentiment_neu_sum,sentiment_neg_sum,
        comment_participation,reply_participation,response_received_rate,sentiment_valence,
        n_sentiment_comments::DOUBLE/nullif(n_comments_raw,0) afe_text_coverage,
        list_avg([comment_participation,reply_participation,response_received_rate]) AFELO_social_proxy,
        {score_expr} AFELO_proxy,
        list_count([comment_participation,reply_participation,response_received_rate,sentiment_valence]) afelo_n_components,
        {qs(afelo_mode)} afelo_status,{qs(FORMULA_VERSION)} formula_version FROM inputs''')
    save('chapter_quality',f'''WITH joined AS (
        SELECT b.*,c.TS_partial_catalog,c.TS,c.RV,c.AV,c.IF_raw,c."IF",c.COElO,c.coelo_n_components,
          a.comment_participation,a.reply_participation,a.response_received_rate,a.sentiment_valence,
          a.n_sentiment_comments,a.sentiment_pos_sum,a.sentiment_neu_sum,a.sentiment_neg_sum,a.afe_text_coverage,
          a.AFELO_social_proxy,a.AFELO_proxy,a.afelo_n_components,a.afelo_status,
          x.KL,x.CFA,x.mastery_attempted_partial,x.problem_participation,x.acelo_n_components,x.acelo_status,x.ACELO,
          c.user_id IS NOT NULL coelo_row_found,a.user_id IS NOT NULL afelo_row_found,x.user_id IS NOT NULL acelo_row_found
        FROM chapter_inputs b LEFT JOIN coelo_components c USING(user_id,course_id,lo_key)
        LEFT JOIN afelo_components a USING(user_id,course_id,lo_key) LEFT JOIN acelo_components x USING(user_id,course_id,lo_key))
      SELECT *,CASE WHEN COELO IS NOT NULL AND AFELO_proxy IS NOT NULL AND ACELO IS NOT NULL
        THEN sqrt(pow(1-COELO,2)+pow(1-AFELO_proxy,2)+pow(1-ACELO,2))/sqrt(3.0) END LOQV,
        CASE WHEN NOT coelo_row_found OR NOT afelo_row_found OR NOT acelo_row_found THEN 'MISSING_COMPONENT_ROW'
             WHEN {qs(afelo_mode)}='deferred' THEN 'PENDING_AFELO'
             WHEN COELO IS NULL OR AFELO_proxy IS NULL OR ACELO IS NULL THEN 'INCOMPLETE_COMPONENTS' ELSE 'SCORABLE' END chapter_status,
        {qs(FORMULA_VERSION)} formula_version FROM joined''')
    # Keep a nested list of all per-chapter fields as well as scalar aggregates.
    summands=','.join(f'sum({name}) {name}_sum_observed' for name in COUNT_FIELDS)
    diagnostic_means=','.join(f'avg({name}) {name}_mean_chapter' for name in ['TS','RV','AV','IF_raw','"IF"','KL','CFA','mastery_attempted_partial','problem_participation']
        ).replace('"IF"_mean_chapter','IF_mean_chapter')
    aggregate_sql=f'''SELECT user_id,course_id,count(*) n_chapters_observed,
        count(LOQV) n_chapters_scored,avg(COELO) coelo_mean_chapter,avg(AFELO_proxy) afelo_proxy_mean_chapter,
        avg(ACELO) acelo_mean_chapter,{diagnostic_means},{summands},
        CASE WHEN count(LOQV)=count(*) THEN avg(LOQV) END q,
        list(chapter_quality ORDER BY chapter_major) chapter_details
      FROM chapter_quality WHERE hash(user_id)%32={{shard}} GROUP BY user_id,course_id'''
    aggregate_dir=stage/'user_course_aggregates';aggregate_dir.mkdir(exist_ok=True)
    final_dir=tables/'user_course_final';final_dir.mkdir(exist_ok=True)
    # Nested per-chapter structs cannot spill efficiently in one global list()
    # aggregate. Bound both the aggregation and the final join by user shards.
    pair_dir=stage/'eligible_pair_shards';pair_done=pair_dir/'_SUCCESS.json'
    if not pair_done.exists() or json.loads(pair_done.read_text()).get('time_sha256')!=manifest['time_manifest_sha256']:
        import shutil
        if pair_dir.exists():shutil.rmtree(pair_dir)
        pair_dir.mkdir()
        con.execute('SET partitioned_write_max_open_files=8')
        con.execute(f'''COPY (SELECT user_id,course_id,enroll_time,course_end,hash(user_id)%32 user_shard
            FROM time_pairs WHERE time_eligible) TO {qs(pair_dir)}
            (FORMAT PARQUET,COMPRESSION ZSTD,PARTITION_BY(user_shard),ROW_GROUP_SIZE 50000)''')
        pair_done.write_text(json.dumps({'time_sha256':manifest['time_manifest_sha256']}))
    final_sql=f'''WITH chapter_counts AS (SELECT course_id,count(*) n_chapters_catalog FROM lo_catalog GROUP BY 1),
      joined AS (SELECT t.user_id,t.course_id,t.enroll_time,t.course_end,s.split,c.n_chapters_catalog,
        coalesce(a.n_chapters_observed,0) n_chapters_observed,coalesce(a.n_chapters_scored,0) n_chapters_scored,
        c.n_chapters_catalog-coalesce(a.n_chapters_observed,0) n_chapters_unobserved,
        a.* EXCLUDE(user_id,course_id,n_chapters_observed,n_chapters_scored)
        FROM pair_slice t LEFT JOIN course_splits s USING(course_id) LEFT JOIN chapter_counts c USING(course_id)
        LEFT JOIN aggregate_slice a USING(user_id,course_id))
      SELECT *,CASE WHEN q<0.2 THEN 'E' WHEN q<0.4 THEN 'G' WHEN q>=0.4 THEN 'A' END y,
        CASE WHEN n_chapters_observed=0 THEN 'NO_OBSERVED_CHAPTER'
             WHEN {qs(afelo_mode)}='deferred' THEN 'PENDING_AFELO'
             WHEN q IS NULL THEN 'INCOMPLETE_OBSERVED_CHAPTERS' ELSE 'LABELED_OBSERVED_CHAPTERS' END label_status,
        {qs(FORMULA_VERSION)} formula_version FROM joined'''
    for shard in range(32):
        aggregate_path=save(f'part-{shard:03d}',aggregate_sql.format(shard=shard),aggregate_dir)
        view('aggregate_slice',aggregate_path)
        pair_path=pair_dir/f'user_shard={shard}'
        if list(pair_path.glob('*.parquet')):
            view('pair_slice',pair_path/'*.parquet')
        else:
            con.execute('CREATE OR REPLACE VIEW pair_slice AS SELECT user_id,course_id,enroll_time,course_end FROM time_pairs WHERE false')
        save(f'part-{shard:03d}',final_sql,final_dir)
    view('user_course_aggregates',aggregate_dir/'*.parquet')
    view('user_course_final',final_dir/'*.parquet')
    summary={
        'chapter_rows':con.execute('SELECT count(*) FROM chapter_quality').fetchone()[0],
        'user_course_rows':con.execute('SELECT count(*) FROM user_course_final').fetchone()[0],
        'afelo_mode':afelo_mode,
        'chapter_status':dictionaries(con,'SELECT chapter_status,count(*) n_rows FROM chapter_quality GROUP BY 1'),
        'label_status':dictionaries(con,'SELECT label_status,count(*) n_rows FROM user_course_final GROUP BY 1 ORDER BY n_rows DESC'),
        'acelo_status':dictionaries(con,'SELECT acelo_status,count(*) n_rows FROM acelo_components GROUP BY 1'),
        'split_summary':dictionaries(con,'SELECT split,count(*) n_courses,min(course_start) first_start,max(course_start) last_start FROM course_splits GROUP BY 1 ORDER BY first_start'),
        'class_distribution':dictionaries(con,'SELECT split,y,count(*) n_rows FROM user_course_final WHERE y IS NOT NULL GROUP BY 1,2 ORDER BY 1,2'),
    }
    overlap=dictionaries(con,"""SELECT count(*) n_train_courses,count(*) FILTER(WHERE course_end>(SELECT min(course_start) FROM course_splits WHERE split='test')) n_train_ending_after_first_test_start
        FROM course_splits WHERE split='train'""")[0]
    summary['temporal_overlap']=overlap
    checks={
        'chapter_tables_unique':all(con.execute(f'SELECT count(*)=count(DISTINCT ({KEYS})) FROM {name}').fetchone()[0] for name in ['coelo_components','afelo_components','acelo_components','chapter_quality']),
        'chapter_joins_preserve_count':con.execute('SELECT count(*)=(SELECT count(*) FROM chapter_numeric) FROM chapter_quality').fetchone()[0],
        'final_user_course_unique':con.execute('SELECT count(*)=count(DISTINCT (user_id,course_id)) FROM user_course_final').fetchone()[0],
        'all_time_eligible_pairs_preserved':summary['user_course_rows']==con.execute('SELECT count(*) FROM time_pairs WHERE time_eligible').fetchone()[0],
        'final_rows_have_split_metadata':con.execute('SELECT count(*)=0 FROM user_course_final WHERE split IS NULL').fetchone()[0],
        'unobserved_users_not_assigned_zero_or_label':con.execute('SELECT count(*)=0 FROM user_course_final WHERE n_chapters_observed=0 AND (q IS NOT NULL OR y IS NOT NULL OR coelo_mean_chapter IS NOT NULL OR acelo_mean_chapter IS NOT NULL)').fetchone()[0],
        'nested_details_match_observed_chapter_count':con.execute('SELECT count(*)=0 FROM user_course_final WHERE coalesce(len(chapter_details),0)<>n_chapters_observed').fetchone()[0],
        'label_thresholds_correct':con.execute("SELECT count(*)=0 FROM user_course_final WHERE y IS DISTINCT FROM CASE WHEN q<0.2 THEN 'E' WHEN q<0.4 THEN 'G' WHEN q>=0.4 THEN 'A' END").fetchone()[0],
        'no_label_when_observed_chapter_incomplete':con.execute('SELECT count(*)=0 FROM user_course_final WHERE n_chapters_scored<n_chapters_observed AND q IS NOT NULL').fetchone()[0],
    }
    if afelo_mode=='deferred':
        checks['deferred_afelo_keeps_loqv_q_y_null']=con.execute('SELECT (SELECT count(*) FROM chapter_quality WHERE AFELO_proxy IS NOT NULL OR LOQV IS NOT NULL)=0 AND (SELECT count(*) FROM user_course_final WHERE q IS NOT NULL OR y IS NOT NULL)=0').fetchone()[0]
    for table in ['coelo_components','afelo_components','acelo_components']:
        checks[f'{table}_has_all_observed_keys']=con.execute(f'''SELECT count(*)=0
            FROM chapter_inputs b ANTI JOIN {table} c USING(user_id,course_id,lo_key)''').fetchone()[0]
    assert all(checks.values()),checks
    summary['validation']=checks
    manifest['finalization']={'afelo_mode':afelo_mode,'sentiment_path':str(sentiment_path) if sentiment_path else None,
        'sentiment_sha256':sha256(sentiment_path) if sentiment_path else None,'checks':checks}
    manifest['engine_source_sha256']=globals().get('ENGINE_SOURCE_SHA256') or sha256(__file__)
    (out/'run_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2,default=str))
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,default=str))
    (out/'validation.json').write_text(json.dumps({'base_checks':manifest['base_checks'],'final_checks':checks},ensure_ascii=False,indent=2))
    (out/'_SUCCESS.json').write_text(json.dumps({'afelo_mode':afelo_mode,'manifest_sha256':sha256(out/'run_manifest.json')},indent=2))
    write_report(out)
    con.close();print('COMPLETE',json.dumps(summary,ensure_ascii=False,default=str),flush=True)
    return summary


def write_report(out):
    out=Path(out);s=json.loads((out/'summary.json').read_text());c=json.loads((out/'protocol_constants.json').read_text())
    fmt=lambda n:f'{int(n):,}'.replace(',','.')
    pending=s['afelo_mode']=='deferred'
    intro=('Bản hiện tại hoàn thành COELO và ACELO theo schema mục 12. AFELO được hoãn theo yêu cầu; '
        'đã tính thành phần xã hội, giữ sentiment và AFELO hoàn chỉnh là null. Vì vậy LOQV, q và y đang chờ AFELO.'
        if pending else f'AFELO đã được hoàn thiện ở mode {s["afelo_mode"]}; LOQV và nhãn chỉ tính khi đủ điều kiện coverage.')
    lines=['# Các bảng chất lượng cuối kỳ theo user–course–chapter','',intro,'',
        f'- Số user–course–chapter quan sát được: {fmt(s["chapter_rows"])}.',
        f'- Số user–course ở bảng cuối: {fmt(s["user_course_rows"])}.',
        f'- P95_IF_raw fit trên train: {c["P95_IF_raw_training"]:.12g}.','',
        '| Bảng | Grain | Nội dung |','|---|---|---|',
        '| coelo_components.parquet | user–course–chapter | Counts, thời lượng, mẫu số; TS, RV, AV, IF; COELO |',
        '| afelo_components.parquet | user–course–chapter | Comment/reply, social proxy, sentiment, AFELO và trạng thái |',
        '| acelo_components.parquet | user–course–chapter | Điểm, attempts, provenance fallback; KL, CFA, ACELO |',
        '| chapter_quality.parquet | user–course–chapter | Toàn bộ feature con và chỉ số, LOQV và trạng thái |',
        '| user_course_final/*.parquet | user–course | Một bảng chia 32 file; feature tổng hợp, ba chỉ số trung bình, q/y và chapter_details |','',
        'Các file nằm trong `tables/`. Mọi user–course đủ điều kiện thời gian từ notebook 05 đều được giữ trong bảng cuối. '
        'Người không có chương quan sát được có chỉ số null và trạng thái NO_OBSERVED_CHAPTER.','',
        'Bảng chapter_quality lấy chapter_inputs làm bảng gốc, LEFT JOIN ba bảng thành phần theo user–course–lo_key. '
        'Bảng cuối lấy mọi cặp đủ điều kiện thời gian làm gốc rồi LEFT JOIN kết quả tổng hợp. '
        'Giá trị chỉ số null không làm mất dòng; các cờ *_row_found phân biệt thiếu bản ghi với có bản ghi nhưng thiếu chỉ số.','',
        '## Quy tắc tính','',
        '- Cửa sổ mỗi người: `[enroll_time, course_end]`. Mọi event được lọc trước khi aggregate.',
        '- Chương không có log không tạo dòng và không tự gán LOQV=1; đây là lựa chọn người dùng.',
        '- COELO = mean_non_null(TS, RV, AV, IF), tối thiểu 2 thành phần. TS dùng caption proxy đầy đủ; '
        'giữ TS_partial_catalog để kiểm toán. IF chuẩn hóa bằng P95 của các dòng user–chapter cuối kỳ train.',
        '- ACELO = mean_non_null(KL, CFA). KL chỉ có khi đủ mẫu số điểm của toàn chương và không còn score chưa giải được. '
        'CFA_ONLY được gắn cờ riêng, không che phần KL thiếu.',
        '- Score hợp lệ được giữ, kể cả điểm từng phần. Chỉ dùng is_correct × max_score khi score không hợp lệ và max_score hợp lệ.',
        ('- AFELO_social_proxy là diagnostic riêng. AFELO_proxy đang chờ sentiment ở mode deferred.' if pending else
         '- AFELO_proxy = mean_non_null(comment_participation, reply_participation, response_received_rate, sentiment_valence). '
         'Chương không có comment giữ sentiment/response rate null; vẫn dùng các thành phần tham gia xã hội còn lại.'),
        '- Khi hoàn thiện AFELO: LOQV tính từng chương; q chỉ lấy trung bình khi mọi chương quan sát của người đó đều có LOQV. '
        'Không tính LOQV từ vector ba chỉ số đã trung bình.',
        '- chapter_details là danh sách struct, giữ nguyên dữ liệu từng chương. Các feature scalar cộng có hậu tố sum_observed, '
        'tỷ lệ/chỉ số trung bình chương có hậu tố mean_chapter; không cộng tỷ lệ.',
        '- Không tự đặt n_problems_available=1 hoặc điểm tối đa=1 khi thiếu catalog.',
        '', '## Trạng thái bảng cuối','', '| Trạng thái | Số dòng |','|---|---:|']
    lines.extend(f'| {r["label_status"]} | {fmt(r["n_rows"])} |' for r in s['label_status'])
    lines += ['', '## Tách train/dev/test và giới hạn thời gian','',
        'Khóa được xếp theo observed course_start rồi course_id, chia 70/10/20 trước khi fit P95. '
        'Không nhóm theo tên; course_id là đơn vị chia theo lựa chọn người dùng. Split hiện là bản sơ bộ để dựng chỉ số, '
        'cần rà lại khi hoàn thiện nhãn và đánh giá mô hình.',
        f'Train có {s["temporal_overlap"]["n_train_courses"]} khóa; '
        f'{s["temporal_overlap"]["n_train_ending_after_first_test_start"]} khóa kết thúc sau ngày mở đầu tiên của test. '
        'Theo review, cần kiểm tra thêm phương án loại các khóa chồng thời gian nếu chạy thí nghiệm vận hành.',
        'Đây là bảng cuối kỳ phục vụ nhãn/kiểm toán, không phải X của P1–P4. Không đưa chỉ số nhãn, q/y, cờ thiếu '
        'hoặc thông tin cuối kỳ vào X. Mốc course_end suy ra từ log là mốc hồi cứu.',
        '`user_problem` là trạng thái tổng hợp tại submit_time; không khôi phục được timestamp từng attempt. '
        'Nguồn video dùng khoảng thu thập công bố; các nguồn có độ phủ khác nhau. Không có log chưa chứng minh người học không hoạt động.',
        '', '## Bổ sung sentiment sau' if pending else '## Nguồn sentiment','',
        ('Cung cấp parquet có cột `comment_id, p_pos, p_neu, p_neg`, mỗi comment một dòng, xác suất tổng bằng 1; '
         'ghi kèm model/revision/ngôn ngữ và kiểm định. Chỉ dự đoán các comment trong `tables/afelo_comment_links.parquet`.' if pending else
         'AFELO đã dùng xác suất sentiment theo comment_id; đường dẫn và SHA256 nằm trong run_manifest.json, '
         'mục finalization. Notebook 07 lưu model/revision, xác suất, cờ truncation/ngôn ngữ và báo cáo riêng.'),
        ('Sau đó gọi ' if pending else 'Đã gọi ')+ '`finalize_quality(output_root, sentiment_path=..., afelo_mode="sentiment")`. '
        'Hàm chỉ đọc các bảng đã tạo, không quét lại video/bài tập. Mode social_only là lựa chọn riêng, không được bật ngầm.',
        '', '## Kiểm tra','',f'Đạt {sum(s["validation"].values())}/{len(s["validation"])} kiểm tra cuối, cùng các kiểm tra nền và fixtures '
        'trong validation.json. run_manifest.json giữ SQL, fingerprint nguồn và cấu hình.','']
    (out/'bao_cao_final_quality.md').write_text('\n'.join(lines),encoding='utf-8')


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--raw-root');p.add_argument('--model-root');p.add_argument('--time-root');p.add_argument('--output-root')
    a=p.parse_args();run_build(a.raw_root,a.model_root,a.time_root,a.output_root)
