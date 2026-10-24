"""Integration check: appending future logs must not change earlier features.

Uses the real notebook-09 window/aggregation functions and explicit fixture
schemas. All rows live in memory; no private Parquet files are needed.
"""
import importlib.util
import json
import math
from pathlib import Path

import duckdb

BASE=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('phase_features_engine',BASE/'09_build_phase_features.py')
engine=importlib.util.module_from_spec(spec);spec.loader.exec_module(engine)


def run_check():
    c=duckdb.connect()
    user='U_cutoff_fixture'
    shard=c.execute('SELECT hash(?)%32',[user]).fetchone()[0]
    empty=next('U_empty_'+str(i) for i in range(500) if c.execute('SELECT hash(?)%32',['U_empty_'+str(i)]).fetchone()[0]==shard)
    q=engine.qs
    c.execute(f"CREATE TABLE pair_slice AS SELECT user_id,'C_fixture' course_id,TIMESTAMP '2020-01-01' enroll_time,TIMESTAMP '2020-01-05' course_end FROM (VALUES ({q(user)}),({q(empty)})) t(user_id)")
    c.execute("CREATE TABLE course_splits AS SELECT 'C_fixture' course_id,'train' split")
    c.execute("CREATE TABLE course_encoded AS SELECT 'C_fixture' course_id,2 course_n_videos,2 course_n_exercises,2 course_n_problems,1 course_n_chapters,1 teacher_set_code,1 school_set_code,1 field_set_code")
    c.execute(f"CREATE TABLE static_users AS SELECT user_id,1995 year_of_birth,1 gender_code,{shard} user_shard FROM pair_slice")
    c.execute('CREATE TABLE enc_learner AS SELECT user_id token,row_number() OVER(ORDER BY user_id) code FROM pair_slice')
    schemas={
      'video_events':'''user_id VARCHAR,course_id VARCHAR,lo_key VARCHAR,video_id VARCHAR,
          start_point DOUBLE,end_point DOUBLE,speed DOUBLE,event_time TIMESTAMP,valid_measurement BOOLEAN''',
      'problem_slice':'''log_id VARCHAR,problem_id VARCHAR,user_id VARCHAR,is_correct INTEGER,
          attempts BIGINT,score DOUBLE,submit_time VARCHAR,course_id VARCHAR,lo_key VARCHAR,
          exercise_id VARCHAR,problem_max_score DOUBLE,n_exercises BIGINT,event_time TIMESTAMP,
          score_valid BOOLEAN,fallback_possible BOOLEAN,efficiency_valid BOOLEAN,user_shard BIGINT''',
      'comments':'''comment_id VARCHAR,user_id VARCHAR,resource_id VARCHAR,text VARCHAR,
          event_time TIMESTAMP,course_id VARCHAR,lo_key VARCHAR,normalized_text VARCHAR,
          exact_duplicate_rank BIGINT,enroll_time TIMESTAMP,course_end TIMESTAMP''',
      'replies':'''reply_id VARCHAR,user_id VARCHAR,event_time TIMESTAMP,parent_comment_id VARCHAR,
          parent_time TIMESTAMP,course_id VARCHAR,lo_key VARCHAR''',
    }
    for name,columns in schemas.items():c.execute(f'CREATE TABLE {name} ({columns})')
    c.execute('CREATE TABLE sentiment(comment_id VARCHAR,p_pos DOUBLE,p_neu DOUBLE,p_neg DOUBLE)')
    c.execute(f"INSERT INTO video_events VALUES ({q(user)},'C_fixture','LO_fixture','V_fixture',0,60,2,TIMESTAMP '2020-01-01 23:59:50',true)")
    c.execute(f'''INSERT INTO problem_slice(log_id,problem_id,user_id,is_correct,attempts,score,submit_time,
      course_id,lo_key,exercise_id,problem_max_score,n_exercises,event_time,score_valid,fallback_possible,efficiency_valid,user_shard)
      VALUES ('L_before','Pm_before',{q(user)},1,2,1,'2020-01-01 10:00:00','C_fixture','LO_fixture','Ex_before',1,1,
      TIMESTAMP '2020-01-01 10:00:00',true,true,true,{shard})''')
    c.execute(f'''INSERT INTO comments VALUES ('Cm_before',{q(user)},'V_fixture','Good',TIMESTAMP '2020-01-01 12:00:00',
      'C_fixture','LO_fixture','Good',1,TIMESTAMP '2020-01-01',TIMESTAMP '2020-01-05')''')
    c.execute("INSERT INTO sentiment VALUES ('Cm_before',0.7,0.2,0.1)")
    c.execute(f"INSERT INTO replies VALUES ('R_before',{q(user)},TIMESTAMP '2020-01-01 18:00:00','Cm_before',TIMESTAMP '2020-01-01 12:00:00','C_fixture','LO_fixture')")

    def snapshot(phase):
        engine.build_windows(c,phase,shard);engine.aggregate_windows(c)
        result=c.execute(engine.feature_sql(phase)+' ORDER BY user_id').fetchall()
        return result,{row[0]:dict(zip([x[0] for x in c.description],row)) for row in result}

    before,rows=snapshot('P1')
    checks={
      'production_engine_clips_crossing_segment':abs(rows[user]['video_media_seconds']-20)<1e-12,
      'wall_seconds_respect_speed_and_cutoff':abs(rows[user]['video_wall_seconds']-10)<1e-12,
      'unique_seconds_use_only_observed_content':abs(rows[user]['video_unique_seconds']-20)<1e-12,
      'observed_reply_counted_before_cutoff':rows[user]['n_replies_received']==1,
      'empty_phase_pair_preserved':rows[empty]['n_events']==0 and rows[empty]['sentiment_p_positive_mean'] is None,
    }
    # These include large values and negative sentiment to expose accidental
    # use of full-window aggregates, and a reply to an already observed comment.
    c.execute(f"INSERT INTO video_events VALUES ({q(user)},'C_fixture','LO_fixture','V_future',0,1000,1,TIMESTAMP '2020-01-03',true)")
    c.execute(f'''INSERT INTO problem_slice(log_id,problem_id,user_id,is_correct,attempts,score,submit_time,
      course_id,lo_key,exercise_id,problem_max_score,n_exercises,event_time,score_valid,fallback_possible,efficiency_valid,user_shard)
      VALUES ('L_future','Pm_future',{q(user)},0,999,0,'2020-01-03','C_fixture','LO_fixture','Ex_future',1,1,
      TIMESTAMP '2020-01-03',true,true,true,{shard})''')
    c.execute(f'''INSERT INTO comments VALUES ('Cm_future',{q(user)},'V_future','Bad',TIMESTAMP '2020-01-03',
      'C_fixture','LO_fixture','Bad',1,TIMESTAMP '2020-01-01',TIMESTAMP '2020-01-05')''')
    c.execute("INSERT INTO sentiment VALUES ('Cm_future',0,0,1)")
    c.execute(f"INSERT INTO replies VALUES ('R_future',{q(user)},TIMESTAMP '2020-01-03','Cm_before',TIMESTAMP '2020-01-01 12:00:00','C_fixture','LO_fixture')")
    after,_=snapshot('P1')
    checks['all_55_features_unchanged_by_future_logs']=before==after
    _,later=snapshot('P3')
    checks['later_phase_can_use_new_problem_state']=later[user]['problem_attempts']==1001
    checks['later_phase_can_use_new_sentiment']=abs(later[user]['sentiment_p_negative_mean']-0.55)<1e-12
    checks['later_phase_can_use_new_reply']=later[user]['n_replies_received']==2
    result={'checks':checks,'source_sha256':engine.sha256(BASE/'09_build_phase_features.py'),
      'n_features_compared':len(engine.NUMERIC)+len(engine.CATEGORICAL),
      'fixture':'In-memory events before/after 2020-01-02 P1 cutoff, including a crossing video segment.'}
    assert all(checks.values()),result
    engine.dump_json(BASE/'phase_features/temporal_invariance_validation.json',result)
    print(json.dumps(result,indent=2));c.close()
    return result


if __name__=='__main__':run_check()
