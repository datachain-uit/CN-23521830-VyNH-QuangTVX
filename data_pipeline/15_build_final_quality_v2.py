"""Build final quality v2 without modifying v1. Python 3.9 + DuckDB 1.3.2.

Stages: prepare (times/events/components), infer (reuse pinned sentiment cache),
finalize (P95 indicators, two weighted distances, all eligible user-course rows).
All epoch conversion uses UTC, matching notebooks 01-09; no timezone offset fix.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import time

import duckdb

ROOT = Path(__file__).resolve().parent
VERSION = 'final-quality-v2-p95-signed-fa-2026-10-20'
KEY = 'user_id,course_id,lo_key'
POLICY = {
    'version': VERSION,
    'time': 'enrollment day 00:00:00; course last activity day 23:59:59; original event timestamps',
    'cutoffs': {'P1': [1, 4], 'P2': [1, 2], 'P3': [3, 4], 'P4': [9, 10]},
    'TS': 'sum positive content seconds including repeats / P95 of the same total across users of the same video, clipped [0,1]; chapter mean over catalog videos (unwatched=0)',
    'IF': 'sum problem attempts / P95 of positive sums among users of same chapter-course, clipped [0,1]; no log transform',
    'FV': '30-minute user-resource sessions from video/problem/comment/reply; NVL/TNVL; clip(FV_raw/P95_positive(FV_raw),0,1)',
    'FA': '(positive_comments-negative_comments)/all comments of chapter; 0.5+0.5*clip(FA_raw/P95_nonzero(abs(FA_raw)),-1,1)',
    'FA_missing': 'no inferred comments => NULL; all-neutral evidence => 0.5; reply text is not separately inferred',
    'ACELO': 'sum_correct(max(0.1,1-0.1*(attempts-1)))/catalog_problem_count',
    'COELO': 'mean available TS,RV,AV,IF; at least 2 components, unchanged coverage rule',
    'AFELO': 'mean available FV,FA only; no CP/RP/response rate',
    'LOQV': 'sqrt(sum_available(w*(1-index)^2)/sum_available(w)); weights=(1,alpha,1)',
    'CQV': 'mean of available chapter LOQV including synthetic chapter -1; no requirement that all chapters have all indices',
    'labels': 'E:q<0.2; G:0.2<=q<0.4; A:q>=0.4; all missing => NULL',
    'P95_reference': 'final-window eligible observed users; positive values (absolute nonzero for FA); not an input scaler for P1-P4',
    'zero_missing': 'valid zero is preserved; missing stays NULL; positive singleton P95 equals its value',
    'rewind': 'start_point>end_point counted as rewind proxy; contributes no watch duration, remains timestamped activity',
    'unknown_chapter': 'course_id:CH-1; catalog denominators NULL; linked replies inherit parent chapter/resource',
    'alphas': [0.1, 1.0],
}


def qs(x):
    return "'" + str(x).replace("'", "''") + "'"


def dump(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.partial')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    tmp.replace(path)


def load_module(filename):
    spec = importlib.util.spec_from_file_location('helper_' + filename[:2], ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Build:
    def __init__(self, out, memory='1500MB', threads=3):
        self.out = Path(out).resolve()
        self.stage = self.out / '_stage'
        self.tables = self.out / 'tables'
        for p in [self.out, self.stage, self.tables, self.out / '_scratch']:
            p.mkdir(parents=True, exist_ok=True)
        self.con = duckdb.connect()
        for setting in [f"memory_limit={qs(memory)}", f'threads={threads}',
                        "TimeZone='UTC'", 'preserve_insertion_order=false',
                        f'temp_directory={qs(self.out / "_scratch")}', "max_temp_directory_size='2GB'"]:
            self.con.execute('SET ' + setting)
        self.log_path = self.out / 'step_checkpoints.json'
        self.log = json.loads(self.log_path.read_text()) if self.log_path.exists() else {}
        inventory = []
        for source in [ROOT/'input',ROOT/'review_time_phases/tables',ROOT/'final_quality_tables/_stage',
                       ROOT/'eda_engagement_features_comparison/_stage/problem_rows',ROOT/'afelo_sentiment_cardiff/text_predictions']:
            for p in sorted(source.rglob('*.parquet')):
                st = p.stat();inventory.append([str(p),st.st_size,st.st_mtime_ns])
        self.base = hashlib.sha256(json.dumps({'policy':POLICY,'inputs':inventory,
          'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},sort_keys=True).encode()).hexdigest()
        self.views()
        dump(self.out / 'formula_policy.json', POLICY)

    def view(self, name, path):
        self.con.execute(f'CREATE OR REPLACE VIEW "{name}" AS SELECT * FROM read_parquet({qs(path)},union_by_name=true)')

    def views(self):
        for name, path in {
            'old_times': ROOT / 'review_time_phases/tables/user_course_phase_times.parquet',
            'old_bounds': ROOT / 'review_time_phases/tables/course_time_bounds.parquet',
            'catalog': ROOT / 'input/moocubex_model/processed/dim_lo_catalog/*.parquet',
            'resources': ROOT / 'input/moocubex_model/processed/dim_lo_resource/*.parquet',
            'course_comments': ROOT / 'input/moocubex_model/processed/bridge_course_comment/*.parquet',
            'comments': ROOT / 'input/moocubex_parquet/tables/entities/translated_comment/*.parquet',
            'replies': ROOT / 'input/moocubex_parquet/tables/entities/reply/*.parquet',
            'comment_reply': ROOT / 'input/moocubex_parquet/tables/relations/comment_reply/*.parquet',
            'video_source': ROOT / 'final_quality_tables/_stage/video_events.parquet',
        }.items():
            self.view(name, path)

    def rows(self, sql):
        cur = self.con.execute(sql)
        return [dict(zip([x[0] for x in cur.description], r)) for r in cur.fetchall()]

    def save(self, name, sql, table=False, folder=None):
        path = (Path(folder) if folder else self.tables if table else self.stage) / (name + '.parquet')
        path.parent.mkdir(parents=True, exist_ok=True)
        signature = hashlib.sha256((self.base + sql).encode()).hexdigest()
        tick = time.monotonic()
        cache_key = str(path.relative_to(self.out))
        if not path.exists() or self.log.get(cache_key, {}).get('signature') != signature:
            tmp = path.with_suffix('.parquet.partial')
            self.con.execute(f'COPY ({sql}) TO {qs(tmp)} (FORMAT PARQUET,COMPRESSION ZSTD,ROW_GROUP_SIZE 50000)')
            tmp.replace(path)
            self.log[cache_key] = {'signature': signature, 'sql': sql, 'elapsed_seconds': round(time.monotonic()-tick, 3)}
            dump(self.log_path, self.log)
            print(name, 'built', self.log[cache_key]['elapsed_seconds'], flush=True)
        else:
            print(name, 'reused', flush=True)
        self.view(name, path)
        return path

    def prepare(self):
        # Resource ambiguity within one course is kept as chapter -1.
        self.con.execute('''CREATE TEMP TABLE resource_map AS SELECT resource_id,min(course_id) course_id,
          CASE WHEN count(DISTINCT lo_key)=1 THEN min(lo_key) END lo_key
          FROM resources GROUP BY 1 HAVING count(DISTINCT course_id)=1''')
        self.save('comments_prepared', '''WITH typed AS (
          SELECT c.id comment_id,'U_'||cast(c.user_id AS VARCHAR) user_id,c.resource_id,c.text,
            cast(to_timestamp(c.create_time/1000.0) AS TIMESTAMP) event_time,
            coalesce(r.course_id,cc.course_id) course_id,
            coalesce(r.lo_key,coalesce(r.course_id,cc.course_id)||':CH-1') lo_key,
            nullif(regexp_replace(trim(c.text),'\\s+',' ','g'),'') normalized_text
          FROM comments c LEFT JOIN resource_map r USING(resource_id)
          LEFT JOIN course_comments cc ON c.id=cc.comment_id
          WHERE c.user_id IS NOT NULL AND c.create_time BETWEEN 0 AND 4102444800000
            AND coalesce(r.course_id,cc.course_id) IS NOT NULL
            AND (r.course_id IS NULL OR cc.course_id IS NULL OR r.course_id=cc.course_id)
        ), ids AS (SELECT *,row_number() OVER(PARTITION BY comment_id ORDER BY event_time,user_id,resource_id,text) rn FROM typed)
        SELECT * EXCLUDE(rn),row_number() OVER(PARTITION BY user_id,resource_id,event_time,normalized_text ORDER BY comment_id) exact_duplicate_rank
        FROM ids WHERE rn=1''')
        self.con.execute('''CREATE TEMP TABLE parent_map AS SELECT reply_id,min(comment_id) parent_comment_id
          FROM comment_reply GROUP BY 1 HAVING count(DISTINCT comment_id)=1''')
        self.save('replies_prepared', '''WITH typed AS (
          SELECT r.id reply_id,r.user_id,r.text,try_cast(r.create_time AS TIMESTAMP) event_time,
            p.parent_comment_id,c.event_time parent_time,c.course_id,c.lo_key,c.resource_id,
            row_number() OVER(PARTITION BY r.id ORDER BY r.create_time,r.user_id,r.text) rn
          FROM replies r JOIN parent_map p ON r.id=p.reply_id
          JOIN comments_prepared c ON p.parent_comment_id=c.comment_id)
          SELECT * EXCLUDE(rn) FROM typed WHERE rn=1 AND user_id IS NOT NULL AND event_time>=parent_time''')
        # Keep all previously valid course events, add newly mapped social events.
        self.save('course_time_bounds', '''WITH additions AS (
          SELECT course_id,min(event_time) first_time,max(event_time) last_time FROM
          (SELECT course_id,event_time FROM comments_prepared UNION ALL SELECT course_id,event_time FROM replies_prepared)
          GROUP BY 1)
          SELECT b.course_id,b.course_name,b.course_start old_course_start,b.course_end old_course_end,
            least(b.course_start,a.first_time) course_start,
            greatest(b.course_end,a.last_time) course_end_raw,
            date_trunc('day',greatest(b.course_end,a.last_time))+INTERVAL '1 day'-INTERVAL '1 second' course_end
          FROM old_bounds b LEFT JOIN additions a USING(course_id)''', table=True)
        helper = load_module('05_build_review_time_phases.py')
        cuts = ','.join(f"CASE WHEN time_eligible THEN {helper.cutoff_sql('enroll_time','course_end',p)} END cutoff_{p}" for p in helper.PHASES)
        self.save('user_course_phase_times', f'''WITH base AS (
          SELECT o.user_id,o.course_id,o.enroll_time enroll_time_raw,date_trunc('day',o.enroll_time) enroll_time,
            b.course_start,b.course_end_raw,b.course_end,o.course_exists_in_catalog,
            o.time_eligible old_time_eligible,o.course_end old_course_end
          FROM old_times o LEFT JOIN course_time_bounds b USING(course_id)
        ), classified AS (SELECT *,{helper.time_status_sql()} time_status FROM base), eligible AS (
          SELECT *,time_status='ELIGIBLE_TIME' time_eligible FROM classified)
          SELECT *,date_diff('microsecond',enroll_time,course_end) observation_span_us,{cuts} FROM eligible''', table=True)
        self.save('course_splits', '''WITH ranked AS (
          SELECT b.*,row_number() OVER(ORDER BY course_start,course_id) pos,count(*) OVER() n_courses
          FROM course_time_bounds b JOIN (SELECT DISTINCT course_id FROM user_course_phase_times WHERE time_eligible) e USING(course_id))
          SELECT course_id,course_start,course_end,CASE WHEN pos<=floor(n_courses*0.7) THEN 'train'
          WHEN pos<=floor(n_courses*0.8) THEN 'dev' ELSE 'test' END split FROM ranked''', table=True)
        mapped = ROOT / 'eda_engagement_features_comparison/_stage/problem_rows'
        self.view('mapped_problems', mapped / '**/*.parquet')
        self.save('active_candidates', '''SELECT user_id,course_id FROM mapped_problems GROUP BY 1,2
          UNION SELECT user_id,course_id FROM video_source UNION SELECT user_id,course_id FROM comments_prepared
          UNION SELECT user_id,course_id FROM replies_prepared''')
        self.save('active_pairs', '''SELECT t.user_id,t.course_id,t.enroll_time,t.course_end,s.split
          FROM user_course_phase_times t JOIN active_candidates a USING(user_id,course_id)
          JOIN course_splits s USING(course_id) WHERE t.time_eligible''')
        for name, source in [('video_window','video_source'),('comment_window','comments_prepared'),('reply_window','replies_prepared')]:
            self.save(name, f'''SELECT e.* FROM {source} e JOIN active_pairs a USING(user_id,course_id)
              WHERE e.event_time BETWEEN a.enroll_time AND a.course_end''')
        self.save('afelo_comment_links', '''SELECT * FROM comment_window
          WHERE normalized_text IS NOT NULL AND exact_duplicate_rank=1''', table=True)
        self.save('chapter_comment_totals', '''SELECT course_id,lo_key,count(*) n_comments_chapter_total
          FROM comment_window GROUP BY 1,2''', table=True)
        self.save('video_union', '''WITH intervals AS (
          SELECT DISTINCT user_id,course_id,lo_key,video_id,start_point,end_point FROM video_window WHERE valid_measurement
        ), previous AS (SELECT *,max(end_point) OVER(PARTITION BY user_id,course_id,lo_key,video_id ORDER BY start_point,end_point
          ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) prev_end FROM intervals), numbered AS (
          SELECT *,sum(CASE WHEN prev_end IS NULL OR start_point>prev_end THEN 1 ELSE 0 END)
          OVER(PARTITION BY user_id,course_id,lo_key,video_id ORDER BY start_point,end_point) island FROM previous)
          SELECT user_id,course_id,lo_key,video_id,island,min(start_point) interval_start,max(end_point) interval_end
          FROM numbered GROUP BY 1,2,3,4,5''')
        self.save('user_video_components', '''WITH counts AS (
          SELECT user_id,course_id,lo_key,video_id,
            count(*) n_video_events,count(*) FILTER(WHERE valid_measurement) n_valid_video_segments,
            count(*) FILTER(WHERE NOT valid_measurement) n_invalid_video_segments,
            count(*) FILTER(WHERE start_point>end_point AND isfinite(start_point) AND isfinite(end_point)) n_rewind_segments,
            count(*) FILTER(WHERE start_point=end_point) n_zero_length_segments,
            coalesce(sum(end_point-start_point) FILTER(WHERE valid_measurement),0) total_content_seconds,
            min(event_time) first_video_time,max(event_time) last_video_time FROM video_window GROUP BY 1,2,3,4
          ), uniq AS (SELECT user_id,course_id,lo_key,video_id,sum(interval_end-interval_start) unique_content_seconds
            FROM video_union GROUP BY 1,2,3,4)
          SELECT c.*,coalesce(u.unique_content_seconds,0) unique_content_seconds,
            c.total_content_seconds-coalesce(u.unique_content_seconds,0) rewatch_content_seconds
          FROM counts c LEFT JOIN uniq u USING(user_id,course_id,lo_key,video_id)''', table=True)
        self.save('video_p95_reference', '''SELECT course_id,lo_key,video_id,count(*) n_users_with_video_event,
          count(*) FILTER(WHERE total_content_seconds>0) n_users_positive_time,
          quantile_cont(total_content_seconds,0.95) FILTER(WHERE total_content_seconds>0) p95_total_content_seconds
          FROM user_video_components GROUP BY 1,2,3''', table=True)
        self.save('user_video_ts', '''SELECT v.*,r.n_users_positive_time,r.p95_total_content_seconds,
          CASE WHEN total_content_seconds=0 THEN 0 WHEN p95_total_content_seconds>0
            THEN least(1.0,total_content_seconds/p95_total_content_seconds) END TS_video
          FROM user_video_components v LEFT JOIN video_p95_reference r USING(course_id,lo_key,video_id)''', table=True)
        self.save('video_chapter', '''SELECT user_id,course_id,lo_key,count(*) n_videos_with_event,
          count(*) FILTER(WHERE total_content_seconds>0) n_videos_watched,
          sum(n_video_events) n_video_events,sum(n_valid_video_segments) n_valid_video_segments,
          sum(n_invalid_video_segments) n_invalid_video_segments,sum(n_rewind_segments) n_rewind_segments,
          sum(n_zero_length_segments) n_zero_length_segments,sum(total_content_seconds) total_content_seconds,
          sum(unique_content_seconds) unique_content_seconds,sum(rewatch_content_seconds) rewatch_content_seconds,
          sum(TS_video) ts_video_sum,min(first_video_time) first_video_time,max(last_video_time) last_video_time
          FROM user_video_ts GROUP BY 1,2,3''')
        # Bound memory: one user shard, one event sort at a time.
        for shard in range(32):
            self.con.execute('DROP TABLE IF EXISTS problem_accepted')
            source = mapped / f'user_shard={shard}/*.parquet'
            self.con.execute(f'''CREATE TEMP TABLE problem_accepted AS SELECT p.* FROM read_parquet({qs(source)}) p
              JOIN active_pairs a USING(user_id,course_id) WHERE p.event_time BETWEEN a.enroll_time AND a.course_end''')
            self.save(f'problem_{shard:02d}', '''SELECT user_id,course_id,lo_key,count(*) n_problem_events,
              count(DISTINCT problem_id) n_problems_attempted,count(DISTINCT exercise_id) n_exercises_attempted,
              sum(attempts) total_problem_attempts,count(*) FILTER(WHERE is_correct=1) n_problems_correct,
              count(*) FILTER(WHERE is_correct IS NULL OR is_correct NOT IN (0,1)) invalid_correctness,
              count(*) FILTER(WHERE attempts IS NULL OR attempts<=0 OR NOT isfinite(attempts) OR attempts<>floor(attempts)) invalid_attempts,
              count(*) FILTER(WHERE is_correct=1 AND (attempts IS NULL OR attempts<=0 OR NOT isfinite(attempts) OR attempts<>floor(attempts))) correct_invalid_attempts,
              coalesce(sum(CASE WHEN is_correct=1 AND attempts>0 AND isfinite(attempts) AND attempts=floor(attempts)
                THEN greatest(1.0,11.0-attempts) END),0) correct_weight_tenths_sum,
              min(event_time) first_problem_time,max(event_time) last_problem_time
              FROM problem_accepted GROUP BY 1,2,3''', folder=self.stage / 'problem_features')
            self.save(f'visits_{shard:02d}', f'''WITH events AS (
              SELECT user_id,course_id,lo_key,exercise_id resource_id,event_time,'problem' source FROM problem_accepted
              UNION ALL SELECT user_id,course_id,lo_key,video_id,event_time,'video' FROM video_window WHERE hash(user_id)%32={shard}
              UNION ALL SELECT user_id,course_id,lo_key,coalesce(resource_id,'discussion:'||course_id),event_time,'comment' FROM comment_window WHERE hash(user_id)%32={shard}
              UNION ALL SELECT user_id,course_id,lo_key,coalesce(resource_id,'discussion:'||course_id),event_time,'reply' FROM reply_window WHERE hash(user_id)%32={shard}
            ), edges AS (SELECT *,lag(event_time) OVER(PARTITION BY user_id,course_id,lo_key,resource_id ORDER BY event_time,source) prev_time FROM events)
              SELECT user_id,course_id,lo_key,count(*) n_visit_source_events,
                count(*) FILTER(WHERE prev_time IS NULL OR event_time-prev_time>INTERVAL '30 minutes') n_visits,
                count(DISTINCT resource_id) n_resources_with_visit,
                min(event_time) first_observed_event,max(event_time) last_observed_event
              FROM edges GROUP BY 1,2,3''', folder=self.stage / 'visit_features')
        self.con.execute('DROP TABLE problem_accepted')
        self.view('problem_features', self.stage / 'problem_features/*.parquet')
        self.view('visit_features', self.stage / 'visit_features/*.parquet')
        self.save('social_chapter', '''WITH cs AS (
          SELECT user_id,course_id,lo_key,count(*) n_comments_all,
            count(*) FILTER(WHERE normalized_text IS NOT NULL AND exact_duplicate_rank=1) n_comments_for_sentiment
          FROM comment_window GROUP BY 1,2,3), rs AS (
          SELECT user_id,course_id,lo_key,count(*) n_replies_authored FROM reply_window GROUP BY 1,2,3)
          SELECT coalesce(c.user_id,r.user_id) user_id,coalesce(c.course_id,r.course_id) course_id,
            coalesce(c.lo_key,r.lo_key) lo_key,coalesce(n_comments_all,0) n_comments_all,
            coalesce(n_comments_for_sentiment,0) n_comments_for_sentiment,coalesce(n_replies_authored,0) n_replies_authored
          FROM cs c FULL JOIN rs r USING(user_id,course_id,lo_key)''')
        numeric = {
            'v': ['n_videos_with_event','n_videos_watched','n_video_events','n_valid_video_segments','n_invalid_video_segments',
                  'n_rewind_segments','n_zero_length_segments','total_content_seconds','unique_content_seconds','rewatch_content_seconds','ts_video_sum'],
            'p': ['n_problem_events','n_problems_attempted','n_exercises_attempted','total_problem_attempts','n_problems_correct',
                  'invalid_correctness','invalid_attempts','correct_invalid_attempts','correct_weight_tenths_sum'],
            's': ['n_comments_all','n_comments_for_sentiment','n_replies_authored'],
        }
        fields = ','.join(f'coalesce({a}.{x},0) {x}' for a, cols in numeric.items() for x in cols)
        self.save('chapter_inputs', f'''SELECT k.*,a.enroll_time,a.course_end,a.split,c.course_name,
          coalesce(c.chapter_major,-1) chapter_major,c.lo_key IS NOT NULL has_catalog,
          c.n_videos_available,c.n_exercises_available,c.n_problems_available,
          v.user_id IS NOT NULL has_video,p.user_id IS NOT NULL has_problem,s.user_id IS NOT NULL has_social,
          {fields}
          FROM visit_features k JOIN active_pairs a USING(user_id,course_id)
          LEFT JOIN catalog c USING(course_id,lo_key) LEFT JOIN video_chapter v USING(user_id,course_id,lo_key)
          LEFT JOIN problem_features p USING(user_id,course_id,lo_key) LEFT JOIN social_chapter s USING(user_id,course_id,lo_key)''')
        old = load_module('07_compute_afelo_sentiment.py')
        (self.out / 'sentiment').mkdir(exist_ok=True)
        old.prepare_comments(self.con, self.tables / 'afelo_comment_links.parquet', self.out / 'sentiment')
        self.view('unique_texts', self.out / 'sentiment/_stage/unique_texts.parquet')
        self.view('old_predictions', ROOT / 'afelo_sentiment_cardiff/text_predictions/*.parquet')
        self.save('sentiment_missing_texts', '''SELECT u.* FROM unique_texts u
          ANTI JOIN old_predictions p USING(text_hash)''')
        stats = self.rows('''SELECT (SELECT count(*) FROM unique_texts) unique_texts,
          (SELECT count(*) FROM sentiment_missing_texts) new_texts,
          (SELECT count(*) FROM afelo_comment_links) comments_to_score''')[0]
        dump(self.out / 'prepare_summary.json', stats)
        dump(self.out / '_PREPARED.json',{'signature':self.base,'version':VERSION})
        print('PREPARED', json.dumps(stats), flush=True)

    def infer(self, threads=4, batch_size=24, checkpoint_size=512, device='auto'):
        # Same revision, preprocessing, fp32 inference and argmax as v1.
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        lock = json.loads((ROOT / 'afelo_sentiment_cardiff/model_lock.json').read_text())
        dump(self.out / 'sentiment/model_lock.json', lock)
        torch.set_num_threads(threads)
        torch.manual_seed(0)
        if device == 'auto':
            device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        tokenizer = AutoTokenizer.from_pretrained(lock['snapshot_path'], local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(lock['snapshot_path'], local_files_only=True)
        model.eval().to(device)
        labels = {int(k): v.lower() for k,v in model.config.id2label.items()}
        assert labels == {0:'negative',1:'neutral',2:'positive'}
        rows = self.con.execute(f'''SELECT text_hash,model_text,has_cjk FROM read_parquet(
          {qs(self.stage / 'sentiment_missing_texts.parquet')}) ORDER BY length(model_text),text_hash''').fetchall()
        out = self.out / 'sentiment/text_predictions'
        out.mkdir(parents=True,exist_ok=True)
        dump(self.out / 'sentiment/inference_config.json', {'revision':lock['revision'],'device':device,'dtype':'float32',
          'threads':threads,'batch_size':batch_size,'checkpoint_size':checkpoint_size,'max_length':512,'cache_reused':str(ROOT / 'afelo_sentiment_cardiff/text_predictions')})
        started = time.monotonic()
        for part, start in enumerate(range(0,len(rows),checkpoint_size)):
            path = out / f'part-{part:05d}.parquet'
            chunk = rows[start:start+checkpoint_size]
            if path.exists():
                saved = self.con.execute(f'SELECT text_hash FROM read_parquet({qs(path)})').fetchall()
                assert [x[0] for x in saved] == [x[0] for x in chunk], 'Sentiment checkpoint source changed'
                continue
            results = []
            for pos in range(0,len(chunk),batch_size):
                batch = chunk[pos:pos+batch_size]
                texts = [x[1] for x in batch]
                ids = tokenizer(texts, truncation=False,padding=False,verbose=False)['input_ids']
                inputs = tokenizer(texts,truncation=True,max_length=512,padding=True,return_tensors='pt')
                inputs = {k:v.to(device) for k,v in inputs.items()}
                with torch.inference_mode():
                    probs = torch.softmax(model(**inputs).logits.float(),dim=-1).cpu().double().tolist()
                for row,tokens,values in zip(batch,ids,probs):
                    values = [v/sum(values) for v in values]
                    results.append({'text_hash':row[0],'p_neg':values[0],'p_neu':values[1],'p_pos':values[2],
                      'predicted_label':labels[max(range(3),key=lambda i:values[i])],
                      'confidence':max(values),'entropy':-sum(v*math.log(v) for v in values if v>0),
                      'n_tokens':len(tokens),'n_tokens_used':min(len(tokens),512),'was_truncated':len(tokens)>512,'has_cjk':row[2]})
            tmp = path.with_suffix('.jsonl')
            with tmp.open('w') as f:
                for row in results:
                    f.write(json.dumps(row)+'\n')
            partial = path.with_suffix('.parquet.partial')
            self.con.execute(f'COPY (SELECT * FROM read_json_auto({qs(tmp)})) TO {qs(partial)} (FORMAT PARQUET,COMPRESSION ZSTD)')
            partial.replace(path)
            tmp.unlink()
            print(f'SENTIMENT {min(start+checkpoint_size,len(rows))}/{len(rows)} elapsed={time.monotonic()-started:.1f}s',flush=True)
        dump(self.out / 'sentiment/_SUCCESS.json', {'new_predictions':len(rows),'revision':lock['revision']})

    def finalize(self):
        prepared=self.out/'_PREPARED.json'
        if not prepared.exists() or json.loads(prepared.read_text()).get('signature')!=self.base:
            raise RuntimeError('Inputs or builder changed; rerun --stage prepare before finalizing')
        (self.out/'_SUCCESS.json').unlink(missing_ok=True)
        for name in ['chapter_inputs','video_window','comment_window','reply_window']:
            self.view(name,self.stage / (name+'.parquet'))
        for name in ['user_course_phase_times','course_time_bounds','course_splits','afelo_comment_links','chapter_comment_totals','user_video_ts']:
            self.view(name,self.tables / (name+'.parquet'))
        self.view('text_keys',self.out / 'sentiment/_stage/comment_text_keys.parquet')
        self.view('old_predictions',ROOT / 'afelo_sentiment_cardiff/text_predictions/*.parquet')
        new_parts = list((self.out / 'sentiment/text_predictions').glob('*.parquet'))
        prediction_sql = 'SELECT * FROM old_predictions'
        if new_parts:
            self.view('new_predictions',self.out / 'sentiment/text_predictions/*.parquet')
            prediction_sql += ' UNION ALL SELECT * FROM new_predictions'
        self.con.execute('CREATE VIEW predictions AS '+prediction_sql)
        assert self.con.execute('SELECT count(*)=count(DISTINCT text_hash) FROM predictions').fetchone()[0]
        missing = self.con.execute('''SELECT count(*) FROM text_keys k ANTI JOIN predictions p USING(text_hash)''').fetchone()[0]
        if missing:
            raise RuntimeError(f'{missing} comments still need inference; run --stage infer before finalize')
        self.save('comment_sentiment', '''SELECT c.comment_id,c.user_id,c.course_id,c.lo_key,c.event_time,
          p.* FROM afelo_comment_links c JOIN text_keys k USING(comment_id) JOIN predictions p USING(text_hash)''',table=True)
        self.save('sentiment_chapter', '''SELECT user_id,course_id,lo_key,count(*) n_sentiment_comments,
          count(*) FILTER(WHERE predicted_label='positive') n_positive_comments,
          count(*) FILTER(WHERE predicted_label='negative') n_negative_comments,
          count(*) FILTER(WHERE predicted_label='neutral') n_neutral_comments,
          sum(p_pos) sentiment_pos_sum,sum(p_neu) sentiment_neu_sum,sum(p_neg) sentiment_neg_sum
          FROM comment_sentiment GROUP BY 1,2,3''')
        self.save('chapter_raw', '''WITH joined AS (
          SELECT b.*,coalesce(s.n_sentiment_comments,0) n_sentiment_comments,
            coalesce(s.n_positive_comments,0) n_positive_comments,coalesce(s.n_negative_comments,0) n_negative_comments,
            coalesce(s.n_neutral_comments,0) n_neutral_comments,s.sentiment_pos_sum,s.sentiment_neu_sum,s.sentiment_neg_sum,
            coalesce(t.n_comments_chapter_total,0) n_comments_chapter_total,
            CASE WHEN n_videos_available>0 THEN ts_video_sum/n_videos_available END TS,
            n_videos_watched::DOUBLE/nullif(n_videos_available,0) RV,
            n_exercises_attempted::DOUBLE/nullif(n_exercises_available,0) AV,
            CASE WHEN n_problems_available>0 AND invalid_attempts=0 THEN total_problem_attempts END IF_raw,
            CASE WHEN n_problems_available>0 AND invalid_correctness=0 THEN n_problems_correct::DOUBLE/n_problems_available END KL,
            CASE WHEN n_problems_correct>0 AND correct_invalid_attempts=0
              THEN correct_weight_tenths_sum/(10.0*n_problems_correct) END CFA,
            CASE WHEN n_problems_available>0 AND invalid_correctness=0 AND correct_invalid_attempts=0
              THEN correct_weight_tenths_sum/(10.0*n_problems_available) END ACELO
          FROM chapter_inputs b LEFT JOIN sentiment_chapter s USING(user_id,course_id,lo_key)
          LEFT JOIN chapter_comment_totals t USING(course_id,lo_key))
          SELECT *,CASE WHEN n_sentiment_comments>0 AND n_sentiment_comments=n_comments_for_sentiment AND n_comments_chapter_total>0
            THEN (n_positive_comments-n_negative_comments)::DOUBLE/n_comments_chapter_total END FA_raw
          FROM joined''')
        self.save('chapter_p95_reference', '''SELECT course_id,lo_key,count(*) n_users_observed,
          count(IF_raw) n_users_if_observed,count(*) FILTER(WHERE IF_raw>0) n_users_if_positive,
          quantile_cont(IF_raw,0.95) FILTER(WHERE IF_raw>0) p95_IF_raw,
          sum(n_visits) total_visits_chapter,count(*) FILTER(WHERE n_visits>0) n_users_visits_positive,
          quantile_cont(n_visits,0.95) FILTER(WHERE n_visits>0) p95_n_visits,
          count(FA_raw) n_users_fa_observed,count(*) FILTER(WHERE abs(FA_raw)>0) n_users_fa_nonzero,
          quantile_cont(abs(FA_raw),0.95) FILTER(WHERE abs(FA_raw)>0) p95_abs_FA_raw
          FROM chapter_raw GROUP BY 1,2''',table=True)
        self.save('chapter_quality', f'''WITH normalized AS (
          SELECT b.*,r.* EXCLUDE(course_id,lo_key),
            CASE WHEN IF_raw IS NULL THEN NULL WHEN IF_raw=0 THEN 0
              WHEN p95_IF_raw>0 THEN least(1.0,IF_raw/p95_IF_raw) END "IF",
            n_visits::DOUBLE/nullif(total_visits_chapter,0) FV_raw,
            p95_n_visits/nullif(total_visits_chapter,0) p95_FV_raw,
            CASE WHEN n_visits IS NULL THEN NULL WHEN n_visits=0 THEN 0
              WHEN p95_n_visits>0 THEN least(1.0,n_visits::DOUBLE/p95_n_visits) END FV,
            CASE WHEN FA_raw IS NULL THEN NULL WHEN FA_raw=0 THEN 0.5
              WHEN p95_abs_FA_raw>0 THEN 0.5+0.5*least(1.0,greatest(-1.0,FA_raw/p95_abs_FA_raw)) END FA
          FROM chapter_raw b LEFT JOIN chapter_p95_reference r USING(course_id,lo_key)
        ), indices AS (
          SELECT *,list_count([TS,RV,AV,"IF"]) coelo_n_components,
            CASE WHEN list_count([TS,RV,AV,"IF"])>=2 THEN list_avg([TS,RV,AV,"IF"]) END COELO,
            list_count([FV,FA]) afelo_n_components,list_avg([FV,FA]) AFELO
          FROM normalized
        ), distances AS (
          SELECT *,list_count([COELO,AFELO,ACELO]) loqv_n_components,
            CASE WHEN COELO IS NOT NULL THEN 'C' ELSE '' END||CASE WHEN AFELO IS NOT NULL THEN 'F' ELSE '' END||CASE WHEN ACELO IS NOT NULL THEN 'A' ELSE '' END component_mask,
            sqrt((coalesce(pow(1-COELO,2),0)+0.1*coalesce(pow(1-AFELO,2),0)+coalesce(pow(1-ACELO,2),0))/
              nullif((COELO IS NOT NULL)::INT+0.1*(AFELO IS NOT NULL)::INT+(ACELO IS NOT NULL)::INT,0)) LOQV_alpha_0p1,
            sqrt((coalesce(pow(1-COELO,2),0)+coalesce(pow(1-AFELO,2),0)+coalesce(pow(1-ACELO,2),0))/
              nullif((COELO IS NOT NULL)::INT+(AFELO IS NOT NULL)::INT+(ACELO IS NOT NULL)::INT,0)) LOQV_alpha_1
          FROM indices)
          SELECT *,{qs(VERSION)} formula_version FROM distances''',table=True)
        self.save('coelo_components','''SELECT user_id,course_id,lo_key,chapter_major,n_videos_available,n_exercises_available,n_problems_available,
          n_videos_watched,n_exercises_attempted,n_problems_attempted,total_content_seconds,unique_content_seconds,rewatch_content_seconds,
          n_video_events,n_valid_video_segments,n_invalid_video_segments,n_rewind_segments,n_zero_length_segments,
          total_problem_attempts,invalid_attempts,IF_raw,p95_IF_raw,TS,RV,AV,"IF",coelo_n_components,COELO,formula_version FROM chapter_quality''',table=True)
        self.save('afelo_components','''SELECT user_id,course_id,lo_key,chapter_major,n_visits,total_visits_chapter,FV_raw,p95_FV_raw,p95_n_visits,FV,
          n_comments_all,n_comments_for_sentiment,n_sentiment_comments,n_positive_comments,n_negative_comments,n_neutral_comments,
          n_comments_chapter_total,n_replies_authored,FA_raw,p95_abs_FA_raw,FA,afelo_n_components,AFELO,formula_version FROM chapter_quality''',table=True)
        self.save('acelo_components','''SELECT user_id,course_id,lo_key,chapter_major,n_problems_available,n_problems_attempted,n_problems_correct,
          total_problem_attempts,correct_weight_tenths_sum,invalid_correctness,correct_invalid_attempts,KL,CFA,ACELO,formula_version FROM chapter_quality''',table=True)
        sums = ['n_videos_with_event','n_videos_watched','n_video_events','n_valid_video_segments','n_invalid_video_segments','n_rewind_segments',
          'n_zero_length_segments','total_content_seconds','unique_content_seconds','rewatch_content_seconds','n_problem_events','n_problems_attempted',
          'n_exercises_attempted','total_problem_attempts','n_problems_correct','invalid_correctness','invalid_attempts','correct_invalid_attempts',
          'correct_weight_tenths_sum','n_comments_all','n_comments_for_sentiment','n_replies_authored','n_visits','n_visit_source_events',
          'n_resources_with_visit','n_sentiment_comments','n_positive_comments','n_negative_comments','n_neutral_comments']
        means = ['TS','RV','AV','IF_raw','IF','KL','CFA','FV_raw','FV','FA_raw','FA','COELO','AFELO','ACELO']
        extra = ','.join([f'sum("{x}") {x}_sum_observed' for x in sums]+[f'avg("{x}") {x}_mean_chapter' for x in means])
        aggregate_sql=f'''SELECT user_id,course_id,count(*) n_chapters_observed,
          count(*) FILTER(WHERE chapter_major=-1) n_unknown_chapters,
          count(COELO) n_chapters_coelo,count(AFELO) n_chapters_afelo,count(ACELO) n_chapters_acelo,
          count(*) FILTER(WHERE loqv_n_components=3) n_chapters_full_components,
          count(LOQV_alpha_0p1) n_chapters_scored_alpha_0p1,count(LOQV_alpha_1) n_chapters_scored_alpha_1,
          avg(LOQV_alpha_0p1) q_alpha_0p1,avg(LOQV_alpha_1) q_alpha_1,
          min(first_observed_event) first_observed_event,max(last_observed_event) last_observed_event,
          {extra},list(chapter_quality ORDER BY chapter_major,lo_key) chapter_details
          FROM chapter_quality WHERE hash(user_id)%32={{shard}} GROUP BY 1,2'''
        aggregate_dir=self.stage/'user_course_aggregates'
        for shard in range(32):
            self.save(f'aggregate-{shard:02d}',aggregate_sql.format(shard=shard),folder=aggregate_dir)
        self.view('user_course_aggregates',aggregate_dir/'*.parquet')
        # All time-eligible enrollment pairs survive. Only observed pairs have aggregates.
        final_dir = self.tables / 'user_course_final'
        for shard in range(32):
            self.view('aggregate_slice',aggregate_dir/f'aggregate-{shard:02d}.parquet')
            self.save(f'part-{shard:02d}', f'''WITH merged AS (
              SELECT t.*,s.split,coalesce(a.n_chapters_observed,0) n_chapters_observed,
                coalesce(a.n_unknown_chapters,0) n_unknown_chapters,
                a.* EXCLUDE(user_id,course_id,n_chapters_observed,n_unknown_chapters)
              FROM user_course_phase_times t LEFT JOIN course_splits s USING(course_id)
              LEFT JOIN aggregate_slice a USING(user_id,course_id)
              WHERE t.time_eligible AND hash(t.user_id)%32={shard})
              SELECT *,CASE WHEN q_alpha_0p1<0.2 THEN 'E' WHEN q_alpha_0p1<0.4 THEN 'G' WHEN q_alpha_0p1>=0.4 THEN 'A' END y_alpha_0p1,
                CASE WHEN q_alpha_1<0.2 THEN 'E' WHEN q_alpha_1<0.4 THEN 'G' WHEN q_alpha_1>=0.4 THEN 'A' END y_alpha_1,
                CASE WHEN n_chapters_observed=0 THEN 'NO_OBSERVED_CHAPTER'
                  WHEN q_alpha_0p1 IS NULL THEN 'NO_AVAILABLE_LOQV' ELSE 'LABELED_OBSERVED_CHAPTERS' END label_status,
                {qs(VERSION)} formula_version
              FROM merged''',folder=final_dir)
        self.view('user_course_final',final_dir / '*.parquet')
        for suffix,alpha in [('0p1',0.1),('1',1.0)]:
            self.save('user_course_labels',f'''SELECT user_id,course_id,split,enroll_time,course_end,cutoff_P1,cutoff_P2,cutoff_P3,cutoff_P4,
              n_chapters_observed,n_unknown_chapters,n_chapters_full_components,n_chapters_coelo,n_chapters_afelo,n_chapters_acelo,
              COELO_mean_chapter COELO,AFELO_mean_chapter AFELO,ACELO_mean_chapter ACELO,
              q_alpha_{suffix} q,y_alpha_{suffix} y,label_status,{alpha}::DOUBLE alpha_afelo,formula_version FROM user_course_final''',
              folder=self.out / f'variants/alpha_{suffix}')
        self.audit()

    def audit(self):
        checks = {}
        def check(name,sql):
            checks[name] = bool(self.con.execute(sql).fetchone()[0])
        check('all_time_pairs_preserved','SELECT (SELECT count(*) FROM user_course_phase_times)=(SELECT count(*) FROM old_times)')
        check('one_row_per_enrollment','SELECT count(*)=count(DISTINCT(user_id,course_id)) FROM user_course_phase_times')
        check('all_eligible_pairs_in_final','SELECT (SELECT count(*) FROM user_course_final)=(SELECT count(*) FROM user_course_phase_times WHERE time_eligible)')
        check('final_unique_keys','SELECT count(*)=count(DISTINCT(user_id,course_id)) FROM user_course_final')
        check('chapter_unique_keys','SELECT count(*)=count(DISTINCT(user_id,course_id,lo_key)) FROM chapter_quality')
        check('comment_predictions_complete','SELECT (SELECT count(*) FROM comment_sentiment)=(SELECT count(*) FROM afelo_comment_links)')
        check('comment_ids_unique','SELECT count(*)=count(DISTINCT comment_id) FROM comment_sentiment')
        check('sentiment_probabilities_valid','''SELECT count(*)=0 FROM comment_sentiment WHERE p_neg IS NULL OR p_neu IS NULL OR p_pos IS NULL
          OR p_neg<0 OR p_neu<0 OR p_pos<0 OR abs(p_neg+p_neu+p_pos-1)>1e-6''')
        check('sentiment_chapter_counts_complete','SELECT count(*)=0 FROM chapter_quality WHERE n_sentiment_comments<>n_comments_for_sentiment')
        check('no_duplicate_problem_observations','SELECT count(*)=0 FROM chapter_quality WHERE n_problem_events<>n_problems_attempted')
        check('no_invalid_problem_inputs','SELECT count(*)=0 FROM chapter_quality WHERE invalid_correctness>0 OR invalid_attempts>0')
        check('problem_counts_within_catalog','SELECT count(*)=0 FROM chapter_quality WHERE n_problems_attempted>n_problems_available OR n_problems_correct>n_problems_available')
        check('acelo_product_matches_direct','''SELECT count(*)=0 FROM chapter_quality WHERE n_problems_correct>0 AND abs(ACELO-KL*CFA)>1e-12''')
        check('zero_correct_keeps_cfa_null','''SELECT count(*)=0 FROM chapter_quality WHERE n_problems_available>0 AND n_problems_correct=0 AND (ACELO<>0 OR CFA IS NOT NULL)''')
        check('no_problem_catalog_acelo_null','SELECT count(*)=0 FROM chapter_quality WHERE coalesce(n_problems_available,0)=0 AND ACELO IS NOT NULL')
        check('no_comment_fa_null','SELECT count(*)=0 FROM chapter_quality WHERE n_sentiment_comments=0 AND (FA IS NOT NULL OR FA_raw IS NOT NULL)')
        check('neutral_fa_half','SELECT count(*)=0 FROM chapter_quality WHERE FA_raw=0 AND FA<>0.5')
        check('signed_fa_preserved','''SELECT count(*)=0 FROM chapter_quality WHERE (FA_raw<0 AND FA>=0.5) OR (FA_raw>0 AND FA<=0.5)''')
        check('unknown_catalog_denominators_null','''SELECT count(*)=0 FROM chapter_quality WHERE chapter_major=-1 AND
          (n_videos_available IS NOT NULL OR n_exercises_available IS NOT NULL OR n_problems_available IS NOT NULL OR COELO IS NOT NULL OR ACELO IS NOT NULL)''')
        check('nonnegative_video_time','SELECT count(*)=0 FROM user_video_ts WHERE total_content_seconds<0 OR unique_content_seconds<0 OR rewatch_content_seconds < -1e-8')
        check('rewind_not_in_watch_duration','SELECT count(*)=0 FROM video_window WHERE start_point>end_point AND valid_measurement')
        check('all_video_events_accounted','SELECT (SELECT count(*) FROM video_window)=(SELECT sum(n_video_events) FROM chapter_quality)')
        for name in ['TS','RV','AV','IF','KL','CFA','ACELO','FV','FA','COELO','AFELO','LOQV_alpha_0p1','LOQV_alpha_1']:
            check(name+'_in_unit_interval',f'SELECT count(*)=0 FROM chapter_quality WHERE "{name}" IS NOT NULL AND (NOT isfinite("{name}") OR "{name}" < -1e-10 OR "{name}" > 1+1e-10)')
        for source in ['video_window','comment_window','reply_window']:
            check(source+'_within_window',f'''SELECT count(*)=0 FROM {source} e JOIN user_course_phase_times t USING(user_id,course_id)
              WHERE e.event_time<t.enroll_time OR e.event_time>t.course_end''')
        check('all_enrollments_at_midnight',"SELECT count(*)=0 FROM user_course_phase_times WHERE enroll_time<>date_trunc('day',enroll_time)")
        check('all_course_ends_end_of_day',"SELECT count(*)=0 FROM course_time_bounds WHERE course_end<>date_trunc('day',course_end)+INTERVAL '1 day'-INTERVAL '1 second'")
        check('cutoffs_strictly_ordered','''SELECT count(*)=0 FROM user_course_phase_times WHERE time_eligible AND NOT
          (enroll_time<cutoff_P1 AND cutoff_P1<cutoff_P2 AND cutoff_P2<cutoff_P3 AND cutoff_P3<cutoff_P4 AND cutoff_P4<course_end)''')
        check('no_observed_pairs_not_fabricated','SELECT count(*)=0 FROM user_course_final WHERE n_chapters_observed=0 AND (q_alpha_0p1 IS NOT NULL OR q_alpha_1 IS NOT NULL)')
        check('nested_chapters_complete','SELECT count(*)=0 FROM user_course_final WHERE coalesce(len(chapter_details),0)<>n_chapters_observed')
        check('user_q_is_mean_chapter_distances','''SELECT count(*)=0 FROM user_course_final u JOIN
          (SELECT user_id,course_id,avg(LOQV_alpha_0p1) q0,avg(LOQV_alpha_1) q1 FROM chapter_quality GROUP BY 1,2) c USING(user_id,course_id)
          WHERE abs(u.q_alpha_0p1-c.q0)>1e-12 OR abs(u.q_alpha_1-c.q1)>1e-12''')
        for suffix in ['0p1','1']:
            check('class_thresholds_'+suffix,f'''SELECT count(*)=0 FROM user_course_final WHERE y_alpha_{suffix} IS DISTINCT FROM
              CASE WHEN q_alpha_{suffix}<0.2 THEN 'E' WHEN q_alpha_{suffix}<0.4 THEN 'G' WHEN q_alpha_{suffix}>=0.4 THEN 'A' END''')
        summary = {
            'formula_version':VERSION,'client_date':'2026-10-20',
            'times':self.rows('''SELECT count(*) input_pairs,count(*) FILTER(WHERE time_eligible) eligible_pairs,
              count(*) FILTER(WHERE time_eligible AND NOT old_time_eligible) newly_eligible_pairs,
              count(*) FILTER(WHERE NOT time_eligible AND old_time_eligible) lost_eligible_pairs,
              count(DISTINCT course_id) FILTER(WHERE time_eligible) eligible_courses FROM user_course_phase_times''')[0],
            'chapters':self.rows('''SELECT count(*) "rows",count(DISTINCT(user_id,course_id)) observed_pairs,
              count(*) FILTER(WHERE chapter_major=-1) unknown_chapter_rows,count(*) FILTER(WHERE loqv_n_components=3) full_components,
              count(*) FILTER(WHERE loqv_n_components<3) partial_components,
              count(COELO) coelo_available,count(AFELO) afelo_available,count(ACELO) acelo_available,
              avg(COELO) coelo_mean,avg(AFELO) afelo_mean,avg(ACELO) acelo_mean,
              sum(n_rewind_segments) rewind_segments FROM chapter_quality''')[0],
            'comments':self.rows('''SELECT count(*) comments_scored,count(*) FILTER(WHERE lo_key LIKE '%:CH-1') unknown_chapter_comments FROM comment_sentiment''')[0],
            'replies':self.rows("SELECT count(*) replies,count(*) FILTER(WHERE lo_key LIKE '%:CH-1') unknown_chapter_replies FROM reply_window")[0],
            'status_counts':self.rows('SELECT label_status,count(*) n FROM user_course_final GROUP BY 1 ORDER BY 1'),
            'label_counts':{},'q_statistics':{},'cutoff_changes':[],
            'validation':checks,
        }
        for suffix in ['0p1','1']:
            summary['label_counts'][suffix] = self.rows(f'SELECT y_alpha_{suffix} y,count(*) n FROM user_course_final GROUP BY 1 ORDER BY 1')
            summary['q_statistics'][suffix] = self.rows(f'''SELECT count(q_alpha_{suffix}) n_labeled,min(q_alpha_{suffix}) min_q,
              quantile_cont(q_alpha_{suffix},0.25) p25_q,median(q_alpha_{suffix}) median_q,avg(q_alpha_{suffix}) mean_q,
              quantile_cont(q_alpha_{suffix},0.75) p75_q,max(q_alpha_{suffix}) max_q FROM user_course_final''')[0]
        for phase in ['P1','P2','P3','P4']:
            summary['cutoff_changes'] += self.rows(f'''SELECT {qs(phase)} phase,count(*) pairs_compared,
              count(*) FILTER(WHERE n.cutoff_{phase}<>o.cutoff_{phase}) changed_cutoffs,
              median(date_diff('microsecond',o.cutoff_{phase},n.cutoff_{phase})/3600000000.0) median_shift_hours,
              min(date_diff('microsecond',o.cutoff_{phase},n.cutoff_{phase})/3600000000.0) min_shift_hours,
              max(date_diff('microsecond',o.cutoff_{phase},n.cutoff_{phase})/3600000000.0) max_shift_hours
              FROM user_course_phase_times n JOIN old_times o USING(user_id,course_id) WHERE n.time_eligible AND o.time_eligible''')
        summary['alpha_label_changes'] = self.rows('''SELECT y_alpha_0p1,y_alpha_1,count(*) n FROM user_course_final
          WHERE q_alpha_0p1 IS NOT NULL GROUP BY 1,2 ORDER BY 1,2''')
        summary['component_masks'] = self.rows('SELECT component_mask,count(*) n FROM chapter_quality GROUP BY 1 ORDER BY 1')
        self.view('v1_labels',ROOT / 'weighted_loqv_alpha_0p1/tables/user_course_loqv_weighted.parquet')
        cols = [r[0] for r in self.con.execute('DESCRIBE v1_labels').fetchall()]
        ycol = 'y_weighted' if 'y_weighted' in cols else 'y'
        summary['v1_v2_label_transition'] = self.rows(f'''SELECT coalesce(o.{ycol},'NULL') old_y,coalesce(n.y_alpha_0p1,'NULL') new_y,count(*) n
          FROM v1_labels o JOIN user_course_final n USING(user_id,course_id) GROUP BY 1,2 ORDER BY 1,2''')
        dump(self.out / 'summary.json',summary)
        dump(self.out / 'validation.json',checks)
        self.write_report(summary)
        # Record immutable inputs and source code for reproducibility.
        inventory = []
        for root in [ROOT/'input',ROOT/'review_time_phases/tables',ROOT/'final_quality_tables/_stage',ROOT/'afelo_sentiment_cardiff/text_predictions']:
            for p in sorted(root.rglob('*.parquet')):
                stat = p.stat();inventory.append({'path':str(p),'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns})
        dump(self.out / 'run_manifest.json',{'formula_policy':POLICY,'input_inventory':inventory,
          'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'runtime':{'duckdb':duckdb.__version__},
          'model_lock':json.loads((ROOT/'afelo_sentiment_cardiff/model_lock.json').read_text()),
          'role':'final-window target tables; existing phase feature tables have not been overwritten',
          'outputs':sorted(str(p.relative_to(self.out)) for p in self.out.rglob('*.parquet'))})
        assert all(checks.values()), {k:v for k,v in checks.items() if not v}
        dump(self.out / '_SUCCESS.json',{'version':VERSION,'checks_passed':len(checks),'labeled_pairs':summary['q_statistics']['0p1']['n_labeled']})
        print('COMPLETE',json.dumps({'times':summary['times'],'chapters':summary['chapters'],'checks':len(checks)},ensure_ascii=False),flush=True)

    def write_report(self,s):
        fmt=lambda x:f'{int(x):,}'.replace(',','.')
        lines=['# Final quality v2','', 'Ngày xây dựng: 20/10/2026. Đơn vị quan sát cuối kỳ: `user_id × course_id`. Bảng chương: `user_id × course_id × lo_key`.',
          '', '## Phạm vi và các thay đổi', '',
          '- Giữ nguyên raw và toàn bộ bảng v1. Các file v2 nằm riêng trong thư mục này.',
          '- Enrollment về 00:00:00; course-end về 23:59:59 của ngày có hoạt động cuối. Không cộng/trừ timezone vào timestamp event.',
          '- Course-end giữ các nguồn hoạt động hợp lệ của v1 và xét thêm comment/reply được khôi phục. Cutoff cá nhân được tính lại ở P1=25%, P2=50%, P3=75%, P4=90% quãng enrollment đến course-end.',
          '- Comment nối được course nhưng thiếu chapter dùng `course_id:CH-1`; reply kế thừa chapter và resource của comment cha. Catalog của chapter -1 giữ NULL.',
          '- Segment start > end được đếm là `n_rewind_segments` (proxy tua ngược), không cộng giây xem. Segment start = end được đếm riêng. Tất cả vẫn giữ vai trò event có timestamp.',
          '', '## Công thức', '',
          'TS_video = clip(total_content_seconds / P95_positive(total_content_seconds cùng video), 0, 1). Thời lượng là giây nội dung có xem lại, không chia speed. TS chapter là tổng TS_video chia số video catalog; video chưa xem đóng góp 0. Không dùng duration caption.',
          '', 'RV = video có thời lượng xem hợp lệ / video catalog. AV = exercise đã thử / exercise catalog. IF = clip(tổng attempts problem / P95_positive của tổng attempts trong cùng chapter-course, 0, 1). COELO = mean_non_null(TS,RV,AV,IF), giữ điều kiện ít nhất 2 thành phần.',
          '', 'FV: gom event video, problem, comment, reply thành phiên user-resource; một phiên mới khi gap > 30 phút. NVL là số phiên; FV_raw=NVL/tổng NVL chapter. FV=clip(FV_raw/P95_positive(FV_raw),0,1), tương đương NVL/P95_positive(NVL). Problem chỉ có submit_time tổng hợp; số phiên không tái tạo được timestamp từng attempt.',
          '', 'FA_raw=(n_positive_comments_user−n_negative_comments_user)/n_comments_chapter_total. Mẫu số đếm comment sau xác định course/chapter và cửa sổ người học, trước lọc text rỗng/trùng; gồm neutral và comment chưa inference, không chỉ user có nhãn. Tử số đếm argmax label của comment có nội dung hợp lệ, khử trùng chính xác. FA=0.5+0.5×clip(FA_raw/P95_nonzero(|FA_raw|),−1,1). FA không có bằng chứng sentiment giữ NULL; có bằng chứng nhưng hiệu positive−negative bằng 0 nhận 0.5. Chỉ infer comment, reply đóng góp phiên FV và được lưu count, không tự thêm sentiment reply vào FA.',
          '', 'AFELO=mean_non_null(FV,FA). CP, RP và response rate không tham gia. Giữ xác suất sentiment và counts để kiểm toán. Model sentiment là proxy cho appreciation, không phải like/dislike thật.',
          '', 'KL=S/N; CFA=sum_correct(max(0.1,1−0.1×(attempts−1)))/S; ACELO=sum_correct(f)/N. S=0,N>0: KL=ACELO=0,CFA=NULL. N=0/NULL: ACELO=NULL. Không dùng score/max_score trong công thức v2.',
          '', 'LOQV_alpha = sqrt(sum_available(w×(1−index)^2)/sum_available(w)), với w=(1,alpha,1). Hai alpha: 0.1 và 1. CQV/q = mean_non_null(LOQV các chapter), gồm chapter -1; không lấy khoảng cách từ vector trung bình các chỉ số. q nhỏ hơn là tốt hơn. E:q<0.2; G:0.2≤q<0.4; A:q≥0.4.',
          '', '## Quy mô', '',
          f'- User-course đủ thời gian: **{fmt(s["times"]["eligible_pairs"])}**; thêm **{fmt(s["times"]["newly_eligible_pairs"])}** so với v1.',
          f'- Chapter quan sát: **{fmt(s["chapters"]["rows"])}**; user-course có chapter: **{fmt(s["chapters"]["observed_pairs"])}**.',
          f'- Chapter -1: **{fmt(s["chapters"]["unknown_chapter_rows"])}**. Comment được inference ở chapter -1: **{fmt(s["comments"]["unknown_chapter_comments"])}**.',
          f'- Tua ngược trong cửa sổ v2: **{fmt(s["chapters"]["rewind_segments"])}** segment.',
          '', '| Alpha AFELO | Nhãn E | Nhãn G | Nhãn A | Chưa có nhãn | q trung bình |','|---|---:|---:|---:|---:|---:|']
        for suffix in ['0p1','1']:
            d={r['y']:r['n'] for r in s['label_counts'][suffix]}
            lines.append(f'| {suffix.replace("p",".")} | {fmt(d.get("E",0))} | {fmt(d.get("G",0))} | {fmt(d.get("A",0))} | {fmt(d.get(None,0))} | {s["q_statistics"][suffix]["mean_q"]:.6f} |')
        lines += ['', '## Thay đổi cutoff P1-P4', '', '| Phase | Số cutoff thay đổi (cặp cùng đủ điều kiện) | Trung vị lệch giờ v2−v1 |', '|---|---:|---:|']
        for r in s['cutoff_changes']:
            lines.append(f'| {r["phase"]} | {fmt(r["changed_cutoffs"])} | {r["median_shift_hours"]:.6f} |')
        lines += ['', 'Các bảng feature P1-P4 cũ chưa được ghi đè. Cần xây lại feature theo cutoff v2 trước khi huấn luyện; không dùng thống kê P95 cuối kỳ làm scaler cho đầu vào phase.',
          '', '## File và cách đọc', '',
          '- `tables/user_course_phase_times.parquet`: giữ mọi cặp ghi danh, mốc raw/chuẩn hóa, trạng thái thời gian và cutoff mới.',
          '- `tables/coelo_components.parquet`, `afelo_components.parquet`, `acelo_components.parquet`: thành phần và mẫu số theo user-course-chapter.',
          '- `tables/user_video_ts.parquet`, `video_p95_reference.parquet`, `chapter_p95_reference.parquet`: nguồn chuẩn hóa để kiểm toán.',
          '- `tables/chapter_quality.parquet`: tất cả thành phần, ba chỉ số, mask độ phủ và hai LOQV.',
          '- `tables/user_course_final/*.parquet`: một bảng chia 32 file, giữ mọi cặp đủ thời gian; feature cộng/trung bình, ba chỉ số trung bình, cả hai q/y và danh sách struct `chapter_details` chứa đầy đủ dữ liệu chương.',
          '- `variants/alpha_0p1/user_course_labels.parquet`, `variants/alpha_1/user_course_labels.parquet`: hai bảng nhãn phụ; cùng keys/cutoff/split, khác alpha/q/y. Join với bảng cuối hoặc chapter theo keys để lấy feature chi tiết.',
          '', '```python', 'import duckdb', 'con = duckdb.connect()',
          'con.execute("SET TimeZone=\'UTC\'")',
          'con.execute("CREATE VIEW quality AS SELECT * FROM read_parquet(\'working_13-10/final_quality_v2/tables/user_course_final/*.parquet\')")',
          'con.execute("SELECT user_id, course_id, q_alpha_0p1, y_alpha_0p1 FROM quality WHERE q_alpha_0p1 IS NOT NULL LIMIT 5").fetchall()', '```',
          '', '## Hạn chế và chống leakage', '',
          '- Không có event sau khi lọc không chứng minh người học không hoạt động; user-course không có chapter vẫn giữ q/y=NULL. Không impute các chỉ số thiếu.',
          '- P95 dùng user có giá trị dương; FA dùng trị tuyệt đối khác 0. Nhóm nhỏ/đơn lẻ có ngưỡng không ổn định; lưu kích thước nhóm để đánh giá độ nhạy. Điểm là mức tương đối trong cohort cuối kỳ, không phải chuẩn năng lực tuyệt đối.',
          '- Dữ liệu video/problem/social có thời gian tracking khác nhau. Chuẩn hóa ngày không khôi phục được log chưa thu thập, cũng không sửa sai timezone chưa có bằng chứng.',
          '- Phân chia course vẫn dựa thứ tự course_start, 70/10/20, course không giao giữa split; user có thể lặp giữa split và thời gian course có thể chồng nhau. Chưa tuyên bố split là mô phỏng triển khai hoàn toàn theo thời gian.',
          '- Không dùng q/y, index cuối kỳ hoặc thống kê P95 label làm feature dự đoán sớm. Các phép fit/impute/scaler đầu vào mô hình phải fit trên train, theo đúng thông tin có tại cutoff.',
          '- Việc bỏ thành phần NULL cho phép chấm từ ít chiều hơn; cần sử dụng component_mask và số chapter/chiều có dữ liệu để báo cáo độ phủ. Alpha triệt tiêu ở chapter chỉ có AFELO.',
          '- Sentiment giữ model/revision và preprocessing v1; cache text cũ được tái sử dụng, text mới được inference thực tế. Không tự gán sentiment theo emoji hoặc từ khóa.',
          '', '## Kiểm tra', '', f'Đạt {sum(s["validation"].values())}/{len(s["validation"])} kiểm tra. Chi tiết: `validation.json`, `summary.json`, `run_manifest.json`, `formula_policy.json`.', '']
        (self.out/'BAO_CAO_FINAL_QUALITY_V2.md').write_text('\n'.join(lines),encoding='utf-8')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage',choices=['prepare','infer','finalize','all'],default='all')
    ap.add_argument('--output',default=str(ROOT / 'final_quality_v2'))
    ap.add_argument('--threads',type=int,default=4)
    ap.add_argument('--device',choices=['auto','cpu','mps'],default='auto')
    args = ap.parse_args()
    b = Build(args.output,threads=min(args.threads,3))
    if args.stage in ['prepare','all']:
        b.prepare()
    if args.stage in ['infer','all']:
        b.infer(threads=args.threads,device=args.device)
    if args.stage in ['finalize','all']:
        b.finalize()


if __name__ == '__main__':
    main()
