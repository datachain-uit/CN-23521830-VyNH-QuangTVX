"""Build the CD3 CQ feature view on the frozen V4 train/dev/FULL_A spine.

No model training, test snapshots, new labels, or sentiment inference. Raw data
remain local. Run with the Python 3.12 environment and requirements-v4_2.txt.
"""
from __future__ import annotations

import argparse
import datetime as dt
import gc
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import duckdb
import numpy as np
import pandas as pd

from cq_tempo_features import (
    CANDIDATES, COMMENT_FEATURES, PROBLEM_FEATURES, VIDEO_FEATURES,
    CQRawEarlyViews, add_hour_cyclical, build_cd3_phase_features,
    held_out_columns, qa23_audit, raw_early_columns,
)

VERSION = "phase_features_v4_2"
CD3_COMMIT = "5b14052ec6329c33f4d9f5324a118f7192b1a579"
LABEL_VERSION = "final-quality-v2-p95-signed-fa-2026-10-20"
META = ["user_id", "course_id", "course_family_id", "enrollment_id", "row_id",
        "phase", "split", "enroll_time", "cutoff_time", "course_end"]
TARGETS = ["q_alpha_0p1", "y_alpha_0p1", "q_alpha_1", "y_alpha_1"]
MASKS = [f"{f}_observed_mask" for f in ("video", "problem", "comment")]
RAW = list(CANDIDATES) + MASKS + ["phase_fraction"]
PHASES = ["P1", "P2", "P3", "P4"]
ROOT = Path(__file__).resolve().parent


def qs(value):
    return "'" + str(value).replace("'", "''") + "'"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(2**20), b""):
            h.update(b)
    return h.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, default=str).encode()).hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str,
                              allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def json_records(frame):
    # pandas' JSON writer consistently translates NaN/NaT to JSON null.
    return json.loads(frame.to_json(orient="records", date_format="iso"))


def cardiff_hash(text):
    words = []
    for word in text.split(" "):
        word = "@user" if word.startswith("@") and len(word) > 1 else word
        word = "http" if word.startswith("http") else word
        words.append(word)
    return hashlib.sha256(" ".join(words).encode("utf-8")).hexdigest()


class Builder:
    def __init__(self, root, output, threads=2, memory="1200MB", batch_size=1500,
                 snapshot_splits=("train", "dev")):
        self.root, self.out = Path(root).resolve(), Path(output).resolve()
        self.snapshot_splits = tuple(snapshot_splits)
        if not self.snapshot_splits or len(set(self.snapshot_splits)) != len(self.snapshot_splits) or not set(self.snapshot_splits) <= {"train", "dev", "test"}:
            raise ValueError("Invalid snapshot splits")
        self.include_full_a = "train" in self.snapshot_splits
        if self.out == (self.root / "phase_features_v4").resolve():
            raise ValueError("Output must be separate from frozen V4")
        self.out.mkdir(parents=True, exist_ok=True)
        self.stage = self.out / "_stage"
        self.stage.mkdir(exist_ok=True)
        (self.out / "_scratch").mkdir(exist_ok=True)
        self.con = duckdb.connect(":memory:")
        for setting in [f"threads={int(threads)}", f"memory_limit={qs(memory)}",
                        "TimeZone='UTC'", "preserve_insertion_order=false",
                        f"temp_directory={qs(self.out / '_scratch')}",
                        "max_temp_directory_size='3GB'"]:
            self.con.execute("SET " + setting)
        self.con.create_function("cardiff_hash", cardiff_hash, ["VARCHAR"], "VARCHAR")
        self.batch_size = batch_size
        self.checks, self.sources, self.steps = {}, {}, {}
        self.started = time.monotonic()
        self.checkpoint = self.out / "checkpoints.json"
        self.cache = json.loads(self.checkpoint.read_text()) if self.checkpoint.exists() else {}

    def check(self, name, condition, evidence=None):
        self.checks[name] = {"status": "PASS" if bool(condition) else "FAIL", "evidence": evidence}
        dump(self.out / "audit/validation.json", self.checks)
        if not condition:
            raise AssertionError(f"{name}: {evidence}")

    def sqlcheck(self, name, sql):
        self.check(name, self.con.execute(sql).fetchone()[0])

    def source(self, name, pattern):
        paths = sorted(self.root.glob(pattern))
        if not paths:
            raise FileNotFoundError(pattern)
        for p in paths:
            s = p.stat()
            self.sources[str(p)] = {"bytes": s.st_size, "mtime_ns": s.st_mtime_ns}
        self.con.execute(f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM read_parquet({qs(self.root / pattern)})")

    def write(self, path, sql):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".parquet.partial")
        self.con.execute(f"COPY ({sql}) TO {qs(tmp)} (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 10000)")
        tmp.replace(path)

    def frame_write(self, path, frame):
        self.con.register("output_frame", frame)
        try:
            self.write(path, "SELECT * FROM output_frame ORDER BY row_id")
        finally:
            self.con.unregister("output_frame")

    def materialize(self, name, sql):
        path = self.stage / f"{name}.parquet"
        key = canonical_hash([self.signature, sql])
        tick = time.monotonic()
        if not (path.exists() and self.cache.get(name) == {"key": key, "sha256": sha256(path)}):
            self.write(path, sql)
            self.cache[name] = {"key": key, "sha256": sha256(path)}
            dump(self.checkpoint, self.cache)
        self.con.execute(f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM read_parquet({qs(path)})")
        n = self.con.execute(f"SELECT count(*) FROM {name}").fetchone()[0]
        self.steps[name] = {"rows": n, "seconds": round(time.monotonic() - tick, 3),
                            "sql": sql, "sha256": self.cache[name]["sha256"]}
        print(f"{name}: {n:,} rows ({self.steps[name]['seconds']:.1f}s)", flush=True)

    def prepare(self):
        for phase in PHASES:
            for split in self.snapshot_splits:
                self.source(f"sp_{phase}_{split}", f"phase_features_v4/supervised_raw/{phase}/{split}.parquet")
        if self.include_full_a:
            self.source("sp_full_a", "phase_features_v4/fulltrain_a/train_raw.parquet")
        self.d06_source = self.root / "phase_features_v4/audit/cutoff_sample_20_pairs.parquet"
        if self.d06_source.exists():
            self.source("d06_original", "phase_features_v4/audit/cutoff_sample_20_pairs.parquet")
        tables = {"courses": "entities/course", "people": "entities/user",
                  "raw_video": "relations/user_video", "raw_problem": "relations/user_problem",
                  "problems": "entities/problem", "comments": "entities/translated_comment",
                  "course_comments": "relations/course_comment", "ct": "relations/course_teacher",
                  "teachers_src": "entities/teacher", "cs": "relations/course_school",
                  "schools_src": "entities/school"}
        for name, rel in tables.items():
            self.source(name, f"input/moocubex_parquet/tables/{rel}/*.parquet")
        self.source("pred_old", "afelo_sentiment_cardiff/text_predictions/*.parquet")
        self.source("pred_new", "final_quality_v2/sentiment/text_predictions/*.parquet")
        self.con.execute("CREATE VIEW predictions AS SELECT * FROM pred_old UNION ALL BY NAME SELECT * FROM pred_new")
        self.sqlcheck("sentiment_cache_unique_hash", "SELECT count(*)=count(DISTINCT text_hash) FROM predictions")
        cols = ",".join(META + TARGETS)
        union = " UNION ALL ".join(f"SELECT {cols} FROM sp_{p}_{s}" for p in PHASES for s in self.snapshot_splits)
        if self.include_full_a:
            union += f" UNION ALL SELECT {cols} FROM sp_full_a"
        self.con.execute(f"CREATE VIEW source_spine AS {union}")
        self.signature = canonical_hash({"sources": self.sources, "builder": sha256(__file__),
                                         "helper": sha256(ROOT / "cq_tempo_features.py"),
                                         "version": VERSION, "batch_size": self.batch_size})
        spine_cols = ",".join("cast(enroll_time AS TIMESTAMP) AS enroll_time" if c=="enroll_time" else c for c in META+TARGETS)
        self.materialize("spine", f"SELECT {spine_cols} FROM source_spine ORDER BY row_id")
        self.sqlcheck("spine_unique_rows", "SELECT count(*)=count(DISTINCT row_id) AND count(*)=count(DISTINCT(enrollment_id,phase)) FROM spine")
        self.sqlcheck("spine_allowed_splits", "SELECT count(*)=0 FROM spine WHERE split NOT IN ("+",".join(qs(s) for s in self.snapshot_splits)+") OR (phase='FULL_A' AND split<>'train')")
        self.sqlcheck("spine_four_early_phases", "SELECT count(*)=0 FROM (SELECT enrollment_id FROM spine WHERE phase<>'FULL_A' GROUP BY 1 HAVING count(*)<>4 OR count(DISTINCT phase)<>4)")
        self.sqlcheck("spine_family_disjoint", "SELECT count(*)=0 FROM (SELECT course_family_id FROM spine GROUP BY 1 HAVING count(DISTINCT split)>1)")
        self.materialize("cohort", """SELECT enrollment_id,user_id,course_id,split,min(enroll_time) enroll_time,
            max(cutoff_time) max_cutoff FROM spine GROUP BY 1,2,3,4 ORDER BY course_id,enrollment_id""")
        self.sqlcheck("cohort_unique_enrollment_and_pair", "SELECT count(*)=count(DISTINCT enrollment_id) AND count(*)=count(DISTINCT(user_id,course_id)) FROM cohort")
        self.materialize("resources", """SELECT DISTINCT c.id course_id,r.resource_id resource_id
            FROM courses c,UNNEST(c.resource) t(r) WHERE c.id IN (SELECT course_id FROM cohort)
            AND r.resource_id IS NOT NULL ORDER BY course_id,resource_id""")
        # CD3 countDistinct(context_id) treats the entire array as a value. JSON
        # is a canonical, hashable representation; do not explode into concepts.
        self.materialize("catalog", """SELECT r.course_id,
            CASE WHEN starts_with(p.problem_id,'Pm_') THEN p.problem_id ELSE 'Pm_'||p.problem_id END problem_id,
            p.exercise_id,json(p.context_id)::VARCHAR context_id,p.score::DOUBLE full_score
            FROM problems p JOIN resources r ON p.exercise_id=r.resource_id
            WHERE starts_with(r.resource_id,'Ex_') ORDER BY r.course_id,problem_id""")
        self.sqlcheck("catalog_unambiguous_course_problem", "SELECT count(*)=count(DISTINCT(course_id,problem_id)) FROM catalog")
        self.materialize("enrollments", """SELECT e.enrollment_id,p.gender,p.year_of_birth,e.enroll_time
            FROM cohort e LEFT JOIN people p ON p.id=e.user_id ORDER BY e.enrollment_id""")
        self.sqlcheck("enrollment_context_one_per_cohort", "SELECT (SELECT count(*) FROM enrollments)=(SELECT count(*) FROM cohort) AND (SELECT count(*)=count(DISTINCT enrollment_id) FROM enrollments)")
        self.materialize("course_teacher", "SELECT DISTINCT course_id,teacher_id FROM ct WHERE course_id IN (SELECT course_id FROM cohort)")
        self.materialize("teachers", "SELECT id teacher_id,org_name,about FROM teachers_src WHERE id IN (SELECT teacher_id FROM course_teacher)")
        self.materialize("course_school", "SELECT DISTINCT course_id,school_id FROM cs WHERE course_id IN (SELECT course_id FROM cohort)")
        self.materialize("schools", "SELECT id school_id,about,motto FROM schools_src WHERE id IN (SELECT school_id FROM course_school)")
        for table, key in [("teachers", "teacher_id"), ("schools", "school_id")]:
            self.sqlcheck(table + "_unique_profile", f"SELECT count(*)=count(DISTINCT {key}) FROM {table}")
        # Scalar UNNEST avoids retaining/copying each large parent sequence for
        # every child row (lateral joins can exhaust memory on the real corpus).
        video_sql = """WITH items AS (
            SELECT u.user_id,unnest(u.seq) item FROM raw_video u
            SEMI JOIN (SELECT DISTINCT user_id FROM cohort) users USING(user_id)),
            segments AS (SELECT e.enrollment_id,e.course_id,r.resource_id,
                unnest(list_filter(item.segment,s -> s.local_start_time BETWEEN epoch(e.enroll_time) AND epoch(e.max_cutoff))) seg
                FROM items v JOIN resources r ON item.video_id=r.resource_id
                JOIN cohort e ON e.user_id=v.user_id AND e.course_id=r.course_id
                WHERE starts_with(r.resource_id,'V_'))
            SELECT enrollment_id,course_id,resource_id,
            cast(to_timestamp(seg.local_start_time) AS TIMESTAMP) event_time,
            seg.start_point start_point,seg.end_point end_point,seg.speed speed FROM segments
            ORDER BY course_id,enrollment_id,event_time,resource_id"""
        # Source video Parquet files each contain one large nested row group.
        # Read one file at a time so four decoded groups cannot coexist in RAM.
        video_parts = []
        for i, path in enumerate(sorted(self.root.glob("input/moocubex_parquet/tables/relations/user_video/*.parquet"))):
            name = f"video_source_{i:03d}"
            self.materialize(name, video_sql.replace("FROM raw_video u", f"FROM read_parquet({qs(path)}) u"))
            video_parts.append(name)
        self.materialize("video", "SELECT * FROM ("+" UNION ALL ".join("SELECT * FROM "+n for n in video_parts)+") ORDER BY course_id,enrollment_id,event_time,resource_id")
        problem_sql = """WITH relevant AS MATERIALIZED (
            SELECT p.user_id,p.problem_id,p.score,p.is_correct,p.attempts,p.submit_time FROM raw_problem p
            SEMI JOIN (SELECT DISTINCT user_id FROM cohort) u USING(user_id)
            SEMI JOIN (SELECT DISTINCT problem_id FROM catalog) c USING(problem_id)),
            mapped AS MATERIALIZED (SELECT p.*,c.course_id,c.exercise_id,c.context_id,c.full_score
                FROM relevant p JOIN catalog c USING(problem_id))
            SELECT e.enrollment_id,e.course_id,p.problem_id,
            p.exercise_id,p.context_id,p.score,p.full_score,p.is_correct,p.attempts,
            try_cast(p.submit_time AS TIMESTAMP) submit_time
            FROM mapped p JOIN cohort e USING(user_id,course_id)
            WHERE try_cast(p.submit_time AS TIMESTAMP) BETWEEN e.enroll_time AND e.max_cutoff
            ORDER BY e.course_id,e.enrollment_id,submit_time,p.problem_id"""
        problem_parts = []
        for i, path in enumerate(sorted(self.root.glob("input/moocubex_parquet/tables/relations/user_problem/*.parquet"))):
            name = f"problem_source_{i:03d}"
            self.materialize(name, problem_sql.replace("FROM raw_problem p", f"FROM read_parquet({qs(path)}) p"))
            problem_parts.append(name)
        self.materialize("problem", "SELECT * FROM ("+" UNION ALL ".join("SELECT * FROM "+n for n in problem_parts)+") ORDER BY course_id,enrollment_id,submit_time,problem_id")
        self.sqlcheck("problem_one_row_per_enrollment_problem", "SELECT count(*)=count(DISTINCT(enrollment_id,problem_id)) FROM problem")
        self.materialize("comment", """WITH typed AS MATERIALIZED (
            SELECT c.id comment_id,'U_'||cast(c.user_id AS VARCHAR) user_id,c.resource_id,c.text,
                cast(to_timestamp(c.create_time/1000.0) AS TIMESTAMP) event_time
            FROM comments c WHERE c.id IS NOT NULL AND c.user_id IS NOT NULL
                AND c.create_time BETWEEN 0 AND 4102444800000
                AND 'U_'||cast(c.user_id AS VARCHAR) IN (SELECT user_id FROM cohort)),
            mappings AS (SELECT c.comment_id,m.course_id FROM typed c JOIN course_comments m USING(comment_id)
                WHERE m.course_id IN (SELECT course_id FROM cohort)
                UNION SELECT c.comment_id,r.course_id FROM typed c JOIN resources r USING(resource_id)),
            selected AS (SELECT e.enrollment_id,e.course_id,c.comment_id,c.resource_id,c.event_time,c.text,
                cardiff_hash(c.text) text_hash
                FROM typed c JOIN mappings m USING(comment_id) JOIN cohort e USING(user_id,course_id)
                WHERE c.event_time BETWEEN e.enroll_time AND e.max_cutoff)
            SELECT s.*,p.p_pos positive_score,p.p_neg negative_score,p.p_neu neutral_score,
                lower(p.predicted_label) sentiment_label
            FROM selected s LEFT JOIN predictions p USING(text_hash)
            ORDER BY s.course_id,s.enrollment_id,s.comment_id""")
        self.sqlcheck("comment_dedup_does_not_hide_source_conflicts", "SELECT count(*)=count(DISTINCT(enrollment_id,course_id,comment_id)) FROM comment")
        self.sqlcheck("video_valid_inputs_finite", """SELECT count(*)=0 FROM video WHERE
            end_point>=start_point AND speed>0 AND
            (NOT isfinite(start_point) OR NOT isfinite(end_point) OR NOT isfinite(speed))""")
        self.sqlcheck("problem_inputs_finite", """SELECT count(*)=0 FROM problem WHERE full_score>0 AND attempts>=0
            AND (NOT isfinite(full_score) OR (score IS NOT NULL AND NOT isfinite(score)))""")
        dump(self.out / "audit/join_cardinality.json", self.steps)

    def build_raw(self):
        ids = self.con.execute("SELECT enrollment_id FROM cohort ORDER BY course_id,enrollment_id").df()
        contexts = {k: self.con.execute("SELECT * FROM " + k).df() for k in
                    ["enrollments", "resources", "catalog", "course_teacher", "teachers", "course_school", "schools"]}
        parts = []
        n_batches = (len(ids) + self.batch_size - 1) // self.batch_size
        for i in range(n_batches):
            selected = ids.iloc[i*self.batch_size:(i+1)*self.batch_size]
            self.con.register("batch_ids", selected)
            sp = self.con.execute("SELECT s.* FROM spine s JOIN batch_ids USING(enrollment_id) ORDER BY s.row_id").df()
            ev = [self.con.execute(f"SELECT e.* FROM {k} e JOIN batch_ids USING(enrollment_id)").df()
                  for k in ["video", "problem", "comment"]]
            courses = sp.course_id.unique()
            ctx = [contexts["enrollments"][contexts["enrollments"].enrollment_id.isin(selected.enrollment_id)]]
            ctx += [contexts[k][contexts[k].course_id.isin(courses)] for k in ["resources", "catalog", "course_teacher"]]
            ctx += [contexts["teachers"], contexts["course_school"][contexts["course_school"].course_id.isin(courses)], contexts["schools"]]
            raw = build_cd3_phase_features(sp, *ev, *ctx)
            if len(raw) != len(sp) or raw.row_id.duplicated().any() or set(raw.columns) != set(META+TARGETS+RAW):
                raise AssertionError("CD3 output schema/cardinality mismatch")
            numbers = raw[list(CANDIDATES)].drop(columns=["most_common_day"]).select_dtypes(include="number").to_numpy(dtype=float,na_value=np.nan)
            if np.isinf(numbers).any():
                raise AssertionError("Nonfinite raw feature")
            path = self.stage / "raw_parts" / f"part-{i:05d}.parquet"
            self.frame_write(path, raw[META+TARGETS+RAW])
            parts.append(path)
            print(f"raw CD3 batch {i+1}/{n_batches}: {len(raw):,} snapshots", flush=True)
            del sp, ev, ctx, raw
            gc.collect()
        self.con.execute("CREATE VIEW raw_all AS SELECT * FROM read_parquet(["+",".join(qs(p) for p in parts)+"])")
        self.sqlcheck("raw_exact_spine_keys_metadata_and_labels", f"""SELECT count(*)=0 FROM raw_all r FULL JOIN spine s USING(row_id)
            WHERE r.row_id IS NULL OR s.row_id IS NULL OR {' OR '.join('r.'+x+' IS DISTINCT FROM s.'+x for x in META+TARGETS)}""")
        self.sqlcheck("raw_one_row_per_spine", "SELECT (SELECT count(*) FROM raw_all)=(SELECT count(*) FROM spine) AND (SELECT count(*)=count(DISTINCT row_id) FROM raw_all)")
        if self.include_full_a:
            self.sqlcheck("full_a_same_train_cohort", """SELECT count(*)=0 FROM
                (SELECT DISTINCT enrollment_id FROM raw_all WHERE phase<>'FULL_A' AND split='train') b
                FULL JOIN (SELECT enrollment_id FROM raw_all WHERE phase='FULL_A') a USING(enrollment_id)
                WHERE b.enrollment_id IS NULL OR a.enrollment_id IS NULL""")
            self.sqlcheck("full_a_lambda_one", "SELECT count(*)=0 FROM raw_all WHERE phase='FULL_A' AND (phase_fraction<>1 OR cutoff_time<>course_end OR split<>'train')")
        counts = ["segment_count", "unique_video_count", "video_active_day_count", "problem_count", "exercise_count",
                  "problem_context_count", "correct_problem_count", "incorrect_problem_count", "attempts_sum",
                  "comment_count", "unique_commented_resource_count", "positive_comment_count", "neutral_comment_count", "negative_comment_count"]
        lag = ",".join(f"lag(coalesce({c},0),1,0) OVER w previous_{c}" for c in counts)
        self.sqlcheck("cumulative_counts_nondecreasing", f"""SELECT count(*)=0 FROM
            (SELECT *,{lag} FROM raw_all WINDOW w AS (PARTITION BY enrollment_id ORDER BY CASE phase WHEN 'FULL_A' THEN 'P5' ELSE phase END))
            WHERE {' OR '.join(f'coalesce({c},0)<previous_{c}' for c in counts)}""")
        self.outputs = []
        exports = [("pool_b/train", "split='train' AND phase<>'FULL_A'")] if self.include_full_a else []
        exports += [(f"pool_b/{s}_phase{i}", f"split={qs(s)} AND phase='P{i}'")
                    for s in self.snapshot_splits if s != "train" for i in range(1,5)]
        if self.include_full_a:
            exports.append(("fulltrain_a/train", "phase='FULL_A'"))
        for name, where in exports:
            path = self.out / "raw_features" / (name + ".parquet")
            self.write(path, "SELECT * FROM raw_all WHERE " + where + " ORDER BY row_id")
            self.outputs.append((name, path))

    def audit_events(self):
        for family, table, timecol, validity in [
            ("video", "video", "event_time", "end_point>=start_point AND speed>0"),
            ("problem", "problem", "submit_time", "full_score>0 AND attempts>=0"),
            ("comment", "comment", "event_time", "comment_id IS NOT NULL")]:
            cols = "sum((e.event_time+((end_point-start_point)/speed)*INTERVAL '1 second'>s.cutoff_time)::BIGINT) crossing_segments," if family=="video" else ""
            self.write(self.out / f"audit/{family}_event_bounds.parquet", f"""SELECT s.row_id,count(*) n_events,
                min(e.{timecol}) first_event,max(e.{timecol}) last_event,{cols}
                count(*) FILTER(WHERE e.{timecol}<s.enroll_time OR e.{timecol}>s.cutoff_time) out_of_bounds
                FROM {table} e JOIN spine s USING(enrollment_id) WHERE {validity}
                AND e.{timecol} BETWEEN s.enroll_time AND s.cutoff_time GROUP BY s.row_id""")
        for family, count in [("video","r.segment_count"), ("problem","r.correct_problem_count+r.incorrect_problem_count"), ("comment","r.comment_count")]:
            self.con.execute(f"CREATE VIEW bounds_{family} AS SELECT * FROM read_parquet({qs(self.out / ('audit/'+family+'_event_bounds.parquet'))})")
            self.sqlcheck(family+"_counts_match_selected_events", f"""SELECT count(*)=0 FROM raw_all r LEFT JOIN bounds_{family} b USING(row_id)
                WHERE coalesce({count},0)<>coalesce(b.n_events,0) OR r.{family}_observed_mask<>(coalesce(b.n_events,0)>0)::INTEGER""")
            self.sqlcheck(family+"_availability_within_frozen_window", f"""SELECT count(*)=0 FROM bounds_{family} b JOIN spine s USING(row_id)
                WHERE b.first_event<s.enroll_time OR b.last_event>s.cutoff_time OR b.out_of_bounds<>0""")
        missing = self.con.execute("SELECT phase,split,count(*) n," + ",".join(f"count(*) FILTER(WHERE {c} IS NULL) null_{c}" for c in CANDIDATES)
                                   + ",sum((video_observed_mask+problem_observed_mask+comment_observed_mask=0)::BIGINT) absent_all FROM raw_all GROUP BY 1,2 ORDER BY 1,2").df()
        dump(self.out / "audit/null_and_absence.json", json_records(missing))
        coverage = self.con.execute("""SELECT count(*) n_comments,count(*) FILTER(WHERE sentiment_label IS NOT NULL) with_sentiment,
            count(*) FILTER(WHERE text IS NULL OR trim(text)='') empty_text,
            count(*) FILTER(WHERE sentiment_label IS NULL AND text IS NOT NULL AND trim(text)<>'') nonempty_without_cached_sentiment FROM comment""").df()
        dump(self.out / "audit/sentiment_coverage.json", json_records(coverage))

    def compare_d06(self):
        pairs = [("segment_count","n_video_segments"),("segment_duration_seconds","video_media_seconds"),
                 ("avg_segment_duration","video_mean_segment_seconds"),("avg_playback_speed","video_mean_speed"),
                 ("unique_video_count","n_videos_watched"),("video_active_day_count","video_active_days"),
                 ("problem_count","n_problem_submissions"),("exercise_count","n_exercises_attempted"),
                 ("attempts_sum","problem_attempts"),("correct_problem_count","n_correct_problems"),
                 ("comment_count","n_comments"),("avg_comment_length_chars","comment_mean_characters")]
        available = {r[0] for r in self.con.execute("DESCRIBE sp_P1_train").fetchall()}
        if not all(old in available for _,old in pairs):
            dump(self.out / "audit/d06_comparison.json", {"status":"unavailable_in_fixture", "human_signoff":False})
            return
        if self.d06_source.exists():
            self.con.execute("""CREATE TEMP TABLE comparison_ids AS SELECT DISTINCT c.enrollment_id FROM cohort c
                JOIN d06_original d USING(enrollment_id)""")
        else:
            self.con.execute("CREATE TEMP TABLE comparison_ids AS SELECT enrollment_id FROM cohort WHERE false")
        reused = self.con.execute("SELECT count(*) FROM comparison_ids").fetchone()[0]
        # Original D06 can contain test enrollments. Restrict to train/dev, and
        # explicitly distinguish supplemental pairs from original D06 pairs.
        self.con.execute(f"""INSERT INTO comparison_ids SELECT enrollment_id FROM cohort
            WHERE enrollment_id NOT IN (SELECT enrollment_id FROM comparison_ids)
            ORDER BY sha256(enrollment_id) LIMIT {max(0,20-reused)}""")
        oldcols = ",".join(["row_id"]+[old for _,old in pairs])
        union = " UNION ALL ".join(f"SELECT {oldcols} FROM sp_{p}_{s}" for p in PHASES for s in ["train","dev"])
        self.con.execute("CREATE VIEW comparison_v4 AS "+union)
        self.con.execute("""CREATE TEMP TABLE d06_new AS SELECT r.* FROM raw_all r
            JOIN comparison_ids USING(enrollment_id) WHERE phase<>'FULL_A'""")
        self.con.execute("CREATE TEMP TABLE d06_old AS SELECT o.* FROM comparison_v4 o JOIN d06_new r USING(row_id)")
        comparisons = []
        for new,old in pairs:
            comparisons.append(f"""SELECT r.row_id,r.enrollment_id,r.phase,r.split,{qs(new)} cd3_feature,{qs(old)} v4_feature,
                r.{new} cd3_value,o.{old} v4_value,r.{new}-o.{old} delta,
                r.{new} IS NOT DISTINCT FROM o.{old} exactly_equal
                FROM d06_new r JOIN d06_old o USING(row_id)""")
        self.write(self.out/"audit/d06_feature_comparison.parquet", " UNION ALL ".join(comparisons))
        self.write(self.out/"audit/d06_sample_spine.parquet", "SELECT s.* FROM spine s JOIN comparison_ids USING(enrollment_id) WHERE phase<>'FULL_A' ORDER BY enrollment_id,phase")
        summary = self.con.execute(f"""SELECT cd3_feature,v4_feature,count(*) snapshots,
            sum(exactly_equal::INTEGER) exactly_equal,max(abs(delta)) max_absolute_delta
            FROM read_parquet({qs(self.out/'audit/d06_feature_comparison.parquet')}) GROUP BY 1,2 ORDER BY 1""").df()
        dump(self.out/"audit/d06_comparison.json", {"status":"automatic_comparison", "original_train_dev_pairs":reused,
            "total_pairs":self.con.execute("SELECT count(*) FROM comparison_ids").fetchone()[0],
            "test_pairs_excluded":True, "human_signoff":False, "summary":json_records(summary),
            "interpretation":"Differences are expected from full vs clipped segments, event validity, comment dedup/text filtering and NULL absence. Equal feature names do not imply identical event populations."})

    def ready(self):
        train_path = self.out / "raw_features/pool_b/train.parquet"
        train = self.con.execute(f"SELECT * FROM read_parquet({qs(train_path)}) ORDER BY row_id").df()
        cyc = add_hour_cyclical(train)
        cols = raw_early_columns(cyc)
        self.check("base_inventory_76_74", len(cols["learner"])==76 and len(cols["selector"])==74)
        print("QA23: auditing 74 columns on train P1–P4", flush=True)
        audit = qa23_audit(cyc, "q_alpha_1", cols["selector"])
        drop = held_out_columns(audit)
        dump(self.out / "audit/qa23_by_phase.json", json_records(audit))
        dump(self.out / "audit/held_out_columns.json", {"columns": drop, "threshold": .5,
            "rule": "union of unreviewed flagged columns across train P1-P4", "status": "pending_teacher_review" if drop else "none"})
        del cyc
        views = CQRawEarlyViews().fit(train, drop=drop)
        prep = {"dataset_version": VERSION, "method": "CD3 V1-Median", "fit_split": "train",
                "fit_population": "TRAIN_POOLED_P1_P4", "fit_rows": len(train),
                "static_fit_rows": train.enrollment_id.nunique(), "fit_order": "row_id ascending",
                "fit_row_ids_sha256": canonical_hash(train.row_id.tolist()), "held_out_columns": drop,
                "columns": views.cols, "center": views.v1.center, "scale": views.v1.scale,
                "log_columns": views.v1.log_columns, "category_maps": views.maps,
                "null_policy": "NULL -> 0 after robust scaling; no missing indicators",
                "all_missing_numeric_train": [c for c in views.v1.columns if c not in views.v1.center]}
        dump(self.out / "preprocessing.json", prep)
        for name, columns in views.cols.items():
            dump(self.out / f"{name}_feature_columns.json", {"dataset_version": VERSION, "columns": columns,
                "columns_sha256": canonical_hash(columns), "categorical_columns": ["gender"] if name=="learner" else []})
        dump(self.out / "feature_columns.json", {"dataset_version": VERSION, "raw_feature_columns": list(CANDIDATES),
            "metadata_columns": META, "target_columns": TARGETS, "mask_columns": MASKS,
            "learner": views.cols["learner"], "selector": views.cols["selector"], "held_out_columns": drop})
        self.check("views_exact_selector_subset", views.cols["learner"]==views.cols["selector"]+["gender","phase_fraction"])
        for name, path in self.outputs:
            frame = train if name=="pool_b/train" else self.con.execute(f"SELECT * FROM read_parquet({qs(path)}) ORDER BY row_id").df()
            matrices = views.transform(frame)
            learner, selector = matrices["learner"], matrices["selector"]
            self.check(name+"_finite_ready", np.isfinite(learner.to_numpy()).all())
            self.check(name+"_shared_v1_values", learner[views.cols["selector"]].equals(selector))
            self.check(name+"_phase_fraction_unscaled", learner.phase_fraction.equals(frame.phase_fraction.astype(float)))
            result = pd.concat([frame[META+TARGETS], learner], axis=1)
            self.frame_write(self.out / "model_ready" / (name+".parquet"), result)
            del matrices, learner, selector, result
            if name!="pool_b/train":
                del frame
        self.prep, self.held_out = prep, drop
        del train
        gc.collect()

    def finalize(self):
        # Hash source files once; retain the complete source inventory and confirm
        # the files did not change while this build was running.
        print("Final validation and source checksums", flush=True)
        for path, rec in self.sources.items():
            p = Path(path)
            self.check("source_unchanged_"+canonical_hash(path)[:12], p.stat().st_size==rec["bytes"] and p.stat().st_mtime_ns==rec["mtime_ns"])
            rec["sha256"] = sha256(p)
        dump(self.out / "provenance/source_manifest.json", self.sources)
        provenance = self.out / "provenance"
        for name in [Path(__file__).name, "cq_tempo_features.py", "test_cq_tempo_features.py"]:
            shutil.copyfile(ROOT/name, provenance/name)
        summary = self.con.execute("SELECT phase,split,count(*) n_rows,count(DISTINCT enrollment_id) enrollments FROM raw_all GROUP BY 1,2 ORDER BY 1,2").df()
        dump(self.out / "summary.json", {"counts": json_records(summary), "raw_feature_count": 85,
            "learner_columns": len(self.prep["columns"]["learner"]), "selector_columns": len(self.prep["columns"]["selector"]),
            "held_out_columns": self.held_out, "checks_passed": len(self.checks), "seconds": time.monotonic()-self.started})
        manifest = {"dataset_version": VERSION, "label_version": LABEL_VERSION, "primary_target": "y_alpha_1",
            "source_spine": "phase_features_v4", "feature_source": "CD3", "cd3_commit": CD3_COMMIT,
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(), "input_signature": self.signature,
            "test_data_read": False, "test_in_bundle": False, "sentiment_inference_run": False,
            "model_training_run": False, "software": {"python": sys.version, "duckdb": duckdb.__version__,
                "pandas": pd.__version__, "numpy": np.__version__},
            "helper_sha256": sha256(ROOT/"cq_tempo_features.py"), "builder_sha256": sha256(__file__),
            "raw_paths": [str(p.relative_to(self.out)) for _,p in self.outputs],
            "ready_paths": ["model_ready/"+n+".parquet" for n,_ in self.outputs],
            "limitations": ["Frozen V4 enrollment-day lower bounds and activity-based cutoffs, not CD3 course calendar.",
                "Video clock follows V4; no Asia/Shanghai shift.",
                "CD3 counts full segments selected by start timestamp, including the tail after cutoff.",
                "Source user_problem rows are aggregate states at submit_time, not individual attempts.",
                "Comment text uses existing translated_comment source; Cardiff cache is reused without new inference.",
                "context_id is the complete JSON array, matching CD3 countDistinct(array), not a chapter ID.",
                "QA23 NMI uses ntile and is sensitive to ties; holdout uses rank-min correlation only.",
                "D06 comparison is automatic; human D06 signoff and teacher approval remain separate."]}
        dump(self.out / "run_manifest.json", manifest)
        (self.out / "load_cq_v4_2.py").write_text(LOADER, encoding="utf-8")
        (self.out / "README.md").write_text(f"""# CQ V4 2 CD3 feature view

Built from frozen V4 train/dev/FULL_A. See run_manifest.json for provenance and limitations.
Raw: 85 CD3 features, 3 audit masks and phase_fraction, plus frozen metadata/targets.
Main learner: {len(self.prep['columns']['learner'])} columns; selector: {len(self.prep['columns']['selector'])}.
QA23 held out: {', '.join(self.held_out) or 'none'}; raw retains all columns.
V1 and gender maps are fitted only on Pool B train P1–P4; FULL_A is transform-only.
Feature order is defined by *_feature_columns.json. ID, targets and masks never enter X.

raw_features/pool_b/train.parquet contains pooled early train snapshots.
raw_features/pool_b/dev_phase1.parquet through dev_phase4.parquet contain dev.
raw_features/fulltrain_a/train.parquet contains lambda=1 train snapshots.
model_ready/ mirrors these paths and contains the shared V1 learner columns.
load_cq_v4_2.py loads either view with frozen metadata and q/y aliases; it rejects test.
The raw matrix supports independent preprocessing checks and the D13 include-held-out sensitivity run.
LR gender one-hot, D13/model training and human D06 approval belong to subsequent work.

Reproduce from data_pipeline: python 25_build_phase_features_v4_2.py
Use requirements-v4_2.txt. Existing stage checkpoints are checked against source/code signatures and file hashes.
All datasets and row-level audits remain private local artifacts.
""", encoding="utf-8")
        # Exclude resumable scratch/stage files from the delivery checksum set.
        paths = [p for p in sorted(self.out.rglob("*")) if p.is_file() and not any(
            s in p.relative_to(self.out).parts for s in ["_stage", "_scratch", "__pycache__"])
                 and p.name not in ["SHA256SUMS.txt", "_SUCCESS.json"] and not p.name.endswith(".partial")]
        (self.out / "SHA256SUMS.txt").write_text("".join(f"{sha256(p)}  {p.relative_to(self.out).as_posix()}\n" for p in paths))
        dump(self.out / "_SUCCESS.json", {"dataset_version": VERSION, "checks_passed": len(self.checks),
             "manifest_sha256": sha256(self.out / "run_manifest.json"), "checksums_sha256": sha256(self.out / "SHA256SUMS.txt")})
        print(f"SUCCESS: {self.out}; {len(self.checks)} checks", flush=True)

    def run(self):
        if self.snapshot_splits != ("train", "dev"):
            raise ValueError("Use 27_build_test_features_v4_2.py to extend a frozen train/dev bundle")
        (self.out / "_SUCCESS.json").unlink(missing_ok=True)
        try:
            self.prepare()
            self.build_raw()
            self.audit_events()
            self.compare_d06()
            self.ready()
            self.finalize()
        finally:
            self.con.close()


LOADER = '''"""Read local CD3 V4.2 raw/ready data; test requires a completed test extension."""
from pathlib import Path
import json
import duckdb

ROOT = Path(__file__).resolve().parent

def feature_columns(view="learner", root=None):
    if view not in ("learner", "selector"):
        raise ValueError("view must be learner or selector")
    return json.loads((Path(root or ROOT)/(view+"_feature_columns.json")).read_text())["columns"]

def load(split="train", phase=None, view="learner", raw=False, root=None, connection=None):
    root = Path(root or ROOT)
    if split not in ("train", "dev", "test"):
        raise ValueError("split must be train, dev or test")
    if split == "test" and not json.loads((root/"run_manifest.json").read_text()).get("test_in_bundle",False):
        raise ValueError("Test is not built in this bundle; test remains locked")
    if phase == "FULL_A":
        if split != "train":
            raise ValueError("FULL_A is train only")
        name = "fulltrain_a/train.parquet"
    elif split in ("dev", "test"):
        if phase not in ("P1", "P2", "P3", "P4"):
            raise ValueError("Specify "+split+" phase P1–P4")
        name = "pool_b/"+split+"_phase"+phase[1]+".parquet"
    else:
        if phase is not None and phase not in ("P1", "P2", "P3", "P4"):
            raise ValueError("Unknown phase")
        name = "pool_b/train.parquet"
    path = root/("raw_features" if raw else "model_ready")/name
    con = connection or duckdb.connect()
    rel = con.read_parquet(str(path))
    if phase is not None:
        rel = rel.filter("phase='"+phase+"'")
    if raw:
        return rel
    config = json.loads((root/"feature_columns.json").read_text())
    cols = config["metadata_columns"] + feature_columns(view, root)
    return rel.project(",".join('"'+c+'"' for c in cols)+",q_alpha_1 AS q,y_alpha_1 AS y")
'''


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT/"phase_features_v4_2")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--memory", default="1200MB")
    parser.add_argument("--batch-size", type=int, default=1500)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    Builder(args.root, args.output, args.threads, args.memory, args.batch_size).run()
