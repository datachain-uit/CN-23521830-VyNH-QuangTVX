"""Materialize cutoff-time feature datasets for two temporal scenarios.

Prerequisite: run ``analyze_temporal_blocks.py`` first.

Scenarios
---------
timeline_only
    Only enrollments with a course-specific shifted schedule template.  The
    window ends at the assigned offering end date.
hybrid
    All enrollments.  The same timeline windows are used where available;
    unmatched enrollments use the enrollment-anchored pseudo-offering built by
    ``analyze_temporal_blocks.py``.  Its duration is selected from the
    observed start-month template (or the 131-day median fallback).  This is
    explicitly a proxy, not an observed course end date.

Every feature is calculated from events satisfying
``enroll_time <= event_time <= cutoff_time``.  Labels are deliberately not
joined here: final labels must retain their full-outcome horizon.
"""

import json
import hashlib
import os
import sys
from pathlib import Path
from datetime import datetime, timezone

PROJECT_ROOT = os.environ.get("PROJECT_ROOT")
if not PROJECT_ROOT or not (Path(PROJECT_ROOT) / "common").is_dir():
    _candidates = list(Path("/Workspace/Users").glob("*/CQ/Feature_extraction"))
    if len(_candidates) != 1 or not (_candidates[0] / "common").is_dir():
        raise RuntimeError("Set PROJECT_ROOT to the absolute CQ/Feature_extraction workspace path.")
    PROJECT_ROOT = str(_candidates[0])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pyspark.sql import SparkSession, Window, functions as F

from common.pipeline_logging import (
    flush_json_log,
    get_logger,
    log_dataframe,
    log_event,
    log_run_context,
    log_run_finished,
    log_write,
    start_run_timer,
    write_parquet,
)
from common.protocol_config import load_protocol_config, path_from_config, phase_pairs


PROTOCOL = load_protocol_config()
OUTPUT_BASE = PROTOCOL["output_base"]
RAW_BASE = PROTOCOL["raw_base"]
ANALYSIS_BASE = path_from_config(PROTOCOL, "analysis_base")
FEATURE_BASE = path_from_config(PROTOCOL, "feature_base")
CONTRACT_BASE = path_from_config(PROTOCOL, "contract_base")
RELEASE_VERSION = "scenario_phase_features_v1"
PHASES = phase_pairs(PROTOCOL)
_block_running_total = 0.0
BLOCK_RATIOS = []
for _block_name, _block_ratio in PROTOCOL["temporal_blocks"].items():
    _block_running_total += float(_block_ratio)
    BLOCK_RATIOS.append((_block_name, _block_running_total))
BLOCK_RATIOS = tuple(BLOCK_RATIOS)
FEATURE_KEYS = ("enrollment_id", "user_id", "course_id", "phase", "temporal_block", "timeline_source")


spark = (
    SparkSession.builder.appName("build_scenario_phase_features")
    .config("spark.sql.session.timeZone", "UTC")
    .getOrCreate()
)
logger = get_logger("build_scenario_phase_features", f"{OUTPUT_BASE}/logs")
log_run_context(
    logger,
    spark,
    {
        "output_base": OUTPUT_BASE,
        "scenarios": ["timeline_only", "hybrid"],
        "phases": dict(PHASES),
        "block_targets": dict(BLOCK_RATIOS),
        "event_time_rule": "enroll_time <= event_time <= cutoff_time",
    },
)
started = start_run_timer()


def write_yaml_json(path, payload):
    """JSON is valid YAML and avoids an undeclared YAML dependency."""
    dbutils.fs.put(path, json.dumps(payload, ensure_ascii=False, indent=2, default=str), True)


def inventory_entry(source_id, path, df, primary_keys, foreign_keys, event_time_field=None, availability_time_field=None):
    schema = [{"name": field.name, "type": field.dataType.simpleString()} for field in df.schema.fields]
    return {
        "source_id": source_id,
        "path_or_uri": path,
        "format": "parquet",
        "compression": "snappy",
        "primary_keys": primary_keys,
        "foreign_keys": foreign_keys,
        "event_time_field": event_time_field,
        "availability_time_field": availability_time_field,
        "ingestion_time_field": None,
        "row_count": df.count(),
        "byte_size": None,
        "sha256": None,
        "checksum_status": "unavailable_file_digest_not_computed_by_spark_connect",
        "schema_hash": hashlib.sha256(json.dumps(schema, sort_keys=True).encode("utf-8")).hexdigest(),
        "schema": schema,
        "extraction_time_utc": datetime.now(timezone.utc).isoformat(),
    }


def raw_inventory_entry(source_id, path, data_format, primary_keys, foreign_keys, event_time_field=None):
    """Declare raw provenance even where file-byte checksums await G1 freeze."""
    return {
        "source_id": source_id,
        "path_or_uri": path,
        "format": data_format,
        "compression": None,
        "primary_keys": primary_keys,
        "foreign_keys": foreign_keys,
        "event_time_field": event_time_field,
        "ingestion_time_field": None,
        "row_count": None,
        "byte_size": None,
        "sha256": None,
        "checksum_status": "unavailable_pending_immutable_g1_release",
        "schema_hash": None,
        "schema": None,
        "extraction_time_utc": datetime.now(timezone.utc).isoformat(),
    }


def add_temporal_block(units):
    """Allocate indivisible units sequentially; ties use course/unit ID only."""
    total = units.agg(F.sum("enrollment_count").alias("total_enrollment_count"))
    ordering = Window.orderBy("evaluation_end_date", "course_id", "unit_id").rowsBetween(
        Window.unboundedPreceding, Window.currentRow
    )
    allocated = (
        units.crossJoin(F.broadcast(total))
        .withColumn("cumulative_enrollment_count", F.sum("enrollment_count").over(ordering))
        .withColumn(
            "enrollment_midpoint_fraction",
            (F.col("cumulative_enrollment_count") - F.col("enrollment_count") / F.lit(2.0))
            / F.col("total_enrollment_count"),
        )
        .withColumn("temporal_block", temporal_block_expression())
        .drop("total_enrollment_count")
    )
    return allocated


def temporal_block_expression():
    expression = F.lit(BLOCK_RATIOS[-1][0])
    for block, boundary in reversed(BLOCK_RATIOS[:-1]):
        expression = F.when(F.col("enrollment_midpoint_fraction") <= F.lit(boundary), F.lit(block)).otherwise(expression)
    return expression


def add_phase_cutoffs(windows):
    phase_specs = spark.createDataFrame(PHASES, ["phase", "phase_ratio"])
    # End is exclusive (midnight after the end date) so a date-based course
    # end includes the complete final calendar day.
    return (
        windows.filter(F.col("window_end_exclusive") > F.col("enroll_time"))
        .crossJoin(F.broadcast(phase_specs))
        .withColumn(
            "cutoff_time",
            F.to_timestamp(
                F.from_unixtime(
                    F.unix_timestamp("enroll_time")
                    + (F.unix_timestamp("window_end_exclusive") - F.unix_timestamp("enroll_time"))
                    * F.col("phase_ratio")
                )
            ),
        )
        .withColumn("window_duration_seconds", F.unix_timestamp("window_end_exclusive") - F.unix_timestamp("enroll_time"))
    )


def select_events_until_cutoff(events, phases):
    # The clean event tables already retain their source ``enroll_time``.
    # Keep the enrollment timestamp from the phase spine under a distinct name;
    # otherwise Spark Connect sees two equally named columns after the join.
    phase_columns = list(FEATURE_KEYS) + ["cutoff_time"]
    phase_spine = phases.select(
        *phase_columns,
        F.col("enroll_time").alias("cutoff_enroll_time"),
    )
    return (
        events.join(phase_spine, ["enrollment_id", "user_id", "course_id"], "inner")
        .filter(
            (F.col("availability_time") >= F.col("cutoff_enroll_time"))
            & (F.col("availability_time") <= F.col("cutoff_time"))
        )
        .drop("cutoff_enroll_time")
    )


def video_features(events, course_metadata):
    keys = list(FEATURE_KEYS)
    x = (
        events.withColumn("segment_duration", (F.col("end_point") - F.col("start_point")).cast("double"))
        .withColumn("real_watch_time", F.try_divide("segment_duration", "speed"))
        .withColumn("day_of_week", F.dayofweek("event_time"))
        .withColumn("hour", F.hour("event_time"))
        .withColumn("watch_day", F.to_date("event_time"))
        .withColumn("days_from_enroll", F.datediff("event_time", "enroll_time").cast("double"))
    )
    base = (
        x.groupBy(*keys)
        .agg(
            F.sum("segment_duration").alias("segment_duration_seconds"),
            F.avg("segment_duration").alias("avg_segment_duration"),
            F.avg("real_watch_time").alias("avg_real_watch_time"),
            F.count("*").alias("segment_count"),
            F.avg("speed").alias("avg_playback_speed"),
            F.avg("day_of_week").alias("avg_day_of_week"),
            F.countDistinct("resource_id").alias("unique_video_count"),
            F.countDistinct("watch_day").alias("video_active_day_count"),
            F.min("days_from_enroll").alias("first_video_day_from_enroll"),
            F.max("days_from_enroll").alias("last_video_day_from_enroll"),
            F.stddev("days_from_enroll").alias("video_day_from_enroll_std"),
            F.min("segment_duration").alias("min_segment_duration"),
            F.max("segment_duration").alias("max_segment_duration"),
            F.stddev("segment_duration").alias("segment_duration_std"),
            F.coalesce(F.variance("speed"), F.lit(0.0)).alias("var_playback_speed"),
            F.avg(F.when(F.col("speed") > 1, 1.0).otherwise(0.0)).alias("fast_forward_ratio"),
        )
        .withColumn("avg_segment_count_per_video", F.try_divide("segment_count", "unique_video_count"))
        .withColumn("video_repeat_segment_ratio", F.try_divide(F.col("segment_count") - F.col("unique_video_count"), "segment_count"))
        .withColumn("video_events_per_active_day", F.try_divide("segment_count", "video_active_day_count"))
        .join(course_metadata.select("course_id", F.col("video_counts").alias("course_video_count")), "course_id", "left")
        .withColumn("video_catalog_coverage", F.try_divide("unique_video_count", "course_video_count"))
    )

    def mode(column_name, output_name):
        counts = x.groupBy(*keys, column_name).count()
        rank = Window.partitionBy(*keys).orderBy(F.desc("count"), F.asc(column_name))
        return counts.withColumn("rank", F.row_number().over(rank)).filter("rank = 1").select(
            *keys, F.col(column_name).alias(output_name)
        )

    event_order = Window.partitionBy(*keys).orderBy("event_time", "resource_id")
    cumulative = event_order.rowsBetween(Window.unboundedPreceding, Window.currentRow)
    sessions = (
        x.withColumn("previous_event_time", F.lag("event_time").over(event_order))
        .withColumn(
            "new_session",
            F.when(
                F.col("previous_event_time").isNull()
                | ((F.unix_timestamp("event_time") - F.unix_timestamp("previous_event_time")) > F.lit(30 * 60)),
                F.lit(1),
            ).otherwise(F.lit(0)),
        )
        .withColumn("session_id", F.sum("new_session").over(cumulative))
    )
    session_std = (
        sessions.groupBy(*keys, "session_id")
        .agg(
            (
                F.max(F.unix_timestamp("event_time") + F.col("real_watch_time"))
                - F.min(F.unix_timestamp("event_time"))
            ).alias("session_duration_seconds")
        )
        .groupBy(*keys)
        .agg(F.coalesce(F.stddev("session_duration_seconds"), F.lit(0.0)).alias("session_duration_std"))
    )
    return base.join(mode("hour", "most_common_hour"), keys, "left").join(
        mode("watch_day", "most_common_day"), keys, "left"
    ).join(session_std, keys, "left")


def problem_features(events, problem_catalog_totals):
    keys = list(FEATURE_KEYS)
    x = (events.withColumn("score_clean", F.coalesce(F.col("score").cast("double"), F.lit(0.0)))
        .withColumn("score_fraction", F.try_divide("score_clean", "full_score"))
        .withColumn("submit_day_from_enroll", F.datediff("event_time", "enroll_time").cast("double"))
        .withColumn("is_correct_clean", F.when(F.col("is_correct") == 1, F.lit(1.0)).otherwise(F.lit(0.0))))
    return x.groupBy(*keys).agg(
        F.countDistinct("problem_id").alias("problem_count"),
        F.countDistinct("exercise_id").alias("exercise_count"),
        F.countDistinct("context_id").alias("problem_context_count"),
        F.sum("score_clean").alias("score_sum"),
        F.avg("score_clean").alias("score_mean"),
        F.stddev("score_clean").alias("score_std"),
        F.avg("score_fraction").alias("score_fraction_mean"),
        F.stddev("score_fraction").alias("score_fraction_std"),
        F.sum("is_correct_clean").alias("correct_problem_count"),
        F.sum(F.when(F.col("is_correct_clean") == 0, 1.0).otherwise(0.0)).alias("incorrect_problem_count"),
        F.avg("is_correct_clean").alias("correct_ratio"),
        F.stddev("is_correct_clean").alias("correct_ratio_std"),
        F.sum(F.coalesce(F.col("attempts").cast("double"), F.lit(0.0))).alias("attempts_sum"),
        F.avg(F.coalesce(F.col("attempts").cast("double"), F.lit(0.0))).alias("attempts_mean"),
        F.stddev(F.coalesce(F.col("attempts").cast("double"), F.lit(0.0))).alias("attempts_std"),
        F.min("submit_day_from_enroll").alias("first_submit_day_from_enroll"),
        F.max("submit_day_from_enroll").alias("last_submit_day_from_enroll"),
        F.avg("submit_day_from_enroll").alias("avg_submit_day_from_enroll"),
        F.stddev("submit_day_from_enroll").alias("submit_day_from_enroll_std"),
    ).join(problem_catalog_totals, "course_id", "left").withColumn(
        "problem_catalog_coverage", F.try_divide("problem_count", "course_problem_count")
    ).withColumn("problem_score_catalog_coverage", F.try_divide("score_sum", "course_problem_full_score_sum"))


def comment_features(events, course_metadata):
    keys = list(FEATURE_KEYS)
    text_column = "text_translated" if "text_translated" in events.columns else "text"
    x = (events.withColumn("text_length", F.length(F.trim(F.col(text_column))).cast("double"))
        .withColumn("comment_day_from_enroll", F.datediff("event_time", "enroll_time").cast("double")))
    return x.groupBy(*keys).agg(
        F.count("*").alias("comment_count"),
        F.countDistinct("resource_id").alias("unique_commented_resource_count"),
        F.avg("text_length").alias("avg_comment_length_chars"),
        F.min("text_length").alias("min_comment_length_chars"),
        F.max("text_length").alias("max_comment_length_chars"),
        F.stddev("text_length").alias("comment_length_std_chars"),
        F.avg("positive_score").alias("avg_positive_score"),
        F.avg("negative_score").alias("avg_negative_score"),
        F.avg("neutral_score").alias("avg_neutral_score"),
        F.stddev("positive_score").alias("positive_score_std"),
        F.stddev("negative_score").alias("negative_score_std"),
        F.stddev("neutral_score").alias("neutral_score_std"),
        F.sum(F.when(F.col("sentiment_label") == "positive", 1).otherwise(0)).alias("positive_comment_count"),
        F.sum(F.when(F.col("sentiment_label") == "neutral", 1).otherwise(0)).alias("neutral_comment_count"),
        F.sum(F.when(F.col("sentiment_label") == "negative", 1).otherwise(0)).alias("negative_comment_count"),
        F.avg(F.when(F.col("sentiment_label") == "positive", 1.0).otherwise(0.0)).alias("positive_comment_ratio"),
        F.avg(F.when(F.col("sentiment_label") == "negative", 1.0).otherwise(0.0)).alias("negative_comment_ratio"),
        F.min("comment_day_from_enroll").alias("first_comment_day_from_enroll"),
        F.max("comment_day_from_enroll").alias("last_comment_day_from_enroll"),
        F.avg("comment_day_from_enroll").alias("avg_comment_day_from_enroll"),
        F.stddev("comment_day_from_enroll").alias("comment_day_from_enroll_std"),
    ).join(course_metadata.select("course_id", F.col("resource_count").alias("course_resource_count")), "course_id", "left").withColumn(
        "commented_resource_catalog_coverage", F.try_divide("unique_commented_resource_count", "course_resource_count")
    )


def merge_features(phase_windows, video, problem, comment, enrollment_context, course_context, teacher_context, school_context):
    keys = list(FEATURE_KEYS)
    spine = phase_windows.select("scenario", *keys, "cutoff_time", "window_end_exclusive", "window_duration_seconds")
    return (
        spine.join(video, keys, "left")
        .join(problem, keys, "left")
        .join(comment, keys, "left")
        .join(enrollment_context, "enrollment_id", "left")
        .join(course_context, "course_id", "left")
        .join(teacher_context, "course_id", "left")
        .join(school_context, "course_id", "left")
        .withColumn(
            "age_at_enroll",
            F.when(F.col("year_of_birth").isNotNull(), F.year("enroll_time_context") - F.col("year_of_birth").cast("int")),
        )
        .drop("enroll_time_context")
        .withColumn("video_observed_mask", F.col("segment_count").isNotNull().cast("int"))
        .withColumn("problem_observed_mask", F.col("problem_count").isNotNull().cast("int"))
        .withColumn("comment_observed_mask", F.col("comment_count").isNotNull().cast("int"))
    )


timeline_assignments = spark.read.parquet(f"{ANALYSIS_BASE}/shifted_offering_assignments/")

# Keep the no-timeline population aligned with the temporal analysis policy:
# the first observed enrollment in each course anchors a sequential inferred
# offering; later enrollments are assigned until the inferred end.  The old
# global-template fallback only assigned enrollments that happened to fall in
# one generic calendar window, silently leaving most no-timeline enrollments
# out of the hybrid feature population.
anchored_fallback_assignments = (
    spark.read.parquet(f"{ANALYSIS_BASE}/enrollment_anchored_pseudo_offering_assignments/")
    .select(
        "enrollment_id", "user_id", "course_id", "enroll_time",
        F.col("pseudo_offering_id").alias("offering_id"),
        F.col("pseudo_start_date").alias("offering_start_date"),
        F.col("pseudo_end_date").alias("offering_end_date"),
        "duration_days", "duration_source",
    )
    .withColumn("timeline_source", F.lit("enrollment_anchored_proxy"))
)
# A secondary fallback is retained for enrollments belonging to a course that
# has some schedule metadata but did not match any shifted run.  It must not
# override the anchored assignment for a genuinely no-timeline course.
global_fallback_assignments = (
    spark.read.parquet(f"{ANALYSIS_BASE}/global_template_offering_assignments/")
    .select(
        "enrollment_id", "user_id", "course_id", "enroll_time", "offering_id",
        "offering_start_date", "offering_end_date",
    )
    .withColumn("duration_days", F.datediff("offering_end_date", "offering_start_date"))
    .withColumn("duration_source", F.lit("global_template_fallback"))
    .withColumn("timeline_source", F.lit("global_template_fallback"))
    .join(anchored_fallback_assignments.select("enrollment_id"), "enrollment_id", "left_anti")
)
fallback_assignments = anchored_fallback_assignments.unionByName(global_fallback_assignments)
enrollments = spark.read.parquet(f"{OUTPUT_BASE}/enrollments/")
course_summary = spark.read.parquet(f"{OUTPUT_BASE}/course_summary/")
teacher_summary = spark.read.parquet(f"{OUTPUT_BASE}/course_teacher_summary/")
school_summary = spark.read.parquet(f"{OUTPUT_BASE}/course_school_summary/")
course_problem_catalog = spark.read.parquet(f"{OUTPUT_BASE}/course_problem_catalog/")
video_events = spark.read.parquet(f"{OUTPUT_BASE}/video_events_clean/").withColumn(
    "availability_time", F.col("event_time")
)
problem_events = spark.read.parquet(f"{OUTPUT_BASE}/problem_events_clean/")
if "availability_time" not in problem_events.columns:
    problem_events = problem_events.withColumn("availability_time", F.to_timestamp("submit_time"))
comment_events = spark.read.parquet(path_from_config(PROTOCOL, "comment_events_clean") + "/").withColumn(
    "availability_time", F.col("event_time")
)

# Catalog/context data are joined separately from event aggregates.  The full
# corpus enrollment count is deliberately excluded because it leaks the future.
problem_catalog_totals = course_problem_catalog.groupBy("course_id").agg(
    F.countDistinct("problem_id").alias("course_problem_count"),
    F.sum("full_score").alias("course_problem_full_score_sum"),
)
enrollment_context = enrollments.select(
    "enrollment_id", "gender", "year_of_birth", F.col("enroll_time").alias("enroll_time_context")
)
course_context = course_summary.select("course_id", "resource_count", "video_counts", "ex_counts")
teacher_context = teacher_summary.select(
    "course_id", "teacher_count", "teacher_bio_coverage", "teacher_avg_bio_length_chars", "teacher_org_count"
)
school_context = school_summary.select(
    "course_id", "school_count", "school_bio_coverage", "school_avg_bio_length_chars",
    "school_motto_coverage", "school_avg_motto_length_chars"
)
STATIC_CONTEXT_FEATURES = (
    "gender", "year_of_birth", "age_at_enroll",
    "resource_count", "video_counts", "ex_counts",
    "teacher_count", "teacher_bio_coverage", "teacher_avg_bio_length_chars", "teacher_org_count",
    "school_count", "school_bio_coverage", "school_avg_bio_length_chars",
    "school_motto_coverage", "school_avg_motto_length_chars",
)
OBSERVATION_MASK_FEATURES = ("video_observed_mask", "problem_observed_mask", "comment_observed_mask")

# The availability rule is explicit and auditable. Problem scoring is confirmed
# automatic at submission; split and label-availability audits remain separate.
availability_policy = {
    "version": "availability_time_v2_locked_auto_grading",
    "video_events": "availability_time = event_time (learner interaction is immediately observable)",
    "comment_events": "availability_time = event_time; sentiment enrichment must not alter availability time",
    "problem_events": "availability_time = submit_time; automatic scoring makes score/is_correct available on submission",
    "primary_protocol_status": "event rules are locked; release checksum audit remains required",
}
log_event(logger, "availability_time_policy", **availability_policy)

source_inventory = [
    inventory_entry(
        "enrollments",
        f"{OUTPUT_BASE}/enrollments/",
        enrollments,
        ["enrollment_id"],
        ["user_id", "course_id"],
        "enroll_time",
    ),
    inventory_entry("course_summary", f"{OUTPUT_BASE}/course_summary/", course_summary, ["course_id"], []),
    inventory_entry(
        "course_teacher_summary", f"{OUTPUT_BASE}/course_teacher_summary/", teacher_summary, ["course_id"], ["course_id"]
    ),
    inventory_entry(
        "course_school_summary", f"{OUTPUT_BASE}/course_school_summary/", school_summary, ["course_id"], ["course_id"]
    ),
    inventory_entry(
        "course_problem_catalog",
        f"{OUTPUT_BASE}/course_problem_catalog/",
        course_problem_catalog,
        ["course_id", "problem_id"],
        ["course_id", "exercise_id"],
    ),
    inventory_entry(
        "course_specific_offering_assignments",
        f"{ANALYSIS_BASE}/shifted_offering_assignments/",
        timeline_assignments,
        ["enrollment_id"],
        ["user_id", "course_id", "offering_id"],
        "enroll_time",
    ),
    inventory_entry(
        "enrollment_anchored_proxy_assignments",
        f"{ANALYSIS_BASE}/enrollment_anchored_pseudo_offering_assignments/",
        anchored_fallback_assignments,
        ["enrollment_id"],
        ["user_id", "course_id", "offering_id"],
        "enroll_time",
    ),
    inventory_entry(
        "global_template_secondary_assignments",
        f"{ANALYSIS_BASE}/global_template_offering_assignments/",
        global_fallback_assignments,
        ["enrollment_id"],
        ["user_id", "course_id", "offering_id"],
        "enroll_time",
    ),
    inventory_entry(
        "video_events_clean",
        f"{OUTPUT_BASE}/video_events_clean/",
        video_events,
        ["enrollment_id", "event_time", "resource_id"],
        ["user_id", "course_id"],
        "event_time",
        "availability_time",
    ),
    inventory_entry(
        "problem_events_clean",
        f"{OUTPUT_BASE}/problem_events_clean/",
        problem_events,
        ["enrollment_id", "event_time", "problem_id"],
        ["user_id", "course_id"],
        "event_time",
        "availability_time",
    ),
    inventory_entry(
        "comment_events_clean",
        path_from_config(PROTOCOL, "comment_events_clean") + "/",
        comment_events,
        ["enrollment_id", "comment_id"],
        ["user_id", "course_id", "resource_id"],
        "event_time",
        "availability_time",
    ),
]
raw_source_inventory = [
    raw_inventory_entry("raw_user", f"{RAW_BASE}/3/user.json", "json", ["id"], []),
    raw_inventory_entry("raw_course_resources", f"{RAW_BASE}/course.csv", "csv", ["course_id", "resource_id"], ["course_id"]),
    raw_inventory_entry("raw_course_schedule", f"{RAW_BASE}/course_limit.csv", "csv", ["course_id"], ["course_id"], "start_date/end_date"),
    raw_inventory_entry("raw_user_problem", f"{RAW_BASE}/3/user-problem.json", "json", ["user_id", "problem_id", "submit_time"], ["user_id", "problem_id"], "submit_time"),
    raw_inventory_entry("raw_problem", f"{RAW_BASE}/3/problem.json", "json", ["problem_id"], ["exercise_id"]),
    raw_inventory_entry("raw_user_video", f"{RAW_BASE}/3/user-video.json", "json", ["user_id"], ["user_id"], "segment.local_start_time"),
    raw_inventory_entry("raw_comment", f"{RAW_BASE}/3/comment.json", "json", ["id"], ["user_id", "resource_id"], "create_time"),
    raw_inventory_entry("raw_comment_sentiment", f"{RAW_BASE}/3/comments_twitter_roberta_base.json", "json", ["id"], ["id"]),
    raw_inventory_entry("raw_teacher", f"{RAW_BASE}/3/teacher.json", "json", ["id"], []),
    raw_inventory_entry("raw_school", f"{RAW_BASE}/3/school.json", "json", ["id"], []),
    raw_inventory_entry("raw_course_score_structure", f"{RAW_BASE}/course_ScoreStruct.csv", "csv", ["id", "activities"], ["id"]),
]
write_yaml_json(
    f"{CONTRACT_BASE}/data_inventory.yaml",
    {"version": RELEASE_VERSION, "raw_sources": raw_source_inventory, "processed_sources": source_inventory},
)
write_yaml_json(
    f"{CONTRACT_BASE}/entity_key_spec.yaml",
    {
        "version": "canonical_entity_keys_v1",
        "learner_id": "user_id",
        "course_family_id": "course_id; source has no separate course-family key",
        "course_id": "course_id",
        "offering_id": "offering_id for observed assignments; pseudo_offering_id for enrollment-anchored proxy assignments",
        "enrollment_id": "deterministic user_id::course_id::UTC enroll_time string (process_user.py)",
        "chapter_id": "not modelled; chapter logic intentionally removed",
        "resource_id": "course resource identifier",
        "event_id": {
            "video": "enrollment_id + resource_id + event_time + segment coordinates",
            "problem": "enrollment_id + problem_id + submit_time",
            "comment": "comment_id",
        },
    },
)
write_yaml_json(
    f"{CONTRACT_BASE}/phase_spec.yaml",
    {
        "version": "phase_cutoff_calendar_proxy_v3_cumulative_prefix",
        "phases": [{"phase": phase, "ratio": ratio, "cumulative": True} for phase, ratio in PHASES],
        "window_start": "enroll_time",
        "window_end": "window_end_exclusive",
        "event_inclusion_rule": "enroll_time <= availability_time <= cutoff_time",
        "prefix_sequence_rule": "For prediction phase Pk, slots P1..Pk are available cumulative prefixes; P(k+1)..P4 are future-unavailable (NULL plus phase_available_mask=0).",
        "timeline_only": "course_specific shifted offering end date",
        "hybrid_fallback": "enrollment-anchored pseudo offering; reliable start-month duration template or 131-day median fallback; proxy only",
        "status": "locked_grouped_temporal_calendar_policy",
    },
)
write_yaml_json(
    f"{CONTRACT_BASE}/cq_label_spec.yaml",
    {
        "version": PROTOCOL["labels"]["cq_canonical_version"],
        "task": "CQ",
        "label_artifact": path_from_config(PROTOCOL, "cq_labels") + "/",
        "label_field": "CQ_label_final",
        "rule": "final TRIAD from full learner-course behavior; fixed across P1-P4",
        "label_availability_field": "label_availability_time",
        "label_availability_source_field": "label_availability_source",
        "required_fields": ["enrollment_id", "COELO_final", "AFELO_final", "ACELO_final", "TRIAD_distance_final", "CQ_label_final", "observed_dimension_mask", "label_availability_time", "label_rule_version"],
        "feature_dependency_rule": "phase features may only use availability_time <= cutoff_time",
    },
)
write_yaml_json(
    f"{CONTRACT_BASE}/lo_label_spec.yaml",
    {
        "version": PROTOCOL["labels"]["lo_version"],
        "task": "LO_operational",
        "required_fields": ["enrollment_id", "performance_score", "LO_performance_label_3", "LO_performance_label_5", "label_availability_time", "label_rule_version", "proxy_exclusion_reason"],
        "label_artifact": path_from_config(PROTOCOL, "lo_labels") + "/",
        "primary_status": "operational_behavioural_label_enabled",
        "decision": PROTOCOL["labels"]["lo_decision"],
        "rule": "video*watch_percent + assignment*mean_assignment_ratio + exam*sum_exam_ratio; final-chapter exercises are exam when exam weight > 0",
        "semantic_scope": PROTOCOL["labels"]["lo_semantics"],
        "exclusion_reason_field": "proxy_exclusion_reason",
        "label_availability_field": "label_availability_time",
    },
)
write_yaml_json(
    f"{CONTRACT_BASE}/logging_spec.yaml",
    {
        "version": "structured_jsonl_v1",
        "log_directory": f"{OUTPUT_BASE}/logs",
        "required_run_context": ["run_id", "stage", "scenario", "paths", "Spark version", "configuration"],
        "required_events": [
            "availability_time_policy",
            "dataframe_metrics",
            "write",
            "phase_builder_audit",
            "exclusion_log",
            "scenario_materialized",
            "run_finished",
        ],
        "failure_policy": "record error status and reason; do not silently fall back",
    },
)

timeline_events = timeline_assignments.select(
    "enrollment_id", "user_id", "course_id", "enroll_time", "offering_id", "offering_end_date"
).withColumn("timeline_source", F.lit("course_specific_timeline")).withColumn(
    "unit_id", F.concat(F.lit("timeline__"), F.col("offering_id"))
)
timeline_units = timeline_events.groupBy("unit_id", "course_id", "offering_end_date", "timeline_source").agg(
    F.count("*").alias("enrollment_count")
).withColumnRenamed("offering_end_date", "evaluation_end_date")

proxy_events = fallback_assignments.select(
    "enrollment_id", "user_id", "course_id", "enroll_time", "offering_id",
    "offering_start_date", "offering_end_date", "duration_source", "timeline_source",
).withColumn("unit_id", F.concat(F.lit("proxy__"), F.col("offering_id")))
proxy_units = proxy_events.groupBy(
    "unit_id", "course_id", "offering_end_date", "timeline_source"
).agg(
    F.count("*").alias("enrollment_count")
).withColumnRenamed("offering_end_date", "evaluation_end_date")

timeline_blocks = add_temporal_block(timeline_units)
hybrid_blocks = add_temporal_block(timeline_units.unionByName(proxy_units))


def scenario_windows(scenario_name, block_units, include_proxy):
    block_lookup = block_units.select("unit_id", "temporal_block", "timeline_source")
    timeline_windows = (
        timeline_events.join(block_lookup, ["unit_id", "timeline_source"], "inner")
        .withColumn("window_end_exclusive", F.to_timestamp(F.date_add("offering_end_date", 1)))
        .withColumn("scenario", F.lit(scenario_name))
        .select(
            "scenario", "enrollment_id", "user_id", "course_id", "enroll_time", "temporal_block", "timeline_source",
            "offering_id", F.col("offering_end_date").alias("window_end_date"), "window_end_exclusive",
        )
    )
    if not include_proxy:
        return timeline_windows
    proxy_windows = (
        proxy_events.join(block_lookup, ["unit_id", "timeline_source"], "inner")
        .withColumn("window_end_date", F.col("offering_end_date"))
        .withColumn("window_end_exclusive", F.to_timestamp(F.date_add("window_end_date", 1)))
        .withColumn("scenario", F.lit(scenario_name))
        .withColumn("offering_id", F.col("unit_id"))
        .select(
            "scenario", "enrollment_id", "user_id", "course_id", "enroll_time", "temporal_block", "timeline_source",
            "offering_id", "window_end_date", "window_end_exclusive",
        )
    )
    return timeline_windows.unionByName(proxy_windows)


scenarios = (
    ("timeline_only", scenario_windows("timeline_only", timeline_blocks, include_proxy=False), timeline_blocks),
    ("hybrid", scenario_windows("hybrid", hybrid_blocks, include_proxy=True), hybrid_blocks),
)

feature_dictionary = []
split_registry_scenarios = []
scenario_exclusion_frames = []

for scenario_name, windows, blocks in scenarios:
    # Assignment generation can contain byte-for-byte duplicate enrollment
    # windows.  Retain one canonical window and record every removed copy.
    raw_windows = windows
    windows = raw_windows.dropDuplicates()
    duplicate_exclusions = (
        raw_windows.exceptAll(windows)
        .withColumn("exclusion_reason", F.lit("duplicate_exact_enrollment_window"))
        .select("scenario", "enrollment_id", "user_id", "course_id", "timeline_source", "exclusion_reason")
    )
    invalid_window_exclusions = (
        windows.filter(F.col("window_end_exclusive") <= F.col("enroll_time"))
        .withColumn("exclusion_reason", F.lit("invalid_proxy_window"))
        .select("scenario", "enrollment_id", "user_id", "course_id", "timeline_source", "exclusion_reason")
    )
    exclusions = duplicate_exclusions.unionByName(invalid_window_exclusions)
    scenario_exclusion_frames.append(exclusions)
    phases = add_phase_cutoffs(windows)
    video_selected = select_events_until_cutoff(video_events, phases)
    problem_selected = select_events_until_cutoff(problem_events, phases)
    comment_selected = select_events_until_cutoff(comment_events, phases)
    video = video_features(video_selected, course_context)
    problem = problem_features(problem_selected, problem_catalog_totals)
    comment = comment_features(comment_selected, course_context)
    merged = merge_features(
        phases, video, problem, comment,
        enrollment_context, course_context, teacher_context, school_context,
    )
    log_event(
        logger,
        "feature_schema",
        scenario=scenario_name,
        video_feature_count=len(video.columns) - len(FEATURE_KEYS),
        problem_feature_count=len(problem.columns) - len(FEATURE_KEYS),
        comment_feature_count=len(comment.columns) - len(FEATURE_KEYS),
        static_context_feature_count=len(STATIC_CONTEXT_FEATURES),
        merged_column_count=len(merged.columns),
    )
    phase_builder_audit = (
        merged.groupBy("scenario", "temporal_block", "timeline_source", "phase")
        .agg(
            F.countDistinct("enrollment_id").alias("enrollment_count"),
            F.sum(
                F.when(
                    (F.col("video_observed_mask") == 0)
                    & (F.col("problem_observed_mask") == 0)
                    & (F.col("comment_observed_mask") == 0),
                    1,
                ).otherwise(0)
            ).alias("zero_event_count"),
            F.avg(
                F.lit(3)
                - F.col("video_observed_mask")
                - F.col("problem_observed_mask")
                - F.col("comment_observed_mask")
            ).alias("mean_missing_count"),
        )
        .withColumn("risk_set_count", F.lit(None).cast("long"))
        .withColumn("post_outcome_count", F.lit(None).cast("long"))
        .withColumn("mean_observed_chapters", F.lit(None).cast("double"))
        .withColumn("mean_unavailable_count", F.lit(0.0))
        .withColumn("label_audit_status", F.lit("not_joined; labels are materialized separately"))
    )

    # Source events are selected by the same cutoff predicate used for every
    # feature aggregation.  Persist conformance counts rather than assuming
    # that a future code edit preserves this invariant.
    def cutoff_violations(events):
        return events.filter(
            (F.col("availability_time") < F.col("enroll_time"))
            | (F.col("availability_time") > F.col("cutoff_time"))
        )

    phase_cardinality_violations = phases.groupBy("enrollment_id").agg(
        F.countDistinct("phase").alias("phase_count")
    ).filter(F.col("phase_count") != F.lit(len(PHASES)))
    conformance = phase_cardinality_violations.agg(
        F.count(F.lit(1)).alias("violation_count")
    ).withColumn("check_name", F.lit("feature_phase_cardinality"))
    for check_name, events in (
        ("video_cutoff_equality", video_selected),
        ("problem_cutoff_equality", problem_selected),
        ("comment_cutoff_equality", comment_selected),
    ):
        conformance = conformance.unionByName(
            cutoff_violations(events).agg(F.count(F.lit(1)).alias("violation_count")).withColumn(
                "check_name", F.lit(check_name)
            )
        )
    conformance = conformance.withColumn(
        "status", F.when(F.col("violation_count") == 0, F.lit("pass")).otherwise(F.lit("fail"))
    )

    scenario_base = f"{FEATURE_BASE}/{scenario_name}"
    paths = {
        "blocks": f"{scenario_base}/temporal_blocks/",
        "windows": f"{scenario_base}/enrollment_windows/",
        "phases": f"{scenario_base}/enrollment_phase_cutoffs/",
        "prefix_manifest": f"{scenario_base}/historical_prefix_manifest/",
        "video": f"{scenario_base}/video_phase_features/",
        "problem": f"{scenario_base}/problem_phase_features/",
        "comment": f"{scenario_base}/comment_phase_features/",
        "merged": f"{scenario_base}/merged_phase_features/",
        "phase_audit": f"{scenario_base}/phase_builder_audit/",
        "conformance": f"{scenario_base}/phase_builder_audit/conformance_checks/",
        "exclusions": f"{scenario_base}/exclusion_log/",
    }
    write_parquet(blocks, paths["blocks"])
    write_parquet(windows, paths["windows"])
    write_parquet(phases, paths["phases"])
    write_parquet(phases, paths["prefix_manifest"])
    write_parquet(video, paths["video"])
    write_parquet(problem, paths["problem"])
    write_parquet(comment, paths["comment"])
    write_parquet(merged, paths["merged"])
    write_parquet(phase_builder_audit, paths["phase_audit"])
    write_parquet(conformance, paths["conformance"])
    write_parquet(exclusions, paths["exclusions"])

    block_summary = blocks.groupBy("temporal_block", "timeline_source").agg(
        F.count("*").alias("unit_count"),
        F.sum("enrollment_count").alias("enrollment_count"),
        F.min("evaluation_end_date").alias("min_evaluation_end_date"),
        F.max("evaluation_end_date").alias("max_evaluation_end_date"),
    )
    window_rows = log_dataframe(logger, f"{scenario_name}_enrollment_windows", windows, ("enrollment_id",))
    phase_rows = log_dataframe(logger, f"{scenario_name}_phase_cutoffs", phases, ("enrollment_id", "phase"))
    video_rows = log_dataframe(logger, f"{scenario_name}_video_phase_features", video, ("enrollment_id", "phase"))
    problem_rows = log_dataframe(logger, f"{scenario_name}_problem_phase_features", problem, ("enrollment_id", "phase"))
    comment_rows = log_dataframe(logger, f"{scenario_name}_comment_phase_features", comment, ("enrollment_id", "phase"))
    merged_rows = log_dataframe(logger, f"{scenario_name}_merged_phase_features", merged, ("enrollment_id", "phase"))
    summary_rows = log_dataframe(logger, f"{scenario_name}_block_summary", block_summary, ("temporal_block", "timeline_source"))
    phase_audit_rows = log_dataframe(logger, f"{scenario_name}_phase_builder_audit", phase_builder_audit, ("temporal_block", "timeline_source", "phase"))
    conformance_rows = log_dataframe(logger, f"{scenario_name}_feature_conformance", conformance, ("check_name",))
    exclusion_rows = log_dataframe(logger, f"{scenario_name}_exclusion_log", exclusions, ("enrollment_id",))
    log_write(logger, f"{scenario_name}_enrollment_windows", paths["windows"], window_rows)
    log_write(logger, f"{scenario_name}_phase_cutoffs", paths["phases"], phase_rows)
    log_write(logger, f"{scenario_name}_historical_prefix_manifest", paths["prefix_manifest"], phase_rows)
    log_write(logger, f"{scenario_name}_video_phase_features", paths["video"], video_rows)
    log_write(logger, f"{scenario_name}_problem_phase_features", paths["problem"], problem_rows)
    log_write(logger, f"{scenario_name}_comment_phase_features", paths["comment"], comment_rows)
    log_write(logger, f"{scenario_name}_merged_phase_features", paths["merged"], merged_rows)
    log_write(logger, f"{scenario_name}_block_summary", paths["blocks"], summary_rows)
    log_write(logger, f"{scenario_name}_phase_builder_audit", paths["phase_audit"], phase_audit_rows)
    log_write(logger, f"{scenario_name}_feature_conformance", paths["conformance"], conformance_rows)
    log_write(logger, f"{scenario_name}_exclusion_log", paths["exclusions"], exclusion_rows)
    log_event(logger, "scenario_materialized", scenario=scenario_name, output_base=scenario_base)

    split_registry_scenarios.append(
        {
            "scenario": scenario_name,
            "ordering_field": "offering_end_date" if scenario_name == "timeline_only" else "offering_end_date | enrollment_month_end",
            "group_unit": "offering_id" if scenario_name == "timeline_only" else "offering_id | course_id x enrollment_month",
            "block_summary": [row.asDict(recursive=True) for row in block_summary.orderBy("temporal_block", "timeline_source").collect()],
        }
    )
    if scenario_name == "timeline_only":
        for feature_df, source_id in ((video, "video_events_clean"), (problem, "problem_events_clean"), (comment, "comment_events_clean")):
            for field in feature_df.schema.fields:
                if field.name not in FEATURE_KEYS:
                    feature_dictionary.append(
                        {
                            "feature_name": field.name,
                            "task_scope": "both",
                            "source_id": source_id,
                            "entity_level": "enrollment_phase",
                            "data_type": field.dataType.simpleString(),
                            "semantic_definition": "Cumulative aggregation of events available at phase cutoff.",
                            "event_time_rule": "enroll_time <= event_time <= cutoff_time",
                            "availability_time_rule": availability_policy[source_id.replace("_clean", "")],
                            "aggregation_window": "P1/P2/P3/P4 cumulative cutoff-time window",
                            "normalization": "none",
                            "missing_semantics": "NULL with observed-mask=0 when no event block is observed",
                            "structural_absence_rule": "not encoded as zero; use block observed mask",
                            "used_by_CQ_label": "indirect",
                            "used_by_LO_label": "proxy",
                            "future_dependency": "no",
                            "allowed_phases": [phase for phase, _ in PHASES],
                            "decision": "primary",
                            "reason": "Event availability is locked by source rule; cutoff conformance is written with each feature release.",
                            "version": RELEASE_VERSION,
                        }
                    )
        merged_types = {field.name: field.dataType.simpleString() for field in merged.schema.fields}
        for feature_name in STATIC_CONTEXT_FEATURES:
            feature_dictionary.append(
                {
                    "feature_name": feature_name,
                    "task_scope": "both",
                    "source_id": "enrollments|course_summary|course_teacher_summary|course_school_summary",
                    "entity_level": "enrollment_phase",
                    "data_type": merged_types[feature_name],
                    "semantic_definition": "Enrollment or course context joined without event aggregation.",
                    "event_time_rule": None,
                    "availability_time_rule": "course/enrollment metadata available at enrollment; verify metadata snapshot timing before primary experiment",
                    "aggregation_window": "not applicable",
                    "normalization": "none",
                    "missing_semantics": "NULL means unavailable source metadata",
                    "structural_absence_rule": "not encoded as zero unless represented as an explicit count",
                    "used_by_CQ_label": "indirect",
                    "used_by_LO_label": "proxy",
                    "future_dependency": "no; course enrollment_count excluded",
                    "allowed_phases": [phase for phase, _ in PHASES],
                    "decision": "ablation_only",
                    "reason": "Metadata snapshot/ingestion time is not observed; retain as a controlled context ablation.",
                    "version": RELEASE_VERSION,
                }
            )
        for feature_name in OBSERVATION_MASK_FEATURES:
            feature_dictionary.append(
                {
                    "feature_name": feature_name,
                    "task_scope": "both",
                    "source_id": "merged_phase_features",
                    "entity_level": "enrollment_phase",
                    "data_type": merged_types[feature_name],
                    "semantic_definition": "One when at least one event from the feature family was observed before cutoff.",
                    "event_time_rule": "enroll_time <= availability_time <= cutoff_time",
                    "availability_time_rule": "inherits the corresponding event family rule",
                    "aggregation_window": "P1/P2/P3/P4 cumulative cutoff-time window",
                    "normalization": "none",
                    "missing_semantics": "0 means no observed event; it is not a numeric imputation",
                    "structural_absence_rule": "retain with its feature family",
                    "used_by_CQ_label": "indirect",
                    "used_by_LO_label": "proxy",
                    "future_dependency": "no",
                    "allowed_phases": [phase for phase, _ in PHASES],
                    "decision": "primary",
                    "reason": "Mask is required to distinguish observed zero/event absence from future-unavailable slots.",
                    "version": RELEASE_VERSION,
                }
            )

all_exclusions = scenario_exclusion_frames[0].unionByName(scenario_exclusion_frames[1])
global_exclusion_path = f"{FEATURE_BASE}/exclusion_log/"
write_parquet(all_exclusions, global_exclusion_path)
global_exclusion_rows = log_dataframe(logger, "scenario_feature_exclusion_log", all_exclusions, ("scenario", "enrollment_id"))
log_write(logger, "scenario_feature_exclusion_log", global_exclusion_path, global_exclusion_rows)

write_yaml_json(
    f"{CONTRACT_BASE}/feature_dictionary.yaml",
    {
        "version": RELEASE_VERSION,
        "availability_policy": availability_policy,
        "features": feature_dictionary,
    },
)
write_yaml_json(
    f"{CONTRACT_BASE}/split_registry.yaml",
    {
        "split_version": PROTOCOL["split"]["version"],
        "scheme": "rolling_origin_expanding",
        "target_block_ratios": BLOCK_RATIOS,
        "ordering_status": "locked: offering_end_date is equivalent to configured label_availability_time (end date + one day)",
        "tie_breaker": "stable course_id, unit_id only within identical evaluation end date",
        "outer_folds": [
            {"window": "W1", "train_blocks": ["A"], "validation_blocks": ["B"], "test_blocks": ["C"]},
            {"window": "W2", "train_blocks": ["A", "B", "C"], "validation_blocks": ["D"], "test_blocks": ["E"]},
            {"window": "W3", "train_blocks": ["A", "B", "C", "D", "E"], "validation_blocks": ["F"], "test_blocks": ["G"]},
        ],
        "random_split": False,
        "scenarios": split_registry_scenarios,
    },
)
write_yaml_json(
    f"{CONTRACT_BASE}/data_release_manifest.json",
    {
        "data_release_id": RELEASE_VERSION,
        "parent_release_id": None,
        "raw_checksums": {item["source_id"]: item["sha256"] for item in raw_source_inventory},
        "processed_checksums": {item["source_id"]: item["sha256"] for item in source_inventory},
        "checksum_status": "pending_release_audit; run validation.audit_processed_release before locking",
        "schema_version": "scenario_phase_schema_v2_context_and_coverage",
        "feature_dictionary_version": RELEASE_VERSION,
        "cq_label_version": PROTOCOL["labels"]["cq_canonical_version"],
        "lo_label_version": PROTOCOL["labels"]["lo_version"],
        "phase_version": "phase_cutoff_calendar_proxy_v3_cumulative_prefix",
        "exclusion_version": "invalid_proxy_window_v1",
        "split_version": PROTOCOL["split"]["version"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "release_status": "auditable_pending_checksum_lock",
    },
)
log_event(
    logger,
    "contract_bundle_written",
    contract_base=CONTRACT_BASE,
    release_version=RELEASE_VERSION,
    primary_protocol_status="event_and_split_policy_locked_pending_release_checksum_audit",
)

log_run_finished(logger, started)
flush_json_log(logger, spark)
