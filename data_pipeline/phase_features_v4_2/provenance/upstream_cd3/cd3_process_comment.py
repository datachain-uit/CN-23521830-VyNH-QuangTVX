from pyspark.sql import SparkSession, functions as F
from common.pipeline_logging import flush_json_log, get_logger, log_dataframe, log_run_context, log_run_finished, log_write, start_run_timer, write_parquet
from common.protocol_config import load_protocol_config, path_from_config


PROTOCOL = load_protocol_config()
RAW_BASE = PROTOCOL["raw_base"]
OUTPUT_BASE = PROTOCOL["output_base"]
COMMENT_OUTPUT_BASE = f"{OUTPUT_BASE}/comment_v1"
COURSE_COMMENT_PATH = f"{RAW_BASE}/3/course-comment.txt"


spark = (
    SparkSession.builder.appName("process_comment_v1")
    .config("spark.driver.memory", "10g")
    .config("spark.executor.memory", "10g")
    .config("spark.sql.session.timeZone", "UTC")
    .getOrCreate()
)
logger = get_logger("process_comment_v1", f"{OUTPUT_BASE}/logs")
log_run_context(logger, spark, {
    "raw_base": RAW_BASE,
    "output_base": OUTPUT_BASE,
    "comment_output_base": COMMENT_OUTPUT_BASE,
    "course_comment_path": COURSE_COMMENT_PATH,
    "comment_join_policy": "course_comment_mapping__raw_comment_presence",
    "sentiment_policy": "not_loaded__afelo_presence_only",
})
run_started_at = start_run_timer()

comment_raw = spark.read.json(f"{RAW_BASE}/3/comment.json")
course_comment_lines = spark.read.text(COURSE_COMMENT_PATH)
log_dataframe(logger, "comment_raw", comment_raw, ("id", "user_id"))
log_dataframe(logger, "course_comment_lines", course_comment_lines, ())

comments = (
    comment_raw.select(
        F.col("id").alias("comment_id"),
        F.col("user_id").cast("string").alias("raw_user_id"),
        F.col("text").alias("text_original"),
        "resource_id",
        F.to_timestamp("create_time").alias("event_time"),
    )
    .withColumn(
        "user_id",
        F.when(F.col("raw_user_id").startswith("U_"), F.col("raw_user_id"))
        .otherwise(F.concat(F.lit("U_"), F.col("raw_user_id"))),
    )
    .drop("raw_user_id")
)

course_comments = (
    course_comment_lines
    .select(F.regexp_replace(F.trim("value"), "^\\ufeff", "").alias("line"))
    .filter(F.length("line") > 0)
    .withColumn("parts", F.split("line", r"\s+"))
    .filter(F.size("parts") >= 2)
    .select(
        F.col("parts").getItem(0).alias("course_id"),
        F.col("parts").getItem(1).alias("comment_id"),
    )
    .filter(F.col("course_id").startswith("C_") & F.col("comment_id").startswith("Cm_"))
    .dropDuplicates(["course_id", "comment_id"])
)
log_dataframe(logger, "course_comments", course_comments, ("course_id", "comment_id"))

# v2 deliberately does not read the incomplete sentiment artifact.  The raw
# comment corpus is the source of truth for AFELO comment presence.  Keep a
# stable nullable sentiment schema for future enrichment without using it in
# the v2 label.
comments_enriched = (
    comments
    .filter(
        F.col("comment_id").isNotNull()
        & F.col("user_id").isNotNull()
        & F.col("event_time").isNotNull()
    )
    .withColumn("text_translated", F.col("text_original"))
    .withColumn("negative_score", F.lit(None).cast("double"))
    .withColumn("neutral_score", F.lit(None).cast("double"))
    .withColumn("positive_score", F.lit(None).cast("double"))
    .withColumn("sentiment_label", F.lit(None).cast("string"))
    .withColumn("sentiment_available", F.lit(0).cast("int"))
)
log_dataframe(logger, "comments_enriched", comments_enriched, ("comment_id", "user_id"))

enrollments = spark.read.parquet(f"{OUTPUT_BASE}/enrollments/")
course_resources = spark.read.parquet(f"{OUTPUT_BASE}/course_resources_clean/")

# Prefer the explicit course-comment mapping because many valid comments have
# resource_id = NULL.  Keep the resource catalog mapping as a fallback for
# comments not present in course-comment.txt.
mapped_by_course_comment = comments_enriched.join(course_comments, "comment_id", "inner")
mapped_by_resource = (
    comments_enriched.filter(F.col("resource_id").isNotNull())
    .join(course_resources.select("course_id", "resource_id"), "resource_id", "inner")
)
comment_course_candidates = (
    mapped_by_course_comment
    .unionByName(mapped_by_resource, allowMissingColumns=True)
    .select(*comments_enriched.columns, "course_id")
    .dropDuplicates(["comment_id", "course_id"])
)
log_dataframe(logger, "comment_course_candidates", comment_course_candidates, ("comment_id", "course_id"))

comment_events_joined = comment_course_candidates.join(enrollments, ["user_id", "course_id"], "inner")
comment_events_before_enroll = (
    comment_events_joined.filter(F.col("event_time") < F.col("enroll_time"))
    .withColumn("exclusion_reason", F.lit("comment_before_enrollment"))
)
comment_events_clean = comment_events_joined.filter(F.col("event_time") >= F.col("enroll_time"))

# Keep valid comments that still cannot be connected to a course for audit/use
# cases; they must not silently inflate AFELO.
comments_unmapped = comments_enriched.join(
    comment_course_candidates.select("comment_id").distinct(),
    "comment_id",
    "left_anti",
)

event_rows = log_dataframe(logger, "comment_events_clean_v2", comment_events_clean, ("comment_id", "enrollment_id", "course_id"))
unmapped_rows = log_dataframe(logger, "comments_unmapped", comments_unmapped, ("comment_id",))
enriched_path = path_from_config(PROTOCOL, "comment_enriched") + "/"
events_path = path_from_config(PROTOCOL, "comment_events_clean") + "/"
unmapped_path = path_from_config(PROTOCOL, "comment_unmapped") + "/"
exclusion_path = f"{COMMENT_OUTPUT_BASE}/comment_events_exclusion_log/"
write_parquet(comments_enriched, enriched_path)
write_parquet(comment_events_clean, events_path)
write_parquet(comments_unmapped, unmapped_path)
write_parquet(
    comment_events_before_enroll.select("comment_id", "user_id", "course_id", "enrollment_id", "event_time", "exclusion_reason"),
    exclusion_path,
)
log_write(logger, "comments_enriched", enriched_path, log_dataframe(logger, "comments_enriched_output", comments_enriched, ("comment_id",)))
log_write(logger, "comment_events_clean_v2", events_path, event_rows)
log_write(logger, "comments_unmapped", unmapped_path, unmapped_rows)
log_write(
    logger,
    "comment_events_exclusion_log",
    exclusion_path,
    log_dataframe(logger, "comment_events_exclusion_log", comment_events_before_enroll, ("exclusion_reason",)),
)
log_run_finished(logger, run_started_at)
flush_json_log(logger, spark)
