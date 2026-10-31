from pyspark.sql import SparkSession, functions as F
from common.pipeline_logging import flush_json_log, get_logger, log_dataframe, log_run_context, log_run_finished, log_write, start_run_timer, write_parquet
from common.protocol_config import load_protocol_config

PROTOCOL = load_protocol_config()
RAW_BASE = PROTOCOL["raw_base"]
OUTPUT_BASE = PROTOCOL["output_base"]


spark = (
    SparkSession.builder.appName("process_video")
    .config("spark.driver.memory", "10g")
    .config("spark.executor.memory", "10g")
    .config("spark.sql.session.timeZone", "UTC")
    .getOrCreate()
)
logger = get_logger("process_video", f"{OUTPUT_BASE}/logs")
log_run_context(logger, spark, {"raw_base": RAW_BASE, "output_base": OUTPUT_BASE})
run_started_at = start_run_timer()

user_video = spark.read.json(f"{RAW_BASE}/3/user-video.json")
enrollments = spark.read.parquet(f"{OUTPUT_BASE}/enrollments/")
course_resources = spark.read.parquet(f"{OUTPUT_BASE}/course_resources_clean/")
log_dataframe(logger, "user_video_raw", user_video, ("user_id",))
log_dataframe(logger, "enrollments_input", enrollments, ("enrollment_id",))

video_events = (
    user_video.withColumn("video", F.explode("seq"))
    .withColumn("segment", F.explode("video.segment"))
    .select(
        "user_id",
        F.col("video.video_id").alias("video_id"),
        F.col("segment.start_point").alias("start_point"),
        F.col("segment.end_point").alias("end_point"),
        F.col("segment.speed").alias("speed"),
        F.col("segment.local_start_time").alias("local_start_time"),
    )
    # Source timestamps represent Asia/Shanghai wall-clock time; store canonical UTC.
    .withColumn(
        "event_time",
        F.to_utc_timestamp(F.from_unixtime("local_start_time").cast("timestamp"), "Asia/Shanghai"),
    )
    .drop("local_start_time")
)
log_dataframe(logger, "video_events_flattened", video_events, ("user_id", "video_id"))

video_events_joined = (
    video_events.join(
        course_resources.filter(F.col("resource_type") == "video").select("course_id", "resource_id"),
        F.col("video_id") == F.col("resource_id"),
        "inner",
    )
    .drop("video_id")
    .join(enrollments, ["user_id", "course_id"], "inner")
)
video_events_excluded = (
    video_events_joined.filter(
        F.col("start_point").isNull()
        | F.col("end_point").isNull()
        | (F.col("end_point") < F.col("start_point"))
        | F.col("speed").isNull()
        | (F.col("speed") <= 0)
        | F.col("event_time").isNull()
        | (F.col("event_time") < F.col("enroll_time"))
    )
    .withColumn(
        "exclusion_reason",
        F.when(F.col("start_point").isNull() | F.col("end_point").isNull(), F.lit("missing_segment_boundary"))
        .when(F.col("end_point") < F.col("start_point"), F.lit("negative_segment_duration"))
        .when(F.col("speed").isNull() | (F.col("speed") <= 0), F.lit("invalid_playback_speed"))
        .when(F.col("event_time").isNull(), F.lit("missing_event_time"))
        .otherwise(F.lit("event_before_enrollment")),
    )
)
video_events_clean = video_events_joined.filter(
    (F.col("end_point") >= F.col("start_point"))
    & (F.col("speed") > 0)
    & F.col("event_time").isNotNull()
    & (F.col("event_time") >= F.col("enroll_time"))
)

clean_rows = log_dataframe(logger, "video_events_clean", video_events_clean, ("enrollment_id", "course_id", "resource_id"))
output_path = f"{OUTPUT_BASE}/video_events_clean/"
exclusion_path = f"{OUTPUT_BASE}/video_events_exclusion_log/"
write_parquet(video_events_clean, output_path)
write_parquet(
    video_events_excluded.select("enrollment_id", "user_id", "course_id", "resource_id", "event_time", "exclusion_reason"),
    exclusion_path,
)
log_write(logger, "video_events_clean", output_path, clean_rows)
log_write(
    logger,
    "video_events_exclusion_log",
    exclusion_path,
    log_dataframe(logger, "video_events_exclusion_log", video_events_excluded, ("exclusion_reason",)),
)
log_run_finished(logger, run_started_at)
flush_json_log(logger, spark)
