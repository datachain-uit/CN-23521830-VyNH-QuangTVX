from pyspark.sql import SparkSession, functions as F
from common.pipeline_logging import flush_json_log, get_logger, log_dataframe, log_run_context, log_run_finished, log_write, start_run_timer, write_parquet
from common.protocol_config import load_protocol_config

PROTOCOL = load_protocol_config()
RAW_BASE = PROTOCOL["raw_base"]
OUTPUT_BASE = PROTOCOL["output_base"]


spark = (
    SparkSession.builder.appName("process_problem")
    .config("spark.driver.memory", "10g")
    .config("spark.executor.memory", "10g")
    .config("spark.sql.session.timeZone", "UTC")
    .getOrCreate()
)
logger = get_logger("process_problem", f"{OUTPUT_BASE}/logs")
log_run_context(logger, spark, {"raw_base": RAW_BASE, "output_base": OUTPUT_BASE})
run_started_at = start_run_timer()

problem = (
    spark.read.json(f"{RAW_BASE}/3/problem.json")
    .withColumnRenamed("score", "full_score")
    .withColumn("problem_id", F.concat(F.lit("Pm_"), F.col("problem_id")))
    .select("exercise_id", "problem_id", "full_score", "context_id")
)
user_problem = spark.read.json(f"{RAW_BASE}/3/user-problem.json")
enrollments = spark.read.parquet(f"{OUTPUT_BASE}/enrollments/")
course_resources = spark.read.parquet(f"{OUTPUT_BASE}/course_resources_clean/")
log_dataframe(logger, "user_problem_raw", user_problem, ("user_id", "problem_id"))

course_problem_catalog = (
    problem.join(
        # ``chapter`` is retained only for the LO-performance proxy: the
        # reference LO pipeline treats exercises in the final chapter as exam
        # exercises.  It is not a feature column.
        course_resources.filter(F.col("resource_type") == "exercise").select("course_id", "resource_id", "chapter"),
        F.col("exercise_id") == F.col("resource_id"),
        "inner",
    )
    .select("course_id", "problem_id", "exercise_id", "context_id", "full_score", "chapter")
    .dropDuplicates(["course_id", "problem_id"])
)
log_dataframe(logger, "course_problem_catalog", course_problem_catalog, ("course_id", "problem_id"))

problem_events_joined = (
    user_problem.join(course_problem_catalog, "problem_id", "inner")
    .join(enrollments, ["user_id", "course_id"], "inner")
    # Problem scoring is automatic: score/is_correct are available at the same
    # instant as submit_time, which is therefore the availability timestamp.
    .withColumn("availability_time", F.to_timestamp("submit_time"))
    .withColumn("event_time", F.col("availability_time"))
)
problem_events_excluded = (
    problem_events_joined.filter(
        F.col("availability_time").isNull()
        | (F.col("availability_time") < F.col("enroll_time"))
        | F.col("full_score").isNull()
        | (F.col("full_score") <= 0)
        | (F.col("attempts") < 0)
        | F.col("attempts").isNull()
    )
    .withColumn(
        "exclusion_reason",
        F.when(F.col("availability_time").isNull(), F.lit("missing_submit_time"))
        .when(F.col("availability_time") < F.col("enroll_time"), F.lit("event_before_enrollment"))
        .when(F.col("full_score").isNull(), F.lit("missing_full_score"))
        .when(F.col("full_score") <= 0, F.lit("nonpositive_full_score"))
        .otherwise(F.lit("invalid_or_missing_attempts")),
    )
)
problem_events_clean = problem_events_joined.filter(
    F.col("availability_time").isNotNull()
    & (F.col("availability_time") >= F.col("enroll_time"))
    & (F.col("full_score") > 0)
    & (F.col("attempts") >= 0)
)

event_rows = log_dataframe(logger, "problem_events_clean", problem_events_clean, ("enrollment_id", "course_id", "problem_id"))
catalog_path = f"{OUTPUT_BASE}/course_problem_catalog/"
events_path = f"{OUTPUT_BASE}/problem_events_clean/"
exclusion_path = f"{OUTPUT_BASE}/problem_events_exclusion_log/"
write_parquet(course_problem_catalog, catalog_path)
write_parquet(problem_events_clean, events_path)
write_parquet(
    problem_events_excluded.select("enrollment_id", "user_id", "course_id", "problem_id", "availability_time", "exclusion_reason"),
    exclusion_path,
)
log_write(logger, "course_problem_catalog", catalog_path, log_dataframe(logger, "course_problem_catalog_output", course_problem_catalog, ("course_id", "problem_id")))
log_write(logger, "problem_events_clean", events_path, event_rows)
log_write(
    logger,
    "problem_events_exclusion_log",
    exclusion_path,
    log_dataframe(logger, "problem_events_exclusion_log", problem_events_excluded, ("exclusion_reason",)),
)
log_run_finished(logger, run_started_at)
flush_json_log(logger, spark)
