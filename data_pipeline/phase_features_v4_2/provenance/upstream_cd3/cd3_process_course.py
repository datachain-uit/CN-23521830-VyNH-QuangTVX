from pyspark.sql import SparkSession, functions as F
from common.pipeline_logging import flush_json_log, get_logger, log_dataframe, log_run_context, log_write, write_parquet
from common.protocol_config import load_protocol_config

PROTOCOL = load_protocol_config()
RAW_BASE = PROTOCOL["raw_base"]
OUTPUT_BASE = PROTOCOL["output_base"]


spark = (
    SparkSession.builder.appName("process_course")
    .config("spark.driver.memory", "10g")
    .config("spark.executor.memory", "10g")
    .config("spark.sql.session.timeZone", "UTC")
    .getOrCreate()
)
logger = get_logger("process_course", f"{OUTPUT_BASE}/logs")
log_run_context(logger, spark, {"raw_base": RAW_BASE, "output_base": OUTPUT_BASE})

enrollments = spark.read.parquet(f"{OUTPUT_BASE}/enrollments/")
log_dataframe(logger, "enrollments_input", enrollments, ("enrollment_id", "course_id"))
enrolled_courses = enrollments.select("course_id").distinct()

# `course.csv` is already resource-level in this dataset: course, chapter, resource.
course_raw = (
    spark.read.option("header", True)
    .option("quote", '"')
    .option("escape", '"')
    .option("multiLine", True)
    .option("mode", "PERMISSIVE")
    .csv(f"{RAW_BASE}/course.csv")
    .withColumnRenamed("id", "course_id")
)
log_dataframe(logger, "course_raw", course_raw, ("course_id", "resource_id"))

course_resources_clean = (
    course_raw.select("course_id", "chapter", "resource_id")
    .filter(F.col("course_id").isNotNull() & F.col("resource_id").isNotNull())
    .withColumn(
        "resource_type",
        F.when(F.col("resource_id").startswith("V_"), F.lit("video"))
        .when(F.col("resource_id").startswith("Ex_"), F.lit("exercise"))
        .otherwise(F.lit("other")),
    )
    .join(enrolled_courses, "course_id", "inner")
    .dropDuplicates(["course_id", "resource_id"])
)
resource_rows = log_dataframe(logger, "course_resources_clean", course_resources_clean, ("course_id", "resource_id"))

course_summary = (
    course_resources_clean.groupBy("course_id").agg(
        F.countDistinct("resource_id").alias("resource_count"),
        F.countDistinct(F.when(F.col("resource_type") == "video", F.col("resource_id"))).alias("video_counts"),
        F.countDistinct(F.when(F.col("resource_type") == "exercise", F.col("resource_id"))).alias("ex_counts"),
    )
    .join(enrollments.groupBy("course_id").agg(F.count("*").alias("enrollment_count")), "course_id", "left")
    .fillna(0, ["video_counts", "ex_counts", "enrollment_count"])
)

summary_rows = log_dataframe(logger, "course_summary", course_summary, ("course_id",))
resource_path = f"{OUTPUT_BASE}/course_resources_clean/"
summary_path = f"{OUTPUT_BASE}/course_summary/"
write_parquet(course_resources_clean, resource_path)
write_parquet(course_summary, summary_path)
log_write(logger, "course_resources_clean", resource_path, resource_rows)
log_write(logger, "course_summary", summary_path, summary_rows)
flush_json_log(logger, spark)
