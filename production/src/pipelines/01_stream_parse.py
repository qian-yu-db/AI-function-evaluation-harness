# Databricks notebook source
# MAGIC %md
# MAGIC # deeds_pipeline · Stage 1 — Stream parse documents
# MAGIC
# MAGIC Streaming notebook that reads new files from the source UC volume and
# MAGIC writes raw `ai_parse_document` VARIANT rows to `deeds_parsed`.
# MAGIC `Trigger.AvailableNow` so each run processes whatever has arrived since
# MAGIC the last run, then stops. Checkpoints make re-runs idempotent.
# MAGIC
# MAGIC **Inputs** (job parameters):
# MAGIC - `catalog`, `schema`, `input_volume`, `checkpoint_volume`
# MAGIC
# MAGIC **Output:** `{catalog}.{schema}.deeds_parsed`

# COMMAND ----------

from pyspark.sql.functions import expr, regexp_extract

dbutils.widgets.text("catalog", "")
dbutils.widgets.text("schema", "unstructured_documents")
dbutils.widgets.text("input_volume", "deeds")
dbutils.widgets.text("checkpoint_volume", "deeds_checkpoints")
dbutils.widgets.text("image_subfolder", "images")
dbutils.widgets.text("table_suffix", "_dab")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
INPUT_VOLUME = dbutils.widgets.get("input_volume")
CHECKPOINT_VOLUME = dbutils.widgets.get("checkpoint_volume")
IMAGE_SUBFOLDER = dbutils.widgets.get("image_subfolder")
TABLE_SUFFIX = dbutils.widgets.get("table_suffix")

assert CATALOG, "catalog parameter is required"

VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{INPUT_VOLUME}/"
CHECKPOINT_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{CHECKPOINT_VOLUME}/01_parse"
# Page images live as a sibling subfolder of the source documents in the input
# volume. ai_parse_document writes one rendered image per page; metadata.pages[].image_uri
# in the parsed VARIANT points back at these so the review UI can overlay
# citation bboxes on the source page.
IMAGE_OUTPUT_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{INPUT_VOLUME}/{IMAGE_SUBFOLDER}/"
OUTPUT_TABLE = f"{CATALOG}.{SCHEMA}.deeds_parsed{TABLE_SUFFIX}"

print(f"Reading from:    {VOLUME_PATH}")
print(f"Checkpoint at:   {CHECKPOINT_PATH}")
print(f"Image output:    {IMAGE_OUTPUT_PATH}")
print(f"Writing to:      {OUTPUT_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Streaming read + parse + write
# MAGIC
# MAGIC `repartition` by `crc32(path)` parallelizes `ai_parse_document` across
# MAGIC executors. `mergeSchema=true` so a future runtime that adds fields to
# MAGIC the parsed VARIANT does not break the write.

# COMMAND ----------

from pyspark.sql.types import StructType, StructField, StringType, TimestampType, LongType, BinaryType

binary_file_schema = StructType([
    StructField("path", StringType()),
    StructField("modificationTime", TimestampType()),
    StructField("length", LongType()),
    StructField("content", BinaryType()),
])

files_df = (
    spark.readStream.format("binaryFile")
    .schema(binary_file_schema)
    .option("pathGlobFilter", "*.{pdf,PDF,tif,TIF,tiff,TIFF,jpg,jpeg,png}")
    .option("recursiveFileLookup", "true")
    .load(VOLUME_PATH)
)

parsed_df = (
    files_df
    .repartition(8, expr("crc32(path) % 8"))
    .withColumn(
        "parsed",
        expr(
            "ai_parse_document(content, map("
            "'version', '2.0', "
            "'descriptionElementTypes', '*', "
            f"'imageOutputPath', '{IMAGE_OUTPUT_PATH}'"
            "))"
        ),
    )
    .withColumn("image_name", regexp_extract("path", r"([^/]+)\.[^/.]+$", 1))
    .select("path", "image_name", "length", "modificationTime", "parsed")
)

query = (
    parsed_df.writeStream.format("delta")
    .queryName("deeds_parse_stream")
    .outputMode("append")
    .option("checkpointLocation", CHECKPOINT_PATH)
    .option("mergeSchema", "true")
    .trigger(availableNow=True)
    .toTable(OUTPUT_TABLE)
)

# No query.awaitTermination() — per Databricks docs, the Jobs service tracks
# the active streaming query and prevents task completion until it finishes.
# Calling awaitTermination() here would disrupt backlog metrics and job
# notifications. The Trigger.AvailableNow on the query makes the run bounded.