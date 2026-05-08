# Databricks notebook source
# MAGIC %md
# MAGIC # deeds_pipeline · Stage 2 — Stream extract entities + per-doc parse stats
# MAGIC
# MAGIC Reads `deeds_parsed` as a Delta stream, calls `ai_extract` v2.1 (with
# MAGIC `enableConfidenceScores` + `enableCitations`), computes per-doc parse
# MAGIC confidence statistics inline, and writes the wide `deeds_extracted_flat`
# MAGIC table. Independent checkpoint from Stage 1 so a failure here never
# MAGIC re-runs the (expensive) `ai_parse_document` step.
# MAGIC
# MAGIC No LLM judge — all scoring is deferred to Job 2 (batch analytics).
# MAGIC
# MAGIC **Inputs** (job parameters):
# MAGIC - `catalog`, `schema`, `checkpoint_volume`
# MAGIC
# MAGIC **Source:** `{catalog}.{schema}.deeds_parsed`
# MAGIC **Output:** `{catalog}.{schema}.deeds_extracted_flat`

# COMMAND ----------

import sys
from pathlib import Path

# Allow `from shared.extract_schema import ...` when running as a Databricks
# notebook task (the bundle deploys `src/` as the working directory).
NOTEBOOK_DIR = Path(__file__).resolve().parent if "__file__" in globals() else None
if NOTEBOOK_DIR is not None:
    SRC_DIR = NOTEBOOK_DIR.parent  # production/src
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))

# When run interactively in the workspace UI without __file__, fall back to a
# workspace-relative path. Bundles deploy notebooks under
# /Workspace/Users/<user>/.bundle/<bundle>/<target>/files/src/...
if NOTEBOOK_DIR is None:
    workspace_src = "/Workspace" + (
        dbutils.notebook.entry_point.getDbutils()
        .notebook()
        .getContext()
        .notebookPath()
        .get()
        .rsplit("/", 2)[0]
    )
    if workspace_src not in sys.path:
        sys.path.insert(0, workspace_src)

from shared.config import EXTRACT_FIELDS  # noqa: E402
from shared.extract_schema import EXTRACT_INSTRUCTIONS, EXTRACT_SCHEMA  # noqa: E402

from pyspark.sql import functions as F  # noqa: E402
from pyspark.sql.functions import col, expr  # noqa: E402

# COMMAND ----------

dbutils.widgets.text("catalog", "")
dbutils.widgets.text("schema", "unstructured_documents")
dbutils.widgets.text("checkpoint_volume", "deeds_checkpoints")
dbutils.widgets.text("table_suffix", "_dab")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
CHECKPOINT_VOLUME = dbutils.widgets.get("checkpoint_volume")
TABLE_SUFFIX = dbutils.widgets.get("table_suffix")

assert CATALOG, "catalog parameter is required"

INPUT_TABLE = f"{CATALOG}.{SCHEMA}.deeds_parsed{TABLE_SUFFIX}"
OUTPUT_TABLE = f"{CATALOG}.{SCHEMA}.deeds_extracted_flat{TABLE_SUFFIX}"
CHECKPOINT_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{CHECKPOINT_VOLUME}/02_extract"

print(f"Reading from:    {INPUT_TABLE}")
print(f"Checkpoint at:   {CHECKPOINT_PATH}")
print(f"Writing to:      {OUTPUT_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## foreachBatch handler
# MAGIC
# MAGIC `foreachBatch` lets us run multi-step DataFrame ops per micro-batch:
# MAGIC ai_extract, then explode element rows for per-page stats, then join
# MAGIC back. Trigger.AvailableNow means each "batch" is just whatever new
# MAGIC `deeds_parsed` rows have arrived since the last run.

# COMMAND ----------

# Pre-build the ai_extract SQL expression once. Single-quote escaping so the
# JSON schema and instructions interpolate safely into a SQL `expr(...)` call.
_extract_options_sql = (
    "map("
    "'version', '2.1', "
    "'enableConfidenceScores', 'true', "
    "'enableCitations', 'true', "
    f"'instructions', '{EXTRACT_INSTRUCTIONS.replace(chr(39), chr(39) * 2)}'"
    ")"
)
_extract_schema_sql = EXTRACT_SCHEMA.replace("'", "''").strip()
AI_EXTRACT_EXPR = (
    f"ai_extract(parsed, '{_extract_schema_sql}', {_extract_options_sql})"
)


def _per_field_select_exprs():
    """SELECT clauses for the wide extraction columns lifted from the ai_extract VARIANT."""
    parts = []
    for f in EXTRACT_FIELDS:
        parts.append(f"try_cast(extracted:response:{f}:value AS STRING) AS {f}")
        parts.append(
            f"try_cast(extracted:response:{f}:confidence_score AS DOUBLE) AS {f}_extract_conf"
        )
        parts.append(
            f"try_cast(extracted:response:{f}:citation_ids AS ARRAY<INT>) AS {f}_citation_ids"
        )
    return parts


def _doc_stats_df(parsed_df):
    """Per-doc parse confidence stats by exploding the VARIANT element list.

    Returns a DataFrame keyed by image_name with the per-doc + per-page roll-up
    columns expected by deeds_extracted_flat (mirrors notebooks/01 cell 5).
    """
    elements_df = (
        parsed_df.select(
            "image_name",
            expr("try_cast(parsed:document:elements AS ARRAY<VARIANT>) AS elements"),
            expr("size(try_cast(parsed:error_status AS ARRAY<VARIANT>)) AS error_status_count"),
        )
        .selectExpr(
            "image_name",
            "error_status_count",
            "posexplode_outer(elements) AS (element_idx, element)",
        )
        .selectExpr(
            "image_name",
            "error_status_count",
            "try_cast(element:bbox[0]:page_id AS INT) AS page_id",
            "try_cast(element:confidence AS DOUBLE) AS confidence",
        )
    )

    # Per-page stats (same doc only — no cross-batch aggregation issues)
    page_df = (
        elements_df.groupBy("image_name", "page_id")
        .agg(
            F.avg("confidence").alias("page_conf_mean"),
            F.stddev("confidence").alias("page_conf_stddev"),
        )
    )

    # Per-doc roll-ups derived from per-page
    page_rollup = (
        page_df.groupBy("image_name")
        .agg(
            F.min("page_conf_mean").alias("worst_page_mean"),
            F.max("page_conf_stddev").alias("max_page_stddev"),
        )
    )
    worst_page_id = (
        page_df.alias("p")
        .join(page_rollup.alias("r"), "image_name")
        .where(col("p.page_conf_mean") == col("r.worst_page_mean"))
        .groupBy("image_name")
        .agg(F.min("p.page_id").alias("worst_page_id"))
    )

    # Per-doc element-level stats
    doc_stats = (
        elements_df.groupBy("image_name")
        .agg(
            F.first("error_status_count").alias("error_status_count"),
            F.countDistinct("page_id").alias("n_pages"),
            F.count(F.lit(1)).alias("n_elements"),
            F.avg("confidence").alias("conf_mean"),
            F.stddev("confidence").alias("conf_stddev"),
            F.min("confidence").alias("conf_min"),
            F.max("confidence").alias("conf_max"),
            F.expr("percentile_approx(confidence, 0.10)").alias("conf_p10"),
            F.expr("percentile_approx(confidence, 0.25)").alias("conf_p25"),
            F.expr("percentile_approx(confidence, 0.50)").alias("conf_p50"),
            F.expr("percentile_approx(confidence, 0.75)").alias("conf_p75"),
            F.expr("percentile_approx(confidence, 0.90)").alias("conf_p90"),
            F.avg(F.when(col("confidence") < 0.5, 1.0).otherwise(0.0)).alias("low_conf_pct"),
        )
        .withColumn("conf_iqr", col("conf_p75") - col("conf_p25"))
    )

    return (
        doc_stats
        .join(page_rollup, "image_name", "left")
        .join(worst_page_id, "image_name", "left")
    )


def process_batch(batch_df, batch_id):
    if batch_df.isEmpty():
        print(f"[batch {batch_id}] empty, skipping")
        return

    # ai_extract — wide projection of values, confidences, citation_ids per field
    extracted = (
        batch_df.withColumn("extracted", expr(AI_EXTRACT_EXPR))
        .selectExpr(
            "image_name",
            *_per_field_select_exprs(),
            # Document-level metadata — when ai_extract is called on the parsed
            # VARIANT, citations carry bbox+page_id; preserve as JSON string for
            # downstream review-queue rendering.
            "extracted:metadata:citations AS citations",
            "extracted:metadata:pages     AS citation_pages",
            "try_cast(extracted:metadata:chunk_type AS STRING) AS citation_chunk_type",
            "try_cast(extracted:error_message AS STRING) AS extraction_error_message",
        )
    )

    # Per-doc parse confidence stats from the parsed VARIANT
    stats = _doc_stats_df(batch_df)

    final = extracted.join(stats, "image_name", "left")

    (
        final.write.format("delta")
        .mode("append")
        .option("mergeSchema", "true")
        .saveAsTable(OUTPUT_TABLE)
    )
    print(f"[batch {batch_id}] wrote {final.count()} rows to {OUTPUT_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run the stream

# COMMAND ----------

query = (
    spark.readStream.format("delta").table(INPUT_TABLE)
    .writeStream
    .foreachBatch(process_batch)
    .option("checkpointLocation", CHECKPOINT_PATH)
    .trigger(availableNow=True)
    .start()
)

query.awaitTermination()

# COMMAND ----------

n_extracted = spark.table(OUTPUT_TABLE).count()
print(f"{OUTPUT_TABLE}: {n_extracted} rows total")