# Databricks notebook source
# MAGIC %md
# MAGIC # deeds_pipeline · Batch analytics (no LLM judge)
# MAGIC
# MAGIC Daily batch job. Reads `deeds_extracted_flat`, computes corpus-relative
# MAGIC outliers and field-level deterministic flags, writes the human-review
# MAGIC queues. **No LLM judge** — every signal is logic-based.
# MAGIC
# MAGIC Replaces `notebooks/02_eval_no_ground_truth.py` for production. Adds
# MAGIC two new field-level signals from the production plan:
# MAGIC
# MAGIC - `<field>_no_citation` — non-null value with empty `citation_ids`,
# MAGIC   gated on `extract_conf < no_citation_gate_conf` to avoid flagging
# MAGIC   confident inferences without explicit grounding.
# MAGIC - `<field>_format_specific_violation` — narrow per-field format rules
# MAGIC   (currently `RecordingDate` length-8 only — extensible).
# MAGIC
# MAGIC **Inputs** (job parameters):
# MAGIC - `catalog`, `schema`
# MAGIC - `low_extract_conf_threshold`, `high_conf_threshold`, `no_citation_gate_conf`
# MAGIC - `mlflow_experiment`
# MAGIC
# MAGIC **Outputs:**
# MAGIC - `{catalog}.{schema}.deeds_review_flags` — per-doc, sortable `review_priority`
# MAGIC - `{catalog}.{schema}.deeds_field_review` — per-cell review queue with
# MAGIC   resolved bbox citations
# MAGIC - MLflow run under `{mlflow_experiment}`

# COMMAND ----------

# MAGIC %pip install --quiet --upgrade "mlflow[databricks]>=3.1.0"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import json
import sys
from functools import reduce
from operator import add
from pathlib import Path

# Allow imports from production/src/shared/
NOTEBOOK_DIR = Path(__file__).resolve().parent if "__file__" in globals() else None
if NOTEBOOK_DIR is not None:
    SRC_DIR = NOTEBOOK_DIR.parent
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))
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

import mlflow  # noqa: E402
import pandas as pd  # noqa: E402
from mlflow.entities import Feedback  # noqa: E402
from mlflow.genai.scorers import scorer  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402
from pyspark.sql.functions import col  # noqa: E402

from shared.config import EXTRACT_FIELDS  # noqa: E402

# COMMAND ----------

dbutils.widgets.text("catalog", "")
dbutils.widgets.text("schema", "unstructured_documents")
dbutils.widgets.text("low_extract_conf_threshold", "0.7")
dbutils.widgets.text("high_conf_threshold", "0.8")
dbutils.widgets.text("no_citation_gate_conf", "0.95")
dbutils.widgets.text("mlflow_experiment", "")
dbutils.widgets.text("table_suffix", "_dab")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
LOW_EXTRACT_CONF_THRESHOLD = float(dbutils.widgets.get("low_extract_conf_threshold"))
HIGH_CONF_THRESHOLD = float(dbutils.widgets.get("high_conf_threshold"))
NO_CITATION_GATE_CONF = float(dbutils.widgets.get("no_citation_gate_conf"))
MLFLOW_EXPERIMENT = dbutils.widgets.get("mlflow_experiment")
TABLE_SUFFIX = dbutils.widgets.get("table_suffix")

assert CATALOG, "catalog parameter is required"
assert MLFLOW_EXPERIMENT, "mlflow_experiment parameter is required"

# Suffixed table names — keep DAB-written tables distinct from the dev notebooks.
EXTRACTED_FLAT_TABLE = f"deeds_extracted_flat{TABLE_SUFFIX}"
REVIEW_FLAGS_TABLE = f"deeds_review_flags{TABLE_SUFFIX}"
FIELD_REVIEW_TABLE = f"deeds_field_review{TABLE_SUFFIX}"

spark.sql(f"USE CATALOG {CATALOG}")
spark.sql(f"USE SCHEMA {SCHEMA}")

mlflow.set_tracking_uri("databricks")
mlflow.set_experiment(MLFLOW_EXPERIMENT)

print(
    f"catalog={CATALOG}  schema={SCHEMA}  table_suffix={TABLE_SUFFIX!r}  "
    f"low_extract_conf={LOW_EXTRACT_CONF_THRESHOLD}  "
    f"high_conf={HIGH_CONF_THRESHOLD}  "
    f"no_citation_gate={NO_CITATION_GATE_CONF}"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Doc-level corpus outliers
# MAGIC
# MAGIC Computed against the all-time corpus in `deeds_extracted_flat` (per the
# MAGIC production plan's baseline policy).

# COMMAND ----------

extracted_flat = spark.table(EXTRACTED_FLAT_TABLE)

corpus_stats_row = spark.sql(
    f"""
    SELECT
        percentile_approx(conf_mean, 0.10)         AS conf_mean_p10,
        percentile_approx(conf_stddev, 0.25)       AS conf_stddev_p25,
        percentile_approx(conf_stddev, 0.75)       AS conf_stddev_p75,
        percentile_approx(worst_page_mean, 0.10)   AS worst_page_mean_p10,
        AVG(conf_mean)                             AS conf_mean_corpus_avg,
        STDDEV(conf_mean)                          AS conf_mean_corpus_std
    FROM {EXTRACTED_FLAT_TABLE}
    """
).collect()[0].asDict()

iqr = (corpus_stats_row["conf_stddev_p75"] or 0.0) - (corpus_stats_row["conf_stddev_p25"] or 0.0)
high_var_fence = (corpus_stats_row["conf_stddev_p75"] or 0.0) + 1.5 * iqr

print("Corpus baseline:")
for k, v in corpus_stats_row.items():
    print(f"  {k}: {v}")
print(f"  conf_stddev IQR upper fence: {high_var_fence}")

outlier_flags = (
    extracted_flat
    .withColumn(
        "outlier_low_mean",
        F.when(col("conf_mean") <= F.lit(corpus_stats_row["conf_mean_p10"]), F.lit(True))
        .otherwise(F.lit(False)),
    )
    .withColumn(
        "outlier_high_variance",
        F.when(col("conf_stddev") > F.lit(high_var_fence), F.lit(True)).otherwise(F.lit(False)),
    )
    .withColumn(
        "outlier_bad_page",
        F.when(
            col("worst_page_mean") <= F.lit(corpus_stats_row["worst_page_mean_p10"]),
            F.lit(True),
        ).otherwise(F.lit(False)),
    )
    .withColumn("has_parse_error", col("error_status_count") > 0)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Heuristic shape checks (extraction-side)
# MAGIC
# MAGIC Validates format only on non-null values. Empty fields pass shape
# MAGIC because absence is legitimate when the entity isn't in the document.

# COMMAND ----------

DATE_RE = r"^\d{8}$"
ANY_DIGIT_RE = r".*\d.*"

shape_checked = (
    outlier_flags
    .withColumn(
        "recording_date_shape_ok",
        F.when(col("RecordingDate").isNull() | (F.length("RecordingDate") == 0), F.lit(True))
        .otherwise(col("RecordingDate").rlike(DATE_RE)),
    )
    .withColumn(
        "book_number_parsed_shape_ok",
        F.when(col("BookNumberParsed").isNull() | (F.length("BookNumberParsed") == 0), F.lit(True))
        .otherwise(col("BookNumberParsed").rlike(ANY_DIGIT_RE)),
    )
    .withColumn(
        "page_number_shape_ok",
        F.when(col("PageNumber").isNull() | (F.length("PageNumber") == 0), F.lit(True))
        .otherwise(col("PageNumber").rlike(ANY_DIGIT_RE)),
    )
    .withColumn(
        "document_number_shape_ok",
        F.when(col("DocumentNumber").isNull() | (F.length("DocumentNumber") == 0), F.lit(True))
        .otherwise(col("DocumentNumber").rlike(ANY_DIGIT_RE)),
    )
    .withColumn(
        "extraction_shape_ok",
        col("recording_date_shape_ok")
        & col("book_number_parsed_shape_ok")
        & col("page_number_shape_ok")
        & col("document_number_shape_ok"),
    )
    .withColumn(
        "extraction_completeness",
        reduce(
            add,
            [
                F.when(col(f).isNotNull() & (F.length(col(f)) > 0), F.lit(1.0)).otherwise(F.lit(0.0))
                for f in EXTRACT_FIELDS
            ],
        )
        / F.lit(float(len(EXTRACT_FIELDS))),
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Field-level low-confidence flags

# COMMAND ----------

field_flagged = shape_checked
for f in EXTRACT_FIELDS:
    field_flagged = field_flagged.withColumn(
        f"{f}_low_extract_conf",
        col(f).isNotNull()
        & (F.length(col(f)) > 0)
        & col(f"{f}_extract_conf").isNotNull()
        & (col(f"{f}_extract_conf") < F.lit(LOW_EXTRACT_CONF_THRESHOLD)),
    )

field_flagged = (
    field_flagged
    .withColumn(
        "low_conf_field_count",
        reduce(
            add,
            [
                F.coalesce(col(f"{f}_low_extract_conf").cast("int"), F.lit(0))
                for f in EXTRACT_FIELDS
            ],
        ),
    )
    .withColumn(
        "low_conf_fields",
        F.array_compact(
            F.array(
                *[
                    F.when(col(f"{f}_low_extract_conf"), F.lit(f)).otherwise(
                        F.lit(None).cast("string")
                    )
                    for f in EXTRACT_FIELDS
                ]
            )
        ),
    )
    .withColumn(
        "min_extract_conf",
        F.least(*[F.coalesce(col(f"{f}_extract_conf"), F.lit(1.0)) for f in EXTRACT_FIELDS]),
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. NEW — `<field>_no_citation` flags
# MAGIC
# MAGIC Fires when a non-null extracted value has empty `citation_ids` AND its
# MAGIC `extract_conf` is below `NO_CITATION_GATE_CONF`. The gate avoids
# MAGIC flagging high-confidence values the model claimed without explicit
# MAGIC grounding (often legitimate inferences from page layout).

# COMMAND ----------

for f in EXTRACT_FIELDS:
    field_flagged = field_flagged.withColumn(
        f"{f}_no_citation",
        col(f).isNotNull()
        & (F.length(col(f)) > 0)
        & (F.coalesce(F.size(col(f"{f}_citation_ids")), F.lit(0)) == 0)
        & col(f"{f}_extract_conf").isNotNull()
        & (col(f"{f}_extract_conf") < F.lit(NO_CITATION_GATE_CONF)),
    )

field_flagged = (
    field_flagged
    .withColumn(
        "no_citation_field_count",
        reduce(
            add,
            [
                F.coalesce(col(f"{f}_no_citation").cast("int"), F.lit(0))
                for f in EXTRACT_FIELDS
            ],
        ),
    )
    .withColumn(
        "no_citation_fields",
        F.array_compact(
            F.array(
                *[
                    F.when(col(f"{f}_no_citation"), F.lit(f)).otherwise(
                        F.lit(None).cast("string")
                    )
                    for f in EXTRACT_FIELDS
                ]
            )
        ),
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. NEW — `<field>_format_specific_violation` flags
# MAGIC
# MAGIC Narrow rules. Only `RecordingDate` length-8 is encoded today (the safe
# MAGIC starting point per the production plan). Add new rules as field
# MAGIC patterns become well-understood.

# COMMAND ----------

field_flagged = (
    field_flagged
    .withColumn(
        "RecordingDate_format_violation",
        col("RecordingDate").isNotNull()
        & (F.length("RecordingDate") > 0)
        & (F.length(F.regexp_replace(col("RecordingDate"), r"[^0-9]", "")) != F.lit(8)),
    )
    # Other fields: no specific rules yet — explicit False so columns exist
    .withColumn("DocumentTitle_format_violation", F.lit(False))
    .withColumn("BookNumberType_format_violation", F.lit(False))
    .withColumn("BookNumberParsed_format_violation", F.lit(False))
    .withColumn("PageNumber_format_violation", F.lit(False))
    .withColumn("DocumentNumber_format_violation", F.lit(False))
)

field_flagged = (
    field_flagged
    .withColumn(
        "format_violation_count",
        reduce(
            add,
            [
                F.coalesce(col(f"{f}_format_violation").cast("int"), F.lit(0))
                for f in EXTRACT_FIELDS
            ],
        ),
    )
    .withColumn(
        "format_violation_fields",
        F.array_compact(
            F.array(
                *[
                    F.when(col(f"{f}_format_violation"), F.lit(f)).otherwise(
                        F.lit(None).cast("string")
                    )
                    for f in EXTRACT_FIELDS
                ]
            )
        ),
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. MLflow eval — pass-through scorers, no judge

# COMMAND ----------

eval_pdf = field_flagged.toPandas()


def _to_int_list(v):
    if v is None:
        return []
    return [int(x) for x in v]


def _to_str_list(v):
    if v is None:
        return []
    return [str(x) for x in v]


eval_data = []
for _, row in eval_pdf.iterrows():
    extracted = {f: row[f] for f in EXTRACT_FIELDS}
    eval_data.append(
        {
            "inputs": {
                "image_name": row["image_name"],
                "parse_stats": {
                    "conf_mean": float(row["conf_mean"]) if pd.notna(row["conf_mean"]) else None,
                    "conf_stddev": float(row["conf_stddev"])
                    if pd.notna(row["conf_stddev"])
                    else None,
                    "worst_page_mean": float(row["worst_page_mean"])
                    if pd.notna(row["worst_page_mean"])
                    else None,
                    "max_page_stddev": float(row["max_page_stddev"])
                    if pd.notna(row["max_page_stddev"])
                    else None,
                    "error_status_count": int(row["error_status_count"])
                    if pd.notna(row["error_status_count"])
                    else 0,
                },
                "extracted": extracted,
                "shape_ok": bool(row["extraction_shape_ok"]),
                "completeness": float(row["extraction_completeness"]),
                "low_conf_field_count": int(row["low_conf_field_count"])
                if pd.notna(row["low_conf_field_count"]) else 0,
                "low_conf_fields": _to_str_list(row.get("low_conf_fields")),
                "no_citation_field_count": int(row["no_citation_field_count"])
                if pd.notna(row["no_citation_field_count"]) else 0,
                "no_citation_fields": _to_str_list(row.get("no_citation_fields")),
                "format_violation_count": int(row["format_violation_count"])
                if pd.notna(row["format_violation_count"]) else 0,
                "format_violation_fields": _to_str_list(row.get("format_violation_fields")),
                "min_extract_conf": float(row["min_extract_conf"])
                if pd.notna(row["min_extract_conf"])
                else 1.0,
            },
            "outputs": {
                "extracted": extracted,
                "extract_conf": {
                    f: (
                        float(row[f"{f}_extract_conf"])
                        if pd.notna(row[f"{f}_extract_conf"])
                        else None
                    )
                    for f in EXTRACT_FIELDS
                },
                "citation_ids": {
                    f: _to_int_list(row.get(f"{f}_citation_ids")) for f in EXTRACT_FIELDS
                },
            },
        }
    )


@scorer
def parse_conf_mean(inputs, outputs):
    val = inputs["parse_stats"]["conf_mean"]
    return Feedback(
        value=float(val) if val is not None else 0.0,
        rationale=f"AVG element confidence = {val}",
    )


@scorer
def parse_conf_stddev(inputs, outputs):
    val = inputs["parse_stats"]["conf_stddev"]
    return Feedback(
        value=float(val) if val is not None else 0.0,
        rationale=f"STDDEV element confidence = {val}",
    )


@scorer
def parse_worst_page_mean(inputs, outputs):
    val = inputs["parse_stats"]["worst_page_mean"]
    return Feedback(
        value=float(val) if val is not None else 0.0,
        rationale=f"Worst page mean = {val}",
    )


@scorer
def extraction_shape_ok(inputs, outputs):
    return Feedback(
        value=bool(inputs["shape_ok"]),
        rationale="Shape check: regex/digit rules on non-null fields only.",
    )


@scorer
def extraction_completeness(inputs, outputs):
    return Feedback(
        value=float(inputs["completeness"]),
        rationale="non_null_fields / 6 (informational)",
    )


@scorer
def min_extract_conf(inputs, outputs):
    val = inputs.get("min_extract_conf")
    return Feedback(value=float(val) if val is not None else 1.0, rationale=f"min over {EXTRACT_FIELDS}")


@scorer
def low_conf_field_count(inputs, outputs):
    return Feedback(
        value=int(inputs["low_conf_field_count"]),
        rationale=(
            f"threshold={LOW_EXTRACT_CONF_THRESHOLD}; "
            f"flagged_fields={inputs.get('low_conf_fields') or []}"
        ),
    )


@scorer
def no_citation_field_count(inputs, outputs):
    return Feedback(
        value=int(inputs["no_citation_field_count"]),
        rationale=(
            f"gate_conf={NO_CITATION_GATE_CONF}; "
            f"flagged_fields={inputs.get('no_citation_fields') or []}"
        ),
    )


@scorer
def format_violation_count(inputs, outputs):
    return Feedback(
        value=int(inputs["format_violation_count"]),
        rationale=f"flagged_fields={inputs.get('format_violation_fields') or []}",
    )


results = mlflow.genai.evaluate(
    data=eval_data,
    scorers=[
        parse_conf_mean,
        parse_conf_stddev,
        parse_worst_page_mean,
        extraction_shape_ok,
        extraction_completeness,
        min_extract_conf,
        low_conf_field_count,
        no_citation_field_count,
        format_violation_count,
    ],
)

print(f"Run ID:  {results.run_id}")
print(f"Metrics: {results.metrics}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. `deeds_review_flags`
# MAGIC
# MAGIC `review_priority` ranges 0..24 — field-level signals dominate so a doc
# MAGIC with N suspicious cells outranks a doc with one general flag.

# COMMAND ----------

review_flags = (
    field_flagged
    .withColumn(
        "review_priority",
        F.coalesce(col("outlier_low_mean").cast("int"), F.lit(0))
        + F.coalesce(col("outlier_high_variance").cast("int"), F.lit(0))
        + F.coalesce(col("outlier_bad_page").cast("int"), F.lit(0))
        + F.coalesce(col("has_parse_error").cast("int"), F.lit(0))
        + F.coalesce((~col("extraction_shape_ok")).cast("int"), F.lit(0))
        + F.coalesce(col("low_conf_field_count"), F.lit(0))
        + F.coalesce(col("no_citation_field_count"), F.lit(0))
        + F.coalesce(col("format_violation_count"), F.lit(0)),
    )
    .select(
        "image_name",
        "conf_mean",
        "conf_stddev",
        "worst_page_mean",
        "worst_page_id",
        "max_page_stddev",
        "error_status_count",
        "outlier_low_mean",
        "outlier_high_variance",
        "outlier_bad_page",
        "has_parse_error",
        "extraction_shape_ok",
        "extraction_completeness",
        "min_extract_conf",
        "low_conf_field_count",
        "low_conf_fields",
        "no_citation_field_count",
        "no_citation_fields",
        "format_violation_count",
        "format_violation_fields",
        *[f"{f}_low_extract_conf" for f in EXTRACT_FIELDS],
        *[f"{f}_no_citation" for f in EXTRACT_FIELDS],
        *[f"{f}_format_violation" for f in EXTRACT_FIELDS],
        "review_priority",
    )
)

(
    review_flags.write.mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(REVIEW_FLAGS_TABLE)
)

display(spark.table(REVIEW_FLAGS_TABLE).orderBy(F.col("review_priority").desc()))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. `deeds_field_review` — cell-level queue
# MAGIC
# MAGIC One row per (doc, field) flagged for review. Inclusion: any of
# MAGIC `low_extract_conf`, `no_citation`, `shape_violation`,
# MAGIC `format_specific_violation`. Each row carries the cited bbox regions
# MAGIC resolved from the doc's `citations` array for click-to-region triage.

# COMMAND ----------

field_review_pdf = (
    field_flagged
    .select(
        "image_name",
        *[
            c
            for f in EXTRACT_FIELDS
            for c in (
                f,
                f"{f}_extract_conf",
                f"{f}_citation_ids",
                f"{f}_low_extract_conf",
                f"{f}_no_citation",
                f"{f}_format_violation",
            )
        ],
        "recording_date_shape_ok",
        "book_number_parsed_shape_ok",
        "page_number_shape_ok",
        "document_number_shape_ok",
        col("citations").cast("string").alias("citations_json"),
    )
    .toPandas()
)

SHAPE_OK_COLUMN = {
    "RecordingDate": "recording_date_shape_ok",
    "BookNumberParsed": "book_number_parsed_shape_ok",
    "PageNumber": "page_number_shape_ok",
    "DocumentNumber": "document_number_shape_ok",
}


def _resolve_citations(citations_json, ids):
    if not ids or not citations_json:
        return []
    try:
        all_citations = json.loads(citations_json)
    except (TypeError, ValueError):
        return []
    if not isinstance(all_citations, list):
        return []
    id_set = {int(i) for i in ids}
    return [c for c in all_citations if isinstance(c, dict) and int(c.get("id", -1)) in id_set]


field_review_rows = []
for _, row in field_review_pdf.iterrows():
    for f in EXTRACT_FIELDS:
        low_conf = bool(row[f"{f}_low_extract_conf"])
        no_cit = bool(row[f"{f}_no_citation"])
        fmt_viol = bool(row[f"{f}_format_violation"])
        shape_col = SHAPE_OK_COLUMN.get(f)
        shape_violation = (shape_col is not None) and (row.get(shape_col) is False)

        if not (low_conf or no_cit or fmt_viol or shape_violation):
            continue

        reasons = []
        if low_conf:
            reasons.append(f"low_extract_conf<{LOW_EXTRACT_CONF_THRESHOLD}")
        if no_cit:
            reasons.append(f"no_citation_below_conf<{NO_CITATION_GATE_CONF}")
        if shape_violation:
            reasons.append("shape_violation")
        if fmt_viol:
            reasons.append("format_specific_violation")

        cit_ids = _to_int_list(row.get(f"{f}_citation_ids"))
        field_review_rows.append(
            {
                "image_name": row["image_name"],
                "field": f,
                "value": row[f] if pd.notna(row[f]) else None,
                "extract_conf": (
                    float(row[f"{f}_extract_conf"])
                    if pd.notna(row[f"{f}_extract_conf"])
                    else None
                ),
                "citation_ids": cit_ids,
                "cited_regions": json.dumps(_resolve_citations(row.get("citations_json"), cit_ids)),
                "reasons": reasons,
            }
        )

if field_review_rows:
    review_pdf = pd.DataFrame(field_review_rows)
    review_sdf = spark.createDataFrame(review_pdf)
    (
        review_sdf.write.mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(FIELD_REVIEW_TABLE)
    )
    display(
        spark.table(FIELD_REVIEW_TABLE).orderBy(
            F.col("extract_conf").asc_nulls_first(), F.col("image_name")
        )
    )
else:
    print("No fields flagged for review.")
    spark.createDataFrame(
        [],
        "image_name STRING, field STRING, value STRING, extract_conf DOUBLE, "
        "citation_ids ARRAY<INT>, cited_regions STRING, reasons ARRAY<STRING>",
    ).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(FIELD_REVIEW_TABLE)