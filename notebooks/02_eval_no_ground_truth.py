# Databricks notebook source
# MAGIC %md
# MAGIC # 02 — Evaluate Without Ground Truth
# MAGIC
# MAGIC Operational evaluation that runs after `01_parse_and_extract`. Two layers of signal,
# MAGIC both deterministic — no LLM judge, no per-doc API cost:
# MAGIC
# MAGIC 1. **Doc-level corpus outliers** on parse confidence (low mean, high variance, bad page,
# MAGIC    parse errors) — coarse "look at this whole doc" pointer.
# MAGIC 2. **Field-level extraction confidence** from `ai_extract` v2.1 — fine-grained "look at
# MAGIC    this specific cell" pointer. Each field's `<field>_extract_conf` is in [0, 1];
# MAGIC    anything below `LOW_EXTRACT_CONF_THRESHOLD` is flagged.
# MAGIC
# MAGIC **Important:** empty/null extracted fields are *not* automatically failures. A blank
# MAGIC value is correct when the entity does not appear in the document. Field-level low-conf
# MAGIC flags only fire on **non-null** values (a missing field has no confidence to flag).
# MAGIC
# MAGIC **Outputs:**
# MAGIC - MLflow run under `/Users/q.yu@databricks.com/deeds_poc_eval_no_gt`
# MAGIC - `fins_genai.unstructured_documents.deeds_review_flags` — per-doc signals + sortable `review_priority`
# MAGIC - `fins_genai.unstructured_documents.deeds_field_review` — long-format per-(doc, field) cells flagged for review, with cited bbox/page resolved

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup

# COMMAND ----------

# MAGIC %pip install --quiet --upgrade "mlflow[databricks]>=3.1.0"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import json
from functools import reduce
from operator import add

import mlflow
import pandas as pd
from mlflow.entities import Feedback
from mlflow.genai.scorers import scorer
from pyspark.sql import functions as F
from pyspark.sql.functions import col

CATALOG = "fins_genai"
SCHEMA = "unstructured_documents"
EXPERIMENT = "/Users/q.yu@databricks.com/mlflow_experiments/deeds_poc_eval_no_gt"

# Field-level extract confidence threshold — tune empirically. 0.7 is a safe starting point.
# A non-null field with confidence below this gets flagged for human review.
LOW_EXTRACT_CONF_THRESHOLD = 0.7

spark.sql(f"USE CATALOG {CATALOG}")
spark.sql(f"USE SCHEMA {SCHEMA}")

mlflow.set_tracking_uri("databricks")
mlflow.set_experiment(EXPERIMENT)

EXTRACT_FIELDS = [
    "DocumentTitle",
    "BookNumberType",
    "BookNumberParsed",
    "PageNumber",
    "DocumentNumber",
    "RecordingDate",
]

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Corpus-relative outlier flags (parse-side)
# MAGIC
# MAGIC Computed against the **current corpus only** — small N, but the relative ranking still
# MAGIC tells you which doc to look at first. None of these flags filter rows out.

# COMMAND ----------

extracted_flat = spark.table("deeds_extracted_flat")

# Corpus quantiles — use percentile_approx with a list arg
corpus_stats_row = spark.sql(
    """
    SELECT
        percentile_approx(conf_mean, 0.10) AS conf_mean_p10,
        percentile_approx(conf_stddev, 0.25) AS conf_stddev_p25,
        percentile_approx(conf_stddev, 0.75) AS conf_stddev_p75,
        percentile_approx(worst_page_mean, 0.10) AS worst_page_mean_p10,
        AVG(conf_mean) AS conf_mean_corpus_avg,
        STDDEV(conf_mean) AS conf_mean_corpus_std
    FROM deeds_extracted_flat
    """
).collect()[0].asDict()

iqr = (corpus_stats_row["conf_stddev_p75"] or 0.0) - (corpus_stats_row["conf_stddev_p25"] or 0.0)
high_var_fence = (corpus_stats_row["conf_stddev_p75"] or 0.0) + 1.5 * iqr

print("Corpus stats:")
for k, v in corpus_stats_row.items():
    print(f"  {k}: {v}")
print(f"  conf_stddev IQR upper fence: {high_var_fence}")

# Outlier flags
outlier_flags = (
    extracted_flat
    .withColumn(
        "outlier_low_mean",
        F.when(
            col("conf_mean") <= F.lit(corpus_stats_row["conf_mean_p10"]),
            F.lit(True),
        ).otherwise(F.lit(False)),
    )
    .withColumn(
        "outlier_high_variance",
        F.when(
            col("conf_stddev") > F.lit(high_var_fence),
            F.lit(True),
        ).otherwise(F.lit(False)),
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
# MAGIC ## 3. Heuristic shape checks (extraction-side)
# MAGIC
# MAGIC Validate format **only when a field is non-null**. Fully-empty extractions pass shape —
# MAGIC absence is treated as a legitimate "entity not in the document" rather than a failure.

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
    # Informational completeness — not used for review_priority
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
# MAGIC ## 3b. Field-level extraction confidence flags
# MAGIC
# MAGIC `ai_extract` v2.1 returns a `<field>_extract_conf` per scalar field. We flag any
# MAGIC **non-null** field whose confidence is below `LOW_EXTRACT_CONF_THRESHOLD`. Null
# MAGIC fields are *not* flagged here — a missing field has no confidence to flag, and a
# MAGIC genuine absence in the source document should not generate a review item.
# MAGIC
# MAGIC Two derived columns are added per doc:
# MAGIC - `low_conf_field_count` — number of fields below the threshold (0–6)
# MAGIC - `low_conf_fields` — array of field names that fired, for direct human triage

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
            [F.coalesce(col(f"{f}_low_extract_conf").cast("int"), F.lit(0)) for f in EXTRACT_FIELDS],
        ),
    )
    .withColumn(
        "low_conf_fields",
        F.array_compact(
            F.array(
                *[
                    F.when(col(f"{f}_low_extract_conf"), F.lit(f)).otherwise(F.lit(None).cast("string"))
                    for f in EXTRACT_FIELDS
                ]
            )
        ),
    )
    # Minimum field confidence per doc — useful as a single sortable signal in MLflow
    .withColumn(
        "min_extract_conf",
        F.least(*[F.coalesce(col(f"{f}_extract_conf"), F.lit(1.0)) for f in EXTRACT_FIELDS]),
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Build the per-doc evaluation dataset
# MAGIC
# MAGIC One record per document. Output is `outputs.extracted` so we can run `mlflow.genai.evaluate`
# MAGIC with pre-computed outputs (no `predict_fn`).

# COMMAND ----------

doc_text_df = (
    spark.table("deeds_parsed_elements")
    .filter(col("content").isNotNull())
    .groupBy("image_name")
    .agg(F.concat_ws("\n\n", F.collect_list("content")).alias("doc_text"))
    .withColumn("doc_text", F.substring("doc_text", 1, 80000))
)

eval_pdf = (
    field_flagged
    .join(doc_text_df, "image_name", "left")
    .toPandas()
)

print(f"Eval rows: {len(eval_pdf)}")


def _to_int_list(v):
    """Spark ARRAY<INT> -> list[int]. None or empty -> []."""
    if v is None:
        return []
    return [int(x) for x in v]


def _to_str_list(v):
    """Spark ARRAY<STRING> -> list[str]. None or empty -> []."""
    if v is None:
        return []
    return [str(x) for x in v]


eval_data = []
for _, row in eval_pdf.iterrows():
    extracted = {f: row[f] for f in EXTRACT_FIELDS}
    extracted_conf = {
        f: (float(row[f"{f}_extract_conf"]) if row[f"{f}_extract_conf"] is not None else None)
        for f in EXTRACT_FIELDS
    }
    extracted_citations = {
        f: _to_int_list(row.get(f"{f}_citation_ids")) for f in EXTRACT_FIELDS
    }
    eval_data.append(
        {
            "inputs": {
                "image_name": row["image_name"],
                "document_text": row["doc_text"] or "",
                "parse_stats": {
                    "conf_mean": float(row["conf_mean"]) if row["conf_mean"] is not None else None,
                    "conf_stddev": float(row["conf_stddev"]) if row["conf_stddev"] is not None else None,
                    "worst_page_mean": float(row["worst_page_mean"]) if row["worst_page_mean"] is not None else None,
                    "max_page_stddev": float(row["max_page_stddev"]) if row["max_page_stddev"] is not None else None,
                    "error_status_count": int(row["error_status_count"]) if row["error_status_count"] is not None else 0,
                },
                "extracted": extracted,
                "shape_ok": bool(row["extraction_shape_ok"]),
                "completeness": float(row["extraction_completeness"]),
                "low_conf_field_count": int(row["low_conf_field_count"]),
                "low_conf_fields": _to_str_list(row.get("low_conf_fields")),
                "min_extract_conf": float(row["min_extract_conf"]) if row["min_extract_conf"] is not None else 1.0,
            },
            # extract_conf and citation_ids belong with the model's *output* — they are the
            # extractor's own claims (value + grounding), not deterministic context.
            "outputs": {
                "extracted": extracted,
                "extract_conf": extracted_conf,
                "citation_ids": extracted_citations,
            },
        }
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Custom scorers (deterministic — no LLM judge)

# COMMAND ----------

@scorer
def parse_conf_mean(inputs, outputs):
    val = inputs["parse_stats"]["conf_mean"]
    return Feedback(value=float(val) if val is not None else 0.0, rationale=f"AVG element confidence = {val}")


@scorer
def parse_conf_stddev(inputs, outputs):
    val = inputs["parse_stats"]["conf_stddev"]
    return Feedback(value=float(val) if val is not None else 0.0, rationale=f"STDDEV element confidence = {val}")


@scorer
def parse_worst_page_mean(inputs, outputs):
    val = inputs["parse_stats"]["worst_page_mean"]
    return Feedback(value=float(val) if val is not None else 0.0, rationale=f"Worst page mean = {val}")


@scorer
def extraction_shape_ok(inputs, outputs):
    return Feedback(value=bool(inputs["shape_ok"]), rationale="Shape check: regex/digit rules only fire on non-null fields.")


@scorer
def extraction_completeness(inputs, outputs):
    return Feedback(value=float(inputs["completeness"]), rationale="non_null_fields / 6 (informational)")


@scorer
def min_extract_conf(inputs, outputs):
    """Minimum per-field extraction confidence — single sortable signal in MLflow."""
    val = inputs.get("min_extract_conf")
    return Feedback(value=float(val) if val is not None else 1.0, rationale=f"min over {EXTRACT_FIELDS}")


@scorer
def low_conf_field_count(inputs, outputs):
    """Number of non-null fields with extract_conf below threshold."""
    return Feedback(
        value=int(inputs["low_conf_field_count"]),
        rationale=(
            f"threshold={LOW_EXTRACT_CONF_THRESHOLD}; "
            f"flagged_fields={inputs.get('low_conf_fields') or []}"
        ),
    )


# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Run `mlflow.genai.evaluate`

# COMMAND ----------

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
    ],
)

print(f"Run ID:  {results.run_id}")
print(f"Metrics: {results.metrics}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Build `deeds_review_flags`
# MAGIC
# MAGIC Joins the corpus-relative outlier flags + shape check + field-level confidence flags.

# COMMAND ----------

# review_priority combines:
#   - 4 doc-level outlier flags (cap 4)
#   - !extraction_shape_ok (cap 1)
#   - low_conf_field_count (cap 6)
# Total range: 0..11. Field-level signal dominates so a doc with 5 low-conf fields
# outranks a doc with no field issues but a single doc-level outlier.
review_flags = (
    field_flagged
    .withColumn(
        "review_priority",
        F.coalesce(col("outlier_low_mean").cast("int"), F.lit(0))
        + F.coalesce(col("outlier_high_variance").cast("int"), F.lit(0))
        + F.coalesce(col("outlier_bad_page").cast("int"), F.lit(0))
        + F.coalesce(col("has_parse_error").cast("int"), F.lit(0))
        + F.coalesce((~col("extraction_shape_ok")).cast("int"), F.lit(0))
        + F.coalesce(col("low_conf_field_count"), F.lit(0)),
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
        *[f"{f}_low_extract_conf" for f in EXTRACT_FIELDS],
        "review_priority",
    )
)

(
    review_flags.write.mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("deeds_review_flags")
)

display(spark.table("deeds_review_flags").orderBy(F.col("review_priority").desc()))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Long-format `deeds_field_review` — cell-level review queue
# MAGIC
# MAGIC One row per (document, field) flagged for review. Each row carries the value, the
# MAGIC extract confidence, the citation IDs the extractor claimed, and the resolved bbox /
# MAGIC page metadata for click-to-region triage.
# MAGIC
# MAGIC A field is included if **either**:
# MAGIC - `<field>_low_extract_conf = true` (non-null value with confidence below threshold), or
# MAGIC - the document's overall extraction shape check failed for this specific field.
# MAGIC
# MAGIC Reviewers sort by `extract_conf ASC` to triage worst-first.

# COMMAND ----------

# Pull the per-doc citations VARIANT alongside the per-field columns so we can resolve
# citation_ids -> bbox/page on the driver.
field_review_pdf = (
    field_flagged
    .select(
        "image_name",
        *[c for f in EXTRACT_FIELDS for c in (f, f"{f}_extract_conf", f"{f}_citation_ids", f"{f}_low_extract_conf")],
        "recording_date_shape_ok",
        "book_number_parsed_shape_ok",
        "page_number_shape_ok",
        "document_number_shape_ok",
        col("citations").cast("string").alias("citations_json"),
    )
    .toPandas()
)

# Per-field shape-violation map (only populated for the four fields that have shape rules)
SHAPE_OK_COLUMN = {
    "RecordingDate": "recording_date_shape_ok",
    "BookNumberParsed": "book_number_parsed_shape_ok",
    "PageNumber": "page_number_shape_ok",
    "DocumentNumber": "document_number_shape_ok",
}


def _resolve_citations(citations_json, ids):
    """Filter the doc's citations VARIANT to those whose id is in ids. Returns list of dicts."""
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
        shape_col = SHAPE_OK_COLUMN.get(f)
        shape_violation = (shape_col is not None) and (row.get(shape_col) is False)
        if not (low_conf or shape_violation):
            continue
        reasons = []
        if low_conf:
            reasons.append(f"low_extract_conf<{LOW_EXTRACT_CONF_THRESHOLD}")
        if shape_violation:
            reasons.append("shape_violation")
        cit_ids = _to_int_list(row.get(f"{f}_citation_ids"))
        field_review_rows.append(
            {
                "image_name": row["image_name"],
                "field": f,
                "value": row[f] if pd.notna(row[f]) else None,
                "extract_conf": (
                    float(row[f"{f}_extract_conf"]) if pd.notna(row[f"{f}_extract_conf"]) else None
                ),
                "citation_ids": cit_ids,
                "cited_regions": _resolve_citations(row.get("citations_json"), cit_ids),
                "reasons": reasons,
            }
        )

if field_review_rows:
    review_pdf = pd.DataFrame(field_review_rows)
    # Serialize cited_regions as JSON string so Spark Delta is happy with the variable shape
    review_pdf["cited_regions"] = review_pdf["cited_regions"].apply(json.dumps)
    review_sdf = spark.createDataFrame(review_pdf)
    (
        review_sdf.write.mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable("deeds_field_review")
    )
    display(
        spark.table("deeds_field_review").orderBy(
            F.col("extract_conf").asc_nulls_first(), F.col("image_name")
        )
    )
else:
    print("No fields flagged for review.")
    spark.createDataFrame(
        [],
        "image_name STRING, field STRING, value STRING, extract_conf DOUBLE, "
        "citation_ids ARRAY<INT>, cited_regions STRING, reasons ARRAY<STRING>",
    ).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable("deeds_field_review")