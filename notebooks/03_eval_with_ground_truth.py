# Databricks notebook source
# MAGIC %md
# MAGIC # 03 — Evaluate With Ground Truth
# MAGIC
# MAGIC Offline evaluation against `GroundTruth.csv`. Loads ground truth, joins with
# MAGIC `deeds_extracted_flat`, applies a hybrid per-field comparator (fuzzy for free-form
# MAGIC `DocumentTitle`, strict for IDs and dates), and reports per-field
# MAGIC accuracy / precision / recall / F1 with **absence as a class** (blank-when-absent
# MAGIC counts as a true negative).
# MAGIC
# MAGIC Each comparison row also carries the model's own `extract_conf` and `citation_ids`
# MAGIC (from `ai_extract` v2.1) so error analysis can sort by *high-confidence-wrong* — the
# MAGIC most actionable error class. A separate calibration step bins confidence vs.
# MAGIC correctness to sanity-check whether the model's confidence claim is meaningful.
# MAGIC
# MAGIC **Outputs:**
# MAGIC - `fins_genai.unstructured_documents.deeds_ground_truth` — transposed GT
# MAGIC - `fins_genai.unstructured_documents.deeds_extracted_vs_gt` — per-cell comparison + extract_conf + cited_regions + high_conf_wrong
# MAGIC - `fins_genai.unstructured_documents.deeds_eval_metrics_gt` — per-field accuracy/precision/recall/F1
# MAGIC - `fins_genai.unstructured_documents.deeds_calibration_gt` — confidence-bin vs correctness rate
# MAGIC - MLflow run under `/Users/q.yu@databricks.com/deeds_poc_eval_gt`

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup

# COMMAND ----------

# MAGIC %pip install --quiet --upgrade "mlflow[databricks]>=3.1.0" pandas
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import json
import math
import re
import mlflow
import pandas as pd
from mlflow.entities import Feedback
from mlflow.genai.scorers import scorer, Correctness
from pyspark.sql import functions as F
from pyspark.sql.functions import col, expr

CATALOG = "fins_genai"
SCHEMA = "unstructured_documents"
EXPERIMENT = "/Users/q.yu@databricks.com/deeds_poc_eval_gt"
JUDGE_MODEL = "databricks:/databricks-claude-sonnet-4-6"

# Default ground-truth location. The repo's data/GroundTruth.xlsx must be uploaded here
GROUND_TRUTH_PATH = "/Volumes/fins_genai/unstructured_documents/deeds/ground_truth/GroundTruth.csv"

spark.sql(f"USE CATALOG {CATALOG}")
spark.sql(f"USE SCHEMA {SCHEMA}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.deeds_artifacts")

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
# MAGIC ## 2. Load & transpose ground truth → `deeds_ground_truth`
# MAGIC
# MAGIC The ground truth csv is column-oriented (one column per document, fields as rows). Empty cells
# MAGIC are preserved as **null** because null is meaningful here ("entity not in document").

# COMMAND ----------

def _normalize_image_name(s):
    """Canonical join key: lowercase, replace runs of [\\s:_-] with one underscore."""
    if s is None:
        return None
    return re.sub(r"[\s:_\-]+", "_", str(s).strip()).lower()


def _stringify(v):
    """Preserve leading zeros from csv; treat blanks/NaN as None."""
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    s = str(v).strip()
    if s == "" or s.lower() == "nan":
        return None
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return s

# Load CSV ground truth file
gt_pdf_raw = pd.read_csv(GROUND_TRUTH_PATH, header=None)

# Row 0 has field names in col 0 and ImageNames across; the remaining rows have field values.
header_row = gt_pdf_raw.iloc[0].tolist()
image_names = [str(x).strip() for x in header_row[1:] if pd.notna(x) and str(x).strip()]
n_docs = len(image_names)
print(f"GT documents: {n_docs} -> {image_names}")

records = {name: {"image_name_raw": name, "join_key": _normalize_image_name(name)} for name in image_names}

for _, row in gt_pdf_raw.iloc[1:].iterrows():
    field = row.iloc[0]
    if pd.isna(field):
        continue
    field = str(field).strip()
    if field not in EXTRACT_FIELDS and field != "ImageName":
        continue
    if field == "ImageName":
        continue
    for i, name in enumerate(image_names, start=1):
        records[name][field] = _stringify(row.iloc[i])

gt_pdf = pd.DataFrame(list(records.values()))
# Ensure all expected columns exist
for f in EXTRACT_FIELDS:
    if f not in gt_pdf.columns:
        gt_pdf[f] = None

gt_pdf = gt_pdf[["image_name_raw", "join_key"] + EXTRACT_FIELDS]
display(spark.createDataFrame(gt_pdf))

(
    spark.createDataFrame(gt_pdf)
    .write.mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("deeds_ground_truth")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Join ground truth to extraction output

# COMMAND ----------

# Add the same normalized key to the extraction side — runs of [whitespace : _ -] -> '_'.
# Also carry per-field extract_conf and citation_ids plus the doc-level citations VARIANT
# so we can resolve cited bbox regions per cell in §5.
extracted = (
    spark.table("deeds_extracted_flat")
    .withColumn(
        "join_key",
        F.regexp_replace(F.lower(col("image_name")), r"[\s:_\-]+", "_"),
    )
    .select(
        "join_key",
        "image_name",
        *[col(f).alias(f"pred_{f}") for f in EXTRACT_FIELDS],
        *[col(f"{f}_extract_conf").alias(f"pred_{f}_conf") for f in EXTRACT_FIELDS],
        *[col(f"{f}_citation_ids").alias(f"pred_{f}_citation_ids") for f in EXTRACT_FIELDS],
        col("citations").cast("string").alias("citations_json"),
    )
)

gt = (
    spark.table("deeds_ground_truth")
    .select(
        "join_key",
        col("image_name_raw").alias("gt_image_name"),
        *[col(f).alias(f"gt_{f}") for f in EXTRACT_FIELDS],
    )
)

joined = gt.join(extracted, "join_key", "inner")
print(f"Joined rows: {joined.count()} (expected {n_docs})")
display(joined)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Per-field comparators (fuzzy for `DocumentTitle`, strict elsewhere)
# MAGIC
# MAGIC Normalization rules (applied in SQL):
# MAGIC - `DocumentTitle`: lowercase, collapse whitespace, strip non-alphanumeric — fuzzy via Levenshtein similarity ≥ 0.85
# MAGIC - `BookNumberType`: trim + uppercase, strict equal
# MAGIC - `BookNumberParsed` / `PageNumber` / `DocumentNumber`: keep digits only, strip leading zeros, strict equal
# MAGIC - `RecordingDate`: keep digits only, must be 8-digit, strict equal

# COMMAND ----------

DIGIT_FIELDS = ["BookNumberParsed", "PageNumber", "DocumentNumber"]

def norm_title(c):
    return F.lower(F.regexp_replace(F.regexp_replace(c, r"[^a-zA-Z0-9 ]+", " "), r"\s+", " "))

def norm_type(c):
    return F.upper(F.trim(c))


def norm_digits(c):
    """Digits only, leading zeros stripped (empty becomes null)."""
    digits_only = F.regexp_replace(c, r"[^0-9]", "")
    stripped = F.regexp_replace(digits_only, r"^0+", "")
    return F.when((stripped == F.lit("")) & (digits_only != F.lit("")), F.lit("0")).otherwise(stripped)


def norm_date(c):
    return F.regexp_replace(c, r"[^0-9]", "")


# Build per-field exact_match / fuzzy_match columns
classified = joined

for fld in EXTRACT_FIELDS:
    gt_col = col(f"gt_{fld}")
    pred_col = col(f"pred_{fld}")

    if fld == "DocumentTitle":
        gt_norm = norm_title(gt_col)
        pred_norm = norm_title(pred_col)
        max_len = F.greatest(F.length(gt_norm), F.length(pred_norm))
        sim = F.when(
            max_len == 0,
            F.lit(1.0),
        ).otherwise(
            F.lit(1.0) - F.levenshtein(gt_norm, pred_norm).cast("double") / max_len.cast("double")
        )
        classified = classified.withColumn(f"{fld}_levenshtein_similarity", sim)
        exact = (gt_norm == pred_norm) & gt_col.isNotNull() & pred_col.isNotNull()
        fuzzy = (sim >= F.lit(0.85)) & gt_col.isNotNull() & pred_col.isNotNull()
        match = fuzzy
    elif fld == "BookNumberType":
        gt_norm = norm_type(gt_col)
        pred_norm = norm_type(pred_col)
        match = (gt_norm == pred_norm) & gt_col.isNotNull() & pred_col.isNotNull()
        exact = match
        fuzzy = match
    elif fld == "RecordingDate":
        gt_norm = norm_date(gt_col)
        pred_norm = norm_date(pred_col)
        valid_pred = (F.length(pred_norm) == 8) & pred_col.isNotNull()
        match = (gt_norm == pred_norm) & gt_col.isNotNull() & valid_pred
        exact = match
        fuzzy = match
    elif fld in DIGIT_FIELDS:
        gt_norm = norm_digits(gt_col)
        pred_norm = norm_digits(pred_col)
        match = (gt_norm == pred_norm) & gt_col.isNotNull() & pred_col.isNotNull()
        exact = match
        fuzzy = match
    else:
        match = (gt_col == pred_col) & gt_col.isNotNull() & pred_col.isNotNull()
        exact = match
        fuzzy = match

    classified = classified.withColumn(f"{fld}_exact_match", exact)
    classified = classified.withColumn(f"{fld}_fuzzy_match", fuzzy)

    # Absence-aware classification — TP / TN / FP / FN
    cls = (
        F.when(gt_col.isNull() & pred_col.isNull(), F.lit("TN"))
        .when(gt_col.isNull() & pred_col.isNotNull(), F.lit("FP"))
        .when(gt_col.isNotNull() & pred_col.isNull(), F.lit("FN"))
        .when(match, F.lit("TP"))
        .otherwise(F.lit("FP_FN"))  # both non-null, mismatch — counts as both
    )
    classified = classified.withColumn(f"{fld}_classification", cls)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Build `deeds_extracted_vs_gt` (long format, one row per (doc, field))
# MAGIC
# MAGIC Each row carries:
# MAGIC - The classification (TP / TN / FP / FN / FP_FN), exact and fuzzy match flags, and
# MAGIC   for `DocumentTitle` the raw Levenshtein similarity.
# MAGIC - `extract_conf` — the model's per-field confidence claim from `ai_extract` v2.1.
# MAGIC - `citation_ids` and resolved `cited_regions` (JSON-serialized bbox + page_id) so a
# MAGIC   reviewer can click straight to the cited source region.
# MAGIC - `high_conf_wrong` — the boolean shortcut for the worst error class:
# MAGIC   `extract_conf >= 0.8 AND classification IN ('FP', 'FN', 'FP_FN')`. Sort by this
# MAGIC   descending to find errors the model was confident about (most damaging in production).

# COMMAND ----------

HIGH_CONF_THRESHOLD = 0.8

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


def _to_int_list(v):
    if v is None:
        return []
    return [int(x) for x in v]


# Pull the classified rows to the driver so we can do citation resolution per cell. With
# n_docs * len(EXTRACT_FIELDS) <= a few hundred rows this is small.
classified_pdf = classified.toPandas()

per_cell_records = []
for _, row in classified_pdf.iterrows():
    citations_json = row.get("citations_json")
    for fld in EXTRACT_FIELDS:
        cls = row[f"{fld}_classification"]
        is_wrong = cls in ("FP", "FN", "FP_FN")
        conf = row.get(f"pred_{fld}_conf")
        conf = float(conf) if conf is not None and not pd.isna(conf) else None
        cit_ids = _to_int_list(row.get(f"pred_{fld}_citation_ids"))
        per_cell_records.append(
            {
                "join_key": row["join_key"],
                "image_name": row["image_name"],
                "field": fld,
                "gt_value": (None if pd.isna(row[f"gt_{fld}"]) else row[f"gt_{fld}"]),
                "pred_value": (None if pd.isna(row[f"pred_{fld}"]) else row[f"pred_{fld}"]),
                "exact_match": bool(row[f"{fld}_exact_match"]) if row[f"{fld}_exact_match"] is not None else None,
                "fuzzy_match": bool(row[f"{fld}_fuzzy_match"]) if row[f"{fld}_fuzzy_match"] is not None else None,
                "levenshtein_similarity": (
                    float(row.get("DocumentTitle_levenshtein_similarity"))
                    if fld == "DocumentTitle" and pd.notna(row.get("DocumentTitle_levenshtein_similarity"))
                    else None
                ),
                "classification": cls,
                "extract_conf": conf,
                "citation_ids": cit_ids,
                "cited_regions": json.dumps(_resolve_citations(citations_json, cit_ids)),
                "high_conf_wrong": bool(is_wrong and conf is not None and conf >= HIGH_CONF_THRESHOLD),
            }
        )

per_cell_pdf = pd.DataFrame(per_cell_records)
per_cell_sdf = spark.createDataFrame(per_cell_pdf)

(
    per_cell_sdf.write.mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("deeds_extracted_vs_gt")
)

# Worst errors first: high_conf_wrong DESC, then lowest conf among non-TP rows
display(
    spark.table("deeds_extracted_vs_gt").orderBy(
        F.col("high_conf_wrong").desc(),
        F.col("classification").isin("TP", "TN").asc(),  # errors before correct
        F.col("extract_conf").desc_nulls_last(),
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Per-field corpus metrics — accuracy / precision / recall / F1

# COMMAND ----------

metrics_df = spark.sql(
    """
    WITH counts AS (
        SELECT
            field,
            SUM(CASE WHEN classification = 'TP' THEN 1 ELSE 0 END) AS tp,
            SUM(CASE WHEN classification = 'TN' THEN 1 ELSE 0 END) AS tn,
            SUM(CASE WHEN classification = 'FP' THEN 1 ELSE 0 END) AS fp,
            SUM(CASE WHEN classification = 'FN' THEN 1 ELSE 0 END) AS fn,
            SUM(CASE WHEN classification = 'FP_FN' THEN 1 ELSE 0 END) AS fp_fn,
            COUNT(*) AS n
        FROM deeds_extracted_vs_gt
        GROUP BY field
    )
    SELECT
        field,
        n,
        tp, tn, fp, fn, fp_fn,
        ROUND((tp + tn) / NULLIF(n, 0), 4) AS accuracy,
        ROUND(tp / NULLIF(tp + fp + fp_fn, 0), 4) AS precision,
        ROUND(tp / NULLIF(tp + fn + fp_fn, 0), 4) AS recall,
        ROUND(
            2.0 * tp / NULLIF(2 * tp + fp + fn + 2 * fp_fn, 0),
            4
        ) AS f1
    FROM counts
    ORDER BY field
    """
)
display(metrics_df)

# Persist the per-field metrics so reviewers don't need to re-run the SQL
(
    metrics_df.write.mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("deeds_eval_metrics_gt")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. MLflow `genai.evaluate` with absence-aware `expected_facts`

# COMMAND ----------

# Reuse the classified pandas DF already collected in §5.
joined_pdf = classified_pdf


def _build_expected_facts(row):
    facts = []
    for fld in EXTRACT_FIELDS:
        gt_val = row[f"gt_{fld}"]
        if pd.isna(gt_val) or gt_val is None:
            facts.append(f"{fld} is not present in the document")
        else:
            facts.append(f"{fld} is {gt_val}")
    return facts


eval_data = []
for _, row in joined_pdf.iterrows():
    extracted = {f: (None if pd.isna(row[f"pred_{f}"]) else row[f"pred_{f}"]) for f in EXTRACT_FIELDS}
    eval_data.append(
        {
            "inputs": {
                "image_name": row["image_name"],
                "document_text": row["image_name"],  # placeholder; per-field scorers don't need text
            },
            "outputs": {"extracted": extracted},
            "expectations": {"expected_facts": _build_expected_facts(row)},
        }
    )

# Per-field comparator description — labels the method that was used so the MLflow
# rationale string is self-explanatory in the UI without forcing reviewers to read code.
COMPARATOR_DESCRIPTION = {
    "DocumentTitle":    ("levenshtein_sim>=0.85", "lowercase + alnum + collapse_ws"),
    "BookNumberType":   ("strict_equal",          "trim + uppercase"),
    "BookNumberParsed": ("strict_equal",          "digits_only + strip_leading_zeros"),
    "PageNumber":       ("strict_equal",          "digits_only + strip_leading_zeros"),
    "DocumentNumber":   ("strict_equal",          "digits_only + strip_leading_zeros"),
    "RecordingDate":    ("strict_equal",          "digits_only (must be 8-digit)"),
}


def _make_field_scorer(field_name: str, classified_pdf: pd.DataFrame):
    """Build a per-field MLflow scorer.

    The match logic itself was already computed in §4 (PySpark) and persisted to
    `deeds_extracted_vs_gt` in §5. This scorer is a pass-through, but the rationale
    string carries the GT, prediction, comparator, normalization, and (for
    DocumentTitle) the raw similarity — so the MLflow per-row view is informative
    instead of just echoing the classification label.
    """
    comparator_kind, normalization = COMPARATOR_DESCRIPTION.get(field_name, ("", ""))

    # Per-doc lookup with everything the rationale needs
    lookup = {}
    for _, row in classified_pdf.iterrows():
        gt = row.get(f"gt_{field_name}")
        pred = row.get(f"pred_{field_name}")
        sim = (
            row.get("DocumentTitle_levenshtein_similarity")
            if field_name == "DocumentTitle"
            else None
        )
        lookup[row["image_name"]] = {
            "classification": row.get(f"{field_name}_classification"),
            "gt": (None if pd.isna(gt) else gt),
            "pred": (None if pd.isna(pred) else pred),
            "similarity": float(sim) if sim is not None and not pd.isna(sim) else None,
        }

    @scorer(name=f"{field_name}_correct")
    def _scorer(inputs, outputs, expectations):
        info = lookup.get(inputs["image_name"], {})
        cls = info.get("classification")
        is_correct = cls in ("TP", "TN")
        sim = info.get("similarity")
        sim_part = f", levenshtein_sim={sim:.3f}" if sim is not None else ""
        rationale = (
            f"classification={cls}; "
            f"comparator={comparator_kind} on {normalization}; "
            f"gt={info.get('gt')!r}, pred={info.get('pred')!r}{sim_part}"
        )
        return Feedback(value=bool(is_correct), rationale=rationale)

    return _scorer


field_scorers = [_make_field_scorer(f, joined_pdf) for f in EXTRACT_FIELDS]

results = mlflow.genai.evaluate(
    data=eval_data,
    scorers=[Correctness(model=JUDGE_MODEL), *field_scorers],
)

print(f"Run ID:  {results.run_id}")
print(f"Metrics: {results.metrics}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Log corpus-level precision/recall/F1 to the same MLflow run

# COMMAND ----------

with mlflow.start_run(run_id=results.run_id):
    for row in metrics_df.collect():
        for metric in ("accuracy", "precision", "recall", "f1"):
            v = row[metric]
            if v is not None:
                mlflow.log_metric(f"{row['field']}_{metric}", float(v))

print("Logged per-field accuracy/precision/recall/F1 to the eval run.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Calibration check — does `extract_conf` correlate with correctness?
# MAGIC
# MAGIC Bins per-cell `extract_conf` and computes the correctness rate per bin. The basic
# MAGIC sanity check we want to answer: *when the model says it's confident, is it usually
# MAGIC right?* If accuracy stays flat across bins, the confidence score is uninformative
# MAGIC and the production review threshold needs to be much higher than the apparent score
# MAGIC would suggest.
# MAGIC
# MAGIC With a 6-doc × 6-field corpus this is at most 36 cells (fewer once nulls are dropped),
# MAGIC so treat the result as an early sanity check rather than a calibrated curve. Rerun
# MAGIC after each batch growth and watch the trend.
# MAGIC
# MAGIC `correct` is defined as `classification IN ('TP', 'TN')`. Cells where `extract_conf`
# MAGIC is null (the field wasn't extracted) are excluded — there's no confidence to bin.

# COMMAND ----------

CALIBRATION_BINS = [
    (0.00, 0.50, "0.0-0.5"),
    (0.50, 0.80, "0.5-0.8"),
    (0.80, 0.95, "0.8-0.95"),
    (0.95, 1.01, "0.95-1.0"),
]

calibration_rows = []
for lo, hi, label in CALIBRATION_BINS:
    bin_df = spark.sql(
        f"""
        SELECT
            COUNT(*) AS n,
            SUM(CASE WHEN classification IN ('TP', 'TN') THEN 1 ELSE 0 END) AS n_correct,
            SUM(CASE WHEN classification IN ('FP', 'FN', 'FP_FN') THEN 1 ELSE 0 END) AS n_wrong
        FROM deeds_extracted_vs_gt
        WHERE extract_conf IS NOT NULL
          AND extract_conf >= {lo}
          AND extract_conf <  {hi}
        """
    ).collect()[0]
    n = int(bin_df["n"] or 0)
    n_correct = int(bin_df["n_correct"] or 0)
    accuracy = (n_correct / n) if n else None
    calibration_rows.append(
        {
            "bin": label,
            "lo": lo,
            "hi": hi,
            "n": n,
            "n_correct": n_correct,
            "n_wrong": int(bin_df["n_wrong"] or 0),
            "accuracy": accuracy,
        }
    )

calibration_pdf = pd.DataFrame(calibration_rows)
display(spark.createDataFrame(calibration_pdf))

(
    spark.createDataFrame(calibration_pdf)
    .write.mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("deeds_calibration_gt")
)

# Log to the same MLflow run so calibration tracks alongside the per-field metrics
with mlflow.start_run(run_id=results.run_id):
    for r in calibration_rows:
        # Skip empty bins to keep the metrics view clean
        if r["n"] == 0:
            continue
        bin_key = r["bin"].replace(".", "_").replace("-", "_to_")
        mlflow.log_metric(f"calibration_{bin_key}_n", float(r["n"]))
        if r["accuracy"] is not None:
            mlflow.log_metric(f"calibration_{bin_key}_accuracy", float(r["accuracy"]))

# Quick verdict: monotonic-up accuracy across bins = well-calibrated; flat or non-monotonic
# = confidence is unreliable and review thresholds must be set conservatively.
non_empty = [r for r in calibration_rows if r["n"] > 0 and r["accuracy"] is not None]
if len(non_empty) >= 2:
    accs = [r["accuracy"] for r in non_empty]
    is_monotonic = all(accs[i] <= accs[i + 1] for i in range(len(accs) - 1))
    print(f"Calibration accuracies (low→high conf): {[round(a, 3) for a in accs]}")
    print(f"Monotonic non-decreasing: {is_monotonic}  (True = sane calibration)")
else:
    print("Not enough bins populated to assess monotonicity.")