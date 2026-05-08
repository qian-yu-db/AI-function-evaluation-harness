# Databricks notebook source
# MAGIC %md
# MAGIC # 01 — Parse & Extract DED Documents
# MAGIC
# MAGIC Batch pipeline that runs `ai_parse_document` over scanned U.S. real-estate Deed (DED)
# MAGIC documents in a UC Volume, captures per-element / per-page / per-doc confidence
# MAGIC statistics as **descriptive signals** (never used to filter), then runs `ai_extract`
# MAGIC directly on the parsed VARIANT to populate the header schema.
# MAGIC
# MAGIC **Outputs (all in `fins_genai.unstructured_documents`):**
# MAGIC
# MAGIC | Table | Purpose |
# MAGIC |---|---|
# MAGIC | `deeds_parsed` | Raw `ai_parse_document` VARIANT per file |
# MAGIC | `deeds_parsed_elements` | Exploded elements with confidence/bbox — analytics only, no filtering |
# MAGIC | `deeds_parsed_pages` | Per-page confidence stats |
# MAGIC | `deeds_extracted_flat` | One row per document: image_name + 6 extracted fields + confidence stats |
# MAGIC
# MAGIC **Requirements:** DBR 17.1+ (for `ai_parse_document`) or serverless SQL.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.functions import col, expr, lit

CATALOG = "fins_genai"
SCHEMA = "unstructured_documents"
VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/deeds/"

spark.sql(f"USE CATALOG {CATALOG}")
spark.sql(f"USE SCHEMA {SCHEMA}")

# Extraction schema mirrors data/header_schema.txt with per-field descriptions to guide
# the model. Descriptions are intentionally specific to recorded U.S. real-estate deeds.
EXTRACT_SCHEMA = """
{
  "DocumentTitle": {
    "type": "string",
    "description": "The title or document type printed on the deed, usually in large or bold text near the top of the first page (e.g., 'WARRANTY DEED', 'CORRECTION QUITCLAIM DEED', 'GENERAL WARRANTY DEED', 'BENEFICIARY (TRANSFER ON DEATH) DEED'). Capture the verbatim heading. Do not include unrelated boilerplate like 'STATE OF ...' or 'KNOW ALL MEN BY THESE PRESENTS'."
  },
  "BookNumberType": {
    "type": "string",
    "description": "The short code or prefix label that precedes/qualifies the book number where the deed is recorded — typically a single letter or 1-3 character abbreviation (e.g., 'R', 'B', 'BK', 'OR', 'DR'). Look for patterns like 'Book R', 'BK 8710', 'Vol. B', or recording-stamp text. Return null if no book-number type is shown."
  },
  "BookNumberParsed": {
    "type": "string",
    "description": "The book/volume number itself where this deed is recorded, as an integer string (digits only, no leading zeros unless explicitly printed). Found near the recording stamp or in phrases like 'recorded in Book 8710'. Return null if the document is filed by document number only and has no book reference."
  },
  "PageNumber": {
    "type": "string",
    "description": "The page or page range where the deed is recorded within the book (e.g., '995', '0839', '2085-2087', 'Page 1 of 2'). Preserve the original formatting including any leading zeros and hyphenated ranges as printed. Return null if no page reference is shown."
  },
  "DocumentNumber": {
    "type": "string",
    "description": "The unique document/instrument number assigned by the county recorder, as printed on the document or recording stamp (e.g., '2023011643', '25008569', '01161191'). Preserve leading zeros exactly as shown. This is distinct from any internal book/page reference."
  },
  "RecordingDate": {
    "type": "string",
    "description": "The date the document was officially recorded by the county recorder, formatted as 8 digits in YYYYMMDD (e.g., '20230316' for March 16, 2023). Look for the recording stamp / 'Recorded on' text — do NOT use the document's signing date or notarization date if those differ from the recording date."
  }
}
"""

EXTRACT_INSTRUCTIONS = (
    "These are scanned U.S. real-estate deed (DED) documents recorded with a county recorder. "
    "Use only what is printed on the document — do not infer or guess. "
    "Return null for any field that is not present. "
    "RecordingDate must be exactly 8 digits in YYYYMMDD format."
)

print(f"Source volume: {VOLUME_PATH}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Parse stage — `ai_parse_document` over the volume
# MAGIC
# MAGIC All rows are retained, including those with non-null `error_status`. Confidence and
# MAGIC structure are kept inside the VARIANT for downstream consumers.

# COMMAND ----------

raw_files = (
    spark.read.format("binaryFile")
    .option("pathGlobFilter", "*.{pdf,PDF,tif,TIF,tiff,TIFF,jpg,jpeg,png}")
    .option("recursiveFileLookup", "true")
    .load(VOLUME_PATH)
)

parsed = (
    raw_files
    .withColumn(
        "parsed",
        expr(
            "ai_parse_document(content, "
            "map('version', '2.0', 'imageOutputPath', '/Volumes/fins_genai/unstructured_documents/deeds/images/', 'descriptionElementTypes', '*'))"
        ),
    )
    # image_name == file stem without extension, used as the join key everywhere downstream
    .withColumn("image_name", F.regexp_extract("path", r"([^/]+)\.[^/.]+$", 1))
    .select("path", "image_name", "length", "modificationTime", "parsed")
)

(
    parsed.write.mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("deeds_parsed")
)

print(f"deeds_parsed rows: {spark.table('deeds_parsed').count()}")
display(spark.sql("SELECT image_name, length, parsed:error_status AS error_status FROM deeds_parsed"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Element flatten + low-confidence analytics
# MAGIC
# MAGIC Uses `transform(try_cast(... AS ARRAY), e -> ...)` + `posexplode_outer` so non-text
# MAGIC elements (figures, tables) and any null fields are preserved.
# MAGIC
# MAGIC OCR confidence is left-skewed (heavy mass near 1.0), so a pure z-score makes a poor
# MAGIC global threshold. The `conf_tier` column combines three complementary signals — any
# MAGIC of them firing is enough to be flagged:
# MAGIC
# MAGIC | Signal | Catches | Rule |
# MAGIC |---|---|---|
# MAGIC | Absolute floor | Universally bad elements regardless of corpus shape | `confidence < ABS_FLOOR` (default 0.5) |
# MAGIC | Corpus percentile | Worst slice of this run, distribution-agnostic | `confidence < corpus_p10` |
# MAGIC | Doc-internal z-score | Local anomalies in otherwise-clean documents | `doc_z < −1.5` |
# MAGIC
# MAGIC Tier logic:
# MAGIC - `low` — absolute floor OR corpus-percentile fires
# MAGIC - `borderline` — only the doc-z signal fires
# MAGIC - `ok` — none fire
# MAGIC - `na` — confidence is null (e.g., figure with no text)
# MAGIC
# MAGIC `corpus_z` and `doc_z` are kept as raw diagnostic columns for ad-hoc analysis even
# MAGIC though the tier no longer hinges on the corpus z-score alone.

# COMMAND ----------

from pyspark.sql.window import Window

# Tunables — adjust empirically as the corpus grows
ABS_FLOOR = 0.5         # confidence < this -> always 'low' regardless of distribution
CORPUS_PERCENTILE = 0.10  # confidence < this corpus quantile -> 'low'
DOC_Z_THRESHOLD = -1.5    # doc-internal z-score < this -> 'borderline'

elements_raw = (
    spark.table("deeds_parsed")
    # Pull elements as ARRAY<VARIANT> so posexplode_outer works
    .withColumn(
        "elements",
        expr("try_cast(parsed:document:elements AS ARRAY<VARIANT>)"),
    )
    .selectExpr(
        "path",
        "image_name",
        "posexplode_outer(elements) AS (element_idx, element)",
    )
    .selectExpr(
        "path",
        "image_name",
        "element_idx",
        "try_cast(element:bbox[0]:page_id AS INT) AS page_id",
        "try_cast(element:type AS STRING) AS type",
        "try_cast(element:content AS STRING) AS content",
        "try_cast(element:confidence AS DOUBLE) AS confidence",
        "element:bbox AS bbox",
    )
)

# Corpus mean / std / pX as broadcast scalars — one pass over all rows
corpus_row = elements_raw.agg(
    F.avg("confidence").alias("corpus_mean"),
    F.stddev("confidence").alias("corpus_std"),
    F.expr(f"percentile_approx(confidence, {CORPUS_PERCENTILE})").alias("corpus_pX"),
).collect()[0]
CORPUS_MEAN = corpus_row["corpus_mean"]
CORPUS_STD = corpus_row["corpus_std"]
CORPUS_PX = corpus_row["corpus_pX"]
print(
    f"corpus_mean={CORPUS_MEAN}  corpus_std={CORPUS_STD}  "
    f"corpus_p{int(CORPUS_PERCENTILE * 100)}={CORPUS_PX}  "
    f"abs_floor={ABS_FLOOR}  doc_z_threshold={DOC_Z_THRESHOLD}"
)

# Per-document mean/std via window — added to every element row in the same partition
doc_window = Window.partitionBy("image_name")

elements_df = (
    elements_raw
    .withColumn("doc_mean", F.avg("confidence").over(doc_window))
    .withColumn("doc_std", F.stddev("confidence").over(doc_window))
    # Diagnostic z-scores — kept for ad-hoc analysis even though the tier no longer
    # depends on the corpus z-score (skewed distributions make it unreliable as a gate).
    .withColumn(
        "corpus_z",
        F.when(
            (F.col("confidence").isNotNull()) & (F.lit(CORPUS_STD) > 0),
            (F.col("confidence") - F.lit(CORPUS_MEAN)) / F.lit(CORPUS_STD),
        ),
    )
    .withColumn(
        "doc_z",
        F.when(
            (F.col("confidence").isNotNull()) & (F.col("doc_std") > 0),
            (F.col("confidence") - F.col("doc_mean")) / F.col("doc_std"),
        ),
    )
    # Three independent flags
    .withColumn("flag_abs_floor", F.col("confidence") < F.lit(ABS_FLOOR))
    .withColumn(
        "flag_corpus_pX",
        (F.lit(CORPUS_PX).isNotNull())
        & (F.col("confidence") < F.lit(CORPUS_PX)),
    )
    .withColumn("flag_doc_z", F.col("doc_z") < F.lit(DOC_Z_THRESHOLD))
    # Layered tier: any "low" signal -> low; only doc-z fires -> borderline; else ok / na
    .withColumn(
        "conf_tier",
        F.when(F.col("confidence").isNull(), F.lit("na"))
        .when(
            F.coalesce(F.col("flag_abs_floor"), F.lit(False))
            | F.coalesce(F.col("flag_corpus_pX"), F.lit(False)),
            F.lit("low"),
        )
        .when(F.coalesce(F.col("flag_doc_z"), F.lit(False)), F.lit("borderline"))
        .otherwise(F.lit("ok")),
    )
)

(
    elements_df.write.mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("deeds_parsed_elements")
)

print(f"deeds_parsed_elements rows: {spark.table('deeds_parsed_elements').count()}")

# Per-doc tier counts — quick eyeball of where low-confidence concentrates
display(
    spark.sql(
        """
        SELECT image_name,
               COUNT(*)                                              AS n_elements,
               COUNT(confidence)                                     AS n_with_conf,
               ROUND(AVG(confidence), 4)                             AS conf_mean,
               SUM(CASE WHEN conf_tier = 'low'        THEN 1 ELSE 0 END) AS n_low,
               SUM(CASE WHEN conf_tier = 'borderline' THEN 1 ELSE 0 END) AS n_borderline,
               SUM(CASE WHEN conf_tier = 'ok'         THEN 1 ELSE 0 END) AS n_ok,
               SUM(CASE WHEN conf_tier = 'na'         THEN 1 ELSE 0 END) AS n_na
        FROM deeds_parsed_elements
        GROUP BY image_name
        ORDER BY n_low DESC, image_name
        """
    )
)

# A peek at the actually-flagged elements so a reviewer can see what `low` looks like
display(
    spark.sql(
        """
        SELECT image_name, page_id, element_idx, type,
               ROUND(confidence, 4) AS confidence,
               ROUND(corpus_z, 3)   AS corpus_z,
               ROUND(doc_z, 3)      AS doc_z,
               conf_tier,
               SUBSTRING(content, 1, 120) AS content_preview
        FROM deeds_parsed_elements
        WHERE conf_tier IN ('low', 'borderline')
        ORDER BY conf_tier DESC, confidence ASC
        LIMIT 50
        """
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Per-page confidence stats → `deeds_parsed_pages`

# COMMAND ----------

pages_df = (
    spark.table("deeds_parsed_elements")
    .groupBy("path", "image_name", "page_id")
    .agg(
        F.count(F.lit(1)).alias("n_elements"),
        F.avg("confidence").alias("conf_mean"),
        F.stddev("confidence").alias("conf_stddev"),
        F.min("confidence").alias("conf_min"),
        F.max("confidence").alias("conf_max"),
    )
)

(
    pages_df.write.mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("deeds_parsed_pages")
)

display(
    spark.sql(
        """
        SELECT image_name, page_id, n_elements,
               ROUND(conf_mean, 4) AS conf_mean,
               ROUND(conf_stddev, 4) AS conf_stddev,
               ROUND(conf_min, 4) AS conf_min,
               ROUND(conf_max, 4) AS conf_max
        FROM deeds_parsed_pages
        ORDER BY image_name, page_id
        """
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Per-doc confidence statistics
# MAGIC
# MAGIC Distribution stats + per-page roll-ups (`worst_page_*`, `max_page_stddev`) so
# MAGIC reviewers can spot "one bad page in an otherwise-good doc". All descriptive — no
# MAGIC filtering applied.

# COMMAND ----------

# Doc-level distribution stats over elements
doc_conf = (
    spark.table("deeds_parsed_elements")
    .groupBy("image_name")
    .agg(
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
        F.avg(F.when(F.col("confidence") < 0.5, 1.0).otherwise(0.0)).alias("low_conf_pct"),
    )
    .withColumn("conf_iqr", F.col("conf_p75") - F.col("conf_p25"))
)

# Per-page roll-ups: worst page mean + which page it was, plus max page stddev
page_rollups = (
    spark.table("deeds_parsed_pages")
    .groupBy("image_name")
    .agg(
        F.min("conf_mean").alias("worst_page_mean"),
        F.max("conf_stddev").alias("max_page_stddev"),
    )
)

# argmin: page_id where page conf_mean equals the worst_page_mean
worst_page_id = (
    spark.table("deeds_parsed_pages").alias("p")
    .join(page_rollups.alias("r"), "image_name")
    .where(F.col("p.conf_mean") == F.col("r.worst_page_mean"))
    .groupBy("image_name")
    .agg(F.min("p.page_id").alias("worst_page_id"))
)

# error_status_count from raw VARIANT
errors = spark.sql(
    """
    SELECT image_name,
           COALESCE(SIZE(try_cast(parsed:error_status AS ARRAY<VARIANT>)), 0) AS error_status_count
    FROM deeds_parsed
    """
)

doc_stats = (
    doc_conf
    .join(page_rollups, "image_name", "left")
    .join(worst_page_id, "image_name", "left")
    .join(errors, "image_name", "left")
)

display(doc_stats)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Extract stage — `ai_extract` v2.1 on the parsed VARIANT
# MAGIC
# MAGIC v2.1 enables two new options:
# MAGIC - `enableConfidenceScores` → each scalar field returns `{value, confidence_score}`; score in [0, 1]
# MAGIC - `enableCitations` → each field returns `citation_ids`; document-level `metadata.citations` carries
# MAGIC   bbox + page_id (because we pass the parsed VARIANT, not raw text), plus `metadata.pages.image_uri`
# MAGIC   for direct reviewer rendering.

# COMMAND ----------

# v2.1 always wraps scalar leaves as {value, confidence_score?, citation_ids?}.
# Pull the value, the per-field confidence, and the per-field citation_ids array.
# Keep document-level metadata.citations and metadata.pages as VARIANT so the bbox/page_id
# /image_uri structure is preserved end-to-end for reviewers and downstream notebooks.
EXTRACT_FIELDS = [
    "DocumentTitle",
    "BookNumberType",
    "BookNumberParsed",
    "PageNumber",
    "DocumentNumber",
    "RecordingDate",
]


def _field_select_exprs(fields):
    exprs = []
    for f in fields:
        exprs.append(f"try_cast(extracted:response:{f}:value AS STRING) AS {f}")
        exprs.append(f"try_cast(extracted:response:{f}:confidence_score AS DOUBLE) AS {f}_extract_conf")
        exprs.append(
            f"try_cast(extracted:response:{f}:citation_ids AS ARRAY<INT>) AS {f}_citation_ids"
        )
    return exprs


extracted_raw = (
    spark.table("deeds_parsed")
    .withColumn(
        "extracted",
        expr(
            f"""
            ai_extract(
                parsed,
                '{EXTRACT_SCHEMA.replace("'", "''").strip()}',
                map(
                    'version',                '2.1',
                    'enableConfidenceScores', 'true',
                    'enableCitations',        'true',
                    'instructions',           '{EXTRACT_INSTRUCTIONS.replace("'", "''")}'
                )
            )
            """
        ),
    )
    .selectExpr(
        "image_name",
        *_field_select_exprs(EXTRACT_FIELDS),
        # Document-level metadata: citations carry bbox + page_id when input is a parsed VARIANT;
        # pages carry image_uri so reviewers can render the source page directly.
        "extracted:metadata:citations AS citations",
        "extracted:metadata:pages     AS citation_pages",
        "try_cast(extracted:metadata:chunk_type AS STRING) AS citation_chunk_type",
        "try_cast(extracted:error_message AS STRING) AS extraction_error_message",
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Build `deeds_extracted_flat` — extraction + confidence stats joined

# COMMAND ----------

extracted_flat = extracted_raw.join(doc_stats, "image_name", "left")

(
    extracted_flat.write.mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("deeds_extracted_flat")
)

display(spark.table("deeds_extracted_flat"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Distribution plots — parse vs extract confidence per document
# MAGIC
# MAGIC Two panels, side by side, **sharing the same doc-axis** so each row reads left-to-right
# MAGIC for a single document. Documents are sorted by median parse confidence ascending —
# MAGIC worst sit on top.
# MAGIC
# MAGIC **Left — `ai_parse_document` (per-element distribution)**
# MAGIC - Violin: shape of the per-element confidence distribution (thin tail toward 0 means a
# MAGIC   few bad elements; wide bulge means uniformly mediocre OCR).
# MAGIC - Box overlay: median + IQR.
# MAGIC - Small dark dots: every parsed element, jittered.
# MAGIC - Vertical thresholds: `abs_floor` (red) and `corpus_p10` (orange) — parse tier classifier reference.
# MAGIC
# MAGIC **Right — `ai_extract` v2.1 (6 per-field confidence scores)**
# MAGIC - One colored marker per field per doc · the legend names the field.
# MAGIC - Short black bar: median of the 6 fields for that doc.
# MAGIC - Vertical reference at `0.7` — the threshold notebook 02 will use to flag a non-null
# MAGIC   field for review.

# COMMAND ----------

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Parse-side: per-element confidence
plot_pdf = (
    spark.table("deeds_parsed_elements")
    .filter(F.col("confidence").isNotNull())
    .select("image_name", "confidence")
    .toPandas()
)

# Extract-side: 6 per-field confidence scores per doc
extract_conf_pdf = (
    spark.table("deeds_extracted_flat")
    .select("image_name", *[f"{f}_extract_conf" for f in EXTRACT_FIELDS])
    .toPandas()
    .set_index("image_name")
)

# Distinct, qualitative palette for the 6 fields
FIELD_COLORS = {
    "DocumentTitle":    "#d62728",
    "BookNumberType":   "#ff7f0e",
    "BookNumberParsed": "#2ca02c",
    "PageNumber":       "#9467bd",
    "DocumentNumber":   "#8c564b",
    "RecordingDate":    "#17becf",
}
EXTRACT_REVIEW_THRESHOLD = 0.7  # informational; matches LOW_EXTRACT_CONF_THRESHOLD in notebook 02

if plot_pdf.empty:
    print("No non-null confidence values to plot.")
else:
    # Sort docs by median parse confidence — worst on top so the eye lands on them first
    medians = plot_pdf.groupby("image_name")["confidence"].median().sort_values()
    docs = list(medians.index)
    data = [plot_pdf.loc[plot_pdf["image_name"] == d, "confidence"].values for d in docs]

    fig, (ax_parse, ax_extract) = plt.subplots(
        1, 2,
        sharey=True,
        figsize=(15, 0.7 * len(docs) + 2.5),
        gridspec_kw={"width_ratios": [1.6, 1.0]},
    )

    # ---- LEFT panel: ai_parse_document per-element distribution ----
    parts = ax_parse.violinplot(
        data, positions=range(len(docs)), vert=False, widths=0.85, showextrema=False
    )
    for body in parts["bodies"]:
        body.set_facecolor("#9ecae1")
        body.set_edgecolor("#3182bd")
        body.set_alpha(0.55)

    ax_parse.boxplot(
        data,
        positions=range(len(docs)),
        vert=False,
        widths=0.25,
        showmeans=True,
        meanline=False,
        flierprops=dict(marker="o", markersize=3, markerfacecolor="#222", alpha=0.6),
        medianprops=dict(color="#08306b", linewidth=2),
        meanprops=dict(marker="D", markeredgecolor="#000", markerfacecolor="#fff", markersize=5),
    )

    rng = np.random.default_rng(42)
    for i, vals in enumerate(data):
        jitter = rng.uniform(-0.18, 0.18, size=len(vals))
        ax_parse.scatter(vals, np.full(len(vals), i) + jitter, s=8, alpha=0.45, color="#08306b")

    ax_parse.axvline(ABS_FLOOR, color="#e6550d", linestyle="--", linewidth=1.4, label=f"abs_floor = {ABS_FLOOR}")
    if CORPUS_PX is not None:
        ax_parse.axvline(
            CORPUS_PX,
            color="#fdae6b",
            linestyle="--",
            linewidth=1.4,
            label=f"corpus_p{int(CORPUS_PERCENTILE * 100)} = {CORPUS_PX:.3f}",
        )

    ax_parse.set_yticks(range(len(docs)))
    ax_parse.set_yticklabels(docs)
    ax_parse.set_xlabel("element confidence")
    ax_parse.set_xlim(min(0.0, plot_pdf["confidence"].min() - 0.02), 1.02)
    ax_parse.set_title("ai_parse_document · per-element confidence")
    ax_parse.grid(axis="x", linestyle=":", alpha=0.4)
    ax_parse.legend(loc="lower left", fontsize=9)

    # ---- RIGHT panel: ai_extract per-field confidence ----
    extract_jitter = rng.uniform(-0.12, 0.12, size=(len(docs), len(EXTRACT_FIELDS)))
    legend_seen = set()
    for i, doc in enumerate(docs):
        if doc not in extract_conf_pdf.index:
            continue
        # Per-doc median bar — short black tick at the median of the 6 fields
        vals_for_doc = [
            float(extract_conf_pdf.loc[doc, f"{f}_extract_conf"])
            for f in EXTRACT_FIELDS
            if pd.notna(extract_conf_pdf.loc[doc, f"{f}_extract_conf"])
        ]
        if vals_for_doc:
            med = float(np.median(vals_for_doc))
            ax_extract.plot(
                [med, med], [i - 0.25, i + 0.25],
                color="#08306b", linewidth=2.5, solid_capstyle="butt", zorder=4,
            )
        # 6 colored markers
        for j, f in enumerate(EXTRACT_FIELDS):
            val = extract_conf_pdf.loc[doc, f"{f}_extract_conf"]
            if val is None or pd.isna(val):
                continue
            ax_extract.scatter(
                float(val),
                i + extract_jitter[i, j],
                s=85,
                marker="o",
                facecolor=FIELD_COLORS[f],
                edgecolor="black",
                linewidth=0.6,
                alpha=0.95,
                zorder=5,
                label=(f if f not in legend_seen else None),
            )
            legend_seen.add(f)

    ax_extract.axvline(
        EXTRACT_REVIEW_THRESHOLD, color="#969696", linestyle="--", linewidth=1.4,
        label=f"review threshold = {EXTRACT_REVIEW_THRESHOLD}",
    )
    ax_extract.set_xlabel("ai_extract field confidence")
    ax_extract.set_xlim(0.0, 1.02)
    ax_extract.set_title("ai_extract v2.1 · per-field confidence")
    ax_extract.grid(axis="x", linestyle=":", alpha=0.4)
    ax_extract.legend(loc="lower left", fontsize=8, ncol=2)

    plt.tight_layout()
    display(fig)
    plt.close(fig)  # prevent Databricks from auto-rendering after display() — fixes the duplicate plot

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Smoke checks

# COMMAND ----------

n_parsed = spark.table("deeds_parsed").count()
n_elements = spark.table("deeds_parsed_elements").count()
n_extracted = spark.table("deeds_extracted_flat").count()
n_with_title = spark.table("deeds_extracted_flat").filter(F.col("DocumentTitle").isNotNull()).count()

print(f"deeds_parsed:           {n_parsed} rows")
print(f"deeds_parsed_elements:  {n_elements} rows")
print(f"deeds_extracted_flat:   {n_extracted} rows ({n_with_title} with non-null DocumentTitle)")

assert n_parsed == n_extracted, "Row count mismatch between parsed and extracted_flat"
assert n_elements > 0, "No elements were flattened — check ai_parse_document output"
print("OK")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Spot-check v2.1 outputs
# MAGIC
# MAGIC Per-field bare value, per-field confidence score, per-field citation IDs, and the
# MAGIC document-level citations array (bbox + page_id when input is the parsed VARIANT).

# COMMAND ----------

display(
    spark.table("deeds_extracted_flat").selectExpr(
        "image_name",
        *[c for f in EXTRACT_FIELDS for c in (f, f"{f}_extract_conf", f"{f}_citation_ids")],
        "citation_chunk_type",
    )
)

# COMMAND ----------

display(
    spark.table("deeds_extracted_flat").selectExpr(
        "image_name",
        "citations",
        "citation_pages",
    )
)

# COMMAND ----------

