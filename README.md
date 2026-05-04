# Deed Document Extraction — Databricks POC

End-to-end POC for parsing scanned U.S. real-estate deed documents (PDFs and
TIFFs) on Databricks and extracting structured header fields. Built around
**`ai_parse_document`** for OCR + layout, **`ai_extract` v2.1** (with
per-field confidence scores and citations) for entity extraction, and
**MLflow GenAI** for evaluation.

The goal is a daily production pipeline that:

1. Parses each new document → flattens layout elements → extracts six header
   fields (`DocumentTitle`, `BookNumberType`, `BookNumberParsed`,
   `PageNumber`, `DocumentNumber`, `RecordingDate`).
2. Surfaces per-document and per-cell quality signals — *without* using an
   LLM judge in the per-document hot path. Logic-based scorers
   (extraction confidence, citation presence, format rules, corpus-relative
   outliers) drive a sortable human-review queue.
3. Optionally evaluates against ground truth offline — fuzzy match for
   free-form titles, strict normalized match for IDs and dates, with
   absence-as-a-class (TP / TN / FP / FN / FP_FN per cell).

## Repository layout

```
.
├── notebooks/                  # Dev notebooks — interactive workflow
│   ├── 01_parse_and_extract.py        # Parse + extract + per-doc / per-page stats
│   ├── 02_eval_no_ground_truth.py     # Operational eval (uses LLM judge)
│   └── 03_eval_with_ground_truth.py   # Offline eval against your ground truth CSV
└── production/                 # Databricks Asset Bundle for daily prod
    ├── databricks.yml          # Bundle config (dev + prod targets)
    ├── README.md               # Deploy + run instructions
    ├── resources/
    │   ├── jobs.yml            # deeds_streaming + deeds_analytics
    │   └── volumes.yml         # checkpoint volume
    └── src/
        ├── pipelines/          # Streaming: parse + extract (Trigger.AvailableNow)
        ├── analytics/          # Batch: corpus outliers + review queue, NO judge
        └── shared/             # EXTRACT_FIELDS, ai_extract schema/instructions
```

## End-to-end flow

```
                 PDFs / TIFFs in UC Volume
                            │
         ┌──────────────────┴──────────────────┐
         │  ai_parse_document (DBR 17.1+)      │  ← writes page images for citation rendering
         │  → deeds_parsed (raw VARIANT)       │
         └──────────────────┬──────────────────┘
                            │
         ┌──────────────────┴──────────────────┐
         │  ai_extract v2.1                    │  ← enableConfidenceScores + enableCitations
         │  → deeds_extracted_flat             │     plus per-doc parse-confidence stats
         └───────────┬─────────────┬───────────┘
                     │             │
        ┌────────────┘             └─────────────┐
        ▼                                        ▼
  Operational eval (no GT)               Offline eval (with GT)
  notebooks/02 OR production analytics   notebooks/03
        │                                        │
        ▼                                        ▼
  deeds_review_flags    (per-doc)        deeds_extracted_vs_gt   (per-cell)
  deeds_field_review    (per-cell)       deeds_eval_metrics_gt   (P/R/F1)
                                          deeds_calibration_gt   (conf vs correctness)
```

## Quick start

### Prerequisites

- Databricks workspace with **DBR 17.1+** (required for `ai_parse_document`) or serverless compute.
- Unity Catalog access — the project assumes:
  - Catalog: `fins_genai` (prod) / `dev_fins_genai` (dev) — change in
    `production/databricks.yml` and the dev notebooks if you use different names.
  - Schema: `unstructured_documents`
  - Volume `deeds/` populated with your input PDFs/TIFFs.
- Databricks CLI configured with a profile (`DEFAULT` by default).

### Bring your own data

This repo does not ship sample documents or ground truth. To run the
notebooks end-to-end:

1. Upload your PDFs/TIFFs to `/Volumes/<catalog>/unstructured_documents/deeds/`.
2. (Optional) Upload a ground-truth CSV to a sub-path under the same volume,
   e.g. `/Volumes/<catalog>/unstructured_documents/deeds/ground_truth/GroundTruth.csv`.
   The CSV is column-oriented: row 0 = `ImageName` plus one column per
   document; subsequent rows = field values per document. Field names must
   match the six entities listed above. Blank cells indicate "entity not
   present in this document" — null is a meaningful value, not missing
   data.

### Run interactively (dev path)

1. Open `notebooks/01_parse_and_extract.py` in your workspace and run it.
   This produces `deeds_parsed`, `deeds_parsed_elements`, `deeds_parsed_pages`,
   and `deeds_extracted_flat`, plus a confidence-distribution plot per document.
2. Run `notebooks/02_eval_no_ground_truth.py` for the operational quality view
   (corpus outliers + field-level confidence flags + citation-aware LLM judge +
   `deeds_review_flags` + `deeds_field_review`).
3. Optionally run `notebooks/03_eval_with_ground_truth.py` against your
   ground-truth CSV for accuracy / precision / recall / F1 per field plus a
   confidence-vs-correctness calibration check.

### Deploy the production bundle

```bash
cd production
databricks bundle validate -t dev
databricks bundle deploy   -t dev

# Manual run for first-time validation
databricks bundle run deeds_streaming -t dev
databricks bundle run deeds_analytics -t dev

# Promote when ready
databricks bundle deploy -t prod
```

Dev jobs deploy `PAUSED`; prod deploys `UNPAUSED` and runs daily 02:00 UTC
(streaming) → 03:00 UTC (analytics). See `production/README.md` for full
deploy details and tunable variables.

## Key design decisions

### Confidence is descriptive, not a filter

Element-level confidence from `ai_parse_document` is recorded with three
parallel quality flags (absolute floor, corpus-percentile, doc-internal
z-score) but **never used to drop rows**. Reviewers see the signal and
decide; the pipeline never silently discards a document.

### Field-level (not just doc-level) review queue

`ai_extract` v2.1's per-field `confidence_score` and `citation_ids` enable a
per-cell queue (`deeds_field_review`) rather than just a per-doc flag list.
A reviewer sees "look at the BookNumberType cell of this doc" with the cited
bbox + page image rendered for click-to-region triage.

### Absence-as-a-class

Empty / null is a meaningful prediction — many entities (e.g., `BookNumberType`)
are legitimately not present in some deeds. The comparator and metrics
explicitly classify each cell as TP / TN / FP / FN / FP_FN and treat
`(null, null)` as a true negative.

### No LLM judge in production

The dev evaluation (`notebooks/02`) uses an `mlflow.genai.judges.make_judge`
categorical judge for absence-aware plausibility. The production bundle
**drops the judge** and replaces it with three deterministic logic-based
signals: `field_low_extract_conf`, `field_no_citation` (non-null with empty
citation_ids and conf below a gate), `field_format_specific_violation`.
Cost / latency / reproducibility favor logic for a daily 10k-doc pipeline.

### Hybrid comparator for ground-truth eval

Per-field comparator selection — fuzzy (Levenshtein ≥ 0.85) for free-form
`DocumentTitle`, strict normalized exact for IDs and dates. A 1-digit
difference in a document number is a real error and must not be hidden by
fuzzy logic.

## Requirements

- **Databricks Runtime 17.1+** for `ai_parse_document`
- **Databricks SQL warehouse Pro+** or serverless for AI Functions
- **`mlflow[databricks] >= 3.1.0`** for `mlflow.genai.evaluate` and `make_judge`
- **`ai_extract` version 2.1** for confidence scores + citations
- A **Unity Catalog** volume for source documents and rendered page images

Streaming is configured to use `Trigger.AvailableNow` so the daily job
processes only files that arrived since the last run, then stops — cheaper
than continuous streaming for batch cadences.

## Extracted entities

The six fields the pipeline extracts from each deed are:

| Field | Type | Notes |
|---|---|---|
| `DocumentTitle` | string | Document type heading at top of page 1 (e.g. "WARRANTY DEED") |
| `BookNumberType` | string | Short prefix for the recording book (e.g. `R`, `BK`) |
| `BookNumberParsed` | string (digits) | Book / volume number |
| `PageNumber` | string | Page or page range, formatted as printed (e.g. `0839`, `2085-2087`) |
| `DocumentNumber` | string | County recorder's instrument number, leading zeros preserved |
| `RecordingDate` | string | Always 8 digits in `YYYYMMDD` |

The `ai_extract` schema (with full per-field guidance prompts) lives in
`production/src/shared/extract_schema.py` and is mirrored inline in
`notebooks/01_parse_and_extract.py`.

## Contributing / extending

- **Add a new entity** to extraction: edit `EXTRACT_SCHEMA` and
  `EXTRACT_FIELDS` in `notebooks/01_parse_and_extract.py` and
  `production/src/shared/`. Add corresponding GT column. The downstream
  notebooks iterate over `EXTRACT_FIELDS` so most code adapts automatically.
- **Tune review thresholds**: edit the bundle variables in
  `production/databricks.yml` (`low_extract_conf_threshold`,
  `high_conf_threshold`, `no_citation_gate_conf`).
- **Add a new format rule** (e.g., 4-digit minimum on `DocumentNumber`):
  edit the analytics notebook in `production/src/analytics/`.

## License

POC code; license to be added before public release.
