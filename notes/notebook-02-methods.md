# Notebook 02 — Evaluate Without Ground Truth: Methods

## Purpose

Score and rank documents and individual extracted cells for human review **without** any ground truth. Runs after `01_parse_and_extract`. Outputs are purely additive — nothing is filtered or overwritten.

## Inputs / outputs

**Inputs**
- `deeds_extracted_flat` — per-doc extraction output + parse confidence aggregates + `ai_extract` v2.1 per-field `extract_conf` and `citation_ids`
- `deeds_parsed_elements` — per-element parsed text (only used by §4 to assemble the eval dataset's `document_text` field for traceability)

**Outputs**
- MLflow run under `/Users/q.yu@databricks.com/deeds_poc_eval_no_gt`
- `deeds_review_flags` — per-doc table with sortable `review_priority`
- `deeds_field_review` — long-format per-(doc, field) cell-level review queue, with cited bbox/page resolved

## Two signal layers (deterministic — no LLM judge)

| Layer | Source | Granularity | Catches |
|---|---|---|---|
| Doc-level corpus outliers | `ai_parse_document` confidence aggregates | per document | "this whole doc is OCR'd badly" |
| Field-level extract confidence | `ai_extract` v2.1 `<field>_extract_conf` | per cell | "this specific cell is uncertain" |

Both layers are pure logic / statistics — no per-doc LLM API cost. An earlier version had a third layer (an absence-aware LLM judge) but it was removed for cost reasons. If you need a deeper "is this null actually missing in the source?" check, see the cross-confidence sibling notebook (`02_eval_no_gt_with_cross_conf.py`), which uses parse-side bbox confidence to flag the same failure mode without an extra LLM call.

## Design rules

- **Empty/null fields are not failures.** A blank value is correct when the entity isn't in the document. Heuristic shape rules and field-level confidence flags only fire on **non-null** values; a missing field has no confidence to flag.
- **Corpus-relative, not against a persisted baseline.** Every flag is computed against the current run's corpus, so each batch is self-describing.
- **No filtering.** All signals are additive columns + a sortable `review_priority`.
- **No LLM judge.** All scorers are deterministic / statistical — `mlflow.genai.evaluate` here is a structured logging vehicle for the deterministic signals, not a model-as-judge runner.

## Methods by cell

### Cell 4 — Setup
- Constants: `LOW_EXTRACT_CONF_THRESHOLD = 0.7` (only field-level tunable), `EXTRACT_FIELDS` (canonical 6-entity list).

### Cell 6 — Doc-level outlier flags
Four Booleans per document, computed corpus-relative:

| Flag | Rule |
|---|---|
| `outlier_low_mean` | `conf_mean ≤ corpus_p10(conf_mean)` |
| `outlier_high_variance` | `conf_stddev > Q3 + 1.5·IQR(conf_stddev)` (Tukey upper fence) |
| `outlier_bad_page` | `worst_page_mean ≤ corpus_p10(worst_page_mean)` |
| `has_parse_error` | `error_status_count > 0` |

IQR (not z-score) on stddev because OCR-variance distributions aren't normal.

### Cell 8 — Heuristic shape checks
Regex/digit format rules per field, **only fire on non-null values**:
- `RecordingDate` — `^\d{8}$`
- `BookNumberParsed` / `PageNumber` / `DocumentNumber` — must contain a digit
- AND-rolled into `extraction_shape_ok`
- `extraction_completeness = non_null_count / 6` is informational only — never used in priority math.

### Cell 10 — Field-level low-conf flags
Per field:
```
<field>_low_extract_conf =
   value IS NOT NULL
   AND extract_conf IS NOT NULL
   AND extract_conf < LOW_EXTRACT_CONF_THRESHOLD
```
Plus per-doc roll-ups:
- `low_conf_field_count` (int 0–6)
- `low_conf_fields` (array of field names that fired)
- `min_extract_conf` (single sortable signal; nulls coalesced to 1.0)

The non-null guard is what makes "absence is OK" actually work.

### Cell 12 — Eval dataset construction
One record per document, `mlflow.genai.evaluate`-shaped:
- `inputs` carries everything the deterministic scorers need (parse_stats, extracted, shape_ok, completeness, low-conf signals) plus `document_text` for traceability in the MLflow run UI.
- `outputs` carries the model's claims: `extracted`, `extract_conf`, `citation_ids`. Putting confidence/citations under outputs frames them as claims to verify, not context to trust.

### Cell 14 — Scorers (deterministic only)
**Pass-through scorers** (`@scorer` wrappers around precomputed values): `parse_conf_mean`, `parse_conf_stddev`, `parse_worst_page_mean`, `extraction_shape_ok`, `extraction_completeness`, `min_extract_conf`, `low_conf_field_count`. They exist so MLflow shows each metric per row and aggregates them across the corpus. No model-as-judge calls — the previous absence-aware LLM judge was removed for cost.

### Cell 16 — `mlflow.genai.evaluate`
Single call, seven scorers, no `predict_fn` (outputs are pre-computed). Logs all metrics to the experiment.

### Cell 18 — `deeds_review_flags`
Per-doc table. `review_priority` formula:
```
review_priority =
  outlier_low_mean              (0–1)
+ outlier_high_variance         (0–1)
+ outlier_bad_page              (0–1)
+ has_parse_error               (0–1)
+ !extraction_shape_ok          (0–1)
+ low_conf_field_count          (0–6)
```
Range 0–11. Field-level term dominates by design — many uncertain cells outranks one general-issue flag, matching how a human triages.

### Cell 20 — `deeds_field_review`
Long-format cell-level queue. Inclusion rule: row is included if **either** `<field>_low_extract_conf == True` **OR** the per-field shape check failed. Each row carries `extract_conf`, `citation_ids`, and `cited_regions` (JSON-stringified bboxes resolved from the doc's `citations` array). `reasons` array surfaces *why* the cell is in the queue.

Cited regions are JSON-stringified rather than struct columns because some cells cite many regions, some cite none — JSON keeps the Delta schema simple.

## Reviewer workflow

1. Open `deeds_review_flags`, sort `review_priority DESC` → which doc to look at first.
2. Read its `low_conf_fields` array → starting list of cells to verify.
3. Open `deeds_field_review`, filter on `image_name`, sort `extract_conf ASC` → cell-by-cell with model value, confidence, reasons, and bbox(es).
4. Use `cited_regions[*].coord` + `page_id` to jump to the source page region.

## Knobs

| Knob | Default | When to change |
|---|---|---|
| `LOW_EXTRACT_CONF_THRESHOLD` | 0.7 | Raise if `min_extract_conf` clusters near 1.0; lower if it never fires |
| Tukey fence multiplier on `conf_stddev` | 1.5·IQR | Lower (e.g. 1.0) for stricter flagging, higher (3.0) for looser |

## Failure modes to watch

- **`low_conf_field_count` always 0** — `LOW_EXTRACT_CONF_THRESHOLD` is too low. Raise it.
- **Same docs flag every batch via `outlier_*`** — corpus-relative flags only catch *relative* outliers. If every doc is uniformly bad, none flag. Pair with absolute thresholds (e.g. `conf_mean < 0.6`) once you have a calibrated baseline.
- **Confident-over-OCR-garbage cells slip through** — pure field-level `extract_conf` cannot catch the case where the extractor reasoned cleanly over mis-OCR'd text. Run the cross-confidence sibling notebook (`02_eval_no_gt_with_cross_conf.py`) instead, which compares `extract_conf` to parse-side bbox confidence.

## Related docs

- `docs/plan-cross-confidence-check.md` — future work on cross-checking `ai_extract` confidence against `ai_parse_document` element confidence via bbox overlap.
