# Notebook 02 (Cross-Conf variant) — Methods

## Purpose

Sibling of `02_eval_no_ground_truth.py`. Adds a **fourth signal layer** that cross-checks `ai_extract` field-level confidence against `ai_parse_document` element-level confidence over the cited bbox region. Catches the failure mode neither score alone can flag:

> **The model is confident over OCR garbage.** `extract_conf` is high because the extractor reasoned cleanly over what it saw, but the source element was mis-OCR'd in the first place — so the value is wrong with high confidence.

The three layers from the original notebook 02 are unchanged. See `docs/notebook-02-methods.md` for those, and `docs/plan-cross-confidence-check.md` for the original design proposal this implements.

## Inputs / outputs

**Inputs** — same as the original notebook 02:
- `deeds_extracted_flat` — extraction output + per-field `extract_conf` and `citation_ids`
- `deeds_parsed_elements` — per-element parsed text + bbox + confidence

**Outputs** — suffixed `_xconf` so the original notebook 02's tables stay intact:
- MLflow run under `/Users/q.yu@databricks.com/mlflow_experiments/deeds_poc_eval_no_gt_xconf`
- `deeds_review_flags_xconf` — per-doc table with the new `min_source_conf`, `calibration_mismatch_field_count`, and per-field `<field>_source_conf` / `<field>_calibration_mismatch` / `<field>_model_uncertain_clean_source` / `<field>_consistent_low_source` / `<field>_citation_unmatched` columns
- `deeds_field_review_xconf` — long-format per-(doc, field) cell-level review queue with `source_conf`, `calibration_mismatch`, sibling flags, and `citation_unmatched` per row

## Three signal layers (all deterministic — no LLM judge)

| Layer | Source | Granularity | Catches |
|---|---|---|---|
| Doc-level corpus outliers | `ai_parse_document` confidence aggregates | per document | "this whole doc is OCR'd badly" |
| Field-level extract confidence | `ai_extract` v2.1 `<field>_extract_conf` | per cell | "this specific cell is uncertain" |
| **Cross-confidence (new)** | **`ai_extract` cited bbox ↔ `ai_parse_document` element confidence via IoU** | **per cell** | **"model is confident over OCR garbage"** |

All three are pure logic / statistics — no per-doc LLM API cost. The previous absence-aware LLM judge was removed for cost; the cross-confidence layer subsumes its main signal (calibration mismatch) at the field level.

## What is new (vs `02_eval_no_ground_truth.py`)

Sections 1–3b are identical (only the experiment path and table-name constants differ in section 1). The new and changed sections:

### §3c — Cross-confidence: bbox-overlap `<field>_source_conf` (NEW)

Vectorized PySpark pipeline (per `plan-cross-confidence-check.md` lines 110–121, the production-shape variant):

1. **Explode `<field>_citation_ids`** into long format `(image_name, field, citation_id)` — one pass per field, then `unionByName`.
2. **Decode + explode `metadata.citations`** (a VARIANT) into long format `(image_name, citation_id, page_id, coord)` — uses `from_json` with an explicit schema so the pipeline doesn't depend on VARIANT-accessor syntax.
3. **Join field-citations to citation-bboxes** on `(image_name, citation_id)`.
4. **Inner-join to `deeds_parsed_elements`** on `(image_name, page_id)`. Element bbox is parsed from VARIANT defensively — array form `[x0, y0, x1, y1]` first, falls back to struct form `{x0, y0, x1, y1}`.
5. **Apply `iou_udf`**, filter `iou >= IOU_THRESHOLD`, aggregate `min(confidence)` per `(image_name, field)` → `source_conf`. `min` (not `mean`) because a single low-confidence element in the cited region is enough to taint the source.
6. **Pivot to wide on `field`** so the result is one row per `image_name` with `<field>_source_conf` columns; left-join back onto `field_flagged`.

A driver-side IoU sanity check runs before step 1 — identical boxes → 1.0, disjoint → 0.0, 50% overlap → 1/3, swapped-axis input → 1.0. Asserts fail loud if the math drifts.

A `<field>_citation_unmatched` boolean is set when a field has non-empty `citation_ids` but no parsed element overlaps any cited bbox (`source_conf IS NULL`). Surface for triage; not rolled into `review_priority` for the first cut — usually means the extractor hallucinated a region.

### §3d — Calibration-mismatch flags (NEW)

Three booleans per `(doc, field)`, only when **both** `extract_conf` and `source_conf` are non-null:

| Flag | Rule | Reading |
|---|---|---|
| `<field>_calibration_mismatch` | `extract_conf > 0.8 AND source_conf < 0.5` | Confident over OCR garbage — top of queue |
| `<field>_model_uncertain_clean_source` | `extract_conf < 0.6 AND source_conf > 0.85` | Source clean, model hesitated — schema/instruction issue |
| `<field>_consistent_low_source` | `extract_conf < 0.6 AND source_conf < 0.6` | Both low — already flagged by other layers |

Roll-ups: `calibration_mismatch_field_count` (0–6) and `min_source_conf` (single sortable signal).

### §4 — Eval dataset (CHANGED)

Adds `source_conf` (per-field map), `min_source_conf`, and `calibration_mismatch_field_count` to `inputs`. `outputs` is unchanged — the model's claims (extracted, extract_conf, citation_ids) stay framed as outputs to verify, not context to trust.

### §5 — Scorers (CHANGED)

Adds two new pass-through scorers on top of the seven deterministic scorers from `02_eval_no_ground_truth.py`:
- `min_source_conf` — single sortable signal in MLflow
- `calibration_mismatch_field_count` — 0–6 count of fields where high `extract_conf` met low `source_conf`

No LLM judge — all scorers are deterministic. The three calibration flags at the field level cover what the previous judge was meant to catch.

### §6 — `genai.evaluate` (CHANGED)

Same call shape, now passes 9 scorers: the original 7 deterministic ones + the 2 new cross-conf scorers.

### §7 — `deeds_review_flags_xconf` (CHANGED)

Adds `calibration_mismatch_field_count` to `review_priority`:

```
review_priority =
  outlier_low_mean              (0–1)
+ outlier_high_variance         (0–1)
+ outlier_bad_page              (0–1)
+ has_parse_error               (0–1)
+ !extraction_shape_ok          (0–1)
+ low_conf_field_count          (0–6)
+ calibration_mismatch_field_count   (0–6)   <-- NEW
```

New range 0–17. Field-level signals dominate; a doc with multiple calibration mismatches outranks a doc with only doc-level outliers — matches how a human triages.

The per-field cross-confidence columns (`<field>_source_conf`, three derived booleans, `<field>_citation_unmatched`) are projected onto the table for cell-level inspection without joining back to `deeds_field_review_xconf`.

### §8 — `deeds_field_review_xconf` (CHANGED)

Inclusion rule extended: a row is included if **any** of `<field>_low_extract_conf`, shape violation, **or** `<field>_calibration_mismatch` is true.

Per-row schema gains: `source_conf DOUBLE`, `calibration_mismatch BOOLEAN`, `model_uncertain_clean_source BOOLEAN`, `consistent_low_source BOOLEAN`, `citation_unmatched BOOLEAN`. The `reasons` array can now contain `"calibration_mismatch"`, `"model_uncertain_clean_source"`, `"consistent_low_source"`.

Display ordering: `calibration_mismatch DESC, extract_conf ASC, image_name` — confident-over-garbage cells float to the top; otherwise lowest-confidence first.

## Reviewer workflow

1. Open `deeds_review_flags_xconf`, sort `review_priority DESC`. Top doc(s) are highest priority.
2. Look at `min_source_conf` and `calibration_mismatch_field_count` to gauge how much of the priority is cross-confidence-driven.
3. Open `deeds_field_review_xconf`, filter on `image_name`, default sort surfaces `calibration_mismatch = TRUE` rows first. These are the cells where the model claimed high confidence but the underlying OCR was poor.
4. Use `cited_regions[*].coord` + `page_id` to jump to the cited bbox in the source page; visually compare model value, OCR'd text, and the actual scan.
5. `model_uncertain_clean_source` cells flag a different remediation — clean OCR, hesitant model. Usually a schema/instruction issue, not a parsing one.
6. `citation_unmatched = TRUE` is rare and usually means the extractor cited a region the parser did not produce — investigate the source page directly.

## Knobs

| Knob | Default | When to change |
|---|---|---|
| `IOU_THRESHOLD` | 0.3 | Lower (e.g. 0.0 for any-intersection) if cited bboxes are much smaller than parsed elements (extractor cites a phrase inside a paragraph). Raise toward 0.5 if elements heavily overlap. |
| `CALIBRATION_MISMATCH_EXTRACT_CONF` / `_SOURCE_CONF` | 0.8 / 0.5 | Tune by inspecting the (extract_conf, source_conf) joint distribution; default catches the worst quadrant. |
| `MODEL_UNCERTAIN_EXTRACT_CONF` / `_SOURCE_CONF` | 0.6 / 0.85 | Adjust together; this flag is informational unless you're running a schema-tuning loop. |
| `CONSISTENT_LOW_CONF` | 0.6 | Should match `LOW_EXTRACT_CONF_THRESHOLD` ± 0.1. |
| Output table suffix `_xconf` | hardcoded | Drop the suffix once you decide this notebook replaces the original. |

## Failure modes to watch

- **`source_conf` all null after §3c.** Either the citation VARIANT decode failed or the bbox VARIANT decode failed. The notebook prints a sample of each at the top of §3c — inspect the output and adjust the schemas. Most common cause: `bbox` is in `[x, y, w, h]` form rather than `[x0, y0, x1, y1]`. The IoU UDF normalizes axis order but not box parameterization, so `_bbox_to_array` would need a third fallback.
- **Massive `citation_unmatched` count.** Either IoU threshold is too high or page_id is mismatched between citation and parsed element. Lower `IOU_THRESHOLD` to 0.0 (any intersection) as a diagnostic; if matches still fail, inspect a few citation `page_id` values vs `deeds_parsed_elements.page_id` for the same doc.
- **`calibration_mismatch_field_count` is always 0.** Either thresholds are too tight or the corpus genuinely has no confident-over-garbage cells. Raise `CALIBRATION_MISMATCH_SOURCE_CONF` to 0.7 as a diagnostic.
- **IoU UDF asserts fail at notebook start.** Math regression — do not modify the unit-test expected values; fix `iou_udf` instead. The asserts are the contract.

## Verification

1. **IoU UDF asserts** in §3c run on every notebook execution and must pass.
2. **Sample VARIANT print** in §3c shows the live structure of `citations` and `bbox` — confirm both decode to non-null arrays of 4 numbers.
3. **End-to-end smoke on the 6-doc POC.** Expected pattern: PDFs (clean) get `min_source_conf` close to 1.0; TIFFs (scanned) show lower `min_source_conf`, especially on `BookNumberType` / `PageNumber`. At least one cell with non-empty `citation_ids` should resolve to a non-null `source_conf`.
4. **GT-mode cross-check (optional, manual).** After running notebook 03, join `deeds_extracted_vs_gt` to `deeds_field_review_xconf` on `(image_name, field)` and check whether `calibration_mismatch = TRUE` rows skew toward `classification IN ('FP', 'FN', 'FP_FN')` rather than TP/TN. If yes, the cross-check is meaningful on this corpus and the threshold defaults are good.

## Related docs

- `docs/notebook-02-methods.md` — the unchanged three-layer methodology this builds on.
- `docs/plan-cross-confidence-check.md` — the original design proposal.
- `docs/notebook-03-methods.md` — sibling with-GT eval; the GT-mode cross-check above lives here.
