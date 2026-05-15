# Notebook 03 — Evaluate With Ground Truth: Methods

## Purpose

Offline evaluation of extraction quality against `data/GroundTruth.csv`. Produces per-field accuracy / precision / recall / F1, a per-cell comparison table with the model's own confidence claims attached, and a calibration check on whether `extract_conf` correlates with correctness. Independent of notebook 02 — both can run after notebook 01 in any order.

## Inputs / outputs

**Inputs**
- `deeds_extracted_flat` — extraction output + per-field `extract_conf` and `citation_ids` from notebook 01
- `data/GroundTruth.csv` (uploaded to a Volume) — column-oriented, one column per document

**Outputs**
- `deeds_ground_truth` — transposed GT, row-per-document
- `deeds_extracted_vs_gt` — per-cell comparison + `extract_conf` + `cited_regions` + `high_conf_wrong`
- `deeds_eval_metrics_gt` — per-field accuracy / precision / recall / F1
- `deeds_calibration_gt` — confidence-bin vs correctness rate
- MLflow run under `/Users/q.yu@databricks.com/deeds_poc_eval_gt`

## Core methodology

| Aspect | Approach |
|---|---|
| **Comparator** | Hybrid: fuzzy (Levenshtein ≥ 0.85) for `DocumentTitle`; strict equal for IDs and dates after normalization |
| **Absence handling** | Blank-when-absent counts as **true negative** (correct). The classification has 5 outcomes: TP, TN, FP, FN, FP_FN |
| **Architecture** | Table-first. Match logic runs in PySpark and persists to `deeds_extracted_vs_gt`; MLflow scorers are pass-throughs that surface the precomputed classifications with informative rationale |
| **Calibration** | Bins `extract_conf` into 4 ranges, computes correctness rate per bin, prints monotonicity verdict |

## Design rules

- **Null is a meaningful value.** GT records blank when the entity isn't in the document. Both the comparator and the metrics treat (null, null) as a correct true-negative — not a missing answer.
- **Per-field comparators, not one-size-fits-all.** Free-form text gets fuzzy; numeric IDs get strict-after-normalization. A single-digit difference in `DocumentNumber` is a real error and must not be hidden by fuzzy logic.
- **Both-non-null mismatch counts twice.** A wrong prediction over a non-null GT is `FP_FN` — penalized in both precision and recall denominators (and 2× in F1 denom). One wrong prediction blocked the right one.
- **Table-first.** `deeds_extracted_vs_gt` is a deliverable for downstream consumers. Match logic lives in PySpark / SQL; MLflow scorers carry the same data with informative per-row rationale.

## Methods by cell

### Cell 4 — Setup
- `GROUND_TRUTH_PATH` points at the Volume location (CSV).
- Same `EXTRACT_FIELDS` as notebooks 01 / 02. Different MLflow experiment.

### Cell 6 — Load + transpose CSV → `deeds_ground_truth`
- `pd.read_csv` with `header=None` so we can address by row/column index.
- **`_normalize_image_name`** — canonical join key: lowercase, replace runs of `[\s:_-]+` with `_`. Resolves CSV/file mismatches like `ARPULA:2023 00023192` → `arpula_2023_00023192` matching `ARPULA-2023_00023192.TIF`.
- **`_stringify`** — preserves leading zeros (e.g. `01161191`), treats blanks/NaN as `None`. Strips `.0` from pandas-read floats that were really integers in the source.
- Blanks land as **null** in Delta, never empty strings.

### Cell 8 — Join GT to extraction → `joined`
- Both sides get the same normalized `join_key`.
- The extraction side carries `pred_<field>`, `pred_<field>_conf`, `pred_<field>_citation_ids`, and the doc-level `citations` (cast to JSON string for portability through `toPandas`).
- Inner join on `join_key`. Smoke check: row count == number of GT documents.

### Cell 10 — Per-field comparators + classification → `classified`
Four normalizers:

| Function | Used for | Rule |
|---|---|---|
| `norm_title` | `DocumentTitle` | lowercase, strip non-alphanumeric, collapse whitespace |
| `norm_type` | `BookNumberType` | trim + uppercase |
| `norm_digits` | `BookNumberParsed`, `PageNumber`, `DocumentNumber` | digits-only, leading zeros stripped (special-case "all zeros" → `0`) |
| `norm_date` | `RecordingDate` | digits-only |

Per-field columns added:
- `<field>_exact_match` — strict equality after normalization
- `<field>_fuzzy_match` — same as exact for everything except `DocumentTitle`, where it uses `1 - levenshtein(a,b) / max(|a|,|b|) >= 0.85`. The raw similarity is also stored as `DocumentTitle_levenshtein_similarity`.
- `<field>_classification` — categorical TP / TN / FP / FN / FP_FN.

### Cell 12 — Build `deeds_extracted_vs_gt` (long format)
- Pivots from one-row-per-doc to one-row-per-(doc, field).
- **`_resolve_citations`** filters the doc's `citations` array to only the entries whose `id` is in this cell's `citation_ids`, returning resolved bbox + page entries.
- Each row carries: `gt_value`, `pred_value`, match flags, `levenshtein_similarity` (DocumentTitle only), `classification`, `extract_conf`, `citation_ids`, `cited_regions` (JSON-stringified bboxes), `high_conf_wrong`.
- **`high_conf_wrong = (classification ∈ {FP, FN, FP_FN}) AND (extract_conf ≥ 0.8)`** — the headline error signal. The model was confident AND wrong; these are the cells that hurt in production.
- Display ordered by `high_conf_wrong DESC` then errors-before-correct then `extract_conf DESC`.

### Cell 14 — Per-field corpus metrics → `deeds_eval_metrics_gt`
SQL aggregation over the long-format table. Per field:
```
accuracy  = (tp + tn) / n
precision = tp / (tp + fp + fp_fn)
recall    = tp / (tp + fn + fp_fn)
f1        = 2*tp / (2*tp + fp + fn + 2*fp_fn)
```
`fp_fn` appears in both precision and recall denominators because a both-non-null mismatch is simultaneously a hallucination and a miss.

### Cell 16 — MLflow `genai.evaluate`
- **`_build_expected_facts`** — for each field, generates `"<field> is <value>"` if GT has a value, OR `"<field> is not present in the document"` if GT is null. The negative statement is what enables `Correctness` to evaluate legitimate absences.
- **`_make_field_scorer`** — factory producing per-field pass-through scorers. The match logic was already computed in cell 10; the scorer surfaces the classification with a rich rationale string (`"classification=TP; comparator=levenshtein_sim>=0.85 on lowercase + alnum + collapse_ws; gt='THIS DEED', pred='THIS DEED', levenshtein_sim=1.000"`). MLflow per-row view becomes informative without recomputing.
- **`COMPARATOR_DESCRIPTION`** — a small dict labeling each field's comparator and normalization, used in the rationale strings. Single source of truth for the human-readable description.
- Scorers passed to `evaluate()`: built-in `Correctness` + 6 per-field scorers.

### Cell 18 — Log corpus metrics to the same MLflow run
Reuses the run created by `evaluate` (`run_id=results.run_id`) so per-field corpus accuracy / precision / recall / F1 land alongside the scorer-level metrics. F1 isn't a row-wise mean and can't be derived from the scorer view — must be logged from the SQL aggregation.

### Cell 20 — Calibration check
Bins `extract_conf` into `[0, 0.5)`, `[0.5, 0.8)`, `[0.8, 0.95)`, `[0.95, 1.0]`. Per bin: count of TP+TN vs FP+FN+FP_FN, accuracy. Excludes null-confidence cells.

Three outputs:
1. `deeds_calibration_gt` Delta table (per-bin counts and accuracy).
2. MLflow metrics `calibration_<bin>_n` and `calibration_<bin>_accuracy` on the same run; empty bins skipped.
3. Printed monotonicity verdict — if accuracy increases as confidence increases, the score is meaningful; if flat or non-monotonic, `extract_conf` is unreliable and any production review threshold based on it must be set conservatively.

With ≤36 cells, treat the verdict as a sanity signal not a calibrated curve. Watch the trend across batches.

## Reviewer workflow

1. Open `deeds_eval_metrics_gt`. Identify which fields are best/worst by F1.
2. Open `deeds_extracted_vs_gt`, sort `high_conf_wrong DESC` then `extract_conf DESC`. Top rows are the most damaging errors.
3. For each row, use `cited_regions` JSON to jump to the cited bbox / page in the source document. Compare to `gt_value` and `pred_value`.
4. Open `deeds_calibration_gt`. Check the monotonicity verdict from cell 20's output. If False, treat `extract_conf` as advisory only.
5. The MLflow run UI shows scorer metrics, per-field correctness, and calibration metrics in one view — useful for cross-version comparisons when schema or instructions change.

## Knobs

| Knob | Default | When to change |
|---|---|---|
| Levenshtein threshold for `DocumentTitle` | 0.85 | Tighten (0.9) if false-fuzzy-matches appear; loosen (0.75) if OCR drift is heavy |
| `HIGH_CONF_THRESHOLD` | 0.8 | Raise once the calibration check confirms scores are meaningful at higher bins |
| `CALIBRATION_BINS` | 4 bins | Add bins as the corpus grows (more data = finer resolution) |
| `JUDGE_MODEL` for `Correctness` | `databricks-claude-sonnet-4-6` | Switch up if the built-in correctness scorer underperforms |

## Failure modes to watch

- **Inner join produces fewer rows than expected** → `_normalize_image_name` doesn't cover a new naming pattern. Print mismatched keys, extend the regex.
- **All `FP_FN` and no `TP`** → comparator normalization is too strict; check `norm_*` outputs on a few examples.
- **Calibration verdict says non-monotonic** → confidence is unreliable on this corpus; raise `HIGH_CONF_THRESHOLD` and the production review cutoff in notebook 02.
- **`Correctness` scorer disagrees with per-field scorers** → judge model is reading the negative `expected_facts` differently than the strict comparator. Investigate per-row; usually the comparator is correct and the judge is being lenient.

## Related docs

- `docs/notebook-02-methods.md` — sibling no-GT eval methodology.
- `docs/plan-cross-confidence-check.md` — future work cross-checking `ai_extract` confidence vs `ai_parse_document` element confidence.
