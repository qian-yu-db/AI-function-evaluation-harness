# Plan — Cross-Confidence Check (`ai_extract` vs `ai_parse_document`)

**Status:** Implemented as a sibling notebook — `notebooks/02_eval_no_gt_with_cross_conf.py` (writes to `deeds_review_flags_xconf` / `deeds_field_review_xconf`). See `docs/notebook-02-cross-conf-methods.md` for the as-built methodology.
**Owner:** TBD
**Depends on:** Notebooks 01 (parse + extract), 02 (no-GT eval). Optional pairing with 03 (with-GT eval).

## Why

Two confidence scores live in this pipeline and they measure different things:

| Score | Source | What it captures |
|---|---|---|
| `confidence` | `ai_parse_document` per element | OCR / layout quality of a parsed element (per-element double in [0, 1]) |
| `extract_conf` | `ai_extract` v2.1 per field | Model's **reasoning** confidence over already-parsed text (per-field double in [0, 1]) |

Each score alone misses a critical failure mode that the *combination* catches:

> **The model is confident about OCR garbage.**
> `extract_conf` is high because the extractor reasoned cleanly over what it saw, but the source element was mis-OCR'd in the first place — so the value is wrong with high confidence. Neither single score flags this; together they do.

The complement is also informative — high source confidence with low extract confidence usually means the model genuinely couldn't decide between alternatives even though the text was clean. That's a different remediation (model upgrade or schema clarification), and the cross-check distinguishes them.

## What gets built

A new column `<field>_source_conf` on `deeds_extracted_flat` (and equivalently a per-cell column on `deeds_extracted_vs_gt` and `deeds_field_review`) — the aggregated confidence of the *parsed elements* whose bbox overlaps the extractor's cited region for that field. Plus three derived flags:

| Flag | Rule | Reading |
|---|---|---|
| `calibration_mismatch` | `extract_conf > 0.8 AND source_conf < 0.5` | Confident over OCR garbage — top-of-queue |
| `model_uncertain_clean_source` | `extract_conf < 0.6 AND source_conf > 0.85` | Source is clean but model hesitated — schema/instruction issue |
| `consistent_low_source` | `extract_conf < 0.6 AND source_conf < 0.6` | Both are low — known-bad cell, expected to be flagged anyway |

The new column slots into:
- `deeds_review_flags` — adds `n_calibration_mismatch_fields` to `review_priority`
- `deeds_field_review` (notebook 02 §8) — adds `source_conf` and `calibration_mismatch` per cell, sortable to the top
- `deeds_extracted_vs_gt` (notebook 03 §5) — same per-cell columns; in the GT setting, calibration mismatches that turn out to be FP are the highest-cost error class

## Algorithm

For each `(image_name, field)`:

1. **Resolve the field's cited regions.** Look up `<field>_citation_ids` (an `ARRAY<INT>`) in the doc's `metadata.citations` array (already pulled into `deeds_extracted_flat.citations` in notebook 01). Each entry has `id`, `bbox: [{coord: [x0, y0, x1, y1], page_id}]`. A field can cite multiple regions.

2. **Bbox-overlap-join to `deeds_parsed_elements`.** For each cited bbox `(coord, page_id)`, find parsed elements where:
   - `parsed_elements.page_id == cited.page_id`
   - bbox of the parsed element overlaps the cited bbox (IoU > 0.3 as a starting threshold; tune empirically — for OCR-mostly cases even any-intersection is acceptable since elements rarely overlap each other)

3. **Aggregate the source elements' confidence.** Take the *minimum* across overlapping elements (not the mean) — a single low-confidence element in the cited region is enough to taint the source. Fallback: if no parsed element overlaps the cited bbox, set `source_conf = NULL` and surface a separate flag `citation_unmatched`.

4. **Persist** as `<field>_source_conf` on `deeds_extracted_flat`. Compute the three derived flags downstream.

## Implementation sketch

The algorithm has two non-trivial pieces: **parsing the bbox VARIANT** and **bbox overlap math**. Both are doable in pure SQL, but a Python UDF over a small joined DataFrame is clearer for the first cut.

### Skeleton (pseudocode-ish PySpark)

```python
import json
from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType, BooleanType

def iou(a, b):
    """a, b = [x0, y0, x1, y1]. Returns IoU in [0, 1]."""
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0

# Per-doc, per-field source_conf — driver-side join (small data)
def compute_source_conf(extracted_flat_pdf, parsed_elements_pdf,
                        iou_threshold=0.3):
    # parsed_elements_pdf indexed by (image_name, page_id) for quick filtering
    out = []
    for _, row in extracted_flat_pdf.iterrows():
        citations = json.loads(row["citations"]) if row["citations"] else []
        cit_by_id = {int(c["id"]): c for c in citations if isinstance(c, dict)}
        for f in EXTRACT_FIELDS:
            ids = row[f"{f}_citation_ids"] or []
            if not ids:
                out.append((row["image_name"], f, None, None, False))
                continue
            confs = []
            for cid in ids:
                cit = cit_by_id.get(int(cid))
                if not cit or not cit.get("bbox"):
                    continue
                for box_entry in cit["bbox"]:
                    coord = box_entry["coord"]
                    page_id = box_entry["page_id"]
                    page_elems = parsed_elements_pdf.query(
                        "image_name == @row.image_name and page_id == @page_id"
                    )
                    for _, elem in page_elems.iterrows():
                        elem_box = parse_bbox(elem["bbox"])
                        if iou(coord, elem_box) >= iou_threshold:
                            if elem["confidence"] is not None:
                                confs.append(float(elem["confidence"]))
            source_conf = min(confs) if confs else None
            unmatched = (source_conf is None and ids)
            out.append((row["image_name"], f, source_conf, unmatched, False))
    return out  # turn into Spark DF, pivot back wide, join to extracted_flat
```

### Vectorized variant (preferred for production)

If the corpus grows beyond a few hundred docs, pull the whole join into a single Spark DataFrame:

1. Explode `<field>_citation_ids` per field — long format `(image_name, field, citation_id)`.
2. Join `metadata.citations` exploded — long format `(image_name, citation_id, page_id, coord)`.
3. Cross-join to `deeds_parsed_elements` filtered by `image_name` and `page_id`.
4. Compute IoU in a Python UDF or Spark expression.
5. Filter `iou >= threshold`, aggregate `min(confidence)` grouped by `(image_name, field)`.
6. Pivot back to one column per field.

The driver-side approach is fine for the 6-doc POC. Switch to the vectorized one when adding the second customer's batch.

## Edge cases to handle

| Case | Handling |
|---|---|
| Field has no citation_ids (extractor didn't cite) | `source_conf = NULL`, do not flag — this is already covered by the no-citations heuristic in notebook 02's judge prompt |
| Citation bbox doesn't overlap any parsed element | `source_conf = NULL`, set `citation_unmatched = TRUE` — surface as a separate signal; usually means the extractor hallucinated a region |
| Cited bbox spans multiple parsed elements | Take `min(confidence)` across all overlapping elements |
| Field cites multiple citation_ids | Aggregate `min(confidence)` across all the cited regions (one bad region is enough to taint) |
| Page ID mismatch | Citation has `page_id`, parsed element has `page_id` — must match; if either is null, skip and treat as unmatched |
| Bbox coordinate order (`[x0, y0, x1, y1]` vs `[x, y, w, h]`) | Verify which Databricks emits at runtime — assume `[x0, y0, x1, y1]` per the docs but defensively swap if `x1 < x0` after parsing |
| `ai_extract` v2.0 input (no citations) | Skip the whole cross-check; surface `source_conf = NULL` for all rows and a notebook-level warning |

## Verification plan

1. **Unit-test IoU function** in isolation — feed known boxes (perfect overlap, zero overlap, partial). Spot-check against hand-computed values.
2. **End-to-end on the 6-doc POC.** Confirm that the documents we already know are mostly clean (the PDFs) get `source_conf` close to 1.0 across fields, and that the TIFFs (which are scanned and OCR'd) show lower `source_conf` values especially on the smaller fonts (e.g. `BookNumberType`, `PageNumber`).
3. **Calibration mismatch sanity.** Manually craft one synthetic extraction with a deliberately wrong value over a low-confidence element — confirm it surfaces with `calibration_mismatch = TRUE`.
4. **GT-mode confirmation (optional).** In notebook 03, compare `calibration_mismatch` cells against `classification`. If the cross-check is meaningful, calibration mismatches should disproportionately be `FP` / `FN` / `FP_FN`, not `TP` / `TN`.

## When this becomes worth building

Build it when one of these is true:
- A `high_conf_wrong` cell appears in the GT eval that the field-level `extract_conf` flag missed (i.e., extract_conf was high but the value is wrong) — that's the smoking gun for cross-check value.
- The corpus grows past ~50 docs and field-level review queues get noisy — calibration_mismatch acts as the top-priority filter.
- The pipeline is going to production and "the model was confident in a wrong value" becomes a customer-impacting failure mode.

For the 6-doc POC, skip it. Field-level confidence + judge + heuristic shape is enough signal.

## Open questions

1. **IoU threshold.** 0.3 is a guess. Need to measure on a real batch — for many extractors the cited bbox is *much smaller* than the parsed element (extractor cites a phrase inside a paragraph), so any-intersection may be more appropriate. Decide empirically.
2. **`min` vs `mean` aggregation across overlapping elements.** `min` is correct for the calibration-mismatch use case (any low-conf source poisons the cell). For other use cases (e.g., reporting average source confidence) the mean is more interpretable. Default to `min`; expose `mean` as a secondary column if it's useful.
3. **Field-level vs cell-level.** The doc here treats it as field-level (one `source_conf` per `(doc, field)`). For nested array extractions (not relevant to deeds but relevant to invoices later), it should be element-level. Future-proof the schema by adding the column on the long-format tables (`deeds_field_review`, `deeds_extracted_vs_gt`) first.
