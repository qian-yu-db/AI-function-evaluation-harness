# deeds_pipeline · docs

Rendered HTML walkthroughs of the four notebooks plus the overall confidence/evaluation framework. Each page contains restrained prose, inline SVG diagrams, and per-section tables — no fancy CSS, no external assets, no JavaScript.

**Live site:** <https://qian-yu-db.github.io/AI-function-evaluation-harness/>

## Pages

| Doc | What it covers |
|---|---|
| [Confidence and evaluation](https://qian-yu-db.github.io/AI-function-evaluation-harness/confidence-and-evaluation.html) | Top-level overview. How parse-side and extract-side confidence relate, how they fuse at the cited bbox, the composite `review_priority` formula, and how every signal becomes an MLflow scorer. **Start here.** |
| [Notebook 01 — parse and extract](https://qian-yu-db.github.io/AI-function-evaluation-harness/notebook-01-workflow.html) | The batch pipeline that produces the four base tables (`deeds_parsed`, `deeds_parsed_elements`, `deeds_parsed_pages`, `deeds_extracted_flat`). Includes the three-signal `conf_tier` rule, the six-field extraction schema, and the per-element / per-field distribution plots. |
| [Notebook 02 (xconf) — cross-confidence workflow](https://qian-yu-db.github.io/AI-function-evaluation-harness/notebook-02-cross-conf-workflow.html) | The sibling of `02_eval_no_ground_truth` that adds the bbox-overlap layer. Six-step vectorized join from citation_ids to parse-side element confidence, IoU geometry asserts, and the three calibration-mismatch flags. |
| [Notebook 03 — evaluate with ground truth](https://qian-yu-db.github.io/AI-function-evaluation-harness/notebook-03-workflow.html) | Per-cell ground-truth evaluation. Comparators per field, the five-outcome classification (TP / TN / FP / FN / FP_FN), per-field accuracy / precision / recall / F1, and the binned `extract_conf` calibration check. |

## Reading order

If you're new to this repo, read in this order:

1. **Confidence and evaluation** — the framework. Why parse-side and extract-side confidence are different objects and how the pipeline reasons about both.
2. **Notebook 01** — how the base tables are produced. Everything downstream reads `deeds_extracted_flat`.
3. **Notebook 02 (xconf)** — the production-shape evaluation. The cross-confidence layer is the most important signal added on top of notebook 01.
4. **Notebook 03** — only when you have a labeled corpus. Calibration check tells you whether `extract_conf` is trustworthy enough to gate review.

## Related (source, in the repo)

- `notebooks/` — the actual `.py` notebooks that these docs describe.
- `production/` — the Databricks Asset Bundle that runs the same pipeline as a scheduled job (parse → extract → analytics).
