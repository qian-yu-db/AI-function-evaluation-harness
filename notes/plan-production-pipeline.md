# Plan — Production Pipeline (Daily Cadence)

**Status:** Future work. Not yet implemented.
**Goal:** Daily ingest of new DED documents → parse → extract → review queue. **No LLM judge in the per-document path** — costs and latency disqualify it for production. Human review remains the safety net; logic-based scorers prioritize the queue.

## Architecture

Two cooperating jobs, coupled by Delta tables:

```
                    Volume: /Volumes/.../deeds/  (new files arrive here)
                                  │
                  ┌───────────────┴───────────────────────┐
                  │ Job 1 — Streaming pipeline (daily)    │
                  │ Trigger.AvailableNow                  │
                  └───────────────┬───────────────────────┘
                                  │
            Stage 1: parse                    checkpoint A
                                  │
                                  ▼
                              deeds_parsed
                              (raw VARIANT per file)
                                  │
            Stage 2: extract (v2.1)           checkpoint B
                                  │
                                  ▼
                       deeds_extracted_flat
                       (values, extract_conf, citation_ids,
                        per-page conf stats, citations)
                                  │
                                  ▼
                       deeds_parsed_elements
                       deeds_parsed_pages
                       (element-level + per-page stats)

                                  │
                  ┌───────────────┴───────────────────────┐
                  │ Job 2 — Batch analytics (daily)       │
                  │ Depends on Job 1 success              │
                  └───────────────┬───────────────────────┘
                                  │
                  ┌───────────────┼───────────────────────┐
                  ▼               ▼                       ▼
          Corpus outliers   Field-level low-conf     deeds_review_flags
          (vs all-time      flags                    (per-doc)
           deeds_extracted_                            deeds_field_review
           flat)                                     (per-cell queue)
                                  │
                                  ▼
                       Reviewer UI
                       joins deeds_review_flags × deeds_review_audit
                       (audit table is owned separately by the UI)
```

## Why no LLM judge

`make_judge` is well-suited to **offline evaluation against a small batch** (notebook 02 today), but cost and latency rule it out for the per-document production path:

- **Cost.** Roughly one LLM call per document per run. At 10k docs/day on a Sonnet-class judge, that's a meaningful spend with no proportional accuracy gain.
- **Latency.** A streaming stage that calls an LLM serially per row is much slower than one that runs only Spark + `ai_parse_document` + `ai_extract`. The latter two are already model-backed but optimized for batch.
- **Marginal value.** The judge's main contribution is distinguishing "legitimately absent" from "model missed it". Logic-based signals (citation presence, shape rules, confidence thresholds) cover the same ground deterministically — see "Replacing the judge" below.
- **Reproducibility.** Logic-based scorers are deterministic; the judge is not. Production teams want repeatable triage queues.

The judge stays in **notebook 02** for offline batch evaluation when you want a second-opinion read on a sample. It does not run on every doc every day.

## Replacing the judge with logic-based scorers

The judge produced one categorical signal: `looks_correct | partial | looks_incorrect`. Decompose into four deterministic signals already available in the data:

| Replacement signal | Rule | What the judge would have said |
|---|---|---|
| `field_low_extract_conf` | `extract_conf < LOW_EXTRACT_CONF_THRESHOLD` (e.g. 0.7) on a non-null field | "model is uncertain about this value" |
| `field_no_citation` | non-null value AND `citation_ids` empty | "model didn't cite any source for this value" — the no-grounding case |
| `field_shape_violation` | regex / digit rule fails on a non-null value | "this value is malformed" |
| `field_format_specific_violations` | additional rules: `RecordingDate` not 8 digits; `BookNumberType` not in known prefix list; etc. | catches obvious wrong values without semantic reasoning |

`field_no_citation` is the most important new signal — it catches "model fabricated a value" cases that the original (judge-driven) prompt was carrying. With it, you keep the absence-aware semantics deterministically: a non-null value with no `citation_ids` is suspicious.

What you lose without the judge:
- **Semantic plausibility.** The judge could read the parsed text and notice "this DocumentTitle says CORRECTION QUITCLAIM DEED but the document is clearly a Warranty Deed." Logic can't catch that.
- **Soft OCR-drift tolerance.** The judge marked minor character drift as `partial` rather than `looks_incorrect`. Logic just sees a string mismatch.

Both are fine for a queue-prioritization role — humans are the actual safety net. The reviewer reads the cell and the cited region; they catch the semantic errors. The pipeline's job is to surface the suspect cells in the right order.

## Job 1 — streaming pipeline

### Stage 1: parse

```python
files_df = (
    spark.readStream.format("binaryFile")
    .option("pathGlobFilter", "*.{pdf,PDF,tif,TIF,tiff,TIFF,jpg,jpeg,png}")
    .option("recursiveFileLookup", "true")
    .load(VOLUME_PATH)
)

parsed_df = (
    files_df
    .repartition(8, expr("crc32(path) % 8"))
    .withColumn("parsed", expr(
        "ai_parse_document(content, map('version','2.0','descriptionElementTypes','*'))"
    ))
    .withColumn("image_name", regexp_extract("path", r"([^/]+)\.[^/.]+$", 1))
)

(
    parsed_df.writeStream.format("delta")
    .outputMode("append")
    .option("checkpointLocation", f"{CHECKPOINT_ROOT}/01_parse")
    .option("mergeSchema", "true")
    .trigger(availableNow=True)
    .toTable(f"{CATALOG}.{SCHEMA}.deeds_parsed")
)
```

Key choices:
- **`Trigger.AvailableNow`** — process all pending files then stop. Cheaper than continuous streaming.
- **`repartition` by crc32(path)** — parallelizes `ai_parse_document` across executors.
- **`mergeSchema = true`** — Databricks runtime upgrades may add fields to the parsed VARIANT; non-breaking.
- **Checkpoint** ensures exactly-once: re-runs skip already-parsed files.

### Stage 2: extract + per-page stats

Reads `deeds_parsed` as a stream, calls `ai_extract` v2.1, derives per-doc / per-page confidence stats inline (not as a separate batch step), writes to `deeds_extracted_flat`. `deeds_parsed_elements` and `deeds_parsed_pages` are written from the same stream as side-tables.

```python
extract_options = (
    "map('version', '2.1', "
    "'enableConfidenceScores', 'true', "
    "'enableCitations', 'true', "
    "'instructions', 'These are scanned U.S. real-estate deed documents...')"
)

extracted_df = (
    spark.readStream.format("delta").table("deeds_parsed")
    .withColumn(
        "extracted",
        expr(f"ai_extract(parsed, '{EXTRACT_SCHEMA}', {extract_options})"),
    )
    # ... select <field>:value, <field>:confidence_score, <field>:citation_ids per field
    # ... union with per-doc parse confidence aggregates from deeds_parsed_elements
)
```

Key choices:
- **Pin `version='2.1'` explicitly** — protects against silent runtime-upgrade behavior changes.
- **No LLM judge call in this stage.** All scoring deferred to Job 2.
- **Independent checkpoint from Stage 1** — if extract fails (model endpoint down, schema change), only Stage 2 re-runs. `ai_parse_document` is the most expensive call; never re-run it because of a downstream failure.
- **Element flatten + per-page stats are doc-local**; safe in streaming. Corpus-relative aggregates are NOT computed here — they belong in Job 2.

## Job 2 — batch analytics (no judge)

Runs after Job 1's streaming pipeline completes. Reads the persisted Delta tables, computes corpus-relative outliers, writes the review queue tables.

Replicates notebook 02's logic but:
- **Drops the LLM judge entirely.** No `make_judge`, no per-doc judge prompt, no `extraction_plausibility` column.
- **Adds `field_no_citation` and `field_format_specific_violations`** as new logic-based signals.
- Uses the same corpus-relative outlier framework (`outlier_low_mean`, `outlier_high_variance`, `outlier_bad_page`, `has_parse_error`).

### Updated `review_priority` formula (no judge)

```
review_priority =
  outlier_low_mean              (0–1)
+ outlier_high_variance         (0–1)
+ outlier_bad_page              (0–1)
+ has_parse_error               (0–1)
+ !extraction_shape_ok          (0–1)
+ low_conf_field_count          (0 to 6)   ← from extract_conf < threshold
+ no_citation_field_count       (0 to 6)   ← non-null value, empty citation_ids
+ format_violation_count        (0 to 6)   ← field-specific format rules
```

Range 0–24. Field-level signals (last three terms) dominate, which is correct — the queue prioritizes specific cells, not whole documents.

### Updated `deeds_field_review` inclusion rule

A `(doc, field)` row is included if **any** of:
- `<field>_low_extract_conf` (non-null value below conf threshold)
- `<field>_no_citation` (non-null value with empty `citation_ids`)
- `<field>_shape_violation` (regex / digit rule failed on non-null value)

Each row's `reasons` array surfaces which signals fired (e.g. `["low_extract_conf<0.7", "no_citation"]`).

## Corpus baseline policy

Compute corpus-relative outliers against **all-time `deeds_extracted_flat`**. Simple, deterministic, captures drift slowly. Move to a persisted baseline only when batch composition changes meaningfully (new county, new document type) and old outlier definitions become noisy.

## Review-state separation

`deeds_review_flags` and `deeds_field_review` are owned by the analytics pipeline and **overwritten or merged** each run. Reviewer triage state (approved / needs follow-up / dismissed / verified) lives in a separate table `deeds_review_audit`, owned by the reviewer UI:

```
deeds_review_audit (
  image_name STRING,
  field STRING,                    -- nullable; doc-level rows have NULL field
  status STRING,                   -- 'approved' | 'needs_follow_up' | 'dismissed'
  reviewed_by STRING,
  reviewed_at TIMESTAMP,
  notes STRING
)
```

The UI shows analytics × audit join. Re-running analytics never clobbers a reviewer's work.

## Schedule

| Job | Cadence | Compute | Cost driver |
|---|---|---|---|
| Streaming pipeline | Daily 02:00 UTC | Photon job cluster (8 workers, autoscale) | `ai_parse_document` + `ai_extract` per new file |
| Batch analytics | Daily 03:00 UTC, depends on Job 1 success | SQL warehouse Pro (small) | Pure SQL aggregations — cheap |
| GT eval (notebook 03) | On-demand or weekly | SQL warehouse | Built-in `Correctness` (judge-based) — only when ground truth refreshed |

## Verification plan

For the first production batch:
1. **Stage 1 idempotency.** Re-run streaming twice in a row. Second run should write zero new rows (checkpoint working).
2. **Stage 2 isolation.** Force a Stage 2 failure (intentional bad schema), re-run. Stage 1 must NOT re-execute (checkpoints separated).
3. **Job 2 freshness.** Confirm corpus outliers update when a new doc lands (re-run Job 2 after streaming).
4. **No-judge equivalence check.** For 50–100 docs that were also evaluated in notebook 02 with the judge, compare `review_priority` rankings with-judge vs without-judge. Spearman correlation > 0.7 means logic-based signals capture the same prioritization. If correlation is weak, revisit the `field_no_citation` and shape-rule coverage.
5. **Audit table separation.** Manually mark a doc `approved` in `deeds_review_audit`. Re-run Job 2. Confirm the audit row survived.

## Open questions

1. **Should `field_no_citation` fire if `extract_conf` is also high?** Possibly not — a high-confidence value the model claims to know with no citation could be a legitimate inference (e.g., the field was on a page that wasn't in the cited bbox set). Consider gating: `field_no_citation = no_citation AND extract_conf < 0.95`. Watch behavior on the first batch and tune.
2. **How aggressive is "format_specific_violations"?** Start narrow: `RecordingDate` length-8 check is the safe one. `BookNumberType` against a known prefix list (`R`, `B`, `BK`, etc.) is risky if a county uses a new prefix.
3. **Cross-confidence flag** (see `plan-cross-confidence-check.md`) is a much stronger signal than any of these and is also logic-based. Consider including it from day 1 of production rather than as a future iteration — particularly the `calibration_mismatch` flag (high `extract_conf` over a low-confidence parsed region).

## Related docs

- `docs/notebook-02-methods.md` — current no-GT eval (judge included; the judge piece is what gets dropped in production).
- `docs/notebook-03-methods.md` — GT eval (offline, runs on demand; not part of daily production).
- `docs/plan-cross-confidence-check.md` — future cross-confidence work; the strongest deterministic signal of all and a natural addition to the no-judge production pipeline.
