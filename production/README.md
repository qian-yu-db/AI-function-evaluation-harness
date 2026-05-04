# deeds_pipeline — Production DAB

Daily production pipeline for the deed-document extraction workflow. Packages two
jobs as a Databricks Asset Bundle:

- **`deeds_streaming`** — runs daily at 02:00 UTC. Two streaming tasks
  (`Trigger.AvailableNow`):
  - `parse` reads new files from the `deeds` volume, calls `ai_parse_document`,
    writes raw VARIANT rows to `deeds_parsed`.
  - `extract` reads `deeds_parsed` as a stream, calls `ai_extract` v2.1 with
    confidence + citation options, computes per-doc parse-confidence stats, and
    writes `deeds_extracted_flat`.
  Each task has its own checkpoint so failure in `extract` does not re-trigger
  the (expensive) parse stage.

- **`deeds_analytics`** — runs daily at 03:00 UTC. Single batch task that reads
  `deeds_extracted_flat`, computes corpus-relative outlier flags + field-level
  low-conf / no-citation / format-violation flags, and writes
  `deeds_review_flags` and `deeds_field_review`. **No LLM judge** — all signals
  are deterministic.

See `docs/plan-production-pipeline.md` (repo root `docs/`) for the full design
rationale, including why the LLM judge was dropped and what the field-level
deterministic signals replace it with.

## Targets

| Target | Catalog | Schema | Cron status |
|---|---|---|---|
| `dev` (default) | `dev_fins_genai` | `unstructured_documents` | `PAUSED` |
| `prod` | `fins_genai` | `unstructured_documents` | `UNPAUSED` |

Both targets use the `DEFAULT` Databricks CLI profile.

## Prerequisites

1. The input volume `/Volumes/{catalog}/unstructured_documents/deeds/` must
   already exist with documents in it. The bundle does NOT create it
   (intentional — production-managed input volume).
2. The catalog (`dev_fins_genai` for dev, `fins_genai` for prod) must exist.
3. The MLflow experiment path must be writable by the deploying user / SP.
4. Workspace must support serverless jobs compute (the bundle does not
   declare a job cluster).

## Common commands

```bash
# Validate
databricks bundle validate -t dev
databricks bundle validate -t prod

# Deploy
databricks bundle deploy -t dev
databricks bundle deploy -t prod

# Run manually (e.g. for first-time validation in dev)
databricks bundle run deeds_streaming -t dev
databricks bundle run deeds_analytics -t dev

# Tear down
databricks bundle destroy -t dev
```

## File layout

```
production/
├── databricks.yml                   # bundle root
├── resources/
│   ├── volumes.yml                  # deeds_checkpoints
│   └── jobs.yml                     # deeds_streaming + deeds_analytics
└── src/
    ├── shared/
    │   ├── config.py                # EXTRACT_FIELDS canonical list
    │   └── extract_schema.py        # ai_extract schema + instructions
    ├── pipelines/
    │   ├── 01_stream_parse.py
    │   └── 02_stream_extract.py
    └── analytics/
        └── 01_batch_analytics.py
```

## Tunable variables

All configurable via `databricks.yml` `variables` block (or `--var` on the CLI):

| Variable | Default | Purpose |
|---|---|---|
| `low_extract_conf_threshold` | 0.7 | Field-level conf below this triggers a review flag |
| `high_conf_threshold` | 0.8 | High-confidence-wrong threshold for analytics surfacing |
| `no_citation_gate_conf` | 0.95 | Below this, a non-null field with empty citations is flagged |

## Related docs

- `../docs/plan-production-pipeline.md` — design rationale.
- `../docs/notebook-02-methods.md` — sibling no-GT eval method (the bundle drops the judge piece).
- `../docs/plan-cross-confidence-check.md` — natural future extension; flag slots into the analytics notebook without restructuring.
