"""Constants shared across the deeds_pipeline DAB notebooks.

The values mirror those in the dev notebooks under `notebooks/01..03`. Keeping
them in one module here avoids drift between the streaming pipeline and the
batch analytics notebook within the bundle.
"""

# Canonical extraction field list — must stay in sync with EXTRACT_SCHEMA below
# and with the dev notebooks. Order matters where it determines column ordering
# in derived tables.
EXTRACT_FIELDS = [
    "DocumentTitle",
    "BookNumberType",
    "BookNumberParsed",
    "PageNumber",
    "DocumentNumber",
    "RecordingDate",
]

# Field groupings used by the analytics notebook
DIGIT_FIELDS = ["BookNumberParsed", "PageNumber", "DocumentNumber"]
