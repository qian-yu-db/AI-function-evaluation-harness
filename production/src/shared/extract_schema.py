"""ai_extract v2.1 schema and instructions for deed documents.

Lifted verbatim from `notebooks/01_parse_and_extract.py` cell 4. Kept in this
shared module so the production streaming notebook (Stage 2) and any future
maintenance notebooks reference the same source. If the dev notebook 01 is
updated, sync this module too.
"""

# Per-field descriptions intentionally specific to recorded U.S. real-estate
# deeds. Descriptions guide ai_extract; do not remove without re-evaluating.
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
