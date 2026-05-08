# Databricks notebook source
# MAGIC %md
# MAGIC # Visualize `ai_extract` Citations with bboxes
# MAGIC
# MAGIC Sibling of `02-visualize-bbox-ai-parse-document-outputs.py`. Where that notebook
# MAGIC overlays the bboxes of *parsed elements* (one box per `ai_parse_document` element),
# MAGIC this notebook overlays the bboxes of **`ai_extract` citations** — one box per
# MAGIC `citation_id` that the extractor claimed grounds an extracted value.
# MAGIC
# MAGIC Boxes are colored by the **field** that cites them (DocumentTitle, BookNumberType,
# MAGIC BookNumberParsed, PageNumber, DocumentNumber, RecordingDate). Hovering a box shows
# MAGIC the extracted value, the per-field `extract_conf`, and the citation id.
# MAGIC
# MAGIC **Source:** `<catalog>.<schema>.deeds_extracted_flat` (or its `_xconf` / `_dab` sibling).
# MAGIC The `citations` and `citation_pages` VARIANT columns plus the six
# MAGIC `<field>_citation_ids` arrays are everything we need.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup

# COMMAND ----------

dbutils.widgets.text("catalog", "fins_genai")
dbutils.widgets.text("schema", "unstructured_documents")
dbutils.widgets.text("table_name", "deeds_extracted_flat")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
TABLE_NAME = dbutils.widgets.get("table_name")
EXTRACTED_FLAT_TABLE = f"{CATALOG}.{SCHEMA}.{TABLE_NAME}"

EXTRACT_FIELDS = [
    "DocumentTitle",
    "BookNumberType",
    "BookNumberParsed",
    "PageNumber",
    "DocumentNumber",
    "RecordingDate",
]

# Same palette as notebook 01's per-field plot, so the two views read consistently.
FIELD_COLORS = {
    "DocumentTitle":    "#d62728",
    "BookNumberType":   "#ff7f0e",
    "BookNumberParsed": "#2ca02c",
    "PageNumber":       "#9467bd",
    "DocumentNumber":   "#8c564b",
    "RecordingDate":    "#17becf",
}
DEFAULT_COLOR = "#7f7f7f"  # citation cited by no field (shouldn't happen but defensive)

print(f"Reading from: {EXTRACTED_FLAT_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Pull-down menu — select a document

# COMMAND ----------

from pyspark.sql import functions as F

extracted_df = spark.table(EXTRACTED_FLAT_TABLE)

# Use image_name as the join key (every downstream column keys off it).
image_names = sorted(
    [row["image_name"] for row in extracted_df.select("image_name").distinct().collect()]
)

dbutils.widgets.dropdown(
    name="image_name",
    defaultValue=image_names[0] if image_names else "",
    choices=image_names if image_names else [""],
    label="Select image_name",
)

selected_image = dbutils.widgets.get("image_name")
print(f"Selected: {selected_image}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Pull citations + per-field info for the selected doc

# COMMAND ----------

import json

# Cast both VARIANTs to string so we can json.loads them on the driver — same pattern as
# notebooks/02 § 8 and notebooks/03 § 5. Avoids depending on Variant-accessor syntax that
# differs across DBR versions.
selected_row = (
    extracted_df.filter(F.col("image_name") == selected_image)
    .selectExpr(
        "image_name",
        *[c for f in EXTRACT_FIELDS for c in (f, f"{f}_extract_conf", f"{f}_citation_ids")],
        "cast(citations AS string) AS citations_json",
        "cast(citation_pages AS string) AS citation_pages_json",
    )
    .first()
)

if selected_row is None:
    raise SystemExit(f"No row in {EXTRACTED_FLAT_TABLE} for image_name={selected_image!r}")


def _safe_json_loads(s):
    if s is None or s == "":
        return None
    try:
        return json.loads(s)
    except (TypeError, ValueError) as exc:
        print(f"WARN: could not parse VARIANT JSON: {exc}")
        return None


citations = _safe_json_loads(selected_row["citations_json"]) or []
citation_pages = _safe_json_loads(selected_row["citation_pages_json"]) or []

# Build the field → citation_id reverse map: which fields cite which id?
def _to_int_list(v):
    if v is None:
        return []
    return [int(x) for x in v]


field_info = {}
for f in EXTRACT_FIELDS:
    field_info[f] = {
        "value": selected_row[f],
        "extract_conf": (
            float(selected_row[f"{f}_extract_conf"])
            if selected_row[f"{f}_extract_conf"] is not None
            else None
        ),
        "citation_ids": _to_int_list(selected_row[f"{f}_citation_ids"]),
    }

# Reverse index: citation_id → list[field]
citation_to_fields = {}
for f, info in field_info.items():
    for cid in info["citation_ids"]:
        citation_to_fields.setdefault(int(cid), []).append(f)

print(f"Document: {selected_image}")
print(f"  pages:               {len(citation_pages)}")
print(f"  citations (total):   {len(citations)}")
print(f"  fields with non-empty citation_ids: "
      f"{[f for f, info in field_info.items() if info['citation_ids']]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Render pages with citation bboxes overlaid
# MAGIC
# MAGIC One panel per page. For each citation whose bbox has a matching `page_id`, draw a
# MAGIC labeled rectangle colored by the field that cited it. Hover a box to see the
# MAGIC extracted value, the per-field `extract_conf`, and the citation id.

# COMMAND ----------

import base64
import os
from typing import Dict, List, Optional, Tuple

from IPython.display import HTML, display
from PIL import Image


class CitationRenderer:
    """Renders ai_extract citation bboxes on top of rendered page images.

    Lighter than the parse-element renderer in
    `02-visualize-bbox-ai-parse-document-outputs.py` because:
    - Citation bboxes are sparse (≤ 6 fields × maybe a few cites each), so no
      tooltip width calculation, no element-list table.
    - Color is keyed by field, not element type.
    - Tooltip shows extracted value + extract_conf + citation id.
    """

    def __init__(self, field_colors: Dict[str, str], default_color: str = DEFAULT_COLOR):
        self.field_colors = field_colors
        self.default_color = default_color

    # ---------- image utilities ----------

    @staticmethod
    def _get_image_dimensions(image_path: str) -> Optional[Tuple[int, int]]:
        try:
            if os.path.exists(image_path):
                with Image.open(image_path) as img:
                    return img.size
            return None
        except Exception as e:
            print(f"WARN: dimensions for {image_path}: {e}")
            return None

    @staticmethod
    def _load_image_as_base64(image_path: str) -> Optional[str]:
        try:
            if not os.path.exists(image_path):
                return None
            with open(image_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode("utf-8")
            ext = os.path.splitext(image_path)[1].lower()
            mime = "image/png" if ext == ".png" else "image/jpeg"
            return f"data:{mime};base64,{img_b64}"
        except Exception as e:
            print(f"WARN: load {image_path}: {e}")
            return None

    @staticmethod
    def _escape(text: str) -> str:
        return (
            (text or "")
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#39;")
            .replace("\n", "<br>")
        )

    # ---------- color picking ----------

    def _color_for_citation(self, citation_id: int, citation_to_fields: Dict[int, List[str]]) -> Tuple[str, List[str]]:
        fields = citation_to_fields.get(int(citation_id), [])
        if not fields:
            return self.default_color, []
        # If a citation is shared across fields, color by the first one (rare).
        return self.field_colors.get(fields[0], self.default_color), fields

    # ---------- main render ----------

    def render(
        self,
        image_name: str,
        citations: List[Dict],
        citation_pages: List[Dict],
        field_info: Dict[str, Dict],
        citation_to_fields: Dict[int, List[str]],
    ) -> None:
        if not citation_pages:
            display(HTML("<p style='color:#b00;'>No citation_pages on this document. "
                        "Check that ai_parse_document was run with imageOutputPath set "
                        "before ai_extract.</p>"))
            return

        # Header with field summary
        display(HTML(self._render_header(image_name, field_info, citations, citation_pages)))

        # Build page_id → list[citation] mapping once
        per_page = {}  # page_id → list of (citation, bbox_entry)
        for cit in citations:
            cid = int(cit.get("id", -1))
            for box in cit.get("bbox", []) or []:
                pid = box.get("page_id")
                if pid is None:
                    continue
                per_page.setdefault(int(pid), []).append((cit, box))

        # One panel per page
        for page in sorted(citation_pages, key=lambda p: int(p.get("id", 0))):
            page_id = int(page.get("id", 0))
            page_html = self._render_page(
                page=page,
                page_id=page_id,
                page_citations=per_page.get(page_id, []),
                citation_to_fields=citation_to_fields,
                field_info=field_info,
            )
            display(HTML(page_html))

    def _render_header(
        self,
        image_name: str,
        field_info: Dict[str, Dict],
        citations: List[Dict],
        citation_pages: List[Dict],
    ) -> str:
        # Per-field rows: name swatch, value, extract_conf, citation_ids
        rows = []
        for f, info in field_info.items():
            color = self.field_colors.get(f, self.default_color)
            value = info["value"] if info["value"] not in (None, "") else "<i style='color:#999'>null</i>"
            conf = (
                f"{info['extract_conf']:.3f}"
                if info["extract_conf"] is not None
                else "<i style='color:#999'>—</i>"
            )
            cids = ", ".join(str(x) for x in info["citation_ids"]) if info["citation_ids"] else (
                "<i style='color:#999'>—</i>"
            )
            rows.append(f"""
                <tr>
                    <td style="padding:6px 10px; vertical-align:top;">
                        <span style="display:inline-block; width:10px; height:10px; background:{color};
                                     border:1px solid #333; margin-right:6px; vertical-align:middle;"></span>
                        <code style="font-size:13px;">{f}</code>
                    </td>
                    <td style="padding:6px 10px; font-family:'Segoe UI',sans-serif; font-size:13px;">
                        {self._escape(str(value)) if isinstance(value, str) else value}
                    </td>
                    <td style="padding:6px 10px; font-family:monospace; font-size:12px; color:#444;">{conf}</td>
                    <td style="padding:6px 10px; font-family:monospace; font-size:12px; color:#444;">{cids}</td>
                </tr>
            """)

        return f"""
        <div style="background:#f8f9fa; border:1px solid #dee2e6; border-radius:8px; padding:18px; margin:14px 0;">
            <h2 style="margin:0 0 10px 0; color:#1f2937; font-family:'Segoe UI',sans-serif;">
                {self._escape(image_name)}
            </h2>
            <div style="color:#4b5563; font-family:'Segoe UI',sans-serif; font-size:13px; margin-bottom:12px;">
                <strong>{len(citations)}</strong> citation{'s' if len(citations) != 1 else ''} ·
                <strong>{len(citation_pages)}</strong> page{'s' if len(citation_pages) != 1 else ''}
            </div>
            <table style="border-collapse:collapse; width:100%; font-family:'Segoe UI',sans-serif;">
                <thead>
                    <tr style="background:#e9ecef; font-size:11px; letter-spacing:.06em; text-transform:uppercase; color:#495057;">
                        <th style="padding:8px 10px; text-align:left; border-bottom:1px solid #adb5bd;">Field</th>
                        <th style="padding:8px 10px; text-align:left; border-bottom:1px solid #adb5bd;">Extracted value</th>
                        <th style="padding:8px 10px; text-align:left; border-bottom:1px solid #adb5bd;">extract_conf</th>
                        <th style="padding:8px 10px; text-align:left; border-bottom:1px solid #adb5bd;">citation_ids</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(rows)}
                </tbody>
            </table>
        </div>
        """

    def _render_page(
        self,
        page: Dict,
        page_id: int,
        page_citations: List[Tuple[Dict, Dict]],
        citation_to_fields: Dict[int, List[str]],
        field_info: Dict[str, Dict],
    ) -> str:
        image_uri = page.get("image_uri", "")
        if not image_uri:
            return f"""
            <div style="background:#fff3cd; border:1px solid #ffc107; padding:12px; border-radius:6px; margin:14px 0;">
                <strong>Page {page_id + 1}:</strong> no <code>image_uri</code> on this page.
                Re-run <code>ai_parse_document</code> with <code>imageOutputPath</code> set so
                <code>metadata.pages[].image_uri</code> is populated.
            </div>
            """

        img_b64 = self._load_image_as_base64(image_uri)
        if img_b64 is None:
            return f"""
            <div style="background:#f8d7da; border:1px solid #f5c6cb; padding:12px; border-radius:6px; color:#721c24; margin:14px 0;">
                <strong>Page {page_id + 1}:</strong> could not load image at
                <code>{self._escape(image_uri)}</code>.
            </div>
            """

        dims = self._get_image_dimensions(image_uri)
        if dims is None:
            orig_w, orig_h = 1024, 1024
        else:
            orig_w, orig_h = dims

        max_display_width = 1024
        scale = min(1.0, max_display_width / orig_w) if orig_w > 0 else 1.0
        disp_w = int(orig_w * scale)
        disp_h = int(orig_h * scale)

        # Build overlays
        overlays = []
        container_id = f"page_{page_id}_{id(self)}"
        for cit, box in page_citations:
            cid = int(cit.get("id", -1))
            coord = box.get("coord") or []
            if len(coord) < 4:
                continue
            x0, y0, x1, y1 = (float(c) for c in coord[:4])
            # Normalize axis order — ai_extract is supposed to emit [x0,y0,x1,y1]
            # but we sort defensively in case of flipped corners.
            x0, x1 = sorted((x0, x1))
            y0, y1 = sorted((y0, y1))
            sx0, sy0 = x0 * scale, y0 * scale
            sx1, sy1 = x1 * scale, y1 * scale
            w = sx1 - sx0
            h = sy1 - sy0
            if w <= 0 or h <= 0:
                continue

            color, fields = self._color_for_citation(cid, citation_to_fields)
            # Label: list of citing field names if any
            label_text = (",".join(fields) if fields else "—") + f" #{cid}"

            # Tooltip: per-field block for each field that cites this citation
            tooltip_blocks = []
            for f in (fields or []):
                info = field_info.get(f, {})
                value = info.get("value")
                conf = info.get("extract_conf")
                conf_text = f"{conf:.3f}" if conf is not None else "—"
                value_text = self._escape(str(value)) if value not in (None, "") else "<i>null</i>"
                tooltip_blocks.append(f"""
                    <div style="margin:6px 0; padding-bottom:6px; border-bottom:1px solid #eee;">
                        <div style="font-family:monospace; font-size:11px; color:{self.field_colors.get(f, self.default_color)};
                                    font-weight:700; letter-spacing:.04em; text-transform:uppercase; margin-bottom:2px;">
                            {f}
                        </div>
                        <div style="font-family:'Segoe UI',sans-serif; font-size:12px; color:#222;">{value_text}</div>
                        <div style="font-family:monospace; font-size:11px; color:#666;">extract_conf = {conf_text}</div>
                    </div>
                """)
            if not tooltip_blocks:
                tooltip_blocks.append("""
                    <div style="font-family:'Segoe UI',sans-serif; font-size:12px; color:#666;">
                        Citation present but no field references it. Check the citation_ids
                        arrays — this is usually a no-op artifact.
                    </div>
                """)

            label_top_offset = -16 if sy0 >= 16 else 2
            overlays.append(f"""
                <div class="cit-overlay cit-{container_id}"
                     style="position:absolute; left:{sx0:.1f}px; top:{sy0:.1f}px;
                            width:{w:.1f}px; height:{h:.1f}px;
                            border:2px solid {color}; background:{color}25;
                            box-sizing:border-box; cursor:pointer;
                            transition:all .15s ease;">
                    <div style="background:{color}; color:white; padding:1px 4px;
                                font-family:monospace; font-size:9px; font-weight:700;
                                position:absolute; top:{label_top_offset}px; left:0;
                                white-space:nowrap; border-radius:2px;
                                box-shadow:0 1px 2px rgba(0,0,0,.3);
                                pointer-events:none;
                                max-width:{max(50, w-4):.0f}px;
                                overflow:hidden; z-index:1000;">
                        {self._escape(label_text)}
                    </div>
                    <div class="cit-tooltip"
                         style="position:absolute; left:8px; top:{h:.1f}px;
                                background:rgba(255,255,255,.98); color:#333;
                                border:2px solid #ccc; padding:10px;
                                border-radius:6px; font-size:12px;
                                width:380px; max-width:380px;
                                z-index:10000; pointer-events:none;
                                box-shadow:0 4px 12px rgba(0,0,0,.15);
                                display:none; line-height:1.4;
                                max-height:400px; overflow-y:auto;">
                        <div style="font-weight:bold; color:#0066cc; margin-bottom:6px;
                                    padding-bottom:6px; border-bottom:1px solid #ddd;
                                    font-family:'Segoe UI',sans-serif;">
                            Citation #{cid} · page {page_id + 1}
                        </div>
                        {''.join(tooltip_blocks)}
                    </div>
                </div>
            """)

        styles = f"""
        <style>
            .cit-{container_id}:hover {{
                background: rgba(255,255,0,.3) !important;
                border-width: 3px !important;
                z-index: 1001 !important;
            }}
            .cit-{container_id}:hover .cit-tooltip {{ display: block !important; }}
            .cit-{container_id} {{ z-index: 100; }}
            .cit-{container_id}:hover {{ z-index: 9999 !important; }}
        </style>
        """

        header = f"""
        <div style="background:#e3f2fd; border:1px solid #2196f3; border-radius:6px;
                    padding:10px 14px; margin:14px 0 8px 0;
                    font-family:'Segoe UI',sans-serif; color:#0d47a1; font-size:13px;">
            <strong>Page {page_id + 1}</strong> · {len(page_citations)} citation
            bbox{'es' if len(page_citations) != 1 else ''} ·
            original {orig_w}×{orig_h}px · scale {scale:.3f}
        </div>
        """

        return f"""
        {header}
        {styles}
        <div style="position:relative; display:inline-block; border:2px solid #333;
                    border-radius:8px; overflow:visible; background:#fff; margin-bottom:24px;">
            <img src="{img_b64}" style="display:block; width:{disp_w}px; height:{disp_h}px;"
                 alt="Page {page_id + 1}">
            {''.join(overlays)}
        </div>
        """


# COMMAND ----------

renderer = CitationRenderer(field_colors=FIELD_COLORS)
renderer.render(
    image_name=selected_image,
    citations=citations,
    citation_pages=citation_pages,
    field_info=field_info,
    citation_to_fields=citation_to_fields,
)

# COMMAND ----------
