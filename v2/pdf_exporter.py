"""pdf_exporter.py - one-click PDF inspection report (RODIC V2, reportlab).

Input is a list of finding dicts:

    {
      "id": 1, "frame_index": 120, "t_sec": 4.8, "box": (x1, y1, x2, y2),
      "image": np.ndarray BGR | None,       # crop or annotated frame
      "defect_class": "Crack" | None, "confidence": 0.81 | None,
      "crack_width_mm": 0.6 | None, "width_resolvable": True,
      "source": "stub_not_evaluated" | "...",
      "recommendation": Recommendation.to_dict()   # from irc_recommender
    }

A disclaimer banner is printed on page 1 of every report. Reports for large runs
list every finding in a table but only show images for the ``max_detail``
highest-severity findings, so a 10,000-image batch still produces a usable PDF.
"""

from __future__ import annotations

import datetime as _dt
import io
import os
from typing import Iterable, Optional
from xml.sax.saxutils import escape

import cv2
import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image as RLImage,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from irc_recommender import DISCLAIMER, SEVERITY_ORDER

SEVERITY_COLOURS = {
    "none": colors.HexColor("#6b7280"),
    "cosmetic": colors.HexColor("#2f855a"),
    "watch": colors.HexColor("#b7791f"),
    "significant": colors.HexColor("#c05621"),
    "urgent": colors.HexColor("#c53030"),
}


def _sev_rank(finding: dict) -> int:
    sev = (finding.get("recommendation") or {}).get("severity", "none")
    return SEVERITY_ORDER.index(sev) if sev in SEVERITY_ORDER else 0


def _fmt(value, spec: str = "{}", missing: str = "n/a") -> str:
    return missing if value is None else spec.format(value)


def _where(finding: dict) -> str:
    """Where the finding came from: file name (batch) and/or video time (live / video)."""
    name = finding.get("source_file")
    t = finding.get("t_sec")
    parts = []
    if name:
        parts.append(os.path.basename(str(name)))
    if t is not None and (not name or t > 0):
        parts.append(f"t={t:.1f}s")
    if not parts:
        parts.append(f"frame {_fmt(finding.get('frame_index'))}")
    return " @ ".join(parts)


def _rl_image(bgr: np.ndarray, max_w_mm: float = 80, max_h_mm: float = 60) -> RLImage:
    h, w = bgr.shape[:2]
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise ValueError("Could not encode image for PDF.")
    scale = min(max_w_mm * mm / w, max_h_mm * mm / h)
    return RLImage(io.BytesIO(buf.tobytes()), width=w * scale, height=h * scale)


def export_report(
    path: str,
    findings: Iterable[dict],
    meta: Optional[dict] = None,
    max_detail: int = 100,
) -> str:
    """Write the PDF to ``path`` and return the path."""
    findings = sorted(list(findings), key=lambda f: (-_sev_rank(f), -(f.get("confidence") or 0.0)))
    meta = dict(meta or {})
    styles = getSampleStyleSheet()
    small = ParagraphStyle("small", parent=styles["BodyText"], fontSize=8, leading=10)
    body = ParagraphStyle("body", parent=styles["BodyText"], fontSize=9, leading=12)
    banner = ParagraphStyle("banner", parent=styles["BodyText"], fontSize=9, leading=12,
                            textColor=colors.white)

    doc = SimpleDocTemplate(
        path, pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm,
        topMargin=15 * mm, bottomMargin=15 * mm,
        title="RODIC InfraAI Inspection Report", author="RODIC InfraAI",
    )
    story = [Paragraph("RODIC InfraAI - Inspection Report (screening aid)", styles["Title"])]

    banner_table = Table([[Paragraph(escape(DISCLAIMER), banner)]], colWidths=[180 * mm])
    banner_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#744210")),
        ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story += [banner_table, Spacer(1, 6 * mm)]

    meta.setdefault("generated", _dt.datetime.now().strftime("%Y-%m-%d %H:%M"))
    meta_rows = [[escape(str(k)), escape(str(v))] for k, v in meta.items()]
    meta_rows.append(["findings", str(len(findings))])
    meta_table = Table(meta_rows, colWidths=[40 * mm, 140 * mm])
    meta_table.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#4a5568")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story += [meta_table, Spacer(1, 4 * mm)]

    counts = {s: 0 for s in SEVERITY_ORDER}
    for f in findings:
        counts[(f.get("recommendation") or {}).get("severity", "none")] += 1
    summary = Table(
        [["severity"] + SEVERITY_ORDER, ["count"] + [str(counts[s]) for s in SEVERITY_ORDER]],
        colWidths=[25 * mm] + [31 * mm] * len(SEVERITY_ORDER),
    )
    summary.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#edf2f7")),
    ]))
    story += [Paragraph("Summary", styles["Heading3"]), summary, Spacer(1, 5 * mm)]

    header = ["#", "source", "class", "conf.", "width mm", "severity", "basis"]
    rows = [header]
    for i, f in enumerate(findings, start=1):
        rec = f.get("recommendation") or {}
        rows.append([
            str(f.get("id", i)),
            Paragraph(escape(_where(f)), small),
            Paragraph(escape(str(f.get("defect_class") or "unclassified")), small),
            _fmt(f.get("confidence"), "{:.2f}"),
            _fmt(f.get("crack_width_mm"), "{:.2f}"),
            rec.get("severity", "none"),
            Paragraph(escape(str(rec.get("severity_basis", ""))), small),
        ])
    table = Table(rows, repeatRows=1,
                  colWidths=[10 * mm, 28 * mm, 30 * mm, 14 * mm, 18 * mm, 22 * mm, 58 * mm])
    style = [
        ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#edf2f7")),
        ("FONTSIZE", (0, 0), (-1, -1), 8), ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]
    for r, f in enumerate(findings, start=1):
        sev = (f.get("recommendation") or {}).get("severity", "none")
        style.append(("TEXTCOLOR", (5, r), (5, r), SEVERITY_COLOURS.get(sev, colors.black)))
    table.setStyle(TableStyle(style))
    story += [Paragraph("All findings", styles["Heading3"]), table]

    detail = findings[:max_detail]
    if detail:
        story += [PageBreak(), Paragraph("Findings in detail (highest severity first)", styles["Heading3"])]
        if len(findings) > len(detail):
            story.append(Paragraph(
                f"Showing detail for the {len(detail)} highest-severity findings of {len(findings)}.", small))
        for i, f in enumerate(detail, start=1):
            rec = f.get("recommendation") or {}
            text = [Paragraph(
                f"<b>Finding {escape(str(f.get('id', i)))}</b> - {escape(str(f.get('defect_class') or 'unclassified'))}"
                f" - severity <b>{escape(rec.get('severity', 'none'))}</b><br/>{escape(_where(f))}", body)]
            text.append(Paragraph(escape(str(rec.get("severity_basis", ""))), small))
            for action in rec.get("actions", []):
                text.append(Paragraph("- " + escape(action), small))
            for note in rec.get("notes", []):
                text.append(Paragraph("<i>Note: " + escape(note) + "</i>", small))
            text.append(Paragraph(f"Source: {escape(str(f.get('source', 'n/a')))}", small))
            image = f.get("image")
            cells = [[_rl_image(image) if image is not None else "", text]]
            block = Table(cells, colWidths=[85 * mm, 95 * mm])
            block.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"),
                                       ("LINEBELOW", (0, 0), (-1, -1), 0.3, colors.lightgrey)]))
            story += [KeepTogether([block]), Spacer(1, 3 * mm)]

    rec_refs = next(
        (r for r in ((f.get("recommendation") or {}).get("references") for f in findings) if r), None)
    if rec_refs:
        story += [Spacer(1, 4 * mm), Paragraph("Standards to consult (verify current edition)", styles["Heading4"])]
        story += [Paragraph("- " + escape(r), small) for r in rec_refs]

    doc.build(story)
    return path
