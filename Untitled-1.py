#!/usr/bin/env python3
"""
Pole Image Analyzer
====================
Analyzes JPG images of electrical poles using Claude Vision AI.
- Classifies each image by category (pole tag, brand height, GL, base, PRI, SEC, COMM, equipment, anchors, spans)
- Deduplicates similar/repeat images using perceptual hashing
- Prioritizes pole tag + number as first page
- Emphasizes measurements throughout
- Outputs a clean, structured PDF report

Usage:
    python pole_analyzer.py <image_dir> [--output report.pdf] [--threshold 10]

    image_dir    : folder containing JPG/JPEG files
    --output     : output PDF filename (default: pole_report.pdf)
    --threshold  : perceptual hash similarity threshold 0-64 (default: 8, lower = stricter dedup)
"""

import os
import sys
import json
import base64
import argparse
import hashlib
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import anthropic
import imagehash
from PIL import Image

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage,
    Table, TableStyle, PageBreak, HRFlowable, KeepTogether
)
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor

# ── Category definitions ─────────────────────────────────────────────────────

CATEGORIES = [
    "pole_tag",
    "brand_height",
    "GL",
    "base_of_pole",
    "PRI",
    "SEC",
    "COMM",
    "equipment",
    "anchors",
    "spans",
    "unknown",
]

CATEGORY_DISPLAY = {
    "pole_tag":      "Pole Tag / Identification",
    "brand_height":  "Brand Height",
    "GL":            "Ground Line (GL)",
    "base_of_pole":  "Base of Pole",
    "PRI":           "Primary (PRI)",
    "SEC":           "Secondary (SEC)",
    "COMM":          "Communications (COMM)",
    "equipment":     "Equipment",
    "anchors":       "Anchors / Guys",
    "spans":         "Spans (Each Direction)",
    "unknown":       "Uncategorized",
}

CATEGORY_ORDER = [
    "pole_tag", "brand_height", "GL", "base_of_pole",
    "PRI", "SEC", "COMM", "equipment", "anchors", "spans", "unknown",
]

# ── Image deduplication ───────────────────────────────────────────────────────

def compute_phash(image_path: str) -> imagehash.ImageHash:
    try:
        img = Image.open(image_path).convert("RGB")
        return imagehash.phash(img, hash_size=16)
    except Exception as e:
        print(f"  [warn] Could not hash {image_path}: {e}")
        return None


def deduplicate_images(image_paths: list[str], threshold: int = 8) -> list[str]:
    """Remove near-duplicate images using perceptual hashing."""
    print(f"\n[dedup] Checking {len(image_paths)} images for duplicates (threshold={threshold})...")
    hashes = {}
    kept = []
    removed = 0

    for path in image_paths:
        h = compute_phash(path)
        if h is None:
            kept.append(path)
            continue
        is_dup = False
        for existing_path, existing_hash in hashes.items():
            diff = h - existing_hash
            if diff <= threshold:
                print(f"  [dup] {os.path.basename(path)} ≈ {os.path.basename(existing_path)} (diff={diff})")
                is_dup = True
                removed += 1
                break
        if not is_dup:
            hashes[path] = h
            kept.append(path)

    print(f"[dedup] Kept {len(kept)}, removed {removed} duplicates.")
    return kept


# ── Claude Vision analysis ────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert electrical utility pole inspector AI.
You analyze photos of utility poles and their components.
You identify specific features and extract any visible measurements or labels.
Always respond with valid JSON only — no markdown, no preamble."""

ANALYSIS_PROMPT = """Analyze this utility pole photo carefully.

Identify which category BEST describes the primary subject of this photo:
- pole_tag        : A tag, sticker, or plate with the pole ID number/label
- brand_height    : The branded or stamped height marking on the pole
- GL              : Ground Line marking on the pole
- base_of_pole    : The base/bottom section of the pole at ground level
- PRI             : Primary electrical conductors / wires at top
- SEC             : Secondary electrical conductors / service drops
- COMM            : Communications cables (telephone, cable TV, fiber)
- equipment       : Transformers, switches, cutouts, capacitors, arrestors, or other mounted equipment
- anchors         : Guy wires, anchors, or down guys attached to the pole
- spans           : The wire spans going in different directions from the pole
- unknown         : Cannot determine / other

Respond ONLY with this exact JSON structure:
{
  "category": "<one of the categories above>",
  "confidence": <0.0-1.0>,
  "pole_number": "<pole ID/tag number if visible, else null>",
  "measurements": [
    {"label": "<measurement name>", "value": "<value with units>"}
  ],
  "notes": "<brief description of what is visible, 1-2 sentences>",
  "measurement_summary": "<concise text emphasizing all numeric measurements found, or empty string>"
}"""


def encode_image(image_path: str) -> tuple[str, str]:
    """Return (base64_data, media_type)."""
    with open(image_path, "rb") as f:
        data = base64.standard_b64encode(f.read()).decode("utf-8")
    ext = Path(image_path).suffix.lower()
    media_type = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png"
    return data, media_type


def analyze_image(client: anthropic.Anthropic, image_path: str) -> dict:
    """Send image to Claude and return parsed analysis."""
    print(f"  [AI] Analyzing: {os.path.basename(image_path)}")
    b64, media_type = encode_image(image_path)
    try:
        response = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=800,
            system=SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64}},
                    {"type": "text", "text": ANALYSIS_PROMPT},
                ]
            }]
        )
        raw = response.content[0].text.strip()
        result = json.loads(raw)
        result["_path"] = image_path
        result["_filename"] = os.path.basename(image_path)
        return result
    except json.JSONDecodeError as e:
        print(f"  [warn] JSON parse error for {image_path}: {e}")
        return {
            "_path": image_path, "_filename": os.path.basename(image_path),
            "category": "unknown", "confidence": 0.0,
            "pole_number": None, "measurements": [],
            "notes": "Analysis failed.", "measurement_summary": ""
        }
    except Exception as e:
        print(f"  [error] API error for {image_path}: {e}")
        return {
            "_path": image_path, "_filename": os.path.basename(image_path),
            "category": "unknown", "confidence": 0.0,
            "pole_number": None, "measurements": [],
            "notes": f"Error: {e}", "measurement_summary": ""
        }


# ── PDF Generation ────────────────────────────────────────────────────────────

DARK_BG    = HexColor("#1a1a2e")
ACCENT     = HexColor("#e8c96d")      # amber/gold
ACCENT2    = HexColor("#4fc3f7")      # sky blue
TEXT_LIGHT = HexColor("#f0f0f0")
SUBTEXT    = HexColor("#aaaaaa")
SECTION_BG = HexColor("#16213e")
ROW_ALT    = HexColor("#0f3460")
WHITE      = colors.white
BLACK      = colors.black


class NumberedCanvas(canvas.Canvas):
    """Canvas that adds page numbers and a footer stripe."""
    def __init__(self, *args, **kwargs):
        canvas.Canvas.__init__(self, *args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self._draw_footer(num_pages)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def _draw_footer(self, page_count):
        self.saveState()
        w, h = letter
        # Footer bar
        self.setFillColor(DARK_BG)
        self.rect(0, 0, w, 30, fill=1, stroke=0)
        # Page number
        self.setFillColor(SUBTEXT)
        self.setFont("Helvetica", 8)
        page_num = self._pageNumber
        self.drawRightString(w - 30, 10, f"Page {page_num} of {page_count}")
        self.drawString(30, 10, "POLE INSPECTION REPORT  •  CONFIDENTIAL")
        self.restoreState()


def make_styles():
    base = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "PoleTitle",
        fontName="Helvetica-Bold",
        fontSize=28,
        textColor=ACCENT,
        spaceAfter=6,
        alignment=TA_CENTER,
    )
    subtitle_style = ParagraphStyle(
        "PoleSubtitle",
        fontName="Helvetica",
        fontSize=11,
        textColor=SUBTEXT,
        spaceAfter=4,
        alignment=TA_CENTER,
    )
    section_header = ParagraphStyle(
        "SectionHeader",
        fontName="Helvetica-Bold",
        fontSize=14,
        textColor=ACCENT,
        spaceBefore=14,
        spaceAfter=6,
        borderPad=4,
    )
    label_style = ParagraphStyle(
        "LabelStyle",
        fontName="Helvetica-Bold",
        fontSize=9,
        textColor=ACCENT2,
        spaceAfter=2,
    )
    value_style = ParagraphStyle(
        "ValueStyle",
        fontName="Helvetica",
        fontSize=9,
        textColor=TEXT_LIGHT,
        spaceAfter=2,
    )
    measurement_style = ParagraphStyle(
        "MeasurementStyle",
        fontName="Helvetica-Bold",
        fontSize=10,
        textColor=ACCENT,
        spaceAfter=2,
    )
    notes_style = ParagraphStyle(
        "NotesStyle",
        fontName="Helvetica-Oblique",
        fontSize=8,
        textColor=SUBTEXT,
        spaceAfter=4,
    )
    toc_style = ParagraphStyle(
        "TOCStyle",
        fontName="Helvetica",
        fontSize=10,
        textColor=TEXT_LIGHT,
        spaceAfter=3,
    )
    pole_number_style = ParagraphStyle(
        "PoleNumber",
        fontName="Helvetica-Bold",
        fontSize=20,
        textColor=WHITE,
        spaceAfter=4,
        alignment=TA_CENTER,
    )
    return {
        "title": title_style,
        "subtitle": subtitle_style,
        "section": section_header,
        "label": label_style,
        "value": value_style,
        "measurement": measurement_style,
        "notes": notes_style,
        "toc": toc_style,
        "pole_number": pole_number_style,
    }


def dark_page_background(canvas_obj, doc):
    """Draw dark background on every page."""
    canvas_obj.saveState()
    w, h = letter
    canvas_obj.setFillColor(DARK_BG)
    canvas_obj.rect(0, 0, w, h, fill=1, stroke=0)
    # Subtle top accent bar
    canvas_obj.setFillColor(ACCENT)
    canvas_obj.rect(0, h - 4, w, 4, fill=1, stroke=0)
    canvas_obj.restoreState()


def build_cover_page(styles: dict, pole_number: str, image_count: int,
                     categories_found: list[str], timestamp: str) -> list:
    story = []
    story.append(Spacer(1, 1.2 * inch))
    story.append(Paragraph("⚡ POLE INSPECTION", styles["title"]))
    story.append(Paragraph("AI-Assisted Field Analysis Report", styles["subtitle"]))
    story.append(Spacer(1, 0.3 * inch))

    # Pole number box
    if pole_number:
        data = [[Paragraph(f"POLE  #{pole_number}", styles["pole_number"])]]
        t = Table(data, colWidths=[5 * inch])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), SECTION_BG),
            ("BOX", (0, 0), (-1, -1), 2, ACCENT),
            ("TOPPADDING", (0, 0), (-1, -1), 16),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 16),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ]))
        story.append(t)
        story.append(Spacer(1, 0.3 * inch))

    # Summary table
    cats_str = ", ".join(CATEGORY_DISPLAY.get(c, c) for c in categories_found if c != "unknown")
    summary_data = [
        [Paragraph("Generated", styles["label"]),  Paragraph(timestamp, styles["value"])],
        [Paragraph("Total Images", styles["label"]), Paragraph(str(image_count), styles["value"])],
        [Paragraph("Categories Found", styles["label"]), Paragraph(cats_str or "—", styles["value"])],
    ]
    t2 = Table(summary_data, colWidths=[1.8 * inch, 4 * inch])
    t2.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), SECTION_BG),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [SECTION_BG, ROW_ALT]),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("BOX", (0, 0), (-1, -1), 1, ACCENT),
        ("LINEBELOW", (0, 0), (-1, -2), 0.5, SUBTEXT),
    ]))
    story.append(t2)
    story.append(PageBreak())
    return story


def build_toc(styles: dict, categories_found: list[str]) -> list:
    story = []
    story.append(Paragraph("TABLE OF CONTENTS", styles["section"]))
    story.append(HRFlowable(width="100%", thickness=1, color=ACCENT, spaceAfter=8))
    for i, cat in enumerate(CATEGORY_ORDER, 1):
        if cat in categories_found:
            display = CATEGORY_DISPLAY.get(cat, cat)
            story.append(Paragraph(f"{'▸':>3}  {display}", styles["toc"]))
    story.append(PageBreak())
    return story


def image_card(img_path: str, analysis: dict, styles: dict, img_width: float = 3.8 * inch) -> list:
    """Build a card for a single image with metadata."""
    elements = []

    # Image
    try:
        pil_img = Image.open(img_path)
        orig_w, orig_h = pil_img.size
        aspect = orig_h / orig_w
        img_height = img_width * aspect
        # Cap height
        max_h = 3.0 * inch
        if img_height > max_h:
            img_height = max_h
            img_width = img_height / aspect
        rl_img = RLImage(img_path, width=img_width, height=img_height)
    except Exception as e:
        rl_img = None

    # Metadata column
    meta_rows = []
    filename = analysis.get("_filename", "")
    meta_rows.append([Paragraph("File", styles["label"]), Paragraph(filename, styles["value"])])

    conf = analysis.get("confidence", 0)
    conf_pct = f"{conf*100:.0f}%"
    meta_rows.append([Paragraph("Confidence", styles["label"]), Paragraph(conf_pct, styles["value"])])

    if analysis.get("pole_number"):
        meta_rows.append([
            Paragraph("Pole #", styles["label"]),
            Paragraph(str(analysis["pole_number"]), styles["measurement"])
        ])

    # Measurements — emphasized
    measurements = analysis.get("measurements", [])
    if measurements:
        meas_lines = "  |  ".join(f"{m['label']}: {m['value']}" for m in measurements)
        meta_rows.append([
            Paragraph("Measurements", styles["label"]),
            Paragraph(meas_lines, styles["measurement"])
        ])

    meas_summary = analysis.get("measurement_summary", "")
    if meas_summary:
        meta_rows.append([
            Paragraph("Summary", styles["label"]),
            Paragraph(meas_summary, styles["measurement"])
        ])

    notes = analysis.get("notes", "")
    if notes:
        meta_rows.append([Paragraph("Notes", styles["label"]), Paragraph(notes, styles["notes"])])

    meta_table = Table(meta_rows, colWidths=[1.0 * inch, 2.6 * inch])
    meta_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("LINEBELOW", (0, 0), (-1, -2), 0.3, HexColor("#333355")),
    ]))

    if rl_img:
        card_data = [[rl_img, meta_table]]
        card = Table(card_data, colWidths=[img_width + 0.15 * inch, 3.8 * inch])
    else:
        card_data = [[Paragraph("(image unavailable)", styles["notes"]), meta_table]]
        card = Table(card_data, colWidths=[1.5 * inch, 3.8 * inch])

    card.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), SECTION_BG),
        ("BOX", (0, 0), (-1, -1), 1, HexColor("#334466")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
    ]))

    elements.append(card)
    elements.append(Spacer(1, 0.15 * inch))
    return elements


def build_section(category: str, analyses: list[dict], styles: dict) -> list:
    story = []
    display = CATEGORY_DISPLAY.get(category, category.upper())
    story.append(Paragraph(f"▸  {display}", styles["section"]))
    story.append(HRFlowable(width="100%", thickness=1, color=ACCENT, spaceAfter=6))

    # Collect all measurements for a section summary banner
    all_meas = []
    for a in analyses:
        for m in a.get("measurements", []):
            all_meas.append(f"{m['label']}: {m['value']}")

    if all_meas:
        meas_text = "  ●  ".join(all_meas)
        banner_data = [[Paragraph(f"📏  {meas_text}", styles["measurement"])]]
        banner = Table(banner_data, colWidths=[6.8 * inch])
        banner.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), HexColor("#1a2a4a")),
            ("BOX", (0, 0), (-1, -1), 1.5, ACCENT2),
            ("TOPPADDING", (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ]))
        story.append(banner)
        story.append(Spacer(1, 0.1 * inch))

    for a in analyses:
        story.extend(image_card(a["_path"], a, styles))

    story.append(Spacer(1, 0.2 * inch))
    return story


def generate_pdf(grouped: dict[str, list], pole_number: str,
                 output_path: str, image_count: int):
    print(f"\n[pdf] Building report → {output_path}")
    styles = make_styles()
    timestamp = datetime.now().strftime("%Y-%m-%d  %H:%M")
    categories_found = [c for c in CATEGORY_ORDER if grouped.get(c)]

    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        leftMargin=0.65 * inch,
        rightMargin=0.65 * inch,
        topMargin=0.6 * inch,
        bottomMargin=0.5 * inch,
        title="Pole Inspection Report",
        author="Pole Analyzer AI",
    )

    story = []

    # Cover
    story.extend(build_cover_page(styles, pole_number, image_count, categories_found, timestamp))

    # TOC
    story.extend(build_toc(styles, categories_found))

    # Sections in priority order
    for cat in CATEGORY_ORDER:
        items = grouped.get(cat)
        if not items:
            continue
        story.extend(build_section(cat, items, styles))
        story.append(PageBreak())

    doc.build(
        story,
        onFirstPage=dark_page_background,
        onLaterPages=dark_page_background,
        canvasmaker=NumberedCanvas,
    )
    print(f"[pdf] Done! Report saved to: {output_path}")


# ── Main pipeline ─────────────────────────────────────────────────────────────

def find_pole_number(analyses: list[dict]) -> str:
    """Find the most confidently detected pole number across all analyses."""
    candidates = []
    for a in analyses:
        pn = a.get("pole_number")
        conf = a.get("confidence", 0)
        if pn:
            candidates.append((conf, pn))
    if not candidates:
        return "UNKNOWN"
    candidates.sort(reverse=True)
    return candidates[0][1]


def main():
    parser = argparse.ArgumentParser(description="Pole Image Analyzer — AI PDF Report Generator")
    parser.add_argument("image_dir", help="Directory containing JPG/JPEG pole images")
    parser.add_argument("--output", default="pole_report.pdf", help="Output PDF filename")
    parser.add_argument("--threshold", type=int, default=8,
                        help="Perceptual hash dedup threshold 0-64 (default 8)")
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    if not image_dir.is_dir():
        print(f"[error] Not a directory: {image_dir}")
        sys.exit(1)

    # Gather images
    exts = {".jpg", ".jpeg", ".JPG", ".JPEG"}
    all_images = [str(p) for p in sorted(image_dir.iterdir()) if p.suffix in exts]
    if not all_images:
        print(f"[error] No JPG images found in {image_dir}")
        sys.exit(1)
    print(f"[scan] Found {len(all_images)} images in {image_dir}")

    # Deduplicate
    unique_images = deduplicate_images(all_images, threshold=args.threshold)

    # Analyze with Claude
    client = anthropic.Anthropic()  # uses ANTHROPIC_API_KEY from env
    analyses = []
    print(f"\n[analyze] Sending {len(unique_images)} images to Claude Vision...")
    for img_path in unique_images:
        result = analyze_image(client, img_path)
        analyses.append(result)

    # Determine pole number
    pole_number = find_pole_number(analyses)
    print(f"\n[result] Detected pole number: {pole_number}")

    # Sort: pole_tag first, then by CATEGORY_ORDER
    def sort_key(a):
        cat = a.get("category", "unknown")
        try:
            idx = CATEGORY_ORDER.index(cat)
        except ValueError:
            idx = len(CATEGORY_ORDER)
        return (idx, -a.get("confidence", 0))

    analyses.sort(key=sort_key)

    # Group by category
    grouped = defaultdict(list)
    for a in analyses:
        cat = a.get("category", "unknown")
        if cat not in CATEGORIES:
            cat = "unknown"
        grouped[cat].append(a)

    # Print summary
    print("\n[summary] Category breakdown:")
    for cat in CATEGORY_ORDER:
        items = grouped.get(cat, [])
        if items:
            print(f"  {CATEGORY_DISPLAY[cat]:<35} {len(items)} image(s)")

    # Build PDF
    output_path = args.output
    generate_pdf(grouped, pole_number, output_path, len(analyses))


if __name__ == "__main__":
    main()