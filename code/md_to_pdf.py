#!/usr/bin/env python3
"""
Convert the analysis_results.md to a formatted PDF using reportlab.
Parses markdown elements and embeds referenced PNG figures.
"""

import os
import re
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
    PageBreak, KeepTogether
)
from reportlab.lib import colors

BASE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'output', 'analysis_summary'
)
MD_PATH = os.path.join(BASE_DIR, 'analysis_results.md')
PDF_PATH = os.path.join(BASE_DIR, 'analysis_results.pdf')
FIG_DIR = os.path.join(BASE_DIR, 'figures')


def get_styles():
    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        'DocTitle', parent=styles['Title'],
        fontSize=22, leading=28, spaceAfter=20,
        textColor=HexColor('#1a1a2e')
    ))
    styles.add(ParagraphStyle(
        'H2', parent=styles['Heading2'],
        fontSize=16, leading=22, spaceBefore=24, spaceAfter=10,
        textColor=HexColor('#16213e'), borderWidth=0,
        borderColor=HexColor('#0f3460'), borderPadding=4,
    ))
    styles.add(ParagraphStyle(
        'H3', parent=styles['Heading3'],
        fontSize=13, leading=18, spaceBefore=16, spaceAfter=8,
        textColor=HexColor('#1a1a2e'),
    ))
    styles.add(ParagraphStyle(
        'Body', parent=styles['Normal'],
        fontSize=10, leading=14, spaceAfter=8,
    ))
    styles.add(ParagraphStyle(
        'BulletItem', parent=styles['Normal'],
        fontSize=10, leading=14, spaceAfter=4,
        leftIndent=24, bulletIndent=12,
    ))
    styles.add(ParagraphStyle(
        'NumberedItem', parent=styles['Normal'],
        fontSize=10, leading=14, spaceAfter=4,
        leftIndent=24, bulletIndent=12,
    ))
    styles.add(ParagraphStyle(
        'TableCell', parent=styles['Normal'],
        fontSize=8.5, leading=11,
    ))
    styles.add(ParagraphStyle(
        'TableHeader', parent=styles['Normal'],
        fontSize=8.5, leading=11, textColor=colors.white,
    ))
    return styles


def md_inline(text):
    """Convert inline markdown (bold, italic, code) to reportlab XML."""
    # Bold + italic
    text = re.sub(r'\*\*\*(.+?)\*\*\*', r'<b><i>\1</i></b>', text)
    # Bold
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    # Italic
    text = re.sub(r'\*(.+?)\*', r'<i>\1</i>', text)
    # Inline code
    text = re.sub(r'`(.+?)`', r'<font face="Courier" size="9">\1</font>', text)
    # Subscripts - handle log₂FC, log₁₀, etc
    text = text.replace('₂', '<sub>2</sub>')
    text = text.replace('₁₀', '<sub>10</sub>')
    text = text.replace('₀', '<sub>0</sub>')
    # Other unicode
    text = text.replace('≥', '>=')
    text = text.replace('≤', '<=')
    text = text.replace('→', '->')
    text = text.replace('↔', '<->')
    text = text.replace('√', 'sqrt')
    text = text.replace('~', '~')
    text = text.replace('—', ' -- ')
    text = text.replace('–', '-')
    return text


def parse_table(lines):
    """Parse markdown table lines into list of lists."""
    rows = []
    for line in lines:
        line = line.strip()
        if line.startswith('|') and line.endswith('|'):
            cells = [c.strip() for c in line.split('|')[1:-1]]
            # Skip separator rows
            if all(re.match(r'^[-:]+$', c) for c in cells):
                continue
            rows.append(cells)
    return rows


def build_table(rows, styles):
    """Build a reportlab Table from parsed rows."""
    if not rows:
        return None

    header = rows[0]
    body = rows[1:]

    # Build cell paragraphs
    table_data = []
    header_cells = [Paragraph(md_inline(c), styles['TableHeader']) for c in header]
    table_data.append(header_cells)

    for row in body:
        # Pad if needed
        while len(row) < len(header):
            row.append('')
        cells = [Paragraph(md_inline(c), styles['TableCell']) for c in row[:len(header)]]
        table_data.append(cells)

    # Calculate column widths
    avail = 6.5 * inch
    n_cols = len(header)
    col_widths = [avail / n_cols] * n_cols

    t = Table(table_data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#1a1a2e')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTSIZE', (0, 0), (-1, -1), 8.5),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#cccccc')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, HexColor('#f5f5f5')]),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]))
    return t


def build_pdf():
    styles = get_styles()

    with open(MD_PATH, 'r') as f:
        md_text = f.read()

    lines = md_text.split('\n')
    story = []
    i = 0

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Skip empty lines
        if not stripped:
            i += 1
            continue

        # H1 title
        if stripped.startswith('# ') and not stripped.startswith('## '):
            title = stripped[2:]
            story.append(Paragraph(md_inline(title), styles['DocTitle']))
            story.append(Spacer(1, 12))
            i += 1
            continue

        # H2
        if stripped.startswith('## '):
            text = stripped[3:]
            story.append(Spacer(1, 6))
            story.append(Paragraph(md_inline(text), styles['H2']))
            i += 1
            continue

        # H3
        if stripped.startswith('### '):
            text = stripped[4:]
            story.append(Paragraph(md_inline(text), styles['H3']))
            i += 1
            continue

        # H4
        if stripped.startswith('#### '):
            text = stripped[5:]
            story.append(Paragraph(f'<b>{md_inline(text)}</b>', styles['Body']))
            story.append(Spacer(1, 4))
            i += 1
            continue

        # Image
        img_match = re.match(r'!\[.*?\]\((.+?)\)', stripped)
        if img_match:
            img_path = os.path.join(BASE_DIR, img_match.group(1))
            if os.path.exists(img_path):
                # Get image dimensions and scale to fit page width
                from PIL import Image as PILImage
                pil_img = PILImage.open(img_path)
                img_w, img_h = pil_img.size
                max_w = 6.2 * inch
                max_h = 7.0 * inch
                scale = min(max_w / img_w, max_h / img_h)
                w = img_w * scale
                h = img_h * scale

                img = Image(img_path, width=w, height=h)
                story.append(Spacer(1, 8))
                story.append(img)
                story.append(Spacer(1, 8))
            else:
                story.append(Paragraph(
                    f'<i>[Missing figure: {img_match.group(1)}]</i>', styles['Body']))
            i += 1
            continue

        # Table - collect all consecutive table lines
        if stripped.startswith('|'):
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith('|'):
                table_lines.append(lines[i])
                i += 1
            rows = parse_table(table_lines)
            if rows:
                t = build_table(rows, styles)
                if t:
                    story.append(Spacer(1, 6))
                    story.append(t)
                    story.append(Spacer(1, 6))
            continue

        # Numbered list
        num_match = re.match(r'^(\d+)\.\s+(.+)', stripped)
        if num_match:
            num = num_match.group(1)
            text = num_match.group(2)
            story.append(Paragraph(
                f'{num}. {md_inline(text)}', styles['NumberedItem']))
            i += 1
            continue

        # Bullet list
        if stripped.startswith('- '):
            text = stripped[2:]
            story.append(Paragraph(
                f'&bull; {md_inline(text)}', styles['BulletItem']))
            i += 1
            continue

        # Regular paragraph - collect consecutive non-special lines
        para_lines = []
        while i < len(lines):
            l = lines[i].strip()
            if not l or l.startswith('#') or l.startswith('|') or l.startswith('- ') or \
               l.startswith('![') or re.match(r'^\d+\.\s', l):
                break
            para_lines.append(l)
            i += 1

        if para_lines:
            text = ' '.join(para_lines)
            story.append(Paragraph(md_inline(text), styles['Body']))

    # Build
    doc = SimpleDocTemplate(
        PDF_PATH,
        pagesize=letter,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
        title='Spatial Transcriptomic Analysis of the Pregnant Mouse Brain',
        author='Bhatt Lab',
    )
    n_elements = len(story)
    print(f"Building PDF with {n_elements} elements...")
    doc.build(story)
    size_mb = os.path.getsize(PDF_PATH) / 1e6
    print(f"PDF written: {PDF_PATH} ({size_mb:.1f} MB, {n_elements} elements)")


if __name__ == '__main__':
    build_pdf()
