from __future__ import annotations

import re
from pathlib import Path

from docx import Document
from docx.enum.text import WD_BREAK
from docx.shared import Inches, Pt


def add_inline(paragraph, text: str) -> None:
    text = text.replace('`', '')
    parts = re.split(r'(\*\*.*?\*\*)', text)
    for part in parts:
        if not part:
            continue
        if part.startswith('**') and part.endswith('**'):
            run = paragraph.add_run(part[2:-2])
            run.bold = True
        else:
            paragraph.add_run(part)


def convert(md_path: Path, docx_path: Path) -> None:
    doc = Document()
    section = doc.sections[0]
    section.top_margin = Inches(0.8)
    section.bottom_margin = Inches(0.8)
    section.left_margin = Inches(0.9)
    section.right_margin = Inches(0.9)
    styles = doc.styles
    styles['Normal'].font.name = 'Calibri'
    styles['Normal'].font.size = Pt(10.5)
    styles['Normal'].paragraph_format.space_after = Pt(4)
    styles['Normal'].paragraph_format.line_spacing = 1.08

    in_code = False
    for raw in md_path.read_text(encoding='utf-8-sig').splitlines():
        line = raw.rstrip()
        if line.startswith('```'):
            in_code = not in_code
            continue
        if not line.strip():
            continue
        if in_code:
            p = doc.add_paragraph()
            r = p.add_run(line)
            r.font.name = 'Consolas'
            r.font.size = Pt(9)
            continue
        if line.startswith('# '):
            doc.add_heading(line[2:].strip(), level=1)
        elif line.startswith('## '):
            doc.add_heading(line[3:].strip(), level=2)
        elif line.startswith('### '):
            doc.add_heading(line[4:].strip(), level=3)
        elif line.startswith('> '):
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Inches(0.25)
            add_inline(p, line[2:].strip())
            for run in p.runs:
                run.italic = True
        elif line.startswith('- '):
            p = doc.add_paragraph(style='List Bullet')
            add_inline(p, line[2:].strip())
        elif re.match(r'^\d+\.\s+', line):
            p = doc.add_paragraph(style='List Number')
            add_inline(p, re.sub(r'^\d+\.\s+', '', line))
        elif line.startswith('|') and line.endswith('|'):
            p = doc.add_paragraph()
            r = p.add_run(line)
            r.font.name = 'Consolas'
            r.font.size = Pt(8)
        else:
            p = doc.add_paragraph()
            add_inline(p, line)
    doc.save(docx_path)
    print('WROTE', docx_path)


ROOT = Path(r'E:\Cabbage')
docs = ROOT / 'docs'
convert(docs / 'COMPAG-D-26-10878_Cover_Letter.md', docs / 'COMPAG-D-26-10878_Cover_Letter.docx')
convert(docs / 'COMPAG-D-26-10878_Revised_Manuscript_Text.md', docs / 'COMPAG-D-26-10878_Revised_Manuscript_Text.docx')
