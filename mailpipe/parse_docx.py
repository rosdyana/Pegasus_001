"""Stage 2: .docx to Markdown, keeping headings, lists, and tables."""
from __future__ import annotations

import io
import re

import docx
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph

from .parse_xlsx import md_table


def _heading_level(p: Paragraph) -> int:
    name = (p.style.name if p.style is not None else "") or ""
    m = re.match(r"(?:Heading|標題)\s*(\d)", name, re.I)
    if m:
        return int(m.group(1))
    if name.lower() == "title":
        return 1
    return 0


def _list_prefix(p: Paragraph) -> str | None:
    name = ((p.style.name if p.style is not None else "") or "").lower()
    ppr = p._p.pPr
    has_num = ppr is not None and ppr.find(qn("w:numPr")) is not None
    if "list number" in name:
        return "1. "
    if "list" in name or has_num:
        return "- "
    return None


def _table_md(t: Table) -> str:
    rows = []
    for row in t.rows:
        rows.append([c.text.strip() or None for c in row.cells])
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [(r + [None] * width)[:width] for r in rows]
    header = [c or f"col_{i + 1}" for i, c in enumerate(rows[0])]
    return md_table(header, rows[1:])


def docx_to_markdown(data: bytes) -> str:
    d = docx.Document(io.BytesIO(data))
    out: list[str] = []
    for el in d.element.body.iterchildren():
        if el.tag == qn("w:p"):
            p = Paragraph(el, d)
            text = p.text.strip()
            if not text:
                continue
            level = _heading_level(p)
            if level:
                out.append(f"{'#' * min(level, 6)} {text}")
                continue
            prefix = _list_prefix(p)
            out.append(f"{prefix}{text}" if prefix else text)
        elif el.tag == qn("w:tbl"):
            md = _table_md(Table(el, d))
            if md:
                out.append(md)
    return "\n\n".join(out)
