"""Stage 7a: chunking per content type, with metadata, rebuilt only when the source changes."""
from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path

import pandas as pd

from .lang import estimate_tokens
from .parse_xlsx import md_table
from .store import Store

log = logging.getLogger(__name__)


def _sig(*parts) -> str:
    return hashlib.sha1("\x1f".join("" if p is None else str(p) for p in parts).encode()).hexdigest()[:16]


def _hard_split(s: str, target: int) -> list[str]:
    out = []
    while estimate_tokens(s) > target:
        cut = max(1, int(len(s) * target / estimate_tokens(s)))
        space = s.rfind(" ", cut // 2, cut)
        cut = space + 1 if space > 0 else cut
        out.append(s[:cut].rstrip())
        s = s[cut:]
    if s.strip():
        out.append(s)
    return out


def _pack(units: list[str], target: int, joiner: str) -> list[str]:
    out, cur = [], ""
    for u in units:
        if cur and estimate_tokens(cur) + estimate_tokens(u) > target:
            out.append(cur)
            cur = u
        else:
            sep = "" if joiner == " " and re.search(r"[。！？；]$", cur) else joiner
            cur = f"{cur}{sep}{u}" if cur else u
    if cur:
        out.append(cur)
    return out


def split_text(text: str, target: int) -> list[str]:
    """Paragraph-packing split at ~target tokens; long paragraphs split by sentence, then hard."""
    pieces: list[str] = []
    for para in re.split(r"\n\s*\n", text.strip()):
        para = para.strip()
        if not para:
            continue
        if estimate_tokens(para) <= target:
            pieces.append(para)
            continue
        units = []
        for sent in re.split(r"(?<=[。！？；])|(?<=[.!?;])\s+|\n", para):
            if sent and sent.strip():
                units += _hard_split(sent.strip(), target)
        pieces += _pack(units, target, " ")
    return _pack(pieces, target, "\n\n")


def _md_sections(md: str) -> list[tuple[list[str], str]]:
    """Split markdown into (heading_path, text) sections."""
    path: list[str] = []
    sections: list[tuple[list[str], str]] = []
    buf: list[str] = []

    def flush():
        if any(x.strip() for x in buf):
            sections.append((list(path), "\n".join(buf).strip()))
        buf.clear()

    for line in md.split("\n"):
        m = re.match(r"^(#{1,6})\s+(.*)", line)
        if m:
            flush()
            level = len(m.group(1))
            path[:] = path[:level - 1] + [m.group(2).strip()]
        else:
            buf.append(line)
    flush()
    return sections


class Chunker:
    def __init__(self, store: Store, cfg):
        self.store = store
        self.cfg = cfg
        self.target = cfg.chunk.target_tokens
        self.added = 0
        self.removed = 0

    def _replace(self, owner_key: str, signature: str, chunks: list[dict]) -> None:
        n = self.store.scalar("SELECT COUNT(*) FROM chunks WHERE owner_key=?", (owner_key,))
        self.store.db.execute("DELETE FROM chunks WHERE owner_key=?", (owner_key,))
        self.removed += n or 0
        for i, c in enumerate(chunks):
            c["chunk_id"] = hashlib.sha1(f"{owner_key}|{i}|{c['text']}".encode()).hexdigest()[:20]
            c["owner_key"] = owner_key
            c["token_count"] = estimate_tokens(c["text"])
            c["embedded"] = 0
            self.store.insert("chunks", c)
        self.added += len(chunks)
        self.store.mark("chunk", owner_key, "done", signature=signature)

    def run(self) -> dict:
        sigs = self.store.signatures("chunk")
        live: set[str] = set()

        emails = self.store.q(
            "SELECT e.*, l.category FROM emails e LEFT JOIN labels l USING(email_id) "
            "JOIN files f ON f.file_id=e.file_id WHERE f.status='parsed'")
        for e in emails:
            key = f"email:{e['email_id']}"
            live.add(key)
            sig = _sig(e["thread_id"], e["category"], e["language"], e["body_clean"], e["subject"])
            if sigs.get(key) != sig:
                self._replace(key, sig, self.email_chunks(e))

        for t in self.store.q("SELECT * FROM threads WHERE summary IS NOT NULL"):
            key = f"thread:{t['thread_id']}"
            live.add(key)
            sig = _sig(t["summary"])
            if sigs.get(key) != sig:
                self._replace(key, sig, [self.thread_chunk(t)])

        atts = self.store.q(
            "SELECT a.*, e.sent_at, e.sender, e.thread_id, e.language, l.category FROM attachments a "
            "LEFT JOIN emails e USING(email_id) LEFT JOIN labels l USING(email_id) "
            "WHERE a.status='parsed' AND a.kind IN ('docx','xlsx')")
        for a in atts:
            key = f"att:{a['attachment_id']}"
            live.add(key)
            sig = _sig(a["thread_id"], a["category"], a["text_md"], a["summary"])
            if sigs.get(key) != sig:
                self._replace(key, sig, self.attachment_chunks(a))

        # owners that no longer exist (e.g. thread regrouped)
        for key in set(sigs) - live:
            n = self.store.scalar("SELECT COUNT(*) FROM chunks WHERE owner_key=?", (key,))
            self.store.db.execute("DELETE FROM chunks WHERE owner_key=?", (key,))
            self.store.db.execute("DELETE FROM progress WHERE stage='chunk' AND item_id=?", (key,))
            self.removed += n or 0
        self.store.commit()
        return {"added": self.added, "removed": self.removed}

    # -- builders ------------------------------------------------------------
    def _meta(self, row, source_type: str, **extra) -> dict:
        return {"email_id": row["email_id"], "thread_id": row["thread_id"], "attachment_id": None,
                "source_type": source_type, "date": row["sent_at"], "sender": row["sender"],
                "category": row["category"], "language": row["language"], **extra}

    def email_chunks(self, e) -> list[dict]:
        header = (f"[{(e['sent_at'] or '')[:16]}] From: {e['sender'] or '?'} | Subject: {e['subject'] or ''}"
                  f" | Category: {e['category'] or 'unlabeled'}")
        body = e["body_clean"] or ""
        if not body.strip():
            return [{**self._meta(e, "email"), "text": header}]
        budget = max(100, self.target - estimate_tokens(header))
        return [{**self._meta(e, "email"), "text": f"{header}\n{part}"} for part in split_text(body, budget)]

    def thread_chunk(self, t) -> dict:
        text = f"Thread summary: {t['topic'] or ''} ({t['first_at'] or '?'} to {t['last_at'] or '?'}, " \
               f"{t['email_count']} emails)\n{t['summary']}"
        cat = self.store.scalar("SELECT l.category FROM emails e JOIN labels l USING(email_id) WHERE thread_id=? "
                                "GROUP BY l.category ORDER BY COUNT(*) DESC LIMIT 1", (t["thread_id"],))
        return {"email_id": None, "thread_id": t["thread_id"], "attachment_id": None, "source_type": "thread_summary",
                "date": t["last_at"], "sender": None, "category": cat, "language": t["language"], "text": text}

    def attachment_chunks(self, a) -> list[dict]:
        meta = {"email_id": a["email_id"], "thread_id": a["thread_id"], "attachment_id": a["attachment_id"],
                "date": a["sent_at"], "sender": a["sender"], "category": a["category"], "language": a["language"]}
        out: list[dict] = []
        if a["summary"]:
            out.append({**meta, "source_type": "attachment_summary",
                        "text": f"Summary of {a['filename']}:\n{a['summary']}"})
        if a["kind"] == "docx":
            for path, text in _md_sections(a["text_md"] or ""):
                prefix = " > ".join([f"Document: {a['filename']}", *path])
                budget = max(100, self.target - estimate_tokens(prefix))
                out += [{**meta, "source_type": "docx", "text": f"{prefix}\n{part}"} for part in split_text(text, budget)]
            return out
        for s in self.store.q("SELECT * FROM sheets WHERE attachment_id=?", (a["attachment_id"],)):
            out += self.sheet_chunks(a["filename"], s, meta)
        return out

    def sheet_chunks(self, filename: str, s, meta: dict) -> list[dict]:
        cols = json.loads(s["columns_json"] or "[]")
        try:
            df = pd.read_parquet(Path(s["parquet_path"]))
        except Exception as e:
            log.warning("cannot read %s: %s", s["parquet_path"], e)
            return []
        samples = []
        for c in cols[:20]:
            vals = [v for v in df[c].dropna().astype(str).unique()[:3]] if c in df else []
            samples.append(f"{c}: {', '.join(vals)}" if vals else c)
        summary = (f"Sheet '{s['name']}' in {filename}: {s['n_rows']} rows x {s['n_cols']} columns.\n"
                   f"Columns and sample values:\n" + "\n".join(f"- {x}" for x in samples))
        out = [{**meta, "source_type": "sheet_summary", "text": summary}]
        rows = df.astype(object).where(df.notna(), None).values.tolist()
        step = self.cfg.chunk.sheet_rows_per_chunk
        i = 0
        while i < len(rows):
            n = step
            # wide sheets: shrink the row group so a chunk stays near 2x target
            while n > 1 and estimate_tokens(md_table(cols, rows[i:i + n])) > self.target * 2:
                n //= 2
            group = rows[i:i + n]
            text = f"Sheet '{s['name']}' in {filename}, rows {i + 1}-{i + len(group)}:\n{md_table(cols, group)}"
            out.append({**meta, "source_type": "sheet_rows", "text": text})
            i += len(group)
        return out


def export_jsonl(store: Store, path: Path) -> int:
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for r in store.q("SELECT chunk_id, email_id, thread_id, attachment_id, source_type, date, sender, category, "
                         "language, token_count, text FROM chunks ORDER BY owner_key, rowid"):
            f.write(json.dumps(dict(r), ensure_ascii=False) + "\n")
            n += 1
    return n
