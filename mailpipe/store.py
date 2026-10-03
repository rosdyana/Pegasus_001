"""SQLite schema and access helpers."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

SCHEMA = """
CREATE TABLE IF NOT EXISTS files (
    file_id      INTEGER PRIMARY KEY,
    path         TEXT NOT NULL,
    sha256       TEXT NOT NULL UNIQUE,
    kind         TEXT NOT NULL,              -- msg | xlsx
    status       TEXT NOT NULL,              -- pending | parsed | failed
    error        TEXT,
    processed_at TEXT
);
CREATE TABLE IF NOT EXISTS emails (
    email_id        INTEGER PRIMARY KEY,
    file_id         INTEGER REFERENCES files(file_id),
    parent_email_id INTEGER REFERENCES emails(email_id),
    thread_id       TEXT,
    message_id      TEXT,
    in_reply_to     TEXT,
    conversation_index TEXT,
    conversation_topic TEXT,
    subject         TEXT,
    sender          TEXT,
    sender_email    TEXT,
    to_addrs        TEXT,
    cc_addrs        TEXT,
    sent_at         TEXT,
    body_raw        TEXT,
    body_clean      TEXT,
    body_source     TEXT,                    -- plain | html | rtf | none
    language        TEXT
);
CREATE INDEX IF NOT EXISTS ix_emails_thread ON emails(thread_id);
CREATE INDEX IF NOT EXISTS ix_emails_file ON emails(file_id);
CREATE TABLE IF NOT EXISTS threads (
    thread_id    TEXT PRIMARY KEY,
    topic        TEXT,
    first_at     TEXT,
    last_at      TEXT,
    email_count  INTEGER,
    members_hash TEXT,
    language     TEXT,
    summary      TEXT,
    summary_json TEXT,
    summary_hash TEXT,                       -- members_hash the summary was built from
    method       TEXT,
    model        TEXT
);
CREATE TABLE IF NOT EXISTS attachments (
    attachment_id INTEGER PRIMARY KEY,
    email_id      INTEGER REFERENCES emails(email_id),
    file_id       INTEGER REFERENCES files(file_id),   -- set for standalone .xlsx
    sha256        TEXT,
    filename      TEXT,
    mime          TEXT,
    size          INTEGER,
    kind          TEXT,                      -- docx | xlsx | msg | other
    status        TEXT,                      -- parsed | unsupported | failed | inline
    error         TEXT,
    text_md       TEXT,
    summary       TEXT
);
CREATE INDEX IF NOT EXISTS ix_att_email ON attachments(email_id);
CREATE TABLE IF NOT EXISTS sheets (
    sheet_id      INTEGER PRIMARY KEY,
    attachment_id INTEGER REFERENCES attachments(attachment_id),
    file_id       INTEGER REFERENCES files(file_id),
    name          TEXT,
    header_row    INTEGER,
    n_rows        INTEGER,
    n_cols        INTEGER,
    columns_json  TEXT,
    parquet_path  TEXT,
    preview_md    TEXT,
    quality_flag  TEXT
);
CREATE TABLE IF NOT EXISTS entities (
    entity_id     INTEGER PRIMARY KEY,
    email_id      INTEGER,
    attachment_id INTEGER,
    type          TEXT,
    value         TEXT,
    normalized    TEXT,
    confidence    REAL,
    method        TEXT,                      -- regex | dateparser | gliner
    context       TEXT
);
CREATE INDEX IF NOT EXISTS ix_ent_email ON entities(email_id);
CREATE TABLE IF NOT EXISTS labels (
    email_id      INTEGER PRIMARY KEY,
    category      TEXT,
    confidence    REAL,
    needs_reply   INTEGER,
    priority      TEXT,
    language      TEXT,
    method        TEXT,
    model         TEXT,
    fallback_used INTEGER
);
CREATE TABLE IF NOT EXISTS reviews (
    review_id       INTEGER PRIMARY KEY,
    item_type       TEXT NOT NULL,           -- category | entity | action | summary | missing_entities
    item_id         TEXT NOT NULL,
    is_correct      INTEGER,
    corrected_value TEXT,
    rating          INTEGER,
    comment         TEXT,
    reviewer        TEXT,
    reviewed_at     TEXT,
    UNIQUE(item_type, item_id)
);
CREATE TABLE IF NOT EXISTS action_items (
    action_id       INTEGER PRIMARY KEY,
    email_id        INTEGER,
    thread_id       TEXT,
    action          TEXT,
    owner           TEXT,
    due_date        TEXT,
    source_email_id INTEGER,
    method          TEXT,
    model           TEXT
);
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id      TEXT PRIMARY KEY,
    email_id      INTEGER,
    thread_id     TEXT,
    attachment_id INTEGER,
    source_type   TEXT,                      -- email | thread_summary | docx | sheet_summary | sheet_rows | attachment
    owner_key     TEXT NOT NULL,             -- what produced the chunk, e.g. email:12, thread:ab12
    date          TEXT,
    sender        TEXT,
    category      TEXT,
    language      TEXT,
    text          TEXT,
    token_count   INTEGER,
    embedded      INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS ix_chunks_owner ON chunks(owner_key);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text, content='chunks', content_rowid='rowid', tokenize='trigram'
);
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, text) VALUES (new.rowid, new.text);
END;
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.rowid, old.text);
END;
CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE OF text ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.rowid, old.text);
    INSERT INTO chunks_fts(rowid, text) VALUES (new.rowid, new.text);
END;
-- per-item stage progress: lets every stage resume and be re-run alone
CREATE TABLE IF NOT EXISTS progress (
    stage      TEXT NOT NULL,
    item_id    TEXT NOT NULL,
    status     TEXT NOT NULL,                -- done | failed | failed_oom
    signature  TEXT,
    method     TEXT,
    model      TEXT,
    error      TEXT,
    updated_at TEXT,
    PRIMARY KEY (stage, item_id)
);
CREATE TABLE IF NOT EXISTS opencc_log (
    item_type TEXT, item_id TEXT, changed INTEGER, PRIMARY KEY (item_type, item_id)
);
CREATE TABLE IF NOT EXISTS runs (
    run_id          INTEGER PRIMARY KEY,
    started_at      TEXT,
    finished_at     TEXT,
    config_hash     TEXT,
    stats_json      TEXT,
    vram_peaks_json TEXT
);
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Store:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript(SCHEMA)
        self.db.commit()

    # -- generic helpers -------------------------------------------------
    def q(self, sql: str, params: Iterable[Any] = ()) -> list[sqlite3.Row]:
        return self.db.execute(sql, tuple(params)).fetchall()

    def one(self, sql: str, params: Iterable[Any] = ()) -> sqlite3.Row | None:
        return self.db.execute(sql, tuple(params)).fetchone()

    def scalar(self, sql: str, params: Iterable[Any] = ()) -> Any:
        row = self.one(sql, params)
        return row[0] if row else None

    def insert(self, table: str, row: dict[str, Any]) -> int:
        cols = ", ".join(row)
        marks = ", ".join("?" for _ in row)
        cur = self.db.execute(f"INSERT INTO {table} ({cols}) VALUES ({marks})", tuple(row.values()))
        return cur.lastrowid

    def upsert(self, table: str, row: dict[str, Any], key: Iterable[str]) -> None:
        cols = ", ".join(row)
        marks = ", ".join("?" for _ in row)
        keys = list(key)
        updates = ", ".join(f"{c}=excluded.{c}" for c in row if c not in keys)
        self.db.execute(
            f"INSERT INTO {table} ({cols}) VALUES ({marks}) ON CONFLICT({', '.join(keys)}) DO UPDATE SET {updates}",
            tuple(row.values()),
        )

    def update(self, table: str, key_col: str, key: Any, **values: Any) -> None:
        sets = ", ".join(f"{c}=?" for c in values)
        self.db.execute(f"UPDATE {table} SET {sets} WHERE {key_col}=?", (*values.values(), key))

    def commit(self) -> None:
        self.db.commit()

    def close(self) -> None:
        self.db.commit()
        self.db.close()

    # -- progress ----------------------------------------------------------
    def mark(self, stage: str, item_id: Any, status: str = "done", *, signature: str | None = None,
             method: str | None = None, model: str | None = None, error: str | None = None) -> None:
        self.upsert("progress", {
            "stage": stage, "item_id": str(item_id), "status": status, "signature": signature,
            "method": method, "model": model, "error": error, "updated_at": now(),
        }, key=("stage", "item_id"))

    def done_ids(self, stage: str) -> set[str]:
        return {r[0] for r in self.q("SELECT item_id FROM progress WHERE stage=? AND status='done'", (stage,))}

    def signatures(self, stage: str) -> dict[str, str]:
        return {r[0]: r[1] for r in self.q(
            "SELECT item_id, signature FROM progress WHERE stage=? AND status='done'", (stage,))}

    def reset_stage(self, stage: str) -> None:
        self.db.execute("DELETE FROM progress WHERE stage=?", (stage,))

    # -- runs ----------------------------------------------------------------
    def start_run(self, config_hash: str) -> int:
        run_id = self.insert("runs", {"started_at": now(), "config_hash": config_hash})
        self.commit()
        return run_id

    def finish_run(self, run_id: int, stats: dict, vram_peaks: dict) -> None:
        self.update("runs", "run_id", run_id, finished_at=now(),
                    stats_json=json.dumps(stats, ensure_ascii=False, default=str),
                    vram_peaks_json=json.dumps(vram_peaks))
        self.commit()
