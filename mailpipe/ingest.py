"""Stage 2 persistence: parsed files into SQLite (emails, attachments, sheets)."""
from __future__ import annotations

import json
import logging
from pathlib import Path

from .parse_docx import docx_to_markdown
from .parse_msg import ParsedAttachment, ParsedEmail, parse_msg_file
from .parse_xlsx import parse_xlsx, workbook_markdown, write_parquet
from .store import Store, now

log = logging.getLogger(__name__)

PARSEABLE = {"docx", "xlsx", "msg"}


def ingest_file(store: Store, file_row, tables_dir: Path) -> None:
    """Parse one source file inside a savepoint; on error nothing is left half-written."""
    file_id, path = file_row["file_id"], Path(file_row["path"])
    store.db.execute("SAVEPOINT ingest")
    try:
        if file_row["kind"] == "msg":
            insert_email(store, parse_msg_file(path), file_id=file_id, parent_id=None, tables_dir=tables_dir)
        else:
            att_id = store.insert("attachments", {
                "email_id": None, "file_id": file_id, "sha256": file_row["sha256"], "filename": path.name,
                "mime": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                "size": path.stat().st_size, "kind": "xlsx", "status": "pending",
            })
            _parse_xlsx_into(store, att_id, file_id, path.read_bytes(), file_row["sha256"], tables_dir)
        store.db.execute("RELEASE ingest")
        store.update("files", "file_id", file_id, status="parsed", error=None, processed_at=now())
    except Exception as e:
        store.db.execute("ROLLBACK TO ingest")
        store.db.execute("RELEASE ingest")
        store.update("files", "file_id", file_id, status="failed", error=f"{type(e).__name__}: {e}",
                     processed_at=now())
        log.warning("parse failed for %s: %s", path, e)
    store.commit()


def insert_email(store: Store, pe: ParsedEmail, *, file_id: int, parent_id: int | None, tables_dir: Path) -> int:
    email_id = store.insert("emails", {
        "file_id": file_id, "parent_email_id": parent_id, "message_id": pe.message_id,
        "in_reply_to": pe.in_reply_to, "conversation_index": pe.conversation_index,
        "conversation_topic": pe.conversation_topic, "subject": pe.subject, "sender": pe.sender,
        "sender_email": pe.sender_email, "to_addrs": pe.to_addrs, "cc_addrs": pe.cc_addrs,
        "sent_at": pe.sent_at, "body_raw": pe.body, "body_source": pe.body_source,
    })
    for pa in pe.attachments:
        insert_attachment(store, email_id, file_id, pa, tables_dir)
    return email_id


def insert_attachment(store: Store, email_id: int, file_id: int, pa: ParsedAttachment, tables_dir: Path) -> None:
    row = {"email_id": email_id, "file_id": None, "sha256": pa.sha256, "filename": pa.filename,
           "mime": pa.mime, "size": pa.size, "kind": pa.kind}
    if pa.error:
        store.insert("attachments", {**row, "status": "failed", "error": pa.error})
        return
    if pa.inline:
        store.insert("attachments", {**row, "status": "inline"})
        return
    if pa.kind not in PARSEABLE:
        store.insert("attachments", {**row, "status": "unsupported"})
        return
    if pa.sha256:
        dup = store.one("SELECT attachment_id, text_md FROM attachments WHERE sha256=? AND status='parsed'",
                        (pa.sha256,))
        if dup is not None:
            store.insert("attachments", {**row, "status": "duplicate", "text_md": dup["text_md"],
                                         "error": f"same content as attachment {dup['attachment_id']}"})
            return
    att_id = store.insert("attachments", {**row, "status": "pending"})
    try:
        if pa.kind == "msg":
            insert_email(store, pa.nested, file_id=file_id, parent_id=email_id, tables_dir=tables_dir)
            store.update("attachments", "attachment_id", att_id, status="parsed")
        elif pa.kind == "docx":
            store.update("attachments", "attachment_id", att_id, status="parsed", text_md=docx_to_markdown(pa.data))
        elif pa.kind == "xlsx":
            _parse_xlsx_into(store, att_id, None, pa.data, pa.sha256, tables_dir)
    except Exception as e:
        store.update("attachments", "attachment_id", att_id, status="failed", error=f"{type(e).__name__}: {e}")
        log.info("attachment %s failed: %s", pa.filename, e)


def _parse_xlsx_into(store: Store, att_id: int, file_id: int | None, data: bytes, sha: str, tables_dir: Path) -> None:
    wb = parse_xlsx(data)
    for i, sheet in enumerate(wb.sheets):
        pq = write_parquet(sheet, tables_dir, sha, i)
        store.insert("sheets", {
            "attachment_id": att_id, "file_id": file_id, "name": sheet.name, "header_row": sheet.header_row,
            "n_rows": sheet.n_rows, "n_cols": sheet.n_cols, "columns_json": json.dumps(sheet.columns, ensure_ascii=False),
            "parquet_path": str(pq), "preview_md": sheet.preview_md(), "quality_flag": sheet.quality_flag,
        })
    note = f"hidden sheets skipped: {', '.join(wb.hidden_skipped)}" if wb.hidden_skipped else None
    store.update("attachments", "attachment_id", att_id, status="parsed", text_md=workbook_markdown(wb), error=note)
