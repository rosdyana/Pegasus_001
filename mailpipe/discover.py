"""Stage 1: scan the input folder, hash files, build the work queue."""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

from .store import Store

log = logging.getLogger(__name__)

KINDS = {".msg": "msg", ".xlsx": "xlsx", ".xlsm": "xlsx"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def scan(input_dir: Path) -> list[Path]:
    files = [
        p for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in KINDS and not p.name.startswith("~$")
    ]
    return sorted(files)


@dataclass
class DiscoverResult:
    found: int = 0
    new: int = 0
    skipped_done: int = 0
    duplicates: int = 0
    retry_failed: int = 0


def discover(store: Store, input_dir: Path, limit: int | None = None) -> DiscoverResult:
    res = DiscoverResult()
    paths = scan(input_dir)
    res.found = len(paths)
    queued = 0
    seen_this_run: set[str] = set()
    for path in paths:
        digest = sha256_file(path)
        if digest in seen_this_run:
            res.duplicates += 1
            continue
        seen_this_run.add(digest)
        row = store.one("SELECT file_id, status, path FROM files WHERE sha256=?", (digest,))
        if row is None:
            if limit is not None and queued >= limit:
                continue
            store.insert("files", {"path": str(path), "sha256": digest, "kind": KINDS[path.suffix.lower()],
                                   "status": "pending"})
            res.new += 1
            queued += 1
        elif row["status"] == "parsed":
            if row["path"] != str(path) and Path(row["path"]).exists():
                res.duplicates += 1
            else:
                if row["path"] != str(path):  # file was moved or renamed
                    store.update("files", "file_id", row["file_id"], path=str(path))
                res.skipped_done += 1
        elif row["status"] in ("failed", "pending"):
            # retry earlier failures; a failure is usually a bug fixed since
            if limit is not None and queued >= limit:
                continue
            store.update("files", "file_id", row["file_id"], status="pending", error=None, path=str(path))
            res.retry_failed += row["status"] == "failed"
            queued += 1
    store.commit()
    log.info("discover: %s", res)
    return res
