"""Stage 7b: embeddings via Ollama into LanceDB, plus hybrid search (vectors + FTS5 trigram)."""
from __future__ import annotations

import logging
from pathlib import Path

import httpx

from .gpu import OllamaAdmin, OOMError, Phase, is_oom
from .store import Store

log = logging.getLogger(__name__)

TABLE = "chunks"
META_COLS = ["chunk_id", "email_id", "thread_id", "attachment_id", "source_type", "date", "sender",
             "category", "language", "text"]


class Embedder:
    def __init__(self, cfg, phase: Phase | None = None):
        self.cfg = cfg
        self.base = cfg.llm.base_url.rstrip("/").removesuffix("/v1")
        self.model = cfg.embedding.model
        self.http = httpx.Client(timeout=600)
        self.phase = phase
        self.cpu = bool(phase and phase.device == "cpu")
        self.events: list[str] = []
        if phase is not None and cfg.llm.api == "ollama":
            phase.on_exit(lambda: OllamaAdmin(cfg.llm.base_url).unload(self.model))

    def _embed(self, texts: list[str]) -> list[list[float]]:
        body = {"model": self.model, "input": texts, "keep_alive": "10m", "truncate": True}
        if self.cpu:
            body["options"] = {"num_gpu": 0}
        r = self.http.post(f"{self.base}/api/embed", json=body)
        if r.status_code >= 400:
            err = r.text
            if is_oom(RuntimeError(err)):
                raise OOMError(err)
            raise RuntimeError(f"embed {r.status_code}: {err[:300]}")
        return r.json()["embeddings"]

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed with the OOM ladder: reload+retry, halve batch, CPU (num_gpu=0)."""
        out: list[list[float]] = []
        batch = self.cfg.embedding.batch_size
        i = 0
        retried = False
        while i < len(texts):
            chunk = texts[i:i + batch]
            try:
                out += self._embed(chunk)
                i += len(chunk)
                retried = False
            except OOMError as e:
                if not retried:
                    self.events.append(f"OOM at batch {batch}: reload and retry")
                    OllamaAdmin(self.cfg.llm.base_url).unload(self.model)
                    retried = True
                elif batch > 1:
                    batch //= 2
                    self.events.append(f"OOM: batch size halved to {batch}")
                elif not self.cpu and self.cfg.embedding.cpu_fallback:
                    self.cpu = True
                    self.events.append("OOM at batch 1: switching embeddings to CPU")
                    OllamaAdmin(self.cfg.llm.base_url).unload(self.model)
                else:
                    raise OOMError(f"embedding failed at batch size 1: {e}") from e
        return out


def open_table(vectors_dir: Path, create_dim: int | None = None):
    import lancedb

    db = lancedb.connect(str(vectors_dir))
    names = db.list_tables() if hasattr(db, "list_tables") else db.table_names()
    names = list(getattr(names, "tables", names))
    if TABLE in names:
        return db.open_table(TABLE)
    if create_dim is None:
        return None
    import pyarrow as pa

    schema = pa.schema([
        pa.field("vector", pa.list_(pa.float32(), create_dim)),
        pa.field("chunk_id", pa.string()), pa.field("email_id", pa.int64()), pa.field("thread_id", pa.string()),
        pa.field("attachment_id", pa.int64()), pa.field("source_type", pa.string()), pa.field("date", pa.string()),
        pa.field("sender", pa.string()), pa.field("category", pa.string()), pa.field("language", pa.string()),
        pa.field("text", pa.string()),
    ])
    return db.create_table(TABLE, schema=schema)


def _quote_list(ids: list[str]) -> str:
    return ", ".join("'" + i.replace("'", "''") + "'" for i in ids)


def sync_vectors(store: Store, vectors_dir: Path, embedder: Embedder, progress=None) -> dict:
    """Embed chunks with embedded=0 and drop vectors whose chunk no longer exists."""
    stats = {"embedded": 0, "deleted": 0}
    table = open_table(vectors_dir)
    live = {r[0] for r in store.q("SELECT chunk_id FROM chunks")}
    if table is not None:
        existing = set(table.to_arrow().column("chunk_id").to_pylist()) if table.count_rows() else set()
        stale = list(existing - live)
        for i in range(0, len(stale), 500):
            table.delete(f"chunk_id IN ({_quote_list(stale[i:i + 500])})")
        stats["deleted"] = len(stale)
        # rows marked embedded but missing in LanceDB (e.g. vectors dir removed) get re-embedded
        missing = [c for c in live if c not in existing]
        for i in range(0, len(missing), 500):
            part = missing[i:i + 500]
            store.db.execute(f"UPDATE chunks SET embedded=0 WHERE chunk_id IN ({','.join('?' * len(part))})", part)
    else:
        store.db.execute("UPDATE chunks SET embedded=0")
    rows = store.q("SELECT * FROM chunks WHERE embedded=0 ORDER BY rowid")
    step = max(64, embedder.cfg.embedding.batch_size * 4)
    for i in range(0, len(rows), step):
        part = rows[i:i + step]
        try:
            vecs = embedder.embed([r["text"] for r in part])
        except OOMError as e:
            for r in part:
                store.mark("embed", r["chunk_id"], "failed_oom", error=str(e)[:300])
            store.commit()
            continue
        if table is None:
            table = open_table(vectors_dir, create_dim=len(vecs[0]))
        records = [{"vector": v, **{c: r[c] for c in META_COLS}} for r, v in zip(part, vecs)]
        ids = [r["chunk_id"] for r in part]
        table.delete(f"chunk_id IN ({_quote_list(ids)})")
        table.add(records)
        store.db.executemany("UPDATE chunks SET embedded=1 WHERE chunk_id=?", [(c,) for c in ids])
        store.commit()
        stats["embedded"] += len(part)
        if progress:
            progress(len(part))
    stats["events"] = embedder.events
    return stats


def email_vectors(vectors_dir: Path) -> dict[int, list[float]]:
    """Mean of each email's body-chunk vectors (for clustering)."""
    import numpy as np

    table = open_table(vectors_dir)
    if table is None or not table.count_rows():
        return {}
    df = table.to_pandas(columns=["vector", "email_id", "source_type"]) if _supports_columns(table) else table.to_pandas()
    df = df[(df["source_type"] == "email") & df["email_id"].notna()]
    out = {}
    for eid, g in df.groupby("email_id"):
        out[int(eid)] = np.mean(np.stack(g["vector"].to_list()), axis=0).tolist()
    return out


def _supports_columns(table) -> bool:
    import inspect

    try:
        return "columns" in inspect.signature(table.to_pandas).parameters
    except (TypeError, ValueError):
        return False


def search(store: Store, vectors_dir: Path, embedder: Embedder, query: str, k: int = 5) -> list[dict]:
    """Hybrid search: vector top-k merged with FTS5 trigram hits via reciprocal rank fusion."""
    scores: dict[str, float] = {}
    table = open_table(vectors_dir)
    if table is not None:
        vec = embedder.embed([query])[0]
        for rank, r in enumerate(table.search(vec).limit(k * 4).to_list()):
            scores[r["chunk_id"]] = scores.get(r["chunk_id"], 0) + 1 / (60 + rank)
    for rank, cid in enumerate(fts_search(store, query, k * 4)):
        scores[cid] = scores.get(cid, 0) + 1 / (60 + rank)
    best = sorted(scores, key=scores.get, reverse=True)[:k]
    rows = {r["chunk_id"]: dict(r) for r in store.q(
        f"SELECT * FROM chunks WHERE chunk_id IN ({','.join('?' * len(best))})", best)} if best else {}
    return [{**rows[c], "score": round(scores[c], 4)} for c in best if c in rows]


def fts_search(store: Store, query: str, k: int) -> list[str]:
    terms = [t for t in query.split() if t]
    long_terms = [t for t in terms if len(t) >= 3]
    if long_terms:
        match = " OR ".join('"' + t.replace('"', '""') + '"' for t in long_terms)
        try:
            return [r[0] for r in store.q(
                "SELECT c.chunk_id FROM chunks_fts f JOIN chunks c ON c.rowid=f.rowid "
                "WHERE chunks_fts MATCH ? ORDER BY rank LIMIT ?", (match, k))]
        except Exception as e:
            log.debug("fts failed: %s", e)
    # trigram needs >= 3 characters; shorter queries fall back to LIKE
    if terms:
        where = " OR ".join("text LIKE ?" for _ in terms)
        return [r[0] for r in store.q(f"SELECT chunk_id FROM chunks WHERE {where} LIMIT ?",
                                      [*(f"%{t}%" for t in terms), k])]
    return []
