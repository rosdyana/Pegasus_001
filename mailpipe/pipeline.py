"""Two-pass orchestration: all CPU work first, then GPU phases grouped by model."""
from __future__ import annotations

import logging
import shutil
import time
from contextlib import contextmanager
from pathlib import Path

from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn

from . import analyze, categorize, chunk, embed, report
from .clean import Boilerplate, clean_body, first_pass
from .discover import discover
from .extract import GlinerRunner, extract_cpu
from .gpu import GPUManager
from .ingest import ingest_file
from .lang import detect_language
from .llm import LLM, LLMStats
from .store import Store
from .thread import build_threads

log = logging.getLogger(__name__)
console = Console()

REDO_STAGES = {
    "clean": ["clean", "regex", "gliner", "classify", "summarize"],
    "entities": ["regex", "gliner"],
    "categorize": ["classify"],
    "analyze": ["summarize", "attach_summary"],
    "embed": [],
}


class Paths:
    def __init__(self, out: Path):
        self.out = out
        self.db = out / "mailpipe.db"
        self.vectors = out / "vectors"
        self.tables = out / "tables"
        self.review = out / "review"
        self.logs = out / "logs"
        self.chunks_jsonl = out / "chunks.jsonl"
        self.discovery = out / "topic_discovery.md"


@contextmanager
def bar(desc: str, total: int):
    with Progress(TextColumn(f"[bold]{desc}"), BarColumn(), MofNCompleteColumn(), TimeElapsedColumn(),
                  console=console, transient=False) as p:
        task = p.add_task(desc, total=total)
        yield lambda n=1: p.advance(task, n)


class Pipeline:
    def __init__(self, cfg, *, force: bool = False, limit: int | None = None, cpu_only: bool = False,
                 redo: list[str] | None = None):
        self.cfg = cfg
        self.force = force
        self.limit = limit
        self.cpu_only = cpu_only
        self.redo = redo or []
        self.paths = Paths(Path(cfg.output_dir))
        self.stats: dict = {"timings_s": {}}
        self.llm_stats = LLMStats()

    @contextmanager
    def timed(self, name: str):
        t0 = time.monotonic()
        try:
            yield
        finally:
            self.stats["timings_s"][name] = round(time.monotonic() - t0, 1)

    def wipe(self) -> None:
        p = self.paths
        for f in (p.db, Path(f"{p.db}-wal"), Path(f"{p.db}-shm"), p.chunks_jsonl, p.discovery):
            f.unlink(missing_ok=True)
        for d in (p.vectors, p.tables):
            shutil.rmtree(d, ignore_errors=True)
        console.print("[yellow]--force: cleared previous results[/]")

    def apply_redo(self, store: Store) -> None:
        for name in self.redo:
            for stage in REDO_STAGES[name]:
                store.reset_stage(stage)
            if name == "categorize":
                store.db.execute("DELETE FROM labels")
                store.db.execute("DELETE FROM action_items WHERE source_email_id IS NOT NULL")
            if name == "analyze":
                store.db.execute("UPDATE threads SET summary=NULL, summary_json=NULL, summary_hash=NULL")
                store.db.execute("UPDATE attachments SET summary=NULL")
            if name in ("entities", "clean"):
                store.db.execute("DELETE FROM entities")
            if name == "embed":
                store.db.execute("UPDATE chunks SET embedded=0")
                shutil.rmtree(self.paths.vectors, ignore_errors=True)
            console.print(f"[yellow]--redo {name}: cleared {', '.join(REDO_STAGES[name]) or 'vectors'}[/]")
        store.commit()

    # ---------------------------------------------------------------------------
    def run(self) -> dict:
        cfg, p = self.cfg, self.paths
        p.out.mkdir(parents=True, exist_ok=True)
        if self.force:
            self.wipe()
        store = Store(p.db)
        self.apply_redo(store)
        run_id = store.start_run(cfg.hash())
        gpu = GPUManager(cfg, cpu_only=self.cpu_only)
        t_start = time.monotonic()
        try:
            with self.timed("discover"):
                d = discover(store, Path(cfg.input_dir), self.limit)
                self.stats["discover"] = d.__dict__
                console.print(f"discover: {d.found} found, {d.new} new, {d.skipped_done} already done, "
                              f"{d.duplicates} duplicates, {d.retry_failed} retried")
            with self.timed("pass1_cpu"):
                self.pass1(store)
            with self.timed("p2_gliner"):
                self.p2_gliner(store, gpu)
            with self.timed("p3_llm"):
                labels_changed = self.p3_llm(store, gpu)
            with self.timed("chunk"):
                self.stats["chunks"] = chunk.Chunker(store, cfg).run()
                self.stats["chunks"]["exported"] = chunk.export_jsonl(store, p.chunks_jsonl)
            with self.timed("p4_embed"):
                self.p4_embed(store, gpu)
            with self.timed("p5_discovery"):
                self.p5_discovery(store, gpu, labels_changed)
        finally:
            self.stats["llm"] = self.llm_stats.__dict__
            self.stats["total_s"] = round(time.monotonic() - t_start, 1)
            vram = gpu.report()
            gpu.close()
            store.finish_run(run_id, self.stats, vram)
        with self.timed("report"):
            paths = report.write_reports(store, cfg, p, run_id)
        console.print(f"[green]report:[/] {paths[0]}")
        store.close()
        return self.stats

    # -- pass 1: CPU -------------------------------------------------------------
    def pass1(self, store: Store) -> None:
        cfg, p = self.cfg, self.paths
        pending = store.q("SELECT * FROM files WHERE status='pending' ORDER BY file_id")
        if pending:
            with bar("parse", len(pending)) as adv:
                for f in pending:
                    ingest_file(store, f, p.tables)
                    adv()
        self.stats["files"] = {r["status"]: r["n"] for r in store.q(
            "SELECT status, COUNT(*) n FROM files GROUP BY status")}

        # clean: boilerplate statistics come from all emails, cleaning is applied to new ones
        done = store.done_ids("clean")
        emails = store.q("SELECT email_id, sender_email, body_raw FROM emails")
        todo = [e for e in emails if str(e["email_id"]) not in done]
        if todo:
            bp = Boilerplate(cfg.clean.boilerplate_min_emails, cfg.clean.boilerplate_min_same_sender,
                             cfg.clean.disclaimers)
            for e in emails:
                bp.observe(e["sender_email"], first_pass(e["body_raw"]))
            for e in todo:
                body = clean_body(e["body_raw"], e["sender_email"], bp)
                subj = store.scalar("SELECT subject FROM emails WHERE email_id=?", (e["email_id"],)) or ""
                store.update("emails", "email_id", e["email_id"], body_clean=body,
                             language=detect_language(f"{subj}\n{body}"))
                store.mark("clean", e["email_id"])
            store.commit()
        self.stats["cleaned"] = len(todo)

        n_threads = build_threads(store)
        self.stats["threads"] = n_threads

        # regex + date entities
        done = store.done_ids("regex")
        n = 0
        for e in store.q("SELECT email_id, subject, body_clean, sent_at FROM emails"):
            key = f"e{e['email_id']}"
            if key in done:
                continue
            text = f"{e['subject'] or ''}\n{e['body_clean'] or ''}"
            self._store_entities(store, e["email_id"], None, extract_cpu(text, e["sent_at"], cfg.regex), text,
                                 methods=("regex", "relative", "dateparser"))
            store.mark("regex", key)
            n += 1
        for a in store.q("SELECT a.attachment_id, a.email_id, a.text_md, e.sent_at FROM attachments a "
                         "LEFT JOIN emails e USING(email_id) WHERE a.status='parsed' AND a.text_md IS NOT NULL"):
            key = f"a{a['attachment_id']}"
            if key in done:
                continue
            text = a["text_md"][:200_000]
            self._store_entities(store, a["email_id"], a["attachment_id"], extract_cpu(text, a["sent_at"], cfg.regex),
                                 text, methods=("regex", "relative", "dateparser"))
            store.mark("regex", key)
            n += 1
        store.commit()
        self.stats["regex_items"] = n

    def _store_entities(self, store: Store, email_id, attachment_id, ents, text: str, methods: tuple[str, ...]) -> None:
        marks = ",".join("?" * len(methods))
        if attachment_id is None:
            store.db.execute(f"DELETE FROM entities WHERE email_id=? AND attachment_id IS NULL AND method IN ({marks})",
                             (email_id, *methods))
        else:
            store.db.execute(f"DELETE FROM entities WHERE attachment_id=? AND method IN ({marks})",
                             (attachment_id, *methods))
        for e in ents:
            store.insert("entities", {"email_id": email_id, "attachment_id": attachment_id, "type": e.type,
                                      "value": e.value, "normalized": e.normalized, "confidence": round(e.confidence, 3),
                                      "method": e.method, "context": e.context(text)})

    # -- P2: GLiNER ----------------------------------------------------------------
    def p2_gliner(self, store: Store, gpu: GPUManager) -> None:
        cfg = self.cfg
        done = store.done_ids("gliner")
        items = []
        for e in store.q("SELECT email_id, subject, body_clean, sent_at FROM emails"):
            if f"e{e['email_id']}" not in done:
                items.append({"key": f"e{e['email_id']}", "email_id": e["email_id"], "attachment_id": None,
                              "text": f"{e['subject'] or ''}\n{e['body_clean'] or ''}", "sent_at": e["sent_at"]})
        for a in store.q("SELECT a.attachment_id, a.email_id, a.text_md, e.sent_at FROM attachments a "
                         "LEFT JOIN emails e USING(email_id) WHERE a.status='parsed' AND a.text_md IS NOT NULL"):
            if f"a{a['attachment_id']}" not in done:
                items.append({"key": f"a{a['attachment_id']}", "email_id": a["email_id"],
                              "attachment_id": a["attachment_id"],
                              "text": a["text_md"][:cfg.gliner.max_chars_per_attachment], "sent_at": a["sent_at"]})
        if not items:
            self.stats["gliner"] = {"items": 0}
            return
        console.print(f"[bold]P2 GLiNER[/] on {len(items)} texts")
        with gpu.phase("gliner", cfg.gliner.vram_gb, cpu_ok=True) as ph:
            runner = GlinerRunner(cfg, ph)

            def on_item(it, ents):
                if ents is None:
                    store.mark("gliner", it["key"], "failed_oom", method=runner.device)
                    return
                self._store_entities(store, it["email_id"], it["attachment_id"], ents, it["text"], methods=("gliner",))
                store.mark("gliner", it["key"], "done", method=runner.device, model=cfg.gliner.model)

            self.stats["gliner"] = runner.run(items, on_item)
            for ev in runner.events:
                ph.note(ev)
        store.commit()

    # -- P3: LLM -------------------------------------------------------------------
    def p3_llm(self, store: Store, gpu: GPUManager) -> bool:
        cfg = self.cfg
        done = store.done_ids("classify")
        emails = [r["email_id"] for r in store.q(
            "SELECT e.email_id FROM emails e JOIN files f USING(file_id) WHERE f.status='parsed' ORDER BY sent_at")
            if str(r["email_id"]) not in done]
        sum_sigs = store.signatures("summarize")
        atts_done = store.done_ids("attach_summary")
        atts = [a for a in analyze.attachments_needing_summary(store, cfg) if str(a) not in atts_done]

        def stale_threads():
            return [t["thread_id"] for t in store.q("SELECT thread_id, members_hash FROM threads")
                    if sum_sigs.get(t["thread_id"]) != t["members_hash"]]

        if not emails and not stale_threads() and not atts:
            self.stats["llm_items"] = {}
            return False
        console.print(f"[bold]P3 LLM[/] {cfg.llm.model}: {len(emails)} emails, "
                      f"{len(stale_threads())} threads, {len(atts)} attachments")
        with gpu.phase("llm", cfg.llm.vram_gb, fallback_need_gb=cfg.llm.fallback_vram_gb) as ph:
            llm = LLM(cfg, ph, self.llm_stats)
            res = {}
            if emails:
                with bar("classify", len(emails)) as adv:
                    res["classify"] = categorize.classify_emails(store, cfg, llm, emails, adv)
            threads = stale_threads()
            if threads:
                with bar("thread summaries", len(threads)) as adv:
                    res["summarize"] = analyze.summarize_threads(store, cfg, llm, threads, adv)
            if atts:
                with bar("attachment summaries", len(atts)) as adv:
                    res["attach_summary"] = analyze.summarize_attachments(store, cfg, llm, atts, adv)
        self.stats["llm_items"] = res
        return bool(emails)

    # -- P4: embeddings ------------------------------------------------------------
    def p4_embed(self, store: Store, gpu: GPUManager) -> None:
        cfg = self.cfg
        pending = store.scalar("SELECT COUNT(*) FROM chunks WHERE embedded=0")
        table = embed.open_table(self.paths.vectors)
        n_vec = table.count_rows() if table is not None else 0
        n_chunks = store.scalar("SELECT COUNT(*) FROM chunks")
        if not pending and n_vec == n_chunks:
            self.stats["embed"] = {"embedded": 0}
            return
        console.print(f"[bold]P4 embeddings[/] {cfg.embedding.model}: {pending} chunks")
        with gpu.phase("embed", cfg.embedding.vram_gb, cpu_ok=cfg.embedding.cpu_fallback) as ph:
            emb = embed.Embedder(cfg, ph)
            with bar("embed", pending) as adv:
                self.stats["embed"] = embed.sync_vectors(store, self.paths.vectors, emb, adv)
            for ev in emb.events:
                ph.note(ev)

    # -- P5: discovery -------------------------------------------------------------
    def p5_discovery(self, store: Store, gpu: GPUManager, labels_changed: bool) -> None:
        cfg, p = self.cfg, self.paths
        if not labels_changed and p.discovery.exists():
            return
        vectors = embed.email_vectors(p.vectors)
        if len(vectors) < cfg.discovery.min_emails:
            p.discovery.write_text(f"# Topic discovery\n\nSkipped: {len(vectors)} embedded emails "
                                   f"(need {cfg.discovery.min_emails}).\n", encoding="utf-8")
            return
        clusters = categorize.cluster_emails(vectors, max(3, len(vectors) // 40))
        groups: dict[int, list[int]] = {}
        for eid, c in clusters.items():
            groups.setdefault(c, []).append(eid)
        names: dict[int, tuple[str, str]] = {}
        real = [c for c in groups if c >= 0]
        if real:
            with gpu.phase("llm_discovery", cfg.llm.vram_gb, fallback_need_gb=cfg.llm.fallback_vram_gb) as ph:
                llm = LLM(cfg, ph, self.llm_stats)
                for c in real:
                    subjects = [store.scalar("SELECT subject FROM emails WHERE email_id=?", (e,)) or ""
                                for e in groups[c][:cfg.discovery.samples_per_cluster]]
                    res, err = categorize.name_cluster(llm, subjects)
                    names[c] = (res.name, res.description) if res else (f"cluster {c}", f"(naming failed: {err})")
        report.write_discovery(store, cfg, p.discovery, groups, names)
        self.stats["discovery"] = {"clusters": len(real), "noise": len(groups.get(-1, []))}
