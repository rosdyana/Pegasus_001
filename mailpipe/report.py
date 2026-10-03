"""Stage 8: results report (HTML + Markdown) and the topic discovery file."""
from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

from jinja2 import Environment, PackageLoader, select_autoescape

from . import analyze, review
from .lang import has_simplified
from .store import Store


def _env() -> Environment:
    env = Environment(loader=PackageLoader("mailpipe", "templates"),
                      autoescape=select_autoescape(["html"]), trim_blocks=True, lstrip_blocks=True)
    env.filters["pct"] = lambda x: "-" if x is None else f"{x * 100:.1f}%"
    env.filters["short"] = lambda s, n=80: (s or "")[:n] + ("…" if s and len(s) > n else "")
    return env


def build_context(store: Store, cfg, run_id: int) -> dict:
    run = store.one("SELECT * FROM runs WHERE run_id=?", (run_id,))
    stats = json.loads(run["stats_json"] or "{}")
    vram = json.loads(run["vram_peaks_json"] or "{}")
    q, s = store.q, store.scalar

    files = {r["status"]: r["n"] for r in q("SELECT status, COUNT(*) n FROM files GROUP BY status")}
    n_msg = s("SELECT COUNT(*) FROM files WHERE kind='msg' AND status='parsed'") or 0
    n_msg_body = s("SELECT COUNT(DISTINCT e.file_id) FROM emails e JOIN files f USING(file_id) "
                   "WHERE f.kind='msg' AND e.parent_email_id IS NULL AND length(trim(e.body_raw))>0") or 0

    cats = q("SELECT category, COUNT(*) n, AVG(confidence) conf FROM labels GROUP BY category ORDER BY n DESC")
    max_cat = max((c["n"] for c in cats), default=1)

    flags = analyze.rule_flags(store, cfg)
    open_actions = q("SELECT a.*, e.subject FROM action_items a LEFT JOIN emails e ON e.email_id=a.email_id "
                     "ORDER BY a.due_date IS NULL, a.due_date LIMIT 100")

    def top_entities(types: tuple[str, ...], n: int = 15):
        marks = ",".join("?" * len(types))
        return q(f"SELECT COALESCE(normalized, value) v, type, COUNT(*) n FROM entities WHERE type IN ({marks}) "
                 f"GROUP BY lower(COALESCE(normalized, value)), type ORDER BY n DESC LIMIT ?", (*types, n))

    failures = q("SELECT path, error FROM files WHERE status='failed'")
    att_failures = q("SELECT filename, kind, error FROM attachments WHERE status='failed'")
    unsupported = q("SELECT kind, lower(substr(filename, instr(filename, '.'))) ext, COUNT(*) n FROM attachments "
                    "WHERE status='unsupported' GROUP BY kind ORDER BY n DESC")
    stage_failures = q("SELECT stage, status, COUNT(*) n, MIN(error) example FROM progress "
                       "WHERE status<>'done' GROUP BY stage, status")

    rng = random.Random(run_id)
    ids = [r[0] for r in q("SELECT email_id FROM emails")]
    samples = []
    for eid in rng.sample(ids, min(10, len(ids))):
        e = store.one("SELECT e.*, l.category, l.confidence FROM emails e LEFT JOIN labels l USING(email_id) "
                      "WHERE email_id=?", (eid,))
        ents = q("SELECT type, value FROM entities WHERE email_id=? AND attachment_id IS NULL LIMIT 20", (eid,))
        samples.append({"e": e, "entities": ents})

    # zh-TW: how often OpenCC had to fix LLM output, and whether anything is still not zh-TW
    opencc = {r["item_type"]: (r["changed"], r["n"]) for r in q(
        "SELECT item_type, SUM(changed) changed, COUNT(*) n FROM opencc_log GROUP BY item_type")}
    profile = cfg.language.opencc_profile
    generated = [r[0] for r in q("SELECT summary FROM threads WHERE summary IS NOT NULL "
                                 "UNION ALL SELECT action FROM action_items "
                                 "UNION ALL SELECT summary FROM attachments WHERE summary IS NOT NULL")]
    still_simplified = sum(1 for t in generated if has_simplified(t, profile))

    fallbacks = s("SELECT COUNT(*) FROM labels WHERE fallback_used=1") or 0
    methods = q("SELECT method, model, COUNT(*) n FROM labels GROUP BY method, model")
    sheet_flags = q("SELECT s.name, a.filename, s.quality_flag FROM sheets s JOIN attachments a USING(attachment_id) "
                    "WHERE s.quality_flag IS NOT NULL")

    return {
        "run": run, "stats": stats, "vram": vram, "cfg": cfg,
        "files": files, "parse_rate": (n_msg_body / n_msg) if n_msg else None, "n_msg": n_msg,
        "counts": {
            "emails": s("SELECT COUNT(*) FROM emails"), "nested": s("SELECT COUNT(*) FROM emails WHERE parent_email_id IS NOT NULL"),
            "threads": s("SELECT COUNT(*) FROM threads"), "attachments": s("SELECT COUNT(*) FROM attachments"),
            "sheets": s("SELECT COUNT(*) FROM sheets"), "entities": s("SELECT COUNT(*) FROM entities"),
            "chunks": s("SELECT COUNT(*) FROM chunks"), "embedded": s("SELECT COUNT(*) FROM chunks WHERE embedded=1"),
            "labels": s("SELECT COUNT(*) FROM labels"), "actions": s("SELECT COUNT(*) FROM action_items"),
        },
        "categories": cats, "max_cat": max_cat,
        "top_senders": q("SELECT sender, COUNT(*) n FROM emails GROUP BY sender_email ORDER BY n DESC LIMIT 10"),
        "busy_threads": q("SELECT thread_id, topic, email_count, last_at FROM threads ORDER BY email_count DESC LIMIT 10"),
        "flags": flags, "open_actions": open_actions,
        "entities": {
            "Companies": top_entities(("company",)), "Products / parts": top_entities(("product model", "part_number")),
            "PO numbers": top_entities(("po_number",)), "People": top_entities(("person",)),
        },
        "failures": failures, "att_failures": att_failures, "unsupported": unsupported,
        "stage_failures": stage_failures, "samples": samples,
        "opencc": opencc, "still_simplified": still_simplified, "generated": len(generated),
        "fallbacks": fallbacks, "methods": methods, "sheet_flags": sheet_flags,
        "metrics": review.compute_metrics(store, cfg),
        "low_conf": s("SELECT COUNT(*) FROM labels WHERE confidence < ? OR category='other' OR fallback_used=1",
                      (cfg.thresholds.review_confidence,)),
    }


def write_reports(store: Store, cfg, paths, run_id: int) -> tuple[Path, Path]:
    ctx = build_context(store, cfg, run_id)
    env = _env()
    html = paths.out / "report.html"
    md = paths.out / "report.md"
    html.write_text(env.get_template("report.html.j2").render(**ctx), encoding="utf-8")
    md.write_text(env.get_template("report.md.j2").render(**ctx), encoding="utf-8")
    return html, md


def write_discovery(store: Store, cfg, path: Path, groups: dict[int, list[int]],
                    names: dict[int, tuple[str, str]]) -> None:
    lines = ["# Topic discovery", "",
             "Clusters of email embeddings (HDBSCAN) next to the categories the LLM assigned. "
             "A cluster spread over several categories hints at overlapping categories; a big cluster "
             "labelled `other` hints at a missing one.", "",
             "| Cluster | Name | Emails | Assigned categories | Sample subjects |", "|---|---|---|---|---|"]
    for c in sorted((c for c in groups if c >= 0), key=lambda c: -len(groups[c])):
        ids = groups[c]
        marks = ",".join("?" * len(ids))
        cats = Counter(r[0] or "unlabeled" for r in store.q(
            f"SELECT l.category FROM emails e LEFT JOIN labels l USING(email_id) WHERE e.email_id IN ({marks})", ids))
        subjects = [store.scalar("SELECT subject FROM emails WHERE email_id=?", (e,)) or "" for e in ids[:5]]
        name, desc = names.get(c, (f"cluster {c}", ""))
        cat_txt = ", ".join(f"{k} ({v})" for k, v in cats.most_common())
        subj_txt = "<br>".join(s.replace("|", "\\|")[:80] for s in subjects)
        lines.append(f"| {c} | **{name}**<br>{desc} | {len(ids)} | {cat_txt} | {subj_txt} |")
    noise = groups.get(-1, [])
    lines += ["", f"Unclustered (noise): {len(noise)} emails.", ""]
    lines += ["## Category totals", "", "| Category | Emails |", "|---|---|"]
    for r in store.q("SELECT category, COUNT(*) n FROM labels GROUP BY category ORDER BY n DESC"):
        lines.append(f"| {r['category']} | {r['n']} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
