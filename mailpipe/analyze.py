"""Stage 6: thread summaries, attachment summaries, and rule-based flags."""
from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta, timezone

from pydantic import BaseModel, Field

from .categorize import output_language
from .gpu import OOMError
from .lang import estimate_tokens, to_zh_tw, trim_to_tokens
from .llm import LLM, LLMError
from .store import Store

log = logging.getLogger(__name__)


class ThreadSummary(BaseModel):
    what_happened: str = Field(description="2-4 sentences")
    current_status: str
    open_questions: list[str] = Field(default_factory=list)
    waiting_on: list[str] = Field(default_factory=list, description="who is waiting on whom, e.g. 'A waits on B for X'")


class AttachmentSummary(BaseModel):
    bullets: list[str] = Field(min_length=1, max_length=5)


SYSTEM_THREAD = """You summarize an email thread for a busy colleague.
Report what happened, the current status, open questions, and who is waiting on whom. Be concrete: names, dates, numbers, part numbers.
{lang_rule}
Respond with JSON only."""

SYSTEM_ATTACH = """Summarize this document in 3 to 5 short bullets: what it is, key figures, dates, and anything that needs action.
{lang_rule}
Respond with JSON only."""


def _summary_lang_rule(out_lang: str) -> str:
    if out_lang == "en":
        return "Write in English."
    return ("請用繁體中文（台灣用語）撰寫，不可使用簡體字。英文產品名稱、料號、公司名稱保持原文。")


def _thread_text(store: Store, thread_id: str) -> str:
    rows = store.q("SELECT e.email_id, e.subject, e.sender, e.sent_at, e.body_clean, l.category "
                   "FROM emails e LEFT JOIN labels l USING(email_id) WHERE thread_id=? "
                   "ORDER BY sent_at, email_id", (thread_id,))
    parts = []
    for r in rows:
        parts.append(f"--- {r['sent_at'] or '?'} | {r['sender'] or '?'} | {r['subject'] or ''}"
                     f"{' | ' + r['category'] if r['category'] else ''}\n{r['body_clean'] or ''}")
    return "\n\n".join(parts)


def summarize_threads(store: Store, cfg, llm: LLM, thread_ids: list[str], progress=None) -> dict:
    counts = {"done": 0, "failed": 0, "failed_oom": 0}
    profile = cfg.language.opencc_profile
    for tid in thread_ids:
        t = store.one("SELECT * FROM threads WHERE thread_id=?", (tid,))
        if t is None:
            continue
        out_lang = output_language(cfg, t["language"])
        system = SYSTEM_THREAD.format(lang_rule=_summary_lang_rule(out_lang))
        text = _thread_text(store, tid)

        def build(max_in: int, text=text, system=system) -> list[dict]:
            # keep the latest messages when trimming: they carry the current status
            budget = max(300, max_in - estimate_tokens(system) - 50)
            body = text if estimate_tokens(text) <= budget else "[...earlier messages omitted]\n" + _tail(text, budget)
            return [{"role": "system", "content": system}, {"role": "user", "content": body}]

        try:
            res, meta = llm.call(build, ThreadSummary)
        except (OOMError, LLMError) as e:
            status = "failed_oom" if isinstance(e, OOMError) else "failed"
            store.mark("summarize", tid, status, error=str(e)[:500])
            counts[status] += 1
            store.commit()
            continue
        raw = res.model_dump()
        fixed = _opencc_tree(raw, profile) if out_lang != "en" else raw
        store.upsert("opencc_log", {"item_type": "thread", "item_id": tid, "changed": int(fixed != raw)},
                     key=("item_type", "item_id"))
        store.update("threads", "thread_id", tid, summary=render_summary(fixed, out_lang),
                     summary_json=json.dumps(fixed, ensure_ascii=False), summary_hash=t["members_hash"],
                     method=f"llm:{meta.method}", model=meta.model)
        store.mark("summarize", tid, "done", signature=t["members_hash"], method=meta.method, model=meta.model)
        store.commit()
        counts["done"] += 1
        if progress:
            progress()
    return counts


def _tail(text: str, max_tokens: int) -> str:
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi) // 2
        if estimate_tokens(text[mid:]) <= max_tokens:
            hi = mid
        else:
            lo = mid + 1
    return text[lo:]


def _opencc_tree(obj, profile: str):
    if isinstance(obj, str):
        return to_zh_tw(obj, profile)
    if isinstance(obj, list):
        return [_opencc_tree(x, profile) for x in obj]
    if isinstance(obj, dict):
        return {k: _opencc_tree(v, profile) for k, v in obj.items()}
    return obj


def render_summary(s: dict, lang: str) -> str:
    L = {"en": ("What happened", "Status", "Open questions", "Waiting on"),
         "zh": ("事件經過", "目前狀態", "待釐清問題", "等待事項")}["en" if lang == "en" else "zh"]
    out = [f"**{L[0]}**: {s['what_happened']}", f"**{L[1]}**: {s['current_status']}"]
    if s.get("open_questions"):
        out.append(f"**{L[2]}**:\n" + "\n".join(f"- {q}" for q in s["open_questions"]))
    if s.get("waiting_on"):
        out.append(f"**{L[3]}**:\n" + "\n".join(f"- {w}" for w in s["waiting_on"]))
    return "\n\n".join(out)


def attachments_needing_summary(store: Store, cfg) -> list[int]:
    rows = store.q("SELECT a.attachment_id, a.kind, length(a.text_md) AS n, "
                   "(SELECT COALESCE(SUM(n_rows),0) FROM sheets s WHERE s.attachment_id=a.attachment_id) AS rows "
                   "FROM attachments a WHERE a.status='parsed' AND a.kind IN ('docx','xlsx')")
    return [r["attachment_id"] for r in rows
            if (r["kind"] == "docx" and (r["n"] or 0) >= cfg.llm.attachment_summary_min_chars)
            or (r["kind"] == "xlsx" and (r["rows"] or 0) >= 100)]


def summarize_attachments(store: Store, cfg, llm: LLM, att_ids: list[int], progress=None) -> dict:
    counts = {"done": 0, "failed": 0, "failed_oom": 0}
    profile = cfg.language.opencc_profile
    for aid in att_ids:
        a = store.one("SELECT a.*, e.language FROM attachments a LEFT JOIN emails e USING(email_id) "
                      "WHERE attachment_id=?", (aid,))
        out_lang = output_language(cfg, a["language"])
        system = SYSTEM_ATTACH.format(lang_rule=_summary_lang_rule(out_lang))
        text = f"File: {a['filename']}\n\n{a['text_md'] or ''}"

        def build(max_in: int, text=text, system=system) -> list[dict]:
            return [{"role": "system", "content": system},
                    {"role": "user", "content": trim_to_tokens(text, max(300, max_in - estimate_tokens(system)))}]

        try:
            res, meta = llm.call(build, AttachmentSummary)
        except (OOMError, LLMError) as e:
            status = "failed_oom" if isinstance(e, OOMError) else "failed"
            store.mark("attach_summary", aid, status, error=str(e)[:500])
            counts[status] += 1
            continue
        bullets = [to_zh_tw(b, profile) if out_lang != "en" else b for b in res.bullets]
        store.upsert("opencc_log", {"item_type": "attachment", "item_id": str(aid),
                                    "changed": int(bullets != res.bullets)}, key=("item_type", "item_id"))
        store.update("attachments", "attachment_id", aid, summary="\n".join(f"- {b}" for b in bullets))
        store.mark("attach_summary", aid, "done", method=meta.method, model=meta.model)
        store.commit()
        counts["done"] += 1
        if progress:
            progress()
    return counts


# --------------------------------------------------------------------------- rule flags

def _as_date(s: str | None) -> date | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s[:10]).date()
    except ValueError:
        return None


def rule_flags(store: Store, cfg, today: date | None = None) -> dict[str, list[dict]]:
    """needs-reply older than N days, deadlines within N days, external senders."""
    today = today or datetime.now(timezone.utc).date()
    th = cfg.thresholds
    overdue = []
    for r in store.q("SELECT e.email_id, e.subject, e.sender, e.sent_at, e.thread_id, l.priority FROM emails e "
                     "JOIN labels l USING(email_id) WHERE l.needs_reply=1"):
        d = _as_date(r["sent_at"])
        # a later email in the thread from someone else counts as a reply
        later = store.scalar("SELECT COUNT(*) FROM emails WHERE thread_id=? AND sent_at>? AND sender<>?",
                             (r["thread_id"], r["sent_at"] or "", r["sender"] or ""))
        if d and not later and (today - d).days > th.needs_reply_days:
            overdue.append({**dict(r), "age_days": (today - d).days})
    deadlines = []
    horizon = today + timedelta(days=th.deadline_days)
    for r in store.q("SELECT a.*, e.subject FROM action_items a LEFT JOIN emails e ON e.email_id=a.email_id "
                     "WHERE a.due_date IS NOT NULL"):
        d = _as_date(r["due_date"])
        if d and d <= horizon:
            deadlines.append({**dict(r), "overdue": d < today})
    for r in store.q("SELECT n.email_id, n.normalized, n.value, e.subject FROM entities n JOIN emails e USING(email_id) "
                     "WHERE n.type IN ('deadline') AND n.normalized IS NOT NULL"):
        d = _as_date(r["normalized"])
        if d and today <= d <= horizon:
            deadlines.append({"email_id": r["email_id"], "action": r["value"], "due_date": r["normalized"],
                              "subject": r["subject"], "overdue": False})
    external = []
    internal = [d.lower().lstrip("@") for d in cfg.internal_domains]
    if internal:
        for r in store.q("SELECT email_id, sender, sender_email, subject FROM emails WHERE sender_email IS NOT NULL"):
            dom = r["sender_email"].split("@")[-1]
            if not any(dom == d or dom.endswith("." + d) for d in internal):
                external.append(dict(r))
    return {"overdue_replies": overdue, "deadlines": deadlines, "external": external}
