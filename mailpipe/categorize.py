"""Stage 5: LLM classification (+ action items in the same call) and topic discovery."""
from __future__ import annotations

import json
import logging
from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, create_model

from .extract import extract_dates
from .lang import estimate_tokens, normalize_width, to_zh_tw, trim_to_tokens
from .llm import LLM, LLMError
from .gpu import OOMError
from .store import Store

log = logging.getLogger(__name__)


class ActionItem(BaseModel):
    action: str
    owner: str | None = None
    due_date: str | None = Field(default=None, description="ISO date YYYY-MM-DD if stated")


def analysis_model(categories: list[str]) -> type[BaseModel]:
    return create_model(
        "EmailAnalysis",
        category=(Literal[tuple(categories)], ...),
        confidence=(float, Field(ge=0, le=1)),
        needs_reply=(bool, ...),
        priority=(Literal["low", "med", "high"], ...),
        language=(Literal["zh", "en", "mixed", "other"], ...),
        action_items=(list[ActionItem], Field(default_factory=list)),
    )


def output_language(cfg, lang: str | None) -> str:
    if lang == "en" and cfg.language.english_only_threads == "en":
        return "en"
    return cfg.language.output


def language_rule(out_lang: str) -> str:
    if out_lang == "en":
        return "Write all generated text (action items) in English."
    return ("所有產生的文字（action items 等）一律使用繁體中文（台灣用語），不可使用簡體字。"
            "英文產品名稱、料號、公司名稱保持原文。")


SYSTEM_CLASSIFY = """You classify business emails for a Taiwanese company. Emails are in Traditional Chinese, English, or both.
Pick exactly one category from the taxonomy. Use "other" only when nothing fits.
confidence: your probability (0-1) that the category is correct.
needs_reply: true if the recipient is expected to answer or act.
priority: high = urgent or blocking, med = needs attention this week, low = FYI.
language: the email's language (zh, en, mixed, other).
action_items: concrete tasks requested in THIS email (not in quoted history). Include owner and due_date (YYYY-MM-DD) only if stated. Empty list if none.
{lang_rule}
Respond with JSON only.

Taxonomy:
{taxonomy}"""


def build_classify_messages(cfg, email, attachments: list, sheets_headers: list[str], entities: list) -> callable:
    taxonomy = "\n".join(f"- {c.name}: {c.description}" for c in cfg.taxonomy)
    out_lang = output_language(cfg, email["language"])
    system = SYSTEM_CLASSIFY.format(taxonomy=taxonomy, lang_rule=language_rule(out_lang))
    head = [
        f"Subject: {email['subject'] or ''}",
        f"From: {email['sender'] or ''}",
        f"Date: {email['sent_at'] or ''}",
    ]
    if attachments:
        head.append("Attachments: " + ", ".join(a["filename"] for a in attachments))
    if sheets_headers:
        head.append("Sheet columns: " + " | ".join(sheets_headers)[:600])
    if entities:
        ents = ", ".join(f"{e['type']}={e['value']}" for e in entities[:25])
        head.append(f"Entities: {ents}")
    head_text = "\n".join(head)
    body = email["body_clean"] or email["body_raw"] or ""

    def build(max_input_tokens: int) -> list[dict]:
        budget = min(2000, max_input_tokens - estimate_tokens(system) - estimate_tokens(head_text) - 50)
        user = f"{head_text}\n\nBody:\n{trim_to_tokens(body, max(200, budget))}"
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    return build


def normalize_due(value: str | None, sent_at: str | None) -> str | None:
    """LLM due dates to ISO; free text like '下週五' is resolved against the email date, else dropped."""
    if not value:
        return None
    try:
        return date.fromisoformat(value.strip()[:10]).isoformat()
    except ValueError:
        pass
    found = extract_dates(normalize_width(value), sent_at)
    return found[0].normalized if found else None


def classify_emails(store: Store, cfg, llm: LLM, email_ids: list[int], progress=None) -> dict:
    Model = analysis_model(cfg.category_names)
    counts = {"done": 0, "failed": 0, "failed_oom": 0}
    profile = cfg.language.opencc_profile
    for eid in email_ids:
        email = store.one("SELECT * FROM emails WHERE email_id=?", (eid,))
        atts = store.q("SELECT attachment_id, filename FROM attachments WHERE email_id=? "
                       "AND status NOT IN ('inline')", (eid,))
        headers = [h for a in atts for s in store.q("SELECT columns_json FROM sheets WHERE attachment_id=?",
                                                    (a["attachment_id"],))
                   for h in json.loads(s["columns_json"] or "[]")[:15]]
        ents = store.q("SELECT type, value FROM entities WHERE email_id=? AND attachment_id IS NULL "
                       "ORDER BY confidence DESC", (eid,))
        build = build_classify_messages(cfg, email, atts, headers, ents)
        try:
            res, meta = llm.call(build, Model)
        except OOMError as e:
            store.mark("classify", eid, "failed_oom", error=str(e)[:500])
            counts["failed_oom"] += 1
            store.commit()
            continue
        except LLMError as e:
            store.mark("classify", eid, "failed", error=str(e)[:500])
            counts["failed"] += 1
            store.commit()
            continue
        store.upsert("labels", {
            "email_id": eid, "category": res.category, "confidence": res.confidence,
            "needs_reply": int(res.needs_reply), "priority": res.priority, "language": res.language,
            "method": f"llm:{meta.method}", "model": meta.model, "fallback_used": int(meta.fallback_used),
        }, key=("email_id",))
        store.db.execute("DELETE FROM action_items WHERE email_id=? AND source_email_id=email_id", (eid,))
        for i, a in enumerate(res.action_items):
            text = to_zh_tw(a.action, profile)
            owner = to_zh_tw(a.owner, profile) if a.owner else None
            due = normalize_due(a.due_date, email["sent_at"])
            store.upsert("opencc_log", {"item_type": "action", "item_id": f"{eid}:{i}",
                                        "changed": int(text != a.action)}, key=("item_type", "item_id"))
            store.insert("action_items", {
                "email_id": eid, "thread_id": email["thread_id"], "action": text, "owner": owner,
                "due_date": due, "source_email_id": eid, "method": f"llm:{meta.method}", "model": meta.model,
            })
        store.mark("classify", eid, "done", method=meta.method, model=meta.model)
        store.commit()
        counts["done"] += 1
        if progress:
            progress()
    return counts


# --------------------------------------------------------------------------- discovery

class ClusterName(BaseModel):
    name: str
    description: str


def cluster_emails(vectors: dict[int, list[float]], min_cluster_size: int) -> dict[int, int]:
    """HDBSCAN over L2-normalized email vectors. Returns email_id -> cluster (-1 = noise)."""
    import numpy as np
    from sklearn.cluster import HDBSCAN

    ids = list(vectors)
    X = np.asarray([vectors[i] for i in ids], dtype="float32")
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
    labels = HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean", copy=True).fit_predict(X)
    return dict(zip(ids, (int(x) for x in labels)))


def name_cluster(llm: LLM, subjects: list[str]) -> tuple[ClusterName | None, str | None]:
    def build(_max: int) -> list[dict]:
        return [
            {"role": "system", "content": "Name the common topic of these email subjects in 2-5 words, "
                                          "and give a one-line description. Use the subjects' main language. JSON only."},
            {"role": "user", "content": "\n".join(f"- {s}" for s in subjects)},
        ]
    try:
        res, _ = llm.call(build, ClusterName)
        return res, None
    except (LLMError, OOMError) as e:
        return None, str(e)
