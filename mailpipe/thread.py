"""Stage 3b: conversation reconstruction.

Emails are joined by (1) the 22-byte Conversation-Index header, (2) In-Reply-To -> Message-ID,
and (3) normalized subject plus overlapping participants. Thread ids are derived from the
earliest member so they stay stable across runs as long as that member stays the earliest.
"""
from __future__ import annotations

import hashlib
import re
from collections import defaultdict

from .lang import normalize_width
from .store import Store

PREFIX = re.compile(r"^\s*((re|fw|fwd|aw|wg|sv|回覆|回复|答覆|答复|轉寄|转发|轉送)\s*(\[\d+\])?\s*[:：]\s*)+", re.I)
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")


def normalize_subject(subject: str | None) -> str:
    s = normalize_width(subject or "")
    s = PREFIX.sub("", s)
    s = re.sub(r"^\s*\[(external|外部)\]\s*", "", s, flags=re.I)
    s = PREFIX.sub("", s)
    return re.sub(r"\s+", " ", s).strip().lower()


def participants(row) -> set[str]:
    out = set()
    for col in ("sender_email", "sender", "to_addrs", "cc_addrs"):
        out.update(a.lower() for a in EMAIL_RE.findall(row[col] or ""))
    return out


class UnionFind:
    def __init__(self):
        self.parent: dict[int, int] = {}

    def find(self, x: int) -> int:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def build_threads(store: Store) -> int:
    rows = store.q("SELECT email_id, message_id, in_reply_to, conversation_index, subject, sender, "
                   "sender_email, to_addrs, cc_addrs, sent_at, language FROM emails ORDER BY email_id")
    uf = UnionFind()
    by_conv: dict[str, int] = {}
    by_msgid: dict[str, int] = {}
    for r in rows:
        uf.find(r["email_id"])
        ci = r["conversation_index"]
        if ci and len(ci) >= 44:
            key = ci[:44]  # 22-byte header, hex
            if key in by_conv:
                uf.union(by_conv[key], r["email_id"])
            else:
                by_conv[key] = r["email_id"]
        if r["message_id"]:
            by_msgid.setdefault(r["message_id"].strip("<> ").lower(), r["email_id"])
    for r in rows:
        irt = (r["in_reply_to"] or "").strip("<> ").lower()
        if irt and irt in by_msgid:
            uf.union(by_msgid[irt], r["email_id"])

    # subject + participant overlap fallback
    by_subject: dict[str, list] = defaultdict(list)
    for r in rows:
        ns = normalize_subject(r["subject"])
        if len(ns) >= 4:
            by_subject[ns].append(r)
    for group in by_subject.values():
        for i, a in enumerate(group):
            pa = participants(a)
            for b in group[i + 1:]:
                if pa & participants(b):
                    uf.union(a["email_id"], b["email_id"])

    members: dict[int, list] = defaultdict(list)
    for r in rows:
        members[uf.find(r["email_id"])].append(r)

    old = {r["thread_id"]: r for r in store.q("SELECT * FROM threads")}
    new_ids = set()
    for group in members.values():
        group.sort(key=lambda r: (r["sent_at"] or "", r["email_id"]))
        root = group[0]
        anchor = root["message_id"] or f"{root['subject']}|{root['sent_at']}|{root['email_id']}"
        thread_id = hashlib.sha1(anchor.encode()).hexdigest()[:12]
        new_ids.add(thread_id)
        ids = sorted(r["email_id"] for r in group)
        members_hash = hashlib.sha1(",".join(map(str, ids)).encode()).hexdigest()[:12]
        langs = {r["language"] for r in group}
        lang = "en" if langs == {"en"} else ("zh" if langs <= {"zh", "unknown"} else "mixed")
        dates = [r["sent_at"] for r in group if r["sent_at"]]
        row = {
            "thread_id": thread_id, "topic": root["subject"], "email_count": len(group),
            "first_at": min(dates) if dates else None, "last_at": max(dates) if dates else None,
            "members_hash": members_hash, "language": lang,
        }
        prev = old.get(thread_id)
        if prev is not None:
            store.update("threads", "thread_id", thread_id, **{k: v for k, v in row.items() if k != "thread_id"})
        else:
            store.insert("threads", row)
        store.db.executemany("UPDATE emails SET thread_id=? WHERE email_id=?", [(thread_id, i) for i in ids])
    stale = set(old) - new_ids
    for tid in stale:
        store.db.execute("DELETE FROM threads WHERE thread_id=?", (tid,))
    store.commit()
    return len(members)
