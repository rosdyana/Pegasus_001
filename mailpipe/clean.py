"""Stage 3a: strip quoted replies, signatures, and disclaimers."""
from __future__ import annotations

import re
from collections import Counter, defaultdict

from .lang import normalize_width

# Lines that open a quoted earlier message. Matched after full-width normalization.
SEPARATORS = [
    re.compile(r"^\s*-{2,}\s*(Original Message|原始郵件|原始邮件|Forwarded message|轉寄郵件)\s*-{2,}", re.I),
    re.compile(r"^\s*_{10,}\s*$"),                              # Outlook's underscore rule before the header
    re.compile(r"^\s*On .{5,200}wrote:\s*$", re.I),
    re.compile(r"^.{2,200}(寫道|写道)\s*:\s*$"),
    re.compile(r"^\s*>"),
]
HEADER_START = re.compile(r"^\s*\**\s*(From|寄件者|寄件人|发件人|發件人)\s*\**\s*:", re.I)
HEADER_FOLLOW = re.compile(
    r"^\s*\**\s*(Sent|Date|To|Cc|Subject|寄件日期|傳送時間|日期|收件者|收件人|副本|主旨|主題|主题|发送时间)\s*\**\s*:", re.I)
SIG_DELIM = re.compile(r"^--\s*$")


def strip_quoted(body: str) -> str:
    lines = body.split("\n")
    norm = [normalize_width(ln) for ln in lines]
    for i, ln in enumerate(norm):
        if i == 0:
            # a body that starts with a header block is a bare forward: keep it
            continue
        if any(p.search(ln) for p in SEPARATORS):
            if ln.lstrip().startswith(">") and not all(x.lstrip().startswith(">") or not x.strip() for x in norm[i:]):
                continue  # inline '>' quote followed by new text: keep
            return "\n".join(lines[:i]).rstrip()
        if HEADER_START.search(ln):
            follow = [x for x in norm[i + 1:i + 6] if x.strip()]
            if sum(1 for x in follow if HEADER_FOLLOW.search(x)) >= 2:
                return "\n".join(lines[:i]).rstrip()
    return body.rstrip()


def strip_signature_delim(body: str) -> str:
    lines = body.split("\n")
    for i in range(len(lines) - 1, -1, -1):
        if SIG_DELIM.match(lines[i]) and len(lines) - i <= 15:
            return "\n".join(lines[:i]).rstrip()
    return body


def paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def _norm_para(p: str) -> str:
    return re.sub(r"\s+", " ", normalize_width(p)).strip().lower()


class Boilerplate:
    """Trailing paragraphs repeated across emails (signatures, disclaimers)."""

    def __init__(self, global_min: int, sender_min: int, disclaimers: list[str]):
        self.global_min = global_min
        self.sender_min = sender_min
        self.disclaimers = {_norm_para(d) for d in disclaimers if d.strip()}
        self.global_counts: Counter[str] = Counter()
        self.sender_counts: dict[str, Counter[str]] = defaultdict(Counter)

    def observe(self, sender: str | None, body: str, tail: int = 6) -> None:
        paras = {_norm_para(p) for p in paragraphs(body)[-tail:]}
        self.global_counts.update(paras)
        if sender:
            self.sender_counts[sender].update(paras)

    def is_boilerplate(self, sender: str | None, para: str) -> bool:
        n = _norm_para(para)
        if not n:
            return True
        if n in self.disclaimers:
            return True
        if self.global_counts[n] >= self.global_min:
            return True
        return bool(sender) and self.sender_counts[sender][n] >= self.sender_min

    def strip(self, sender: str | None, body: str) -> str:
        paras = paragraphs(body)
        kept = list(paras)
        while len(kept) > 1 and self.is_boilerplate(sender, kept[-1]):
            kept.pop()
        # exact-match disclaimers anywhere
        kept = [p for p in kept if _norm_para(p) not in self.disclaimers]
        if len(kept) == len(paras):
            return body.strip()
        return "\n\n".join(kept)


def clean_body(body: str, sender: str | None, bp: Boilerplate) -> str:
    text = strip_quoted(body or "")
    text = strip_signature_delim(text)
    text = bp.strip(sender, text)
    return text.strip()


def first_pass(body: str) -> str:
    """Quote + delimiter stripping only, used to collect boilerplate statistics."""
    return strip_signature_delim(strip_quoted(body or ""))
