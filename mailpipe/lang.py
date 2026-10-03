"""Language detection, token estimates, and zh-TW normalization."""
from __future__ import annotations

import unicodedata
from functools import lru_cache

import regex

CJK = regex.compile(r"\p{Han}")
LETTER = regex.compile(r"[\p{Han}\p{Latin}]")


def cjk_ratio(text: str) -> float:
    letters = LETTER.findall(text or "")
    if not letters:
        return 0.0
    return sum(1 for c in letters if CJK.match(c)) / len(letters)


def estimate_tokens(text: str) -> int:
    """Conservative token estimate for Qwen/Llama-style tokenizers.

    Han characters are counted as one token each; everything else at ~3.5 chars per token.
    Over-estimating is the safe direction: it keeps prompts inside num_ctx.
    """
    if not text:
        return 0
    han = len(CJK.findall(text))
    return han + int((len(text) - han) / 3.5) + 1


def trim_to_tokens(text: str, max_tokens: int) -> str:
    if estimate_tokens(text) <= max_tokens:
        return text
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if estimate_tokens(text[:mid]) <= max_tokens:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo].rstrip() + "\n[...truncated]"


@lru_cache(maxsize=1)
def _detector():
    from lingua import Language, LanguageDetectorBuilder

    return LanguageDetectorBuilder.from_languages(
        Language.CHINESE, Language.ENGLISH, Language.JAPANESE
    ).build()


def detect_language(text: str) -> str:
    """Return 'zh', 'en', 'mixed', 'ja', or 'unknown'."""
    text = (text or "").strip()
    if not text:
        return "unknown"
    ratio = cjk_ratio(text)
    if ratio >= 0.85:
        lang = _detector().detect_language_of(text[:2000])
        return "ja" if lang is not None and lang.name == "JAPANESE" else "zh"
    if ratio <= 0.02:
        lang = _detector().detect_language_of(text[:2000])
        return "en" if lang is None or lang.name == "ENGLISH" else lang.name.lower()[:2]
    return "mixed"


@lru_cache(maxsize=4)
def _opencc(profile: str):
    from opencc import OpenCC

    return OpenCC(profile)


def to_zh_tw(text: str, profile: str = "s2twp") -> str:
    if not text or not CJK.search(text):
        return text
    return _opencc(profile).convert(text)


def has_simplified(text: str, profile: str = "s2twp") -> bool:
    """True if OpenCC would still change the text, i.e. it is not clean zh-TW."""
    return bool(text) and to_zh_tw(text, profile) != text


def normalize_width(text: str) -> str:
    """Full-width ASCII and spaces to half-width (e.g. '：' to ':'), leaving Han untouched."""
    out = []
    for ch in text:
        code = ord(ch)
        if ch == "　":
            out.append(" ")
        elif 0xFF01 <= code <= 0xFF5E:
            out.append(chr(code - 0xFEE0))
        else:
            out.append(ch)
    return unicodedata.normalize("NFC", "".join(out))
