"""Stage 4: entity extraction. Regex and dates on CPU (pass 1), GLiNER on GPU (pass 2)."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta

import regex as re2

from .lang import normalize_width

log = logging.getLogger(__name__)

ROC_DATE = re.compile(r"(?<![\d/])(1[01]\d)/(\d{1,2})/(\d{1,2})(?![\d/])")
ISO_DATE = re.compile(r"(?<!\d)(20\d{2})[/.-](\d{1,2})[/.-](\d{1,2})(?!\d)")
ZH_DATE = re.compile(r"(?:(\d{2,4})\s*年\s*)?(\d{1,2})\s*月\s*(\d{1,2})\s*[日號号]")
WEEKDAYS_ZH = "一二三四五六日天"
REL_ZH = re.compile(r"(下下|下|這|这|本)?\s*(週|周|星期|禮拜|礼拜)\s*([一二三四五六日天])")
REL_WORDS_ZH = {"今天": 0, "明天": 1, "後天": 2, "后天": 2}
REL_EN = re.compile(r"\b(?:(next|this)\s+)?(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b|\b(tomorrow|today)\b",
                    re.I)
EN_DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
EN_MONTH_DATE = re.compile(
    r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+20\d{2})?\b"
    r"|\b\d{1,2}(?:st|nd|rd|th)?\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?(?:,?\s+20\d{2})?\b",
    re.I)


@dataclass
class Entity:
    type: str
    value: str
    normalized: str | None
    confidence: float
    method: str
    start: int = -1
    end: int = -1

    def context(self, text: str, width: int = 40) -> str:
        if self.start < 0:
            return ""
        a, b = max(0, self.start - width), min(len(text), self.end + width)
        return text[a:b].replace("\n", " ")


def _safe_date(y: int, m: int, d: int) -> date | None:
    try:
        return date(y, m, d)
    except ValueError:
        return None


def _base_date(sent_at: str | None) -> date:
    if sent_at:
        try:
            return datetime.fromisoformat(sent_at).date()
        except ValueError:
            pass
    return date.today()


def extract_regex(text: str, patterns: dict[str, str]) -> list[Entity]:
    """Patterns run in config order; a span already matched (e.g. a PO number) is not re-tagged."""
    out = []
    taken: list[tuple[int, int]] = []
    for name, pat in patterns.items():
        try:
            rx = re2.compile(pat, re2.I if name in ("email",) else 0)
        except re2.error as e:
            log.warning("bad regex %s: %s", name, e)
            continue
        for m in rx.finditer(text):
            if any(a <= m.start() and m.end() <= b for a, b in taken):
                continue
            taken.append((m.start(), m.end()))
            value = m.group(0).strip()
            out.append(Entity(name, value, _normalize(name, value), 1.0, "regex", m.start(), m.end()))
    return out


def _normalize(kind: str, value: str) -> str:
    if kind == "po_number":
        return "PO" + re.sub(r"\D", "", value)
    if kind == "amount":
        cur = "USD" if re.search(r"USD|US\$", value) else "TWD"
        num = re.sub(r"[^\d.]", "", value)
        return f"{cur} {num}"
    if kind in ("email",):
        return value.lower()
    if kind == "phone":
        return re.sub(r"[^\d+#]", "", value)
    return value.upper() if kind in ("part_number", "invoice_number") else value


def extract_dates(text: str, sent_at: str | None) -> list[Entity]:
    """Absolute and relative dates, resolved against the email date. Returns ISO dates."""
    base = _base_date(sent_at)
    out: list[Entity] = []
    taken: list[tuple[int, int]] = []

    def add(m, d: date | None, conf: float, method: str):
        if d is None:
            return
        if any(a < m.end() and m.start() < b for a, b in taken):
            return
        taken.append((m.start(), m.end()))
        out.append(Entity("date", m.group(0), d.isoformat(), conf, method, m.start(), m.end()))

    for m in ISO_DATE.finditer(text):
        add(m, _safe_date(int(m.group(1)), int(m.group(2)), int(m.group(3))), 1.0, "regex")
    for m in ROC_DATE.finditer(text):
        add(m, _safe_date(int(m.group(1)) + 1911, int(m.group(2)), int(m.group(3))), 0.9, "regex")
    for m in ZH_DATE.finditer(text):
        y = m.group(1)
        if y:
            year = int(y) + 1911 if len(y) <= 3 else int(y)
        else:
            year = base.year
            cand = _safe_date(year, int(m.group(2)), int(m.group(3)))
            # "1月5日" written in December means next January
            if cand and (base - cand).days > 180:
                year += 1
        add(m, _safe_date(year, int(m.group(2)), int(m.group(3))), 0.9, "regex")
    for m in REL_ZH.finditer(text):
        which, day = m.group(1) or "", m.group(3)
        wd = WEEKDAYS_ZH.index(day)
        wd = 6 if wd >= 6 else wd
        monday = base - timedelta(days=base.weekday())
        weeks = 2 if which == "下下" else (1 if which == "下" else 0)
        add(m, monday + timedelta(weeks=weeks, days=wd), 0.7, "relative")
    for word, delta in REL_WORDS_ZH.items():
        for m in re.finditer(word, text):
            add(m, base + timedelta(days=delta), 0.7, "relative")
    for m in REL_EN.finditer(text):
        if m.group(3):
            add(m, base + timedelta(days=1 if m.group(3).lower() == "tomorrow" else 0), 0.7, "relative")
            continue
        wd = EN_DAYS.index(m.group(2).lower())
        if m.group(1) and m.group(1).lower() == "next":
            d = base - timedelta(days=base.weekday()) + timedelta(weeks=1, days=wd)
        else:
            ahead = (wd - base.weekday()) % 7
            d = base + timedelta(days=ahead)
        add(m, d, 0.6, "relative")
    out.extend(_dateparser_dates(text, base, taken))
    return out


def _dateparser_dates(text: str, base: date, taken: list[tuple[int, int]]) -> list[Entity]:
    """English month-name dates via dateparser (it is noisy on free text, so only on matched spans)."""
    import dateparser

    out = []
    for m in EN_MONTH_DATE.finditer(text):
        if any(a < m.end() and m.start() < b for a, b in taken):
            continue
        d = dateparser.parse(m.group(0), languages=["en"], settings={
            "RELATIVE_BASE": datetime.combine(base, datetime.min.time()), "PREFER_DATES_FROM": "future"})
        if d:
            taken.append((m.start(), m.end()))
            out.append(Entity("date", m.group(0), d.date().isoformat(), 0.9, "dateparser", m.start(), m.end()))
    return out


def extract_cpu(text: str, sent_at: str | None, patterns: dict[str, str]) -> list[Entity]:
    text = normalize_width(text or "")
    ents = extract_regex(text, patterns) + extract_dates(text, sent_at)
    # dedupe on (type, normalized)
    seen, out = set(), []
    for e in ents:
        k = (e.type, e.normalized or e.value)
        if k not in seen:
            seen.add(k)
            out.append(e)
    return out


# --------------------------------------------------------------------------- GLiNER

CJK_RUN = re2.compile(r"\p{Han}+")
WORD = re2.compile(r"\p{Han}+|[\w][\w\-_./]*|\S", re2.UNICODE)


class MixedWordSplitter:
    """Word splitter for mixed zh-TW/English: jieba for Han runs, \\w+ for the rest.

    GLiNER's default whitespace splitter treats a whole Chinese sentence as one word.
    """

    def __init__(self):
        try:
            import jieba

            jieba.setLogLevel(logging.WARNING)
            self._jieba = jieba
        except ImportError:
            self._jieba = None

    def __call__(self, text: str):
        for m in WORD.finditer(text):
            tok = m.group(0)
            if CJK_RUN.fullmatch(tok):
                if self._jieba is not None:
                    for w, a, b in self._jieba.tokenize(tok):
                        yield w, m.start() + a, m.start() + b
                else:
                    for i, ch in enumerate(tok):
                        yield ch, m.start() + i, m.start() + i + 1
            else:
                yield tok, m.start(), m.end()


def windows(text: str, splitter: MixedWordSplitter, size: int, overlap: int) -> list[tuple[int, str]]:
    """Split text into (char_offset, window_text) pieces of at most `size` words."""
    spans = [(a, b) for _, a, b in splitter(text)]
    if len(spans) <= size:
        return [(0, text)] if text.strip() else []
    out, i = [], 0
    step = max(1, size - overlap)
    while i < len(spans):
        a = spans[i][0]
        b = spans[min(i + size, len(spans)) - 1][1]
        out.append((a, text[a:b]))
        if i + size >= len(spans):
            break
        i += step
    return out


def gliner_labels(cfg) -> tuple[list[str], dict[str, str]]:
    labels = list(cfg.labels)
    mapping = {lbl: lbl for lbl in labels}
    if cfg.use_label_aliases:
        for alias, target in cfg.label_aliases.items():
            labels.append(alias)
            mapping[alias] = target
    return labels, mapping


class GlinerRunner:
    """Loads GLiNER inside a GPU phase and runs windowed, length-sorted batches with the OOM ladder."""

    STRUCTURED = ("po_number", "invoice_number", "amount", "email", "phone")

    def __init__(self, cfg, phase):
        self.cfg = cfg.gliner
        # regex owns structured fields; GLiNER often labels a PO number as a product model
        self.structured = [re2.compile(p) for k, p in cfg.regex.items() if k in self.STRUCTURED]
        self.phase = phase
        self.labels, self.label_map = gliner_labels(self.cfg)
        self.splitter = MixedWordSplitter()
        self.batch_size = self.cfg.batch_size
        self.events: list[str] = []
        self.model = None
        self.device = "cpu"
        self._load(phase.device)
        phase.on_exit(self.unload)

    def _load(self, want: str) -> None:
        import torch
        from gliner import GLiNER

        self.device = "cuda" if want == "cuda" and torch.cuda.is_available() else "cpu"
        model = GLiNER.from_pretrained(self.cfg.model, map_location=self.device)
        model.data_processor.words_splitter = self.splitter
        model = model.to(self.device)
        if self.cfg.fp16 and self.device == "cuda":
            model = model.half()
        model.eval()
        self.model = model

    def unload(self) -> None:
        from .gpu import free_torch_memory

        self.model = None
        free_torch_memory()

    def _predict(self, texts: list[str]) -> list[list[dict]]:
        import torch

        with torch.inference_mode():
            return self.model.inference(texts, self.labels, threshold=self.cfg.threshold,
                                        batch_size=self.batch_size, flat_ner=True)

    def predict_batch(self, texts: list[str]) -> list[list[dict]]:
        """OOM ladder: free + retry, halve batch, CPU, then give up (caller marks failed_oom)."""
        from .gpu import OOMError, free_torch_memory, is_oom

        retried = False
        while True:
            try:
                return self._predict(texts)
            except Exception as e:
                if not is_oom(e):
                    raise
                free_torch_memory()
                if not retried:
                    retried = True
                    self.events.append("OOM: freed cache and retried")
                elif self.batch_size > 1:
                    self.batch_size //= 2
                    self.events.append(f"OOM: batch size halved to {self.batch_size}")
                elif self.device == "cuda":
                    self.events.append("OOM at batch 1: moving GLiNER to CPU")
                    self.unload()
                    self._load("cpu")
                else:
                    raise OOMError(f"GLiNER OOM on CPU: {e}") from e

    def run(self, items: list[dict], on_item=None) -> dict:
        """items: {key, email_id, attachment_id, text, sent_at}. Calls on_item(item, entities)."""
        from .gpu import OOMError

        jobs = []  # (item_index, offset, window_text)
        for idx, it in enumerate(items):
            for off, w in windows(it["text"], self.splitter, self.cfg.window_words, self.cfg.window_overlap):
                jobs.append((idx, off, w))
        jobs.sort(key=lambda j: len(j[2]))  # similar lengths together: less padding
        found: dict[int, list[Entity]] = {i: [] for i in range(len(items))}
        failed: set[int] = set()
        step = max(self.batch_size * 4, 16)
        for i in range(0, len(jobs), step):
            group = jobs[i:i + step]
            try:
                preds = self.predict_batch([j[2] for j in group])
            except OOMError as e:
                failed.update(j[0] for j in group)
                self.events.append(str(e)[:200])
                continue
            for (idx, off, _), ents in zip(group, preds):
                for p in ents:
                    found[idx].append(Entity(self.label_map.get(p["label"], p["label"]), p["text"], None,
                                             float(p["score"]), "gliner", off + p["start"], off + p["end"]))
        stats = {"items": len(items), "windows": len(jobs), "failed_oom": len(failed), "entities": 0}
        for idx, it in enumerate(items):
            if idx in failed:
                if on_item:
                    on_item(it, None)
                continue
            seen, ents = set(), []
            for e in sorted(found[idx], key=lambda e: -e.confidence):
                k = (e.type, e.value.strip().lower())
                if k in seen or not e.value.strip():
                    continue
                if any(rx.fullmatch(e.value.strip()) for rx in self.structured):
                    continue
                seen.add(k)
                if e.type == "deadline":
                    ds = extract_dates(e.value, it.get("sent_at"))
                    e.normalized = ds[0].normalized if ds else None
                ents.append(e)
            stats["entities"] += len(ents)
            if on_item:
                on_item(it, ents)
        stats["events"] = self.events
        return stats
