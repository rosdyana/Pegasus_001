"""Stage 2: .msg parsing with extract-msg (metadata, body, attachments, nested messages)."""
from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from email.utils import getaddresses, parseaddr, parsedate_to_datetime
from pathlib import Path

import chardet
import extract_msg
from bs4 import BeautifulSoup
from extract_msg.enums import AttachmentType

log = logging.getLogger(__name__)

PR_CONVERSATION_INDEX = "__substg1.0_00710102"
PR_CONVERSATION_TOPIC = "__substg1.0_0070"
BLOCKS = ["p", "div", "li", "tr", "table", "h1", "h2", "h3", "h4", "h5", "h6", "blockquote", "pre", "hr"]


@dataclass
class ParsedAttachment:
    filename: str
    mime: str | None
    kind: str                       # docx | xlsx | msg | pdf | image | other
    data: bytes | None = None
    sha256: str | None = None
    size: int = 0
    inline: bool = False
    nested: "ParsedEmail | None" = None
    error: str | None = None


@dataclass
class ParsedEmail:
    subject: str | None = None
    sender: str | None = None
    sender_email: str | None = None
    to_addrs: str | None = None
    cc_addrs: str | None = None
    sent_at: str | None = None
    message_id: str | None = None
    in_reply_to: str | None = None
    conversation_index: str | None = None
    conversation_topic: str | None = None
    body: str = ""
    body_source: str = "none"
    attachments: list[ParsedAttachment] = field(default_factory=list)


def html_to_text(html: bytes | str) -> str:
    if isinstance(html, bytes):
        html = decode_bytes(html, prefer_meta=True)
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["style", "script", "head"]):
        tag.decompose()
    for br in soup.find_all("br"):
        br.replace_with("\n")
    # newlines only around block elements, so inline tags (<b>, <span>) do not split sentences
    for el in soup.find_all(BLOCKS):
        el.insert_before("\n")
        el.insert_after("\n")
    for cell in soup.find_all(["td", "th"]):
        cell.insert_after(" | ")
    text = soup.get_text("")
    text = re.sub(r"[ \t\xa0]+", " ", text)
    return squeeze_blank_lines(text)


def decode_bytes(data: bytes, prefer_meta: bool = False) -> str:
    if prefer_meta:
        m = re.search(rb"charset=[\"']?([\w-]+)", data[:4096], re.I)
        if m:
            try:
                return data.decode(m.group(1).decode("ascii"), errors="strict")
            except (LookupError, UnicodeDecodeError):
                pass
    for enc in ("utf-8",):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            pass
    guess = chardet.detect(data[:200_000]).get("encoding") or "big5"
    if guess.lower() in ("big5", "gb2312", "gbk"):
        guess = "big5hkscs" if guess.lower() == "big5" else "gb18030"
    return data.decode(guess, errors="replace")


def squeeze_blank_lines(text: str) -> str:
    lines = [ln.rstrip() for ln in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    out, blank = [], 0
    for ln in lines:
        if ln.strip():
            out.append(ln)
            blank = 0
        else:
            blank += 1
            if blank == 1:
                out.append("")
    return "\n".join(out).strip()


def _addr_list(value: str | None) -> str | None:
    if not value:
        return None
    pairs = getaddresses([value.replace(";", ",")])
    parts = [f"{n} <{a}>" if n else a for n, a in pairs if a or n]
    return "; ".join(parts) or None


def _get(msg, attr: str):
    try:
        v = getattr(msg, attr, None)
        return v.rstrip("\x00") if isinstance(v, str) else v
    except Exception as e:  # extract-msg raises on some malformed properties
        log.debug("property %s unreadable: %s", attr, e)
        return None


def _body(msg) -> tuple[str, str]:
    plain = _get(msg, "body")
    if isinstance(plain, bytes):
        plain = decode_bytes(plain)
    if plain and plain.strip():
        return squeeze_blank_lines(plain), "plain"
    html = _get(msg, "htmlBody")
    if html:
        text = html_to_text(html)
        if text:
            return text, "html"
    try:
        rtf = msg.deencapsulatedRtf
        if rtf is not None:
            if rtf.content_type == "html":
                return html_to_text(rtf.html), "rtf"
            if rtf.content_type == "text":
                text = rtf.text.decode(errors="replace") if isinstance(rtf.text, bytes) else rtf.text
                return squeeze_blank_lines(text), "rtf"
    except Exception as e:
        log.debug("rtf body unreadable: %s", e)
    return "", "none"


def _kind(filename: str, mime: str | None) -> str:
    ext = Path(filename).suffix.lower()
    if ext == ".docx":
        return "docx"
    if ext in (".xlsx", ".xlsm"):
        return "xlsx"
    if ext == ".msg":
        return "msg"
    if ext == ".pdf":
        return "pdf"
    if (mime or "").startswith("image/") or ext in (".png", ".jpg", ".jpeg", ".gif", ".bmp"):
        return "image"
    return "other"


def _attachment(att) -> ParsedAttachment:
    filename = (_get(att, "longFilename") or _get(att, "shortFilename") or _get(att, "name") or "unnamed")
    mime = _get(att, "mimetype")
    try:
        atype = att.type
    except Exception:
        atype = AttachmentType.UNKNOWN
    if atype == AttachmentType.MSG:
        nested = att.data
        pa = ParsedAttachment(filename=filename if filename != "unnamed" else "embedded.msg", mime="application/vnd.ms-outlook",
                              kind="msg")
        try:
            pa.nested = parse_msg_obj(nested)
            try:
                raw = nested.exportBytes()
            except Exception:
                raw = f"{pa.nested.subject}|{pa.nested.sent_at}|{pa.nested.body}".encode()
            pa.sha256 = hashlib.sha256(raw).hexdigest()
            pa.size = len(raw)
        except Exception as e:
            pa.error = f"nested msg: {e}"
        return pa
    pa = ParsedAttachment(filename=filename, mime=mime, kind=_kind(filename, mime))
    if atype != AttachmentType.DATA:
        pa.error = f"unsupported attachment type {getattr(atype, 'name', atype)}"
        return pa
    data = att.data
    if isinstance(data, bytes):
        pa.data = data
        pa.size = len(data)
        pa.sha256 = hashlib.sha256(data).hexdigest()
    pa.inline = bool(_get(att, "hidden")) or (pa.kind == "image" and bool(_get(att, "contentId")))
    return pa


def parse_msg_obj(msg) -> ParsedEmail:
    e = ParsedEmail()
    e.subject = _get(msg, "subject")
    sender = _get(msg, "sender")
    if sender:
        name, addr = parseaddr(sender)
        e.sender = sender.strip()
        e.sender_email = (addr or "").lower() or None
        if not e.sender_email and "@" in (name or ""):
            e.sender_email = name.lower()
    e.to_addrs = _addr_list(_get(msg, "to"))
    e.cc_addrs = _addr_list(_get(msg, "cc"))
    dt = _get(msg, "date")
    if dt is None:
        # unsent/draft messages have no submit time; fall back to the transport header
        try:
            raw = msg.header.get("Date")
            dt = parsedate_to_datetime(raw) if raw else None
        except Exception:
            dt = None
    if dt is not None and hasattr(dt, "isoformat"):
        e.sent_at = dt.isoformat()
    e.message_id = (_get(msg, "messageId") or "").strip() or None
    e.in_reply_to = (_get(msg, "inReplyTo") or "").strip() or None
    try:
        ci = msg.getStream(PR_CONVERSATION_INDEX)
        e.conversation_index = ci.hex() if ci else None
    except Exception:
        pass
    try:
        e.conversation_topic = msg.getStringStream(PR_CONVERSATION_TOPIC)
    except Exception:
        pass
    e.body, e.body_source = _body(msg)
    for att in _get(msg, "attachments") or []:
        try:
            e.attachments.append(_attachment(att))
        except Exception as ex:
            e.attachments.append(ParsedAttachment(filename="unreadable", mime=None, kind="other", error=str(ex)))
    return e


def parse_msg_file(path: Path) -> ParsedEmail:
    msg = extract_msg.openMsg(str(path))
    try:
        return parse_msg_obj(msg)
    finally:
        try:
            msg.close()
        except Exception:
            pass
