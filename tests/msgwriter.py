"""Minimal Outlook .msg writer for test fixtures (CFB v3 + MS-OXMSG property streams).

Only what the tests need: string/binary properties, transport headers, attachments by value,
and embedded messages. Not a general-purpose writer.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field

SECTOR = 512
MINI = 64
CUTOFF = 4096
FREESECT, ENDOFCHAIN, FATSECT, NOSTREAM = 0xFFFFFFFF, 0xFFFFFFFE, 0xFFFFFFFD, 0xFFFFFFFF


# --------------------------------------------------------------------------- CFB

@dataclass
class _Entry:
    name: str
    kind: int                 # 1 storage, 2 stream, 5 root
    data: bytes = b""
    children: list["_Entry"] = field(default_factory=list)
    left: int = NOSTREAM
    right: int = NOSTREAM
    child: int = NOSTREAM
    start: int = ENDOFCHAIN
    index: int = 0


def _tree(d: dict, name: str, kind: int) -> _Entry:
    e = _Entry(name, kind)
    for k, v in d.items():
        e.children.append(_tree(v, k, 1) if isinstance(v, dict) else _Entry(k, 2, data=v))
    return e


def _key(e: _Entry):
    return (len(e.name), e.name.upper())


def _bst(nodes: list[_Entry]) -> int:
    if not nodes:
        return NOSTREAM
    mid = len(nodes) // 2
    n = nodes[mid]
    n.left = _bst(nodes[:mid])
    n.right = _bst(nodes[mid + 1:])
    return n.index


def build_cfb(tree: dict) -> bytes:
    root = _tree(tree, "Root Entry", 5)
    entries: list[_Entry] = []
    queue = [root]
    while queue:
        e = queue.pop(0)
        e.index = len(entries)
        entries.append(e)
        queue.extend(e.children)
    for e in entries:
        if e.children:
            e.child = _bst(sorted(e.children, key=_key))

    # mini stream
    mini = bytearray()
    minifat: list[int] = []
    big: list[_Entry] = []
    for e in entries:
        if e.kind != 2 or not e.data:
            continue
        if len(e.data) < CUTOFF:
            start = len(mini) // MINI
            n = -(-len(e.data) // MINI)
            mini += e.data + b"\0" * (n * MINI - len(e.data))
            minifat += [start + i + 1 for i in range(n - 1)] + [ENDOFCHAIN]
            e.start = start
        else:
            big.append(e)

    sectors: list[bytes] = []
    fat: list[int] = []

    def alloc(data: bytes, marker: int | None = None) -> int:
        n = max(1, -(-len(data) // SECTOR))
        start = len(sectors)
        for i in range(n):
            sectors.append(data[i * SECTOR:(i + 1) * SECTOR].ljust(SECTOR, b"\0"))
            fat.append(marker if marker is not None else (start + i + 1 if i < n - 1 else ENDOFCHAIN))
        return start

    root.start = alloc(bytes(mini)) if mini else ENDOFCHAIN
    root.data = bytes(mini)
    for e in big:
        e.start = alloc(e.data)
    minifat_start = alloc(b"".join(struct.pack("<I", x) for x in minifat)) if minifat else ENDOFCHAIN
    n_minifat = -(-len(minifat) * 4 // SECTOR) if minifat else 0

    dir_bytes = bytearray()
    for e in entries:
        name = e.name.encode("utf-16-le") + b"\0\0"
        size = len(e.data) if e.kind in (2, 5) else 0
        dir_bytes += struct.pack("<64sHBBIII16sIQQIQ", name.ljust(64, b"\0"), len(name), e.kind, 1,
                                 e.left, e.right, e.child, b"\0" * 16, 0, 0, 0,
                                 0 if e.kind == 1 else (e.start if size else ENDOFCHAIN), size)
    while len(dir_bytes) % SECTOR:
        dir_bytes += struct.pack("<64sHBBIII16sIQQIQ", b"\0" * 64, 0, 0, 0, NOSTREAM, NOSTREAM, NOSTREAM,
                                 b"\0" * 16, 0, 0, 0, 0, 0)
    dir_start = alloc(bytes(dir_bytes))

    n_fat = 1
    while (len(sectors) + n_fat) > n_fat * (SECTOR // 4):
        n_fat += 1
    assert n_fat <= 109, "fixture too large"
    fat_start = len(sectors)
    fat += [FATSECT] * n_fat
    fat += [FREESECT] * (n_fat * (SECTOR // 4) - len(fat))
    fat_bytes = b"".join(struct.pack("<I", x) for x in fat)
    for i in range(n_fat):
        sectors.append(fat_bytes[i * SECTOR:(i + 1) * SECTOR])

    difat = [fat_start + i for i in range(n_fat)] + [FREESECT] * (109 - n_fat)
    header = struct.pack(
        "<8s16sHHHHH6sIIIIIIIII",
        bytes.fromhex("D0CF11E0A1B11AE1"), b"\0" * 16, 0x3E, 3, 0xFFFE, 9, 6, b"\0" * 6,
        0, n_fat, dir_start, 0, CUTOFF, minifat_start, n_minifat, ENDOFCHAIN, 0,
    ) + b"".join(struct.pack("<I", x) for x in difat)
    return header.ljust(SECTOR, b"\0") + b"".join(sectors)


# --------------------------------------------------------------------------- MSG

PT_LONG, PT_UNICODE, PT_BINARY, PT_OBJECT = 0x0003, 0x001F, 0x0102, 0x000D


def _props_stream(header: bytes, fixed: dict[int, int], variable: dict[int, bytes | str]) -> bytes:
    out = bytearray(header)
    for tag, val in fixed.items():
        if tag <= 0xFFFF:  # bare property id: PT_LONG
            tag = (tag << 16) | PT_LONG
        out += struct.pack("<IIq", tag, 0x6, val)
    for tag, val in variable.items():
        ptype = tag & 0xFFFF
        size = len(val.encode("utf-16-le")) + 2 if isinstance(val, str) else len(val)
        out += struct.pack("<IIII", tag, 0x6, size, 0)
    return bytes(out)


def _streams(props: dict[int, bytes | str]) -> dict:
    out = {}
    for tag, val in props.items():
        name = f"__substg1.0_{tag:08X}"
        out[name] = val.encode("utf-16-le") if isinstance(val, str) else val
    return out


@dataclass
class Attachment:
    filename: str
    data: bytes | None = None
    mime: str | None = None
    embedded: "Message | None" = None


@dataclass
class Message:
    subject: str
    sender_name: str
    sender_email: str
    to: str
    date: str                       # RFC 2822
    body: str | None = None
    html: str | None = None
    cc: str | None = None
    message_id: str | None = None
    in_reply_to: str | None = None
    conversation_index: bytes | None = None
    conversation_topic: str | None = None
    attachments: list[Attachment] = field(default_factory=list)

    def headers(self) -> str:
        h = [f"From: {self.sender_name} <{self.sender_email}>", f"To: {self.to}"]
        if self.cc:
            h.append(f"Cc: {self.cc}")
        h += [f"Subject: {self.subject}", f"Date: {self.date}"]
        if self.message_id:
            h.append(f"Message-ID: {self.message_id}")
        if self.in_reply_to:
            h.append(f"In-Reply-To: {self.in_reply_to}")
        h.append("MIME-Version: 1.0")
        return "\r\n".join(h) + "\r\n\r\n"

    def tree(self, embedded: bool = False) -> dict:
        var: dict[int, bytes | str] = {
            0x001A001F: "IPM.Note",
            0x0037001F: self.subject,
            0x0C1A001F: self.sender_name,
            0x0C1F001F: self.sender_email,
            0x0E04001F: self.to,
            0x007D001F: self.headers(),
            0x0070001F: self.conversation_topic or self.subject,
        }
        if self.cc:
            var[0x0E03001F] = self.cc
        if self.body is not None:
            var[0x1000001F] = self.body
        if self.html is not None:
            var[0x10130102] = self.html.encode("utf-8")
        if self.message_id:
            var[0x1035001F] = self.message_id
        if self.in_reply_to:
            var[0x1042001F] = self.in_reply_to
        if self.conversation_index:
            var[0x00710102] = self.conversation_index
        from email.utils import parsedate_to_datetime

        dt = parsedate_to_datetime(self.date)
        filetime = int((dt.timestamp() + 11644473600) * 10_000_000)
        fixed = {0x00390040: filetime, 0x0E060040: filetime, 0x0E07: 1}  # submit + delivery time, MSGFLAG_READ
        n = len(self.attachments)
        hdr = (b"\0" * 8 + struct.pack("<IIII", 0, n, 0, n)) + (b"" if embedded else b"\0" * 8)
        t: dict = {"__properties_version1.0": _props_stream(hdr, fixed, var), **_streams(var)}
        if not embedded:
            t["__nameid_version1.0"] = {"__substg1.0_00020102": b"", "__substg1.0_00030102": b"",
                                        "__substg1.0_00040102": b""}
        for i, a in enumerate(self.attachments):
            avar: dict[int, bytes | str] = {0x3707001F: a.filename, 0x3704001F: a.filename[:8],
                                            0x3001001F: a.filename}
            if a.mime:
                avar[0x370E001F] = a.mime
            if a.embedded is not None:
                fixed = {0x3705: 5}
                at = {"__properties_version1.0": _props_stream(b"\0" * 8, fixed, avar), **_streams(avar),
                      "__substg1.0_3701000D": a.embedded.tree(embedded=True)}
            else:
                avar[0x37010102] = a.data or b""
                fixed = {0x3705: 1}
                at = {"__properties_version1.0": _props_stream(b"\0" * 8, fixed, avar), **_streams(avar)}
            t[f"__attach_version1.0_#{i:08X}"] = at
        return t

    def to_bytes(self) -> bytes:
        return build_cfb(self.tree())
