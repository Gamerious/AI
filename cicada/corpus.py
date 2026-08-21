"""Loader for the Liber Primus master transcription (rtkd/iddqd archive).

The archive stores the whole book as one delimited rune stream::

    Word : -   Clause : .   Paragraph : &   Segment : $   Line : /   Page : %

18 ``$`` segments, 74 ``%`` pages.  The book's own page numbering runs
-16..57; the archive's file order runs 0..73, so ``lp_number = index - 17``.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

from .gematria import RUNE_INDEX, to_indices, to_latin

ARCHIVE = os.environ.get("CICADA_ARCHIVE", "/home/user/rtkd/iddqd")

_MASTER = "liber-primus__transcription--master/liber-primus__transcription--master.txt"
_KEYS = "liber-primus__keys/liber-primus__keys.txt"
_TRANSLATION = "liber-primus__translation/liber-primus__translation.txt"
_INDEX = "liber-primus__index/liber-primus__index.txt"

WORD, CLAUSE, PARA, SEG, LINE, PAGE = "-", ".", "&", "$", "/", "%"


def _read(rel: str) -> str:
    with open(os.path.join(ARCHIVE, rel), encoding="utf-8") as fh:
        return fh.read()


def _runes_only(s: str) -> str:
    return "".join(c for c in s if c in RUNE_INDEX)


@dataclass
class Segment:
    """One ``$`` segment - the unit the community assigns keys to (0.0 .. 0.17)."""

    idx: int
    raw: str
    runes: str = field(init=False)
    key_note: str = ""

    def __post_init__(self) -> None:
        self.runes = _runes_only(self.raw)

    @property
    def name(self) -> str:
        return f"0.{self.idx}"

    @property
    def n(self) -> int:
        return len(self.runes)

    def words(self) -> list[str]:
        """Rune words in reading order (delimiters stripped)."""
        flat = re.sub(f"[{re.escape(LINE + PAGE + SEG)}\n]", "", self.raw)
        flat = re.sub(f"[{re.escape(CLAUSE + PARA)}]", WORD, flat)
        return [w for w in (_runes_only(p) for p in flat.split(WORD)) if w]

    def clauses(self) -> list[str]:
        flat = re.sub(f"[{re.escape(LINE + PAGE + SEG + WORD)}\n]", "", self.raw)
        flat = re.sub(f"[{re.escape(PARA)}]", CLAUSE, flat)
        return [c for c in (_runes_only(p) for p in flat.split(CLAUSE)) if c]

    @property
    def solved(self) -> bool:
        return bool(self.key_note) and self.key_note.strip() != "-"


@dataclass
class Page:
    idx: int          # 0..73, order in the archive file
    raw: str
    runes: str = field(init=False)
    segment: int = -1

    def __post_init__(self) -> None:
        self.runes = _runes_only(self.raw)

    @property
    def lp_number(self) -> int:
        """The book's own page number.

        The archive's index numbers the pages 1..74 and the book numbers them
        -16..57, so file position 0 is book page -16.
        """
        return self.idx - 16


class LiberPrimus:
    """The whole book, parsed."""

    def __init__(self, archive: str | None = None) -> None:
        global ARCHIVE
        if archive:
            ARCHIVE = archive
        text = _read(_MASTER)
        body = text.split("Page     : %", 1)[1]
        body = body.replace("§", "")          # single chapter marker, no structure

        self.keys = self._parse_keys()
        parts = body.split(SEG)
        self.segments = [
            Segment(i, raw, key_note=self.keys.get(f"0.{i}", ""))
            for i, raw in enumerate(parts)
            if _runes_only(raw)
        ]

        # Pages and segments are independent markers in the same stream, so a
        # page is attributed by where its runes fall in the concatenated book.
        self.pages = [Page(i, raw) for i, raw in enumerate(body.split(PAGE))]
        bounds, at = [], 0
        for seg in self.segments:
            bounds.append((at, at + seg.n, seg.idx))
            at += seg.n
        at = 0
        for pg in self.pages:
            mid = at + len(pg.runes) // 2
            for lo, hi, idx in bounds:
                if lo <= mid < hi:
                    pg.segment = idx
                    break
            at += len(pg.runes)
        self.translation = _read(_TRANSLATION)
        self.index = _read(_INDEX)

    @staticmethod
    def _parse_keys() -> dict[str, str]:
        out: dict[str, str] = {}
        for line in _read(_KEYS).splitlines():
            m = re.match(r"\+\s*(0\.\d+)\s+(.*)", line.strip())
            if m:
                out[m.group(1)] = m.group(2).strip()
        return out

    @property
    def runes(self) -> str:
        return "".join(s.runes for s in self.segments)

    def segment(self, idx: int) -> Segment:
        return self.segments[idx]

    def known_plaintext(self) -> dict[str, str]:
        """Section id -> plaintext line, from the archive's translation file."""
        out: dict[str, str] = {}
        for line in self.translation.splitlines():
            m = re.match(r"(0\.\d+\.\d+\.\d+)\s+(.+)", line.strip())
            if m:
                out[m.group(1)] = re.sub(r"\s+", " ", m.group(2)).strip()
        return out


def load(archive: str | None = None) -> LiberPrimus:
    return LiberPrimus(archive)
