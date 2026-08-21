"""Reproduction of every publicly solved Liber Primus section.

Each entry is an executable decryption, not a copied answer: the recipe runs
against the raw rune transcription and the result is diffed against the
community's plaintext.  If the toolkit is wrong, ``verify`` says so.

Note on the "clear text F is not encrypted" remark in the community key file:
it is *not* a skip rule.  Recovering the key stream from cipher+plaintext shows
DIVINITY and FIRFUMFERENFE repeating with no gaps at all.  What the remark
describes is a property of the key: ᚠ has gematria index 0, so wherever the key
character is F the shift is zero and that rune passes through unchanged.  That
is why Cicada spelled CIRCUMFERENCE as FIRFUMFERENFE.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass
from typing import Callable, Sequence

from . import ciphers as C
from .analysis import RuneNgram, build_model
from .corpus import LiberPrimus
from .decoder import decode, repeating, running
from .gematria import latin_to_indices, to_indices, to_latin, to_runes


@dataclass
class Recipe:
    segment: int
    label: str
    method: str
    fn: Callable[[Sequence[int], RuneNgram], list[int]]
    ref_section: str
    ref_offset: int = 0


def _vig(word: str) -> Callable[[Sequence[int], RuneNgram], list[int]]:
    key = C.keyword_stream(word)
    return lambda idx, m: decode(idx, repeating(key), m, beam=128)[0]


def _totient_primes(idx: Sequence[int], m: RuneNgram) -> list[int]:
    stream = C.totient_prime_stream(len(idx) + 8)
    return decode(idx, running(stream), m, beam=128)[0]


def _atbash_shift3(idx: Sequence[int], m: RuneNgram) -> list[int]:
    """'Invert Gematria. Key: 3' - atbash first, then shift the result by +3."""
    return C.shift(C.atbash(idx), 3)


def _plain(idx: Sequence[int], m: RuneNgram) -> list[int]:
    return list(idx)


def _atbash(idx: Sequence[int], m: RuneNgram) -> list[int]:
    return C.atbash(idx)


RECIPES: tuple[Recipe, ...] = (
    Recipe(0,  "A WARNING",              "atbash (invert gematria)",                  _atbash,            "0"),
    Recipe(1,  "WELCOME / WISDOM",       "vigenere, key = DIVINITY",                  _vig("DIVINITY"),   "1"),
    Recipe(2,  "SOME WISDOM / KNOW THIS", "plaintext (no cipher)",                    _plain,             "1", 515),
    Recipe(3,  "A KOAN",                 "atbash then shift +3",                      _atbash_shift3,     "2"),
    Recipe(4,  "THE LOSS OF DIVINITY",   "plaintext (no cipher)",                     _plain,             "3"),
    Recipe(5,  "A KOAN (lesson)",        "vigenere, key = FIRFUMFERENFE",             _vig("FIRFUMFERENFE"), "4"),
    Recipe(6,  "AN INSTRUCTION",         "plaintext (no cipher)",                     _plain,             "4", 319),
    Recipe(16, "AN END",                 "running key f(n) = phi(prime_n)",           _totient_primes,    "13"),
    Recipe(17, "A PARABLE",              "plaintext (no cipher)",                     _plain,             "14"),
)

SOLVED_SEGMENTS = tuple(r.segment for r in RECIPES)
OPEN_SEGMENTS = tuple(i for i in range(18) if i not in SOLVED_SEGMENTS)


# ---------------------------------------------------------------------------

def normalise(text: str) -> str:
    """English -> canonical rune-latin so digraphs (NG, TH, EA...) compare fairly."""
    return to_latin(to_runes(latin_to_indices(text)))


def _sections(lp: LiberPrimus) -> dict[str, str]:
    out: dict[str, list[str]] = {}
    for k, v in lp.known_plaintext().items():
        out.setdefault(k.split(".")[1], []).append(v)
    return {k: " ".join(v) for k, v in out.items()}


#: places where the book itself disagrees with the community transcription -
#: the ciphertext of the WISDOM headline really does decrypt to WIDSOM.
KNOWN_ANOMALIES = {(1, 394): ("WIDSOM", "WISDOM")}


def decrypt_segment(lp: LiberPrimus, recipe: Recipe, model: RuneNgram | None = None) -> str:
    seg = lp.segments[recipe.segment]
    model = model or build_model(4)
    return to_latin(to_runes(recipe.fn(to_indices(seg.runes), model)))


def verify(lp: LiberPrimus, model: RuneNgram | None = None) -> list[tuple[Recipe, float, int, str, str]]:
    """-> (recipe, agreement, compared_runes, decrypted, reference)"""
    secs = _sections(lp)
    model = model or build_model(4)
    rows = []
    for r in RECIPES:
        got = decrypt_segment(lp, r, model)
        ref_idx = latin_to_indices(secs[r.ref_section])[r.ref_offset:]
        want = to_latin(to_runes(ref_idx))
        n = min(len(got), len(want))
        # alignment-robust: the reference file carries its own typos (FOLLWNG for
        # FOLLOWNG), and a single dropped rune would otherwise destroy a
        # positional comparison from that point on.
        agree = difflib.SequenceMatcher(None, got[:n], want[:n], autojunk=False).ratio()
        rows.append((r, agree, n, got, want))
    return rows
