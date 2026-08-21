"""Gematria Primus - the alphabet Cicada 3301 built the Liber Primus on.

29 runes of the Anglo-Saxon futhorc.  Each rune carries an index (0..28), a
latin transliteration (some are digraphs) and the n-th prime number.

The whole Liber Primus, every solved page of it, is expressed in operations on
the *index*: atbash is ``28 - i``, every shift cipher is ``(i + k) mod 29``.
The primes matter for the key streams (``f(n) = phi(prime_n)``) and for the
numeric "gematria sum" of a word.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# The table.  Order is load-bearing: index == position in this tuple.
# ---------------------------------------------------------------------------

#: (rune, primary latin, alternate latin spellings, prime)
GEMATRIA_PRIMUS: tuple[tuple[str, str, tuple[str, ...], int], ...] = (
    ("ᚠ", "F",   (),           2),    # ᚠ  feoh
    ("ᚢ", "U",   ("V",),       3),    # ᚢ  ur
    ("ᚦ", "TH",  (),           5),    # ᚦ  thorn
    ("ᚩ", "O",   (),           7),    # ᚩ  os
    ("ᚱ", "R",   (),          11),    # ᚱ  rad
    ("ᚳ", "C",   ("K",),      13),    # ᚳ  cen
    ("ᚷ", "G",   (),          17),    # ᚷ  gyfu
    ("ᚹ", "W",   (),          19),    # ᚹ  wynn
    ("ᚻ", "H",   (),          23),    # ᚻ  haegl
    ("ᚾ", "N",   (),          29),    # ᚾ  nyd
    ("ᛁ", "I",   (),          31),    # ᛁ  is
    ("ᛄ", "J",   (),          37),    # ᛄ  ger
    ("ᛇ", "EO",  (),          41),    # ᛇ  eoh
    ("ᛈ", "P",   (),          43),    # ᛈ  peorth
    ("ᛉ", "X",   (),          47),    # ᛉ  eolh
    ("ᛋ", "S",   ("Z",),      53),    # ᛋ  sigel
    ("ᛏ", "T",   (),          59),    # ᛏ  tir
    ("ᛒ", "B",   (),          61),    # ᛒ  beorc
    ("ᛖ", "E",   (),          67),    # ᛖ  eh
    ("ᛗ", "M",   (),          71),    # ᛗ  man
    ("ᛚ", "L",   (),          73),    # ᛚ  lagu
    ("ᛝ", "NG",  ("ING",),    79),    # ᛝ  ing
    ("ᛟ", "OE",  (),          83),    # ᛟ  ethel
    ("ᛞ", "D",   (),          89),    # ᛞ  daeg
    ("ᚪ", "A",   (),          97),    # ᚪ  ac
    ("ᚫ", "AE",  (),         101),    # ᚫ  aesc
    ("ᚣ", "Y",   (),         103),    # ᚣ  yr
    ("ᛡ", "IA",  ("IO",),    107),    # ᛡ  ior
    ("ᛠ", "EA",  (),         109),    # ᛠ  ear
)

N = len(GEMATRIA_PRIMUS)                                   # 29
RUNES: str = "".join(r for r, _, _, _ in GEMATRIA_PRIMUS)
LATIN: tuple[str, ...] = tuple(l for _, l, _, _ in GEMATRIA_PRIMUS)
PRIMES: tuple[int, ...] = tuple(p for _, _, _, p in GEMATRIA_PRIMUS)

RUNE_INDEX: dict[str, int] = {r: i for i, r in enumerate(RUNES)}
PRIME_INDEX: dict[int, int] = {p: i for i, p in enumerate(PRIMES)}

#: every latin spelling (primary + alternates) -> index, longest first when parsing
_LATIN_TO_INDEX: dict[str, int] = {}
for _i, (_r, _l, _alts, _p) in enumerate(GEMATRIA_PRIMUS):
    _LATIN_TO_INDEX.setdefault(_l, _i)
    for _a in _alts:
        _LATIN_TO_INDEX.setdefault(_a, _i)
#: digraphs must be matched before single letters
_LATIN_KEYS: tuple[str, ...] = tuple(
    sorted(_LATIN_TO_INDEX, key=lambda s: (-len(s), s))
)


# ---------------------------------------------------------------------------
# conversions
# ---------------------------------------------------------------------------

def to_indices(runes: str) -> list[int]:
    """Runes -> gematria indices.  Non-rune characters are dropped."""
    return [RUNE_INDEX[c] for c in runes if c in RUNE_INDEX]


def to_runes(indices) -> str:
    """Gematria indices -> runes."""
    return "".join(RUNES[i % N] for i in indices)


def to_latin(runes: str, sep: str = "") -> str:
    """Runes -> latin transliteration; non-rune characters pass through."""
    out = []
    for c in runes:
        i = RUNE_INDEX.get(c)
        out.append(LATIN[i] + sep if i is not None else c)
    return "".join(out)


def latin_to_indices(text: str) -> list[int]:
    """Latin text -> gematria indices, greedily matching digraphs (TH, EO, NG...).

    Used to turn key words like ``DIVINITY`` into a key stream.  Characters that
    are not part of any spelling (spaces, punctuation) are skipped.
    """
    text = text.upper()
    out: list[int] = []
    i = 0
    while i < len(text):
        for k in _LATIN_KEYS:
            if text.startswith(k, i):
                out.append(_LATIN_TO_INDEX[k])
                i += len(k)
                break
        else:
            i += 1                      # not a rune letter -> skip
    return out


def latin_to_runes(text: str) -> str:
    return to_runes(latin_to_indices(text))


# ---------------------------------------------------------------------------
# gematria arithmetic
# ---------------------------------------------------------------------------

def gematria_sum(runes: str) -> int:
    """Sum of the prime values of every rune in *runes*."""
    return sum(PRIMES[i] for i in to_indices(runes))


def atbash_index(i: int) -> int:
    """Invert the gematria: first rune <-> last rune."""
    return (N - 1) - i


def shift_indices(indices, k: int) -> list[int]:
    return [(i + k) % N for i in indices]
