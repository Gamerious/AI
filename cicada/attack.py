"""Attack battery for the unsolved Liber Primus segments.

None of this finds the answer - the point is that the search is *recorded*.
Every family below is run against every open segment and the best candidate is
scored by the rune n-gram model, so "we tried X" is a claim backed by code
rather than by memory.

Discriminator scale (mean log-prob per rune, 4-gram):
    English in runes  ~ -2.1
    random over 29    ~ -3.85
Anything above about -3.2 deserves a human look.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Iterable, Iterator, Sequence

import numpy as np

from . import ciphers as C
from .analysis import RuneNgram, ioc
from .decoder import decode, repeating, running
from .gematria import N, latin_to_indices, to_latin, to_runes

ENGLISH_SCORE = -2.1
RANDOM_SCORE = -3.85
INTERESTING = -3.25


@dataclass
class Hit:
    family: str
    detail: str
    score: float
    ioc: float
    preview: str

    def __str__(self) -> str:
        return f"{self.score:+7.3f} ioc={self.ioc:5.3f} [{self.family}] {self.detail}: {self.preview[:60]}"


def _hit(family: str, detail: str, out: np.ndarray, model: RuneNgram) -> Hit:
    return Hit(family, detail, model.score(out), ioc(out), to_latin(to_runes(out.tolist())))


# ---------------------------------------------------------------------------
# integer sequences Cicada actually cares about
# ---------------------------------------------------------------------------

def _primes(n: int) -> list[int]:
    return list(itertools.islice(C.primes(), n))


def sequences(n: int) -> dict[str, list[int]]:
    """Named integer streams, already reduced mod 29."""
    pr = _primes(n + 8)
    fib = [0, 1]
    while len(fib) < n + 8:
        fib.append(fib[-1] + fib[-2])
    luc = [2, 1]
    while len(luc) < n + 8:
        luc.append(luc[-1] + luc[-2])
    tot = [C.totient(i) for i in range(1, n + 9)]
    seqs = {
        "primes": pr,
        "phi(prime_n)": [p - 1 for p in pr],
        "phi(n)": tot,
        "n": list(range(1, n + 9)),
        "fibonacci": fib,
        "lucas": luc,
        "triangular": [i * (i + 1) // 2 for i in range(1, n + 9)],
        "squares": [i * i for i in range(1, n + 9)],
        "prime_gaps": [b - a for a, b in zip(pr, pr[1:])],
        "prime_index_sum": list(itertools.accumulate(pr)),
        "phi_prime_sum": list(itertools.accumulate(p - 1 for p in pr)),
        "3301*n": [3301 * i for i in range(1, n + 9)],
        "digits_pi": _digits("pi", n + 8),
        "digits_e": _digits("e", n + 8),
    }
    return {k: [v % N for v in vs] for k, vs in seqs.items()}


def _digits(which: str, n: int) -> list[int]:
    """Decimal digits of pi or e via integer arithmetic (no float error)."""
    prec = n + 20
    if which == "pi":
        # Chudnovsky is overkill; a simple arctan series at integer scale is fine
        scale = 10 ** prec
        def arctan_inv(x: int) -> int:
            total = term = scale // x
            x2, k = x * x, 1
            while term:
                term //= x2
                total += -term // (2 * k + 1) if k % 2 else term // (2 * k + 1)
                k += 1
            return total
        val = 16 * arctan_inv(5) - 4 * arctan_inv(239)
    else:
        scale = 10 ** prec
        val, term, k = 0, scale, 0
        while term:
            val += term
            k += 1
            term //= k
    return [int(c) for c in str(val)][:n]


# ---------------------------------------------------------------------------
# attack families
# ---------------------------------------------------------------------------

def monoalphabetic(cipher: np.ndarray, model: RuneNgram) -> Iterator[Hit]:
    """All 29 shifts x atbash, plus all 812 affine maps."""
    for inv in (False, True):
        base = (N - 1 - cipher) if inv else cipher
        tag = "atbash+" if inv else ""
        for k in range(N):
            yield _hit("mono", f"{tag}shift {k}", (base + k) % N, model)
    for a in range(1, N):
        if np.gcd(a, N) != 1:
            continue
        inv_a = pow(a, -1, N)
        for b in range(N):
            yield _hit("affine", f"a={a} b={b}", (inv_a * (cipher - b)) % N, model)


def running_keys(cipher: np.ndarray, model: RuneNgram,
                 offsets: Iterable[int] = range(0, 40)) -> Iterator[Hit]:
    """Named integer sequences as running keys, both directions, +/- atbash."""
    n = cipher.size
    seqs = sequences(n + max(offsets) + 4)
    for name, seq in seqs.items():
        arr = np.array(seq)
        for off in offsets:
            k = arr[off:off + n]
            if k.size < n:
                continue
            for inv in (False, True):
                base = (N - 1 - cipher) if inv else cipher
                tag = "atbash+" if inv else ""
                yield _hit("stream", f"{tag}{name} off={off} sub", (base - k) % N, model)
                yield _hit("stream", f"{tag}{name} off={off} add", (base + k) % N, model)


def keyword_attack(cipher: np.ndarray, model: RuneNgram,
                   words: Sequence[str]) -> Iterator[Hit]:
    """Vigenere with a dictionary of candidate keywords."""
    n = cipher.size
    for w in words:
        key = latin_to_indices(w.upper())
        if not key:
            continue
        k = np.array(key)[np.arange(n) % len(key)]
        for inv in (False, True):
            base = (N - 1 - cipher) if inv else cipher
            tag = "atbash+" if inv else ""
            yield _hit("keyword", f"{tag}{w} sub", (base - k) % N, model)
            yield _hit("keyword", f"{tag}{w} add", (base + k) % N, model)


def f_rule_streams(cipher: np.ndarray, model: RuneNgram,
                   offsets: Iterable[int] = range(0, 20), beam: int = 6) -> Iterator[Hit]:
    """Same integer streams, but decoded under Cicada's plaintext-F rule.

    Needed because a plain Vigenere decode of an F-rule page desynchronises at
    the first plaintext F and scores like noise - that is exactly how the
    battery misses FIRFUMFERENFE without this pass.
    """
    n = cipher.size
    seqs = sequences(n + max(offsets) + 8)
    for name, seq in seqs.items():
        arr = list(seq)
        for off in offsets:
            k = arr[off:]
            if len(k) < n + 2:
                continue
            for inv in (False, True):
                out, sc = decode(cipher.tolist(), running(k), model, beam=beam, invert=inv)
                tag = "atbash+" if inv else ""
                a = np.array(out)
                yield Hit("stream/F", f"{tag}{name} off={off}", sc, ioc(a),
                          to_latin(to_runes(out)))


def f_rule_keywords(cipher: np.ndarray, model: RuneNgram,
                    words: Sequence[str], beam: int = 6) -> Iterator[Hit]:
    for w in words:
        key = latin_to_indices(w.upper())
        if not key:
            continue
        for inv in (False, True):
            out, sc = decode(cipher.tolist(), repeating(key), model, beam=beam, invert=inv)
            tag = "atbash+" if inv else ""
            yield Hit("keyword/F", f"{tag}{w}", sc, ioc(np.array(out)),
                      to_latin(to_runes(out)))


def transposition(cipher: np.ndarray, model: RuneNgram,
                  cols: Iterable[int] = range(2, 61)) -> Iterator[Hit]:
    for c in cols:
        out = np.array(C.transpose_columnar(cipher.tolist(), c))
        yield _hit("transpose", f"columnar {c}", out, model)
        yield _hit("transpose", f"columnar {c} rev", out[::-1], model)


def autokey(cipher: np.ndarray, model: RuneNgram, max_primer: int = 3) -> Iterator[Hit]:
    for plen in range(1, max_primer + 1):
        for primer in itertools.product(range(N), repeat=plen):
            out = np.array(C.autokey_decrypt(cipher.tolist(), primer, skip_f=False))
            yield _hit("autokey", f"primer {primer}", out, model)


CICADA_WORDS = [
    "CICADA", "LIBERPRIMUS", "INSTAR", "DIVINITY", "CIRCUMFERENCE", "FIRFUMFERENFE",
    "PILGRIM", "PRIMES", "TOTIENT", "SACRED", "WISDOM", "KOAN", "PARABLE", "MOBIUS",
    "SHADOWS", "AETHEREAL", "BUFFERS", "VOID", "CARNAL", "OBSCURA", "FORM", "ANEND",
    "WELCOME", "WARNING", "EMERGE", "TRUTH", "SELF", "DEATH", "ADHERE", "PRESERVE",
    "ANALOG", "CONSUMPTION", "PRESERVATION", "ADHERENCE", "THEDIVINITYWITHIN",
    "AWARNING", "AKOAN", "ANINSTRUCTION", "KNOWTHIS", "SOMEWISDOM", "THELOSSOFDIVINITY",
    "3301", "1033", "AENEAS", "MABINOGION", "AGRIPPA", "KINGARTHUR", "ARTHUR",
]


def run_all(cipher: Sequence[int], model: RuneNgram, words: Sequence[str] | None = None,
            top: int = 12) -> list[Hit]:
    """Run every family and return the highest-scoring candidates."""
    c = np.asarray(cipher)
    words = list(words or CICADA_WORDS)
    best: list[Hit] = []
    for gen in (monoalphabetic(c, model),
                running_keys(c, model),
                keyword_attack(c, model, words),
                f_rule_streams(c, model),
                f_rule_keywords(c, model, words),
                transposition(c, model),
                autokey(c, model, 2)):
        for h in gen:
            best.append(h)
            if len(best) > 4000:
                best.sort(key=lambda x: -x.score)
                del best[top * 4:]
    best.sort(key=lambda x: -x.score)
    return best[:top]
