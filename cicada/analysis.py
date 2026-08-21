"""Statistics and an English language model, both living in rune-index space.

Scoring English *after* transliterating back to latin is a trap: the Gematria
Primus collapses TH/EO/NG/OE/IA/EA/AE into single symbols, so a plaintext
candidate has to be judged over the 29-symbol alphabet it actually lives in.
Everything here therefore trains on English that has been pushed through
``latin_to_indices`` first.
"""

from __future__ import annotations

import os
import re

import numpy as np

from .gematria import N, latin_to_indices, to_indices

DATA = os.path.join(os.path.dirname(__file__), "data")


# ---------------------------------------------------------------------------
# classic cipher statistics
# ---------------------------------------------------------------------------

def ioc(indices) -> float:
    """Index of coincidence, normalised so random == 1.0 over 29 symbols."""
    a = np.asarray(indices)
    n = a.size
    if n < 2:
        return 0.0
    counts = np.bincount(a, minlength=N).astype(np.float64)
    return float((counts * (counts - 1)).sum() / (n * (n - 1)) * N)


def periodic_ioc(indices, period: int) -> float:
    """Average IoC of the *period* interleaved columns - the Friedman test."""
    a = np.asarray(indices)
    vals = [ioc(a[i::period]) for i in range(period) if a[i::period].size > 1]
    return float(np.mean(vals)) if vals else 0.0


def kasiski(indices, max_period: int = 40) -> list[tuple[int, float]]:
    """Periods ranked by column IoC.  A real Vigenere spikes at its key length."""
    return sorted(
        ((p, periodic_ioc(indices, p)) for p in range(1, max_period + 1)),
        key=lambda t: -t[1],
    )


def chi_squared(indices, expected: np.ndarray) -> float:
    a = np.asarray(indices)
    obs = np.bincount(a, minlength=N).astype(np.float64)
    exp = expected * a.size
    exp = np.where(exp <= 0, 1e-9, exp)
    return float(((obs - exp) ** 2 / exp).sum())


def doublets(indices) -> float:
    """Fraction of adjacent equal runes.  English-in-runes sits near 0.02."""
    a = np.asarray(indices)
    return float((a[:-1] == a[1:]).mean()) if a.size > 1 else 0.0


# ---------------------------------------------------------------------------
# language model
# ---------------------------------------------------------------------------

class RuneNgram:
    """Add-k smoothed n-gram log-probabilities over the 29 gematria indices."""

    def __init__(self, order: int = 4) -> None:
        self.order = order
        self.logp: np.ndarray | None = None
        self.unigram: np.ndarray | None = None

    # -- training ----------------------------------------------------------
    def fit(self, indices, k: float = 0.5) -> "RuneNgram":
        a = np.asarray(indices, dtype=np.int64)
        o = self.order
        size = N ** o
        codes = np.zeros(a.size - o + 1, dtype=np.int64)
        for i in range(o):
            codes = codes * N + a[i:a.size - o + 1 + i]
        counts = np.bincount(codes, minlength=size).astype(np.float64)
        counts += k
        ctx = counts.reshape(-1, N).sum(axis=1, keepdims=True)
        self.logp = np.log(counts.reshape(-1, N) / ctx).astype(np.float32).ravel()
        uni = np.bincount(a, minlength=N).astype(np.float64) + k
        self.unigram = (uni / uni.sum()).astype(np.float64)
        return self

    # -- scoring -----------------------------------------------------------
    def score(self, indices) -> float:
        """Mean log-probability per rune.  Higher is more English."""
        a = np.asarray(indices, dtype=np.int64)
        o = self.order
        if a.size < o or self.logp is None:
            return -99.0
        codes = np.zeros(a.size - o + 1, dtype=np.int64)
        for i in range(o):
            codes = codes * N + a[i:a.size - o + 1 + i]
        return float(self.logp[codes].mean())

    # -- persistence -------------------------------------------------------
    def save(self, path: str) -> None:
        np.savez_compressed(path, order=self.order, logp=self.logp, unigram=self.unigram)

    @classmethod
    def load(cls, path: str) -> "RuneNgram":
        z = np.load(path)
        m = cls(int(z["order"]))
        m.logp = z["logp"]
        m.unigram = z["unigram"]
        return m


def english_text(extra: str = "") -> str:
    """The training text: Norvig's big.txt plus anything appended."""
    with open(os.path.join(DATA, "big.txt"), encoding="utf-8", errors="ignore") as fh:
        return fh.read() + "\n" + extra


def build_model(order: int = 4, extra: str = "", cache: bool = True) -> RuneNgram:
    path = os.path.join(DATA, f"english_{order}gram.npz")
    if cache and os.path.exists(path) and not extra:
        return RuneNgram.load(path)
    text = re.sub(r"[^A-Za-z ]+", " ", english_text(extra).upper())
    idx = latin_to_indices(text)
    m = RuneNgram(order).fit(idx)
    if cache and not extra:
        m.save(path)
    return m


# ---------------------------------------------------------------------------
# dictionary in rune space
# ---------------------------------------------------------------------------

def rune_dictionary(min_len: int = 3) -> set[tuple[int, ...]]:
    """English words encoded as index tuples, for word-hit scoring."""
    path = os.path.join(DATA, "words_alpha.txt")
    out: set[tuple[int, ...]] = set()
    with open(path, encoding="utf-8", errors="ignore") as fh:
        for w in fh:
            w = w.strip().upper()
            if len(w) >= min_len:
                out.add(tuple(latin_to_indices(w)))
    return out


def common_words(limit: int = 3000) -> list[tuple[int, ...]]:
    path = os.path.join(DATA, "google-10000-english-usa.txt")
    out = []
    with open(path, encoding="utf-8", errors="ignore") as fh:
        for i, w in enumerate(fh):
            if i >= limit:
                break
            w = w.strip().upper()
            if len(w) >= 3:
                out.append(tuple(latin_to_indices(w)))
    return out
