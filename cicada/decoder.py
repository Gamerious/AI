"""Beam decoder for Cicada's ambiguous ᚠ rule.

Cicada encrypted the polyalphabetic pages with one exception: a *plaintext* ᚠ
is emitted unchanged and does not consume a key character.  That is fine going
forwards, but backwards it is ambiguous - a ᚠ in the ciphertext is either

  (a) an untouched plaintext F, key index stays put, or
  (b) an ordinary rune that happened to encrypt onto index 0.

Both happen: in the WELCOME page 11 of the ᚠ are real Fs and 14 are collisions.
The community resolved these by eye.  This does it by search: every ᚠ forks the
state, states are keyed by (key position, n-gram context) and ranked by an
English model over runes, and the best surviving path is the plaintext.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from .analysis import RuneNgram
from .gematria import N, atbash_index

F = 0


@dataclass
class _State:
    j: int                 # key position
    out: list[int]
    score: float
    ctx: tuple[int, ...]   # last order-1 runes


def decode(
    cipher: Sequence[int],
    key_at: Callable[[int], int],
    model: RuneNgram,
    *,
    beam: int = 96,
    invert: bool = False,
    f_rule: bool = True,
) -> tuple[list[int], float]:
    """Return (plaintext indices, mean log-prob) for the best path.

    ``key_at(j)`` gives the key index for key position *j*, so the same decoder
    drives a repeating keyword and an infinite running key.
    """
    o = model.order
    states = [_State(0, [], 0.0, ())]

    def step(st: _State, p: int, advance: bool) -> _State:
        ctx = (st.ctx + (p,))[-(o - 1):] if o > 1 else ()
        sc = st.score
        if len(st.ctx) == o - 1:
            code = 0
            for v in st.ctx + (p,):
                code = code * N + v
            sc += float(model.logp[code])
        return _State(st.j + (1 if advance else 0), st.out + [p], sc, ctx)

    for c in cipher:
        nxt: dict[tuple, _State] = {}

        def offer(s: _State) -> None:
            k = (s.j, s.ctx)
            cur = nxt.get(k)
            if cur is None or s.score > cur.score:
                nxt[k] = s

        for st in states:
            v = atbash_index(c) if invert else c
            if f_rule and c == F:
                offer(step(st, F, False))                                # real plaintext F
                offer(step(st, (v - key_at(st.j)) % N, True))             # collision
            else:
                offer(step(st, (v - key_at(st.j)) % N, True))
        states = sorted(nxt.values(), key=lambda s: -s.score)[:beam]

    best = max(states, key=lambda s: s.score)
    n = max(1, len(best.out) - o + 1)
    return best.out, best.score / n


def repeating(key: Sequence[int]) -> Callable[[int], int]:
    return lambda j: key[j % len(key)]


def running(stream: Sequence[int]) -> Callable[[int], int]:
    return lambda j: stream[j] if j < len(stream) else 0
