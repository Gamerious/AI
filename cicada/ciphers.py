"""Cipher primitives over the 29-rune Gematria Primus alphabet.

Everything works on *index lists* (0..28).  ``decrypt`` is the direction that
matters; ``encrypt`` exists so reproductions can be checked round-trip.

The Cicada-specific wrinkle is ``skip_f``: on the polyalphabetic pages a
plaintext ``ᚠ`` (F, index 0) is passed through unencrypted and does **not**
consume a key character.  Cicada used this to smuggle the words
"FIRFUMFERENFE"/"DIVINITY" style key hints; the community named it the
"clear text F is not encrypted" rule.
"""

from __future__ import annotations

from itertools import count
from typing import Iterable, Iterator, Sequence

from .gematria import N, atbash_index, latin_to_indices

F = 0  # index of ᚠ


# ---------------------------------------------------------------------------
# key streams
# ---------------------------------------------------------------------------

def primes(limit: int | None = None) -> Iterator[int]:
    """Infinite (or bounded) stream of primes by incremental sieve."""
    seen: dict[int, int] = {}
    q = 2
    made = 0
    while limit is None or made < limit:
        if q not in seen:
            yield q
            made += 1
            seen[q * q] = q
        else:
            step = seen.pop(q)
            nxt = q + step
            while nxt in seen:
                nxt += step
            seen[nxt] = step
        q += 1


def totient(n: int) -> int:
    """Euler phi."""
    result, m, p = n, n, 2
    while p * p <= m:
        if m % p == 0:
            while m % p == 0:
                m //= p
            result -= result // p
        p += 1 if p == 2 else 2
    if m > 1:
        result -= result // m
    return result


def totient_prime_stream(length: int, start: int = 0) -> list[int]:
    """f(n) = phi(prime_n) = prime_n - 1.  The key of the 'AN END' page."""
    out: list[int] = []
    for i, p in enumerate(primes()):
        if i < start:
            continue
        out.append((p - 1) % N)
        if len(out) == length:
            return out
    return out


def prime_stream(length: int, start: int = 0) -> list[int]:
    out: list[int] = []
    for i, p in enumerate(primes()):
        if i < start:
            continue
        out.append(p % N)
        if len(out) == length:
            return out
    return out


def keyword_stream(key: str | Sequence[int]) -> list[int]:
    """Latin keyword -> gematria index key, or pass through an index sequence."""
    if isinstance(key, str):
        return latin_to_indices(key)
    return list(key)


# ---------------------------------------------------------------------------
# ciphers
# ---------------------------------------------------------------------------

def atbash(indices: Iterable[int]) -> list[int]:
    """Self-inverse: ᚠ<->ᛠ.  Cicada calls it 'invert gematria'."""
    return [atbash_index(i) for i in indices]


def shift(indices: Iterable[int], k: int) -> list[int]:
    return [(i + k) % N for i in indices]


def vigenere_decrypt(
    indices: Sequence[int],
    key: Sequence[int],
    *,
    skip_f: bool = True,
    invert: bool = False,
    offset: int = 0,
    subtract: bool = True,
) -> list[int]:
    """Polyalphabetic decryption over 29 runes.

    ``invert``  - apply atbash to each rune before the shift ('invert gematria')
    ``skip_f``  - ᚠ passes through and does not consume a key character
    ``subtract``- normal Vigenere decryption; ``False`` gives Beaufort-ish add
    """
    if not key:
        return list(indices)
    out: list[int] = []
    j = offset
    for c in indices:
        if skip_f and c == F:
            out.append(F)
            continue
        v = atbash_index(c) if invert else c
        k = key[j % len(key)]
        out.append((v - k) % N if subtract else (v + k) % N)
        j += 1
    return out


def vigenere_encrypt(
    indices: Sequence[int],
    key: Sequence[int],
    *,
    skip_f: bool = True,
    invert: bool = False,
    offset: int = 0,
) -> list[int]:
    if not key:
        return list(indices)
    out: list[int] = []
    j = offset
    for p in indices:
        if skip_f and p == F:
            out.append(F)
            continue
        k = key[j % len(key)]
        v = (p + k) % N
        out.append(atbash_index(v) if invert else v)
        j += 1
    return out


def stream_decrypt(
    indices: Sequence[int],
    stream: Sequence[int],
    *,
    skip_f: bool = True,
    invert: bool = False,
    subtract: bool = True,
) -> list[int]:
    """Running-key decryption: one key value per (non-skipped) rune."""
    out: list[int] = []
    j = 0
    for c in indices:
        if skip_f and c == F:
            out.append(F)
            continue
        v = atbash_index(c) if invert else c
        k = stream[j % len(stream)]
        out.append((v - k) % N if subtract else (v + k) % N)
        j += 1
    return out


def affine_decrypt(indices: Iterable[int], a: int, b: int) -> list[int]:
    """i -> a^-1 * (i - b) mod 29.  29 is prime so every a in 1..28 is valid."""
    inv = pow(a, -1, N)
    return [(inv * (i - b)) % N for i in indices]


def autokey_decrypt(
    indices: Sequence[int], primer: Sequence[int], *, skip_f: bool = True
) -> list[int]:
    """Plaintext-autokey: the recovered plaintext feeds the key stream."""
    out: list[int] = []
    key = list(primer)
    j = 0
    for c in indices:
        if skip_f and c == F:
            out.append(F)
            continue
        p = (c - key[j]) % N
        out.append(p)
        key.append(p)
        j += 1
    return out


def transpose_columnar(indices: Sequence[int], cols: int) -> list[int]:
    rows = (len(indices) + cols - 1) // cols
    out: list[int] = []
    for c in range(cols):
        for r in range(rows):
            k = r * cols + c
            if k < len(indices):
                out.append(indices[k])
    return out
