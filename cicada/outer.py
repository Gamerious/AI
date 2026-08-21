"""The outer puzzle chains (2012 / 2013 / 2014) and their reproducible pieces.

The 2012 and 2013 hunts were solved by the community years ago; most of their
steps were stego, .onion fetches and a book code rather than cipher work, so
only the parts that can still be recomputed from the archived artefacts live
here.  ``2014`` is the Liber Primus - see :mod:`cicada.solved`.
"""

from __future__ import annotations

import os

ARCHIVE = os.environ.get("CICADA_ARCHIVE", "/home/user/rtkd/iddqd")

# ---------------------------------------------------------------------------
# 2012 - the Mabinogion cipher
# ---------------------------------------------------------------------------

MABINOGION_KEY = "KCOHTGSMHIRATHOSOTNABCA"


def _letters(s: str) -> str:
    return "".join(c for c in s if c.isalpha())


def mabinogion_decrypt(cipher: str, key: str = MABINOGION_KEY) -> str:
    """Vigenere over A-Z, key index advancing on letters only and **reset at
    every newline**.  Non-letters pass through untouched.

    The per-line reset is the non-obvious part: taken as one continuous stream
    the key looks aperiodic, and only lines up once you notice it restarts.
    """
    out = []
    for line in cipher.split("\n"):
        j = 0
        for ch in line:
            if ch.isalpha():
                base = 65 if ch.isupper() else 97
                k = ord(key[j % len(key)]) - 65
                out.append(chr((ord(ch) - base - k) % 26 + base))
                j += 1
            else:
                out.append(ch)
        out.append("\n")
    return "".join(out)[:-1]


def mabinogion_encrypt(plain: str, key: str = MABINOGION_KEY) -> str:
    out = []
    for line in plain.split("\n"):
        j = 0
        for ch in line:
            if ch.isalpha():
                base = 65 if ch.isupper() else 97
                k = ord(key[j % len(key)]) - 65
                out.append(chr((ord(ch) - base + k) % 26 + base))
                j += 1
            else:
                out.append(ch)
        out.append("\n")
    return "".join(out)[:-1]


def recover_vigenere_key(cipher: str, plain: str, max_len: int = 60) -> str | None:
    """Derive the key from a cipher/plaintext pair, assuming per-line reset.

    The two archive files are wrapped differently, so they do not align
    character for character.  Only the *letter* streams align: the cipher
    supplies the line structure, the plaintext is consumed as a flat stream.
    """
    flat_plain = _letters(plain).upper()
    stream: list[list[int]] = []
    pos = 0
    for lc in cipher.split("\n"):
        row = []
        for ch in lc:
            if ch.isalpha():
                row.append((ord(ch.upper()) - ord(flat_plain[pos])) % 26)
                pos += 1
        stream.append(row)
    flat = max(stream, key=len) if stream else []
    for L in range(1, min(max_len, len(flat)) + 1):
        if all(
            row[i] == flat[i % L] for row in stream for i in range(len(row))
        ):
            return "".join(chr(65 + x) for x in flat[:L])
    return None


def load_mabinogion() -> tuple[str, str]:
    with open(os.path.join(ARCHIVE, "2012/02/mabinogion_transcript"),
              encoding="utf-8", errors="ignore") as fh:
        cipher = fh.read()
    with open(os.path.join(ARCHIVE, "2012/02/mabinogion_translation"),
              encoding="utf-8", errors="ignore") as fh:
        plain = fh.read()
    return cipher, plain


# ---------------------------------------------------------------------------
# 2013 - hidden service byte strings
# ---------------------------------------------------------------------------

def load_byte_strings() -> dict[str, bytes]:
    """The four hex blobs the 2013 hunt published across its hidden services."""
    path = os.path.join(ARCHIVE, "byte-strings/byte-strings")
    out: dict[str, bytes] = {}
    label = None
    with open(path, encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("*\t") or line.startswith("* "):
                label = line.lstrip("*\t ").strip()
            elif line and all(c in "0123456789abcdefABCDEF" for c in line):
                if label:
                    out[label] = bytes.fromhex(line)
    return out
