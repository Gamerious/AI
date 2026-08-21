"""Tests for the Cicada 3301 toolkit.

The important ones are the reproductions: they decrypt the raw rune
transcription and compare against the community plaintext, so a regression in
the gematria table, the F rule or the decoder fails the suite.
"""

from __future__ import annotations

import math
import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cicada import ciphers as C
from cicada.analysis import build_model, ioc
from cicada.corpus import load
from cicada.decoder import decode, repeating, running
from cicada.gematria import (GEMATRIA_PRIMUS, N, PRIMES, RUNES, atbash_index,
                             gematria_sum, latin_to_indices, to_indices,
                             to_latin, to_runes)
from cicada.solved import OPEN_SEGMENTS, RECIPES, decrypt_segment, verify

ARCHIVE = os.environ.get("CICADA_ARCHIVE", "/home/user/rtkd/iddqd")
needs_archive = pytest.mark.skipif(
    not os.path.isdir(ARCHIVE), reason="rune archive not available"
)


# --- gematria ---------------------------------------------------------------

def test_alphabet_is_29_runes():
    assert N == 29 == len(set(RUNES)) == len(GEMATRIA_PRIMUS)


def test_primes_are_the_first_29_primes():
    assert list(PRIMES) == list(C.primes(29))


def test_atbash_is_an_involution():
    assert all(atbash_index(atbash_index(i)) == i for i in range(N))


def test_known_key_derivations():
    # both keys are quoted as index lists in the community key file
    assert latin_to_indices("DIVINITY") == [23, 10, 1, 10, 9, 10, 16, 26]
    assert latin_to_indices("FIRFUMFERENFE") == [0, 10, 4, 0, 1, 19, 0, 18, 4, 18, 9, 0, 18]


def test_digraphs_are_matched_greedily():
    # TH and ING are each a single rune, so THINGS is three runes, not six,
    # and transliterating back drops the I that ING absorbed.
    assert len(latin_to_indices("THINGS")) == 3
    assert to_latin(to_runes(latin_to_indices("THINGS"))) == "THNGS"
    assert len(latin_to_indices("SACRED")) == 6


def test_gematria_sum():
    assert gematria_sum(to_runes(latin_to_indices("CICADA"))) == 13 + 31 + 13 + 97 + 89 + 97


# --- ciphers ----------------------------------------------------------------

def test_totient_of_a_prime():
    assert C.totient(3301) == 3300


def test_totient_prime_stream_is_phi_of_primes():
    assert C.totient_prime_stream(10) == [(p - 1) % N for p in C.primes(10)]


def test_vigenere_round_trip():
    idx = latin_to_indices("THEPRIMESARESACRED")
    key = latin_to_indices("DIVINITY")
    enc = C.vigenere_encrypt(idx, key, skip_f=False)
    assert C.vigenere_decrypt(enc, key, skip_f=False) == idx


def test_affine_round_trip():
    idx = latin_to_indices("ALLTHINGSSHOULDBEENCRYPTED")
    enc = [(5 * i + 7) % N for i in idx]
    assert C.affine_decrypt(enc, 5, 7) == idx


# --- the F rule -------------------------------------------------------------

@needs_archive
def test_plaintext_f_rule_reproduces_the_ciphertext():
    """A plaintext F passes through and does *not* consume a key character."""
    lp = load()
    # take the plaintext as indices, not via latin: transliterating T+H to
    # "TH" and reparsing it would collapse two runes into thorn.
    cipher = to_indices(lp.segments[16].runes)
    plain, _ = decode(cipher, running(C.totient_prime_stream(400)),
                      build_model(4), beam=64)
    key = C.totient_prime_stream(400)
    out, j = [], 0
    for p in plain:
        if p == 0:
            out.append(0)
            continue
        out.append((p + key[j]) % N)
        j += 1
    assert to_runes(out) == lp.segments[16].runes


@needs_archive
def test_ciphertext_f_is_ambiguous():
    """More ciphertext Fs are collisions than genuine plaintext Fs."""
    lp = load()
    model = build_model(4)
    cipher = to_indices(lp.segments[1].runes)
    plain, _ = decode(cipher, repeating(latin_to_indices("DIVINITY")), model, beam=64)
    real = sum(1 for p in plain[:len(cipher)] if p == 0)
    seen = sum(1 for c in cipher if c == 0)
    assert seen > real > 0


# --- reproductions ----------------------------------------------------------

@needs_archive
@pytest.mark.parametrize("recipe", RECIPES, ids=lambda r: f"0.{r.segment}")
def test_solved_section_reproduces(recipe):
    lp = load()
    model = build_model(4)
    rows = {r.segment: (a, n) for r, a, n, _, _ in verify(lp, model)}
    agree, n = rows[recipe.segment]
    assert n > 50
    assert agree > 0.99, f"0.{recipe.segment} only reproduced to {agree:.3%}"


@needs_archive
def test_warning_page_decrypts_to_english():
    lp = load()
    got = decrypt_segment(lp, RECIPES[0], build_model(4))
    assert got.startswith("AWARNNGBELIEUENOTHNG")


@needs_archive
def test_beam_decoder_beats_naive_on_ambiguous_page():
    """Naive decryption desynchronises at the first plaintext F; the beam does not."""
    lp = load()
    model = build_model(4)
    cipher = to_indices(lp.segments[5].runes)
    key = latin_to_indices("FIRFUMFERENFE")
    naive = C.vigenere_decrypt(cipher, key, skip_f=False)
    beamed, _ = decode(cipher, repeating(key), model, beam=64)
    assert to_latin(to_runes(beamed)).startswith("ACOANDURNGALESSONTHEMASTER")
    assert model.score(beamed) > model.score(naive) + 1.0


# --- the anomaly ------------------------------------------------------------

@needs_archive
def test_open_segments_are_flat():
    lp = load()
    for i in OPEN_SEGMENTS:
        seg = lp.segments[i]
        if seg.n < 200:
            continue
        assert abs(ioc(to_indices(seg.runes)) - 1.0) < 0.05


@needs_archive
def test_adjacent_repeats_are_suppressed_on_open_segments():
    lp = load()
    a = np.concatenate([
        np.array(to_indices(lp.segments[i].runes))
        for i in OPEN_SEGMENTS if lp.segments[i].n > 50
    ])
    obs = int((a[:-1] == a[1:]).sum())
    exp = (a.size - 1) / N
    z = (obs - exp) / math.sqrt(exp * (1 - 1 / N))
    assert z < -10, f"expected a strong deficit, got z={z:.2f}"


@needs_archive
def test_wider_gaps_are_at_chance():
    """Only adjacency is suppressed - gap 2..5 sit at 1/29."""
    lp = load()
    a = np.concatenate([
        np.array(to_indices(lp.segments[i].runes))
        for i in OPEN_SEGMENTS if lp.segments[i].n > 50
    ])
    for gap in (2, 3, 4, 5):
        rate = (a[:-gap] == a[gap:]).mean()
        assert abs(rate - 1 / N) < 0.006, f"gap {gap} rate {rate:.4f}"


# --- 2012 ------------------------------------------------------------------

@needs_archive
def test_mabinogion_cipher():
    from cicada.outer import (MABINOGION_KEY, load_mabinogion,
                              mabinogion_decrypt, recover_vigenere_key)
    cipher, plain = load_mabinogion()
    assert recover_vigenere_key(cipher, plain) == MABINOGION_KEY
    letters = lambda s: "".join(c for c in s if c.isalpha()).upper()
    assert letters(mabinogion_decrypt(cipher)) == letters(plain)
