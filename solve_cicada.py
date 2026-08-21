#!/usr/bin/env python3
"""Cicada 3301 - solver and analysis CLI.

    python solve_cicada.py verify         reproduce every solved Liber Primus page
    python solve_cicada.py read [SEG]     print the plaintext of the solved pages
    python solve_cicada.py stats          statistical survey of all 18 segments
    python solve_cicada.py doublets       the adjacency anomaly on the open pages
    python solve_cicada.py attack SEG     run the full attack battery on a segment
    python solve_cicada.py mabinogion     solve the 2012 Mabinogion cipher

The archive of rune transcriptions is expected at $CICADA_ARCHIVE
(default /home/user/rtkd/iddqd, github.com/rtkd/iddqd).
"""

from __future__ import annotations

import argparse
import math
import sys
import textwrap

import numpy as np

from cicada.analysis import build_model, doublets, ioc, periodic_ioc
from cicada.attack import run_all
from cicada.corpus import load
from cicada.gematria import to_indices
from cicada.solved import (OPEN_SEGMENTS, RECIPES, SOLVED_SEGMENTS,
                           decrypt_segment, verify)


def cmd_verify(args) -> int:
    lp, model = load(), build_model(4)
    ok = 0
    for r, agree, n, got, want in verify(lp, model):
        good = agree > 0.99
        ok += good
        print(f"{'OK  ' if good else 'FAIL'} seg 0.{r.segment:<2d} {agree*100:6.2f}% over {n:4d} runes "
              f"| {r.label:24s} | {r.method}")
    print(f"\n{ok}/{len(RECIPES)} solved sections reproduced from the raw rune transcription")
    return 0 if ok == len(RECIPES) else 1


def cmd_read(args) -> int:
    lp, model = load(), build_model(4)
    for r in RECIPES:
        if args.segment is not None and r.segment != args.segment:
            continue
        print(f"\n{'='*78}\nsegment 0.{r.segment}  -  {r.label}\nmethod: {r.method}\n{'='*78}")
        print(textwrap.fill(decrypt_segment(lp, r, model), 78))
    return 0


def cmd_stats(args) -> int:
    lp, model = load(), build_model(4)
    print(f"{'seg':6s} {'runes':>6s} {'state':7s} {'IoC':>6s} {'dbl%':>6s} {'4gram':>7s}   best periods")
    print("-" * 84)
    for s in lp.segments:
        a = np.array(to_indices(s.runes))
        if a.size < 20:
            continue
        state = "SOLVED" if s.idx in SOLVED_SEGMENTS else "open"
        per = sorted(((p, periodic_ioc(a, p)) for p in range(2, 61) if a.size // p >= 25),
                     key=lambda t: -t[1])[:4]
        print(f"0.{s.idx:<4d} {a.size:6d} {state:7s} {ioc(a):6.3f} {doublets(a)*100:6.2f} "
              f"{model.score(a):7.3f}   " + " ".join(f"{p}:{v:.2f}" for p, v in per))
    print("\nreference  Liber Primus plaintext: IoC 1.73  doublets 2.6%   4gram -2.1")
    print("           English in rune space : IoC 1.79")
    print("           uniform random        : IoC 1.00  doublets 3.45%  4gram -3.85")
    return 0


def cmd_doublets(args) -> int:
    lp = load()
    obs = tot = 0
    print("adjacent-rune repeats on the unsolved segments\n")
    print(f"{'seg':6s} {'pairs':>6s} {'found':>6s} {'expected':>9s} {'z':>7s}")
    for i in OPEN_SEGMENTS:
        a = np.array(to_indices(lp.segments[i].runes))
        if a.size < 50:
            continue
        d = int((a[:-1] == a[1:]).sum())
        e = (a.size - 1) / 29
        z = (d - e) / math.sqrt(e * (1 - 1 / 29))
        print(f"0.{i:<4d} {a.size-1:6d} {d:6d} {e:9.1f} {z:7.2f}")
        obs += d
        tot += a.size - 1
    e = tot / 29
    z = (obs - e) / math.sqrt(e * (1 - 1 / 29))
    print(f"{'ALL':6s} {tot:6d} {obs:6d} {e:9.1f} {z:7.2f}")
    print(f"\nthe open pages carry {obs} adjacent repeats where {e:.0f} are expected (z={z:.1f}).")
    print("gap-2 and wider repeats sit at chance, and the off-diagonal bigram table is flat,")
    print("so only *adjacency* is suppressed - see README_CICADA.md.")
    return 0


def cmd_attack(args) -> int:
    lp, model = load(), build_model(4)
    c = to_indices(lp.segments[args.segment].runes)
    print(f"segment 0.{args.segment}: {len(c)} runes\n")
    for h in run_all(c, model, top=args.top):
        print("  ", h)
    print(f"\nEnglish scores about {-2.1}, random about {-3.85}.")
    return 0


def cmd_mabinogion(args) -> int:
    from cicada.outer import (load_mabinogion, mabinogion_decrypt,
                              recover_vigenere_key)
    cipher, plain = load_mabinogion()
    key = recover_vigenere_key(cipher, plain)
    out = mabinogion_decrypt(cipher, key)
    letters = lambda s: "".join(c for c in s if c.isalpha()).upper()
    print(f"recovered key      : {key}  (length {len(key)}, reset at every line)")
    print(f"matches plaintext  : {letters(out) == letters(plain)}")
    print("\n" + textwrap.fill(out[:400], 78))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("verify").set_defaults(fn=cmd_verify)
    r = sub.add_parser("read"); r.add_argument("segment", nargs="?", type=int); r.set_defaults(fn=cmd_read)
    sub.add_parser("stats").set_defaults(fn=cmd_stats)
    sub.add_parser("doublets").set_defaults(fn=cmd_doublets)
    a = sub.add_parser("attack"); a.add_argument("segment", type=int)
    a.add_argument("--top", type=int, default=10); a.set_defaults(fn=cmd_attack)
    sub.add_parser("mabinogion").set_defaults(fn=cmd_mabinogion)
    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
