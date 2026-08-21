# Cicada 3301

A working toolkit for the Cicada 3301 puzzles, plus an honest account of what
is solved and what is not.

```bash
python solve_cicada.py verify        # reproduce every solved Liber Primus page
python solve_cicada.py read          # print the plaintext of those pages
python solve_cicada.py stats         # statistical survey of all 18 segments
python solve_cicada.py doublets      # the adjacency anomaly on the open pages
python solve_cicada.py attack 15     # run the attack battery on a segment
python solve_cicada.py mabinogion    # solve the 2012 Mabinogion cipher
pytest tests/test_cicada.py          # 27 tests, mostly reproductions
```

Rune data comes from the community archive
[`rtkd/iddqd`](https://github.com/rtkd/iddqd); point `$CICADA_ARCHIVE` at a
checkout (default `/home/user/rtkd/iddqd`).  Nothing here is transcribed from
memory: every plaintext below is computed from the raw runes and diffed against
the community's own translation file.

## Status

| hunt | state |
| --- | --- |
| 2012 (first puzzle) | solved by the community in 2012 |
| 2013 (second puzzle) | solved by the community in 2013 |
| 2014 -> Liber Primus | **19% solved.** 2 977 of 15 933 runes.  The other 12 956 are open, and have been since 2014 |

`solve_cicada.py verify` reproduces all nine solved sections:

```
OK   seg 0.0  100.00% |  A WARNING             | atbash (invert gematria)
OK   seg 0.1   99.82% |  WELCOME / WISDOM      | vigenere, key = DIVINITY
OK   seg 0.2  100.00% |  SOME WISDOM/KNOW THIS | plaintext (no cipher)
OK   seg 0.3  100.00% |  A KOAN                | atbash then shift +3
OK   seg 0.4   99.88% |  THE LOSS OF DIVINITY  | plaintext (no cipher)
OK   seg 0.5  100.00% |  A KOAN (lesson)       | vigenere, key = FIRFUMFERENFE
OK   seg 0.6  100.00% |  AN INSTRUCTION        | plaintext (no cipher)
OK   seg 0.16 100.00% |  AN END                | running key f(n) = phi(prime_n)
OK   seg 0.17 100.00% |  A PARABLE             | plaintext (no cipher)
```

The two sub-100% rows are the reference file being wrong, not the decryption -
see *Anomalies*.

## The F rule, and why it makes decryption ambiguous

The community key file annotates the polyalphabetic pages with "clear text F is
not encrypted".  That is a **plaintext** rule: a plaintext ᚠ is emitted
unchanged and does **not** consume a key character.  Re-encrypting the known
plaintext under that rule reproduces the ciphertext exactly (100% on segments
0.5 and 0.16); without it, only 28% and 66%.

Backwards this is ambiguous, because a ᚠ in the ciphertext is either an
untouched plaintext F or an ordinary rune that encrypted onto index 0.  Both
happen, and collisions are the *majority*:

| page | ciphertext ᚠ | genuine plaintext F | collisions |
| --- | --- | --- | --- |
| 0.1 WELCOME | 25 | 11 | 14 |
| 0.5 A KOAN | 5 | 2 | 3 |
| 0.16 AN END | 5 | 1 | 4 |

`cicada/decoder.py` resolves them by search rather than by eye: every ᚠ forks
the state, states are keyed by (key position, n-gram context) and ranked by an
English model built over runes, and the best surviving path is the plaintext.
It recovers segments 0.5 and 0.16 to 100% and 0.1 to 99.8%.

Note the model has to live in **rune space**.  The Gematria Primus collapses
TH, EO, NG, OE, IA, EA and AE into single symbols, so scoring a candidate after
transliterating it back to latin measures the wrong alphabet.  `THINGS` is
three runes (TH + ING + S), not six.

## Anomalies found while reproducing

- **`WIDSOM`.** The WISDOM headline in segment 0.1 genuinely encrypts to
  W-I-D-S-O-M.  The ciphertext is ᛒᛗᚱᚦᚠᛈ; WISDOM would require ᛒᛗᚫᛁᚠᛈ.  The
  community translation silently corrects it.  Given that the book's own first
  page says *"do not edit or change this book ... either the words or their
  numbers, for all is sacred"*, the transposition is worth keeping visible.
- **`FOLLOWNG`.** The reference translation of segment 0.4 drops an O; the book
  spells it out.

## The unsolved 81%

Segments 0.7 - 0.15 have resisted the community since 2014.  Measuring them
says why.

**Every ordinary handle is absent.**  Index of coincidence is 1.000, against
1.787 for English in rune space, 1.730 for the Liber Primus's own plaintext and
1.000 for uniform random.  No lag from 1 to 200
shows a differential spike.  A column-wise chi-squared scan over key lengths
1-60 produces no peak, only the monotone drift in key length that means "no
periodic key".  The off-diagonal bigram table is statistically flat
(chi-squared 841.7 on 811 degrees of freedom, z = +0.76).

**One thing is not random.**  Adjacent identical runes are strongly suppressed:

| | pairs | observed | expected | z |
| --- | --- | --- | --- | --- |
| open segments | 12 939 | **86** | 446.2 | **-17.35** |
| solved encrypted pages | 918 | 27 | 31.7 | -0.84 |
| plaintext pages | 1 095 | 29 | 37.8 | -1.46 |

That is p ~ 1e-67.  It is specific to adjacency: repeats at gap 2, 3, 4 and 5
all sit at chance (3.45%).  It holds within words and across word, line and
page boundaries alike, so it is not a per-word key reset.  It is uniform across
all 29 runes - no single rune is special.  And it reproduces on an
**independent transcription** ([`krisyotam/cicada3301`](https://github.com/krisyotam/cicada3301),
13 136 runes, 89 observed against 452.9 expected, z = -17.40), so it is a
property of the book and not of one transcriber.

Turning that into a generative test: take real Liber Primus plaintext, encrypt
it every way, and ask which output has the measured fingerprint of
IoC 1.000 / doublet ratio 0.19 / flat off-diagonal.

| construction | IoC | doublets/expected | off-diagonal z |
| --- | --- | --- | --- |
| true one-time pad | 1.001 | 0.91 | +0.70 |
| Vigenere, L = 8 | 1.102 | 0.95 | +156.45 |
| Vigenere, L = 40 | 1.012 | 0.95 | +29.71 |
| running key (English) | 1.052 | 1.03 | +31.58 |
| autokey on plaintext | 1.088 | 1.12 | +192.29 |
| autokey on ciphertext | 1.001 | 0.88 | +0.64 |
| cumulative sum | 1.002 | 0.48 | +227.67 |
| OTP + hard no-repeat rule | 1.000 | 0.00 | +9.71 |
| OTP + 81% no-repeat rule | 1.000 | 0.18 | -1.44 |
| **actual unsolved Liber Primus** | **1.000** | **0.19** | **+0.75** |

Only the last construction matches on all three axes.  The unsolved pages
behave like a stream cipher whose key is indistinguishable from random, plus a
partial constraint against repeating a rune immediately.  A cumulative cipher
(`c[i] = c[i-1] + p[i]`, where a doublet means a plaintext F) is the obvious
candidate for that constraint and is **ruled out**: it would predict a doublet
rate equal to the plaintext F rate of 1.63%, not 0.66%, and it wrecks the
off-diagonal table.  Inverting it recovers noise.

## What was searched

`cicada/attack.py` runs seven families against every open segment.  The battery
is validated on positive controls first - it rediscovers atbash on 0.0,
atbash+3 on 0.3, phi(prime_n) on 0.16, DIVINITY on 0.1 and FIRFUMFERENFE on 0.5,
each by a margin of 0.6 to 1.5 nats over the runner-up.  On the open segments:

| family | candidates per segment | best score seen |
| --- | --- | --- |
| all 29 shifts, atbash, all 812 affine maps | 870 | -3.74 |
| 14 integer streams x 40 offsets x 2 directions x atbash | 2 240 | -3.74 |
| 10 000-word keyword Vigenere | 38 496 | -3.68 |
| the same streams under the F rule (beam decoded) | 560 | -3.63 |
| 1 200-word keyword under the F rule | 2 492 | -3.67 |
| columnar transposition, 2-60 columns | 118 | -3.77 |
| autokey, all primers up to length 2 | 870 | -3.71 |

English scores about -2.1 and uniform random about -3.85.  Nothing in roughly
365 000 candidates cleared -3.6, which is what taking a maximum over that many
random draws produces on its own.  The integer streams include primes,
phi(prime_n), phi(n), Fibonacci, Lucas, triangular numbers, squares, prime
gaps, prime partial sums, multiples of 3301 and the digits of pi and e.

This does not prove the pages are unbreakable.  It does mean the answer is not
a keyword, not a named integer sequence, not a transposition and not any
periodic key - which is most of what a classical toolbox contains, and is
consistent with a decade of community failure.

## 2012 and 2013

Those hunts were chains of steganography, hidden services, a phone number and
a book code rather than cipher work, so most steps cannot be recomputed from
archived artefacts.  Two that can:

- **The Mabinogion cipher.**  `2012/02/mabinogion_transcript` is *The Lady of
  the Fountain* under a Vigenere with the 23-letter key
  `KCOHTGSMHIRATHOSOTNABCA`, **reset at every line**.  The per-line reset is
  the part that hides it: read as one stream the key looks aperiodic.
  `cicada/outer.py` recovers the key from the cipher/plaintext pair and the
  round trip is exact over all 9 525 letters.
- **The 2013 hidden-service byte strings.**  Four 256-byte blobs, one of them
  derived from Liber Primus pages 49-51.  They are not RSA moduli - three of
  the four are even - pairwise GCDs are trivial, trial division to 1e5 leaves
  2013+ bit remainders, and byte entropy (7.12-7.28 bits) is what 256 random
  bytes give.  Consistent with signature or ciphertext material, not with
  anything factorable.

## Layout

```
cicada/gematria.py   29-rune alphabet: runes, latin, primes, atbash
cicada/corpus.py     parser for the master transcription (18 segments, 74 pages)
cicada/ciphers.py    atbash, shift, affine, Vigenere, autokey, transposition,
                     prime and totient key streams
cicada/analysis.py   IoC, periodic IoC, chi-squared, doublets, rune n-gram model
cicada/decoder.py    beam decoder for the ambiguous F rule
cicada/solved.py     executable reproductions of all nine solved sections
cicada/attack.py     the attack battery
cicada/outer.py      2012 Mabinogion cipher, 2013 byte strings
solve_cicada.py      CLI
tests/test_cicada.py 27 tests
```

`cicada/data/` holds the English corpus the rune n-gram model trains on
(Norvig's `big.txt`) and two word lists; they are downloaded, not authored here.
