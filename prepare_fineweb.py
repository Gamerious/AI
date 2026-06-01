#!/usr/bin/env python3
"""
Prepare web-scale pretraining data for NEXUS-LM (Stage 1 scale-up).

Streams FineWeb-Edu (high-quality educational web text), trains a fast Rust
BPE tokenizer on it, and writes flat uint16 token .bin files (nanoGPT-style)
for memmap training. Designed so the expensive GPU never waits on the CPU:
tokenization happens once, up front, with all cores (encode_batch).

Extra deps (NOT part of the zero-dep base - install on the pod):
    pip install datasets tokenizers numpy

Usage (on the RunPod box, inside tmux):
    # First real run - small + cheap, proves the pipeline end to end:
    python prepare_fineweb.py --tokens 300M  --vocab-size 32000

    # Full Stage-1 run (~2B tokens for a ~100M model, Chinchilla-ish):
    python prepare_fineweb.py --tokens 2B    --vocab-size 32000

Output (default lm_data_web/):
    train.bin, val.bin              flat uint16 token streams
    tokenizer/fast_tokenizer.json   the trained BPE
    meta.json                       vocab_size, seq_len hint, token counts

Then train on it:
    python train_nexus_lm.py --config base --data-dir lm_data_web --cisa-v2
"""

import os
import sys
import json
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(line_buffering=True)

import numpy as np

from nexus.lm.fast_tokenizer import FastTokenizer


OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lm_data_web")


def parse_count(s: str) -> int:
    """Accept 2B / 300M / 2_000_000_000 / 5e6 style counts."""
    s = str(s).strip().replace("_", "").upper()
    mult = 1
    if s.endswith("B"):
        mult, s = 1_000_000_000, s[:-1]
    elif s.endswith("M"):
        mult, s = 1_000_000, s[:-1]
    elif s.endswith("K"):
        mult, s = 1_000, s[:-1]
    return int(float(s) * mult)


def stream_docs(sample: str, text_key: str = "text"):
    """Yield document strings from FineWeb-Edu via streaming (no full download)."""
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError("prepare_fineweb needs 'datasets': pip install datasets") from e

    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu", name=sample, split="train", streaming=True
    )
    for ex in ds:
        t = ex.get(text_key)
        if t:
            yield t


def main():
    ap = argparse.ArgumentParser(description="Build web-scale .bin data for NEXUS-LM")
    ap.add_argument("--tokens", type=parse_count, default="2B",
                    help="Target TRAIN tokens (e.g. 300M, 2B). ~20x model params is Chinchilla-optimal.")
    ap.add_argument("--vocab-size", type=int, default=32000)
    ap.add_argument("--sample", default="sample-10BT",
                    help="FineWeb-Edu config: sample-10BT / sample-100BT / sample-350BT / CC-MAIN-*")
    ap.add_argument("--val-tokens", type=parse_count, default="5M")
    ap.add_argument("--seq-len", type=int, default=1024, help="Training context hint stored in meta.json")
    ap.add_argument("--tok-train-docs", type=int, default=400_000,
                    help="How many docs to train the tokenizer on (more = slower, marginal gain)")
    ap.add_argument("--chars-per-token", type=float, default=4.5,
                    help="Estimate used to size the streaming buffer")
    ap.add_argument("--out", default=OUT_DIR)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    tok_dir = os.path.join(args.out, "tokenizer")

    target_tokens = args.tokens + args.val_tokens
    target_chars = int(target_tokens * args.chars_per_token)

    print("=" * 70)
    print("  NEXUS-LM Web Data Preparation (FineWeb-Edu)")
    print("=" * 70)
    print(f"  sample:       {args.sample}")
    print(f"  target:       {args.tokens:,} train + {args.val_tokens:,} val tokens")
    print(f"  vocab_size:   {args.vocab_size}", flush=True)

    # ---- Pass 1: stream docs into RAM until we have enough text -----------
    print(f"\n==> Streaming until ~{target_chars/1e9:.1f}B chars "
          f"(~{target_tokens/1e9:.2f}B tokens)...", flush=True)
    docs = []
    n_chars = 0
    t0 = time.time()
    for t in stream_docs(args.sample):
        docs.append(t)
        n_chars += len(t)
        if len(docs) % 100_000 == 0:
            rate = n_chars / max(1e-9, time.time() - t0) / 1e6
            print(f"    {len(docs):,} docs | {n_chars/1e9:.2f}B chars | {rate:.0f} MB/s", flush=True)
        if n_chars >= target_chars:
            break
    print(f"  collected {len(docs):,} docs, {n_chars/1e9:.2f}B chars in {time.time()-t0:.0f}s", flush=True)

    if not docs:
        print("  ERROR: no documents streamed - check --sample / network.", flush=True)
        sys.exit(1)

    # ---- Train tokenizer on a subset --------------------------------------
    n_tok_train = min(args.tok_train_docs, len(docs))
    print(f"\n==> Training fast BPE (vocab={args.vocab_size}) on {n_tok_train:,} docs...", flush=True)
    t1 = time.time()
    tok = FastTokenizer.train(iter(docs[:n_tok_train]), vocab_size=args.vocab_size, length=n_tok_train)
    tok.save(tok_dir)
    assert tok.vocab_size <= 65535, f"vocab {tok.vocab_size} too large for uint16"
    print(f"  done in {time.time()-t1:.0f}s, vocab_size={tok.vocab_size}", flush=True)

    # ---- Encode everything (batched, multi-core) --------------------------
    print(f"\n==> Encoding {len(docs):,} docs -> uint16...", flush=True)
    t2 = time.time()
    eos = tok.eos_id
    chunks = []
    total = 0
    BATCH = 2000
    for i in range(0, len(docs), BATCH):
        for ids in tok.encode_batch(docs[i:i + BATCH]):
            ids.append(eos)
            chunks.append(np.asarray(ids, dtype=np.uint16))
            total += len(ids)
        if (i // BATCH) % 25 == 0:
            done = min(i + BATCH, len(docs))
            print(f"    {done:,}/{len(docs):,} docs | {total/1e9:.3f}B tokens | "
                  f"{time.time()-t2:.0f}s", flush=True)
        if total >= target_tokens:
            break
    arr = np.concatenate(chunks)
    del chunks, docs
    print(f"  encoded {len(arr):,} tokens in {time.time()-t2:.0f}s", flush=True)

    # ---- Split + write ----------------------------------------------------
    val_n = min(args.val_tokens, len(arr) // 20)
    val = arr[-val_n:]
    train = arr[:-val_n] if val_n > 0 else arr
    train.tofile(os.path.join(args.out, "train.bin"))
    val.tofile(os.path.join(args.out, "val.bin"))

    meta = {
        "vocab_size": int(tok.vocab_size),
        "seq_len": int(args.seq_len),
        "n_train_tokens": int(len(train)),
        "n_val_tokens": int(len(val)),
        "dtype": "uint16",
        "tokenizer": "fast",
        "source": f"fineweb-edu/{args.sample}",
    }
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("\n" + "=" * 70)
    print(f"  DONE  train={len(train):,} tok | val={len(val):,} tok | vocab={tok.vocab_size}")
    print(f"  -> {args.out}/ (train.bin, val.bin, tokenizer/, meta.json)")
    print(f"\n  Train on it:")
    print(f"    python train_nexus_lm.py --config base --data-dir {os.path.basename(args.out)} --cisa-v2")
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
