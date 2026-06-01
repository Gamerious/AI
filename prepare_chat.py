#!/usr/bin/env python3
"""
Prepare conversational SFT data for NEXUS-LM (Stage 2: make it a chatbot).

Streams a chat dataset (default: SmolTalk, built for small models), renders it
into a simple ChatML template, and writes a packed uint16 token stream plus a
uint8 loss-mask. Only ASSISTANT tokens are trainable (mask=1); system/user
turns and the role headers are masked out (-100) so the model learns to
*answer*, not to parrot the prompt.

IMPORTANT: uses the SAME tokenizer as the pretrained base (default
lm_data_web/tokenizer) - the vocab must match the model you fine-tune.

Deps: datasets, tokenizers, numpy  (same as the web pipeline)

Usage (after a base model is pretrained):
    python prepare_chat.py --tokens 200M
    python train_nexus_lm.py --data-dir lm_data_chat --resume <base_model.pt> --lr 1e-5 --max-steps 3000

Template per turn:
    <|system|>\n{content}\n            (masked)
    <|user|>\n{content}\n              (masked)
    <|assistant|>\n{content}<|eot|>\n  (header masked, content + <|eot|> trainable)
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


OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lm_data_chat")
DEFAULT_TOK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lm_data_web", "tokenizer")


def parse_count(s):
    s = str(s).strip().replace("_", "").upper()
    mult = 1
    if s.endswith("B"):
        mult, s = 1_000_000_000, s[:-1]
    elif s.endswith("M"):
        mult, s = 1_000_000, s[:-1]
    elif s.endswith("K"):
        mult, s = 1_000, s[:-1]
    return int(float(s) * mult)


def stream_chats(dataset, config, split):
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError("prepare_chat needs 'datasets': pip install datasets") from e
    kw = {"split": split, "streaming": True}
    if config:
        kw["name"] = config
    ds = load_dataset(dataset, **kw)
    for ex in ds:
        msgs = ex.get("messages") or ex.get("conversations")
        if msgs:
            yield msgs


def normalize_msg(m):
    """Return (role, content) from the various field conventions."""
    role = m.get("role") or m.get("from") or ""
    content = m.get("content") or m.get("value") or ""
    role = {"human": "user", "gpt": "assistant", "assistant": "assistant",
            "user": "user", "system": "system"}.get(role, role)
    return role, content


def main():
    ap = argparse.ArgumentParser(description="Build chat SFT data (tokens.bin + mask.bin)")
    ap.add_argument("--tokens", type=parse_count, default="200M", help="Target total tokens")
    ap.add_argument("--val-tokens", type=parse_count, default="2M")
    ap.add_argument("--seq-len", type=int, default=1024, help="Must be <= base model max_seq_len")
    ap.add_argument("--dataset", default="HuggingFaceTB/smoltalk")
    ap.add_argument("--config", default="all", help="dataset config name (SmolTalk: 'all')")
    ap.add_argument("--split", default="train")
    ap.add_argument("--tokenizer", default=DEFAULT_TOK, help="dir with fast_tokenizer.json (same as base)")
    ap.add_argument("--out", default=OUT_DIR)
    args = ap.parse_args()

    if not FastTokenizer.exists(args.tokenizer):
        print(f"  ERROR: tokenizer not found at {args.tokenizer}.")
        print(f"  Run prepare_fineweb.py first (it writes the shared tokenizer), or pass --tokenizer.")
        sys.exit(1)

    os.makedirs(args.out, exist_ok=True)
    tok = FastTokenizer.load(args.tokenizer)
    tok.save(os.path.join(args.out, "tokenizer"))  # self-contained: same tokenizer as base
    print("=" * 70)
    print("  NEXUS-LM Chat (SFT) Data Preparation")
    print("=" * 70)
    print(f"  dataset:    {args.dataset} ({args.config})")
    print(f"  tokenizer:  {args.tokenizer} (vocab {tok.vocab_size})")
    print(f"  target:     {args.tokens:,} tokens, seq_len={args.seq_len}", flush=True)
    assert tok.vocab_size <= 65535

    # Pre-encode the static template pieces once.
    eot = tok.encode("<|eot|>\n")
    hdr = {r: tok.encode(f"<|{r}|>\n") for r in ("system", "user", "assistant")}

    all_tok = []
    all_mask = []
    total = 0
    n_conv = 0
    n_assist_tok = 0
    t0 = time.time()
    gen = stream_chats(args.dataset, args.config, args.split)
    for msgs in gen:
        for m in msgs:
            role, content = normalize_msg(m)
            if not content:
                continue
            h = hdr.get(role, tok.encode(f"<|{role}|>\n"))
            all_tok.extend(h)
            all_mask.extend([0] * len(h))
            c = tok.encode(content)
            if role == "assistant":
                c = c + eot
                all_tok.extend(c)
                all_mask.extend([1] * len(c))   # trainable
                n_assist_tok += len(c)
            else:
                nl = tok.encode("\n")
                all_tok.extend(c + nl)
                all_mask.extend([0] * (len(c) + len(nl)))
        total = len(all_tok)
        n_conv += 1
        if n_conv % 20000 == 0:
            print(f"    {n_conv:,} convs | {total/1e6:.1f}M tokens "
                  f"({100*n_assist_tok/max(1,total):.0f}% trainable) | {time.time()-t0:.0f}s", flush=True)
        if total >= args.tokens + args.val_tokens:
            break

    tokens = np.asarray(all_tok, dtype=np.uint16)
    mask = np.asarray(all_mask, dtype=np.uint8)
    del all_tok, all_mask
    print(f"  packed {len(tokens):,} tokens from {n_conv:,} convs "
          f"({100*n_assist_tok/max(1,len(tokens)):.0f}% trainable)", flush=True)

    # val split from the tail
    val_n = min(args.val_tokens, len(tokens) // 20)
    def dump(name, t, m):
        t.tofile(os.path.join(args.out, f"{name}_tokens.bin"))
        m.tofile(os.path.join(args.out, f"{name}_mask.bin"))

    # Single train/val pair using the canonical filenames the dataset expects.
    train_tok, train_mask = tokens[:-val_n], mask[:-val_n]
    val_tok, val_mask = tokens[-val_n:], mask[-val_n:]
    train_tok.tofile(os.path.join(args.out, "tokens.bin"))
    train_mask.tofile(os.path.join(args.out, "mask.bin"))
    val_tok.tofile(os.path.join(args.out, "val_tokens.bin"))
    val_mask.tofile(os.path.join(args.out, "val_mask.bin"))

    meta = {
        "vocab_size": int(tok.vocab_size),
        "seq_len": int(args.seq_len),
        "n_train_tokens": int(len(train_tok)),
        "n_val_tokens": int(len(val_tok)),
        "trainable_frac": round(n_assist_tok / max(1, len(tokens)), 3),
        "dtype": "uint16",
        "tokenizer": "fast",
        "kind": "chat",
        "source": f"{args.dataset}/{args.config}",
    }
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("\n" + "=" * 70)
    print(f"  DONE  train={len(train_tok):,} tok | val={len(val_tok):,} tok | "
          f"trainable={meta['trainable_frac']*100:.0f}%")
    print(f"  -> {args.out}/ (tokens.bin, mask.bin, val_*.bin, meta.json)")
    print(f"\n  SFT on it (resume the pretrained base):")
    print(f"    python train_nexus_lm.py --data-dir {os.path.basename(args.out)} "
          f"--resume <base.pt> --lr 1e-5 --max-steps 3000")
    print("=" * 70, flush=True)

    sys.stdout.flush()
    os._exit(0)  # avoid datasets streaming-thread std::terminate at teardown


if __name__ == "__main__":
    main()
