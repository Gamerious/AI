#!/usr/bin/env python3
"""
Prepare training data for NEXUS-LM.

Downloads TinyStories dataset (small, clean, proven to work at small scale),
trains a BPE tokenizer, and creates tokenized .pt files.

TinyStories: https://arxiv.org/abs/2305.07759
- Specifically designed for training small LMs (<100M params)
- Simple English stories, clean data
- Models trained on it can generate coherent text

Usage:
    python3 prepare_lm_data.py                    # Full pipeline
    python3 prepare_lm_data.py --vocab-size 16000 # Custom vocab
    python3 prepare_lm_data.py --skip-download     # If data already downloaded
    python3 prepare_lm_data.py --custom-data mytext.txt  # Use your own text
"""

import sys
import os
import argparse
import json
import time
import gzip
import io

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(line_buffering=True)

import torch
import requests

from nexus.lm.tokenizer import BPETokenizer


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lm_data")


def download_tinystories(data_dir):
    """Download TinyStories dataset from HuggingFace."""
    os.makedirs(data_dir, exist_ok=True)

    # TinyStories is available as a simple text dataset
    # We'll download the train and validation splits
    base_url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main"

    for split in ["train", "validation"]:
        output_file = os.path.join(data_dir, f"tinystories_{split}.txt")
        if os.path.exists(output_file):
            size_mb = os.path.getsize(output_file) / (1024 * 1024)
            print(f"  {split}: already exists ({size_mb:.0f} MB)", flush=True)
            continue

        url = f"{base_url}/TinyStoriesV2-GPT4-{split}.txt"
        print(f"  Downloading {split} from HuggingFace...", flush=True)
        print(f"  URL: {url}", flush=True)

        try:
            response = requests.get(url, stream=True, timeout=300)
            response.raise_for_status()

            total = int(response.headers.get('content-length', 0))
            downloaded = 0

            with open(output_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        pct = downloaded / total * 100
                        print(f"\r    {downloaded / 1024 / 1024:.0f} / {total / 1024 / 1024:.0f} MB ({pct:.0f}%)", end="", flush=True)

            print(flush=True)
            size_mb = os.path.getsize(output_file) / (1024 * 1024)
            print(f"    Saved: {output_file} ({size_mb:.0f} MB)", flush=True)

        except Exception as e:
            print(f"\n  Download failed: {e}", flush=True)
            print(f"  You can manually download from: {url}", flush=True)
            print(f"  Or use --custom-data with your own text file", flush=True)
            if os.path.exists(output_file):
                os.remove(output_file)
            return False

    return True


def load_texts(data_dir, split="train", max_stories=None):
    """Load TinyStories texts."""
    path = os.path.join(data_dir, f"tinystories_{split}.txt")
    if not os.path.exists(path):
        print(f"  ERROR: {path} not found!", flush=True)
        return []

    print(f"  Loading {path}...", flush=True)
    stories = []
    current_story = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line == "<|endoftext|>":
                if current_story:
                    stories.append(" ".join(current_story))
                    current_story = []
                    if max_stories and len(stories) >= max_stories:
                        break
            else:
                if line:
                    current_story.append(line)

    if current_story:
        stories.append(" ".join(current_story))

    print(f"  Loaded {len(stories):,} stories", flush=True)
    return stories


def load_custom_texts(path, max_chars=None):
    """Load custom text file (one document per empty line, or entire file)."""
    print(f"  Loading custom data: {path}...", flush=True)

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
        if max_chars:
            content = content[:max_chars]

    # Split on double newlines (paragraph/document boundaries)
    texts = [t.strip() for t in content.split("\n\n") if t.strip()]
    print(f"  Loaded {len(texts):,} text segments ({len(content):,} chars)", flush=True)
    return texts


def tokenize_and_save(tokenizer, texts, output_path, seq_len=512, split_name="train"):
    """Tokenize texts and save as .pt file."""
    print(f"  Tokenizing {len(texts):,} texts (seq_len={seq_len})...", flush=True)

    all_tokens = []
    for text in texts:
        tokens = tokenizer.encode(text, add_bos=True, add_eos=True)
        all_tokens.extend(tokens)

    print(f"  Total tokens: {len(all_tokens):,}", flush=True)

    # Create fixed-length sequences (for efficient batched training)
    n_seqs = len(all_tokens) // seq_len
    if n_seqs == 0:
        print(f"  WARNING: Not enough tokens for seq_len={seq_len}!", flush=True)
        return 0

    # Reshape into sequences
    all_tokens = all_tokens[:n_seqs * seq_len]
    token_tensor = torch.tensor(all_tokens, dtype=torch.long).view(n_seqs, seq_len)

    # For language modeling: input = tokens, labels = tokens (shifted internally in model)
    data = {
        "input_ids": token_tensor,
        "n_sequences": n_seqs,
        "seq_len": seq_len,
        "total_tokens": len(all_tokens),
        "split": split_name,
    }

    torch.save(data, output_path)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Saved: {output_path} ({n_seqs:,} sequences, {size_mb:.1f} MB)", flush=True)
    return n_seqs


def main():
    parser = argparse.ArgumentParser(description="Prepare NEXUS-LM training data")
    parser.add_argument("--vocab-size", type=int, default=16000, help="BPE vocabulary size")
    parser.add_argument("--seq-len", type=int, default=512, help="Sequence length")
    parser.add_argument("--max-stories", type=int, default=None, help="Limit number of stories")
    parser.add_argument("--skip-download", action="store_true", help="Skip download step")
    parser.add_argument("--custom-data", type=str, default=None, help="Use custom text file instead of TinyStories")
    parser.add_argument("--tokenizer-train-size", type=int, default=50000, help="Number of texts to train tokenizer on")
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)
    tokenizer_dir = os.path.join(DATA_DIR, "tokenizer")

    print("=" * 70)
    print("  NEXUS-LM Data Preparation")
    print("=" * 70, flush=True)

    t_start = time.time()

    # ============================================================
    # Step 1: Get text data
    # ============================================================
    if args.custom_data:
        print(f"\n  Step 1: Loading custom data...", flush=True)
        train_texts = load_custom_texts(args.custom_data)
        val_texts = train_texts[-len(train_texts)//10:]  # Last 10% for validation
        train_texts = train_texts[:-len(val_texts)]
    else:
        print(f"\n  Step 1: Getting TinyStories data...", flush=True)
        if not args.skip_download:
            success = download_tinystories(DATA_DIR)
            if not success:
                print("\n  Fallback: creating synthetic training data...", flush=True)
                train_texts = create_synthetic_data()
                val_texts = train_texts[-500:]
                train_texts = train_texts[:-500]
            else:
                train_texts = load_texts(DATA_DIR, "train", args.max_stories)
                val_texts = load_texts(DATA_DIR, "validation", max_stories=5000)
        else:
            train_texts = load_texts(DATA_DIR, "train", args.max_stories)
            val_texts = load_texts(DATA_DIR, "validation", max_stories=5000)

    if not train_texts:
        print("  ERROR: No training data available!", flush=True)
        print("  Use --custom-data <file.txt> to provide your own text", flush=True)
        sys.exit(1)

    # ============================================================
    # Step 2: Train tokenizer
    # ============================================================
    print(f"\n  Step 2: Training BPE tokenizer (vocab={args.vocab_size})...", flush=True)

    tokenizer = BPETokenizer()
    # Use subset for tokenizer training (faster)
    tok_train = train_texts[:args.tokenizer_train_size]
    tokenizer.train(tok_train, target_vocab_size=args.vocab_size)
    tokenizer.save(tokenizer_dir)

    # Quick test
    test_text = "Once upon a time, there was a little cat named Whiskers."
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    print(f"\n  Tokenizer test:")
    print(f"    Input:   '{test_text}'")
    print(f"    Tokens:  {len(encoded)} IDs")
    print(f"    Decoded: '{decoded}'")
    print(f"    Match:   {'YES' if decoded == test_text else 'NO'}", flush=True)

    # ============================================================
    # Step 3: Tokenize and save
    # ============================================================
    print(f"\n  Step 3: Tokenizing and saving...", flush=True)

    n_train = tokenize_and_save(
        tokenizer, train_texts,
        os.path.join(DATA_DIR, "train.pt"),
        seq_len=args.seq_len, split_name="train"
    )

    n_val = tokenize_and_save(
        tokenizer, val_texts,
        os.path.join(DATA_DIR, "val.pt"),
        seq_len=args.seq_len, split_name="val"
    )

    # ============================================================
    # Summary
    # ============================================================
    elapsed = time.time() - t_start

    print(f"\n{'=' * 70}")
    print(f"  DONE!")
    print(f"  Tokenizer:     {tokenizer_dir} (vocab={tokenizer.vocab_size})")
    print(f"  Train data:    {n_train:,} sequences × {args.seq_len} tokens")
    print(f"  Val data:      {n_val:,} sequences × {args.seq_len} tokens")
    print(f"  Total tokens:  {(n_train + n_val) * args.seq_len:,}")
    print(f"  Time:          {elapsed/60:.1f} minutes")
    print(f"{'=' * 70}")
    print(f"\n  Next: python3 train_nexus_lm.py", flush=True)


def create_synthetic_data():
    """Create synthetic stories if download fails."""
    import random
    random.seed(42)

    templates = [
        "Once upon a time, there was a {adj} {animal} named {name}. {name} loved to {verb} in the {place}. One day, {name} found a {object}. It was very {adj2}. {name} was so happy!",
        "There was a {adj} {animal} who lived in a {place}. Every day, the {animal} would {verb}. One morning, something special happened. A {adj2} {object} appeared! The {animal} was amazed.",
        "{name} was a little {animal}. {name} had a {adj} {object}. One day, {name} went to the {place} to {verb}. There, {name} met a {adj2} friend. They played together all day.",
        "In a {adj} {place}, there lived a {animal}. The {animal} wanted to find a {object}. So the {animal} started to {verb}. After a long journey, the {animal} found it! The {animal} was very {adj2}.",
        "A {adj} {animal} named {name} loved to {verb}. Every day, {name} would go to the {place}. One day, {name} saw a {adj2} {object}. It was the best day ever for {name}!",
    ]

    names = ["Luna", "Max", "Bella", "Charlie", "Daisy", "Oliver", "Lucy", "Milo", "Rosie", "Leo", "Lily", "Buddy", "Sophie", "Jack", "Ruby"]
    animals = ["cat", "dog", "rabbit", "bird", "bear", "fox", "mouse", "fish", "turtle", "frog", "owl", "deer", "duck", "pig", "horse"]
    adjs = ["little", "big", "happy", "brave", "curious", "shy", "friendly", "gentle", "clever", "kind", "funny", "sweet", "tiny", "young", "old"]
    adjs2 = ["beautiful", "magical", "wonderful", "special", "amazing", "sparkly", "golden", "colorful", "warm", "soft"]
    verbs = ["play", "sing", "dance", "run", "swim", "jump", "explore", "dream", "laugh", "read", "paint", "cook", "garden", "fly", "climb"]
    places = ["garden", "forest", "park", "meadow", "beach", "mountain", "river", "village", "castle", "farm", "library", "school", "playground"]
    objects = ["flower", "stone", "star", "rainbow", "treasure", "book", "shell", "feather", "butterfly", "key", "hat", "ball", "cookie", "gift"]

    stories = []
    for _ in range(10000):
        template = random.choice(templates)
        story = template.format(
            name=random.choice(names),
            animal=random.choice(animals),
            adj=random.choice(adjs),
            adj2=random.choice(adjs2),
            verb=random.choice(verbs),
            place=random.choice(places),
            object=random.choice(objects),
        )
        stories.append(story)

    print(f"  Generated {len(stories)} synthetic stories", flush=True)
    return stories


if __name__ == "__main__":
    main()
