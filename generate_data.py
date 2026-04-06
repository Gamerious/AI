#!/usr/bin/env python3
"""
Massive Training Data Generator for NEXUS

Generates hundreds of thousands of samples across all 24 tasks,
saves them as memory-mapped .pt files for efficient loading.

Usage:
    python3 generate_data.py               # Generate all datasets
    python3 generate_data.py --quick       # Quick test (1K per task)
"""

import sys
import os
import argparse
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(line_buffering=True)

import torch
from training.tasks import CombinedTaskDataset


def generate_and_save(name, tasks, n_per_task, seq_len, vocab_size, output_dir):
    """Generate dataset and save as .pt file."""
    print(f"\n  Generating '{name}': {n_per_task} samples/task, {len(tasks)} tasks...", flush=True)
    t0 = time.time()

    dataset = CombinedTaskDataset(
        n_samples_per_task=n_per_task,
        seq_len=seq_len,
        vocab_size=vocab_size,
        tasks=tasks,
    )

    # Collect all samples into tensors for efficient storage
    all_input_ids = []
    all_labels = []

    for i in range(len(dataset)):
        sample = dataset[i]
        all_input_ids.append(sample["input_ids"])
        all_labels.append(sample["labels"])

    data = {
        "input_ids": torch.stack(all_input_ids),
        "labels": torch.stack(all_labels),
        "n_samples": len(dataset),
        "seq_len": seq_len,
        "vocab_size": vocab_size,
        "tasks": tasks,
        "n_per_task": n_per_task,
    }

    path = os.path.join(output_dir, f"{name}.pt")
    torch.save(data, path)
    size_mb = os.path.getsize(path) / (1024 * 1024)
    elapsed = time.time() - t0

    print(f"    Saved: {path} ({len(dataset):,} samples, {size_mb:.1f} MB, {elapsed:.1f}s)", flush=True)
    return len(dataset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Quick test with small data")
    parser.add_argument("--seq-len", type=int, default=128, help="Sequence length")
    parser.add_argument("--vocab-size", type=int, default=200, help="Vocabulary size")
    args = parser.parse_args()

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(output_dir, exist_ok=True)

    SEQ_LEN = args.seq_len
    VOCAB = args.vocab_size

    all_tasks = list(CombinedTaskDataset.TASK_CLASSES.keys())

    # Task groups
    math_tasks = ["addition", "subtraction", "multiplication", "modular_arithmetic",
                  "number_comparison", "sequence_sum", "fibonacci"]
    logic_tasks = ["boolean_logic", "implication_chain"]
    pattern_tasks = ["arithmetic_sequence", "geometric_sequence", "palindrome",
                     "pattern_repetition", "mirror_sequence"]
    memory_tasks = ["delayed_copy", "selective_copy", "token_counting", "associative_recall"]
    sequence_tasks = ["sorting", "deduplication", "rotation"]
    language_tasks = ["bracket_matching", "substitution_cipher", "grammar_pattern"]

    hard_tasks = math_tasks + ["sorting", "deduplication", "palindrome",
                                "arithmetic_sequence", "token_counting", "associative_recall"]

    if args.quick:
        n_all, n_hard, n_math = 1000, 1000, 1000
    else:
        n_all = 20000    # 20K per task × 24 tasks ≈ 480K samples
        n_hard = 30000   # 30K per task × ~12 tasks ≈ 360K samples
        n_math = 50000   # 50K per task × 7 tasks ≈ 350K samples

    print("=" * 70)
    print("  NEXUS Massive Data Generator")
    print(f"  Seq len: {SEQ_LEN}, Vocab: {VOCAB}")
    print(f"  Output: {output_dir}/")
    print("=" * 70, flush=True)

    total = 0
    t_start = time.time()

    # 1. Full multi-task dataset (for foundation training)
    total += generate_and_save(
        "train_all_tasks", all_tasks, n_all, SEQ_LEN, VOCAB, output_dir
    )

    # 2. Hard tasks focused (for fine-tuning)
    total += generate_and_save(
        "train_hard_tasks", hard_tasks, n_hard, SEQ_LEN, VOCAB, output_dir
    )

    # 3. Math focused (for arithmetic specialization)
    total += generate_and_save(
        "train_math", math_tasks, n_math, SEQ_LEN, VOCAB, output_dir
    )

    # 4. Logic + Pattern (for reasoning)
    total += generate_and_save(
        "train_logic_pattern", logic_tasks + pattern_tasks, n_all, SEQ_LEN, VOCAB, output_dir
    )

    # 5. Memory + Sequence (for memory tasks)
    total += generate_and_save(
        "train_memory_sequence", memory_tasks + sequence_tasks, n_all, SEQ_LEN, VOCAB, output_dir
    )

    # 6. Language tasks
    total += generate_and_save(
        "train_language", language_tasks, n_all, SEQ_LEN, VOCAB, output_dir
    )

    # 7. Evaluation set (smaller, fresh samples)
    total += generate_and_save(
        "eval_all_tasks", all_tasks, 500, SEQ_LEN, VOCAB, output_dir
    )

    # 8. Short sequence versions (seq_len=64) for curriculum learning
    total += generate_and_save(
        "train_all_short", all_tasks, n_all // 2, 64, VOCAB, output_dir
    )

    elapsed = time.time() - t_start

    print(f"\n{'=' * 70}")
    print(f"  DONE!")
    print(f"  Total samples: {total:,}")
    print(f"  Time: {elapsed / 60:.1f} minutes")
    print(f"  Files in: {output_dir}/")
    print(f"{'=' * 70}", flush=True)

    # List files
    for f in sorted(os.listdir(output_dir)):
        if f.endswith('.pt'):
            size = os.path.getsize(os.path.join(output_dir, f)) / (1024 * 1024)
            print(f"    {f:40s} {size:>8.1f} MB")


if __name__ == "__main__":
    main()
