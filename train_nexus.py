#!/usr/bin/env python3
"""
NEXUS Training & Evaluation Pipeline

Trains NEXUS on diverse tasks and rigorously tests autoregressive generation.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import math
import time

from nexus.model import NexusModel, NexusConfig
from training.tasks import CombinedTaskDataset, BOS, SEP, EOS, PAD, OFFSET
from training.evaluator import TaskEvaluator


def get_lr(step, warmup, max_steps, max_lr, min_lr=1e-6):
    if step < warmup:
        return max_lr * step / max(1, warmup)
    progress = (step - warmup) / max(1, max_steps - warmup)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


def train_phase(model, dataset, max_steps, lr=3e-4, warmup=200, batch_size=32, log_every=250):
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.95))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    model.train()

    step = 0
    running_loss = running_acc = n_acc = 0
    best_acc = 0

    while step < max_steps:
        for batch in loader:
            if step >= max_steps:
                break

            current_lr = get_lr(step, warmup, max_steps, lr)
            for pg in optimizer.param_groups:
                pg["lr"] = current_lr

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs["loss"]

            with torch.no_grad():
                preds = outputs["logits"].argmax(dim=-1)
                mask = labels != -100
                acc = (preds[mask] == labels[mask]).float().mean().item() * 100 if mask.any() else 0

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_loss += loss.item()
            running_acc += acc
            n_acc += 1

            if step % log_every == 0 and step > 0:
                avg_acc = running_acc / n_acc
                best_acc = max(best_acc, avg_acc)
                print(
                    f"  Step {step:>5}/{max_steps} | "
                    f"Loss: {running_loss/n_acc:.4f} | "
                    f"Acc: {avg_acc:.1f}% | "
                    f"Best: {best_acc:.1f}% | "
                    f"LR: {current_lr:.2e}"
                )
                running_loss = running_acc = n_acc = 0

            step += 1

    return best_acc


def generate(model, prefix_tokens, max_new=10, seq_len=64):
    """Pure greedy autoregressive generation. No tricks, no min_new."""
    model.eval()
    device = next(model.parameters()).device
    tokens = list(prefix_tokens)

    for _ in range(max_new):
        if len(tokens) >= seq_len:
            break

        padded = tokens + [PAD] * (seq_len - len(tokens))
        inp = torch.tensor([padded[:seq_len]], dtype=torch.long, device=device)

        with torch.no_grad():
            out = model(input_ids=inp)

        logits = out["logits"][0, len(tokens) - 1]
        pred = logits.argmax().item()

        if pred == EOS or pred == PAD:
            break

        tokens.append(pred)

    return tokens[len(prefix_tokens):]


def test_generation(model, config, label=""):
    print(f"\n{'='*70}")
    print(f"  NEXUS GENERATION TEST {label}")
    print(f"{'='*70}")

    TRUE, FALSE = OFFSET + 11, OFFSET + 12
    AND_OP, OR_OP, XOR_OP, NOT_OP = OFFSET + 20, OFFSET + 21, OFFSET + 22, OFFSET + 23

    tests = []

    # === SIMPLE (single-token output) ===

    # Single-digit addition
    for a, b in [(2, 3), (5, 3), (1, 1), (4, 5), (7, 2), (6, 3)]:
        r = a + b
        if r < 10:
            a_tok = [a + OFFSET]
            b_tok = [b + OFFSET]
            r_tok = [r + OFFSET]
            tests.append((f"Add {a}+{b}={r}", [BOS] + a_tok + [SEP] + b_tok + [SEP], r_tok))

    # Boolean logic
    tests += [
        ("AND T,T=T", [BOS, AND_OP, TRUE, TRUE, SEP], [TRUE]),
        ("AND T,F=F", [BOS, AND_OP, TRUE, FALSE, SEP], [FALSE]),
        ("AND F,F=F", [BOS, AND_OP, FALSE, FALSE, SEP], [FALSE]),
        ("OR F,T=T", [BOS, OR_OP, FALSE, TRUE, SEP], [TRUE]),
        ("OR F,F=F", [BOS, OR_OP, FALSE, FALSE, SEP], [FALSE]),
        ("XOR T,T=F", [BOS, XOR_OP, TRUE, TRUE, SEP], [FALSE]),
        ("XOR T,F=T", [BOS, XOR_OP, TRUE, FALSE, SEP], [TRUE]),
        ("NOT T=F", [BOS, NOT_OP, TRUE, SEP], [FALSE]),
        ("NOT F=T", [BOS, NOT_OP, FALSE, SEP], [TRUE]),
    ]

    # Palindrome check
    tests += [
        ("Pal [1,2,1]=T", [BOS, 1+OFFSET, 2+OFFSET, 1+OFFSET, SEP], [TRUE]),
        ("Pal [1,2,3]=F", [BOS, 1+OFFSET, 2+OFFSET, 3+OFFSET, SEP], [FALSE]),
        ("Pal [3,3]=T", [BOS, 3+OFFSET, 3+OFFSET, SEP], [TRUE]),
        ("Pal [1,2]=F", [BOS, 1+OFFSET, 2+OFFSET, SEP], [FALSE]),
    ]

    # Bracket matching
    OPEN, CLOSE = OFFSET + 30, OFFSET + 31
    tests += [
        ("Bkt ()=T", [BOS, OPEN, CLOSE, SEP], [TRUE]),
        ("Bkt (())=T", [BOS, OPEN, OPEN, CLOSE, CLOSE, SEP], [TRUE]),
        ("Bkt )(=F", [BOS, CLOSE, OPEN, SEP], [FALSE]),
        ("Bkt ((=F", [BOS, OPEN, OPEN, SEP], [FALSE]),
    ]

    # === MEDIUM (multi-token output) ===

    # Two-digit addition
    for a, b in [(12, 7), (25, 30), (33, 44), (50, 25)]:
        a_tok = [int(d) + OFFSET for d in str(a)]
        b_tok = [int(d) + OFFSET for d in str(b)]
        r_tok = [int(d) + OFFSET for d in str(a + b)]
        tests.append((f"Add {a}+{b}={a+b}", [BOS] + a_tok + [SEP] + b_tok + [SEP], r_tok))

    # Subtraction
    for a, b in [(9, 3), (10, 5), (50, 25)]:
        a_tok = [int(d) + OFFSET for d in str(a)]
        b_tok = [int(d) + OFFSET for d in str(b)]
        r_tok = [int(d) + OFFSET for d in str(a - b)]
        tests.append((f"Sub {a}-{b}={a-b}", [BOS] + a_tok + [SEP] + b_tok + [SEP], r_tok))

    # Multiplication
    for a, b in [(3, 4), (2, 5), (7, 8)]:
        a_tok = [int(d) + OFFSET for d in str(a)]
        b_tok = [int(d) + OFFSET for d in str(b)]
        r_tok = [int(d) + OFFSET for d in str(a * b)]
        tests.append((f"Mul {a}*{b}={a*b}", [BOS] + a_tok + [SEP] + b_tok + [SEP], r_tok))

    # Fibonacci
    for a, b in [(1, 1), (1, 2), (2, 3)]:
        seq = [a, b]
        for _ in range(3):
            seq.append(seq[-1] + seq[-2])
            if seq[-1] >= config.vocab_size - OFFSET:
                seq.pop()
                break
        tests.append((
            f"Fib {seq[:2]}→{seq[2:]}",
            [BOS, seq[0]+OFFSET, seq[1]+OFFSET, SEP],
            [s+OFFSET for s in seq[2:]]
        ))

    # Sorting
    for seq in [[3,1,2], [5,3,1,4,2]]:
        tests.append((
            f"Sort {seq}",
            [BOS] + [s+OFFSET for s in seq] + [SEP],
            [s+OFFSET for s in sorted(seq)]
        ))

    # Run all tests
    n_simple = 0
    n_simple_ok = 0
    n_hard = 0
    n_hard_ok = 0
    total_ok = 0

    for name, prefix, expected in tests:
        gen = generate(model, prefix, max_new=len(expected)+3, seq_len=config.max_seq_len)
        match = gen[:len(expected)] == expected

        is_simple = len(expected) == 1

        if match:
            total_ok += 1
            if is_simple:
                n_simple_ok += 1
            else:
                n_hard_ok += 1
            print(f"  [OK] {name}")
        else:
            g = [t-OFFSET if t>=OFFSET else f"[{t}]" for t in gen[:max(len(expected), len(gen[:10]))]]
            e = [t-OFFSET if t>=OFFSET else f"[{t}]" for t in expected]
            print(f"  [XX] {name} | got {g}, want {e}")

        if is_simple:
            n_simple += 1
        else:
            n_hard += 1

    total = len(tests)
    pct = total_ok / max(1, total) * 100
    print(f"\n  Simple (1-token):  {n_simple_ok}/{n_simple} ({n_simple_ok/max(1,n_simple)*100:.0f}%)")
    print(f"  Hard (multi-token): {n_hard_ok}/{n_hard} ({n_hard_ok/max(1,n_hard)*100:.0f}%)")
    print(f"  TOTAL GENERATION:   {total_ok}/{total} ({pct:.1f}%)")
    return pct


def make_dataset(tasks, n_per_task, seq_len, vocab_size):
    """Create dataset and flush stdout to ensure output visibility."""
    import gc
    gc.collect()
    data = CombinedTaskDataset(
        n_samples_per_task=n_per_task, seq_len=seq_len,
        vocab_size=vocab_size, tasks=tasks
    )
    print(f"  Dataset: {len(data)} samples", flush=True)
    return data


def main():
    import gc
    t_start = time.time()

    # Force line-buffered output so we see progress even if process dies
    sys.stdout.reconfigure(line_buffering=True)

    print("="*70)
    print("  NEXUS: Neural EXecution with Unified States")
    print("  Innovation: Cross-Iteration State Attention (CISA)")
    print("="*70, flush=True)

    VOCAB_SIZE = 200
    SEQ_LEN = 64

    config = NexusConfig(
        vocab_size=VOCAB_SIZE,
        d_model=128,
        n_heads=4,
        d_ff=256,
        max_seq_len=SEQ_LEN,
        n_iterations=6,
        max_iterations=8,
    )
    model = NexusModel(config)
    params = model.count_parameters()

    print(f"\n  NEXUS Model Configuration:")
    print(f"  d_model={config.d_model}, n_heads={config.n_heads}, d_ff={config.d_ff}")
    print(f"  Iterations: {config.n_iterations} (shared weights)")
    print(f"  Real params:      {params['total_params']:,}")
    print(f"  Effective params: {params['effective_params']:,} ({config.n_iterations}x multiplier)")
    print(f"  Innovation: Cross-Iteration State Attention (CISA)", flush=True)

    all_tasks = list(CombinedTaskDataset.TASK_CLASSES.keys())

    # ============================================================
    # PHASE 1: Build foundation on ALL tasks
    # ============================================================
    print(f"\n{'#'*70}")
    print(f"  PHASE 1: Multi-Task Foundation (10000 steps)")
    print(f"{'#'*70}", flush=True)
    data = make_dataset(all_tasks, 3000, SEQ_LEN, VOCAB_SIZE)
    train_phase(model, data, max_steps=10000, lr=3e-4, warmup=500,
                batch_size=32, log_every=500)
    del data; gc.collect()
    score1 = test_generation(model, config, "after Phase 1")

    # ============================================================
    # PHASE 2: Focus on hard tasks
    # ============================================================
    hard_tasks = [
        "addition", "subtraction", "multiplication", "fibonacci",
        "sorting", "boolean_logic", "palindrome", "bracket_matching",
        "arithmetic_sequence", "token_counting", "associative_recall",
    ]
    print(f"\n{'#'*70}")
    print(f"  PHASE 2: Hard Task Focus (6000 steps)")
    print(f"{'#'*70}", flush=True)
    data = make_dataset(hard_tasks, 4000, SEQ_LEN, VOCAB_SIZE)
    train_phase(model, data, max_steps=6000, lr=1e-4, warmup=300,
                batch_size=32, log_every=500)
    del data; gc.collect()
    score2 = test_generation(model, config, "after Phase 2")

    # ============================================================
    # PHASE 3: Arithmetic deep dive
    # ============================================================
    arith_tasks = [
        "addition", "subtraction", "multiplication", "fibonacci",
        "arithmetic_sequence",
    ]
    print(f"\n{'#'*70}")
    print(f"  PHASE 3: Arithmetic Focus (4000 steps)")
    print(f"{'#'*70}", flush=True)
    data = make_dataset(arith_tasks, 5000, SEQ_LEN, VOCAB_SIZE)
    train_phase(model, data, max_steps=4000, lr=5e-5, warmup=200,
                batch_size=32, log_every=500)
    del data; gc.collect()
    score3 = test_generation(model, config, "after Phase 3")

    # ============================================================
    # PHASE 4: Final polish on all tasks
    # ============================================================
    print(f"\n{'#'*70}")
    print(f"  PHASE 4: Final Polish (3000 steps)")
    print(f"{'#'*70}", flush=True)
    data = make_dataset(all_tasks, 2000, SEQ_LEN, VOCAB_SIZE)
    train_phase(model, data, max_steps=3000, lr=2e-5, warmup=150,
                batch_size=32, log_every=500)
    del data; gc.collect()
    score4 = test_generation(model, config, "FINAL")

    # ============================================================
    # Save model
    # ============================================================
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config.to_dict(),
    }, "nexus_trained.pt")

    # ============================================================
    # Comprehensive evaluation
    # ============================================================
    print(f"\n{'='*70}")
    print("  COMPREHENSIVE TASK EVALUATION")
    print(f"{'='*70}", flush=True)
    data = make_dataset(all_tasks, 150, SEQ_LEN, VOCAB_SIZE)
    evaluator = TaskEvaluator(model, "cpu")
    results = evaluator.evaluate_all_tasks(data, batch_size=32)
    summary = evaluator.print_results(results, "NEXUS FINAL EVALUATION")
    del data; gc.collect()

    elapsed = time.time() - t_start

    print(f"\n{'='*70}")
    print(f"  NEXUS FINAL RESULTS")
    print(f"{'='*70}")
    print(f"  Architecture:     NEXUS (Cross-Iteration State Attention)")
    print(f"  Real params:      {params['total_params']:,}")
    print(f"  Effective params: {params['effective_params']:,}")
    print(f"  Iterations:       {config.n_iterations}")
    print(f"  Generation:       {score4:.1f}%")
    print(f"  Task Accuracy:    {summary['avg_token_accuracy']:.1f}%")
    print(f"  Training time:    {elapsed/60:.1f} minutes")
    print(f"{'='*70}")
    print(f"\n  Generation progression: {score1:.0f}% → {score2:.0f}% → {score3:.0f}% → {score4:.0f}%", flush=True)


if __name__ == "__main__":
    main()
