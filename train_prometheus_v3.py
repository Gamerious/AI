#!/usr/bin/env python3
"""
Prometheus AI v3 - Fixed train/test distribution mismatch

Key fix: NeuroplasticFFN and Memory used mean(dim=1) which pools over ALL
positions including future PAD tokens. During training PADs contain real output
tokens, during generation they're zeros. This mismatch caused the model to
behave differently during generation.

Fix: Use masked pooling that only considers non-PAD positions.

Training approach:
- Phase 1: Teacher forcing on all tasks (build strong foundation)
- Phase 2: Hard task focus (improve weakest areas)
- Phase 3: Light scheduled sampling (close remaining generation gap)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import math
import random

from prometheus.model import PrometheusModel, PrometheusConfig
from training.tasks import CombinedTaskDataset, BOS, SEP, EOS, PAD, OFFSET
from training.evaluator import TaskEvaluator


def get_lr(step, warmup, max_steps, max_lr, min_lr=1e-6):
    if step < warmup:
        return max_lr * step / max(1, warmup)
    progress = (step - warmup) / max(1, max_steps - warmup)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


def train(model, dataset, max_steps, lr=3e-4, warmup=200, batch_size=32, log_every=250):
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.95))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    model.train()

    step = 0
    running_loss = running_acc = n_acc = 0

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
                diag = model.get_diagnostics()
                print(
                    f"  Step {step:>5}/{max_steps} | Loss: {running_loss/n_acc:.4f} | "
                    f"Acc: {running_acc/n_acc:.1f}% | LR: {current_lr:.2e} | "
                    f"Depth: {diag.get('avg_depth', 0):.1f} | "
                    f"PredErr: {diag.get('pred_error', 0):.3f} | "
                    f"Active: {diag.get('active_fraction', 0):.0%}"
                )
                running_loss = running_acc = n_acc = 0

            step += 1


def generate(model, prefix_tokens, max_new=10, seq_len=64):
    """Greedy autoregressive generation."""
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
        pred = out["logits"][0, len(tokens) - 1].argmax().item()
        if pred == EOS or pred == PAD:
            break
        tokens.append(pred)
    return tokens[len(prefix_tokens):]


def test_generation(model, config):
    print(f"\n{'='*70}")
    print(f"  PROMETHEUS v3 GENERATION TESTS")
    print(f"{'='*70}")

    TRUE, FALSE = OFFSET + 11, OFFSET + 12
    AND_OP, OR_OP, XOR_OP, NOT_OP = OFFSET + 20, OFFSET + 21, OFFSET + 22, OFFSET + 23

    tests = []

    # Addition (simple single-digit results)
    for a, b in [(2, 3), (5, 3), (1, 1), (4, 5), (7, 2), (6, 3)]:
        a_tok = [int(d) + OFFSET for d in str(a)]
        b_tok = [int(d) + OFFSET for d in str(b)]
        r_tok = [int(d) + OFFSET for d in str(a + b)]
        tests.append((f"Add {a}+{b}={a+b}", [BOS] + a_tok + [SEP] + b_tok + [SEP], r_tok))

    # Two-digit addition
    for a, b in [(12, 7), (25, 30), (33, 44)]:
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

    # Logic (6 tests)
    tests += [
        ("AND T,T=T", [BOS, AND_OP, TRUE, TRUE, SEP], [TRUE]),
        ("AND T,F=F", [BOS, AND_OP, TRUE, FALSE, SEP], [FALSE]),
        ("OR F,T=T", [BOS, OR_OP, FALSE, TRUE, SEP], [TRUE]),
        ("XOR T,T=F", [BOS, XOR_OP, TRUE, TRUE, SEP], [FALSE]),
        ("NOT T=F", [BOS, NOT_OP, TRUE, SEP], [FALSE]),
        ("NOT F=T", [BOS, NOT_OP, FALSE, SEP], [TRUE]),
    ]

    # Fibonacci
    for a, b in [(1, 1), (1, 2), (2, 3)]:
        seq = [a, b]
        for _ in range(3):
            seq.append(seq[-1] + seq[-2])
            if seq[-1] >= config.vocab_size - OFFSET: seq.pop(); break
        tests.append((f"Fib {seq[:2]}→{seq[2:]}", [BOS, seq[0]+OFFSET, seq[1]+OFFSET, SEP], [s+OFFSET for s in seq[2:]]))

    # Sorting
    for seq in [[3,1,2], [5,3,1,4,2]]:
        tests.append((f"Sort {seq}", [BOS] + [s+OFFSET for s in seq] + [SEP], [s+OFFSET for s in sorted(seq)]))

    # Palindrome
    tests += [
        ("Pal [1,2,1]=T", [BOS, 1+OFFSET, 2+OFFSET, 1+OFFSET, SEP], [TRUE]),
        ("Pal [1,2,3]=F", [BOS, 1+OFFSET, 2+OFFSET, 3+OFFSET, SEP], [FALSE]),
    ]

    # Brackets
    OPEN, CLOSE = OFFSET + 30, OFFSET + 31
    tests += [
        ("Bkt ()=T", [BOS, OPEN, CLOSE, SEP], [TRUE]),
        ("Bkt (())=T", [BOS, OPEN, OPEN, CLOSE, CLOSE, SEP], [TRUE]),
        ("Bkt )(=F", [BOS, CLOSE, OPEN, SEP], [FALSE]),
    ]

    correct = 0
    for name, prefix, expected in tests:
        gen = generate(model, prefix, max_new=len(expected)+2, seq_len=config.max_seq_len)
        match = gen[:len(expected)] == expected
        if match:
            correct += 1
            print(f"  [OK] {name}")
        else:
            g = [t-OFFSET if t>=OFFSET else f"[{t}]" for t in gen[:max(len(expected), len(gen[:10]))]]
            e = [t-OFFSET if t>=OFFSET else f"[{t}]" for t in expected]
            print(f"  [XX] {name} | got {g}, want {e}")

    pct = correct / max(1, len(tests)) * 100
    print(f"\n  GENERATION: {correct}/{len(tests)} ({pct:.1f}%)")
    return pct


def main():
    print("="*70)
    print("  PROMETHEUS AI v3 - FIXED TRAIN/TEST MISMATCH")
    print("="*70)

    VOCAB_SIZE = 200
    SEQ_LEN = 64

    config = PrometheusConfig(
        vocab_size=VOCAB_SIZE, d_model=160, n_heads=8, d_ff=320,
        max_seq_len=SEQ_LEN, max_depth=12, halt_threshold=0.95,
        n_memory_slots=128, memory_top_k=8, n_read_heads=2,
        n_pred_levels=3, n_settle_steps=3,
        neuron_sparsity=0.80, plastic_rank=20,
    )
    model = PrometheusModel(config)
    params = model.count_parameters()
    print(f"\n  Prometheus v3 Model:")
    print(f"  Real params:      {params['total_params']:,}")
    print(f"  Effective params: {params['effective_params']:,} (fractal x{config.max_depth})")
    print(f"  Active/token:     {params['active_params_per_token']:,} ({params['efficiency_ratio']:.0%})")

    all_tasks = list(CombinedTaskDataset.TASK_CLASSES.keys())

    # Phase 1: Teacher forcing on all tasks
    all_data = CombinedTaskDataset(n_samples_per_task=5000, seq_len=SEQ_LEN,
                                    vocab_size=VOCAB_SIZE, tasks=all_tasks)
    print(f"\n{'#'*70}")
    print(f"  PHASE 1: Multi-Task Training (8000 steps)")
    print(f"{'#'*70}")
    train(model, all_data, max_steps=8000, lr=3e-4, warmup=500, batch_size=32, log_every=500)
    score1 = test_generation(model, config)

    # Phase 2: Hard task focus
    hard_tasks = ["addition", "subtraction", "multiplication", "fibonacci",
                  "sorting", "boolean_logic", "palindrome", "bracket_matching",
                  "arithmetic_sequence", "token_counting", "associative_recall"]
    hard_data = CombinedTaskDataset(n_samples_per_task=8000, seq_len=SEQ_LEN,
                                     vocab_size=VOCAB_SIZE, tasks=hard_tasks)
    print(f"\n{'#'*70}")
    print(f"  PHASE 2: Hard Task Focus (5000 steps)")
    print(f"{'#'*70}")
    train(model, hard_data, max_steps=5000, lr=1e-4, warmup=250, batch_size=32, log_every=500)
    score2 = test_generation(model, config)

    # Phase 3: Final polish
    print(f"\n{'#'*70}")
    print(f"  PHASE 3: Final Polish (3000 steps)")
    print(f"{'#'*70}")
    train(model, all_data, max_steps=3000, lr=3e-5, warmup=150, batch_size=32, log_every=500)
    score3 = test_generation(model, config)

    # Save
    torch.save({"model_state_dict": model.state_dict(), "config": config.to_dict()},
               "prometheus_v3_trained.pt")

    # Full evaluation
    print(f"\n{'='*70}")
    print("  FINAL TASK EVALUATION")
    print(f"{'='*70}")
    test_data = CombinedTaskDataset(n_samples_per_task=200, seq_len=SEQ_LEN,
                                     vocab_size=VOCAB_SIZE, tasks=all_tasks)
    evaluator = TaskEvaluator(model, "cpu")
    results = evaluator.evaluate_all_tasks(test_data, batch_size=32)
    summary = evaluator.print_results(results, "PROMETHEUS v3 FINAL")

    print(f"\n{'='*70}")
    print(f"  PROMETHEUS v3 RESULTS")
    print(f"  Real params:      {params['total_params']:,}")
    print(f"  Effective params: {params['effective_params']:,}")
    print(f"  Generation:       {score3:.1f}%")
    print(f"  Task Accuracy:    {summary['avg_token_accuracy']:.1f}%")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
