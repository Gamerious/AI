#!/usr/bin/env python3
"""
NeuroSpark Causal Training - Clean autoregressive training without information leaks.

Phase 1: Train pure causal MoE transformer (no reasoning/adaptive depth)
Phase 2: Verify autoregressive generation works
Phase 3: Add back properly causal reasoning
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import math
import time
import json

from core.model import NeuroSparkModel, NeuroSparkConfig
from training.tasks import CombinedTaskDataset, BOS, SEP, EOS, PAD, OFFSET
from training.evaluator import TaskEvaluator


def get_lr(step, warmup, max_steps, max_lr, min_lr=1e-6):
    if step < warmup:
        return max_lr * step / max(1, warmup)
    progress = (step - warmup) / max(1, max_steps - warmup)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


def train(model, dataset, max_steps, lr=3e-4, warmup=100, batch_size=32, log_every=100):
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.95))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    model.train()

    step = 0
    best_loss = float("inf")
    running_loss = 0
    running_acc = 0
    n_acc = 0

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
                avg_l = running_loss / n_acc
                avg_a = running_acc / n_acc
                best_loss = min(best_loss, avg_l)
                print(f"  Step {step:>5}/{max_steps} | Loss: {avg_l:.4f} | Acc: {avg_a:.1f}% | LR: {current_lr:.2e}")
                running_loss = running_acc = n_acc = 0

            step += 1

    print(f"  Training complete. Best loss: {best_loss:.4f}")
    return model


def generate(model, prefix_tokens, max_new=10, seq_len=64):
    """Clean autoregressive generation."""
    model.eval()
    tokens = list(prefix_tokens)

    for _ in range(max_new):
        if len(tokens) >= seq_len:
            break
        padded = tokens + [PAD] * (seq_len - len(tokens))
        inp = torch.tensor([padded[:seq_len]], dtype=torch.long)
        with torch.no_grad():
            out = model(input_ids=inp)
        pred = out["logits"][0, len(tokens) - 1].argmax().item()
        if pred == EOS or pred == PAD:
            break
        tokens.append(pred)

    return tokens[len(prefix_tokens):]


def test_generation(model, config):
    """Test autoregressive generation on various tasks."""
    print(f"\n{'='*70}")
    print("  GENERATION TESTS")
    print(f"{'='*70}")

    tests = []

    # Addition tests
    for a, b in [(5, 3), (12, 7), (25, 30), (40, 50), (99, 1), (8, 8), (33, 44), (50, 25)]:
        a_tok = [int(d) + OFFSET for d in str(a)]
        b_tok = [int(d) + OFFSET for d in str(b)]
        r_tok = [int(d) + OFFSET for d in str(a + b)]
        prefix = [BOS] + a_tok + [SEP] + b_tok + [SEP]
        tests.append((f"Add: {a}+{b}={a+b}", prefix, r_tok))

    # Subtraction tests
    for a, b in [(10, 3), (50, 25), (99, 50), (77, 33)]:
        a_tok = [int(d) + OFFSET for d in str(a)]
        b_tok = [int(d) + OFFSET for d in str(b)]
        r_tok = [int(d) + OFFSET for d in str(a - b)]
        prefix = [BOS] + a_tok + [SEP] + b_tok + [SEP]
        tests.append((f"Sub: {a}-{b}={a-b}", prefix, r_tok))

    # Multiplication
    for a, b in [(3, 4), (7, 8), (5, 5), (9, 9)]:
        a_tok = [int(d) + OFFSET for d in str(a)]
        b_tok = [int(d) + OFFSET for d in str(b)]
        r_tok = [int(d) + OFFSET for d in str(a * b)]
        prefix = [BOS] + a_tok + [SEP] + b_tok + [SEP]
        tests.append((f"Mul: {a}*{b}={a*b}", prefix, r_tok))

    # Boolean logic
    TRUE, FALSE = OFFSET + 11, OFFSET + 12
    AND_OP, OR_OP, XOR_OP, NOT_OP = OFFSET + 20, OFFSET + 21, OFFSET + 22, OFFSET + 23
    tests.extend([
        ("AND T,T=T", [BOS, AND_OP, TRUE, TRUE, SEP], [TRUE]),
        ("AND T,F=F", [BOS, AND_OP, TRUE, FALSE, SEP], [FALSE]),
        ("OR F,T=T", [BOS, OR_OP, FALSE, TRUE, SEP], [TRUE]),
        ("XOR T,T=F", [BOS, XOR_OP, TRUE, TRUE, SEP], [FALSE]),
        ("NOT T=F", [BOS, NOT_OP, TRUE, SEP], [FALSE]),
        ("NOT F=T", [BOS, NOT_OP, FALSE, SEP], [TRUE]),
    ])

    # Fibonacci
    for a, b in [(1, 1), (1, 2), (2, 3)]:
        seq = [a, b]
        for _ in range(3):
            seq.append(seq[-1] + seq[-2])
            if seq[-1] >= config.vocab_size - OFFSET:
                seq.pop()
                break
        prefix = [BOS, seq[0] + OFFSET, seq[1] + OFFSET, SEP]
        expected = [s + OFFSET for s in seq[2:]]
        tests.append((f"Fib: {seq[:2]}→{seq[2:]}", prefix, expected))

    # Sorting
    for seq in [[5,3,1,4,2], [9,7,8,6], [3,1,4]]:
        prefix = [BOS] + [s + OFFSET for s in seq] + [SEP]
        expected = [s + OFFSET for s in sorted(seq)]
        tests.append((f"Sort: {seq}→{sorted(seq)}", prefix, expected))

    # Reverse
    for seq in [[1,2,3,4,5], [7,3,7], [9,8,7]]:
        prefix = [BOS] + [s + OFFSET for s in seq] + [SEP]
        expected = [s + OFFSET for s in reversed(seq)]
        tests.append((f"Rev: {seq}→{list(reversed(seq))}", prefix, expected))

    # Palindrome
    for seq, is_pal in [([1,2,3,2,1], True), ([1,2,3], False), ([1,1,1], True)]:
        prefix = [BOS] + [s + OFFSET for s in seq] + [SEP]
        expected = [TRUE if is_pal else FALSE]
        tests.append((f"Pal: {seq}={'T' if is_pal else 'F'}", prefix, expected))

    # Bracket matching
    OPEN, CLOSE = OFFSET + 30, OFFSET + 31
    tests.extend([
        ("Bkt: ()=T", [BOS, OPEN, CLOSE, SEP], [TRUE]),
        ("Bkt: (()=F", [BOS, OPEN, OPEN, CLOSE, SEP], [FALSE]),
        ("Bkt: (())=T", [BOS, OPEN, OPEN, CLOSE, CLOSE, SEP], [TRUE]),
    ])

    # Token counting
    for target, seq in [(1, [1,2,1,3,1]), (3, [3,3,1,2])]:
        count = seq.count(target)
        prefix = [BOS, target + OFFSET, SEP] + [s + OFFSET for s in seq] + [SEP]
        expected = [int(d) + OFFSET for d in str(count)]
        tests.append((f"Count {target} in {seq}={count}", prefix, expected))

    # Run all tests
    correct = 0
    total = len(tests)
    categories = {}

    for name, prefix, expected in tests:
        generated = generate(model, prefix, max_new=len(expected) + 2, seq_len=config.max_seq_len)
        match = generated[:len(expected)] == expected
        cat = name.split(":")[0].strip()
        if cat not in categories:
            categories[cat] = {"correct": 0, "total": 0}
        categories[cat]["total"] += 1

        if match:
            correct += 1
            categories[cat]["correct"] += 1
            print(f"  [OK] {name}")
        else:
            gen_readable = [t - OFFSET if t >= OFFSET else f"[{t}]" for t in generated[:len(expected)]]
            exp_readable = [t - OFFSET if t >= OFFSET else f"[{t}]" for t in expected]
            print(f"  [XX] {name} | got {gen_readable}, want {exp_readable}")

    print(f"\n  --- Category Breakdown ---")
    for cat, stats in sorted(categories.items()):
        pct = stats["correct"] / max(1, stats["total"]) * 100
        print(f"  {cat:<15} {stats['correct']}/{stats['total']} ({pct:.0f}%)")

    overall = correct / max(1, total) * 100
    print(f"\n  OVERALL: {correct}/{total} ({overall:.1f}%)")
    return overall


def main():
    print("="*70)
    print("  NEUROSPARK - CLEAN CAUSAL TRAINING")
    print("="*70)

    VOCAB_SIZE = 200
    SEQ_LEN = 64

    # Clean causal config: NO reasoning, NO adaptive depth
    config = NeuroSparkConfig(
        vocab_size=VOCAB_SIZE,
        d_model=128,
        n_heads=4,
        n_layers=8,       # More layers to compensate for no reasoning
        d_ff=256,
        n_experts=8,
        max_k=3,
        max_seq_len=SEQ_LEN,
        linear_threshold=128,
        dropout=0.0,
        use_reasoning=False,      # Disabled - not causal-safe
        use_adaptive_depth=False,  # Disabled - not needed for clean test
    )

    model = NeuroSparkModel(config)
    params = model.count_parameters()
    print(f"\n  Clean Causal Model:")
    print(f"  Total params:      {params['total_params']:,}")
    print(f"  Active params/tok: {params['active_params_per_token']:,}")

    # Generate ALL task data
    print(f"\n  Generating training data...")
    all_tasks = list(CombinedTaskDataset.TASK_CLASSES.keys())
    train_data = CombinedTaskDataset(n_samples_per_task=3000, seq_len=SEQ_LEN, vocab_size=VOCAB_SIZE, tasks=all_tasks)

    # Phase 1: Full training
    print(f"\n{'#'*70}")
    print(f"  PHASE 1: Full Multi-Task Causal Training (4000 steps)")
    print(f"{'#'*70}")
    model = train(model, train_data, max_steps=4000, lr=3e-4, warmup=200, batch_size=32, log_every=200)

    # Test generation
    score1 = test_generation(model, config)

    # Phase 2: Fine-tune on harder tasks if needed
    if score1 < 80:
        hard_tasks = ["addition", "subtraction", "multiplication", "fibonacci",
                      "sorting", "associative_recall", "arithmetic_sequence",
                      "boolean_logic", "palindrome", "bracket_matching",
                      "token_counting", "implication_chain"]
        hard_data = CombinedTaskDataset(n_samples_per_task=5000, seq_len=SEQ_LEN, vocab_size=VOCAB_SIZE, tasks=hard_tasks)

        print(f"\n{'#'*70}")
        print(f"  PHASE 2: Hard Task Fine-Tuning (3000 steps)")
        print(f"{'#'*70}")
        model = train(model, hard_data, max_steps=3000, lr=1e-4, warmup=100, batch_size=32, log_every=200)
        score2 = test_generation(model, config)
    else:
        score2 = score1

    # Phase 3: Even more training if needed
    if score2 < 70:
        print(f"\n{'#'*70}")
        print(f"  PHASE 3: Extended Training (2000 more steps)")
        print(f"{'#'*70}")
        model = train(model, train_data, max_steps=2000, lr=5e-5, warmup=50, batch_size=32, log_every=200)
        score3 = test_generation(model, config)
    else:
        score3 = score2

    # Save
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config.to_dict(),
    }, "neurospark_trained.pt")

    # Final eval with evaluator
    print(f"\n{'='*70}")
    print("  FINAL TASK EVALUATION")
    print(f"{'='*70}")
    test_data = CombinedTaskDataset(n_samples_per_task=200, seq_len=SEQ_LEN, vocab_size=VOCAB_SIZE, tasks=all_tasks)
    evaluator = TaskEvaluator(model, "cpu")
    results = evaluator.evaluate_all_tasks(test_data, batch_size=32)
    summary = evaluator.print_results(results, "FINAL EVALUATION (Next-Token Prediction)")

    print(f"\n{'='*70}")
    print(f"  Generation Score: {score3:.1f}%")
    print(f"  Task Eval Accuracy: {summary['avg_token_accuracy']:.1f}%")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
