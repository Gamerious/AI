#!/usr/bin/env python3
"""
Prometheus AI v2 - Training with Scheduled Sampling + Beam Search

Key improvements over v1:
1. Scheduled sampling: gradually mix model's own predictions during training
   - Closes the train/generation gap (exposure bias)
2. Beam search for generation: explores multiple hypotheses
3. Curriculum learning: simple tasks first, complex later
4. Generation-focused loss: extra penalty on first output token
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


def scheduled_sampling_step(model, input_ids, labels, ss_ratio=0.0):
    """
    Training step with scheduled sampling.

    ss_ratio: probability of using model's own prediction instead of ground truth
              0.0 = pure teacher forcing, 1.0 = pure autoregressive

    This gradually teaches the model to handle its own (potentially wrong) predictions,
    closing the train/generation gap.
    """
    device = input_ids.device
    B, N = input_ids.shape

    if ss_ratio <= 0.0:
        # Pure teacher forcing
        return model(input_ids=input_ids, labels=labels)

    # Find SEP positions (start of output) for each sample
    sep_positions = []
    for b in range(B):
        seps = (input_ids[b] == SEP).nonzero(as_tuple=True)[0]
        if len(seps) > 0:
            sep_positions.append(seps[-1].item())  # Last SEP before output
        else:
            sep_positions.append(N)  # No SEP found, skip SS

    # Build mixed input: ground truth for input portion,
    # mix of ground truth and model predictions for output portion
    mixed_ids = input_ids.clone()

    model.eval()  # Temporarily for predictions
    with torch.no_grad():
        logits = model(input_ids=input_ids)["logits"]
        model_preds = logits.argmax(dim=-1)  # (B, N) - model's predictions
    model.train()

    # For output positions, randomly replace with model's prediction
    for b in range(B):
        sep_pos = sep_positions[b]
        for t in range(sep_pos + 1, N - 1):  # Output tokens (after SEP, before EOS/PAD)
            if labels[b, t-1] == -100:  # Not a supervised position
                continue
            if random.random() < ss_ratio:
                # Use model's prediction from position t-1 as input at position t
                pred_token = model_preds[b, t-1].item()
                if pred_token != PAD and pred_token != EOS:
                    mixed_ids[b, t] = pred_token

    return model(input_ids=mixed_ids, labels=labels)


def train_with_ss(model, dataset, max_steps, lr=3e-4, warmup=200, batch_size=32,
                  log_every=200, ss_start=0.0, ss_end=0.3, ss_warmup_frac=0.5):
    """
    Train with scheduled sampling.

    ss_start: initial sampling ratio (0 = pure teacher forcing)
    ss_end: final sampling ratio
    ss_warmup_frac: fraction of training to ramp up SS ratio
    """
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

            # Compute scheduled sampling ratio
            ss_progress = min(1.0, step / (max_steps * ss_warmup_frac))
            ss_ratio = ss_start + (ss_end - ss_start) * ss_progress

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            outputs = scheduled_sampling_step(model, input_ids, labels, ss_ratio)
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
                    f"SS: {ss_ratio:.2f} | "
                    f"Depth: {diag.get('avg_depth', 0):.1f} | "
                    f"Active: {diag.get('active_fraction', 0):.0%}"
                )
                running_loss = running_acc = n_acc = 0

            step += 1


def beam_search(model, prefix_tokens, beam_width=3, max_new=10, seq_len=64):
    """
    Beam search generation - explores multiple hypotheses simultaneously.
    Much better than greedy for multi-token outputs.
    """
    model.eval()
    device = next(model.parameters()).device

    # Each beam: (tokens, log_prob)
    beams = [(list(prefix_tokens), 0.0)]
    completed = []

    for _ in range(max_new):
        if not beams:
            break

        candidates = []
        for tokens, score in beams:
            if len(tokens) >= seq_len:
                completed.append((tokens, score))
                continue

            padded = tokens + [PAD] * (seq_len - len(tokens))
            inp = torch.tensor([padded[:seq_len]], dtype=torch.long, device=device)
            with torch.no_grad():
                out = model(input_ids=inp)
            logits = out["logits"][0, len(tokens) - 1]
            log_probs = F.log_softmax(logits, dim=-1)

            # Get top-k candidates
            top_k_lp, top_k_idx = log_probs.topk(beam_width * 2)

            for lp, idx in zip(top_k_lp.tolist(), top_k_idx.tolist()):
                if idx == EOS or idx == PAD:
                    completed.append((tokens, score + lp))
                else:
                    candidates.append((tokens + [idx], score + lp))

        # Keep top beams
        candidates.sort(key=lambda x: x[1], reverse=True)
        beams = candidates[:beam_width]

    # Return best completed or best beam
    all_results = completed + beams
    if not all_results:
        return []

    best = max(all_results, key=lambda x: x[1] / max(1, len(x[0]) - len(prefix_tokens)))
    return best[0][len(prefix_tokens):]


def generate_greedy(model, prefix_tokens, max_new=10, seq_len=64, min_new=0):
    """Greedy generation with optional min_new."""
    model.eval()
    device = next(model.parameters()).device
    tokens = list(prefix_tokens)
    generated = 0
    for _ in range(max_new):
        if len(tokens) >= seq_len:
            break
        padded = tokens + [PAD] * (seq_len - len(tokens))
        inp = torch.tensor([padded[:seq_len]], dtype=torch.long, device=device)
        with torch.no_grad():
            out = model(input_ids=inp)
        logits = out["logits"][0, len(tokens) - 1]

        if generated < min_new:
            logits[EOS] = float('-inf')
            logits[PAD] = float('-inf')

        pred = logits.argmax().item()
        if pred == EOS or pred == PAD:
            break
        tokens.append(pred)
        generated += 1
    return tokens[len(prefix_tokens):]


def test_generation(model, config, use_beam=False):
    print(f"\n{'='*70}")
    mode = "BEAM SEARCH" if use_beam else "GREEDY"
    print(f"  PROMETHEUS GENERATION ({mode})")
    print(f"{'='*70}")

    TRUE, FALSE = OFFSET + 11, OFFSET + 12
    AND_OP, OR_OP, XOR_OP, NOT_OP = OFFSET + 20, OFFSET + 21, OFFSET + 22, OFFSET + 23

    tests = []

    # Addition
    for a, b in [(5, 3), (12, 7), (25, 30), (40, 50), (8, 8), (33, 44)]:
        a_tok = [int(d) + OFFSET for d in str(a)]
        b_tok = [int(d) + OFFSET for d in str(b)]
        r_tok = [int(d) + OFFSET for d in str(a + b)]
        tests.append((f"Add {a}+{b}={a+b}", [BOS] + a_tok + [SEP] + b_tok + [SEP], r_tok))

    # Subtraction
    for a, b in [(10, 3), (50, 25), (77, 33)]:
        a_tok = [int(d) + OFFSET for d in str(a)]
        b_tok = [int(d) + OFFSET for d in str(b)]
        r_tok = [int(d) + OFFSET for d in str(a - b)]
        tests.append((f"Sub {a}-{b}={a-b}", [BOS] + a_tok + [SEP] + b_tok + [SEP], r_tok))

    # Multiplication
    for a, b in [(3, 4), (7, 8), (5, 5)]:
        a_tok = [int(d) + OFFSET for d in str(a)]
        b_tok = [int(d) + OFFSET for d in str(b)]
        r_tok = [int(d) + OFFSET for d in str(a * b)]
        tests.append((f"Mul {a}*{b}={a*b}", [BOS] + a_tok + [SEP] + b_tok + [SEP], r_tok))

    # Logic
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
    for seq in [[5,3,1,4,2], [3,1,4]]:
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
        if use_beam:
            gen = beam_search(model, prefix, beam_width=4, max_new=len(expected)+3, seq_len=config.max_seq_len)
        else:
            gen = generate_greedy(model, prefix, max_new=len(expected)+2, seq_len=config.max_seq_len, min_new=len(expected))

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
    print("  PROMETHEUS AI v2 - SCHEDULED SAMPLING TRAINING")
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
    print(f"\n  Prometheus v2 Model:")
    print(f"  Real params:      {params['total_params']:,}")
    print(f"  Effective params: {params['effective_params']:,} (fractal x{config.max_depth})")
    print(f"  Active/token:     {params['active_params_per_token']:,} ({params['efficiency_ratio']:.0%})")

    all_tasks = list(CombinedTaskDataset.TASK_CLASSES.keys())

    # ===== PHASE 1: Pure teacher forcing on simple tasks =====
    simple_tasks = ["boolean_logic", "number_comparison", "palindrome",
                    "bracket_matching", "implication_chain", "addition",
                    "subtraction"]
    simple_data = CombinedTaskDataset(n_samples_per_task=5000, seq_len=SEQ_LEN,
                                      vocab_size=VOCAB_SIZE, tasks=simple_tasks)

    print(f"\n{'#'*70}")
    print(f"  PHASE 1: Simple Tasks - Teacher Forcing (3000 steps)")
    print(f"{'#'*70}")
    train_with_ss(model, simple_data, max_steps=3000, lr=3e-4, warmup=300,
                  batch_size=32, log_every=500, ss_start=0.0, ss_end=0.0)
    test_generation(model, config, use_beam=False)

    # ===== PHASE 2: All tasks with gradual scheduled sampling =====
    all_data = CombinedTaskDataset(n_samples_per_task=5000, seq_len=SEQ_LEN,
                                    vocab_size=VOCAB_SIZE, tasks=all_tasks)

    print(f"\n{'#'*70}")
    print(f"  PHASE 2: All Tasks + Scheduled Sampling (8000 steps)")
    print(f"{'#'*70}")
    train_with_ss(model, all_data, max_steps=8000, lr=2e-4, warmup=400,
                  batch_size=32, log_every=500, ss_start=0.0, ss_end=0.25,
                  ss_warmup_frac=0.6)
    score2_greedy = test_generation(model, config, use_beam=False)
    score2_beam = test_generation(model, config, use_beam=True)

    # ===== PHASE 3: Hard tasks with aggressive SS =====
    hard_tasks = ["addition", "subtraction", "multiplication", "fibonacci",
                  "sorting", "arithmetic_sequence", "token_counting",
                  "associative_recall", "delayed_copy"]
    hard_data = CombinedTaskDataset(n_samples_per_task=8000, seq_len=SEQ_LEN,
                                     vocab_size=VOCAB_SIZE, tasks=hard_tasks)

    print(f"\n{'#'*70}")
    print(f"  PHASE 3: Hard Tasks + Aggressive SS (5000 steps)")
    print(f"{'#'*70}")
    train_with_ss(model, hard_data, max_steps=5000, lr=1e-4, warmup=250,
                  batch_size=32, log_every=500, ss_start=0.1, ss_end=0.4,
                  ss_warmup_frac=0.5)
    score3_greedy = test_generation(model, config, use_beam=False)
    score3_beam = test_generation(model, config, use_beam=True)

    # ===== PHASE 4: Polish with high SS =====
    print(f"\n{'#'*70}")
    print(f"  PHASE 4: Final Polish with SS (3000 steps)")
    print(f"{'#'*70}")
    train_with_ss(model, all_data, max_steps=3000, lr=3e-5, warmup=150,
                  batch_size=32, log_every=500, ss_start=0.2, ss_end=0.5,
                  ss_warmup_frac=0.4)
    score4_greedy = test_generation(model, config, use_beam=False)
    score4_beam = test_generation(model, config, use_beam=True)

    # Save
    torch.save({"model_state_dict": model.state_dict(), "config": config.to_dict()},
               "prometheus_v2_trained.pt")

    # Full evaluation
    print(f"\n{'='*70}")
    print("  FINAL TASK EVALUATION")
    print(f"{'='*70}")
    test_data = CombinedTaskDataset(n_samples_per_task=200, seq_len=SEQ_LEN,
                                     vocab_size=VOCAB_SIZE, tasks=all_tasks)
    evaluator = TaskEvaluator(model, "cpu")
    results = evaluator.evaluate_all_tasks(test_data, batch_size=32)
    summary = evaluator.print_results(results, "PROMETHEUS v2 FINAL")

    print(f"\n{'='*70}")
    print(f"  PROMETHEUS v2 FINAL RESULTS")
    print(f"  Real params:         {params['total_params']:,}")
    print(f"  Effective params:    {params['effective_params']:,}")
    print(f"  Generation (greedy): {score4_greedy:.1f}%")
    print(f"  Generation (beam):   {score4_beam:.1f}%")
    print(f"  Task Accuracy:       {summary['avg_token_accuracy']:.1f}%")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
