#!/usr/bin/env python3
"""
NEXUS GPU Training Script

Optimized for RTX 4060 (8GB VRAM):
- Mixed precision (fp16) training
- Gradient accumulation
- Cosine LR with warmup
- Saves checkpoints periodically
- Loads pre-generated data from data/ directory
- Generation testing between phases

Usage:
    python3 train_nexus_gpu.py                          # Train large (recommended for 4060)
    python3 train_nexus_gpu.py --config medium          # Train medium (faster)
    python3 train_nexus_gpu.py --config xlarge --bs 12  # Train xlarge (tight fit)
    python3 train_nexus_gpu.py --resume checkpoint.pt   # Resume from checkpoint
"""

import sys
import os
import argparse
import math
import time
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.amp import autocast, GradScaler

from nexus.model import NexusModel, NexusConfig
from training.tasks import CombinedTaskDataset, BOS, SEP, EOS, PAD, OFFSET
from training.evaluator import TaskEvaluator


# ============================================================
# Data Loading
# ============================================================

def load_data(path):
    """Load pre-generated dataset from .pt file."""
    print(f"  Loading {path}...", end=" ", flush=True)
    data = torch.load(path, map_location="cpu", weights_only=False)
    dataset = TensorDataset(data["input_ids"], data["labels"])
    print(f"{len(dataset):,} samples (seq_len={data['seq_len']})", flush=True)
    return dataset, data


def generate_fresh_data(tasks, n_per_task, seq_len, vocab_size):
    """Generate data on-the-fly if pre-generated not available."""
    import gc
    print(f"  Generating fresh data: {n_per_task}/task, {len(tasks)} tasks...", flush=True)
    dataset = CombinedTaskDataset(
        n_samples_per_task=n_per_task, seq_len=seq_len,
        vocab_size=vocab_size, tasks=tasks,
    )
    gc.collect()
    return dataset


# ============================================================
# Learning Rate Schedule
# ============================================================

def get_lr(step, warmup, max_steps, max_lr, min_lr=1e-6):
    if step < warmup:
        return max_lr * step / max(1, warmup)
    progress = (step - warmup) / max(1, max_steps - warmup)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


# ============================================================
# Training Loop
# ============================================================

def train_phase(
    model, dataset, max_steps, lr=3e-4, warmup=500, batch_size=48,
    grad_accum=1, log_every=100, device="cuda", use_amp=True,
    checkpoint_dir="checkpoints", phase_name="phase",
):
    """Train one phase with mixed precision and gradient accumulation."""
    os.makedirs(checkpoint_dir, exist_ok=True)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.95)
    )
    scaler = GradScaler("cuda") if use_amp and device == "cuda" else None
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                        num_workers=2, pin_memory=(device == "cuda"))

    model.train()
    step = 0
    running_loss = running_acc = n_acc = 0
    best_acc = 0
    t_start = time.time()

    while step < max_steps:
        for batch in loader:
            if step >= max_steps:
                break

            current_lr = get_lr(step, warmup, max_steps, lr)
            for pg in optimizer.param_groups:
                pg["lr"] = current_lr

            # Handle both TensorDataset and CombinedTaskDataset
            if isinstance(batch, (list, tuple)):
                input_ids = batch[0].to(device, non_blocking=True)
                labels = batch[1].to(device, non_blocking=True)
            else:
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)

            # Forward pass with mixed precision
            if use_amp and device == "cuda":
                with autocast("cuda"):
                    outputs = model(input_ids=input_ids, labels=labels)
                    loss = outputs["loss"] / grad_accum
                scaler.scale(loss).backward()
            else:
                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs["loss"] / grad_accum
                loss.backward()

            # Accuracy tracking
            with torch.no_grad():
                preds = outputs["logits"].argmax(dim=-1)
                mask = labels != -100
                if mask.any():
                    acc = (preds[mask] == labels[mask]).float().mean().item() * 100
                else:
                    acc = 0

            # Gradient step (with accumulation)
            if (step + 1) % grad_accum == 0 or step == max_steps - 1:
                if scaler:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                optimizer.zero_grad()

            running_loss += loss.item() * grad_accum
            running_acc += acc
            n_acc += 1

            if step % log_every == 0 and step > 0:
                avg_acc = running_acc / n_acc
                best_acc = max(best_acc, avg_acc)
                elapsed = time.time() - t_start
                steps_per_sec = step / max(1, elapsed)
                eta = (max_steps - step) / max(0.01, steps_per_sec)
                print(
                    f"  [{phase_name}] Step {step:>6}/{max_steps} | "
                    f"Loss: {running_loss/n_acc:.4f} | "
                    f"Acc: {avg_acc:.1f}% | "
                    f"Best: {best_acc:.1f}% | "
                    f"LR: {current_lr:.2e} | "
                    f"{steps_per_sec:.1f} step/s | "
                    f"ETA: {eta/60:.0f}min",
                    flush=True,
                )
                running_loss = running_acc = n_acc = 0

            # Save checkpoint every 2000 steps
            if step > 0 and step % 2000 == 0:
                ckpt_path = os.path.join(checkpoint_dir, f"{phase_name}_step{step}.pt")
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "step": step,
                    "best_acc": best_acc,
                    "config": model.config.to_dict(),
                }, ckpt_path)
                print(f"  Checkpoint saved: {ckpt_path}", flush=True)

            step += 1

    return best_acc


# ============================================================
# Generation Testing
# ============================================================

def generate(model, prefix_tokens, max_new=10, seq_len=256, device="cuda"):
    """Greedy autoregressive generation."""
    model.eval()
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

    model.train()
    return tokens[len(prefix_tokens):]


def run_generation_tests(model, config, device="cuda", label=""):
    """Run comprehensive generation tests."""
    TRUE, FALSE = OFFSET + 11, OFFSET + 12
    AND_OP, OR_OP, XOR_OP, NOT_OP = OFFSET + 20, OFFSET + 21, OFFSET + 22, OFFSET + 23

    tests = []

    # Single-digit addition
    for a, b in [(2, 3), (5, 3), (1, 1), (4, 5), (7, 2), (6, 3)]:
        r = a + b
        if r < 10:
            tests.append((f"{a}+{b}={r}", [BOS, a+OFFSET, SEP, b+OFFSET, SEP], [r+OFFSET]))

    # Boolean logic
    tests += [
        ("AND T,T=T", [BOS, AND_OP, TRUE, TRUE, SEP], [TRUE]),
        ("AND T,F=F", [BOS, AND_OP, TRUE, FALSE, SEP], [FALSE]),
        ("OR F,T=T", [BOS, OR_OP, FALSE, TRUE, SEP], [TRUE]),
        ("XOR T,T=F", [BOS, XOR_OP, TRUE, TRUE, SEP], [FALSE]),
        ("NOT T=F", [BOS, NOT_OP, TRUE, SEP], [FALSE]),
        ("NOT F=T", [BOS, NOT_OP, FALSE, SEP], [TRUE]),
    ]

    # Multi-digit
    for a, b in [(25, 30), (33, 44), (12, 7)]:
        a_tok = [int(d)+OFFSET for d in str(a)]
        b_tok = [int(d)+OFFSET for d in str(b)]
        r_tok = [int(d)+OFFSET for d in str(a+b)]
        tests.append((f"{a}+{b}={a+b}", [BOS]+a_tok+[SEP]+b_tok+[SEP], r_tok))

    # Fibonacci
    for a, b in [(1,1), (1,2), (2,3)]:
        seq = [a, b]
        for _ in range(3):
            seq.append(seq[-1]+seq[-2])
        tests.append((f"Fib {a},{b}", [BOS, a+OFFSET, b+OFFSET, SEP], [s+OFFSET for s in seq[2:]]))

    # Sorting
    tests.append(("Sort [3,1,2]", [BOS, 3+OFFSET, 1+OFFSET, 2+OFFSET, SEP], [1+OFFSET, 2+OFFSET, 3+OFFSET]))

    ok = total = 0
    print(f"\n  Generation Test {label}:", flush=True)
    for name, prefix, expected in tests:
        gen = generate(model, prefix, max_new=len(expected)+3, seq_len=config.max_seq_len, device=device)
        match = gen[:len(expected)] == expected
        if match:
            ok += 1
            print(f"    [OK] {name}")
        else:
            g = [t-OFFSET if t>=OFFSET else f"[{t}]" for t in gen[:8]]
            e = [t-OFFSET if t>=OFFSET else f"[{t}]" for t in expected]
            print(f"    [XX] {name} | got {g}, want {e}")
        total += 1

    pct = ok / max(1, total) * 100
    print(f"    Score: {ok}/{total} ({pct:.1f}%)", flush=True)
    return pct


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="NEXUS GPU Training")
    parser.add_argument("--config", choices=["small", "medium", "large", "xlarge"], default="large")
    parser.add_argument("--bs", type=int, default=None, help="Batch size (auto if not set)")
    parser.add_argument("--grad-accum", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision")
    parser.add_argument("--cpu", action="store_true", help="Force CPU training")
    args = parser.parse_args()

    sys.stdout.reconfigure(line_buffering=True)

    # Device
    if args.cpu or not torch.cuda.is_available():
        device = "cpu"
        use_amp = False
        print("  Device: CPU", flush=True)
    else:
        device = "cuda"
        use_amp = not args.no_amp
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1024**3
        print(f"  Device: {gpu_name} ({gpu_mem:.1f} GB)", flush=True)

    # Config
    config_map = {
        "small": NexusConfig.small,
        "medium": NexusConfig.medium,
        "large": NexusConfig.large,
        "xlarge": NexusConfig.xlarge,
    }
    config = config_map[args.config]()

    # Default batch sizes per config (tuned for 4060 8GB with AMP)
    default_bs = {
        "small": 128,
        "medium": 96,
        "large": 48,
        "xlarge": 16,
    }
    batch_size = args.bs or default_bs[args.config]

    # Create model
    model = NexusModel(config).to(device)
    params = model.count_parameters()

    print(f"\n{'='*70}")
    print(f"  NEXUS {args.config.upper()} Training")
    print(f"{'='*70}")
    print(f"  Config:      {args.config}")
    print(f"  d_model:     {config.d_model}")
    print(f"  n_heads:     {config.n_heads}")
    print(f"  d_ff:        {config.d_ff}")
    print(f"  iterations:  {config.n_iterations}")
    print(f"  max_seq_len: {config.max_seq_len}")
    print(f"  dropout:     {config.dropout}")
    print(f"  Real params:      {params['total_params']:>12,}")
    print(f"  Effective params: {params['effective_params']:>12,}")
    print(f"  Batch size:  {batch_size}")
    print(f"  Grad accum:  {args.grad_accum}")
    print(f"  Mixed prec:  {use_amp}")
    print(f"{'='*70}", flush=True)

    # Resume from checkpoint
    start_phase = 0
    if args.resume:
        print(f"\n  Resuming from {args.resume}...", flush=True)
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"  Loaded model (best_acc={ckpt.get('best_acc', '?')})", flush=True)

    all_tasks = list(CombinedTaskDataset.TASK_CLASSES.keys())
    data_dir = args.data_dir

    # ============================================================
    # PHASE 1: Foundation (all tasks)
    # ============================================================
    print(f"\n{'#'*70}")
    print(f"  PHASE 1: Multi-Task Foundation")
    print(f"{'#'*70}", flush=True)

    if os.path.exists(os.path.join(data_dir, "train_all_tasks.pt")):
        dataset, _ = load_data(os.path.join(data_dir, "train_all_tasks.pt"))
    else:
        dataset = generate_fresh_data(all_tasks, 10000, config.max_seq_len, config.vocab_size)

    # Scale steps with model size
    base_steps = {"small": 10000, "medium": 15000, "large": 25000, "xlarge": 40000}
    steps = base_steps[args.config]

    train_phase(
        model, dataset, max_steps=steps, lr=3e-4, warmup=min(1000, steps//10),
        batch_size=batch_size, grad_accum=args.grad_accum,
        log_every=250, device=device, use_amp=use_amp,
        phase_name="P1-foundation",
    )
    del dataset

    gen1 = run_generation_tests(model, config, device, "Phase 1")

    # Save Phase 1 checkpoint
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config.to_dict(),
        "generation_score": gen1,
        "phase": 1,
    }, "nexus_phase1.pt")

    # ============================================================
    # PHASE 2: Hard tasks
    # ============================================================
    print(f"\n{'#'*70}")
    print(f"  PHASE 2: Hard Task Focus")
    print(f"{'#'*70}", flush=True)

    hard_tasks = [
        "addition", "subtraction", "multiplication", "fibonacci",
        "sorting", "boolean_logic", "palindrome", "bracket_matching",
        "arithmetic_sequence", "token_counting", "associative_recall",
    ]

    if os.path.exists(os.path.join(data_dir, "train_hard_tasks.pt")):
        dataset, _ = load_data(os.path.join(data_dir, "train_hard_tasks.pt"))
    else:
        dataset = generate_fresh_data(hard_tasks, 10000, config.max_seq_len, config.vocab_size)

    steps2 = base_steps[args.config] // 2
    train_phase(
        model, dataset, max_steps=steps2, lr=1e-4, warmup=min(500, steps2//10),
        batch_size=batch_size, grad_accum=args.grad_accum,
        log_every=250, device=device, use_amp=use_amp,
        phase_name="P2-hard",
    )
    del dataset

    gen2 = run_generation_tests(model, config, device, "Phase 2")

    # ============================================================
    # PHASE 3: Math deep dive
    # ============================================================
    print(f"\n{'#'*70}")
    print(f"  PHASE 3: Math Focus")
    print(f"{'#'*70}", flush=True)

    math_tasks = ["addition", "subtraction", "multiplication", "fibonacci",
                  "arithmetic_sequence", "modular_arithmetic", "sequence_sum"]

    if os.path.exists(os.path.join(data_dir, "train_math.pt")):
        dataset, _ = load_data(os.path.join(data_dir, "train_math.pt"))
    else:
        dataset = generate_fresh_data(math_tasks, 15000, config.max_seq_len, config.vocab_size)

    steps3 = base_steps[args.config] // 3
    train_phase(
        model, dataset, max_steps=steps3, lr=5e-5, warmup=min(300, steps3//10),
        batch_size=batch_size, grad_accum=args.grad_accum,
        log_every=250, device=device, use_amp=use_amp,
        phase_name="P3-math",
    )
    del dataset

    gen3 = run_generation_tests(model, config, device, "Phase 3")

    # ============================================================
    # PHASE 4: Final polish (all tasks, low LR)
    # ============================================================
    print(f"\n{'#'*70}")
    print(f"  PHASE 4: Final Polish")
    print(f"{'#'*70}", flush=True)

    if os.path.exists(os.path.join(data_dir, "train_all_tasks.pt")):
        dataset, _ = load_data(os.path.join(data_dir, "train_all_tasks.pt"))
    else:
        dataset = generate_fresh_data(all_tasks, 5000, config.max_seq_len, config.vocab_size)

    steps4 = base_steps[args.config] // 4
    train_phase(
        model, dataset, max_steps=steps4, lr=2e-5, warmup=min(200, steps4//10),
        batch_size=batch_size, grad_accum=args.grad_accum,
        log_every=250, device=device, use_amp=use_amp,
        phase_name="P4-polish",
    )
    del dataset

    gen4 = run_generation_tests(model, config, device, "FINAL")

    # ============================================================
    # Save final model
    # ============================================================
    final_path = f"nexus_{args.config}_trained.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config.to_dict(),
        "generation_scores": [gen1, gen2, gen3, gen4],
        "params": params,
    }, final_path)
    print(f"\n  Model saved: {final_path}", flush=True)

    # ============================================================
    # Full evaluation
    # ============================================================
    print(f"\n{'='*70}")
    print(f"  COMPREHENSIVE EVALUATION")
    print(f"{'='*70}", flush=True)

    if os.path.exists(os.path.join(data_dir, "eval_all_tasks.pt")):
        eval_data, _ = load_data(os.path.join(data_dir, "eval_all_tasks.pt"))
    else:
        eval_data = generate_fresh_data(all_tasks, 200, config.max_seq_len, config.vocab_size)

    model.eval()
    evaluator = TaskEvaluator(model, device)
    results = evaluator.evaluate_all_tasks(eval_data, batch_size=batch_size)
    summary = evaluator.print_results(results, f"NEXUS {args.config.upper()} FINAL")

    # Final summary
    print(f"\n{'='*70}")
    print(f"  NEXUS {args.config.upper()} - FINAL RESULTS")
    print(f"{'='*70}")
    print(f"  Real params:      {params['total_params']:,}")
    print(f"  Effective params: {params['effective_params']:,}")
    print(f"  Token Accuracy:   {summary['avg_token_accuracy']:.1f}%")
    print(f"  Seq Accuracy:     {summary['avg_sequence_accuracy']:.1f}%")
    print(f"  Generation:       P1={gen1:.0f}% → P2={gen2:.0f}% → P3={gen3:.0f}% → P4={gen4:.0f}%")
    print(f"  Model saved:      {final_path}")
    print(f"{'='*70}", flush=True)

    # Save results JSON
    results_json = {
        "config": args.config,
        "params": params,
        "token_accuracy": summary["avg_token_accuracy"],
        "sequence_accuracy": summary["avg_sequence_accuracy"],
        "generation_scores": [gen1, gen2, gen3, gen4],
    }
    with open(f"nexus_{args.config}_results.json", "w") as f:
        json.dump(results_json, f, indent=2)


if __name__ == "__main__":
    main()
