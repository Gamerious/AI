#!/usr/bin/env python3
"""
NeuroSpark AI - Comprehensive Training & Evaluation Pipeline

Trains the model on 24 diverse tasks across 6 categories,
then evaluates performance on held-out test data.

Training strategy:
1. Phase 1: Warmup on easy tasks (copy, patterns)
2. Phase 2: Full multi-task training on all 24 tasks
3. Phase 3: Hard task fine-tuning (math, reasoning)
4. Evaluation on held-out test set with per-task breakdown
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
import time
import math
import json
from typing import Dict, List
from collections import defaultdict

from core.model import NeuroSparkModel, NeuroSparkConfig
from training.tasks import CombinedTaskDataset
from training.evaluator import TaskEvaluator


def get_lr(step: int, warmup: int, max_steps: int, max_lr: float, min_lr: float = 1e-6) -> float:
    """Cosine decay with linear warmup."""
    if step < warmup:
        return max_lr * step / max(1, warmup)
    progress = (step - warmup) / max(1, max_steps - warmup)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


def train_phase(
    model: NeuroSparkModel,
    dataset,
    phase_name: str,
    max_steps: int,
    batch_size: int = 32,
    lr: float = 3e-4,
    warmup_steps: int = 50,
    grad_clip: float = 1.0,
    log_interval: int = 50,
    eval_interval: int = 200,
    evaluator: TaskEvaluator = None,
    eval_dataset: CombinedTaskDataset = None,
) -> List[Dict]:
    """Train one phase."""
    print(f"\n{'#'*80}")
    print(f"  PHASE: {phase_name}")
    print(f"  Steps: {max_steps} | LR: {lr} | Batch: {batch_size}")
    print(f"{'#'*80}\n")

    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.95), eps=1e-8
    )

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    model.train()

    history = []
    step = 0
    best_loss = float("inf")
    running_loss = 0.0
    running_ce = 0.0
    running_acc = 0.0
    n_accumulated = 0

    epoch = 0
    while step < max_steps:
        epoch += 1
        for batch in dataloader:
            if step >= max_steps:
                break

            # Learning rate schedule
            current_lr = get_lr(step, warmup_steps, max_steps, lr)
            for pg in optimizer.param_groups:
                pg["lr"] = current_lr

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            # Forward
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs["loss"]

            # Token accuracy
            with torch.no_grad():
                preds = outputs["logits"].argmax(dim=-1)
                mask = labels != -100
                if mask.any():
                    acc = (preds[mask] == labels[mask]).float().mean().item() * 100
                else:
                    acc = 0.0

            # Backward
            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            # Track stats
            running_loss += loss.item()
            running_ce += outputs.get("ce_loss", loss).item()
            running_acc += acc
            n_accumulated += 1

            # Log
            if step % log_interval == 0 and step > 0:
                avg_loss = running_loss / n_accumulated
                avg_ce = running_ce / n_accumulated
                avg_acc = running_acc / n_accumulated
                diag = model.get_diagnostics()

                print(
                    f"  Step {step:>5}/{max_steps} | "
                    f"Loss: {avg_loss:.4f} | CE: {avg_ce:.4f} | "
                    f"Acc: {avg_acc:.1f}% | LR: {current_lr:.2e} | "
                    f"GradNorm: {grad_norm:.2f} | "
                    f"Depth: {diag.get('avg_depth', 0):.1f} | "
                    f"Reason: {diag.get('reasoning_steps', 0)}"
                )

                history.append({
                    "step": step, "loss": avg_loss, "ce_loss": avg_ce,
                    "accuracy": avg_acc, "lr": current_lr,
                })

                if avg_loss < best_loss:
                    best_loss = avg_loss

                running_loss = 0.0
                running_ce = 0.0
                running_acc = 0.0
                n_accumulated = 0

            # Periodic evaluation
            if eval_interval > 0 and step % eval_interval == 0 and step > 0:
                if evaluator and eval_dataset:
                    print(f"\n  --- Mid-training evaluation at step {step} ---")
                    results = evaluator.evaluate_all_tasks(eval_dataset, batch_size=32)
                    summary = evaluator.print_results(results, f"EVAL @ Step {step}")
                    model.train()  # Back to training mode

            step += 1

    print(f"\n  Phase '{phase_name}' complete. Best loss: {best_loss:.4f}")
    return history


def main():
    print("="*80)
    print("  NEUROSPARK AI - COMPREHENSIVE TRAINING & EVALUATION")
    print("="*80)

    # ---- Configuration ----
    VOCAB_SIZE = 200
    SEQ_LEN = 64
    DEVICE = "cpu"

    config = NeuroSparkConfig(
        vocab_size=VOCAB_SIZE,
        d_model=128,
        n_heads=4,
        n_layers=6,
        d_ff=256,
        n_experts=8,
        max_k=3,
        max_seq_len=SEQ_LEN,
        linear_threshold=128,
        n_thoughts=8,
        max_reasoning_steps=4,
        dropout=0.0,
    )

    model = NeuroSparkModel(config).to(DEVICE)
    params = model.count_parameters()
    print(f"\n  Model: NeuroSpark")
    print(f"  Total params:       {params['total_params']:,}")
    print(f"  Active params/tok:  {params['active_params_per_token']:,}")
    print(f"  Efficiency:         {params['efficiency_ratio']:.1%}")

    # ---- Generate Datasets ----
    print(f"\n{'='*80}")
    print("  GENERATING TRAINING DATA (24 tasks)")
    print(f"{'='*80}")

    # Easy tasks for warmup
    easy_tasks = [
        "mirror_sequence", "delayed_copy", "pattern_repetition",
        "boolean_logic", "grammar_pattern",
    ]

    # All tasks
    all_tasks = list(CombinedTaskDataset.TASK_CLASSES.keys())

    # Hard tasks for fine-tuning
    hard_tasks = [
        "addition", "subtraction", "multiplication",
        "fibonacci", "sorting", "associative_recall",
        "implication_chain", "arithmetic_sequence",
    ]

    print("\n  --- Training Set ---")
    train_dataset = CombinedTaskDataset(
        n_samples_per_task=2000, seq_len=SEQ_LEN, vocab_size=VOCAB_SIZE, tasks=all_tasks
    )

    print("\n  --- Test Set ---")
    test_dataset = CombinedTaskDataset(
        n_samples_per_task=200, seq_len=SEQ_LEN, vocab_size=VOCAB_SIZE, tasks=all_tasks
    )

    easy_train = CombinedTaskDataset(
        n_samples_per_task=1000, seq_len=SEQ_LEN, vocab_size=VOCAB_SIZE, tasks=easy_tasks
    )

    hard_train = CombinedTaskDataset(
        n_samples_per_task=3000, seq_len=SEQ_LEN, vocab_size=VOCAB_SIZE, tasks=hard_tasks
    )

    evaluator = TaskEvaluator(model, DEVICE)

    # ---- Pre-training Evaluation (baseline) ----
    print(f"\n{'='*80}")
    print("  BASELINE EVALUATION (untrained model)")
    print(f"{'='*80}")
    baseline_results = evaluator.evaluate_all_tasks(test_dataset, batch_size=32)
    baseline_summary = evaluator.print_results(baseline_results, "BASELINE (random weights)")

    # ---- Phase 1: Warmup on easy tasks ----
    phase1_history = train_phase(
        model, easy_train,
        phase_name="WARMUP (easy tasks)",
        max_steps=300,
        batch_size=32,
        lr=5e-4,
        warmup_steps=30,
        log_interval=50,
        eval_interval=0,  # Skip mid-eval for speed
    )

    # ---- Phase 2: Full multi-task training ----
    phase2_history = train_phase(
        model, train_dataset,
        phase_name="MULTI-TASK TRAINING (all 24 tasks)",
        max_steps=2000,
        batch_size=32,
        lr=3e-4,
        warmup_steps=100,
        log_interval=100,
        eval_interval=500,
        evaluator=evaluator,
        eval_dataset=test_dataset,
    )

    # ---- Phase 3: Hard task fine-tuning ----
    phase3_history = train_phase(
        model, hard_train,
        phase_name="HARD TASK FINE-TUNING (math + reasoning)",
        max_steps=1500,
        batch_size=32,
        lr=1e-4,
        warmup_steps=50,
        log_interval=100,
        eval_interval=500,
        evaluator=evaluator,
        eval_dataset=test_dataset,
    )

    # ---- Final Evaluation ----
    print(f"\n{'='*80}")
    print("  FINAL COMPREHENSIVE EVALUATION")
    print(f"{'='*80}")
    final_results = evaluator.evaluate_all_tasks(test_dataset, batch_size=32)
    final_summary = evaluator.print_results(final_results, "FINAL RESULTS (after all training)")

    # ---- Improvement Summary ----
    print(f"\n{'='*80}")
    print("  IMPROVEMENT SUMMARY")
    print(f"{'='*80}")
    print(f"  Baseline Token Accuracy:  {baseline_summary['avg_token_accuracy']:.1f}%")
    print(f"  Final Token Accuracy:     {final_summary['avg_token_accuracy']:.1f}%")
    print(f"  Improvement:              +{final_summary['avg_token_accuracy'] - baseline_summary['avg_token_accuracy']:.1f}%")
    print()
    print(f"  Baseline Seq Accuracy:    {baseline_summary['avg_sequence_accuracy']:.1f}%")
    print(f"  Final Seq Accuracy:       {final_summary['avg_sequence_accuracy']:.1f}%")
    print(f"  Improvement:              +{final_summary['avg_sequence_accuracy'] - baseline_summary['avg_sequence_accuracy']:.1f}%")
    print()
    print(f"  Baseline Loss:            {baseline_summary['avg_loss']:.4f}")
    print(f"  Final Loss:               {final_summary['avg_loss']:.4f}")

    # ---- Per-task improvement ----
    print(f"\n  --- Per-Task Improvements ---")
    print(f"  {'Task':<25} {'Before':>10} {'After':>10} {'Change':>10}")
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10}")

    for task_name in sorted(final_results.keys()):
        if task_name in baseline_results:
            before = baseline_results[task_name]["token_accuracy"]
            after = final_results[task_name]["token_accuracy"]
            change = after - before
            marker = "+++" if change > 20 else "++" if change > 10 else "+" if change > 0 else ""
            print(f"  {task_name:<25} {before:>9.1f}% {after:>9.1f}% {change:>+9.1f}% {marker}")

    # ---- Adaptive Behavior Analysis ----
    print(f"\n  --- Adaptive Behavior ---")
    for task_name in ["boolean_logic", "addition", "sorting", "fibonacci"]:
        if task_name in final_results:
            r = final_results[task_name]
            print(f"  {task_name:<25} Depth: {r['avg_depth']:.1f} | "
                  f"Reasoning: {r['avg_reasoning_steps']:.1f} | "
                  f"Accuracy: {r['token_accuracy']:.1f}%")

    # ---- Save results ----
    all_results = {
        "baseline": baseline_summary,
        "final": final_summary,
        "per_task_final": {k: v for k, v in final_results.items()},
        "config": config.to_dict(),
        "params": params,
    }

    with open("training_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved to training_results.json")

    # Save model
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config.to_dict(),
    }, "neurospark_trained.pt")
    print(f"  Model saved to neurospark_trained.pt")

    print(f"\n{'='*80}")
    print("  TRAINING & EVALUATION COMPLETE")
    print(f"{'='*80}\n")

    return model, final_results


if __name__ == "__main__":
    model, results = main()
