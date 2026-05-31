"""
Comprehensive Task Evaluation Framework

Evaluates a model on each task type independently,
measuring accuracy, loss, and adaptive behavior metrics.
Model-agnostic: works with any module exposing the task-model interface
(used by the NEXUS task model).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, List, Optional
from collections import defaultdict
import time

from training.tasks import CombinedTaskDataset, PAD


class TaskEvaluator:
    """Evaluates model performance on each task independently."""

    def __init__(self, model: nn.Module, device: str = "cpu"):
        self.model = model
        self.device = device

    @torch.no_grad()
    def evaluate_task(
        self,
        dataset,
        task_name: str,
        batch_size: int = 32,
        max_batches: int = 50,
    ) -> Dict[str, float]:
        """Evaluate on a single task."""
        self.model.eval()
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        total_loss = 0.0
        total_correct = 0
        total_tokens = 0
        total_seq_correct = 0
        total_sequences = 0
        total_time = 0.0
        depth_sum = 0.0
        reasoning_steps_sum = 0.0
        n_batches = 0

        for batch in loader:
            if n_batches >= max_batches:
                break

            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)

            start = time.time()
            outputs = self.model(input_ids=input_ids, labels=labels)
            elapsed = time.time() - start
            total_time += elapsed

            # Loss
            if "ce_loss" in outputs:
                total_loss += outputs["ce_loss"].item()
            elif "loss" in outputs:
                total_loss += outputs["loss"].item()

            # Token-level accuracy (only on labeled positions)
            logits = outputs["logits"]
            preds = logits.argmax(dim=-1)
            mask = labels != -100

            if mask.any():
                correct = (preds[mask] == labels[mask]).sum().item()
                total_correct += correct
                total_tokens += mask.sum().item()

                # Sequence-level accuracy (entire output must be correct)
                for i in range(input_ids.shape[0]):
                    seq_mask = mask[i]
                    if seq_mask.any():
                        seq_correct = (preds[i][seq_mask] == labels[i][seq_mask]).all().item()
                        total_seq_correct += seq_correct
                        total_sequences += 1

            # Diagnostics
            diag = self.model.get_diagnostics()
            depth_sum += diag.get("avg_depth", 0)
            reasoning_steps_sum += diag.get("reasoning_steps", 0)
            n_batches += 1

        if n_batches == 0:
            return {"error": "no batches processed"}

        return {
            "task": task_name,
            "loss": total_loss / n_batches,
            "token_accuracy": total_correct / max(1, total_tokens) * 100,
            "sequence_accuracy": total_seq_correct / max(1, total_sequences) * 100,
            "avg_depth": depth_sum / n_batches,
            "avg_reasoning_steps": reasoning_steps_sum / n_batches,
            "throughput_ms": total_time / n_batches * 1000,
            "n_sequences": total_sequences,
        }

    def evaluate_all_tasks(
        self,
        combined_dataset: CombinedTaskDataset,
        batch_size: int = 32,
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate on all tasks in a combined dataset."""
        results = {}

        for task_name, task_dataset in combined_dataset.task_datasets.items():
            if len(task_dataset) == 0:
                continue
            result = self.evaluate_task(task_dataset, task_name, batch_size)
            results[task_name] = result

        return results

    @staticmethod
    def print_results(results: Dict[str, Dict[str, float]], title: str = "EVALUATION"):
        """Pretty print evaluation results."""
        print(f"\n{'='*80}")
        print(f"  {title}")
        print(f"{'='*80}")

        # Group by category
        categories = {
            "MATH": ["addition", "subtraction", "multiplication", "modular_arithmetic",
                     "number_comparison", "sequence_sum", "fibonacci"],
            "LOGIC": ["boolean_logic", "implication_chain"],
            "PATTERN": ["arithmetic_sequence", "geometric_sequence", "palindrome",
                       "pattern_repetition", "mirror_sequence"],
            "MEMORY": ["delayed_copy", "selective_copy", "token_counting",
                      "associative_recall"],
            "SEQUENCE": ["sorting", "deduplication", "rotation"],
            "LANGUAGE": ["bracket_matching", "substitution_cipher", "grammar_pattern"],
        }

        overall_tok_acc = 0.0
        overall_seq_acc = 0.0
        overall_loss = 0.0
        n_tasks = 0

        for category, task_names in categories.items():
            cat_results = [(n, results[n]) for n in task_names if n in results]
            if not cat_results:
                continue

            print(f"\n  --- {category} ---")
            print(f"  {'Task':<25} {'Loss':>8} {'Tok Acc':>10} {'Seq Acc':>10} {'Depth':>8} {'Reason':>8}")
            print(f"  {'-'*25} {'-'*8} {'-'*10} {'-'*10} {'-'*8} {'-'*8}")

            for name, r in cat_results:
                print(f"  {name:<25} {r['loss']:>8.4f} {r['token_accuracy']:>9.1f}% "
                      f"{r['sequence_accuracy']:>9.1f}% {r['avg_depth']:>8.1f} "
                      f"{r['avg_reasoning_steps']:>8.1f}")
                overall_tok_acc += r["token_accuracy"]
                overall_seq_acc += r["sequence_accuracy"]
                overall_loss += r["loss"]
                n_tasks += 1

        if n_tasks > 0:
            print(f"\n  {'='*80}")
            print(f"  OVERALL ({n_tasks} tasks):")
            print(f"    Avg Token Accuracy:    {overall_tok_acc / n_tasks:.1f}%")
            print(f"    Avg Sequence Accuracy: {overall_seq_acc / n_tasks:.1f}%")
            print(f"    Avg Loss:              {overall_loss / n_tasks:.4f}")
            print(f"  {'='*80}")

        return {
            "avg_token_accuracy": overall_tok_acc / max(1, n_tasks),
            "avg_sequence_accuracy": overall_seq_acc / max(1, n_tasks),
            "avg_loss": overall_loss / max(1, n_tasks),
            "n_tasks": n_tasks,
        }
