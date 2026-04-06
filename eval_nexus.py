#!/usr/bin/env python3
"""Quick evaluation of saved NEXUS model."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(line_buffering=True)

import torch
from nexus.model import NexusModel, NexusConfig
from training.tasks import CombinedTaskDataset
from training.evaluator import TaskEvaluator

print("Loading NEXUS model...", flush=True)
ckpt = torch.load("nexus_trained.pt", map_location="cpu")
config = NexusConfig.from_dict(ckpt["config"])
model = NexusModel(config)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
print(f"  Loaded: {sum(p.numel() for p in model.parameters()):,} params", flush=True)

all_tasks = list(CombinedTaskDataset.TASK_CLASSES.keys())
data = CombinedTaskDataset(
    n_samples_per_task=200, seq_len=64, vocab_size=200, tasks=all_tasks
)
print(f"  Eval data: {len(data)} samples", flush=True)

evaluator = TaskEvaluator(model, "cpu")
results = evaluator.evaluate_all_tasks(data, batch_size=32)
summary = evaluator.print_results(results, "NEXUS FINAL EVALUATION")

print(f"\n  Task Accuracy: {summary['avg_token_accuracy']:.1f}%")
print(f"  Seq Accuracy:  {summary['avg_sequence_accuracy']:.1f}%")
