"""
NeuroSpark Benchmarks

Compares NeuroSpark against a standard dense Transformer to demonstrate
efficiency gains while maintaining quality.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from typing import Dict
from core.model import NeuroSparkModel, NeuroSparkConfig


class StandardTransformer(nn.Module):
    """Vanilla transformer for baseline comparison."""

    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, max_seq_len):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=0.0, batch_first=True, activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

    def forward(self, input_ids, labels=None):
        B, N = input_ids.shape
        positions = torch.arange(N, device=input_ids.device).unsqueeze(0)
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        x = self.transformer(x)
        logits = self.lm_head(x)

        result = {"logits": logits}
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), labels.view(-1))
            result["loss"] = loss
        return result


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def benchmark_throughput(model, input_ids, n_runs=20, warmup=5):
    """Measure forward pass throughput."""
    model.eval()
    with torch.no_grad():
        # Warmup
        for _ in range(warmup):
            model(input_ids=input_ids)

        # Benchmark
        start = time.time()
        for _ in range(n_runs):
            model(input_ids=input_ids)
        elapsed = (time.time() - start) / n_runs

    return elapsed


def benchmark_training_step(model, input_ids, labels, n_runs=10, warmup=3):
    """Measure training step throughput."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model.train()

    # Warmup
    for _ in range(warmup):
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs["loss"]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Benchmark
    start = time.time()
    for _ in range(n_runs):
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs["loss"]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    elapsed = (time.time() - start) / n_runs

    return elapsed


def run_benchmarks():
    print("\n" + "="*70)
    print("  NEUROSPARK vs STANDARD TRANSFORMER BENCHMARK")
    print("="*70)

    # Configuration
    vocab_size = 1000
    d_model = 128
    n_heads = 4
    n_layers = 6
    d_ff = 256
    max_seq_len = 512

    # Create models
    config = NeuroSparkConfig(
        vocab_size=vocab_size, d_model=d_model, n_heads=n_heads,
        n_layers=n_layers, d_ff=d_ff, n_experts=8, max_k=2,
        max_seq_len=max_seq_len, linear_threshold=128,
        n_thoughts=4, max_reasoning_steps=3,
    )
    neurospark = NeuroSparkModel(config)

    standard = StandardTransformer(
        vocab_size=vocab_size, d_model=d_model, n_heads=n_heads,
        n_layers=n_layers, d_ff=d_ff, max_seq_len=max_seq_len,
    )

    # Parameter comparison
    ns_params = neurospark.count_parameters()
    std_params = count_params(standard)

    print(f"\n--- Parameter Comparison ---")
    print(f"  Standard Transformer:  {std_params:>10,} params (all active)")
    print(f"  NeuroSpark total:      {ns_params['total_params']:>10,} params")
    print(f"  NeuroSpark active:     {ns_params['active_params_per_token']:>10,} params/token")
    print(f"  NeuroSpark efficiency: {ns_params['efficiency_ratio']:.1%} active")

    # Throughput benchmarks
    print(f"\n--- Inference Throughput ---")
    for seq_len in [32, 64, 128, 256]:
        input_ids = torch.randint(0, vocab_size, (4, seq_len))
        labels = torch.randint(0, vocab_size, (4, seq_len))

        ns_time = benchmark_throughput(neurospark, input_ids, n_runs=15)
        std_time = benchmark_throughput(standard, input_ids, n_runs=15)

        speedup = std_time / ns_time if ns_time > 0 else float('inf')
        print(f"  Seq len {seq_len:>4}: Standard={std_time*1000:.1f}ms | "
              f"NeuroSpark={ns_time*1000:.1f}ms | "
              f"{'Speedup' if speedup >= 1 else 'Slowdown'}: {speedup:.2f}x")

    # Training step benchmarks
    print(f"\n--- Training Step ---")
    input_ids = torch.randint(0, vocab_size, (4, 64))
    labels = torch.randint(0, vocab_size, (4, 64))

    ns_train = benchmark_training_step(neurospark, input_ids, labels, n_runs=8)
    std_train = benchmark_training_step(standard, input_ids, labels, n_runs=8)

    print(f"  Standard:   {std_train*1000:.1f}ms/step")
    print(f"  NeuroSpark: {ns_train*1000:.1f}ms/step")

    # Quality comparison (loss after N steps)
    print(f"\n--- Learning Speed (same data, 100 steps) ---")
    from training.trainer import SyntheticDataset, NeuroSparkTrainer

    dataset = SyntheticDataset(vocab_size=vocab_size, seq_len=64, n_samples=512, task="pattern")

    # Train NeuroSpark
    ns_model = NeuroSparkModel(config)
    ns_trainer = NeuroSparkTrainer(
        ns_model, config, lr=1e-3, warmup_steps=10, max_steps=100, log_interval=25
    )
    print("\n  Training NeuroSpark:")
    ns_history = ns_trainer.train(dataset, batch_size=16)

    # Train Standard
    std_model = StandardTransformer(
        vocab_size=vocab_size, d_model=d_model, n_heads=n_heads,
        n_layers=n_layers, d_ff=d_ff, max_seq_len=max_seq_len
    )
    std_optimizer = torch.optim.AdamW(std_model.parameters(), lr=1e-3)
    std_model.train()

    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, drop_last=True)

    print("\n  Training Standard Transformer:")
    std_losses = []
    step = 0
    for batch in dataloader:
        if step >= 100:
            break
        outputs = std_model(input_ids=batch["input_ids"], labels=batch["labels"])
        loss = outputs["loss"]
        std_optimizer.zero_grad()
        loss.backward()
        std_optimizer.step()
        if step % 25 == 0:
            print(f"  Step {step:4d} | Loss: {loss.item():.4f}")
            std_losses.append(loss.item())
        step += 1

    # Summary
    print(f"\n--- SUMMARY ---")
    print(f"  NeuroSpark has {ns_params['total_params']/std_params:.1f}x more total params")
    print(f"  But only uses {ns_params['active_params_per_token']/std_params:.1%} of them per token")
    if ns_history:
        ns_final = ns_history[-1]["loss"]
        print(f"  NeuroSpark final loss: {ns_final:.4f}")
    if std_losses:
        print(f"  Standard final loss:   {std_losses[-1]:.4f}")

    # Adaptive behavior
    print(f"\n--- Adaptive Behavior ---")
    ns_model.eval()
    with torch.no_grad():
        simple_input = torch.zeros(1, 16, dtype=torch.long)  # Simple input
        complex_input = torch.randint(0, vocab_size, (1, 64))  # Complex input

        _ = ns_model(input_ids=simple_input)
        simple_diag = ns_model.get_diagnostics()

        _ = ns_model(input_ids=complex_input)
        complex_diag = ns_model.get_diagnostics()

    print(f"  Simple input  → Depth: {simple_diag.get('avg_depth', 'N/A'):.1f}, "
          f"Reasoning steps: {simple_diag.get('reasoning_steps', 'N/A')}")
    print(f"  Complex input → Depth: {complex_diag.get('avg_depth', 'N/A'):.1f}, "
          f"Reasoning steps: {complex_diag.get('reasoning_steps', 'N/A')}")

    print("\n" + "="*70)
    print("  BENCHMARK COMPLETE")
    print("="*70)


if __name__ == "__main__":
    run_benchmarks()
