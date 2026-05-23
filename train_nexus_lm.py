#!/usr/bin/env python3
"""
NEXUS-LM Training Script

Trains NEXUS as a real language model on text data.

Usage:
    # Full pipeline (download data → train tokenizer → train model):
    python3 prepare_lm_data.py
    python3 train_nexus_lm.py

    # Options:
    python3 train_nexus_lm.py --config base          # 37M params (recommended for 4060)
    python3 train_nexus_lm.py --config small          # 22M params (faster)
    python3 train_nexus_lm.py --config large --bs 4   # 80M params (tight fit)
    python3 train_nexus_lm.py --resume checkpoint.pt  # Resume training
"""

import sys
import os
import argparse
import math
import time
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(line_buffering=True)

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from nexus.lm.model import NexusLM, NexusLMConfig
from nexus.lm.tokenizer import BPETokenizer


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lm_data")


def get_lr(step, warmup, max_steps, max_lr, min_lr=1e-6):
    if step < warmup:
        return max_lr * step / max(1, warmup)
    progress = (step - warmup) / max(1, max_steps - warmup)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


def train(
    model, train_data, val_data, max_steps, lr=3e-4, warmup=1000,
    batch_size=16, grad_accum=1, log_every=100, eval_every=1000,
    device="cuda", use_amp=True, checkpoint_dir="lm_checkpoints",
    tokenizer=None,
):
    """Train the language model."""
    os.makedirs(checkpoint_dir, exist_ok=True)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=0.1, betas=(0.9, 0.95)
    )

    use_scaler = use_amp and device == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_scaler else None

    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, drop_last=True,
        num_workers=2 if device == "cuda" else 0,
        pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, drop_last=True) if val_data else None

    model.train()
    step = 0
    running_loss = 0
    n_loss = 0
    best_val_loss = float('inf')
    t_start = time.time()

    while step < max_steps:
        for batch in train_loader:
            if step >= max_steps:
                break

            current_lr = get_lr(step, warmup, max_steps, lr)
            for pg in optimizer.param_groups:
                pg["lr"] = current_lr

            input_ids = batch[0].to(device, non_blocking=True)
            labels = input_ids.clone()  # For LM, labels = input (model shifts internally)

            if use_scaler:
                with torch.amp.autocast("cuda"):
                    outputs = model(input_ids=input_ids, labels=labels)
                    loss = outputs["loss"] / grad_accum
                scaler.scale(loss).backward()
            else:
                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs["loss"] / grad_accum
                loss.backward()

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
            n_loss += 1

            # Logging
            if step % log_every == 0 and step > 0:
                avg_loss = running_loss / n_loss
                ppl = math.exp(min(avg_loss, 20))
                elapsed = time.time() - t_start
                steps_per_sec = step / max(1, elapsed)
                eta = (max_steps - step) / max(0.01, steps_per_sec)
                print(
                    f"  Step {step:>6}/{max_steps} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"PPL: {ppl:.1f} | "
                    f"LR: {current_lr:.2e} | "
                    f"{steps_per_sec:.1f} step/s | "
                    f"ETA: {eta/60:.0f}min",
                    flush=True,
                )
                running_loss = n_loss = 0

            # Evaluation
            if step % eval_every == 0 and step > 0:
                val_loss = evaluate(model, val_loader, device, use_amp)
                val_ppl = math.exp(min(val_loss, 20))
                print(f"  [EVAL] Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.1f}", flush=True)

                # Generate sample
                if tokenizer:
                    generate_sample(model, tokenizer, device)

                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save({
                        "model_state_dict": model.state_dict(),
                        "config": model.config.to_dict(),
                        "step": step,
                        "val_loss": val_loss,
                    }, os.path.join(checkpoint_dir, "best.pt"))
                    print(f"  [BEST] Saved (val_loss={val_loss:.4f})", flush=True)

                model.train()

            # Checkpoint every 5000 steps
            if step > 0 and step % 5000 == 0:
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": model.config.to_dict(),
                    "step": step,
                }, os.path.join(checkpoint_dir, f"step_{step}.pt"))

            step += 1

    return best_val_loss


@torch.no_grad()
def evaluate(model, val_loader, device, use_amp):
    """Evaluate on validation set."""
    if val_loader is None:
        return 0.0
    model.eval()
    total_loss = 0
    n = 0
    for batch in val_loader:
        input_ids = batch[0].to(device)
        labels = input_ids.clone()
        if use_amp and device == "cuda":
            with torch.amp.autocast("cuda"):
                outputs = model(input_ids=input_ids, labels=labels)
        else:
            outputs = model(input_ids=input_ids, labels=labels)
        total_loss += outputs["loss"].item()
        n += 1
        if n >= 50:  # Quick eval
            break
    return total_loss / max(1, n)


@torch.no_grad()
def generate_sample(model, tokenizer, device, prompt="Once upon a time"):
    """Generate a text sample for monitoring training progress."""
    model.eval()
    input_ids = torch.tensor([tokenizer.encode(prompt, add_bos=True)], dtype=torch.long, device=device)

    output_ids = model.generate(
        input_ids, max_new_tokens=100, temperature=0.8, top_k=50, top_p=0.9
    )

    text = tokenizer.decode(output_ids[0].tolist())
    # Clean up for display
    text = text[:300]
    print(f"  [GEN] \"{text}\"", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Train NEXUS Language Model")
    parser.add_argument("--config", choices=["tiny", "small", "base", "large"], default="base")
    parser.add_argument("--bs", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()

    # Device
    if args.cpu or not torch.cuda.is_available():
        device = "cpu"
        use_amp = False
        print("  Device: CPU (slow - use GPU for real training!)", flush=True)
    else:
        device = "cuda"
        use_amp = not args.no_amp
        print(f"  Device: {torch.cuda.get_device_name(0)}", flush=True)

    # Load tokenizer
    tokenizer_dir = os.path.join(DATA_DIR, "tokenizer")
    if not os.path.exists(os.path.join(tokenizer_dir, "tokenizer.json")):
        print("  ERROR: Tokenizer not found! Run prepare_lm_data.py first.", flush=True)
        sys.exit(1)
    tokenizer = BPETokenizer.load(tokenizer_dir)
    print(f"  Tokenizer: vocab_size={tokenizer.vocab_size}", flush=True)

    # Load data
    train_path = os.path.join(DATA_DIR, "train.pt")
    val_path = os.path.join(DATA_DIR, "val.pt")

    if not os.path.exists(train_path):
        print("  ERROR: Training data not found! Run prepare_lm_data.py first.", flush=True)
        sys.exit(1)

    print(f"  Loading training data...", flush=True)
    train_data_raw = torch.load(train_path, map_location="cpu", weights_only=False)
    train_dataset = TensorDataset(train_data_raw["input_ids"])
    seq_len = train_data_raw["seq_len"]
    print(f"  Train: {len(train_dataset):,} sequences × {seq_len} tokens", flush=True)

    val_dataset = None
    if os.path.exists(val_path):
        val_data_raw = torch.load(val_path, map_location="cpu", weights_only=False)
        val_dataset = TensorDataset(val_data_raw["input_ids"])
        print(f"  Val:   {len(val_dataset):,} sequences × {val_data_raw['seq_len']} tokens", flush=True)

    # Config
    config_map = {"tiny": NexusLMConfig.tiny, "small": NexusLMConfig.small,
                  "base": NexusLMConfig.base, "large": NexusLMConfig.large}
    config = config_map[args.config]()
    config.vocab_size = tokenizer.vocab_size
    config.max_seq_len = seq_len

    # Default batch sizes (tuned for 4060 8GB with AMP; tiny for CPU)
    default_bs = {"tiny": 8, "small": 16, "base": 8, "large": 4}
    batch_size = args.bs or default_bs[args.config]

    # Default steps
    default_steps = {"tiny": 5000, "small": 30000, "base": 50000, "large": 80000}
    max_steps = args.max_steps or default_steps[args.config]

    # Create model
    model = NexusLM(config).to(device)
    params = model.count_parameters()

    if args.resume:
        print(f"  Resuming from {args.resume}...", flush=True)
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])

    print(f"\n{'='*70}")
    print(f"  NEXUS-LM {args.config.upper()} Training")
    print(f"{'='*70}")
    print(f"  d_model:     {config.d_model}")
    print(f"  n_heads:     {config.n_heads}")
    print(f"  d_ff:        {config.d_ff}")
    print(f"  iterations:  {config.n_iterations} (CISA depth)")
    print(f"  max_seq_len: {config.max_seq_len}")
    print(f"  vocab_size:  {config.vocab_size}")
    print(f"  Real params:      {params['total_params']:>12,}")
    print(f"  Effective params: {params['effective_params']:>12,}")
    print(f"  Batch size:  {batch_size} (× {args.grad_accum} grad accum)")
    print(f"  Max steps:   {max_steps:,}")
    print(f"  Mixed prec:  {use_amp}")
    print(f"{'='*70}", flush=True)

    # Initial sample
    print(f"\n  Initial generation (random weights):", flush=True)
    generate_sample(model, tokenizer, device)

    # Train
    print(f"\n  Starting training...", flush=True)
    best_val_loss = train(
        model, train_dataset, val_dataset,
        max_steps=max_steps, lr=args.lr, warmup=min(2000, max_steps // 10),
        batch_size=batch_size, grad_accum=args.grad_accum,
        log_every=100, eval_every=2000,
        device=device, use_amp=use_amp,
        tokenizer=tokenizer,
    )

    # Save final model
    final_path = f"nexus_lm_{args.config}.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config.to_dict(),
        "best_val_loss": best_val_loss,
        "params": params,
    }, final_path)

    # Final generation
    print(f"\n{'='*70}")
    print(f"  FINAL GENERATION SAMPLES")
    print(f"{'='*70}", flush=True)
    prompts = [
        "Once upon a time",
        "The little cat",
        "One day, a brave",
        "In a magical forest",
        "The children were playing",
    ]
    for prompt in prompts:
        print(f"\n  Prompt: \"{prompt}\"", flush=True)
        generate_sample(model, tokenizer, device, prompt)

    print(f"\n{'='*70}")
    print(f"  NEXUS-LM {args.config.upper()} Training Complete!")
    print(f"  Best val loss: {best_val_loss:.4f} (PPL: {math.exp(min(best_val_loss, 20)):.1f})")
    print(f"  Model saved:   {final_path}")
    print(f"{'='*70}", flush=True)


if __name__ == "__main__":
    main()
