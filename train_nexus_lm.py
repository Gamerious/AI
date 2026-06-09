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
    tokenizer=None, resume_ckpt=None,
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

    if resume_ckpt is not None:
        if "optimizer_state_dict" in resume_ckpt:
            optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
            print(f"  Loaded optimizer state from checkpoint", flush=True)
        if "step" in resume_ckpt:
            step = int(resume_ckpt["step"])
            print(f"  Resuming from step {step}", flush=True)
        if "val_loss" in resume_ckpt:
            best_val_loss = float(resume_ckpt["val_loss"])

    t_start = time.time()
    t_offset_steps = step  # for accurate step/s & ETA after resume

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
                steps_per_sec = max(1, step - t_offset_steps) / max(1, elapsed)
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
        # Pure LM loss for PPL (with adaptive halting, "loss" includes the
        # ponder cost and would inflate the perplexity).
        total_loss += outputs.get("lm_loss", outputs["loss"]).item()
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
    parser.add_argument("--detach-state", action="store_true",
                        help="Reproduce the original buggy CISA (GRU frozen, no grad into state)")
    parser.add_argument("--n-iter", type=int, default=None,
                        help="Override n_iterations (for ablations, e.g. 1)")
    parser.add_argument("--gate-init", type=float, default=None,
                        help="Override temporal_gate init logit (-2.0=12%%, -1.0=27%%, 0.0=50%%)")
    # NEXUS-2 fixes. A1-A4 are now the DEFAULTS (in NexusLMConfig); the
    # --no-* forms turn them off for ablations.
    parser.add_argument("--gate-mode", choices=["logbias", "channel"], default=None,
                        help="A1: spatial/temporal mixing (default: channel)")
    parser.add_argument("--nope-temporal", action=argparse.BooleanOptionalAction, default=None,
                        help="A2: NoPE query for temporal path (default: on)")
    parser.add_argument("--qk-norm", action=argparse.BooleanOptionalAction, default=None,
                        help="A3: QK-Norm on q/k (default: on)")
    parser.add_argument("--stabilize", action=argparse.BooleanOptionalAction, default=None,
                        help="A4: no-tanh state_init + GRU-input RMSNorm + ReZero iter_emb (default: on)")
    parser.add_argument("--deep-supervision", action=argparse.BooleanOptionalAction, default=None,
                        help="LM loss after every iteration (default: off)")
    parser.add_argument("--cisa-v2", action="store_true",
                        help="Legacy shortcut: A1-A4 (now defaults) + deep supervision")
    # NEXUS-3
    parser.add_argument("--current-state", action=argparse.BooleanOptionalAction, default=None,
                        help="B1: temporal path sees the CURRENT state s_t (default: on)")
    parser.add_argument("--cross-state", action=argparse.BooleanOptionalAction, default=None,
                        help="S1: cross-position state attention (default: off)")
    parser.add_argument("--halting", action=argparse.BooleanOptionalAction, default=None,
                        help="S2: ACT adaptive per-token iteration depth (default: off)")
    parser.add_argument("--ponder-weight", type=float, default=None,
                        help="Weight of the ACT ponder cost in the loss (default 0.01)")
    parser.add_argument("--sdpa", action=argparse.BooleanOptionalAction, default=None,
                        help="Flash attention via F.scaled_dot_product_attention (default: on)")
    parser.add_argument("--grad-checkpoint", action="store_true",
                        help="Recompute iterations in backward (fits K=8 / large on 8GB)")
    args = parser.parse_args()

    # --cisa-v2 shortcut (legacy): A1-A4 are defaults now, just adds deep supervision
    if args.cisa_v2 and args.deep_supervision is None:
        args.deep_supervision = True

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
    config.detach_state_history = args.detach_state
    if args.n_iter is not None:
        config.n_iterations = args.n_iter
    if args.gate_init is not None:
        config.temporal_gate_init = args.gate_init
    # Only override config defaults when a flag was given explicitly
    for arg_name, cfg_name in [
        ("gate_mode", "gate_mode"), ("nope_temporal", "nope_temporal"),
        ("qk_norm", "qk_norm"), ("stabilize", "stabilize"),
        ("deep_supervision", "deep_supervision"),
        ("current_state", "include_current_state"),
        ("cross_state", "cross_state"), ("halting", "adaptive_halting"),
        ("ponder_weight", "ponder_weight"), ("sdpa", "use_sdpa"),
    ]:
        val = getattr(args, arg_name)
        if val is not None:
            setattr(config, cfg_name, val)
    config.grad_checkpoint = args.grad_checkpoint

    # Default batch sizes (tuned for 4060 8GB with AMP; tiny for CPU)
    default_bs = {"tiny": 8, "small": 16, "base": 8, "large": 4}
    batch_size = args.bs or default_bs[args.config]

    # Default steps
    default_steps = {"tiny": 5000, "small": 30000, "base": 50000, "large": 80000}
    max_steps = args.max_steps or default_steps[args.config]

    # Run name -> isolates checkpoints/final model per variant (no clobbering!)
    # Suffixes mark DEVIATIONS from the current defaults (A1-A4 + B1 on).
    run_name = args.config
    if config.detach_state_history:
        run_name += "_detach"
    if args.n_iter is not None:
        run_name += f"_iter{args.n_iter}"
    if args.gate_init is not None:
        run_name += f"_gate{args.gate_init}"
    if config.gate_mode != "channel":
        run_name += "_logbias"
    if not config.nope_temporal:
        run_name += "_ropet"
    if not config.qk_norm:
        run_name += "_noqknorm"
    if not config.stabilize:
        run_name += "_nostab"
    if not config.include_current_state:
        run_name += "_nocur"
    if config.deep_supervision:
        run_name += "_ds"
    if config.cross_state:
        run_name += "_xstate"
    if config.adaptive_halting:
        run_name += "_act"
    if run_name == args.config:
        run_name += "_v3"  # the NEXUS-3 default bundle
    checkpoint_dir = f"lm_checkpoints_{run_name}"

    # Create model
    model = NexusLM(config).to(device)
    params = model.count_parameters()

    resume_ckpt = None
    if args.resume:
        print(f"  Resuming from {args.resume}...", flush=True)
        resume_ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(resume_ckpt["model_state_dict"])

    print(f"\n{'='*70}")
    print(f"  NEXUS-LM {args.config.upper()} Training")
    print(f"{'='*70}")
    print(f"  d_model:     {config.d_model}")
    print(f"  n_heads:     {config.n_heads}")
    print(f"  d_ff:        {config.d_ff}")
    print(f"  iterations:  {config.n_iterations} (CISA depth)")
    print(f"  CISA state:  {'DETACHED (buggy/frozen GRU)' if config.detach_state_history else 'LEARNED (fixed - GRU trains)'}")
    print(f"  NEXUS-2:     gate={config.gate_mode} nope={config.nope_temporal} qk_norm={config.qk_norm} "
          f"stabilize={config.stabilize} deep_sup={config.deep_supervision}")
    print(f"  NEXUS-3:     current_state={config.include_current_state} cross_state={config.cross_state} "
          f"halting={config.adaptive_halting} sdpa={config.use_sdpa} grad_ckpt={config.grad_checkpoint}")
    print(f"  run name:    {run_name}  ->  checkpoints in {checkpoint_dir}/")
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
        device=device, use_amp=use_amp, checkpoint_dir=checkpoint_dir,
        tokenizer=tokenizer, resume_ckpt=resume_ckpt,
    )

    # Save final model
    final_path = f"nexus_lm_{run_name}.pt"
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
