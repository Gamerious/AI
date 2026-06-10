#!/usr/bin/env python3
"""S3 2x2 matrix: does routing supervised plans across positions help?

Trains the four variants
    base          (v3 defaults)
    plan          (--plan: state channel predicts tokens i+2..i+1+H)
    xstate        (--cross-state: positions read others' latest states)
    plan_xstate   (both: tokens read their predecessors' PLANS)
with identical data order and step budget, then reports val lm-loss/PPL,
plan-loss (do states learn the future?) and the learned gates.

Headline hypothesis: plan x cross_state is SUPERADDITIVE - a plan is worth
more if others can read it. A CPU/tiny run gives a first directional signal
only; the deciding run is base config on the 4060.

Usage:
    python run_s3_matrix.py                          # tiny, CPU-friendly
    python run_s3_matrix.py --config base --bs 8 --steps 20000   # 4060
    # gate hypothesis (CPU pilot: gates barely opened -> synergy couldn't show):
    python run_s3_matrix.py --config base --bs 8 --steps 20000 \
        --variants xstate,plan_xstate --gate-init -1.0
"""
import sys, os, json, math, time, argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(line_buffering=True)

import torch
from torch.utils.data import DataLoader, TensorDataset

from nexus.lm.model import NexusLM, NexusLMConfig, CISAttention
from train_nexus_lm import get_lr

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lm_data")

VARIANTS = {
    "base":        dict(),
    "plan":        dict(plan_states=True),
    "xstate":      dict(cross_state=True),
    "plan_xstate": dict(plan_states=True, cross_state=True),
}


@torch.no_grad()
def evaluate(model, val_loader, device, max_batches):
    model.eval()
    lm_sum, plan_sum, n, n_plan = 0.0, 0.0, 0, 0
    for i, (ids,) in enumerate(val_loader):
        if i >= max_batches:
            break
        out = model(input_ids=ids.to(device), labels=ids.to(device).clone())
        lm_sum += out["lm_loss"].item()
        n += 1
        if "plan_loss" in out:
            plan_sum += out["plan_loss"].item()
            n_plan += 1
    model.train()
    return lm_sum / max(1, n), (plan_sum / n_plan) if n_plan else None


def gates(model):
    for m in model.modules():
        if isinstance(m, CISAttention):
            g = {"temporal_gate": torch.sigmoid(m.temporal_gate.detach()).mean().item()}
            if hasattr(m, "xstate_gate"):
                g["xstate_gate"] = torch.sigmoid(m.xstate_gate.detach()).mean().item()
            return g
    return {}


def run_variant(name, overrides, args, train_ids, val_loader, device, vocab_size, seq_len):
    torch.manual_seed(args.seed)
    cfg = getattr(NexusLMConfig, args.config)()
    cfg.vocab_size = vocab_size
    cfg.max_seq_len = seq_len
    for k, v in overrides.items():
        setattr(cfg, k, v)
    if args.plan_horizon is not None and cfg.plan_states:
        cfg.plan_horizon = args.plan_horizon
    if args.gate_init is not None:
        # Seeds BOTH gates (temporal + xstate). The CPU pilot showed gates
        # barely opening within the step budget (0.119 -> ~0.128) - this
        # tests whether the routing path needs a more open start.
        cfg.temporal_gate_init = args.gate_init

    model = NexusLM(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1,
                            betas=(0.9, 0.95))

    # identical batch order for every variant
    loader = DataLoader(TensorDataset(train_ids), batch_size=args.bs, shuffle=True,
                        drop_last=True, generator=torch.Generator().manual_seed(args.seed))

    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    print(f"\n=== {name}  ({n_params:,} params) ===", flush=True)
    model.train()
    step, t0, history = 0, time.time(), []
    while step < args.steps:
        for (ids,) in loader:
            if step >= args.steps:
                break
            lr = get_lr(step, args.warmup, args.steps, args.lr)
            for pg in opt.param_groups:
                pg["lr"] = lr
            ids = ids.to(device, non_blocking=True)
            if use_amp:
                with torch.amp.autocast("cuda"):
                    out = model(input_ids=ids, labels=ids.clone())
                scaler.scale(out["loss"]).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                out = model(input_ids=ids, labels=ids.clone())
                out["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            opt.zero_grad()
            step += 1

            if step % args.eval_every == 0 or step == args.steps:
                vl, pl = evaluate(model, val_loader, device,
                                  args.eval_batches if step < args.steps else args.final_eval_batches)
                g = gates(model)
                history.append({"step": step, "val_lm": vl, "val_plan": pl, **g})
                rate = step / (time.time() - t0)
                msg = f"  step {step:>6} | val lm {vl:.4f} (ppl {math.exp(min(vl, 20)):.2f})"
                if pl is not None:
                    msg += f" | plan {pl:.4f}"
                msg += f" | gates {g} | {rate:.1f} st/s"
                print(msg, flush=True)

    final = history[-1]
    ckpt_path = None
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        ckpt_path = os.path.join(args.save_dir, f"{args.config}_{name}.pt")
        torch.save({"model_state_dict": model.state_dict(), "config": cfg.to_dict(),
                    "step": args.steps, "val_loss": final["val_lm"]}, ckpt_path)
        print(f"  saved -> {ckpt_path}", flush=True)
    return {"name": name, "params": n_params, "history": history,
            "final_val_lm": final["val_lm"], "final_ppl": math.exp(min(final["val_lm"], 20)),
            "final_plan": final["val_plan"], "gates": gates(model), "checkpoint": ckpt_path}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", choices=["tiny", "small", "base", "large"], default="tiny")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--bs", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--plan-horizon", type=int, default=None)
    p.add_argument("--gate-init", type=float, default=None,
                   help="override gate init logit for temporal AND xstate gates "
                        "(default -2.0 = 12%%; -1.0 = 27%% opens the state paths faster)")
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--eval-batches", type=int, default=30)
    p.add_argument("--final-eval-batches", type=int, default=120)
    p.add_argument("--variants", type=str, default=",".join(VARIANTS),
                   help="comma-separated subset of: " + ",".join(VARIANTS))
    p.add_argument("--save-dir", type=str, default="s3_matrix_models",
                   help="save the final model per variant here ('' disables; "
                        "checkpoints are chat_nexus.py-compatible)")
    p.add_argument("--out", type=str, default=None,
                   help="results JSON (default: auto-named from config/steps/gate)")
    args = p.parse_args()
    if args.out is None:
        gate = f"_gate{args.gate_init}" if args.gate_init is not None else ""
        args.out = f"s3_matrix_{args.config}_{args.steps}{gate}.json"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_raw = torch.load(os.path.join(DATA_DIR, "train.pt"), map_location="cpu", weights_only=False)
    val_raw = torch.load(os.path.join(DATA_DIR, "val.pt"), map_location="cpu", weights_only=False)
    seq_len = train_raw["seq_len"]
    with open(os.path.join(DATA_DIR, "tokenizer", "tokenizer.json")) as f:
        vocab_size = json.load(f)["vocab_size"]
    val_loader = DataLoader(TensorDataset(val_raw["input_ids"]), batch_size=args.bs,
                            shuffle=False, drop_last=True)
    print(f"device={device} config={args.config} steps={args.steps} bs={args.bs} "
          f"vocab={vocab_size} seq={seq_len} seed={args.seed} "
          f"gate_init={args.gate_init if args.gate_init is not None else 'default(-2.0)'}",
          flush=True)

    results = []
    for name in args.variants.split(","):
        results.append(run_variant(name, VARIANTS[name], args, train_raw["input_ids"],
                                   val_loader, device, vocab_size, seq_len))
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\n{'=' * 74}")
    print(f"{'variant':<14} {'params':>10} {'val lm':>8} {'PPL':>7} {'plan':>7}  gates")
    print("-" * 74)
    base_lm = next((r["final_val_lm"] for r in results if r["name"] == "base"), None)
    for r in results:
        d = f" ({r['final_val_lm'] - base_lm:+.4f})" if base_lm and r["name"] != "base" else ""
        plan = f"{r['final_plan']:.3f}" if r["final_plan"] is not None else "-"
        print(f"{r['name']:<14} {r['params']:>10,} {r['final_val_lm']:>8.4f} "
              f"{r['final_ppl']:>7.2f} {plan:>7}  {r['gates']}{d}")
    # superadditivity check
    by = {r["name"]: r["final_val_lm"] for r in results}
    if all(k in by for k in VARIANTS):
        d_plan = by["base"] - by["plan"]
        d_x = by["base"] - by["xstate"]
        d_combo = by["base"] - by["plan_xstate"]
        synergy = d_combo - (d_plan + d_x)
        print(f"\nplan effect {d_plan:+.4f} | xstate effect {d_x:+.4f} | "
              f"combo {d_combo:+.4f} | SYNERGY {synergy:+.4f} "
              f"({'superadditive' if synergy > 0 else 'sub/additive'})")
    print(f"results -> {args.out}")


if __name__ == "__main__":
    main()
