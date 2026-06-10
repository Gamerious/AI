#!/usr/bin/env python3
"""NEXUS-3 architecture test battery.

Verifies, on a tiny CPU model, for every architecture variant:
  1. CAUSALITY      changing future tokens never changes past logits
                    (the Prometheus failure mode - must hold for cross_state!)
  2. KV-CACHE       incremental decoding is EXACTLY full recomputation
  3. GRADIENTS      state path trains (and is frozen with --detach-state)
  4. SDPA PARITY    flash attention == manual attention
  5. GRAD CKPT      checkpointed backward == normal backward
  6. HALTING        ACT bookkeeping sane, ponder cost differentiable
  7. LEGACY LOAD    old checkpoint configs restore old behavior
"""
import sys, os, copy
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(line_buffering=True)

import torch
from nexus.lm.model import NexusLM, NexusLMConfig, NexusKVCache

torch.manual_seed(0)
DEVICE = "cpu"
PASS, FAIL = 0, []


def check(name, cond, detail=""):
    global PASS
    if cond:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL.append(name)
        print(f"  FAIL {name}  {detail}")


def make(seed=0, **overrides):
    torch.manual_seed(seed)
    cfg = NexusLMConfig(
        vocab_size=64, d_model=32, n_heads=2, d_ff=64,
        max_seq_len=64, n_iterations=3, max_iterations=8, dropout=0.0,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    m = NexusLM(cfg)
    m.eval()
    return m


VARIANTS = {
    "legacy":         dict(gate_mode="logbias", nope_temporal=False, qk_norm=False,
                           stabilize=False, include_current_state=False, use_sdpa=False),
    "v3_default":     dict(),
    "v3_logbias":     dict(gate_mode="logbias"),
    "xstate":         dict(cross_state=True),
    "xstate_logbias": dict(cross_state=True, gate_mode="logbias"),
    "act":            dict(adaptive_halting=True),
    "act_xstate":     dict(adaptive_halting=True, cross_state=True),
}

B, N = 2, 24
ids = torch.randint(0, 64, (B, N))

# ---------------------------------------------------------------- causality
print("\n[1] Causality: past logits invariant to future tokens")
for name, ov in VARIANTS.items():
    m = make(**ov)
    with torch.no_grad():
        full = m(ids)["logits"]
        corrupted = ids.clone()
        corrupted[:, N // 2:] = torch.randint(0, 64, (B, N - N // 2))
        cor = m(corrupted)["logits"]
    delta = (full[:, :N // 2] - cor[:, :N // 2]).abs().max().item()
    check(f"causal[{name}]", delta < 1e-5, f"max delta {delta:.2e}")

# ---------------------------------------------------------- cache exactness
print("\n[2] KV cache: incremental == full recompute")
for name, ov in VARIANTS.items():
    m = make(**ov)
    with torch.no_grad():
        full = m(ids)["logits"]

        # (a) prefill in one chunk
        cache = NexusKVCache(m.config.n_iterations)
        pre = m(ids, kv_cache=cache)["logits"]
        d_pre = (full - pre).abs().max().item()

        # (b) prefill half, decode the rest token by token
        cache = NexusKVCache(m.config.n_iterations)
        half = N // 2
        outs = [m(ids[:, :half], kv_cache=cache)["logits"]]
        for j in range(half, N):
            outs.append(m(ids[:, j:j + 1], kv_cache=cache,
                          pos_offset=cache.seq_len)["logits"])
        inc = torch.cat(outs, dim=1)
        d_inc = (full - inc).abs().max().item()
    check(f"cache_prefill[{name}]", d_pre < 1e-5, f"max delta {d_pre:.2e}")
    check(f"cache_decode[{name}]", d_inc < 1e-4, f"max delta {d_inc:.2e}")

# greedy generation must agree between cached and uncached paths
for name in ["v3_default", "xstate", "act", "act_xstate", "xstate_logbias"]:
    m = make(**VARIANTS[name])
    prompt = ids[:1, :5]
    g1 = m.generate(prompt, max_new_tokens=20, temperature=0, eos_id=-1, use_cache=True)
    g2 = m.generate(prompt, max_new_tokens=20, temperature=0, eos_id=-1, use_cache=False)
    check(f"greedy_cache[{name}]", torch.equal(g1, g2),
          f"\n    cache:   {g1[0].tolist()}\n    nocache: {g2[0].tolist()}")

# window overflow: generation far past max_seq_len must not crash and must
# keep the cache within bounds (rebuild path)
m = make(max_seq_len=16, **VARIANTS["xstate"])
g = m.generate(ids[:1, :5], max_new_tokens=40, temperature=0, eos_id=-1, use_cache=True)
check("cache_window_overflow", g.shape[1] == 45, f"got {g.shape[1]} tokens")

# ------------------------------------------------------------- gradient flow
print("\n[3] Gradient flow into the state path")
m = make()
m.train()
out = m(ids, labels=ids.clone())
out["loss"].backward()
gru_g = m.cell.state_update.W_z.weight.grad
init_g = m.state_init.weight.grad
kst_g = m.cell.attention.k_state_proj.weight.grad
check("grad[GRU W_z]", gru_g is not None and gru_g.norm() > 0)
check("grad[state_init]", init_g is not None and init_g.norm() > 0)
check("grad[k_state_proj]", kst_g is not None and kst_g.norm() > 0)

m = make(detach_state_history=True)
m.train()
m(ids, labels=ids.clone())["loss"].backward()
g = m.cell.state_update.W_z.weight.grad
check("grad[detach -> GRU frozen]", g is None or g.norm() == 0,
      f"norm {0 if g is None else g.norm().item():.2e}")

# all-params check on the full feature set (ReZero zero-inits excepted:
# iter_embeddings sit behind tanh(iter_scale)=0 at init by design)
m = make(cross_state=True, adaptive_halting=True)
m.train()
m(ids, labels=ids.clone())["loss"].backward()
allowed_zero = {"cell.iter_embeddings.weight"}
dead = [n for n, p in m.named_parameters()
        if (p.grad is None or p.grad.norm() == 0) and n not in allowed_zero]
check("grad[all params live (xstate+act)]", not dead, f"dead: {dead}")

# ---------------------------------------------------------------- SDPA parity
print("\n[4] SDPA (flash) == manual attention")
for name in ["v3_default", "xstate", "act_xstate"]:
    m = make(**VARIANTS[name])
    with torch.no_grad():
        a = m(ids)["logits"]
        m.cell.attention.use_sdpa = False
        b = m(ids)["logits"]
    d = (a - b).abs().max().item()
    check(f"sdpa[{name}]", d < 1e-4, f"max delta {d:.2e}")

# ------------------------------------------------------------ grad checkpoint
print("\n[5] Gradient checkpointing parity")
m1 = make(cross_state=True)
m2 = copy.deepcopy(m1)
m2.config.grad_checkpoint = True
for m in (m1, m2):
    m.train()
l1 = m1(ids, labels=ids.clone())["loss"]
l1.backward()
l2 = m2(ids, labels=ids.clone())["loss"]
l2.backward()
check("ckpt[loss equal]", torch.allclose(l1, l2, atol=1e-6),
      f"{l1.item():.6f} vs {l2.item():.6f}")
worst = max(
    (p1.grad - p2.grad).abs().max().item()
    for (n, p1), (_, p2) in zip(m1.named_parameters(), m2.named_parameters())
    if p1.grad is not None and p2.grad is not None
)
check("ckpt[grads equal]", worst < 1e-5, f"max grad delta {worst:.2e}")

# --------------------------------------------------------------------- halting
print("\n[6] Adaptive halting (ACT)")
m = make(adaptive_halting=True)
m.train()
out = m(ids, labels=ids.clone())
K = m.config.n_iterations
check("act[ponder in output]", "ponder_cost" in out and torch.isfinite(out["ponder_cost"]))
check("act[loss = lm + ponder]",
      torch.allclose(out["loss"], out["lm_loss"] + m.config.ponder_weight * out["ponder_cost"]))
d = m.get_diagnostics()["avg_depth"]
check("act[1 <= avg_depth <= K]", 1.0 <= d <= K, f"avg_depth {d}")
out["loss"].backward()
hg = m.halt_head.weight.grad
check("act[halt head learns]", hg is not None and hg.norm() > 0)

# ----------------------------------------------------------------- legacy load
print("\n[7] Legacy checkpoint configs restore old behavior")
old_dict = {"vocab_size": 64, "d_model": 32, "n_heads": 2, "d_ff": 64,
            "max_seq_len": 64, "n_iterations": 3, "max_iterations": 8,
            "dropout": 0.0, "rope_theta": 10000.0, "temporal_gate_init": -2.0}
cfg = NexusLMConfig.from_dict(old_dict)
check("legacy[gate_mode]", cfg.gate_mode == "logbias")
check("legacy[A2-A4 off]", not cfg.nope_temporal and not cfg.qk_norm and not cfg.stabilize)
check("legacy[current_state off]", not cfg.include_current_state)
check("legacy[new features off]", not cfg.cross_state and not cfg.adaptive_halting)

# state dict of a legacy-shaped model must load into a fresh instance
m_old = NexusLM(cfg)
m_new = NexusLM(NexusLMConfig.from_dict(old_dict))
m_new.load_state_dict(m_old.state_dict())
check("legacy[state_dict roundtrip]", True)

# ----------------------------------------------------------------- S3 plans
print("\n[8] S3 plan states (state channel predicts the future)")
m = make(plan_states=True, plan_horizon=2)
m.train()
out = m(ids, labels=ids.clone())
check("plan[loss present]", "plan_loss" in out and torch.isfinite(out["plan_loss"]))
check("plan[loss = lm + w*plan]",
      torch.allclose(out["loss"],
                     out["lm_loss"] + m.config.plan_weight * out["plan_loss"], atol=1e-6))

# the plan objective ALONE must reach the GRU/state path (that's the point:
# it shapes what the state channel stores)
m.zero_grad()
out["plan_loss"].backward()
g = m.cell.state_update.W_z.weight.grad
check("plan[gradient reaches GRU]", g is not None and g.norm() > 0)

# triple combo: lm + plan + ponder decompose exactly
m = make(plan_states=True, plan_horizon=1, adaptive_halting=True)
m.train()
out = m(ids, labels=ids.clone())
expected = (out["lm_loss"] + m.config.plan_weight * out["plan_loss"]
            + m.config.ponder_weight * out["ponder_cost"])
check("plan[loss = lm + plan + ponder]", torch.allclose(out["loss"], expected, atol=1e-6))

# sequences too short for any horizon: plan term silently absent, no crash
out = m(ids[:, :2], labels=ids[:, :2].clone())
check("plan[short seq safe]", "loss" in out and "plan_loss" not in out)

# eval averages all horizons
m.eval()
with torch.no_grad():
    out = m(ids, labels=ids.clone())
check("plan[eval finite]", torch.isfinite(out["plan_loss"]))

# plan heads are train-time only: generation (cached & uncached) still agrees
g1 = m.generate(ids[:1, :5], max_new_tokens=12, temperature=0, eos_id=-1, use_cache=True)
g2 = m.generate(ids[:1, :5], max_new_tokens=12, temperature=0, eos_id=-1, use_cache=False)
check("plan[greedy cache match]", torch.equal(g1, g2))

# full S1+S3 stack: every parameter lives (H=1 so the single plan head is
# deterministically sampled)
m = make(plan_states=True, plan_horizon=1, cross_state=True)
m.train()
m(ids, labels=ids.clone())["loss"].backward()
dead = [n for n, p in m.named_parameters()
        if (p.grad is None or p.grad.norm() == 0) and n not in allowed_zero]
check("plan[all params live (plan+xstate)]", not dead, f"dead: {dead}")

# -------------------------------------------------------------- train smoke
print("\n[9] Training smoke test (overfit one batch, full feature set)")
m = make(cross_state=True, adaptive_halting=True, deep_supervision=True,
         plan_states=True, plan_horizon=2)
m.train()
opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
losses = []
for _ in range(80):
    out = m(ids, labels=ids.clone())
    opt.zero_grad()
    out["loss"].backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    opt.step()
    losses.append(out["lm_loss"].item())
check("train[loss drops >50%]", losses[-1] < 0.5 * losses[0],
      f"{losses[0]:.3f} -> {losses[-1]:.3f}")
check("train[loss finite]", all(torch.isfinite(torch.tensor(losses))))
print(f"       lm_loss {losses[0]:.3f} -> {losses[-1]:.3f}, "
      f"avg_depth {m.get_diagnostics()['avg_depth']:.2f}")

print(f"\n{'=' * 50}")
if FAIL:
    print(f"{PASS} passed, {len(FAIL)} FAILED: {FAIL}")
    sys.exit(1)
print(f"ALL {PASS} CHECKS PASSED")
