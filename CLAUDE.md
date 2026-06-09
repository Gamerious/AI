# NEXUS-LM Project Context

This file gives a new Claude Code session the full context to continue work on **NEXUS-LM**, a novel language model architecture.

**Active development branch:** `claude/nice-thompson-95zuzz` (NEXUS-3; older history on `claude/advanced-ai-architecture-VSb4j`)
**User language:** German (mixed German/English OK). The user prefers concise, technical answers.
**User hardware:** Laptop with RTX 4060 (8GB VRAM).

---

## What This Project Is

NEXUS-LM is a from-scratch language model with one core innovation: **CISA (Cross-Iteration State Attention)**. It is the third iteration in this repo, replacing the failed Prometheus and NeuroSpark architectures.

### Why NEXUS exists
- **NeuroSpark** (in `core/`, `modules/`) was too complex - 4 competing innovations interfered with each other.
- **Prometheus** (in `prometheus/`) had a generation gap: 85% train accuracy, 17% generation accuracy. Root cause: mean pooling over PAD tokens + global information leaks.
- **NEXUS** is the working solution: strict causality, no global pooling, ONE focused innovation (CISA).

### Core innovation: CISA (Cross-Iteration State Attention)
Each token position maintains a **persistent state across recursive iterations** and can attend to its own state history via attention (incl. the current state, B1). Implementation: `nexus/lm/model.py` (`CISAttention`), task-model version in `nexus/attention.py`.

The cell is applied K times (3-4 iterations depending on config) with shared weights. Honest framing: weight sharing multiplies **effective depth/compute**, not capacity — "effective params = total × K" is NOT equivalent to a K-layer transformer (see ALBERT/Universal Transformer). The iso-FLOPs baseline comparison is still open.

### NEXUS-3 additions (this branch)
- **B1 `include_current_state`** (default on): temporal path sees the current GRU state s_t; previously the freshest state was invisible for one iteration and the path was dead at iteration 0.
- **S1 `cross_state`** (flag `--cross-state`): Cross-Position State Attention — positions attend causally to OTHER positions' latest states ("read the neighbors' conclusions, not their surface"). The only state path that routes genuinely new information. THE candidate for a real innovation; needs the ablation run.
- **S2 `adaptive_halting`** (flag `--halting`): ACT-style learned per-token iteration depth (UT-style frozen blending + ponder cost, `--ponder-weight`, default 0.01). `get_diagnostics()["avg_depth"]` reports the learned depth.
- **Exact KV cache** for generation (K per-iteration caches; strict causality makes it exact). 2.5x faster even on tiny/CPU, much more at N=512 on GPU.
- **Flash attention** (`F.scaled_dot_product_attention`) on all separate-softmax paths (gate_mode=channel). GPU win; on CPU slightly slower (irrelevant).
- **A1-A4 are now DEFAULTS** (channel gate, NoPE temporal, QK-Norm, stabilize). `--no-qk-norm` etc. for ablations. `temporal_gate_init` stays -2.0.
- State K/V projected **once per state** when appended (was O(K²) recompute).
- Depth-scaled init (1/sqrt(2K)) on residual-out projections; `--grad-checkpoint` for memory-tight configs.
- `NexusLMConfig.from_dict()` maps old checkpoints to their **old** behavior (legacy defaults). Always load configs via `from_dict`, never `NexusLMConfig(**d)`.

---

## Repository Structure

```
nexus/                  # NEW architecture (task-solver version)
  attention.py          # CISA implementation
  cell.py               # Recursive cell with GRU state update
  model.py              # NexusModel + NexusConfig
  lm/                   # LANGUAGE MODEL VERSION
    __init__.py
    tokenizer.py        # BPE tokenizer (zero deps, GPT-2 style)
    model.py            # NexusLM + NexusLMConfig (with RoPE)

train_nexus_lm.py       # MAIN training script (this is what you run)
prepare_lm_data.py      # Downloads TinyStories, trains tokenizer
chat_nexus.py           # Interactive chat with trained model
eval_nexus.py           # Quick eval of a saved model

train_nexus.py          # Task model training (older, not for LM)
train_nexus_gpu.py      # GPU training for task model

training/               # Task-model framework: tasks.py + evaluator.py (model-agnostic)

# REMOVED from main: NeuroSpark (core/, modules/, configs/) and Prometheus
# (prometheus/) were deleted as failed predecessors. They remain on branch
# claude/advanced-ai-architecture-VSb4j if ever needed for reference.
```

---

## Current State (as of last session)

NEXUS-3 architecture work landed (see additions above). All changes are
verified by `test_nexus3.py` — 48 checks: causality for every variant (incl.
cross_state), KV-cache == full recompute, gradient flow, SDPA parity,
grad-checkpoint parity, ACT sanity, legacy-config loading, train smoke test.
**Run this battery after any architecture change.**

Older result (pre-NEXUS-3, CPU sandbox, tiny config): val PPL 20.3 after 3500
steps on TinyStories — undertrained, not broken (TinyStories-1M paper reaches
PPL ~3-4). Old checkpoints load via `from_dict` with their old behavior.

**Next step (user's plan):** Train on the 4060. Priority order:
1. `base` baseline with the new defaults (`_v3` run)
2. The deciding ablation: same run with `--no-current-state --gate-mode logbias ...` vs v3, and `ablation_temporal.py` on the result (does the state path matter at all?)
3. `--cross-state` and `--halting` runs vs the v3 baseline

---

## How to Train (the user's main goal)

### One-time setup
```bash
# CUDA PyTorch for 4060
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install requests numpy

# Verify CUDA
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# Prepare data (5 min, downloads ~22MB TinyStories)
python prepare_lm_data.py --vocab-size 16000
```

### Training commands (recommended for 4060 8GB)
```bash
# RECOMMENDED: base config with NEXUS-3 defaults
python train_nexus_lm.py --config base --bs 8

# The new features (each vs. the plain v3 run = the ablation):
python train_nexus_lm.py --config base --bs 8 --cross-state          # S1
python train_nexus_lm.py --config base --bs 8 --halting              # S2
python train_nexus_lm.py --config base --bs 8 --cross-state --halting

# Memory tight (large config or K=8):
python train_nexus_lm.py --config large --bs 4 --grad-accum 4 --grad-checkpoint

# Resume after interruption (checkpoints live in lm_checkpoints_<runname>/)
python train_nexus_lm.py --config base --bs 8 --resume lm_checkpoints_base_v3/best.pt

# Architecture regression tests (CPU, ~1 min) - run after any model change
python test_nexus3.py
```

### Configs (defined in `nexus/lm/model.py`)
- `tiny`: d_model=192, 3 iters, ~2M real. CPU-trainable.
- `small`: d_model=512, 4 iters, ~22M real.
- `base`: d_model=768, 4 iters, ~37M real. **Recommended for 4060.**
- `large`: d_model=1024, 4 iters, ~80M real. Tight fit, use --grad-checkpoint.

### After training
```bash
python chat_nexus.py             # Interactive generation
# Commands inside chat: /temp <n>, /topk <n>, /topp <n>, /quit
```

---

## Important Implementation Notes

### Things that were fixed (don't reintroduce)
1. **BPE encode was 1000x too slow** — original applied all merges sequentially. Now uses word-level BPE with merge ranking + caching. See `nexus/lm/tokenizer.py:_encode_word_bytes`.
2. **State detach bug** — the original code detached `state_history`, which froze the GRU + state_init at init (zero gradient). The state now stays in the autograd graph (stability handled by A4: QK-Norm, RMSNorm on GRU input, ReZero iter_emb, depth-scaled init). `--detach-state` reproduces the bug for ablations only.
3. **Off-by-one in temporal attention** — the current state s_t was excluded (`[:iteration]` slice), so the freshest state was invisible and the path was dead at iteration 0. Fixed via `include_current_state` (B1, default on).
4. **No KV cache in generate()** — every token recomputed the full window (O(L³·K) per generation). Now: exact per-iteration KV caches. The exactness is regression-tested in `test_nexus3.py`; don't break the cache==full invariant.
5. **Prometheus OOM** — was caused by oversized eval datasets. Don't load all eval data at once.

### Architecture details
- Token embeddings are **weight-tied** to LM head.
- Labels shift is done **inside** the model (`shift_logits = logits[:, :-1]`, `shift_labels = labels[:, 1:]`). Pass labels = input_ids unshifted.
- RoPE is used for position (in LM version, `precompute_freqs_cis` / `apply_rotary_emb`).
- Special tokens: PAD=0, BOS=1, EOS=2, UNK=3, bytes=4-259, BPE merges=260+.

### Gitignore handles automatically
`data/`, `lm_data/`, `lm_checkpoints/`, `checkpoints/`, `*.pt`, `*.pth`, `*.ckpt`, `wandb/`, `logs/`, `.venv/`.

---

## Git Workflow

- **Branch:** All work on `claude/nice-thompson-95zuzz`. Don't push to main.
- **Remote:** `gamerious/ai` on GitHub.
- **Commit style:** Look at recent commits with `git log --oneline -5` and match the style (Conventional Commits: `feat:`, `fix:`, `perf:`, `chore:`).

---

## Communication Style

- User speaks German, mixes English technical terms. Answer in German.
- User wants honest assessments, not cheerleading. They asked "ist das model gut für die größe?" and appreciated the honest "undertrained" answer with comparison data.
- Keep answers concise. The user runs commands themselves and reports back.
- Don't suggest creating unnecessary files (docs, READMEs) unless asked.
