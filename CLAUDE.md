# NEXUS-LM Project Context

This file gives a new Claude Code session the full context to continue work on **NEXUS-LM**, a novel language model architecture.

**Active development branch:** `claude/advanced-ai-architecture-VSb4j`
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
Each token position maintains a **persistent state across recursive iterations** and can attend to its own state history via attention. Combined with standard causal attention via joint softmax. Implementation: `nexus/attention.py`.

The cell is applied K times (3-8 iterations depending on config) with shared weights. This multiplies effective parameter count: a 13M real-params model has ~50M effective params.

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

# Old work - DO NOT TOUCH unless asked
core/, modules/         # NeuroSpark (old)
prometheus/             # Prometheus (failed predecessor)
```

---

## Current State (as of last session)

Training was done on CPU in a cloud sandbox with the **tiny** config (2.3M real params):
- 3500 steps total on TinyStories (27K stories, 5.3M tokens)
- Final val PPL: **20.3** (decent but undertrained - only ~0.7 epochs)
- Best checkpoint: `lm_checkpoints/best.pt`
- Generations are grammatical, coherent kindergarten stories

**Why PPL 20 is not impressive:** The user understands this. TinyStories-1M paper achieves PPL ~3-4 with proper training. The model is undertrained, not broken.

**Next step (user's plan):** Train on their 4060 with `base` or `large` config for real results.

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
# RECOMMENDED: base config - 13M params (50M effective), ~6-8h
python train_nexus_lm.py --config base --bs 16

# Quick test: small - 6M params, ~2h
python train_nexus_lm.py --config small --bs 32

# Max quality: large - 23M params (90M effective), ~24h, tight VRAM
python train_nexus_lm.py --config large --bs 4 --grad-accum 4

# Resume after interruption
python train_nexus_lm.py --config base --bs 16 --resume lm_checkpoints/best.pt
```

### Configs (defined in `nexus/lm/model.py`)
- `tiny`: d_model=192, 3 iters, 2.3M real / 6.9M effective. CPU-trainable.
- `small`: d_model=256, 4 iters, 6M real / 22M effective.
- `base`: d_model=384, 5 iters, 13M real / 50M effective. **Sweet spot for 4060.**
- `large`: d_model=512, 6 iters, 23M real / 90M effective.

### After training
```bash
python chat_nexus.py             # Interactive generation
# Commands inside chat: /temp <n>, /topk <n>, /topp <n>, /quit
```

---

## Important Implementation Notes

### Things that were fixed (don't reintroduce)
1. **BPE encode was 1000x too slow** — original applied all merges sequentially. Now uses word-level BPE with merge ranking + caching. See `nexus/lm/tokenizer.py:_encode_word_bytes`.
2. **State gradient explosion** — `state_history` uses `.detach().clone()` to break the gradient graph between iterations. See `nexus/lm/model.py`.
3. **Prometheus OOM** — was caused by oversized eval datasets. Don't load all eval data at once.

### Architecture details
- Token embeddings are **weight-tied** to LM head.
- Labels shift is done **inside** the model (`shift_logits = logits[:, :-1]`, `shift_labels = labels[:, 1:]`). Pass labels = input_ids unshifted.
- RoPE is used for position (in LM version, `precompute_freqs_cis` / `apply_rotary_emb`).
- Special tokens: PAD=0, BOS=1, EOS=2, UNK=3, bytes=4-259, BPE merges=260+.

### Gitignore handles automatically
`data/`, `lm_data/`, `lm_checkpoints/`, `checkpoints/`, `*.pt`, `*.pth`, `*.ckpt`, `wandb/`, `logs/`, `.venv/`.

---

## Git Workflow

- **Branch:** All work on `claude/advanced-ai-architecture-VSb4j`. Don't push to main.
- **Remote:** `gamerious/ai` on GitHub.
- **Commit style:** Look at recent commits with `git log --oneline -5` and match the style (Conventional Commits: `feat:`, `fix:`, `perf:`, `chore:`).

---

## Communication Style

- User speaks German, mixes English technical terms. Answer in German.
- User wants honest assessments, not cheerleading. They asked "ist das model gut für die größe?" and appreciated the honest "undertrained" answer with comparison data.
- Keep answers concise. The user runs commands themselves and reports back.
- Don't suggest creating unnecessary files (docs, READMEs) unless asked.
