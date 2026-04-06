"""
NEXUS: Neural EXecution with Unified States

A novel AI architecture built around ONE genuinely new idea:
Cross-Iteration State Attention (CISA).

Each position maintains a persistent "state" that evolves across recursive
iterations. When computing attention at iteration t, each position can attend
to its OWN state history from iterations 0..t-1, in addition to the normal
causal attention over other positions.

This gives each position a "thought history" - it can see how its understanding
evolved, enabling self-correction and iterative refinement.

Why this is novel:
- Universal Transformers share weights but iterations are independent
- Memory Networks use global memory, not per-position state history
- PonderNet has adaptive depth but no iteration memory
- NEXUS: per-position state history accessible via attention = unique

Architecture:
    Input → Embedding + Position Encoding
            ↓
    ┌─── NEXUS Cell (shared weights, run K times) ───┐
    │                                                  │
    │  1. Cross-Iteration State Attention (CISA)       │
    │     Q = current representation                   │
    │     K,V = [causal positions] + [own past states] │
    │                                                  │
    │  2. GRU State Update                             │
    │     Selectively update persistent state           │
    │                                                  │
    │  3. SwiGLU FFN                                   │
    │     Standard but effective                        │
    │                                                  │
    │  4. Iteration Embedding                          │
    │     Tells cell which iteration we're on           │
    │                                                  │
    └──────────────────────────────────────────────────┘
            ↓
    LayerNorm → LM Head (weight-tied)

Design principles:
- NO global pooling (avoids train/test mismatch)
- Strict causal masking (generation works correctly)
- Shared weights across iterations (parameter efficient)
- Per-position state (no cross-position information leaks)
- Simple components that compose cleanly
"""
