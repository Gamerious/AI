"""
NeuroSpark Configuration Presets

Each preset is optimized for a different use case.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.model import NeuroSparkConfig


# For unit testing and CI
TINY = NeuroSparkConfig(
    vocab_size=1000, d_model=64, n_heads=4, n_layers=4,
    d_ff=128, n_experts=4, max_k=2, max_seq_len=256,
    linear_threshold=64, n_thoughts=4, max_reasoning_steps=3,
)

# For rapid prototyping on a single GPU
SMALL = NeuroSparkConfig(
    vocab_size=32000, d_model=256, n_heads=8, n_layers=8,
    d_ff=512, n_experts=8, max_k=4, max_seq_len=2048,
    n_thoughts=8, max_reasoning_steps=4,
)

# Comparable to GPT-2 small (124M dense → ~85M active)
BASE = NeuroSparkConfig(
    vocab_size=50257, d_model=768, n_heads=12, n_layers=12,
    d_ff=3072, n_experts=16, max_k=4, max_seq_len=4096,
    n_thoughts=16, max_reasoning_steps=6,
)

# Comparable to GPT-2 medium (350M dense → ~150M active)
LARGE = NeuroSparkConfig(
    vocab_size=50257, d_model=1024, n_heads=16, n_layers=24,
    d_ff=4096, n_experts=32, max_k=4, max_seq_len=8192,
    n_thoughts=32, max_reasoning_steps=8,
)

# Maximum efficiency mode (no reasoning, no adaptive depth)
EFFICIENT = NeuroSparkConfig(
    vocab_size=32000, d_model=512, n_heads=8, n_layers=12,
    d_ff=1024, n_experts=16, max_k=2, max_seq_len=4096,
    use_reasoning=False, use_adaptive_depth=False,
)
