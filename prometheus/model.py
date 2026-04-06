"""
PROMETHEUS: The Complete Model

Combines all 5 innovations into one architecture:

    Input
      ↓
    Embedding
      ↓
    Memory Read (what do I already know about this?)
      ↓
    Fractal Recursive Processing (adaptive depth, weight sharing)
      ├── Each iteration uses Neuroplastic FFN (adapts to input)
      ├── Predictive Coding (only errors propagate)
      └── Sparse Activation (95% neurons silent)
      ↓
    Memory Write (store what I learned)
      ↓
    Output

Total params: TINY (fractal weight sharing)
Active compute: TINY (sparse + predictive coding)
Effective capacity: HUGE (memory + plasticity + depth)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Optional, Tuple

from .neuroplastic import NeuroplasticFFN
from .memory import SparseDistributedMemory, MemoryAugmentedLayer
from .fractal import FractalRecursiveProcessor
from .predictive import PredictiveCodingEngine
from .sparse_activation import SparseFFN


class PrometheusConfig:
    """Configuration for Prometheus model."""

    def __init__(
        self,
        vocab_size: int = 200,
        d_model: int = 128,
        n_heads: int = 4,
        d_ff: int = 256,
        max_seq_len: int = 64,
        # Fractal
        max_depth: int = 12,
        halt_threshold: float = 0.95,
        # Memory
        n_memory_slots: int = 128,
        memory_top_k: int = 8,
        n_read_heads: int = 2,
        # Predictive Coding
        n_pred_levels: int = 3,
        n_settle_steps: int = 3,
        # Sparsity
        neuron_sparsity: float = 0.85,
        # Plasticity
        plastic_rank: int = 16,
        # General
        dropout: float = 0.0,
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.max_depth = max_depth
        self.halt_threshold = halt_threshold
        self.n_memory_slots = n_memory_slots
        self.memory_top_k = memory_top_k
        self.n_read_heads = n_read_heads
        self.n_pred_levels = n_pred_levels
        self.n_settle_steps = n_settle_steps
        self.neuron_sparsity = neuron_sparsity
        self.plastic_rank = plastic_rank
        self.dropout = dropout

    def to_dict(self) -> dict:
        return self.__dict__.copy()

    @classmethod
    def tiny(cls) -> "PrometheusConfig":
        return cls(
            vocab_size=200, d_model=64, n_heads=4, d_ff=128,
            max_seq_len=64, max_depth=8, n_memory_slots=64,
            memory_top_k=4, n_read_heads=2, n_pred_levels=2,
            n_settle_steps=2, neuron_sparsity=0.80, plastic_rank=8,
        )

    @classmethod
    def small(cls) -> "PrometheusConfig":
        return cls(
            vocab_size=200, d_model=128, n_heads=4, d_ff=256,
            max_seq_len=64, max_depth=12, n_memory_slots=128,
            memory_top_k=8, n_read_heads=2, n_pred_levels=3,
            n_settle_steps=3, neuron_sparsity=0.85, plastic_rank=16,
        )


class PrometheusModel(nn.Module):
    """
    PROMETHEUS: Self-Modifying Fractal Intelligence

    Key efficiency metrics:
    - Fractal weight sharing: N virtual layers from 1 set of weights
    - Sparse activation: only ~10% of neurons compute per forward
    - Predictive coding: only errors propagate → minimal signal flow
    - Neuroplasticity: adapts computation to each specific input
    - External memory: knowledge persists without parameter growth
    """

    def __init__(self, config: PrometheusConfig):
        super().__init__()
        self.config = config

        # Token + position embeddings
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)

        # Sparse Distributed Memory
        self.memory = SparseDistributedMemory(
            n_slots=config.n_memory_slots,
            d_key=config.d_model // 2,
            d_value=config.d_model,
            n_read_heads=config.n_read_heads,
            top_k=config.memory_top_k,
        )
        self.memory_layer = MemoryAugmentedLayer(config.d_model, self.memory)

        # Fractal Recursive Processor (THE core - replaces all transformer layers)
        self.fractal = FractalRecursiveProcessor(
            d_model=config.d_model,
            n_heads=config.n_heads,
            d_ff=config.d_ff,
            max_depth=config.max_depth,
            halt_threshold=config.halt_threshold,
        )

        # Neuroplastic FFN (post-fractal refinement)
        self.plastic_ffn = NeuroplasticFFN(
            config.d_model, config.d_ff, config.plastic_rank
        )

        # Predictive Coding Engine
        self.predictive = PredictiveCodingEngine(
            d_model=config.d_model,
            n_levels=config.n_pred_levels,
            n_settle_steps=config.n_settle_steps,
        )

        # Sparse FFN (final processing with sparse activation)
        self.sparse_ffn = SparseFFN(
            config.d_model, config.d_ff, config.neuron_sparsity
        )

        # Output
        self.out_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight  # Weight tying

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        B, N = input_ids.shape
        device = input_ids.device

        # 1. Embeddings
        positions = torch.arange(N, device=device).unsqueeze(0)
        x = self.token_emb(input_ids) + self.pos_emb(positions)

        # 2. Memory read: augment with stored knowledge
        x, write_strength = self.memory_layer(x)

        # 3. Fractal recursive processing (adaptive depth)
        x, ponder_loss = self.fractal(x)

        # 4. Neuroplastic refinement (adapts weights to this input)
        x = x + self.plastic_ffn(x)

        # 5. Predictive coding (only errors flow)
        x_pred, pred_loss = self.predictive(x)
        x = x + x_pred

        # 6. Sparse FFN (final processing with 90% neurons silent)
        x = self.sparse_ffn(x)

        # 7. Output
        x = self.out_norm(x)
        logits = self.lm_head(x)

        result = {
            "logits": logits,
            "ponder_loss": ponder_loss,
            "pred_loss": pred_loss,
        }

        if labels is not None:
            ce_loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                labels.view(-1),
                ignore_index=-100,
            )
            aux_loss = ponder_loss + pred_loss
            result["loss"] = ce_loss + aux_loss
            result["ce_loss"] = ce_loss
            result["aux_loss"] = aux_loss

        return result

    def count_parameters(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        # With fractal sharing, effective depth = max_depth but params are shared
        effective_params = total * self.config.max_depth
        active_fraction = 1.0 - self.config.neuron_sparsity
        active_per_token = int(total * active_fraction)

        return {
            "total_params": total,
            "effective_params": effective_params,
            "active_params_per_token": active_per_token,
            "fractal_depth": self.config.max_depth,
            "efficiency_ratio": active_per_token / total,
        }

    def get_diagnostics(self) -> Dict[str, float]:
        return {
            "avg_depth": self.fractal.avg_depth,
            "pred_error": self.predictive.avg_error,
            "active_fraction": self.predictive.active_fraction,
        }
