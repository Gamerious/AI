"""
NEXUS Model

The complete architecture:
    Embedding → Recursive Cell × K → Output

Simple. Clean. Novel (via CISA). Efficient (shared weights).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List
from dataclasses import dataclass

from .cell import NexusCell


@dataclass
class NexusConfig:
    """Configuration for NEXUS model."""
    vocab_size: int = 200
    d_model: int = 128
    n_heads: int = 4
    d_ff: int = 256
    max_seq_len: int = 64
    n_iterations: int = 6      # Number of recursive iterations (K)
    max_iterations: int = 8    # Max possible (for embeddings)
    dropout: float = 0.0

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, d: dict) -> "NexusConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def tiny(cls) -> "NexusConfig":
        """~200K params, for quick experiments."""
        return cls(d_model=64, n_heads=4, d_ff=128, n_iterations=4)

    @classmethod
    def small(cls) -> "NexusConfig":
        """~350K params, good balance on CPU."""
        return cls(d_model=128, n_heads=4, d_ff=256, n_iterations=6)

    @classmethod
    def medium(cls) -> "NexusConfig":
        """~1.5M params, for moderate GPU."""
        return cls(d_model=256, n_heads=8, d_ff=512, n_iterations=8, max_seq_len=128)

    @classmethod
    def large(cls) -> "NexusConfig":
        """~6M real / ~48M effective. Fits 4060 (8GB) with batch_size=48."""
        return cls(
            d_model=512, n_heads=8, d_ff=1536,
            max_seq_len=256, n_iterations=8, max_iterations=12,
            dropout=0.1,
        )

    @classmethod
    def xlarge(cls) -> "NexusConfig":
        """~15M real / ~120M effective. Fits 4060 with batch_size=16."""
        return cls(
            d_model=768, n_heads=12, d_ff=2048,
            max_seq_len=256, n_iterations=8, max_iterations=12,
            dropout=0.1,
        )


class NexusModel(nn.Module):
    """
    NEXUS: Neural EXecution with Unified States

    The complete model. One recursive cell with shared weights,
    applied K times. Each iteration refines understanding using
    Cross-Iteration State Attention (CISA).

    Key properties:
    - Weight sharing: K effective layers from 1 set of weights
    - CISA: per-position state history for iterative refinement
    - Strict causality: no information leaks, generation works
    - No global pooling: no train/test distribution mismatch
    - GRU states: persistent memory across iterations
    """

    def __init__(self, config: NexusConfig):
        super().__init__()
        self.config = config

        # Token + position embeddings
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)

        # State initialization: project embedding to initial state
        self.state_init = nn.Linear(config.d_model, config.d_model)

        # THE recursive cell (shared weights across all iterations)
        self.cell = NexusCell(
            d_model=config.d_model,
            n_heads=config.n_heads,
            d_ff=config.d_ff,
            max_iterations=config.max_iterations,
            dropout=config.dropout,
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

        # 2. Initialize persistent state from embedding
        state = torch.tanh(self.state_init(x))

        # 3. Recursive processing with Cross-Iteration State Attention
        state_history: List[torch.Tensor] = []

        for iteration in range(self.config.n_iterations):
            # Store current state in history (for future iterations to attend to)
            state_history.append(state.detach().clone())

            # One iteration of the shared cell
            x, state = self.cell(x, state, state_history, iteration)

        # 4. Output
        x = self.out_norm(x)
        logits = self.lm_head(x)

        result = {"logits": logits}

        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                labels.view(-1),
                ignore_index=-100,
            )
            result["loss"] = loss

        return result

    def get_diagnostics(self) -> Dict[str, float]:
        """Compatible with TaskEvaluator."""
        return {
            "avg_depth": float(self.config.n_iterations),
            "reasoning_steps": float(self.config.n_iterations),
        }

    def count_parameters(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        effective = total * self.config.n_iterations
        return {
            "total_params": total,
            "effective_params": effective,
            "iterations": self.config.n_iterations,
            "efficiency_multiplier": self.config.n_iterations,
        }
