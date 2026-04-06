"""
NEXUS Recursive Cell

One cell, shared weights, run K times.
Each iteration refines the representation using:
1. Cross-Iteration State Attention (the novel part)
2. GRU-style state update (persistent memory)
3. SwiGLU FFN (effective nonlinear processing)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

from .attention import CrossIterationStateAttention


class GRUStateUpdate(nn.Module):
    """
    GRU-style update for the persistent per-position state.

    Unlike standard GRU which processes sequences, this takes:
    - current state s_{t-1}
    - new information h_t (from attention + FFN)
    And produces updated state s_t using gates.

    The gates control:
    - reset: how much of the old state to forget
    - update: how much to blend old state with new information

    This prevents catastrophic forgetting across iterations while
    allowing selective updates - crucial for iterative refinement.
    """

    def __init__(self, d_model: int):
        super().__init__()
        # Update gate: how much to keep from old state vs new
        self.W_z = nn.Linear(d_model * 2, d_model)
        # Reset gate: how much of old state to expose to candidate
        self.W_r = nn.Linear(d_model * 2, d_model)
        # Candidate new state
        self.W_h = nn.Linear(d_model * 2, d_model)

    def forward(self, state: torch.Tensor, new_info: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: (B, N, D) previous state
            new_info: (B, N, D) new information from attention + FFN

        Returns:
            (B, N, D) updated state
        """
        combined = torch.cat([state, new_info], dim=-1)

        z = torch.sigmoid(self.W_z(combined))  # Update gate
        r = torch.sigmoid(self.W_r(combined))  # Reset gate

        # Candidate new state (using reset-gated old state)
        candidate_input = torch.cat([r * state, new_info], dim=-1)
        h_candidate = torch.tanh(self.W_h(candidate_input))

        # Blend old and new
        new_state = (1 - z) * state + z * h_candidate
        return new_state


class NexusCell(nn.Module):
    """
    The core recursive processing cell.

    Applied K times with shared weights. Each application:
    1. Adds iteration embedding (tells cell which iteration)
    2. Cross-Iteration State Attention (novel: sees own past states)
    3. SwiGLU FFN
    4. GRU state update

    The cell processes (representation, state) → (new_representation, new_state)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        max_iterations: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model

        # Pre-norm for attention
        self.attn_norm = nn.LayerNorm(d_model)

        # Cross-Iteration State Attention
        self.attention = CrossIterationStateAttention(
            d_model=d_model,
            n_heads=n_heads,
            max_iterations=max_iterations,
            dropout=dropout,
        )

        # Pre-norm for FFN
        self.ffn_norm = nn.LayerNorm(d_model)

        # SwiGLU FFN
        self.ffn_gate = nn.Linear(d_model, d_ff, bias=False)
        self.ffn_up = nn.Linear(d_model, d_ff, bias=False)
        self.ffn_down = nn.Linear(d_ff, d_model, bias=False)

        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.ffn_dropout = nn.Dropout(dropout)

        # GRU state update
        self.state_update = GRUStateUpdate(d_model)

        # Iteration embeddings
        self.iter_embeddings = nn.Embedding(max_iterations, d_model)

    def forward(
        self,
        x: torch.Tensor,
        state: torch.Tensor,
        state_history: List[torch.Tensor],
        iteration: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        One iteration of processing.

        Args:
            x: (B, N, D) current representation
            state: (B, N, D) current persistent state
            state_history: list of past states [(B, N, D), ...]
            iteration: current iteration index

        Returns:
            new_x: (B, N, D) updated representation
            new_state: (B, N, D) updated persistent state
        """
        B, N, D = x.shape

        # Add iteration embedding
        iter_emb = self.iter_embeddings(
            torch.full((B, N), iteration, dtype=torch.long, device=x.device)
        )
        x = x + iter_emb

        # Cross-Iteration State Attention with residual
        attn_out = self.attention(self.attn_norm(x), state_history, iteration)
        x = x + self.attn_dropout(attn_out)

        # SwiGLU FFN with residual
        h = self.ffn_norm(x)
        ffn_out = self.ffn_down(F.silu(self.ffn_gate(h)) * self.ffn_up(h))
        x = x + self.ffn_dropout(ffn_out)

        # Update persistent state
        new_state = self.state_update(state, x)

        return x, new_state
