"""
Cross-Iteration State Attention (CISA)

THE core innovation of NEXUS.

Standard causal attention at iteration t for position i:
    Attend to positions 0..i at iteration t

CISA at iteration t for position i:
    Attend to positions 0..i at iteration t  (spatial context)
    + attend to own states at iterations 0..t-1  (temporal context)

The "temporal" part gives each position a HISTORY of its own processing.
This enables:
    1. Self-correction: "I thought X before, but now I see Y"
    2. Iterative refinement: "Building on my previous understanding..."
    3. Confidence tracking: "My state keeps changing → I'm uncertain"

Complexity analysis:
    Standard: O(N² · H) per iteration, O(N² · H · T) total
    CISA:     O((N + T)² · H) per iteration... NO!

    Actually, the state history is per-position (not cross-position),
    so it's: O(N² · H + N · T · H) per iteration = O(N²H + NTH)
    For small T (4-8), this is essentially O(N²H) - same as standard!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Optional, Tuple


class CrossIterationStateAttention(nn.Module):
    """
    Multi-head attention with cross-iteration state history.

    At iteration t, for each position i:
    - Q comes from current representation x[i]
    - K, V come from:
        (a) Causal positions at current iteration: x[0..i]
        (b) Own state history: state[i, 0], state[i, 1], ..., state[i, t-1]

    The state history keys/values use SEPARATE projections from spatial K/V,
    allowing the model to learn different attention patterns for "what did
    I think before?" vs "what do my neighbors say?".
    """

    def __init__(self, d_model: int, n_heads: int, max_iterations: int = 8, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.max_iterations = max_iterations
        self.scale = self.d_head ** -0.5

        # Spatial attention: Q, K, V for attending to other positions
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

        # Temporal attention: K, V for attending to own past states
        # Separate projections because temporal context is semantically different
        self.k_state_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_state_proj = nn.Linear(d_model, d_model, bias=False)

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # Attention dropout
        self.attn_dropout = nn.Dropout(dropout)

        # Learned bias: how much to attend to temporal vs spatial
        # Starts small so the model first learns basic spatial attention
        self.temporal_gate = nn.Parameter(torch.tensor(-2.0))

    def forward(
        self,
        x: torch.Tensor,
        state_history: List[torch.Tensor],
        iteration: int,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, N, D) current representation
            state_history: list of (B, N, D) tensors, one per past iteration
            iteration: current iteration index (0-based)

        Returns:
            (B, N, D) attended representation
        """
        B, N, D = x.shape
        H = self.n_heads
        dh = self.d_head

        # === Spatial attention (standard causal) ===
        q = self.q_proj(x).view(B, N, H, dh).transpose(1, 2)   # (B, H, N, dh)
        k = self.k_proj(x).view(B, N, H, dh).transpose(1, 2)   # (B, H, N, dh)
        v = self.v_proj(x).view(B, N, H, dh).transpose(1, 2)   # (B, H, N, dh)

        # Spatial attention scores with causal mask
        spatial_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, H, N, N)
        causal_mask = torch.triu(
            torch.ones(N, N, device=x.device, dtype=torch.bool), diagonal=1
        )
        spatial_scores.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        # === Temporal attention (to own past states) ===
        if len(state_history) > 0 and iteration > 0:
            # Stack past states: (B, T, N, D) where T = number of past iterations
            past_states = torch.stack(state_history[:iteration], dim=1)  # (B, T, N, D)
            T = past_states.shape[1]

            # Project past states to K, V
            k_temporal = self.k_state_proj(past_states)  # (B, T, N, D)
            v_temporal = self.v_state_proj(past_states)  # (B, T, N, D)

            # Reshape: we want per-position temporal attention
            # For position i, temporal K/V are state_history[:, :, i, :]
            # q: (B, H, N, dh), k_temporal needs to be (B, H, N, T, dh)
            k_temporal = k_temporal.view(B, T, N, H, dh).permute(0, 3, 2, 1, 4)  # (B, H, N, T, dh)
            v_temporal = v_temporal.view(B, T, N, H, dh).permute(0, 3, 2, 1, 4)  # (B, H, N, T, dh)

            # Temporal scores: each position attends to its own past states
            # q[:, :, i, :] @ k_temporal[:, :, i, :, :].T → (B, H, N, T)
            q_expanded = q.unsqueeze(-2)  # (B, H, N, 1, dh)
            temporal_scores = torch.matmul(
                q_expanded, k_temporal.transpose(-2, -1)
            ).squeeze(-2) * self.scale  # (B, H, N, T)

            # Apply temporal gate (learned, starts small)
            temporal_bias = torch.sigmoid(self.temporal_gate)
            temporal_scores = temporal_scores + temporal_bias.log()

            # === Combine spatial and temporal via joint softmax ===
            # Concatenate scores: (B, H, N, N + T)
            all_scores = torch.cat([spatial_scores, temporal_scores], dim=-1)
            all_weights = self.attn_dropout(F.softmax(all_scores, dim=-1))

            # Split weights back
            spatial_weights = all_weights[:, :, :, :N]       # (B, H, N, N)
            temporal_weights = all_weights[:, :, :, N:]       # (B, H, N, T)

            # Spatial output
            spatial_out = torch.matmul(spatial_weights, v)     # (B, H, N, dh)

            # Temporal output: per-position weighted sum of past state values
            temporal_weights_exp = temporal_weights.unsqueeze(-1)  # (B, H, N, T, 1)
            temporal_out = (temporal_weights_exp * v_temporal).sum(dim=-2)  # (B, H, N, dh)

            out = spatial_out + temporal_out
        else:
            # First iteration: no temporal context, just spatial
            spatial_weights = self.attn_dropout(F.softmax(spatial_scores, dim=-1))
            out = torch.matmul(spatial_weights, v)  # (B, H, N, dh)

        # Reshape and project output
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        return self.out_proj(out)
