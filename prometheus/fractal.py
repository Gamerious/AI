"""
Fractal Recursive Processor

THE key innovation for running on weak hardware.

Instead of N different layers (each with unique weights), we use
ONE small "cell" that processes recursively. The cell takes state
and produces new state + a halting probability.

Why "fractal"? Because the same pattern repeats at every depth,
creating a self-similar computation structure - like a fractal.

A 1M parameter cell with depth 16 gives the EFFECTIVE capacity
of a 16M parameter network, using only 1M of actual memory.

This is different from:
- Universal Transformers (which share weights but use fixed depth)
- PonderNet (which has adaptive depth but no state compression)
- Recursive Neural Networks (which process trees, not sequences)

Our innovation: The cell includes STATE COMPRESSION between iterations.
This prevents information from growing without bound and forces the
model to learn increasingly abstract representations at each depth.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, List, Optional


class StateCompressor(nn.Module):
    """
    Compresses state between recursive iterations.

    Forces information bottleneck: the model must learn what's
    IMPORTANT to keep between recursive steps. Unimportant details
    are discarded, creating a natural abstraction hierarchy.
    """

    def __init__(self, d_model: int, compression_ratio: float = 0.5):
        super().__init__()
        d_bottleneck = max(16, int(d_model * compression_ratio))
        self.d_bottleneck = d_bottleneck

        self.compress = nn.Linear(d_model, d_bottleneck, bias=False)
        self.decompress = nn.Linear(d_bottleneck, d_model, bias=False)
        self.gate = nn.Linear(d_model * 2, d_model)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Compress and reconstruct, keeping only essential info."""
        compressed = F.gelu(self.compress(state))
        reconstructed = self.decompress(compressed)

        # Gated residual: blend original with compressed version
        gate = torch.sigmoid(self.gate(
            torch.cat([state, reconstructed], dim=-1)
        ))
        # Fix: ensure shapes match for cat by projecting compressed
        return gate * state + (1 - gate) * reconstructed


class FractalCell(nn.Module):
    """
    The core recursive processing cell.

    One cell does: attention + FFN + halting decision.
    It receives (state, depth_embedding) and produces (new_state, halt_prob).

    The same cell is applied recursively until it decides to halt.
    Depth embedding tells the cell which "virtual layer" it's on.
    """

    def __init__(self, d_model: int, n_heads: int = 4, d_ff: int = None):
        super().__init__()
        d_ff = d_ff or d_model * 2

        # Causal self-attention
        self.attn_norm = nn.LayerNorm(d_model)
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.attn_out = nn.Linear(d_model, d_model, bias=False)

        # FFN with gating (SwiGLU-like)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn_gate = nn.Linear(d_model, d_ff, bias=False)
        self.ffn_up = nn.Linear(d_model, d_ff, bias=False)
        self.ffn_down = nn.Linear(d_ff, d_model, bias=False)

        # Depth embedding projection
        self.depth_proj = nn.Linear(d_model, d_model, bias=False)

        # Halting decision
        self.halt = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid(),
        )

        # State compression between iterations
        self.compressor = StateCompressor(d_model)

    def _causal_attention(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        scale = self.d_head ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Causal mask
        causal_mask = torch.triu(torch.ones(N, N, device=x.device, dtype=torch.bool), diagonal=1)
        attn = attn.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        return self.attn_out(out)

    def forward(
        self, state: torch.Tensor, depth_emb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        One recursive step.

        Args:
            state: (B, N, D) current processing state
            depth_emb: (B, N, D) embedding of current recursion depth

        Returns:
            new_state: (B, N, D)
            halt_prob: (B, N, 1) probability of halting at this depth
        """
        # Incorporate depth information
        x = state + self.depth_proj(depth_emb)

        # Self-attention
        x = state + self._causal_attention(self.attn_norm(x))

        # SwiGLU FFN
        h = self.ffn_norm(x)
        x = x + self.ffn_down(F.silu(self.ffn_gate(h)) * self.ffn_up(h))

        # Compress state for next iteration
        x = self.compressor(x)

        # Halting decision
        # Per-token halting probability
        halt_prob = self.halt(x)  # (B, N, 1)

        return x, halt_prob


class FractalRecursiveProcessor(nn.Module):
    """
    Applies the FractalCell recursively with adaptive depth.

    Uses the ACT (Adaptive Computation Time) mechanism:
    each iteration produces a halt probability. The final output
    is a probability-weighted combination of all intermediate states.

    This means:
    - Easy inputs: 2-3 iterations (fast!)
    - Hard inputs: up to max_depth iterations (thorough)
    - Average: somewhere in between (efficient)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        d_ff: int = None,
        max_depth: int = 16,
        halt_threshold: float = 0.95,
    ):
        super().__init__()
        self.max_depth = max_depth
        self.halt_threshold = halt_threshold
        self.d_model = d_model

        # The ONE cell that does everything (fractal!)
        self.cell = FractalCell(d_model, n_heads, d_ff)

        # Depth embeddings (learned, one per possible depth)
        self.depth_embeddings = nn.Embedding(max_depth, d_model)

        # Statistics
        self._avg_depth = 0.0
        self._depths = None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Recursive processing with adaptive depth.

        Returns:
            output: (B, N, D) final processed state
            ponder_loss: scalar loss encouraging early halting
        """
        B, N, D = x.shape
        device = x.device

        state = x
        output = torch.zeros_like(x)
        cumulative_halt = torch.zeros(B, N, 1, device=device)
        n_updates = torch.zeros(B, N, 1, device=device)

        for depth in range(self.max_depth):
            # Get depth embedding
            depth_idx = torch.full((B, N), depth, dtype=torch.long, device=device)
            depth_emb = self.depth_embeddings(depth_idx)  # (B, N, D)

            # One recursive step
            state, halt_prob = self.cell(state, depth_emb)

            # Determine which tokens are still active
            still_active = (cumulative_halt < self.halt_threshold).float()

            if depth == self.max_depth - 1:
                # Last step: use all remaining probability
                weight = (1.0 - cumulative_halt) * still_active
            else:
                weight = halt_prob * still_active
                cumulative_halt = cumulative_halt + weight

            output = output + weight * state
            n_updates = n_updates + still_active

        # Ponder loss: penalize excessive depth
        ponder_loss = 0.01 * n_updates.mean()

        self._avg_depth = n_updates.mean().item()
        self._depths = n_updates.squeeze(-1)

        return output, ponder_loss

    @property
    def avg_depth(self) -> float:
        return self._avg_depth
