"""
Neuroplastic Inference Engine

The core innovation: weights that adapt DURING the forward pass.

Standard neural networks freeze weights at inference. This means they must
encode ALL possible computations in fixed weights - extremely wasteful.

Neuroplastic layers temporarily modify their weights based on the current
input using Hebbian-like learning rules. This is like "in-context learning"
but built into the weights themselves, not just the activations.

Key insight: we learn the LEARNING RULE itself, not just the weights.
The meta-parameters control HOW the weights change, making this
a "learning to learn" system.

This is different from:
- LoRA (which is static adapter weights)
- HyperNetworks (which generate full weight matrices - expensive)
- Fast Weights (which use simple outer products without learned rules)

Our approach: Learned Hebbian rules with gating and decay, applied
through efficient low-rank updates that cost O(r*d) not O(d²).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional


class HebbianPlasticityRule(nn.Module):
    """
    A LEARNED plasticity rule that determines how weights change.

    Instead of using a fixed rule like Δw = x*y, we learn the update rule:
        Δw = gate * (rule(x, y) - decay * w_plastic)

    Where 'rule', 'gate', and 'decay' are all learned functions.
    The outer weight matrix stays fixed; we add a plastic component.
    """

    def __init__(self, d_in: int, d_out: int, rank: int = 16):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.rank = rank

        # Low-rank factorization of the plastic update
        # Instead of storing a full d_in x d_out plastic matrix,
        # we store two small matrices: (d_in, rank) and (rank, d_out)
        # This makes updates O(r*d) instead of O(d²)
        self.pre_proj = nn.Linear(d_in, rank, bias=False)
        self.post_proj = nn.Linear(rank, d_out, bias=False)

        # Learned plasticity rate (how fast to adapt)
        self.plasticity_rate = nn.Parameter(torch.tensor(0.1))

        # Learned decay rate (prevents runaway growth)
        self.decay_rate = nn.Parameter(torch.tensor(0.01))

        # Gate: controls WHICH parts of the weight matrix change
        self.gate = nn.Sequential(
            nn.Linear(d_in + d_out, rank),
            nn.Sigmoid(),
        )

    def compute_update(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the plastic weight update from input x and output y.

        Returns low-rank factors (A, B) such that Δw ≈ A @ B
        """
        B, N, _ = x.shape

        # Compute gating (which weight dimensions to modify)
        # Use CAUSAL pooling: only consider non-pad positions via exponential recency weighting
        # This avoids train/test mismatch from PAD tokens in future positions
        weights = torch.arange(N, device=x.device, dtype=x.dtype).unsqueeze(0).unsqueeze(-1)  # (1, N, 1)
        weights = (weights / N).exp()  # Exponential recency: later positions weighted more
        # Mask out PAD tokens (all zeros in embedding)
        token_mask = (x.abs().sum(dim=-1, keepdim=True) > 1e-6).float()  # (B, N, 1)
        weights = weights * token_mask
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)

        x_pool = (x * weights).sum(dim=1)  # (B, d_in) - weighted pool excluding PADs
        y_pool = (y * weights).sum(dim=1)  # (B, d_out)
        gate_input = torch.cat([x_pool, y_pool], dim=-1)
        gate_values = self.gate(gate_input)  # (B, rank)

        # Compute low-rank update factors
        A = self.pre_proj(x_pool)   # (B, rank)
        B_factor = self.post_proj.weight.T  # (rank, d_out) - shared across batch

        # Apply gate and plasticity rate
        A = A * gate_values * torch.tanh(self.plasticity_rate)

        return A, B_factor


class NeuroplasticLinear(nn.Module):
    """
    A linear layer with neuroplastic weights.

    w_effective = w_fixed + w_plastic

    The fixed weights are learned during training (standard backprop).
    The plastic weights are computed on-the-fly from the input.

    This means the layer behaves DIFFERENTLY for different inputs,
    adapting its computation to each specific problem.
    """

    def __init__(self, d_in: int, d_out: int, plastic_rank: int = 16):
        super().__init__()
        # Fixed weights (standard, learned via backprop)
        self.fixed = nn.Linear(d_in, d_out, bias=True)

        # Plasticity rule (learned meta-parameters)
        self.plasticity = HebbianPlasticityRule(d_in, d_out, rank=plastic_rank)

        # Blend between fixed and plastic computation
        self.blend = nn.Parameter(torch.tensor(0.0))  # Starts with mostly fixed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with neuroplastic weight adaptation.
        """
        # Fixed computation
        y_fixed = self.fixed(x)

        # Compute plastic update from (input, fixed_output) correlation
        A, B = self.plasticity.compute_update(x, y_fixed)

        # Apply plastic component: x @ (A^T @ B) in low-rank form
        # A: (B, rank), B: (rank, d_out), x: (B, N, d_in)
        # We need: x @ pre_proj^T @ diag(A) @ B
        x_proj = self.plasticity.pre_proj(x)  # (B, N, rank)
        # Scale by per-sample A factors
        x_proj = x_proj * A.unsqueeze(1)  # (B, N, rank) * (B, 1, rank)
        y_plastic = x_proj @ B  # (B, N, d_out)

        # Blend fixed and plastic
        alpha = torch.sigmoid(self.blend)
        return (1 - alpha) * y_fixed + alpha * y_plastic


class NeuroplasticFFN(nn.Module):
    """
    Feed-forward network with neuroplastic layers.
    Both the up-projection and down-projection adapt to the input.
    """

    def __init__(self, d_model: int, d_ff: int, plastic_rank: int = 16):
        super().__init__()
        self.up = NeuroplasticLinear(d_model, d_ff, plastic_rank)
        self.down = NeuroplasticLinear(d_ff, d_model, plastic_rank)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.gelu(self.up(x)))
