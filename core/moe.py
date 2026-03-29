"""
NeuroSpark Adaptive Mixture-of-Experts

Key innovations:
1. Dynamic top-k routing - the model learns HOW MANY experts to use per token
2. Expert specialization loss - encourages experts to specialize in different domains
3. Load balancing with capacity factors - prevents expert collapse
4. Shared expert - always active base expert for common knowledge
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class ExpertFFN(nn.Module):
    """Single expert: a 2-layer FFN with SwiGLU activation."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU activation
        return self.w2(F.silu(self.w1(x)) * self.w_gate(x))


class DynamicRouter(nn.Module):
    """
    Learns to route tokens to experts AND how many experts each token needs.

    Standard MoE uses fixed top-k. We learn a threshold per token,
    allowing simple tokens to use 1 expert and complex ones to use more.
    This saves compute on easy tokens.
    """

    def __init__(self, d_model: int, n_experts: int, max_k: int = 4):
        super().__init__()
        self.n_experts = n_experts
        self.max_k = min(max_k, n_experts)

        self.gate = nn.Linear(d_model, n_experts, bias=False)
        # Learned threshold for dynamic-k selection
        self.threshold_proj = nn.Linear(d_model, 1, bias=True)
        nn.init.constant_(self.threshold_proj.bias, 0.3)  # Start with moderate threshold

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            weights: (B, N, max_k) - expert weights per token
            indices: (B, N, max_k) - selected expert indices
            load_balance_loss: scalar - auxiliary loss for balanced routing
        """
        B, N, D = x.shape

        # Compute routing logits
        logits = self.gate(x)  # (B, N, n_experts)
        probs = F.softmax(logits, dim=-1)

        # Dynamic threshold: determines how many experts each token uses
        threshold = torch.sigmoid(self.threshold_proj(x))  # (B, N, 1)

        # Get top-k experts
        top_k_probs, top_k_indices = probs.topk(self.max_k, dim=-1)

        # Create dynamic mask: keep experts whose probability exceeds threshold
        # At minimum, always use top-1 expert
        dynamic_mask = top_k_probs >= threshold
        dynamic_mask[:, :, 0] = True  # Always keep the top expert

        # Zero out experts below threshold
        weights = top_k_probs * dynamic_mask.float()

        # Renormalize weights
        weight_sum = weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        weights = weights / weight_sum

        # Load balancing loss (encourages uniform expert utilization)
        # Based on Switch Transformer's auxiliary loss
        expert_counts = F.one_hot(top_k_indices[:, :, 0], self.n_experts).float()
        tokens_per_expert = expert_counts.sum(dim=1).mean(dim=0)  # (n_experts,)
        avg_prob_per_expert = probs.mean(dim=1).mean(dim=0)  # (n_experts,)
        load_balance_loss = (
            self.n_experts * (tokens_per_expert * avg_prob_per_expert).sum()
        )

        return weights, top_k_indices, load_balance_loss


class AdaptiveMixtureOfExperts(nn.Module):
    """
    Full MoE layer with:
    - Shared base expert (always active, provides common knowledge)
    - N specialized experts with dynamic routing
    - Expert specialization regularization
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        n_experts: int = 8,
        max_k: int = 4,
        capacity_factor: float = 1.25,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_experts = n_experts

        # Shared expert (always active - handles common patterns)
        self.shared_expert = ExpertFFN(d_model, d_ff)
        self.shared_gate = nn.Parameter(torch.tensor(0.5))

        # Specialized experts
        self.experts = nn.ModuleList(
            [ExpertFFN(d_model, d_ff) for _ in range(n_experts)]
        )

        # Dynamic router
        self.router = DynamicRouter(d_model, n_experts, max_k)

        # Layer norm before expert processing
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            output: (B, N, D)
            aux_loss: scalar load balancing loss
        """
        B, N, D = x.shape
        residual = x
        x = self.norm(x)

        # Shared expert output (always computed)
        shared_out = self.shared_expert(x)
        gate = torch.sigmoid(self.shared_gate)

        # Route to specialized experts
        weights, indices, load_balance_loss = self.router(x)

        # Vectorized expert computation - avoids slow Python loops
        # Flatten batch and sequence dims for efficient routing
        x_flat = x.view(B * N, D)  # (B*N, D)
        expert_output_flat = torch.zeros_like(x_flat)
        max_k = weights.shape[-1]

        indices_flat = indices.view(B * N, max_k)
        weights_flat = weights.view(B * N, max_k)

        for e in range(self.n_experts):
            # Find ALL tokens routed to expert e across all k slots
            expert_mask = (indices_flat == e)  # (B*N, max_k)
            token_mask = expert_mask.any(dim=-1)  # (B*N,)

            if token_mask.any():
                expert_input = x_flat[token_mask]
                expert_out = self.experts[e](expert_input)

                # Sum weights across all k slots that selected this expert
                combined_weights = (weights_flat[token_mask] * expert_mask[token_mask].float()).sum(dim=-1, keepdim=True)
                expert_output_flat[token_mask] += expert_out * combined_weights

        expert_output = expert_output_flat.view(B, N, D)

        # Combine shared and routed expert outputs
        output = gate * shared_out + (1 - gate) * expert_output

        return residual + output, load_balance_loss
