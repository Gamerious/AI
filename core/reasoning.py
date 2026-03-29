"""
NeuroSpark Emergent Reasoning Module

Key innovation: Built-in chain-of-thought reasoning as a differentiable module.
Instead of relying on prompt engineering for reasoning, this module:

1. Generates internal "thought tokens" in a latent space
2. Iteratively refines them through a recurrent reasoning loop
3. Uses a confidence estimator to know when to stop thinking
4. Compresses reasoning into a fixed-size representation

This gives the model built-in reasoning ability without explicit prompting,
and the reasoning depth adapts to problem complexity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class ThoughtGenerator(nn.Module):
    """Generates initial latent thought tokens from input."""

    def __init__(self, d_model: int, n_thoughts: int = 8):
        super().__init__()
        self.n_thoughts = n_thoughts
        self.thought_proj = nn.Linear(d_model, d_model * n_thoughts)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, D) input representation
        Returns:
            thoughts: (B, n_thoughts, D) initial thought tokens
        """
        # Pool input to create thought seeds
        pooled = x.mean(dim=1)  # (B, D)
        thoughts = self.thought_proj(pooled)  # (B, D * n_thoughts)
        thoughts = thoughts.view(-1, self.n_thoughts, x.shape[-1])
        return self.norm(thoughts)


class ReasoningStep(nn.Module):
    """Single step of iterative reasoning: cross-attend to input, then self-refine."""

    def __init__(self, d_model: int, n_heads: int = 4):
        super().__init__()
        self.d_head = d_model // n_heads
        self.n_heads = n_heads

        # Cross-attention: thoughts attend to input
        self.cross_q = nn.Linear(d_model, d_model, bias=False)
        self.cross_k = nn.Linear(d_model, d_model, bias=False)
        self.cross_v = nn.Linear(d_model, d_model, bias=False)
        self.cross_out = nn.Linear(d_model, d_model, bias=False)

        # Self-attention among thoughts
        self.self_q = nn.Linear(d_model, d_model, bias=False)
        self.self_k = nn.Linear(d_model, d_model, bias=False)
        self.self_v = nn.Linear(d_model, d_model, bias=False)
        self.self_out = nn.Linear(d_model, d_model, bias=False)

        # FFN for refinement
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def _multihead_attention(
        self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
        q_proj: nn.Linear, k_proj: nn.Linear, v_proj: nn.Linear,
        out_proj: nn.Linear,
    ) -> torch.Tensor:
        B = Q.shape[0]
        NQ, NK = Q.shape[1], K.shape[1]

        q = q_proj(Q).view(B, NQ, self.n_heads, self.d_head).transpose(1, 2)
        k = k_proj(K).view(B, NK, self.n_heads, self.d_head).transpose(1, 2)
        v = v_proj(V).view(B, NK, self.n_heads, self.d_head).transpose(1, 2)

        scale = self.d_head ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, NQ, -1)
        return out_proj(out)

    def forward(
        self, thoughts: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        # Cross-attention to input context
        residual = thoughts
        thoughts = self.norm1(thoughts)
        thoughts = residual + self._multihead_attention(
            thoughts, context, context,
            self.cross_q, self.cross_k, self.cross_v, self.cross_out
        )

        # Self-attention among thoughts
        residual = thoughts
        thoughts = self.norm2(thoughts)
        thoughts = residual + self._multihead_attention(
            thoughts, thoughts, thoughts,
            self.self_q, self.self_k, self.self_v, self.self_out
        )

        # FFN refinement
        residual = thoughts
        thoughts = residual + self.ffn(self.norm3(thoughts))

        return thoughts


class ConfidenceEstimator(nn.Module):
    """Estimates reasoning confidence to decide when to stop thinking."""

    def __init__(self, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, thoughts: torch.Tensor) -> torch.Tensor:
        """Returns confidence score: (B, 1)"""
        pooled = thoughts.mean(dim=1)  # (B, D)
        return self.net(pooled)  # (B, 1)


class EmergentReasoningModule(nn.Module):
    """
    Complete reasoning module with adaptive thinking depth.

    Architecture:
    1. Generate initial thought tokens from input
    2. Iteratively refine thoughts (cross-attend to input + self-attend)
    3. Check confidence after each step
    4. Stop when confident or max steps reached
    5. Compress final thoughts into reasoning output
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        n_thoughts: int = 8,
        max_reasoning_steps: int = 6,
        confidence_threshold: float = 0.9,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_steps = max_reasoning_steps
        self.confidence_threshold = confidence_threshold
        self.n_thoughts = n_thoughts

        self.thought_generator = ThoughtGenerator(d_model, n_thoughts)
        self.reasoning_steps = nn.ModuleList(
            [ReasoningStep(d_model, n_heads) for _ in range(max_reasoning_steps)]
        )
        self.confidence = ConfidenceEstimator(d_model)

        # Compress thoughts back into sequence representation
        self.compress = nn.Sequential(
            nn.Linear(d_model * n_thoughts, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )

        # Gate to blend reasoning output with original
        self.reasoning_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid(),
        )

        self._n_steps_used = 0
        self._confidence_score = 0.0

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, N, D) input representation

        Returns:
            output: (B, N, D) reasoning-enhanced representation
            reasoning_loss: scalar regularization loss
        """
        B, N, D = x.shape

        # Generate initial thoughts
        thoughts = self.thought_generator(x)  # (B, n_thoughts, D)

        # Iterative reasoning
        n_steps = 0
        total_confidence_loss = torch.tensor(0.0, device=x.device)

        for step in self.reasoning_steps:
            thoughts = step(thoughts, x)
            n_steps += 1

            conf = self.confidence(thoughts)  # (B, 1)
            # Encourage high confidence (penalize unnecessary steps)
            total_confidence_loss += (1.0 - conf).mean()

            # During inference, stop early if confident
            if not self.training and conf.mean() > self.confidence_threshold:
                break

        self._n_steps_used = n_steps
        self._confidence_score = conf.mean().item()

        # Compress thoughts to a fixed representation
        thoughts_flat = thoughts.view(B, -1)  # (B, n_thoughts * D)
        reasoning_repr = self.compress(thoughts_flat)  # (B, D)

        # Broadcast reasoning to all positions and gate with original
        reasoning_expanded = reasoning_repr.unsqueeze(1).expand(-1, N, -1)
        gate = self.reasoning_gate(torch.cat([x, reasoning_expanded], dim=-1))
        output = x + gate * reasoning_expanded

        # Regularization: encourage early stopping
        reasoning_loss = 0.01 * total_confidence_loss / self.max_steps

        return output, reasoning_loss

    @property
    def steps_used(self) -> int:
        return self._n_steps_used

    @property
    def confidence_score(self) -> float:
        return self._confidence_score
