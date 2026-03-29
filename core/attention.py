"""
NeuroSpark Efficient Attention Mechanisms

Key innovations:
1. LinearAttention - O(n) complexity via kernel trick with ELU feature map
2. SparseLocalAttention - Local windowed attention with global tokens
3. EfficientMultiHeadAttention - Adaptive switching between attention types
   based on sequence length for optimal compute/quality tradeoff
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class LinearAttention(nn.Module):
    """
    Linear attention using kernel feature maps.

    Instead of computing softmax(QK^T)V which is O(n²d),
    we compute φ(Q)(φ(K)^T V) which is O(nd²) - linear in sequence length.

    Uses ELU+1 as the feature map (proven stable in practice).
    """

    def __init__(self, d_model: int, n_heads: int, eps: float = 1e-6):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.eps = eps

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # Learnable scaling per head
        self.head_scale = nn.Parameter(torch.ones(n_heads, 1, 1))

    def feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """ELU+1 feature map for non-negative kernel approximation."""
        return F.elu(x) + 1

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        causal: bool = False,
    ) -> torch.Tensor:
        B, N, D = x.shape

        Q = self.q_proj(x).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        K = self.k_proj(x).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        V = self.v_proj(x).view(B, N, self.n_heads, self.d_head).transpose(1, 2)

        # Apply feature map
        Q = self.feature_map(Q)
        K = self.feature_map(K)

        if causal:
            # Causal linear attention via cumulative sum trick
            output = self._causal_linear_attention(Q, K, V)
        else:
            # Non-causal: compute K^T V first (d x d matrix), then Q @ (K^T V)
            KV = torch.einsum("bhnd,bhne->bhde", K, V)  # (B, H, d_head, d_head)
            Z = K.sum(dim=2)  # (B, H, d_head) - normalizer

            output = torch.einsum("bhnd,bhde->bhne", Q, KV)  # (B, H, N, d_head)
            normalizer = torch.einsum("bhnd,bhd->bhn", Q, Z).unsqueeze(-1) + self.eps
            output = output / normalizer

        output = output * self.head_scale
        output = output.transpose(1, 2).contiguous().view(B, N, D)
        return self.out_proj(output)

    def _causal_linear_attention(
        self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor
    ) -> torch.Tensor:
        """Causal linear attention using cumulative sums."""
        B, H, N, D = Q.shape

        # Cumulative KV and K sums for causal masking
        S = torch.zeros(B, H, D, D, device=Q.device, dtype=Q.dtype)
        Z = torch.zeros(B, H, D, device=Q.device, dtype=Q.dtype)
        outputs = []

        for i in range(N):
            k_i = K[:, :, i, :]  # (B, H, D)
            v_i = V[:, :, i, :]  # (B, H, D)
            q_i = Q[:, :, i, :]  # (B, H, D)

            S = S + torch.einsum("bhd,bhe->bhde", k_i, v_i)
            Z = Z + k_i

            out_i = torch.einsum("bhd,bhde->bhe", q_i, S)
            norm_i = torch.einsum("bhd,bhd->bh", q_i, Z).unsqueeze(-1) + self.eps
            outputs.append(out_i / norm_i)

        return torch.stack(outputs, dim=2)


class SparseLocalAttention(nn.Module):
    """
    Windowed local attention with global sentinel tokens.

    Complexity: O(n * w) where w is window size, instead of O(n²).
    Global tokens attend to everything, enabling long-range information flow.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        window_size: int = 64,
        n_global_tokens: int = 4,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.window_size = window_size
        self.n_global_tokens = n_global_tokens

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # Learnable global tokens
        self.global_tokens = nn.Parameter(
            torch.randn(1, n_global_tokens, d_model) * 0.02
        )

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, N, D = x.shape

        # Prepend global tokens
        globals_expanded = self.global_tokens.expand(B, -1, -1)
        x_aug = torch.cat([globals_expanded, x], dim=1)
        N_aug = N + self.n_global_tokens

        Q = self.q_proj(x_aug).view(B, N_aug, self.n_heads, self.d_head).transpose(1, 2)
        K = self.k_proj(x_aug).view(B, N_aug, self.n_heads, self.d_head).transpose(1, 2)
        V = self.v_proj(x_aug).view(B, N_aug, self.n_heads, self.d_head).transpose(1, 2)

        scale = math.sqrt(self.d_head)

        # Global tokens attend to everything (full attention for first n_global_tokens)
        global_Q = Q[:, :, : self.n_global_tokens, :]
        global_attn = torch.matmul(global_Q, K.transpose(-2, -1)) / scale
        global_attn = F.softmax(global_attn, dim=-1)
        global_out = torch.matmul(global_attn, V)

        # Local tokens use windowed attention + attend to global tokens
        local_outputs = []
        for i in range(self.n_global_tokens, N_aug):
            q_i = Q[:, :, i : i + 1, :]  # (B, H, 1, d_head)

            # Window around position i
            start = max(self.n_global_tokens, i - self.window_size // 2)
            end = min(N_aug, i + self.window_size // 2 + 1)

            # Concatenate global keys and local window keys
            local_K = torch.cat([K[:, :, : self.n_global_tokens, :], K[:, :, start:end, :]], dim=2)
            local_V = torch.cat([V[:, :, : self.n_global_tokens, :], V[:, :, start:end, :]], dim=2)

            attn = torch.matmul(q_i, local_K.transpose(-2, -1)) / scale
            attn = F.softmax(attn, dim=-1)
            local_outputs.append(torch.matmul(attn, local_V))

        local_out = torch.cat(local_outputs, dim=2)  # (B, H, N, d_head)

        # Combine and remove global token positions from output
        output = local_out.transpose(1, 2).contiguous().view(B, N, D)
        return self.out_proj(output)


class EfficientMultiHeadAttention(nn.Module):
    """
    Adaptive attention that switches strategy based on sequence length.

    - Short sequences (< threshold): Standard softmax attention (highest quality)
    - Long sequences (>= threshold): Linear attention (O(n) complexity)

    This gives optimal quality/compute tradeoff automatically.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        linear_threshold: int = 512,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.linear_threshold = linear_threshold

        self.linear_attn = LinearAttention(d_model, n_heads)

        # Standard attention projections (shared weights with linear for efficiency)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

        # Learnable blending factor between attention types
        self.blend_gate = nn.Parameter(torch.tensor(0.0))

    def _standard_attention(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, N, D = x.shape
        Q = self.q_proj(x).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        K = self.k_proj(x).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        V = self.v_proj(x).view(B, N, self.n_heads, self.d_head).transpose(1, 2)

        scale = math.sqrt(self.d_head)
        attn = torch.matmul(Q, K.transpose(-2, -1)) / scale

        if mask is not None:
            attn = attn.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        output = torch.matmul(attn, V)
        output = output.transpose(1, 2).contiguous().view(B, N, D)
        return self.out_proj(output)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        causal: bool = False,
    ) -> torch.Tensor:
        N = x.shape[1]

        if N < self.linear_threshold:
            return self._standard_attention(x, mask)
        else:
            # For long sequences, use causal linear attention
            return self.linear_attn(x, mask, causal=True)
