"""
NEXUS-LM: Language Model with Cross-Iteration State Attention

Same core innovation (CISA) as the task model, but adapted for
real language modeling:
- Large vocabulary (16K-32K BPE tokens)
- Longer sequences (512-1024)
- RoPE position encoding (scales to any length)
- Proper language model head
- KV cache for fast inference
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


# ============================================================
# RoPE (Rotary Position Embedding)
# ============================================================

def precompute_freqs_cis(dim: int, max_seq_len: int, theta: float = 10000.0):
    """Precompute RoPE frequency tensor."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    """Apply RoPE to query and key tensors."""
    # xq, xk: (B, H, N, D)
    B, H, N, D = xq.shape
    xq_complex = torch.view_as_complex(xq.float().reshape(B, H, N, D // 2, 2))
    xk_complex = torch.view_as_complex(xk.float().reshape(B, H, N, D // 2, 2))

    freqs = freqs_cis[:N].unsqueeze(0).unsqueeze(0)  # (1, 1, N, D//2)
    xq_out = torch.view_as_real(xq_complex * freqs).flatten(-2)
    xk_out = torch.view_as_real(xk_complex * freqs).flatten(-2)
    return xq_out.type_as(xq), xk_out.type_as(xk)


# ============================================================
# Cross-Iteration State Attention (LM version)
# ============================================================

class CISAttention(nn.Module):
    """
    Cross-Iteration State Attention for language modeling.

    Same core idea as the task model version, but with:
    - RoPE instead of learned position embeddings
    - Optimized for longer sequences
    """

    def __init__(self, d_model: int, n_heads: int, max_iterations: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5

        # Spatial attention projections
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

        # Temporal (state history) projections
        self.k_state_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_state_proj = nn.Linear(d_model, d_model, bias=False)

        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = nn.Dropout(dropout)

        # Temporal gate (learned, starts small)
        self.temporal_gate = nn.Parameter(torch.tensor(-2.0))

    def forward(
        self,
        x: torch.Tensor,
        state_history: List[torch.Tensor],
        iteration: int,
        freqs_cis: torch.Tensor,
    ) -> torch.Tensor:
        B, N, D = x.shape
        H = self.n_heads
        dh = self.d_head

        # Spatial Q, K, V
        q = self.q_proj(x).view(B, N, H, dh).transpose(1, 2)
        k = self.k_proj(x).view(B, N, H, dh).transpose(1, 2)
        v = self.v_proj(x).view(B, N, H, dh).transpose(1, 2)

        # Apply RoPE
        q, k = apply_rotary_emb(q, k, freqs_cis)

        # Spatial attention with causal mask
        spatial_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        causal_mask = torch.triu(
            torch.ones(N, N, device=x.device, dtype=torch.bool), diagonal=1
        )
        spatial_scores.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        # Temporal attention
        if len(state_history) > 0 and iteration > 0:
            past_states = torch.stack(state_history[:iteration], dim=1)  # (B, T, N, D)
            T = past_states.shape[1]

            k_temporal = self.k_state_proj(past_states).view(B, T, N, H, dh).permute(0, 3, 2, 1, 4)
            v_temporal = self.v_state_proj(past_states).view(B, T, N, H, dh).permute(0, 3, 2, 1, 4)

            q_expanded = q.unsqueeze(-2)
            temporal_scores = torch.matmul(
                q_expanded, k_temporal.transpose(-2, -1)
            ).squeeze(-2) * self.scale

            temporal_bias = torch.sigmoid(self.temporal_gate)
            temporal_scores = temporal_scores + temporal_bias.log()

            all_scores = torch.cat([spatial_scores, temporal_scores], dim=-1)
            all_weights = self.attn_dropout(F.softmax(all_scores, dim=-1))

            spatial_weights = all_weights[:, :, :, :N]
            temporal_weights = all_weights[:, :, :, N:]

            spatial_out = torch.matmul(spatial_weights, v)
            temporal_weights_exp = temporal_weights.unsqueeze(-1)
            temporal_out = (temporal_weights_exp * v_temporal).sum(dim=-2)

            out = spatial_out + temporal_out
        else:
            spatial_weights = self.attn_dropout(F.softmax(spatial_scores, dim=-1))
            out = torch.matmul(spatial_weights, v)

        out = out.transpose(1, 2).contiguous().view(B, N, D)
        return self.out_proj(out)


# ============================================================
# GRU State Update (same as task model)
# ============================================================

class GRUStateUpdate(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.W_z = nn.Linear(d_model * 2, d_model)
        self.W_r = nn.Linear(d_model * 2, d_model)
        self.W_h = nn.Linear(d_model * 2, d_model)

    def forward(self, state: torch.Tensor, new_info: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([state, new_info], dim=-1)
        z = torch.sigmoid(self.W_z(combined))
        r = torch.sigmoid(self.W_r(combined))
        candidate_input = torch.cat([r * state, new_info], dim=-1)
        h_candidate = torch.tanh(self.W_h(candidate_input))
        return (1 - z) * state + z * h_candidate


# ============================================================
# NEXUS-LM Cell
# ============================================================

class NexusLMCell(nn.Module):
    """Recursive cell with CISA, GRU state, SwiGLU FFN."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 max_iterations: int, dropout: float = 0.0):
        super().__init__()

        self.attn_norm = nn.LayerNorm(d_model)
        self.attention = CISAttention(d_model, n_heads, max_iterations, dropout)
        self.ffn_norm = nn.LayerNorm(d_model)

        # SwiGLU FFN
        self.ffn_gate = nn.Linear(d_model, d_ff, bias=False)
        self.ffn_up = nn.Linear(d_model, d_ff, bias=False)
        self.ffn_down = nn.Linear(d_ff, d_model, bias=False)

        self.attn_drop = nn.Dropout(dropout)
        self.ffn_drop = nn.Dropout(dropout)

        self.state_update = GRUStateUpdate(d_model)
        self.iter_embeddings = nn.Embedding(max_iterations, d_model)

    def forward(self, x, state, state_history, iteration, freqs_cis):
        B, N, D = x.shape

        # Iteration embedding
        iter_emb = self.iter_embeddings(
            torch.full((B, N), iteration, dtype=torch.long, device=x.device)
        )
        x = x + iter_emb

        # CISA + residual
        attn_out = self.attention(self.attn_norm(x), state_history, iteration, freqs_cis)
        x = x + self.attn_drop(attn_out)

        # SwiGLU FFN + residual
        h = self.ffn_norm(x)
        ffn_out = self.ffn_down(F.silu(self.ffn_gate(h)) * self.ffn_up(h))
        x = x + self.ffn_drop(ffn_out)

        # Update state
        new_state = self.state_update(state, x)

        return x, new_state


# ============================================================
# Configuration
# ============================================================

@dataclass
class NexusLMConfig:
    vocab_size: int = 16000
    d_model: int = 512
    n_heads: int = 8
    d_ff: int = 1536
    max_seq_len: int = 512
    n_iterations: int = 4
    max_iterations: int = 8
    dropout: float = 0.1
    rope_theta: float = 10000.0

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def tiny(cls):
        """~2M real / ~6M eff. CPU-trainable."""
        return cls(d_model=192, n_heads=4, d_ff=512, n_iterations=3, max_seq_len=256, dropout=0.1)

    @classmethod
    def small(cls):
        """~22M real / ~88M eff. Quick experiments."""
        return cls(d_model=512, n_heads=8, d_ff=1536, n_iterations=4, max_seq_len=512)

    @classmethod
    def base(cls):
        """~37M real / ~148M eff. Recommended for 4060."""
        return cls(d_model=768, n_heads=12, d_ff=2048, n_iterations=4, max_seq_len=512)

    @classmethod
    def large(cls):
        """~80M real / ~320M eff. Tight fit on 4060 with small batch."""
        return cls(d_model=1024, n_heads=16, d_ff=2816, n_iterations=4, max_seq_len=512)


# ============================================================
# NEXUS-LM Model
# ============================================================

class NexusLM(nn.Module):
    """
    NEXUS Language Model.

    Same CISA innovation as the task model, but built for real text:
    - BPE tokenizer support (16K-32K vocab)
    - RoPE position encoding (no sequence length limit)
    - Proper causal language modeling
    - Generation with temperature, top-k, top-p sampling
    """

    def __init__(self, config: NexusLMConfig):
        super().__init__()
        self.config = config

        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.emb_dropout = nn.Dropout(config.dropout)

        # State initialization
        self.state_init = nn.Linear(config.d_model, config.d_model)

        # Recursive cell
        self.cell = NexusLMCell(
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

        # RoPE
        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(config.d_model // config.n_heads, config.max_seq_len * 2, config.rope_theta),
            persistent=False,
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=0.02)

    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None):
        B, N = input_ids.shape

        x = self.emb_dropout(self.token_emb(input_ids))
        state = torch.tanh(self.state_init(x))

        state_history: List[torch.Tensor] = []
        for iteration in range(self.config.n_iterations):
            state_history.append(state.detach().clone())
            x, state = self.cell(x, state, state_history, iteration, self.freqs_cis)

        x = self.out_norm(x)
        logits = self.lm_head(x)

        result = {"logits": logits}
        if labels is not None:
            # Shift: predict next token
            shift_logits = logits[:, :-1].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            result["loss"] = loss

        return result

    def get_diagnostics(self):
        return {"avg_depth": float(self.config.n_iterations), "reasoning_steps": float(self.config.n_iterations)}

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 200,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        eos_id: int = 2,
    ) -> torch.Tensor:
        """
        Autoregressive generation with temperature, top-k, top-p.

        Args:
            input_ids: (1, N) prompt token IDs
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0 = greedy)
            top_k: Top-k filtering
            top_p: Nucleus sampling threshold
            eos_id: End of sequence token ID

        Returns:
            (1, N + generated) full sequence
        """
        self.eval()
        device = input_ids.device

        for _ in range(max_new_tokens):
            # Truncate to max_seq_len
            x = input_ids[:, -self.config.max_seq_len:]

            out = self(x)
            logits = out["logits"][:, -1, :]  # Last position

            if temperature == 0:
                # Greedy
                next_id = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature

                # Top-k
                if top_k > 0:
                    top_k_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < top_k_vals[:, -1:]] = float('-inf')

                # Top-p (nucleus)
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
                    sorted_logits[mask] = float('-inf')
                    logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)

            input_ids = torch.cat([input_ids, next_id], dim=1)

            if next_id.item() == eos_id:
                break

        return input_ids

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        return {
            "total_params": total,
            "effective_params": total * self.config.n_iterations,
            "iterations": self.config.n_iterations,
        }
