"""
NEXUS-LM: Language Model with Cross-Iteration State Attention

Same core innovation (CISA) as the task model, but adapted for
real language modeling:
- Large vocabulary (16K-32K BPE tokens)
- Longer sequences (512-1024)
- RoPE position encoding (scales to any length)
- Proper language model head
- KV cache for fast incremental generation (exact, per-iteration caches)

NEXUS-3 additions (this file):
- B1  include_current_state: the temporal path sees the CURRENT state s_t
      (previously only s_0..s_{t-1}; the freshest state was invisible and the
      temporal path was dead at iteration 0).
- S1  cross_state: Cross-Position State Attention. Positions attend causally
      to the *states* of other positions (their distilled conclusions), not
      just their surface representations. This is the path that routes NEW
      information, unlike the own-history path.
- S2  adaptive_halting: ACT-style learned per-token iteration depth
      (Universal-Transformer-style frozen blending + ponder cost).
- S3  plan_states: the state channel is trained to predict the FUTURE
      (tokens i+2..i+1+H), turning it into an explicit per-token plan.
      With cross_state on, decoding routes these supervised plans across
      positions - tokens read their predecessors' plans. Related lines
      (MTP/DeepSeek-V3, Belief State Transformer, NextLat, Semformer) all
      supervise futures in the residual stream or auxiliary latents that
      are NOT attended to by other positions; the routed plan channel is
      the novel combination here.
- Per-state K/V projections are computed ONCE when a state is added to the
  history (was: recomputed over the whole history every iteration, O(K^2)).
- F.scaled_dot_product_attention (flash) for all separate-softmax paths.
- Optional per-iteration gradient checkpointing (memory O(1) in K).
- Depth-scaled init on residual-out projections (GPT-2 style, 1/sqrt(2K)).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.checkpoint import checkpoint as _grad_checkpoint
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


def _rope_rotate(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    # x: (B, H, N, D), freqs: (1, 1, N, D//2) complex
    B, H, N, D = x.shape
    xc = torch.view_as_complex(x.float().reshape(B, H, N, D // 2, 2))
    return torch.view_as_real(xc * freqs).flatten(-2).type_as(x)


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor,
                     offset: int = 0):
    """Apply RoPE to query and key tensors. `offset` = absolute position of
    the first token in the chunk (needed for incremental decoding)."""
    N = xq.shape[2]
    freqs = freqs_cis[offset:offset + N].unsqueeze(0).unsqueeze(0)
    return _rope_rotate(xq, freqs), _rope_rotate(xk, freqs)


def apply_rotary_single(x: torch.Tensor, freqs_cis: torch.Tensor, offset: int = 0):
    """RoPE for a single tensor (used for cross-state keys)."""
    N = x.shape[2]
    freqs = freqs_cis[offset:offset + N].unsqueeze(0).unsqueeze(0)
    return _rope_rotate(x, freqs)


class RMSNorm(nn.Module):
    """RMSNorm over the last dim (version-proof, learnable gain)."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        dt = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight).to(dt)


# ============================================================
# KV cache for incremental decoding
# ============================================================

class NexusKVCache:
    """Per-iteration KV cache.

    NEXUS runs the same cell K times, so a token's spatial K/V differ per
    iteration -> we keep K separate caches. Strict causality makes this
    exact: earlier positions never change when new tokens are appended.
    """

    def __init__(self, n_iterations: int):
        self.k_spatial: List[Optional[torch.Tensor]] = [None] * n_iterations
        self.v_spatial: List[Optional[torch.Tensor]] = [None] * n_iterations
        self.k_xstate: List[Optional[torch.Tensor]] = [None] * n_iterations
        self.v_xstate: List[Optional[torch.Tensor]] = [None] * n_iterations

    @property
    def seq_len(self) -> int:
        k0 = self.k_spatial[0]
        return 0 if k0 is None else k0.shape[2]

    @staticmethod
    def _append(buf: List[Optional[torch.Tensor]], t: int, new: torch.Tensor) -> torch.Tensor:
        buf[t] = new if buf[t] is None else torch.cat([buf[t], new], dim=2)
        return buf[t]

    def update_spatial(self, t: int, k: torch.Tensor, v: torch.Tensor):
        return self._append(self.k_spatial, t, k), self._append(self.v_spatial, t, v)

    def update_xstate(self, t: int, k: torch.Tensor, v: torch.Tensor):
        return self._append(self.k_xstate, t, k), self._append(self.v_xstate, t, v)


# ============================================================
# Cross-Iteration State Attention (LM version)
# ============================================================

class CISAttention(nn.Module):
    """
    Cross-Iteration State Attention for language modeling.

    Three K/V sources, one query:
      spatial   - causal attention over the current representations (RoPE)
      temporal  - each position over its OWN state history (no positions ->
                  NoPE query by default)
      xstate    - (S1, optional) causal attention over the latest STATES of
                  all positions: "read your neighbors' conclusions, not just
                  their surface". RoPE'd (positions matter here).

    gate_mode "channel": separate softmax per path + channel-wise sigmoid
    gates (real gradient signal, flash-attention compatible).
    gate_mode "logbias": original joint softmax with log-sigmoid biases.
    """

    def __init__(self, d_model: int, n_heads: int, max_iterations: int, dropout: float = 0.0,
                 temporal_gate_init: float = -2.0, gate_mode: str = "logbias",
                 nope_temporal: bool = False, qk_norm: bool = False,
                 include_current_state: bool = True, cross_state: bool = False,
                 use_sdpa: bool = True):
        super().__init__()
        assert d_model % n_heads == 0
        self.gate_mode = gate_mode
        self.nope_temporal = nope_temporal
        self.qk_norm = qk_norm
        self.include_current_state = include_current_state
        self.cross_state = cross_state
        self.use_sdpa = use_sdpa
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5

        # Spatial attention projections
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

        # Temporal (own state history) projections
        self.k_state_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_state_proj = nn.Linear(d_model, d_model, bias=False)

        # S1: cross-position state projections ("neighbor conclusions" are
        # semantically different from "my own history" -> separate weights)
        if cross_state:
            self.k_xstate_proj = nn.Linear(d_model, d_model, bias=False)
            self.v_xstate_proj = nn.Linear(d_model, d_model, bias=False)

        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = nn.Dropout(dropout)

        # Gates (learned, start small). Channel-wise (D,) in "channel" mode,
        # scalar in the original "logbias" mode.
        def _make_gate():
            if gate_mode == "channel":
                return nn.Parameter(torch.full((d_model,), float(temporal_gate_init)))
            return nn.Parameter(torch.tensor(float(temporal_gate_init)))

        self.temporal_gate = _make_gate()
        if cross_state:
            self.xstate_gate = _make_gate()

        # A3: QK-Norm (per-head RMSNorm on queries/keys before the dot-product).
        if qk_norm:
            self.q_norm = RMSNorm(self.d_head)
            self.k_norm = RMSNorm(self.d_head)
            self.k_state_norm = RMSNorm(self.d_head)
            if cross_state:
                self.k_xstate_norm = RMSNorm(self.d_head)

        # Runtime switch: force spatial-only (used for the 0-cost ablation test
        # "does the cross-iteration state path contribute anything at all?").
        self.disable_temporal = False

    # ---- state projections, computed ONCE per state (was O(K^2)) ----

    def project_state(self, state: torch.Tensor, freqs_cis: torch.Tensor,
                      pos_offset: int = 0):
        """Project a state to per-head K/V for the temporal path (and, if
        enabled, the cross-state path). Called once when the state is added
        to the history; with shared weights the result never changes."""
        B, N, D = state.shape
        H, dh = self.n_heads, self.d_head

        k_t = self.k_state_proj(state).view(B, N, H, dh).transpose(1, 2)
        v_t = self.v_state_proj(state).view(B, N, H, dh).transpose(1, 2)
        if self.qk_norm:
            k_t = self.k_state_norm(k_t)

        if not self.cross_state:
            return k_t, v_t, None, None

        kx = self.k_xstate_proj(state).view(B, N, H, dh).transpose(1, 2)
        if self.qk_norm:
            kx = self.k_xstate_norm(kx)
        kx = apply_rotary_single(kx, freqs_cis, pos_offset)  # positions matter here
        vx = self.v_xstate_proj(state).view(B, N, H, dh).transpose(1, 2)
        return k_t, v_t, kx, vx

    # ---- helpers ----

    @staticmethod
    def _blocked_mask(N: int, L: int, device) -> Optional[torch.Tensor]:
        """Bool mask (N, L), True = blocked. None if no masking needed
        (single-query decode attends to everything cached)."""
        if N == 1:
            return None
        blocked = torch.triu(torch.ones(N, N, device=device, dtype=torch.bool), diagonal=1)
        past = L - N
        if past > 0:
            blocked = torch.cat(
                [torch.zeros(N, past, device=device, dtype=torch.bool), blocked], dim=1
            )
        return blocked

    def _sdpa(self, q, k, v, blocked, dropout_p):
        if blocked is None:
            return F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
        if k.shape[2] == q.shape[2]:
            return F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=dropout_p)
        return F.scaled_dot_product_attention(q, k, v, attn_mask=~blocked, dropout_p=dropout_p)

    def _manual_attn(self, q, k, v, blocked):
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if blocked is not None:
            scores = scores.masked_fill(blocked, float('-inf'))
        weights = self.attn_dropout(F.softmax(scores, dim=-1))
        return torch.matmul(weights, v)

    # ---- forward ----

    def forward(
        self,
        x: torch.Tensor,
        temporal_kv: Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]],
        xstate_kv: Optional[Tuple[torch.Tensor, torch.Tensor]],
        iteration: int,
        freqs_cis: torch.Tensor,
        kv_cache: Optional[NexusKVCache] = None,
        pos_offset: int = 0,
    ) -> torch.Tensor:
        B, N, D = x.shape
        H, dh = self.n_heads, self.d_head

        # Spatial Q, K, V
        q = self.q_proj(x).view(B, N, H, dh).transpose(1, 2)
        k = self.k_proj(x).view(B, N, H, dh).transpose(1, 2)
        v = self.v_proj(x).view(B, N, H, dh).transpose(1, 2)

        # A3: QK-Norm before RoPE / dot-products
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # A2: keep a pre-RoPE query for the temporal (iteration-axis) path
        q_nope = q
        q, k = apply_rotary_emb(q, k, freqs_cis, offset=pos_offset)

        if kv_cache is not None:
            k, v = kv_cache.update_spatial(iteration, k, v)
        L = k.shape[2]

        # Temporal entries (pre-projected, one per stored state)
        ks_t, vs_t = temporal_kv
        if not self.include_current_state and len(ks_t) > 0:
            # Legacy behavior: the current state s_t is invisible to attention.
            ks_t, vs_t = ks_t[:-1], vs_t[:-1]
        if self.disable_temporal:
            ks_t, vs_t = (), ()
        T = len(ks_t)

        has_temporal = T > 0
        has_xstate = xstate_kv is not None and not self.disable_temporal

        blocked = self._blocked_mask(N, L, x.device)
        dropout_p = self.attn_dropout.p if self.training else 0.0

        joint = self.gate_mode != "channel" and (has_temporal or has_xstate)

        if not joint:
            # --- separate softmaxes (channel mode, or no state paths active) ---
            if self.use_sdpa:
                out = self._sdpa(q, k, v, blocked, dropout_p)
            else:
                out = self._manual_attn(q, k, v, blocked)
            out = out.transpose(1, 2).reshape(B, N, D)

            if has_temporal:
                k_temporal = torch.stack(ks_t, dim=3)  # (B, H, N, T, dh)
                v_temporal = torch.stack(vs_t, dim=3)
                q_t = (q_nope if self.nope_temporal else q).unsqueeze(-2)
                t_scores = torch.matmul(
                    q_t, k_temporal.transpose(-2, -1)
                ).squeeze(-2) * self.scale  # (B, H, N, T)
                t_weights = self.attn_dropout(F.softmax(t_scores, dim=-1))
                t_out = (t_weights.unsqueeze(-1) * v_temporal).sum(dim=-2)
                t_out = t_out.transpose(1, 2).reshape(B, N, D)
                out = out + torch.sigmoid(self.temporal_gate) * t_out

            if has_xstate:
                kx, vx = xstate_kv
                if self.use_sdpa:
                    x_out = self._sdpa(q, kx, vx, blocked, dropout_p)
                else:
                    x_out = self._manual_attn(q, kx, vx, blocked)
                x_out = x_out.transpose(1, 2).reshape(B, N, D)
                out = out + torch.sigmoid(self.xstate_gate) * x_out
        else:
            # --- original logbias: joint softmax across all active paths ---
            spatial_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            if blocked is not None:
                spatial_scores = spatial_scores.masked_fill(blocked, float('-inf'))
            pieces = [spatial_scores]

            if has_xstate:
                kx, vx = xstate_kv
                x_scores = torch.matmul(q, kx.transpose(-2, -1)) * self.scale
                if blocked is not None:
                    x_scores = x_scores.masked_fill(blocked, float('-inf'))
                pieces.append(x_scores + F.logsigmoid(self.xstate_gate))

            if has_temporal:
                k_temporal = torch.stack(ks_t, dim=3)
                v_temporal = torch.stack(vs_t, dim=3)
                q_t = (q_nope if self.nope_temporal else q).unsqueeze(-2)
                t_scores = torch.matmul(
                    q_t, k_temporal.transpose(-2, -1)
                ).squeeze(-2) * self.scale
                pieces.append(t_scores + F.logsigmoid(self.temporal_gate))

            all_weights = self.attn_dropout(F.softmax(torch.cat(pieces, dim=-1), dim=-1))

            idx = L
            out = torch.matmul(all_weights[..., :L], v)
            if has_xstate:
                Lx = xstate_kv[0].shape[2]
                out = out + torch.matmul(all_weights[..., idx:idx + Lx], vx)
                idx += Lx
            if has_temporal:
                t_weights = all_weights[..., idx:]
                out = out + (t_weights.unsqueeze(-1) * v_temporal).sum(dim=-2)
            out = out.transpose(1, 2).reshape(B, N, D)

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
                 max_iterations: int, dropout: float = 0.0,
                 temporal_gate_init: float = -2.0, gate_mode: str = "logbias",
                 nope_temporal: bool = False, qk_norm: bool = False,
                 stabilize: bool = False, include_current_state: bool = True,
                 cross_state: bool = False, use_sdpa: bool = True):
        super().__init__()
        self.stabilize = stabilize

        self.attn_norm = nn.LayerNorm(d_model)
        self.attention = CISAttention(d_model, n_heads, max_iterations, dropout,
                                      temporal_gate_init=temporal_gate_init,
                                      gate_mode=gate_mode, nope_temporal=nope_temporal,
                                      qk_norm=qk_norm,
                                      include_current_state=include_current_state,
                                      cross_state=cross_state, use_sdpa=use_sdpa)
        self.ffn_norm = nn.LayerNorm(d_model)

        # SwiGLU FFN
        self.ffn_gate = nn.Linear(d_model, d_ff, bias=False)
        self.ffn_up = nn.Linear(d_model, d_ff, bias=False)
        self.ffn_down = nn.Linear(d_ff, d_model, bias=False)

        self.attn_drop = nn.Dropout(dropout)
        self.ffn_drop = nn.Dropout(dropout)

        self.state_update = GRUStateUpdate(d_model)
        self.iter_embeddings = nn.Embedding(max_iterations, d_model)

        # A4: ReZero-scaled iteration embedding + RMSNorm on the GRU input.
        if stabilize:
            self.iter_scale = nn.Parameter(torch.zeros(1))
            self.state_norm = RMSNorm(d_model)

    def forward(self, x, state, temporal_kv, xstate_kv, iteration, freqs_cis,
                kv_cache=None, pos_offset: int = 0):
        B, N, D = x.shape

        # Iteration embedding (A4: ReZero-scaled so it doesn't inflate the stream)
        iter_emb = self.iter_embeddings(
            torch.full((B, N), iteration, dtype=torch.long, device=x.device)
        )
        if self.stabilize:
            x = x + torch.tanh(self.iter_scale) * iter_emb
        else:
            x = x + iter_emb

        # CISA + residual
        attn_out = self.attention(self.attn_norm(x), temporal_kv, xstate_kv,
                                  iteration, freqs_cis, kv_cache, pos_offset)
        x = x + self.attn_drop(attn_out)

        # SwiGLU FFN + residual
        h = self.ffn_norm(x)
        ffn_out = self.ffn_down(F.silu(self.ffn_gate(h)) * self.ffn_up(h))
        x = x + self.ffn_drop(ffn_out)

        # Update state (A4: normalize the GRU input so its scale doesn't drift)
        gru_in = self.state_norm(x) if self.stabilize else x
        new_state = self.state_update(state, gru_in)

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
    # Initial value of the temporal_gate logit. sigmoid(-2.0)=0.12 (original,
    # heavily throttles the cross-iteration/state path). Higher = state path
    # contributes more from the start. -1.0=0.27, 0.0=0.50.
    temporal_gate_init: float = -2.0
    # A1: how spatial & temporal attention are mixed.
    #  "logbias" = original (gate as log-bias in a shared softmax -> gate gets ~0 gradient).
    #  "channel" = separate softmaxes + multiplicative channel-wise gate (default;
    #              also unlocks flash attention for the spatial path).
    gate_mode: str = "channel"
    # A2: the temporal path uses the pre-RoPE (NoPE) query (the iteration
    # axis has no positional meaning, so RoPE there is just noise). Default on.
    nope_temporal: bool = True
    # A3: RMSNorm on q/k (and state keys) before the dot-product -> balances
    # spatial/temporal key scales, allows higher LR. Default on.
    qk_norm: bool = True
    # A4: training-stability bundle: state_init without tanh, RMSNorm on the GRU
    # input, and a ReZero-scaled iteration embedding (iter_scale zero-init). Default on.
    stabilize: bool = True
    # Deep supervision: compute the LM loss after EVERY iteration (ascending
    # weights), so each iteration gets an O(1) gradient signal. Train-time only.
    deep_supervision: bool = False
    # If True, reproduce the (buggy) original behavior: state history is
    # detached, so the GRU state-update + state_init get NO gradient and stay
    # frozen at init. Default False = correct CISA (BPTT flows through the
    # cross-iteration state, GRU actually learns).
    detach_state_history: bool = False
    # B1: temporal attention also sees the CURRENT state s_t (the GRU output
    # summarizing everything so far). Old behavior (False) hid the freshest
    # state for one full iteration and left the temporal path dead at iter 0.
    include_current_state: bool = True
    # S1: Cross-Position State Attention. Positions attend causally to the
    # latest STATES of other positions (their conclusions, not their surface).
    # The only path that routes genuinely new information across positions.
    cross_state: bool = False
    # S2: ACT-style adaptive halting: learned per-token iteration depth.
    # Easy tokens exit early, hard tokens use all n_iterations.
    adaptive_halting: bool = False
    ponder_weight: float = 0.01     # weight of the ACT ponder cost in the loss
    halt_bias_init: float = -1.0    # initial halt-head bias (sigmoid(-1)=0.27)
    # S3: plan states. The final state of position i is trained to predict
    # tokens i+2 .. i+1+plan_horizon (i+1 is the LM loss's job). This turns
    # the state channel into an explicit PLAN of the future instead of a
    # recap of the past - and cross_state attention then routes plans across
    # positions at decode time ("read your predecessors' plans").
    # Train-time only; zero inference cost.
    plan_states: bool = False
    plan_horizon: int = 4           # how many tokens beyond next to predict
    plan_weight: float = 0.1        # weight of the plan loss
    # Use F.scaled_dot_product_attention (flash) where the math allows it
    # (all separate-softmax paths). Numerically equivalent; big speed/memory win.
    use_sdpa: bool = True
    # Recompute each iteration in backward instead of storing activations
    # (memory O(1) in K, ~1.3-2x slower step). For large configs / K=8 on 8GB.
    grad_checkpoint: bool = False

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, d):
        kwargs = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        # Checkpoints saved before these fields existed were trained with the
        # OLD behavior -> restore that, not today's defaults.
        legacy_defaults = {
            "gate_mode": "logbias",
            "nope_temporal": False,
            "qk_norm": False,
            "stabilize": False,
            "include_current_state": False,
            "cross_state": False,
            "adaptive_halting": False,
        }
        for k, v in legacy_defaults.items():
            if k not in d:
                kwargs[k] = v
        return cls(**kwargs)

    @classmethod
    def tiny(cls):
        """~2M real params. CPU-trainable."""
        return cls(d_model=192, n_heads=4, d_ff=512, n_iterations=3, max_seq_len=256, dropout=0.1)

    @classmethod
    def small(cls):
        """~22M real params. Quick experiments."""
        return cls(d_model=512, n_heads=8, d_ff=1536, n_iterations=4, max_seq_len=512)

    @classmethod
    def base(cls):
        """~37M real params. Recommended for 4060."""
        return cls(d_model=768, n_heads=12, d_ff=2048, n_iterations=4, max_seq_len=512)

    @classmethod
    def large(cls):
        """~80M real params. Tight fit on 4060 with small batch."""
        return cls(d_model=1024, n_heads=16, d_ff=2816, n_iterations=4, max_seq_len=512)


# ============================================================
# NEXUS-LM Model
# ============================================================

_HALT_EPS = 0.01  # ACT halting threshold (halt once cumulative prob > 1 - eps)


class NexusLM(nn.Module):
    """
    NEXUS Language Model.

    Same CISA innovation as the task model, but built for real text:
    - BPE tokenizer support (16K-32K vocab)
    - RoPE position encoding (no sequence length limit)
    - Proper causal language modeling
    - Exact KV-cached generation with temperature, top-k, top-p sampling
    """

    def __init__(self, config: NexusLMConfig):
        super().__init__()
        assert config.n_iterations <= config.max_iterations
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
            temporal_gate_init=config.temporal_gate_init,
            gate_mode=config.gate_mode,
            nope_temporal=config.nope_temporal,
            qk_norm=config.qk_norm,
            stabilize=config.stabilize,
            include_current_state=config.include_current_state,
            cross_state=config.cross_state,
            use_sdpa=config.use_sdpa,
        )

        # S2: halting head decides per position whether to keep iterating
        if config.adaptive_halting:
            self.halt_head = nn.Linear(config.d_model * 2, 1)

        # S3: plan heads - one projection per future offset, output through
        # the tied LM head (cheap). Supervises the STATE channel, not x.
        if config.plan_states:
            self.plan_norm = RMSNorm(config.d_model)
            self.plan_projs = nn.ModuleList(
                nn.Linear(config.d_model, config.d_model, bias=False)
                for _ in range(config.plan_horizon)
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

        # Depth-scaled init (GPT-2 style): the shared cell contributes
        # 2*n_iterations residual additions -> scale the residual-out
        # projections by 1/sqrt(2K) so the stream doesn't blow up at init.
        with torch.no_grad():
            resid_scale = (2 * config.n_iterations) ** -0.5
            self.cell.attention.out_proj.weight.mul_(resid_scale)
            self.cell.ffn_down.weight.mul_(resid_scale)
            if config.adaptive_halting:
                self.halt_head.bias.fill_(config.halt_bias_init)

        self._last_avg_depth = float(config.n_iterations)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=0.02)

    def _lm_loss(self, logits, labels):
        # Shift: predict next token
        shift_logits = logits[:, :-1].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        return F.cross_entropy(
            shift_logits.view(-1, self.config.vocab_size),
            shift_labels.view(-1),
            ignore_index=-100,
        )

    def _plan_loss(self, state, labels) -> Optional[torch.Tensor]:
        """S3: the final state of position i must predict tokens i+2..i+1+H.

        Training samples ONE horizon per step (unbiased estimate of the mean,
        constant memory: a single extra (B,N,V) logits tensor). Eval averages
        all horizons (under no_grad, so memory is freed per horizon).
        """
        N = labels.shape[1]
        valid = [h for h in range(1, self.config.plan_horizon + 1) if N > 1 + h]
        if not valid:
            return None
        if self.training:
            valid = [valid[int(torch.randint(len(valid), (1,)))]]

        s = self.plan_norm(state)
        losses = []
        for h in valid:
            off = 1 + h
            plan_logits = self.lm_head(self.plan_projs[h - 1](s[:, :-off]))
            losses.append(F.cross_entropy(
                plan_logits.reshape(-1, self.config.vocab_size),
                labels[:, off:].reshape(-1),
                ignore_index=-100,
            ))
        return torch.stack(losses).mean()

    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None,
                kv_cache: Optional[NexusKVCache] = None, pos_offset: int = 0):
        """Forward pass. With `kv_cache` set this is an incremental
        (prefill/decode) step: K/V get appended to the cache and `pos_offset`
        is the absolute position of the first token in `input_ids`."""
        cfg = self.config
        B, N = input_ids.shape

        x = self.emb_dropout(self.token_emb(input_ids))
        # A4: state_init without the saturating tanh
        state = self.state_init(x) if cfg.stabilize else torch.tanh(self.state_init(x))

        halting = cfg.adaptive_halting
        if halting:
            cum_halt = x.new_zeros(B, N)
            n_updates = x.new_zeros(B, N)
            remainders = x.new_zeros(B, N)

        deep = cfg.deep_supervision and self.training and labels is not None
        if deep:
            ds_weights = torch.linspace(0.5, 1.0, cfg.n_iterations, device=x.device)
        ds_loss, ds_wsum, logits = 0.0, 0.0, None

        use_ckpt = (cfg.grad_checkpoint and self.training
                    and torch.is_grad_enabled() and kv_cache is None)

        ks_list: List[torch.Tensor] = []
        vs_list: List[torch.Tensor] = []

        for iteration in range(cfg.n_iterations):
            # Project the state entering this iteration ONCE; later iterations
            # reuse the projection (shared weights -> identical result).
            s_in = state.detach() if cfg.detach_state_history else state
            k_t, v_t, kx, vx = self.cell.attention.project_state(
                s_in, self.freqs_cis, pos_offset
            )
            ks_list.append(k_t)
            vs_list.append(v_t)

            xstate_kv = None
            if cfg.cross_state:
                if kv_cache is not None:
                    kx, vx = kv_cache.update_xstate(iteration, kx, vx)
                xstate_kv = (kx, vx)

            # Immutable snapshot: required for correct recomputation under
            # gradient checkpointing (the lists keep growing afterwards).
            temporal_kv = (tuple(ks_list), tuple(vs_list))

            if halting:
                # ACT bookkeeping (t2t Universal Transformer style), computed
                # BEFORE the transformation from the pre-update [x, state].
                p = torch.sigmoid(
                    self.halt_head(torch.cat([x, state], dim=-1))
                ).squeeze(-1)  # (B, N)
                still = (cum_halt < 1.0 - _HALT_EPS).to(x.dtype)
                if iteration == cfg.n_iterations - 1:
                    new_halted = still
                    still_next = torch.zeros_like(still)
                else:
                    crossing = (cum_halt + p * still > 1.0 - _HALT_EPS).to(x.dtype)
                    new_halted = crossing * still
                    still_next = (1.0 - crossing) * still
                cum_halt = cum_halt + p * still_next
                r = (1.0 - cum_halt) * new_halted
                cum_halt = cum_halt + r
                remainders = remainders + r
                n_updates = n_updates + still
                update_w = (p * still_next + r).unsqueeze(-1)  # (B, N, 1)

            if use_ckpt:
                x_run, state_run = _grad_checkpoint(
                    self.cell, x, state, temporal_kv, xstate_kv, iteration,
                    self.freqs_cis, None, pos_offset, use_reentrant=False,
                )
            else:
                x_run, state_run = self.cell(
                    x, state, temporal_kv, xstate_kv, iteration,
                    self.freqs_cis, kv_cache, pos_offset,
                )

            if halting:
                # Frozen blending: halted positions (w=0) keep x/state as-is.
                x = update_w * x_run + (1.0 - update_w) * x
                state = update_w * state_run + (1.0 - update_w) * state
            else:
                x, state = x_run, state_run

            if deep:
                # Deep supervision: LM loss after every iteration (ascending weight)
                logits = self.lm_head(self.out_norm(x))
                w = ds_weights[iteration]
                ds_loss = ds_loss + w * self._lm_loss(logits, labels)
                ds_wsum = ds_wsum + w

            # Everyone halted -> remaining iterations are no-ops for x/state.
            # (Not with a kv_cache: future tokens still need per-iteration K/V.)
            if (halting and not deep and kv_cache is None
                    and bool((cum_halt >= 1.0 - _HALT_EPS).all())):
                break

        if logits is None:
            logits = self.lm_head(self.out_norm(x))

        result = {"logits": logits}

        if halting:
            ponder_cost = (n_updates + remainders).mean()
            result["ponder_cost"] = ponder_cost
            self._last_avg_depth = float(n_updates.detach().float().mean())

        if labels is not None:
            lm = (ds_loss / ds_wsum) if deep else self._lm_loss(logits, labels)
            result["lm_loss"] = lm
            total = lm
            if cfg.plan_states:
                plan_loss = self._plan_loss(state, labels)
                if plan_loss is not None:
                    result["plan_loss"] = plan_loss
                    total = total + cfg.plan_weight * plan_loss
            if halting:
                total = total + cfg.ponder_weight * ponder_cost
            result["loss"] = total

        return result

    def get_diagnostics(self):
        return {"avg_depth": self._last_avg_depth,
                "reasoning_steps": self._last_avg_depth}

    # ---- sampling ----

    @staticmethod
    def _sample_next(logits: torch.Tensor, temperature: float, top_k: int,
                     top_p: float) -> torch.Tensor:
        if temperature == 0:
            return logits.argmax(dim=-1, keepdim=True)
        logits = logits / temperature

        if top_k > 0:
            kth = torch.topk(logits, min(top_k, logits.size(-1)))[0][:, -1:]
            logits = logits.masked_fill(logits < kth, float('-inf'))

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            probs = F.softmax(sorted_logits, dim=-1)
            cumulative = torch.cumsum(probs, dim=-1)
            sorted_logits[cumulative - probs >= top_p] = float('-inf')
            logits = torch.full_like(logits, float('-inf')).scatter(
                1, sorted_indices, sorted_logits
            )

        return torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 200,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        eos_id: int = 2,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """
        Autoregressive generation with temperature, top-k, top-p.

        With use_cache=True (default) each new token costs one single-position
        forward against per-iteration KV caches (exact - strict causality means
        earlier positions never change). When the window fills up, the cache is
        rebuilt from the most recent 3/4 of max_seq_len.
        """
        self.eval()
        if not use_cache:
            return self._generate_nocache(
                input_ids, max_new_tokens, temperature, top_k, top_p, eos_id
            )

        max_ctx = self.config.max_seq_len
        ids = input_ids

        cache = NexusKVCache(self.config.n_iterations)
        context = ids[:, -max_ctx:]
        logits = self(context, kv_cache=cache)["logits"][:, -1, :]

        for _ in range(max_new_tokens):
            next_id = self._sample_next(logits, temperature, top_k, top_p)
            ids = torch.cat([ids, next_id], dim=1)
            if next_id.item() == eos_id:
                break

            if cache.seq_len + 1 > max_ctx:
                # Window full: rebuild the cache from the recent context
                # (25% stride keeps rebuilds rare).
                cache = NexusKVCache(self.config.n_iterations)
                context = ids[:, -(max_ctx * 3 // 4):]
                logits = self(context, kv_cache=cache)["logits"][:, -1, :]
            else:
                logits = self(next_id, kv_cache=cache,
                              pos_offset=cache.seq_len)["logits"][:, -1, :]

        return ids

    @torch.no_grad()
    def _generate_nocache(self, input_ids, max_new_tokens, temperature,
                          top_k, top_p, eos_id):
        """Original full-recompute generation (fallback / reference)."""
        for _ in range(max_new_tokens):
            x = input_ids[:, -self.config.max_seq_len:]
            logits = self(x)["logits"][:, -1, :]
            next_id = self._sample_next(logits, temperature, top_k, top_p)
            input_ids = torch.cat([input_ids, next_id], dim=1)
            if next_id.item() == eos_id:
                break
        return input_ids

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        return {
            "total_params": total,
            # Honest framing: weight sharing multiplies DEPTH/compute, not
            # capacity. Kept for backward compat with older scripts.
            "effective_params": total * self.config.n_iterations,
            "iterations": self.config.n_iterations,
        }
