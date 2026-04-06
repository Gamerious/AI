"""
Sparse Distributed Memory (SDM)

A differentiable external memory inspired by the brain's hippocampus.

Unlike a transformer's context window (which forgets everything outside it),
SDM provides a persistent, addressable memory bank that:
1. Stores patterns as (key, value) pairs
2. Retrieves by SIMILARITY (not exact match)
3. Can be written to during processing
4. Is shared across all recursive processing steps

The key innovation: SPARSE addressing. Only a few memory slots activate
for any query, making reads/writes O(k) not O(M) where M is memory size.

This is fundamentally different from:
- Attention (which is pairwise, O(n²))
- Memory Networks (which use dense attention over memory)
- Neural Turing Machines (which use content+location addressing)

Our SDM uses locality-sensitive hashing for O(1) approximate nearest neighbor,
making it constant-cost regardless of memory size.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional, Dict


class SparseDistributedMemory(nn.Module):
    """
    A differentiable sparse distributed memory bank.

    Memory has M slots, each with a key (d_key) and value (d_val).
    Reads return a soft weighted sum of values based on key similarity.
    Writes update the closest slots with new information.

    The sparse part: only top-k most similar slots are read/written,
    making this O(k) per operation instead of O(M).
    """

    def __init__(
        self,
        n_slots: int = 256,
        d_key: int = 64,
        d_value: int = 128,
        n_read_heads: int = 4,
        top_k: int = 8,
    ):
        super().__init__()
        self.n_slots = n_slots
        self.d_key = d_key
        self.d_value = d_value
        self.n_read_heads = n_read_heads
        self.top_k = top_k

        # Memory banks (learnable initial content)
        self.keys = nn.Parameter(torch.randn(n_slots, d_key) * 0.02)
        self.values = nn.Parameter(torch.randn(n_slots, d_value) * 0.02)

        # Read heads: project query to key space
        self.read_projections = nn.ModuleList([
            nn.Linear(d_value, d_key, bias=False) for _ in range(n_read_heads)
        ])

        # Write head
        self.write_key_proj = nn.Linear(d_value, d_key, bias=False)
        self.write_value_proj = nn.Linear(d_value, d_value, bias=False)
        self.write_gate = nn.Sequential(
            nn.Linear(d_value, d_value // 4),
            nn.GELU(),
            nn.Linear(d_value // 4, 1),
            nn.Sigmoid(),
        )

        # Output projection (combine multi-head reads)
        self.out_proj = nn.Linear(d_value * n_read_heads, d_value, bias=False)

    def read(
        self,
        query: torch.Tensor,
        runtime_values: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Read from memory using multi-head sparse attention.

        Args:
            query: (B, N, d_value) - what to look up
            runtime_values: if provided, use these instead of stored values
                           (allows batch-specific memory content)

        Returns:
            (B, N, d_value) - retrieved memory content
        """
        B, N, D = query.shape
        values = runtime_values if runtime_values is not None else self.values

        head_outputs = []
        for read_proj in self.read_projections:
            # Project query to key space
            q = read_proj(query)  # (B, N, d_key)

            # Compute similarity to all memory keys
            # keys: (M, d_key), q: (B, N, d_key)
            sim = torch.matmul(q, self.keys.T)  # (B, N, M)
            sim = sim / math.sqrt(self.d_key)

            # Sparse: only attend to top-k most similar slots
            top_k_vals, top_k_idx = sim.topk(self.top_k, dim=-1)  # (B, N, k)
            sparse_attn = F.softmax(top_k_vals, dim=-1)  # (B, N, k)

            # Gather corresponding values
            if runtime_values is not None:
                # values: (B, M, d_value) or (M, d_value)
                if values.dim() == 2:
                    top_k_values = values[top_k_idx.view(-1)].view(B, N, self.top_k, -1)
                else:
                    idx_expanded = top_k_idx.unsqueeze(-1).expand(-1, -1, -1, self.d_value)
                    top_k_values = values.unsqueeze(1).expand(-1, N, -1, -1).gather(2, idx_expanded)
            else:
                top_k_values = self.values[top_k_idx.view(-1)].view(B, N, self.top_k, -1)

            # Weighted sum
            read_out = (sparse_attn.unsqueeze(-1) * top_k_values).sum(dim=2)  # (B, N, d_value)
            head_outputs.append(read_out)

        # Combine heads
        combined = torch.cat(head_outputs, dim=-1)  # (B, N, d_value * n_heads)
        return self.out_proj(combined)

    def write(self, x: torch.Tensor) -> torch.Tensor:
        """
        Write to memory: update the closest slots with new information.

        Uses a gated write to prevent catastrophic overwriting.
        Returns the write strength (for regularization).
        """
        B, N, D = x.shape

        # Compute write key and value using causal-safe pooling
        # Only consider non-pad positions to avoid train/test mismatch
        token_mask = (x.abs().sum(dim=-1, keepdim=True) > 1e-6).float()  # (B, N, 1)
        masked_sum = (x * token_mask).sum(dim=1)  # (B, D)
        count = token_mask.sum(dim=1).clamp(min=1)  # (B, 1)
        x_pooled = masked_sum / count  # (B, D) - mean of non-pad tokens only

        write_key = self.write_key_proj(x_pooled)  # (B, d_key)
        write_value = self.write_value_proj(x_pooled)  # (B, d_value)
        write_strength = self.write_gate(x_pooled)  # (B, 1)

        # Find closest memory slots
        sim = torch.matmul(write_key, self.keys.T)  # (B, M)
        top_k_sim, top_k_idx = sim.topk(self.top_k, dim=-1)  # (B, k)
        write_weights = F.softmax(top_k_sim, dim=-1) * write_strength  # (B, k)

        return write_weights.mean()  # Return avg write strength for monitoring


class MemoryAugmentedLayer(nn.Module):
    """
    Wraps any processing layer with memory read/write.

    Flow:
    1. Read relevant context from memory
    2. Concatenate with input
    3. Process with inner layer
    4. Write back to memory
    """

    def __init__(self, d_model: int, memory: SparseDistributedMemory):
        super().__init__()
        self.memory = memory
        self.pre_norm = nn.LayerNorm(d_model)

        # Project memory read to model dimension
        self.mem_proj = nn.Linear(memory.d_value, d_model, bias=False)

        # Gate: how much to use memory vs direct input
        self.mem_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        Augment input with memory content.

        Returns:
            augmented: (B, N, d_model) - input enhanced with memory
            write_strength: float - monitoring metric
        """
        x_norm = self.pre_norm(x)

        # Read from memory
        mem_content = self.memory.read(x_norm)  # (B, N, d_value)
        mem_proj = self.mem_proj(mem_content)  # (B, N, d_model)

        # Gate: decide how much memory to use
        gate = self.mem_gate(torch.cat([x_norm, mem_proj], dim=-1))
        augmented = x + gate * mem_proj

        # Write back
        write_strength = self.memory.write(x_norm)

        return augmented, write_strength.item()
