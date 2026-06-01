"""
Memmap token dataset for web-scale NEXUS-LM training (nanoGPT-style).

The data pipeline (prepare_fineweb.py) writes one flat uint16 stream of token
ids to <data_dir>/train.bin and val.bin. Training reads non-overlapping
seq_len windows straight out of the memory-mapped file - near-zero CPU/RAM
cost per batch, so the GPU never waits on tokenization.

The dataset returns a 1-tuple ``(tokens,)`` to stay drop-in compatible with
the existing trainer, which reads ``batch[0]`` (same shape as TensorDataset).
"""

import os
import json

import numpy as np
import torch
from torch.utils.data import Dataset


class MemmapTokenDataset(Dataset):
    """Non-overlapping contiguous seq_len windows over a flat uint16 .bin."""

    def __init__(self, bin_path: str, seq_len: int, dtype: str = "uint16"):
        self.bin_path = bin_path
        self.seq_len = seq_len
        self.np_dtype = np.dtype(dtype)
        n_bytes = os.path.getsize(bin_path)
        self.n_tokens = n_bytes // self.np_dtype.itemsize
        self.n = self.n_tokens // seq_len
        if self.n == 0:
            raise ValueError(
                f"{bin_path} has {self.n_tokens:,} tokens, fewer than seq_len={seq_len}"
            )
        # Opened lazily so each DataLoader worker gets its own memmap handle
        # (np.memmap doesn't survive pickling across worker processes).
        self._data = None

    def _mm(self):
        if self._data is None:
            self._data = np.memmap(self.bin_path, dtype=self.np_dtype, mode="r")
        return self._data

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        start = i * self.seq_len
        window = np.asarray(self._mm()[start:start + self.seq_len]).astype(np.int64)
        return (torch.from_numpy(window),)


class ChatMemmapDataset(Dataset):
    """seq_len windows over a packed chat stream, with a SFT loss mask.

    Reads two parallel files written by prepare_chat.py:
      tokens.bin (uint16)  - the packed ChatML token stream
      mask.bin   (uint8)   - 1 where the token is a trainable assistant token, else 0

    Returns ``(tokens, labels)`` where labels == tokens on assistant tokens and
    -100 (ignored by the loss) everywhere else. The trainer passes labels[1] as
    the model's labels instead of cloning the input.
    """

    def __init__(self, tokens_path: str, mask_path: str, seq_len: int, dtype: str = "uint16"):
        self.tokens_path = tokens_path
        self.mask_path = mask_path
        self.seq_len = seq_len
        self.np_dtype = np.dtype(dtype)
        n_bytes = os.path.getsize(tokens_path)
        self.n_tokens = n_bytes // self.np_dtype.itemsize
        self.n = self.n_tokens // seq_len
        if self.n == 0:
            raise ValueError(f"{tokens_path}: {self.n_tokens:,} tokens < seq_len={seq_len}")
        self._tok = None
        self._mask = None

    def _mm(self):
        if self._tok is None:
            self._tok = np.memmap(self.tokens_path, dtype=self.np_dtype, mode="r")
            self._mask = np.memmap(self.mask_path, dtype=np.uint8, mode="r")
        return self._tok, self._mask

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        tok, mask = self._mm()
        s = i * self.seq_len
        t = np.asarray(tok[s:s + self.seq_len]).astype(np.int64)
        m = np.asarray(mask[s:s + self.seq_len]).astype(np.bool_)
        labels = np.where(m, t, -100).astype(np.int64)
        return (torch.from_numpy(t), torch.from_numpy(labels))


def load_meta(data_dir: str) -> dict:
    """Load meta.json written by prepare_fineweb.py / prepare_chat.py."""
    with open(os.path.join(data_dir, "meta.json"), "r") as f:
        return json.load(f)


def has_bin_data(data_dir: str) -> bool:
    return os.path.exists(os.path.join(data_dir, "train.bin")) or \
        os.path.exists(os.path.join(data_dir, "tokens.bin"))


def is_chat_data(data_dir: str) -> bool:
    return os.path.exists(os.path.join(data_dir, "tokens.bin")) and \
        os.path.exists(os.path.join(data_dir, "mask.bin"))
