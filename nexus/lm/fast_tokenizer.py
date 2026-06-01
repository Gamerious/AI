"""
FastTokenizer - HuggingFace-tokenizers (Rust) backend for NEXUS-LM.

Drop-in replacement for BPETokenizer with the SAME interface
(.vocab_size / .encode / .decode / .pad_id / .bos_id / .eos_id / .save / .load),
but tokenizes at GB/s instead of MB/s. Used for the web-scale data pipeline
(FineWeb-Edu) where the pure-Python BPE would bottleneck a fast GPU.

Special token ids are FIXED to match the rest of the codebase:
    0 = <pad>, 1 = <bos>, 2 = <eos>, 3 = <unk>

Requires `tokenizers` (pip install tokenizers). The model itself stays
tokenizer-agnostic - it only ever sees integer ids, weight-tied to vocab_size.
"""

import os
from typing import List, Iterable


FILENAME = "fast_tokenizer.json"
SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]  # -> ids 0,1,2,3


class FastTokenizer:
    def __init__(self, tk):
        self._tk = tk
        self.vocab_size = tk.get_vocab_size()

    @classmethod
    def train(cls, text_iter: Iterable[str], vocab_size: int = 32000, length=None) -> "FastTokenizer":
        """Train a byte-level BPE on an iterator of strings."""
        try:
            from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
        except ImportError as e:
            raise ImportError(
                "FastTokenizer needs the 'tokenizers' package: pip install tokenizers"
            ) from e

        tk = Tokenizer(models.BPE(unk_token="<unk>"))
        tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tk.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=SPECIAL_TOKENS,  # added first -> ids 0..3
            show_progress=True,
        )
        tk.train_from_iterator(text_iter, trainer=trainer, length=length)
        return cls(tk)

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        ids = self._tk.encode(text).ids
        if add_bos:
            ids = [1] + ids
        if add_eos:
            ids = ids + [2]
        return ids

    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        """Parallel (multi-core) batch encode - this is where the Rust speed pays off."""
        return [e.ids for e in self._tk.encode_batch(texts)]

    def decode(self, token_ids: List[int]) -> str:
        return self._tk.decode(list(token_ids), skip_special_tokens=True)

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        self._tk.save(os.path.join(path, FILENAME))
        print(f"  FastTokenizer saved: {path} (vocab_size={self.vocab_size})")

    @classmethod
    def load(cls, path: str) -> "FastTokenizer":
        try:
            from tokenizers import Tokenizer
        except ImportError as e:
            raise ImportError(
                "FastTokenizer needs the 'tokenizers' package: pip install tokenizers"
            ) from e
        f = path if path.endswith(".json") else os.path.join(path, FILENAME)
        return cls(Tokenizer.from_file(f))

    @staticmethod
    def exists(path: str) -> bool:
        f = path if path.endswith(".json") else os.path.join(path, FILENAME)
        return os.path.exists(f)

    @property
    def pad_id(self) -> int:
        return 0

    @property
    def bos_id(self) -> int:
        return 1

    @property
    def eos_id(self) -> int:
        return 2
