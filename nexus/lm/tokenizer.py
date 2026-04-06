"""
Byte-Pair Encoding (BPE) Tokenizer for NEXUS-LM

Zero external dependencies. Implements:
- Byte-level BPE (like GPT-2) - works with ANY language
- Train from text corpus
- Save/load vocabulary
- Encode text → token IDs
- Decode token IDs → text

Vocab structure:
  0: <pad>
  1: <bos>
  2: <eos>
  3: <unk>
  4-259: raw bytes (0x00-0xFF)
  260+: BPE merges
"""

import json
import os
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple


# Special tokens
PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"

SPECIAL_TOKENS = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]
N_SPECIAL = len(SPECIAL_TOKENS)
N_BYTES = 256  # Raw byte tokens
MERGE_START = N_SPECIAL + N_BYTES  # = 260


class BPETokenizer:
    """
    Byte-level BPE tokenizer.

    Works by:
    1. Converting text to UTF-8 bytes
    2. Splitting on word boundaries (spaces, punctuation)
    3. Iteratively merging most frequent byte pairs
    """

    def __init__(self):
        self.merges: List[Tuple[int, int]] = []  # Ordered list of merge pairs
        self.vocab: Dict[int, bytes] = {}  # token_id → bytes
        self.vocab_size: int = MERGE_START
        self._build_base_vocab()

    def _build_base_vocab(self):
        """Build base vocabulary: special tokens + 256 byte tokens."""
        self.vocab = {}
        # Special tokens
        for i, tok in enumerate(SPECIAL_TOKENS):
            self.vocab[i] = tok.encode("utf-8")
        # Byte tokens
        for i in range(N_BYTES):
            self.vocab[N_SPECIAL + i] = bytes([i])

    def _text_to_byte_ids(self, text: str) -> List[int]:
        """Convert text to a list of byte-level token IDs."""
        return [b + N_SPECIAL for b in text.encode("utf-8")]

    def _get_pair_counts(self, sequences: List[List[int]]) -> Counter:
        """Count all adjacent pairs across all sequences."""
        counts = Counter()
        for seq in sequences:
            for i in range(len(seq) - 1):
                counts[(seq[i], seq[i + 1])] += 1
        return counts

    def _merge_pair(self, sequences: List[List[int]], pair: Tuple[int, int], new_id: int) -> List[List[int]]:
        """Replace all occurrences of pair with new_id in all sequences."""
        result = []
        for seq in sequences:
            new_seq = []
            i = 0
            while i < len(seq):
                if i < len(seq) - 1 and seq[i] == pair[0] and seq[i + 1] == pair[1]:
                    new_seq.append(new_id)
                    i += 2
                else:
                    new_seq.append(seq[i])
                    i += 1
            result.append(new_seq)
        return result

    def train(self, texts: List[str], target_vocab_size: int = 16000, verbose: bool = True):
        """
        Train BPE tokenizer on a list of texts.

        Args:
            texts: List of training texts
            target_vocab_size: Desired vocabulary size (including special + byte tokens)
            verbose: Print progress
        """
        n_merges = target_vocab_size - MERGE_START
        if n_merges <= 0:
            return

        if verbose:
            print(f"  Training BPE tokenizer: target vocab={target_vocab_size}, merges={n_merges}")

        # Split texts into words (preserve spaces as part of tokens)
        # GPT-2 style: split on word boundaries, keep leading spaces
        word_pattern = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?[^\s\w]+|\s+""")

        # Tokenize all words to byte sequences
        word_freqs: Counter = Counter()
        for text in texts:
            words = word_pattern.findall(text)
            for word in words:
                word_freqs[word] += 1

        if verbose:
            print(f"  Unique words: {len(word_freqs):,}")

        # Convert words to byte ID sequences, weighted by frequency
        sequences = []
        weights = []
        for word, freq in word_freqs.items():
            byte_ids = self._text_to_byte_ids(word)
            if len(byte_ids) >= 2:
                sequences.append(byte_ids)
                weights.append(freq)

        # Iterative merging
        self.merges = []
        next_id = MERGE_START

        for merge_idx in range(n_merges):
            if not sequences:
                break

            # Count pairs (weighted)
            pair_counts: Counter = Counter()
            for seq, w in zip(sequences, weights):
                for i in range(len(seq) - 1):
                    pair_counts[(seq[i], seq[i + 1])] += w

            if not pair_counts:
                break

            # Find most frequent pair
            best_pair = pair_counts.most_common(1)[0][0]
            best_count = pair_counts[best_pair]

            if best_count < 2:
                break

            # Create new token
            self.merges.append(best_pair)
            self.vocab[next_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]

            # Apply merge
            sequences = self._merge_pair(sequences, best_pair, next_id)
            next_id += 1

            if verbose and (merge_idx + 1) % 500 == 0:
                print(f"    Merge {merge_idx+1}/{n_merges}: "
                      f"({best_pair[0]}, {best_pair[1]}) → {next_id-1} "
                      f"(count={best_count:,})")

        self.vocab_size = next_id
        if verbose:
            print(f"  Done! Final vocab size: {self.vocab_size}")

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        """Encode text to token IDs."""
        if not text:
            tokens = []
        else:
            # Convert to byte IDs
            tokens = self._text_to_byte_ids(text)

            # Apply merges in order
            for pair_idx, pair in enumerate(self.merges):
                new_id = MERGE_START + pair_idx
                i = 0
                new_tokens = []
                while i < len(tokens):
                    if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                        new_tokens.append(new_id)
                        i += 2
                    else:
                        new_tokens.append(tokens[i])
                        i += 1
                tokens = new_tokens

        if add_bos:
            tokens = [1] + tokens  # BOS = 1
        if add_eos:
            tokens = tokens + [2]  # EOS = 2
        return tokens

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back to text."""
        byte_list = []
        for tid in token_ids:
            if tid < N_SPECIAL:
                continue  # Skip special tokens
            if tid in self.vocab:
                byte_list.append(self.vocab[tid])
            else:
                byte_list.append(b"?")  # Unknown token

        raw_bytes = b"".join(byte_list)
        return raw_bytes.decode("utf-8", errors="replace")

    def save(self, path: str):
        """Save tokenizer to directory."""
        os.makedirs(path, exist_ok=True)
        data = {
            "vocab_size": self.vocab_size,
            "merges": self.merges,
        }
        with open(os.path.join(path, "tokenizer.json"), "w") as f:
            json.dump(data, f)
        print(f"  Tokenizer saved: {path} (vocab_size={self.vocab_size})")

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        """Load tokenizer from directory."""
        tok = cls()
        with open(os.path.join(path, "tokenizer.json"), "r") as f:
            data = json.load(f)
        tok.merges = [tuple(p) for p in data["merges"]]
        tok.vocab_size = data["vocab_size"]
        # Rebuild vocab from merges
        for pair_idx, pair in enumerate(tok.merges):
            new_id = MERGE_START + pair_idx
            tok.vocab[new_id] = tok.vocab[pair[0]] + tok.vocab[pair[1]]
        return tok

    @property
    def pad_id(self) -> int:
        return 0

    @property
    def bos_id(self) -> int:
        return 1

    @property
    def eos_id(self) -> int:
        return 2
