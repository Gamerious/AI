"""
NeuroSpark Comprehensive Task Suite

A diverse collection of synthetic tasks to test and train the AI across
multiple cognitive domains:

1. MATH TASKS:
   - Addition, subtraction, multiplication
   - Modular arithmetic
   - Number comparison (greater/less)
   - Sequence summation
   - Fibonacci detection
   - Prime detection patterns

2. LOGIC TASKS:
   - Boolean logic (AND, OR, XOR, NOT)
   - Implication chains
   - Set operations (union, intersection patterns)
   - If-then reasoning

3. PATTERN TASKS:
   - Sequence continuation (arithmetic, geometric)
   - Palindrome detection
   - Pattern repetition
   - Mirror sequences

4. MEMORY TASKS:
   - Copy with delay
   - Selective copy (copy only certain tokens)
   - Associative recall
   - Token counting

5. SEQUENCE TASKS:
   - Sorting
   - Reversing
   - Deduplication
   - Rotation

6. LANGUAGE-LIKE TASKS:
   - Grammar pattern learning (subject-verb-object)
   - Bracket matching
   - Substitution cipher
"""

import torch
from torch.utils.data import Dataset
import random
import math
from typing import List, Dict, Tuple, Optional


# Special tokens
PAD = 0
SEP = 1   # Separator between input and output
BOS = 2   # Beginning of sequence
EOS = 3   # End of sequence
OFFSET = 10  # Numbers start at this offset to avoid collision with special tokens


class TaskBase(Dataset):
    """Base class for all tasks."""

    def __init__(self, n_samples: int, seq_len: int, vocab_size: int):
        self.n_samples = n_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.data = []
        self._generate()

    def _generate(self):
        raise NotImplementedError

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def _pad(self, tokens: List[int], length: int) -> torch.Tensor:
        """Pad or truncate to fixed length."""
        tokens = tokens[:length]
        tokens = tokens + [PAD] * (length - len(tokens))
        return torch.tensor(tokens, dtype=torch.long)

    def _make_sample(self, input_tokens: List[int], output_tokens: List[int]) -> Dict:
        """Create input_ids and labels for NEXT-TOKEN prediction (causal/autoregressive).

        Format: [BOS] input [SEP] output [EOS] [PAD...]
        Labels are shifted: at position i, the label is the token at position i+1.
        We only compute loss on the output portion (after SEP).
        """
        full_seq = [BOS] + input_tokens + [SEP] + output_tokens + [EOS]

        # For next-token prediction: label[i] = full_seq[i+1]
        # We only supervise the output part: from SEP position onward
        n_prefix = len(input_tokens) + 1  # BOS + input (positions before SEP)

        # Labels shifted by 1: predict next token
        labels = [-100] * n_prefix  # Don't supervise input prefix
        # From SEP onward, each position predicts the next token
        for i in range(n_prefix, len(full_seq) - 1):
            labels.append(full_seq[i + 1])
        labels.append(-100)  # Last token has no next token to predict

        # Pad both
        input_ids = self._pad(full_seq, self.seq_len)
        labels_padded = labels + [-100] * (self.seq_len - len(labels))
        labels_tensor = torch.tensor(labels_padded[:self.seq_len], dtype=torch.long)

        return {"input_ids": input_ids, "labels": labels_tensor}


# ============================================================
# MATH TASKS
# ============================================================

class AdditionTask(TaskBase):
    """Learn to add two numbers represented as token sequences.
    Input: num1 tokens + SEP + num2 tokens → Output: sum tokens"""

    def _generate(self):
        max_num = min(100, (self.vocab_size - OFFSET) // 2)
        for _ in range(self.n_samples):
            a = random.randint(0, max_num)
            b = random.randint(0, max_num)
            result = a + b

            # Encode numbers as individual digit tokens
            a_tokens = [int(d) + OFFSET for d in str(a)]
            b_tokens = [int(d) + OFFSET for d in str(b)]
            r_tokens = [int(d) + OFFSET for d in str(result)]

            self.data.append(self._make_sample(a_tokens + [SEP] + b_tokens, r_tokens))


class SubtractionTask(TaskBase):
    """Learn subtraction: a - b where a >= b."""

    def _generate(self):
        max_num = min(100, (self.vocab_size - OFFSET) // 2)
        for _ in range(self.n_samples):
            a = random.randint(0, max_num)
            b = random.randint(0, a)  # Ensure non-negative result
            result = a - b

            a_tokens = [int(d) + OFFSET for d in str(a)]
            b_tokens = [int(d) + OFFSET for d in str(b)]
            r_tokens = [int(d) + OFFSET for d in str(result)]

            self.data.append(self._make_sample(a_tokens + [SEP] + b_tokens, r_tokens))


class MultiplicationTask(TaskBase):
    """Learn multiplication of small numbers."""

    def _generate(self):
        max_num = min(20, (self.vocab_size - OFFSET) // 4)
        for _ in range(self.n_samples):
            a = random.randint(0, max_num)
            b = random.randint(0, max_num)
            result = a * b

            a_tokens = [int(d) + OFFSET for d in str(a)]
            b_tokens = [int(d) + OFFSET for d in str(b)]
            r_tokens = [int(d) + OFFSET for d in str(result)]

            self.data.append(self._make_sample(a_tokens + [SEP] + b_tokens, r_tokens))


class ModularArithmeticTask(TaskBase):
    """Learn modular arithmetic: a mod b."""

    def _generate(self):
        max_num = min(50, (self.vocab_size - OFFSET) // 2)
        for _ in range(self.n_samples):
            b = random.randint(2, max(3, max_num // 2))
            a = random.randint(0, max_num)
            result = a % b

            a_tokens = [int(d) + OFFSET for d in str(a)]
            b_tokens = [int(d) + OFFSET for d in str(b)]
            r_tokens = [int(d) + OFFSET for d in str(result)]

            self.data.append(self._make_sample(a_tokens + [SEP] + b_tokens, r_tokens))


class NumberComparisonTask(TaskBase):
    """Learn to compare numbers: output 1 if a > b, else 0."""

    def _generate(self):
        max_num = min(100, (self.vocab_size - OFFSET) // 2)
        TRUE_TOKEN = OFFSET + 11  # Represents 'true'
        FALSE_TOKEN = OFFSET + 12  # Represents 'false'

        for _ in range(self.n_samples):
            a = random.randint(0, max_num)
            b = random.randint(0, max_num)
            result = TRUE_TOKEN if a > b else FALSE_TOKEN

            a_tokens = [int(d) + OFFSET for d in str(a)]
            b_tokens = [int(d) + OFFSET for d in str(b)]

            self.data.append(self._make_sample(a_tokens + [SEP] + b_tokens, [result]))


class SequenceSumTask(TaskBase):
    """Sum a sequence of single-digit numbers."""

    def _generate(self):
        for _ in range(self.n_samples):
            length = random.randint(2, min(8, self.seq_len // 4))
            numbers = [random.randint(1, 9) for _ in range(length)]
            total = sum(numbers)

            input_tokens = [n + OFFSET for n in numbers]
            result_tokens = [int(d) + OFFSET for d in str(total)]

            self.data.append(self._make_sample(input_tokens, result_tokens))


class FibonacciTask(TaskBase):
    """Given first two Fibonacci numbers, predict the next N."""

    def _generate(self):
        for _ in range(self.n_samples):
            a, b = random.randint(1, 5), random.randint(1, 5)
            seq = [a, b]
            for _ in range(4):
                seq.append(seq[-1] + seq[-2])
                if seq[-1] >= self.vocab_size - OFFSET:
                    seq.pop()
                    break

            input_tokens = [seq[0] + OFFSET, seq[1] + OFFSET]
            output_tokens = [s + OFFSET for s in seq[2:]]

            if output_tokens:
                self.data.append(self._make_sample(input_tokens, output_tokens))


# ============================================================
# LOGIC TASKS
# ============================================================

class BooleanLogicTask(TaskBase):
    """Learn boolean operations: AND, OR, XOR, NOT."""

    def _generate(self):
        # Encode: operation_token, a, b → result
        AND_OP = OFFSET + 20
        OR_OP = OFFSET + 21
        XOR_OP = OFFSET + 22
        NOT_OP = OFFSET + 23
        TRUE = OFFSET + 11
        FALSE = OFFSET + 12

        for _ in range(self.n_samples):
            a = random.choice([True, False])
            b = random.choice([True, False])
            op = random.choice(['and', 'or', 'xor', 'not'])

            a_tok = TRUE if a else FALSE
            b_tok = TRUE if b else FALSE

            if op == 'and':
                result = a and b
                input_tokens = [AND_OP, a_tok, b_tok]
            elif op == 'or':
                result = a or b
                input_tokens = [OR_OP, a_tok, b_tok]
            elif op == 'xor':
                result = a ^ b
                input_tokens = [XOR_OP, a_tok, b_tok]
            else:  # not
                result = not a
                input_tokens = [NOT_OP, a_tok]

            r_tok = TRUE if result else FALSE
            self.data.append(self._make_sample(input_tokens, [r_tok]))


class ImplicationChainTask(TaskBase):
    """If A→B and B→C and A is true, what is C?
    Tests multi-step logical reasoning."""

    def _generate(self):
        TRUE = OFFSET + 11
        FALSE = OFFSET + 12
        IMPLIES = OFFSET + 24

        for _ in range(self.n_samples):
            # Chain of 2-4 implications
            chain_len = random.randint(2, 4)
            values = [random.choice([True, False])]

            input_tokens = [TRUE if values[0] else FALSE]
            for i in range(chain_len):
                # Each step can flip or keep the value
                flip = random.choice([True, False])
                if flip:
                    values.append(not values[-1])
                else:
                    values.append(values[-1])
                input_tokens.extend([IMPLIES, TRUE if values[-1] else FALSE])

            # Output: final value
            self.data.append(self._make_sample(input_tokens, [TRUE if values[-1] else FALSE]))


# ============================================================
# PATTERN TASKS
# ============================================================

class ArithmeticSequenceTask(TaskBase):
    """Continue an arithmetic sequence: given first terms, predict next."""

    def _generate(self):
        for _ in range(self.n_samples):
            start = random.randint(1, 20)
            step = random.randint(1, 5)
            n_given = random.randint(3, 5)
            n_predict = random.randint(2, 3)

            seq = [start + i * step for i in range(n_given + n_predict)]
            if max(seq) >= self.vocab_size - OFFSET:
                continue

            input_tokens = [s + OFFSET for s in seq[:n_given]]
            output_tokens = [s + OFFSET for s in seq[n_given:]]

            self.data.append(self._make_sample(input_tokens, output_tokens))


class GeometricSequenceTask(TaskBase):
    """Continue a geometric sequence (with small ratios)."""

    def _generate(self):
        for _ in range(self.n_samples):
            start = random.randint(1, 5)
            ratio = random.randint(2, 3)
            n_given = 3
            n_predict = 2

            seq = [start * (ratio ** i) for i in range(n_given + n_predict)]
            if max(seq) >= self.vocab_size - OFFSET:
                continue

            input_tokens = [s + OFFSET for s in seq[:n_given]]
            output_tokens = [s + OFFSET for s in seq[n_given:]]

            self.data.append(self._make_sample(input_tokens, output_tokens))


class PalindromeTask(TaskBase):
    """Detect if a sequence is a palindrome. Output TRUE/FALSE."""

    def _generate(self):
        TRUE = OFFSET + 11
        FALSE = OFFSET + 12

        for _ in range(self.n_samples):
            length = random.randint(3, min(8, self.seq_len // 3))

            if random.random() < 0.5:
                # Generate palindrome
                half = [random.randint(OFFSET, OFFSET + 9) for _ in range(length // 2)]
                if length % 2 == 1:
                    seq = half + [random.randint(OFFSET, OFFSET + 9)] + half[::-1]
                else:
                    seq = half + half[::-1]
                result = TRUE
            else:
                # Generate non-palindrome
                seq = [random.randint(OFFSET, OFFSET + 9) for _ in range(length)]
                while seq == seq[::-1]:
                    seq = [random.randint(OFFSET, OFFSET + 9) for _ in range(length)]
                result = FALSE

            self.data.append(self._make_sample(seq, [result]))


class PatternRepetitionTask(TaskBase):
    """Given a pattern, repeat it N times.
    Input: pattern + SEP + count → Output: repeated pattern"""

    def _generate(self):
        for _ in range(self.n_samples):
            pattern_len = random.randint(2, 4)
            repeats = random.randint(2, 3)

            pattern = [random.randint(OFFSET, OFFSET + 9) for _ in range(pattern_len)]
            count_token = repeats + OFFSET

            output = pattern * repeats
            if len(output) + pattern_len + 5 > self.seq_len:
                continue

            self.data.append(self._make_sample(pattern + [SEP] + [count_token], output))


class MirrorSequenceTask(TaskBase):
    """Output the mirror (reverse) of the input sequence."""

    def _generate(self):
        for _ in range(self.n_samples):
            length = random.randint(3, min(10, self.seq_len // 3))
            seq = [random.randint(OFFSET, OFFSET + 9) for _ in range(length)]
            self.data.append(self._make_sample(seq, seq[::-1]))


# ============================================================
# MEMORY TASKS
# ============================================================

class DelayedCopyTask(TaskBase):
    """Copy input after a delay of noise tokens. Tests working memory."""

    def _generate(self):
        NOISE = OFFSET + 15  # Noise token

        for _ in range(self.n_samples):
            length = random.randint(3, min(8, self.seq_len // 4))
            delay = random.randint(2, min(5, self.seq_len // 4))

            seq = [random.randint(OFFSET, OFFSET + 9) for _ in range(length)]
            noise = [NOISE] * delay

            self.data.append(self._make_sample(seq + noise, seq))


class SelectiveCopyTask(TaskBase):
    """Copy only tokens that match a marker. Tests selective attention."""

    def _generate(self):
        MARKER = OFFSET + 16

        for _ in range(self.n_samples):
            length = random.randint(6, min(12, self.seq_len // 3))
            seq = []
            selected = []

            for _ in range(length):
                tok = random.randint(OFFSET, OFFSET + 9)
                if random.random() < 0.4:
                    seq.extend([MARKER, tok])
                    selected.append(tok)
                else:
                    seq.append(tok)

            if selected and len(seq) + len(selected) + 5 < self.seq_len:
                self.data.append(self._make_sample(seq, selected))


class TokenCountingTask(TaskBase):
    """Count occurrences of a specific token in a sequence."""

    def _generate(self):
        for _ in range(self.n_samples):
            length = random.randint(5, min(15, self.seq_len // 3))
            target = random.randint(OFFSET, OFFSET + 5)

            seq = [random.randint(OFFSET, OFFSET + 5) for _ in range(length)]
            count = seq.count(target)

            input_tokens = [target, SEP] + seq
            count_tokens = [int(d) + OFFSET for d in str(count)]

            self.data.append(self._make_sample(input_tokens, count_tokens))


class AssociativeRecallTask(TaskBase):
    """Given key-value pairs, recall value for a query key."""

    def _generate(self):
        PAIR_SEP = OFFSET + 17

        for _ in range(self.n_samples):
            n_pairs = random.randint(2, min(5, self.seq_len // 6))
            keys = random.sample(range(OFFSET, OFFSET + 20), n_pairs)
            values = [random.randint(OFFSET, OFFSET + 9) for _ in range(n_pairs)]

            # Build input: k1 v1 PAIR_SEP k2 v2 PAIR_SEP ... SEP query_key
            input_tokens = []
            for k, v in zip(keys, values):
                input_tokens.extend([k, v, PAIR_SEP])

            # Query a random key
            query_idx = random.randint(0, n_pairs - 1)
            input_tokens.append(keys[query_idx])

            output_tokens = [values[query_idx]]

            if len(input_tokens) + 5 < self.seq_len:
                self.data.append(self._make_sample(input_tokens, output_tokens))


# ============================================================
# SEQUENCE TASKS
# ============================================================

class SortingTask(TaskBase):
    """Sort a sequence of numbers in ascending order."""

    def _generate(self):
        for _ in range(self.n_samples):
            length = random.randint(3, min(8, self.seq_len // 3))
            seq = [random.randint(OFFSET, OFFSET + 20) for _ in range(length)]
            sorted_seq = sorted(seq)
            self.data.append(self._make_sample(seq, sorted_seq))


class DeduplicationTask(TaskBase):
    """Remove duplicates from a sequence, keeping first occurrence."""

    def _generate(self):
        for _ in range(self.n_samples):
            length = random.randint(4, min(10, self.seq_len // 3))
            seq = [random.randint(OFFSET, OFFSET + 5) for _ in range(length)]

            seen = set()
            deduped = []
            for t in seq:
                if t not in seen:
                    seen.add(t)
                    deduped.append(t)

            self.data.append(self._make_sample(seq, deduped))


class RotationTask(TaskBase):
    """Rotate a sequence by N positions."""

    def _generate(self):
        for _ in range(self.n_samples):
            length = random.randint(4, min(10, self.seq_len // 3))
            seq = [random.randint(OFFSET, OFFSET + 9) for _ in range(length)]
            rotation = random.randint(1, length - 1)

            rotated = seq[rotation:] + seq[:rotation]
            input_tokens = seq + [SEP] + [rotation + OFFSET]

            self.data.append(self._make_sample(input_tokens, rotated))


# ============================================================
# LANGUAGE-LIKE TASKS
# ============================================================

class BracketMatchingTask(TaskBase):
    """Check if brackets are balanced. Output TRUE/FALSE."""

    def _generate(self):
        TRUE = OFFSET + 11
        FALSE = OFFSET + 12
        OPEN = OFFSET + 30
        CLOSE = OFFSET + 31

        for _ in range(self.n_samples):
            length = random.randint(2, min(10, self.seq_len // 3))

            if random.random() < 0.5:
                # Generate balanced brackets
                seq = []
                depth = 0
                for i in range(length):
                    if depth == 0 or (random.random() < 0.5 and i < length - 1):
                        seq.append(OPEN)
                        depth += 1
                    else:
                        seq.append(CLOSE)
                        depth -= 1
                # Close remaining
                while depth > 0:
                    seq.append(CLOSE)
                    depth -= 1
                result = TRUE
            else:
                # Generate unbalanced brackets
                seq = [random.choice([OPEN, CLOSE]) for _ in range(length)]
                # Verify it's actually unbalanced
                depth = 0
                balanced = True
                for t in seq:
                    if t == OPEN:
                        depth += 1
                    else:
                        depth -= 1
                    if depth < 0:
                        balanced = False
                        break
                if depth != 0:
                    balanced = False
                result = TRUE if balanced else FALSE

            self.data.append(self._make_sample(seq, [result]))


class SubstitutionCipherTask(TaskBase):
    """Apply a simple Caesar cipher: shift each token by a constant."""

    def _generate(self):
        for _ in range(self.n_samples):
            length = random.randint(3, min(8, self.seq_len // 3))
            shift = random.randint(1, 5)

            seq = [random.randint(OFFSET, OFFSET + 9) for _ in range(length)]
            shifted = [(t - OFFSET + shift) % 10 + OFFSET for t in seq]

            input_tokens = [shift + OFFSET, SEP] + seq
            self.data.append(self._make_sample(input_tokens, shifted))


class GrammarPatternTask(TaskBase):
    """Learn simple grammar: Subject-Verb-Object patterns.
    Given subject + verb, predict valid object class."""

    def _generate(self):
        # Define "word classes" as token ranges
        SUBJECTS = list(range(OFFSET + 40, OFFSET + 45))  # 5 subjects
        VERBS = list(range(OFFSET + 45, OFFSET + 50))      # 5 verbs
        OBJECTS = list(range(OFFSET + 50, OFFSET + 55))     # 5 objects

        # Define grammar rules: certain subjects + verbs → certain objects
        rules = {}
        for s in SUBJECTS:
            for v in VERBS:
                # Deterministic mapping based on subject and verb
                obj_idx = ((s - OFFSET) + (v - OFFSET)) % len(OBJECTS)
                rules[(s, v)] = OBJECTS[obj_idx]

        for _ in range(self.n_samples):
            s = random.choice(SUBJECTS)
            v = random.choice(VERBS)
            o = rules[(s, v)]

            self.data.append(self._make_sample([s, v], [o]))


# ============================================================
# COMBINED DATASET
# ============================================================

class CombinedTaskDataset(Dataset):
    """Combines all tasks into a single dataset for multi-task training."""

    TASK_CLASSES = {
        # Math
        "addition": AdditionTask,
        "subtraction": SubtractionTask,
        "multiplication": MultiplicationTask,
        "modular_arithmetic": ModularArithmeticTask,
        "number_comparison": NumberComparisonTask,
        "sequence_sum": SequenceSumTask,
        "fibonacci": FibonacciTask,
        # Logic
        "boolean_logic": BooleanLogicTask,
        "implication_chain": ImplicationChainTask,
        # Pattern
        "arithmetic_sequence": ArithmeticSequenceTask,
        "geometric_sequence": GeometricSequenceTask,
        "palindrome": PalindromeTask,
        "pattern_repetition": PatternRepetitionTask,
        "mirror_sequence": MirrorSequenceTask,
        # Memory
        "delayed_copy": DelayedCopyTask,
        "selective_copy": SelectiveCopyTask,
        "token_counting": TokenCountingTask,
        "associative_recall": AssociativeRecallTask,
        # Sequence
        "sorting": SortingTask,
        "deduplication": DeduplicationTask,
        "rotation": RotationTask,
        # Language-like
        "bracket_matching": BracketMatchingTask,
        "substitution_cipher": SubstitutionCipherTask,
        "grammar_pattern": GrammarPatternTask,
    }

    def __init__(
        self,
        n_samples_per_task: int = 1000,
        seq_len: int = 64,
        vocab_size: int = 200,
        tasks: Optional[List[str]] = None,
    ):
        self.seq_len = seq_len
        self.vocab_size = vocab_size

        if tasks is None:
            tasks = list(self.TASK_CLASSES.keys())

        self.task_datasets = {}
        self.all_data = []
        self.task_labels = []  # Track which task each sample belongs to

        for task_name in tasks:
            if task_name in self.TASK_CLASSES:
                print(f"  Generating task: {task_name}...", end=" ")
                dataset = self.TASK_CLASSES[task_name](n_samples_per_task, seq_len, vocab_size)
                self.task_datasets[task_name] = dataset
                n = len(dataset)
                self.all_data.extend([dataset[i] for i in range(n)])
                self.task_labels.extend([task_name] * n)
                print(f"{n} samples")

        print(f"  Total: {len(self.all_data)} samples across {len(self.task_datasets)} tasks")

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
        return self.all_data[idx]

    def get_task_name(self, idx: int) -> str:
        return self.task_labels[idx]
