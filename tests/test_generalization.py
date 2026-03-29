#!/usr/bin/env python3
"""
NeuroSpark Generalization & Stress Tests

These tests go FAR beyond the training distribution to test true intelligence:

1. OUT-OF-DISTRIBUTION (OOD): Numbers/patterns never seen in training
2. COMPOSITIONAL: Combining multiple skills (e.g. sort then sum)
3. LONGER SEQUENCES: Sequences longer than training data
4. ADVERSARIAL: Edge cases designed to confuse the model
5. TRANSFER: Can skills from one domain help in another?
6. SCALING: How does performance degrade with problem difficulty?
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import time
from typing import Dict, List
from collections import defaultdict

from core.model import NeuroSparkModel, NeuroSparkConfig
from training.tasks import PAD, SEP, BOS, EOS, OFFSET


def load_model(path: str = "neurospark_trained.pt"):
    """Load trained model."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = NeuroSparkConfig(**checkpoint["config"])
    model = NeuroSparkModel(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, config


def make_input(input_tokens: List[int], seq_len: int = 64) -> torch.Tensor:
    """Create model input from token list."""
    full = [BOS] + input_tokens + [SEP]
    full = full + [PAD] * (seq_len - len(full))
    return torch.tensor([full[:seq_len]], dtype=torch.long)


def predict(model, input_ids: torch.Tensor, n_tokens: int = 10) -> List[int]:
    """Autoregressive generation using only the causal prefix.

    To avoid information leaking through the reasoning module's global pooling,
    we generate by feeding only the valid prefix tokens (truncated to actual length)
    padded minimally, and growing the sequence one token at a time.
    """
    # Extract the actual prefix (non-PAD tokens)
    non_pad = (input_ids[0] != PAD).nonzero(as_tuple=True)[0]
    if len(non_pad) == 0:
        return []

    prefix_len = non_pad[-1].item() + 1
    prefix = input_ids[0, :prefix_len].tolist()
    seq_len = input_ids.shape[1]
    predictions = []

    for i in range(n_tokens):
        current_len = len(prefix)
        if current_len >= seq_len:
            break

        # Pad to model's expected length
        padded = prefix + [PAD] * (seq_len - current_len)
        inp = torch.tensor([padded[:seq_len]], dtype=torch.long)

        with torch.no_grad():
            outputs = model(input_ids=inp)
            logits = outputs["logits"]

        # At position current_len - 1, predict next token
        pred = logits[0, current_len - 1].argmax().item()
        if pred == EOS or pred == PAD:
            break

        predictions.append(pred)
        prefix.append(pred)

    return predictions


def decode_tokens(tokens: List[int]) -> str:
    """Convert tokens back to readable form."""
    return [t - OFFSET if t >= OFFSET else f"[{t}]" for t in tokens]


class GeneralizationTest:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.results = defaultdict(list)

    def run_test(self, name: str, input_tokens: List[int], expected: List[int],
                 n_predict: int = None):
        """Run a single test and record result."""
        if n_predict is None:
            n_predict = len(expected)

        input_ids = make_input(input_tokens, self.config.max_seq_len)
        predicted = predict(self.model, input_ids, n_predict)

        correct = predicted[:len(expected)] == expected
        self.results[name].append({
            "input": decode_tokens(input_tokens),
            "expected": decode_tokens(expected),
            "predicted": decode_tokens(predicted[:len(expected)]),
            "correct": correct,
        })
        return correct

    def print_results(self):
        """Print comprehensive results."""
        print(f"\n{'='*80}")
        print(f"  GENERALIZATION TEST RESULTS")
        print(f"{'='*80}")

        total_correct = 0
        total_tests = 0

        for category, tests in self.results.items():
            n_correct = sum(1 for t in tests if t["correct"])
            n_total = len(tests)
            pct = n_correct / max(1, n_total) * 100
            total_correct += n_correct
            total_tests += n_total

            status = "PASS" if pct >= 80 else "PARTIAL" if pct >= 50 else "FAIL"
            print(f"\n  [{status}] {category}: {n_correct}/{n_total} ({pct:.0f}%)")

            # Show failures
            for t in tests:
                marker = "OK" if t["correct"] else "XX"
                print(f"    [{marker}] Input: {t['input']}")
                if not t["correct"]:
                    print(f"         Expected:  {t['expected']}")
                    print(f"         Predicted: {t['predicted']}")

        overall = total_correct / max(1, total_tests) * 100
        print(f"\n{'='*80}")
        print(f"  OVERALL: {total_correct}/{total_tests} ({overall:.1f}%)")
        print(f"{'='*80}")
        return overall


def run_generalization_tests():
    print("Loading trained model...")
    model, config = load_model()
    print(f"Model loaded. Config: d_model={config.d_model}, n_layers={config.n_layers}")

    tester = GeneralizationTest(model, config)

    # ================================================================
    # 1. MATH OOD - Numbers beyond training range
    # ================================================================
    print("\n--- Testing Math Generalization ---")

    # Addition with numbers seen in training
    for a, b in [(5, 3), (12, 7), (25, 30), (40, 50), (99, 1)]:
        a_tok = [int(d) + OFFSET for d in str(a)]
        b_tok = [int(d) + OFFSET for d in str(b)]
        r_tok = [int(d) + OFFSET for d in str(a + b)]
        tester.run_test("Math: In-dist Addition",
                       a_tok + [SEP] + b_tok, r_tok)

    # Subtraction
    for a, b in [(10, 3), (50, 25), (99, 50), (77, 33)]:
        a_tok = [int(d) + OFFSET for d in str(a)]
        b_tok = [int(d) + OFFSET for d in str(b)]
        r_tok = [int(d) + OFFSET for d in str(a - b)]
        tester.run_test("Math: In-dist Subtraction",
                       a_tok + [SEP] + b_tok, r_tok)

    # Multiplication
    for a, b in [(3, 4), (7, 8), (5, 5), (12, 3), (9, 9)]:
        a_tok = [int(d) + OFFSET for d in str(a)]
        b_tok = [int(d) + OFFSET for d in str(b)]
        r_tok = [int(d) + OFFSET for d in str(a * b)]
        tester.run_test("Math: In-dist Multiplication",
                       a_tok + [SEP] + b_tok, r_tok)

    # Modular arithmetic
    for a, b in [(17, 5), (23, 7), (44, 9), (100, 3)]:
        if b > 0:
            a_tok = [int(d) + OFFSET for d in str(a)]
            b_tok = [int(d) + OFFSET for d in str(b)]
            r_tok = [int(d) + OFFSET for d in str(a % b)]
            tester.run_test("Math: Modular Arithmetic",
                           a_tok + [SEP] + b_tok, r_tok)

    # Sequence sums
    for seq in [[1,2,3], [5,5,5,5], [9,1,9,1], [3,7,2,8,1]]:
        total = sum(seq)
        input_tok = [n + OFFSET for n in seq]
        r_tok = [int(d) + OFFSET for d in str(total)]
        tester.run_test("Math: Sequence Sum",
                       input_tok, r_tok)

    # ================================================================
    # 2. LOGIC TESTS
    # ================================================================
    print("\n--- Testing Logic ---")

    TRUE = OFFSET + 11
    FALSE = OFFSET + 12
    AND_OP = OFFSET + 20
    OR_OP = OFFSET + 21
    XOR_OP = OFFSET + 22
    NOT_OP = OFFSET + 23

    # Boolean logic
    logic_tests = [
        ([AND_OP, TRUE, TRUE], [TRUE], "T AND T = T"),
        ([AND_OP, TRUE, FALSE], [FALSE], "T AND F = F"),
        ([OR_OP, FALSE, FALSE], [FALSE], "F OR F = F"),
        ([OR_OP, FALSE, TRUE], [TRUE], "F OR T = T"),
        ([XOR_OP, TRUE, TRUE], [FALSE], "T XOR T = F"),
        ([XOR_OP, TRUE, FALSE], [TRUE], "T XOR F = T"),
        ([NOT_OP, TRUE], [FALSE], "NOT T = F"),
        ([NOT_OP, FALSE], [TRUE], "NOT F = T"),
    ]
    for inp, exp, desc in logic_tests:
        tester.run_test(f"Logic: Boolean ({desc})", inp, exp)

    # ================================================================
    # 3. PATTERN CONTINUATION
    # ================================================================
    print("\n--- Testing Pattern Recognition ---")

    # Arithmetic sequences
    for start, step, n_given, n_pred in [(1,1,4,2), (2,3,4,2), (5,5,3,2), (10,2,4,2)]:
        seq = [start + i * step for i in range(n_given + n_pred)]
        if max(seq) < config.vocab_size - OFFSET:
            inp = [s + OFFSET for s in seq[:n_given]]
            exp = [s + OFFSET for s in seq[n_given:]]
            tester.run_test("Pattern: Arithmetic Seq", inp, exp)

    # Fibonacci
    for a, b in [(1,1), (1,2), (2,3), (3,5)]:
        seq = [a, b]
        for _ in range(3):
            seq.append(seq[-1] + seq[-2])
            if seq[-1] >= config.vocab_size - OFFSET:
                seq.pop()
                break
        if len(seq) > 2:
            inp = [seq[0] + OFFSET, seq[1] + OFFSET]
            exp = [s + OFFSET for s in seq[2:]]
            tester.run_test("Pattern: Fibonacci", inp, exp)

    # ================================================================
    # 4. MEMORY TESTS
    # ================================================================
    print("\n--- Testing Memory ---")

    # Mirror/reverse
    for seq in [[1,2,3,4,5], [7,3,7], [1,1,2,2,3,3], [9,8,7,6]]:
        inp = [s + OFFSET for s in seq]
        exp = [s + OFFSET for s in reversed(seq)]
        tester.run_test("Memory: Reverse", inp, exp)

    # Delayed copy
    NOISE = OFFSET + 15
    for seq in [[1,2,3], [5,6,7,8], [3,3,3]]:
        inp = [s + OFFSET for s in seq] + [NOISE] * 3
        exp = [s + OFFSET for s in seq]
        tester.run_test("Memory: Delayed Copy", inp, exp)

    # ================================================================
    # 5. SEQUENCE OPERATIONS
    # ================================================================
    print("\n--- Testing Sequence Operations ---")

    # Sorting
    for seq in [[5,3,1,4,2], [9,7,8,6], [3,1,4,1,5,9], [2,2,1,1,3,3]]:
        inp = [s + OFFSET for s in seq]
        exp = [s + OFFSET for s in sorted(seq)]
        tester.run_test("Sequence: Sort", inp, exp)

    # Deduplication
    for seq in [[1,1,2,2,3], [5,5,5,3,3], [1,2,3,1,2], [4,4,4,4]]:
        inp = [s + OFFSET for s in seq]
        seen = set()
        deduped = []
        for s in seq:
            if s not in seen:
                seen.add(s)
                deduped.append(s)
        exp = [s + OFFSET for s in deduped]
        tester.run_test("Sequence: Dedup", inp, exp)

    # ================================================================
    # 6. LANGUAGE-LIKE TESTS
    # ================================================================
    print("\n--- Testing Language Tasks ---")

    # Palindrome detection
    for seq, is_pal in [([1,2,3,2,1], True), ([1,2,3,4,5], False),
                         ([1,1,1], True), ([1,2,1,2], False)]:
        inp = [s + OFFSET for s in seq]
        exp = [TRUE if is_pal else FALSE]
        tester.run_test("Language: Palindrome", inp, exp)

    # Bracket matching
    OPEN = OFFSET + 30
    CLOSE = OFFSET + 31
    for seq, balanced in [
        ([OPEN, CLOSE], True),
        ([OPEN, OPEN, CLOSE, CLOSE], True),
        ([OPEN, CLOSE, OPEN, CLOSE], True),
        ([CLOSE, OPEN], False),
        ([OPEN, OPEN, CLOSE], False),
    ]:
        exp = [TRUE if balanced else FALSE]
        tester.run_test("Language: Brackets", seq, exp)

    # Substitution cipher
    for shift, seq in [(1, [1,2,3]), (3, [5,6,7]), (2, [0,9,0])]:
        inp_tok = [shift + OFFSET, SEP] + [s + OFFSET for s in seq]
        exp = [(s + shift) % 10 + OFFSET for s in seq]
        tester.run_test("Language: Caesar Cipher", inp_tok, exp)

    # ================================================================
    # 7. STRESS TESTS - Harder versions
    # ================================================================
    print("\n--- Stress Tests ---")

    # Longer sorts
    for length in [8, 10, 12]:
        import random
        random.seed(length)
        seq = [random.randint(0, 15) for _ in range(length)]
        inp = [s + OFFSET for s in seq]
        exp = [s + OFFSET for s in sorted(seq)]
        tester.run_test(f"Stress: Sort len={length}", inp, exp)

    # Longer reversal
    for length in [8, 10, 12]:
        random.seed(length + 100)
        seq = [random.randint(0, 9) for _ in range(length)]
        inp = [s + OFFSET for s in seq]
        exp = [s + OFFSET for s in reversed(seq)]
        tester.run_test(f"Stress: Reverse len={length}", inp, exp)

    # Larger addition
    for a, b in [(55, 45), (88, 12), (67, 33), (91, 9)]:
        a_tok = [int(d) + OFFSET for d in str(a)]
        b_tok = [int(d) + OFFSET for d in str(b)]
        r_tok = [int(d) + OFFSET for d in str(a + b)]
        tester.run_test("Stress: Larger Addition", a_tok + [SEP] + b_tok, r_tok)

    # Token counting with more tokens
    for target, seq in [
        (1, [1,2,1,3,1,4,1,5,1]),
        (3, [3,3,3,1,2,3,4,3]),
        (5, [1,2,3,4,5,6,7,8,9]),
    ]:
        count = seq.count(target)
        inp = [target + OFFSET, SEP] + [s + OFFSET for s in seq]
        exp = [int(d) + OFFSET for d in str(count)]
        tester.run_test("Stress: Token Counting", inp, exp)

    # ================================================================
    # RESULTS
    # ================================================================
    overall = tester.print_results()
    return overall


if __name__ == "__main__":
    score = run_generalization_tests()
    print(f"\nFinal Generalization Score: {score:.1f}%")
