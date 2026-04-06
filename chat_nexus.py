#!/usr/bin/env python3
"""
NEXUS-LM Interactive Chat / Text Generation

Usage:
    python3 chat_nexus.py                              # Auto-detect model
    python3 chat_nexus.py --model nexus_lm_base.pt     # Specific model
    python3 chat_nexus.py --temperature 1.0 --top-k 100  # Adjust sampling

Commands during chat:
    /temp <value>     - Change temperature (0=greedy, 1.0=creative)
    /topk <value>     - Change top-k
    /topp <value>     - Change top-p
    /max <value>      - Change max tokens
    /reset            - Clear context
    /quit             - Exit
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from nexus.lm.model import NexusLM, NexusLMConfig
from nexus.lm.tokenizer import BPETokenizer


def load_model(model_path, device="auto"):
    """Load NEXUS-LM model and tokenizer."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"  Loading model: {model_path}")
    print(f"  Device: {device}")

    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    config = NexusLMConfig.from_dict(ckpt["config"])
    model = NexusLM(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    params = model.count_parameters()
    print(f"  Params: {params['total_params']:,} real / {params['effective_params']:,} effective")
    print(f"  CISA iterations: {config.n_iterations}")

    tokenizer_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lm_data", "tokenizer")
    tokenizer = BPETokenizer.load(tokenizer_dir)
    print(f"  Vocab: {tokenizer.vocab_size} tokens")

    return model, tokenizer, config, device


def generate_text(model, tokenizer, prompt, max_tokens=200, temperature=0.8,
                  top_k=50, top_p=0.9, device="cpu"):
    """Generate text from prompt."""
    input_ids = torch.tensor(
        [tokenizer.encode(prompt, add_bos=True)],
        dtype=torch.long, device=device
    )

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

    full_text = tokenizer.decode(output_ids[0].tolist())
    # Return only the generated part
    generated = full_text[len(prompt):]
    return generated


def find_model():
    """Auto-detect available model files."""
    candidates = [
        "nexus_lm_base.pt",
        "nexus_lm_small.pt",
        "nexus_lm_large.pt",
        "lm_checkpoints/best.pt",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def main():
    parser = argparse.ArgumentParser(description="NEXUS-LM Interactive Generation")
    parser.add_argument("--model", type=str, default=None, help="Model path")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    model_path = args.model or find_model()
    if not model_path or not os.path.exists(model_path):
        print("  No trained model found!")
        print("  Train one first:")
        print("    python3 prepare_lm_data.py")
        print("    python3 train_nexus_lm.py")
        sys.exit(1)

    device = "cpu" if args.cpu else "auto"
    model, tokenizer, config, device = load_model(model_path, device)

    # Settings
    temperature = args.temperature
    top_k = args.top_k
    top_p = args.top_p
    max_tokens = args.max_tokens

    print(f"\n{'='*60}")
    print(f"  NEXUS-LM Interactive Generation")
    print(f"  Temperature: {temperature} | Top-k: {top_k} | Top-p: {top_p}")
    print(f"  Max tokens: {max_tokens}")
    print(f"  Commands: /temp /topk /topp /max /reset /quit")
    print(f"{'='*60}\n")

    while True:
        try:
            prompt = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not prompt:
            continue

        # Handle commands
        if prompt.startswith("/"):
            parts = prompt.split()
            cmd = parts[0].lower()

            if cmd == "/quit":
                print("Bye!")
                break
            elif cmd == "/temp" and len(parts) > 1:
                temperature = float(parts[1])
                print(f"  Temperature → {temperature}")
            elif cmd == "/topk" and len(parts) > 1:
                top_k = int(parts[1])
                print(f"  Top-k → {top_k}")
            elif cmd == "/topp" and len(parts) > 1:
                top_p = float(parts[1])
                print(f"  Top-p → {top_p}")
            elif cmd == "/max" and len(parts) > 1:
                max_tokens = int(parts[1])
                print(f"  Max tokens → {max_tokens}")
            elif cmd == "/reset":
                print("  Context reset.")
            else:
                print(f"  Unknown command: {cmd}")
            continue

        # Generate
        print(f"\nNEXUS> ", end="", flush=True)
        generated = generate_text(
            model, tokenizer, prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            device=device,
        )
        print(generated)
        print()


if __name__ == "__main__":
    main()
