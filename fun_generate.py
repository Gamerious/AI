#!/usr/bin/env python3
"""Fun generation samples from final NEXUS-LM."""
import sys, os, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(line_buffering=True)

from nexus.lm.model import NexusLM, NexusLMConfig
from nexus.lm.tokenizer import BPETokenizer

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lm_data")
CKPT = "nexus_lm_base.pt"

device = "cuda" if torch.cuda.is_available() else "cpu"
tok = BPETokenizer.load(os.path.join(DATA_DIR, "tokenizer"))
ckpt = torch.load(CKPT, map_location=device, weights_only=False)
cfg = NexusLMConfig(**ckpt["config"])
model = NexusLM(cfg).to(device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
print(f"Loaded {CKPT}  (best_val_loss={ckpt.get('best_val_loss','?'):.4f})\n", flush=True)

prompts = [
    # Klassisch TinyStories
    ("Klassisch", "Once upon a time, there was a dragon", 0.8),
    # Was passiert mit aktuellen Themen?
    ("Tech", "Tim opened his laptop and", 0.8),
    # Negative Emotion
    ("Sad", "The little boy was very sad because", 0.8),
    # Spannung / Mystery
    ("Mystery", "It was a dark and stormy night. Suddenly,", 0.8),
    # Dialog-Setup
    ("Dialog", '"I have a secret," said Lily. "What is it?" asked Tom.', 0.8),
    # Conflict/Resolution
    ("Conflict", "The two friends had a big fight, but then", 0.8),
    # Edge case: nonsense
    ("Nonsense", "The purple banana was learning to fly because", 0.9),
    # Längere Story testen
    ("Long", "Sue lost her favorite toy. She looked everywhere", 0.8),
    # Sehr kurzer Prompt
    ("Short", "Hello", 0.8),
    # Greedy decode (Temp 0)
    ("Greedy", "Once upon a time, a wise old owl", 0.0),
    # Hohe Kreativität
    ("Creative", "Once upon a time", 1.2),
]

@torch.no_grad()
def gen(prompt, temp=0.8, n=150):
    ids = torch.tensor([tok.encode(prompt, add_bos=True)], dtype=torch.long, device=device)
    out = model.generate(
        ids, max_new_tokens=n,
        temperature=temp if temp > 0 else 1.0,
        top_k=50 if temp > 0 else 1,
        top_p=0.9 if temp > 0 else 1.0,
    )
    return tok.decode(out[0].tolist())

for label, p, t in prompts:
    print(f"=== [{label}] (temp={t}) Prompt: {p!r}", flush=True)
    print(gen(p, temp=t), flush=True)
    print(flush=True)
