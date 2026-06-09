#!/usr/bin/env python3
"""Quick generation test for NEXUS-LM best checkpoint."""
import sys, os, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(line_buffering=True)

from nexus.lm.model import NexusLM, NexusLMConfig
from nexus.lm.tokenizer import BPETokenizer

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lm_data")
CKPT = "lm_checkpoints/best.pt"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}", flush=True)

tok = BPETokenizer.load(os.path.join(DATA_DIR, "tokenizer"))
ckpt = torch.load(CKPT, map_location=device, weights_only=False)
cfg = NexusLMConfig.from_dict(ckpt["config"])
model = NexusLM(cfg).to(device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
print(f"Loaded best.pt (step {ckpt.get('step','?')}, val_loss {ckpt.get('val_loss','?'):.4f})\n", flush=True)

prompts = [
    "Once upon a time",
    "The little cat",
    "Lily and Tom found a",
    "In the dark forest,",
    "The brave knight",
    "Tim wanted to bake a cake, so",
    "Mom said, \"You can't go outside because",
    "When the rain stopped,",
]

@torch.no_grad()
def gen(prompt, temp=0.8, top_k=50, top_p=0.9, n=120):
    ids = torch.tensor([tok.encode(prompt, add_bos=True)], dtype=torch.long, device=device)
    out = model.generate(ids, max_new_tokens=n, temperature=temp, top_k=top_k, top_p=top_p)
    return tok.decode(out[0].tolist())

for p in prompts:
    print(f"--- Prompt: {p!r}", flush=True)
    print(gen(p), flush=True)
    print(flush=True)
