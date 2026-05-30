#!/usr/bin/env python3
"""Diagnose: bekommen ALLE Parameter Gradienten? (Verdacht: GRU-State ist tot)"""
import sys, os, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nexus.lm.model import NexusLM, NexusLMConfig

torch.manual_seed(0)
cfg = NexusLMConfig.tiny()
cfg.vocab_size = 256
model = NexusLM(cfg)
model.train()

# Ein forward+backward auf Zufallsdaten
ids = torch.randint(0, cfg.vocab_size, (2, 32))
out = model(input_ids=ids, labels=ids.clone())
out["loss"].backward()

print(f"Loss: {out['loss'].item():.4f}\n")
print(f"{'Parameter':<45} {'grad-norm':>14}")
print("-" * 62)
dead = []
for name, p in model.named_parameters():
    if p.grad is None:
        g = None
        flag = "  <-- KEIN GRAD (None)"
        dead.append(name)
    else:
        g = p.grad.norm().item()
        flag = "  <-- NULL-GRAD" if g == 0.0 else ""
        if g == 0.0:
            dead.append(name)
    gs = "None" if g is None else f"{g:.3e}"
    print(f"{name:<45} {gs:>14}{flag}")

print("\n" + "=" * 62)
if dead:
    print(f"TOTE PARAMETER (kein Lernsignal): {len(dead)}")
    for d in dead:
        print(f"   - {d}")
else:
    print("Alle Parameter bekommen Gradienten. Alles gut.")
