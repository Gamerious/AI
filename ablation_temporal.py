#!/usr/bin/env python3
"""0-cost ablation: does the cross-iteration temporal/state path contribute anything?

Loads a trained checkpoint, measures val loss/PPL once normally and once with the
temporal path forced off (out = spatial_out). A tiny delta => the state path is
dead weight on this data (confirms the critic's prediction).
"""
import sys, os, math, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(line_buffering=True)

from torch.utils.data import DataLoader, TensorDataset
from nexus.lm.model import NexusLM, NexusLMConfig, CISAttention

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lm_data")
CKPT = sys.argv[1] if len(sys.argv) > 1 else "nexus_lm_small_fixed.pt"
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- load model ---
ckpt = torch.load(CKPT, map_location=device, weights_only=False)
cfg = NexusLMConfig.from_dict(ckpt["config"])
model = NexusLM(cfg).to(device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
print(f"Loaded {CKPT}  (best_val_loss={ckpt.get('best_val_loss','?')})  device={device}", flush=True)
print(f"  config: d_model={cfg.d_model} n_iter={cfg.n_iterations} gate_init={getattr(cfg,'temporal_gate_init','?')}", flush=True)

# --- learned gate value(s) ---
for m in model.modules():
    if isinstance(m, CISAttention):
        g = torch.sigmoid(m.temporal_gate.detach()).float()
        print(f"  gelerntes temporal_gate: logit={m.temporal_gate.item():.4f} -> sigmoid={g.mean().item():.4f} "
              f"(= {g.mean().item()*100:.1f}% temporal-Anteil)", flush=True)

# --- val data ---
val_raw = torch.load(os.path.join(DATA_DIR, "val.pt"), map_location="cpu", weights_only=False)
val_loader = DataLoader(TensorDataset(val_raw["input_ids"]), batch_size=16, shuffle=False, drop_last=True)
n_batches = len(val_loader)
print(f"  Val: {len(val_raw['input_ids']):,} seqs, {n_batches} batches\n", flush=True)


@torch.no_grad()
def eval_ppl():
    total, n = 0.0, 0
    for (ids,) in val_loader:
        ids = ids.to(device)
        with torch.amp.autocast("cuda", enabled=(device == "cuda")):
            out = model(input_ids=ids, labels=ids.clone())
        total += out["loss"].item()
        n += 1
    loss = total / max(1, n)
    return loss, math.exp(min(loss, 20))


def set_temporal(enabled: bool):
    for m in model.modules():
        if isinstance(m, CISAttention):
            m.disable_temporal = not enabled


print("Messe... (voller Val-Set, beide Varianten)", flush=True)
set_temporal(True)
loss_on, ppl_on = eval_ppl()
print(f"  [temporal AN ]  Val Loss {loss_on:.5f}  |  PPL {ppl_on:.4f}", flush=True)

set_temporal(False)
loss_off, ppl_off = eval_ppl()
print(f"  [temporal AUS]  Val Loss {loss_off:.5f}  |  PPL {ppl_off:.4f}", flush=True)

d_loss = loss_off - loss_on
d_pct = (ppl_off - ppl_on) / ppl_on * 100
print(f"\n{'='*60}")
print(f"  DELTA durch Abschalten des State-Pfads:")
print(f"    Loss:  {d_loss:+.5f}")
print(f"    PPL:   {ppl_off-ppl_on:+.4f}  ({d_pct:+.2f}%)")
print(f"{'='*60}")
if abs(d_pct) < 0.5:
    print("  -> WINZIG (<0.5%): der State-Pfad traegt praktisch nichts bei.")
    print("     CISAs Kernidee ist auf diesen Daten redundant (Kritiker bestaetigt).")
else:
    print("  -> SPUERBAR: der State-Pfad traegt doch etwas bei -> A-Fixes lohnen sich.")
