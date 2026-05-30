#!/usr/bin/env python3
"""Test 0: bekommt das temporal_gate im channel-Modus (A1) echtes Gradientensignal,
im Vergleich zum alten logbias-Modus (quasi 0)?"""
import sys, os, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nexus.lm.model import NexusLM, NexusLMConfig, CISAttention

def gate_grad(gate_mode, nope_temporal=False):
    torch.manual_seed(0)
    cfg = NexusLMConfig.tiny()
    cfg.vocab_size = 256
    cfg.gate_mode = gate_mode
    cfg.nope_temporal = nope_temporal
    m = NexusLM(cfg); m.train()
    ids = torch.randint(0, cfg.vocab_size, (2, 32))
    m(input_ids=ids, labels=ids.clone())["loss"].backward()
    for mod in m.modules():
        if isinstance(mod, CISAttention):
            g = mod.temporal_gate.grad
            return g.abs().mean().item(), tuple(mod.temporal_gate.shape)

g_log, sh_log = gate_grad("logbias")
g_ch,  sh_ch  = gate_grad("channel", nope_temporal=True)

print(f"  logbias (alt):   gate{list(sh_log)}  mean|grad| = {g_log:.3e}")
print(f"  channel (A1+A2): gate{list(sh_ch)}   mean|grad| = {g_ch:.3e}")
ratio = g_ch / max(g_log, 1e-12)
print(f"\n  Verhaeltnis channel/logbias: {ratio:,.0f}x")
print("  -> A1 greift: Gate bekommt echtes Signal." if ratio > 100
      else "  -> kaum Unterschied, A1 pruefen.")
