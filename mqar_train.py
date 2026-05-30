#!/usr/bin/env python3
"""MQAR (Multi-Query Associative Recall) — der eigentliche State-Test.

Sequenz: K Schluessel-Wert-Paare, dann Q Anfragen (Schluessel) -> Modell muss den
zugehoerigen Wert vorhersagen. Reine Recall-Aufgabe: ein Modell OHNE Gedaechtnis
(UT, temporal-Pfad aus) sollte scheitern, eines MIT State sollte es loesen.

Vergleicht die CISA-Varianten iso-compute:
    python mqar_train.py --mode ut    # kein State (UT-Baseline)
    python mqar_train.py --mode gru   # CISA wie gefixt (logbias)
    python mqar_train.py --mode v2    # NEXUS-2 (channel+nope+qk_norm+stabilize)
"""
import sys, os, argparse, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(line_buffering=True)
from nexus.lm.model import NexusLM, NexusLMConfig, CISAttention

N_KEYS, N_VALUES = 64, 64           # disjoint token ranges
VOCAB = N_KEYS + N_VALUES


def make_batch(B, n_pairs, n_queries, g, device):
    L = 2 * n_pairs + 2 * n_queries
    ids = torch.zeros(B, L, dtype=torch.long)
    labels = torch.full((B, L), -100, dtype=torch.long)
    for b in range(B):
        keys = torch.randperm(N_KEYS, generator=g)[:n_pairs]
        vals = torch.randint(0, N_VALUES, (n_pairs,), generator=g) + N_KEYS
        seq = []
        for i in range(n_pairs):
            seq += [keys[i].item(), vals[i].item()]
        qidx = torch.randint(0, n_pairs, (n_queries,), generator=g)
        for j in range(n_queries):
            qi = qidx[j].item()
            seq += [keys[qi].item(), vals[qi].item()]
        ids[b] = torch.tensor(seq)
        for j in range(n_queries):
            ans = 2 * n_pairs + 2 * j + 1     # answer (value) position
            labels[b, ans] = ids[b, ans]
    return ids.to(device), labels.to(device)


@torch.no_grad()
def accuracy(model, n_pairs, n_queries, g, device, n_batches=20, B=64):
    model.eval()
    correct = total = 0
    for _ in range(n_batches):
        ids, labels = make_batch(B, n_pairs, n_queries, g, device)
        logits = model(input_ids=ids)["logits"]
        mask = labels != -100                       # answer positions
        # logits[:, p-1] predicts token at p
        pred = logits[:, :-1].argmax(-1)
        tgt = ids[:, 1:]
        m = mask[:, 1:]
        correct += (pred[m] == tgt[m]).sum().item()
        total += m.sum().item()
    model.train()
    return correct / max(1, total)


def build_model(mode, n_iter, d_model, device):
    cfg = NexusLMConfig(vocab_size=VOCAB, d_model=d_model, n_heads=4, d_ff=d_model * 3,
                        n_iterations=n_iter, max_seq_len=256, dropout=0.0)
    if mode == "v2":
        cfg.gate_mode = "channel"; cfg.nope_temporal = True
        cfg.qk_norm = True; cfg.stabilize = True; cfg.deep_supervision = True
    model = NexusLM(cfg).to(device)
    if mode == "ut":                                # disable the state path entirely
        for m in model.modules():
            if isinstance(m, CISAttention):
                m.disable_temporal = True
    return model, cfg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["ut", "gru", "v2"], default="gru")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--n-iter", type=int, default=4)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--pairs", type=int, default=16)
    p.add_argument("--queries", type=int, default=4)
    p.add_argument("--bs", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    g = torch.Generator().manual_seed(0)
    model, cfg = build_model(args.mode, args.n_iter, args.d_model, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"MQAR  mode={args.mode}  params={n_params:,}  n_iter={cfg.n_iterations}  "
          f"d_model={cfg.d_model}  device={device}", flush=True)
    print(f"  task: {args.pairs} pairs, {args.queries} queries, vocab={VOCAB}\n", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    model.train()
    for step in range(1, args.steps + 1):
        ids, labels = make_batch(args.bs, args.pairs, args.queries, g, device)
        loss = model(input_ids=ids, labels=labels)["loss"]
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 200 == 0 or step == 1:
            acc = accuracy(model, args.pairs, args.queries, g, device)
            print(f"  step {step:>5} | loss {loss.item():.4f} | recall-acc {acc*100:5.1f}%", flush=True)

    acc = accuracy(model, args.pairs, args.queries, g, device, n_batches=50)
    print(f"\n  FINAL recall accuracy [{args.mode}]: {acc*100:.1f}%", flush=True)


if __name__ == "__main__":
    main()
