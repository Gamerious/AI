#!/usr/bin/env python3
"""NEXUS-LM TinyStories Web Playground.

Vom Laptop starten, am Handy bedienen:

    python webapp.py --share                       # Auto-detect Checkpoint, oeffentlicher Link
    python webapp.py --ckpt lm_checkpoints/best.pt --share
    python webapp.py --share --password geheim123   # Link mit Passwortschutz

--share erzeugt eine https://xxxx.gradio.live URL, die du direkt im
iPhone-Browser oeffnen kannst. Das Modell rechnet auf dem Laptop (GPU),
das Handy ist nur die Fernbedienung.
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from nexus.lm.model import NexusLM, NexusLMConfig
from nexus.lm.tokenizer import BPETokenizer
import gradio as gr

HERE = os.path.dirname(os.path.abspath(__file__))


def find_model():
    """Auto-detect available checkpoint (same order as chat_nexus.py)."""
    candidates = [
        "nexus_lm_base.pt", "nexus_lm_small.pt", "nexus_lm_large.pt",
        "lm_checkpoints/best.pt",
    ]
    for c in candidates:
        if os.path.exists(os.path.join(HERE, c)):
            return os.path.join(HERE, c)
    return None


def main():
    ap = argparse.ArgumentParser(description="NEXUS-LM Web Playground")
    ap.add_argument("--ckpt", type=str, default=None,
                    help="Checkpoint-Pfad (Default: auto-detect)")
    ap.add_argument("--tokenizer", type=str, default=None,
                    help="Tokenizer-Verzeichnis (Default: lm_data/tokenizer)")
    ap.add_argument("--share", action="store_true",
                    help="Oeffentlichen gradio.live-Link erzeugen (fuers Handy)")
    ap.add_argument("--password", type=str, default=None,
                    help="Optionaler Passwortschutz fuer den Share-Link")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--cpu", action="store_true", help="CPU erzwingen")
    args = ap.parse_args()

    ckpt_path = args.ckpt or find_model()
    if not ckpt_path or not os.path.exists(ckpt_path):
        sys.exit("Kein Checkpoint gefunden. Mit --ckpt <pfad> angeben oder erst trainieren.")

    tok_dir = args.tokenizer or os.path.join(HERE, "lm_data", "tokenizer")
    if not os.path.exists(tok_dir):
        sys.exit(f"Tokenizer nicht gefunden: {tok_dir}  (mit --tokenizer setzen)")

    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading model: {ckpt_path}", flush=True)
    tok = BPETokenizer.load(tok_dir)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = NexusLMConfig.from_dict(ckpt["config"])
    model = NexusLM(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    params = model.count_parameters()
    val = ckpt.get("best_val_loss", None)
    val_str = f"{val:.4f}" if isinstance(val, (int, float)) else "?"
    print(f"Loaded ({params['total_params']:,} real / {params['effective_params']:,} eff, "
          f"{cfg.n_iterations} iters, val_loss={val_str}, device={device})", flush=True)

    @torch.no_grad()
    def generate(prompt, max_new_tokens, temperature, top_k, top_p):
        if not prompt.strip():
            return "(Bitte einen Prompt eingeben)"
        ids = torch.tensor(
            [tok.encode(prompt, add_bos=True)], dtype=torch.long, device=device
        )
        if temperature <= 0.001:
            out = model.generate(ids, max_new_tokens=int(max_new_tokens),
                                 temperature=0.0, top_k=1, top_p=1.0)
        else:
            out = model.generate(ids, max_new_tokens=int(max_new_tokens),
                                 temperature=float(temperature),
                                 top_k=int(top_k), top_p=float(top_p))
        return tok.decode(out[0].tolist())

    EXAMPLES = [
        ["Once upon a time, there was a little dragon", 200, 0.8, 50, 0.9],
        ["The little cat was feeling", 200, 0.8, 50, 0.9],
        ['"I have a secret," whispered Lily.', 200, 0.8, 50, 0.9],
        ["Sue lost her favorite toy. She looked everywhere", 250, 0.8, 50, 0.9],
        ["It was a dark and stormy night.", 200, 0.9, 50, 0.9],
        ["One day, a brave knight", 200, 0.8, 50, 0.9],
    ]

    pm = params['total_params'] / 1e6
    em = params['effective_params'] / 1e6
    with gr.Blocks(title="NEXUS-LM Playground") as demo:
        gr.Markdown("# NEXUS-LM Playground")
        gr.Markdown(
            f"{pm:.0f}M-Parameter Story-Generator ({em:.0f}M effektiv, CISA-Architektur, "
            f"{cfg.n_iterations} Iterationen). Gib einen Story-Anfang ein und lass das Modell weitererzaehlen."
        )

        with gr.Row():
            with gr.Column(scale=2):
                prompt = gr.Textbox(label="Prompt", value="Once upon a time", lines=3)
                with gr.Row():
                    temperature = gr.Slider(0.0, 1.5, value=0.8, step=0.05,
                                            label="Temperature (0 = greedy)")
                    max_new_tokens = gr.Slider(20, 400, value=200, step=10,
                                               label="Max neue Tokens")
                with gr.Row():
                    top_k = gr.Slider(1, 200, value=50, step=1, label="Top-k")
                    top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Top-p")
                btn = gr.Button("Generieren", variant="primary")

            with gr.Column(scale=3):
                out = gr.Textbox(label="Generierte Story", lines=15)

        gr.Examples(
            examples=EXAMPLES,
            inputs=[prompt, max_new_tokens, temperature, top_k, top_p],
        )

        btn.click(generate, [prompt, max_new_tokens, temperature, top_k, top_p], out)

    auth = ("nexus", args.password) if args.password else None
    demo.launch(
        server_name="0.0.0.0" if args.share else "127.0.0.1",
        server_port=args.port,
        share=args.share,
        auth=auth,
        inbrowser=not args.share,
    )


if __name__ == "__main__":
    main()
