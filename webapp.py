#!/usr/bin/env python3
"""NEXUS-LM TinyStories Web Playground."""
import sys, os, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nexus.lm.model import NexusLM, NexusLMConfig
from nexus.lm.tokenizer import BPETokenizer
import gradio as gr

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lm_data")
CKPT = "nexus_lm_base.pt"

print("Loading model...", flush=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
tok = BPETokenizer.load(os.path.join(DATA_DIR, "tokenizer"))
ckpt = torch.load(CKPT, map_location=device, weights_only=False)
cfg = NexusLMConfig(**ckpt["config"])
model = NexusLM(cfg).to(device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
print(f"Loaded {CKPT}  (val_loss={ckpt.get('best_val_loss','?'):.4f})  device={device}", flush=True)


@torch.no_grad()
def generate(prompt, max_new_tokens, temperature, top_k, top_p):
    if not prompt.strip():
        return "(Bitte einen Prompt eingeben)"
    ids = torch.tensor(
        [tok.encode(prompt, add_bos=True)], dtype=torch.long, device=device
    )
    if temperature <= 0.001:
        out = model.generate(ids, max_new_tokens=int(max_new_tokens),
                             temperature=1.0, top_k=1, top_p=1.0)
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

with gr.Blocks(title="NEXUS-LM TinyStories Playground") as demo:
    gr.Markdown("# NEXUS-LM TinyStories Playground")
    gr.Markdown(
        "37M-Parameter Story-Generator (CISA-Architektur, trainiert auf TinyStories). "
        "Gib einen Story-Anfang ein und lass das Modell weitererzählen."
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

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)
