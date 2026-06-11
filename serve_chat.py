#!/usr/bin/env python3
"""
NEXUS-LM — ChatGPT-style local web app.

Serves the trained 155M CISA chatbot (SFT) behind a streaming chat UI that
runs entirely on the local RTX 4060. No cloud, no API keys.

The model's built-in `generate()` has no repetition penalty (it loops) and no
KV cache, so this server implements its own incremental sampling loop with:
  - repetition penalty + no-repeat-ngram blocking (kills the "...is the capital
    of France and is the capital of France" loops)
  - temperature / top-k / top-p
  - ChatML multi-turn prompt assembly + clean stop at <|eot|> / <|user|>
  - token-by-token SSE streaming for a real ChatGPT feel

Usage:
    .venv\\Scripts\\python.exe serve_chat.py
    .venv\\Scripts\\python.exe serve_chat.py --model pod_backup/nexus_lm_xl_sft.pt --port 8000
    .venv\\Scripts\\python.exe serve_chat.py --cpu          # force CPU
"""

import os
import re
import sys
import json
import time
import shutil
import argparse
import threading
import subprocess
import webbrowser

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import asyncio
from urllib.parse import parse_qs

import torch
import torch.nn.functional as F
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse, Response
from starlette.concurrency import run_in_threadpool

from nexus.lm.model import NexusLM, NexusLMConfig
from nexus.lm.fast_tokenizer import FastTokenizer

# ----------------------------------------------------------------------------
# Globals (populated in load_model)
# ----------------------------------------------------------------------------
MODEL = None
TOK = None
CFG = None
DEVICE = "cpu"
INFO = {}
GEN_LOCK = asyncio.Lock()  # one generation at a time on the GPU (async-cancel-safe)

EOS_ID = 2
# These markers are normal (byte-level BPE) text, NOT special tokens, so we
# detect end-of-turn by scanning the decoded string.
STOP_STRINGS = ["<|eot|>", "<|user|>", "<|system|>", "<|assistant|>", "<|endoftext|>"]


# ----------------------------------------------------------------------------
# Model loading
# ----------------------------------------------------------------------------
def human(n):
    for unit, div in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if n >= div:
            return f"{n/div:.0f}{unit}" if n / div >= 100 else f"{n/div:.1f}{unit}"
    return str(n)


def load_model(model_path, tok_path, device):
    global MODEL, TOK, CFG, DEVICE, INFO
    DEVICE = device
    print(f"  Loading model:     {model_path}")
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    CFG = NexusLMConfig.from_dict(ckpt["config"])
    MODEL = NexusLM(CFG)
    MODEL.load_state_dict(ckpt["model_state_dict"])
    MODEL.to(device).eval()

    print(f"  Loading tokenizer: {tok_path}")
    TOK = FastTokenizer.load(tok_path)

    p = MODEL.count_parameters()
    val_loss = ckpt.get("best_val_loss", ckpt.get("val_loss"))
    val_ppl = f"{float(torch.exp(torch.tensor(val_loss))):.1f}" if val_loss else "?"
    dev_name = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"
    INFO = {
        "params": p["total_params"], "params_h": human(p["total_params"]),
        "effective": p["effective_params"], "effective_h": human(p["effective_params"]),
        "n_iter": CFG.n_iterations, "vocab": CFG.vocab_size,
        "ctx": CFG.max_seq_len, "val_ppl": val_ppl, "device": dev_name,
    }
    print(f"  Params:  {p['total_params']:,} real / {p['effective_params']:,} effective")
    print(f"  CISA:    n_iter={CFG.n_iterations} | ctx={CFG.max_seq_len} | vocab={CFG.vocab_size}")
    print(f"  Val-PPL: {val_ppl} | device: {dev_name}")
    if device == "cuda":
        warm = torch.tensor([[1, 100, 200]], device=device)
        with torch.no_grad():
            MODEL(warm)
        torch.cuda.synchronize()
    print("  Model ready.\n")


# ----------------------------------------------------------------------------
# Prompt assembly (ChatML — must match prepare_chat.py)
# ----------------------------------------------------------------------------
def build_prompt(messages, system=None):
    parts = []
    if system:
        parts.append(f"<|system|>\n{system}\n")
    for m in messages:
        role, content = m.get("role"), (m.get("content") or "")
        if role == "user":
            parts.append(f"<|user|>\n{content}\n")
        elif role == "assistant":
            parts.append(f"<|assistant|>\n{content}<|eot|>\n")
    parts.append("<|assistant|>\n")  # cue the model to answer
    return "".join(parts)


# ----------------------------------------------------------------------------
# Sampling helpers
# ----------------------------------------------------------------------------
def apply_repetition_penalty(logits, seq_ids, penalty):
    if penalty == 1.0:
        return
    ids = torch.unique(seq_ids)
    vals = logits[0, ids]
    logits[0, ids] = torch.where(vals < 0, vals * penalty, vals / penalty)


def banned_ngram_tokens(seq, n):
    """Token ids that would complete a previously-seen n-gram given the tail."""
    if n <= 0 or len(seq) < n:
        return []
    prefix = tuple(seq[-(n - 1):]) if n > 1 else ()
    banned = []
    for i in range(len(seq) - n + 1):
        if tuple(seq[i:i + n - 1]) == prefix:
            banned.append(seq[i + n - 1])
    return banned


def _stop_prefix_tail(text):
    """Length of the longest suffix of text that is a proper prefix of a stop
    string — that many chars must be held back, they may still become a stop
    marker once the next token arrives."""
    longest = 0
    for s in STOP_STRINGS:
        for k in range(min(len(s) - 1, len(text)), longest, -1):
            if text.endswith(s[:k]):
                longest = k
                break
    return longest


@torch.no_grad()
def generate_events(prompt_ids, *, max_new_tokens, temperature, top_k, top_p,
                    repetition_penalty, no_repeat_ngram):
    """Incremental sampling loop. Yields {'delta': str} per chunk and a final
    {'done': True, 'tokens': n, 'elapsed': s}."""
    ids = prompt_ids.clone()
    gen_ids = []
    emitted = ""
    text = ""
    stopped = False
    t0 = time.time()

    for _ in range(max_new_tokens):
        x = ids[:, -CFG.max_seq_len:]
        logits = MODEL(x)["logits"][:, -1, :].float()  # (1, V)

        apply_repetition_penalty(logits, ids[0], repetition_penalty)
        if no_repeat_ngram and len(ids[0]) >= no_repeat_ngram:
            ban = banned_ngram_tokens(ids[0].tolist(), int(no_repeat_ngram))
            if ban:
                logits[0, ban] = float("-inf")

        if temperature <= 0:
            next_id = int(logits.argmax(dim=-1))
        else:
            logits = logits / temperature
            if top_k and top_k > 0:
                k = min(int(top_k), logits.size(-1))
                kth = torch.topk(logits, k)[0][:, -1:]
                logits[logits < kth] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            if top_p and top_p < 1.0:
                sp, si = torch.sort(probs, descending=True)
                cum = torch.cumsum(sp, dim=-1)
                remove = (cum - sp) >= top_p
                sp[remove] = 0.0
                probs = torch.zeros_like(probs).scatter(1, si, sp)
                probs = probs / probs.sum(dim=-1, keepdim=True)
            next_id = int(torch.multinomial(probs, num_samples=1))

        if next_id == EOS_ID:
            break

        ids = torch.cat([ids, torch.tensor([[next_id]], device=ids.device)], dim=1)
        gen_ids.append(next_id)

        # Decode the full generated run each step (byte-level decoder buffers
        # partial UTF-8 safely) and diff against what we've already sent.
        text = TOK.decode(gen_ids)

        stop_at = -1
        for s in STOP_STRINGS:
            p = text.find(s)
            if p != -1:
                stop_at = p if stop_at == -1 else min(stop_at, p)
        if stop_at != -1:
            final = text[:stop_at]
            if len(final) > len(emitted):
                yield {"delta": final[len(emitted):]}
            stopped = True
            break

        # hold back a tail that could still become a stop marker, so a marker
        # split across tokens is never partially emitted
        safe = len(text) - _stop_prefix_tail(text)
        if safe > len(emitted):
            yield {"delta": text[len(emitted):safe]}
            emitted = text[:safe]

    if not stopped and len(text) > len(emitted):
        yield {"delta": text[len(emitted):]}  # flush held-back tail

    yield {"done": True, "tokens": len(gen_ids), "elapsed": round(time.time() - t0, 3)}


# ----------------------------------------------------------------------------
# FastAPI app
# ----------------------------------------------------------------------------
app = FastAPI(title="NEXUS-LM Chat")

TOKEN = None  # optional shared secret (--token); gates all routes when set
PUBLIC_PATHS = ("/apple-touch-icon.png", "/favicon.ico")


class TokenGate:
    """Pure ASGI middleware — deliberately NOT @app.middleware("http"):
    Starlette's BaseHTTPMiddleware proxies receive/send and breaks
    request.is_disconnected() for streaming responses, so a stopped generation
    left GEN_LOCK held forever (stop -> regenerate deadlock)."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not TOKEN or scope["path"] in PUBLIC_PATHS:
            return await self.app(scope, receive, send)

        qs = parse_qs(scope.get("query_string", b"").decode("latin-1"))
        from_query = (qs.get("key") or [None])[0]
        cookie = next((v.decode("latin-1") for k, v in scope.get("headers", []) if k == b"cookie"), "")
        m = re.search(r"(?:^|;\s*)nexus_key=([^;]+)", cookie)
        from_cookie = m.group(1) if m else None

        if TOKEN not in (from_query, from_cookie):
            body = json.dumps({"error": "unauthorized — open /?key=<token>"}).encode()
            await send({"type": "http.response.start", "status": 401,
                        "headers": [(b"content-type", b"application/json")]})
            await send({"type": "http.response.body", "body": body})
            return

        wrapped = send
        if from_query == TOKEN and from_cookie != TOKEN:
            # remember the key so the UI's fetch() calls work without ?key=
            async def wrapped(message, _send=send):
                if message["type"] == "http.response.start":
                    message.setdefault("headers", []).append(
                        (b"set-cookie",
                         f"nexus_key={TOKEN}; Max-Age=15552000; Path=/; SameSite=Lax".encode()))
                await _send(message)

        await self.app(scope, receive, wrapped)


app.add_middleware(TokenGate)


@app.get("/")
def index():
    return FileResponse(os.path.join(HERE, "webchat", "index.html"))


def _icon_png(size=180):
    """Home-screen icon (accent square, white 'N') as a tiny pure-python PNG."""
    import zlib
    import struct
    bg, fg = (16, 163, 127), (255, 255, 255)
    s = size
    x1, x2, y1, y2 = .30 * s, .70 * s, .28 * s, .72 * s   # N geometry
    half, dhalf = .07 * s, .085 * s                       # stroke half-widths
    raw = bytearray()
    for y in range(s):
        raw += b"\x00"  # PNG filter byte per scanline
        for x in range(s):
            on_n = y1 <= y <= y2 and (
                abs(x - x1) <= half or abs(x - x2) <= half
                or abs(x - (x1 + (y - y1) / (y2 - y1) * (x2 - x1))) <= dhalf)
            raw += bytes(fg if on_n else bg)

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))

    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", s, s, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(bytes(raw)))
            + chunk(b"IEND", b""))


_ICON = None


@app.get("/apple-touch-icon.png")
@app.get("/favicon.ico")
def icon():
    global _ICON
    if _ICON is None:
        _ICON = _icon_png()
    return Response(content=_ICON, media_type="image/png")


@app.get("/api/info")
def info():
    return JSONResponse(INFO)


@app.post("/api/chat")
async def chat(req: Request):
    body = await req.json()
    messages = body.get("messages", [])
    system = body.get("system") or None
    params = dict(
        max_new_tokens=int(body.get("max_new_tokens", 256)),
        temperature=float(body.get("temperature", 0.7)),
        top_k=int(body.get("top_k", 50)),
        top_p=float(body.get("top_p", 0.9)),
        repetition_penalty=float(body.get("repetition_penalty", 1.3)),
        no_repeat_ngram=int(body.get("no_repeat_ngram", 3)),
    )

    prompt = build_prompt(messages, system)
    ids = torch.tensor([TOK.encode(prompt, add_bos=True)], dtype=torch.long, device=DEVICE)

    async def event_stream():
        # Serialize GPU access with an async lock. We step the (blocking) sync
        # token generator one token at a time in a threadpool, and between tokens
        # check whether the client went away (Stop button / tab close). On
        # disconnect or cancellation the `finally` closes the generator and
        # releases the lock -> no permanent deadlock (the old threading.Lock
        # held across a `yield` never got released on disconnect).
        # Manual acquire with timeout: even if the lock ever leaks again, the
        # client gets an error instead of hanging forever.
        try:
            await asyncio.wait_for(GEN_LOCK.acquire(), timeout=120)
        except asyncio.TimeoutError:
            yield f"data: {json.dumps({'error': 'Server beschäftigt — bitte erneut versuchen.'})}\n\n"
            return
        gen = generate_events(ids, **params)
        try:
            while True:
                if await req.is_disconnected():
                    break
                try:
                    # next(gen, None): StopIteration must not cross the thread
                    # boundary (becomes "coroutine raised StopIteration")
                    ev = await run_in_threadpool(next, gen, None)
                except Exception as e:  # surface model errors to the UI
                    yield f"data: {json.dumps({'error': str(e)})}\n\n"
                    break
                if ev is None:
                    break
                yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
        finally:
            try:
                gen.close()
            except Exception:
                pass  # generator may still be mid-step in the threadpool
            GEN_LOCK.release()

    return StreamingResponse(event_stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ----------------------------------------------------------------------------
# Cloudflare quick tunnel (public https URL, works from any network)
# ----------------------------------------------------------------------------
def start_tunnel(port):
    candidates = [shutil.which("cloudflared"),
                  os.path.join(os.environ.get("LOCALAPPDATA", ""), "cloudflared", "cloudflared.exe")]
    exe = next((c for c in candidates if c and os.path.exists(c)), None)
    if not exe:
        print("  ERROR: cloudflared.exe not found (PATH or %LOCALAPPDATA%\\cloudflared\\) — no tunnel.")
        return None
    proc = subprocess.Popen(
        [exe, "tunnel", "--url", f"http://127.0.0.1:{port}", "--no-autoupdate"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace")

    def pump():
        for line in proc.stdout:
            m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", line)
            if m:
                extra = f"/?key={TOKEN}" if TOKEN else ""
                print(f"\n  PUBLIC URL (iPhone etc.):  {m.group(0)}{extra}\n", flush=True)

    threading.Thread(target=pump, daemon=True).start()
    return proc


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def find_model():
    for c in ["pod_backup/nexus_lm_xl_sft.pt", "nexus_lm_xl_sft.pt",
              "nexus_lm_xl_web_qknorm_stab.pt", "nexus_lm_base.pt"]:
        if os.path.exists(os.path.join(HERE, c)):
            return os.path.join(HERE, c)
    return None


def find_tokenizer():
    for c in ["pod_backup/tokenizer", "lm_data_web/tokenizer", "lm_data/tokenizer"]:
        p = os.path.join(HERE, c)
        if FastTokenizer.exists(p):
            return p
    return os.path.join(HERE, "pod_backup/tokenizer")


def main():
    ap = argparse.ArgumentParser(description="NEXUS-LM local chat web app")
    ap.add_argument("--model", default=None, help="path to .pt checkpoint")
    ap.add_argument("--tokenizer", default=None, help="dir with fast_tokenizer.json")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--cpu", action="store_true", help="force CPU")
    ap.add_argument("--no-open", action="store_true", help="don't auto-open the browser")
    ap.add_argument("--tunnel", action="store_true",
                    help="expose via Cloudflare quick tunnel (public https URL)")
    ap.add_argument("--token", default=None,
                    help="require this key (?key=... once, then cookie) on all routes")
    args = ap.parse_args()

    global TOKEN
    TOKEN = args.token or None

    model_path = args.model or find_model()
    if not model_path or not os.path.exists(model_path):
        print("  ERROR: no model checkpoint found.")
        print("  Expected pod_backup/nexus_lm_xl_sft.pt — pass --model <path>.")
        sys.exit(1)
    tok_path = args.tokenizer or find_tokenizer()
    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("  NEXUS-LM — local ChatGPT-style web app")
    print("=" * 60)
    load_model(model_path, tok_path, device)

    url = f"http://{args.host}:{args.port}"
    print(f"  Open in your browser:  {url}\n")
    if not args.no_open:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()

    tunnel = start_tunnel(args.port) if args.tunnel else None
    try:
        import uvicorn
        uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    finally:
        if tunnel:
            tunnel.terminate()


if __name__ == "__main__":
    main()
