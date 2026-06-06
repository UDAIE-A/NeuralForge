#!/usr/bin/env python3
"""
NeuralForge Studio - FastAPI backend.

Serves a single-page UI with a ChatGPT-style chat on one side and a training
admin panel on the other, with live metrics over WebSocket.

Run:
    python -m webui.server
    # then open http://127.0.0.1:8000
"""

import os
import sys
import glob
import json
import time
import queue
import pickle
import asyncio
import threading
from typing import Optional

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from neuralforge.core import NeuralForge, ModelConfig
from neuralforge.tokenizer import BPETokenizer
from neuralforge.tokenizer.char_tokenizer import CharTokenizer
from neuralforge.training import Trainer, create_dataloaders
from neuralforge.training.trainer import get_gpu_stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

app = FastAPI(title="NeuralForge Studio")

# ----------------------------------------------------------------------------
# Shared training state (updated by the training thread, read by the metrics WS)
# ----------------------------------------------------------------------------
TRAIN = {
    "running": False,
    "status": "idle",
    "error": None,
    "params": {},
    "epoch": 0, "num_epochs": 0,
    "batch": 0, "total_batches": 0,
    "global_step": 0,
    "loss": None, "lr": None, "tokens_per_sec": 0,
    "eta": 0, "elapsed": 0,
    "best_val_loss": None,
    "loss_history": [],   # [{step, loss}]
}
TRAIN_LOCK = threading.Lock()
STOP_FLAG = {"stop": False}
TRAIN_THREAD: Optional[threading.Thread] = None

# Cache of loaded inference models: path -> (model, tokenizer)
MODEL_CACHE = {}
MODEL_LOCK = threading.Lock()


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def list_checkpoints():
    """Find .pt checkpoints under checkpoints/ and runs/."""
    found = []
    for base in ("checkpoints", "runs"):
        for p in glob.glob(os.path.join(ROOT, base, "**", "*.pt"), recursive=True):
            rel = os.path.relpath(p, ROOT).replace("\\", "/")
            found.append({
                "path": rel,
                "name": rel,
                "size_mb": round(os.path.getsize(p) / 1024 / 1024, 1),
                "mtime": os.path.getmtime(p),
            })
    found.sort(key=lambda d: d["mtime"], reverse=True)
    return found


def list_datafiles():
    files = []
    for p in sorted(glob.glob(os.path.join(ROOT, "data", "*.txt"))):
        rel = os.path.relpath(p, ROOT).replace("\\", "/")
        files.append({"path": rel, "name": os.path.basename(p),
                      "size_kb": round(os.path.getsize(p) / 1024)})
    return files


def load_tokenizer(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    if "char_to_id" in data:
        return CharTokenizer.load(path)
    return BPETokenizer.load(path)


def load_model_for_inference(ckpt_rel):
    """Load (and cache) a model + tokenizer for a checkpoint."""
    with MODEL_LOCK:
        if ckpt_rel in MODEL_CACHE:
            return MODEL_CACHE[ckpt_rel]

        ckpt_path = os.path.join(ROOT, ckpt_rel)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_rel}")

        device = torch.device("cuda")
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        config = checkpoint["config"]
        config.device = "cuda"

        tok_path = os.path.join(os.path.dirname(ckpt_path), "tokenizer.pkl")
        if not os.path.exists(tok_path):
            raise FileNotFoundError("tokenizer.pkl not found next to checkpoint")
        tokenizer = load_tokenizer(tok_path)

        model = NeuralForge(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device).eval()

        MODEL_CACHE[ckpt_rel] = (model, tokenizer)
        return model, tokenizer


def _reset_train_state(params, num_epochs):
    with TRAIN_LOCK:
        TRAIN.update({
            "running": True, "status": "starting", "error": None, "params": params,
            "epoch": 0, "num_epochs": num_epochs, "batch": 0, "total_batches": 0,
            "global_step": 0, "loss": None, "lr": None, "tokens_per_sec": 0,
            "eta": 0, "elapsed": 0, "best_val_loss": None, "loss_history": [],
        })


def _metrics_cb(m):
    with TRAIN_LOCK:
        TRAIN.update({
            "status": "training",
            "epoch": m["epoch"], "num_epochs": m["num_epochs"],
            "batch": m["batch"], "total_batches": m["total_batches"],
            "global_step": m["global_step"],
            "loss": round(m["loss"], 4), "lr": m["lr"],
            "tokens_per_sec": int(m["tokens_per_sec"]),
            "eta": m["eta"], "elapsed": m["elapsed"],
        })
        # Keep a light loss history for the chart (cap length).
        hist = TRAIN["loss_history"]
        hist.append({"step": m["global_step"], "loss": round(m["loss"], 4)})
        if len(hist) > 2000:
            del hist[: len(hist) - 2000]


def _training_worker(params):
    """Runs in a background thread."""
    try:
        STOP_FLAG["stop"] = False
        num_epochs = int(params["epochs"])
        _reset_train_state(params, num_epochs)

        data_rel = params["data"]
        data_path = os.path.join(ROOT, data_rel)
        with open(data_path, "r", encoding="utf-8") as f:
            text = f.read()

        ckpt_dir = os.path.join(ROOT, params.get("checkpoint_dir", "checkpoints"))
        os.makedirs(ckpt_dir, exist_ok=True)

        # Tokenizer
        with TRAIN_LOCK:
            TRAIN["status"] = "building tokenizer"
        if params.get("char"):
            tokenizer = CharTokenizer()
            tokenizer.train(text)
        else:
            tokenizer = BPETokenizer()
            tokenizer.train(text, vocab_size=int(params.get("vocab_size", 8000)))
        tokenizer.save(os.path.join(ckpt_dir, "tokenizer.pkl"))

        # Config + model
        config = ModelConfig.from_preset(params["preset"])
        config.max_seq_len = int(params["seq_len"])
        config.batch_size = int(params["batch_size"])
        config.learning_rate = float(params["lr"])
        config.vocab_size = len(tokenizer)
        config.device = "cuda"

        with TRAIN_LOCK:
            TRAIN["status"] = "building model"
        model = NeuralForge(config)

        train_loader, val_loader = create_dataloaders(
            data_path, None, tokenizer,
            seq_len=config.max_seq_len, batch_size=config.batch_size,
            num_workers=0,  # threads + Windows: keep it single-process
        )

        trainer = Trainer(
            model=model, config=config,
            train_loader=train_loader, val_loader=val_loader,
            checkpoint_dir=ckpt_dir,
            compile_model=False,  # avoid compile latency inside the server
            metrics_callback=_metrics_cb,
            should_stop=lambda: STOP_FLAG["stop"],
        )
        trainer.train(num_epochs=num_epochs)

        with TRAIN_LOCK:
            TRAIN["running"] = False
            TRAIN["status"] = "stopped" if STOP_FLAG["stop"] else "complete"
            TRAIN["best_val_loss"] = (
                round(trainer.best_val_loss, 4)
                if trainer.best_val_loss != float("inf") else None
            )
        # New checkpoints exist - drop inference cache so they show up fresh.
        with MODEL_LOCK:
            MODEL_CACHE.clear()
    except Exception as e:  # noqa
        import traceback
        traceback.print_exc()
        with TRAIN_LOCK:
            TRAIN["running"] = False
            TRAIN["status"] = "error"
            TRAIN["error"] = str(e)


# ----------------------------------------------------------------------------
# REST endpoints
# ----------------------------------------------------------------------------
@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC, "index.html"))


@app.get("/api/info")
def info():
    gpu = {}
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        gpu = {"name": props.name, "total_gb": round(props.total_memory / 1024**3, 1)}
    return {
        "cuda": torch.cuda.is_available(),
        "gpu": gpu,
        "presets": ["tiny", "small", "base", "large", "xl", "xxl"],
        "checkpoints": list_checkpoints(),
        "datafiles": list_datafiles(),
    }


@app.get("/api/checkpoints")
def checkpoints():
    return list_checkpoints()


@app.post("/api/train/start")
async def train_start(req: dict):
    global TRAIN_THREAD
    if TRAIN["running"]:
        return JSONResponse({"ok": False, "error": "Training already running"}, status_code=409)
    if not torch.cuda.is_available():
        return JSONResponse({"ok": False, "error": "CUDA not available"}, status_code=400)
    required = ["preset", "data", "epochs", "seq_len", "batch_size", "lr"]
    for r in required:
        if r not in req:
            return JSONResponse({"ok": False, "error": f"Missing field: {r}"}, status_code=400)
    TRAIN_THREAD = threading.Thread(target=_training_worker, args=(req,), daemon=True)
    TRAIN_THREAD.start()
    return {"ok": True}


@app.post("/api/train/stop")
def train_stop():
    if not TRAIN["running"]:
        return {"ok": False, "error": "No training running"}
    STOP_FLAG["stop"] = True
    return {"ok": True}


@app.get("/api/train/status")
def train_status():
    with TRAIN_LOCK:
        return dict(TRAIN)


# ----------------------------------------------------------------------------
# WebSockets
# ----------------------------------------------------------------------------
@app.websocket("/ws/metrics")
async def ws_metrics(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            gpu = get_gpu_stats()
            with TRAIN_LOCK:
                payload = dict(TRAIN)
            payload["gpu"] = gpu
            await ws.send_text(json.dumps(payload))
            await asyncio.sleep(0.4)
    except (WebSocketDisconnect, RuntimeError):
        return


@app.websocket("/ws/chat")
async def ws_chat(ws: WebSocket):
    await ws.accept()
    loop = asyncio.get_event_loop()
    try:
        while True:
            raw = await ws.receive_text()
            req = json.loads(raw)
            ckpt = req.get("checkpoint")
            prompt = req.get("prompt", "")
            opts = {
                "max_new_tokens": int(req.get("max_tokens", 200)),
                "temperature": float(req.get("temperature", 0.8)),
                "top_k": (int(req["top_k"]) if req.get("top_k") else None),
                "top_p": (float(req["top_p"]) if req.get("top_p") else None),
                "repetition_penalty": float(req.get("repetition_penalty", 1.0)),
            }
            if not ckpt:
                await ws.send_text(json.dumps({"type": "error", "message": "No model selected"}))
                continue
            try:
                model, tokenizer = load_model_for_inference(ckpt)
            except Exception as e:
                await ws.send_text(json.dumps({"type": "error", "message": str(e)}))
                continue

            await ws.send_text(json.dumps({"type": "start"}))

            # Run generation in a thread; stream decoded deltas via a queue.
            q: "queue.Queue" = queue.Queue()

            def worker():
                try:
                    ids = tokenizer.encode(prompt, add_special_tokens=False)
                    x = torch.tensor([ids], dtype=torch.long, device="cuda")
                    gen_ids = []
                    prev_text = ""
                    for tok in model.generate_stream(x, **opts):
                        gen_ids.append(tok)
                        text = tokenizer.decode(gen_ids)
                        delta = text[len(prev_text):]
                        prev_text = text
                        if delta:
                            q.put(("delta", delta))
                    q.put(("done", None))
                except Exception as e:  # noqa
                    q.put(("error", str(e)))

            threading.Thread(target=worker, daemon=True).start()

            while True:
                kind, val = await loop.run_in_executor(None, q.get)
                if kind == "delta":
                    await ws.send_text(json.dumps({"type": "token", "text": val}))
                elif kind == "done":
                    await ws.send_text(json.dumps({"type": "done"}))
                    break
                else:
                    await ws.send_text(json.dumps({"type": "error", "message": val}))
                    break
    except (WebSocketDisconnect, RuntimeError):
        return


# Serve static assets (if any are added later)
if os.path.isdir(STATIC):
    app.mount("/static", StaticFiles(directory=STATIC), name="static")


if __name__ == "__main__":
    import uvicorn
    print("\n  NeuralForge Studio -> http://127.0.0.1:8000\n")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
