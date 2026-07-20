#!/usr/bin/env python3
"""
NeuralForge Auto-Tutor Bridge -- the live wire between the agent and the model.

Runs an autonomous teach-loop and broadcasts every step over a pub/sub bus that
the web UI subscribes to:

    probe   -> we sent "hi" and the model replied; here is what it said
    teach   -> reply was gibberish/off-target, so we taught the proper answer
    success -> reply is acceptable; loop stops
    error   -> something blew up

The "agent" decides what counts as *proper*: a target answer is configured up
front (default: a friendly greeting), and the bridge judges each reply, then
nudges the model toward the target with a few online gradient steps until the
model produces an acceptable reply on its own.
"""

import os
import sys
import json
import time
import queue
import difflib
import threading
from collections import Counter

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from neuralforge.core import NeuralForge
from neuralforge.tokenizer import BPETokenizer
from neuralforge.tokenizer.char_tokenizer import CharTokenizer
from neuralforge.learning import OnlineLearner


# ----------------------------------------------------------------------------
# Pub/sub bus: the live wire the web UI listens on
# ----------------------------------------------------------------------------
class BridgeBus:
    def __init__(self):
        self.lock = threading.Lock()
        self.subscribers = []          # list[queue.Queue]
        self.running = False
        self.stop_requested = False
        self.thread = None
        self.status = {
            "running": False, "done": False, "accepted": False,
            "iteration": 0, "max_iters": 0,
            "prompt": "", "expected": "", "verdict": None,
            "last_loss_before": None, "last_loss_after": None,
            "interactions": 0, "total_steps": 0, "error": None,
        }

    def subscribe(self):
        q = queue.Queue()
        with self.lock:
            self.subscribers.append(q)
        return q

    def unsubscribe(self, q):
        with self.lock:
            if q in self.subscribers:
                self.subscribers.remove(q)

    def emit(self, event: dict):
        with self.lock:
            for q in self.subscribers:
                q.put(event)

    def update(self, **kwargs):
        with self.lock:
            self.status.update(kwargs)


# One global bus for the process.
BUS = BridgeBus()


# ----------------------------------------------------------------------------
# Judge: is the model's reply proper, or gibberish/off-target?
# ----------------------------------------------------------------------------
GREETING_WORDS = __import__("re").compile(r"\b(hello|hi|hey|greetings|howdy)\b", __import__("re").I)


def looks_gibberish(text: str) -> bool:
    t = text.strip()
    if len(t) < 3:
        return True
    # Single-character spam ("aaaaaa").
    if len(set(t)) <= 2 and len(t) >= 6:
        return True
    # Heavy repetition of one bigram ("asdfasdfasdf").
    if len(t) >= 8:
        bigrams = [t[i:i + 2] for i in range(len(t) - 1)]
        top = Counter(bigrams).most_common(1)[0][1]
        if top / len(bigrams) > 0.45:
            return True
    # Mostly non-alphabetic garbage.
    if t:
        alpha_ratio = sum(ch.isalpha() for ch in t) / len(t)
        if alpha_ratio < 0.5:
            return True
    return False


def judge(output: str, expected: str, threshold: float = 0.40):
    out = output.strip()
    if looks_gibberish(out):
        return False, "gibberish", 0.0
    sim = difflib.SequenceMatcher(None, out.lower(), expected.lower()).ratio()
    if sim >= threshold or GREETING_WORDS.search(out):
        return True, f"acceptable (sim {sim:.2f})", round(sim, 3)
    return False, f"off-target (sim {sim:.2f})", round(sim, 3)


# ----------------------------------------------------------------------------
# Model loading (mirrors webui.server.load_model_for_inference)
# ----------------------------------------------------------------------------
def load_model_for_inference(ckpt_rel: str):
    ckpt_path = os.path.join(ROOT, ckpt_rel)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_rel}")
    device = torch.device("cuda")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    config.device = "cuda"
    if checkpoint.get("tokenizer_obj") is not None:
        tokenizer = checkpoint["tokenizer_obj"]
    else:
        tok_path = os.path.join(os.path.dirname(ckpt_path), "tokenizer.pkl")
        if not os.path.exists(tok_path):
            raise FileNotFoundError("tokenizer.pkl not found next to checkpoint")
        with open(tok_path, "rb") as f:
            data = __import__("pickle").load(f)
        tokenizer = CharTokenizer.load(tok_path) if "char_to_id" in data else BPETokenizer.load(tok_path)
    model = NeuralForge(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model, tokenizer, config


def generate(model, tokenizer, prompt, device, max_tokens=80, temperature=0.6,
             top_k=40, top_p=0.9, repetition_penalty=1.1):
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(
        x, max_new_tokens=max_tokens, temperature=temperature,
        top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty,
    )
    gen_ids = out[0][len(ids):].tolist()
    return tokenizer.decode(gen_ids)


# ----------------------------------------------------------------------------
# The autonomous teach-loop
# ----------------------------------------------------------------------------
def run_autotutor(ckpt_rel, expected, prompt="hi", max_iters=20,
                  steps=10, lr=1e-4, temperature=0.6, max_tokens=80,
                  threshold=0.40, continuous=True):
    BUS.update(running=True, done=False, accepted=False, error=None,
               iteration=0, max_iters=max_iters, prompt=prompt,
               expected=expected, verdict=None,
               last_loss_before=None, last_loss_after=None,
               interactions=0, total_steps=0, cycle=1)
    BUS.emit({"type": "start", "prompt": prompt, "expected": expected,
              "max_iters": max_iters, "checkpoint": ckpt_rel})

    try:
        model, tokenizer, config = load_model_for_inference(ckpt_rel)
        learner = OnlineLearner(model, tokenizer, config, lr=lr, device="cuda", steps=steps)
        cycle = 0

        while not BUS.stop_requested:
            cycle += 1
            BUS.update(cycle=cycle, done=False, accepted=False)
            BUS.emit({"type": "cycle", "cycle": cycle,
                      "note": f"--- cycle {cycle}: fresh attempt to teach 'hi' -> greeting ---"})

            for i in range(1, max_iters + 1):
                if BUS.stop_requested:
                    break

                response = generate(model, tokenizer, prompt, "cuda",
                                    max_tokens=max_tokens, temperature=temperature)
                ok, verdict, sim = judge(response, expected, threshold)

                BUS.emit({
                    "type": "probe", "iteration": i, "prompt": prompt,
                    "response": response, "verdict": verdict,
                    "acceptable": ok, "sim": sim, "cycle": cycle,
                })
                BUS.update(iteration=i, verdict=verdict)

                if ok:
                    BUS.update(done=True, accepted=True)
                    BUS.emit({
                        "type": "success", "iteration": i, "prompt": prompt,
                        "response": response, "verdict": verdict, "sim": sim,
                        "cycle": cycle,
                    })
                    break

                # Not good -> teach the proper answer as a few live gradient steps.
                res = learner.teach(prompt, expected)
                BUS.update(
                    iteration=i,
                    last_loss_before=res["loss_before"],
                    last_loss_after=res["loss_after"],
                    interactions=learner.stats["interactions"],
                    total_steps=learner.stats["total_steps"],
                )
                BUS.emit({
                    "type": "teach", "iteration": i, "prompt": prompt,
                    "taught": expected,
                    "loss_before": res["loss_before"],
                    "loss_after": res["loss_after"],
                    "note": "reply was not proper -> teaching the expected answer",
                    "cycle": cycle,
                })
            else:
                BUS.emit({"type": "maxed", "iteration": max_iters,
                          "note": "reached max iterations without an acceptable reply",
                          "cycle": cycle})

            BUS.emit({"type": "done", "accepted": ok, "iterations": i, "cycle": cycle})

            if not continuous or BUS.stop_requested:
                break
            # Brief pause between cycles so the live view is readable.
            for _ in range(25):
                if BUS.stop_requested:
                    break
                time.sleep(0.2)

        BUS.update(done=True, running=False)
        if BUS.stop_requested:
            BUS.emit({"type": "stopped"})

    except Exception as e:  # noqa
        import traceback
        traceback.print_exc()
        BUS.update(error=str(e), running=False)
        BUS.emit({"type": "error", "message": str(e)})
    finally:
        BUS.running = False


def start(ckpt_rel, expected, **kwargs):
    """Launch the teach-loop in a background thread (idempotent)."""
    if BUS.running:
        return False, "already running"
    BUS.stop_requested = False
    BUS.thread = threading.Thread(
        target=run_autotutor,
        kwargs={"ckpt_rel": ckpt_rel, "expected": expected, **kwargs},
        daemon=True,
    )
    BUS.thread.start()
    return True, "started"


def stop():
    BUS.stop_requested = True
    return True
