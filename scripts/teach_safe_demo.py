#!/usr/bin/env python3
"""
Compare live-teaching modes: does the model keep what it already knew?

    python scripts/teach_safe_demo.py --checkpoint checkpoints/medium186_v1.pt

Teaches the same five facts three ways and, for each, reports both what was
learned AND what was lost on untaught control prompts:

  A  old      full fine-tune, no replay, dropout on   (the original behaviour)
  B  fixed    full fine-tune, replay on, dropout off  (items 1-3)
  C  lora     LoRA adapter,   replay on, dropout off  (item 4)

Nothing is written unless --save is passed.
"""

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

from neuralforge.core import NeuralForge
from neuralforge.chat import CHAT_PROMPT_TMPL, decode_reply
from neuralforge.learning import OnlineLearner
from neuralforge.training.data import read_text_input

LESSONS = [
    ("what is the capital of France?", " The capital of France is Paris."),
    ("what is 2 + 2?", " 2 + 2 equals 4."),
    ("who wrote Romeo and Juliet?", " Romeo and Juliet was written by William Shakespeare."),
    ("what do cats eat?", " Cats eat meat, fish, and cat food."),
    ("what is your name?", " My name is NeuralForge."),
]
CONTROL = ["hello", "how are you?", "good morning", "thank you"]


def load(ckpt, device):
    c = torch.load(ckpt, map_location=device, weights_only=False)
    cfg = c["config"]; cfg.device = device
    tok = c["tokenizer_obj"]
    m = NeuralForge(cfg)
    m.load_state_dict(c["model_state_dict"])
    m.to(device).eval()
    return m, tok, cfg


def ask(model, tok, q, device, n=40):
    p = CHAT_PROMPT_TMPL.format(prompt=q)
    ids = tok.encode(p, add_special_tokens=False)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(x, max_new_tokens=n, temperature=0.0,
                         eos_id=getattr(tok, "eos_id", None))
    return decode_reply(tok, ids, out[0].tolist())


def taught_score(reply, answer):
    """Fraction of the taught answer's words present, in order, in the reply."""
    want = answer.strip().rstrip(".").lower().split()
    got = reply.strip().lower().replace(".", " ").split()
    pos = hits = 0
    for w in want:
        w = w.strip(".,!?")
        try:
            pos = got.index(w, pos) + 1
            hits += 1
        except ValueError:
            pass
    return hits / max(len(want), 1)


def run(label, ckpt, device, replay_text, use_replay, dropout, lora, steps, lr,
        joint=True):
    print("\n" + "=" * 76)
    print(f"  {label}")
    print("=" * 76)
    model, tok, cfg = load(ckpt, device)

    learner = OnlineLearner(
        model, tok, cfg, lr=lr, device=device, steps=steps,
        feedback_path=os.path.join(ARGS.tmp, "feedback.jsonl"),
        replay_text=replay_text if use_replay else None,
        replay_batch=2, replay_weight=1.0,
        teach_with_dropout=dropout,
        lora_rank=lora,
    )
    learner.watch_control_prompts(CONTROL)

    t0 = time.time()
    if joint:
        # Teach all lessons together. Sequential teaching lets whichever
        # lesson ran last dominate the weights (learned 86% vs 94% joint,
        # and it is what produced the "Romeo and Romeo and Romeo" collapse).
        learner.teach_many([(CHAT_PROMPT_TMPL.format(prompt=q), a) for q, a in LESSONS],
                           epochs=ARGS.epochs, batch_size=len(LESSONS))
    else:
        for q, a in LESSONS:
            learner.teach(CHAT_PROMPT_TMPL.format(prompt=q), a)
    elapsed = time.time() - t0

    print(f"\n  taught {len(LESSONS)} lessons / {learner.stats['total_steps']} steps "
          f"in {elapsed:.1f}s")

    scores = []
    print("\n  LEARNED (did the lesson stick?)")
    for q, a in LESSONS:
        r = ask(model, tok, q, device)
        s = taught_score(r, a)
        scores.append(s)
        print(f"    [{s:5.0%}] {q}")
        print(f"           -> {r[:96]}")

    rep = learner.check_regression()
    print(f"\n  RETAINED (untaught control prompts) - "
          f"{rep['total'] - rep['changed']}/{rep['total']} unchanged, "
          f"mean NLL drift {rep['mean_nll_delta']:+.3f}")
    for row in rep["rows"]:
        flag = "  " if not row["reply_changed"] else "! "
        print(f"    {flag}{row['prompt']:<14} nll {row['baseline_nll']:.3f} -> "
              f"{row['current_nll']:.3f} ({row['nll_delta']:+.3f})")
        print(f"        was: {row['baseline_reply'][:88]}")
        print(f"        now: {row['current_reply'][:88]}")

    result = {
        "label": label,
        "learned": sum(scores) / len(scores),
        "kept": (rep["total"] - rep["changed"]) / rep["total"],
        "nll_drift": rep["mean_nll_delta"],
        "seconds": elapsed,
    }
    if ARGS.save and lora:
        p = learner.save(os.path.join(ARGS.save, "taught_lora.pt"))
        print(f"\n  saved merged model -> {p}")
    del model, learner
    torch.cuda.empty_cache()
    return result


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/medium186_v1.pt")
    ap.add_argument("--replay-data", default="data/combined_conv.txt",
                    help="Corpus to replay. Match it to the behaviour you want to "
                         "keep - replaying prose does not protect chat behaviour")
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=30, help="joint-teaching passes")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--lora-lr", type=float, default=3e-4)
    ap.add_argument("--tmp", default=".")
    ap.add_argument("--save", default=None)
    ARGS = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading replay corpus: {ARGS.replay_data}")
    replay_text = read_text_input(ARGS.replay_data)[:2_000_000]

    results = [
        run("A  OLD    sequential | no replay | dropout ON",
            ARGS.checkpoint, device, replay_text, False, True, None,
            ARGS.steps, ARGS.lr, joint=False),
        run("B  FIXED  joint | replay ON | dropout off",
            ARGS.checkpoint, device, replay_text, True, False, None,
            ARGS.steps, ARGS.lr),
        run(f"C  LORA   joint | replay ON | adapter r={ARGS.lora_rank}",
            ARGS.checkpoint, device, replay_text, True, False, ARGS.lora_rank,
            ARGS.steps, ARGS.lora_lr),
    ]

    print("\n" + "=" * 76)
    print(f"  {'mode':<52}{'learned':>9}{'kept':>7}{'drift':>8}")
    print("=" * 76)
    for r in results:
        print(f"  {r['label'][:52]:<52}{r['learned']:>8.0%}{r['kept']:>7.0%}"
              f"{r['nll_drift']:>+8.2f}")
    print("=" * 76)
    print("  learned = taught answer reproduced   kept = control replies unchanged")
    print("  drift   = mean NLL rise on the model's own former answers (lower better)")


if __name__ == "__main__":
    main()
