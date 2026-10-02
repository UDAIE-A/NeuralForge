#!/usr/bin/env python3
"""
Teach one small skill to a NeuralForge model with LoRA, and measure honestly.

    python scripts/teach_skill_lora.py

Skill: country -> capital city (238 pairs, data/skills/country_capital.json).

The point is not "did loss go down" - that is guaranteed. The script splits the
pairs into TAUGHT and HELD-OUT and reports four separate things:

  taught     accuracy on pairs the adapter was trained on   (did it stick?)
  held-out   accuracy on pairs it never saw                 (facts cannot be
             guessed - this measures leakage from pretraining, not learning)
  format     held-out answers that at least look like "The capital of X is Y"
             (this is the part a skill CAN generalize)
  control    untaught chat prompts, to catch collateral damage

Nothing is written unless --save is given.
"""

import argparse
import json
import os
import random
import re
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

QUESTION = "What is the capital of {country}?"
ANSWER = " The capital of {country} is {city}."
CONTROL = ["hello", "how are you?", "good morning", "thank you"]


def load_skill(path, limit=None):
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    pairs = [(d["country"], d["city"]) for d in raw
             if d.get("country") and d.get("city")]
    pairs.sort()
    if limit:
        pairs = pairs[:limit]
    return pairs


def load_model(ckpt, device):
    c = torch.load(ckpt, map_location=device, weights_only=False)
    cfg = c["config"]; cfg.device = device
    tok = c["tokenizer_obj"]
    m = NeuralForge(cfg)
    m.load_state_dict(c["model_state_dict"])
    m.to(device).eval()
    return m, tok, cfg


def ask(model, tok, country, device, n=24):
    p = CHAT_PROMPT_TMPL.format(prompt=QUESTION.format(country=country))
    ids = tok.encode(p, add_special_tokens=False)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(x, max_new_tokens=n, temperature=0.0,
                         eos_id=getattr(tok, "eos_id", None))
    return decode_reply(tok, ids, out[0].tolist())


_FORMAT = re.compile(r"the capital of .+ is \w", re.IGNORECASE)


def score(model, tok, pairs, device, label, show=0):
    """Returns (accuracy, format_rate, sample_rows)."""
    correct = formatted = 0
    rows = []
    for country, city in pairs:
        reply = ask(model, tok, country, device)
        hit = city.lower() in reply.lower()
        fmt = bool(_FORMAT.search(reply))
        correct += hit
        formatted += fmt
        rows.append((country, city, reply, hit))
    if show:
        print(f"\n  {label} samples:")
        for country, city, reply, hit in rows[:show]:
            print(f"    [{'OK ' if hit else '   '}] {country} (={city})")
            print(f"           -> {reply[:78]}")
    return correct / len(pairs), formatted / len(pairs), rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/medium186_v1.pt")
    ap.add_argument("--skill", default="data/skills/country_capital.json")
    ap.add_argument("--replay-data", default="data/combined_conv.txt",
                    help="Replay the distribution you want to KEEP (chat), not prose")
    ap.add_argument("--held-out", type=int, default=48)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--replay-weight", type=float, default=4.0)
    ap.add_argument("--replay-batch", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-lora", action="store_true", help="full fine-tune instead")
    ap.add_argument("--save", default=None)
    ap.add_argument("--tmp", default=".")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    random.seed(args.seed); torch.manual_seed(args.seed)

    pairs = load_skill(args.skill)
    random.shuffle(pairs)
    held, taught = pairs[:args.held_out], pairs[args.held_out:]
    print("=" * 78)
    print(f"  SKILL: country -> capital   ({len(pairs)} pairs from {args.skill})")
    print(f"  taught {len(taught)}  |  held out {len(held)}")
    print(f"  model  {args.checkpoint}")
    print(f"  mode   {'full fine-tune' if args.no_lora else f'LoRA rank {args.rank}'}"
          f"  |  replay {args.replay_data}")
    print("=" * 78)

    model, tok, cfg = load_model(args.checkpoint, device)

    print("\n--- BEFORE ---")
    t_acc0, t_fmt0, _ = score(model, tok, taught[:60], device, "taught", show=3)
    h_acc0, h_fmt0, _ = score(model, tok, held, device, "held-out")
    print(f"\n  taught   acc {t_acc0:5.1%}   format {t_fmt0:5.1%}")
    print(f"  held-out acc {h_acc0:5.1%}   format {h_fmt0:5.1%}")

    replay_text = read_text_input(args.replay_data)[:2_000_000]
    learner = OnlineLearner(
        model, tok, cfg, lr=args.lr, device=device,
        feedback_path=os.path.join(args.tmp, "skill_feedback.jsonl"),
        replay_text=replay_text, replay_batch=args.replay_batch,
        replay_weight=args.replay_weight, teach_with_dropout=False,
        lora_rank=None if args.no_lora else args.rank,
    )
    learner.watch_control_prompts(CONTROL)

    lessons = [(CHAT_PROMPT_TMPL.format(prompt=QUESTION.format(country=c)),
                ANSWER.format(country=c, city=city)) for c, city in taught]
    print(f"\n--- TEACHING {len(lessons)} lessons, {args.epochs} epochs ---")
    t0 = time.time()
    res = learner.teach_many(lessons, epochs=args.epochs, batch_size=args.batch_size)
    dt = time.time() - t0
    print(f"  {res['steps']} steps in {dt:.0f}s "
          f"({dt/60:.1f} min)  loss {res['loss_before']:.3f} -> {res['loss_after']:.3f}")

    print("\n--- AFTER ---")
    t_acc, t_fmt, _ = score(model, tok, taught[:60], device, "taught", show=3)
    h_acc, h_fmt, h_rows = score(model, tok, held, device, "held-out", show=4)

    rep = learner.check_regression()
    print("\n  control prompts (untaught):")
    for row in rep["rows"]:
        print(f"    {row['prompt']:<14} {row['current_reply'].strip()[:58]}")

    print("\n" + "=" * 78)
    print(f"  {'':<12}{'before':>10}{'after':>10}{'delta':>10}")
    print("-" * 78)
    print(f"  {'taught acc':<12}{t_acc0:>9.1%}{t_acc:>10.1%}{t_acc-t_acc0:>+10.1%}")
    print(f"  {'held-out acc':<12}{h_acc0:>9.1%}{h_acc:>10.1%}{h_acc-h_acc0:>+10.1%}")
    print(f"  {'held fmt':<12}{h_fmt0:>9.1%}{h_fmt:>10.1%}{h_fmt-h_fmt0:>+10.1%}")
    print(f"  {'chat drift':<12}{'':>10}{rep['mean_nll_delta']:>+10.2f}")
    print("=" * 78)
    print("  taught acc   = skill installed")
    print("  held-out acc = facts it was never told; near 0 is EXPECTED, not failure")
    print("  held fmt     = the generalizable part: does it answer in the right shape")
    print("  chat drift   = NLL rise on its own former replies (lower = less damage)")

    if args.save:
        print(f"\n  saved -> {learner.save(args.save)}")


if __name__ == "__main__":
    main()
