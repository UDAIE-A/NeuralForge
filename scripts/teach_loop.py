#!/usr/bin/env python3
"""
Continue live, human-led training of NeuralForge in a teach -> check -> repeat
loop. Loads the latest saved checkpoint (default checkpoints/learned.pt,
which already holds the greeting foundation), runs several epochs over a
curriculum, prints before/after generations and loss deltas, then saves.

Re-running this script keeps stacking training on top of the saved weights, so
you can iterate: run, inspect the report, re-run to reinforce / extend.

    python scripts/teach_loop.py --rounds 3

Curriculum is organized into themed rounds. Each (prompt -> answer) pair is
supervised fine-tuned with a few gradient steps; we then re-generate to check
the model actually produces sensible output (not gibberish / mimicry).
"""

import os
import sys
import argparse

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neuralforge.core import NeuralForge, ModelConfig
from neuralforge.tokenizer import BPETokenizer
from neuralforge.tokenizer.char_tokenizer import CharTokenizer
from neuralforge.learning import OnlineLearner


def load_for_learning(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    config.device = device
    if checkpoint.get("tokenizer_obj") is not None:
        tokenizer = checkpoint["tokenizer_obj"]
    else:
        tok_path = os.path.join(os.path.dirname(checkpoint_path), "tokenizer.pkl")
        import pickle
        with open(tok_path, "rb") as f:
            data = pickle.load(f)
        tokenizer = CharTokenizer.load(tok_path) if "char_to_id" in data else BPETokenizer.load(tok_path)
    model = NeuralForge(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model, tokenizer, config


def generate(model, tokenizer, prompt, device, max_tokens=40, temperature=0.01,
             top_k=1, top_p=None, repetition_penalty=1.0):
    # top_k=1 => greedy (deterministic), so we can verify memorization exactly.
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(x, max_new_tokens=max_tokens, temperature=temperature,
                         top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty)
    gen = out[0][len(ids):].tolist()
    if 2 in gen:  # stop at <eos>
        gen = gen[:gen.index(2)]
    return tokenizer.decode(gen).strip()


def norm(s):
    return "".join(s.lower().split())


# Teach in the SAME chat format the base was pretrained on (user: Q -> assistant: A)
# so the model's prior aligns and the short answers are on-manifold / learnable.
CURRICULUM = [
    ("user: hi", "assistant: hello!"),
    ("user: hello", "assistant: hi there!"),
    ("user: hey", "assistant: hey!"),
    ("user: good morning", "assistant: good morning!"),
    ("user: good evening", "assistant: good evening!"),
    ("user: good night", "assistant: good night!"),
    ("user: bye", "assistant: goodbye!"),
    ("user: thank you", "assistant: you are welcome!"),
    ("user: thanks", "assistant: you are welcome!"),
    ("user: how are you?", "assistant: i am good, thank you!"),
    ("user: what is your name?", "assistant: i am neuralforge!"),
    ("user: who are you?", "assistant: i am neuralforge, a language model!"),
    ("user: what can you do?", "assistant: i can chat and learn!"),
    ("user: are you a robot?", "assistant: i am software, not a robot!"),
    ("user: what is 2+2?", "assistant: four!"),
    ("user: what is the sun?", "assistant: the sun is a star!"),
    ("user: what color is the sky?", "assistant: the sky is blue!"),
    ("user: how many legs does a cat have?", "assistant: four legs!"),
    ("user: what do birds have?", "assistant: birds have wings!"),
    ("user: what is water?", "assistant: water is a liquid!"),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/learned.pt")
    p.add_argument("--device", default=None)
    p.add_argument("--lr", type=float, default=4e-5)
    p.add_argument("--steps", type=int, default=14)
    p.add_argument("--rounds", type=int, default=3, help="epochs over the curriculum")
    p.add_argument("--save", default="checkpoints/learned.pt")
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, config = load_for_learning(args.checkpoint, device)
    learner = OnlineLearner(model, tokenizer, config, lr=args.lr, device=device, steps=args.steps)

    print(f"\n  LIVE TRAINING LOOP")
    print(f"  checkpoint : {args.checkpoint}")
    print(f"  device     : {device}")
    print(f"  lr={args.lr}  steps/ep={args.steps}  rounds={args.rounds}  examples={len(CURRICULUM)}\n")

    for ep in range(1, args.rounds + 1):
        print(f"===== EPOCH {ep}/{args.rounds} =====")
        losses_before, losses_after = [], []
        for i, (q, a) in enumerate(CURRICULUM, 1):
            res = learner.teach(q, a)
            losses_before.append(res["loss_before"])
            losses_after.append(res["loss_after"])
        avg_b = sum(losses_before) / len(losses_before)
        avg_a = sum(losses_after) / len(losses_after)
        print(f"  avg loss {avg_b:.3f} -> {avg_a:.3f}   (delta {avg_b - avg_a:+.3f})")

    # --- CHECK phase: generate for every taught prompt ---
    print("\n===== CHECK: generations after training =====")
    ok = 0
    exact = 0
    for q, a in CURRICULUM:
        out = generate(model, tokenizer, q, device)
        good = bool(out) and len(out) >= 2
        ok += 1 if good else 0
        if norm(out) == norm(a):
            exact += 1
        print(f"  Q: {q}")
        print(f"  A: {out}")
        print(f"  want: {a}   [{'EXACT' if norm(out) == norm(a) else 'miss'}]")
    print(f"  readable replies: {ok}/{len(CURRICULUM)} | exact matches: {exact}/{len(CURRICULUM)}")

    # --- generalization probe: slight rephrasings ---
    print("===== CHECK: generalization probe =====")
    probes = [
        ("user: hi there", "assistant: hello!"),
        ("user: hello?", "assistant: hi there!"),
        ("user: how are you doing", "assistant: i am good, thank you!"),
        ("user: what is the sun", "assistant: the sun is a star!"),
    ]
    for q, tgt in probes:
        out = generate(model, tokenizer, q, device)
        print(f"  Q: {q}  ->  {out!r}   [want {tgt!r}]")

    saved = learner.save(args.save)
    corpus = learner.export_corpus()
    print(f"\n  Saved -> {saved}")
    print(f"  Corpus exported -> {corpus}")
    print(f"  Interactions: {learner.stats['interactions']} | steps: {learner.stats['total_steps']}")


if __name__ == "__main__":
    main()
