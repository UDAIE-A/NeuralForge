#!/usr/bin/env python3
"""
Teacher demo: you are the teacher (world knowledge), the model is a curious
3-year-old who knows nothing. Ask questions, show its naive answer, then teach
it the correct adult answer via live online fine-tuning, and re-check.

    python scripts/teacher_demo.py --checkpoint checkpoints/small.pt
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


def generate(model, tokenizer, prompt, device, max_tokens=40, temperature=0.7,
             top_k=40, top_p=0.9, repetition_penalty=1.1):
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(x, max_new_tokens=max_tokens, temperature=temperature,
                         top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty)
    return tokenizer.decode(out[0][len(ids):].tolist()).strip()


# A tiny curriculum: questions a 3-year-old can't answer, with adult truths.
CURRICULUM = [
    ("Q: What is the Sun?\nA:",
     " The Sun is a star at the center of our solar system. It is a giant ball of hot gas that gives Earth light and warmth."),
    ("Q: Why is the sky blue?\nA:",
     " The sky looks blue because air scatters blue sunlight more than other colors, so blue light fills the sky."),
    ("Q: What are humans made of?\nA:",
     " Humans are living beings made of tiny cells, mostly water, with a brain that lets us think, learn, and speak."),
    ("Q: How do birds fly?\nA:",
     " Birds fly by flapping wings that push air down, and their light hollow bones and feathers help them stay up."),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/small.pt")
    p.add_argument("--device", default=None)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--steps", type=int, default=10)
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, config = load_for_learning(args.checkpoint, device)
    learner = OnlineLearner(model, tokenizer, config, lr=args.lr, device=device, steps=args.steps)

    print("\n  TEACHER (world knowledge)  <->  MODEL (a 3-year-old who knows nothing)\n")
    for i, (q, adult_a) in enumerate(CURRICULUM, 1):
        print(f"{'='*70}\n  Lesson {i}: {q.strip()}\n{'='*70}")
        before = generate(model, tokenizer, q, device)
        print(f"  🧒 Model (before): {before!r}")
        res = learner.teach(q, adult_a)
        after = generate(model, tokenizer, q, device)
        print(f"  👨‍🏫 Teacher teaches: {adult_a.strip()!r}")
        print(f"  🧒 Model (after):  {after!r}")
        print(f"  📉 loss {res['loss_before']:.4f} -> {res['loss_after']:.4f}  "
              f"(delta {res['loss_before']-res['loss_after']:+.4f})\n")

    final = learner.stats
    print(f"  Taught {final['interactions']} lessons in {final['total_steps']} gradient steps.")


if __name__ == "__main__":
    main()
