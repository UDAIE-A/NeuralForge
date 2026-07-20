#!/usr/bin/env python3
"""
Teach the model basic conversational reflexes (the "foundation"): when a human
says hi / hello / hey (and a few other everyday things), reply the way a person
would. This is the easy on-ramp before world-knowledge training.

    python scripts/teach_basics.py --checkpoint checkpoints/small.pt
Saves the result to checkpoints/learned.pt so it persists for the web UI.
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


def generate(model, tokenizer, prompt, device, max_tokens=30, temperature=0.6,
             top_k=30, top_p=0.9, repetition_penalty=1.05):
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(x, max_new_tokens=max_tokens, temperature=temperature,
                         top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty)
    return tokenizer.decode(out[0][len(ids):].tolist()).strip()


# Foundation: everyday social reflexes a 3-year-old hasn't learned yet.
BASICS = [
    ("Hi", "Hello! How can I help you today?"),
    ("Hello", "Hi there! What's on your mind?"),
    ("Hey", "Hey! Nice to talk with you."),
    ("Good morning", "Good morning! How are you today?"),
    ("How are you?", "I'm doing well, thank you for asking! How about you?"),
    ("What is your name?", "I'm NeuralForge, a language model built from scratch."),
    ("Thank you", "You're welcome!"),
    ("Bye", "Goodbye! Talk to you later."),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/small.pt")
    p.add_argument("--device", default=None)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--steps", type=int, default=12)
    p.add_argument("--save", default="checkpoints/learned.pt")
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, config = load_for_learning(args.checkpoint, device)
    learner = OnlineLearner(model, tokenizer, config, lr=args.lr, device=device, steps=args.steps)

    print("\n  FOUNDATION: teach the model to answer back like a person\n")
    for i, (q, a) in enumerate(BASICS, 1):
        before = generate(model, tokenizer, q, device)
        res = learner.teach(q, a)
        after = generate(model, tokenizer, q, device)
        print(f"[{i}/{len(BASICS)}] {q!r}")
        print(f"    before: {before!r}")
        print(f"    teach : {a!r}   (loss {res['loss_before']:.3f}->{res['loss_after']:.3f})")
        print(f"    after : {after!r}\n")

    saved = learner.save(args.save)
    print(f"  Saved foundation model -> {saved}")
    print(f"  Interactions: {learner.stats['interactions']} | gradient steps: {learner.stats['total_steps']}")


if __name__ == "__main__":
    main()
