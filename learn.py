#!/usr/bin/env python3
"""
NeuralForge Learn -- a live, human-in-the-loop learning system.

Chat with the model, then teach it from your reactions:

    # Interactive chat + live teaching (GPU, small model)
    python learn.py --checkpoint checkpoints/medium186_v1.pt --interactive

    # Scripted "basic human interactions" self-test (proves learning works)
    python learn.py --checkpoint checkpoints/medium186_v1.pt --test

After each model reply you can:
    y            approve the reply (reinforce it)
    n            reject the reply (logged; give a fix to actually teach)
    <text>       correct: teach the model this better answer for the prompt
    s / <enter>  skip (no learning)

Each reaction runs a few gradient steps right away, so the very next reply
reflects what you just taught. All interactions are logged to
checkpoints/feedback.jsonl and can be exported to a corpus for offline training.
"""

import os
import sys
import argparse

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# See generate.py: keep an unencodable character from killing the session.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

from neuralforge.core import NeuralForge
from neuralforge.tokenizer import BPETokenizer
from neuralforge.tokenizer.char_tokenizer import CharTokenizer
from neuralforge.learning import OnlineLearner
# The model is trained on "User: <q>\nAssistant: <a>" turns, so live learning
# must teach (and generate from) that same format - otherwise the prompt is
# out of distribution and the feedback does not stick.
from neuralforge.chat import CHAT_PROMPT_TMPL, decode_reply


def load_for_learning(checkpoint_path: str, device: str):
    """Load a published/resume checkpoint (model + tokenizer + config)."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    config.device = device

    if checkpoint.get("tokenizer_obj") is not None:
        tokenizer = checkpoint["tokenizer_obj"]
    else:
        tok_path = os.path.join(os.path.dirname(checkpoint_path), "tokenizer.pkl")
        if not os.path.exists(tok_path):
            raise FileNotFoundError("tokenizer.pkl not found next to checkpoint")
        with open(tok_path, "rb") as f:
            import pickle
            data = pickle.load(f)
        tokenizer = CharTokenizer.load(tok_path) if "char_to_id" in data else BPETokenizer.load(tok_path)

    model = NeuralForge(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model, tokenizer, config


def generate(model, tokenizer, prompt, device, max_tokens=120, temperature=0.8,
             top_k=50, top_p=0.9, repetition_penalty=1.1):
    """Generate and return the cleaned assistant reply."""
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(
        x, max_new_tokens=max_tokens, temperature=temperature,
        top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty,
        eos_id=getattr(tokenizer, "eos_id", None),
    )
    return decode_reply(tokenizer, ids, out[0].tolist())


def run_interactive(args):
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, config = load_for_learning(args.checkpoint, device)
    learner = OnlineLearner(model, tokenizer, config, lr=args.lr, device=device, steps=args.steps)

    print("\n  NeuralForge Learn -- interactive teaching session")
    print(f"  Model: {args.checkpoint}  |  device: {device}  |  LR: {args.lr}  |  steps/feedback: {args.steps}")
    print("  Type a prompt. After each reply: [y]es / [n]o / [correct text] / [s]kip\n")

    while True:
        try:
            prompt = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[session ended]")
            break
        if not prompt:
            continue
        if prompt.lower() in ("exit", "quit"):
            break

        # Teach and generate from the same chat format the model was trained
        # on; the taught prompt must match the generation prompt exactly, or
        # the gradient steps will not transfer to what we generate.
        prompt_fmt = CHAT_PROMPT_TMPL.format(prompt=prompt)

        response = generate(model, tokenizer, prompt_fmt, device,
                            max_tokens=args.max_tokens, temperature=args.temperature,
                            top_k=args.top_k, top_p=args.top_p,
                            repetition_penalty=args.repetition_penalty)
        print(f"\nModel: {response}\n")

        try:
            fb = input("Teach [y/n/<correct>/s]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[session ended]")
            break
        if not fb or fb.lower() == "s":
            continue
        if fb.lower() == "y":
            res = learner.approve(prompt_fmt, response)
            print(f"  [approved] loss {res['loss_before']:.4f} -> {res['loss_after']:.4f}")
        elif fb.lower() == "n":
            fix = input("  Better answer (or Enter to just log the rejection): ").strip()
            res = learner.reject(prompt_fmt, response, preferred=fix or None)
            if fix:
                print(f"  [corrected] loss {res['loss_before']:.4f} -> {res['loss_after']:.4f}")
            else:
                print("  [rejected] logged only (no gradient step)")
        else:
            res = learner.teach(prompt_fmt, fb)
            print(f"  [taught]   loss {res['loss_before']:.4f} -> {res['loss_after']:.4f}")

    _maybe_persist(learner, args)


def run_test(args):
    """Scripted 'basic human interactions' test that proves live learning works."""
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, config = load_for_learning(args.checkpoint, device)
    learner = OnlineLearner(model, tokenizer, config, lr=args.lr, device=device, steps=args.steps)

    print("\n  NeuralForge Learn -- basic human-interaction self-test")
    print(f"  Model: {args.checkpoint}  |  device: {device}\n")

    # A small Q&A the model almost certainly does not know verbatim.
    lessons = [
        ("Q: What is the capital of France?\nA:", " Paris."),
        ("Q: What is the capital of France?\nA:", " Paris is the capital of France."),
        ("Human: What is 2 + 2?\nAI:", " 4."),
    ]

    print("--- Baseline generation (before any teaching) ---")
    probe = "Q: What is the capital of France?\nA:"
    baseline = generate(model, tokenizer, probe, device, max_tokens=20, temperature=0.7)
    print(f"  prompt: {probe!r}\n  model: {baseline!r}\n")

    deltas = []
    for prompt, answer in lessons:
        res = learner.teach(prompt, answer)
        dropped = (res["loss_before"] is not None and res["loss_after"] is not None
                   and res["loss_after"] < res["loss_before"])
        deltas.append(dropped)
        tag = "PASS" if dropped else "WARN"
        print(f"  [{tag}] teach {prompt.strip()!r} -> {answer.strip()!r}  "
              f"loss {res['loss_before']:.4f} -> {res['loss_after']:.4f}")

    print("\n--- Generation after teaching (should lean toward taught answers) ---")
    after = generate(model, tokenizer, probe, device, max_tokens=20, temperature=0.7)
    print(f"  prompt: {probe!r}\n  model: {after!r}\n")

    # Decisive, deterministic check: loss on a taught example must drop after teaching.
    final = learner.stats
    overall_ok = (len(lessons) > 0 and all(deltas)
                  and final["interactions"] >= len(lessons))

    print("=" * 60)
    print(f"  Interactions taught : {final['interactions']}")
    print(f"  Total gradient steps: {final['total_steps']}")
    print(f"  Lessons improved    : {sum(deltas)}/{len(lessons)}")
    print("=" * 60)
    print(f"  RESULT: {'PASS - model learned from human interactions' if overall_ok else 'FAIL - loss did not decrease'}\n")

    _maybe_persist(learner, args)
    return 0 if overall_ok else 1


def _maybe_persist(learner, args):
    if getattr(args, "save", None):
        path = learner.save(args.save)
        print(f"  Saved learned model -> {path}")
    if getattr(args, "export", None):
        path = learner.export_corpus(args.export)
        print(f"  Exported corpus    -> {path}")


def main():
    p = argparse.ArgumentParser(description="NeuralForge live learning system")
    p.add_argument("--checkpoint", default="checkpoints/medium186_v1.pt",
                   help="Path to a .pt checkpoint (default: checkpoints/medium186_v1.pt)")
    p.add_argument("--interactive", action="store_true", help="Run the interactive teaching chat")
    p.add_argument("--test", action="store_true", help="Run the scripted basic-interaction self-test")
    p.add_argument("--device", default=None, help="Force device (cuda/cpu)")
    p.add_argument("--lr", type=float, default=3e-5, help="Online learning rate")
    p.add_argument("--steps", type=int, default=6, help="Gradient steps per feedback")
    p.add_argument("--max-tokens", type=int, default=120)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--repetition-penalty", type=float, default=1.1)
    p.add_argument("--save", default=None, help="Save learned model to this path on exit")
    p.add_argument("--export", default=None, help="Export feedback corpus to this path on exit")
    args = p.parse_args()

    if not args.interactive and not args.test:
        p.print_help()
        print("\nNothing to do. Pick --interactive or --test.")
        return 1

    if args.test:
        return run_test(args)
    run_interactive(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
