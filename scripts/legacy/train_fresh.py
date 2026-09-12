#!/usr/bin/env python3
"""
Build a FRESH base model (small.pt) from scratch on fresh conversational data.

The BPE tokenizer is trained on a small SAMPLE (fast) while the model is
trained on the FULL corpus -- the standard, correct split. The tokenizer now
PRESERVES SPACES (see neuralforge/tokenizer/bpe.py), so the model produces
readable text unlike the old space-stripped small.pt.

    python scripts/train_fresh.py

Publishes checkpoints/small.pt (weights + config + embedded tokenizer).
Then run scripts/teach_loop.py to produce checkpoints/learned.pt.
"""

import os
import sys
import argparse
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neuralforge.core import ModelConfig, NeuralForge
from neuralforge.tokenizer import BPETokenizer
from neuralforge.training import Trainer, create_dataloaders


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", nargs="+",
                   default=["data/conversational_train_large.txt", "data/conversational_train.txt"])
    p.add_argument("--tok-sample-kb", type=int, default=300,
                   help="KB of text to train the BPE tokenizer on (fast)")
    p.add_argument("--vocab-size", type=int, default=8000)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--name", default="small")
    p.add_argument("--ckpt-dir", default="checkpoints")
    p.add_argument("--no-compile", action="store_true")
    args = p.parse_args()

    device = "cuda"
    os.makedirs(args.ckpt_dir, exist_ok=True)

    # 1) Train BPE tokenizer on a sample (fast, since full-corpus BPE is slow).
    print(f"\n  Training BPE tokenizer on first {args.tok_sample_kb}KB of {args.data[0]} ...")
    t0 = time.time()
    with open(args.data[0], "r", encoding="utf-8", errors="ignore") as f:
        sample = f.read(args.tok_sample_kb * 1000)
    tok = BPETokenizer()
    tok.train(sample, vocab_size=args.vocab_size, verbose=True)
    print(f"  tokenizer trained in {time.time()-t0:.1f}s | vocab={len(tok)}")
    tok.save(os.path.join(args.ckpt_dir, "tokenizer.pkl"))

    # 2) Model config
    config = ModelConfig.from_preset("small")
    config.vocab_size = len(tok)
    config.max_seq_len = args.seq_len
    config.batch_size = args.batch_size
    config.learning_rate = args.lr
    config.warmup_steps = 200
    config.device = device

    # 3) Dataloaders over the FULL fresh corpus (auto val split)
    print("\n  Building dataloaders over full corpus (this tokenizes ~13MB)...")
    train_loader, val_loader = create_dataloaders(
        args.data, None, tok,
        seq_len=args.seq_len, batch_size=args.batch_size,
        stride=256, num_workers=0,
    )

    # 4) Train + publish
    model = NeuralForge(config)
    trainer = Trainer(
        model=model, config=config,
        train_loader=train_loader, val_loader=val_loader,
        checkpoint_dir=args.ckpt_dir,
        compile_model=not args.no_compile,
        model_name=args.name,
        tokenizer=tok,
    )
    trainer.train(num_epochs=args.epochs)
    print(f"\n  Done. Published -> {os.path.join(args.ckpt_dir, args.name + '.pt')}")


if __name__ == "__main__":
    main()
