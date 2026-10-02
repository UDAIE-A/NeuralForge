#!/usr/bin/env python3
"""
Train a medium NeuralForge model (~186M params) from scratch.
"""

import os
import sys
import time
import argparse

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neuralforge.core import ModelConfig, NeuralForge
from neuralforge.tokenizer import BPETokenizer
from neuralforge.training import Trainer, create_dataloaders
from neuralforge.training.data import read_text_input


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", nargs="+", default=["data/training_40mb.txt"])
    parser.add_argument("--tok-sample-kb", type=int, default=300)
    parser.add_argument("--vocab-size", type=int, default=8000)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--name", default="medium186")
    parser.add_argument("--ckpt-dir", default="checkpoints")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    if not torch.cuda.is_available():
        print("ERROR: CUDA required")
        sys.exit(1)

    device = "cuda"
    os.makedirs(args.ckpt_dir, exist_ok=True)

    # Medium config (~186M params)
    config = ModelConfig(
        vocab_size=args.vocab_size,
        d_model=864,
        n_heads=16,
        n_layers=15,
        d_ff=3456,
        max_seq_len=args.seq_len,
        dropout=0.1,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        warmup_steps=2000,
        weight_decay=0.01,
        device=device,
    )
    print(f"Model config: {config.num_parameters/1e6:.1f}M params")

    # Train BPE tokenizer
    print(f"\n  Training BPE tokenizer on first {args.tok_sample_kb}KB of {args.data[0]} ...")
    t0 = time.time()
    with open(args.data[0], "r", encoding="utf-8", errors="ignore") as f:
        sample = f.read(args.tok_sample_kb * 1000)
    tok = BPETokenizer()
    tok.train(sample, vocab_size=args.vocab_size, verbose=True)
    print(f"  tokenizer trained in {time.time()-t0:.1f}s | vocab={len(tok)}")
    tok.save(os.path.join(args.ckpt_dir, "tokenizer.pkl"))
    config.vocab_size = len(tok)

    # Dataloaders
    print("\n  Building dataloaders...")
    train_text = read_text_input(args.data[0])
    train_loader, val_loader = create_dataloaders(
        train_text, None, tok,
        seq_len=args.seq_len, batch_size=args.batch_size,
        stride=args.seq_len // 2, num_workers=0,
    )

    # Model
    model = NeuralForge(config)
    print(f"\nModel: {model.count_parameters()/1e6:.1f}M parameters")

    # Trainer
    trainer = Trainer(
        model=model, config=config,
        train_loader=train_loader, val_loader=val_loader,
        checkpoint_dir=args.ckpt_dir,
        compile_model=not args.no_compile,
        model_name=args.name,
        tokenizer=tok,
    )

    trainer.train(num_epochs=args.epochs)
    print(f"\nDone. Model saved to {os.path.join(args.ckpt_dir, args.name + '.pt')}")


if __name__ == "__main__":
    main()