#!/usr/bin/env python3
"""
NeuralForge Training Script

Usage:
    python train.py --preset tiny --data data/corpus.txt
    python train.py --preset small --data data/corpus.txt --epochs 20
    python train.py --preset small --data data/corpus.txt --char  # fast char-level
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from neuralforge.core import ModelConfig, NeuralForge
from neuralforge.tokenizer import BPETokenizer
from neuralforge.tokenizer.char_tokenizer import CharTokenizer
from neuralforge.training import Trainer, create_dataloaders


def main():
    parser = argparse.ArgumentParser(description='Train NeuralForge model')
    parser.add_argument('--preset', type=str, default='tiny',
                       choices=['tiny', 'small', 'base', 'large', 'xl', 'xxl'],
                       help='Model size preset')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to training data (text file)')
    parser.add_argument('--val-data', type=str, default=None,
                       help='Path to validation data (optional)')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--vocab-size', type=int, default=8000,
                       help='Tokenizer vocabulary size (for BPE)')
    parser.add_argument('--seq-len', type=int, default=512,
                       help='Sequence length for training')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-4,
                       help='Learning rate')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint')
    parser.add_argument('--char', action='store_true',
                       help='Use character-level tokenizer (fast, no BPE training)')
    
    args = parser.parse_args()
    
    # GPU check
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. This model requires an NVIDIA GPU for training.")
        sys.exit(1)
    
    # Get model config
    config = ModelConfig.from_preset(args.preset)
    config.max_seq_len = args.seq_len
    config.batch_size = args.batch_size
    config.learning_rate = args.lr
    config.device = "cuda"
    
    # Read training data
    if os.path.exists(args.data):
        with open(args.data, 'r', encoding='utf-8') as f:
            train_text = f.read()
    else:
        train_text = args.data
    
    # Train tokenizer
    if args.char:
        print("\n  Using character-level tokenizer (instant)...")
        tokenizer = CharTokenizer()
        tokenizer.train(train_text, verbose=True)
    else:
        print("\n  Training BPE tokenizer (slow)...")
        tokenizer = BPETokenizer()
        tokenizer.train(train_text, vocab_size=args.vocab_size, verbose=True)
    
    config.vocab_size = len(tokenizer)
    tokenizer.save(os.path.join(args.checkpoint_dir, 'tokenizer.pkl'))
    
    # Header
    print()
    print("=" * 60)
    print("  NEURALFORGE TRAINING")
    print("=" * 60)
    print(f"  Preset:      {args.preset}")
    print(f"  Parameters:  {config.num_parameters / 1e6:.2f}M")
    print(f"  Vocab:       {config.vocab_size} ({'char' if args.char else 'BPE'})")
    print(f"  Seq length:  {config.max_seq_len}")
    print(f"  Batch size:  {args.batch_size}")
    print(f"  LR:          {config.learning_rate}")
    print(f"  Epochs:      {args.epochs}")
    print(f"  GPU:         {torch.cuda.get_device_name(0)}")
    print(f"  Data:        {len(train_text):,} chars / {len(train_text.split()):,} words")
    print("=" * 60)
    print()
    
    # Create model
    model = NeuralForge(config)
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        args.data, args.val_data, tokenizer,
        seq_len=config.max_seq_len, batch_size=config.batch_size
    )
    
    # Train
    trainer = Trainer(
        model=model, config=config,
        train_loader=train_loader, val_loader=val_loader,
        checkpoint_dir=args.checkpoint_dir,
    )
    
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    trainer.train(num_epochs=args.epochs)
    
    print(f"\n  Generate: python generate.py --checkpoint {args.checkpoint_dir}/epoch_{args.epochs}.pt")


if __name__ == '__main__':
    main()
