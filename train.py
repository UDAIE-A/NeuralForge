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
from neuralforge.training.data import read_text_input


def load_resume_artifacts(resume_path: str):
    """Load config + tokenizer from a saved checkpoint for true continuation."""
    checkpoint = torch.load(resume_path, map_location='cpu', weights_only=False)
    config = checkpoint['config']
    if checkpoint.get('tokenizer_obj') is not None:
        tokenizer = checkpoint['tokenizer_obj']
    else:
        tokenizer_path = os.path.join(os.path.dirname(resume_path), 'tokenizer.pkl')
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError("tokenizer.pkl not found next to resume checkpoint")
        import pickle
        with open(tokenizer_path, 'rb') as f:
            tok_data = pickle.load(f)
        if 'char_to_id' in tok_data:
            tokenizer = CharTokenizer.load(tokenizer_path)
        else:
            tokenizer = BPETokenizer.load(tokenizer_path)
    return config, tokenizer


def main():
    parser = argparse.ArgumentParser(description='Train NeuralForge model')
    parser.add_argument('--preset', type=str, default='tiny',
                       choices=['tiny', 'small', 'base', 'large', 'xl', 'xxl'],
                       help='Model size preset')
    parser.add_argument('--data', type=str, nargs='+', required=True,
                       help='Path(s) to training data (text file)')
    parser.add_argument('--val-data', type=str, nargs='+', default=None,
                       help='Path(s) to validation data (optional)')
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
    parser.add_argument('--no-compile', action='store_true',
                       help='Disable torch.compile (auto-skipped if Triton is missing)')
    parser.add_argument('--name', type=str, default=None,
                       help='Model name for checkpoint files (default: preset name). '
                            'Produces <name>.pt (final), <name>_train.pt, <name>_best.pt')

    args = parser.parse_args()
    model_name = args.name or args.preset
    
    # GPU check
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. This model requires an NVIDIA GPU for training.")
        sys.exit(1)
    
    # Read training data
    train_input = args.data if len(args.data) > 1 else args.data[0]
    val_input = None
    if args.val_data:
        val_input = args.val_data if len(args.val_data) > 1 else args.val_data[0]
    train_text = read_text_input(train_input)

    if args.resume:
        print(f"\n  Resuming from checkpoint: {args.resume}")
        config, tokenizer = load_resume_artifacts(args.resume)
        config.device = "cuda"
        print("  Loaded model config and tokenizer from checkpoint")
    else:
        # Get model config
        config = ModelConfig.from_preset(args.preset)
        config.max_seq_len = args.seq_len
        config.batch_size = args.batch_size
        config.learning_rate = args.lr
        config.device = "cuda"

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

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    tokenizer.save(os.path.join(args.checkpoint_dir, 'tokenizer.pkl'))
    
    # Header
    print()
    print("=" * 60)
    print("  NEURALFORGE TRAINING")
    print("=" * 60)
    print(f"  Preset:      {args.preset}{' (resume config loaded)' if args.resume else ''}")
    print(f"  Parameters:  {config.num_parameters / 1e6:.2f}M")
    tok_kind = 'char' if isinstance(tokenizer, CharTokenizer) else 'BPE'
    print(f"  Vocab:       {config.vocab_size} ({tok_kind})")
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
        train_input, val_input, tokenizer,
        seq_len=config.max_seq_len, batch_size=config.batch_size
    )
    
    # Train
    trainer = Trainer(
        model=model, config=config,
        train_loader=train_loader, val_loader=val_loader,
        checkpoint_dir=args.checkpoint_dir,
        compile_model=not args.no_compile,
        model_name=model_name,
        tokenizer=tokenizer,
    )
    
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    trainer.train(num_epochs=args.epochs)

    print(f"\n  Generate: python generate.py --checkpoint {args.checkpoint_dir}/{model_name}.pt")


if __name__ == '__main__':
    main()
