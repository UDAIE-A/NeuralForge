#!/usr/bin/env python3
"""
NeuralForge Training Script

Usage:
    python train.py --preset tiny --data data/corpus.txt
    python train.py --preset small --data data/corpus.txt --epochs 20
    python train.py --preset small --data data/corpus.txt --char  # fast char-level
"""

import argparse
import json
import os
import random
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


def apply_config_file(parser, args, argv):
    """Fold a JSON run config into `args` without overriding explicit flags.

    Precedence: explicit command-line flag > config file > argparse default.
    This is what replaced the family of near-identical scripts/train_*.py
    files - each of those was really just a frozen set of these values, and
    every bug fixed in the shared trainer had to be fixed in all of them.
    """
    if not args.config:
        return args
    with open(args.config, encoding='utf-8') as f:
        cfg = json.load(f)

    valid = {a.dest for a in parser._actions}
    unknown = set(cfg) - valid
    if unknown:
        raise SystemExit(f"Unknown key(s) in {args.config}: {sorted(unknown)}")

    # Which dests the user typed explicitly on the command line.
    explicit = set()
    for action in parser._actions:
        for opt in action.option_strings:
            if any(a == opt or a.startswith(opt + '=') for a in argv):
                explicit.add(action.dest)

    applied = []
    for key, value in cfg.items():
        if key in explicit or key == 'config':
            continue
        setattr(args, key, value)
        applied.append(key)
    print(f"  Config: {args.config} -> {', '.join(sorted(applied))}")
    return args


def main():
    parser = argparse.ArgumentParser(description='Train NeuralForge model')
    parser.add_argument('--config', type=str, default=None,
                       help='JSON file of run settings (see configs/). Any key '
                            'matching a flag below; explicit flags still win')
    parser.add_argument('--preset', type=str, default='tiny',
                       choices=['tiny', 'small', 'base', 'large', 'xl', 'xxl'],
                       help='Model size preset')
    parser.add_argument('--data', type=str, nargs='+', default=None,
                       help='Path(s) to training data (text file). Required, but '
                            'may come from --config instead')
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
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducible runs (torch, cuda, python, numpy)')
    parser.add_argument('--grad-accum', type=int, default=1,
                       help='Gradient accumulation steps (simulates a larger batch size)')
    parser.add_argument('--num-workers', type=int, default=None,
                       help='DataLoader workers (default: 0 on Windows, 8 elsewhere)')
    parser.add_argument('--warmup-steps', type=int, default=None,
                       help='LR warmup steps (default: adaptive, capped at ~10%% of the run)')
    parser.add_argument('--stride', type=int, default=None,
                       help='Sliding-window stride (default: seq_len, i.e. no overlap). '
                            'A stride below seq_len shows every token multiple times '
                            'per epoch and accelerates memorization')
    parser.add_argument('--dropout', type=float, default=None,
                       help='Dropout probability (default: the preset value, 0.1). '
                            'Raise it when validation loss stalls above training loss')
    parser.add_argument('--early-stopping', type=int, default=None,
                       help='Stop after N consecutive evaluations with no validation '
                            'improvement (default: off)')
    parser.add_argument('--val-fraction', type=float, default=0.05,
                       help='Fraction of the corpus held out for validation when '
                            '--val-data is not given (default: 0.05)')
    parser.add_argument('--save-interval', type=int, default=None,
                       help='Checkpoint save interval in optimizer steps '
                            '(default: adaptive, ~30 saves per run)')
    parser.add_argument('--max-steps', type=int, default=None,
                       help='Hard cap on optimizer steps, regardless of --epochs. '
                            'Lets a run be sized in tokens rather than passes')

    arch = parser.add_argument_group(
        'architecture overrides',
        'Override individual preset dimensions to define a custom model size')
    arch.add_argument('--d-model', type=int, default=None)
    arch.add_argument('--n-heads', type=int, default=None)
    arch.add_argument('--n-layers', type=int, default=None)
    arch.add_argument('--d-ff', type=int, default=None)

    args = apply_config_file(parser, parser.parse_args(), sys.argv[1:])
    if args.data is None:
        parser.error('--data is required (pass it directly or in --config)')
    if isinstance(args.data, str):
        args.data = [args.data]
    model_name = args.name or args.preset

    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        random.seed(args.seed)
        try:
            import numpy as np
            np.random.seed(args.seed)
        except ImportError:
            pass
        print(f"  Seed: {args.seed}")
    
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
        # The architecture is fixed by the checkpoint, but these are training
        # knobs the user can legitimately change on a continuation run. They
        # used to be accepted and silently ignored.
        for flag, attr, given in (
            ('--lr', 'learning_rate', args.lr != parser.get_default('lr')),
            ('--batch-size', 'batch_size', args.batch_size != parser.get_default('batch_size')),
            ('--dropout', 'dropout', args.dropout is not None),
        ):
            if given:
                value = getattr(args, attr if attr != 'learning_rate' else 'lr')
                setattr(config, attr, value)
                print(f"  Override from {flag}: {attr} = {value}")
        if args.seq_len != parser.get_default('seq_len'):
            print(f"  Note: --seq-len is fixed at {config.max_seq_len} by the "
                  f"checkpoint's RoPE tables; ignoring {args.seq_len}")
    else:
        # Get model config
        config = ModelConfig.from_preset(args.preset)
        config.max_seq_len = args.seq_len
        config.batch_size = args.batch_size
        config.learning_rate = args.lr
        config.device = "cuda"
        if args.dropout is not None:
            config.dropout = args.dropout
        # Architecture overrides let a custom size (e.g. d_model=864,
        # n_layers=15) be expressed as a config file instead of its own script.
        for flag, attr in (('d_model', 'd_model'), ('n_heads', 'n_heads'),
                           ('n_layers', 'n_layers'), ('d_ff', 'd_ff')):
            value = getattr(args, flag)
            if value is not None:
                setattr(config, attr, value)
        config.__post_init__()   # re-validate d_model % n_heads and d_head

        # Train tokenizer
        if args.char:
            print("\n  Using character-level tokenizer (instant)...")
            tokenizer = CharTokenizer()
            tokenizer.train(train_text, verbose=True)
        else:
            print("\n  Training BPE tokenizer...")
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
    print(f"  Batch size:  {args.batch_size}"
          f"{f' x {args.grad_accum} grad-accum' if args.grad_accum > 1 else ''}")
    print(f"  LR:          {config.learning_rate}")
    print(f"  Dropout:     {config.dropout}")
    print(f"  Epochs:      {args.epochs}")
    print(f"  GPU:         {torch.cuda.get_device_name(0)}")
    print(f"  Data:        {len(train_text):,} chars / {len(train_text.split()):,} words")
    print("=" * 60)
    print()
    
    # Create model
    model = NeuralForge(config)
    
    # Create dataloaders. Pass the already-read text so the corpus is only
    # read from disk once (read_text_input is a no-op on plain strings).
    # stride defaults to seq_len: no overlap, so each token is seen once per
    # epoch. The old seq_len/2 default silently doubled every token's exposure.
    stride = args.stride if args.stride else config.max_seq_len
    train_loader, val_loader = create_dataloaders(
        train_text, val_input, tokenizer,
        seq_len=config.max_seq_len, batch_size=config.batch_size,
        stride=stride,
        num_workers=args.num_workers if args.num_workers is not None else (0 if os.name == 'nt' else 8),
        val_fraction=args.val_fraction,
    )

    # Train
    trainer = Trainer(
        model=model, config=config,
        train_loader=train_loader, val_loader=val_loader,
        checkpoint_dir=args.checkpoint_dir,
        compile_model=not args.no_compile,
        gradient_accumulation_steps=args.grad_accum,
        save_interval=args.save_interval,
        model_name=model_name,
        tokenizer=tokenizer,
        warmup_steps=args.warmup_steps,
        early_stopping_patience=args.early_stopping,
        max_steps=args.max_steps,
    )
    
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    trainer.train(num_epochs=args.epochs)

    print(f"\n  Generate: python generate.py --checkpoint {args.checkpoint_dir}/{model_name}.pt")


if __name__ == '__main__':
    main()
