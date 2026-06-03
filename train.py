#!/usr/bin/env python3
"""
NeuralForge Training Script

Usage:
    python train.py --preset tiny --data data/corpus.txt
    python train.py --preset small --data data/corpus.txt --epochs 20
"""

import argparse
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from neuralforge.core import ModelConfig, NeuralForge
from neuralforge.tokenizer import BPETokenizer
from neuralforge.training import Trainer, create_dataloaders


def generate_sample_text(model, tokenizer, prompt="The ", max_tokens=100):
    """Generate sample text during training."""
    model.eval()
    device = next(model.parameters()).device
    
    # Encode prompt
    tokens = tokenizer.encode(prompt, add_special_tokens=False)
    x = torch.tensor([tokens], dtype=torch.long, device=device)
    
    # Generate
    with torch.no_grad():
        generated = model.generate(x, max_new_tokens=max_tokens, temperature=0.8, top_k=50)
    
    # Decode
    text = tokenizer.decode(generated[0].tolist())
    return text


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
                       help='Tokenizer vocabulary size')
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
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Get model config
    config = ModelConfig.from_preset(args.preset)
    config.vocab_size = args.vocab_size
    config.max_seq_len = args.seq_len
    config.batch_size = args.batch_size
    config.learning_rate = args.lr
    config.device = args.device
    
    print(f"\n{'='*60}")
    print(f"NeuralForge Training")
    print(f"{'='*60}")
    print(f"Model preset: {args.preset}")
    print(f"Estimated parameters: {config.num_parameters / 1e6:.2f}M")
    print(f"Vocabulary size: {config.vocab_size}")
    print(f"Sequence length: {config.max_seq_len}")
    print(f"Batch size: {config.batch_size}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Device: {config.device}")
    print(f"{'='*60}\n")
    
    # Train tokenizer
    print("Training tokenizer...")
    tokenizer = BPETokenizer()
    
    # Read training data for tokenizer
    if os.path.exists(args.data):
        with open(args.data, 'r', encoding='utf-8') as f:
            train_text = f.read()
    else:
        train_text = args.data
    
    tokenizer.train(train_text, vocab_size=config.vocab_size, verbose=True)
    tokenizer.save(os.path.join(args.checkpoint_dir, 'tokenizer.pkl'))
    print(f"Tokenizer saved to {args.checkpoint_dir}/tokenizer.pkl\n")
    
    # Update config with actual tokenizer vocab size
    config.vocab_size = len(tokenizer)
    
    # Create model
    print("Creating model...")
    model = NeuralForge(config)
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        args.data,
        args.val_data,
        tokenizer,
        seq_len=config.max_seq_len,
        batch_size=config.batch_size
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        checkpoint_dir=args.checkpoint_dir,
    )
    
    # Resume if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Train
    trainer.train(num_epochs=args.epochs)
    
    print(f"\nTraining complete! Model saved to {args.checkpoint_dir}/")
    print(f"To generate text, use: python generate.py --checkpoint {args.checkpoint_dir}/best_model.pt")


if __name__ == '__main__':
    import torch  # Import here to avoid circular import
    main()
