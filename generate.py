#!/usr/bin/env python3
"""
NeuralForge Text Generation Script

Usage:
    python generate.py --checkpoint checkpoints/best_model.pt --prompt "The "
    python generate.py --checkpoint checkpoints/best_model.pt --interactive
"""

import argparse
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from neuralforge.core import NeuralForge, ModelConfig
from neuralforge.tokenizer import BPETokenizer
from neuralforge.tokenizer.char_tokenizer import CharTokenizer


def main():
    parser = argparse.ArgumentParser(description='Generate text with NeuralForge')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--tokenizer', type=str, default=None,
                       help='Path to tokenizer (default: checkpoint_dir/tokenizer.pkl)')
    parser.add_argument('--prompt', type=str, default="The ",
                       help='Prompt for text generation')
    parser.add_argument('--max-tokens', type=int, default=200,
                       help='Maximum tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8,
                       help='Sampling temperature')
    parser.add_argument('--top-k', type=int, default=50,
                       help='Top-k sampling')
    parser.add_argument('--interactive', action='store_true',
                       help='Interactive mode')
    
    args = parser.parse_args()
    
    # GPU check
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. This model requires an NVIDIA GPU.")
        sys.exit(1)
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    device = torch.device("cuda")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    # Get config
    config = checkpoint['config']
    config.device = "cuda"
    
    # Load tokenizer
    tokenizer_path = args.tokenizer or os.path.join(
        os.path.dirname(args.checkpoint), 'tokenizer.pkl'
    )
    print(f"Loading tokenizer: {tokenizer_path}")
    
    # Try char tokenizer first, fall back to BPE
    try:
        tokenizer = CharTokenizer.load(tokenizer_path)
        print("  (character-level tokenizer)")
    except (KeyError, Exception):
        tokenizer = BPETokenizer.load(tokenizer_path)
        print("  (BPE tokenizer)")
    
    # Create model
    config.vocab_size = len(tokenizer)
    model = NeuralForge(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"\nModel loaded: {model.count_parameters():,} parameters")
    print(f"Device: {device}")
    print(f"{'='*60}\n")
    
    if args.interactive:
        # Interactive mode
        print("Interactive mode. Type your prompt and press Enter.")
        print("Type 'quit' to exit.\n")
        
        while True:
            try:
                prompt = input("You: ")
                if prompt.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not prompt:
                    continue
                
                # Encode
                tokens = tokenizer.encode(prompt, add_special_tokens=False)
                x = torch.tensor([tokens], dtype=torch.long, device=device)
                
                # Generate
                with torch.no_grad():
                    generated = model.generate(
                        x,
                        max_new_tokens=args.max_tokens,
                        temperature=args.temperature,
                        top_k=args.top_k
                    )
                
                # Decode
                response = tokenizer.decode(generated[0].tolist())
                print(f"NeuralForge: {response}\n")
                
            except KeyboardInterrupt:
                print("\nExiting...")
                break
    else:
        # Single generation
        print(f"Prompt: {args.prompt}")
        print(f"{'-'*60}")
        
        # Encode
        tokens = tokenizer.encode(args.prompt, add_special_tokens=False)
        x = torch.tensor([tokens], dtype=torch.long, device=device)
        
        # Generate
        with torch.no_grad():
            generated = model.generate(
                x,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k
            )
        
        # Decode
        text = tokenizer.decode(generated[0].tolist())
        # Clean up non-printable characters
        text = ''.join(c if c.isprintable() or c in '\n\r\t' else ' ' for c in text)
        print(f"Generated: {text}")


if __name__ == '__main__':
    main()
