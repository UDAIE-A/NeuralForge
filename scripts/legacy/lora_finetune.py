#!/usr/bin/env python3
"""
LoRA fine-tuning for NeuralForge base models.
Usage:
    python scripts/lora_finetune.py --base-model checkpoints/base_v2.pt --data data/tinystories_100mb.txt --steps 500 --rank 16
"""

import argparse
import os
import sys
import time
import random

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neuralforge.core import NeuralForge, ModelConfig
from neuralforge.tokenizer import BPETokenizer
from neuralforge.training.data import read_text_input
from neuralforge.training import create_dataloaders


class LoRALinear(nn.Module):
    """Low-Rank Adaptation wrapper for a Linear layer."""
    
    def __init__(self, base_layer: nn.Linear, rank: int = 16, alpha: float = 32.0, dropout: float = 0.0):
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        in_features = base_layer.in_features
        out_features = base_layer.out_features
        
        # LoRA matrices: A (in_features, rank), B (rank, out_features)
        self.lora_A = nn.Linear(in_features, rank, bias=False, device=base_layer.weight.device)
        self.lora_B = nn.Linear(rank, out_features, bias=False, device=base_layer.weight.device)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Initialize A with normal, B with zeros (so initial output is 0)
        nn.init.normal_(self.lora_A.weight, std=0.02)
        nn.init.zeros_(self.lora_B.weight)
        
        # Freeze base layer
        for p in base_layer.parameters():
            p.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        lora_out = self.lora_B(self.dropout(self.lora_A(x))) * self.scaling
        return base_out + lora_out


def inject_lora(model: NeuralForge, rank: int = 16, alpha: float = 32.0, 
                target_modules=("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
                dropout: float = 0.0):
    """Replace target linear layers with LoRA versions."""
    lora_layers = []
    
    for name, module in model.named_modules():
        for target in target_modules:
            if name.endswith(target) and isinstance(module, nn.Linear):
                parent_name = name.rsplit(".", 1)[0]
                parent = model.get_submodule(parent_name)
                child_name = name.rsplit(".", 1)[1]
                
                lora_layer = LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout)
                setattr(parent, child_name, lora_layer)
                lora_layers.append((name, lora_layer))
                print(f"  Injected LoRA into {name} (rank={rank}, alpha={alpha})")
    
    return lora_layers


def count_lora_params(model: NeuralForge):
    """Count trainable LoRA parameters."""
    lora_params = 0
    total_params = 0
    for n, p in model.named_parameters():
        total_params += p.numel()
        if 'lora' in n:
            lora_params += p.numel()
    return lora_params, total_params


def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tune NeuralForge model")
    parser.add_argument("--base-model", type=str, required=True, help="Path to base model checkpoint")
    parser.add_argument("--data", type=str, required=True, help="Training data file")
    parser.add_argument("--output", type=str, default="checkpoints/lora_adapter.pt", help="Output adapter path")
    parser.add_argument("--rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--alpha", type=float, default=32.0, help="LoRA alpha")
    parser.add_argument("--dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--steps", type=int, default=500, help="Training steps")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=512, help="Sequence length")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--warmup", type=int, default=50, help="Warmup steps")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--val-split", type=float, default=0.05, help="Validation split")
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        random.seed(args.seed)

    if not torch.cuda.is_available():
        print("ERROR: CUDA required")
        sys.exit(1)

    device = torch.device("cuda")

    # Load base model
    print(f"Loading base model: {args.base_model}")
    checkpoint = torch.load(args.base_model, map_location='cpu', weights_only=False)
    config = checkpoint['config']
    config.device = "cuda"
    tokenizer = checkpoint['tokenizer_obj']

    model = NeuralForge(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    # Inject LoRA
    print(f"\nInjecting LoRA (rank={args.rank}, alpha={args.alpha})...")
    lora_layers = inject_lora(model, rank=args.rank, alpha=args.alpha, dropout=args.dropout)
    
    lora_params, total_params = count_lora_params(model)
    print(f"\nTrainable LoRA params: {lora_params:,} ({lora_params/1e6:.2f}M / {total_params/1e6:.1f}M = {100*lora_params/total_params:.2f}%)")

    # Prepare data
    print(f"\nLoading data: {args.data}")
    text = read_text_input(args.data)
    
    val_size = int(len(text) * args.val_split)
    train_text = text[:-val_size]
    val_text = text[-val_size:]
    
    print(f"Train: {len(train_text):,} chars, Val: {len(val_text):,} chars")

    train_loader, val_loader = create_dataloaders(
        train_text, val_text, tokenizer,
        seq_len=args.seq_len, batch_size=args.batch_size,
        stride=args.seq_len // 2, num_workers=0,
    )

    # Only optimize LoRA params
    lora_params_list = [p for n, p in model.named_parameters() if 'lora' in n]
    optimizer = torch.optim.AdamW(lora_params_list, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    
    # Scheduler
    def get_lr(step):
        if step < args.warmup:
            return args.lr * step / args.warmup
        progress = (step - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr * 0.5 * (1 + math.cos(math.pi * progress))
    
    scaler = torch.amp.GradScaler('cuda')
    model.train()
    
    print(f"\nTraining for {args.steps} steps...")
    print(f"Batches per epoch: {len(train_loader)}")
    print(f"Effective batch size: {args.batch_size * args.grad_accum}")
    print("=" * 60)

    step = 0
    best_val_loss = float('inf')
    train_iter = iter(train_loader)
    start_time = time.time()

    while step < args.steps:
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x = x.to(device)
        y = y.to(device)

        with torch.amp.autocast('cuda'):
            _, loss, _ = model(x, targets=y)
            loss = loss / args.grad_accum

        scaler.scale(loss).backward()

        if (step + 1) % args.grad_accum == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(lora_params_list, 1.0)
            
            # Update LR
            lr = get_lr(step)
            for g in optimizer.param_groups:
                g['lr'] = lr
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # Logging
        if step % 25 == 0:
            elapsed = time.time() - start_time
            tok_s = (step + 1) * args.batch_size * args.seq_len * args.grad_accum / elapsed
            current_lr = optimizer.param_groups[0]['lr']
            print(f"  Step {step:5d}/{args.steps} | Loss: {loss.item()*args.grad_accum:.4f} | LR: {current_lr:.2e} | {tok_s:,.0f} tok/s | GPU: {torch.cuda.memory_allocated()/1024**3:.1f}GB")

        # Validation
        if step % 100 == 0 and step > 0:
            model.eval()
            val_loss = 0
            val_batches = 0
            with torch.no_grad():
                for vx, vy in val_loader:
                    vx, vy = vx.to(device), vy.to(device)
                    with torch.amp.autocast('cuda'):
                        _, vl, _ = model(vx, targets=vy)
                    val_loss += vl.item()
                    val_batches += 1
                    if val_batches >= 50:  # quick val
                        break
            
            val_loss /= val_batches
            print(f"  >> Val Loss: {val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # Save LoRA adapter
                adapter_state = {n: p.detach().cpu() for n, p in model.named_parameters() if 'lora' in n}
                torch.save({
                    'adapter_state': adapter_state,
                    'config': {'rank': args.rank, 'alpha': args.alpha, 'target_modules': list(args.target_modules) if hasattr(args, 'target_modules') else ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]},
                    'step': step,
                    'val_loss': val_loss,
                }, args.output)
                print(f"  >> Saved adapter to {args.output}")
            
            model.train()

        step += 1

    total_time = time.time() - start_time
    print("=" * 60)
    print(f"Done in {total_time/60:.1f} min | Best val loss: {best_val_loss:.4f}")
    print(f"Adapter saved to: {args.output}")


if __name__ == "__main__":
    import math
    main()