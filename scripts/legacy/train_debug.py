#!/usr/bin/env python3
"""
Minimal continued pretraining - verbose debugging.
"""

import torch
import torch.nn as nn
import math
import time
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neuralforge.core import NeuralForge, ModelConfig
from neuralforge.tokenizer import BPETokenizer
from neuralforge.training.data import read_text_input
from neuralforge.training import create_dataloaders


def main():
    seq_len = 512
    batch_size = 2
    grad_accum = 16
    lr = 1.5e-4
    warmup_steps = 100
    epochs = 1
    device = torch.device("cuda")
    
    # Load checkpoint
    print("Loading checkpoint...")
    ckpt = torch.load('checkpoints/medium186_v1.pt', map_location='cpu', weights_only=False)
    config = ckpt['config']
    tokenizer = ckpt['tokenizer_obj']
    vocab_size = len(tokenizer)
    
    config.vocab_size = vocab_size
    config.max_seq_len = seq_len
    config.device = "cuda"
    config.learning_rate = lr
    config.warmup_steps = 100
    config.weight_decay = 0.01
    
    model = NeuralForge(config)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.train()
    
    print(f"Model: {model.count_parameters()/1e6:.1f}M params")
    
    # Data - SMALL subset for testing
    print("Loading data...")
    text = read_text_input('data/pretrain_100mb.txt')
    # Use only first 10M chars for quick test
    text = text[:10_000_000]
    print(f"Using {len(text)/1e6:.1f}M chars for test")
    
    train_loader, val_loader = create_dataloaders(
        text, None, tokenizer,
        seq_len=512, batch_size=2,
        stride=256, num_workers=0,
    )
    
    # Optimizer
    optimizer = model.get_optimizer(config)
    
    # Scheduler
    optimizer_steps_per_epoch = len(train_loader) // grad_accum
    total_steps = optimizer_steps_per_epoch * 1
    
    def get_lr(step):
        if step < 100:
            return lr * step / 100
        progress = (step - 100) / max(1, total_steps - 100)
        return lr * 0.5 * (1 + math.cos(math.pi * progress))
    
    scaler = torch.amp.GradScaler('cuda')
    
    print(f"Total optimizer steps: {total_steps}")
    print(f"Batches per epoch: {len(train_loader)}")
    print("=" * 60)
    
    global_step = 0
    start_time = time.time()
    
    for epoch in range(1):
        print(f"\nEpoch {epoch+1}/1")
        epoch_start = time.time()
        
        for batch_idx, (x, y) in enumerate(train_loader):
            batch_start = time.time()
            
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            
            with torch.amp.autocast('cuda'):
                logits, loss, _ = model(x, targets=y)
                loss = loss / grad_accum
            
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                current_lr = get_lr(global_step)
                for g in optimizer.param_groups:
                    g['lr'] = current_lr
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                
                # Logging EVERY step for debugging
                elapsed = time.time() - start_time
                tok_s = global_step * 2 * 512 * 16 / max(elapsed, 0.001)
                print(f"Step {global_step:4d} | Loss: {loss.item()*grad_accum:.4f} | LR: {current_lr:.2e} | {tok_s:,.0f} tok/s | GPU: {torch.cuda.memory_allocated()/1024**3:.1f}GB")
                sys.stdout.flush()
                
                # Save checkpoint every 50 steps
                if global_step % 50 == 0:
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'global_step': global_step,
                        'config': config,
                    }, 'checkpoints/medium186_debug.pt')
                    print(f"  >> Checkpoint saved at step {global_step}")
                
                if global_step >= 200:  # Stop after 200 steps for testing
                    print("Reached 200 steps, stopping test")
                    break
            
            if global_step >= 200:
                break
        
        if global_step >= 200:
            break
    
    print(f"\nDone: {global_step} steps in {(time.time()-start_time)/60:.1f} min")
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'tokenizer_obj': ckpt['tokenizer_obj'],
    }, 'checkpoints/medium186_test.pt')


if __name__ == "__main__":
    import math
    import sys
    start_time = time.time()
    ckpt = torch.load('checkpoints/medium186_v1.pt', map_location='cpu', weights_only=False)
    config = ckpt['config']
    tokenizer = ckpt['tokenizer_obj']
    vocab_size = len(tokenizer)
    
    config.vocab_size = vocab_size
    config.max_seq_len = 512
    config.device = "cuda"
    config.learning_rate = 1.5e-4
    config.warmup_steps = 100
    config.weight_decay = 0.01
    config.batch_size = 2
    
    model = NeuralForge(config)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to('cuda')
    model.train()
    
    text = read_text_input('data/pretrain_100mb.txt')[:10_000_000]
    train_loader, _ = create_dataloaders(
        text, None, tokenizer,
        seq_len=512, batch_size=2,
        stride=256, num_workers=0,
    )
    
    optimizer = model.get_optimizer(config)
    
    total_steps = len(train_loader) // 16
    def get_lr(step):
        if step < 100:
            return 1.5e-4 * step / 100
        progress = (step - 100) / max(1, total_steps - 100)
        return 1.5e-4 * 0.5 * (1 + math.cos(math.pi * progress))
    
    scaler = torch.amp.GradScaler('cuda')
    global_step = 0
    start_time = time.time()
    
    for batch_idx, (x, y) in enumerate(train_loader):
        x = x.to('cuda', non_blocking=True)
        y = y.to('cuda', non_blocking=True)
        
        with torch.amp.autocast('cuda'):
            logits, loss, _ = model(x, targets=y)
            loss = loss / 16
        
        scaler.scale(loss).backward()
        
        if (batch_idx + 1) % 16 == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            current_lr = get_lr(global_step)
            for g in optimizer.param_groups:
                g['lr'] = current_lr
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            
            elapsed = time.time() - start_time
            tok_s = global_step * 2 * 512 * 16 / max(elapsed, 0.001)
            print(f"Step {global_step:4d} | Loss: {loss.item()*16:.4f} | LR: {current_lr:.2e} | {tok_s:,.0f} tok/s | GPU: {torch.cuda.memory_allocated()/1024**3:.1f}GB")
            sys.stdout.flush()
            
            if global_step % 50 == 0:
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'global_step': global_step,
                    'config': config,
                }, 'checkpoints/medium186_debug.pt')
                print(f"  >> Checkpoint saved at step {global_step}")
            
            if global_step >= 200:
                break
    
    print(f"\nDone: {global_step} steps")
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'tokenizer_obj': ckpt['tokenizer_obj'],
    }, 'checkpoints/medium186_test.pt')


if __name__ == "__main__":
    import math
    import torch
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from neuralforge.core import NeuralForge, ModelConfig
    from neuralforge.tokenizer import BPETokenizer
    from neuralforge.training.data import read_text_input
    from neuralforge.training import create_dataloaders
    main()