#!/usr/bin/env python3
"""
Full continued pretraining for medium186 - 100M tokens target.
"""

import torch
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
    # Config
    seq_len = 512
    batch_size = 4
    grad_accum = 8
    lr = 1.5e-4
    warmup_steps = 500
    weight_decay = 0.01
    epochs = 1
    device = torch.device("cuda")
    max_steps = 10000  # ~100M tokens / (4*512*8) = ~6100 steps per epoch
    
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
    config.warmup_steps = warmup_steps
    config.weight_decay = weight_decay
    config.batch_size = batch_size
    
    model = NeuralForge(config)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.train()
    
    print(f"Model: {model.count_parameters()/1e6:.1f}M params")
    
    # Data - full 100MB
    print("Loading data...")
    text = read_text_input('data/pretrain_100mb.txt')
    print(f"Training on {len(text)/1e6:.1f}M chars")
    
    train_loader, val_loader = create_dataloaders(
        text, None, tokenizer,
        seq_len=seq_len, batch_size=batch_size,
        stride=seq_len // 2, num_workers=0,
    )
    
    optimizer = model.get_optimizer(config)
    
    optimizer_steps_per_epoch = len(train_loader) // grad_accum
    total_steps = min(max_steps, optimizer_steps_per_epoch * epochs)
    
    def get_lr(step):
        if step < warmup_steps:
            return lr * step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return lr * 0.5 * (1 + math.cos(math.pi * progress))
    
    scaler = torch.amp.GradScaler('cuda')
    
    print(f"Total optimizer steps: {total_steps}")
    print(f"Batches per epoch: {len(train_loader)}")
    print(f"Optimizer steps per epoch: {optimizer_steps_per_epoch}")
    print("=" * 60)
    
    global_step = 0
    start_time = time.time()
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        epoch_start = time.time()
        
        for batch_idx, (x, y) in enumerate(train_loader):
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
                
                # Logging
                if global_step % 25 == 0:
                    elapsed = time.time() - start_time
                    tok_s = global_step * batch_size * seq_len * grad_accum / elapsed
                    print(f"  Step {global_step:5d}/{total_steps} | Loss: {loss.item()*grad_accum:.4f} | LR: {current_lr:.2e} | {tok_s:,.0f} tok/s | GPU: {torch.cuda.memory_allocated()/1024**3:.1f}GB")
                    sys.stdout.flush()
                
                # Validation
                if global_step % 500 == 0:
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
                            if val_batches >= 20:
                                break
                    val_loss /= max(val_batches, 1)
                    print(f"  >> Val Loss: {val_loss:.4f} | Step: {global_step}")
                    sys.stdout.flush()
                    
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        torch.save({
                            'model_state_dict': model.state_dict(),
                            'config': config,
                            'tokenizer_obj': tokenizer,
                            'meta': {'name': 'medium186_100m', 'step': global_step, 'val_loss': val_loss},
                        }, 'checkpoints/medium186_100m_best.pt')
                        print(f"  >> Best model saved (val_loss={val_loss:.4f})")
                    
                    model.train()
                
                # Checkpoint
                if global_step % 1000 == 0:
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'global_step': global_step,
                        'config': config,
                    }, 'checkpoints/medium186_100m_train.pt')
                    print(f"  >> Checkpoint saved at step {global_step}")
                
                if global_step >= total_steps:
                    break
        
        if global_step >= total_steps:
            break
    
    # Final save
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'tokenizer_obj': tokenizer,
        'meta': {
            'name': 'medium186_100m',
            'epochs': epochs,
            'total_steps': global_step,
            'total_tokens': global_step * batch_size * seq_len * grad_accum,
            'best_val_loss': best_val_loss,
        },
    }, 'checkpoints/medium186_100m.pt')
    
    total_time = time.time() - start_time
    print(f"\nDone in {total_time/60:.1f} min | Steps: {global_step} | Tokens: {global_step * batch_size * seq_len * grad_accum:,} | Best val: {best_val_loss:.4f}")
    print(f"Saved to checkpoints/medium186_100m.pt")


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
    
    # Config
    seq_len = 512
    batch_size = 4
    grad_accum = 8
    lr = 1.5e-4
    warmup_steps = 500
    weight_decay = 0.01
    epochs = 1
    max_steps = 10000
    device = torch.device("cuda")
    start_time = time.time()
    
    # Load checkpoint
    print("Loading checkpoint...")
    ckpt = torch.load('checkpoints/medium186_v1.pt', map_location='cpu', weights_only=False)
    config = ckpt['config']
    tokenizer = ckpt['tokenizer_obj']
    vocab_size = len(tokenizer)
    
    config.vocab_size = vocab_size
    config.max_seq_len = 512
    config.device = "cuda"
    config.learning_rate = lr
    config.warmup_steps = 500
    config.weight_decay = weight_decay
    config.batch_size = 4
    
    model = NeuralForge(config)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.train()
    
    print(f"Model: {model.count_parameters()/1e6:.1f}M params")
    
    text = read_text_input('data/pretrain_100mb.txt')
    print(f"Training on {len(text)/1e6:.1f}M chars")
    
    train_loader, val_loader = create_dataloaders(
        text, None, tokenizer,
        seq_len=512, batch_size=4,
        stride=256, num_workers=0,
    )
    
    optimizer = model.get_optimizer(config)
    
    optimizer_steps_per_epoch = len(train_loader) // 8
    total_steps = min(max_steps, optimizer_steps_per_epoch * epochs)
    
    def get_lr(step):
        if step < warmup_steps:
            return lr * step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return lr * 0.5 * (1 + math.cos(math.pi * progress))
    
    scaler = torch.amp.GradScaler('cuda')
    
    print(f"Total optimizer steps: {total_steps}")
    print(f"Batches per epoch: {len(train_loader)}")
    print(f"Optimizer steps per epoch: {optimizer_steps_per_epoch}")
    print("=" * 60)
    
    global_step = 0
    start_time = time.time()
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        for batch_idx, (x, y) in enumerate(train_loader):
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
                
                if global_step % 25 == 0:
                    elapsed = time.time() - start_time
                    tok_s = global_step * batch_size * seq_len * grad_accum / elapsed
                    print(f"  Step {global_step:5d}/{total_steps} | Loss: {loss.item()*grad_accum:.4f} | LR: {current_lr:.2e} | {tok_s:,.0f} tok/s | GPU: {torch.cuda.memory_allocated()/1024**3:.1f}GB")
                    sys.stdout.flush()
                
                if global_step % 500 == 0:
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
                            if val_batches >= 20:
                                break
                    val_loss /= max(val_batches, 1)
                    print(f"  >> Val Loss: {val_loss:.4f} | Step: {global_step}")
                    sys.stdout.flush()
                    
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        torch.save({
                            'model_state_dict': model.state_dict(),
                            'config': config,
                            'tokenizer_obj': tokenizer,
                            'meta': {'name': 'medium186_100m', 'step': global_step, 'val_loss': val_loss},
                        }, 'checkpoints/medium186_100m_best.pt')
                        print(f"  >> Best model saved (val_loss={val_loss:.4f})")
                    
                    model.train()
                
                if global_step % 1000 == 0:
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'global_step': global_step,
                        'config': config,
                    }, 'checkpoints/medium186_100m_train.pt')
                    print(f"  >> Checkpoint saved at step {global_step}")
                
                if global_step >= total_steps:
                    break
        
        if global_step >= total_steps:
            break
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'tokenizer_obj': tokenizer,
        'meta': {
            'name': 'medium186_100m',
            'epochs': epochs,
            'total_steps': global_step,
            'total_tokens': global_step * batch_size * seq_len * grad_accum,
            'best_val_loss': best_val_loss,
        },
    }, 'checkpoints/medium186_100m.pt')
    
    total_time = time.time() - start_time
    print(f"\nDone in {total_time/60:.1f} min | Steps: {global_step} | Tokens: {global_step * batch_size * seq_len * grad_accum:,} | Best val: {best_val_loss:.4f}")


if __name__ == "__main__":
    import math
    import torch
    import os
    import sys
    import time
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from neuralforge.core import NeuralForge, ModelConfig
    from neuralforge.tokenizer import BPETokenizer
    from neuralforge.training.data import read_text_input
    from neuralforge.training import create_dataloaders
    
    # Config
    seq_len = 512
    batch_size = 4
    grad_accum = 8
    lr = 1.5e-4
    warmup_steps = 500
    weight_decay = 0.01
    epochs = 1
    max_steps = 10000
    device = torch.device("cuda")
    start_time = time.time()
    
    ckpt = torch.load('checkpoints/medium186_v1.pt', map_location='cpu', weights_only=False)
    config = ckpt['config']
    tokenizer = ckpt['tokenizer_obj']
    vocab_size = len(tokenizer)
    
    config.vocab_size = vocab_size
    config.max_seq_len = 512
    config.device = "cuda"
    config.learning_rate = lr
    config.warmup_steps = 500
    config.weight_decay = weight_decay
    config.batch_size = 4
    
    model = NeuralForge(config)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.train()
    
    print(f"Model: {model.count_parameters()/1e6:.1f}M params")
    
    text = read_text_input('data/pretrain_100mb.txt')
    print(f"Training on {len(text)/1e6:.1f}M chars")
    
    train_loader, val_loader = create_dataloaders(
        text, None, tokenizer,
        seq_len=512, batch_size=4,
        stride=256, num_workers=0,
    )
    
    optimizer = model.get_optimizer(config)
    
    optimizer_steps_per_epoch = len(train_loader) // 8
    total_steps = min(max_steps, optimizer_steps_per_epoch * epochs)
    
    def get_lr(step):
        if step < warmup_steps:
            return lr * step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return lr * 0.5 * (1 + math.cos(math.pi * progress))
    
    scaler = torch.amp.GradScaler('cuda')
    
    print(f"Total optimizer steps: {total_steps}")
    print(f"Batches per epoch: {len(train_loader)}")
    print(f"Optimizer steps per epoch: {optimizer_steps_per_epoch}")
    print("=" * 60)
    
    global_step = 0
    start_time = time.time()
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        for batch_idx, (x, y) in enumerate(train_loader):
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
                
                if global_step % 25 == 0:
                    elapsed = time.time() - start_time
                    tok_s = global_step * batch_size * seq_len * grad_accum / elapsed
                    print(f"  Step {global_step:5d}/{total_steps} | Loss: {loss.item()*grad_accum:.4f} | LR: {current_lr:.2e} | {tok_s:,.0f} tok/s | GPU: {torch.cuda.memory_allocated()/1024**3:.1f}GB")
                    sys.stdout.flush()
                
                if global_step % 500 == 0:
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
                            if val_batches >= 20:
                                break
                    val_loss /= max(val_batches, 1)
                    print(f"  >> Val Loss: {val_loss:.4f} | Step: {global_step}")
                    sys.stdout.flush()
                    
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        torch.save({
                            'model_state_dict': model.state_dict(),
                            'config': config,
                            'tokenizer_obj': tokenizer,
                            'meta': {'name': 'medium186_100m', 'step': global_step, 'val_loss': val_loss},
                        }, 'checkpoints/medium186_100m_best.pt')
                        print(f"  >> Best model saved (val_loss={val_loss:.4f})")
                    
                    model.train()
                
                if global_step % 1000 == 0:
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'global_step': global_step,
                        'config': config,
                    }, 'checkpoints/medium186_100m_train.pt')
                    print(f"  >> Checkpoint saved at step {global_step}")
                
                if global_step >= total_steps:
                    break
        
        if global_step >= total_steps:
            break
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'tokenizer_obj': tokenizer,
        'meta': {
            'name': 'medium186_100m',
            'epochs': epochs,
            'total_steps': global_step,
            'total_tokens': global_step * batch_size * seq_len * grad_accum,
            'best_val_loss': best_val_loss,
        },
    }, 'checkpoints/medium186_100m.pt')
    
    total_time = time.time() - start_time
    print(f"\nDone in {total_time/60:.1f} min | Steps: {global_step} | Tokens: {global_step * batch_size * seq_len * grad_accum:,} | Best val: {best_val_loss:.4f}")