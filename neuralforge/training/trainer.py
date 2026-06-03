"""
Training loop for NeuralForge models.
"""

import os
import time
import math
import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from torch.utils.data import DataLoader

from ..core.model import NeuralForge
from ..core.config import ModelConfig


class CosineScheduleWithWarmup:
    """Learning rate scheduler with cosine decay and warmup."""
    
    def __init__(self, optimizer, warmup_steps: int, max_steps: int, min_lr: float = 1e-6):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr = min_lr
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.step_num = 0
    
    def step(self):
        self.step_num += 1
        lr = self._get_lr()
        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            param_group['lr'] = lr * base_lr
    
    def _get_lr(self) -> float:
        if self.step_num < self.warmup_steps:
            return self.step_num / self.warmup_steps
        progress = (self.step_num - self.warmup_steps) / (self.max_steps - self.warmup_steps)
        progress = min(progress, 1.0)
        return self.min_lr + 0.5 * (1.0 - self.min_lr) * (1.0 + math.cos(math.pi * progress))


class Trainer:
    """
    Trainer for NeuralForge models.
    
    Handles the full training loop with:
    - Mixed precision training
    - Gradient accumulation
    - Learning rate scheduling
    - Checkpointing
    - Logging
    """
    
    def __init__(
        self,
        model: NeuralForge,
        config: ModelConfig,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        checkpoint_dir: str = "checkpoints",
        log_interval: int = 10,
        eval_interval: int = 500,
        save_interval: int = 1000,
        gradient_accumulation_steps: int = 1,
    ):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.checkpoint_dir = checkpoint_dir
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.save_interval = save_interval
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        # Setup device
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print(f"Using device: {self.device}")
        
        # Setup optimizer
        self.optimizer = model.get_optimizer(config)
        
        # Setup scheduler
        total_steps = len(train_loader) * config.warmup_steps  # Assuming some epochs
        self.scheduler = CosineScheduleWithWarmup(
            self.optimizer,
            warmup_steps=config.warmup_steps,
            max_steps=total_steps
        )
        
        # Mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if self.device.type == 'cuda' else None
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Training state
        self.global_step = 0
        self.best_val_loss = float('inf')
    
    def train_epoch(self, epoch: int):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        start_time = time.time()
        
        self.optimizer.zero_grad()
        
        for batch_idx, (x, y) in enumerate(self.train_loader):
            x = x.to(self.device)
            y = y.to(self.device)
            
            # Forward pass with mixed precision
            with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                logits, loss, _ = self.model(x, targets=y)
                loss = loss / self.gradient_accumulation_steps
            
            # Backward pass
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            total_loss += loss.item() * self.gradient_accumulation_steps
            
            # Gradient accumulation step
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1
                
                # Logging
                if self.global_step % self.log_interval == 0:
                    avg_loss = total_loss / (batch_idx + 1)
                    lr = self.scheduler._get_lr() * self.scheduler.base_lrs[0]
                    elapsed = time.time() - start_time
                    tokens_per_sec = (batch_idx + 1) * x.size(0) * x.size(1) / elapsed
                    
                    print(f"Step {self.global_step} | Epoch {epoch} | "
                          f"Loss: {avg_loss:.4f} | LR: {lr:.2e} | "
                          f"Tokens/s: {tokens_per_sec:.0f}")
                
                # Evaluation
                if self.global_step % self.eval_interval == 0 and self.val_loader:
                    val_loss = self.evaluate()
                    print(f"Step {self.global_step} | Validation Loss: {val_loss:.4f}")
                    
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.save_checkpoint(f"best_model.pt")
                    self.model.train()
                
                # Save checkpoint
                if self.global_step % self.save_interval == 0:
                    self.save_checkpoint(f"checkpoint_{self.global_step}.pt")
        
        return total_loss / len(self.train_loader)
    
    @torch.no_grad()
    def evaluate(self) -> float:
        """Evaluate on validation set."""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        for x, y in self.val_loader:
            x = x.to(self.device)
            y = y.to(self.device)
            
            with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                _, loss, _ = self.model(x, targets=y)
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / max(num_batches, 1)
    
    def train(self, num_epochs: int = 10):
        """Full training loop."""
        print(f"\nStarting training for {num_epochs} epochs")
        print(f"Model parameters: {self.model.count_parameters():,}")
        print(f"Training samples: {len(self.train_loader.dataset):,}")
        print(f"Batch size: {self.train_loader.batch_size}")
        print(f"Gradient accumulation steps: {self.gradient_accumulation_steps}")
        print("=" * 60)
        
        for epoch in range(1, num_epochs + 1):
            print(f"\nEpoch {epoch}/{num_epochs}")
            print("-" * 60)
            
            train_loss = self.train_epoch(epoch)
            print(f"Epoch {epoch} | Train Loss: {train_loss:.4f}")
            
            # Save at end of epoch
            self.save_checkpoint(f"epoch_{epoch}.pt")
        
        print("\n" + "=" * 60)
        print("Training complete!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': {
                'step_num': self.scheduler.step_num,
            },
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
            'config': self.config,
        }, path)
        print(f"Saved checkpoint: {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.step_num = checkpoint['scheduler_state_dict']['step_num']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        print(f"Loaded checkpoint from step {self.global_step}")
