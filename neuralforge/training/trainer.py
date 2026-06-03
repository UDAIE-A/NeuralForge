"""
Training loop for NeuralForge models with visual dashboard.
"""

import os
import sys
import time
import math
import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List
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


def format_time(seconds: float) -> str:
    """Format seconds into human readable time."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        m = int(seconds // 60)
        s = int(seconds % 60)
        return f"{m}m {s}s"
    else:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        return f"{h}h {m}m"


def get_gpu_stats() -> Dict[str, float]:
    """Get GPU memory and utilization stats."""
    if not torch.cuda.is_available():
        return {}
    
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(', ')
            return {
                'gpu_util': float(parts[0]),
                'mem_used': float(parts[1]),
                'mem_total': float(parts[2]),
                'temp': float(parts[3]),
            }
    except Exception:
        pass
    
    # Fallback to PyTorch
    mem_used = torch.cuda.memory_allocated() / 1024**2
    mem_reserved = torch.cuda.memory_reserved() / 1024**2
    return {
        'gpu_util': 0,
        'mem_used': mem_used,
        'mem_total': 0,
        'temp': 0,
    }


def make_bar(progress: float, width: int = 30, fill: str = "#", empty: str = "-") -> str:
    """Create a progress bar."""
    filled = int(width * progress)
    bar = fill * filled + empty * (width - filled)
    return f"[{bar}]"


class Trainer:
    """
    Trainer for NeuralForge models with visual dashboard.
    """
    
    def __init__(
        self,
        model: NeuralForge,
        config: ModelConfig,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        checkpoint_dir: str = "checkpoints",
        log_interval: int = 1,
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
        
        # Setup device - GPU only
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available. This model requires a GPU for training.")
        self.device = torch.device("cuda")
        self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = model.get_optimizer(config)
        
        # Setup scheduler
        total_steps = len(train_loader) * config.warmup_steps
        self.scheduler = CosineScheduleWithWarmup(
            self.optimizer,
            warmup_steps=config.warmup_steps,
            max_steps=total_steps
        )
        
        # Mixed precision
        self.scaler = torch.amp.GradScaler('cuda')
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Training state
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.epoch_losses: List[float] = []
        self.train_start_time = None
    
    def _print_header(self, num_epochs: int):
        """Print training header with model info."""
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1024**3
        n_params = self.model.count_parameters()
        total_batches = len(self.train_loader) * num_epochs
        
        print()
        print("=" * 70)
        print("  NEURALFORGE TRAINING DASHBOARD")
        print("=" * 70)
        print()
        print(f"  GPU:        {gpu_name} ({gpu_mem:.1f} GB)")
        print(f"  Model:      {n_params:,} parameters ({n_params/1e6:.2f}M)")
        print(f"  Dataset:    {len(self.train_loader.dataset):,} samples")
        print(f"  Batch size: {self.train_loader.batch_size}")
        print(f"  Batches:    {len(self.train_loader)} per epoch")
        print(f"  Epochs:     {num_epochs}")
        print(f"  Total steps: {total_batches}")
        print(f"  Seq len:    {self.config.max_seq_len}")
        print(f"  LR:         {self.config.learning_rate}")
        print()
        print("=" * 70)
        print()
    
    def _print_dashboard(self, epoch: int, num_epochs: int, batch_idx: int, 
                         loss: float, lr: float, elapsed: float, batch_time: float):
        """Print live training dashboard."""
        total_batches = len(self.train_loader)
        batch_progress = (batch_idx + 1) / total_batches
        epoch_progress = ((epoch - 1) + batch_progress) / num_epochs
        
        # ETA calculation
        if batch_idx > 0:
            avg_batch_time = elapsed / (batch_idx + 1)
            remaining_batches = (total_batches - batch_idx - 1) + (num_epochs - epoch) * total_batches
            eta = avg_batch_time * remaining_batches
        else:
            eta = 0
        
        # Tokens per second
        tokens_per_sec = (batch_idx + 1) * self.train_loader.batch_size * self.config.max_seq_len / max(elapsed, 0.001)
        
        # GPU stats
        gpu_stats = get_gpu_stats()
        gpu_util = gpu_stats.get('gpu_util', 0)
        mem_used = gpu_stats.get('mem_used', 0)
        mem_total = gpu_stats.get('mem_total', 0)
        temp = gpu_stats.get('temp', 0)
        
        # Build display
        bar = make_bar(batch_progress, width=40)
        epoch_bar = make_bar(epoch_progress, width=20)
        
        # Clear and print
        lines = []
        lines.append("\033[2J\033[H")  # Clear screen
        lines.append("=" * 70)
        lines.append("  NEURALFORGE TRAINING DASHBOARD")
        lines.append("=" * 70)
        lines.append("")
        
        # GPU info
        if mem_total > 0:
            mem_pct = (mem_used / mem_total) * 100
            lines.append(f"  GPU: {torch.cuda.get_device_name(0)}")
            lines.append(f"  utilization: {gpu_util:.0f}%  |  memory: {mem_used:.0f}/{mem_total:.0f} MB ({mem_pct:.0f}%)  |  temp: {temp:.0f}C")
        else:
            lines.append(f"  GPU: {torch.cuda.get_device_name(0)}")
        lines.append("")
        
        # Training progress
        lines.append(f"  Epoch:      {epoch}/{num_epochs} {epoch_bar} {epoch_progress*100:.1f}%")
        lines.append(f"  Batch:      {batch_idx+1}/{total_batches} {bar} {batch_progress*100:.1f}%")
        lines.append("")
        
        # Metrics
        lines.append(f"  Loss:       {loss:.4f}")
        lines.append(f"  LR:         {lr:.2e}")
        lines.append(f"  Tokens/s:   {tokens_per_sec:,.0f}")
        lines.append(f"  Step:       {self.global_step}")
        lines.append("")
        
        # Time
        lines.append(f"  Elapsed:    {format_time(elapsed)}")
        lines.append(f"  ETA:        {format_time(eta)}")
        lines.append(f"  Batch time: {batch_time*1000:.0f}ms")
        lines.append("")
        
        # Loss history (last 10 epochs)
        if self.epoch_losses:
            lines.append("  Loss history:")
            max_loss = max(self.epoch_losses) if self.epoch_losses else 1
            min_loss = min(self.epoch_losses) if self.epoch_losses else 0
            for i, l in enumerate(self.epoch_losses[-10:], max(1, len(self.epoch_losses)-9)):
                bar_len = int(30 * (l - min_loss) / max(max_loss - min_loss, 0.001))
                loss_bar = "#" * bar_len
                lines.append(f"    Epoch {i:3d}: {l:.4f} |{loss_bar}")
        
        lines.append("")
        lines.append("=" * 70)
        lines.append("  Press Ctrl+C to stop training")
        lines.append("=" * 70)
        
        sys.stdout.write("\n".join(lines))
        sys.stdout.flush()
    
    def train_epoch(self, epoch: int, num_epochs: int):
        """Train for one epoch with visual dashboard."""
        self.model.train()
        total_loss = 0
        epoch_start = time.time()
        batch_times = []
        
        self.optimizer.zero_grad()
        
        for batch_idx, (x, y) in enumerate(self.train_loader):
            batch_start = time.time()
            
            x = x.to(self.device)
            y = y.to(self.device)
            
            # Forward pass with mixed precision
            with torch.amp.autocast('cuda'):
                logits, loss, _ = self.model(x, targets=y)
                loss = loss / self.gradient_accumulation_steps
            
            # Backward pass
            self.scaler.scale(loss).backward()
            
            total_loss += loss.item() * self.gradient_accumulation_steps
            
            # Gradient accumulation step
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1
            
            batch_time = time.time() - batch_start
            batch_times.append(batch_time)
            
            # Update dashboard every batch
            avg_loss = total_loss / (batch_idx + 1)
            lr = self.scheduler._get_lr() * self.scheduler.base_lrs[0]
            elapsed = time.time() - epoch_start
            
            self._print_dashboard(epoch, num_epochs, batch_idx, avg_loss, lr, elapsed, batch_time)
            
            # Evaluation
            if self.global_step % self.eval_interval == 0 and self.val_loader:
                val_loss = self.evaluate()
                print(f"\n  >> Validation Loss: {val_loss:.4f}")
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint("best_model.pt")
                self.model.train()
            
            # Save checkpoint
            if self.global_step % self.save_interval == 0:
                self.save_checkpoint(f"step_{self.global_step}.pt")
        
        avg_epoch_loss = total_loss / len(self.train_loader)
        self.epoch_losses.append(avg_epoch_loss)
        
        # Print epoch summary
        epoch_time = time.time() - epoch_start
        avg_batch = sum(batch_times) / len(batch_times) if batch_times else 0
        print(f"\n  Epoch {epoch}/{num_epochs} complete | Loss: {avg_epoch_loss:.4f} | Time: {format_time(epoch_time)} | Avg batch: {avg_batch*1000:.0f}ms")
        
        return avg_epoch_loss
    
    @torch.no_grad()
    def evaluate(self) -> float:
        """Evaluate on validation set."""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        for x, y in self.val_loader:
            x = x.to(self.device)
            y = y.to(self.device)
            
            with torch.amp.autocast('cuda'):
                _, loss, _ = self.model(x, targets=y)
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / max(num_batches, 1)
    
    def train(self, num_epochs: int = 10):
        """Full training loop."""
        self.train_start_time = time.time()
        self._print_header(num_epochs)
        
        try:
            for epoch in range(1, num_epochs + 1):
                train_loss = self.train_epoch(epoch, num_epochs)
                
                # Save at end of epoch
                self.save_checkpoint(f"epoch_{epoch}.pt")
            
            # Final summary
            total_time = time.time() - self.train_start_time
            print()
            print("=" * 70)
            print("  TRAINING COMPLETE")
            print("=" * 70)
            print(f"  Total time:    {format_time(total_time)}")
            print(f"  Final loss:    {self.epoch_losses[-1]:.4f}")
            print(f"  Best val loss: {self.best_val_loss:.4f}")
            print(f"  Total steps:   {self.global_step}")
            print(f"  Checkpoints:   {self.checkpoint_dir}/")
            print("=" * 70)
            
        except KeyboardInterrupt:
            total_time = time.time() - self.train_start_time
            print()
            print("=" * 70)
            print("  TRAINING STOPPED BY USER")
            print("=" * 70)
            print(f"  Time:     {format_time(total_time)}")
            print(f"  Steps:    {self.global_step}")
            print(f"  Last loss: {self.epoch_losses[-1]:.4f if self.epoch_losses else 'N/A'}")
            print(f"  Resuming: python train.py --resume {self.checkpoint_dir}/epoch_{len(self.epoch_losses)}.pt")
            print("=" * 70)
    
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
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.step_num = checkpoint['scheduler_state_dict']['step_num']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        print(f"  Resumed from step {self.global_step}")
