"""
Data loading utilities for NeuralForge training.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Optional, Tuple


class TextDataset(Dataset):
    """
    Dataset for language modeling.
    
    Loads text files and tokenizes them into fixed-length sequences.
    """
    
    def __init__(
        self,
        data: str,
        tokenizer,
        seq_len: int = 512,
        stride: int = 256
    ):
        """
        Args:
            data: Raw text string or path to text file
            tokenizer: Tokenizer instance
            seq_len: Sequence length for training
            stride: Stride for sliding window
        """
        self.seq_len = seq_len
        self.stride = stride
        self.tokenizer = tokenizer
        
        # Load data
        if os.path.exists(data):
            with open(data, 'r', encoding='utf-8') as f:
                text = f.read()
        else:
            text = data
        
        # Tokenize
        print(f"Tokenizing {len(text)} characters...")
        self.tokens = tokenizer.encode(text, add_special_tokens=False)
        print(f"Got {len(self.tokens)} tokens")
        
        # Calculate number of sequences
        self.num_sequences = max(0, (len(self.tokens) - seq_len) // stride + 1)
        print(f"Created {self.num_sequences} sequences of length {seq_len}")
    
    def __len__(self) -> int:
        return self.num_sequences
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.stride
        end = start + self.seq_len + 1
        
        # Get sequence
        chunk = self.tokens[start:end]

        # Pad if necessary. Inputs are padded with <pad> (0), but the matching
        # targets are padded with -1 so cross_entropy(ignore_index=-1) skips
        # them - otherwise the model would be trained to predict <pad>.
        pad = self.seq_len + 1 - len(chunk)
        x_ids = chunk[:-1] if pad <= 0 else chunk[:-1] + [0] * pad
        y_ids = chunk[1:] if pad <= 0 else chunk[1:] + [-1] * pad

        # Input and target (shifted by 1)
        x = torch.tensor(x_ids[:self.seq_len], dtype=torch.long)
        y = torch.tensor(y_ids[:self.seq_len], dtype=torch.long)

        return x, y


def create_dataloaders(
    train_data: str,
    val_data: Optional[str],
    tokenizer,
    seq_len: int = 512,
    batch_size: int = 32,
    stride: int = 256,
    num_workers: int = 8
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Create train and validation dataloaders.
    
    Args:
        train_data: Training text or file path
        val_data: Validation text or file path (optional)
        tokenizer: Tokenizer instance
        seq_len: Sequence length
        batch_size: Batch size
        stride: Stride for sliding window
        num_workers: Number of data loading workers
        
    Returns:
        (train_loader, val_loader)
    """
    train_dataset = TextDataset(train_data, tokenizer, seq_len, stride)
    
    # Cap batch size to dataset size
    effective_batch_size = min(batch_size, len(train_dataset))
    if effective_batch_size < batch_size:
        print(f"  Warning: batch_size {batch_size} > dataset size {len(train_dataset)}, using {effective_batch_size}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=effective_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True,
        prefetch_factor=4
    )
    
    val_loader = None
    if val_data:
        val_dataset = TextDataset(val_data, tokenizer, seq_len, stride)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    
    return train_loader, val_loader
