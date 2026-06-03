"""
NeuralForge: A from-scratch language model.

Built from the ground up with no dependencies on external models.
"""

from .core import NeuralForge, ModelConfig
from .tokenizer import BPETokenizer
from .training import Trainer, TextDataset, DataLoader

__version__ = "0.1.0"
__all__ = ['NeuralForge', 'ModelConfig', 'BPETokenizer', 'Trainer', 'TextDataset', 'DataLoader']
