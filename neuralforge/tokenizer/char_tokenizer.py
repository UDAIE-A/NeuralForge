"""
Character-level tokenizer - instant training, no BPE needed.
"""

import os
import pickle
from typing import List, Dict


class CharTokenizer:
    """Simple character-level tokenizer. Fast to train."""
    
    def __init__(self):
        self.char_to_id: Dict[str, int] = {}
        self.id_to_char: Dict[int, str] = {}
        self.special_tokens = {'<pad>': 0, '<bos>': 1, '<eos>': 2, '<unk>': 3}
        self.is_trained = False
    
    def train(self, text: str, verbose: bool = False):
        """Train on text - instant, just collect unique chars."""
        chars = sorted(set(text))
        
        self.char_to_id = dict(self.special_tokens)
        for i, c in enumerate(chars):
            self.char_to_id[c] = len(self.char_to_id)
        
        self.id_to_char = {v: k for k, v in self.char_to_id.items()}
        self.is_trained = True
        
        if verbose:
            print(f"  Tokenizer: {len(self.char_to_id)} tokens (characters)")
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text to token IDs."""
        tokens = []
        if add_special_tokens:
            tokens.append(self.special_tokens['<bos>'])
        
        for c in text:
            tokens.append(self.char_to_id.get(c, self.special_tokens['<unk>']))
        
        if add_special_tokens:
            tokens.append(self.special_tokens['<eos>'])
        
        return tokens
    
    def decode(self, ids: List[int]) -> str:
        """Decode token IDs to text."""
        chars = []
        for id in ids:
            if id in self.id_to_char:
                c = self.id_to_char[id]
                if c not in self.special_tokens.values():
                    chars.append(c)
        return ''.join(chars)
    
    def save(self, path: str):
        """Save tokenizer."""
        with open(path, 'wb') as f:
            pickle.dump({
                'char_to_id': self.char_to_id,
                'is_trained': self.is_trained,
            }, f)
    
    @classmethod
    def load(cls, path: str) -> 'CharTokenizer':
        """Load tokenizer."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        tokenizer = cls()
        tokenizer.char_to_id = data['char_to_id']
        tokenizer.id_to_char = {v: k for k, v in tokenizer.char_to_id.items()}
        tokenizer.is_trained = data['is_trained']
        return tokenizer
    
    def __len__(self):
        return len(self.char_to_id)
