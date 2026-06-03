"""
Byte Pair Encoding (BPE) Tokenizer - Implemented from scratch.

This tokenizer learns subword units from training data, similar to how
GPT-2 and other modern language models tokenize text.
"""

import re
import json
import os
from collections import Counter, defaultdict
from typing import List, Dict, Optional, Tuple
import pickle


class BPETokenizer:
    """
    Byte Pair Encoding tokenizer.
    
    Learns a vocabulary of subword tokens from training data.
    Starts with individual bytes and merges the most frequent pairs.
    """
    
    def __init__(self):
        self.merges: List[Tuple[str, str]] = []
        self.vocab: Dict[str, int] = {}
        self.inverse_vocab: Dict[int, str] = {}
        self.special_tokens = {
            '<pad>': 0,
            '<bos>': 1,
            '<eos>': 2,
            '<unk>': 3,
        }
        self.is_trained = False
    
    def _get_stats(self, ids: List[List[str]]) -> Dict[Tuple[str, str], int]:
        """Count frequency of adjacent pairs."""
        counts = Counter()
        for word_ids in ids:
            for i in range(len(word_ids) - 1):
                pair = (word_ids[i], word_ids[i + 1])
                counts[pair] += 1
        return counts
    
    def _merge(self, ids: List[str], pair: Tuple[str, str]) -> List[str]:
        """Merge all occurrences of pair in the token list."""
        new_ids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                new_ids.append(pair[0] + pair[1])
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids
    
    def _build_vocab(self):
        """Build vocabulary from learned merges."""
        # Start with byte-level characters (0-255)
        vocab = {bytes([i]).decode('latin-1'): i for i in range(256)}
        
        # Add special tokens
        for token, idx in self.special_tokens.items():
            vocab[token] = 256 + idx
        
        # Add merged tokens
        offset = 256 + len(self.special_tokens)
        for i, (first, second) in enumerate(self.merges):
            merged = first + second
            if merged not in vocab:
                vocab[merged] = offset + i
        
        self.vocab = vocab
        self.inverse_vocab = {v: k for k, v in vocab.items()}
    
    def train(self, text: str, vocab_size: int = 32000, verbose: bool = False):
        """
        Train BPE tokenizer on text.
        
        Args:
            text: Training text
            vocab_size: Target vocabulary size (including byte tokens and special tokens)
            verbose: Print training progress
        """
        if verbose:
            print(f"Training BPE tokenizer on {len(text)} characters...")
        
        # Split into words (simple whitespace + punctuation splitting)
        # Each word becomes a list of byte characters
        words = re.findall(r'\S+', text.lower())
        
        # Convert each word to list of byte characters
        corpus = []
        for word in words:
            word_bytes = [bytes([b]).decode('latin-1') for b in word.encode('utf-8')]
            corpus.append(word_bytes)
        
        # Calculate number of merges needed
        # 256 byte tokens + 4 special tokens + N merges = vocab_size
        num_merges = vocab_size - 256 - len(self.special_tokens)
        
        self.merges = []
        
        for i in range(num_merges):
            # Count all adjacent pairs
            stats = self._get_stats(corpus)
            
            if not stats:
                if verbose:
                    print(f"No more pairs to merge after {i} merges")
                break
            
            # Find most frequent pair
            best_pair = max(stats, key=stats.get)
            best_count = stats[best_pair]
            
            if best_count < 2:
                if verbose:
                    print(f"Best pair frequency < 2 after {i} merges, stopping")
                break
            
            # Merge this pair everywhere in corpus
            corpus = [self._merge(word_ids, best_pair) for word_ids in corpus]
            
            self.merges.append(best_pair)
            
            if verbose and (i + 1) % 100 == 0:
                print(f"Merge {i+1}/{num_merges}: {best_pair} -> {best_pair[0]+best_pair[1]} (freq: {best_count})")
        
        # Build vocabulary
        self._build_vocab()
        self.is_trained = True
        
        if verbose:
            print(f"Tokenizer trained: {len(self.vocab)} tokens, {len(self.merges)} merges")
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text
            add_special_tokens: Add BOS/EOS tokens
            
        Returns:
            List of token IDs
        """
        if not self.is_trained:
            raise RuntimeError("Tokenizer must be trained before encoding")
        
        tokens = []
        
        if add_special_tokens:
            tokens.append(self.special_tokens['<bos>'])
        
        # Process each word
        words = re.findall(r'\S+', text.lower())
        
        for word in words:
            # Convert to byte characters
            word_bytes = [bytes([b]).decode('latin-1') for b in word.encode('utf-8')]
            
            # Apply learned merges
            for first, second in self.merges:
                word_bytes = self._merge(word_bytes, (first, second))
            
            # Convert to IDs
            for token in word_bytes:
                if token in self.vocab:
                    tokens.append(self.vocab[token])
                else:
                    tokens.append(self.special_tokens['<unk>'])
        
        if add_special_tokens:
            tokens.append(self.special_tokens['<eos>'])
        
        return tokens
    
    def decode(self, ids: List[int]) -> str:
        """
        Decode token IDs to text.
        
        Args:
            ids: List of token IDs
            
        Returns:
            Decoded text
        """
        if not self.is_trained:
            raise RuntimeError("Tokenizer must be trained before decoding")
        
        tokens = []
        for id in ids:
            if id in self.inverse_vocab:
                token = self.inverse_vocab[id]
                # Skip special tokens in output
                if token not in self.special_tokens:
                    tokens.append(token)
            else:
                tokens.append('<unk>')
        
        # Join and clean up
        text = ''.join(tokens)
        # Try to decode as UTF-8
        try:
            text_bytes = text.encode('latin-1')
            text = text_bytes.decode('utf-8')
        except (UnicodeDecodeError, UnicodeEncodeError):
            pass
        
        return text
    
    def save(self, path: str):
        """Save tokenizer to file."""
        data = {
            'merges': self.merges,
            'vocab': self.vocab,
            'special_tokens': self.special_tokens,
            'is_trained': self.is_trained,
        }
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(data, f)
    
    @classmethod
    def load(cls, path: str) -> 'BPETokenizer':
        """Load tokenizer from file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        tokenizer = cls()
        tokenizer.merges = data['merges']
        tokenizer.vocab = data['vocab']
        tokenizer.special_tokens = data['special_tokens']
        tokenizer.is_trained = data['is_trained']
        tokenizer.inverse_vocab = {v: k for k, v in tokenizer.vocab.items()}
        
        return tokenizer
    
    def __len__(self) -> int:
        return len(self.vocab)
