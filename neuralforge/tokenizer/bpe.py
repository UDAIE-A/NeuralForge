"""
Byte Pair Encoding (BPE) Tokenizer - Fast implementation.
"""

import re
import os
import time
from collections import Counter, defaultdict
from typing import List, Dict, Optional, Tuple
import pickle
import heapq


class BPETokenizer:
    """Fast BPE tokenizer."""
    
    def __init__(self):
        self.merges: List[Tuple[str, str]] = []
        self.vocab: Dict[str, int] = {}
        self.inverse_vocab: Dict[int, str] = {}
        self.special_tokens = {'<pad>': 0, '<bos>': 1, '<eos>': 2, '<unk>': 3}
        self.is_trained = False
    
    def _get_stats_fast(self, corpus: List[Tuple[str, ...]]) -> Counter:
        """Count pair frequencies efficiently."""
        counts = Counter()
        for word in corpus:
            for i in range(len(word) - 1):
                counts[(word[i], word[i + 1])] += 1
        return counts
    
    def _merge_pair(self, corpus: List[Tuple[str, ...]], pair: Tuple[str, str]) -> List[Tuple[str, ...]]:
        """Merge a pair in all words."""
        new_corpus = []
        a, b = pair
        for word in corpus:
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                    new_word.append(a + b)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_corpus.append(tuple(new_word))
        return new_corpus
    
    def _build_vocab(self):
        """Build vocabulary from learned merges."""
        vocab = {bytes([i]).decode('latin-1'): i for i in range(256)}
        for token, idx in self.special_tokens.items():
            vocab[token] = 256 + idx
        offset = 256 + len(self.special_tokens)
        for i, (first, second) in enumerate(self.merges):
            merged = first + second
            if merged not in vocab:
                vocab[merged] = offset + i
        self.vocab = vocab
        self.inverse_vocab = {v: k for k, v in vocab.items()}
    
    def train(self, text: str, vocab_size: int = 32000, verbose: bool = False):
        """Train BPE tokenizer on text."""
        if verbose:
            print(f"  Training BPE on {len(text):,} characters...")
        
        t0 = time.time()
        
        # Split into words
        words = re.findall(r'\S+', text.lower())
        
        # Convert to tuples of byte characters
        corpus = []
        for word in words:
            word_bytes = tuple(bytes([b]).decode('latin-1') for b in word.encode('utf-8'))
            corpus.append(word_bytes)
        
        if verbose:
            print(f"  Tokenized into {len(corpus):,} words in {time.time()-t0:.1f}s")
        
        num_merges = vocab_size - 256 - len(self.special_tokens)
        self.merges = []
        
        t0 = time.time()
        for i in range(num_merges):
            stats = self._get_stats_fast(corpus)
            if not stats:
                break
            
            best_pair = max(stats, key=stats.get)
            best_count = stats[best_pair]
            
            if best_count < 2:
                break
            
            corpus = self._merge_pair(corpus, best_pair)
            self.merges.append(best_pair)
            
            if verbose and (i + 1) % 500 == 0:
                elapsed = time.time() - t0
                per_merge = elapsed / (i + 1)
                eta = per_merge * (num_merges - i - 1) / 60
                merged = best_pair[0] + best_pair[1]
                merged = ''.join(c if c.isprintable() else '?' for c in merged)
                print(f"    Merge {i+1}/{num_merges}: {merged} | ETA: {eta:.1f}min")
        
        self._build_vocab()
        self.is_trained = True
        
        if verbose:
            print(f"  Tokenizer done: {len(self.vocab)} tokens in {time.time()-t0:.1f}s")
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text to token IDs."""
        if not self.is_trained:
            raise RuntimeError("Tokenizer must be trained before encoding")
        
        tokens = []
        if add_special_tokens:
            tokens.append(self.special_tokens['<bos>'])
        
        words = re.findall(r'\S+', text.lower())
        for word in words:
            word_bytes = [bytes([b]).decode('latin-1') for b in word.encode('utf-8')]
            for first, second in self.merges:
                new_bytes = []
                j = 0
                while j < len(word_bytes):
                    if j < len(word_bytes) - 1 and word_bytes[j] == first and word_bytes[j + 1] == second:
                        new_bytes.append(first + second)
                        j += 2
                    else:
                        new_bytes.append(word_bytes[j])
                        j += 1
                word_bytes = new_bytes
            
            for token in word_bytes:
                tokens.append(self.vocab.get(token, self.special_tokens['<unk>']))
        
        if add_special_tokens:
            tokens.append(self.special_tokens['<eos>'])
        
        return tokens
    
    def decode(self, ids: List[int]) -> str:
        """Decode token IDs to text."""
        if not self.is_trained:
            raise RuntimeError("Tokenizer must be trained before decoding")
        
        tokens = []
        for id in ids:
            if id in self.inverse_vocab:
                token = self.inverse_vocab[id]
                if token not in self.special_tokens:
                    tokens.append(token)
            else:
                tokens.append('<unk>')
        
        text = ''.join(tokens)
        try:
            text_bytes = text.encode('latin-1')
            text = text_bytes.decode('utf-8')
        except (UnicodeDecodeError, UnicodeEncodeError):
            pass
        
        return text
    
    def save(self, path: str):
        """Save tokenizer to file."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'merges': self.merges,
                'vocab': self.vocab,
                'special_tokens': self.special_tokens,
                'is_trained': self.is_trained,
            }, f)
    
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
