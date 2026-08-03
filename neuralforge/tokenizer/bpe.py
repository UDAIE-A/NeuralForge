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
        self.space_id = None
        self._ranks: Optional[Dict[Tuple[str, str], int]] = None
    
    def _build_vocab(self):
        """Build vocabulary from learned merges.

        IDs 0..len(special_tokens)-1 are reserved for the special tokens
        (e.g. <unk> == 3), and the 256 byte tokens start right after them.
        Earlier versions let the byte tokens take IDs 0-3, so the <unk>/<eos>
        ids everyone refers to silently pointed at real control bytes - OOV
        input became byte 0x03 instead of the actual <unk> token.
        """
        base = len(self.special_tokens)
        vocab = {bytes([i]).decode('latin-1'): base + i for i in range(256)}
        for token, idx in self.special_tokens.items():
            vocab[token] = idx
        # IDs must stay contiguous (0..len(vocab)-1): a duplicate merge
        # string is skipped, so assigning by merge-list position would leave
        # a gap and push later ids beyond len(vocab), out of the embedding's
        # range. A running counter keeps every id valid.
        offset = base + 256
        next_id = offset
        for first, second in self.merges:
            merged = first + second
            if merged not in vocab:
                vocab[merged] = next_id
                next_id += 1
        self.vocab = vocab
        self.inverse_vocab = {v: k for k, v in vocab.items()}
        self.space_id = self.vocab.get(' ')
    
    def train(self, text: str, vocab_size: int = 32000, verbose: bool = False):
        """Train BPE tokenizer on text.

        Merge selection uses a max-heap with lazy deletion plus per-word pair
        counters and a pair->words index, so each merge only rescans the words
        that actually contain the winning pair - not the entire corpus every
        iteration (which was O(merges x corpus) and crawled on real data).
        Counts are verified-exact against a full recount; on equal-count ties
        the lexicographically smallest pair wins, so the merge *order* may
        differ from older full-recount versions even though every chosen pair
        is a true global max.
        """
        if verbose:
            print(f"  Training BPE on {len(text):,} characters...")
        
        t0 = time.time()
        
        # Split into words. Case is preserved so the model can learn and
        # reproduce capital letters; lowercasing here silently removed every
        # uppercase character from the vocabulary.
        words = re.findall(r'\S+', text)

        # Convert to tuples of byte characters. A space token is inserted
        # BETWEEN words so the model can learn and reproduce word boundaries
        # (otherwise decoded text has no spaces, e.g. "thesunisastar").
        corpus = []
        for word in words:
            word_bytes = tuple(bytes([b]).decode('latin-1') for b in word.encode('utf-8'))
            corpus.append(word_bytes)
            corpus.append((' ',))  # word separator token
        if corpus:
            corpus = corpus[:-1]  # drop trailing separator
        
        if verbose:
            print(f"  Tokenized into {len(corpus):,} words in {time.time()-t0:.1f}s")
        
        # Per-word counters of adjacent pairs, the global totals, and a
        # pair -> word-indices index so merges only touch affected words.
        pair_to_words: Dict[Tuple[str, str], set] = defaultdict(set)
        pair_total: Counter = Counter()
        word_counters: List[Counter] = []
        for i, word in enumerate(corpus):
            counter = Counter()
            for j in range(len(word) - 1):
                pair = (word[j], word[j + 1])
                counter[pair] += 1
                pair_total[pair] += 1
                pair_to_words[pair].add(i)
            word_counters.append(counter)
        
        # Max-heap of (-count, pair). Entries go stale whenever a count
        # changes; they're skipped lazily when popped.
        heap = [(-count, pair) for pair, count in pair_total.items()]
        heapq.heapify(heap)
        
        num_merges = vocab_size - 256 - len(self.special_tokens)
        self.merges = []
        
        t0 = time.time()
        for i in range(num_merges):
            # Pop until we reach a fresh entry whose count is still current.
            best_pair = None
            best_count = 0
            while heap:
                neg_count, pair = heap[0]
                if pair_total.get(pair, 0) == -neg_count:
                    best_pair, best_count = pair, -neg_count
                    break
                heapq.heappop(heap)
            
            if best_pair is None or best_count < 2:
                break
            
            a, b = best_pair
            merged = a + b
            
            # Only the words that contain the pair can change. The snapshot is
            # safe: every (a, b) occurrence is consumed by the merge, so no
            # affected word can produce a new (a, b) and re-enter this list.
            for idx in list(pair_to_words[best_pair]):
                word = corpus[idx]
                old_counts = word_counters[idx]
                
                # Build the merged word in a single pass.
                new_word = []
                j = 0
                n = len(word)
                while j < n:
                    if j < n - 1 and word[j] == a and word[j + 1] == b:
                        new_word.append(merged)
                        j += 2
                    else:
                        new_word.append(word[j])
                        j += 1
                new_word = tuple(new_word)
                corpus[idx] = new_word
                
                # Re-count just this word's pairs and diff against the old.
                new_counts: Counter = Counter()
                for j in range(len(new_word) - 1):
                    new_counts[(new_word[j], new_word[j + 1])] += 1
                
                keys = set(old_counts) | set(new_counts)
                for pair in keys:
                    delta = new_counts.get(pair, 0) - old_counts.get(pair, 0)
                    if delta:
                        pair_total[pair] += delta
                        if pair_total[pair] <= 0:
                            del pair_total[pair]
                        else:
                            heapq.heappush(heap, (-pair_total[pair], pair))
                    if new_counts.get(pair, 0) > 0:
                        pair_to_words[pair].add(idx)
                    else:
                        pair_to_words[pair].discard(idx)
                word_counters[idx] = new_counts
            
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
    
    def _ensure_ranks(self):
        """Lazily build pair -> merge-rank map for the fast encoder."""
        if self._ranks is None:
            self._ranks = {pair: rank for rank, pair in enumerate(self.merges)}

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text to token IDs.

        Rank-based encoding: repeatedly merge the adjacent pair with the
        lowest merge rank, which reproduces learning-order application but is
        O(word) per word instead of O(merges x word). On a real ~1200-merge
        tokenizer this cut corpus tokenization from ~37s/MB to ~1s/MB.
        """
        if not self.is_trained:
            raise RuntimeError("Tokenizer must be trained before encoding")
        self._ensure_ranks()
        ranks = self._ranks
        
        tokens = []
        if add_special_tokens:
            tokens.append(self.bos_id)
        
        words = re.findall(r'\S+', text)
        for wi, word in enumerate(words):
            word_bytes = [bytes([b]).decode('latin-1') for b in word.encode('utf-8')]
            if not word_bytes:
                continue
            # Merge the lowest-rank adjacent pair until none remains.
            while True:
                best_idx = -1
                best_rank = float('inf')
                for j in range(len(word_bytes) - 1):
                    rank = ranks.get((word_bytes[j], word_bytes[j + 1]))
                    if rank is not None and rank < best_rank:
                        best_rank, best_idx = rank, j
                if best_idx < 0:
                    break
                word_bytes[best_idx:best_idx + 2] = [word_bytes[best_idx] + word_bytes[best_idx + 1]]

            for token in word_bytes:
                tokens.append(self.vocab.get(token, self.unk_id))
            if wi != len(words) - 1 and self.space_id is not None:
                tokens.append(self.space_id)
        
        if add_special_tokens:
            tokens.append(self.eos_id)
        
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
        tokenizer.space_id = tokenizer.vocab.get(' ')
        tokenizer._ranks = None
        return tokenizer
    
    def __len__(self) -> int:
        return len(self.vocab)

    # The real ids live in the vocab (a saved tokenizer may predate the
    # special-id layout fix), so resolve through it, not special_tokens.
    def _spec_id(self, name: str) -> int:
        return self.vocab.get(name, self.special_tokens[name])

    @property
    def pad_id(self) -> int:
        return self._spec_id('<pad>')

    @property
    def bos_id(self) -> int:
        return self._spec_id('<bos>')

    @property
    def eos_id(self) -> int:
        return self._spec_id('<eos>')

    @property
    def unk_id(self) -> int:
        return self._spec_id('<unk>')
