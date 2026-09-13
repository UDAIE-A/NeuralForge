import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .config import ModelConfig


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the two halves of the last dimension: (x1, x2) -> (-x2, x1)."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embedding to (B, n_heads, T, d_head) tensor.

    cos/sin are (T, d_head) and broadcast over batch and heads.
    """
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return x * cos + rotate_half(x) * sin


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with causal masking."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.d_head = config.d_head
        self.d_model = config.d_model
        self.dropout_p = config.dropout
        
        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        rope: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, T, C = x.shape

        # Project to Q, K, V
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # Rotary position embedding on the new tokens' q/k before they enter
        # the cache. RoPE is relative, so caching already-rotated keys is correct.
        if rope is not None:
            cos, sin = rope
            q = apply_rotary(q, cos, sin)
            k = apply_rotary(k, cos, sin)

        # Handle KV cache for efficient generation
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)
        
        new_cache = (k, v)

        T_q = q.shape[2]
        T_k = k.shape[2]
        dropout_p = self.dropout_p if self.training else 0

        # Prefer the is_causal fast path - it lets SDPA dispatch to the
        # FlashAttention kernel without materializing a mask. is_causal aligns
        # the triangle to the top-left, which is only correct when query and
        # key lengths match (training, or the first generation step). During
        # incremental decoding T_q==1 and the single query must see every
        # cached key, so no mask is needed there either.
        if T_q == T_k:
            att = F.scaled_dot_product_attention(
                q, k, v, dropout_p=dropout_p, is_causal=True
            )
        elif T_q == 1:
            att = F.scaled_dot_product_attention(
                q, k, v, dropout_p=dropout_p, is_causal=False
            )
        else:
            # Multi-token step with a non-empty cache: build an offset causal
            # mask so each new query attends to the cache plus earlier new tokens.
            offset = T_k - T_q
            causal_mask = torch.ones(T_q, T_k, device=x.device, dtype=torch.bool).tril(diagonal=offset)
            att = F.scaled_dot_product_attention(
                q, k, v, attn_mask=causal_mask, dropout_p=dropout_p, is_causal=False
            )
        
        out = att.transpose(1, 2).contiguous().view(B, T, C)
        out = self.resid_dropout(self.o_proj(out))
        
        return out, new_cache


class RMSNorm(nn.Module):
    """Root-mean-square layer normalization (no mean subtraction, no bias)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm.type_as(x) * self.weight


class FeedForward(nn.Module):
    """SwiGLU feed-forward network (gated SiLU), as used in LLaMA."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.up_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.down_proj = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.gate_proj(x)) * self.up_proj(x)
        x = self.down_proj(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Single transformer block with pre-norm architecture."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln1 = RMSNorm(config.d_model)
        self.attn = MultiHeadAttention(config)
        self.ln2 = RMSNorm(config.d_model)
        self.ffn = FeedForward(config)
    
    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        rope: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # Pre-norm architecture
        residual = x
        x = self.ln1(x)
        attn_out, new_cache = self.attn(x, kv_cache, rope)
        x = residual + attn_out
        
        residual = x
        x = self.ln2(x)
        x = residual + self.ffn(x)
        
        return x, new_cache


class NeuralForge(nn.Module):
    """NeuralForge: A GPT-style decoder-only transformer.
    
    Built from scratch with no dependencies on external models.
    Architecture: Token embedding + RoPE + Transformer blocks + LM head
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Token embedding. Positions come from rotary embeddings (RoPE) applied
        # inside attention, so there is no learned positional embedding table.
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.emb_dropout = nn.Dropout(config.dropout)

        # Rotary frequencies for RoPE (d_head must be even).
        assert config.d_head % 2 == 0, "d_head must be even for rotary embeddings"
        inv_freq = 1.0 / (10000 ** (torch.arange(0, config.d_head, 2).float() / config.d_head))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Precompute the cos/sin tables once (positions 0..max_seq_len) instead
        # of rebuilding them on every forward call. Generation steps then just
        # slice the needed window.
        pos = torch.arange(config.max_seq_len, dtype=torch.float32)
        freqs = torch.outer(pos, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("_rope_cos", emb.cos(), persistent=False)
        self.register_buffer("_rope_sin", emb.sin(), persistent=False)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])
        
        # Final layer norm
        self.ln_f = RMSNorm(config.d_model)
        
        # Language model head (weight tied with token embedding)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Weight tying
        self.lm_head.weight = self.tok_emb.weight
        
        # Initialize weights
        self.apply(self._init_weights)

        # GPT-2 style scaled init for the residual projections: with N blocks
        # the residual stream accumulates 2*N contributions, so scale those
        # output projections by 1/sqrt(2*N) to keep activation variance stable
        # in deep models.
        residual_scale = (2 * config.n_layers) ** -0.5
        for name, param in self.named_parameters():
            if name.endswith('o_proj.weight') or name.endswith('down_proj.weight'):
                torch.nn.init.normal_(param, mean=0.0, std=0.02 * residual_scale)

        # Print parameter count
        n_params = sum(p.numel() for p in self.parameters())
        print(f"NeuralForge initialized: {n_params/1e6:.2f}M parameters")
    
    def _init_weights(self, module):
        """Initialize weights with scaled normal distribution."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def _rope(self, seq_len: int, offset: int, device, dtype):
        """Build (cos, sin) of shape (seq_len, d_head) for positions
        [offset, offset+seq_len) by slicing the precomputed tables."""
        cos = self._rope_cos[offset:offset + seq_len].to(device=device, dtype=dtype)
        sin = self._rope_sin[offset:offset + seq_len].to(device=device, dtype=dtype)
        return cos, sin

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        kv_caches: Optional[list] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[list]]:
        """
        Forward pass.
        
        Args:
            idx: Token indices (B, T)
            targets: Target indices for loss computation (B, T)
            kv_caches: KV caches for each layer (for generation)
            use_cache: Whether to return updated caches
            
        Returns:
            logits: (B, T, vocab_size)
            loss: Scalar loss if targets provided
            new_caches: Updated KV caches if use_cache
        """
        B, T = idx.shape
        device = idx.device

        # During cached generation only the new tokens are passed in, so RoPE
        # positions must be offset by however much is already in the cache.
        past_len = kv_caches[0][0].size(2) if kv_caches is not None else 0
        x = self.emb_dropout(self.tok_emb(idx))

        # Rotary cos/sin for the current tokens' absolute positions.
        rope = self._rope(T, past_len, device, x.dtype)

        # Transformer blocks
        new_caches = []
        for i, block in enumerate(self.blocks):
            cache = kv_caches[i] if kv_caches is not None else None
            x, new_cache = block(x, cache, rope)
            if use_cache:
                new_caches.append(new_cache)
        
        # Final layer norm
        x = self.ln_f(x)
        
        # Language model head
        logits = self.lm_head(x)
        
        # Compute loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )
        
        return logits, loss, new_caches if use_cache else None
    
    def _next_input(self, idx: torch.Tensor, kv_caches: Optional[list]):
        """Pick the tokens to feed this step, and the cache to feed them with.

        Returns (idx_cond, kv_caches). Three cases:

          * no cache yet      -> feed the (truncated) prompt
          * cache has room    -> feed just the last token
          * cache is full     -> slide the window

        Sliding matters: RoPE positions are read out of a table of exactly
        max_seq_len rows, and during cached decoding the position index is the
        cache length. Once that reached max_seq_len the table slice came back
        EMPTY and generation died with "shape [B, 1, C] is invalid for input of
        size 0" - so a model could never emit more tokens than its context
        length. Here the cache is dropped and re-primed from the tail of the
        sequence, which re-bases every position into the valid range.

        Re-priming keeps three quarters of the window so it costs one extra
        forward pass per max_seq_len/4 tokens, not one per token. Cached keys
        carry rotations for their original absolute positions, so they cannot
        simply be evicted from the front - the surviving keys would sit at the
        wrong relative distance from the new query.
        """
        max_len = self.config.max_seq_len
        if kv_caches is None:
            return idx[:, -max_len:], None
        if kv_caches[0][0].size(2) < max_len:
            return idx[:, -1:], kv_caches
        keep = max(1, (max_len * 3) // 4)
        return idx[:, -keep:], None

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = None,
        repetition_penalty: float = 1.0,
        eos_id: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate text autoregressively.

        Args:
            idx: Starting token indices (B, T)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature; <= 0 means greedy (argmax)
            top_k: If set, only sample from top-k tokens
            top_p: If set, nucleus sampling - keep the smallest set of tokens
                whose cumulative probability exceeds top_p
            repetition_penalty: >1.0 discourages repeating tokens already in
                the sequence (1.0 disables it)
            eos_id: If set, generation stops as soon as this token is sampled

        Returns:
            Generated token indices (B, T + tokens_generated)
        """
        self.eval()
        kv_caches = None
        # One seen-token set per batch row; a shared set would leak row 0's
        # tokens into every other row's penalty.
        seen_tokens: Optional[List[set]] = None

        for _ in range(max_new_tokens):
            idx_cond, kv_caches = self._next_input(idx, kv_caches)

            # Forward pass with cache
            logits, _, kv_caches = self.forward(
                idx_cond,
                kv_caches=kv_caches,
                use_cache=True
            )

            # Get logits for last position
            logits = logits[:, -1, :]

            # Repetition penalty: divide logits of already-seen tokens (CTRL
            # style - positive logits shrink, negative logits grow).
            if repetition_penalty != 1.0:
                if seen_tokens is None:
                    seen_tokens = [set(row.tolist()) for row in idx]
                for b in range(idx.size(0)):
                    seen = torch.tensor(sorted(seen_tokens[b]), device=idx.device)
                    scores = logits[b, seen]
                    logits[b, seen] = torch.where(
                        scores > 0, scores / repetition_penalty, scores * repetition_penalty
                    )

            if temperature > 0:
                logits = logits / temperature

                # Top-k filtering
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float('-inf')

                # Top-p (nucleus) filtering
                if top_p is not None and 0 < top_p < 1.0:
                    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
                    cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    # Mask tokens past the nucleus, always keeping the top token.
                    remove = cum_probs > top_p
                    remove[..., 1:] = remove[..., :-1].clone()
                    remove[..., 0] = False
                    sorted_logits[remove] = float('-inf')
                    logits = sorted_logits.scatter(-1, sorted_idx, sorted_logits)

                # Sample
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                idx_next = logits.argmax(dim=-1, keepdim=True)

            # Append
            idx = torch.cat([idx, idx_next], dim=1)

            if eos_id is not None and (idx_next == eos_id).any():
                break
            if seen_tokens is not None:
                for b in range(idx.size(0)):
                    seen_tokens[b].add(int(idx_next[b].item()))

        return idx

    @torch.no_grad()
    def generate_stream(
        self,
        idx: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = None,
        repetition_penalty: float = 1.0,
        eos_id: Optional[int] = None
    ):
        """Like generate(), but yields each new token id one at a time.

        Batch size must be 1. Intended for streaming UIs. The eos token (if
        it ends the sequence) is NOT yielded.
        """
        self.eval()
        assert idx.size(0) == 1, "generate_stream supports batch size 1"
        kv_caches = None
        seen_tokens: Optional[set] = None

        for _ in range(max_new_tokens):
            idx_cond, kv_caches = self._next_input(idx, kv_caches)
            logits, _, kv_caches = self.forward(idx_cond, kv_caches=kv_caches, use_cache=True)
            logits = logits[:, -1, :]

            if repetition_penalty != 1.0:
                if seen_tokens is None:
                    seen_tokens = set(idx[0].tolist())
                seen = torch.tensor(sorted(seen_tokens), device=idx.device)
                scores = logits[0, seen]
                logits[0, seen] = torch.where(
                    scores > 0, scores / repetition_penalty, scores * repetition_penalty
                )

            if temperature > 0:
                logits = logits / temperature

                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float('-inf')

                if top_p is not None and 0 < top_p < 1.0:
                    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
                    cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    remove = cum_probs > top_p
                    remove[..., 1:] = remove[..., :-1].clone()
                    remove[..., 0] = False
                    sorted_logits[remove] = float('-inf')
                    logits = sorted_logits.scatter(-1, sorted_idx, sorted_logits)

                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                idx_next = logits.argmax(dim=-1, keepdim=True)

            idx = torch.cat([idx, idx_next], dim=1)
            nxt = int(idx_next.item())
            if eos_id is not None and nxt == eos_id:
                break
            if seen_tokens is not None:
                seen_tokens.add(nxt)
            yield nxt

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_optimizer(self, config: ModelConfig):
        """Create optimizer with weight decay."""
        # Separate parameters for weight decay
        decay_params = []
        no_decay_params = []
        
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if 'bias' in name or 'ln' in name or 'emb' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        optimizer_groups = [
            {'params': decay_params, 'weight_decay': config.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]
        
        return torch.optim.AdamW(
            optimizer_groups,
            lr=config.learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8
        )
