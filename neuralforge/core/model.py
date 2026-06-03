import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .config import ModelConfig


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with causal masking."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.d_head = config.d_head
        self.d_model = config.d_model
        
        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        # Causal mask
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.max_seq_len, config.max_seq_len))
            .view(1, 1, config.max_seq_len, config.max_seq_len)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, T, C = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        
        # Handle KV cache for efficient generation
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)
        
        new_cache = (k, v)
        
        # Compute attention scores
        scale = math.sqrt(self.d_head)
        
        # Use Flash Attention for speed
        T_q = q.shape[2]
        T_k = k.shape[2]
        
        # Create causal mask for flash attention
        causal_mask = torch.tril(torch.ones(T_q, T_k, device=x.device, dtype=torch.bool))
        
        att = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=causal_mask,
            dropout_p=self.config.dropout if self.training else 0,
            is_causal=False
        )
        
        out = att.transpose(1, 2).contiguous().view(B, T, C)
        out = self.resid_dropout(self.o_proj(out))
        
        return out, new_cache


class FeedForward(nn.Module):
    """Position-wise feed-forward network with GELU activation."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.d_model, config.d_ff)
        self.fc2 = nn.Linear(config.d_ff, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Single transformer block with pre-norm architecture."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = MultiHeadAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.ffn = FeedForward(config)
    
    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # Pre-norm architecture
        residual = x
        x = self.ln1(x)
        attn_out, new_cache = self.attn(x, kv_cache)
        x = residual + attn_out
        
        residual = x
        x = self.ln2(x)
        x = residual + self.ffn(x)
        
        return x, new_cache


class NeuralForge(nn.Module):
    """NeuralForge: A GPT-style decoder-only transformer.
    
    Built from scratch with no dependencies on external models.
    Architecture: Token embedding + Positional embedding + Transformer blocks + LM head
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Token and position embeddings
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        self.emb_dropout = nn.Dropout(config.dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])
        
        # Final layer norm
        self.ln_f = nn.LayerNorm(config.d_model)
        
        # Language model head (weight tied with token embedding)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Weight tying
        self.lm_head.weight = self.tok_emb.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
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
        
        # Token and position embeddings
        tok_emb = self.tok_emb(idx)
        positions = torch.arange(0, T, dtype=torch.long, device=device).unsqueeze(0)
        pos_emb = self.pos_emb(positions)
        x = self.emb_dropout(tok_emb + pos_emb)
        
        # Transformer blocks
        new_caches = []
        for i, block in enumerate(self.blocks):
            cache = kv_caches[i] if kv_caches is not None else None
            x, new_cache = block(x, cache)
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
    
    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: Optional[int] = 50
    ) -> torch.Tensor:
        """
        Generate text autoregressively.
        
        Args:
            idx: Starting token indices (B, T)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: If set, only sample from top-k tokens
            
        Returns:
            Generated token indices (B, T + max_new_tokens)
        """
        self.eval()
        kv_caches = None
        
        for _ in range(max_new_tokens):
            # First step: feed the full prompt; subsequent steps: only last token
            if kv_caches is None:
                idx_cond = idx if idx.size(1) <= self.config.max_seq_len else \
                           idx[:, -self.config.max_seq_len:]
            else:
                idx_cond = idx[:, -1:]
            
            # Forward pass with cache
            logits, _, kv_caches = self.forward(
                idx_cond,
                kv_caches=kv_caches,
                use_cache=True
            )
            
            # Get logits for last position
            logits = logits[:, -1, :] / temperature
            
            # Top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append
            idx = torch.cat([idx, idx_next], dim=1)
        
        return idx
    
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
