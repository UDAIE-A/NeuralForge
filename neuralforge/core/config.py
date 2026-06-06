from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """Configuration for NeuralForge model.
    
    Start small (millions of params) and scale up to billions.
    """
    # Vocabulary
    vocab_size: int = 32000
    
    # Model dimensions
    d_model: int = 512      # Hidden dimension
    n_heads: int = 8        # Number of attention heads
    n_layers: int = 6       # Number of transformer blocks
    d_ff: int = 2048        # Feed-forward inner dimension
    
    # Sequence
    max_seq_len: int = 2048
    dropout: float = 0.1
    
    # Training
    learning_rate: float = 3e-4
    batch_size: int = 32
    warmup_steps: int = 4000
    weight_decay: float = 0.01
    
    # Device
    device: str = "cuda"
    
    def __post_init__(self):
        """Calculate derived values and validate."""
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        self.d_head = self.d_model // self.n_heads
    
    @property
    def num_parameters(self) -> int:
        """Estimate total parameters."""
        # Embedding
        emb = self.vocab_size * self.d_model
        # Positional
        pos = self.max_seq_len * self.d_model
        # Per transformer block
        attn = 4 * self.d_model * self.d_model  # Q, K, V, O projections
        ffn = 2 * self.d_model * self.d_ff       # Two linear layers
        ln = 4 * self.d_model                     # Two layer norms (weight + bias)
        block = attn + ffn + ln
        # Output head (tied with embedding, so 0)
        total = emb + pos + self.n_layers * block
        return total
    
    @classmethod
    def tiny(cls) -> "ModelConfig":
        """Tiny model (~2M params) for testing."""
        return cls(
            vocab_size=8000,
            d_model=128,
            n_heads=4,
            n_layers=4,
            d_ff=512,
            max_seq_len=512,
        )
    
    @classmethod
    def small(cls) -> "ModelConfig":
        """Small model (~11M params) for initial training."""
        return cls(
            vocab_size=16000,
            d_model=256,
            n_heads=8,
            n_layers=8,
            d_ff=1024,
            max_seq_len=1024,
        )
    
    @classmethod
    def base(cls) -> "ModelConfig":
        """Base model (~110M params)."""
        return cls(
            vocab_size=32000,
            d_model=768,
            n_heads=12,
            n_layers=12,
            d_ff=3072,
            max_seq_len=2048,
        )
    
    @classmethod
    def large(cls) -> "ModelConfig":
        """Large model (~340M params)."""
        return cls(
            vocab_size=32000,
            d_model=1024,
            n_heads=16,
            n_layers=24,
            d_ff=4096,
            max_seq_len=2048,
        )
    
    @classmethod
    def xl(cls) -> "ModelConfig":
        """XL model (~1.7B params)."""
        return cls(
            vocab_size=32000,
            d_model=2048,
            n_heads=32,
            n_layers=32,
            d_ff=8192,
            max_seq_len=4096,
        )
    
    @classmethod
    def xxl(cls) -> "ModelConfig":
        """XXL model (~16B params)."""
        return cls(
            vocab_size=64000,
            d_model=4096,
            n_heads=32,
            n_layers=80,
            d_ff=16384,
            max_seq_len=4096,
        )
    
    @classmethod
    def from_preset(cls, name: str) -> "ModelConfig":
        """Get config from preset name."""
        presets = {
            "tiny": cls.tiny,
            "small": cls.small,
            "base": cls.base,
            "large": cls.large,
            "xl": cls.xl,
            "xxl": cls.xxl,
        }
        if name not in presets:
            raise ValueError(f"Unknown preset: {name}. Available: {list(presets.keys())}")
        return presets[name]()
