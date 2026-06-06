# NeuralForge

A language model built from the ground up with no dependencies on external models.

## Overview

NeuralForge is a GPT-style decoder-only transformer language model implemented entirely from scratch. No pre-trained weights, no external model dependencies - pure PyTorch from random initialization.

### Features

- **Pure implementation**: No dependencies on existing model weights or architectures
- **Scalable**: From ~2M to 16B+ parameters
- **Modern architecture**: Rotary position embeddings (RoPE), SwiGLU feed-forward, RMSNorm
- **Dual tokenizers**: BPE tokenizer or fast character-level tokenizer
- **Flash Attention**: SDPA `is_causal` fast path for faster training
- **torch.compile**: JIT-compiled training by default (falls back to eager)
- **Rich sampling**: temperature, top-k, top-p (nucleus), and repetition penalty
- **Visual dashboard**: Real-time training metrics, GPU stats, loss trends
- **Auto validation split**: Holds out a slice of the data when none is given
- **Checkpoint rotation**: Keeps the latest few checkpoints plus the best model
- **GPU-only**: CUDA required for training
- **KV-cache**: Efficient autoregressive text generation

## Quick Start

### 1. Setup

```bash
git clone https://github.com/UDAIE-A/NeuralForge.git
cd NeuralForge
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cu126
```

### 2. Prepare data

Put your training data in a text file:

```bash
# Any text file works - books, articles, code, etc.
# Larger data = better results
```

### 3. Train

```bash
# Fast training with character-level tokenizer (recommended for quick tests)
python train.py --preset small --data data/train.txt --epochs 100 --batch-size 64 --seq-len 512 --char

# Better quality with BPE tokenizer (slower tokenizer training)
python train.py --preset small --data data/train.txt --epochs 100 --batch-size 64 --seq-len 512
```

### 4. Generate text

```bash
python generate.py --checkpoint checkpoints/epoch_100.pt --prompt "Alice" --max-tokens 200

# Better quality sampling: nucleus sampling + repetition penalty
python generate.py --checkpoint checkpoints/epoch_100.pt --prompt "Alice" \
    --max-tokens 200 --top-p 0.9 --repetition-penalty 1.2

# Interactive mode
python generate.py --checkpoint checkpoints/epoch_100.pt --interactive
```

## Model Sizes

Parameter counts use each preset's default vocab size (the embedding scales
with the actual tokenizer vocab at train time).

| Preset | Parameters | d_model | n_heads | n_layers | d_ff  | VRAM (approx) |
|--------|------------|---------|---------|----------|-------|---------------|
| tiny   | ~2M        | 128     | 4       | 4        | 512   | ~2 GB         |
| small  | ~12M       | 256     | 8       | 8        | 1024  | ~4 GB         |
| base   | ~138M      | 768     | 12      | 12       | 3072  | ~8 GB         |
| large  | ~435M      | 1024    | 16      | 24       | 4096  | ~12 GB        |
| xl     | ~2.2B      | 2048    | 32      | 32       | 8192  | ~24 GB        |
| xxl    | ~22B       | 4096    | 32      | 80       | 16384 | multi-GPU     |

## Tokenizers

### Character-level (`--char`)
- Instant training (no tokenizer learning needed)
- Faster training on small datasets
- Lower quality output
- Good for quick experiments

### BPE (default)
- Learns subword units from data
- Better quality output
- Slower tokenizer training
- Recommended for serious training

## Project Structure

```
neuralforge/
├── core/
│   ├── config.py          # Model configuration (tiny to xxl)
│   └── model.py           # Transformer: RoPE, SwiGLU, RMSNorm, Flash Attention
├── tokenizer/
│   ├── bpe.py             # BPE tokenizer from scratch
│   └── char_tokenizer.py  # Character-level tokenizer
├── training/
│   ├── data.py            # Dataset and DataLoader
│   └── trainer.py         # Training with visual dashboard
├── train.py               # Main training script
├── generate.py            # Text generation
├── data/                  # Training data
└── checkpoints/           # Saved models
```

## Training Dashboard

Real-time metrics during training:
- Epoch and batch progress bars
- Loss value and trend sparkline
- GPU utilization, memory, temperature
- Tokens per second throughput
- ETA and elapsed time
- Ctrl+C saves checkpoint and shows resume command

## Requirements

- Python 3.8+
- PyTorch 2.0+
- NVIDIA GPU with CUDA support
- 4-12 GB VRAM depending on model size

## Roadmap

- [x] Transformer architecture from scratch
- [x] BPE tokenizer
- [x] Character-level tokenizer
- [x] Flash Attention
- [x] Rotary position embeddings (RoPE)
- [x] SwiGLU feed-forward + RMSNorm
- [x] top-p / repetition-penalty sampling
- [x] torch.compile training
- [x] Visual training dashboard
- [x] GPU-only training
- [ ] Multi-GPU training
- [ ] Gradient checkpointing for large models
- [ ] Mixture of Experts for scaling
- [ ] RLHF alignment
- [ ] Instruction tuning

## Checkpoint compatibility

The model architecture changed (RoPE, SwiGLU, RMSNorm), so checkpoints
trained before that switch cannot be loaded by the current code. The previous
architecture is preserved at the `v0-legacy-arch` tag:

```bash
# Generate from old (pre-RoPE) checkpoints
git checkout v0-legacy-arch

# Return to the current architecture
git checkout main
```

New checkpoints trained on `main` are the way forward and should produce
better results.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Author

**UDAIE-A** - [GitHub](https://github.com/UDAIE-A)
