# NeuralForge

A language model built from the ground up with no dependencies on external models.

## Overview

NeuralForge is a GPT-style decoder-only transformer language model implemented entirely from scratch. The goal is to build a model that starts small (millions of parameters) and can scale up to 120B+ parameters.

### Features

- **Pure implementation**: No dependencies on existing model weights or architectures
- **Scalable**: From 10M to 120B+ parameters
- **BPE Tokenizer**: Byte Pair Encoding tokenizer trained from scratch
- **Modern architecture**: Pre-norm transformer with GELU activation
- **Efficient generation**: KV-cache for fast autoregressive generation

## Architecture

```
Token Embedding + Positional Embedding
        ↓
   Dropout
        ↓
┌─────────────────┐
│ Transformer Block │ × N layers
│  ├─ Layer Norm   │
│  ├─ Self-Attention│
│  ├─ Residual     │
│  ├─ Layer Norm   │
│  ├─ Feed-Forward │
│  └─ Residual     │
└─────────────────┘
        ↓
   Layer Norm
        ↓
   LM Head (weight-tied)
        ↓
     Logits
```

## Quick Start

### 1. Install dependencies

```bash
pip install torch
```

### 2. Prepare training data

Create a text file with your training data:

```bash
# Example: use any text corpus
cp data/sample.txt data/train.txt
```

### 3. Train tokenizer and model

```bash
# Train a tiny model (~10M params)
python train.py --preset tiny --data data/train.txt --epochs 10

# Train a small model (~50M params)
python train.py --preset small --data train.txt --epochs 20 --vocab-size 16000
```

### 4. Generate text

```bash
# Single generation
python generate.py --checkpoint checkpoints/best_model.pt --prompt "The "

# Interactive mode
python generate.py --checkpoint checkpoints/best_model.pt --interactive
```

## Model Sizes

| Preset | Parameters | d_model | n_heads | n_layers | d_ff |
|--------|------------|---------|---------|----------|------|
| tiny   | ~10M       | 128     | 4       | 4        | 512  |
| small  | ~50M       | 256     | 8       | 8        | 1024 |
| base   | ~250M      | 768     | 12      | 12       | 3072 |
| large  | ~1B        | 1024    | 16      | 24       | 4096 |
| xl     | ~8B        | 2048    | 32      | 32       | 8192 |
| xxl    | ~70B       | 4096    | 32      | 80       | 16384|

## Project Structure

```
neuralforge/
├── core/
│   ├── __init__.py
│   ├── config.py        # Model configuration
│   └── model.py         # Transformer architecture
├── tokenizer/
│   ├── __init__.py
│   └── bpe.py           # BPE tokenizer
├── training/
│   ├── __init__.py
│   ├── data.py          # Data loading
│   └── trainer.py       # Training loop
├── inference/
│   ├── __init__.py
│   └── generate.py      # Text generation
├── data/
│   └── sample.txt       # Sample training data
├── train.py             # Main training script
├── generate.py          # Main generation script
├── requirements.txt
└── README.md
```

## Training Tips

1. **Start small**: Begin with the `tiny` preset to verify everything works
2. **Use enough data**: More data = better generalization
3. **Monitor validation loss**: Stop when it starts increasing (overfitting)
4. **Scale gradually**: Move from tiny → small → base as you get better results

## Roadmap

- [x] Basic transformer architecture
- [x] BPE tokenizer from scratch
- [x] Training pipeline
- [x] Text generation
- [ ] Multi-GPU training
- [ ] Gradient checkpointing for large models
- [ ] Mixture of Experts for scaling
- [ ] RLHF alignment
- [ ] Instruction tuning

## License

MIT
