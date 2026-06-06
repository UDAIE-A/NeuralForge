# NeuralForge Usage Guide

Everything you need to train a model and generate text: full argument
reference, runnable examples, recommended settings, and troubleshooting.

- [Setup](#setup)
- [Quick start](#quick-start)
- [Training (`train.py`)](#training-trainpy)
  - [Arguments](#train-arguments)
  - [Examples](#train-examples)
- [Generation (`generate.py`)](#generation-generatepy)
  - [Arguments](#generate-arguments)
  - [Examples](#generate-examples)
- [Model presets](#model-presets)
- [Choosing a tokenizer](#choosing-a-tokenizer)
- [Sampling guide](#sampling-guide)
- [Recommended settings for a 12 GB GPU (RTX 3060)](#recommended-settings-for-a-12-gb-gpu-rtx-3060)
- [Checkpoints](#checkpoints)
- [Best-use-case recipes](#best-use-case-recipes)
- [Troubleshooting](#troubleshooting)

---

## Setup

```bash
git clone https://github.com/UDAIE-A/NeuralForge.git
cd NeuralForge
python -m venv venv
.\venv\Scripts\Activate.ps1            # Windows PowerShell
pip install torch --index-url https://download.pytorch.org/whl/cu126
```

Requirements: Python 3.8+, PyTorch 2.0+, and an NVIDIA GPU with CUDA
(training and generation are GPU-only).

---

## Quick start

```bash
# 1. Train a small model fast (character tokenizer) on a single book
python train.py --preset small --data data/alice.txt --char --epochs 30 --seq-len 256 --batch-size 24

# 2. Generate from the result
python generate.py --checkpoint checkpoints/epoch_30.pt --prompt "Alice " --max-tokens 200 --top-p 0.9
```

---

## Training (`train.py`)

```bash
python train.py --data <file> [options]
```

### Train arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--data` | path/str | **required** | Training text file (or a literal string of text). |
| `--preset` | choice | `tiny` | Model size: `tiny`, `small`, `base`, `large`, `xl`, `xxl`. |
| `--val-data` | path | `None` | Validation text file. If omitted, the last 5% of the training text is auto-held-out for validation. |
| `--epochs` | int | `10` | Number of passes over the data. |
| `--vocab-size` | int | `8000` | Target vocab for the **BPE** tokenizer (ignored with `--char`). |
| `--seq-len` | int | `512` | Context length (tokens per training sequence). |
| `--batch-size` | int | `32` | Sequences per step. Lower this first if you run out of VRAM. |
| `--lr` | float | `3e-4` | Peak learning rate (cosine decay with warmup). |
| `--checkpoint-dir` | path | `checkpoints` | Where checkpoints and the tokenizer are saved. |
| `--resume` | path | `None` | Resume training from a checkpoint `.pt`. |
| `--char` | flag | off | Use the instant character-level tokenizer instead of BPE. |

**Set in code (not CLI flags)** — defaults in `neuralforge/training/trainer.py`:
`compile_model=True` (torch.compile), `keep_last=3` (checkpoint rotation),
`gradient_accumulation_steps=1`, `eval_interval=500`, `save_interval=1000`.
Data loading uses `num_workers=8` (see `neuralforge/training/data.py`).

### Train examples

```bash
# Fast experiment — char tokenizer, single book
python train.py --preset small --data data/dracula.txt --char --epochs 50 --seq-len 256 --batch-size 24

# Serious run — BPE tokenizer on the big combined corpus
python train.py --preset small --data data/train_large.txt --vocab-size 8000 --epochs 20 --seq-len 384 --batch-size 12

# Explicit validation file
python train.py --preset small --data data/train.txt --val-data data/sample.txt --char --epochs 30

# Resume an interrupted run
python train.py --preset small --data data/train_large.txt --char --epochs 20 --resume checkpoints/epoch_8.pt

# Custom checkpoint directory and learning rate
python train.py --preset tiny --data data/alice.txt --char --epochs 100 --lr 5e-4 --checkpoint-dir runs/alice_tiny
```

During training you get a live dashboard: epoch/batch progress, loss + trend
sparkline, GPU utilization/memory/temperature, tokens/sec, and ETA. Press
`Ctrl+C` to save an `*_interrupted.pt` checkpoint and print a resume command.

---

## Generation (`generate.py`)

```bash
python generate.py --checkpoint <file> [options]
```

### Generate arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--checkpoint` | path | **required** | Model checkpoint `.pt` to load. |
| `--tokenizer` | path | `<checkpoint_dir>/tokenizer.pkl` | Tokenizer file. Auto-detected as char or BPE. |
| `--prompt` | str | `"The "` | Starting text. |
| `--max-tokens` | int | `200` | Number of new tokens to generate. |
| `--temperature` | float | `0.8` | Randomness. Lower = focused, higher = creative. |
| `--top-k` | int | `50` | Sample only from the k most likely tokens. |
| `--top-p` | float | `None` | Nucleus sampling: keep the smallest set of tokens with cumulative probability ≥ p (e.g. `0.9`). |
| `--repetition-penalty` | float | `1.0` | `>1.0` discourages repeating tokens already generated (try `1.1`–`1.3`). |
| `--interactive` | flag | off | Prompt-response loop; type `quit` to exit. |

### Generate examples

```bash
# Basic
python generate.py --checkpoint checkpoints/epoch_50.pt --prompt "It was " --max-tokens 200

# Higher-quality sampling (recommended): nucleus + repetition penalty
python generate.py --checkpoint checkpoints/epoch_50.pt --prompt "It was " \
    --max-tokens 300 --temperature 0.8 --top-p 0.9 --repetition-penalty 1.2

# More deterministic / focused
python generate.py --checkpoint checkpoints/epoch_50.pt --prompt "Chapter 1 " \
    --temperature 0.5 --top-k 20

# Interactive chat-style loop
python generate.py --checkpoint checkpoints/epoch_50.pt --interactive --top-p 0.9 --repetition-penalty 1.2

# Point at a specific tokenizer
python generate.py --checkpoint runs/alice_tiny/epoch_100.pt --tokenizer runs/alice_tiny/tokenizer.pkl --prompt "Alice "
```

---

## Model presets

Parameter counts use each preset's default vocab; the embedding scales with
the actual tokenizer vocab at train time.

| Preset | Params | d_model | n_heads | n_layers | d_ff  | Fits 12 GB? |
|--------|--------|---------|---------|----------|-------|-------------|
| `tiny` | ~2M    | 128     | 4       | 4        | 512   | ✅ easily |
| `small`| ~12M   | 256     | 8       | 8        | 1024  | ✅ comfortable |
| `base` | ~138M  | 768     | 12      | 12       | 3072  | ⚠️ small batch/seq only |
| `large`| ~435M  | 1024    | 16      | 24       | 4096  | ❌ |
| `xl`   | ~2.2B  | 2048    | 32      | 32       | 8192  | ❌ |
| `xxl`  | ~22B   | 4096    | 32      | 80       | 16384 | ❌ (multi-GPU) |

---

## Choosing a tokenizer

| | Character (`--char`) | BPE (default) |
|---|---|---|
| Tokenizer training | Instant | Slower (learns merges) |
| Output quality | Lower | Higher |
| Best for | Quick experiments, tiny data | Serious runs, larger corpora |
| Vocab control | Fixed (unique chars) | `--vocab-size` (e.g. 8000) |

Rule of thumb: prototype with `--char`, train your keeper model with BPE.

---

## Sampling guide

- **temperature** — scales randomness. `0.5` = safe/repetitive, `0.8` = balanced, `1.0+` = adventurous.
- **top-k** — hard cap on candidate tokens. Small (10–20) = focused, large (50+) = diverse.
- **top-p** (nucleus) — adaptive alternative to top-k; `0.9`–`0.95` is a good range. You can combine with top-k or pass `--top-k 0`-style by leaving top-k and relying on top-p.
- **repetition-penalty** — `1.0` off; `1.1`–`1.3` curbs loops (very helpful for small models).

Good default for readable output:
`--temperature 0.8 --top-p 0.9 --repetition-penalty 1.2`.

---

## Recommended settings for a 12 GB GPU (RTX 3060)

`small` is the sweet spot. Start conservative, then raise `--batch-size`
until you're near ~11 GB (watch the dashboard's GPU memory), then back off.

```bash
# Recommended keeper run (BPE) on the big corpus
python train.py --preset small --data data/train_large.txt \
    --vocab-size 8000 --epochs 20 --seq-len 384 --batch-size 12

# Fast iteration (char)
python train.py --preset small --data data/train_large.txt \
    --char --epochs 15 --seq-len 256 --batch-size 24

# Pushing 'base' on 12 GB — only with tight settings
python train.py --preset base --data data/train_large.txt \
    --char --epochs 5 --seq-len 256 --batch-size 4
```

VRAM tips:
1. **Lower `--batch-size` first**, then `--seq-len`, if you hit OOM.
2. Mixed precision (AMP) is always on, so memory is already optimized.
3. 32 GB system RAM is plenty for BPE training on `train_large.txt` and 8 data workers — VRAM is the bottleneck, not RAM.

---

## Checkpoints

- Saved to `--checkpoint-dir` (default `checkpoints/`): `epoch_<n>.pt` each epoch,
  `step_<n>.pt` periodically, `best_model.pt` on best validation loss, and
  `epoch_<n>_interrupted.pt` on `Ctrl+C`.
- The tokenizer is saved alongside as `tokenizer.pkl` — keep it with the
  checkpoint; generation needs the matching tokenizer.
- **Rotation:** only the newest few `epoch_*`/`step_*` are kept (default 3);
  `best_model.pt` and `*interrupted*` are always preserved.
- **Architecture note:** the model now uses RoPE + SwiGLU + RMSNorm.
  Checkpoints from before that change won't load on `main`; check out the
  `v0-legacy-arch` tag to use them, then `git checkout main` to return.

---

## Best-use-case recipes

**A. "I just want to see it work" (minutes)**
```bash
python train.py --preset tiny --data data/alice.txt --char --epochs 50 --seq-len 128 --batch-size 32
python generate.py --checkpoint checkpoints/epoch_50.pt --prompt "Alice " --top-p 0.9
```

**B. "A decent model overnight on my 3060"**
```bash
python train.py --preset small --data data/train_large.txt --vocab-size 8000 --epochs 20 --seq-len 384 --batch-size 12
python generate.py --checkpoint checkpoints/best_model.pt --prompt "The " --max-tokens 300 --top-p 0.9 --repetition-penalty 1.2
```

**C. "Train on my own text"**
```bash
# Point --data at any .txt file (book, articles, code, chat logs...)
python train.py --preset small --data path/to/my_corpus.txt --char --epochs 30 --seq-len 256 --batch-size 16
```

**D. "Resume after Ctrl+C"**
```bash
python train.py --preset small --data data/train_large.txt --char --epochs 20 --resume checkpoints/epoch_7_interrupted.pt
```

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `CUDA out of memory` | Lower `--batch-size`, then `--seq-len`. Use a smaller preset. |
| `CUDA not available` | Training/generation require an NVIDIA GPU + CUDA build of PyTorch. |
| Old checkpoint won't load | It's the legacy architecture — `git checkout v0-legacy-arch`. |
| Garbled / wrong-vocab output | Tokenizer doesn't match the checkpoint. Pass the correct `--tokenizer`. |
| Output repeats/loops | Add `--repetition-penalty 1.2` and/or `--top-p 0.9`. |
| `torch.compile` warning | Harmless — it falls back to eager if Triton isn't available. |
| Tiny dataset warning | `batch_size` was capped to the dataset size; use more data or smaller `--seq-len`. |
