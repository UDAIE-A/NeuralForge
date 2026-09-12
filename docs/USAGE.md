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
python generate.py --checkpoint checkpoints/small.pt --prompt "Alice " --max-tokens 200 --top-p 0.9
```

---

## Training (`train.py`)

```bash
python train.py --data <file> [options]
```

### Train arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--config` | path | `None` | JSON file of run settings (see `configs/`). Any key matching a flag below; an explicit flag still wins. |
| `--data` | path/str | **required** | Training text file (or a literal string of text). May be supplied via `--config` instead. |
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
| `--name` | str | `<preset>` | Base model name. Produces `<name>.pt`, `<name>_train.pt`, and `<name>_best.pt`. |
| `--grad-accum` | int | `1` | Gradient accumulation steps — simulates a larger batch without more VRAM. |
| `--num-workers` | int | platform | DataLoader workers (`0` on Windows, `8` elsewhere). |
| `--warmup-steps` | int | adaptive | LR warmup steps (default: capped at ~10% of the run). |
| `--stride` | int | `seq_len` | Sliding-window stride between training sequences. The default gives **non-overlapping** windows, so each token is seen once per epoch. A smaller stride shows every token multiple times and accelerates memorization. |
| `--dropout` | float | preset (`0.1`) | Dropout probability. Raise it when validation loss stalls above training loss. |
| `--early-stopping` | int | off | Stop after N consecutive evaluations with no validation improvement. |
| `--val-fraction` | float | `0.05` | Fraction of the corpus held out for validation when `--val-data` is not given. |
| `--max-steps` | int | `None` | Hard cap on optimizer steps regardless of `--epochs`, so a run can be sized in tokens rather than passes. |
| `--d-model`, `--n-heads`, `--n-layers`, `--d-ff` | int | preset | Override individual preset dimensions to define a custom model size. |
| `--no-compile` | flag | off | Disable `torch.compile` (auto-skipped if Triton is missing). |
| `--seed` | int | `None` | Random seed for reproducible runs (torch/cuda/python/numpy). |
| `--save-interval` | int | adaptive | Checkpoint save interval in optimizer steps (default: ~30 saves per run, so 300 steps = every 10, 3000 = every 100). |

**Set in code (not CLI flags)** — defaults in `neuralforge/training/trainer.py`:
`compile_model=True` (torch.compile), `eval_interval=500`.
The LR warmup is adaptive: with no `--warmup-steps`, it's capped at ~10% of
the run so short training sessions still reach the full LR and cosine decay.
Checkpoint saves are also adaptive: without `--save-interval`, the rolling
`<name>_train.pt` is written ~30 times per run (300 steps → every 10, 3000 →
every 100).

### Train examples

```bash
# Fast experiment — char tokenizer, single book
python train.py --preset small --data data/dracula.txt --char --epochs 50 --seq-len 256 --batch-size 24

# Serious run — BPE tokenizer on the big combined corpus
python train.py --preset small --data data/train_large.txt --vocab-size 8000 --epochs 20 --seq-len 384 --batch-size 12

# Explicit validation file
python train.py --preset small --data data/train.txt --val-data data/sample.txt --char --epochs 30

# Resume an interrupted run
python train.py --preset small --data data/train_large.txt --char --epochs 20 --resume checkpoints/small_train.pt

# Custom checkpoint directory and learning rate
python train.py --preset tiny --data data/alice.txt --char --epochs 100 --lr 5e-4 --checkpoint-dir runs/alice_tiny
```

During training you get a live dashboard: epoch/batch progress, loss + trend
sparkline, GPU utilization/memory/temperature, tokens/sec, and ETA. Press
`Ctrl+C` to save `<name>_train.pt` and print a resume command.

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
| `--seed` | int | `None` | Random seed for reproducible generation. |

### Generate examples

```bash
# Basic
python generate.py --checkpoint checkpoints/small.pt --prompt "It was " --max-tokens 200

# Higher-quality sampling (recommended): nucleus + repetition penalty
python generate.py --checkpoint checkpoints/small.pt --prompt "It was " \
    --max-tokens 300 --temperature 0.8 --top-p 0.9 --repetition-penalty 1.2

# More deterministic / focused
python generate.py --checkpoint checkpoints/small.pt --prompt "Chapter 1 " \
    --temperature 0.5 --top-k 20

# Interactive chat-style loop
python generate.py --checkpoint checkpoints/small.pt --interactive --top-p 0.9 --repetition-penalty 1.2

# Point at a specific tokenizer
python generate.py --checkpoint runs/alice_tiny/tiny.pt --prompt "Alice "
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
| Tokenizer training | Instant | Fast (heap-based merge learning) |
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
1. **Lower `--batch-size` first**, then `--seq-len`, if you hit OOM. To keep a large effective batch without the VRAM, raise `--grad-accum` instead.
2. Mixed precision (AMP) is always on, so memory is already optimized.
3. 32 GB system RAM is plenty for BPE training on `train_large.txt` — VRAM is the bottleneck, not RAM.

---

## Checkpoints

- Saved to `--checkpoint-dir` (default `checkpoints/`): one rolling
  `<name>_train.pt` while training, `<name>_best.pt` for the best validation
  snapshot, and `<name>.pt` as the published final model.
- The tokenizer is saved alongside as `tokenizer.pkl`. Published `<name>.pt`
  models also embed the tokenizer, so generation can usually load them without
  a separate `--tokenizer` argument.
- On successful completion, `<name>.pt` is published from the **best-validation**
  weights (not the last epoch's) whenever a validation split exists; its
  `meta['weights_from']` records which. Only the bulky resumable
  `<name>_train.pt` is deleted — `<name>_best.pt` is kept.
- **Architecture note:** the model now uses RoPE + SwiGLU + RMSNorm.
  Checkpoints from before that change won't load on `main`; check out the
  `v0-legacy-arch` tag to use them, then `git checkout main` to return.

---

## Best-use-case recipes

**A. "I just want to see it work" (minutes)**
```bash
python train.py --preset tiny --data data/alice.txt --char --epochs 50 --seq-len 128 --batch-size 32
python generate.py --checkpoint checkpoints/tiny.pt --prompt "Alice " --top-p 0.9
```

**B. "A decent model overnight on my 3060"**
```bash
python train.py --preset small --data data/train_large.txt --vocab-size 8000 --epochs 20 --seq-len 384 --batch-size 12
python generate.py --checkpoint checkpoints/small_best.pt --prompt "The " --max-tokens 300 --top-p 0.9 --repetition-penalty 1.2
```

**C. "Train on my own text"**
```bash
# Point --data at any .txt file (book, articles, code, chat logs...)
python train.py --preset small --data path/to/my_corpus.txt --char --epochs 30 --seq-len 256 --batch-size 16
```

**D. "Resume after Ctrl+C"**
```bash
python train.py --preset small --data data/train_large.txt --char --epochs 20 --resume checkpoints/small_train.pt
```

---

## Changing clothes in a photo (`scripts/change_clothes.py`)

Separate from the text model. Takes one photo of a person and repaints only the
clothes from a text prompt - no training. Face, hair, skin and background are kept
pixel-identical. Uses a clothing-segmentation model for the mask and a Stable
Diffusion inpainting checkpoint to fill it (downloads ~2 GB once).

```bash
pip install -r requirements-image.txt
python scripts/change_clothes.py photo.jpg "a black leather jacket and grey trousers" --cover-arms
python scripts/change_clothes.py photo.jpg "a red floral summer dress" --num 4
python scripts/change_clothes.py photo.jpg "grey hoodie" --parts upper
```

| Flag | Meaning |
|---|---|
| `--parts upper\|lower\|full` | which clothes to replace (default `full`) |
| `--cover-arms` / `--cover-legs` | also repaint bare skin - needed for sleeves over a tank top, pants over shorts |
| `--num N` / `--seed S` | number of variations / reproducible seed |
| `--sdxl` | SDXL inpainting at 1024 px (slower, ~7 GB download) |
| `--model ID` | any diffusers inpainting checkpoint |
| `--mask-only` | write `<name>_mask.png` and stop, to check what will be repainted |

Outputs go to `outputs/clothes/`. ~8 s per image on an RTX 3060 at 576x768.
Only use it on photos of yourself or people who have agreed to it.

Weights live in `checkpoints/` (gitignored). Fetch them once with
`python scripts/download_image_models.py` (~7 GB); after that everything runs offline.

### Teaching it one person (`scripts/train_identity_lora.py`)

DreamBooth-style LoRA on Realistic Vision 5.1 from a folder of photos of one person.
Auto-crops a person shot and a face close-up per photo, trains only rank-16 adapters on
the UNet attention (3.2M params, ~2.2 GB VRAM, ~7 min for 800 steps on a 3060), then
renders sample images.

```bash
python scripts/train_identity_lora.py --images "C:/photos/me" --name me
python scripts/train_identity_lora.py --name me --sample-only --prompt "photo of ohwx person hiking"
python scripts/change_clothes.py photo.jpg "a navy suit" --lora checkpoints/lora/me
```

Output: `checkpoints/lora/me/` with `pytorch_lora_weights.safetensors`, `token.txt`
(`ohwx person` - use it in prompts), `train_crops/` (inspect these!) and `samples/`.
Likeness scales with data: 3 clear faces gives "same kind of person"; 10-15 sharp,
varied, overlay-free face photos gives a recognisable identity. Then raise `--steps`
to 1200-1500. Only for your own photos or people who have agreed.

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
