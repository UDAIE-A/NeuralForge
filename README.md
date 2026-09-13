# NeuralForge

A language model built from the ground up with no dependencies on external models.

## Overview

NeuralForge is a GPT-style decoder-only transformer language model implemented entirely from scratch. No pre-trained weights, no external model dependencies - pure PyTorch from random initialization.

### Features

- **Pure implementation**: No dependencies on existing model weights or architectures
- **Scalable**: From ~2M to 16B+ parameters
- **Modern architecture**: Rotary position embeddings (RoPE), SwiGLU feed-forward, RMSNorm
- **Dual tokenizers**: whitespace-preserving BPE (space-prefixed tokens, real newlines) or a fast character-level tokenizer
- **Flash Attention**: SDPA `is_causal` fast path for faster training
- **torch.compile**: JIT-compiled training by default (falls back to eager)
- **Rich sampling**: temperature, top-k, top-p (nucleus), and repetition penalty
- **Visual dashboard**: Real-time training metrics, GPU stats, loss trends
- **Auto validation split**: Holds out a slice of the data when none is given
- **Named model artifacts**: train to `<name>_train.pt`, keep `<name>_best.pt`, publish `<name>.pt` from the best-validation weights
- **Overfitting controls**: non-overlapping windows by default, `--dropout`, and `--early-stopping` on validation loss
- **Config-driven runs**: `train.py --config configs/<name>.json` instead of a script per model size
- **GPU-only**: CUDA required for training
- **KV-cache**: Efficient autoregressive text generation

## Documentation

📖 **[Full Usage Guide → docs/USAGE.md](docs/USAGE.md)** — complete argument
reference for `train.py` and `generate.py`, runnable examples, sampling guide,
recommended settings for 12 GB GPUs, best-use-case recipes, and troubleshooting.

## NeuralForge Studio (Web UI)

A professional, ChatGPT-style web interface with a **side-by-side** layout:
chat with your model on one side, train/track/tune it on the other — all live.

```bash
pip install -r requirements.txt          # installs fastapi + uvicorn
python -m webui.server                    # then open http://127.0.0.1:8000
```

- **Chat panel** — pick a checkpoint and generate, with token-by-token streaming
  and live sampling controls (temperature, top-k, top-p, repetition penalty).
- **Admin panel** — configure and launch training (preset, data, tokenizer,
  epochs, seq-len, batch, lr…), then watch **live** loss curve, epoch/batch
  progress, tokens/sec, ETA, best validation loss, and GPU utilization /
  memory / temperature. Stop a run any time.

## NeuralForge Learn (live, human-in-the-loop learning)

Teach the model *during* a conversation. Every reaction you give — approve,
reject, or a corrected answer — becomes a few gradient steps that nudge the
model toward what you want, immediately. No big dataset, no offline run needed.

### The catch, and what fixes it

Naive online teaching destroys the model. Measured on a 186M checkpoint —
teaching five facts with six full-parameter steps each turned an **untaught**
`hello` from

```
Hello! It's nice to talk to you. How can I help?
```

into `Romeo and Romeo and Romeo and Romeo...`, while held-out loss on
pretraining prose moved only **+0.7%**. Perplexity on prose does not detect
chat-behaviour collapse, so `OnlineLearner` ships four guards:

| guard | what it does |
|---|---|
| **Replay** (`replay_text=`) | mixes real corpus batches into every teaching step |
| **Joint teaching** (`teach_many`) | trains all lessons together — teaching them one after another lets the last one dominate |
| **Dropout off** (default) | makes teaching deterministic and the reported loss delta honest |
| **LoRA mode** (`lora_rank=`) | trains a ~2.5% adapter, leaving base weights untouched |

Two results worth knowing before you use this:

**Replay only preserves what it contains.** Replaying prose while your control
prompts are chat does almost nothing. Same config, swapping the replay corpus:

```
replay = training_40mb.txt (prose)  ->  chat drift +0.65   "Romeo and Romeo and..."
replay = combined_conv.txt (chat)   ->  chat drift +0.04   "Hello! Ready to chat?"
```

**Teach jointly, not sequentially.** Sequential teaching scored 89% on the
lessons; joint teaching scored 94% *and* cut drift 4×.

```
mode                                          learned   drift
sequential | no replay | dropout ON               89%   +0.42
joint      | replay ON | dropout off              94%   +0.11
joint      | replay ON | LoRA r=16                94%   +0.10
```

Reproduce it (writes nothing without `--save`):

```bash
python scripts/teach_safe_demo.py
```

### Teaching a skill with LoRA

```bash
python scripts/teach_skill_lora.py                    # country -> capital, 238 pairs
python scripts/teach_skill_lora.py --no-lora          # full fine-tune instead
python scripts/teach_skill_lora.py --save checkpoints/skilled.pt
```

Splits the data into taught / held-out and reports them separately, because
the two measure different things: held-out accuracy on *facts* the model was
never told is expected to stay near zero — the part a skill can actually
generalize is the answer **format**.

### CLI

```bash
# Interactive teaching chat (GPU, small model)
python learn.py --checkpoint checkpoints/small.pt --interactive

# Scripted self-test
python learn.py --checkpoint checkpoints/small.pt --test
```

In the interactive session, after each reply type `y` (approve), `n` (reject,
then optionally a fix), `<text>` (teach that better answer), or `s` (skip).
Interactions are logged to `checkpoints/feedback.jsonl` and can be exported
with `--export data/learned_interactions.txt` for a full offline run.

> `--test` only proves that gradient descent reduces loss on the example it
> just trained on — it cannot fail. Use `scripts/teach_safe_demo.py` to see
> whether teaching actually helped without breaking something else.

### Web UI

In NeuralForge Studio, each assistant reply has a feedback toolbar (👍 approve,
👎 reject, ✏ correct). Reacting runs live gradient steps on the selected model;
the **Live Learning** card shows interactions/steps and last loss, and can
**Save** the learned weights or **Export** the corpus.

Implementation: `neuralforge/learning/` — `online.py` (`teach` / `teach_many` /
`approve` / `reject`, masked next-token loss, JSONL store), `replay.py`
(`ReplayBuffer`, `RegressionProbe`), `lora.py` (`inject_lora`, `merge_lora`).

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
# From a run config (recommended - see configs/)
python train.py --config configs/small.json

# Or entirely from flags. Character-level tokenizer for quick tests:
python train.py --preset small --data data/train.txt --epochs 100 --batch-size 64 --seq-len 512 --char

# BPE tokenizer (default), with overfitting guards on
python train.py --preset small --data data/train.txt --epochs 100 --batch-size 64 \
    --seq-len 512 --early-stopping 3 --dropout 0.15
```

Run configs in `configs/` hold the settings that used to live in a family of
near-identical `scripts/train_*.py` files. Explicit flags override the file.

### 4. Generate text

```bash
python generate.py --checkpoint checkpoints/small.pt --prompt "Alice" --max-tokens 200

# Better quality sampling: nucleus sampling + repetition penalty
python generate.py --checkpoint checkpoints/small.pt --prompt "Alice" \
    --max-tokens 200 --top-p 0.9 --repetition-penalty 1.2

# Interactive mode
python generate.py --checkpoint checkpoints/small.pt --interactive
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
- **Whitespace-preserving**: spaces and newlines are part of the token stream,
  so the vocabulary learns space-prefixed units (`" the"`) and can reproduce
  line structure — which is what lets chat turns (`"User: …
Assistant: …"`)
  survive a round-trip
- Better quality output
- Recommended for serious training

Tokenizers trained before this change are version 1; they collapsed newlines
into a single space token. They keep their original behaviour when loaded, so
existing checkpoints still work, but a **retrain is needed to benefit** — the
vocabulary itself has no whitespace in it.

## Project Structure

```
neuralforge/
├── core/
│   ├── config.py          # Model configuration (tiny to xxl)
│   └── model.py           # Transformer: RoPE, SwiGLU, RMSNorm, Flash Attention
├── tokenizer/
│   ├── bpe.py             # Whitespace-preserving BPE from scratch
│   └── char_tokenizer.py  # Character-level tokenizer
├── training/
│   ├── data.py            # Dataset and DataLoader
│   └── trainer.py         # Training loop, dashboard, early stopping
├── learning/
│   ├── online.py          # OnlineLearner: teach / teach_many / approve / reject
│   ├── replay.py          # ReplayBuffer + RegressionProbe
│   └── lora.py            # LoRALinear, inject_lora, merge_lora
└── chat.py                # Chat prompt template + reply cleanup

train.py                   # Main training script (--config aware)
generate.py                # Text generation
learn.py                   # Interactive teaching CLI
configs/                   # Run configs for train.py --config
scripts/
├── verify_changes.py      # 52-check end-to-end verification
├── teach_safe_demo.py     # Teaching modes compared (learned vs retained)
├── teach_skill_lora.py    # Teach one small skill with LoRA
├── lora_finetune_v2.py    # Offline LoRA fine-tune, assistant-only loss
└── legacy/                # Superseded per-model training scripts
data/                      # Training data (gitignored)
checkpoints/               # Saved models
```

## LoRA fine-tuning

Offline instruction tuning with loss applied only to assistant tokens:

```bash
python scripts/lora_finetune_v2.py     --base-model checkpoints/medium186_v1.pt     --data data/oasst_clean.txt     --rank 32 --steps 2000
```

Role markers are derived from the tokenizer rather than hardcoded. This matters:
with hardcoded ids, pointing the script at a different vocabulary makes every
mask empty, the loss exactly `0.0`, and the run a **silent no-op** that still
prints a plausible loss curve. Windows containing no assistant tokens are now
skipped and counted rather than trained on as zeros.

Masking is per-turn — it supervises every assistant turn and stops each at the
next `User:`. The previous single-span version masked everything after the
first `Assistant:` to the end of the window, which on multi-turn data trained
the model to write the user's turns too (measured at 22.3% of windows on
`oasst_clean.txt`; now 0%).

## Verification

```bash
python scripts/verify_changes.py
```

52 checks covering tokenizer round-trip, merge correctness against a naive
reference, legacy-tokenizer compatibility, generation past the context window,
best-weight publishing, early stopping, and config precedence. Exits non-zero
on failure; takes about a minute and runs real short GPU training runs.

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
 - [x] Live online fine-tuning (NeuralForge Learn)
 - [x] Replay + LoRA guards against catastrophic forgetting
 - [x] LoRA fine-tuning with assistant-only loss masking
 - [x] Config-driven training runs
 - [x] Early stopping on validation loss
- [x] GPU-only training
- [ ] Conversation-boundary packing for instruction data
- [ ] Multi-GPU training
- [ ] Gradient checkpointing for large models
- [ ] Mixture of Experts for scaling
- [ ] RLHF alignment
- [x] Instruction tuning (LoRA, `scripts/lora_finetune_v2.py`)

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

### Tokenizer versions

Tokenizers are version-stamped. Version 1 (anything trained before the
whitespace fix) split on `\S+` and rejoined words with a single space token,
which deleted every newline, tab and blank line from the corpus before the
model saw it — so a `"User: q\nAssistant: a"` turn arrived as
`"User: q Assistant: a"` and no turn boundary could ever be learned:

```
v1:  encode("Assistant:") == encode("\nAssistant:") == encode(" Assistant:")
     -> [5779, 62] for all three
```

Version-1 tokenizers keep their original behaviour when loaded, so existing
checkpoints still work. But the vocabulary itself contains no whitespace, so
**a retrain is required to benefit** — LoRA cannot fix it, because it adapts
weights, not the vocabulary.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Author

**UDAIE-A** - [GitHub](https://github.com/UDAIE-A)
