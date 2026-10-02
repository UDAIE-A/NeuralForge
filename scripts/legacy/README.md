# Legacy training scripts

These are superseded. Each one hand-rolled its own copy of the training loop,
LR schedule and checkpointing rather than calling `neuralforge.training.Trainer`,
so every bug fixed in the shared trainer had to be fixed in all of them —
and in practice was not. None of them got the best-checkpoint publish fix, the
no-overlap stride default, or early stopping.

They are kept only for reference. **Use `train.py --config configs/<name>.json`.**

| Legacy script | Replacement |
|---|---|
| `train_fresh.py` | `python train.py --config configs/small.json` |
| `train_medium.py` | `python train.py --config configs/medium.json` |
| `train_medium186.py` | `python train.py --config configs/medium186.json` |
| `train_100m.py`, `train_continue_medium.py` | `python train.py --config configs/pretrain_100m.json` |
| `train_overnight.py` | `python train.py --config configs/conversational.json` |
| `train_debug.py` | `python train.py --config configs/debug.json` |
| `lora_finetune.py` | `scripts/lora_finetune_v2.py` (adds assistant-only loss masking) |

Anything a script could express that `train.py` could not is now a flag:

- custom architectures → `--d-model --n-heads --n-layers --d-ff`
- token-budgeted runs → `--max-steps`
- overfitting controls → `--dropout --stride --early-stopping --val-fraction`

The `--tok-sample-kb` option several scripts carried is gone: BPE training no
longer needs a truncated sample, so the tokenizer always sees the full corpus.
