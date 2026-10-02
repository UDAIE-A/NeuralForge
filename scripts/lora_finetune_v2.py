#!/usr/bin/env python3
"""
LoRA fine-tuning for NeuralForge with proper assistant-only loss masking.
"""

import argparse
import os
import sys
import time
import random
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neuralforge.core import NeuralForge, ModelConfig
from neuralforge.tokenizer import BPETokenizer
from neuralforge.training.data import read_text_input
from neuralforge.training import create_dataloaders
from neuralforge.chat import decode_reply
from neuralforge.learning.lora import (
    LoRALinear, inject_lora, count_lora_params, DEFAULT_TARGETS,
)


def role_markers(tokenizer):
    """Token ids for the "Assistant:" and "User:" markers, from the tokenizer.

    These used to be hardcoded as [5779, 62], which is correct for exactly one
    tokenizer. Point the script at any other vocabulary and the marker is never
    found, every mask is all zeros, and compute_masked_loss returns
    0.0 / clamp(0, min=1) == 0.0 - a silent no-op run that still prints a
    plausible loss curve. Deriving them removes that trap.
    """
    assistant = tokenizer.encode("Assistant:", add_special_tokens=False)
    user = tokenizer.encode("User:", add_special_tokens=False)
    if not assistant or not user:
        raise ValueError("Tokenizer produced an empty role marker")
    return assistant, user


def _find(seq, marker, start=0):
    n = len(marker)
    for i in range(start, len(seq) - n + 1):
        if seq[i:i + n] == marker:
            return i
    return -1


def build_assistant_mask(tokenizer, input_ids, device, markers=None):
    """Mask that is 1 only on Assistant response tokens.

    Supervises EVERY assistant turn in the window and stops each one at the
    next "User:" marker. The previous version masked everything after the
    FIRST "Assistant:" to the end of the window, which on multi-turn data
    trained the model to generate the user's turns too - measured at 22.3% of
    windows on oasst_clean.txt, and exactly the "keeps writing the next turn"
    behaviour that generation then has to strip back out.

    The mask is indexed by LOSS position, not input position: position i
    predicts targets[i] == input[i+1], so reply tokens input[start:end] are
    supervised at positions start-1 .. end-2. Masking input positions directly
    skipped the first token of every reply - the one that decides how the
    answer opens - and supervised the token after the reply instead.
    """
    assistant, user = markers if markers else role_markers(tokenizer)
    mask = torch.zeros_like(input_ids, dtype=torch.float32, device=device)

    for b in range(input_ids.shape[0]):
        seq = input_ids[b].tolist()
        pos = 0
        while True:
            a = _find(seq, assistant, pos)
            if a < 0:
                break
            start = a + len(assistant)
            nxt = _find(seq, user, start)
            end = nxt if nxt >= 0 else len(seq)
            # A reply cut off by the window end continues into targets[-1].
            mask[b, start - 1:end - 1 if nxt >= 0 else end] = 1.0
            pos = end if nxt >= 0 else len(seq)
            if pos >= len(seq):
                break
    return mask


def compute_masked_loss(logits, targets, mask):
    """Cross entropy loss only on masked positions."""
    # logits: (B, T, V), targets: (B, T), mask: (B, T)
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    loss = loss_fct(logits.view(-1, logits.size(-1)), targets.view(-1))
    loss = loss * mask.view(-1)
    denom = mask.sum()
    if denom == 0:
        # No supervised token anywhere in the batch. Returning 0.0 here would
        # be a silent no-op step; surface it as NaN-free zero WITH a flag so
        # the caller can skip the batch rather than "train" on nothing.
        return loss.sum() * 0.0, 0
    return loss.sum() / denom, int(denom.item())


# Evaluation prompts (same every time)
EVAL_PROMPTS = [
    "User: Hello\nAssistant:",
    "User: How are you?\nAssistant:",
    "User: What is a neural network?\nAssistant:",
    "User: Write a short poem about coding\nAssistant:",
    "User: Can you explain monopsony in economics?\nAssistant:",
    "User: How do I make a pizza?\nAssistant:",
    "User: What is Python?\nAssistant:",
    "User: Tell me a joke\nAssistant:",
]


def evaluate_model(model, tokenizer, device, max_new_tokens=100):
    """Generate from fixed prompts for consistent comparison."""
    model.eval()
    results = []
    for prompt in EVAL_PROMPTS:
        tokens = tokenizer.encode(prompt, add_special_tokens=False)
        x = torch.tensor([tokens], dtype=torch.long, device=device)
        with torch.no_grad():
            out = model.generate(x, max_new_tokens=max_new_tokens, temperature=0.7,
                                 top_p=0.9, eos_id=getattr(tokenizer, 'eos_id', None))
        reply = decode_reply(tokenizer, tokens, out[0].tolist())
        results.append((prompt, reply))
    model.train()
    return results


def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tune NeuralForge with assistant-only loss")
    parser.add_argument("--base-model", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--output", type=str, default="checkpoints/lora_adapter.pt")
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=32.0)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-split", type=float, default=0.05)
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--eval-every", type=int, default=250)
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        random.seed(args.seed)

    if not torch.cuda.is_available():
        print("ERROR: CUDA required")
        sys.exit(1)

    device = torch.device("cuda")

    # Load base model
    print(f"Loading base model: {args.base_model}")
    checkpoint = torch.load(args.base_model, map_location='cpu', weights_only=False)
    config = checkpoint['config']
    config.device = "cuda"
    tokenizer = checkpoint['tokenizer_obj']

    model = NeuralForge(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    # Inject LoRA - attention + MLP (better for instruction following)
    target_modules = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    print(f"\nInjecting LoRA into {target_modules} (rank={args.rank}, alpha={args.alpha})...")
    lora_layers = inject_lora(model, rank=args.rank, alpha=args.alpha, 
                              target_modules=target_modules, dropout=args.dropout)
    
    lora_params, total_params = count_lora_params(model)
    print(f"\nTrainable LoRA params: {lora_params:,} ({lora_params/1e6:.2f}M / {total_params/1e6:.1f}M = {100*lora_params/total_params:.2f}%)")

    # Prepare data
    print(f"\nLoading data: {args.data}")
    text = read_text_input(args.data)
    
    val_size = int(len(text) * args.val_split)
    train_text = text[:-val_size]
    val_text = text[-val_size:]
    
    print(f"Train: {len(train_text):,} chars, Val: {len(val_text):,} chars")

    train_loader, val_loader = create_dataloaders(
        train_text, val_text, tokenizer,
        seq_len=args.seq_len, batch_size=args.batch_size,
        stride=args.seq_len, num_workers=0,
    )

    # Only optimize LoRA params
    lora_params_list = [p for n, p in model.named_parameters() if 'lora' in n]
    optimizer = torch.optim.AdamW(lora_params_list, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    
    def get_lr(step):
        if step < args.warmup:
            return args.lr * step / args.warmup
        progress = (step - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr * 0.5 * (1 + math.cos(math.pi * progress))
    
    markers = role_markers(tokenizer)
    print(f"Role markers from tokenizer: Assistant:{markers[0]}  User:{markers[1]}")

    scaler = torch.amp.GradScaler('cuda')
    model.train()
    skipped = 0
    
    print(f"\nTraining for {args.steps} steps...")
    print(f"Batches per epoch: {len(train_loader)}")
    print(f"Effective batch size: {args.batch_size * args.grad_accum}")
    print("=" * 60)

    step = 0
    best_val_loss = float('inf')
    train_iter = iter(train_loader)
    start_time = time.time()

    while step < args.steps:
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x = x.to(device)
        y = y.to(device)

        mask = build_assistant_mask(tokenizer, x, device, markers)
        if mask.sum() == 0:
            # No assistant tokens in this window - a real gradient is
            # impossible, so skip instead of taking a zero-loss "step".
            skipped += 1
            step += 1
            continue

        with torch.amp.autocast('cuda'):
            logits, _, _ = model(x, targets=y)
            loss, _ = compute_masked_loss(logits, y, mask)
            loss = loss / args.grad_accum

        scaler.scale(loss).backward()

        if (step + 1) % args.grad_accum == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(lora_params_list, 1.0)
            
            lr = get_lr(step)
            for g in optimizer.param_groups:
                g['lr'] = lr
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # Logging
        if step % 25 == 0:
            elapsed = time.time() - start_time
            tok_s = (step + 1) * args.batch_size * args.seq_len * args.grad_accum / elapsed
            current_lr = optimizer.param_groups[0]['lr']
            mask_ratio = mask.mean().item() if 'mask' in locals() else 0
            print(f"  Step {step:5d}/{args.steps} | Loss: {loss.item()*args.grad_accum:.4f} | LR: {current_lr:.2e} | Mask: {mask_ratio:.2%} | skip {skipped} | {tok_s:,.0f} tok/s | GPU: {torch.cuda.memory_allocated()/1024**3:.1f}GB")

        # Validation with masked loss
        if step % args.eval_every == 0 and step > 0:
            model.eval()
            val_loss = 0
            val_batches = 0
            with torch.no_grad():
                for vx, vy in val_loader:
                    vx, vy = vx.to(device), vy.to(device)
                    with torch.amp.autocast('cuda'):
                        logits, _, _ = model(vx, targets=vy)
                    v_mask = build_assistant_mask(tokenizer, vx, device, markers)
                    if v_mask.sum() == 0:
                        continue
                    vl, _ = compute_masked_loss(logits, vy, v_mask)
                    val_loss += vl.item()
                    val_batches += 1
                    if val_batches >= 50:
                        break
            
            val_loss /= max(val_batches, 1)
            print(f"  >> Val Loss (assistant-only): {val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                adapter_state = {n: p.detach().cpu() for n, p in model.named_parameters() if 'lora' in n}
                torch.save({
                    'adapter_state': adapter_state,
                    'config': {'rank': args.rank, 'alpha': args.alpha, 'target_modules': target_modules},
                    'step': step,
                    'val_loss': val_loss,
                }, args.output)
                print(f"  >> Saved best adapter to {args.output}")
            
            # Generation evaluation
            print(f"  >> Generation eval at step {step}:")
            results = evaluate_model(model, tokenizer, device)
            for prompt, reply in results:
                # Truncate for display, safe encode
                display = reply[:120] + ("..." if len(reply) > 120 else "")
                display = display.encode('ascii', 'replace').decode('ascii')
                prompt_short = prompt.split(chr(10))[0].encode('ascii', 'replace').decode('ascii')
                print(f"     {prompt_short} -> {display}")
            
            model.train()

        # Periodic checkpoint
        if step % args.save_every == 0 and step > 0:
            ckpt_path = args.output.replace('.pt', f'_step{step}.pt')
            adapter_state = {n: p.detach().cpu() for n, p in model.named_parameters() if 'lora' in n}
            torch.save({
                'adapter_state': adapter_state,
                'config': {'rank': args.rank, 'alpha': args.alpha, 'target_modules': target_modules},
                'step': step,
                'val_loss': val_loss if 'val_loss' in locals() else None,
            }, ckpt_path)
            print(f"  >> Checkpoint saved: {ckpt_path}")

        step += 1

    total_time = time.time() - start_time
    print("=" * 60)
    print(f"Done in {total_time/60:.1f} min | Best val loss: {best_val_loss:.4f}")
    print(f"Final adapter: {args.output}")


if __name__ == "__main__":
    main()