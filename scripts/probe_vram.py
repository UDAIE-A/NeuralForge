#!/usr/bin/env python3
"""Probe peak VRAM for a given preset to decide what fits on this GPU."""
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neuralforge.core import ModelConfig, NeuralForge

PRESET = sys.argv[1] if len(sys.argv) > 1 else "base"
VOCAB = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
BATCH = int(sys.argv[3]) if len(sys.argv) > 3 else 8
SEQ = int(sys.argv[4]) if len(sys.argv) > 4 else 256

cfg = ModelConfig.from_preset(PRESET)
cfg.vocab_size = VOCAB
cfg.max_seq_len = SEQ
cfg.device = "cuda"
cfg.batch_size = BATCH

print(f"Preset={PRESET} params~{cfg.num_parameters/1e6:.1f}M vocab={VOCAB} batch={BATCH} seq={SEQ}")
torch.cuda.reset_peak_memory_stats()
model = NeuralForge(cfg).cuda()
model.train()
x = torch.randint(0, VOCAB, (BATCH, SEQ), device="cuda")
y = torch.randint(0, VOCAB, (BATCH, SEQ), device="cuda")

opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
with torch.cuda.amp.autocast(dtype=torch.float16):
    logits = model(x)[0]
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1))
loss.backward()
opt.step()
torch.cuda.synchronize()

peak = torch.cuda.max_memory_allocated() / 1e9
total = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"peak VRAM used: {peak:.2f} GB / {total:.2f} GB  -> {'OK' if peak < total*0.92 else 'TOO BIG'}")
