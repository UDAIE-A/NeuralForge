#!/usr/bin/env python3
"""
Overnight training script for NeuralForge conversational model.
Run this before going to bed - it'll train while you sleep!
"""

import subprocess
import sys

# Training configuration for overnight run
# Using 'base' preset for better quality, or 'small' for faster training
COMMAND = [
    sys.executable, "train.py",
    "--preset", "small",
    "--data", "data/combined_train.txt",
    "--vocab-size", "10000",
    "--epochs", "10",
    "--seq-len", "256",
    "--batch-size", "20",
    "--lr", "3e-4",
    "--name", "conversational_overnight",
    "--checkpoint-dir", "checkpoints/conversational"
]

print("=" * 70)
print("  NEURALFORGE OVERNIGHT TRAINING")
print("=" * 70)
print("  This will train the model on conversational data while you sleep.")
print("  The training will take several hours depending on your GPU.")
print("=" * 70)
print()
print("  Command:", " ".join(COMMAND))
print()

# Run the training
subprocess.run(COMMAND)
