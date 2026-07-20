#!/usr/bin/env python3
"""
Merge conversational data with literary data for training NeuralForge
to have better day-to-day conversations while maintaining general knowledge.
"""

import glob
import random

conversation_weight = 0.7
literature_weight = 0.3

def combine_datasets():
    # Read conversational data
    with open('conversational_train_large.txt', 'r', encoding='utf-8') as f:
        conversational = f.readlines()
    
    # Read literary data files
    literary_files = glob.glob('*.txt')
    literary_files = [f for f in literary_files if 'conversational' not in f and 'expand' not in f]
    
    literary = []
    for file in literary_files:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                literary.extend(f.readlines())
        except:
            pass
    
    # Combine with weights
    combined = []
    conv_idx = 0
    lit_idx = 0
    
    while conv_idx < len(conversational) or lit_idx < len(literary):
        if random.random() < conversation_weight and conv_idx < len(conversational):
            combined.append(conversational[conv_idx])
            conv_idx += 1
        elif lit_idx < len(literary):
            combined.append(literary[lit_idx])
            lit_idx += 1
        else:
            break
    
    # Write combined data
    with open('combined_train.txt', 'w', encoding='utf-8') as f:
        f.writelines(combined)
    
    print(f"Combined dataset created:")
    print(f"  Conversational: {conv_idx} lines ({conv_idx/len(combined)*100:.1f}%)")
    print(f"  Literary:       {lit_idx} lines ({lit_idx/len(combined)*100:.1f}%)")
    print(f"  Total:          {len(combined)} lines")
    
    return 'combined_train.txt'

if __name__ == "__main__":
    combine_datasets()
