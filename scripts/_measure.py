import sys, time
sys.path.insert(0, ".")
from neuralforge.tokenizer.bpe import BPETokenizer
from neuralforge.training.data import read_text_input

SAMPLE = "data/conversational_train_large.txt"
with open(SAMPLE, "r", encoding="utf-8", errors="ignore") as f:
    text = f.read()

# 1) BPE train on a 300KB sample
sample = text[:300_000]
t0 = time.time()
tok = BPETokenizer()
tok.train(sample, vocab_size=8000, verbose=False)
dt_train = time.time() - t0
print(f"BPE train on 300KB: {dt_train:.1f}s")

# 2) encode throughput on 1MB
chunk = text[:1_000_000]
t0 = time.time()
ids = tok.encode(chunk, add_special_tokens=False)
dt_enc = time.time() - t0
print(f"encode 1MB ({len(ids)} toks): {dt_enc:.1f}s  -> {len(ids)/dt_enc:,.0f} tok/s")
print(f"  estimated encode of full 13MB: {dt_enc*13:.0f}s ({dt_enc*13/60:.1f} min)")
