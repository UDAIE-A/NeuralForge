import sys, time
sys.path.insert(0, ".")
from neuralforge.tokenizer.bpe import BPETokenizer

with open("data/conversational_train_large.txt", "r", encoding="utf-8", errors="ignore") as f:
    text = f.read(1_000_000)  # 1MB sample
print("sample chars:", len(text))
for vs in (4000, 8000):
    t0 = time.time()
    tok = BPETokenizer()
    tok.train(text, vocab_size=vs, verbose=False)
    dt = time.time() - t0
    print(f"vocab={vs}: trained in {dt:.1f}s  -> extrapolated to 13MB: {dt*13:.0f}s ({dt*13/60:.1f} min)")
