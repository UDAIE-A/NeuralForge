import time, sys
sys.path.insert(0, '.')
from neuralforge.tokenizer import BPETokenizer

txt = open('data/clean_corpus.txt', encoding='utf-8', errors='ignore').read()
for mb in [0.5, 1]:
    n = int(mb * 1024 * 1024)
    s = txt[:n]
    t = time.time()
    tk = BPETokenizer()
    tk.train(s, vocab_size=8000, verbose=False)
    dt = time.time() - t
    print(f'{mb}MB BPE train -> {len(tk.vocab)} tokens in {dt:.1f}s', flush=True)
