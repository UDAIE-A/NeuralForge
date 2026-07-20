import sys
sys.path.insert(0, ".")
from neuralforge.tokenizer.bpe import BPETokenizer

tok = BPETokenizer()
tok.train("Hello world! The sun is a star. How are you today?", vocab_size=200, verbose=False)
print("space_id:", tok.space_id)
for s in ["Hello world!", "The sun is a star.", "how are you today?"]:
    ids = tok.encode(s, add_special_tokens=False)
    out = tok.decode(ids)
    print(f"in={s!r}  out={out!r}  has_space={(' ' in out)}")
