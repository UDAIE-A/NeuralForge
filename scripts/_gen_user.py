import sys
sys.path.insert(0, ".")
from scripts.teach_loop import load_for_learning, generate

device = "cuda"
model, tok, cfg = load_for_learning("checkpoints/small.pt", device)
for q in ["user: hi", "user: what is the sun?", "user: how are you?",
          "user: what is 2+2?", "user: what can you do?",
          "user: are you a robot?", "user: what color is the sky?"]:
    out = generate(model, tok, q, device, max_tokens=60,
                   temperature=0.7, top_k=40, top_p=0.9, repetition_penalty=1.1)
    print(f"  {q!r:28} -> {out!r}")
