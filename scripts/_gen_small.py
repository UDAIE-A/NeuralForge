import sys
sys.path.insert(0, ".")
from scripts.teach_loop import load_for_learning, generate

device = "cuda"
model, tok, cfg = load_for_learning("checkpoints/small.pt", device)
print("loaded fresh small.pt | vocab", len(tok), "| space_id", tok.space_id)
for q in ["hi", "hello", "how are you?", "what is the sun?", "tell me a joke",
          "what is the capital of France?", "good morning"]:
    out = generate(model, tok, q, device, max_tokens=60,
                   temperature=0.8, top_k=40, top_p=0.9, repetition_penalty=1.1)
    print(f"  {q!r:32} -> {out!r}")
