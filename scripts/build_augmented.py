import os

# 20 curriculum Q&A in the chat format the base was pretrained on.
QA = [
    ("user: hi", "assistant: hello!"),
    ("user: hello", "assistant: hi there!"),
    ("user: hey", "assistant: hey!"),
    ("user: good morning", "assistant: good morning!"),
    ("user: good evening", "assistant: good evening!"),
    ("user: good night", "assistant: good night!"),
    ("user: bye", "assistant: goodbye!"),
    ("user: thank you", "assistant: you are welcome!"),
    ("user: thanks", "assistant: you are welcome!"),
    ("user: how are you?", "assistant: i am good, thank you!"),
    ("user: what is your name?", "assistant: i am neuralforge!"),
    ("user: who are you?", "assistant: i am neuralforge, a language model!"),
    ("user: what can you do?", "assistant: i can chat and learn!"),
    ("user: are you a robot?", "assistant: i am software, not a robot!"),
    ("user: what is 2+2?", "assistant: four!"),
    ("user: what is the sun?", "assistant: the sun is a star!"),
    ("user: what color is the sky?", "assistant: the sky is blue!"),
    ("user: how many legs does a cat have?", "assistant: four legs!"),
    ("user: what do birds have?", "assistant: birds have wings!"),
    ("user: what is water?", "assistant: water is a liquid!"),
]

REPEAT = 40
out_path = "data/augmented_train.txt"

parts = []
for f in ["data/conversational_train_large.txt", "data/conversational_train.txt"]:
    with open(f, "r", encoding="utf-8", errors="ignore") as fh:
        parts.append(fh.read().strip())

block = []
for _ in range(REPEAT):
    for q, a in QA:
        block.append(f"{q}\n{a}")
parts.append("\n\n".join(block))

with open(out_path, "w", encoding="utf-8") as fh:
    fh.write("\n\n".join(parts))

print(f"Wrote {out_path}: {sum(len(p) for p in parts):,} chars, Q&A repeated {REPEAT}x")
