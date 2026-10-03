"""Quick probe: does the learned checkpoint greet properly?

Uses the shared chat helpers and the curriculum scorer rather than carrying
private copies of either - the old local `similar()` here passed on a single
loose substring hit, so it reported success against unrelated replies.
"""

import os
import re
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

from neuralforge.core import NeuralForge
from neuralforge.chat import CHAT_PROMPT_TMPL, decode_reply

CHECKPOINT = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/learned.pt"

# Reuse the curriculum's scorer so this probe and the curriculum agree.
_src = open("scripts/teach_basics_loop.py", encoding="utf-8").read()
_ns = {"re": re}
exec(_src[_src.index("def _content_words"):_src.index("def main()")], _ns)
similar = _ns["similar"]


def generate(model, tokenizer, prompt, device, max_tokens=120, temperature=0.7):
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(x, max_new_tokens=max_tokens, temperature=temperature,
                         eos_id=getattr(tokenizer, "eos_id", None))
    return decode_reply(tokenizer, ids, out[0].tolist())


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    config = checkpoint["config"]
    config.device = device
    tokenizer = checkpoint["tokenizer_obj"]
    model = NeuralForge(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    answer = "Hello! How can I help you today?"
    reply = generate(model, tokenizer, CHAT_PROMPT_TMPL.format(prompt="hello"), device)

    print(f"Checkpoint: {CHECKPOINT}")
    print(f"Tokenizer : version {getattr(tokenizer, 'version', 1)}")
    print("Prompt    : hello")
    print(f"Expected  : {answer}")
    print(f"Got       : {reply}")
    print(f"Similar   : {similar(reply, answer)}")


if __name__ == "__main__":
    main()
