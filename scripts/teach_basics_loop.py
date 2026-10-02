#!/usr/bin/env python3
"""
Teach the model the full conversational knowledge curriculum via LIVE learning.

Every pass:
  1. Teach the WHOLE curriculum (small steps), shuffled order.
  2. Test every lesson at chat temperature.
  3. Re-teach only the FAILING lessons (more steps), shuffled, repeatedly,
     until they pass or rounds run out.

The BEST-scoring checkpoint is saved, so degradation never destroys progress.

    python scripts/teach_basics_loop.py --checkpoint checkpoints/tiny.pt --passes 8
    python scripts/teach_basics_loop.py --checkpoint checkpoints/learned.pt --passes 8
"""

import os
import re
import sys
import random
import argparse

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neuralforge.core import NeuralForge
from neuralforge.tokenizer import BPETokenizer
from neuralforge.tokenizer.char_tokenizer import CharTokenizer
from neuralforge.learning import OnlineLearner
from neuralforge.chat import CHAT_PROMPT_TMPL, decode_reply
from data.knowledge_curriculum import CURRICULUM, all_lessons
from data.general_knowledge import GENERAL_KNOWLEDGE

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")


def load_for_learning(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    config.device = device
    if checkpoint.get("tokenizer_obj") is not None:
        tokenizer = checkpoint["tokenizer_obj"]
    else:
        import pickle
        tok_path = os.path.join(os.path.dirname(checkpoint_path), "tokenizer.pkl")
        with open(tok_path, "rb") as f:
            data = pickle.load(f)
        tokenizer = CharTokenizer.load(tok_path) if "char_to_id" in data else BPETokenizer.load(tok_path)
    model = NeuralForge(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model, tokenizer, config


def generate(model, tokenizer, prompt, device, max_tokens=120, temperature=0.0):
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(
        x, max_new_tokens=max_tokens, temperature=temperature,
        eos_id=getattr(tokenizer, "eos_id", None),
    )
    return decode_reply(tokenizer, ids, out[0].tolist())


def safe(s):
    return s.encode("cp1252", errors="replace").decode("cp1252")


def _content_words(text):
    """Lowercased word tokens, punctuation stripped, stopwords removed."""
    words = re.findall(r"[a-z0-9']+", text.lower())
    return [w for w in words if len(w) > 2 and w not in _STOPWORDS]


_STOPWORDS = {
    "the", "and", "are", "for", "you", "your", "that", "this", "with", "was",
    "not", "but", "can", "has", "have", "its", "it's", "they", "them", "our",
    "who", "how", "what", "when", "where", "why", "from", "into", "than",
    "then", "there", "their", "these", "those", "will", "would", "about",
}


def similar(reply, target, min_coverage=0.6):
    """Does the reply actually contain the target answer?

    The previous version passed on ONE loose substring hit anywhere in the
    reply (`hits >= max(1, len(words)//2)` with bare `str.find`), so "Hi" was
    scored a PASS against "Yes, cats are cute and independent." It reported
    100% while the model was answering at random.

    This requires most of the target's content words to appear as WHOLE words,
    in order, and rejects a reply that pads a few right words into a wall of
    unrelated text.
    """
    r, t = reply.strip().lower(), target.strip().lower()
    if not t:
        return False
    if t in r:
        return True

    target_words = _content_words(t)
    if not target_words:
        # Nothing but stopwords (e.g. "I am fine") - fall back to exact-ish match.
        return t.rstrip(".!?") in r.rstrip(".!?")

    reply_words = _content_words(r)
    if not reply_words:
        return False

    # Whole-word overlap, order-independent so a correct paraphrase ("The
    # capital of France is Paris") still counts.
    reply_set = set(reply_words)
    hits = sum(1 for w in set(target_words) if w in reply_set)
    if hits / len(set(target_words)) < min_coverage:
        return False
    # Guard against a reply that buries the answer in unrelated filler.
    return hits / len(reply_set) >= 0.3


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/tiny.pt")
    p.add_argument("--device", default=None)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--save", default="checkpoints/learned.pt")
    p.add_argument("--passes", type=int, default=20)
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--fail-steps", type=int, default=10)
    p.add_argument("--rounds", type=int, default=4)
    p.add_argument("--temp", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, config = load_for_learning(args.checkpoint, device)
    learner = OnlineLearner(model, tokenizer, config, lr=args.lr, device=device,
                            steps=1)

    lessons = all_lessons()
    prompts = [CHAT_PROMPT_TMPL.format(prompt=q) for q, _ in lessons]
    answers = [a for _, a in lessons]
    n = len(lessons)
    by_topic = []
    i = 0
    for topic, topic_lessons in CURRICULUM + GENERAL_KNOWLEDGE:
        by_topic.append((topic, i, i + len(topic_lessons)))
        i += len(topic_lessons)
    order = list(range(n))

    print("\n  FULL KNOWLEDGE TEACHING: teach -> test -> re-teach failures")
    print(f"  Model: {args.checkpoint}  |  device: {device}  |  lessons: {n}  "
          f"|  passes: {args.passes}  |  test temp: {args.temp}\n")

    def run_tests():
        """Test every lesson. Results are indexed BY LESSON ID, not by
        iteration position.

        `order` is reshuffled every pass, and the old version appended results
        in visit order while dump_status, the per-topic slices and the
        re-teach selector all indexed them as lesson ids. So the status file
        paired each prompt with a random other lesson's reply, per-topic
        scores were meaningless, and - worst - the re-teach loop taught the
        wrong lessons. Preallocating by lesson id fixes all three.
        """
        statuses = [False] * n
        replies = [""] * n
        for idx in order:
            r = generate(model, tokenizer, prompts[idx], device,
                         temperature=args.temp)
            statuses[idx] = similar(r, answers[idx])
            replies[idx] = r
        return statuses, replies

    def dump_status(statuses, replies):
        """statuses/replies are indexed by lesson id (see run_tests)."""
        with open("checkpoints/curriculum_status.txt", "w", encoding="utf-8") as f:
            for idx, ok in enumerate(statuses):
                prompt, expected = lessons[idx]
                f.write(f"{'PASS' if ok else 'FAIL'} | {prompt}\n")
                f.write(f"    expected: {expected}\n")
                f.write(f"    got     : {replies[idx]}\n")

    baseline_statuses, baseline_replies = run_tests()
    best = sum(baseline_statuses)
    print(f"  Loaded checkpoint already passes {best}/{n} "
          f"(baseline, tested at temp {args.temp})\n")

    final_statuses, final_replies = baseline_statuses, baseline_replies
    for pidx in range(1, args.passes + 1):
        random.shuffle(order)
        for idx in order:
            learner.steps = args.steps
            learner.teach(prompts[idx], answers[idx])

        statuses, replies = run_tests()
        print(f"  pass {pidx:2d} | after teach-all: {sum(statuses)}/{n}")

        for rnd in range(1, args.rounds + 1):
            fails = [idx for idx in order if not statuses[idx]]
            if not fails:
                break
            random.shuffle(fails)
            learner.steps = args.fail_steps
            for idx in fails:
                learner.teach(prompts[idx], answers[idx])
            statuses, replies = run_tests()
            print(f"  pass {pidx:2d} | re-teach round {rnd}: taught {len(fails)}, "
                  f"now {sum(statuses)}/{n}")

        passed = sum(statuses)
        final_statuses, final_replies = statuses, replies
        per_topic = []
        for topic, lo, hi in by_topic:
            t = sum(statuses[lo:hi])
            per_topic.append(f"{topic}: {t}/{hi - lo}")
        print(f"  pass {pidx} complete: {passed}/{n}  "
              f"({'  '.join(safe(x) for x in per_topic)})\n")

        if passed > best:
            best = passed
            learner.save(args.save)
            dump_status(statuses, replies)
            print(f"  *** new best ({passed}/{n}) saved -> {args.save}\n")

        if passed == n:
            break

    dump_status(final_statuses, final_replies)

    print("=" * 60)
    print(f"  Curriculum     : {best}/{n} lessons passing (best)")
    print(f"  Passes run     : {pidx}")
    print(f"  Interactions   : {learner.stats['interactions']}")
    print(f"  Gradient steps : {learner.stats['total_steps']}")
    print(f"  Saved best     : {args.save}")
    print(f"  Status dump    : checkpoints/curriculum_status.txt")
    print("=" * 60)


if __name__ == "__main__":
    main()
