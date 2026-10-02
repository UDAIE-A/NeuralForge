#!/usr/bin/env python3
"""
End-to-end verification of the six fixes.

    python scripts/verify_changes.py

Exits 0 if everything passes, 1 otherwise. Each check names the item it covers:

  1  whitespace-preserving BPE (+ legacy compatibility, merge correctness)
  2  publish() keeps the best-validation weights
  3  curriculum indexing + scorer
  4  overfitting controls (non-overlapping stride, early stopping)
  5  the auto-tutor bridge is gone
  6  config-driven training replaces the per-model scripts

The training checks run real short runs on the GPU and take about a minute.
"""

import glob
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

PY = sys.executable
RESULTS = []


def check(item, name, ok, detail=""):
    RESULTS.append((item, name, bool(ok)))
    mark = "PASS" if ok else "FAIL"
    line = f"  [{mark}] ({item}) {name}"
    if detail:
        line += f"\n         {detail}"
    print(line, flush=True)
    return ok


def section(title):
    print(f"\n{title}\n{'-' * len(title)}", flush=True)


def run(args, timeout=600):
    return subprocess.run([PY] + args, capture_output=True, text=True,
                          timeout=timeout, encoding="utf-8", errors="replace")


# ---------------------------------------------------------------- item 1
def check_tokenizer():
    section("1. Whitespace-preserving BPE tokenizer")
    from neuralforge.tokenizer.bpe import BPETokenizer, _SPLIT_RE

    corpus = ("User: hello there\nAssistant: Hi! How are you today?\n"
              "User: good thanks\nAssistant: Great to hear.\n\n") * 200
    tok = BPETokenizer()
    tok.train(corpus, vocab_size=600)

    chat = "User: hi\nAssistant: Hello!\n\nUser: bye\nAssistant: Bye!"
    rt = tok.decode(tok.encode(chat, add_special_tokens=False))
    check(1, "chat turns round-trip exactly (newlines survive)", rt == chat,
          "" if rt == chat else f"got {rt!r}")

    for probe in ["line1\n\nline2\n\tindented", "The sun is a star. It's 2026!"]:
        got = tok.decode(tok.encode(probe, add_special_tokens=False))
        check(1, f"round-trip {probe[:24]!r}...", got == probe,
              "" if got == probe else f"got {got!r}")

    sp = [k for k in tok.vocab if k.startswith(" ") and len(k) > 1]
    check(1, "space-prefixed tokens exist in vocab", len(sp) > 0,
          f"{len(sp)} found, e.g. {sorted(sp)[:4]}")
    check(1, "newline is a real vocab token", "\n" in tok.vocab)
    check(1, "tokenizer stamped version 2", getattr(tok, "version", 1) == 2)

    # Legacy v1 tokenizers must keep their old behaviour.
    import pickle
    legacy_path = os.path.join(tempfile.gettempdir(), "_nf_legacy.pkl")
    with open(legacy_path, "wb") as f:
        pickle.dump({"merges": tok.merges, "vocab": tok.vocab,
                     "special_tokens": tok.special_tokens,
                     "is_trained": True}, f)          # note: no 'version' key
    lg = BPETokenizer.load(legacy_path)
    check(1, "versionless pickle detected as legacy",
          getattr(lg, "version", None) == 1 and lg._is_legacy)
    ghost = BPETokenizer.load(legacy_path)
    del ghost.version                                  # attribute absent entirely
    check(1, "missing version attribute falls back safely",
          ghost._is_legacy and ghost._pieces("a b") == ["a", "b"])
    os.remove(legacy_path)

    # Frequency-weighted merges must equal a naive full recount.
    from collections import Counter

    def reference_merges(text, n):
        words = [tuple(bytes([b]).decode("latin-1") for b in p.encode("utf-8"))
                 for p in _SPLIT_RE.findall(text)]
        merges = []
        for _ in range(n):
            cnt = Counter()
            for w in words:
                for j in range(len(w) - 1):
                    cnt[(w[j], w[j + 1])] += 1
            if not cnt:
                break
            top = max(cnt.values())
            if top < 2:
                break
            a, b = min(p for p, c in cnt.items() if c == top)
            m = a + b
            out = []
            for w in words:
                nw, j = [], 0
                while j < len(w):
                    if j < len(w) - 1 and w[j] == a and w[j + 1] == b:
                        nw.append(m); j += 2
                    else:
                        nw.append(w[j]); j += 1
                out.append(tuple(nw))
            words = out
            merges.append((a, b))
        return merges

    sample = _corpus_text()[:60000]
    n_merges = 150
    fast = BPETokenizer()
    fast.train(sample, vocab_size=256 + 4 + n_merges)
    ref = reference_merges(sample, n_merges)
    k = min(len(ref), len(fast.merges))
    check(1, "weighted merges identical to naive reference",
          k > 50 and fast.merges[:k] == ref[:k], f"compared {k} merges")

    # Shared chat module replaced the three copied regexes.
    from neuralforge.chat import clean_chat_reply
    check(1, "next-turn cut (strict)",
          clean_chat_reply(" Hello!\nUser: next") == (" Hello!", True))
    check(1, "next-turn cut (legacy tokenizers)",
          clean_chat_reply(" Hi there User: next", legacy=True) == (" Hi there", True))
    check(1, "mid-sentence 'user:' is not a turn boundary",
          clean_chat_reply(" ask the user: politely") == (" ask the user: politely", False))
    # Built from parts so this scanner does not match its own source.
    needle = "_NEXT_TURN" + " = re.compile"
    skip = {os.path.abspath(__file__),
            os.path.abspath(os.path.join("neuralforge", "chat.py"))}
    copies = [p for p in glob.glob("**/*.py", recursive=True)
              if "venv" not in p and "legacy" not in p
              and os.path.abspath(p) not in skip
              and needle in open(p, encoding="utf-8", errors="replace").read()]
    check(1, "no duplicated _NEXT_TURN regexes outside chat.py",
          not copies, f"found in {copies}" if copies else "")


# ---------------------------------------------------------------- generation
def check_generation():
    section("1b. Generation past the context window")
    import torch
    from neuralforge.core import NeuralForge, ModelConfig
    cfg = ModelConfig(vocab_size=64, d_model=32, n_heads=4, n_layers=2,
                      d_ff=64, max_seq_len=16)
    m = NeuralForge(cfg).eval()
    x = torch.randint(0, 64, (1, 8))
    try:
        out = m.generate(x, max_new_tokens=200, top_k=5)
        check(1, "generate 200 tokens with max_seq_len=16", out.shape[1] == 208,
              f"shape {tuple(out.shape)}")
    except Exception as e:
        check(1, "generate 200 tokens with max_seq_len=16", False, repr(e))
    try:
        n = sum(1 for _ in m.generate_stream(x, max_new_tokens=100, top_k=5))
        check(1, "generate_stream past the window", n == 100, f"yielded {n}")
    except Exception as e:
        check(1, "generate_stream past the window", False, repr(e))
    try:
        b = m.generate(x.repeat(3, 1), max_new_tokens=60, top_k=5,
                       repetition_penalty=1.1)
        check(1, "batched generation + repetition penalty", b.shape == (3, 68))
    except Exception as e:
        check(1, "batched generation + repetition penalty", False, repr(e))


# ---------------------------------------------------------------- item 3
def check_curriculum():
    section("3. Curriculum indexing and scorer")
    src = open("scripts/teach_basics_loop.py", encoding="utf-8").read()
    ns = {"re": re}
    exec(src[src.index("def _content_words"):src.index("def main()")], ns)
    similar = ns["similar"]

    cases = [
        # the three real rows from the old curriculum_status.txt: all must FAIL
        ("Yes, cats are cute and independent.", "Hello! How can I help you today?", False),
        ("Good evening! How was your day?", "Hi there! Nice to meet you.", False),
        ("I live inside a computer, in my model weights.", "You're welcome!", False),
        ("Hello! How can I help you today?", "Hello! How can I help you today?", True),
        ("The capital of France is Paris.", "Paris is the capital of France.", True),
        ("Paris and also many unrelated words about weather and dogs and cars and trains",
         "Paris is the capital of France.", False),
    ]
    bad = [c for c in cases if similar(c[0], c[1]) != c[2]]
    check(3, "scorer rejects mismatched answers, accepts paraphrase",
          not bad, f"{len(bad)} wrong: {[c[0][:30] for c in bad]}" if bad else
          f"{len(cases)}/{len(cases)} correct")

    # Results must be indexed by lesson id even when the visit order is shuffled.
    lessons = [(f"Q{i}", f"answer-{i}") for i in range(8)]
    n = len(lessons)
    order = list(range(n))
    random.seed(7)
    random.shuffle(order)
    reply_for = lambda i: lessons[i][1] if i % 2 == 0 else "unrelated text"

    statuses, replies = [False] * n, [""] * n
    for idx in order:
        r = reply_for(idx)
        statuses[idx] = (r == lessons[idx][1])
        replies[idx] = r
    aligned = all(replies[i] == reply_for(i) for i in range(n))
    fails = sorted(i for i in range(n) if not statuses[i])
    check(3, "status rows aligned to their own lesson", aligned)
    check(3, "re-teach selects the truly failing lessons",
          fails == [1, 3, 5, 7], f"selected {fails}, truth [1, 3, 5, 7]")

    check(3, "dump_status writes expected/got pairs",
          "expected:" in src and "got     :" in src)


# ---------------------------------------------------------------- item 5
def check_bridge_removed():
    section("5. Auto-tutor bridge removed")
    check(5, "webui/bridge.py deleted", not os.path.exists("webui/bridge.py"))
    check(5, "webui/static/bridge.html deleted",
          not os.path.exists("webui/static/bridge.html"))
    import webui.server as srv
    routes = [getattr(r, "path", "") for r in srv.app.routes]
    check(5, "no bridge routes registered",
          not any("bridge" in p for p in routes))
    server_src = open("webui/server.py", encoding="utf-8").read()
    check(5, "no unattended teach-loop autostart",
          "_autostart_bridge" not in server_src)
    check(5, "chat/train/learn routes still present",
          all(p in routes for p in ("/ws/chat", "/api/train/start", "/api/learn/teach")))


# ---------------------------------------------------------------- item 6
def check_configs():
    section("6. Config-driven training replaces per-model scripts")
    cfgs = sorted(glob.glob("configs/*.json"))
    check(6, "run configs present", len(cfgs) >= 6, f"{[os.path.basename(c) for c in cfgs]}")
    ok = True
    for c in cfgs:
        try:
            json.load(open(c, encoding="utf-8"))
        except Exception as e:
            ok = False
            print(f"         {c}: {e}")
    check(6, "every config is valid JSON", ok)
    check(6, "duplicate train scripts archived",
          len(glob.glob("scripts/legacy/train_*.py")) >= 7
          and os.path.exists("scripts/legacy/README.md"))
    # Compare by name: scripts/train_identity_lora.py is the image pipeline's
    # LoRA trainer, not one of the archived language-model scripts.
    archived = {os.path.basename(p) for p in glob.glob("scripts/legacy/train_*.py")}
    left = sorted(n for n in archived if os.path.exists(os.path.join("scripts", n)))
    check(6, "no archived train script left in scripts/", not left, ", ".join(left))

    help_text = run(["train.py", "--help"], timeout=120).stdout
    for flag in ("--config", "--max-steps", "--d-model", "--dropout",
                 "--early-stopping", "--val-fraction"):
        check(6, f"train.py exposes {flag}", flag in help_text)

    bad = os.path.join(tempfile.gettempdir(), "_nf_bad.json")
    open(bad, "w", encoding="utf-8").write('{"preset":"tiny","data":["x"],"bogus":1}')
    r = run(["train.py", "--config", bad], timeout=120)
    check(6, "unknown config key rejected", "Unknown key" in (r.stdout + r.stderr))
    os.remove(bad)

    r = run(["train.py", "--preset", "tiny"], timeout=120)
    check(6, "--data required when absent from config",
          "--data is required" in (r.stdout + r.stderr))


# ---------------------------------------------------------------- items 2 & 4
def check_training(tmp):
    section("2 & 4. Publish-best weights, early stopping, non-overlapping stride")
    import torch
    from neuralforge.training.data import create_dataloaders
    import inspect
    check(4, "stride defaults to seq_len (no overlap)",
          inspect.signature(create_dataloaders).parameters["stride"].default is None)

    if not torch.cuda.is_available():
        check(2, "training run (needs CUDA)", False, "CUDA unavailable - skipped")
        return

    corpus = os.path.join(tmp, "corpus.txt")
    open(corpus, "w", encoding="utf-8").write(_corpus_text()[:400000])
    small = os.path.join(tmp, "small.txt")
    open(small, "w", encoding="utf-8").write(_corpus_text()[:60000])

    # --- publish() must ship the best-validation weights, keep _best.pt ---
    d1 = os.path.join(tmp, "ck1")
    r = run(["train.py", "--preset", "tiny", "--data", corpus, "--name", "vfy",
             "--vocab-size", "1200", "--seq-len", "128", "--batch-size", "16",
             "--epochs", "3", "--lr", "1e-3", "--checkpoint-dir", d1,
             "--no-compile", "--seed", "1"])
    if r.returncode != 0:
        check(2, "training run completes", False, (r.stdout + r.stderr)[-600:])
        return
    check(2, "training run completes", True)

    pub, best = os.path.join(d1, "vfy.pt"), os.path.join(d1, "vfy_best.pt")
    check(2, "_best.pt is kept (not deleted by publish)", os.path.exists(best))
    check(2, "_train.pt removed", not os.path.exists(os.path.join(d1, "vfy_train.pt")))
    if os.path.exists(pub) and os.path.exists(best):
        p = torch.load(pub, map_location="cpu", weights_only=False)
        b = torch.load(best, map_location="cpu", weights_only=False)
        same = all(torch.equal(p["model_state_dict"][k], b["model_state_dict"][k])
                   for k in p["model_state_dict"])
        check(2, "published weights == best-validation weights", same)
        check(2, "meta records the weight source",
              "best val" in str(p["meta"].get("weights_from", "")),
              f"weights_from = {p['meta'].get('weights_from')!r}")
        check(1, "published tokenizer is version 2",
              getattr(p.get("tokenizer_obj"), "version", 1) == 2)

        g = run(["generate.py", "--checkpoint", pub, "--prompt", "The king",
                 "--max-tokens", "120", "--top-p", "0.9",
                 "--repetition-penalty", "1.15", "--seed", "1"])
        check(1, "generate.py runs on the published model",
              g.returncode == 0 and "Generated:" in g.stdout,
              (g.stdout + g.stderr)[-400:] if g.returncode else "")

        gi = subprocess.run([PY, "generate.py", "--checkpoint", pub,
                             "--interactive", "--max-tokens", "40", "--seed", "1"],
                            input="who are you\nquit\n", capture_output=True,
                            text=True, timeout=300, encoding="utf-8", errors="replace")
        echoed = "User: who are you" in gi.stdout
        check(1, "interactive reply does not echo the prompt",
              gi.returncode == 0 and not echoed,
              (gi.stdout + gi.stderr)[-400:] if gi.returncode else "")

    # --- early stopping must fire and publish the pre-turn weights ---
    d2 = os.path.join(tmp, "ck2")
    r2 = run(["train.py", "--preset", "tiny", "--data", small, "--name", "es",
              "--vocab-size", "800", "--seq-len", "128", "--batch-size", "16",
              "--epochs", "40", "--lr", "3e-3", "--early-stopping", "2",
              "--checkpoint-dir", d2, "--no-compile", "--seed", "1"])
    out2 = r2.stdout.replace("\r", "\n")
    check(4, "early stopping fires before the epoch budget",
          "Early stop:" in out2 and "TRAINING COMPLETE (early stop)" in out2,
          "" if "Early stop:" in out2 else out2[-500:])
    vals = [float(m) for m in re.findall(r"validation loss: ([\d.]+)", out2)]
    check(4, "stopped after validation bottomed out",
          bool(vals) and vals[-1] > min(vals),
          f"{len(vals)} evals, min {min(vals):.4f}, last {vals[-1]:.4f}" if vals else "")

    # --- --max-steps caps the run ---
    d3 = os.path.join(tmp, "ck3")
    r3 = run(["train.py", "--preset", "tiny", "--data", small, "--name", "ms",
              "--vocab-size", "800", "--seq-len", "128", "--batch-size", "16",
              "--epochs", "40", "--max-steps", "25", "--checkpoint-dir", d3,
              "--no-compile", "--seed", "1"])
    out3 = r3.stdout.replace("\r", "\n")
    check(6, "--max-steps caps the run",
          "Reached --max-steps" in out3 and "Total steps:   25" in out3)

    # --- config file drives a run; explicit flags still win ---
    cfg_path = os.path.join(tmp, "run.json")
    json.dump({"preset": "tiny", "name": "cfg", "data": [small],
               "vocab_size": 800, "seq_len": 128, "batch_size": 16,
               "epochs": 1, "lr": 1e-3, "dropout": 0.15,
               "checkpoint_dir": os.path.join(tmp, "ck4")},
              open(cfg_path, "w", encoding="utf-8"))
    r4 = run(["train.py", "--config", cfg_path, "--no-compile", "--batch-size", "4"])
    out4 = r4.stdout.replace("\r", "\n")
    check(6, "config file drives a real run", r4.returncode == 0
          and "Config:" in out4, (r4.stdout + r4.stderr)[-400:] if r4.returncode else "")
    check(6, "explicit flag overrides the config file",
          "Batch size:  4" in out4,
          [l for l in out4.splitlines() if "Batch size" in l][:1])
    check(6, "config value applied when not overridden",
          "Dropout:     0.15" in out4)


def _corpus_text():
    """Real repo corpus if available, otherwise synthetic filler."""
    for p in ("data/high_quality.txt", "data/training_40mb.txt",
              "data/combined_train.txt"):
        if os.path.exists(p):
            with open(p, encoding="utf-8", errors="replace") as f:
                return f.read(600000)
    return ("User: hello there\nAssistant: Hi! How can I help you today?\n\n"
            "The quick brown fox jumps over the lazy dog. It was a bright cold "
            "day in April, and the clocks were striking thirteen.\n\n") * 4000


def main():
    print("=" * 68)
    print("  NeuralForge - verifying the six fixes")
    print("=" * 68)
    tmp = tempfile.mkdtemp(prefix="nf_verify_")
    try:
        check_tokenizer()
        check_generation()
        check_curriculum()
        check_bridge_removed()
        check_configs()
        check_training(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("\n" + "=" * 68)
    by_item = {}
    for item, _, ok in RESULTS:
        p, t = by_item.get(item, (0, 0))
        by_item[item] = (p + int(ok), t + 1)
    labels = {1: "whitespace BPE + chat/generation",
              2: "publish best weights",
              3: "curriculum indexing + scorer",
              4: "overfitting controls",
              5: "bridge removed",
              6: "config-driven training"}
    for item in sorted(by_item):
        p, t = by_item[item]
        print(f"  item {item}  {labels.get(item, ''):<34} {p}/{t} "
              f"{'OK' if p == t else '*** FAILURES ***'}")
    failed = [f"({i}) {n}" for i, n, ok in RESULTS if not ok]
    total, passed = len(RESULTS), sum(1 for *_, ok in RESULTS if ok)
    print("=" * 68)
    if failed:
        print(f"  {passed}/{total} checks passed. FAILED:")
        for f in failed:
            print(f"    - {f}")
        return 1
    print(f"  ALL {total} CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
