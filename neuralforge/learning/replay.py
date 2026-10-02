"""
Replay buffer and regression probe for live teaching.

Both exist because of the same measured failure. Teaching five facts with six
full-parameter gradient steps each collapsed a 186M model's general chat
behaviour - an untaught "hello" went from

    Hello! It's nice to talk to you. How can I help?

to

    Romeo and Romeo and Romeo and Romeo and ...

while held-out loss on pretraining text moved only +0.7% (1.9157 -> 1.9296).
So: replay keeps the original distribution in the gradient, and the probe
measures the kind of drift that perplexity on prose cannot see.
"""

import random

import torch


class ReplayBuffer:
    """Fixed-length windows of the original training corpus, sampled per step.

    Mixing a replay batch into every teaching step is what keeps the update
    from walking away from the pretraining distribution. The buffer is bounded
    (a slice of the corpus, not all of it) so building it stays cheap.

    REPLAY ONLY PRESERVES WHAT IT CONTAINS. This is the whole game, and it is
    easy to get wrong. Teaching five facts to a 186M model while replaying
    general prose (training_40mb.txt) left control-prompt drift at +0.65 NLL
    and turned an untaught "hello" into "Romeo and Romeo and Romeo...".
    Replaying CONVERSATIONAL data (combined_conv.txt) instead, with everything
    else identical, cut drift to +0.04 and kept "hello" answering like a
    greeting - while the taught facts still landed at 94%.

    So: replay the distribution whose behaviour you care about keeping, not
    whichever corpus happens to be largest.
    """

    def __init__(self, text: str, tokenizer, seq_len: int = 256,
                 max_sequences: int = 256, seed: int = 0):
        # Only tokenize as much text as the requested window count needs.
        approx_chars = int(seq_len * max_sequences * 6)
        ids = tokenizer.encode(text[:approx_chars], add_special_tokens=False)
        self.seq_len = seq_len
        self.windows = []
        for start in range(0, len(ids) - seq_len - 1, seq_len):
            self.windows.append(ids[start:start + seq_len + 1])
            if len(self.windows) >= max_sequences:
                break
        self._rng = random.Random(seed)

    def __len__(self):
        return len(self.windows)

    def sample(self, batch_size: int, device):
        """Return (x, y) next-token pairs, or (None, None) if empty."""
        if not self.windows:
            return None, None
        picks = [self._rng.choice(self.windows)
                 for _ in range(min(batch_size, len(self.windows)))]
        x = torch.tensor([p[:-1] for p in picks], dtype=torch.long, device=device)
        y = torch.tensor([p[1:] for p in picks], dtype=torch.long, device=device)
        return x, y


class RegressionProbe:
    """Watches untaught prompts for collateral damage.

    Records each control prompt's baseline reply, then scores how likely the
    model still finds that exact reply. A rising negative log-likelihood means
    the model has drifted off behaviour it used to have - which is precisely
    what a held-out perplexity number misses, because a handful of chat turns
    barely register against a corpus of prose.
    """

    def __init__(self, model, tokenizer, prompt_template, device,
                 max_new_tokens: int = 40):
        self.model = model
        self.tokenizer = tokenizer
        self.template = prompt_template
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.baseline = {}      # prompt -> (reply_text, reply_ids, nll)

    @torch.no_grad()
    def _reply_nll(self, prompt_ids, reply_ids) -> float:
        """Negative log-likelihood of `reply_ids` following `prompt_ids`."""
        seq = list(prompt_ids) + list(reply_ids)
        if len(seq) < 2 or not reply_ids:
            return float("nan")
        was_training = self.model.training
        self.model.eval()
        x = torch.tensor([seq[:-1]], dtype=torch.long, device=self.device)
        targets = seq[1:]
        a = len(prompt_ids)
        y = torch.tensor([[-1 if i < a - 1 else targets[i] for i in range(len(targets))]],
                         dtype=torch.long, device=self.device)
        _, loss, _ = self.model(x, targets=y)
        if was_training:
            self.model.train()
        return float(loss.item())

    @torch.no_grad()
    def capture(self, prompts, seed: int = 0):
        """Snapshot the model's current answer to each control prompt."""
        was_training = self.model.training
        self.model.eval()
        for q in prompts:
            torch.manual_seed(seed)
            p = self.template.format(prompt=q)
            pids = self.tokenizer.encode(p, add_special_tokens=False)
            x = torch.tensor([pids], dtype=torch.long, device=self.device)
            out = self.model.generate(
                x, max_new_tokens=self.max_new_tokens, temperature=0.0,
                eos_id=getattr(self.tokenizer, "eos_id", None))
            rids = out[0].tolist()[len(pids):]
            text = self.tokenizer.decode(rids)
            self.baseline[q] = (text, rids, self._reply_nll(pids, rids))
        if was_training:
            self.model.train()
        return self.baseline

    @torch.no_grad()
    def check(self, seed: int = 0):
        """Re-score (and re-generate) the control prompts. Returns a report."""
        rows = []
        was_training = self.model.training
        self.model.eval()
        for q, (base_text, base_ids, base_nll) in self.baseline.items():
            p = self.template.format(prompt=q)
            pids = self.tokenizer.encode(p, add_special_tokens=False)
            now_nll = self._reply_nll(pids, base_ids)
            torch.manual_seed(seed)
            x = torch.tensor([pids], dtype=torch.long, device=self.device)
            out = self.model.generate(
                x, max_new_tokens=self.max_new_tokens, temperature=0.0,
                eos_id=getattr(self.tokenizer, "eos_id", None))
            now_text = self.tokenizer.decode(out[0].tolist()[len(pids):])
            rows.append({
                "prompt": q,
                "baseline_reply": base_text,
                "current_reply": now_text,
                "baseline_nll": base_nll,
                "current_nll": now_nll,
                "nll_delta": now_nll - base_nll,
                "reply_changed": now_text.strip() != base_text.strip(),
            })
        if was_training:
            self.model.train()
        deltas = [r["nll_delta"] for r in rows if r["nll_delta"] == r["nll_delta"]]
        return {
            "rows": rows,
            "mean_nll_delta": sum(deltas) / len(deltas) if deltas else float("nan"),
            "changed": sum(1 for r in rows if r["reply_changed"]),
            "total": len(rows),
        }
