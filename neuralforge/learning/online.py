"""
Live online fine-tuning for NeuralForge.

This module lets a human teach the model *during* a conversation. Instead of
collecting a big dataset and launching a full offline run, every piece of
human feedback becomes a few gradient steps that nudge the model toward the
behaviour the human wants -- right away.

Three feedback primitives:

  * demonstrate(prompt, answer)  -> supervised fine-tune on an ideal (prompt -> answer)
  * approve(prompt, response)    -> reinforce a model response the human liked
  * reject(prompt, response, preferred=None)
                                   -> if a `preferred` answer is given, fine-tune
                                      toward it; otherwise the rejection is logged
                                      (we cannot unlearn without a better target)

All interactions are appended to a JSONL feedback log and can later be exported
to a plain-text corpus for a full offline training run via `train.py`.
"""

import os
import json
import time
import threading

import torch
import torch.nn.functional as F
from torch.optim import AdamW

from .lora import inject_lora, freeze_base, lora_state_dict, count_lora_params, merge_lora
from .replay import ReplayBuffer, RegressionProbe


class FeedbackStore:
    """Append-only JSONL store of human interactions."""

    def __init__(self, path: str):
        self.path = path
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)

    def append(self, record: dict):
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def all(self) -> list:
        if not os.path.exists(self.path):
            return []
        out = []
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out


class OnlineLearner:
    """Wraps a NeuralForge model and performs live, few-step fine-tuning."""

    def __init__(
        self,
        model,
        tokenizer,
        config,
        lr: float = 3e-5,
        device: str = None,
        steps: int = 6,
        feedback_path: str = None,
        eos_id: int = None,
        replay_text: str = None,
        replay_batch: int = 2,
        replay_weight: float = 1.0,
        replay_seq_len: int = 256,
        replay_sequences: int = 256,
        teach_with_dropout: bool = False,
        lora_rank: int = None,
        lora_alpha: float = 32.0,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.lr = lr
        self.steps = max(1, int(steps))
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # Dropout during teaching made results non-deterministic (identical
        # lesson sets produced different models run to run) and made the
        # reported delta dishonest: loss_before was measured in eval mode with
        # dropout off, loss_after in train mode with it on, so part of the
        # "improvement" was just noise. Teach in eval mode by default.
        self.teach_with_dropout = teach_with_dropout
        # Appending <eos> to every taught answer teaches the model to STOP
        # after a reply instead of running on and drifting into repetition loops.
        # The id is taken from the tokenizer so BPE and char vocabularies both
        # use their real <eos> (hardcoding 2 is wrong for BPE tokenizers).
        self.eos_id = eos_id if eos_id is not None else getattr(tokenizer, "eos_id", 2)

        # LoRA mode: freeze the base model and train only a small adapter, so
        # a bad lesson can be thrown away instead of having corrupted the
        # weights. Off by default - it mutates the module in place, and the
        # web UI shares one model object between chat and teaching.
        self.lora_rank = lora_rank
        if lora_rank:
            inject_lora(self.model, rank=lora_rank, alpha=lora_alpha)
            self.model.to(self.device)
            params = freeze_base(self.model)
            n_lora, n_total = count_lora_params(self.model)
            print(f"  OnlineLearner: LoRA rank {lora_rank}, training "
                  f"{n_lora/1e6:.2f}M / {n_total/1e6:.1f}M params "
                  f"({100*n_lora/n_total:.2f}%)")
        else:
            params = list(self.model.parameters())

        # Replay: without it, N steps on a single example walk the model off
        # the pretraining distribution and general behaviour collapses.
        self.replay = None
        self.replay_batch = max(1, int(replay_batch))
        self.replay_weight = float(replay_weight)
        if replay_text:
            self.replay = ReplayBuffer(replay_text, tokenizer,
                                       seq_len=replay_seq_len,
                                       max_sequences=replay_sequences)
            print(f"  OnlineLearner: replay buffer with {len(self.replay)} windows "
                  f"of {replay_seq_len} tokens")

        # Regression probe over untaught control prompts (opt-in via
        # watch_control_prompts).
        self.probe = None

        # A dedicated optimizer with a small LR so a handful of online steps
        # can shift behaviour without catastrophically forgetting.
        self.optimizer = AdamW(
            params,
            lr=lr,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=0.0,
        )

        self._lock = threading.Lock()
        self.stats = {
            "interactions": 0,
            "total_steps": 0,
            "last_loss_before": None,
            "last_loss_after": None,
            "last_replay_loss": None,
        }
        self.feedback = FeedbackStore(
            feedback_path or os.path.join("checkpoints", "feedback.jsonl")
        )

    # -- tokenization helpers -------------------------------------------------
    def _encode(self, text: str):
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _masked_targets(self, prompt_ids, seq):
        """Build target ids that ignore the prompt portion.

        We only want to supervise the *answer* tokens (so the model learns to
        produce the answer given the prompt, without re-shaping how it models
        the prompt itself, which would risk forgetting). target[i] is the token
        after input[i] == seq[i+1]; an answer token starts at i == len(prompt)-1.
        """
        a = len(prompt_ids)
        target = seq[1:]
        return [-1 if i < a - 1 else target[i] for i in range(len(target))]

    # -- loss evaluation ------------------------------------------------------
    @torch.no_grad()
    def eval_loss(self, prompt: str, answer: str) -> float:
        """Next-token cross-entropy on (prompt -> answer), prompt masked out."""
        prompt_ids = self._encode(prompt)
        ans_ids = self._encode(answer) + [self.eos_id]
        seq = prompt_ids + ans_ids
        if len(seq) < 2:
            return None
        x = torch.tensor([seq[:-1]], dtype=torch.long, device=self.device)
        y = torch.tensor(
            [self._masked_targets(prompt_ids, seq)], dtype=torch.long, device=self.device
        )
        self.model.eval()
        _, loss, _ = self.model(x, targets=y)
        return float(loss.item())

    # -- one training pass over an example -----------------------------------
    def _train_example(self, prompt: str, answer: str) -> float:
        prompt_ids = self._encode(prompt)
        ans_ids = self._encode(answer) + [self.eos_id]
        seq = prompt_ids + ans_ids
        if len(seq) < 2:
            return None
        x = torch.tensor([seq[:-1]], dtype=torch.long, device=self.device)
        y = torch.tensor(
            [self._masked_targets(prompt_ids, seq)], dtype=torch.long, device=self.device
        )

        # eval() disables dropout but leaves autograd on, so the update is
        # deterministic and loss_before/loss_after are measured alike.
        self.model.train() if self.teach_with_dropout else self.model.eval()
        trainable = [p for p in self.model.parameters() if p.requires_grad]

        last = None
        last_replay = None
        for _ in range(self.steps):
            self.optimizer.zero_grad(set_to_none=True)
            _, lesson_loss, _ = self.model(x, targets=y)
            total = lesson_loss

            # Anchor the update to the original distribution.
            if self.replay is not None and self.replay_weight > 0:
                rx, ry = self.replay.sample(self.replay_batch, self.device)
                if rx is not None:
                    _, replay_loss, _ = self.model(rx, targets=ry)
                    total = lesson_loss + self.replay_weight * replay_loss
                    last_replay = float(replay_loss.item())

            total.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            self.optimizer.step()
            last = float(lesson_loss.item())

        self.stats["total_steps"] += self.steps
        self.stats["last_replay_loss"] = last_replay
        return last

    def _batch(self, pairs):
        """Pad a list of (prompt, answer) into one (x, y) batch.

        Inputs pad with <pad>; targets pad with -1 so cross_entropy ignores
        them, exactly as TextDataset does.
        """
        seqs, masks = [], []
        for prompt, answer in pairs:
            pids = self._encode(prompt)
            aids = self._encode(answer) + [self.eos_id]
            seq = pids + aids
            if len(seq) < 2:
                continue
            seqs.append(seq)
            masks.append(self._masked_targets(pids, seq))
        if not seqs:
            return None, None
        width = max(len(s) for s in seqs) - 1
        pad = getattr(self.tokenizer, "pad_id", 0)
        xs = [s[:-1] + [pad] * (width - len(s) + 1) for s in seqs]
        ys = [m + [-1] * (width - len(m)) for m in masks]
        return (torch.tensor(xs, dtype=torch.long, device=self.device),
                torch.tensor(ys, dtype=torch.long, device=self.device))

    def teach_many(self, pairs, epochs: int = 8, batch_size: int = 4,
                   shuffle: bool = True) -> dict:
        """Teach several lessons JOINTLY rather than one after another.

        Teaching sequentially - N steps on lesson 1, then N on lesson 2 - lets
        whichever lesson ran last dominate the weights, which is why five facts
        taught in a row collapsed an untaught "hello" into "Romeo and Romeo
        and Romeo...". Every step here sees a mix of all the lessons plus a
        replay batch, so no single example can pull the model onto itself.
        """
        import random as _random
        with self._lock:
            pairs = list(pairs)
            before = [self.eval_loss(p, a) for p, a in pairs]

            self.model.train() if self.teach_with_dropout else self.model.eval()
            trainable = [p for p in self.model.parameters() if p.requires_grad]
            order = list(range(len(pairs)))
            steps = 0
            for _ in range(epochs):
                if shuffle:
                    _random.shuffle(order)
                for i in range(0, len(order), batch_size):
                    chunk = [pairs[j] for j in order[i:i + batch_size]]
                    x, y = self._batch(chunk)
                    if x is None:
                        continue
                    self.optimizer.zero_grad(set_to_none=True)
                    _, loss, _ = self.model(x, targets=y)
                    total = loss
                    if self.replay is not None and self.replay_weight > 0:
                        rx, ry = self.replay.sample(self.replay_batch, self.device)
                        if rx is not None:
                            _, rloss, _ = self.model(rx, targets=ry)
                            total = loss + self.replay_weight * rloss
                    total.backward()
                    torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                    self.optimizer.step()
                    steps += 1

            self.stats["total_steps"] += steps
            after = [self.eval_loss(p, a) for p, a in pairs]
            for (p, a), b, af in zip(pairs, before, after):
                self._record("demonstrate", p, a, None, b, af)
            mb = [v for v in before if v is not None]
            ma = [v for v in after if v is not None]
            return {
                "type": "teach_many",
                "lessons": len(pairs),
                "steps": steps,
                "loss_before": sum(mb) / len(mb) if mb else None,
                "loss_after": sum(ma) / len(ma) if ma else None,
            }

    # -- public feedback primitives ------------------------------------------
    def teach(self, prompt: str, answer: str) -> dict:
        """Supervised fine-tune toward an ideal (prompt -> answer)."""
        with self._lock:
            before = self.eval_loss(prompt, answer)
            after = self._train_example(prompt, answer)
            self._record("demonstrate", prompt, answer, None, before, after)
            return self._result(before, after)

    def approve(self, prompt: str, response: str) -> dict:
        """Reinforce a model response the human liked."""
        with self._lock:
            before = self.eval_loss(prompt, response)
            after = self._train_example(prompt, response)
            self._record("approve", prompt, response, None, before, after)
            return self._result(before, after)

    def reject(self, prompt: str, response: str, preferred: str = None) -> dict:
        """Reject a response. If `preferred` is supplied, learn toward it."""
        with self._lock:
            if preferred:
                before = self.eval_loss(prompt, preferred)
                after = self._train_example(prompt, preferred)
                self._record("correct", prompt, preferred, None, before, after)
                return self._result(before, after)
            before = self.eval_loss(prompt, response)
            self._record("reject", prompt, response, None, before, before)
            return {
                "type": "reject",
                "loss_before": before,
                "loss_after": before,
                "note": "logged only; supply a preferred answer to learn from it",
            }

    # -- regression watching --------------------------------------------------
    def watch_control_prompts(self, prompts, prompt_template="User: {prompt}\nAssistant:"):
        """Snapshot untaught prompts so drift can be measured later.

        Held-out perplexity is not a substitute: teaching five facts moved
        pretraining loss +0.7% while an untaught "hello" degenerated into
        "Romeo and Romeo and Romeo...". Watch behaviour, not prose.
        """
        self.probe = RegressionProbe(self.model, self.tokenizer,
                                     prompt_template, self.device)
        return self.probe.capture(list(prompts))

    def check_regression(self):
        """Report drift on the watched prompts. None if none are watched."""
        return self.probe.check() if self.probe is not None else None

    # -- bookkeeping ----------------------------------------------------------
    def _record(self, kind, prompt, answer, rating, before, after):
        self.stats["interactions"] += 1
        self.stats["last_loss_before"] = before
        self.stats["last_loss_after"] = after
        self.feedback.append(
            {
                "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                "type": kind,
                "prompt": prompt,
                "answer": answer,
                "loss_before": before,
                "loss_after": after,
            }
        )

    def _result(self, before, after) -> dict:
        return {
            "type": "teach",
            "loss_before": before,
            "loss_after": after,
            "delta": (before - after) if (before is not None and after is not None) else None,
        }

    def export_corpus(self, path: str = None) -> str:
        """Write all taught examples as a plain-text corpus for offline training."""
        lines = []
        for r in self.feedback.all():
            if r["type"] in ("demonstrate", "approve", "correct"):
                lines.append(f"{r['prompt']}\n{r['answer']}\n\n")
        path = path or os.path.join("data", "learned_interactions.txt")
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write("".join(lines))
        return path

    def save(self, path: str, merge: bool = True):
        """Persist the learned weights (+ config + tokenizer) so progress survives.

        In LoRA mode the adapter is folded into the base weights by default, so
        the result is an ordinary checkpoint that loads into an unmodified
        NeuralForge. Pass merge=False to keep the model wrapped and store only
        the adapter tensors (a few MB) alongside it.
        """
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        meta = {
            "name": "learned",
            "interactions": self.stats["interactions"],
            "total_steps": self.stats["total_steps"],
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "source": "neuralforge.online.OnlineLearner",
            "lora_rank": self.lora_rank,
            "replay": None if self.replay is None else len(self.replay),
        }

        artifact = {"config": self.config, "tokenizer_obj": self.tokenizer, "meta": meta}
        if self.lora_rank and not merge:
            artifact["adapter_state"] = lora_state_dict(self.model)
            artifact["model_state_dict"] = None
        else:
            if self.lora_rank:
                merged = merge_lora(self.model)
                meta["lora_merged_layers"] = merged
                self.lora_rank = None      # the wrapper is gone after merging
            artifact["model_state_dict"] = self.model.state_dict()

        torch.save(artifact, path)
        return path
