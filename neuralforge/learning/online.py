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
        eos_id: int = 2,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.lr = lr
        self.steps = max(1, int(steps))
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # Appending <eos> to every taught answer teaches the model to STOP
        # after a reply instead of running on and drifting into repetition loops.
        self.eos_id = eos_id

        # A dedicated optimizer with a small LR so a handful of online steps
        # can shift behaviour without catastrophically forgetting.
        self.optimizer = AdamW(
            self.model.parameters(),
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

        self.model.train()
        last = None
        for _ in range(self.steps):
            self.optimizer.zero_grad(set_to_none=True)
            _, loss, _ = self.model(x, targets=y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            last = float(loss.item())
        self.stats["total_steps"] += self.steps
        return last

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

    def save(self, path: str):
        """Persist the learned weights (+ config + tokenizer) so progress survives."""
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        artifact = {
            "model_state_dict": self.model.state_dict(),
            "config": self.config,
            "tokenizer_obj": self.tokenizer,
            "meta": {
                "name": "learned",
                "interactions": self.stats["interactions"],
                "total_steps": self.stats["total_steps"],
                "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                "source": "neuralforge.online.OnlineLearner",
            },
        }
        torch.save(artifact, path)
        return path
