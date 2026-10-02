"""
Shared chat-turn formatting and reply cleanup.

The conversational corpus formats turns as "User: <q>\\nAssistant: <a>", so a
model will happily keep writing the NEXT turn ("User: ...") after finishing its
reply. This module owns the prompt template and the cut-at-next-turn logic.

It used to be copy-pasted into generate.py, learn.py and webui/server.py, where
each copy also had to tolerate the version-1 tokenizer silently collapsing every
newline into a space (so the next-turn marker arrived as " User:" rather than
"\\nUser:"). Version-2 tokenizers preserve newlines, so the strict marker works;
the loose one is kept for checkpoints carrying a legacy tokenizer.
"""

import re

CHAT_PROMPT_TMPL = "User: {prompt}\nAssistant:"

# Strict: a role marker that genuinely starts a new line (v2 tokenizers).
_NEXT_TURN_STRICT = re.compile(r"\n\s*(?:user|assistant)\s*:", re.IGNORECASE)

# Loose: v1 tokenizers collapsed newlines to spaces, so the marker shows up as
# " User:" - and a char vocab missing 'U' leaves the "ser:" fragment behind.
_NEXT_TURN_LOOSE = re.compile(
    r"(?:\s|^)u?ser\s*:|(?:\n|^)\s*assistant\s*:", re.IGNORECASE
)

_LEADING_ROLE = re.compile(r"^\s*(?:assistant|u?ser)\s*:\s*", re.IGNORECASE)


def uses_legacy_tokenizer(tokenizer) -> bool:
    """True when the tokenizer cannot represent newlines (version 1 BPE)."""
    return bool(getattr(tokenizer, "_is_legacy", False))


def clean_chat_reply(text: str, legacy: bool = False) -> tuple:
    """Return (cleaned_text, next_turn_found).

    Strips a leading role echo ("Assistant:") and cuts the reply at the first
    marker that starts a new turn. Pass legacy=True for a checkpoint whose
    tokenizer collapses newlines, which needs the looser marker.
    """
    t = _LEADING_ROLE.sub("", text, count=1)
    pattern = _NEXT_TURN_LOOSE if legacy else _NEXT_TURN_STRICT
    m = pattern.search(t)
    if m:
        return t[:m.start()].rstrip(), True
    return t.rstrip(), False


def decode_reply(tokenizer, prompt_ids, out_ids, legacy=None) -> str:
    """Decode only the generated continuation, then clean it.

    Slicing by TOKEN COUNT rather than by decoded prefix length is what makes
    this reliable: decode(prompt_ids) is not guaranteed to reproduce the prompt
    string byte-for-byte, so the old `full[len(prompt):]` string arithmetic
    silently fell through to returning the echoed prompt as part of the reply.
    """
    if legacy is None:
        legacy = uses_legacy_tokenizer(tokenizer)
    gen_ids = list(out_ids)[len(prompt_ids):]
    reply, _ = clean_chat_reply(tokenizer.decode(gen_ids), legacy=legacy)
    return reply
