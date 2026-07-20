#!/usr/bin/env python3
"""Clean training corpora: strip Project Gutenberg boilerplate/license noise
and normalize whitespace, producing a single high-quality corpus.

Run:
    python scripts/clean_data.py
"""
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")

INPUTS = [
    os.path.join(DATA, "books", "train_large.txt"),
    os.path.join(DATA, "conversational_train_large.txt"),
]

OUT = os.path.join(DATA, "clean_corpus.txt")

# ONLY strict Gutenberg boilerplate phrases (whole-line removal).
# NOTE: ordinary novel words like "visit", "public", "license", "copyright"
# are intentionally NOT here -- they appear constantly in real prose.
BOILERPLATE = [
    "project gutenberg",
    "gutenberg",
    "ebook",
    "e-book",
    "www.gutenberg",
    "archive.org",
    "this ebook is for the use of anyone",
    "this etext is for the use of anyone",
    "produced by",
    "produced at",
    "transcribed by",
    "transcriber",
    "proofread",
    "distributed by",
    "redistribution",
    "most recent update",
    "updated editions",
    "search the catalog",
    "literary archive foundation",
    "***",
]


# Matches a whole Gutenberg book: from its "*** START OF THE PROJECT GUTENBERG ***"
# marker through its "*** END OF THE PROJECT GUTENBERG ***" marker, capturing the
# interior (the actual book text). Non-greedy so each book is matched separately
# even when hundreds are concatenated in one file.
_BOOK_RE = re.compile(
    r"\*\*\*\s*start of .*?project gutenberg.*?\*\*\*"   # header marker (incl. title block start)
    r"(.*?)"                                              # <-- book body (captured)
    r"\*\*\*\s*end of .*?project gutenberg.*?\*\*\*",     # footer marker
    re.IGNORECASE | re.DOTALL,
)


def strip_gutenberg(text: str) -> str:
    # Extract the interior of EVERY Gutenberg book in the file and join them.
    bodies = _BOOK_RE.findall(text)
    if bodies:
        return "\n\n".join(b.strip() for b in bodies if b.strip())
    return text


def clean_text(text: str) -> str:
    text = strip_gutenberg(text)
    out_lines = []
    for line in text.splitlines():
        low = line.lower()
        if any(b in low for b in BOILERPLATE):
            continue
        stripped = line.strip()
        if not stripped:
            out_lines.append("")
            continue
        # Drop URLs
        if "http://" in low or "https://" in low or "www." in low:
            continue
        # Drop lines that are just a few symbols (rules, asterisks, brackets)
        if len(stripped) <= 3 and not stripped[0].isalnum():
            continue
        out_lines.append(stripped)
    body = "\n".join(out_lines)
    body = re.sub(r"\n{3,}", "\n\n", body)
    body = re.sub(r"[ \t]+", " ", body)
    return body.strip() + "\n"


def main():
    parts = []
    for path in INPUTS:
        if not os.path.exists(path):
            print(f"  skip (missing): {os.path.basename(path)}")
            continue
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            raw = f.read()
        cleaned = clean_text(raw)
        parts.append(cleaned)
        print(f"  cleaned {path}: {len(raw):,} -> {len(cleaned):,} chars")

    corpus = "\n\n".join(parts)
    corpus = re.sub(r"\n{3,}", "\n\n", corpus)
    os.makedirs(DATA, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(corpus)

    print(f"\nWrote {OUT}")
    print(f"  total chars : {len(corpus):,}")
    print(f"  total words : {len(corpus.split()):,}")


if __name__ == "__main__":
    main()
