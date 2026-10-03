#!/usr/bin/env python3
"""
Convert the Cornell Movie Dialogs corpus into NeuralForge "User:/Assistant:"
conversation pairs and append them to the balanced conversational corpus.

The Cornell corpus is a movie-script conversation dump. Every turn becomes a
chat example: alternate speakers become User/Assistant. Lines are truncated to
keep them short (movie lines can be huge), and only clean dialogue survives.

Usage:
    python data/convert_movie_dialogs.py              # auto-download if missing
    python data/convert_movie_dialogs.py --zip path/movie_dialogs.zip
"""

import argparse
import os
import random
import re
import sys
import urllib.request
import zipfile

CORNELL_URL = "https://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "conversational_train_large.txt")
MAX_LINE_CHARS = 200          # keep replies short, like real chat
MAX_CONVO_CHARS = 2000        # skip absurdly long scene dumps
MAX_TURNS = 6                 # keep only the first N turns of a scene
MIN_TURNS = 2
TARGET_PAIRS = 5000


def download_corpus(dest):
    print(f"Downloading Cornell Movie Dialogs -> {dest}")
    urllib.request.urlretrieve(CORNELL_URL, dest)
    print("  downloaded")


def parse_lines(lines_path):
    """movie_lines.txt -> {line_id: raw_text}."""
    result = {}
    with open(lines_path, encoding="utf-8", errors="replace") as f:
        for row in f:
            parts = row.rstrip("\n").split(" +++$+++ ")
            if len(parts) == 5:
                result[parts[0]] = parts[4].strip()
    return result


def parse_conversations(conv_path):
    """movie_conversations.txt -> list of line-id lists."""
    result = []
    with open(conv_path, encoding="utf-8", errors="replace") as f:
        for row in f:
            parts = row.rstrip("\n").split(" +++$+++ ")
            if len(parts) == 4:
                ids = [x.strip().strip("'\"") for x in parts[3][1:-1].split(",")]
                if len(ids) >= MIN_TURNS:
                    result.append(ids)
    return result


def clean_line(text):
    t = text.strip()
    t = re.sub(r"\s+", " ", t)
    t = t[:MAX_LINE_CHARS]
    return t


def build_pairs():
    if not os.path.exists("movie_lines.txt"):
        zip_path = os.path.join(HERE, "cornell_movie_dialogs_corpus.zip")
        if not os.path.exists(zip_path):
            download_corpus(zip_path)
        print(f"Unzipping {zip_path}")
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(HERE)
        os.remove(zip_path)

    extracted = os.path.join(HERE, "cornell movie-dialogs corpus")
    lines_path = os.path.join(extracted, "movie_lines.txt")
    conv_path = os.path.join(extracted, "movie_conversations.txt")
    lines = parse_lines(lines_path)
    conversations = parse_conversations(conv_path)
    print(f"  {len(lines):,} lines, {len(conversations):,} conversations")

    random.seed(7)
    random.shuffle(conversations)

    pairs = []
    for ids in conversations:
        turns = []
        for lid in ids[:MAX_TURNS]:
            t = clean_line(lines.get(lid, ""))
            if t:
                turns.append(t)
        if len(turns) < MIN_TURNS:
            continue
        if sum(len(t) for t in turns) > MAX_CONVO_CHARS:
            continue
        # Alternate speakers -> User/Assistant. Keep the full exchange so the
        # model learns multi-turn rhythm.
        block = []
        for i, t in enumerate(turns):
            role = "User" if i % 2 == 0 else "Assistant"
            block.append(f"{role}: {t}")
        pairs.append("\n".join(block))
        if len(pairs) >= TARGET_PAIRS:
            break
    return pairs


def merge(pairs):
    corpus_path = OUT
    corpus = open(corpus_path, encoding="utf-8").read().rstrip("\n")
    extra = "\n\n".join(pairs)
    merged = corpus + "\n\n" + extra + "\n"
    with open(corpus_path, "w", encoding="utf-8") as f:
        f.write(merged)
    print(f"Merged {len(pairs):,} movie-dialog exchanges into {corpus_path}")
    print(f"Corpus now: {len(merged):,} chars, {len(merged.splitlines()):,} lines")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default=None, help="Path to an existing Cornell zip")
    ap.add_argument("--pairs", type=int, default=TARGET_PAIRS)
    args = ap.parse_args()
    TARGET_PAIRS = args.pairs
    pairs = build_pairs()
    merge(pairs)
