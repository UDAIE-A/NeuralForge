"""
Crash-safe checkpoint writing.

torch.save straight onto the final path leaves a truncated file behind if the
process dies mid-write (Ctrl-C, OOM kill, power loss) - which is how
checkpoints/test_base_train.pt ended up as 1.2 GB that no longer loads. Write
to a temp file beside the target and swap it in with os.replace, which is
atomic on the same filesystem: the path holds either the old complete file or
the new complete file, never half of one.
"""

import os

import torch


def atomic_save(obj, path: str):
    """torch.save(obj, path) without ever leaving a partial file at `path`."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp = f"{path}.tmp"
    try:
        torch.save(obj, tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def remove_quietly(path: str) -> bool:
    """Delete `path` if it exists. Returns True when something was removed."""
    if not os.path.exists(path):
        return False
    try:
        os.remove(path)
        return True
    except OSError:
        return False
