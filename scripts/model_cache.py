"""Keep loaded models between runs, and let a run be cancelled without killing the process.

Command-line use is unchanged: KEEP is False, so get() just calls the loader and every model
is dropped when the script ends. The Image Studio worker (scripts/studio_worker.py) sets KEEP,
so SegFormer, SAM 2.1, Sapiens, Depth Anything and the inpainting pipeline stay on the GPU
from one job to the next; clear() releases them all. FLUX does not fit beside the rest on a
12 GB card, so change_clothes.py calls drop_except() to make room for it (and to evict it
again before an SD1.5 job).

CANCEL is set by the worker when the user presses Stop. Long loops call check_cancel() and
diffusion calls cancel_callback() every step, so a run stops within one step and the models
stay loaded.
"""

from __future__ import annotations

import gc
import threading

KEEP = False
_cache: dict = {}
_lock = threading.RLock()
CANCEL = threading.Event()


class Cancelled(Exception):
    """Raised inside a run when the user pressed Stop."""


def get(key, loader):
    """The cached object for `key`, loading it with `loader()` the first time (if KEEP)."""
    if not KEEP:
        return loader()
    with _lock:
        if key not in _cache:
            _cache[key] = loader()
        return _cache[key]


def drop(key) -> None:
    with _lock:
        _cache.pop(key, None)


def drop_except(keep) -> int:
    """Release every cached model whose key fails `keep(key)`. Returns how many."""
    with _lock:
        gone = [k for k in _cache if not keep(k)]
        for k in gone:
            del _cache[k]
    if gone:
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    return len(gone)


def clear() -> int:
    """Release every cached model (and the GPU memory behind them). Returns how many."""
    with _lock:
        n = len(_cache)
        _cache.clear()
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    return n


def loaded() -> list[str]:
    with _lock:
        return [k if isinstance(k, str) else ":".join(map(str, k)) for k in _cache]


def check_cancel() -> None:
    if CANCEL.is_set():
        raise Cancelled()


def cancel_callback(pipe, step, timestep, callback_kwargs):
    """diffusers callback_on_step_end: ask the pipeline to stop after this step."""
    if CANCEL.is_set():
        pipe._interrupt = True
    return callback_kwargs
