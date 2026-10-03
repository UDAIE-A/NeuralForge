"""Device choice and log hygiene shared by the image scripts.

pick_device(): the GPU whenever it is usable, else the CPU. "Usable" means CUDA exists and less
than `max_used` of the card's memory is taken - by anything, including a game or another
process, which is what torch.cuda.mem_get_info reports. Call it again before each heavy stage:
a game started halfway through a run is caught before the next model loads.

quiet(): silences MediaPipe/TFLite's C++ log lines (InitGoogle, XNNPACK, "Feedback manager",
NORM_RECT) and transformers' weight-loading bars. Must run before mediapipe is imported.
"""

from __future__ import annotations

import contextlib
import functools
import os
import sys
import warnings


@contextlib.contextmanager
def native_stderr_muted():
    """Point file descriptor 2 at the null device for the duration.

    MediaPipe's C++ core logs through absl straight to fd 2 and ignores GLOG_minloglevel, so
    the only reliable mute is at the descriptor. Python exceptions still propagate normally;
    only what native code writes to stderr inside the block is dropped.
    """
    sys.stderr.flush()
    saved = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 2)
        yield
    finally:
        sys.stderr.flush()
        os.dup2(saved, 2)
        os.close(devnull)
        os.close(saved)


def muted(fn):
    """Decorator form of native_stderr_muted, for the MediaPipe helpers."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with native_stderr_muted():
            return fn(*args, **kwargs)
    return wrapper


def quiet() -> None:
    os.environ.setdefault("GLOG_minloglevel", "2")       # MediaPipe / absl C++: errors only
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")   # TensorFlow Lite
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    # library deprecation notices (torch.jit, diffusers' torch_dtype) say nothing about the run
    warnings.filterwarnings("ignore", category=FutureWarning)
    try:
        from transformers.utils import logging as hf_logging
        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bar()
    except Exception:
        pass


def gpu_memory_used() -> float | None:
    """Fraction of the GPU's memory in use by OTHER processes (a game, a browser...), or None
    without CUDA. This process's own allocations - models the Studio worker keeps loaded - are
    left out, so keeping them resident never trips the CPU fallback by itself."""
    import torch

    if not torch.cuda.is_available():
        return None
    free, total = torch.cuda.mem_get_info()
    ours = torch.cuda.memory_reserved()
    return max(0.0, (total - free - ours) / total)


def pick_device(requested: str = "auto", max_used: float = 0.8, stage: str = "", log=print) -> str:
    """'cuda' or 'cpu'. An explicit --device cuda/cpu is honoured; 'auto' applies the rule."""
    import torch

    label = f" for {stage}" if stage else ""
    if requested == "cpu":
        return "cpu"
    if not torch.cuda.is_available():
        log(f"device{label}: CPU (no CUDA GPU available)")
        return "cpu"
    used = gpu_memory_used()
    if requested == "cuda":
        return "cuda"
    if used is not None and used > max_used:
        log(f"device{label}: CPU - GPU memory is {used:.0%} used (limit {max_used:.0%}); "
            f"this will be much slower")
        return "cpu"
    log(f"device{label}: GPU ({used:.0%} of its memory in use)")
    return "cuda"
