"""Image Studio's long-lived worker: runs change_clothes jobs with the models kept loaded.

Started by webui/image_api.py.

  commands  over a private connection on 127.0.0.1 (multiprocessing.connection, authkey from
            the STUDIO_KEY environment variable; the port is announced as "@@PORT <n>"):
            {"cmd": "run", "id": "...", "argv": [...change_clothes.py arguments...]}
            {"cmd": "cancel"}   stop the running job at the next step; models stay loaded
            {"cmd": "unload"}   release every cached model (also: just kill this process)
  stdout    ordinary log lines of the running job, plus control lines starting with "@@":
            @@PORT <n> | @@READY | @@START <id> | @@END <id> <rc> | @@MODELS <json>

Commands deliberately do not come in on stdin. On Windows a thread blocked reading a pipe
deadlocks any other thread that loads a DLL touching the standard handles - SciPy's BLAS,
imported lazily by transformers, froze the first job forever. A socket has no such tie.

Models stay on the GPU between jobs - the point of this process. Two exits from that:
  * the user's "Unload" button (the server kills this process: guaranteed to free the GPU);
  * other programs (a game) pushing the GPU past --gpu-max-used while no job runs: the cache
    is released automatically so the worker never sits on memory something else needs.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import sys
import threading
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
# stdout is a pipe to the server: without line buffering, job logs arrive in 8 KB lumps
sys.stdout.reconfigure(line_buffering=True)

import model_cache  # noqa: E402
from devices import gpu_memory_used, quiet  # noqa: E402

quiet()
model_cache.KEEP = True

import change_clothes  # noqa: E402  (imports torch; after quiet())

jobs: queue.Queue = queue.Queue()
busy = threading.Lock()


def say(line: str) -> None:
    print(line, flush=True)


def report_models(note: str = "") -> None:
    info = {"models": model_cache.loaded(), "note": note}
    try:
        import torch
        if torch.cuda.is_available():
            info["gpu_reserved_gb"] = round(torch.cuda.memory_reserved() / 1e9, 2)
    except Exception:
        pass
    say("@@MODELS " + json.dumps(info))


def serve_commands(listener) -> None:
    """Accept the server's connection and queue its commands (cancel acts immediately)."""
    conn = listener.accept()
    try:
        while True:
            msg = conn.recv()
            if not isinstance(msg, dict):
                continue
            if msg.get("cmd") == "cancel":
                model_cache.CANCEL.set()
            else:
                jobs.put(msg)
    except (EOFError, OSError):
        pass
    model_cache.CANCEL.set()        # server went away: stop whatever is running, then exit
    jobs.put({"cmd": "exit"})


def watch_gpu(max_used: float) -> None:
    """Release the cache while idle if other programs need the GPU."""
    while True:
        time.sleep(5)
        if not model_cache.loaded():
            continue
        others = gpu_memory_used()
        if others is None or others <= max_used:
            continue
        if busy.acquire(blocking=False):          # never in the middle of a job
            try:
                n = model_cache.clear()
                report_models(f"other programs use {others:.0%} of the GPU - unloaded {n} model(s)")
            finally:
                busy.release()


def run_job(msg: dict) -> None:
    job_id = msg.get("id", "?")
    model_cache.CANCEL.clear()
    say(f"@@START {job_id}")
    rc = 0
    with busy:
        t0 = time.time()
        try:
            change_clothes.main(list(msg.get("argv", [])))
        except model_cache.Cancelled:
            say("stopped - models kept loaded for the next run")
            rc = 130
        except SystemExit as e:                    # argparse errors / "nothing to repaint"
            if e.code not in (0, None):
                say(str(e.code) if not isinstance(e.code, int) else "")
                rc = e.code if isinstance(e.code, int) else 1
        except Exception:
            traceback.print_exc(file=sys.stdout)
            rc = 1
        finally:
            model_cache.CANCEL.clear()
        say(f"finished in {time.time() - t0:.1f}s")
    say(f"@@END {job_id} {rc}")
    report_models()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu-max-used", type=float, default=0.8)
    args = ap.parse_args()
    from multiprocessing.connection import Listener
    listener = Listener(("127.0.0.1", 0), authkey=os.environ.get("STUDIO_KEY", "studio").encode())
    say(f"@@PORT {listener.address[1]}")
    threading.Thread(target=serve_commands, args=(listener,), daemon=True).start()
    threading.Thread(target=watch_gpu, args=(args.gpu_max_used,), daemon=True).start()
    say("@@READY")
    report_models()
    while True:
        msg = jobs.get()
        cmd = msg.get("cmd")
        if cmd == "exit":
            return
        if cmd == "unload":
            with busy:
                n = model_cache.clear()
            report_models(f"unloaded {n} model(s)")
        elif cmd == "run":
            run_job(msg)


if __name__ == "__main__":
    main()
