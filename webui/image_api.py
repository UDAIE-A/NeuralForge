"""Image Studio API.

Change-clothes jobs go to ONE long-lived worker process (scripts/studio_worker.py) that keeps
its models on the GPU between jobs: Stop cancels the running job and keeps them loaded;
Unload kills the worker, which is guaranteed to give the memory back. LoRA training and
sampling still run as one-off subprocesses (and unload the worker first - they need the GPU).
The CLI scripts stay the single source of truth for what a job does.

Mounted by webui/server.py under /api/image; the page is webui/static/image.html at /image.
One job at a time - there is one GPU.
"""

from __future__ import annotations

import atexit
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
CKPT = ROOT / "checkpoints"
OUT = ROOT / "outputs"
UPLOADS = OUT / "uploads"
LORA_DIR = CKPT / "lora"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
STOP_GRACE_S = 10  # Stop waits this long for a clean cancel before killing the worker

router = APIRouter(prefix="/api/image")


# ----------------------------------------------------------------------------- jobs

class Job:
    def __init__(self, kind: str, cmd: list[str], out_dir: Path | None, meta: dict):
        self.id = uuid.uuid4().hex[:10]
        self.kind = kind
        self.cmd = cmd
        self.out_dir = out_dir
        self.meta = meta
        self.log: deque[str] = deque(maxlen=400)
        self.status = "running"
        self.returncode: int | None = None
        self.started = time.time()
        self.ended: float | None = None
        self.proc: subprocess.Popen | None = None

    def start(self):
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
        self.proc = subprocess.Popen(self.cmd, cwd=str(ROOT), env=env, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self):
        assert self.proc and self.proc.stdout
        buf = ""
        for chunk in iter(lambda: self.proc.stdout.read(1), ""):
            if chunk in ("\n", "\r"):
                line = buf.strip()
                buf = ""
                if not line or _NOISE.search(line):
                    continue
                # tqdm progress: keep only the latest line of a bar
                if self.log and _BAR.search(line) and _BAR.search(self.log[-1]):
                    self.log[-1] = line
                else:
                    self.log.append(line)
            else:
                buf += chunk
        if buf.strip():
            self.log.append(buf.strip())
        self.returncode = self.proc.wait()
        self.ended = time.time()
        self.status = "done" if self.returncode == 0 else ("stopped" if self.status == "stopping" else "error")

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.status = "stopping"
            self.proc.terminate()

    def outputs(self) -> list[str]:
        if not self.out_dir or not self.out_dir.is_dir():
            return []
        files = sorted(p for p in self.out_dir.iterdir() if p.suffix.lower() == ".png")
        return [_url(p) for p in files]

    def to_dict(self, full_log: bool = False) -> dict:
        lines = list(self.log)
        return {
            "id": self.id, "kind": self.kind, "status": self.status, "meta": self.meta,
            "elapsed": round((self.ended or time.time()) - self.started, 1),
            "log": lines if full_log else lines[-60:],
            "outputs": self.outputs(), "returncode": self.returncode,
        }


class WorkerJob(Job):
    """A change-clothes job executed by the resident worker instead of its own process."""

    def __init__(self, argv: list[str], out_dir: Path, meta: dict):
        super().__init__("clothes", argv, out_dir, meta)

    def start(self):
        WORKER.run(self)

    def stop(self):
        if self.status == "running":
            self.status = "stopping"
            WORKER.send({"cmd": "cancel"})
            # cancel is only seen between steps; a model download or load never checks it,
            # so if the job is still going after a grace period, kill the worker instead
            threading.Timer(STOP_GRACE_S, self._force_stop).start()

    def _force_stop(self):
        if self.status == "stopping" and WORKER.job is self:
            self.log.append(f"did not stop within {STOP_GRACE_S}s (loading a model?) - killing the worker")
            WORKER.kill()

    def finish(self, rc: int):
        self.returncode = rc
        self.ended = time.time()
        self.status = "done" if rc == 0 else ("stopped" if rc == 130 or self.status == "stopping" else "error")


class Worker:
    """Owns the resident studio_worker.py process and routes its output to the current job."""

    def __init__(self):
        self.proc: subprocess.Popen | None = None
        self.conn = None
        self.job: WorkerJob | None = None
        self.models: dict = {"models": [], "note": ""}
        self.lock = threading.Lock()
        self.ready = threading.Event()

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def _spawn(self):
        key = secrets.token_hex(16)
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1", STUDIO_KEY=key)
        self.ready.clear()
        self.proc = subprocess.Popen([sys.executable, str(SCRIPTS / "studio_worker.py")], cwd=str(ROOT), env=env,
                                     stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     text=True, encoding="utf-8", errors="replace")
        self._key = key
        threading.Thread(target=self._pump, args=(self.proc,), daemon=True).start()
        if not self.ready.wait(120):
            self.kill()
            raise HTTPException(500, "the image worker did not start")

    def _line(self, line: str):
        if line.startswith("@@PORT "):
            from multiprocessing.connection import Client
            self.conn = Client(("127.0.0.1", int(line.split()[1])), authkey=self._key.encode())
        elif line.startswith("@@READY"):
            self.ready.set()
        elif line.startswith("@@MODELS "):
            try:
                self.models = json.loads(line[9:])
            except json.JSONDecodeError:
                pass
        elif line.startswith("@@END ") and self.job is not None:
            parts = line.split()
            self.job.finish(int(parts[2]) if len(parts) > 2 and parts[2].lstrip("-").isdigit() else 1)
            self.job = None
        elif line.startswith("@@"):
            return
        elif self.job is not None:
            log = self.job.log
            if _NOISE.search(line):
                return
            if log and _BAR.search(line) and _BAR.search(log[-1]):
                log[-1] = line          # tqdm: keep only the latest state of a progress bar
            else:
                log.append(line)

    def _pump(self, proc: subprocess.Popen):
        buf = ""
        for ch in iter(lambda: proc.stdout.read(1), ""):
            if ch in ("\n", "\r"):
                if buf.strip():
                    self._line(buf.strip())
                buf = ""
            else:
                buf += ch
        # the process died: whatever was running failed
        if self.job is not None:
            self.job.log.append("the image worker exited")
            self.job.finish(1 if self.job.status != "stopping" else 130)
            self.job = None
        self.models = {"models": [], "note": ""}

    def send(self, msg: dict):
        if self.conn is not None:
            try:
                self.conn.send(msg)
            except OSError:
                pass

    def run(self, job: WorkerJob):
        with self.lock:
            if not self.alive():
                self._spawn()
            self.job = job
            self.send({"cmd": "run", "id": job.id, "argv": job.cmd})

    def kill(self) -> bool:
        """Force quit: the OS takes back every byte of GPU memory the worker held."""
        was = self.alive()
        if self.proc is not None:
            try:
                self.proc.kill()
                self.proc.wait(timeout=15)
            except Exception:
                pass
        self.proc, self.conn = None, None
        self.models = {"models": [], "note": "unloaded"}
        return was

    def info(self) -> dict:
        return {"running": self.alive(), "loaded": self.models.get("models", []),
                "reserved_gb": self.models.get("gpu_reserved_gb"), "note": self.models.get("note", "")}


WORKER = Worker()
atexit.register(WORKER.kill)


def _gpu() -> dict | None:
    """Whole-card usage from nvidia-smi (every process, including games)."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5).stdout
        name, used, total, util = [x.strip() for x in out.splitlines()[0].split(",")]
        return {"name": name, "used_gb": round(int(used) / 1024, 1), "total_gb": round(int(total) / 1024, 1),
                "util": int(util)}
    except Exception:
        return None


_NOISE = re.compile(r"Warning|warn\(|deprecat|Loading weights|Fetching \d+ files|XNNPACK|^I0000|^W0000|Siglip2")
_BAR = re.compile(r"\d+%\|")
JOBS: dict[str, Job] = {}
JOBS_LOCK = threading.Lock()


def _url(p: Path) -> str:
    """Browser URL for a file under outputs/ or checkpoints/lora/ (both mounted by server.py)."""
    p = p.resolve()
    for base, prefix in ((OUT.resolve(), "/outputs"), (LORA_DIR.resolve(), "/lora")):
        try:
            return f"{prefix}/{p.relative_to(base).as_posix()}"
        except ValueError:
            continue
    return ""


def _busy() -> Job | None:
    return next((j for j in JOBS.values() if j.status in ("running", "stopping")), None)


def _launch(kind: str, cmd: list[str], out_dir: Path | None, meta: dict, worker: bool = False) -> dict:
    with JOBS_LOCK:
        if (b := _busy()) is not None:
            raise HTTPException(409, f"a {b.kind} job is already running ({b.id}); one GPU, one job")
        if worker:
            job = WorkerJob(cmd, out_dir, meta)
        else:
            WORKER.kill()   # training/sampling load their own models and need the GPU free
            job = Job(kind, cmd, out_dir, meta)
        JOBS[job.id] = job
        try:
            job.start()
        except Exception:
            del JOBS[job.id]     # never leave a phantom "running" job blocking the GPU
            raise
    return job.to_dict()


def _safe_stem(name: str) -> str:
    stem = Path(name).stem
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_") or "image"
    return stem[:40]


# ----------------------------------------------------------------------------- info

@router.get("/info")
def info():
    loras = []
    if LORA_DIR.is_dir():
        for d in sorted(LORA_DIR.iterdir()):
            if (d / "pytorch_lora_weights.safetensors").exists():
                samples = sorted((d / "samples").glob("*.png")) if (d / "samples").is_dir() else []
                crops = sorted((d / "train_crops").glob("*.jpg")) if (d / "train_crops").is_dir() else []
                token = (d / "token.txt").read_text().strip() if (d / "token.txt").exists() else "ohwx person"
                loras.append({"name": d.name, "token": token, "samples": [_url(p) for p in samples],
                              "crops": [_url(p) for p in crops]})
    return {
        "engines": {
            "flux": (CKPT / "flux2-klein-4b" / "model_index.json").exists(),
            "sam": (CKPT / "sam2.1-large" / "config.json").exists(),
            "sapiens": any((CKPT / "sapiens-seg-1b").glob("*.pt2")),
            "sd15": (CKPT / "sd15-inpaint").is_dir(),
            "controlnet": (CKPT / "controlnet-depth").is_dir() and (CKPT / "controlnet-canny").is_dir(),
            "hands": (CKPT / "hand_landmarker.task").exists(),
            "sd15_base": (CKPT / "sd15-base").is_dir(),
        },
        "loras": loras,
        "busy": (b.to_dict() if (b := _busy()) else None),
        "models": WORKER.info(),
        "gpu": _gpu(),
    }


@router.post("/models/unload")
def unload_models():
    """Force quit the worker: frees all its GPU memory. Refused while a job is running."""
    if (b := _busy()) is not None:
        raise HTTPException(409, f"stop the running {b.kind} job first")
    return {"ok": True, "was_running": WORKER.kill()}


# ----------------------------------------------------------------------------- uploads

@router.post("/upload")
async def upload(file: UploadFile = File(...)):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in IMAGE_EXTS:
        raise HTTPException(400, f"not an image: {file.filename}")
    UPLOADS.mkdir(parents=True, exist_ok=True)
    dest = UPLOADS / f"{_safe_stem(file.filename)}_{uuid.uuid4().hex[:6]}{ext}"
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"path": dest.relative_to(ROOT).as_posix(), "url": _url(dest), "name": file.filename}


@router.post("/upload_many")
async def upload_many(files: list[UploadFile] = File(...)):
    """Photos for LoRA training -> outputs/uploads/train_<id>/ (one folder = one person)."""
    folder = UPLOADS / f"train_{uuid.uuid4().hex[:6]}"
    folder.mkdir(parents=True, exist_ok=True)
    saved = []
    for file in files:
        ext = Path(file.filename or "").suffix.lower()
        if ext not in IMAGE_EXTS:
            continue
        dest = folder / f"{_safe_stem(file.filename)}{ext}"
        with dest.open("wb") as f:
            shutil.copyfileobj(file.file, f)
        saved.append(_url(dest))
    if not saved:
        shutil.rmtree(folder, ignore_errors=True)
        raise HTTPException(400, "no image files in the upload")
    return {"folder": folder.relative_to(ROOT).as_posix(), "count": len(saved), "urls": saved}


# ----------------------------------------------------------------------------- change clothes

class ClothesRequest(BaseModel):
    image: str                      # path returned by /upload
    prompt: str | None = None
    upper: str | None = None
    lower: str | None = None
    parts: str = "full"
    engine: str | None = None       # flux | sd15 | None (auto)
    ref: str | None = None
    num: int = 2
    seed: int | None = None
    steps: int | None = None
    guidance: float | None = None
    grow: int = 20
    no_neckline: bool = False
    cover_arms: bool = False
    cover_legs: bool = False
    expose: bool | None = None        # None: on by itself for bikinis/tube tops/halter necks
    res: int | None = None
    hires: int | None = None
    refine_strength: float | None = None
    no_control: bool = False
    depth_scale: float | None = None
    edge_scale: float | None = None
    negative: str | None = None
    lora: str | None = None         # LoRA name under checkpoints/lora
    lora_scale: float | None = None
    sdxl: bool = False
    mask_only: bool = False
    parser: str = "segformer"       # segformer | sapiens (non-commercial)
    device: str = "auto"            # auto | cuda | cpu


@router.post("/clothes")
def clothes(req: ClothesRequest):
    image = ROOT / req.image
    if not image.is_file():
        raise HTTPException(400, "upload the photo first")
    if not (req.prompt or req.upper or req.lower):
        raise HTTPException(400, "describe the outfit (prompt, or upper/lower)")
    if (req.engine == "flux" or req.ref) and not (CKPT / "flux2-klein-4b" / "model_index.json").exists():
        # otherwise diffusers quietly starts a ~16 GB download from the Hub mid-job
        raise HTTPException(400, "FLUX is not downloaded - run scripts/download_image_models.py --flux, "
                                 "or pick SD 1.5")
    out_dir = OUT / "clothes" / f"{_safe_stem(image.name)}_{uuid.uuid4().hex[:6]}"
    # change_clothes.py arguments; the resident worker runs them in-process
    cmd = [str(image), "--out", str(out_dir), "--num", str(req.num), "--parts", req.parts,
           "--grow", str(req.grow), "--device", req.device]
    if req.prompt and not (req.upper or req.lower):
        cmd.insert(1, req.prompt)
    if req.parser == "sapiens":
        cmd += ["--parser", "sapiens"]
    for flag, val in (("--upper", req.upper), ("--lower", req.lower), ("--engine", req.engine),
                      ("--seed", req.seed), ("--steps", req.steps), ("--guidance", req.guidance),
                      ("--res", req.res), ("--hires", req.hires), ("--refine-strength", req.refine_strength),
                      ("--depth-scale", req.depth_scale), ("--edge-scale", req.edge_scale),
                      ("--negative", req.negative), ("--lora-scale", req.lora_scale)):
        if val not in (None, ""):
            cmd += [flag, str(val)]
    if req.ref:
        ref = ROOT / req.ref
        if not ref.is_file():
            raise HTTPException(400, "reference garment photo not found")
        cmd += ["--ref", str(ref)]
    if req.lora:
        lora = LORA_DIR / req.lora
        if not (lora / "pytorch_lora_weights.safetensors").exists():
            raise HTTPException(400, f"no LoRA named {req.lora}")
        cmd += ["--lora", str(lora)]
    for flag, on in (("--no-neckline", req.no_neckline), ("--cover-arms", req.cover_arms),
                     ("--cover-legs", req.cover_legs), ("--no-control", req.no_control),
                     ("--sdxl", req.sdxl), ("--mask-only", req.mask_only)):
        if on:
            cmd.append(flag)
    if req.expose is not None:
        cmd.append("--expose" if req.expose else "--no-expose")
    return _launch("clothes", cmd, out_dir, {"image": _url(image), "num": req.num, "mask_only": req.mask_only,
                                             "prompt": req.prompt or f"{req.upper or ''} | {req.lower or ''}"},
                   worker=True)


# ----------------------------------------------------------------------------- LoRA training

class TrainRequest(BaseModel):
    folder: str                     # from /upload_many, or any folder path under the repo
    name: str
    token: str = "ohwx person"
    steps: int = 800
    rank: int = 16
    lr: float = 1e-4
    batch: int = 2
    res: int = 512
    min_face: int = 48
    seed: int | None = None


@router.post("/train")
def train(req: TrainRequest):
    folder = ROOT / req.folder
    if not folder.is_dir():
        raise HTTPException(400, "photo folder not found")
    name = re.sub(r"[^A-Za-z0-9_-]+", "_", req.name).strip("_")
    if not name:
        raise HTTPException(400, "give the LoRA a name")
    cmd = [sys.executable, str(SCRIPTS / "train_identity_lora.py"), "--images", str(folder), "--name", name,
           "--token", req.token, "--steps", str(req.steps), "--rank", str(req.rank), "--lr", str(req.lr),
           "--batch", str(req.batch), "--res", str(req.res), "--min-face", str(req.min_face)]
    if req.seed is not None:
        cmd += ["--seed", str(req.seed)]
    return _launch("train", cmd, LORA_DIR / name / "samples", {"name": name, "folder": req.folder})


class SampleRequest(BaseModel):
    name: str
    prompt: str | None = None
    num: int = 2
    lora_scale: float = 0.9
    seed: int | None = None


@router.post("/sample")
def sample(req: SampleRequest):
    if not (LORA_DIR / req.name / "pytorch_lora_weights.safetensors").exists():
        raise HTTPException(400, f"no LoRA named {req.name}")
    cmd = [sys.executable, str(SCRIPTS / "train_identity_lora.py"), "--name", req.name, "--sample-only",
           "--num", str(req.num), "--lora-scale", str(req.lora_scale)]
    if req.prompt:
        cmd += ["--prompt", req.prompt]
    if req.seed is not None:
        cmd += ["--seed", str(req.seed)]
    return _launch("sample", cmd, LORA_DIR / req.name / "samples", {"name": req.name, "prompt": req.prompt})


# ----------------------------------------------------------------------------- job status

@router.get("/jobs")
def jobs():
    return [j.to_dict() for j in sorted(JOBS.values(), key=lambda j: j.started, reverse=True)][:20]


@router.get("/jobs/{job_id}")
def job(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(404, "no such job")
    return JOBS[job_id].to_dict(full_log=True)


@router.post("/jobs/{job_id}/stop")
def stop(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(404, "no such job")
    JOBS[job_id].stop()
    return JSONResponse({"ok": True})
