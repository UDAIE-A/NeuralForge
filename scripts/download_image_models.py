"""Download the image-editing weights into checkpoints/ once, so the image scripts run offline.

  venv/Scripts/python.exe scripts/download_image_models.py                      # SD1.5 stack, ~8.5 GB
  venv/Scripts/python.exe scripts/download_image_models.py --flux               # + FLUX.2 klein 4B, ~16 GB
  venv/Scripts/python.exe scripts/download_image_models.py --flux --flux-dir D:/models/flux2-klein-4b
                       # FLUX on another drive; checkpoints/flux2-klein-4b becomes a junction to it

Fetches (~8.5 GB total):
  checkpoints/segformer-clothes    clothing segmentation (mask for change_clothes.py)
  checkpoints/sd15-inpaint         Realistic Vision 5.1 inpainting (change_clothes.py)
  checkpoints/sd15-base            Realistic Vision 5.1 + sd-vae-ft-mse (train_identity_lora.py)
  checkpoints/controlnet-depth     ControlNet depth: body volume/proportions (change_clothes.py)
  checkpoints/controlnet-canny     ControlNet canny: neck/jewelry edges (change_clothes.py)
  checkpoints/depth-anything-small depth estimator for the depth guide
  checkpoints/hand_landmarker.task MediaPipe hand detector so hands are never repainted
  checkpoints/flux2-klein-4b       (--flux) FLUX.2 klein 4B inpainting, Apache-2.0, ~16 GB
"""

import argparse
import subprocess
import urllib.request
from pathlib import Path

from huggingface_hub import snapshot_download

CKPT = Path(__file__).resolve().parent.parent / "checkpoints"
ap = argparse.ArgumentParser()
ap.add_argument("--flux", action="store_true", help="also fetch FLUX.2 klein 4B (~16 GB)")
ap.add_argument("--flux-dir", type=Path, default=None, help="store FLUX here (another drive) and junction it into checkpoints/")
args = ap.parse_args()

snapshot_download("mattmdjaga/segformer_b2_clothes", local_dir=CKPT / "segformer-clothes",
                  ignore_patterns=["*.msgpack", "*.h5", "*.onnx", "*.bin"])
snapshot_download("Uminosachi/realisticVisionV51_v51VAE-inpainting", local_dir=CKPT / "sd15-inpaint",
                  ignore_patterns=["*.ckpt", "*.bin", "*.msgpack"])
snapshot_download("SG161222/Realistic_Vision_V5.1_noVAE", local_dir=CKPT / "sd15-base",
                  allow_patterns=["*.json", "*.txt", "*.md", "unet/*.safetensors", "text_encoder/*.safetensors",
                                  "tokenizer/*", "scheduler/*"])
snapshot_download("stabilityai/sd-vae-ft-mse", local_dir=CKPT / "sd15-base" / "vae",
                  allow_patterns=["*.json", "diffusion_pytorch_model.safetensors"])
snapshot_download("lllyasviel/control_v11f1p_sd15_depth", local_dir=CKPT / "controlnet-depth",
                  allow_patterns=["config.json", "diffusion_pytorch_model.fp16.safetensors"])
snapshot_download("lllyasviel/control_v11p_sd15_canny", local_dir=CKPT / "controlnet-canny",
                  allow_patterns=["config.json", "diffusion_pytorch_model.fp16.safetensors"])
snapshot_download("depth-anything/Depth-Anything-V2-Small-hf", local_dir=CKPT / "depth-anything-small",
                  allow_patterns=["*.json", "*.safetensors"])
hand = CKPT / "hand_landmarker.task"
if not hand.exists():
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        hand)
if args.flux:
    target = args.flux_dir or CKPT / "flux2-klein-4b"
    snapshot_download("black-forest-labs/FLUX.2-klein-4B", local_dir=target,
                      allow_patterns=["model_index.json", "scheduler/*", "text_encoder/*", "tokenizer/*",
                                      "transformer/*", "vae/*", "LICENSE.md"])  # skips the duplicate root .safetensors
    link = CKPT / "flux2-klein-4b"
    if args.flux_dir and not link.exists():
        # a directory junction needs no admin rights on Windows
        subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target.resolve())], check=True)
print("done ->", CKPT)
