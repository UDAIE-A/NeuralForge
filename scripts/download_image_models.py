"""Download the image-editing weights into checkpoints/ once, so the image scripts run offline.

  venv/Scripts/python.exe scripts/download_image_models.py

Fetches (~8.5 GB total):
  checkpoints/segformer-clothes    clothing segmentation (mask for change_clothes.py)
  checkpoints/sd15-inpaint         Realistic Vision 5.1 inpainting (change_clothes.py)
  checkpoints/sd15-base            Realistic Vision 5.1 + sd-vae-ft-mse (train_identity_lora.py)
  checkpoints/controlnet-depth     ControlNet depth: body volume/proportions (change_clothes.py)
  checkpoints/controlnet-canny     ControlNet canny: neck/jewelry edges (change_clothes.py)
  checkpoints/depth-anything-small depth estimator for the depth guide
  checkpoints/hand_landmarker.task MediaPipe hand detector so hands are never repainted
"""

import urllib.request
from pathlib import Path

from huggingface_hub import snapshot_download

CKPT = Path(__file__).resolve().parent.parent / "checkpoints"

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
print("done ->", CKPT)
