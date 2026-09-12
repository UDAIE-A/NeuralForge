"""Download the image-editing weights into checkpoints/ once, so the image scripts run offline.

  venv/Scripts/python.exe scripts/download_image_models.py

Fetches (~7 GB total):
  checkpoints/segformer-clothes  clothing segmentation (mask for change_clothes.py)
  checkpoints/sd15-inpaint       Realistic Vision 5.1 inpainting (change_clothes.py)
  checkpoints/sd15-base          Realistic Vision 5.1 + sd-vae-ft-mse (train_identity_lora.py)
"""

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
print("done ->", CKPT)
