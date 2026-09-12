"""Change the clothes in a photo of a person from a text prompt. No training.

Pipeline:
  1. Segment the clothing (SegFormer trained on fashion parsing) -> mask.
  2. Inpaint the masked region with Stable Diffusion, conditioned on the prompt.
  3. Composite: original pixels are kept everywhere outside the (feathered) mask,
     so face, hair, skin and background come back pixel-identical.

Usage:
  venv/Scripts/python.exe scripts/change_clothes.py photo.jpg "a black leather jacket and blue jeans"
  venv/Scripts/python.exe scripts/change_clothes.py photo.jpg "a red summer dress" --parts full --num 4
  venv/Scripts/python.exe scripts/change_clothes.py photo.jpg "grey hoodie" --parts upper --sdxl

Weights are loaded from checkpoints/ (run scripts/download_image_models.py once); if that folder
is missing they are fetched from the HF Hub into its cache instead.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter

# Label ids of mattmdjaga/segformer_b2_clothes
LABELS = {
    0: "background", 1: "hat", 2: "hair", 3: "sunglasses", 4: "upper_clothes",
    5: "skirt", 6: "pants", 7: "dress", 8: "belt", 9: "left_shoe", 10: "right_shoe",
    11: "face", 12: "left_leg", 13: "right_leg", 14: "left_arm", 15: "right_arm",
    16: "bag", 17: "scarf",
}
PARTS = {
    "upper": {4, 7, 8, 17},
    "lower": {5, 6, 7, 8},
    "full": {4, 5, 6, 7, 8, 17},
}
# Never paint over these, even if the dilated mask touches them.
PROTECT = {2, 3, 11}  # hair, sunglasses, face

ROOT = Path(__file__).resolve().parent.parent
CKPT = ROOT / "checkpoints"


def local_or_hub(local: Path, hub_id: str) -> str:
    """Prefer a copy under checkpoints/ (see scripts/download_image_models.py); else the Hub id."""
    return str(local) if local.is_dir() else hub_id


SEG_MODEL = local_or_hub(CKPT / "segformer-clothes", "mattmdjaga/segformer_b2_clothes")
# SD1.5-based, tuned for photoreal people. Alternatives: Lykon/dreamshaper-8-inpainting,
# stable-diffusion-v1-5/stable-diffusion-inpainting
SD15_INPAINT = local_or_hub(CKPT / "sd15-inpaint", "Uminosachi/realisticVisionV51_v51VAE-inpainting")
SDXL_INPAINT = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"

NEGATIVE = ("deformed, disfigured, bad anatomy, extra limbs, blurry, low quality, "
            "watermark, text, cropped, duplicate")


def save(img: Image.Image, path: Path) -> Path:
    """Save, falling back to a timestamped name if Windows has the target locked
    (open in Photos / Explorer preview / mid-sync gives OSError 22 or 13)."""
    try:
        img.save(path)
        return path
    except OSError as e:
        alt = path.with_name(f"{path.stem}_{int(time.time())}{path.suffix}")
        print(f"    could not overwrite {path.name} ({e.strerror}); writing {alt.name} instead")
        img.save(alt)
        return alt


def fit_image(img: Image.Image, max_side: int) -> Image.Image:
    """Downscale so the longest side <= max_side, rounded to a multiple of 8."""
    w, h = img.size
    scale = min(1.0, max_side / max(w, h))
    w, h = int(w * scale) // 8 * 8, int(h * scale) // 8 * 8
    return img.resize((w, h), Image.LANCZOS)


def segment(img: Image.Image, device: str) -> np.ndarray:
    from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

    proc = AutoImageProcessor.from_pretrained(SEG_MODEL)
    model = AutoModelForSemanticSegmentation.from_pretrained(SEG_MODEL).to(device).eval()
    with torch.no_grad():
        inputs = proc(images=img, return_tensors="pt").to(device)
        logits = model(**inputs).logits
    logits = torch.nn.functional.interpolate(logits, size=img.size[::-1], mode="bilinear", align_corners=False)
    labels = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
    del model
    torch.cuda.empty_cache()
    return labels


def build_mask(labels: np.ndarray, parts: set[int], grow: int) -> Image.Image:
    import cv2

    mask = np.isin(labels, list(parts)).astype(np.uint8) * 255
    if grow > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1))
        mask = cv2.dilate(mask, k)
    mask[np.isin(labels, list(PROTECT))] = 0
    return Image.fromarray(mask, "L")


def load_pipe(model: str, sdxl: bool, device: str):
    from diffusers import AutoPipelineForInpainting

    pipe = AutoPipelineForInpainting.from_pretrained(model, torch_dtype=torch.float16, variant="fp16" if sdxl else None)
    if sdxl:
        pipe.enable_model_cpu_offload()  # 12 GB is tight for SDXL at 1024
        pipe.enable_vae_slicing()
    else:
        pipe.to(device)
    pipe.set_progress_bar_config(leave=False)
    return pipe


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image", type=Path)
    ap.add_argument("prompt", help="describe the new clothes, e.g. 'a navy suit with a white shirt'")
    ap.add_argument("--parts", choices=PARTS, default="full", help="which clothes to replace (default: full)")
    ap.add_argument("--num", type=int, default=2, help="how many variations to generate")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=7.5)
    ap.add_argument("--grow", type=int, default=12, help="dilate the clothes mask by N px so seams blend")
    ap.add_argument("--cover-arms", action="store_true", help="also repaint bare arms (needed for sleeves/jackets)")
    ap.add_argument("--cover-legs", action="store_true", help="also repaint bare legs (needed for pants over shorts/skirt)")
    ap.add_argument("--negative", default=NEGATIVE)
    ap.add_argument("--sdxl", action="store_true", help="use SDXL inpainting (slower, higher quality, 1024px)")
    ap.add_argument("--model", default=None, help="path or HF id of an inpainting checkpoint (overrides the default)")
    ap.add_argument("--lora", type=Path, default=None, help="identity LoRA dir from scripts/train_identity_lora.py")
    ap.add_argument("--lora-scale", type=float, default=0.8)
    ap.add_argument("--out", type=Path, default=Path("outputs/clothes"))
    ap.add_argument("--mask-only", action="store_true", help="only write the mask, don't run diffusion")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)
    stem = args.image.stem

    img = Image.open(args.image).convert("RGB")
    img = fit_image(img, 1024 if args.sdxl else 768)
    print(f"image {img.size[0]}x{img.size[1]}  device={device}")

    t0 = time.time()
    labels = segment(img, device)
    found = sorted({LABELS[i] for i in np.unique(labels) if i in PARTS["full"]})
    print(f"segmented in {time.time() - t0:.1f}s, clothes found: {found or 'none'}")
    parts = set(PARTS[args.parts])
    if args.cover_arms:
        parts |= {14, 15}
    if args.cover_legs:
        parts |= {12, 13}
    mask = build_mask(labels, parts, args.grow)
    coverage = np.asarray(mask).mean() / 255
    mask_path = save(mask, args.out / f"{stem}_mask.png")
    if coverage < 0.005:
        raise SystemExit(f"mask covers {coverage:.1%} of the image - no '{args.parts}' clothes detected. "
                         f"Try --parts full or check {mask_path}")
    print(f"mask covers {coverage:.1%} of the image -> {mask_path}")
    if args.mask_only:
        return

    pipe = load_pipe(args.model or (SDXL_INPAINT if args.sdxl else SD15_INPAINT), args.sdxl, device)
    subject = "a person"
    if args.lora:
        pipe.load_lora_weights(str(args.lora))
        pipe.fuse_lora(lora_scale=args.lora_scale)
        token_file = args.lora / "token.txt"
        subject = token_file.read_text().strip() if token_file.exists() else subject
    prompt = f"{subject} wearing {args.prompt}, photo, detailed fabric, natural lighting"
    # feathered mask for compositing so the seam between old and new pixels blends
    soft = mask.filter(ImageFilter.GaussianBlur(4))
    base_seed = args.seed if args.seed is not None else int(torch.seed() % 2**31)

    seed = base_seed
    for i in range(args.num):
        t0 = time.time()
        for attempt in range(3):
            gen = torch.Generator(device="cpu").manual_seed(seed)
            out = pipe(
                prompt=prompt, negative_prompt=args.negative, image=img, mask_image=mask,
                width=img.size[0], height=img.size[1],
                num_inference_steps=args.steps, guidance_scale=args.guidance, generator=gen,
                strength=0.99 if args.sdxl else 1.0,
            )
            # the SD1.5 safety checker misfires on ordinary photos of people and returns a
            # black image; bump the seed and try again instead of writing black clothes
            flagged = getattr(out, "nsfw_content_detected", None)
            if flagged and flagged[0]:
                print(f"    seed {seed} flagged by safety checker, retrying with seed {seed + 1}")
                seed += 1
                continue
            break
        else:
            print(f"[{i + 1}/{args.num}] skipped: 3 seeds in a row flagged")
            continue
        result = out.images[0].resize(img.size)
        final = Image.composite(result, img, soft)
        path = save(final, args.out / f"{stem}_{i}_seed{seed}.png")
        print(f"[{i + 1}/{args.num}] {time.time() - t0:.1f}s  seed={seed}  -> {path}")
        seed += 1


if __name__ == "__main__":
    main()
