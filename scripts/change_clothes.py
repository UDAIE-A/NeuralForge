"""Change the clothes in a photo of a person from a text prompt. No training.

Pipeline:
  1. Segment the clothing (SegFormer trained on fashion parsing) -> mask. For upper-body
     changes the mask also covers the neck/shoulders down to the old neckline, so the new
     top can have any collar, straps or sleeves (face and hair are always protected).
  2. Inpaint the masked region with Stable Diffusion (DPM++ 2M Karras), cropped to the
     clothes so the garment is rendered at full resolution.
  3. Refine: upscale to --hires and run a low-strength second pass over the mask for
     sharper fabric and seams.
  4. Composite: original pixels are kept everywhere outside the feathered mask, so face,
     hair, skin and background come back pixel-identical.

Usage:
  venv/Scripts/python.exe scripts/change_clothes.py photo.jpg "a black leather jacket and blue jeans"
  venv/Scripts/python.exe scripts/change_clothes.py photo.jpg --upper "a white linen shirt" --lower "beige chinos"
  venv/Scripts/python.exe scripts/change_clothes.py photo.jpg "a red summer dress" --num 4
  venv/Scripts/python.exe scripts/change_clothes.py photo.jpg "grey hoodie" --parts upper --lora checkpoints/lora/me

--upper/--lower run as two passes, so each garment follows its own description instead of
SD1.5 blending them ("white top, denim skirt" -> denim top). One prompt + --parts full is
best for one-piece outfits (dress, saree, suit).

Weights are loaded from checkpoints/ (run scripts/download_image_models.py once); if that folder
is missing they are fetched from the HF Hub into its cache instead.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageChops, ImageFilter

# Label ids of mattmdjaga/segformer_b2_clothes
LABELS = {
    0: "background", 1: "hat", 2: "hair", 3: "sunglasses", 4: "upper_clothes",
    5: "skirt", 6: "pants", 7: "dress", 8: "belt", 9: "left_shoe", 10: "right_shoe",
    11: "face", 12: "left_leg", 13: "right_leg", 14: "left_arm", 15: "right_arm",
    16: "bag", 17: "scarf",
}
UPPER = {4, 7, 17}
PARTS = {
    "upper": UPPER | {8},
    "lower": {5, 6, 7, 8},
    "full": UPPER | {5, 6, 8},
}
# Never paint over these, even if the dilated mask touches them.
PROTECT = {1, 2, 3, 11}  # hat, hair, sunglasses, face
ARMS, LEGS = {14, 15}, {12, 13}  # also protected unless --cover-arms / --cover-legs
FACE = 11

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

NEGATIVE = ("deformed, disfigured, bad anatomy, extra limbs, mutated hands, blurry, low quality, "
            "lowres, jpeg artifacts, watermark, text, cropped, duplicate, nude, bare skin, "
            "cartoon, painting, 3d render")


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


def bbox(m: np.ndarray):
    ys, xs = np.where(m)
    return None if len(xs) == 0 else (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))


def neckline_zone(labels: np.ndarray) -> np.ndarray:
    """Shoulders/neck box: from mouth level down to just below the old neckline, about three
    face-widths wide. Thin straps and bare shoulders are labelled skin/background by the
    segmenter, so without this the old neckline survives and the new top is forced into the
    same cut. Face pixels are removed again by PROTECT, so starting high costs nothing.
    (The segmenter's 'face' label includes the neck, so its bottom edge is the neckline.)"""
    up = bbox(np.isin(labels, list(UPPER)))
    face = bbox(labels == FACE)
    if up is None or face is None:
        return np.zeros(labels.shape, bool)
    fx0, fy0, fx1, fy1 = face
    fw, fh = fx1 - fx0, fy1 - fy0
    cx = (fx0 + fx1) // 2
    top = fy0 + fh // 2
    bottom = min(labels.shape[0], up[1] + fh // 2)
    zone = np.zeros(labels.shape, bool)
    zone[top:bottom, max(0, int(cx - 1.5 * fw)):min(labels.shape[1], int(cx + 1.5 * fw))] = True
    # keep to the body: straps thinner than --grow get picked up by the dilation from the skin next to them
    return zone & (labels != 0)


def split_dress(labels: np.ndarray) -> np.ndarray:
    """Relabel the lower part of a dress (label 7) as skirt (5) so --upper/--lower can address
    the two halves separately. The split is at ~42% of the dress height (about the hips)."""
    dress = labels == 7
    box = bbox(dress)
    if box is None:
        return labels
    y_split = box[1] + int(0.42 * (box[3] - box[1]))
    out = labels.copy()
    lower = dress.copy()
    lower[:y_split] = False
    out[lower] = 5
    return out


def build_mask(labels: np.ndarray, parts: set[int], grow: int, neckline: bool, protect: set[int]) -> Image.Image:
    import cv2

    mask = np.isin(labels, list(parts))
    zone = neckline_zone(labels) if neckline else np.zeros(labels.shape, bool)
    mask |= zone
    mask = mask.astype(np.uint8) * 255
    if grow > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1))
        mask = cv2.dilate(mask, k)
    # Protect face/hair, minus a few boundary pixels: the segmenter's 'face' runs down the neck
    # to the old neckline, and locking its edge leaves a fringe of the old garment that the
    # model then continues (e.g. a coloured trim). Face/jaw borders hair and background, never
    # clothes, so the erosion changes nothing there.
    keep = np.isin(labels, list(protect))
    # shoulders are labelled 'arm'; inside the neckline zone they must be repaintable or the
    # old straps survive. Face/hair stay protected everywhere.
    keep &= ~(zone & ~np.isin(labels, list(PROTECT)))
    keep = keep.astype(np.uint8)
    e = max(3, grow // 2)  # segmenter boundaries are fuzzy by ~8 px at 768
    keep = cv2.erode(keep, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * e + 1, 2 * e + 1)))
    mask[keep.astype(bool)] = 0
    return Image.fromarray(mask, "L")


def load_pipe(model: str, sdxl: bool, device: str):
    from diffusers import AutoPipelineForInpainting, DPMSolverMultistepScheduler

    pipe = AutoPipelineForInpainting.from_pretrained(model, torch_dtype=torch.float16, variant="fp16" if sdxl else None)
    # DPM++ 2M Karras: sharper fabric than the default sampler at the same step count
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config, algorithm_type="dpmsolver++", use_karras_sigmas=True, final_sigmas_type="sigma_min")
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
    ap.add_argument("prompt", nargs="?", help="the new outfit, e.g. 'a navy suit with a white shirt'")
    ap.add_argument("--upper", help="prompt for the top only (own pass); combine with --lower")
    ap.add_argument("--lower", help="prompt for skirt/pants only (own pass); combine with --upper")
    ap.add_argument("--parts", choices=PARTS, default="full", help="which clothes the single prompt replaces")
    ap.add_argument("--num", type=int, default=2, help="how many variations to generate")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=7.0)
    ap.add_argument("--grow", type=int, default=20, help="dilate the clothes mask by N px so the new garment "
                    "can take its own shape and seams blend")
    ap.add_argument("--no-neckline", action="store_true", help="don't repaint neck/shoulders for upper changes "
                    "(keeps the old neckline exactly)")
    ap.add_argument("--cover-arms", action="store_true", help="also repaint bare arms (needed for sleeves/jackets)")
    ap.add_argument("--cover-legs", action="store_true", help="also repaint bare legs (needed for pants over shorts/skirt)")
    ap.add_argument("--res", type=int, default=768, help="working resolution of the first pass (longest side)")
    ap.add_argument("--hires", type=int, default=1024, help="resolution of the refine pass; 0 disables it")
    ap.add_argument("--refine-strength", type=float, default=0.35, help="how much the refine pass may change (0-1)")
    ap.add_argument("--negative", default=NEGATIVE)
    ap.add_argument("--sdxl", action="store_true", help="use SDXL inpainting (slower, higher quality, 1024px)")
    ap.add_argument("--model", default=None, help="path or HF id of an inpainting checkpoint (overrides the default)")
    ap.add_argument("--lora", type=Path, default=None, help="identity LoRA dir from scripts/train_identity_lora.py")
    ap.add_argument("--lora-scale", type=float, default=0.8)
    ap.add_argument("--out", type=Path, default=Path("outputs/clothes"))
    ap.add_argument("--mask-only", action="store_true", help="only write the mask, don't run diffusion")
    args = ap.parse_args()

    # passes: (parts, prompt). --upper/--lower are separate passes; lower first so the top overlaps it.
    if args.upper or args.lower:
        if args.prompt:
            ap.error("give either a single prompt or --upper/--lower, not both")
        passes = [(p, t) for p, t in (("lower", args.lower), ("upper", args.upper)) if t]
    elif args.prompt:
        passes = [(args.parts, args.prompt)]
    else:
        ap.error("need a prompt or --upper/--lower")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)
    stem = args.image.stem

    src = Image.open(args.image).convert("RGB")
    img = fit_image(src, 1024 if args.sdxl else args.res)
    print(f"image {img.size[0]}x{img.size[1]}  device={device}")

    t0 = time.time()
    labels = segment(img, device)
    found = sorted({LABELS[i] for i in np.unique(labels) if i in PARTS["full"]})
    print(f"segmented in {time.time() - t0:.1f}s, clothes found: {found or 'none'}")

    masks = []
    if len(passes) > 1 or passes[0][0] != "full":
        labels = split_dress(labels)
    for parts_name, text in passes:
        parts = set(PARTS[parts_name])
        protect = set(PROTECT)
        # bare arms/legs: repaint them (sleeves, trousers) or keep hands/feet exactly as they are
        for flag, group in ((args.cover_arms, ARMS), (args.cover_legs, LEGS)):
            if flag:
                parts |= group
            else:
                protect |= group
        neckline = parts_name != "lower" and not args.no_neckline
        mask = build_mask(labels, parts, args.grow, neckline, protect)
        coverage = np.asarray(mask).mean() / 255
        if coverage < 0.005:
            raise SystemExit(f"'{parts_name}' mask covers {coverage:.1%} of the image - nothing to repaint. "
                             f"Clothes found: {found or 'none'}. Try --parts full.")
        print(f"pass '{parts_name}': mask covers {coverage:.1%}  <- {text}")
        masks.append(mask)
    union = Image.fromarray(np.maximum.reduce([np.asarray(m) for m in masks]), "L")
    mask_path = save(union, args.out / f"{stem}_mask.png")
    print(f"mask -> {mask_path}")
    if args.mask_only:
        return

    pipe = load_pipe(args.model or (SDXL_INPAINT if args.sdxl else SD15_INPAINT), args.sdxl, device)
    subject = "a person"
    if args.lora:
        pipe.load_lora_weights(str(args.lora))
        pipe.fuse_lora(lora_scale=args.lora_scale)
        token_file = args.lora / "token.txt"
        subject = token_file.read_text().strip() if token_file.exists() else subject

    def prompt_for(text: str) -> str:
        return (f"{subject} wearing {text}, full body photo, detailed fabric texture, "
                f"natural lighting, photorealistic, sharp focus, high detail")

    base_seed = args.seed if args.seed is not None else int(torch.seed() % 2**31)

    def feather(m: Image.Image, side: int) -> Image.Image:
        """Outward-only feather: 100% new pixels inside the mask, soft falloff outside it.
        Blurring inward too would bleed the old garment's edge into the new one."""
        return ImageChops.lighter(m.filter(ImageFilter.GaussianBlur(side / 64)), m)

    # refine pass runs at a higher resolution on an upscaled copy of the first pass
    hires = args.hires if (args.hires and not args.sdxl and args.hires > max(img.size)) else 0
    if hires:
        img_hi = fit_image(src, hires)
        union_hi = union.resize(img_hi.size, Image.BILINEAR)
        soft_hi = feather(union_hi, hires)

    def run(image, mask_img, text, strength, seed):
        # a blurred mask makes the pipeline's own paste-back soft instead of a hard seam
        soft_in = pipe.mask_processor.blur(mask_img, blur_factor=max(image.size) // 64)
        out = pipe(
            prompt=prompt_for(text), negative_prompt=args.negative, image=image, mask_image=soft_in,
            width=image.size[0], height=image.size[1],
            num_inference_steps=args.steps, guidance_scale=args.guidance,
            generator=torch.Generator(device="cpu").manual_seed(seed), strength=strength,
            # crop to the clothes region so the garment is rendered at full resolution
            padding_mask_crop=32,
        )
        flagged = getattr(out, "nsfw_content_detected", None)
        return None if flagged and flagged[0] else out.images[0].resize(image.size)

    seed = base_seed
    for i in range(args.num):
        t0 = time.time()
        for attempt in range(3):
            cur = img
            for (parts_name, text), mask in zip(passes, masks):
                result = run(cur, mask, text, 1.0, seed)
                if result is None:
                    break
                cur = Image.composite(result, cur, feather(mask, args.res))
            else:
                break  # every pass succeeded
            # the SD1.5 safety checker misfires on ordinary photos of people and returns a
            # black image; bump the seed and try again instead of writing black clothes
            print(f"    seed {seed} flagged by safety checker, retrying with seed {seed + 1}")
            seed += 1
        else:
            print(f"[{i + 1}/{args.num}] skipped: 3 seeds in a row flagged")
            continue
        final = cur
        if hires:
            up = final.resize(img_hi.size, Image.LANCZOS)
            refined = run(up, union_hi, ", ".join(t for _, t in passes), args.refine_strength, seed) or up
            final = Image.composite(refined, img_hi, soft_hi)
        path = save(final, args.out / f"{stem}_{i}_seed{seed}.png")
        print(f"[{i + 1}/{args.num}] {time.time() - t0:.1f}s  seed={seed}  {final.size[0]}x{final.size[1]}  -> {path}")
        seed += 1


if __name__ == "__main__":
    main()
