"""Change the clothes in a photo of a person from a text prompt. No training.

Pipeline:
  1. Segment the clothing (SegFormer trained on fashion parsing) -> mask. For upper-body
     changes the mask also covers the neck/shoulders down to the old neckline, so the new
     top can have any collar, straps or sleeves (face and hair are always protected).
  2. Inpaint the masked region with Stable Diffusion (DPM++ 2M Karras), cropped to the
     clothes so the garment is rendered at full resolution. Two ControlNets guide it:
     a depth map of the photo (body volume and proportions come from the person, not
     from the model's imagination) and an edge map over skin/jewelry areas only (necklace,
     collarbones, arms are redrawn where they are). Hands found by a hand detector are
     never repainted, even with --cover-arms.
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
CN_DEPTH = local_or_hub(CKPT / "controlnet-depth", "lllyasviel/control_v11f1p_sd15_depth")
CN_CANNY = local_or_hub(CKPT / "controlnet-canny", "lllyasviel/control_v11p_sd15_canny")
DEPTH_MODEL = local_or_hub(CKPT / "depth-anything-small", "depth-anything/Depth-Anything-V2-Small-hf")
HAND_MODEL = CKPT / "hand_landmarker.task"  # scripts/download_image_models.py fetches it

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


def depth_map(img: Image.Image, old_clothes: np.ndarray, device: str) -> Image.Image:
    """Depth Anything on the photo. Inside the old garment the map is blurred heavily: the
    coarse body volume is kept (proportions), but the garment's own folds and outline are
    not, so the new outfit is free to hang differently."""
    from transformers import pipeline

    est = pipeline("depth-estimation", model=DEPTH_MODEL, device=0 if device == "cuda" else -1)
    depth = est(img)["depth"].convert("L").resize(img.size)
    del est
    torch.cuda.empty_cache()
    coarse = depth.filter(ImageFilter.GaussianBlur(max(img.size) / 40))
    depth = Image.composite(coarse, depth, Image.fromarray(old_clothes.astype(np.uint8) * 255))
    return depth.convert("RGB")


def edge_map(img: Image.Image, where: np.ndarray) -> Image.Image:
    """Canny edges of the photo, kept only where `where` is True (skin/jewelry regions that
    get repainted). Edges of the old garment must not be passed or its outline comes back."""
    import cv2

    edges = cv2.Canny(cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY), 80, 180)
    edges[~where] = 0
    return Image.fromarray(edges).convert("RGB")


def hand_mask(img: Image.Image, grow: int) -> np.ndarray:
    """Convex hull around every detected hand (MediaPipe hand landmarker), padded by `grow`."""
    if not HAND_MODEL.exists():
        return np.zeros((img.size[1], img.size[0]), bool)
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python as mpp
    from mediapipe.tasks.python import vision

    opts = vision.HandLandmarkerOptions(base_options=mpp.BaseOptions(model_asset_path=str(HAND_MODEL)),
                                        num_hands=2, min_hand_detection_confidence=0.3)
    res = vision.HandLandmarker.create_from_options(opts).detect(
        mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(np.asarray(img))))
    w, h = img.size
    m = np.zeros((h, w), np.uint8)
    for hand in res.hand_landmarks:
        pts = np.array([[int(l.x * w), int(l.y * h)] for l in hand], np.int32)
        cv2.fillConvexPoly(m, cv2.convexHull(pts), 255)
    if grow > 0 and m.any():
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1)))
    return m.astype(bool)


def build_mask(labels: np.ndarray, parts: set[int], grow: int, neckline: bool, protect: set[int],
               hands: np.ndarray | None = None) -> Image.Image:
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
    if hands is not None:
        keep |= hands  # SD1.5 cannot draw hands; keep the real ones whatever else is repainted
    keep = keep.astype(np.uint8)
    e = max(3, grow // 2)  # segmenter boundaries are fuzzy by ~8 px at 768
    keep = cv2.erode(keep, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * e + 1, 2 * e + 1)))
    mask[keep.astype(bool)] = 0
    return Image.fromarray(mask, "L")


def load_pipe(model: str, sdxl: bool, control: bool, device: str):
    from diffusers import (AutoPipelineForInpainting, ControlNetModel, DPMSolverMultistepScheduler,
                           StableDiffusionControlNetInpaintPipeline)

    # SD1.5's safety checker runs on the cropped garment region and rejects ordinary clothed
    # torsos most of the time (returning a black image), so it is disabled.
    common = dict(torch_dtype=torch.float16, safety_checker=None, requires_safety_checker=False)
    if control:
        nets = [ControlNetModel.from_pretrained(m, torch_dtype=torch.float16, variant="fp16") for m in (CN_DEPTH, CN_CANNY)]
        pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(model, controlnet=nets, **common)
    else:
        pipe = AutoPipelineForInpainting.from_pretrained(model, variant="fp16" if sdxl else None, **common)
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
    ap.add_argument("--no-control", action="store_true", help="skip the depth/edge ControlNets (faster, less faithful body)")
    ap.add_argument("--depth-scale", type=float, default=0.7, help="ControlNet weight for the depth map")
    ap.add_argument("--edge-scale", type=float, default=0.6, help="ControlNet weight for the skin/jewelry edge map")
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

    control = not args.no_control and not args.sdxl
    # hands: keep only what the segmenter also calls skin, so fabric between the fingers is repainted
    hands = hand_mask(img, args.grow // 2) & ~np.isin(labels, list(PARTS["full"]))
    if hands.any():
        print(f"hands detected, protected ({hands.mean():.1%} of the image)")
    masks, kept = [], []
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
        mask = build_mask(labels, parts, args.grow, neckline, protect, hands)
        coverage = np.asarray(mask).mean() / 255
        if coverage < 0.005:
            # e.g. --lower on a half-body shot: nothing there, carry on with the other passes
            print(f"pass '{parts_name}': skipped, only {coverage:.1%} of the image is {parts_name} clothing "
                  f"(found: {', '.join(found) or 'none'})")
            continue
        print(f"pass '{parts_name}': mask covers {coverage:.1%}  <- {text}")
        kept.append((parts_name, text))
        masks.append(mask)
    passes = kept
    if not masks:
        raise SystemExit(f"nothing to repaint - clothes found: {', '.join(found) or 'none'}. Try --parts full.")
    union = Image.fromarray(np.maximum.reduce([np.asarray(m) for m in masks]), "L")
    mask_path = save(union, args.out / f"{stem}_mask.png")
    print(f"mask -> {mask_path}")
    if args.mask_only:
        return

    guides = {}
    if control:
        old_clothes = np.isin(labels, list(PARTS["full"]))
        # edges only from neck/collarbone/jewelry (the segmenter's 'face' label) inside the mask -
        # never the old garment, background, or the thin strip beside arms/legs, where an outline
        # edge makes the model draw a fabric hem along the limb
        skin = (np.asarray(union) > 0) & (labels == FACE)
        guides[img.size] = [depth_map(img, old_clothes, device), edge_map(img, skin)]
        save(guides[img.size][0], args.out / f"{stem}_depth.png")
        save(guides[img.size][1], args.out / f"{stem}_edges.png")

    pipe = load_pipe(args.model or (SDXL_INPAINT if args.sdxl else SD15_INPAINT), args.sdxl, control, device)
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
        if control:
            guides[img_hi.size] = [g.resize(img_hi.size, Image.BILINEAR) for g in guides[img.size]]

    def run(image, mask_img, text, strength, seed):
        """Inpaint `mask_img` on `image`. The mask's bounding box (+padding) is cut out and
        rendered at the working resolution so the garment gets the full pixel budget, then
        pasted back through a blurred mask so there is no hard seam."""
        side = max(image.size)
        x0, y0, x1, y1 = pipe.mask_processor.get_crop_region(mask_img, *image.size, pad=side // 24)
        cw, ch = x1 - x0, y1 - y0
        scale = side / max(cw, ch)
        rw, rh = int(cw * scale) // 8 * 8, int(ch * scale) // 8 * 8
        box = (x0, y0, x1, y1)
        crop = image.crop(box).resize((rw, rh), Image.LANCZOS)
        mcrop = mask_img.crop(box).resize((rw, rh), Image.BILINEAR)
        soft_in = pipe.mask_processor.blur(mcrop, blur_factor=side // 64)
        extra = {}
        if control:
            extra = dict(control_image=[g.crop(box).resize((rw, rh), Image.BILINEAR) for g in guides[image.size]],
                         controlnet_conditioning_scale=[args.depth_scale, args.edge_scale])
        out = pipe(
            prompt=prompt_for(text), negative_prompt=args.negative, image=crop, mask_image=soft_in, **extra,
            width=rw, height=rh, num_inference_steps=args.steps, guidance_scale=args.guidance,
            generator=torch.Generator(device="cpu").manual_seed(seed), strength=strength,
        ).images[0]
        result = image.copy()
        result.paste(Image.composite(out, crop, soft_in).resize((cw, ch), Image.LANCZOS), box)
        return result

    seed = base_seed
    for i in range(args.num):
        t0 = time.time()
        cur = img
        for (parts_name, text), mask in zip(passes, masks):
            result = run(cur, mask, text, 1.0, seed)
            cur = Image.composite(result, cur, feather(mask, args.res))
        final = cur
        if hires:
            up = final.resize(img_hi.size, Image.LANCZOS)
            refined = run(up, union_hi, ", ".join(t for _, t in passes), args.refine_strength, seed)
            final = Image.composite(refined, img_hi, soft_hi)
        path = save(final, args.out / f"{stem}_{i}_seed{seed}.png")
        print(f"[{i + 1}/{args.num}] {time.time() - t0:.1f}s  seed={seed}  {final.size[0]}x{final.size[1]}  -> {path}")
        seed += 1


if __name__ == "__main__":
    main()
