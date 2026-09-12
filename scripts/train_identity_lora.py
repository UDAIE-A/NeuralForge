"""Teach Stable Diffusion one person's identity from a handful of photos (DreamBooth-style LoRA).

Only your own photos, or photos of people who have agreed to this.

  venv/Scripts/python.exe scripts/train_identity_lora.py --images "C:/photos/me" --name me
  venv/Scripts/python.exe scripts/train_identity_lora.py --name me --sample-only --prompt "photo of ohwx person on a beach"

What it does:
  1. Prep: finds the person / face in each photo (SegFormer clothes parser), cuts a
     person-centred crop and a face close-up, resizes to --res. Crops are written to
     <out>/train_crops so you can see exactly what was trained on.
  2. Encodes crops (+ horizontal flips) to VAE latents once; the VAE, text encoder and
     base UNet stay frozen. Only LoRA adapters (rank --rank) on the UNet attention
     layers are trained, so this fits comfortably on a 12 GB card at 512 px.
  3. Saves <out>/pytorch_lora_weights.safetensors and renders sample images.

Use the result:
  - text-to-image:   this script with --sample-only
  - change clothes:  scripts/change_clothes.py photo.jpg "a suit" --lora checkpoints/lora/me
"""

from __future__ import annotations

import argparse
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent.parent
CKPT = ROOT / "checkpoints"
BASE = CKPT / "sd15-base"
SEG = CKPT / "segformer-clothes"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
FACE, BACKGROUND = 11, 0


# ----------------------------------------------------------------------------- prep

def square_crop(img: Image.Image, box: tuple[int, int, int, int], margin: float, res: int) -> Image.Image:
    """Square crop centred on `box`, grown by `margin`, clamped to the image, resized to res."""
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    side = max(x1 - x0, y1 - y0) * margin
    side = min(side, min(img.size))
    left = int(min(max(cx - side / 2, 0), img.size[0] - side))
    top = int(min(max(cy - side / 2, 0), img.size[1] - side))
    return img.crop((left, top, left + int(side), top + int(side))).resize((res, res), Image.LANCZOS)


def bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def prepare_crops(images: list[Path], res: int, token: str, out: Path, device: str, min_face: int):
    from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

    proc = AutoImageProcessor.from_pretrained(str(SEG))
    seg = AutoModelForSemanticSegmentation.from_pretrained(str(SEG)).to(device).eval()
    out.mkdir(parents=True, exist_ok=True)
    crops: list[tuple[Image.Image, str]] = []
    for path in images:
        img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        with torch.no_grad():
            logits = seg(**proc(images=img, return_tensors="pt").to(device)).logits
        labels = F.interpolate(logits, size=img.size[::-1], mode="bilinear", align_corners=False).argmax(1)[0].cpu().numpy()
        person = bbox(labels != BACKGROUND)
        face = bbox(labels == FACE)
        made = []
        if person:
            crops.append((square_crop(img, person, 1.15, res), f"photo of {token}"))
            made.append("person")
        if face and (face[3] - face[1]) >= min_face:
            crops.append((square_crop(img, face, 2.4, res), f"close-up photo of {token}, face"))
            made.append("face")
        if not person:
            crops.append((ImageOps.fit(img, (res, res), Image.LANCZOS), f"photo of {token}"))
            made.append("center")
        print(f"  {path.name[:40]:40s} {img.size[0]}x{img.size[1]}  -> {', '.join(made)}")
    for i, (im, cap) in enumerate(crops):
        im.save(out / f"{i:03d}.jpg", quality=95)
        (out / f"{i:03d}.txt").write_text(cap)
    del seg
    torch.cuda.empty_cache()
    return crops


# ----------------------------------------------------------------------------- train

def encode(crops, vae, tokenizer, text_encoder, device):
    """VAE latents for every crop and its mirror, plus the text embedding of its caption."""
    latents, embeds = [], []
    with torch.no_grad():
        for im, cap in crops:
            ids = tokenizer(cap, padding="max_length", max_length=tokenizer.model_max_length,
                            truncation=True, return_tensors="pt").input_ids.to(device)
            emb = text_encoder(ids)[0]
            for flip in (False, True):
                x = ImageOps.mirror(im) if flip else im
                t = torch.from_numpy(np.asarray(x, dtype=np.float32) / 127.5 - 1).permute(2, 0, 1)[None].to(device, vae.dtype)
                lat = vae.encode(t).latent_dist.sample() * vae.config.scaling_factor
                latents.append(lat.cpu())
                embeds.append(emb.cpu())
    return torch.cat(latents), torch.cat(embeds)


def train(args, device):
    from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
    from diffusers.training_utils import cast_training_params
    from peft import LoraConfig
    from transformers import CLIPTextModel, CLIPTokenizer

    images = sorted(p for p in args.images.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not images:
        raise SystemExit(f"no images in {args.images}")
    print(f"{len(images)} photos in {args.images}")
    crops = prepare_crops(images, args.res, args.token, args.out / "train_crops", device, args.min_face)
    print(f"{len(crops)} training crops (x2 with flips) -> {args.out / 'train_crops'}")

    dtype = torch.float16
    tokenizer = CLIPTokenizer.from_pretrained(BASE, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(BASE, subfolder="text_encoder", torch_dtype=dtype).to(device)
    vae = AutoencoderKL.from_pretrained(BASE, subfolder="vae", torch_dtype=dtype).to(device)
    latents, embeds = encode(crops, vae, tokenizer, text_encoder, device)
    del vae, text_encoder
    torch.cuda.empty_cache()

    unet = UNet2DConditionModel.from_pretrained(BASE, subfolder="unet", torch_dtype=dtype).to(device)
    unet.requires_grad_(False)
    unet.enable_gradient_checkpointing()
    unet.add_adapter(LoraConfig(r=args.rank, lora_alpha=args.rank, init_lora_weights="gaussian",
                                target_modules=["to_k", "to_q", "to_v", "to_out.0"]))
    cast_training_params(unet, dtype=torch.float32)  # LoRA weights in fp32, base stays fp16
    params = [p for p in unet.parameters() if p.requires_grad]
    print(f"trainable LoRA params: {sum(p.numel() for p in params) / 1e6:.2f}M  rank={args.rank}")

    sched = DDPMScheduler.from_pretrained(BASE, subfolder="scheduler")
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-2)
    warm = max(1, args.steps // 20)
    lr_at = lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / args.steps)))
    lr_sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    scaler = torch.amp.GradScaler("cuda")

    n = latents.shape[0]
    order: list[int] = []
    unet.train()
    t0 = time.time()
    running = 0.0
    for step in range(args.steps):
        if not order:
            order = random.sample(range(n), n)
        idx = [order.pop() for _ in range(min(args.batch, len(order)))]
        x0 = latents[idx].to(device, dtype)
        cond = embeds[idx].to(device, dtype)
        noise = torch.randn_like(x0)
        t = torch.randint(0, sched.config.num_train_timesteps, (x0.shape[0],), device=device)
        xt = sched.add_noise(x0, noise, t)
        target = noise if sched.config.prediction_type == "epsilon" else sched.get_velocity(x0, noise, t)
        with torch.autocast("cuda", dtype=dtype):
            pred = unet(xt, t, encoder_hidden_states=cond).sample
        loss = F.mse_loss(pred.float(), target.float())
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        lr_sched.step()
        running = loss.item() if step == 0 else 0.98 * running + 0.02 * loss.item()
        if (step + 1) % 50 == 0 or step == 0:
            el = time.time() - t0
            print(f"step {step + 1:5d}/{args.steps}  loss {running:.4f}  lr {lr_sched.get_last_lr()[0]:.2e}  "
                  f"{el:5.0f}s  vram {torch.cuda.max_memory_allocated() / 2**30:.1f}G")

    save_lora(unet, args.out)
    (args.out / "token.txt").write_text(args.token)
    print(f"saved LoRA -> {args.out}  ({time.time() - t0:.0f}s)")
    del unet
    torch.cuda.empty_cache()


def save_lora(unet, out: Path):
    from diffusers import StableDiffusionPipeline
    from diffusers.utils import convert_state_dict_to_diffusers
    from peft.utils import get_peft_model_state_dict

    out.mkdir(parents=True, exist_ok=True)
    sd = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
    StableDiffusionPipeline.save_lora_weights(out, unet_lora_layers=sd, safe_serialization=True)


# ----------------------------------------------------------------------------- sample

DEFAULT_PROMPTS = [
    "photo of {t}, smiling, outdoors, natural light, sharp focus",
    "close-up portrait photo of {t}, studio lighting, 85mm",
    "photo of {t} wearing a black leather jacket, city street at night",
    "photo of {t} wearing a traditional red saree, wedding, bokeh",
]
NEGATIVE = "deformed, disfigured, bad anatomy, extra limbs, blurry, low quality, watermark, text, cartoon, painting"


def sample(args, device):
    from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline

    token = (args.out / "token.txt").read_text().strip() if (args.out / "token.txt").exists() else args.token
    pipe = StableDiffusionPipeline.from_pretrained(BASE, torch_dtype=torch.float16, safety_checker=None).to(device)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config, algorithm_type="dpmsolver++", use_karras_sigmas=True, final_sigmas_type="sigma_min")
    pipe.load_lora_weights(str(args.out))
    pipe.fuse_lora(lora_scale=args.lora_scale)
    pipe.set_progress_bar_config(leave=False)
    prompts = [args.prompt] if args.prompt else [p.format(t=token) for p in DEFAULT_PROMPTS]
    sdir = args.out / "samples"
    sdir.mkdir(exist_ok=True)
    seed = args.seed if args.seed is not None else random.randrange(2**31)
    for i, prompt in enumerate(prompts):
        for j in range(args.num):
            g = torch.Generator("cpu").manual_seed(seed + j)
            im = pipe(prompt, negative_prompt=NEGATIVE, num_inference_steps=25, guidance_scale=6.0,
                      width=args.res, height=args.res, generator=g).images[0]
            path = sdir / f"{i}_{j}_seed{seed + j}.png"
            im.save(path)
            print(f"  {path.name}  <- {prompt}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", type=Path, help="folder of photos of ONE person")
    ap.add_argument("--name", required=True, help="output name -> checkpoints/lora/<name>")
    ap.add_argument("--token", default="ohwx person", help="rare token to bind the identity to")
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--min-face", type=int, default=48, help="min face height (px) to also make a face crop")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--sample-only", action="store_true", help="skip training, just render with the saved LoRA")
    ap.add_argument("--prompt", default=None, help="single prompt to render (default: 4 built-in prompts)")
    ap.add_argument("--num", type=int, default=2, help="images per prompt")
    ap.add_argument("--lora-scale", type=float, default=0.9)
    args = ap.parse_args()
    args.out = CKPT / "lora" / args.name
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    if not args.sample_only:
        if not args.images:
            ap.error("--images is required unless --sample-only")
        train(args, device)
    sample(args, device)


if __name__ == "__main__":
    main()
