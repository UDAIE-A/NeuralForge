"""Full-resolution, edge-accurate person masks: SegFormer for WHAT, SAM 2.1 for WHERE.

The clothes parser (mattmdjaga/segformer_b2_clothes) knows which region is a dress, an arm or
hair, but it only ever sees the photo squashed to 512x512. On a 2274x2880 portrait that is a
~5x downscale plus an aspect-ratio distortion, so every boundary is a staircase of 5-px blocks
and thin parts (straps, fingers) are lost or mislabelled.

This module keeps SegFormer's labels and redraws every boundary:

  1. SegFormer runs on a letterboxed (undistorted) copy; its logits are upsampled to full res.
  2. Each region is re-segmented by SAM 2.1 on a crop around it, prompted with points taken
     from the coarse region (positives) and from its neighbours (negatives). Small regions get
     a zoomed-in crop, so hands and straps are traced at better than original resolution.
     Hands are prompted with their 21 MediaPipe landmarks.
  3. Every region's soft mask is snapped to the photo's own edges with a guided filter at full
     resolution, then the label map is the per-pixel argmax.

edge_alignment() scores a label map by how much of its boundary lies on a real image edge,
so coarse and refined results can be compared with a number instead of by eye.

SAM 2.1 (facebook/sam2.1-hiera-large) is Apache-2.0.
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
CKPT = ROOT / "checkpoints"
SAM_MODEL = CKPT / "sam2.1-large"
SKIN = 18  # extra label: exposed neck/chest skin (SegFormer folds it into 'face')
APPAREL = 19  # extra label: Sapiens' 'Apparel' (accessories / other clothing)
NUM_LABELS = 20
SAPIENS_MODEL = CKPT / "sapiens-seg-1b" / "sapiens_1b_goliath_best_goliath_mIoU_7994_epoch_151_torchscript.pt2"

# Sapiens (Goliath, 28 classes) -> this project's label ids (SegFormer's, plus SKIN/APPAREL).
# Upper/lower arm and the hand fold into one arm label so downstream logic is unchanged;
# the hands are also returned separately. Lips/teeth/tongue are face.
SAPIENS_CLASSES = [
    "Background", "Apparel", "Face_Neck", "Hair", "Left_Foot", "Left_Hand", "Left_Lower_Arm",
    "Left_Lower_Leg", "Left_Shoe", "Left_Sock", "Left_Upper_Arm", "Left_Upper_Leg", "Lower_Clothing",
    "Right_Foot", "Right_Hand", "Right_Lower_Arm", "Right_Lower_Leg", "Right_Shoe", "Right_Sock",
    "Right_Upper_Arm", "Right_Upper_Leg", "Torso", "Upper_Clothing", "Lower_Lip", "Upper_Lip",
    "Lower_Teeth", "Upper_Teeth", "Tongue"]
SAPIENS_TO_LABEL = [0, APPAREL, 11, 2, 12, 14, 14, 12, 9, 12, 14, 12, 6, 13, 15, 15, 13, 10, 13,
                    15, 13, SKIN, 4, 11, 11, 11, 11, 11]
SAPIENS_HANDS = (5, 14)

HAND_TIPS = [4, 8, 12, 16, 20]
HAND_KNUCKLES = [2, 5, 9, 13, 17]


# ----------------------------------------------------------------------------- SegFormer
def segformer_probs(img: Image.Image, seg_model: str, device: str, side: int = 512) -> np.ndarray:
    """Class probabilities (C, H, W) at the image's FULL resolution, from an undistorted pass.

    The image is padded to a square with its own border colour before the processor's 512x512
    resize, so the person keeps their real proportions; the padding is cropped off the logits.
    """
    from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

    w, h = img.size
    s = max(w, h)
    border = np.concatenate([np.asarray(img)[0], np.asarray(img)[-1]]).reshape(-1, 3)
    pad = Image.new("RGB", (s, s), tuple(int(v) for v in np.median(border, 0)))
    ox, oy = (s - w) // 2, (s - h) // 2
    pad.paste(img, (ox, oy))

    proc = AutoImageProcessor.from_pretrained(seg_model)
    model = AutoModelForSemanticSegmentation.from_pretrained(seg_model).to(device).eval()
    with torch.no_grad():
        inputs = proc(images=pad.resize((side, side), Image.BICUBIC), return_tensors="pt",
                      do_resize=False).to(device)
        logits = model(**inputs).logits.float()                       # (1, C, side/4, side/4)
    del model
    # crop the letterbox off at logit resolution, then upsample to the true image size
    f = logits.shape[-1] / s
    x0, y0 = int(round(ox * f)), int(round(oy * f))
    x1, y1 = int(round((ox + w) * f)), int(round((oy + h) * f))
    logits = logits[..., y0:y1, x0:x1]
    probs = torch.nn.functional.interpolate(logits, size=(h, w), mode="bilinear",
                                            align_corners=False).softmax(1)[0]
    return probs.half().cpu().numpy()  # (C, H, W) - float16 halves a ~470 MB array


# ----------------------------------------------------------------------------- Sapiens
def sapiens_probs(img: Image.Image, device: str) -> tuple[np.ndarray, np.ndarray]:
    """Sapiens-1B body-part probabilities mapped to this project's labels, at full resolution.

    Returns (probs (NUM_LABELS, H, W) float16, hand probability (H, W) float16).
    LICENSE: Sapiens is CC-BY-NC-4.0 - non-commercial use only.
    The photo is letterboxed into Sapiens' 768x1024 input (no aspect distortion); the model
    is freed from the GPU before returning so SAM never shares memory with it.
    """
    w, h = img.size
    s = min(768 / w, 1024 / h)
    nw, nh = round(w * s), round(h * s)
    ox, oy = (768 - nw) // 2, (1024 - nh) // 2
    canvas = Image.new("RGB", (768, 1024))
    canvas.paste(img.resize((nw, nh), Image.BICUBIC), (ox, oy))
    x = torch.from_numpy(np.array(canvas)).permute(2, 0, 1).float()
    mean = torch.tensor([123.675, 116.28, 103.53])[:, None, None]
    std = torch.tensor([58.395, 57.12, 57.375])[:, None, None]
    x = ((x - mean) / std)[None].to(device)

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)   # torch.jit.load deprecation notice
        model = torch.jit.load(str(SAPIENS_MODEL), map_location=device).eval()
    with torch.no_grad():
        logits = model(x).float()                          # (1, 28, 512, 384)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    logits = torch.nn.functional.interpolate(logits, size=(1024, 768), mode="bilinear",
                                             align_corners=False)[..., oy:oy + nh, ox:ox + nw]
    p28 = torch.nn.functional.interpolate(logits, size=(h, w), mode="bilinear",
                                          align_corners=False).softmax(1)[0].cpu()
    probs = torch.zeros((NUM_LABELS, h, w), dtype=torch.float32)
    probs.index_add_(0, torch.tensor(SAPIENS_TO_LABEL), p28)
    hands = p28[list(SAPIENS_HANDS)].sum(0)
    return probs.half().numpy(), hands.half().numpy()


# ----------------------------------------------------------------------------- prompts
def interior_points(mask: np.ndarray, k: int) -> list[tuple[int, int]]:
    """Up to k well-spread points deep inside `mask` (farthest-point sampling)."""
    if not mask.any():
        return []
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    dmax = dist.max()
    ys, xs = np.nonzero(dist >= max(1.0, 0.5 * dmax))
    if len(xs) == 0:
        return []
    cand = np.stack([xs, ys], 1).astype(np.float32)
    if len(cand) > 4000:
        cand = cand[np.random.default_rng(0).choice(len(cand), 4000, replace=False)]
    first = np.unravel_index(int(dist.argmax()), dist.shape)
    pts = [np.array([first[1], first[0]], np.float32)]
    while len(pts) < k:
        d = np.min([np.linalg.norm(cand - p, axis=1) for p in pts], axis=0)
        if d.max() < 0.25 * dmax:
            break
        pts.append(cand[int(d.argmax())])
    return [(int(x), int(y)) for x, y in pts]


def pad_box(box, pad: float, w: int, h: int, min_pad: int = 24):
    x0, y0, x1, y1 = box
    px, py = max(min_pad, int((x1 - x0) * pad)), max(min_pad, int((y1 - y0) * pad))
    return max(0, x0 - px), max(0, y0 - py), min(w, x1 + px + 1), min(h, y1 + py + 1)


def mask_box(m: np.ndarray):
    ys, xs = np.nonzero(m)
    return None if len(xs) == 0 else (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))


# ----------------------------------------------------------------------------- SAM
class Sam:
    def __init__(self, device: str, model_dir: Path = SAM_MODEL):
        from transformers import Sam2Model, Sam2Processor
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()  # the checkpoint is tagged sam2_video; Sam2Model loads it fine
        self.proc = Sam2Processor.from_pretrained(model_dir)
        self.model = Sam2Model.from_pretrained(model_dir).to(device).eval()
        self.device = device
        self.calls = 0

    @torch.no_grad()
    def segment(self, photo: Image.Image, crop_box, pos, neg, ref: np.ndarray | None = None,
                use_box: bool = True) -> np.ndarray:
        """SAM logits for the crop (crop-sized float array). Points are in PHOTO coordinates.

        Of SAM's three candidate masks, the one agreeing best with `ref` (the coarse region,
        crop-sized) is kept; without a reference, the one SAM itself scores highest.
        """
        x0, y0, x1, y1 = crop_box
        crop = photo.crop(crop_box)
        pts = [[x - x0, y - y0] for x, y in pos + neg]
        lbl = [1] * len(pos) + [0] * len(neg)
        kw = {"input_points": [[pts]], "input_labels": [[lbl]]}
        if use_box and ref is not None and ref.any():
            bx = mask_box(ref)
            kw["input_boxes"] = [[[bx[0], bx[1], bx[2], bx[3]]]]
        inp = self.proc(images=crop, return_tensors="pt", **kw).to(self.device)
        out = self.model(**inp, multimask_output=True)
        logits = self.proc.post_process_masks(out.pred_masks.cpu(), inp["original_sizes"].cpu(),
                                              binarize=False)[0][0].float().numpy()  # (3, h, w)
        self.calls += 1
        if ref is not None and ref.any():
            ious = [((l > 0) & ref).sum() / max(1, ((l > 0) | ref).sum()) for l in logits]
            best = int(np.argmax(ious))
        else:
            best = int(out.iou_scores[0, 0].argmax())
        return logits[best]


# ----------------------------------------------------------------------------- edges
def guided_snap(soft: np.ndarray, guide: np.ndarray, radius: int, eps: float = 1e-4) -> np.ndarray:
    """Edge-aware smoothing of a soft mask: boundaries move onto the photo's own edges."""
    return cv2.ximgproc.guidedFilter(guide, soft.astype(np.float32), radius, eps)


def edge_alignment(labels: np.ndarray, gray: np.ndarray, tol: int = 2) -> float:
    """Share of label-map boundary pixels lying within `tol` px of a Canny edge of the photo."""
    b = np.zeros(labels.shape, bool)
    b[:-1] |= labels[:-1] != labels[1:]
    b[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    if not b.any():
        return float("nan")
    med = float(np.median(gray))
    edges = cv2.Canny(gray, int(max(0, 0.5 * med)), int(min(255, 1.2 * med)))
    near = cv2.dilate(edges, np.ones((2 * tol + 1, 2 * tol + 1), np.uint8)) > 0
    return float((near & b).sum() / b.sum())


GARMENT_IDS = {1, 3, 4, 5, 6, 7, 8, 9, 10, 16, 17, APPAREL}


def _crop_around(piece: np.ndarray, margin: int = 3):
    ys, xs = np.nonzero(piece)
    h, w = piece.shape
    return (slice(max(0, ys.min() - margin), min(h, ys.max() + margin + 1)),
            slice(max(0, xs.min() - margin), min(w, xs.max() + margin + 1)))


def ring_majority(labels: np.ndarray, piece: np.ndarray, prefer: set[int] | None = None) -> int:
    """Most common label just outside `piece` (a garment from `prefer` wins if any touches)."""
    win = _crop_around(piece)            # work on a crop: pieces are small, the photo is not
    p = piece[win]
    ring = cv2.dilate(p.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) & ~p
    around = labels[win][ring]
    if prefer:
        pref = around[np.isin(around, list(prefer))]
        if len(pref):
            return int(np.bincount(pref).argmax())
    return int(np.bincount(around).argmax()) if len(around) else 0


def fill_enclosed(labels: np.ndarray, photo: Image.Image, reach: int) -> tuple[np.ndarray, int]:
    """Background pixels hemmed in by the person that do not look like the backdrop.

    A thin strap the segmenter lost entirely comes out as background, yet it sits between
    skin, arm, hair and dress - often touching the wall only through a sliver behind the hair,
    so it is not strictly enclosed. The test is therefore: colour unlike the backdrop, AND at
    least 60% of the piece's border is person. A shadow on the wall beside the body fails the
    second test (its border is mostly wall); a gap of wall between arm and waist fails the first.
    Such pieces go to the garment they touch, or else their most common neighbour.
    """
    from scipy.ndimage import binary_fill_holes

    inside = labels > 0
    lab = cv2.cvtColor(np.asarray(photo), cv2.COLOR_RGB2LAB).astype(np.float32)
    backdrop = lab[~binary_fill_holes(inside)]
    bg = np.median(backdrop, 0) if len(backdrop) else np.array([255, 128, 128], np.float32)
    near = cv2.dilate(inside.astype(np.uint8), np.ones((2 * reach + 1,) * 2, np.uint8)) > 0
    unlike = np.linalg.norm(lab - bg, axis=2) > 25
    cand = (~inside) & near & unlike
    n, comp, stats, _ = cv2.connectedComponentsWithStats(cand.astype(np.uint8))
    out = labels.copy()
    filled = 0
    H, W = labels.shape
    for c in range(1, n):
        x, y, w, h = stats[c, :4]
        win = (slice(max(0, y - 3), min(H, y + h + 3)), slice(max(0, x - 3), min(W, x + w + 3)))
        piece = comp[win] == c
        ring = cv2.dilate(piece.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) & ~piece
        if ring.any() and inside[win][ring].mean() >= 0.6:
            sub = out[win]
            sub[piece] = ring_majority(labels[win], piece, GARMENT_IDS)
            filled += 1

    # Fully enclosed holes the global colour test let through: soft, desaturated skin can sit
    # close to a grey-blue backdrop. Such a hole belongs to the person when its colour is
    # nearer the person around it than the backdrop nearest to it; a real gap (sea showing
    # between hand and chest) is nearer the backdrop and stays.
    still = out > 0
    holes = binary_fill_holes(still) & ~still
    n, comp, stats, _ = cv2.connectedComponentsWithStats(holes.astype(np.uint8))
    outside = ~binary_fill_holes(still)
    for c in range(1, n):
        x, y, w, h = stats[c, :4]
        win = (slice(max(0, y - 3), min(H, y + h + 3)), slice(max(0, x - 3), min(W, x + w + 3)))
        piece = comp[win] == c
        ring = cv2.dilate(piece.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) & ~piece
        if not ring.any():
            continue
        colour = lab[win][piece].mean(0)
        person_c = lab[win][ring].mean(0)
        # nearest backdrop: search a widening window around the hole, else the global median
        backdrop_c = bg
        for grow_px in (100, 300, 900):
            far = (slice(max(0, y - grow_px), min(H, y + h + grow_px)),
                   slice(max(0, x - grow_px), min(W, x + w + grow_px)))
            if outside[far].sum() >= 200:
                backdrop_c = lab[far][outside[far]].mean(0)
                break
        if np.linalg.norm(colour - person_c) < np.linalg.norm(colour - backdrop_c):
            # plain majority: preferring garments here let one earring claim a sheet of hair
            sub = out[win]
            sub[piece] = ring_majority(out[win], piece)
            filled += 1
    return out, filled


def remove_specks(labels: np.ndarray, min_px: int) -> np.ndarray:
    """Islands smaller than min_px join whatever surrounds them."""
    out = labels.copy()
    H, W = labels.shape
    for lab in np.unique(labels):
        n, comp, stats, _ = cv2.connectedComponentsWithStats((labels == lab).astype(np.uint8))
        for c in np.flatnonzero(stats[1:, cv2.CC_STAT_AREA] < min_px) + 1:
            x, y, w, h = stats[c, :4]
            win = (slice(max(0, y - 3), min(H, y + h + 3)), slice(max(0, x - 3), min(W, x + w + 3)))
            piece = comp[win] == c
            sub = out[win]
            sub[piece] = ring_majority(sub, piece)
    return out


CLOTH_IDS = (4, 5, 6, 7, 17)          # upper_clothes, skirt, pants, dress, scarf
SKIN_IDS = (11, 12, 13, 14, 15, SKIN)   # face, legs, arms, neck/chest skin


def clean_semantics(coarse: np.ndarray, face_oval: np.ndarray | None, legs_visible: bool,
                    merge_garments: bool = True, photo: Image.Image | None = None,
                    log=print) -> np.ndarray:
    """Fix label mistakes the clothes parser makes on close-up, tilted portraits.

    It was trained on full-body fashion photos; on a head-and-shoulders shot it splits one
    bikini top into dress + upper_clothes + pants, calls an ear an 'arm', and finds 'legs'
    in a picture with none. SAM only sharpens boundaries, so these must be fixed first.
    """
    out = coarse.copy()
    H, W = out.shape

    # 1. One garment, one label: touching cloth pieces take the majority label. (SegFormer
    #    only - Sapiens tells a top from trousers reliably, and those do touch at the waist.)
    n, comp = cv2.connectedComponents(np.isin(out, CLOTH_IDS).astype(np.uint8))
    for c in range(1, n if merge_garments else 1):
        piece = comp == c
        vals = out[piece]
        major = int(np.bincount(vals).argmax())
        if (vals != major).any():
            out[piece] = major
            log(f"  garment pieces {sorted(set(int(v) for v in np.unique(vals)))} merged as {major}")
    # One-piece check (Sapiens): it only knows upper vs lower clothing, so a dress comes out
    # as a top with 'trousers' patches near the hem. Touching upper and lower pieces of the
    # same colour are one garment - a dress. A top and trousers in different colours stay apart.
    if not merge_garments and photo is not None:
        lab_img = cv2.cvtColor(np.asarray(photo), cv2.COLOR_RGB2LAB).astype(np.float32)
        for c in range(1, n):
            piece = comp == c
            up, low = piece & (out == 4), piece & (out == 6)
            if up.sum() < 500 or low.sum() < 500:
                continue
            gap = float(np.linalg.norm(lab_img[up].mean(0) - lab_img[low].mean(0)))
            if gap < 15:
                out[piece & np.isin(out, (4, 6))] = 7
                log(f"  upper + lower clothing of one colour (dE {gap:.1f}) merged into a dress")
    # A stray 'garment' far smaller than the main one and surrounded by skin is a shadow or a
    # skin fold the parser misread (seen: blurred skin beside an elbow called upper_clothes).
    if n > 2:
        areas = np.bincount(comp.ravel())[1:]
        for c in range(1, n):
            if areas[c - 1] >= 0.1 * areas.max():
                continue
            piece = comp == c
            around = ring_majority(out, piece)
            if around in SKIN_IDS or around == 0:
                out[piece] = around
                log(f"  stray garment speck ({int(areas[c - 1])} px) inside skin relabelled")

    # 2. Skin next to the face and above the chin is face (ears, temples), not arm or leg.
    if face_oval is not None:
        x0, y0, x1, y1 = mask_box(face_oval)
        r = max(5, int(0.3 * (x1 - x0)))
        zone = cv2.dilate(face_oval.astype(np.uint8),
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))) > 0
        zone[y1:] = False
        head_skin = zone & np.isin(out, (12, 13, 14, 15, SKIN))
        if head_skin.any():
            out[head_skin] = 11
            log(f"  {int(head_skin.sum())} px of 'arm/leg/skin' beside the face relabelled face")

    # 3. No legs unless the pose detector sees a knee or ankle in the frame.
    if not legs_visible:
        legs = np.isin(out, (12, 13))
        if legs.any():
            n, comp = cv2.connectedComponents(legs.astype(np.uint8))
            for c in range(1, n):
                piece = comp == c
                out[piece] = ring_majority(np.where(legs, 0, out), piece, {14, 15, SKIN}) or SKIN
            log(f"  'legs' with no leg in the pose relabelled ({int(legs.sum())} px)")
    return out


# ----------------------------------------------------------------------------- main entry
def refine(photo: Image.Image, probs: np.ndarray, face_oval: np.ndarray | None,
           hand_landmarks: list[np.ndarray], device: str, labels_map: dict,
           legs_visible: bool = True, extra_hands: np.ndarray | None = None,
           merge_garments: bool = True, log=print) -> dict:
    """Return refined full-res `labels` (uint8), `hands` mask, `person` mask and timings.

    probs: SegFormer class probabilities at full res (C, H, W).
    face_oval: full-res bool mask from the face mesh, or None.
    hand_landmarks: list of (21, 2) arrays of hand landmark pixels at full res.
    """
    W, H = photo.size
    coarse = probs.argmax(0).astype(np.uint8)
    t0 = time.time()
    sam = Sam(device)
    log(f"  SAM 2.1 loaded ({time.time() - t0:.1f}s)")

    # How far SAM may move a boundary beyond the coarse region (it is fixing 5-px blocks and
    # mislabelled slivers, not inventing regions). Without a limit, a SAM pass seeded on a
    # black strap happily returns the whole black dress.
    reach = max(15, round(0.02 * max(W, H)))
    grow = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * reach + 1, 2 * reach + 1))

    # SegFormer's 'face' covers face + neck + chest skin. Split it: the mesh oval is the face,
    # leftovers above the chin are still face (forehead, ears), below the chin are bare skin.
    if face_oval is not None:
        chin = mask_box(face_oval)[3]
        rows = np.arange(H)[:, None]
        is_face = coarse == 11
        coarse = coarse.copy()
        coarse[is_face & ~face_oval & (rows > chin)] = SKIN
        coarse[face_oval] = 11
    coarse = clean_semantics(coarse, face_oval, legs_visible, merge_garments, photo, log)

    def run_region(label: int, region: np.ndarray, k_pos: int = 3) -> np.ndarray | None:
        box = mask_box(region)
        if box is None:
            return None
        cb = pad_box(box, 0.12, W, H)
        x0, y0, x1, y1 = cb
        sub_ref = region[y0:y1, x0:x1]
        pos = interior_points(region, k_pos)
        neg = []
        sub_coarse = coarse[y0:y1, x0:x1]
        for other in np.unique(sub_coarse):
            if other == label:
                continue
            m = np.zeros_like(region)
            m[y0:y1, x0:x1] = sub_coarse == other
            if m.sum() > 0.01 * sub_ref.size:
                neg += interior_points(m, 1)
        lg = sam.segment(photo, cb, pos, neg[:6], ref=sub_ref)
        hard = np.zeros((H, W), bool)
        hard[y0:y1, x0:x1] = lg > 0
        return hard & (cv2.dilate(region.astype(np.uint8), grow) > 0)

    refined: dict[int, np.ndarray] = {}
    for lab in [int(v) for v in np.unique(coarse) if v != 0]:
        region = coarse == lab
        if lab == 11 and face_oval is not None:
            # the mesh is exact for the face itself; ears and temples come from the cleanup
            refined[11] = face_oval | region
            continue
        n, comp, stats, _ = cv2.connectedComponentsWithStats(region.astype(np.uint8))
        acc = np.zeros((H, W), bool)
        # SAM the 3 largest pieces of at least 0.05% of the image; smaller fragments keep
        # SegFormer's label and are only edge-snapped (each SAM pass costs ~10 s on CPU).
        areas = stats[1:, cv2.CC_STAT_AREA]
        big = [int(i) + 1 for i in np.argsort(-areas)[:3] if areas[i] >= 0.0005 * H * W]
        for c in range(1, n):
            part = comp == c
            if c not in big:
                acc |= part
                continue
            t = time.time()
            r = run_region(lab, part)
            if r is not None:
                acc |= r
                log(f"  {labels_map.get(lab, str(lab))} ({time.time() - t:.1f}s)")
        if acc.any():
            refined[lab] = acc

    # Paint back to front, so whatever is physically in front wins an overlap: skin under
    # clothes, straps over skin, arms over the dress they rest on, hair over face and shoulders.
    order = [SKIN, 5, 6, 7, 4, 17, 8, 16, 9, 10, 12, 13, 14, 15, 11, 2, 1, 3]
    labels = np.zeros((H, W), np.uint8)
    for lab in [l for l in np.unique(coarse) if l and int(l) not in refined]:
        labels[coarse == lab] = lab                       # unrefined specks
    for lab in sorted(refined, key=lambda l: order.index(l) if l in order else -1):
        labels[refined[lab]] = lab
    labels, filled = fill_enclosed(labels, photo, reach)
    if filled:
        log(f"  {filled} gap(s) hemmed in by the person and unlike the backdrop given back to it")
    sam_labels = labels.copy()

    # hands: prompted by their own landmarks, with a negative point up the forearm
    hands = np.zeros((H, W), bool)
    for lm in hand_landmarks:
        t = time.time()
        box = (int(lm[:, 0].min()), int(lm[:, 1].min()), int(lm[:, 0].max()), int(lm[:, 1].max()))
        cb = pad_box(box, 0.35, W, H)
        pos = [tuple(map(int, lm[i])) for i in HAND_TIPS + HAND_KNUCKLES]
        wrist, mid = lm[0], lm[9]
        forearm = wrist + 0.9 * (wrist - mid)
        neg = [(int(forearm[0]), int(forearm[1]))]
        ref = np.zeros((cb[3] - cb[1], cb[2] - cb[0]), np.uint8)
        cv2.fillConvexPoly(ref, cv2.convexHull((lm - [cb[0], cb[1]]).astype(np.int32)), 1)
        lg = sam.segment(photo, cb, pos, neg, ref=ref.astype(bool), use_box=False)
        size = int(max(box[2] - box[0], box[3] - box[1]))
        zone = cv2.dilate(ref, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size // 2 | 1,) * 2)) > 0
        hands[cb[1]:cb[3], cb[0]:cb[2]] |= (lg > 0) & zone
        log(f"  hand ({time.time() - t:.1f}s)")

    # Snap to the photo's own edges: guided-filter every label's one-hot channel, re-argmax.
    t = time.time()
    guide = np.asarray(photo).astype(np.float32) / 255
    radius = max(3, round(max(W, H) / 800))
    present = [int(l) for l in np.unique(labels)]
    snapped = np.stack([guided_snap((labels == l).astype(np.float32), guide, radius) for l in present])
    labels = np.asarray(present, np.uint8)[snapped.argmax(0)]
    labels = remove_specks(labels, min_px=max(50, W * H // 20000))
    if extra_hands is not None:
        # the parser's own hand pixels count too, but only on the arms (never on clothing)
        hands |= (extra_hands.astype(np.float32) > 0.5) & np.isin(labels, (14, 15))
    hands_m = guided_snap(hands.astype(np.float32), guide, radius) > 0.5
    # A hand traced from its own landmarks beats whatever garment edge spilled onto it:
    # each hand takes the arm label it is attached to.
    n, comp = cv2.connectedComponents(hands_m.astype(np.uint8))
    for c in range(1, n):
        piece = comp == c
        arm = ring_majority(labels, piece, {14, 15})
        arm = arm if arm in (14, 15) else 14
        labels[piece & ~np.isin(labels, (14, 15))] = arm
    log(f"  edge snapping ({time.time() - t:.1f}s), {sam.calls} SAM passes")
    return {"labels": labels, "sam_labels": sam_labels, "hands": hands_m,
            "person": labels > 0, "sam_calls": sam.calls}
