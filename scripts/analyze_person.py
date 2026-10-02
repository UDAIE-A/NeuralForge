"""Analyze one photo of a person: face, clothing, hands, pose, body proportions and depth.

  venv/Scripts/python.exe scripts/analyze_person.py photo.jpg
  venv/Scripts/python.exe scripts/analyze_person.py photo.jpg --device cpu --threads 4

Everything runs zero-shot on pretrained models - nothing is trained, one picture is enough:

  clothing + body parts   SegFormer clothes parser (18 labels: dress, upper_clothes, skirt, pants,
                          face, hair, arms, legs ...)
  face                    MediaPipe Face Landmarker, 478 points. Its face-oval mask stops at the
                          jaw, unlike the segmenter's 'face' label, which runs down the neck.
  hands                   MediaPipe Hand Landmarker
  pose + silhouette       MediaPipe Pose Landmarker (heavy), 33 joints + a person mask
  depth                   Depth Anything V2 Small

Output, in outputs/analysis/<photo name>/ by default:

  report.json             every measurement and landmark, in ORIGINAL-image pixels
  labels.png              the full label map (pixel value = label id, see LABELS)
  overlay.jpg             the photo with segmentation, skeleton and face mesh drawn on it
  masks/<label>.png       one mask per detected garment / body part, plus face_oval, hands, person
  depth.png               relative depth (brighter = nearer)

Body proportions are measured for garment fitting: widths of the silhouette (arms and hands
removed) at the shoulder, waist and hip lines, and joint-to-joint lengths from the pose. There
is no scale reference in a single photo, so they are reported in pixels and as ratios to the
shoulder-joint width, never in centimetres.

All models are Apache-2.0 except the SegFormer clothes parser (license "other" - see
checkpoints/segformer-clothes/README.md). Run scripts/download_image_models.py once first.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from change_clothes import (  # noqa: E402  (shared label map and model helpers)
    CKPT, DEPTH_MODEL, HAND_MODEL, LABELS as SEG_LABELS, ROOT, SEG_MODEL, bbox, fit_image, hand_mask, segment,
)
from refine_masks import SKIN, edge_alignment, refine, segformer_probs  # noqa: E402

LABELS = {**SEG_LABELS, SKIN: "neck_chest_skin"}

FACE_MODEL = CKPT / "face_landmarker.task"
POSE_MODEL = CKPT / "pose_landmarker_heavy.task"

GARMENTS = {1, 3, 4, 5, 6, 7, 8, 9, 10, 16, 17}  # hat, sunglasses, clothes, belt, shoes, bag, scarf
ARM_IDS = {14, 15}

# MediaPipe pose landmark indices
POSE = {"nose": 0, "l_shoulder": 11, "r_shoulder": 12, "l_elbow": 13, "r_elbow": 14,
        "l_wrist": 15, "r_wrist": 16, "l_hip": 23, "r_hip": 24, "l_knee": 25, "r_knee": 26,
        "l_ankle": 27, "r_ankle": 28}
SKELETON = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24),
            (23, 25), (25, 27), (24, 26), (26, 28)]
# Face-mesh indices for a few named points
FACE_POINTS = {"left_eye": 468, "right_eye": 473, "nose_tip": 1, "mouth_left": 61,
               "mouth_right": 291, "chin": 152, "forehead": 10}

# A distinct colour per segmentation label for the overlay
PALETTE = np.array([
    [0, 0, 0], [255, 200, 0], [120, 70, 20], [60, 60, 60], [230, 30, 30], [30, 160, 230],
    [30, 60, 200], [200, 40, 160], [150, 150, 0], [90, 200, 90], [90, 200, 90], [255, 170, 140],
    [255, 120, 60], [255, 120, 60], [255, 220, 120], [255, 220, 120], [120, 0, 200], [0, 200, 200],
    [255, 150, 200],
], np.uint8)


def mp_image(img: Image.Image):
    import mediapipe as mp
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(np.asarray(img)))


def detect_face(img: Image.Image) -> dict | None:
    """478-point face mesh, a jaw-bounded face mask, and a rough head turn."""
    if not FACE_MODEL.exists():
        return None
    import cv2
    from mediapipe.tasks import python as mpp
    from mediapipe.tasks.python import vision

    opts = vision.FaceLandmarkerOptions(base_options=mpp.BaseOptions(model_asset_path=str(FACE_MODEL)),
                                        num_faces=1, output_facial_transformation_matrixes=True)
    res = vision.FaceLandmarker.create_from_options(opts).detect(mp_image(img))
    if not res.face_landmarks:
        return None
    w, h = img.size
    pts = np.array([[l.x * w, l.y * h] for l in res.face_landmarks[0]], np.float32)
    oval = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(oval, cv2.convexHull(pts.astype(np.int32)), 255)
    face = {"points": pts, "oval": oval.astype(bool)}
    if res.facial_transformation_matrixes:
        r = np.asarray(res.facial_transformation_matrixes[0])[:3, :3]
        face["yaw_deg"] = float(np.degrees(np.arctan2(-r[2, 0], np.hypot(r[2, 1], r[2, 2]))))
        face["pitch_deg"] = float(np.degrees(np.arctan2(r[2, 1], r[2, 2])))
        face["roll_deg"] = float(np.degrees(np.arctan2(r[1, 0], r[0, 0])))
    return face


def detect_hands(img: Image.Image) -> list[np.ndarray]:
    """21 MediaPipe landmarks per detected hand, in pixels."""
    if not HAND_MODEL.exists():
        return []
    from mediapipe.tasks import python as mpp
    from mediapipe.tasks.python import vision

    opts = vision.HandLandmarkerOptions(base_options=mpp.BaseOptions(model_asset_path=str(HAND_MODEL)),
                                        num_hands=2, min_hand_detection_confidence=0.3)
    res = vision.HandLandmarker.create_from_options(opts).detect(mp_image(img))
    w, h = img.size
    return [np.array([[l.x * w, l.y * h] for l in hand], np.float32) for hand in res.hand_landmarks]


def detect_pose(img: Image.Image) -> dict | None:
    """33 body joints (pixels + visibility) and MediaPipe's person silhouette."""
    if not POSE_MODEL.exists():
        return None
    from mediapipe.tasks import python as mpp
    from mediapipe.tasks.python import vision

    opts = vision.PoseLandmarkerOptions(base_options=mpp.BaseOptions(model_asset_path=str(POSE_MODEL)),
                                        num_poses=1, output_segmentation_masks=True)
    res = vision.PoseLandmarker.create_from_options(opts).detect(mp_image(img))
    if not res.pose_landmarks:
        return None
    w, h = img.size
    lms = res.pose_landmarks[0]
    pts = np.array([[l.x * w, l.y * h] for l in lms], np.float32)
    vis = np.array([getattr(l, "visibility", 1.0) or 0.0 for l in lms], np.float32)
    # MediaPipe x/y are normalised to the frame and may fall outside it for cut-off joints
    inside = (pts[:, 0] >= 0) & (pts[:, 0] < w) & (pts[:, 1] >= 0) & (pts[:, 1] < h)
    sil = None
    if res.segmentation_masks:
        sil = res.segmentation_masks[0].numpy_view().squeeze() > 0.5
    return {"points": pts, "visibility": vis, "in_frame": inside, "silhouette": sil}


def estimate_depth(img: Image.Image, device: str) -> Image.Image:
    from transformers import pipeline

    est = pipeline("depth-estimation", model=DEPTH_MODEL, device=0 if device == "cuda" else -1)
    depth = est(img)["depth"].convert("L").resize(img.size)
    del est
    if device == "cuda":
        torch.cuda.empty_cache()
    return depth


def absorb_straps(labels: np.ndarray, max_fraction: float = 0.01) -> tuple[np.ndarray, int]:
    """Give small 'bag' pieces that touch a garment to that garment.

    The clothes parser labels thin dark shoulder straps as 'bag' (seen on a strappy dress:
    both straps came out as label 16). Left alone, a strap is reported as a bag and is not
    part of the dress mask, so repainting the dress would leave the old straps behind.
    A real bag is large or detached from the clothes; straps are small and attached.

    A strap can also be cut in two: the parser may call its middle 'face' (skin), leaving the
    top piece floating above the garment. A thin upright piece with the garment a short way
    below it is joined to the garment along a line as wide as the strap.
    """
    import cv2

    clothes = [4, 7, 5, 6, 17]  # garments a strap could belong to
    out = labels.copy()
    n, comp, stats, _ = cv2.connectedComponentsWithStats((labels == 16).astype(np.uint8))
    moved = 0
    kernel = np.ones((7, 7), np.uint8)
    for c in range(1, n):
        x, y, w, h, area = stats[c]
        piece = comp == c
        if piece.mean() > max_fraction:
            continue
        ring = cv2.dilate(piece.astype(np.uint8), kernel).astype(bool) & ~piece
        touching = labels[ring]
        touching = touching[np.isin(touching, clothes)]
        if len(touching):
            out[piece] = np.bincount(touching).argmax()
            moved += 1
            continue
        if h < 2 * w:
            continue  # not strap-shaped
        # Detached: look for the garment within 1.5x the piece's height below its bottom end.
        ys, xs = np.nonzero(piece)
        bottom = ys.max()
        bx = float(xs[ys >= bottom - 2].mean())
        y1 = min(labels.shape[0], bottom + int(1.5 * h) + 1)
        x0, x1 = max(0, x - 2 * w), min(labels.shape[1], x + 3 * w)
        gy, gx = np.nonzero(np.isin(labels[bottom:y1, x0:x1], clothes))
        if len(gy) == 0:
            continue
        k = int(np.argmin((gy) ** 2 + (gx + x0 - bx) ** 2))
        target = (int(gx[k] + x0), int(gy[k] + bottom))
        garment = int(labels[target[1], target[0]])
        thick = max(2, int(np.median(np.count_nonzero(piece[y:y + h], axis=1))))
        bridge = np.zeros(labels.shape, np.uint8)
        cv2.line(bridge, (int(bx), int(bottom)), target, 1, thickness=thick)
        out[piece] = garment
        out[(bridge > 0) & ~np.isin(labels, [2, 14, 15])] = garment  # never over hair or arms
        moved += 1
    return out, moved


def row_width(mask: np.ndarray, y: int, cx: float) -> tuple[int, int] | None:
    """The contiguous run of `mask` on row y that contains (or is nearest to) column cx."""
    xs = np.flatnonzero(mask[y])
    if len(xs) == 0:
        return None
    breaks = np.flatnonzero(np.diff(xs) > 1)
    starts = np.r_[xs[0], xs[breaks + 1]]
    ends = np.r_[xs[breaks], xs[-1]]
    i = int(np.argmin(np.where((starts <= cx) & (cx <= ends), 0, np.minimum(abs(starts - cx), abs(ends - cx)))))
    return int(starts[i]), int(ends[i])


def proportions(pose: dict, torso: np.ndarray) -> dict:
    """Garment-fitting widths and lengths. `torso` = person mask minus arms, hands and hair."""
    p, ok = pose["points"], pose["visibility"] > 0.5
    dist = lambda a, b: float(np.linalg.norm(p[a] - p[b]))  # noqa: E731
    out: dict = {"units": "pixels of the original image; *_ratio = value / shoulder_joint_width"}
    if not (ok[11] and ok[12]):
        out["note"] = "shoulders not visible - no proportions"
        return out
    shoulder = dist(11, 12)
    out["shoulder_joint_width"] = shoulder
    sy, cx = (p[11, 1] + p[12, 1]) / 2, (p[11, 0] + p[12, 0]) / 2
    h = torso.shape[0]

    def widest(y0, y1, narrow=False):
        best = None
        for y in range(max(0, int(y0)), min(h, int(y1))):
            run = row_width(torso, y, cx)
            if run is None:
                continue
            wdt = run[1] - run[0]
            if best is None or (wdt < best[0] if narrow else wdt > best[0]):
                best = (wdt, y, run)
        return best

    if ok[23] and ok[24]:
        hy = (p[23, 1] + p[24, 1]) / 2
        torso_len = hy - sy
        out["hip_joint_width"] = dist(23, 24)
        out["torso_length"] = float(torso_len)
        # Waist: narrowest silhouette row between 35% and 85% of shoulder->hip.
        waist = widest(sy + 0.35 * torso_len, sy + 0.85 * torso_len, narrow=True)
        # Hip: widest row within 25% of torso length either side of the hip joints.
        hip = widest(hy - 0.25 * torso_len, hy + 0.25 * torso_len)
        for name, m in (("waist", waist), ("hip", hip)):
            if m:
                out[f"{name}_width"] = m[0]
                out[f"{name}_line_y"] = m[1]
                out[f"{name}_span_x"] = list(m[2])
        if waist and hip:
            out["waist_to_hip_ratio"] = round(waist[0] / hip[0], 3)
        for a, b, name in ((11, 13, "l_upper_arm"), (13, 15, "l_forearm"), (12, 14, "r_upper_arm"),
                           (14, 16, "r_forearm"), (23, 25, "l_thigh"), (24, 26, "r_thigh")):
            if ok[a] and ok[b] and pose["in_frame"][a] and pose["in_frame"][b]:
                out[f"{name}_length"] = dist(a, b)
    out["shoulder_tilt_deg"] = float(np.degrees(np.arctan2(p[11, 1] - p[12, 1], p[11, 0] - p[12, 0])))
    for k in list(out):
        if k.endswith(("_width", "_length")) and k != "shoulder_joint_width":
            out[k.replace("_width", "_ratio").replace("_length", "_length_ratio")] = round(out[k] / shoulder, 3)
    return out


def framing(pose: dict | None, labels: np.ndarray) -> dict:
    """What part of the body is in the picture."""
    if pose is None:
        return {}
    seen = lambda i: bool(pose["in_frame"][i] and pose["visibility"][i] > 0.5)  # noqa: E731
    shot = ("full body" if seen(27) or seen(28) else "knees up" if seen(25) or seen(26)
            else "waist up" if seen(23) or seen(24) else "head and shoulders")
    h = labels.shape[0]
    person = labels > 0
    return {"shot": shot,
            "person_touches_bottom_edge": bool(person[h - 1].any()),
            "person_height_fraction": round(float(np.any(person, 1).mean()), 3)}


def draw_overlay(img: Image.Image, labels: np.ndarray, face: dict | None, pose: dict | None,
                 props: dict) -> Image.Image:
    color = PALETTE[labels]
    alpha = (labels > 0)[..., None] * 0.45
    base = np.asarray(img).astype(np.float32)
    out = Image.fromarray((base * (1 - alpha) + color * alpha).astype(np.uint8))
    d = ImageDraw.Draw(out)
    r = max(2, img.size[0] // 300)
    if face is not None:
        for x, y in face["points"][::3]:
            d.point((float(x), float(y)), fill=(255, 255, 255))
    if pose is not None:
        p, ok = pose["points"], pose["visibility"] > 0.5
        for a, b in SKELETON:
            if ok[a] and ok[b]:
                d.line([tuple(p[a]), tuple(p[b])], fill=(0, 255, 120), width=r)
        for i in POSE.values():
            if ok[i]:
                x, y = p[i]
                d.ellipse([x - 2 * r, y - 2 * r, x + 2 * r, y + 2 * r], outline=(0, 255, 120), width=r)
    for name, col in (("waist", (255, 255, 0)), ("hip", (0, 200, 255))):
        if f"{name}_line_y" in props:
            y, (x0, x1) = props[f"{name}_line_y"], props[f"{name}_span_x"]
            d.line([(x0, y), (x1, y)], fill=col, width=r)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image", type=Path)
    ap.add_argument("--out", type=Path, default=None, help="default: outputs/analysis/<image name>")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--threads", type=int, default=4, help="CPU threads for torch (keeps the machine responsive)")
    ap.add_argument("--max-side", type=int, default=1024, help="working resolution (longest side)")
    ap.add_argument("--no-depth", action="store_true", help="skip the depth map")
    ap.add_argument("--fast", action="store_true",
                    help="skip SAM refinement: SegFormer masks at working resolution (blocky edges)")
    args = ap.parse_args()

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else \
        ("cpu" if args.device == "auto" else args.device)
    torch.set_num_threads(max(1, args.threads))
    out_dir = args.out or ROOT / "outputs" / "analysis" / args.image.stem
    (out_dir / "masks").mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    orig = Image.open(args.image).convert("RGB")
    img = fit_image(orig, args.max_side)
    scale = orig.size[0] / img.size[0]          # working pixels -> original pixels
    print(f"{args.image.name}: {orig.size[0]}x{orig.size[1]}, analysing at {img.size[0]}x{img.size[1]} on {device}")

    # Landmark detectors run at working resolution; their points are exact enough to lift.
    face = detect_face(img)
    pose = detect_pose(img)
    depth = None if args.no_depth else estimate_depth(img, device)
    gray_full = np.asarray(orig.convert("L"))
    coarse_full = np.asarray(Image.fromarray(segment(img, device)).resize(orig.size, Image.NEAREST))
    scores = {"segformer_512_upscaled": edge_alignment(coarse_full, gray_full)}

    if args.fast:
        labels = segment(img, device)
        hands = hand_mask(img, grow=0)
    else:
        # Lift everything to full resolution and redraw the boundaries there.
        import cv2
        hand_lms = [lm * scale for lm in detect_hands(img)]
        if face is not None:
            face["points"] = face["points"] * scale
            oval = np.zeros(orig.size[::-1], np.uint8)
            cv2.fillConvexPoly(oval, cv2.convexHull(face["points"].astype(np.int32)), 255)
            face["oval"] = oval.astype(bool)
        if pose is not None:
            pose["points"] = pose["points"] * scale
        if depth is not None:
            depth = depth.resize(orig.size, Image.BILINEAR)
        print("refining at full resolution (SegFormer letterboxed + SAM 2.1 + edge snapping):")
        t = time.time()
        probs = segformer_probs(orig, SEG_MODEL, device)
        scores["segformer_letterboxed_fullres"] = edge_alignment(probs.argmax(0).astype(np.uint8), gray_full)
        print(f"  SegFormer at full res ({time.time() - t:.1f}s)")
        legs_visible = pose is not None and any(
            pose["in_frame"][i] and pose["visibility"][i] > 0.5 for i in (25, 26, 27, 28))
        out = refine(orig, probs, face["oval"] if face is not None else None, hand_lms, device, LABELS,
                     legs_visible=legs_visible)
        del probs
        scores["sam_before_snap"] = edge_alignment(out["sam_labels"], gray_full)
        labels, hands = out["labels"], out["hands"]
        img, scale = orig, 1.0
    labels, straps = absorb_straps(labels)
    if straps:
        print(f"merged {straps} strap piece(s) the parser labelled 'bag' into the garment they hang from")
    scores["final"] = edge_alignment(np.asarray(Image.fromarray(labels).resize(orig.size, Image.NEAREST)), gray_full)

    # Body silhouette without arms, hands and hair: what garments are fitted to.
    person = labels > 0
    torso = person & ~np.isin(labels, list(ARM_IDS | {2})) & ~hands
    props = proportions(pose, torso) if pose is not None else {"note": "no person pose detected"}

    # ---- masks
    present = {}
    for i in np.unique(labels):
        if i == 0:
            continue
        m = labels == i
        present[LABELS[int(i)]] = {"area_fraction": round(float(m.mean()), 4),
                                   "bbox": [round(v * scale) for v in bbox(m)],
                                   "kind": "garment" if i in GARMENTS else "body"}
        Image.fromarray(m.astype(np.uint8) * 255).resize(orig.size, Image.NEAREST).save(
            out_dir / "masks" / f"{LABELS[int(i)]}.png")
    extra = {"person": person, "hands": hands}
    if face is not None:
        extra["face_oval"] = face["oval"]
    for name, m in extra.items():
        Image.fromarray(m.astype(np.uint8) * 255).resize(orig.size, Image.NEAREST).save(
            out_dir / "masks" / f"{name}.png")
    # the whole label map, one byte per pixel (values = LABELS keys), for other scripts to reuse
    Image.fromarray(labels).resize(orig.size, Image.NEAREST).save(out_dir / "labels.png")
    if depth is not None:
        depth.resize(orig.size, Image.BILINEAR).save(out_dir / "depth.png")
    draw_overlay(img, labels, face, pose, props).resize(orig.size, Image.LANCZOS).save(
        out_dir / "overlay.jpg", quality=92)

    # ---- report (all coordinates in original-image pixels)
    s = lambda v: round(float(v) * scale, 1)  # noqa: E731
    report = {
        "image": str(args.image), "size": list(orig.size), "device": device,
        "mode": "fast" if args.fast else "precise",
        "edge_alignment": {k: round(v, 3) for k, v in scores.items()},
        "framing": framing(pose, labels),
        "clothing": {k: v for k, v in present.items() if v["kind"] == "garment"},
        "body_parts": {k: v for k, v in present.items() if v["kind"] == "body"},
        "hands": {"detected": bool(hands.any()), "area_fraction": round(float(hands.mean()), 4)},
        "face": None, "pose": None,
        "proportions": {k: (s(v) if isinstance(v, (int, float)) and k.endswith(("_width", "_length")) else
                            [round(x * scale) for x in v] if k.endswith("_span_x") else
                            round(v * scale) if k.endswith("_line_y") else v)
                        for k, v in props.items()},
    }
    if face is not None:
        fp = face["points"]
        report["face"] = {
            "bbox": [round(v * scale) for v in bbox(face["oval"])],
            "points": {k: [s(fp[i, 0]), s(fp[i, 1])] for k, i in FACE_POINTS.items()},
            "head_turn_deg": {k: round(face[f"{k}_deg"], 1) for k in ("yaw", "pitch", "roll") if f"{k}_deg" in face},
            "mesh_478": [[s(x), s(y)] for x, y in fp],
        }
    if pose is not None:
        report["pose"] = {name: {"xy": [s(pose["points"][i, 0]), s(pose["points"][i, 1])],
                                 "visibility": round(float(pose["visibility"][i]), 3),
                                 "in_frame": bool(pose["in_frame"][i])}
                          for name, i in POSE.items()}
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    # ---- summary
    print("edges     : " + ", ".join(f"{k}={v:.1%}" for k, v in scores.items())
          + "  (share of mask boundary on a real image edge)")
    print(f"framing   : {report['framing'].get('shot', 'n/a')}")
    print(f"clothing  : {', '.join(f'{k} ({v['area_fraction']:.0%})' for k, v in report['clothing'].items()) or 'none'}")
    print(f"body      : {', '.join(report['body_parts']) or 'none'}")
    print(f"face      : {'found, head turn ' + str(report['face']['head_turn_deg']) if face else 'not found'}")
    print(f"hands     : {'found' if report['hands']['detected'] else 'not found'}")
    pr = report["proportions"]
    keys = [k for k in ("shoulder_joint_width", "waist_width", "hip_width", "torso_length") if k in pr]
    print("proportion: " + ", ".join(f"{k}={pr[k]:.0f}px" for k in keys))
    ratios = {k: pr[k] for k in ("waist_ratio", "hip_ratio", "torso_length_ratio") if k in pr}
    if ratios:
        print("ratios    : " + ", ".join(f"{k}={v}" for k, v in ratios.items()) + "  (x shoulder width)")
    print(f"saved     : {out_dir}  ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
