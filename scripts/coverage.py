"""Which parts of the body a photo's clothes actually cover, and which a new garment will.

The name of a garment says almost nothing about what it covers. Two photos of a one-piece
swimsuit come out of the parser as 'dress' and as 'upper_clothes', and a 'top' can be a crop top
showing the whole midriff or a tunic to the thigh. What is reliable is the shape of the cloth:
where it is on the body relative to the body's own landmarks.

So this module separates two things that used to be conflated:

  measured   where the cloth IS in this photo, per body region
  required   where the NEW garment has to be, from GARMENTS[] matched on the prompt

and `plan()` combines them into the set of body regions to repaint. That is what stops the
pipeline painting a garment it has no room for, and what keeps it from repainting a region the new
garment does not reach.

## Regions are found from the cloth, not from image rows

The first version put each region at a fixed fraction of the shoulder-to-hip distance measured
from the pose joints. That is wrong in a way that only shows up on real photos: when a person is
seated, turned away, or filling the frame, the landmark positions describe a body much larger
than the one in the picture, and the 'chest' band lands on the face. Measured coverage then came
out as 0.00 for a garment that plainly covers the chest.

So the regions are anchored to the CLOTH instead. A garment's own top edge is its neckline and its
bottom edge is its hem; the person is cut into slices along the torso axis and each slice is
asked how much cloth it holds. A slice that is bare between two covered ones is the midriff, a
backless top, the gap of a cut-out dress - the gaps are found, not assumed from a table.

The pose joints are used for one thing only: the direction of the torso axis, so a leaning body is
sliced along its own slope rather than along horizontal rows.

Front and back cannot be told apart in a single view - `back` is the same pixels as `chest` - so
only the side actually facing the camera is reported, chosen by where the nose sits between the
shoulders.

Every number is a fraction of a body region, never pixels, so a 4000px and a 600px photo of the
same outfit give the same result.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

# Body regions, in the order they run down the body. `back` is only filled when the person is
# turned away from the camera; otherwise the visible side of the torso is `chest`.
REGIONS = ("neck", "shoulder", "chest", "back", "midriff", "hip", "thigh", "calf")

# How many slices the torso and the legs are cut into. Enough to see a midriff band a few slices
# tall, few enough that a single noisy row cannot invent one.
TORSO_SLICES = 24
LEG_SLICES = 8
# Label ids from mattmdjaga/segformer_b2_clothes, mirrored here so this module can be used without
# importing change_clothes (which imports nothing from it back, but the test scripts do not always
# have the whole pipeline loaded). Change them together.
_FACE, _LLEG, _RLEG, _LARM, _RARM = 11, 12, 13, 14, 15
_SKIN_NECK_CHEST = 18           # refine_masks' own exposed-skin label
PARTS = {
    "skin": {_SKIN_NECK_CHEST, _FACE, _LLEG, _RLEG, _LARM, _RARM},
    # garment pieces that belong to only one half of the body: 'dress' and 'belt' label both, so
    # they are in neither and a one-piece garment never reports a waist gap
    "upper_only": {4, 17},      # upper_clothes, scarf
    "lower_only": {5, 6},       # skirt, pants
    # the limbs are skin wherever they are visible, so a column that runs down an arm finds no
    # cloth above or below it and looks exactly like a bare midriff. Excluding them is what stops a
    # plain dress from reporting a gap it does not have.
    "limbs": {_LLEG, _RLEG, _LARM, _RARM},
}

@dataclass
class BodyMap:
    """Which body regions the photo shows, and how much cloth each one holds."""
    regions: dict[str, np.ndarray] = field(default_factory=dict)      # region -> pixel mask
    spans: dict[str, tuple[int, int]] = field(default_factory=dict)   # region -> (y0, y1)
    body: np.ndarray | None = None              # the whole person mask the regions sit inside
    covered: dict[str, np.ndarray] = field(default_factory=dict)      # region -> [cloth fraction]
    measured_fractions: dict[str, float] = field(default_factory=dict)
    exposed: dict[str, float] = field(default_factory=dict)           # bare-skin fraction
    exposed_gap_cols: float = 0.0
    midriff_source: str = ""
    facing: float = 0.5
    source: str = "none"
    sh: np.ndarray = field(default_factory=lambda: np.zeros(2))
    hip: np.ndarray = field(default_factory=lambda: np.zeros(2))
    scale: float = 0.0                      # shoulder-to-hip distance in px
    leg_origin: np.ndarray | None = None
    leg_end: np.ndarray | None = None
    leg_len: float = 0.0

    def coverage(self, region: str) -> float | None:
        """Fraction of a region covered by cloth, or None when the photo does not show it."""
        v = self.measured_fractions.get(region)
        return None if v is None else float(v)


def _axis_from_pose(pose: dict | None) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Shoulder midpoint, hip midpoint and the distance between them, from the pose joints."""
    if pose is None:
        return None
    p, v = pose["points"], pose["visibility"]
    if not (v[11] > 0.5 and v[12] > 0.5 and v[23] > 0.5 and v[24] > 0.5):
        return None
    sh = np.array([(p[11][0] + p[12][0]) / 2, (p[11][1] + p[12][1]) / 2], float)
    hip = np.array([(p[23][0] + p[24][0]) / 2, (p[23][1] + p[24][1]) / 2], float)
    n = float(np.hypot(*(hip - sh)))
    return sh, hip, n if n > 20 else 0.0



def _facing(pose: dict | None, face: np.ndarray | None) -> float:
    """How far the head is turned from the camera: 0.0 = square on, 1.0 = full profile.

    Front and back are the same pixels in one photo, so this is the only thing that decides which
    side of the torso gets reported as `chest` and which as `back`. It is read off how far the nose
    sits from the midpoint between the shoulders, in shoulder widths, because that is what a profile
    actually does: the nose swings out to one side while the shoulders stay put.

    A real back view also puts the nose there, so this cannot tell a back from a face on its own.
    That is why `back` is only claimed at the confident end of the range, and why `plan()` keeps the
    repaint set the same either way - a wrong guess costs a label on one region, not a garment.

    `face` is a list of (x, y) landmarks such as analyze_person's face['points']. The pose nose is
    preferred because it is already in the pose's coordinate space; the face landmarks are only a
    fallback, and only if they are genuine point pairs, since some callers pass a per-pixel index
    map under the same name and an (H, W) array would read as a face pressed against the edge."""
    p = pose["points"] if pose is not None else None
    nose_x = None
    if p is not None and pose["visibility"][1] > 0.3:
        nose_x = float(p[1][0])
    elif face is not None:
        f = np.asarray(face, float)
        if f.ndim == 2 and 4 <= f.shape[0] <= 1000 and f.shape[1] == 2:
            ys = f[:, 1]
            lo = ys.min() + 0.30 * (ys.max() - ys.min())
            sel = f[ys > lo]
            nose_x = float(sel[:, 0].mean()) if len(sel) else float(f[:, 0].mean())
    if nose_x is None or p is None:
        return 0.5
    if not (pose["visibility"][11] > 0.5 and pose["visibility"][12] > 0.5):
        return 0.5
    sh_mid = (p[11][0] + p[12][0]) / 2
    sh_w = abs(p[11][0] - p[12][0])
    if sh_w < 10:
        return 0.5
    return float(min(1.0, abs(nose_x - sh_mid) / sh_w))


def body_map(labels: np.ndarray, pose: dict | None, cloth_ids: set[int] | None = None,
             face: np.ndarray | None = None, parts: dict | None = None) -> BodyMap:
    """Work out which body regions the photo shows and how much cloth each one holds.

    `cloth_ids` is the label set that counts as clothing (change_clothes.PARTS for the pass). The
    body always comes from the whole person mask, never from the cloth: a bare midriff has to be a
    region of the body, or there is nothing left to describe as bare.

    `parts` carries the label sets that the segmentation already defines - 'skin' for exposed skin,
    and 'upper_only'/'lower_only' for garment pieces that belong to just one half of the body.
    They default to the ids in change_clothes so a caller that knows nothing about label ids still
    gets a sensible answer, but passing them in keeps the two modules from drifting apart."""
    H, W = labels.shape
    b = BodyMap()
    person = (labels > 0).astype(np.uint8)
    if not person.any():
        return b
    cloth = np.isin(labels, sorted(cloth_ids)) if cloth_ids else np.zeros(labels.shape, bool)
    b.facing = _facing(pose, face)

    got = _axis_from_pose(pose)
    if got is not None and got[2] > 0:
        b.sh, b.hip, b.scale = got[0], got[1], got[2]
        b.source = "pose"
    else:
        # No usable pose: fall back to the person's own bounding box as the torso span. The regions
        # are then approximate, which `source` records and `plan()` passes on as `confident: False`.
        ys, xs = np.nonzero(person)
        y0, y1 = int(ys.min()), int(ys.max())
        cx = (int(xs.min()) + int(xs.max())) / 2
        b.sh = np.array([cx, float(y0)], float)
        b.hip = np.array([cx, float(y1)], float)
        b.scale = max(8.0, float(y1 - y0) * 0.45)     # guess: the torso is part of the height
        b.source = "silhouette"

    legs = _legs_from_pose(pose)
    if legs is not None:
        b.leg_origin, b.leg_end, b.leg_len = legs

    _assign_regions(b, labels, person, cloth, dict(PARTS, **parts) if parts else PARTS)
    b.body = person > 0
    return b


def _legs_from_pose(pose):
    """Hip midpoint, knee midpoint and the distance between them, or None if the knees are hidden."""
    if pose is None:
        return None
    p, v = pose["points"], pose["visibility"]
    hips = [i for i in (23, 24) if v[i] > 0.5]
    knees = [i for i in (25, 26) if v[i] > 0.5]
    if not hips or not knees:
        return None
    hip_pt = np.array([float(np.mean([p[i][0] for i in hips])),
                       float(np.mean([p[i][1] for i in hips]))], float)
    knee_pt = np.array([float(np.mean([p[i][0] for i in knees])),
                        float(np.mean([p[i][1] for i in knees]))], float)
    n = float(np.hypot(*(knee_pt - hip_pt)))
    return (hip_pt, knee_pt, n) if n > 20 else None


def _cloth_holes(body: np.ndarray, cloth: np.ndarray, y0: int, y1: int, x0: int, x1: int,
                 min_gap: int) -> tuple[np.ndarray, float]:
    """The bare body pixels that sit BETWEEN two runs of cloth, per column.

    This is the only dependable way to find a midriff. The pose hip joints are the hip *joints*,
    well below the waist, so a band placed at a fixed fraction of the torso lands inside the top
    rather than on the bare strip: in photo_0026 the gap is at 0.63-0.72 of the shoulder-to-hip
    span while a 0.34-0.68 band sits wholly inside the crop top and measures 0.86 covered.

    Holes are found between CONSECUTIVE runs of cloth, never against the edges of the search box,
    for the same reason: the skirt runs on below the hip landmarks, so any column whose cloth
    reaches the bottom of a box ending at the hips would be discarded and every gap missed.

    Returns the mask of hole pixels and the fraction of columns that have one."""
    H, W = body.shape
    y0, y1 = max(0, y0), min(H - 1, y1)
    mask = np.zeros((H, W), bool)
    hits = seen = 0
    for x in range(max(0, x0), min(W - 1, x1) + 1):
        col_body = body[y0:y1 + 1, x]
        if not col_body.any():
            continue
        seen += 1
        col_cloth = (cloth[y0:y1 + 1, x] & col_body).astype(np.int8)
        d = np.diff(np.concatenate(([0], col_cloth, [0])))
        starts = np.flatnonzero(d == 1)
        ends = np.flatnonzero(d == -1)            # exclusive end of each run
        if len(starts) < 2:
            continue
        best_len, best_s, best_e = 0, 0, 0
        for i in range(len(starts) - 1):
            gs, ge = int(ends[i]), int(starts[i + 1])
            if ge <= gs:
                continue
            gap = col_body[gs:ge] & ~col_cloth[gs:ge].astype(bool)
            n = int(gap.sum())
            if n > best_len:
                best_len, best_s, best_e = n, gs, ge
        if best_len >= min_gap:
            hits += 1
            idx = np.arange(best_s, best_e)
            idx = idx[col_body[best_s:best_e] & ~col_cloth[best_s:best_e].astype(bool)]
            mask[y0 + idx, x] = True
    return mask, (hits / seen if seen else 0.0)


def _waist_gap(body: np.ndarray, labels: np.ndarray, upper_only: set[int], lower_only: set[int],
               y0: int, y1: int, x0: int, x1: int, min_gap: int) -> tuple[np.ndarray, float]:
    """The bare band between the bottom of the upper garment and the top of the lower one.

    This is the reliable way to place a midriff, and it needs no anatomy at all. SegFormer and the
    refinement label the pieces separately - a crop top is 'upper_clothes', a skirt is 'skirt' - so
    the waist is simply where the upper piece stops and the lower piece starts, per column. Where
    that leaves a real gap, that gap IS the bare midriff.

    A one-piece dress shares one label for the bodice and the skirt, so there is no boundary to find
    and the caller falls back to the pose band. That is the right answer for a dress: it covers the
    midriff, so there is no gap to protect.

    Labels that count for both (a dress, a belt) are excluded from both sets, otherwise the waist
    would always find the dress touching itself and report no gap."""
    H, W = body.shape
    y0, y1 = max(0, y0), min(H - 1, y1)
    mask = np.zeros((H, W), bool)
    hits = seen = 0
    up = np.isin(labels, sorted(upper_only)) if upper_only else np.zeros(labels.shape, bool)
    lo = np.isin(labels, sorted(lower_only)) if lower_only else np.zeros(labels.shape, bool)
    for x in range(max(0, x0), min(W - 1, x1) + 1):
        col_body = body[y0:y1 + 1, x]
        if not col_body.any():
            continue
        seen += 1
        u = np.flatnonzero(up[y0:y1 + 1, x])
        d = np.flatnonzero(lo[y0:y1 + 1, x])
        if not len(u) or not len(d):
            continue
        bottom = int(u[u < d[0]].max()) if (u < d[0]).any() else -1
        top = int(d[d > u[-1]].min()) if (d > u[-1]).any() else -1
        if bottom < 0 or top < 0 or top - bottom < min_gap:
            continue
        band = col_body[bottom + 1:top]
        if not band.any():
            continue
        hits += 1
        ys = np.arange(bottom + 1, top)
        mask[y0 + ys[band], x] = True
    return mask, (hits / seen if seen else 0.0)



def skin_tone(img: np.ndarray, labels: np.ndarray, skin_ids: set[int] | None = None) -> dict:
    """Measure the subject's skin tone from the photo, so repainted skin can be asked to match.

    Repainting skin with a diffusion model reliably drifts paler and cooler than the original: the
    model's prior for skin is lighter than most real skin, and nothing in the prompt says otherwise.
    So the tone is measured and named rather than left to the prior.

    Only pixels the segmentation already calls skin are sampled, and only the middle of the
    luminance range, because the top and bottom percentiles are specular highlight and cast shadow
    rather than skin colour. Taking a straight mean over a lit forearm returns the highlight.

    Names follow the usual retail depth scale (fair -> very deep) with an undertone from the
    red-minus-blue channel. These are coarse buckets on purpose: the goal is to stop a drift, and a
    wrong bucket by one step is harmless where a fixed 'medium brown' is wrong for everyone."""
    a = np.asarray(img)
    if a.ndim != 3 or a.shape[2] < 3:
        return {"phrase": "", "rgb": None}
    ids = sorted(skin_ids) if skin_ids else sorted(PARTS["skin"])
    m = np.isin(labels, ids) if ids else np.zeros(labels.shape, bool)
    if m.sum() < 200:
        return {"phrase": "", "rgb": None}
    px = a[m][:, :3].astype(np.float32)
    lum = px @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    lo, hi = np.percentile(lum, [20, 80])
    mid = (lum >= lo) & (lum <= hi)
    if mid.sum() < 100:
        return {"phrase": "", "rgb": None}
    core = px[mid]
    rgb = [float(v) for v in np.median(core, axis=0)]
    lum_m = float(np.median(core @ np.array([0.2126, 0.7152, 0.0722], np.float32)))
    warm = rgb[0] - rgb[2]

    if lum_m >= 200:
        depth = "very fair"
    elif lum_m >= 170:
        depth = "fair"
    elif lum_m >= 140:
        depth = "light"
    elif lum_m >= 112:
        depth = "medium-light"
    elif lum_m >= 88:
        depth = "medium"
    elif lum_m >= 66:
        depth = "medium-deep"
    elif lum_m >= 44:
        depth = "deep"
    else:
        depth = "very deep"
    undertone = "warm" if warm > 18 else ("cool" if warm < -4 else "neutral")
    # how much of the visible skin is shadow, a cheap read on how hard the light is
    spread = float(np.percentile(lum, 90) - np.percentile(lum, 10))
    light = "soft even light" if spread < 55 else ("directional light" if spread < 95
                                                   else "hard directional light")
    return {"phrase": f"{depth} {undertone} skin under {light}", "rgb": rgb,
            "depth": depth, "undertone": undertone, "light": light,
            "luminance": lum_m, "warmth": warm, "contrast": spread,
            "samples": int(m.sum())}


def _skin_blob(skin: np.ndarray, y_from: int, min_area: int) -> tuple[np.ndarray, float]:
    """The largest connected patch of exposed skin below `y_from`, and its top edge as a fraction.

    refine_masks already labels the bare torso skin, so a bare midriff arrives as a blob of that
    label rather than something to be inferred from edges. That is far steadier than hunting for
    boundaries between garment labels: in photo_0015 the top and the skirt are one label with no
    seam to find, and in photo_0021 the arm crosses the waist so most of the gap's columns are limb
    columns - both defeat a label-boundary search, and neither defeats the blob.

    Returns an empty mask when there is no such patch, which is the correct answer for a one-piece
    garment that covers the middle."""
    import cv2

    H, W = skin.shape
    work = np.zeros_like(skin)
    work[int(y_from):, :] = skin[int(y_from):, :]
    if not work.any():
        return np.zeros_like(skin), 0.0
    n, lab, stats, _ = cv2.connectedComponentsWithStats(work.astype(np.uint8), 8)
    if n <= 1:
        return np.zeros_like(skin), 0.0
    best, best_area = 0, 0
    for i in range(1, n):
        a = int(stats[i, cv2.CC_STAT_AREA])
        if a >= min_area and a > best_area:
            best, best_area = i, a
    if not best:
        return np.zeros_like(skin), 0.0
    m = lab == best
    top = float(np.flatnonzero(m.any(1)).min())
    return m, top


def _assign_regions(b: BodyMap, labels: np.ndarray, person: np.ndarray, cloth: np.ndarray,
                   parts: dict | None = None) -> None:
    """Fill in the regions: where each one is on this body, and how much of it is cloth.

    Two measurements, and it matters that they are kept apart:

      region masks    the pixels of the body the region occupies
      exposure        how much of that region shows bare skin rather than cloth

    Exposure is read from the label semantics rather than guessed from geometry. The refinement
    already labels exposed neck and chest skin as its own id and the limbs as their own ids, so
    'bare' is a label test with no threshold to tune - which is why a bare midriff measures 1.0 and
    the same band inside a dress measures 0.0.

    The midriff mask comes from the waist gap between the upper and lower garment pieces, because
    the pose hip joints sit well below the waist: a band at a fixed fraction of the shoulder-to-hip
    span lands inside the crop top instead of on the bare strip between it and the skirt."""
    H, W = person.shape
    body = person > 0
    parts = parts or {}
    upper_only = set(parts.get("upper_only") or set())
    lower_only = set(parts.get("lower_only") or set())

    sh, hip, torso = b.sh, b.hip, b.scale
    y_sh, y_hip = int(sh[1]), int(hip[1])
    skin = np.isin(labels, sorted(parts["skin"])) if parts.get("skin") else (~cloth & body)
    limbs = np.isin(labels, sorted(parts["limbs"])) if parts.get("limbs") else np.zeros(labels.shape, bool)
    # the torso without the arms hanging beside it: used to find the waist, not to place the masks
    core = body & ~limbs

    band = core[y_sh:y_hip + 1, :]
    if band.any():
        cols = np.flatnonzero(band.any(0))
        x0, x1 = int(cols.min()), int(cols.max())
    else:
        x0, x1 = int(sh[0]), int(hip[0])
    mid_x = (x0 + x1) // 2
    half = max(4, int((x1 - x0) * 0.30))
    cx0, cx1 = max(0, mid_x - half), min(W - 1, mid_x + half)

    def band_mask(y0f, y1f, ax0=cx0, ax1=cx1):
        m = np.zeros_like(body)
        m[int(y_sh + torso * y0f):int(y_sh + torso * y1f) + 1, max(0, ax0):min(W, ax1 + 1)] = True
        return m & body

    def put(name, mask, y0=None, y1=None):
        """Measure a region: coverage from the cloth in it, exposure from the bare skin in it."""
        m = mask & body
        n = int(m.sum())
        if not n:
            return False
        cov = float((m & cloth).sum()) / n
        exp = float((m & skin).sum()) / n
        b.covered[name] = np.array([cov])
        b.regions[name] = m
        b.measured_fractions[name] = cov
        b.exposed[name] = exp
        if y0 is not None and y1 is not None:
            b.spans[name] = (int(y0), int(y1))
        return True

    def put_band(name, y0f, y1f, ax0=cx0, ax1=cx1):
        y0 = int(y_sh + torso * y0f)
        y1 = int(y_sh + torso * y1f)
        return put(name, band_mask(y0f, y1f, ax0, ax1), y0, y1)

    def put_band(name, y0f, y1f, ax0=cx0, ax1=cx1):
        # No pose means no shoulders and no hips to measure between, so a band placed at a fraction
        # of that span would be anchored to the bounding box instead - which invented a hip and a
        # midriff on a tight head-and-shoulders crop, and `plan()` then protected a waist that was
        # never in shot. Only cloth-derived regions survive without a pose, because a gap between
        # two garments is a fact about the cloth rather than about the anatomy.
        if b.source != "pose":
            return False
        y0 = int(y_sh + torso * y0f)
        y1 = int(y_sh + torso * y1f)
        return put(name, band_mask(y0f, y1f, ax0, ax1), y0, y1)

    put_band("neck", 0.0, 0.16)
    put_band("shoulder", 0.0, 0.16)

    # chest and back are the same pixels in one photo, so both are reported. `facing` only
    # decides which of the two a backless prompt should treat as the bare side.
    put_band("chest", 0.10, 0.36)
    put_band("back", 0.10, 0.36)

    # the midriff: the gap between the upper and lower pieces where there is one, otherwise the
    # band that a one-piece garment covers
    min_gap = max(5, int(torso * 0.03))
    y_end = min(H - 1, y_hip + int(torso * 0.55))
    gap, gap_cols = _waist_gap(body, labels, upper_only, lower_only, y_sh, y_end, cx0, cx1, min_gap)
    how = "waist gap"
    if not (gap_cols > 0.15 and gap.any()) and b.source == "pose":
        # No seam between the garment pieces. Either the middle is covered - a one-piece dress - or
        # the skin shows through a cut-out, which is a real bare midriff (photo_0104). Ask the
        # refinement's own exposed-skin label which of the two this is.
        #
        # Only with a pose. Deciding whether a bare patch is a midriff or a chest is a question about
        # where the patch sits on the body, and without shoulders and hips to measure against the
        # bounding box is no answer at all: on a tight head-and-shoulders crop it read the whole
        # decolletage as a midriff and went on to protect it.
        blob, blob_top = _skin_blob(skin & core, y_sh + int(torso * 0.28),
                                    int(0.0015 * H * W))
        if blob.any():
            how = "skin blob"
            frac = (blob_top - y_sh) / max(1.0, torso)
            if frac < 0.18:
                # the patch starts right under the shoulders: that is a bare chest or back, not a
                # midriff, and naming it as one would protect the wrong region
                ys = np.flatnonzero(blob.any(1))
                for name in ("chest", "back"):
                    put(name, blob & body, int(ys.min()), int(ys.max()))
                put_band("midriff", 0.45, 0.75)
            else:
                gap, gap_cols = blob, 1.0
    if gap_cols > 0.15 and gap.any():
        ys = np.flatnonzero(gap.any(1))
        put("midriff", gap, int(ys.min()), int(ys.max()))
    else:
        put_band("midriff", 0.45, 0.75)
    b.exposed_gap_cols = gap_cols
    b.midriff_source = how

    put_band("hip", 0.75, 1.05)

    if b.leg_origin is not None and b.leg_end is not None:
        lo, le, llen = b.leg_origin, b.leg_end, b.leg_len
        lx0, lx1 = int(min(lo[0], le[0])) - half, int(max(lo[0], le[0])) + half

        def leg_band(f0, f1):
            m = np.zeros_like(body)
            m[int(lo[1] + llen * f0):int(lo[1] + llen * f1) + 1,
              max(0, lx0):min(W, lx1 + 1)] = True
            return m & body

        for nm, f0, f1 in (("thigh", 0.0, 0.55), ("calf", 0.55, 1.0)):
            put(nm, leg_band(f0, f1), int(lo[1] + llen * f0), int(lo[1] + llen * f1))


# region. These are priors about a garment's NORMAL cut; the measured coverage above is what the
# photo actually shows, and only this describes the new garment.
GARMENTS: list[tuple[str, dict]] = [
    ("bikini", {"neck": None, "shoulder": 0.15, "chest": 0.30, "back": 0.10, "midriff": None,
                "hip": 0.25, "thigh": 0.10, "calf": None}),
    ("swimsuit", {"neck": None, "shoulder": 0.25, "chest": 0.55, "back": 0.35, "midriff": None,
                  "hip": 0.55, "thigh": 0.25, "calf": None}),
    ("crop", {"neck": None, "shoulder": 0.20, "chest": 0.40, "back": 0.15, "midriff": None,
              "hip": None, "thigh": None, "calf": None}),
    ("bandeau", {"neck": None, "shoulder": None, "chest": 0.30, "back": None, "midriff": None,
                 "hip": None, "thigh": None, "calf": None}),
    ("top", {"neck": None, "shoulder": 0.30, "chest": 0.65, "back": 0.45, "midriff": 0.30,
             "hip": None, "thigh": None, "calf": None}),
    ("longtop", {"neck": None, "shoulder": 0.35, "chest": 0.75, "back": 0.55, "midriff": 0.80,
                 "hip": 0.35, "thigh": None, "calf": None}),
    ("dress", {"neck": None, "shoulder": 0.35, "chest": 0.75, "back": 0.60, "midriff": 0.85,
               "hip": 0.90, "thigh": 0.35, "calf": None}),
    # A saree is not a dress. It is a blouse over a bare midriff and a drape over the hip, and the
    # bare midriff is the whole point of the garment - so midriff is None here, not 0.85. Treating it
    # as a dress is what paints fabric across the stomach and invents a covered waist.
    ("saree", {"neck": None, "shoulder": None, "chest": None, "back": None, "midriff": None,
               "hip": 0.95, "thigh": 0.60, "calf": 0.25}),
    ("blouse", {"neck": None, "shoulder": 0.35, "chest": 0.50, "back": 0.20, "midriff": None,
                "hip": None, "thigh": None, "calf": None}),
    ("maxi", {"neck": None, "shoulder": 0.35, "chest": 0.80, "back": 0.65, "midriff": 0.90,
              "hip": 1.00, "thigh": 0.80, "calf": 0.40}),
    ("shorts", {"neck": None, "shoulder": None, "chest": None, "back": None, "midriff": None,
                "hip": 0.95, "thigh": 0.35, "calf": None}),
    ("skirt", {"neck": None, "shoulder": None, "chest": None, "back": None, "midriff": 0.15,
               "hip": 0.95, "thigh": 0.30, "calf": None}),
    ("trousers", {"neck": None, "shoulder": None, "chest": None, "back": None, "midriff": None,
                  "hip": 1.00, "thigh": 0.95, "calf": 0.85}),
    ("jacket", {"neck": None, "shoulder": 0.95, "chest": 0.60, "back": 0.80, "midriff": 0.10,
                "hip": 0.10, "thigh": None, "calf": None}),
]

# Prompt words -> garment kind. Order is not enough on its own: 'a white crop top' contains both
# 'crop top' and 'top', and the general 'top' would win and cover the midriff the crop top is
# named for, so a specific kind also silences the general ones.
GARMENT_PATTERNS: list[tuple[str, str]] = [
    (r"\bbikini\b|\btwo-?piece\b|\bbikini set\b|\bthong\b", "bikini"),
    (r"\bone-?piece\b|\bswimsuit\b|\bswim ?wear\b|\bbathing suit\b|\bmonokini\b", "swimsuit"),
    (r"\bcrop ?top\b|\bcropped top\b|\bcut-?out top\b", "crop"),
    (r"\bbandeau\b|\bstrapless\b|\btube top\b|\btube\b|\bbra\b|\bbralette\b", "bandeau"),
    (r"\bmaxi\b|\bgown\b|\bfloor-?length\b|\bankle-?length\b", "maxi"),
    (r"\bsarees?\b|\bsaris?\b", "saree"),
    (r"\bblouses?\b", "blouse"),
    (r"\bshorts\b|\bhot ?pants?\b", "shorts"),
    (r"\bdress(?:es)?\b|\bsalwar\b|\bcheongsam\b|\bkaftan\b|\bkimono\b", "dress"),
    (r"\bskirt\b", "skirt"),
    (r"\btrousers\b|\bpants\b|\bjeans\b|\bleggings\b|\bjoggers?\b|\bsweatpants\b|\bpalazzos?\b",
     "trousers"),
    (r"\bjackets?\b|\bcoats?\b|\bblazers?\b|\bhoodies?\b|\bwindbreakers?\b|"
     r"\bcardigans?\b|\bjumpers?\b|\bparkas?\b", "jacket"),
    (r"\blong (?:top|tunic|shirt)\b|\blongline\b|\btunic\b", "longtop"),
    (r"\bshirts?\b|\bt-?shirts?\b|\btees?\b|\btops?\b|\bkurtis?\b|"
     r"\bsweaters?\b|\bsweatshirts?\b|\btank\b|\bcamisole\b|\bcami\b", "top"),
]

_COMPILED = [(re.compile(p, re.I), kind) for p, kind in GARMENT_PATTERNS]
_SPECIFIC = {"bikini", "swimsuit", "crop", "bandeau", "maxi", "shorts", "trousers", "longtop",
            "saree", "blouse"}

# A broad word is dropped when a narrower one that already covers the same ground is also in the
# prompt: 'a crop top' must not become 'a top' as well, or the midriff gets painted. This is a
# per-group table on purpose rather than 'any specific kind wins', because most kinds are specific
# about a different part of the body: in 'a crop top and a long skirt' the crop top says nothing
# about the skirt, and dropping it would leave the hips uncovered.
_SUPPRESSED_BY = {
    "top": {"crop", "bandeau", "bikini", "swimsuit", "longtop"},
    "skirt": {"dress", "maxi", "saree"},
    "trousers": {"dress", "maxi"},
}


def garments_in(text: str) -> list[str]:
    """The garment kinds a prompt asks for, most specific first.

    'a red bikini top with matching shorts' -> ['bikini', 'shorts']: one prompt can name several
    pieces. A general word ('top') is dropped when a specific garment already accounts for it, so
    'a crop top' does not also become 'a normal top' and cover the midriff."""
    hits = []
    for rx, kind in _COMPILED:
        m = rx.search(text)
        if m:
            hits.append((m.start(), kind))
    hits.sort()
    kinds: list[str] = []
    for _, kind in hits:
        if kind in kinds:
            continue
        if kind in _SUPPRESSED_BY and _SUPPRESSED_BY[kind] & set(kinds):
            continue
        kinds.append(kind)
    # A 'suit' is two pieces, not one garment, so it expands rather than matching a kind of its own:
    # a jacket over the torso and trousers over the legs. Without this a "navy suit" parses as a
    # single 'top' and the trousers are never covered.
    if re.search(r"\bsuit\b", text, re.I) and not re.search(r"\bswimsuit\b|\bbathing suit\b",
                                                            text, re.I):
        for k in ("jacket", "trousers"):
            if k not in kinds:
                kinds.append(k)
    return kinds


def requirement(text: str) -> tuple[dict[str, float], list[str]]:
    """Which regions each requested garment kind has to cover, and the kinds in order."""
    kinds = garments_in(text)
    req: dict[str, float] = {}
    for kind in kinds:
        for region, frac in next(g for k, g in GARMENTS if k == kind).items():
            if frac is not None:
                req[region] = max(req.get(region, 0.0), frac)
    return req, kinds


def measured_coverage(b: BodyMap) -> dict[str, float | None]:
    """What this photo's clothes cover, per region. None = the region is not in the picture."""
    return {r: b.coverage(r) for r in REGIONS if r in b.regions and b.regions[r].any()}


def plan(b: BodyMap, text: str) -> dict:
    """The regions to repaint for `text`, given what this photo is wearing.

    repaint    regions the mask must cover
    keep_bare  regions the new garment does not reach, so the mask must NOT be widened into them
    measured   cloth fraction per region in this photo
    exposed    bare-skin fraction per region in this photo
    required   what the prompt asks for, per region
    kinds      the garment kinds recognised
    confident  False when there was no usable pose, so the regions are a guess

    The keep_bare half is what stops a crop-top prompt from painting the whole torso: a crop does
    not reach the midriff, so the midriff is never widened into whatever the old clothes did there.
    It is also what keeps a saree's bare midriff bare when the old garment was a dress.

    A region the new garment reaches but this photo shows no body for is left out entirely: the
    model cannot repaint a hip that is out of frame, and widening the mask to reach one would paint
    over whatever is at the edge instead."""
    measured = measured_coverage(b)
    required, kinds = requirement(text)

    repaint: set[str] = set()
    keep_bare: set[str] = set()
    if kinds:
        # The prompt named garments, so those decide what changes. The hip is left alone when only a
        # crop top was asked for, because the skirt she is wearing is not part of the request - the
        # alternative, repainting every region that currently has cloth on it, replaces a skirt
        # nobody asked to replace.
        for region in required:
            if region in b.regions and b.regions[region].any():
                repaint.add(region)
    else:
        # Nothing recognisable in the prompt: keep today's behaviour and repaint whatever is worn,
        # which is what the pipeline did before this module existed.
        repaint = {r for r, v in measured.items() if v is not None and v > 0.02}

    for region in measured:
        if region not in b.regions or not b.regions[region].any():
            continue
        r = required.get(region)
        if r is None or r < 0.12:
            keep_bare.add(region)        # the new garment does not reach here

    off_frame = {r for r in REGIONS
                 if r not in b.regions or not b.regions[r].any()}
    return {"repaint": repaint, "keep_bare": keep_bare, "off_frame": off_frame,
            "measured": measured, "exposed": dict(b.exposed), "required": required,
            "kinds": kinds, "midriff_source": b.midriff_source,
            "facing": b.facing, "confident": b.source == "pose", "source": b.source}
