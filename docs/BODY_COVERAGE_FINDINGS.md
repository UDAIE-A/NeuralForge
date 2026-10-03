# Body-shape / coverage work — findings

Measured on the 12 photos in `C:\Users\udayk\Videos\AnyDesk\train` with `analyze_person.py`
(SegFormer clothes + SAM 2.1 + MediaPipe), 2026-10-03.

## Done and verified

**Face detection: 3/12 -> 11/12.** `detect_face()` now takes the pose, aims a crop at the head
(nose + shoulder span), de-rotates by the shoulder-line tilt, and retries at 1.0x/1.8x/2.6x box
size with the detector's confidence floor at 0.5 then 0.0. Only `photo_0077` (full profile,
looking away) still fails — a genuine limit of MediaPipe's detector, not a bug.

Why it mattered: SegFormer's `face` label bleeds down the neck onto the chest. With no face mesh,
the whole chest is "face" and therefore permanently protected, so a bikini top can never be drawn
over it. This is the root cause of most "the garment looks fake" complaints.

**Crop top no longer swallowed by the skirt: `split_top_from_bottom()` in `refine_masks.py`.**
`clean_semantics` step 1 takes the majority label of each connected cloth region. A crop top and
the skirt under it are ONE region (the hem and waistband abut; an arm covers the midriff), so the
skirt wins and `--parts upper` finds nothing. The splitter finds the hem per column and cuts one
line across the blob.

Result on the set: 7/12 photos with both parts detected -> 8/12, and the two that flipped
(`photo_0026`, `photo_0021`) went from `NO-UPPER` to a correct 3.7% / 2.5% upper.
No pixels are removed - the top keeps every pixel, it just stops being glued to the skirt.
Regression check: bikini mask 34.8% and t-shirt 31.3%, identical to before the change.

## Still broken, with the measured reason

`photo_0008`, `photo_0033` — NO-UPPER.
`photo_0099` — NO-LOWER (a one-piece swimsuit; SegFormer calls it `upper_clothes`, and there is
  no lower garment in the frame, so this is arguably correct labelling, not a bug).

- `photo_0008`: top and skirt are already 2 blobs, but EACH blob contains both classes
  (blob1 UP:49913/SK:27833, blob2 UP:189014/SK:800746). The hem is diagonal across a leaning
  body, so per-column cuts fall outside each blob's own row span. A median cut cannot fix a
  diagonal boundary that crosses blob boundaries.
- `photo_0033`: the top IS pure `upper` already but is tiny (8517px, 0.59% of the image) and
  sits at rows 731-873 while the skirt is a separate blob at rows 918-1611. It survives
  clean_semantics but `remove_specks` / SAM pass 3 / edge snapping then eat it.

Both need work in the SAM/edge-snapping stage, not in the label merge.

## Do NOT re-try these (measured, they do not work)

- Per-row skin-gap test: a midriff on a leaning body is a DIAGONAL band, not horizontal.
- Per-column skin-gap test: the midriff is labelled `face` (id 11) by SegFormer, not skin. In
  `photo_0026` the midriff band is 38% `face` by row area.
- Cutting at the hem+gap: in `photo_0026` the crop top's hem (label 4) DIRECTLY ABUTS the skirt's
  waistband (label 5) with zero skin rows between — the arm and the sunglasses cover the midriff.
  There is no gap to find. Only 96 of 564 top columns have a clean ordered boundary, so cutting
  only those leaves the two blobs joined.

## Skin tone

`coverage.skin_tone()` measures the subject's skin from the pixels the segmentation already calls
skin, trimmed to the middle of the luminance range so specular highlight and cast shadow do not set
the colour. It returns a depth (fair -> very deep), an undertone from red-minus-blue, and a light
description from the luminance spread.

Measured on the train set: `medium warm skin under directional light (median rgb 132,96,79)` on
photo_0026, `very fair warm skin under hard directional light (238,207,157)` on photo_0033. The
spread between those two is the point - a fixed "medium brown" would be wrong for both, which is why
it is measured rather than defaulted.

Wired into `prompt_for()`. Opt out with `--no-skin-tone`.

## The coverage map (built — `scripts/coverage.py`, tested by `scripts/test_coverage.py`)

The map answers two separate questions, which were being conflated:

- **measured** — where the cloth IS in this photo, per body region
- **required** — where the NEW garment has to be, from the prompt

`plan()` combines them into `repaint` / `keep_bare`. `keep_bare` is the half that matters: a crop top
does not reach the midriff, so the mask is never widened into whatever the old clothes did there.

Bare-ness is read from **label semantics, not geometry**. `refine_masks` already labels exposed
neck/chest skin as id 18 and the limbs as 12-15, so "bare" is a label test with no threshold to tune:
a bare midriff measures `exposed=1.00` and the same band inside a dress measures `0.07`.

The midriff is located by the **waist gap** between the upper and lower garment pieces, with a
fallback to the largest connected patch of exposed skin for one-piece garments:

| photo | garment | source | covered | exposed |
|---|---|---|---|---|
| 0008 | backless tube top + skirt | skin blob | 0.00 | 1.00 |
| 0009 | halter dress | waist gap | 0.93 | 0.07 |
| 0015 | one-shoulder top + skirt | skin blob | 0.00 | 1.00 |
| 0021 | backless crop top + skirt | waist gap | 0.55 | 0.08 |
| 0026 | crop top + skirt | waist gap | 0.00 | 1.00 |
| 0033 | backless crop top + long skirt | skin blob | 0.00 | 1.00 |
| 0037 | halter dress | waist gap | 1.00 | 0.00 |
| 0104 | cut-out dress | skin blob | 0.00 | 1.00 |

### Approaches that measurably failed (do not re-try)

- **Fixed fractions of the shoulder-to-hip span.** The pose hip joints are the hip *joints*, well
  below the waist, so a "midriff band" at 0.34-0.68 of the span lands *inside the crop top*. In
  `photo_0026` the bare band is at 0.63-0.72 and the fixed band measured 0.86 covered.
- **Slab/slice grid along the torso axis.** A slab wider than it is deep means every slice sees
  almost the same pixels: 322px deep spaced 37px apart, so ~9 consecutive slices were identical and
  the cloth fraction never dipped. Invisible in a unit test, fatal on a real photo.
- **Largest cloth hole per column.** Picks up the arms: a column running down an arm has no cloth
  above or below it and looks exactly like a bare midriff. A plain dress reported a gap it does not
  have until the limbs were excluded from the column range.
- **Box edges to bound a hole.** The skirt runs on below the hip landmarks, so any column whose
  cloth reached the bottom of a box ending at the hips was discarded and every real gap missed.
- **`face['oval']` as face landmarks.** It is a per-pixel index map of shape `(H, W)`, not points.
  Passing it in made `facing` read 1.00 for every photo. Use `face['points']`, or the pose nose.

## Wired into `change_clothes.py`

`--mask-only` over the train set, comparing the mask with and without the plan:

| case | with plan | `--no-coverage` | gates |
|---|---|---|---|
| full / saree on 0026 | 16.9% | 17.9% | deny midriff |
| full / dress on 0026 | 12.9% | 13.3% | deny neck |
| upper / dress on 0026 | 3.7% | 5.5% | allow on, deny neck |
| upper / saree on 0104 | 18.0% | 19.4% | allow on, deny midriff |
| lower / skirt on 0026 | 8.4% | 8.4% | deny neck/shoulder |

Three things this established, each of which was wrong in the first attempt:

- **`allow` must not apply to a `--parts full` pass.** "Replace the whole outfit" means the skirt
  goes too, so clipping the mask to the regions the prompt names deleted the skirt and took the
  coverage from 13.3% to 6.3%. Restriction belongs on `--upper`/`--lower` only.
- **`allow` has to be built from the region SPANS, not the measuring bands.** The bands are 30% of
  torso width so the cloth fraction inside them is clean; using them to gate painting clips the
  garment to a stripe down the middle (kept only 22% of the clothing). Spans are for recall, the
  narrow masks are for precision, and only `deny` uses the narrow ones.
- **`deny` must skip the neck and shoulders when exposing.** The neck band overlaps the top of the
  chest, which is exactly where a bikini's straps go; denying it puts the old straps back.

`MIN_ALLOW_KEEP` (0.40) stops a misplaced region band from deleting a pass: if restricting the
repaint regions would keep under 40% of the pass's clothing, the restriction is reported and
dropped. This is what `photo_0033 --upper` hits - it has no upper-garment label at all.

### Regression

`bikini` / `t-shirt` / `strapless` / `swimsuit` all still run, with 0.1-0.7% of image changed - only
the bare regions that are now protected. The bikini mask keeps the chest exposed above the old
neckline (cups/straps) while leaving the bare midriff strip uncovered.

## Known limitation

- `photo_0021` midriff reads 0.55 covered: her arm crosses the waist, so only 0.08 of the torso
  columns show the gap. Reported by the test as `lim`, not silently tolerated.
- `photo_0033 --upper` finds no upper clothing (pre-existing: the label is missing), and fails the
  same way with `--no-coverage`.

## Next

- `photo_0077` still has no face (full profile defeats MediaPipe FaceLandmarker); `photo_0008` and
  `photo_0033` still have no `NO-UPPER` label, though both now measure their coverage correctly.
- An arm crossing the waist defeats the column-wise waist gap; the skin-blob fallback catches most
  of the rest but not that one.
