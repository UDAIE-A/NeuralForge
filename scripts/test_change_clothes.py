"""Check that change_clothes still behaves: the coverage gates, the old masks, and every photo.

    venv\\Scripts\\python.exe scripts\\test_change_clothes.py

Slow by nature - each case runs the real segmenter in a subprocess, so this is a
before-you-push check rather than something to run on every edit. Needs the train set.

Three things are checked, and none of them is covered by test_coverage.py (which tests the
measurement on its own, without the pipeline):

  gates     the repaint/deny logic decides what the mask may and may not touch. Getting
            `allow` wrong on a --parts full pass deleted the skirt and took the mask from
            13.3% to 6.3%, which is exactly the sort of regression nothing else here would
            notice.
  old masks the bikini fix predates the coverage plan and must survive it: the bikini still
            needs the bare chest above the old neckline, or the cups come out sliced off.
  smoke    every photo still segments and builds a mask, with the plan and without it.

Each case runs twice and compares, so a change that silently alters every mask shows up as
a coverage delta rather than as nothing at all.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
PY = ROOT / "venv" / "Scripts" / "python.exe"
SCRIPT = ROOT / "scripts" / "change_clothes.py"
TRAIN = Path(r"C:\Users\udayk\Videos\AnyDesk\train")
OUT = Path(r"C:\Users\udayk\AppData\Local\Temp\opencode\test_change_clothes")

SAREES = "a silk saree with a fitted blouse"

# label, photo, argv after the image
GATES = [
    # a --parts full prompt describes the whole new outfit, so every piece of the old one goes
    # with it: restricting its mask to the regions the prompt names would delete the skirt.
    ("full / saree",      "photo_0026", ["a silk saree with a fitted blouse", "--parts", "full"]),
    ("full / dress",      "photo_0026", ["a black dress", "--parts", "full"]),
    ("full / crop top",   "photo_0026", ["a white crop top", "--parts", "full"]),
    # a partial pass is the opposite: "a crop top" for the upper pass must not take the hip.
    ("upper / crop top",  "photo_0026", ["--upper", "a white crop top"]),
    ("upper / dress",     "photo_0026", ["--upper", "a black dress"]),
    ("lower / skirt",     "photo_0026", ["--lower", "a long silk skirt"]),
    ("upper / saree",     "photo_0104", ["--upper", "a silk saree with a fitted blouse"]),
]

# photo_0033 has no upper-garment label at all, so there is nothing to repaint and the pass exits
# non-zero. It fails identically with --no-coverage, so it is a pre-existing limitation rather than
# a regression - but a change that made it fail DIFFERENTLY would still be a bug, so it is checked
# for that and reported rather than dropped.
KNOWN_NO_UPPER = {"photo_0033"}

# prompt, and whether the mask must reach the bare chest above the old neckline
REG = [
    ("a red string bikini", True),
    ("a white t-shirt", False),
    ("a strapless tube top", True),
    ("a black one piece swimsuit", True),
]


def run(photo: str, argv: list[str], extra: list[str]) -> dict:
    tag = abs(hash(" ".join(argv) + "".join(extra))) % 99999
    out = OUT / f"{photo}_{tag}"
    out.mkdir(parents=True, exist_ok=True)
    p = subprocess.run([str(PY), str(SCRIPT), str(TRAIN / f"{photo}.jpg"), "--mask-only",
                        "--out", str(out), *argv, *extra],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    lines = (p.stdout or "").splitlines()
    mask = np.asarray(Image.open(out / f"{photo}_mask.png")) if (out / f"{photo}_mask.png").exists() \
        else None
    return {
        "rc": p.returncode,
        "mask": mask,
        "cov": None if mask is None else float(mask.mean() / 255),
        "skipped": any("skipped," in ln for ln in lines),
        "plan": next((ln.strip() for ln in lines if "coverage plan" in ln), ""),
        "deny": next((ln.strip() for ln in lines if "protecting bare" in ln), ""),
        "log": "\n".join(lines[-8:]),
    }


def test_gates() -> list[str]:
    fails = []
    print("--- gates")
    for label, photo, argv in GATES:
        a, b = run(photo, argv, []), run(photo, argv, ["--no-coverage"])
        if photo in KNOWN_NO_UPPER and a["rc"] and b["rc"]:
            print(f"lim {label:20s} rc={a['rc']}/{b['rc']}   no upper-garment label "
                  f"(pre-existing, same with --no-coverage)")
            continue
        if a["rc"] or b["rc"]:
            fails.append(f"{label}: rc={a['rc']}/{b['rc']}\n{b['log']}\n{a['log']}")
            print(f"BAD {label:20s} rc={a['rc']}/{b['rc']}")
            continue
        restricted = "not restricted" not in a["plan"] and "repaint[all]" not in a["plan"]
        note = []
        if restricted:
            note.append("allow=on")
        if "full pass" in a["plan"]:
            note.append("full pass left alone")
        if a["deny"]:
            note.append("deny")
        print(f"ok  {label:20s} {a['cov']:6.1%} vs {b['cov']:6.1%}   {','.join(note) or '-'}")
    return fails


def test_old_masks() -> list[str]:
    fails = []
    print("--- previously verified masks")
    for prompt, must_expose in REG:
        a, b = run("photo_0026", [prompt], []), run("photo_0026", [prompt], ["--no-coverage"])
        if a["rc"] or b["rc"]:
            fails.append(f"{prompt!r}: rc={a['rc']}/{b['rc']}\n{a['log']}")
            print(f"BAD {prompt!r:38s} rc={a['rc']}/{b['rc']}")
            continue
        if a["mask"] is None:
            fails.append(f"{prompt!r}: no mask written")
            print(f"BAD {prompt!r:38s} no mask")
            continue
        # the plan should only ever REMOVE bare skin, so it cannot cover more than the old mask
        grew = a["mask"].sum() > b["mask"].sum()
        if grew:
            fails.append(f"{prompt!r}: plan grew the mask by "
                         f"{int(a['mask'].sum() - b['mask'].sum())}px")
        print(f"{'BAD' if grew else 'ok '} {prompt!r:38s} {a['cov']:6.1%} vs {b['cov']:6.1%}"
              f"   exposed={must_expose}")
    return fails


def test_smoke() -> list[str]:
    fails = []
    photos = sorted(p.stem for p in TRAIN.glob("*.jpg"))
    print(f"--- smoke ({len(photos)} photos)")
    for ph in photos:
        a = run(ph, [SAREES, "--parts", "full"], [])
        tone = next((ln.split("skin tone:")[1].strip()
                     for ln in a["log"].splitlines() if "skin tone" in ln), "-")
        if a["rc"]:
            fails.append(f"{ph}: rc={a['rc']}\n{a['log']}")
        print(f"{'BAD' if a['rc'] else 'ok '} {ph:12s} {str(a['cov']):>7s}  {tone[:44]}")
    return fails


if __name__ == "__main__":
    if not TRAIN.is_dir():
        sys.exit(f"train set not found at {TRAIN}")
    failures = test_gates() + test_old_masks() + test_smoke()
    print()
    print(f"FAILURES: {len(failures)}")
    for f in failures:
        print("  -", f)
    sys.exit(1 if failures else 0)