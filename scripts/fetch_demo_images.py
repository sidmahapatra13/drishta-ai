"""Build the demo image set from real APTOS photographs.

The set is not committed. APTOS is competition data and AGENTS.md keeps
datasets out of git, so this fetches into `local-data/demo/`, which is ignored.

The four ids below are not arbitrary. They are the most unambiguous correctly
predicted case of each grade in `experiments/classification/oof_predictions.npy`,
ranked by distance from the nearest fitted cut point, so none of them sits near
a boundary where a rounding difference would change the grade on stage.

Between them they exercise every path in the pipeline:

    4134b290f5f3   grade 0   gradable, not referable
    0dbaa09a458c   grade 2   gradable, referable - the screening threshold
    4d47300e3ddb   grade 3   borderline capture: enhanced, re-assessed, then graded
    838c87c63422   grade 4   urgent
    <the above, defocused>   ungradable: the gate refuses it and asks for a recapture

The borderline case needed no help - that photograph really is soft, and the
enhancement path is exercised on a real capture rather than a synthetic one.
Only the ungradable case is derived, because APTOS has no image bad enough and
a health worker's blurred capture is honestly reproduced by defocusing a real
one.

    python scripts/fetch_demo_images.py
"""

import subprocess
import sys
import zipfile
from pathlib import Path

from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "local-data" / "demo"

CASES = {
    "4134b290f5f3": "grade 0 - gradable, not referable",
    "0dbaa09a458c": "grade 2 - gradable, referable",
    "4d47300e3ddb": "grade 3 - borderline capture, enhanced then graded",
    "838c87c63422": "grade 4 - urgent",
}

#: Defocus applied to the grade 3 capture to build the recapture case. Radius 4
#: measures focus 0.138 against the gate's 0.35 ungradable floor - unambiguously
#: rejected, without being so blurred that it looks like a synthetic fixture.
BLUR_RADIUS = 4
BLUR_SOURCE = "4d47300e3ddb"


def fetch(image_id: str) -> Path:
    """Download one APTOS training image. Kaggle serves some of them zipped."""
    png = DEST / f"{image_id}.png"
    if png.exists():
        return png

    subprocess.run(
        [
            "uvx", "--from", "kaggle", "kaggle", "competitions", "download",
            "-c", "aptos2019-blindness-detection",
            "-f", f"train_images/{image_id}.png",
            "-p", str(DEST),
        ],
        check=True,
    )

    archive = DEST / f"{image_id}.png.zip"
    if archive.exists():
        with zipfile.ZipFile(archive) as z:
            z.extractall(DEST)
        archive.unlink()
    return png


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)

    for image_id, purpose in CASES.items():
        fetch(image_id)
        print(f"  {image_id}.png  {purpose}")

    ungradable = DEST / "ungradable-defocus.png"
    source = Image.open(DEST / f"{BLUR_SOURCE}.png")
    source.filter(ImageFilter.GaussianBlur(BLUR_RADIUS)).save(ungradable)
    print(f"  {ungradable.name}  ungradable - gate refuses it, recapture requested")

    print(f"\n{len(CASES) + 1} images in {DEST.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
