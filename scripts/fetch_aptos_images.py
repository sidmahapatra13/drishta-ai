"""Fetch the full APTOS training set, for the ablation and the failure gallery.

`fetch_calibration_images.py` pulls forty images one request at a time, which is
right for forty and hopeless for 3,662 - the API round trip dominates. This
takes the competition archive in one request instead and keeps only
`train_images/`; the test images have no public labels and nothing here can use
them.

About 850 MB extracted, into gitignored `local-data/aptos-full/`. APTOS is
competition data: AGENTS.md keeps it out of git, and so does its licence.
Removing it is `rm -rf local-data/aptos-full`.

    python scripts/fetch_aptos_images.py
"""

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "local-data" / "aptos-full"
IMAGES = DEST / "train_images"
COMPETITION = "aptos2019-blindness-detection"

#: The count APTOS ships. Anything else means a partial extract, which would
#: silently shrink the ablation rather than fail.
EXPECTED = 3662


def main() -> int:
    if IMAGES.is_dir() and len(list(IMAGES.glob("*.png"))) == EXPECTED:
        print(f"{EXPECTED} images already in {IMAGES.relative_to(ROOT)}")
        return 0

    DEST.mkdir(parents=True, exist_ok=True)
    archive = DEST / f"{COMPETITION}.zip"

    if not archive.exists():
        print("downloading the competition archive (one request, ~1 GB)")
        subprocess.run(
            ["uvx", "--from", "kaggle", "kaggle", "competitions", "download",
             "-c", COMPETITION, "-p", str(DEST)],
            check=True,
        )

    print("extracting train_images/")
    with zipfile.ZipFile(archive) as z:
        members = [m for m in z.namelist() if m.startswith("train_images/")]
        if not members:
            sys.exit("archive has no train_images/ - the competition layout changed")
        z.extractall(DEST, members=members)

    found = len(list(IMAGES.glob("*.png")))
    if found != EXPECTED:
        sys.exit(f"extracted {found} images, expected {EXPECTED}")

    archive.unlink()
    shutil.rmtree(DEST / "__MACOSX", ignore_errors=True)
    print(f"{found} images in {IMAGES.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
