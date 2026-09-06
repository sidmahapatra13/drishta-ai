"""Fetch the APTOS images the quality gate's focus constant is fitted against.

The gate's focus reference cannot be calibrated on the synthetic test fixture:
its per-pixel noise puts the Laplacian variance two orders of magnitude above a
real photograph's, and a constant fitted on it rejected 100% of real fundus
images. So the constant is fitted on real captures, and this script fetches
them.

Forty images, eight per grade, taken in a stable order from APTOS `train.csv`
so the set is identical on every machine and the fit is reproducible. They land
in gitignored `local-data/calibration/`; APTOS is competition data and
AGENTS.md keeps datasets out of git.

    python scripts/fetch_calibration_images.py
"""

import csv
import subprocess
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "local-data" / "calibration"
PER_GRADE = 8


def kaggle(*args: str) -> None:
    subprocess.run(
        ["uvx", "--from", "kaggle", "kaggle", "competitions", *args],
        check=True,
        capture_output=True,
    )


def unzip_into(directory: Path) -> None:
    for archive in directory.glob("*.zip"):
        with zipfile.ZipFile(archive) as z:
            z.extractall(directory)
        archive.unlink()


def train_labels() -> list[tuple[str, int]]:
    csv_path = DEST / "train.csv"
    if not csv_path.exists():
        kaggle("download", "-c", "aptos2019-blindness-detection",
               "-f", "train.csv", "-p", str(DEST))
        unzip_into(DEST)
    with csv_path.open() as fh:
        return [(r["id_code"], int(r["diagnosis"])) for r in csv.DictReader(fh)]


def selection() -> list[tuple[str, int]]:
    """The first PER_GRADE ids of each grade in file order.

    Deterministic rather than random: a calibration constant that moves when
    someone reruns the fetch is not a calibration constant.
    """
    by_grade: dict[int, list[str]] = defaultdict(list)
    for image_id, grade in train_labels():
        if len(by_grade[grade]) < PER_GRADE:
            by_grade[grade].append(image_id)
    return [(i, g) for g in sorted(by_grade) for i in by_grade[g]]


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    chosen = selection()

    for n, (image_id, grade) in enumerate(chosen, 1):
        target = DEST / f"grade{grade}_{image_id}.png"
        if target.exists():
            continue
        kaggle("download", "-c", "aptos2019-blindness-detection",
               "-f", f"train_images/{image_id}.png", "-p", str(DEST))
        unzip_into(DEST)
        (DEST / f"{image_id}.png").rename(target)
        print(f"  [{n}/{len(chosen)}] {target.name}")

    print(f"\n{len(chosen)} images in {DEST.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
