"""Fit quality.FOCUS_REFERENCE_CONTRAST against real APTOS captures.

The focus score is `focus_measure(image) / FOCUS_REFERENCE_CONTRAST`, clipped
to 0-1, and the gate rejects an image whose focus falls below
`FOCUS_UNGRADABLE`. So the reference has to satisfy two conditions at once:

    every real capture is kept       ->  R <= min(measure over real) / FLOOR
    every sigma>=4 blur is rejected  ->  R >  max(measure over blurred) / FLOOR

which gives a valid interval rather than a single value. This script reports
that interval and picks its geometric midpoint, since the quantity is a ratio.

An empty interval means the measure does not separate real captures from
blurred ones on this set, and is reported as a failure rather than papered over
with a value that satisfies neither condition.

    python scripts/fetch_calibration_images.py
    python scripts/fit_focus_reference.py
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.services import quality  # noqa: E402

IMAGES = ROOT / "local-data" / "calibration"

#: The blur floor the gate must catch. Applied at native resolution, which is
#: how a real defocused capture arrives.
REJECT_FROM_SIGMA = 4
SIGMAS = [1, 2, 3, 4, 6, 8]

#: Exposure levels the fitted reference must hold across. A sharp capture must
#: score the same whether the flash was strong or weak - that is the property
#: the normalisation exists for, so the fit is checked against it.
EXPOSURES = [1.0, 0.6, 0.4, 0.2]


def measure(img: Image.Image) -> float:
    work = quality._working(img)
    return quality.focus_measure(work, quality._retina_mask(work))


def dim(img: Image.Image, factor: float) -> Image.Image:
    arr = np.asarray(img.convert("RGB"), dtype=np.float64) * factor
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def main() -> int:
    paths = sorted(IMAGES.glob("grade*.png"))
    if not paths:
        print(f"No calibration images in {IMAGES}.")
        print("Run: python scripts/fetch_calibration_images.py")
        return 1

    sharp: list[float] = []
    blurred: dict[int, list[float]] = {s: [] for s in SIGMAS}
    exposure_spread: list[float] = []

    for path in paths:
        img = Image.open(path)
        sharp.append(measure(img))
        for sigma in SIGMAS:
            blurred[sigma].append(measure(img.filter(ImageFilter.GaussianBlur(sigma))))

        across = [measure(dim(img, e)) for e in EXPOSURES]
        exposure_spread.append(max(across) / min(across))

    floor = quality.FOCUS_UNGRADABLE
    upper = min(sharp) / floor                       # keep every real capture
    lower = max(blurred[REJECT_FROM_SIGMA]) / floor  # reject every sigma-4 blur

    print(f"{len(paths)} real captures, focus_measure:")
    print(f"  min {min(sharp):.4f}   median {np.median(sharp):.4f}   max {max(sharp):.4f}")
    print("\nBlurred, by sigma:")
    for sigma in SIGMAS:
        vals = blurred[sigma]
        mark = "  <- must be rejected" if sigma >= REJECT_FROM_SIGMA else ""
        print(f"  sigma {sigma}:  min {min(vals):.4f}   max {max(vals):.4f}{mark}")

    print("\nExposure invariance (max/min of focus_measure across "
          f"{EXPOSURES[0]:.1f}-{EXPOSURES[-1]:.1f} brightness):")
    print(f"  worst image {max(exposure_spread):.4f}x   median "
          f"{np.median(exposure_spread):.4f}x   (1.0000x is exact)")

    print(f"\nValid interval for FOCUS_REFERENCE_CONTRAST: ({lower:.4f}, {upper:.4f}]")
    if lower >= upper:
        print("\nEMPTY. The measure does not separate real captures from sigma-4")
        print("blurs on this set. Do not pick a value; the metric needs work.")
        return 1

    fitted = float(np.sqrt(lower * upper))
    print(f"Geometric midpoint: {fitted:.4f}")
    print(f"\nSeparation: sharpest blur {max(blurred[REJECT_FROM_SIGMA]):.4f} vs "
          f"softest real capture {min(sharp):.4f}  "
          f"({min(sharp) / max(blurred[REJECT_FROM_SIGMA]):.1f}x)")
    print(f"\nFOCUS_REFERENCE_CONTRAST = {fitted:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
