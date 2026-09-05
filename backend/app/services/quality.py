"""Image quality assessment - PS requirement 1.

Real implementation, not a stub. These are classical measurements that need no
training data, which is why this module is finished on day one while the
classifier is still a stub.

The MATLAB port (matlab/quality_assessment/) mirrors these signals exactly so
the two implementations stay comparable; this Python version is the fallback
that keeps the demo running if the MATLAB engine is unavailable.
"""

import numpy as np
from PIL import Image

from ..contract import QualityReport, QualityStatus

# Thresholds are deliberately named and grouped so they can be tuned against
# the synthetic-degradation set rather than being scattered magic numbers.
FOCUS_UNGRADABLE = 0.35
FOCUS_BORDERLINE = 0.60
ILLUM_UNGRADABLE = 0.35
FOV_UNGRADABLE = 0.45
SCORE_UNGRADABLE = 0.45
SCORE_BORDERLINE = 0.70

#: Metrics are measured at this longest side, never at native resolution.
#: Laplacian variance scales with pixel count, so the same eye photographed at
#: 4288px and at 640px scores very differently - and a rural programme mixes
#: camera models. Normalising size first is what makes focus comparable between
#: devices rather than a measure of which camera took the picture.
WORK_RESOLUTION = 1024

#: Laplacian variance of a well-focused fundus photograph at WORK_RESOLUTION.
#: Calibrated on 39 APTOS images spanning all five grades against programmatic
#: degradations of the same images: at 15.0 every real photograph is kept and
#: every Gaussian blur of sigma 4 or above is rejected. The valid range is
#: 12-18; 15 sits in the middle of it.
#:
#: The previous value of 500.0 was calibrated against the synthetic test
#: fixture, whose per-pixel noise gives a variance near 4375 where a real
#: photograph gives 6-41. It rejected 100% of real fundus images.
FOCUS_REFERENCE_VARIANCE = 15.0


def _to_gray(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("L"), dtype=np.float64)


def _working(img: Image.Image) -> Image.Image:
    """Downscale to WORK_RESOLUTION so the measurements mean the same thing on
    every camera. Enlarging is pointless - it invents no detail - so an image
    already at or below the working size is measured as it is."""
    longest = max(img.size)
    if longest <= WORK_RESOLUTION:
        return img
    scale = WORK_RESOLUTION / longest
    return img.resize(
        (max(1, round(img.width * scale)), max(1, round(img.height * scale))),
        Image.BILINEAR,
    )


def _retina_mask(img: Image.Image) -> np.ndarray:
    """The circular retinal region. Fundus photographs sit on a black surround,
    so everything outside this mask is padding and must not count toward
    brightness or contrast statistics."""
    gray = _to_gray(img)
    return gray > max(12.0, gray.mean() * 0.12)


def focus_score(img: Image.Image, mask: np.ndarray) -> float:
    """Laplacian variance inside the retina, normalized to 0-1.

    A blurred image has little high-frequency energy, so the variance of the
    second derivative collapses. This is the signal that catches the defocused
    captures a shaky hand produces on a portable camera.
    """
    gray = _to_gray(img)
    lap = (
        -4 * gray[1:-1, 1:-1]
        + gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
    )
    inner = mask[1:-1, 1:-1]
    if inner.sum() < 100:
        return 0.0
    variance = float(lap[inner].var())
    return float(np.clip(variance / FOCUS_REFERENCE_VARIANCE, 0.0, 1.0))


def illumination_score(img: Image.Image, mask: np.ndarray) -> tuple[float, list[str]]:
    """Penalize both under- and over-exposure, and uneven illumination.

    Returns the score plus the specific problems found, because 'too dark' and
    'washed out' need different recapture advice for the health worker.
    """
    gray = _to_gray(img)
    if mask.sum() < 100:
        return 0.0, ["no retinal area detected"]

    pixels = gray[mask]
    mean = pixels.mean() / 255.0
    issues: list[str] = []

    # Ideal mean exposure sits near 0.45; score falls off linearly either side.
    exposure = 1.0 - min(abs(mean - 0.45) / 0.45, 1.0)
    if mean < 0.20:
        issues.append("underexposed")
    elif mean > 0.75:
        issues.append("overexposed")

    # Saturation clipping destroys lesion detail even when the mean looks fine.
    clipped = float((pixels > 250).sum() / pixels.size)
    if clipped > 0.10:
        issues.append("blown highlights")

    # Uneven illumination: compare the brightness of image halves.
    h = gray.shape[0]
    top, bottom = gray[: h // 2], gray[h // 2 :]
    tm, bm = mask[: h // 2], mask[h // 2 :]
    evenness = 1.0
    if tm.sum() > 50 and bm.sum() > 50:
        delta = abs(top[tm].mean() - bottom[bm].mean()) / 255.0
        evenness = 1.0 - min(delta / 0.35, 1.0)
        if delta > 0.28:
            issues.append("uneven illumination")

    score = float(np.clip(0.5 * exposure + 0.3 * evenness + 0.2 * (1 - clipped), 0, 1))
    return score, issues


def field_of_view_score(img: Image.Image, mask: np.ndarray) -> tuple[float, list[str]]:
    """How much of the frame the retina fills, and whether it is centred.

    A partially captured retina can hide the macula entirely, which is exactly
    where referable disease shows up first.
    """
    coverage = float(mask.mean())
    issues: list[str] = []

    # A correctly framed fundus image fills roughly half the frame. Score the
    # coverage against that expectation rather than rewarding a full frame,
    # which usually means the camera was too close.
    fill = float(np.clip(coverage / 0.55, 0.0, 1.0))
    if coverage < 0.25:
        issues.append("retina too small in frame")

    # Centring: the retinal centroid should sit near the middle of the image.
    centring = 1.0
    if mask.sum() > 100:
        ys, xs = np.nonzero(mask)
        cy = ys.mean() / mask.shape[0]
        cx = xs.mean() / mask.shape[1]
        offset = float(np.hypot(cy - 0.5, cx - 0.5))
        centring = 1.0 - min(offset / 0.35, 1.0)
        if offset > 0.22:
            issues.append("retina off-centre")

    return float(np.clip(0.65 * fill + 0.35 * centring, 0, 1)), issues


def assess_quality(img: Image.Image) -> QualityReport:
    """Grade an image as gradable, borderline, or ungradable.

    Borderline images are the ones worth enhancing (see enhancement.py);
    ungradable images are rejected outright and never reach the classifier.
    """
    img = _working(img)
    mask = _retina_mask(img)

    focus = focus_score(img, mask)
    illum, illum_issues = illumination_score(img, mask)
    fov, fov_issues = field_of_view_score(img, mask)

    issues = list(illum_issues + fov_issues)
    if focus < FOCUS_UNGRADABLE:
        issues.insert(0, "severe defocus")
    elif focus < FOCUS_BORDERLINE:
        issues.insert(0, "insufficient focus")

    # Focus is weighted highest: a sharp but dim image can be corrected by
    # enhancement, whereas detail lost to blur cannot be recovered.
    score = float(np.clip(0.45 * focus + 0.30 * illum + 0.25 * fov, 0, 1))

    if (
        score < SCORE_UNGRADABLE
        or focus < FOCUS_UNGRADABLE
        or illum < ILLUM_UNGRADABLE
        or fov < FOV_UNGRADABLE
    ):
        status = QualityStatus.UNGRADABLE
    elif score < SCORE_BORDERLINE:
        status = QualityStatus.BORDERLINE
    else:
        status = QualityStatus.GRADABLE

    return QualityReport(
        focus=round(focus, 4),
        illumination=round(illum, 4),
        field_of_view=round(fov, 4),
        score=round(score, 4),
        status=status,
        issues=issues,
    )
