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

#: Reference focus_measure for a well-focused fundus photograph. Fitted by
#: scripts/fit_focus_reference.py over the 40 real APTOS captures that
#: scripts/fetch_calibration_images.py downloads, eight per grade.
#:
#: The fit has to keep every real capture above FOCUS_UNGRADABLE and push every
#: sigma-4 blur below it, which gives an interval rather than a value:
#: (0.2112, 0.2977]. This is its geometric midpoint. The separation is 1.4x -
#: the softest real capture measures 0.1042, the most contrasty sigma-4 blur
#: 0.0739.
#:
#: That interval is narrow, and worth knowing when tuning: on this set the
#: previous metric had no valid interval at all. Measuring absolute Laplacian
#: variance on the grey channel, the softest real capture scored 5.09 and the
#: most contrasty sigma-4 blur scored 5.40 - overlapping, so no single
#: threshold could both keep the real photographs and reject the blurs. The old
#: value of 15.0 in fact rejected the softest real capture in this set, which
#: only shows up when the fit is checked against 40 images rather than asserted.
FOCUS_REFERENCE_CONTRAST = 0.2508


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
    brightness or contrast statistics.

    Both terms of the threshold are relative to the frame's own levels, and
    that is load-bearing. An absolute floor drops the retina out of its own
    mask on an underexposed capture - measured, coverage fell from 0.904 to
    0.019 across a brightness sweep where the relative threshold held at 0.905
    - so the illumination statistics ended up averaging the brightest surviving
    sliver and stopped responding to exposure at all.

    The percentile term anchors to the retina rather than the frame, so a
    photograph with a large black surround is not thresholded down toward it. A
    frame with no bright region at all yields an empty mask, which the callers
    already report as "no retinal area detected".
    """
    gray = _to_gray(img)
    highlight = float(np.percentile(gray, 99))
    return gray > max(gray.mean() * 0.12, highlight * 0.06)


def _green(img: Image.Image) -> np.ndarray:
    """The green plane, where retinal structure lives.

    Red saturates in the choroid and blue records almost nothing from the
    retinal layers, so vessel and lesion contrast - the high-frequency content
    a focus measure is actually looking for - is carried by green.
    """
    return np.asarray(img.convert("RGB"), dtype=np.float64)[:, :, 1]


def focus_measure(img: Image.Image, mask: np.ndarray) -> float:
    """Relative high-frequency contrast inside the retina. Unbounded.

    A blurred image has little high-frequency energy, so the variance of the
    second derivative collapses. That is the signal that catches the defocused
    captures a shaky hand produces on a portable camera - but raw Laplacian
    variance also collapses when the flash is weak, because the Laplacian is a
    linear operator: scale intensity by k and its variance scales by k squared.
    Measured on real captures, a perfectly sharp photograph at 20% brightness
    scored 0.17 where the same photograph at full brightness scored 1.00, so
    the gate reported severe defocus and sent the health worker to adjust a
    focus ring that was never the problem.

    Dividing by mean retinal luminance removes most of that dependence. The
    exponent is 1, not the 2 the algebra suggests, and that is an empirical
    result rather than a derivation: 8-bit quantization contributes a floor to
    the Laplacian variance that does not scale with exposure, so a full square
    normalisation over-corrects at low light. Measured over 40 real APTOS
    captures by scripts/fit_focus_reference.py, sweeping the exponent:

        exponent   blur separation   spread over 1.0-0.2 brightness
          0.0          1.38x               7.72x
          1.0          1.41x               1.51x
          2.0          0.84x               3.46x

    Exponent 1 is the only setting that improves blur separation over the
    unnormalised metric while also holding steady across exposure. Note the
    residual 1.5x is upward - a dark capture now reads slightly sharper, not
    blurred - so the metric can no longer manufacture the false "severe
    defocus" that sent a health worker to refocus a sharp photograph.
    """
    green = _green(img)
    lap = (
        -4 * green[1:-1, 1:-1]
        + green[:-2, 1:-1]
        + green[2:, 1:-1]
        + green[1:-1, :-2]
        + green[1:-1, 2:]
    )
    inner = mask[1:-1, 1:-1]
    if inner.sum() < 100:
        return 0.0

    luminance = float(green[mask].mean())
    if luminance < 1.0:
        return 0.0
    return float(lap[inner].var()) / luminance


def focus_score(img: Image.Image, mask: np.ndarray) -> float:
    """focus_measure against the fitted reference, clipped to 0-1."""
    return float(
        np.clip(focus_measure(img, mask) / FOCUS_REFERENCE_CONTRAST, 0.0, 1.0)
    )


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

    # Multiplicative in exposure, not additive. An additive composition floors
    # at 0.50 for any even, unclipped frame however dark it is, which put
    # ILLUM_UNGRADABLE = 0.35 permanently out of reach and left a pitch-black
    # retina scoring half marks. A retina that cannot be seen is not half
    # quality, and no amount of evenness rescues a black frame.
    score = float(np.clip(exposure * (0.6 * evenness + 0.4 * (1 - clipped)), 0, 1))
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

    # The health worker acts on the first thing this list says, so it is
    # ordered by how far each metric sits below its own floor rather than by
    # the order the metrics happen to be computed in. Inserting focus at the
    # front unconditionally is what made a dark capture read "severe defocus,
    # underexposed" and sent them to refocus a sharp photograph.
    tagged: list[tuple[str, str]] = []
    if focus < FOCUS_UNGRADABLE:
        tagged.append(("focus", "severe defocus"))
    elif focus < FOCUS_BORDERLINE:
        tagged.append(("focus", "insufficient focus"))
    tagged += [("illumination", issue) for issue in illum_issues]
    tagged += [("field_of_view", issue) for issue in fov_issues]

    shortfall = {
        "focus": (FOCUS_UNGRADABLE - focus) / FOCUS_UNGRADABLE,
        "illumination": (ILLUM_UNGRADABLE - illum) / ILLUM_UNGRADABLE,
        "field_of_view": (FOV_UNGRADABLE - fov) / FOV_UNGRADABLE,
    }
    # Stable, so metrics that are equally far below their floor keep the order
    # above and a metric that is above its floor never outranks one below it.
    issues = [text for _, text in sorted(tagged, key=lambda t: -shortfall[t[0]])]

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
