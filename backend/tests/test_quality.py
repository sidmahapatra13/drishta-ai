"""Quality gate properties.

The gate is the most-watched moment of the demo and the one place where a wrong
message sends a health worker to do the wrong thing. These tests pin the
properties that make its output trustworthy, separately from the calibration
constants, which can only be fitted against real photographs.

The synthetic fixture cannot tell us anything about calibration - its per-pixel
noise puts its Laplacian variance two orders of magnitude above a real
photograph's. It can tell us about invariance, which is arithmetic rather than
calibration, so that is all it is used for here.
"""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.contract import QualityStatus
from app.services import quality

from test_pipeline import synthetic_fundus

DEMO_DIR = Path(__file__).resolve().parents[2] / "local-data" / "demo"


def demo(name: str) -> Image.Image:
    if not (DEMO_DIR / name).exists():
        pytest.skip(f"{name} not fetched")
    return Image.open(DEMO_DIR / name)


def dimmed(name: str, factor: float) -> Image.Image:
    """A real capture at a fraction of its brightness, standing in for a weak
    flash or an undilated pupil. Linear scaling, so nothing but exposure
    changes - the photograph stays exactly as sharp as it was."""
    if not (DEMO_DIR / name).exists():
        pytest.skip(f"{name} not fetched")
    arr = np.asarray(Image.open(DEMO_DIR / name).convert("RGB"), dtype=np.float64)
    return Image.fromarray(np.clip(arr * factor, 0, 255).astype(np.uint8))


def measure_focus(img: Image.Image) -> float:
    work = quality._working(img)
    return quality.focus_measure(work, quality._retina_mask(work))


@pytest.mark.skipif(
    not DEMO_DIR.exists(), reason="real demo captures are not in git; fetch them first"
)
def test_focus_does_not_collapse_when_a_capture_is_dark():
    """A sharp photograph is sharp whether it was taken bright or dark.

    The Laplacian is linear, so scaling intensity by k scales its variance by k
    squared. Measuring absolute variance therefore measures exposure as much as
    focus, which is what produced the false "severe defocus" on a perfectly
    focused capture.

    Exact invariance is not achievable on 8-bit data - quantization contributes
    a variance floor that does not scale with exposure - so the bound here is
    the measured one rather than an aspirational 1.0. Across the 40 calibration
    captures the worst retention at 20% brightness is 0.41 and the median 0.67,
    where the previous metric retained 0.17.

    This cannot be tested on the synthetic fixture. Its high-frequency content
    is additive Gaussian noise applied before the brightness scaling, so its
    measure falls linearly with exposure by construction, telling us about the
    fixture rather than about the metric.
    """
    full = measure_focus(demo("4134b290f5f3.png"))
    dark = measure_focus(dimmed("4134b290f5f3.png", 0.2))

    assert dark > full * 0.4, f"focus fell from {full:.4f} to {dark:.4f}"


@pytest.mark.skipif(
    not DEMO_DIR.exists(), reason="real demo captures are not in git; fetch them first"
)
@pytest.mark.parametrize("name", ["4134b290f5f3.png", "0dbaa09a458c.png", "838c87c63422.png"])
def test_no_sharp_capture_is_called_severely_defocused_when_dimmed(name):
    """The claim that sent a health worker to refocus a sharp photograph.

    Severe defocus is the issue that pushes an image below FOCUS_UNGRADABLE and
    rejects it outright, so a false one both names the wrong problem and
    attributes the rejection to the wrong metric.
    """
    report = quality.assess_quality(dimmed(name, 0.2))
    assert "severe defocus" not in report.issues, report.issues


@pytest.mark.skipif(
    not DEMO_DIR.exists(), reason="real demo captures are not in git; fetch them first"
)
def test_sharp_dark_capture_is_not_reported_as_defocused():
    """The bug this fixes: a dark capture was told to refocus.

    Nothing about a correctly focused photograph changes when the flash is too
    weak, so no focus issue may be raised for one. This needs a real capture -
    the synthetic fixture's per-pixel noise pins its focus at the ceiling, so
    the misreport cannot be reproduced on it.
    """
    issues = quality.assess_quality(dimmed("4134b290f5f3.png", 0.2)).issues
    assert not any("focus" in issue for issue in issues), issues


@pytest.mark.skipif(
    not DEMO_DIR.exists(), reason="real demo captures are not in git; fetch them first"
)
def test_recapture_names_the_dominant_failure_first():
    """'Recapture, too dark' is the most actionable thing this system can say.

    The issue list is what the recapture sentence is built from, so its first
    entry decides what the health worker actually does next. Today a sharp but
    dark capture reads 'severe defocus, underexposed' and sends them to adjust
    a focus ring that was never the problem.
    """
    issues = quality.assess_quality(dimmed("4134b290f5f3.png", 0.2)).issues
    assert issues[0] == "underexposed", issues


def test_retina_mask_survives_underexposure():
    """The mask decides which pixels the illumination statistics average over.

    An absolute intensity floor drops the dark retina out of its own mask, so
    the metric ends up measuring the brightest surviving sliver and stops
    responding. Coverage must track the retina, not the exposure.
    """
    bright = quality._working(synthetic_fundus(brightness=1.0))
    dark = quality._working(synthetic_fundus(brightness=0.12))

    assert quality._retina_mask(dark).mean() == pytest.approx(
        quality._retina_mask(bright).mean(), rel=0.05
    )


def test_darkness_alone_reaches_the_illumination_gate():
    """ILLUM_UNGRADABLE was unreachable: the additive floor terms guaranteed
    0.50 for any even, unclipped frame, so a pitch-black retina scored 0.50.
    A retina that cannot be seen is not half quality."""
    report = quality.assess_quality(synthetic_fundus(brightness=0.12))
    assert report.illumination < quality.ILLUM_UNGRADABLE


def test_a_well_exposed_image_keeps_a_high_illumination_score():
    """The counterweight to the test above - the fix must not reject good images."""
    report = quality.assess_quality(synthetic_fundus())
    assert report.illumination > 0.6


@pytest.mark.skipif(
    not DEMO_DIR.exists(), reason="real demo captures are not in git; fetch them first"
)
@pytest.mark.parametrize(
    "name,expected",
    [
        ("4134b290f5f3.png", QualityStatus.GRADABLE),
        ("0dbaa09a458c.png", QualityStatus.GRADABLE),
        ("4d47300e3ddb.png", QualityStatus.GRADABLE),
        ("838c87c63422.png", QualityStatus.GRADABLE),
        ("0097f532ac9f.png", QualityStatus.BORDERLINE),
        ("ungradable-defocus.png", QualityStatus.UNGRADABLE),
    ],
)
def test_demo_captures_keep_their_status(name, expected):
    """The captures the demo runs on, each pinned to the path it exercises.

    The borderline case has to stay borderline or the enhancement path stops
    being exercised on stage, and the ungradable one has to stay ungradable or
    the quality gate never fires in front of a judge."""
    path = DEMO_DIR / name
    if not path.exists():
        pytest.skip(f"{name} not fetched")
    assert quality.assess_quality(Image.open(path)).status is expected
