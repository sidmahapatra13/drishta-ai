"""Contract and quality-gate tests.

Not aiming for coverage - aiming for 'nothing breaks live in front of judges'.
The quality gate is the most-watched moment of the demo, so it gets the most
tests.
"""

import numpy as np
from PIL import Image

from app.contract import Eye, QualityStatus
from app.services import fusion, pipeline, quality
from app.services.classifier import predict_dr


def synthetic_fundus(size: int = 512, blur: int = 0, brightness: float = 1.0) -> Image.Image:
    """A crude stand-in for a fundus photograph: a textured disc on black."""
    yy, xx = np.mgrid[0:size, 0:size]
    cy = cx = size / 2
    r = np.hypot(yy - cy, xx - cx)
    mask = r < size * 0.45

    rng = np.random.default_rng(0)
    arr = np.zeros((size, size, 3), dtype=np.float64)
    arr[:, :, 0] = 150 + rng.normal(0, 25, (size, size))
    arr[:, :, 1] = 80 + rng.normal(0, 20, (size, size))
    arr[:, :, 2] = 40 + rng.normal(0, 15, (size, size))

    # Vessel-like streaks give the image high-frequency content to focus on.
    for _ in range(40):
        y0, x0 = rng.integers(0, size, 2)
        ang = rng.uniform(0, 2 * np.pi)
        for t in range(80):
            y, x = int(y0 + t * np.sin(ang)), int(x0 + t * np.cos(ang))
            if 0 <= y < size and 0 <= x < size:
                arr[y, x] *= 0.55

    arr *= mask[:, :, None]
    arr *= brightness
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    if blur:
        from PIL import ImageFilter

        img = img.filter(ImageFilter.GaussianBlur(blur))
    return img


def test_good_image_is_gradable():
    report = quality.assess_quality(synthetic_fundus())
    assert report.status is not QualityStatus.UNGRADABLE
    assert 0 <= report.focus <= 1


def test_blurred_image_is_rejected():
    report = quality.assess_quality(synthetic_fundus(blur=12))
    assert report.status is QualityStatus.UNGRADABLE
    assert report.issues


def test_dark_image_is_rejected():
    report = quality.assess_quality(synthetic_fundus(brightness=0.10))
    assert report.status is QualityStatus.UNGRADABLE


def test_ungradable_never_carries_a_prediction():
    """The single most important safety property in the system."""
    result = pipeline.analyze_eye(synthetic_fundus(blur=12), Eye.LEFT, "test")
    assert result.grade is None
    assert result.referable is None
    assert result.confidence is None
    assert result.recapture_required
    assert result.recapture_reason


def test_worse_eye_drives_referral():
    left = pipeline.analyze_eye(synthetic_fundus(), Eye.LEFT, "t")
    right = pipeline.analyze_eye(synthetic_fundus(), Eye.RIGHT, "t")
    left.grade, left.referable = 0, False
    right.grade, right.referable = 4, True

    merged = fusion.fuse("t", "P1", left, right)
    assert merged.grade == 4
    assert merged.referable is True
    assert merged.priority.value == "urgent"


def test_referral_threshold_matches_ps():
    """PS fixes referable at Level 2+. Not a tunable."""
    assert [fusion.is_referable(g) for g in range(5)] == [False, False, True, True, True]


def test_classifier_is_deterministic():
    """Demo rehearsals must reproduce exactly."""
    img = synthetic_fundus()
    assert predict_dr(img)["grade"] == predict_dr(img)["grade"]


def test_contract_shape_survives_serialization():
    result = fusion.fuse(
        "t", "P1", pipeline.analyze_eye(synthetic_fundus(), Eye.LEFT, "t"), None
    )
    assert "screening_id" in result.model_dump_json()
