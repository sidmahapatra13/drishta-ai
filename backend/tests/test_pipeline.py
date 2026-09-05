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


def test_review_round_trip(tmp_path, monkeypatch):
    """The doctor's verdict must persist and move the case out of the queue.

    This is the human-in-the-loop step, and it is a demo beat, so it gets a test.
    """
    from app import db

    monkeypatch.setattr(db, "DB_PATH", tmp_path / "t.sqlite")
    db.init()

    result = fusion.fuse(
        "s1", "IND-1", pipeline.analyze_eye(synthetic_fundus(), Eye.LEFT, "s1"), None
    )
    with db.session() as conn:
        conn.execute("INSERT INTO patients (patient_ref) VALUES ('IND-1')")
        db.save_screening(conn, result)

    with db.session() as conn:
        assert db.review_queue(conn)[0]["review_status"] == "pending"

    from app.contract import ReviewAction

    with db.session() as conn:
        db.save_review(conn, "s1", ReviewAction(action="confirm", reviewer_ref="DR-1"))

    with db.session() as conn:
        assert db.review_queue(conn)[0]["review_status"] == "reviewed"
        assert db.get_screening(conn, "s1")["patient_ref"] == "IND-1"


def test_enhancement_runs_on_a_borderline_image():
    """Enhancement was unreachable while the quality gate rejected every real
    photograph, so this path had never executed. np.asarray on a PIL image is
    read-only, and the CLAHE step assigns into it.
    """
    from app.services import enhancement

    out = enhancement.enhance(synthetic_fundus())
    assert out.size == synthetic_fundus().size
    assert out.mode == "RGB"
