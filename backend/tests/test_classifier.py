"""Tests for the trained DR classifier.

The stub remains the fallback when weights are absent - a teammate who has not
run `gh release download` still gets a working demo - so these tests separate
what must hold either way from what only holds once the real network is loaded.
"""

import json

import numpy as np
import pytest

from app.services import classifier
from app.services.classifier import predict_dr
from test_pipeline import synthetic_fundus

needs_weights = pytest.mark.skipif(
    not classifier.MODEL_PATH.exists(), reason="model weights not fetched"
)

#: 8x8 mean-pooled signature of Ben Graham preprocessing on the synthetic
#: fundus, recorded from the OpenCV path the network was trained with. A PIL
#: reimplementation moves this by 11.8, so the tolerance below catches a swapped
#: implementation while tolerating minor OpenCV version noise.
PREPROCESS_SIGNATURE = [
    [126.1, 108.9, 132.4, 133.6, 134.6, 132.0, 108.7, 126.1],
    [109.1, 148.7, 128.6, 128.2, 128.5, 129.5, 148.2, 108.9],
    [132.3, 129.2, 127.2, 129.7, 127.8, 128.8, 128.3, 132.7],
    [134.7, 128.3, 128.6, 127.4, 129.1, 127.4, 128.3, 133.8],
    [134.4, 128.9, 127.9, 128.6, 128.4, 128.8, 128.5, 134.4],
    [132.0, 128.8, 127.5, 128.7, 130.0, 127.4, 128.8, 132.4],
    [108.8, 148.1, 129.0, 129.1, 126.9, 128.8, 148.7, 108.9],
    [126.1, 108.8, 132.5, 134.6, 134.6, 131.7, 109.3, 126.2],
]


@needs_weights
def test_real_model_is_used_when_weights_are_present():
    """The trained network grades the image, not the day-one stub."""
    out = predict_dr(synthetic_fundus())
    assert out["model_version"] != "stub-v0"
    assert 0 <= out["grade"] <= 4
    assert out["referable"] == (out["grade"] >= 2)


@needs_weights
def test_real_model_is_deterministic():
    """Demo rehearsals must reproduce exactly."""
    img = synthetic_fundus()
    assert predict_dr(img)["raw_logit"] == predict_dr(img)["raw_logit"]


@needs_weights
def test_referral_cut_is_the_grade_two_boundary():
    """The bug this project shipped with: a case graded 2 while the referral
    flag said no, for any output between two independently fitted thresholds.
    Referable is grade 2 and above, so the two are one number."""
    card = json.loads(classifier.CARD_PATH.read_text())
    assert card["referable"]["threshold"] == card["grade_thresholds"][1]


@needs_weights
def test_grade_boundaries_follow_the_tuned_cut_points():
    """Expected grades are read off the fitted cut points by hand, not
    recomputed the way the implementation does."""
    _, cuts = classifier._loaded()
    assert [round(float(c), 3) for c in cuts] == [0.613, 1.298, 2.475, 3.248]

    for raw, expected in [(-1.0, 0), (0.9, 1), (1.8, 2), (2.8, 3), (4.0, 4)]:
        assert int(np.argmax(classifier.grade_likelihood(raw, cuts))) == expected


@needs_weights
def test_grade_likelihood_is_a_distribution():
    _, cuts = classifier._loaded()
    probs = classifier.grade_likelihood(1.8, cuts)
    assert len(probs) == 5
    assert all(0.0 <= p <= 1.0 for p in probs)
    assert abs(sum(probs) - 1.0) < 1e-3


def test_preprocessing_matches_what_the_network_was_trained_on():
    """Guards the silent failure: preprocessing that drifts from training costs
    accuracy without raising anything. Measured on real APTOS images, a PIL
    reimplementation moved the ordinal output by up to 1.83 - wider than a grade
    boundary."""
    out = classifier.preprocess(synthetic_fundus())
    assert out.shape == (classifier.SIZE, classifier.SIZE, 3)
    assert out.dtype == np.uint8

    pooled = out.reshape(8, 57, 8, 57, 3).mean(axis=(1, 3, 4))
    assert np.allclose(pooled, np.array(PREPROCESS_SIGNATURE), atol=2.0)


def test_falls_back_to_the_stub_when_weights_are_absent(monkeypatch, tmp_path):
    """A checkout without the weights must still run the demo, and must say so."""
    monkeypatch.setattr(classifier, "MODEL_PATH", tmp_path / "absent.onnx")
    classifier._loaded.cache_clear()
    try:
        out = predict_dr(synthetic_fundus())
        assert out["model_version"] == "stub-v0"
        assert 0 <= out["grade"] <= 4
        assert out["referable"] == (out["grade"] >= 2)
        assert abs(sum(out["probabilities"]) - 1.0) < 1e-3
    finally:
        classifier._loaded.cache_clear()
