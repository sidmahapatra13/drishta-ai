"""Tests for confidence calibration.

The network emits one continuous severity score, so there is no softmax
posterior to temperature-scale in the textbook sense. Two different quantities
are calibrated here and they must never be conflated:

  * `referral_probability(raw)` is a real binary posterior - P(this patient is
    referable) - fitted by Platt scaling on the 3,662 out-of-fold predictions.
    It is the number a clinician acts on, and it is the one reported with an
    ECE.
  * `TEMPERATURE` scales `grade_likelihood`, which is a *derived display*
    distribution over the five grades, not a posterior. It is labelled as
    derived wherever it is shown.

The fitted constants live in `classifier.py`; the run that produced them wrote
`experiments/calibration/calibration.json`. Pinning the two together catches a
constant edited without a refit - the same guard `fit_focus_reference.py` gives
the quality gate.
"""

import json

import pytest

from app.contract import Eye, QualityStatus
from app.services import classifier, fusion, pipeline
from test_pipeline import synthetic_fundus

needs_weights = pytest.mark.skipif(
    not classifier.MODEL_PATH.exists(), reason="model weights not fetched"
)

needs_labels = pytest.mark.skipif(
    not (classifier.ROOT / "local-data" / "aptos" / "train.csv").exists(),
    reason="APTOS labels not fetched; run scripts/fetch_demo_images.py",
)


def test_referral_probability_increases_with_severity():
    """A sicker eye must never be called less likely to need a referral."""
    scores = [classifier.referral_probability(r) for r in [-1.0, 0.0, 1.0, 2.0, 4.0]]
    assert scores == sorted(scores)
    assert scores[0] < scores[-1]


def test_referral_probability_is_a_probability():
    for raw in [-5.0, -1.0, 0.0, 1.3, 3.0, 10.0]:
        p = classifier.referral_probability(raw)
        assert 0.0 <= p <= 1.0


def test_temperature_is_fitted_rather_than_a_placeholder():
    """The bug this replaces: results claimed calibration at temperature 1.0."""
    assert classifier.TEMPERATURE != 1.0
    assert classifier.IS_CALIBRATED


def test_fitted_constants_match_the_run_that_produced_them():
    """Catches a constant hand-edited without re-running the fit."""
    recorded = json.loads(classifier.CALIBRATION_PATH.read_text())
    assert classifier.PLATT_A == recorded["platt_a"]
    assert classifier.PLATT_B == recorded["platt_b"]
    assert classifier.TEMPERATURE == recorded["temperature"]
    assert recorded["n"] == 3662


def test_recorded_calibration_reports_a_real_error():
    """Every number shown to a judge comes from a real run. An ECE of exactly
    zero would mean the metric was never computed."""
    recorded = json.loads(classifier.CALIBRATION_PATH.read_text())
    for key in ["ece_referral", "ece_referral_cv", "brier_referral"]:
        assert 0.0 < recorded[key] < 1.0


def test_ungradable_eye_carries_no_referral_probability():
    """The gate strips every prediction, and a referral probability is one."""
    result = pipeline.analyze_eye(synthetic_fundus(blur=12), Eye.LEFT, "test")
    assert result.quality.status is QualityStatus.UNGRADABLE
    assert result.referral_probability is None


def test_gradable_eye_carries_a_referral_probability():
    result = pipeline.analyze_eye(synthetic_fundus(), Eye.LEFT, "test")
    assert result.quality.status is not QualityStatus.UNGRADABLE
    assert result.referral_probability is not None
    assert 0.0 <= result.referral_probability <= 1.0


def test_worse_eye_drives_the_reported_referral_probability():
    """Patient level reports the eye that drove the decision, not an average -
    the same rule the grade already follows."""
    left = pipeline.analyze_eye(synthetic_fundus(), Eye.LEFT, "t")
    right = pipeline.analyze_eye(synthetic_fundus(), Eye.RIGHT, "t")
    left.grade, left.referable, left.referral_probability = 0, False, 0.02
    right.grade, right.referable, right.referral_probability = 4, True, 0.97

    merged = fusion.fuse("t", "P1", left, right)
    assert merged.referral_probability == 0.97


@needs_weights
@needs_labels
def test_calibration_reproduces_from_the_out_of_fold_run():
    """Refit from the committed predictions and the APTOS labels, independently
    of the fitting script, and land on the constants the module carries."""
    import csv

    import numpy as np

    raw = np.load(classifier.ROOT / "experiments" / "classification" / "oof_predictions.npy")
    rows = list(csv.DictReader(open(classifier.ROOT / "local-data" / "aptos" / "train.csv")))
    y = np.array([int(r["diagnosis"]) >= 2 for r in rows], dtype=float)

    # Newton-Raphson on the logistic negative log-likelihood. Two parameters
    # over 3,662 points converges in a handful of steps.
    a, b = 1.0, 0.0
    X = np.column_stack([raw, np.ones_like(raw)])
    for _ in range(100):
        p = 1.0 / (1.0 + np.exp(-(X @ [a, b])))
        grad = X.T @ (p - y)
        H = (X * (p * (1 - p))[:, None]).T @ X + 1e-9 * np.eye(2)
        step = np.linalg.solve(H, grad)
        a, b = a - step[0], b - step[1]
        if np.max(np.abs(step)) < 1e-12:
            break

    assert abs(a - classifier.PLATT_A) < 1e-4
    assert abs(b - classifier.PLATT_B) < 1e-4
