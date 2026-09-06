"""Guards on the external validation record.

These do not re-run inference - that needs the 516 IDRiD images, which are not
in git. They guard the thing that actually goes wrong with an external number:
it gets hand-edited, or it gets produced by a reimplementation of preprocessing
that the shipped model never used, or it quietly gets averaged together with the
APTOS figure until nobody can say which dataset the headline came from.

An external number that is wrong is worse than no external number at all.
"""

import json

import numpy as np
import pytest

from app.services import classifier

RECORD = classifier.ROOT / "experiments" / "validation" / "idrid.json"

def test_external_result_names_its_dataset_and_stays_separate_from_aptos():
    """APTOS and IDRiD are two numbers, never one. The model card holds the
    cross-validated APTOS figure; this file holds the external one."""
    record = json.loads(RECORD.read_text())
    card = json.loads(classifier.CARD_PATH.read_text())

    assert "IDRiD" in record["dataset"]
    assert record["trained_on"] != record["dataset"]
    assert record["splits"]["all"]["qwk"] != card["qwk"]


def test_external_validation_covers_the_official_split():
    record = json.loads(RECORD.read_text())
    assert record["splits"]["all"]["n"] == 516
    assert record["splits"]["official_test"]["n"] == 103


@pytest.mark.parametrize("split", ["all", "official_test"])
def test_reported_metrics_follow_from_the_reported_confusion_matrix(split):
    """Recompute sensitivity and specificity from the matrix beside them.

    Catches a number edited by hand, and catches a metric computed against a
    different threshold than the one reported.
    """
    s = json.loads(RECORD.read_text())["splits"][split]
    m = np.array(s["confusion_matrix"])

    assert m.sum() == s["n"]

    # Referable is grade 2 and above, on both axes.
    tp = int(m[2:, 2:].sum())
    fn = int(m[2:, :2].sum())
    fp = int(m[:2, 2:].sum())
    tn = int(m[:2, :2].sum())

    assert s["referable"]["tp"] == tp
    assert s["referable"]["fn"] == fn
    assert abs(s["referable"]["sensitivity"] - tp / (tp + fn)) < 1e-9
    assert abs(s["referable"]["specificity"] - tn / (tn + fp)) < 1e-9


def test_external_validation_used_the_shipped_preprocessing():
    """The failure the handoff warns about: anyone who reimplements
    `classifier.preprocess` produces a wrong external number.

    The run records a signature of the preprocessing it actually used. If the
    shipped preprocessing changes, this fails and the validation has to be run
    again rather than silently describing a pipeline that no longer exists.
    """
    import sys

    sys.path.insert(0, str(classifier.ROOT / "scripts"))
    from validate_idrid import preprocess_signature

    record = json.loads(RECORD.read_text())

    assert np.allclose(
        np.array(preprocess_signature()),
        np.array(record["preprocess_signature"]),
        atol=1e-3,
    )
    assert record["model_version"] == classifier.MODEL_VERSION


def test_calibration_transfer_is_measured_not_assumed():
    """The APTOS-fitted calibration is reported against IDRiD rather than
    assumed to carry over. The two populations differ in referable prevalence,
    so a drift is expected - it has to be a measured number either way."""
    record = json.loads(RECORD.read_text())
    transfer = record["calibration_transfer"]

    assert 0.0 < transfer["ece_on_idrid"] < 1.0
    assert transfer["platt_a"] == classifier.PLATT_A
    assert transfer["platt_b"] == classifier.PLATT_B
