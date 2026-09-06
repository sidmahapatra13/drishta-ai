"""Guards on the ablation record.

The ablation exists to support one PS claim: that the integrated pipeline beats
any single technique used alone. A table like that is worth nothing if its
baseline quietly drifts from the system that ships, or if two rows measured on
different footings are printed as though they were comparable.

That second failure is the live risk here. The exported network is fold 0, so it
trained on four fifths of APTOS. Any row produced by running it over all 3,662
images is in-sample and cannot sit beside an out-of-fold row. The record
therefore keeps two sections on two footings, and these tests keep them apart.
"""

import json

import numpy as np
import pytest

from app.services import classifier

RECORD = classifier.ROOT / "experiments" / "ablation" / "ablation.json"

#: The ablation has not been run yet - it needs the full APTOS image set, which
#: is a multi-gigabyte fetch. These skip until `scripts/ablation.py` writes the
#: record, and stop skipping on their own the moment it does. Delete this guard
#: once the record is committed: from then on its absence is a real failure.
pending = pytest.mark.skipif(
    not RECORD.exists(),
    reason="run scripts/fetch_aptos_images.py then scripts/ablation.py",
)


def record() -> dict:
    return json.loads(RECORD.read_text())


def variants(section: str) -> dict:
    return record()[section]["variants"]


@pending
def test_decision_rule_baseline_is_the_system_that_ships():
    """Every delta in that table is measured from this row, so it has to be the
    real pipeline rather than a re-derivation of it."""
    card = json.loads(classifier.CARD_PATH.read_text())
    full = variants("decision_rule")["full"]

    assert full["removes"] is None
    assert abs(full["qwk"] - card["qwk"]) < 1e-9
    assert full["referable"]["tp"] == card["referable"]["tp"]
    assert full["referable"]["fn"] == card["referable"]["fn"]


@pending
@pytest.mark.parametrize(
    "section,name",
    [
        ("decision_rule", "full"),
        ("decision_rule", "naive_rounding"),
        ("decision_rule", "nearest_centre_argmax"),
        ("input", "original"),
        ("input", "clahe_into_classifier"),
    ],
)
def test_each_variant_follows_from_its_own_confusion_matrix(section, name):
    variant = variants(section)[name]
    m = np.array(variant["confusion_matrix"])

    assert m.sum() == record()[section]["n"]

    tp = int(m[2:, 2:].sum())
    fn = int(m[2:, :2].sum())
    fp = int(m[:2, 2:].sum())
    tn = int(m[:2, :2].sum())

    assert variant["referable"]["tp"] == tp
    assert abs(variant["referable"]["sensitivity"] - tp / (tp + fn)) < 1e-9
    assert abs(variant["referable"]["specificity"] - tn / (tn + fp)) < 1e-9


@pending
def test_every_variant_names_what_it_changes():
    """A row that does not say which technique it drops is not an ablation."""
    for section in ("decision_rule", "input"):
        for name, variant in variants(section).items():
            if variant["removes"] is None:
                continue
            assert variant["removes"], f"{section}.{name} does not say what it removes"


@pending
def test_the_two_sections_are_measured_on_different_footings_and_say_so():
    """The comparison that must never be made silently: an in-sample row read
    against an out-of-fold one."""
    decision, inputs = record()["decision_rule"], record()["input"]

    assert decision["n"] == 3662
    assert "out-of-fold" in decision["footing"]

    # The input ablation runs on fold 0's held-out split only, which is the one
    # fifth the exported network never trained on.
    assert inputs["n"] < decision["n"]
    assert abs(inputs["n"] - decision["n"] / 5) < decision["n"] * 0.02
    assert "held out" in inputs["footing"]


@pending
def test_what_was_not_measured_is_declared():
    """A 5-class softmax head cannot be compared without retraining one. Saying
    so on the record stops the table implying a comparison that never ran."""
    assert record()["not_attempted"]


@pending
def test_quality_gate_effect_is_measured_on_both_sides():
    """The gate's claim is that it removes images the model gets wrong. That is
    only a claim if the error rate on refused and on retained sit side by side."""
    gate = record()["quality_gate"]
    assert gate["refused"] + gate["retained"] == record()["decision_rule"]["n"]
    for key in ["error_rate_refused", "error_rate_retained"]:
        assert 0.0 <= gate[key] <= 1.0
