"""Guards on the failure record.

Showing your own failures only counts if the set shown is the one the stated
rule produces. A gallery quietly filtered to the flattering cases is worse than
no gallery, so the selection rule is executable here rather than a sentence in a
caption: these tests re-derive the shown ids from the full list and check they
match.
"""

import json

import pytest

from app.contract import REFERABLE_FROM_GRADE
from app.services import classifier

RECORD = classifier.ROOT / "experiments" / "failures" / "failures.json"
SIGHT_THREATENING = 3


@pytest.fixture(scope="module")
def record() -> dict:
    return json.loads(RECORD.read_text())


def test_every_recorded_failure_is_actually_a_false_negative(record):
    assert record["failures"]
    for f in record["failures"]:
        assert f["true_grade"] >= REFERABLE_FROM_GRADE
        assert f["predicted_grade"] < REFERABLE_FROM_GRADE


def test_per_severity_rows_reconcile_with_the_headline(record):
    """The table and the summary above it must not be able to disagree."""
    rows = record["by_severity"]
    assert sum(r["n"] for r in rows) == record["referable"]["n"]
    assert sum(r["missed"] for r in rows) == record["referable"]["missed"]
    assert len(record["failures"]) == record["referable"]["missed"]

    for r in rows:
        assert r["caught"] + r["missed"] == r["n"]
        assert abs(r["sensitivity"] - r["caught"] / r["n"]) < 1e-12


def test_sight_threatening_is_grade_three_and_above(record):
    severe = [r for r in record["by_severity"] if r["grade"] >= SIGHT_THREATENING]
    assert record["sight_threatening"]["n"] == sum(r["n"] for r in severe)
    assert record["sight_threatening"]["missed"] == sum(r["missed"] for r in severe)


def test_the_captures_shown_are_the_ones_the_rule_selects(record):
    """Re-derive the selection instead of trusting the caption: every
    sight-threatening miss first, then the lowest severity scores."""
    expected = sorted(
        record["failures"], key=lambda f: (not f["sight_threatening"], f["raw"])
    )
    assert record["shown"] == [f["id_code"] for f in expected[: len(record["shown"])]]

    # Nothing sight-threatening may be left out while a milder miss is shown.
    severe = [f["id_code"] for f in record["failures"] if f["sight_threatening"]]
    assert set(severe) <= set(record["shown"])


def test_each_failure_carries_the_calibrated_referral_probability(record):
    """The gallery quotes the same calibrated probability the product shows."""
    for f in record["failures"]:
        expected = round(classifier.referral_probability(f["raw"]), 4)
        assert abs(f["referral_probability"] - expected) < 1e-9
