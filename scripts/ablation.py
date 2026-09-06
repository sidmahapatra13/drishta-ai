"""Ablate the pipeline one technique at a time.

The PS asks for evidence that the integrated pipeline beats any single
technique. This produces it, and it is careful about one thing that would
otherwise make the whole table meaningless.

**Two footings, never mixed.** The exported network is fold 0 (see
`experiments/classification/train_aptos.py::export_onnx`), so it trained on four
fifths of APTOS. Any figure from running it across all 3,662 images is in-sample
and cannot be read against an out-of-fold number:

  * *Decision-rule ablations* change how a score becomes a grade. They need no
    inference at all - they re-decide the existing out-of-fold predictions, so
    all 3,662 images stay honestly held out.
  * *The input ablation* changes what the network sees, so it must run the
    network. It is restricted to fold 0's own held-out fifth, recovered here
    rather than assumed: those are exactly the images whose out-of-fold
    prediction was produced by fold 0, so the exported network reproduces them
    and reproduces nothing else. The script verifies the recovered split is
    about a fifth of the data and stops if it is not.

Not attempted: ordinal head against a 5-class softmax head. That needs a second
training run, and a table that implied the comparison had happened would be
worse than one that admits it did not.

    python scripts/fetch_aptos_images.py
    python scripts/ablation.py
"""

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.contract import REFERABLE_FROM_GRADE  # noqa: E402
from app.services import classifier, enhancement, quality  # noqa: E402

OOF = ROOT / "experiments" / "classification" / "oof_predictions.npy"
LABELS = ROOT / "local-data" / "aptos" / "train.csv"
IMAGES = ROOT / "local-data" / "aptos-full" / "train_images"
OUT_DIR = ROOT / "experiments" / "ablation"

#: How close the exported network has to land to a recorded out-of-fold value
#: for that image to be one it was held out from. The backend's ONNX output
#: matches the training run's own predictions to 0.0106, and a different fold's
#: model lands orders of magnitude further away, so this separates cleanly.
SAME_MODEL = 0.05


def qwk(actual: np.ndarray, predicted: np.ndarray, n: int = 5) -> float:
    O = np.zeros((n, n))
    for a, p in zip(actual, predicted):
        O[a, p] += 1
    W = np.array([[(i - j) ** 2 for j in range(n)] for i in range(n)]) / (n - 1) ** 2
    ha, hp = np.bincount(actual, minlength=n), np.bincount(predicted, minlength=n)
    E = np.outer(ha, hp) * O.sum() / (ha.sum() * hp.sum())
    return float(1.0 - (W * O).sum() / (W * E).sum())


def summarise(truth: np.ndarray, pred: np.ndarray, removes: str | None) -> dict:
    matrix = np.zeros((5, 5), dtype=int)
    for a, p in zip(truth, pred):
        matrix[a, p] += 1

    t, q = truth >= REFERABLE_FROM_GRADE, pred >= REFERABLE_FROM_GRADE
    tp, fn = int((t & q).sum()), int((t & ~q).sum())
    fp, tn = int((~t & q).sum()), int((~t & ~q).sum())

    return {
        "removes": removes,
        "qwk": qwk(truth, pred),
        "confusion_matrix": matrix.tolist(),
        "referable": {
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "sensitivity": tp / (tp + fn),
            "specificity": tn / (tn + fp),
        },
    }


def decide(raw: np.ndarray, cuts: np.ndarray, rule: str) -> np.ndarray:
    """Turn severity scores into grades three different ways."""
    if rule == "tuned":
        return np.digitize(raw, cuts)
    if rule == "rounding":
        # Equally spaced boundaries at .5, 1.5, 2.5, 3.5 - what you get without
        # fitting the cut points to anything.
        return np.clip(np.rint(raw), 0, 4).astype(int)
    if rule == "nearest":
        edges = np.concatenate(([cuts[0] - 1.0], cuts, [cuts[-1] + 1.0]))
        centres = (edges[:-1] + edges[1:]) / 2
        return np.argmin(np.abs(raw[:, None] - centres[None, :]), axis=1)
    raise ValueError(rule)


def main() -> None:
    if not IMAGES.exists():
        sys.exit(f"missing {IMAGES} - run scripts/fetch_aptos_images.py first")

    oof = np.load(OOF).astype(np.float64)
    rows = list(csv.DictReader(LABELS.open()))
    truth = np.array([int(r["diagnosis"]) for r in rows])
    ids = [r["id_code"] for r in rows]
    cuts = np.array(json.loads(classifier.CARD_PATH.read_text())["grade_thresholds"])

    # --- Decision rule: no inference, so every image stays held out ---------
    decision = {
        "footing": (
            "out-of-fold predictions - every image scored by a model that did "
            "not train on it"
        ),
        "n": int(len(oof)),
        "variants": {
            "full": summarise(truth, decide(oof, cuts, "tuned"), None),
            "naive_rounding": summarise(
                truth, decide(oof, cuts, "rounding"), "tuned cut points"
            ),
            "nearest_centre_argmax": summarise(
                truth, decide(oof, cuts, "nearest"), "the interval decision rule"
            ),
        },
    }

    # --- Inference over the originals, for the gate and the fold-0 recovery --
    print(f"grading {len(ids)} images")
    released = np.empty(len(ids))
    gated = np.zeros(len(ids), dtype=bool)
    started = time.time()
    for i, image_id in enumerate(ids):
        img = Image.open(IMAGES / f"{image_id}.png").convert("RGB")
        gated[i] = quality.assess_quality(img).status.value == "ungradable"
        released[i] = classifier.predict_dr(img)["raw_logit"]
        if (i + 1) % 500 == 0:
            rate = (time.time() - started) / (i + 1)
            print(f"  {i + 1}/{len(ids)}  {rate:.2f}s/image  "
                  f"eta {rate * (len(ids) - i - 1) / 60:.1f}m")

    # The recovery assumes the two populations separate: images fold 0 was held
    # out from reproduce their recorded score, images scored by another fold do
    # not. Print the distribution rather than assert it - two folds can agree by
    # chance on an easy image, and if that happens often the threshold is
    # picking up images fold 0 trained on.
    gap = np.abs(released - oof)
    print("\n|exported - recorded| distribution:")
    edges = [0, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0, np.inf]
    for lo, hi in zip(edges[:-1], edges[1:]):
        n = int(((gap >= lo) & (gap < hi)).sum())
        print(f"  {lo:>5} - {hi:<5} {n:>5}  {'#' * (n * 40 // len(gap))}")

    held_out = gap < SAME_MODEL
    share = held_out.mean()
    print(f"\nfold-0 held-out split recovered: {held_out.sum()} images ({share:.1%})")
    if not 0.15 < share < 0.25:
        sys.exit(
            f"recovered {share:.1%} of the data as fold 0's validation split; "
            "expected about a fifth. The exported network is not the model that "
            "produced these out-of-fold predictions, and the input ablation "
            "would be measured on the wrong footing."
        )

    # --- Input ablation: paired, on fold 0's own held-out fifth --------------
    subset = np.flatnonzero(held_out)
    print(f"re-grading {len(subset)} held-out images with CLAHE applied")
    enhanced = np.empty(len(subset))
    for j, i in enumerate(subset):
        img = Image.open(IMAGES / f"{ids[i]}.png").convert("RGB")
        enhanced[j] = classifier.predict_dr(enhancement.enhance(img))["raw_logit"]

    inputs = {
        "footing": (
            "fold 0's held out fifth - the only images the exported network "
            "never trained on. Paired: the same images through both paths."
        ),
        "n": int(len(subset)),
        "variants": {
            "original": summarise(
                truth[subset], decide(released[subset], cuts, "tuned"), None
            ),
            "clahe_into_classifier": summarise(
                truth[subset],
                decide(enhanced, cuts, "tuned"),
                "the un-enhanced input the network was trained on",
            ),
        },
    }

    # --- Quality gate: does it refuse the images the model gets wrong? -------
    shipped = decide(oof, cuts, "tuned")
    wrong = shipped != truth
    gate = {
        "refused": int(gated.sum()),
        "retained": int((~gated).sum()),
        "error_rate_refused": float(wrong[gated].mean()) if gated.any() else 0.0,
        "error_rate_retained": float(wrong[~gated].mean()),
        "note": (
            "Error rate is against the out-of-fold prediction, so the gate is "
            "judged on images the model was honestly held out from."
        ),
    }

    record = {
        "dataset": "APTOS 2019",
        "claim": (
            "The integrated pipeline is compared against itself with one "
            "technique removed at a time."
        ),
        "decision_rule": decision,
        "input": inputs,
        "quality_gate": gate,
        "not_attempted": {
            "five_class_softmax_head": (
                "Requires training a second network. The ordinal head is not "
                "compared against a 5-class softmax head anywhere in this "
                "record, and no such comparison should be read into it."
            )
        },
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "ablation.json").write_text(json.dumps(record, indent=2) + "\n")

    print(f"\n{'variant':<26} {'QWK':>7} {'sens':>7} {'spec':>7}")
    for section in ("decision_rule", "input"):
        print(f"-- {section}, n={record[section]['n']}")
        for name, v in record[section]["variants"].items():
            print(f"{name:<26} {v['qwk']:>7.4f} {v['referable']['sensitivity']:>7.4f} "
                  f"{v['referable']['specificity']:>7.4f}")
    print(f"\nquality gate refused {gate['refused']}/{len(ids)}")
    print(f"  error rate on refused  {gate['error_rate_refused']:.3f}")
    print(f"  error rate on retained {gate['error_rate_retained']:.3f}")
    print(f"\nwrote {OUT_DIR / 'ablation.json'}")


if __name__ == "__main__":
    main()
