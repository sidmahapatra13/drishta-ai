"""External validation of the shipped network against IDRiD.

APTOS says the model learned something. IDRiD says whether it transfers. They
are two numbers and this script never merges them: the APTOS figure stays in
`matlab/models/model_card.json`, the external one is written here.

Why IDRiD is a real external test and not a second split:

  * The network trained on APTOS alone. No IDRiD image, in any split, was seen
    during training or threshold fitting - so all 516 graded images are held
    out, not just the official 103.
  * Different cameras, different centre, different country's screening
    population. Referable prevalence is 63% here against 41% in APTOS.

Inference goes through `classifier.predict_dr`, which is the exact path the
running product uses - the same `preprocess`, the same ONNX session, the same
tuned cut points. This is not a convenience. A reimplementation of preprocessing
moved the ordinal output by 1.83 on real captures, wider than a grade boundary,
so an external number produced by a lookalike pipeline would describe a model
nobody ships. The run records a signature of the preprocessing it used and
`test_validation.py` checks it against the shipped one.

    python scripts/validate_idrid.py

Takes roughly ten minutes on CPU: 516 images at 4288x2848.
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
from app.services import classifier, quality  # noqa: E402

IDRID = ROOT / "local-data" / "idrid" / "B.%20Disease%20Grading" / "B. Disease Grading"
IMAGES = IDRID / "1. Original Images"
TRUTH = IDRID / "2. Groundtruths"
OUT_DIR = ROOT / "experiments" / "validation"

SPLITS = {
    "train": ("a. Training Set", "a. IDRiD_Disease Grading_Training Labels.csv"),
    "test": ("b. Testing Set", "b. IDRiD_Disease Grading_Testing Labels.csv"),
}


def load_split(name: str) -> list[tuple[Path, int]]:
    folder, labels = SPLITS[name]
    rows = [
        r
        for r in csv.DictReader((TRUTH / labels).open())
        if r["Image name"].strip()
    ]
    out = []
    for r in rows:
        path = IMAGES / folder / f"{r['Image name'].strip()}.jpg"
        if not path.exists():
            sys.exit(f"missing image {path}")
        out.append((path, int(r["Retinopathy grade"])))
    return out


def probe_image() -> Image.Image:
    """A fixed synthetic fundus used only to sign the preprocessing path.

    Defined once, here, and imported by `test_validation.py` so the signature in
    the record and the signature the test recomputes cannot drift apart. A real
    IDRiD frame would be a better probe but is not in git, and a guard that only
    runs where the dataset happens to be present is not a guard.
    """
    rng = np.random.default_rng(20260906)
    yy, xx = np.mgrid[0:512, 0:512]
    disc = np.hypot(yy - 256, xx - 256) < 230
    arr = np.zeros((512, 512, 3), dtype=np.float64)
    arr[..., 0], arr[..., 1], arr[..., 2] = 160.0, 90.0, 45.0
    arr += rng.normal(0, 20, (512, 512, 3))
    arr *= disc[..., None]
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def preprocess_signature() -> list[list[float]]:
    """An 8x8 mean-pooled fingerprint of `classifier.preprocess` on that probe."""
    out = classifier.preprocess(probe_image())
    pooled = out.reshape(8, 57, 8, 57, 3).mean(axis=(1, 3, 4))
    return [[round(float(v), 3) for v in row] for row in pooled]


def qwk(actual: np.ndarray, predicted: np.ndarray, n: int = 5) -> float:
    """Quadratic weighted kappa, the metric the APTOS figure is quoted in."""
    O = np.zeros((n, n))
    for a, p in zip(actual, predicted):
        O[a, p] += 1
    W = np.array([[(i - j) ** 2 for j in range(n)] for i in range(n)]) / (n - 1) ** 2
    ha, hp = np.bincount(actual, minlength=n), np.bincount(predicted, minlength=n)
    E = np.outer(ha, hp) * O.sum() / (ha.sum() * hp.sum())
    return float(1.0 - (W * O).sum() / (W * E).sum())


def roc_auc(score: np.ndarray, positive: np.ndarray) -> float:
    """Mann-Whitney U, with ties taking the average rank."""
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=np.float64)
    ranks[order] = np.arange(1, len(score) + 1)
    s = score[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = (i + j + 2) / 2
        i = j + 1
    n_pos = float(positive.sum())
    n_neg = float(len(positive) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[positive == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def ece(p: np.ndarray, y: np.ndarray, bins: int = 15) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (p >= lo) & (p < hi if hi < 1.0 else p <= hi)
        if sel.sum():
            total += sel.sum() / len(p) * abs(y[sel].mean() - p[sel].mean())
    return float(total)


def summarise(truth: np.ndarray, pred: np.ndarray, raw: np.ndarray) -> dict:
    matrix = np.zeros((5, 5), dtype=int)
    for a, p in zip(truth, pred):
        matrix[a, p] += 1

    ref_true = truth >= REFERABLE_FROM_GRADE
    ref_pred = pred >= REFERABLE_FROM_GRADE
    tp = int((ref_true & ref_pred).sum())
    fn = int((ref_true & ~ref_pred).sum())
    fp = int((~ref_true & ref_pred).sum())
    tn = int((~ref_true & ~ref_pred).sum())

    return {
        "n": int(len(truth)),
        "qwk": round(qwk(truth, pred), 6),
        "confusion_matrix": matrix.tolist(),
        "referable_prevalence": round(float(ref_true.mean()), 6),
        "referable": {
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "sensitivity": tp / (tp + fn),
            "specificity": tn / (tn + fp),
            "roc_auc": round(roc_auc(raw, ref_true.astype(int)), 6),
        },
    }


def main() -> None:
    if not IDRID.exists():
        sys.exit(f"missing {IDRID}")

    rows = [(p, g, "train") for p, g in load_split("train")]
    rows += [(p, g, "test") for p, g in load_split("test")]
    print(f"{len(rows)} images")

    raw, truth, pred, split, gated = [], [], [], [], 0
    started = time.time()
    for i, (path, grade, which) in enumerate(rows, 1):
        img = Image.open(path).convert("RGB")

        # Reported as an observation, not applied. The headline is graded on
        # every image so it stays comparable with published IDRiD numbers;
        # refusing images would flatter the result on a basis nobody else used.
        if quality.assess_quality(img).status.value == "ungradable":
            gated += 1

        out = classifier.predict_dr(img)
        raw.append(out["raw_logit"])
        pred.append(out["grade"])
        truth.append(grade)
        split.append(which)

        if i % 50 == 0:
            rate = (time.time() - started) / i
            print(f"  {i}/{len(rows)}  {rate:.2f}s/image  eta {rate * (len(rows) - i) / 60:.1f}m")

    raw = np.array(raw)
    truth = np.array(truth)
    pred = np.array(pred)
    split = np.array(split)

    if len(set(pred.tolist())) == 1:
        sys.exit("every image received the same grade - the network did not load")

    official = split == "test"
    referable = (truth >= REFERABLE_FROM_GRADE).astype(float)
    transferred = np.array([classifier.referral_probability(r) for r in raw])

    record = {
        "dataset": "IDRiD Disease Grading (Indian Diabetic Retinopathy Image Dataset)",
        "trained_on": "APTOS 2019",
        "source": "kaggle: aaryapatel98/indian-diabetic-retinopathy-image-dataset",
        "licence": "CC BY 4.0",
        "model_version": classifier.MODEL_VERSION,
        "threshold_version": classifier.THRESHOLD_VERSION,
        "preprocess_signature": preprocess_signature(),
        "inference": "classifier.predict_dr - the same path the product runs",
        "splits": {
            "all": summarise(truth, pred, raw),
            "official_test": summarise(truth[official], pred[official], raw[official]),
        },
        "quality_gate": {
            "would_refuse": gated,
            "of": len(rows),
            "note": (
                "Observed, not applied. Every image was graded so the figures "
                "stay comparable with published IDRiD results."
            ),
        },
        "calibration_transfer": {
            "platt_a": classifier.PLATT_A,
            "platt_b": classifier.PLATT_B,
            "fitted_on": "APTOS out-of-fold",
            "ece_on_idrid": round(ece(transferred, referable), 6),
            "mean_predicted": round(float(transferred.mean()), 6),
            "actual_prevalence": round(float(referable.mean()), 6),
            "note": (
                "Calibration is population-specific. IDRiD's referable "
                "prevalence differs from APTOS's, so the APTOS-fitted "
                "calibration is expected to drift here; the drift is measured "
                "rather than assumed, and the shipped calibration is not "
                "refitted on IDRiD."
            ),
        },
        "caveat": (
            "External validation on a public dataset. Not clinical validation, "
            "and not a claim of performance in deployment."
        ),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "idrid.json").write_text(json.dumps(record, indent=2) + "\n")

    a = record["splits"]["all"]
    print(f"\n--- IDRiD, all {a['n']} graded images (none seen in training) ---")
    print(f"QWK                      {a['qwk']:.4f}")
    print(f"Referable sensitivity    {a['referable']['sensitivity']:.4f}")
    print(f"Referable specificity    {a['referable']['specificity']:.4f}")
    print(f"Referable ROC-AUC        {a['referable']['roc_auc']:.4f}")
    print(f"Referable prevalence     {a['referable_prevalence']:.4f}")
    o = record["splits"]["official_test"]
    print(f"\n--- official test split, n={o['n']} ---")
    print(f"QWK                      {o['qwk']:.4f}")
    print(f"Referable sensitivity    {o['referable']['sensitivity']:.4f}")
    print(f"Referable specificity    {o['referable']['specificity']:.4f}")
    t = record["calibration_transfer"]
    print(f"\nECE of APTOS calibration on IDRiD   {t['ece_on_idrid']:.4f}")
    print(f"mean predicted {t['mean_predicted']:.4f} vs actual {t['actual_prevalence']:.4f}")
    print(f"quality gate would refuse {gated}/{len(rows)}")
    print(f"\nwrote {OUT_DIR / 'idrid.json'}")


if __name__ == "__main__":
    main()
