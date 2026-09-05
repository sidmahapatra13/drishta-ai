"""DR severity grading - PS requirement 3.

EfficientNet-B0 with an ordinal regression head, trained on APTOS 2019 (5-fold,
QWK 0.902, referable sensitivity 95.5% at 91.5% specificity) and exported to
ONNX. The network is not in git; fetch it with

    gh release download model-v1 --dir matlab/models

When the weights are absent this module falls back to the deterministic day-one
stub, so a fresh checkout still runs the demo end to end instead of crashing.
The stub is clearly marked in `model_version`, so nothing downstream can mistake
a rehearsal result for a real one.
"""

import hashlib
import json
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

from ..contract import GRADE_LABELS, REFERABLE_FROM_GRADE

MODEL_DIR = Path(__file__).resolve().parents[3] / "matlab" / "models"
MODEL_PATH = MODEL_DIR / "dr_effnetb0_ordinal.onnx"
CARD_PATH = MODEL_DIR / "model_card.json"

#: Training resolution. Baked into the exported graph's spatial dimensions.
SIZE = 456

STUB_VERSION = "stub-v0"


@lru_cache(maxsize=1)
def _loaded() -> tuple | None:
    """The ONNX session and its tuned cut points, or None when unavailable.

    Cut points are read from the model card rather than hardcoded: they are
    fitted per training run, and a stale copy here would grade every patient
    against boundaries the network was never calibrated to.
    """
    if not (MODEL_PATH.exists() and CARD_PATH.exists()):
        return None
    card = json.loads(CARD_PATH.read_text())
    session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
    return session, np.asarray(card["grade_thresholds"], dtype=np.float64)


def _versions() -> tuple[str, str]:
    loaded = _loaded()
    if loaded is None:
        return STUB_VERSION, STUB_VERSION
    return "dr-effnetb0-ordinal-v1", f"referable-grade2-t{loaded[1][1]:.3f}"


MODEL_VERSION, THRESHOLD_VERSION = _versions()


def preprocess(img: Image.Image) -> np.ndarray:
    """Ben Graham preprocessing, identical to training.

    OpenCV rather than PIL, and that is load-bearing rather than incidental:
    cv2.resize's INTER_LINEAR aliases on the ~6x downscale from a full-size
    fundus photograph, where PIL's BILINEAR antialiases, and the network was
    trained on the aliased version. Measured across nine APTOS images spanning
    all five grades, a PIL reimplementation moved the ordinal output by up to
    1.83 - wider than a grade boundary, enough to read grade 4 as grade 3 -
    while this path reproduces the recorded out-of-fold predictions to 0.011.
    """
    arr = np.asarray(img.convert("RGB"))

    # Crop to the retinal disc. The black surround carries no signal and its
    # size varies between cameras.
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    mask = gray > gray.mean() * 0.12
    if mask.any():
        ys, xs = np.nonzero(mask)
        arr = arr[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]

    arr = cv2.resize(arr, (SIZE, SIZE))

    # Subtract a heavy blur. What survives is local contrast - lesions - with
    # the camera's colour cast and illumination gradient removed.
    blur = cv2.GaussianBlur(arr, (0, 0), SIZE / 30)
    return cv2.addWeighted(arr, 4, blur, -4, 128)


def grade_likelihood(raw: float, cuts: np.ndarray) -> list[float]:
    """A display distribution over the five grades, derived from one scalar.

    The model emits a single continuous severity value, not five probabilities,
    so there is no posterior to read off. This is a softmax over distance from
    the centre of each grade's interval: reproducible, monotone in `raw`, and
    explicitly derived.

    It must be labelled "grade likelihood (derived)" wherever it is shown, never
    "probability of disease". The referral decision does not consult it - that
    comes from the grade, which comes from the tuned cut points.
    """
    # Pad the open-ended outer intervals so grades 0 and 4 get finite centres.
    edges = np.concatenate(([cuts[0] - 1.0], cuts, [cuts[-1] + 1.0]))
    centres = (edges[:-1] + edges[1:]) / 2
    spread = float(np.mean(np.diff(cuts))) / 2

    logits = -np.abs(raw - centres) / spread
    weights = np.exp(logits - logits.max())
    return [round(float(p), 4) for p in weights / weights.sum()]


def _stub_seed(img: Image.Image) -> int:
    """Hash the whole image, not a prefix.

    A fundus photograph opens with the black surround, so the leading bytes are
    identical across images - seeding from a prefix gave every patient the same
    grade.
    """
    return int(hashlib.sha256(img.tobytes()).hexdigest()[:8], 16)


def _stub_predict(img: Image.Image) -> dict:
    """Deterministic stand-in used when the weights are not present."""
    rng = np.random.default_rng(_stub_seed(img))

    # Skew toward the grades that matter in a demo (0 and 2) so the workflow is
    # exercised end to end.
    grade = int(rng.choice([0, 1, 2, 3, 4], p=[0.35, 0.15, 0.30, 0.12, 0.08]))

    probs = rng.dirichlet(np.full(5, 0.4))
    probs = probs * 0.25
    probs[grade] += 0.75
    probs = probs / probs.sum()

    return {
        "grade": grade,
        "grade_label": GRADE_LABELS[grade],
        "probabilities": [round(float(p), 4) for p in probs],
        "raw_logit": float(grade + rng.normal(0, 0.2)),
        "referable": grade >= REFERABLE_FROM_GRADE,
        "model_version": STUB_VERSION,
        "threshold_version": STUB_VERSION,
    }


def predict_dr(img: Image.Image) -> dict:
    """Return {grade, probabilities, raw_logit, ...}.

    Contract note: callers must not consult this for an ungradable image.
    fusion.gate() strips any prediction that slips through.
    """
    loaded = _loaded()
    if loaded is None:
        return _stub_predict(img)

    session, cuts = loaded
    batch = np.ascontiguousarray(
        preprocess(img).transpose(2, 0, 1), dtype=np.float32
    )[None] / 255.0
    raw = float(session.run(["grade"], {"fundus": batch})[0].ravel()[0])

    # The grade is the referral decision: cuts[1] is the grade-2 boundary and
    # was fitted to clear the screening sensitivity target, so a case can never
    # be graded 2 while being withheld from referral.
    grade = int(np.digitize(raw, cuts))

    return {
        "grade": grade,
        "grade_label": GRADE_LABELS[grade],
        "probabilities": grade_likelihood(raw, cuts),
        "raw_logit": raw,
        "referable": grade >= REFERABLE_FROM_GRADE,
        "model_version": MODEL_VERSION,
        "threshold_version": THRESHOLD_VERSION,
    }


def calibrate(probabilities: list[float], temperature: float = 1.0) -> list[float]:
    """Temperature scaling.

    A temperature of 1.0 is a no-op. The real value is fitted on the validation
    set on day 3 - raw output is over-confident and must not be shown to a
    clinician as if it were a probability of disease.
    """
    logits = np.log(np.clip(probabilities, 1e-9, 1.0)) / temperature
    exp = np.exp(logits - logits.max())
    return [round(float(p), 4) for p in exp / exp.sum()]
