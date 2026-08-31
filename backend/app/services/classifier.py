"""DR severity grading - PS requirement 3.

STUB. Replaced on day 2 by the EfficientNet-B0 ordinal-regression model
trained on APTOS (Kaggle) and imported into MATLAB via ONNX.

The stub exists so the frontend, report generator, review queue and RAG
assistant can all be built against real contract shapes on day one instead of
waiting for training to finish. It is deterministic - the same image always
produces the same grade - so demo rehearsals are reproducible.
"""

import hashlib

import numpy as np
from PIL import Image

from ..contract import GRADE_LABELS, REFERABLE_FROM_GRADE

MODEL_VERSION = "stub-v0"
THRESHOLD_VERSION = "stub-v0"

#: Replaced by thresholds optimized on the validation folds. The real model
#: regresses a continuous grade and cuts it at four tuned boundaries; the
#: referable boundary is tuned for sensitivity, accepting specificity cost,
#: because a missed referable case is worse than a false alarm in screening.
ORDINAL_THRESHOLDS = [0.5, 1.5, 2.5, 3.5]


def _seed(img: Image.Image) -> int:
    return int(hashlib.sha256(img.tobytes()[:4096]).hexdigest()[:8], 16)


def predict_dr(img: Image.Image) -> dict:
    """Return {grade, probabilities, raw_logit}.

    Contract note: callers must not consult this for an ungradable image.
    fusion.gate() strips any prediction that slips through.
    """
    rng = np.random.default_rng(_seed(img))

    # Skew the stub toward the grades that matter in a demo (0 and 2) so the
    # workflow is exercised end to end without waiting for the real model.
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
        "model_version": MODEL_VERSION,
        "threshold_version": THRESHOLD_VERSION,
    }


def calibrate(probabilities: list[float], temperature: float = 1.0) -> list[float]:
    """Temperature scaling.

    STUB temperature of 1.0 is a no-op. The real value is fitted on the
    validation set on day 3 - raw softmax output is over-confident and must not
    be shown to a clinician as if it were a probability of disease.
    """
    logits = np.log(np.clip(probabilities, 1e-9, 1.0)) / temperature
    exp = np.exp(logits - logits.max())
    return [round(float(p), 4) for p in exp / exp.sum()]
