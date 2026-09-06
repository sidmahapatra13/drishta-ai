"""End-to-end orchestration: image in, structured result out.

This is the module that enforces the order of operations the PS requires:

    quality -> (enhance borderline) -> gate -> grade -> calibrate
            -> lesions -> Grad-CAM -> fuse

Swapping a stub for a real implementation means editing one import here. That
is the whole point of the day-one skeleton: the demo never breaks while the
intelligence is filled in behind it.
"""

import uuid
from pathlib import Path

from PIL import Image

from ..contract import (
    Eye,
    EyeResult,
    PatientResult,
    QualityStatus,
    ScreeningRequest,
)
from . import classifier, enhancement, explain, fusion, lesions, quality

ASSET_DIR = Path(__file__).resolve().parents[2] / "local-data" / "assets"


def _save(img: Image.Image, screening_id: str, name: str) -> str:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    path = ASSET_DIR / f"{screening_id}_{name}.png"
    img.save(path, format="PNG")
    return f"/assets/{path.name}"


def analyze_eye(img: Image.Image, eye: Eye, screening_id: str) -> EyeResult:
    """Run one fundus image through the full pipeline."""
    report = quality.assess_quality(img)

    # Borderline images get one enhancement attempt and a re-assessment.
    # Ungradable images are never enhanced into gradability.
    working, report = enhancement.enhance_if_borderline(
        img, report, quality.assess_quality
    )

    result = EyeResult(
        eye=eye,
        quality=report,
        model_version=classifier.MODEL_VERSION,
        threshold_version=classifier.THRESHOLD_VERSION,
        image_url=_save(working, screening_id, f"{eye.value}_image"),
    )

    if report.status is QualityStatus.UNGRADABLE:
        # Return early. fusion.gate() will fill in the recapture message and
        # guarantee no prediction is attached.
        return fusion.gate(result)

    # The network was trained on un-enhanced fundus photographs, so CLAHE output
    # is an input distribution it has never seen. Preprocessing drift of exactly
    # this kind measured 1.83 on the ordinal scale - wider than a grade boundary.
    # Enhancement is for the health worker to look at and for the quality
    # re-assessment; the classifier reads the image as captured.
    prediction = classifier.predict_dr(img)
    result.grade = prediction["grade"]
    result.grade_label = prediction["grade_label"]
    result.referable = prediction["referable"]
    result.probabilities = classifier.calibrate(prediction["probabilities"])
    result.calibrated = classifier.IS_CALIBRATED
    result.confidence = max(result.probabilities)

    # The calibrated posterior, from the same score that produced the grade, so
    # the two can never disagree about the same eye.
    result.referral_probability = classifier.referral_probability(
        prediction["raw_logit"]
    )

    result.lesions = lesions.segment_lesions(working, result.grade)

    # Grad-CAM has to run on the image the classifier actually read, or the
    # attention map explains a prediction that was never made.
    heatmap = explain.gradcam(img, result.grade)
    if heatmap is not None:
        result.gradcam_url = _save(heatmap, screening_id, f"{eye.value}_gradcam")

    return fusion.gate(result)


def run_screening(
    req: ScreeningRequest,
    left: Image.Image | None,
    right: Image.Image | None,
) -> PatientResult:
    """Screen one patient. At least one eye must be supplied."""
    if left is None and right is None:
        raise ValueError("At least one fundus image is required.")

    screening_id = uuid.uuid4().hex[:12]
    return fusion.fuse(
        screening_id=screening_id,
        patient_ref=req.patient_ref,
        left=analyze_eye(left, Eye.LEFT, screening_id) if left else None,
        right=analyze_eye(right, Eye.RIGHT, screening_id) if right else None,
    )
