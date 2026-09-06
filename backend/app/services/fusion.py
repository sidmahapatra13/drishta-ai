"""Evidence fusion: quality gate enforcement and patient-level referral.

This module is real from day one - it contains no stubs. It is the safety
boundary of the system, so the rules live in one place rather than being
re-implemented by every caller.
"""

from ..contract import (
    GRADE_LABELS,
    REFERABLE_FROM_GRADE,
    Eye,
    EyeResult,
    PatientResult,
    Priority,
    QualityStatus,
)


def is_referable(grade: int) -> bool:
    return grade >= REFERABLE_FROM_GRADE


def priority_for(grade: int) -> Priority:
    if grade >= 4:
        return Priority.URGENT
    if grade >= REFERABLE_FROM_GRADE:
        return Priority.REVIEW
    return Priority.LOW


def gate(result: EyeResult) -> EyeResult:
    """Enforce the quality gate.

    An ungradable image must never carry a grade. A confident 'No DR' on a
    blurred photograph is the single most dangerous output this system could
    produce, so we strip the prediction here rather than trusting every caller
    to check the status first.
    """
    if result.quality.status is QualityStatus.UNGRADABLE:
        result.grade = None
        result.grade_label = None
        result.referable = None
        result.confidence = None
        result.probabilities = None
        result.referral_probability = None
        result.lesions = []
        result.recapture_required = True
        if not result.recapture_reason:
            issues = ", ".join(result.quality.issues) or "insufficient image quality"
            result.recapture_reason = f"Image ungradable: {issues}."
    return result


def fuse(
    screening_id: str,
    patient_ref: str,
    left: EyeResult | None,
    right: EyeResult | None,
) -> PatientResult:
    """Combine both eyes into a single referral decision.

    The worse eye drives the outcome. A patient with one healthy eye and one
    proliferative eye is an urgent referral, not an average of the two.
    """
    left = gate(left) if left else None
    right = gate(right) if right else None

    result = PatientResult(
        screening_id=screening_id,
        patient_ref=patient_ref,
        left=left,
        right=right,
    )

    eyes = [(Eye.LEFT, left), (Eye.RIGHT, right)]
    result.recapture_eyes = [e for e, r in eyes if r and r.recapture_required]
    result.recapture_required = bool(result.recapture_eyes)

    graded = [r for _, r in eyes if r and r.grade is not None]
    if not graded:
        result.summary = (
            "No gradable image available. Please recapture and repeat screening."
        )
        return result

    worst = max(graded, key=lambda r: r.grade or 0)
    result.grade = worst.grade
    result.grade_label = GRADE_LABELS[worst.grade]
    result.referable = is_referable(worst.grade)
    result.priority = priority_for(worst.grade)
    # Report the confidence of the eye that actually drove the decision, not an
    # average across eyes that may disagree.
    result.confidence = worst.confidence
    result.referral_probability = worst.referral_probability

    result.summary = _summarize(result, worst)
    return result


def _summarize(result: PatientResult, worst: EyeResult) -> str:
    parts = [f"{result.grade_label} (Grade {result.grade}) in the {worst.eye.value} eye."]

    if result.referable:
        parts.append("Referable - ophthalmologist review recommended.")
    else:
        parts.append("Non-referable at this screening. Routine follow-up.")

    if worst.lesions:
        found = ", ".join(sorted({l.type.value for l in worst.lesions if l.count > 0}))
        if found:
            parts.append(f"Detected evidence: {found}.")

    if result.recapture_required:
        eyes = " and ".join(e.value for e in result.recapture_eyes)
        parts.append(f"Recapture required for the {eyes} eye.")

    return " ".join(parts)
