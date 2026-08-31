"""The frozen result contract for SIH26038.

Every module in the pipeline produces or consumes these shapes. They are frozen
so that six people can build in parallel against a stub predictor: the frontend,
the report generator, the RAG assistant and the review queue all code against
this file, not against whatever the model happens to return today.

Changing anything here means announcing it to the whole team.
"""

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

# --- Vocabulary -------------------------------------------------------------

GRADE_LABELS = {
    0: "No DR",
    1: "Mild NPDR",
    2: "Moderate NPDR",
    3: "Severe NPDR",
    4: "Proliferative DR",
}

#: The International Clinical DR Severity Scale calls grade 2 and above
#: referable. The PS fixes this threshold, so it is not a tunable.
REFERABLE_FROM_GRADE = 2


class QualityStatus(str, Enum):
    GRADABLE = "gradable"
    BORDERLINE = "borderline"
    UNGRADABLE = "ungradable"


class LesionType(str, Enum):
    MICROANEURYSM = "microaneurysm"
    HEMORRHAGE = "hemorrhage"
    EXUDATE = "exudate"
    NEOVASCULARIZATION = "neovascularization"


class Priority(str, Enum):
    URGENT = "urgent"
    REVIEW = "review"
    LOW = "low"


class Eye(str, Enum):
    LEFT = "left"
    RIGHT = "right"


# --- Per-eye pieces ---------------------------------------------------------


class QualityReport(BaseModel):
    """Output of `assess_quality(image)`. Drives the gate, not the diagnosis."""

    focus: float = Field(ge=0, le=1)
    illumination: float = Field(ge=0, le=1)
    field_of_view: float = Field(ge=0, le=1)
    score: float = Field(ge=0, le=1)
    status: QualityStatus
    issues: list[str] = []
    enhancement_applied: bool = False


class Lesion(BaseModel):
    """One lesion class detected in one eye.

    `count` is detector output, not a clinically authoritative lesion count.
    The UI must label it as such.
    """

    type: LesionType
    count: int = Field(ge=0)
    area_px: int = Field(ge=0)
    mask_url: str | None = None
    confidence: float = Field(ge=0, le=1)


class EyeResult(BaseModel):
    """The complete analysis of one fundus image.

    When `quality.status` is UNGRADABLE the classifier is never run, so `grade`,
    `referable` and `confidence` are all None. Enforced in the service layer
    (see services/fusion.py), never left to the UI to remember.
    """

    eye: Eye
    quality: QualityReport

    grade: int | None = Field(default=None, ge=0, le=4)
    grade_label: str | None = None
    referable: bool | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    calibrated: bool = False
    probabilities: list[float] | None = None

    lesions: list[Lesion] = []
    gradcam_url: str | None = None
    overlay_url: str | None = None
    image_url: str | None = None

    model_version: str
    threshold_version: str

    #: Set when the eye needs to be photographed again.
    recapture_required: bool = False
    recapture_reason: str | None = None


# --- Patient level ----------------------------------------------------------


class PatientResult(BaseModel):
    """Both eyes fused into one referral decision.

    The worse eye drives the outcome: a patient with one healthy eye and one
    proliferative eye is an urgent referral, not an average.
    """

    screening_id: str
    patient_ref: str
    left: EyeResult | None = None
    right: EyeResult | None = None

    grade: int | None = None
    grade_label: str | None = None
    referable: bool | None = None
    priority: Priority | None = None
    confidence: float | None = None

    recapture_required: bool = False
    recapture_eyes: list[Eye] = []

    summary: str = ""
    disclaimer: str = (
        "AI-assisted screening output. This is not a diagnosis and does not "
        "replace examination by a qualified ophthalmologist."
    )


# --- Requests ---------------------------------------------------------------


class ScreeningRequest(BaseModel):
    patient_ref: str
    age: int | None = Field(default=None, ge=0, le=120)
    sex: Literal["M", "F", "O"] | None = None
    diabetes_duration_years: float | None = Field(default=None, ge=0)


class ReviewAction(BaseModel):
    """An ophthalmologist's verdict on an AI result. Human-in-the-loop."""

    action: Literal["confirm", "override", "request_recapture", "refer"]
    reviewer_ref: str
    override_grade: int | None = Field(default=None, ge=0, le=4)
    notes: str | None = None


class ClinicalQuestion(BaseModel):
    screening_id: str | None = None
    question: str


class Citation(BaseModel):
    title: str
    source: str
    snippet: str


class ClinicalAnswer(BaseModel):
    """RAG output. Grounded explanation only - it never sets a grade."""

    answer: str
    citations: list[Citation] = []
    grounded_in_model_output: bool = True
