"""FastAPI application for the DR screening platform."""

import io

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError

from . import db
from .contract import (
    ClinicalQuestion,
    PatientResult,
    ReviewAction,
    ScreeningRequest,
)
from .services import pipeline, rag, simulation
from .services.pipeline import ASSET_DIR

MAX_IMAGE_BYTES = 20 * 1024 * 1024

# Create the schema at import time. Doing this in a startup hook means the
# tables are missing under TestClient and under any runner that skips lifespan.
db.init()

app = FastAPI(
    title="AI-Assisted DR Screening",
    description="SIH26038 - rural diabetic retinopathy screening and referral.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


async def _read_image(upload: UploadFile | None, label: str) -> Image.Image | None:
    """Decode an upload, rejecting oversized and non-image files with a clear
    message rather than a stack trace - these are demo failure modes we test."""
    if upload is None:
        return None
    data = await upload.read()
    if not data:
        return None
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(413, f"{label} image exceeds the 20 MB limit.")
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
        return img.convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(415, f"{label} file is not a readable image.")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": app.version}


@app.post("/screening/analyze", response_model=PatientResult)
async def analyze(
    patient_ref: str = Form(...),
    age: int | None = Form(None),
    sex: str | None = Form(None),
    diabetes_duration_years: float | None = Form(None),
    left: UploadFile | None = File(None),
    right: UploadFile | None = File(None),
) -> PatientResult:
    """Screen one patient. Quality gate runs before any grading."""
    req = ScreeningRequest(
        patient_ref=patient_ref,
        age=age,
        sex=sex,
        diabetes_duration_years=diabetes_duration_years,
    )
    left_img = await _read_image(left, "Left")
    right_img = await _read_image(right, "Right")

    try:
        result = pipeline.run_screening(req, left_img, right_img)
    except ValueError as exc:
        raise HTTPException(400, str(exc))

    with db.session() as conn:
        db.upsert_patient(conn, req)
        db.save_screening(conn, result)
    return result


@app.get("/screening/{screening_id}")
def get_screening(screening_id: str) -> dict:
    with db.session() as conn:
        result = db.get_screening(conn, screening_id)
    if result is None:
        raise HTTPException(404, "Screening not found.")
    return result


@app.get("/patients/{patient_ref}/history")
def history(patient_ref: str) -> list[dict]:
    with db.session() as conn:
        return db.patient_history(conn, patient_ref)


@app.get("/review/queue")
def queue() -> list[dict]:
    """Ophthalmologist review queue, most urgent first."""
    with db.session() as conn:
        return db.review_queue(conn)


@app.post("/review/{screening_id}")
def submit_review(screening_id: str, action: ReviewAction) -> dict:
    """Record a clinician's verdict. This is the human-in-the-loop step."""
    with db.session() as conn:
        if db.get_screening(conn, screening_id) is None:
            raise HTTPException(404, "Screening not found.")
        db.save_review(conn, screening_id, action)
    return {"status": "recorded", "screening_id": screening_id}


@app.post("/clinical/ask")
def clinical_ask(question: ClinicalQuestion) -> dict:
    """Evidence-grounded clinical explanation. Never sets or changes a grade."""
    model_output = None
    if question.screening_id:
        with db.session() as conn:
            model_output = db.get_screening(conn, question.screening_id)
    return rag.ask(question.question, model_output).model_dump()


@app.get("/simulation/scenarios")
def scenarios() -> list[dict]:
    return [simulation.run(s) for s in simulation.DEFAULT_SCENARIOS]


@app.post("/simulation/run")
def run_simulation(config: dict) -> dict:
    try:
        return simulation.run(simulation.Scenario(**config))
    except TypeError as exc:
        raise HTTPException(400, f"Invalid scenario configuration: {exc}")


@app.get("/assets/{filename}")
def asset(filename: str) -> FileResponse:
    path = (ASSET_DIR / filename).resolve()
    if not path.is_file() or ASSET_DIR.resolve() not in path.parents:
        raise HTTPException(404, "Asset not found.")
    return FileResponse(path)
