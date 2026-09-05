# AGENTS.md

SIH26038 — AI-assisted diabetic retinopathy screening for rural India.
Read `docs/superpowers/specs/` for scope, cuts and risks before proposing work.

## What this is

A screening and referral platform, not a DR classifier. A health worker
photographs a diabetic patient's retina at a rural health centre; the system
checks the image is gradable, grades DR 0–4, shows the evidence behind that
grade, and puts referable patients in front of a scarce ophthalmologist in
priority order. It never replaces the ophthalmologist.

Deadline is days, not weeks. The failure mode is not a weak model — it is no
working demo.

## Rules that are not negotiable

**An ungradable image never carries a grade.** Enforced in
`backend/app/services/fusion.py::gate`, not in the UI. A confident "No DR" on a
blurred photograph is the most dangerous output this system can produce.

**Referable is grade 2 and above.** Fixed by the problem statement. It is not a
tunable, and `contract.py::REFERABLE_FROM_GRADE` is the only place it is written.

**Never claim clinical validation.** The defensible claim is benchmark
performance, external validation on IDRiD, calibrated confidence, explainable
outputs, and a human-in-the-loop workflow. A strong public-dataset split is not
clinical validation.

**Grad-CAM shows influence, not lesions.** Never describe a heatmap as proof a
lesion exists. That is why lesion segmentation runs alongside it.

**Every number shown to a judge comes from a real run.** Including the
simulation dashboard and every metric on every slide. Never type a plausible
number in as a placeholder — if it isn't measured yet, leave it visibly empty.

**Neovascularization is out of scope.** No public pixel-level NV masks exist.
Say so explicitly; do not quietly omit it and do not fake it.

**Lesion counts are detector output**, not clinically authoritative counts.
Label them that way wherever they appear.

## Architecture

The result contract in `backend/app/contract.py` is frozen. Six people build
against it while the model behind it is still a stub. Changing it means telling
the whole team.

```
React (frontend/) → FastAPI (backend/) → SQLite
                          ├→ vision pipeline (MATLAB, ONNX net from Kaggle)
                          └→ RAG (curated markdown + numpy cosine)
Simulink/SimEvents → scenario JSON → dashboard
```

Pipeline order, enforced in `services/pipeline.py`:
`quality → enhance if borderline → gate → grade → calibrate → lesions →
Grad-CAM → fuse`.

Swapping a stub for the real thing means editing one module in
`backend/app/services/` and keeping the return shape. `pipeline.py` does not
change. The demo must never break while intelligence is filled in behind it.

## Stubs

`lesions.py`, `rag.py`, `simulation.py` are stubs. `rag.py` and
`simulation.py` are deterministic — the same input always yields the same
result — so demo rehearsals reproduce. Seeds hash the **whole** image: a fundus
photo opens with the black surround, so hashing a prefix gives every patient the
same result.

`lesions.py` is the exception, and deliberately: it returns nothing at all. A
stub that invents per-class counts puts a fabricated clinical finding into the
UI's evidence panel and into the patient summary. Leave it silent until the
detector is real.

Real already: `quality.py`, `enhancement.py`, `fusion.py`, `classifier.py`,
`explain.py`.

## Working here

```bash
cd backend && PYTHONPATH=. ../.venv/bin/python -m uvicorn app.main:app --reload
cd frontend && npm run dev
cd backend && PYTHONPATH=. ../.venv/bin/python -m pytest tests/ -q
```

Python 3.14 no longer puts the working directory on `sys.path`, hence
`PYTHONPATH=.`.

- Match the surrounding style. Comments explain *why*, not *what*.
- Touch only what the task requires. Mention unrelated problems; don't fix them.
- Verify before claiming. Run the thing, read the output, then say it works.
- Prefer the simplest approach that solves the problem. No speculative
  abstraction, no configurability nobody asked for.
- Don't commit datasets, model weights, or patient data.

## Medical judgment is not yours to make

Do not decide clinical thresholds, whether a dataset may be used, whether a
metric constitutes validation, or whether a result is medically correct. Surface
the question. Every ML component gets checked against real dataset labels.
