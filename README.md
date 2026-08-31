# SIH26038 — AI-Assisted Diabetic Retinopathy Screening

Explainable AI screening and referral for rural India. Team Smarty Pants.

**Capture → Check → Analyze → Explain → Refer.**

A rural health worker photographs a diabetic patient's retina at a primary
health centre. The system checks whether the image is good enough to grade,
enhances it or asks for a recapture, grades diabetic retinopathy 0–4, shows the
lesion evidence and attention map behind that grade, and puts referable
patients in front of a scarce ophthalmologist in priority order.

It does not replace the ophthalmologist. It brings screening closer to the
patient and makes each specialist review faster.

## Status

Day-one skeleton. The pipeline runs end to end; the intelligence is being
filled in behind a frozen contract.

| Component | State |
|---|---|
| Result contract (`backend/app/contract.py`) | **Frozen** — build against this |
| Quality assessment, enhancement (CLAHE) | **Real** |
| Quality gate + patient-level fusion | **Real** |
| DR classifier, lesions, Grad-CAM | Stub — replaced day 2–3 |
| RAG assistant, deployment simulation | Stub — replaced day 3 |

Stubs are deterministic: the same image always yields the same result, so demo
rehearsals reproduce exactly.

## Run it

Two terminals. Python 3.14 no longer puts the working directory on `sys.path`,
so `PYTHONPATH=.` is required.

```bash
# terminal 1 — API on :8000
python3 -m venv .venv && ./.venv/bin/pip install -r backend/requirements.txt
cd backend && PYTHONPATH=. ../.venv/bin/python -m uvicorn app.main:app --reload

# terminal 2 — UI on :5173
cd frontend && npm install && npm run dev
```

```bash
cd backend && PYTHONPATH=. ../.venv/bin/python -m pytest tests/ -q   # 8 tests
```

## Layout

```
backend/app/contract.py       the frozen result contract — read this first
backend/app/services/         quality, enhancement, classifier, lesions,
                              explain, fusion, pipeline, rag, simulation
frontend/src/api.ts           typed mirror of the contract
matlab/                       MATLAB pipeline (see matlab/README.md)
simulink/                     deployment model (see simulink/README.md)
rag_knowledge_base/           clinical documents (see its README)
docs/superpowers/specs/       the 4-day design spec — scope, cuts, risks
```

## How to swap a stub for the real thing

Edit one module in `backend/app/services/`, keep the return shape, run the
tests. `services/pipeline.py` orchestrates and does not need to change. The
demo never breaks while the intelligence is filled in — that is the whole point
of the skeleton.

## Ground rules

- **Never claim clinical validation.** The defensible claim is benchmark
  performance, external validation on IDRiD, calibrated confidence, explainable
  outputs, and a human-in-the-loop workflow.
- **Grad-CAM shows influence, not lesions.** Never describe a heatmap as proof.
- **Every number on every slide comes from a real run.** Including the
  simulation dashboard.
- **An ungradable image never gets a grade.** Enforced in `services/fusion.py`,
  not left to the UI.
- **Neovascularization is out of scope** — no public pixel-level masks exist.
  Say so rather than omitting it quietly.

## Tracks

| Owner | Track |
|---|---|
| P1 | Preprocessing, classifier, thresholds, calibration, Grad-CAM, integration |
| P2 | MATLAB quality, enhancement, classical lesion/vessel/optic disc |
| P3 | FastAPI, SQLite, report generation |
| P4 | React screening flow, explainability, doctor queue |
| P5 | Simulink/SimEvents, scenarios |
| P6 | RAG knowledge base, evaluation write-up, ablation table, demo script |

Full plan, scope cuts and risk register: `docs/superpowers/specs/`.
