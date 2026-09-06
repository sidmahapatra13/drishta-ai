# DRishta — AI-powered Diabetic Retinopathy Detection (SIH2026038)

Explainable AI screening and referral for rural India. Team Smarty Pants.

**Capture → Check → Analyze → Explain → Refer.**

A rural health worker photographs a diabetic patient's retina at a primary
health centre. The system checks whether the image is good enough to grade,
enhances it or asks for a recapture, grades diabetic retinopathy 0–4, shows the
lesion evidence and attention map behind that grade, and puts referable
patients in front of a scarce ophthalmologist in priority order.

It does not replace the ophthalmologist. It brings screening closer to the
patient and makes each specialist review faster.

## The pipeline

```
quality → enhance if borderline → gate → grade → calibrate → lesions → Grad-CAM → fuse
```

Enforced in `backend/app/services/pipeline.py`. An ungradable image stops at the
gate and never reaches the classifier. At patient level the worse eye drives
referral.

## Results

EfficientNet-B0 with an ordinal regression head — one continuous severity output
cut at four tuned boundaries, rather than a 5-class softmax that would treat
confusing grade 0 for 4 the same as 3 for 4. Five-fold cross-validated on APTOS
2019 (3,662 images).

| Metric | Value |
|---|---|
| Quadratic weighted kappa | **0.902** |
| Referable sensitivity (grade ≥ 2) | **95.5%** |
| Referable specificity | 91.5% |
| Referable ROC-AUC | 0.979 |

Every figure above is read from `matlab/models/model_card.json`, written by the
training run itself. The referral cut *is* the grade-2 boundary, fitted to clear
90% sensitivity — in screening a missed referable patient costs more than a
false alarm — so the grade shown and the referral flag beside it cannot
disagree.

**This is cross-validation on a public dataset, not clinical validation.**

## What it misses

That 95.5% is an average over very different failures, so it is broken out by
how severe the missed eye actually was. All 67 misses are listed by image id in
`experiments/failures/failures.json`.

| True severity | n | Missed | Referral sensitivity |
|---|---|---|---|
| 2 — Moderate NPDR | 999 | 64 | 93.6% |
| 3 — Severe NPDR | 193 | **0** | **100%** |
| 4 — Proliferative DR | 295 | 3 | **99.0%** |
| **Sight-threatening (3+)** | **488** | **3** | **99.4%** |

The misses are concentrated where they cost least: 64 of 67 are grade 2, the
mildest referable grade, sitting just under a boundary tuned to catch it. Three
proliferative eyes were missed and no severe NPDR was. This is reported because
a screening tool that misses moderate disease and one that misses proliferative
disease are not the same tool, and one sensitivity figure cannot tell them
apart.

## Does it transfer?

The model was trained on APTOS alone, then graded all 516 images of IDRiD — an
Indian dataset, a different camera and a different clinic — none of which it had
seen. The two numbers are reported side by side and never merged.

| | APTOS (5-fold CV) | IDRiD (external) |
|---|---|---|
| Quadratic weighted kappa | 0.902 | **0.746** |
| Referable sensitivity | 95.5% | **83.9%** |
| Referable specificity | 91.5% | **87.6%** |
| Referable ROC-AUC | 0.979 | **0.932** |

What holds is the ranking: ROC-AUC falls only 0.047, so the model still orders
patients by severity on a population it has never seen. What degrades is where
the cut points sit — sensitivity drops 11.6 points, because thresholds fitted to
APTOS's 41% referable prevalence meet IDRiD's 63%. That is what external
validation is for, and it is why one number is never quoted for both.

Confidence is calibrated on APTOS by Platt scaling: **ECE 0.018** (5-fold
cross-validated, n = 3,662). That calibration is population-specific and does
not transfer — measured at **ECE 0.092** on IDRiD, under-predicting referral by
7.9 points. It is reported, not silently refitted.

Full records and figures in `experiments/validation/` and
`experiments/calibration/`.

## Status

The pipeline runs end to end; the intelligence is being filled in behind a
frozen contract.

| Component | State |
|---|---|
| Result contract (`backend/app/contract.py`) | **Frozen** — build against this |
| Quality assessment, enhancement (CLAHE) | **Real** |
| Quality gate + patient-level fusion | **Real** |
| DR classifier | **Real** — trained network, weights released |
| Grad-CAM | **Real** — no gradients needed; see `services/explain.py` |
| Lesions | Stub, and silent: it reports nothing rather than inventing counts |
| RAG assistant, deployment simulation | Stub |

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
cd backend && PYTHONPATH=. ../.venv/bin/python -m pytest tests/ -q   # 39 tests
```

Model weights are not in git. Fetch the trained network into `matlab/models/`:

```bash
gh release download model-v1 --dir matlab/models
```

Neither are the demo images. Six real APTOS captures covering every path the
pipeline can take — not referable, referable, severe, urgent, a soft capture
that goes through enhancement, and one the quality gate refuses:

```bash
python scripts/fetch_demo_images.py
```

The quality gate's focus reference is fitted against real photographs, not the
synthetic test fixture. To refit it — or to check the fit still holds:

```bash
python scripts/fetch_calibration_images.py   # 40 APTOS captures, 8 per grade
python scripts/fit_focus_reference.py        # reports the valid interval
```

## Layout

```
backend/app/contract.py       the frozen result contract — read this first
backend/app/services/         quality, enhancement, classifier, lesions,
                              explain, fusion, pipeline, rag, simulation
frontend/src/api.ts           typed mirror of the contract
frontend/src/components/      one component per file, CSS beside each
frontend/src/styles/          design tokens and type scale
experiments/classification/   APTOS training run — script, notebook, OOF preds
experiments/calibration/      fitted calibration + reliability diagram
experiments/validation/       IDRiD external validation + comparison figure
experiments/failures/         every miss, by severity, with the ones shown
experiments/ablation/         one technique removed at a time
scripts/                      demo set, calibration set, quality-gate and
                              confidence fits, external validation
matlab/models/                ONNX network + model_card.json (the metrics)
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
