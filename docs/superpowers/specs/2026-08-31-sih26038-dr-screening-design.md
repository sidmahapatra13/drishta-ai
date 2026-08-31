# SIH26038 — 4-Day Design Spec

**Date:** 2026-08-31
**Deadline:** 4 days (internal round)
**Effective headcount:** 1 primary (lead) + 5 teammates on parallel, low-coupling tracks
**Strategy:** Option A — training on Kaggle (PyTorch) → ONNX → MATLAB owns the vision pipeline, explainability, metrics, and Simulink.

---

## 1. The Governing Constraint

Four days. The failure mode is not a weak model — it is **no end-to-end demo**. Therefore:

> **Day 1 ships a complete, running, ugly system with a stub predictor.**
> Days 2–4 replace stubs with real components, one at a time, never breaking the demo.

Every module is swapped behind a frozen JSON contract (§4). At no point after Day 1 end is the demo non-functional. This is the single most important rule in this spec.

---

## 2. Scope Decisions

### 2.1 In scope (must work by Day 4)

| # | Component | PS requirement |
|---|---|---|
| 1 | Image quality assessment: focus, illumination, field-of-view | Req 1 |
| 2 | Adaptive enhancement: CLAHE, illumination normalization, denoising + re-check | Req 1 |
| 3 | Ungradable rejection with recapture feedback | Req 1 |
| 4 | DR grading 0–4 (EfficientNet-B0, ordinal regression head) | Req 3 |
| 5 | Referable DR (Level 2+) with tuned threshold, sensitivity/specificity | Req 3 |
| 6 | Grad-CAM attention maps | Req 4 |
| 7 | Lesion evidence: microaneurysms, exudates, hemorrhages (classical morphology) | Req 2, 4 |
| 8 | Vessel segmentation + optic disc localization (classical) | Req 2 |
| 9 | Confidence calibration (temperature scaling) + ECE/reliability diagram | Req 4 |
| 10 | Automated annotated report (PDF) | Req 4 |
| 11 | Health-worker screening UI + ophthalmologist review queue | Product |
| 12 | Clinical RAG assistant with real citations | Differentiator |
| 13 | Simulink/SimEvents district deployment simulation | Req 5 |
| 14 | **External validation on IDRiD (Indian data)** | "validation against published benchmarks" |
| 15 | **Ablation table** (integrated pipeline vs single techniques) | "outperforms any single-technique approach" |

Items 14 and 15 are explicit PS deliverables that are easy to forget. They are cheap. Do not drop them.

### 2.2 Explicitly cut, with reasons

| Cut | Reason |
|---|---|
| PostgreSQL + pgvector | Hours of setup, zero judge-visible payoff. **SQLite** (single file) + numpy cosine similarity over ~40 KB chunks. |
| U-Net for vessel segmentation | Vessels are solvable classically (green channel → CLAHE → Frangi/top-hat) in half a day with no labels. |
| U-Net for lesions | **Stretch goal only** (§3, Day 3 fork). Classical morphology gives demo-quality overlays in ~4h vs ~1.5 days. |
| DDR dataset (13,673 images) | No GPU-hours available. Excluded from training entirely. |
| Messidor | Redundant — IDRiD already supplies external validation, and on Indian data, which is strictly better for this PS. |
| **Neovascularization detection** | **No public pixel-level NV masks exist.** Scoped out *explicitly and on a slide*, with justification. Do not attempt; do not silently omit. |
| Docker, auth, ONNX-in-production serving | Zero demo value in 4 days. |
| Multilingual UI, offline/edge sync | Presented as architecture/roadmap only, never claimed as implemented. |

### 2.3 Honesty constraints (non-negotiable)

- No claim of clinical validation. The defensible claim is: *benchmark performance + external validation on IDRiD + calibrated confidence + explainable outputs + human-in-the-loop workflow.*
- Grad-CAM is described as "regions that influenced the prediction," never as lesion proof.
- Every number on every slide comes from a real run. No fabricated demo text — including the Simulink dashboard.
- Lesion counts are labeled as detector output, not clinically authoritative counts.

---

## 3. Four-Day Plan

### Day 0 — Tonight, ~1 hour, strictly parallel

These are long-pole items that block everything. Start all three before doing anything else.

1. **Claim MATLAB.** Check the SIH portal for MathWorks-sponsored licenses first (this is usually the gating item and can take days to approve). In parallel start the 30-day trial download (~15 GB, hours). Required toolboxes: Image Processing, Computer Vision, Deep Learning, Statistics & ML, Simulink, **SimEvents** (separately licensed — verify).
2. **Kaggle setup.** Account verified for GPU (30 h/week, P100/T4). APTOS 2019 is hosted natively on Kaggle — no download needed. Request IDRiD access (IEEE DataPort) *tonight*; approval is not instant.
3. **Repo + contract.** Scaffold and push. Freeze the JSON contract (§4) and send it to all 5 teammates so they start Day 1 against a stub, not against a promise.

### Day 1 — Skeleton end-to-end (the critical day)

**Lead:** repo scaffold; frozen contract; **stub predictor** returning contract-shaped fake results; FastAPI wired; React shell that uploads two images and renders a result. *By end of Day 1 the full pipe runs on a laptop with fake numbers.*
**In parallel:** kick off APTOS training on Kaggle — Ben Graham preprocessing (circle-crop, resize 456, subtract Gaussian blur), EfficientNet-B0, **ordinal regression head** (single continuous output + 4 optimized thresholds, not 5-class softmax), 5-fold, ~15 epochs.
**Teammates:** backend endpoints against stub; frontend screens against stub; Simulink model started; RAG KB documents gathered.

### Day 2 — Real intelligence

- Real model replaces stub. Threshold tuning for referable DR on the validation folds.
- Referable sensitivity/specificity/ROC-AUC/QWK/confusion matrix.
- **IDRiD external validation** (train APTOS → test IDRiD 516 graded images).
- Quality gate: classical metrics (Laplacian variance for focus, histogram statistics for illumination, retinal mask coverage for FOV) + **synthetic degradation** to generate Good/Borderline/Ungradable labels (APTOS has no quality labels — programmatically blur, over/under-expose, and crop known-good images).
- MATLAB enhancement path: `adapthisteq` (CLAHE) → illumination normalization → denoise → re-assess.
- Grad-CAM via MATLAB `gradCAM` on the ONNX-imported network (one built-in call).

### Day 3 — Evidence, explanation, scale

- Classical lesion detection: exudates (bright-lesion morphology with optic-disc suppression), hemorrhages (dark-lesion), microaneurysms (top-hat + candidate filtering). Vessel segmentation + optic disc localization.
- **Fork point:** if Day 2 finished early, train a single multi-class U-Net on the IDRiD segmentation subset (54 training images, patch-based, ~1 h on Kaggle) producing MA/HE/EX channels. Otherwise ship classical and say so.
- Confidence calibration (temperature scaling) + ECE + reliability diagram.
- Evidence fusion → structured result → PDF report.
- Doctor review queue with priority sorting; confirm/override/refer actions.
- RAG assistant answering over the curated KB with real source citations.
- Simulink scenarios run and results exported.

### Day 4 — Integration and hardening

- End-to-end integration; failure-mode tests (§6).
- **Ablation table** produced (§2.1 item 15).
- Demo assets frozen: one good image, one deliberately poor image, one referable case, one non-referable case.
- Metrics dashboard populated with real numbers.
- Two full timed rehearsals of the 5-minute demo.

---

## 4. The Frozen Contract

This is what unblocks 6 people working in parallel. Freeze it Day 0; change it only with an announcement to everyone.

### Per-eye result

```json
{
  "eye": "left",
  "quality": {
    "focus": 0.94, "illumination": 0.87, "field_of_view": 0.96,
    "score": 0.92,
    "status": "gradable",
    "issues": [],
    "enhancement_applied": false
  },
  "grade": 2,
  "grade_label": "Moderate NPDR",
  "referable": true,
  "confidence": 0.917,
  "calibrated": true,
  "probabilities": [0.01, 0.05, 0.80, 0.10, 0.04],
  "lesions": [
    {"type": "microaneurysm", "count": 12, "area_px": 340, "mask_url": "...", "confidence": 0.71}
  ],
  "gradcam_url": "...",
  "overlay_url": "...",
  "model_version": "dr-effnetb0-ordinal-v1",
  "threshold_version": "referable-t0.53"
}
```

`status` is one of `gradable` | `borderline` | `ungradable`. When `ungradable`, **`grade`, `referable`, and `confidence` are `null`** — the classifier is never run on a rejected image. This is the quality gate, and it must be enforced in the service layer, not the UI.

### Patient-level fusion

Missing from the source README and required for a coherent referral decision: the **worse eye drives referral**. `patient.referable = left.referable OR right.referable`; `patient.grade = max(left.grade, right.grade)`; priority = urgent (grade 4) / review (grade 2–3) / low (grade 0–1). If one eye is ungradable and the other is referable, the patient is referable *and* flagged for recapture of the failed eye.

### Module interfaces

```
assess_quality(image)          -> {focus, illumination, field_of_view, score, status, issues}
enhance(image)                 -> enhanced_image
predict_dr(image)              -> {grade, probabilities, raw_logit}
calibrate(probabilities)       -> calibrated_probabilities
segment_lesions(image)         -> {masks, lesion_types, counts, confidence}
explain(image, model)          -> gradcam_map
fuse(quality, dr, lesions)     -> structured_result
```

Each is independently testable and independently swappable. Nothing imports another module's internals.

---

## 5. Architecture

```
React (Vite + TS + Tailwind)
        │ REST
FastAPI  ──── SQLite (patients, screenings, predictions, lesions, reports)
        │
        ├── Vision service ──→ MATLAB (quality, enhancement, Grad-CAM,
        │                       lesions, vessels, OD, calibration, metrics)
        │                       ← ONNX network imported from Kaggle training
        └── RAG service ────→ curated markdown KB + local embeddings

Simulink / SimEvents  (offline; exports scenario results as JSON → dashboard)
```

**MATLAB integration:** MATLAB Engine API for Python during development. If the engine proves flaky under time pressure, fall back to subprocess invocation of a MATLAB script with JSON in/out — same contract, less coupling. Decide by end of Day 2; do not spend more than 2 hours debugging the engine.

**Simulink degradation path:** if SimEvents is not in the trial license, implement the discrete-event queue simulation as a MATLAB script with identical parameters and outputs. The dashboard consumes JSON either way and does not care. State honestly on the slide which was used.

---

## 6. Testing

Not aiming for coverage — aiming for "nothing breaks live in front of judges."

**Contract tests:** every module returns contract-shaped output for a known input. Runs in CI from Day 1.

**Quality gate (the demo's most-watched moment):** blurry, dark, overexposed, poor-FOV, artifact-heavy, and borderline-recoverable images. The deliberately-poor demo image must reliably produce *Image ungradable — please recapture* every single run. Test it 20 times.

**ML:** one held-out example per grade 0–4; a near-threshold case; confirm ungradable inputs never reach the classifier.

**Application failure modes:** missing second eye, unsupported file type, oversized image, backend timeout, model unavailable, RAG unavailable. Each must degrade with a message, never a stack trace or a spinner that hangs.

**Simulation:** high patient volume, low bandwidth, few doctors — confirm the bottleneck indicator actually changes.

---

## 7. Work Split (6 people, 4 days)

Each track is self-contained with a defined artifact, so nobody blocks on anyone after the contract is frozen.

| Person | Track | Day-4 artifact |
|---|---|---|
| **P1 (lead)** | Preprocessing, classifier, thresholds, calibration, Grad-CAM, integration | Trained model + metrics + working pipeline |
| **P2** | MATLAB quality assessment, enhancement, classical lesion/vessel/OD | MATLAB functions matching the contract (Python fallback if license is late) |
| **P3** | FastAPI, SQLite, image storage, PDF report generation | Running API against the contract |
| **P4** | React: screening flow, result screen, explainability tabs, doctor queue | Working UI |
| **P5** | Simulink/SimEvents, three scenarios, results export | Model + scenario JSON + dashboard numbers |
| **P6** | RAG knowledge base, evaluation write-up, **ablation table**, demo script, slides | Working assistant + the two PS-mandated evidence artifacts |

P2 must not be blocked by the MATLAB license. They build in Python against the contract from Day 1 and port to MATLAB when the license lands.

---

## 8. Risk Register

| Risk | Likelihood | Mitigation |
|---|---|---|
| MATLAB license doesn't arrive in time | **High** | Every MATLAB module has a Python implementation behind the same contract. Demo works either way; MATLAB is presented as the scientific pipeline it genuinely is once available. |
| SimEvents not licensed | High | MATLAB-script discrete-event sim, same parameters and outputs. |
| IDRiD access not approved in 4 days | Medium | Request Day 0. Fallback: report APTOS held-out only and state plainly that external validation is pending — do not fake it. |
| Kaggle GPU quota exhausted | Low | 30 h/week is ample for B0 on 3,662 images. Keep image size at 456 and don't chase B3. |
| Sensitivity target (>90%) missed | Medium | Threshold is *tuned for sensitivity* on validation, accepting specificity cost — correct for screening, and defensible on clinical grounds. Report what was actually measured. |
| Integration collapses on Day 4 | Medium | Mitigated structurally by the Day-1 skeleton rule. This is why Day 1 exists. |
| Demo fails live | Medium | Pre-recorded fallback video of the full flow, plus frozen demo images that are tested repeatedly. |

---

## 9. What "Winning" Looks Like

Internal-round judges give roughly 10 minutes and reward **coverage of the PS bullets plus a demo that visibly works**, not marginal AUC. The pitch:

> Capture → Check → Analyze → Explain → Refer.

The differentiator is not the classifier — any team can train one. It is the **integrated pipeline**: quality gate that rejects bad images, adaptive enhancement, severity grading, lesion-level evidence, Grad-CAM, calibrated confidence, automated report, human-in-the-loop referral, evidence-grounded clinical RAG, and a district-scale Simulink model — with **external validation on Indian data** and an **honest statement of limitations**.

The last item is a differentiator, not a weakness. Most teams will overclaim clinical validation. Stating precisely what was and was not validated reads as competence to any judge who knows the domain.
