# Frontend rebuild and quality-gate correction

**Date:** 2026-09-06
**Deadline:** 2026-09-12 — live demo and submission on the same day
**Scope:** Tracks 1 and 2 of six. The remaining four (external validation and
calibration, PDF report, lesion detection, Simulink and RAG) get their own
specs.

This spec covers the two pieces of work that share a subject: what the system
tells a health worker when it refuses to grade an image, and the interface that
tells them. They are specified together because the second is worthless if the
first is wrong — a beautifully rendered sentence naming the wrong problem is
worse than an ugly one naming the right problem.

---

## 1. The quality gate is measuring the wrong thing

### 1.1 What was measured

Four real APTOS captures from `local-data/demo/`, brightness scaled linearly
from 1.0 to 0.05, run through the current `quality.assess_quality`. The probe
recorded every intermediate the score is built from, not just the score.

Three defects, in the order they matter.

**Focus is a contrast metric, not a focus metric.** `focus_score` is the
variance of the Laplacian of raw luminance. The Laplacian is a linear operator,
so scaling image intensity by *k* scales the Laplacian by *k* and its variance
by *k²*. A perfectly sharp photograph measured at 20% brightness:

| brightness | 0dbaa09a458c | 4134b290f5f3 | 838c87c63422 |
|---|---|---|---|
| 1.00 | 1.000 | 1.000 | 1.000 |
| 0.60 | 0.543 | 0.547 | 0.517 |
| 0.40 | 0.329 | 0.314 | 0.301 |
| 0.20 | 0.170 | 0.175 | 0.146 |

`FOCUS_UNGRADABLE` is 0.35. A sharp but dim capture is therefore reported as
severely defocused, and `assess_quality` inserts `"severe defocus"` at position
0 of the issue list, ahead of the `"underexposed"` that also fired. The health
worker is told to refocus a camera that was never out of focus.

This is not only a darkness problem. It means focus scores partly measure
exposure on every image, so a dim capture from a cheap handheld camera is
penalised for its camera rather than its sharpness — the exact cross-device
comparability that `WORK_RESOLUTION` was introduced to protect.

**The retina mask discards the dark retina.** `_retina_mask` thresholds at
`max(12.0, gray.mean() * 0.12)`. The absolute floor wins on any underexposed
frame, so retinal pixels darker than 12 are excluded from the mask that
`illumination_score` averages over. Measured on `4d47300e3ddb`:

| brightness | mask, with the 12.0 floor | mask, relative threshold only |
|---|---|---|
| 1.00 | 0.904 | 0.907 |
| 0.30 | 0.682 | 0.906 |
| 0.20 | 0.267 | 0.905 |
| 0.12 | **0.019** | 0.905 |

Illumination ends up measuring the brightest surviving sliver of a dark image,
which is why the score plateaus near 0.56 instead of continuing to fall. This
is the plateau recorded as an open question in the session-two handoff.

**`ILLUM_UNGRADABLE = 0.35` is unreachable.** The score composes additively:

```
0.5 * exposure + 0.3 * evenness + 0.2 * (1 - clipped)
```

An evenly lit, unclipped frame scores at least 0.50 whatever its exposure. A
pitch-black retina scores 0.50. Darkness alone can never trip the illumination
branch of the gate; the composite score catches it instead, which is why the
image is still correctly rejected and why this went unnoticed.

The handoff records that `"underexposed"` never fires. That is incorrect — it
fires from 50% brightness downward on every image measured. The defect is the
ordering, not the detection.

### 1.2 The correction

Four changes, shipped together.

**Focus becomes exposure-invariant, measured on the green channel.** Green
carries the retinal structure: red saturates in the choroid and blue records
almost nothing from the retinal layers, which is why retinal image quality work
reads sharpness off green. Normalising Laplacian variance by the square of mean
retinal luminance cancels the *k²* exactly, turning an absolute contrast
measurement into a relative one:

```
focus_raw = var(laplacian(green[mask])) / mean(green[mask])**2
```

`FOCUS_REFERENCE_VARIANCE` is refitted against this new quantity. It is no
longer a variance in intensity units; it is renamed to say so.

**The retina mask survives underexposure.** The absolute floor of 12.0 is
replaced by a threshold relative to the frame's own dynamic range, keeping the
existing guard that an empty or structureless frame reports no retina rather
than a garbage mask.

**Illumination composes multiplicatively.**

```
exposure * (0.6 * evenness + 0.4 * (1 - clipped))
```

A retina that cannot be seen is not 50% quality. No amount of evenness rescues
a black frame, and `ILLUM_UNGRADABLE` becomes reachable.

**Issues are ranked by severity, not by insertion order.** Each failing metric
is scored by how far below its own floor it sits, and the issue list is sorted
by that margin. The recapture sentence then leads with the dominant failure.

### 1.3 Acceptance

These are gate thresholds, so the change is not complete until it is shown not
to have moved anything it should not have.

- All five demo captures keep the status they have today. `4d47300e3ddb` stays
  borderline and still routes through enhancement; `ungradable-defocus.png`
  stays ungradable.
- The existing calibration constraint holds on the 39 APTOS images: every real
  photograph is kept, every Gaussian blur of sigma 4 or above is rejected.
- Focus is flat across the brightness sweep — a sharp image scores within a
  narrow band from 1.0 down to 0.2 brightness, where today it falls by 6x.
- A dark capture's recapture sentence names underexposure first and does not
  claim defocus.
- The 23 existing tests still pass, and new tests pin each property above.

Threshold values are the user's call, not the implementation's. The before and
after table is presented for sign-off rather than shipped silently.

---

## 2. The interface

### 2.1 The problem with the current one

`App.tsx` is 538 lines holding nine components, and `index.css` is 316 lines of
global classes. That is survivable. Three things are not.

**Two users are rendered identically.** A health worker at a primary health
centre and an ophthalmologist triaging a queue have almost nothing in common:
one needs a single instruction in a low-light room, the other needs density and
speed. Both currently receive the same stack of equal-weight `.panel` boxes.

**The refusal is not the hero.** The most defensible thing this system does is
decline to answer. Today it renders as a bordered warning box, below two figure
images, inside the per-eye panel, below the fold. A judge watching the demo sees
a form, then some pictures, then eventually a yellow box.

**Everything is the same weight.** Eleven `.eyebrow` labels — tracked-out
uppercase mono — sit above eleven panels sharing one border-radius and one
border colour. Nothing on the page tells you what to read first.

### 2.2 What stays

The design direction is kept and sharpened, not replaced.

The palette is derived from fundus anatomy rather than a brand deck: the near
black of the vitreous surround, the orange-red of the retina, the cream-yellow
of the optic disc. The interface is dark because that is how retinal images are
actually read. IBM Plex is chosen for technical contexts and for its Devanagari
companion, which is what regional-language support will need. The severity
ladder — the ordinal scale drawn with the referral threshold as a real line —
is the signature element and is strengthened, not removed.

The mono/sans rule is kept and enforced more strictly than it is today: a
measured value is set in mono, and nothing that is not a measurement is.

### 2.3 What changes

**The verdict gets its own column.** The screening result becomes a two-column
instrument: capture state on the left (image, Grad-CAM, quality readings),
decision on the right. The right column is the page's centre of gravity and
changes completely between the two outcomes.

Graded:

```
┌──────────────────────────────┬────────────────────────────────┐
│  [fundus]      [attention]   │  Moderate NPDR                 │
│                              │  grade 2 · referable           │
│  focus         ▓▓▓▓▓▓▓░  .71 │                                │
│  illumination  ▓▓▓▓▓▓▓▓  .83 │  0  No apparent DR             │
│  field of view ▓▓▓▓▓▓▓░  .79 │  1  Mild NPDR                  │
│                              │  ═══════ referral threshold    │
│                              │  2  Moderate NPDR      ◄ here  │
│                              │  3  Severe NPDR                │
│                              │  4  Proliferative DR           │
└──────────────────────────────┴────────────────────────────────┘
```

Refused:

```
┌──────────────────────────────┬────────────────────────────────┐
│  [fundus]                    │  Too dark to grade             │
│                              │                                │
│  focus         ▓▓▓▓▓▓▓░  .71 │  Increase the flash and take   │
│  illumination  ▓▓░░░░░░  .19 │  the left eye again.           │
│                └── below the │                                │
│                    line      │  No grade is reported for an   │
│  field of view ▓▓▓▓▓▓▓▓  .83 │  ungradable image.             │
│                              │                                │
│                              │  [ Retake left eye ]           │
└──────────────────────────────┴────────────────────────────────┘
```

The refusal occupies the position a grade would have occupied, at the same
size. That is the whole point: the system is not failing, it is reporting.

**Quality readings become gauges with the gate drawn on them.** The three thin
progress bars become tracks with the ungradable and borderline bands marked, so
a reading shows not just its value but which side of the line it sits on and by
how much. This is the visual counterpart of the section 1 fix: once focus stops
responding to brightness, the gauge that fails is the one that should.

**Eyebrow labels are cut from eleven to the few that carry structure.** A label
that only restates the heading below it is removed.

**A type scale replaces ad-hoc sizes.** One modular scale in tokens; no inline
`fontSize` anywhere.

**Provenance moves into the chrome.** `model_version` and `threshold_version`
are shown persistently rather than being absent from the UI entirely. A judge
should never have to ask which model produced a number.

### 2.4 Structure

```
src/
  App.tsx                  shell, tab state, routing only
  api.ts                   unchanged - frozen contract mirror
  lib/grade.ts             GRADES, colourFor, pct, issue phrasing
  styles/tokens.css        palette, type scale, spacing, radii
  styles/base.css          reset, elements, focus rings
  components/
    ProvenanceBar.tsx      model and threshold version
    ScreeningForm.tsx      patient fields, eye inputs, submit
    PatientResultView.tsx  composes the two columns
    VerdictPanel.tsx       grade or refusal - the right column
    SeverityLadder.tsx     the signature element
    QualityGauges.tsx      three readings against the gate
    RecaptureNotice.tsx    the refusal, at verdict weight
    EyePanel.tsx           capture state - the left column
    LesionEvidence.tsx     silent until lesions.py is real
    ReviewQueue.tsx        ophthalmologist triage
    CaseView.tsx           queue detail plus review actions
    SimulationTable.tsx    district deployment scenarios
```

Each component owns a `.css` file beside it, importing tokens. Plain classes,
not CSS modules: the existing stylesheet is global and readable, and a
half-migration to modules six days from the deadline buys nothing.

`tailwindcss` and `@tailwindcss/vite` are removed from `package.json`. They have
never been imported in `index.css` and are absent from the Vite config, so the
choice is between wiring up a second styling system with six days left or
deleting two unused dependencies. The token system already exists and carries
its reasoning in comments.

### 2.5 Copy

The interface speaks in instructions, not diagnoses. It addresses the person
holding the camera.

- `Too dark to grade` / `Increase the flash and take the left eye again.`
- Never `Error`, never `Failed`, never an apology. The instrument reports.
- A grade 0 is stated as flatly as a grade 4. No congratulation, no green tick.
- Grad-CAM keeps its caption: regions that influenced the prediction, not proof
  of a lesion.
- The disclaimer stays on every result.

Recapture wording is derived from the ranked issue list of section 1.2, so the
sentence the health worker reads is generated from the metric that actually
failed rather than from a fixed string.

### 2.6 Acceptance

- `App.tsx` is under 100 lines; no component file exceeds 150.
- `npm run build` passes with no TypeScript errors; `npm run lint` is clean.
- The five demo captures each render correctly end to end, including the
  refusal, verified by screenshot rather than by assertion.
- Keyboard focus is visible on every interactive element; the reduced-motion
  query is respected; the layout reflows to a single column on a tablet.
- No inline `fontSize` remains in any component.
- The frozen contract in `api.ts` is unchanged. This is a presentation-layer
  rebuild; if it needs a contract change, that is a signal the change is wrong.

---

## 3. Order of work

The quality gate lands first. Its output is the copy the interface renders, so
building the refusal view against the current wrong sentence would mean
building it twice.

1. Quality gate correction, with the before/after table for sign-off.
2. Tokens, type scale, base styles; Tailwind removed.
3. Components extracted from `App.tsx` behind unchanged behaviour.
4. The verdict column, gauges and refusal view — the new design.
5. Screenshot pass against all five demo captures.

## 4. Risks

**Retuning the gate changes demo behaviour.** Mitigated by the acceptance
criteria in 1.3: the five demo captures must keep their current statuses, and
that is checked before anything else is built on top.

**A frontend rewrite breaks a working demo.** Mitigated by ordering: components
are extracted with behaviour unchanged and verified before the new design is
applied, so there is a working checkpoint between the two.

**Scope creep into the other four tracks.** The lesion evidence panel and the
report button are built as present-but-silent, guarding on absence exactly as
the current UI does. They light up when their tracks land.
