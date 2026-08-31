# Simulink / SimEvents deployment model

Models the district telemedicine pipeline so the resource allocation that
serves 100,000+ patients a year can be found rather than asserted.

## Owner
P5.

## Model

```
Patient Generator -> Image Acquisition -> Network/Bandwidth Delay
  -> Quality Assessment -> AI Processing -> Risk Classification
       |-> non-referable -> END
       |-> referable -> Doctor Queue -> Doctor Server -> Review -> END
```

## Parameters

Mirror `backend/app/services/simulation.py::Scenario` exactly, so the analytical
stub and the Simulink model stay comparable:
`patientsPerYear, healthCentres, doctors, bandwidthMbps, imagesPerPatient,
imageSizeMB, qualityCheckSec, aiInferenceSec, doctorReviewSec, referralRate,
screeningStations, captureMinutesPerPatient, doctorReviewHoursPerDay`

## Outputs

Export to `results/<scenario>.json` in the shape the dashboard already consumes:
patients/day, images/day, queue length (mean and max), waiting time, and
utilization for capture, network, AI and doctor, plus the bottleneck resource.

## If SimEvents is not licensed

Write the same discrete-event simulation as a MATLAB script with identical
parameters and outputs. The dashboard reads JSON either way. State honestly on
the slide which was used - and the numbers on the dashboard must always come
from a run, never be typed in.

## Current stub finding

The analytical stub reports **bandwidth as the binding constraint**, not doctor
capacity: AI triage cuts specialist reads to the referable fraction (~22%),
which is exactly the value proposition. Confirm or refute this with the real
model - a refutation is just as good a slide.
