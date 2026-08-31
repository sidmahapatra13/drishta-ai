# MATLAB pipeline

MATLAB owns the scientific pipeline: quality assessment, enhancement, classical
segmentation, Grad-CAM, calibration, metrics, and Simulink. Training happens on
Kaggle in PyTorch because Parallel Computing Toolbox requires CUDA and the
team's hardware is Apple Silicon; the trained network is exported to ONNX and
imported here with `importNetworkFromONNX`.

This is not a workaround around the MATLAB requirement. The analysis pipeline,
the explainability layer and the validation are all MATLAB, and that is what
the problem statement asks for.

## Owner
P2, with P1 on Grad-CAM and calibration.

## Contract

Every function mirrors the Python implementation in `backend/app/services/` and
must return the same fields, so the two can be compared and either can back the
API:

```matlab
assessQuality(img)      % -> struct(focus, illumination, fieldOfView, score, status, issues)
enhance(img)            % -> enhanced image  (adapthisteq, illumination norm, denoise)
predictDR(img, net)     % -> struct(grade, probabilities)
segmentLesions(img)     % -> struct(masks, lesionTypes, counts, confidence)
explainPrediction(img, net)  % -> Grad-CAM map via built-in gradCAM()
calibrateConfidence(p, T)    % -> temperature-scaled probabilities
```

`status` is one of `gradable`, `borderline`, `ungradable` - spelled exactly as
in `backend/app/contract.py`.

## Order of work

1. `quality_assessment/` - port `services/quality.py`. Same signals, same
   thresholds, so the outputs can be diffed against the Python reference.
2. `preprocessing/` - `adapthisteq` for CLAHE, illumination normalization,
   denoising. Ben Graham preprocessing (circle crop, resize, subtract Gaussian
   blur) for the classifier input.
3. `explainability/` - `gradCAM(net, img, label)`. One built-in call.
4. `segmentation/` - classical first: green channel, top-hat, Frangi for
   vessels; bright-lesion morphology for exudates with optic disc suppressed.
5. `evaluation/` - confusion matrix, ROC, QWK, sensitivity/specificity at the
   referable threshold, reliability diagram, ECE.

## Not attempted

Neovascularization detection. No public pixel-level NV masks exist, so it is
scoped out explicitly rather than faked. Say so on the slide.
