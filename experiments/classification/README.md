# APTOS classifier training

## Run it on Kaggle

1. kaggle.com → **Create → Notebook**
2. **File → Import Notebook** → `train_aptos.ipynb`

   Import the notebook rather than pasting the `.py`. Pasting invites hand-edits,
   and an edit to the dataset path is what dropped an f-string prefix and broke
   the first run. The path is auto-detected now; you should not need to touch it.
3. Right panel → **Add Input** → search "APTOS 2019 Blindness Detection" → add
   the competition dataset
4. Right panel → **Session options**:
   - Accelerator: **GPU T4 x2** — *not* P100. Kaggle's PyTorch build is
     compiled for sm_70 and above; the P100 is sm_60 and will not run.
   - Internet: **On** — pretrained weights download at startup
5. **Save Version → Save & Run All (Commit)** — *not* the interactive Run All
   button.

   This matters more than it looks. An interactive session's `/kaggle/working`
   is scratch: close the tab, and the fold checkpoints written so far are gone
   with the container. A committed version runs detached for up to 12 hours and
   saves its output automatically. Start it, close the laptop, collect the
   output later.
6. Download from the Output tab of the finished version:
   - `dr_effnetb0_ordinal.onnx` → `matlab/models/`
   - `model_card.json` → the numbers for the metrics slide

Free tier gives 30 GPU-hours a week, so there is room to re-run.

## Why it is not slow any more

The first run took five hours, and the GPU was not the reason. `preprocess()`
decodes a 3000x2000 PNG, and it was being called from `__getitem__` — so every
image was decoded 60 times, once per epoch per fold. Measured locally that is
~104 ms per call against ~0.2 ms to read an already-preprocessed image, and
the T4 spent the run waiting on the CPU.

`build_cache()` now runs `preprocess()` once over the dataset into a memmapped
uint8 array in `/kaggle/temp` (~2.3 GB, scratch, not part of the saved output).
`preprocess()` is deterministic and augmentation still happens per
`__getitem__`, so this changes speed and nothing else about the model.

Per-epoch wall time is printed as the run goes. If a fold's epochs are slow,
that is now genuinely the GPU, and `SIZE` is the lever.

## What comes back

`model_card.json` carries the thresholds and the measured metrics: QWK,
confusion matrix, and the referable operating point with its sensitivity,
specificity and ROC-AUC.

`referable.threshold` is the grade-2 cut point itself, not a separately tuned
number. Referable DR is grade 2 and above, so the two cannot be fitted
independently - doing so put 2.5% of APTOS into a band that graded 2 while
withholding referral. `fit_thresholds` now maximizes QWK subject to that one
boundary clearing the sensitivity target, so the grade and the referral flag
can never disagree.

If the `referable` block carries a `note`, no feasible cut reached 90%
sensitivity. Report the operating point actually achieved. Do not claim the
target.

`oof_predictions.npy` is rewritten after every fold, so a run that dies at fold
4 still leaves the finished folds' predictions and checkpoints in the output.

## Then

- Import the ONNX net in MATLAB with `importNetworkFromONNX`, and run
  `gradCAM` against it — that is the explainability layer.
- Replace `backend/app/services/classifier.py`, keeping the return shape.
- Run the same model against IDRiD's 516 graded images. That is external
  validation on Indian data, and it is a separate number from the APTOS one.
  Never report them as the same thing.
