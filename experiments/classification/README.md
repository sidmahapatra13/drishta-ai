# APTOS classifier training

## Run it on Kaggle

1. kaggle.com → **Create → Notebook**
2. **File → Import Notebook**, or paste `train_aptos.py` into one cell
3. Right panel → **Add Input** → search "APTOS 2019 Blindness Detection" → add
   the competition dataset
4. Right panel → **Session options**:
   - Accelerator: **GPU T4 x2** or **P100**
   - Internet: **On** — pretrained weights download at startup
5. **Run All**. Expect roughly 1.5–2 hours for 5 folds.
6. Download from the Output tab:
   - `dr_effnetb0_ordinal.onnx` → `matlab/models/`
   - `model_card.json` → the numbers for the metrics slide

Free tier gives 30 GPU-hours a week, so this costs about 6% of the quota. There
is room to re-run.

## What comes back

`model_card.json` carries the thresholds and the measured metrics: QWK,
confusion matrix, and the referable operating point with its sensitivity,
specificity and ROC-AUC.

If `referable.threshold` is `null`, no cut point reached 90% sensitivity.
Report the operating point actually achieved. Do not claim the target.

## Then

- Import the ONNX net in MATLAB with `importNetworkFromONNX`, and run
  `gradCAM` against it — that is the explainability layer.
- Replace `backend/app/services/classifier.py`, keeping the return shape.
- Run the same model against IDRiD's 516 graded images. That is external
  validation on Indian data, and it is a separate number from the APTOS one.
  Never report them as the same thing.
