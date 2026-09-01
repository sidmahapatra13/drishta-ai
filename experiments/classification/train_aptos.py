"""APTOS 2019 diabetic retinopathy grading — Kaggle training script.

Run on Kaggle with GPU and Internet both ON (pretrained weights need internet).
Attach the "APTOS 2019 Blindness Detection" competition dataset.

Use the **T4 x2** accelerator, not P100. Kaggle's current PyTorch build is
compiled for sm_70 and above; the P100 is sm_60 and will not run.

Produces, in /kaggle/working:
    dr_effnetb0_ordinal.onnx   the network, for MATLAB importNetworkFromONNX
    model_card.json            thresholds and metrics — the numbers for slides

Two choices here differ from the obvious approach, and both matter:

1. Ben Graham preprocessing. Circle-crop the retina, then subtract a heavy
   Gaussian blur. This removes the per-camera colour cast that otherwise
   dominates the signal, and it is the single largest accuracy lever on APTOS.

2. Ordinal regression, not 5-class softmax. DR grades are ordered - confusing
   grade 3 for 4 is a smaller error than confusing 0 for 4 - and softmax
   cross-entropy treats all confusions alike. One continuous output with four
   tuned cut points optimizes quadratic weighted kappa directly.

3. Preprocessing is cached to disk before training, not run per __getitem__.
   The first run took five hours because it decoded each 3000x2000 PNG sixty
   times - twelve epochs across five folds - and the T4 sat waiting on the CPU.

Save this as a **version** (Save Version -> Save & Run All), not an interactive
Run All. A batch run survives the browser closing; an interactive session's
/kaggle/working does not.
"""

import glob
import json
import os
import time

import cv2
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from scipy.optimize import minimize
from sklearn.metrics import cohen_kappa_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset

OUT = "/kaggle/working"
SIZE = 456
BATCH = 16
EPOCHS = 12
FOLDS = 5
LR = 3e-4
SEED = 42
#: Kaggle gives four cores. With preprocessing cached the loader only has to
#: augment and cast, but four workers still keeps the GPU fed at batch 16.
WORKERS = 4

#: The PS fixes referable DR at grade 2 and above. Not a tunable.
REFERABLE_FROM = 2
#: Screening misses cost more than false alarms, so the operating point is
#: chosen to clear this sensitivity, then take the best specificity available.
TARGET_SENSITIVITY = 0.90

torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def find_data() -> str:
    """Locate the attached APTOS dataset.

    Kaggle has moved competition inputs between /kaggle/input/<slug> and
    /kaggle/input/competitions/<slug>, and hand-editing the path is what broke
    this script the first time it ran. Search instead of hard-coding, and fail
    with the actual directory listing rather than an OpenCV assertion 400
    images later.
    """
    for candidate in glob.glob("/kaggle/input/**/train.csv", recursive=True):
        root = os.path.dirname(candidate)
        if os.path.isdir(f"{root}/train_images"):
            print(f"dataset: {root}")
            return root

    available = glob.glob("/kaggle/input/*") + glob.glob("/kaggle/input/*/*")
    raise FileNotFoundError(
        "Could not find train.csv beside a train_images/ directory under "
        f"/kaggle/input. Found: {available or 'nothing - no dataset attached'}. "
        "Add the APTOS 2019 Blindness Detection competition data via Add Input."
    )


DATA = find_data()


def preprocess(path: str) -> np.ndarray:
    """Ben Graham preprocessing: circle-crop, resize, subtract local average."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Crop to the retinal disc - the black surround carries no signal and
    # varies in size between cameras.
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = gray > gray.mean() * 0.12
    if mask.any():
        ys, xs = np.nonzero(mask)
        img = img[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]

    img = cv2.resize(img, (SIZE, SIZE))

    # Subtract a heavy blur. What survives is local contrast - lesions - with
    # the camera's colour cast and illumination gradient removed.
    blur = cv2.GaussianBlur(img, (0, 0), SIZE / 30)
    img = cv2.addWeighted(img, 4, blur, -4, 128)

    return img


class Unprocessed(Dataset):
    """Yields preprocessed images in id order. Used once, to fill the cache."""

    def __init__(self, ids):
        self.ids = ids

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i):
        return torch.from_numpy(preprocess(f"{DATA}/train_images/{self.ids[i]}.png"))


def build_cache(ids) -> np.ndarray:
    """Preprocess every image once into a memmapped uint8 array.

    preprocess() is deterministic, so hoisting it out of the training loop
    changes nothing about the model - it only stops the same PNG being decoded
    once per epoch per fold. Augmentation still happens per __getitem__.

    Written to /kaggle/temp, not /kaggle/working: it is 2.3 GB of scratch and
    has no business in the saved output. Workers memmap it rather than each
    holding a copy.
    """
    path = "/kaggle/temp/preprocessed.npy"
    os.makedirs("/kaggle/temp", exist_ok=True)
    cache = np.lib.format.open_memmap(
        path, mode="w+", dtype=np.uint8, shape=(len(ids), SIZE, SIZE, 3)
    )

    # A DataLoader, not a ProcessPoolExecutor: multiprocessing pools are
    # unreliable inside a notebook kernel, and torch's worker handling is
    # already the mechanism this script depends on. shuffle is off, so batches
    # arrive in id order and row i is ids[i].
    started = time.time()
    loader = DataLoader(Unprocessed(ids), batch_size=32, num_workers=WORKERS)
    at = 0
    for batch in loader:
        cache[at : at + len(batch)] = batch.numpy()
        at += len(batch)
        if at % 512 == 0:
            print(f"  cached {at}/{len(ids)}", flush=True)
    cache.flush()
    assert at == len(ids), f"cached {at} of {len(ids)} images"

    print(f"cache built in {time.time() - started:.0f}s -> {path}")
    return np.load(path, mmap_mode="r")


class Retinas(Dataset):
    def __init__(self, df: pd.DataFrame, cache: np.ndarray, train: bool):
        self.df = df.reset_index(drop=True)
        self.cache = cache
        self.train = train

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = self.cache[row.cache_row]

        if self.train:
            # Fundus images have no canonical orientation, so flips and
            # rotations are label-preserving. Nothing that changes colour -
            # preprocessing already normalized it.
            if np.random.rand() > 0.5:
                img = np.fliplr(img)
            if np.random.rand() > 0.5:
                img = np.flipud(img)
            img = np.rot90(img, np.random.randint(4))

        img = np.ascontiguousarray(img.transpose(2, 0, 1), dtype=np.float32) / 255.0
        return torch.from_numpy(img), torch.tensor(row.diagnosis, dtype=torch.float32)


def build_model() -> nn.Module:
    """EfficientNet-B0 with a single continuous output."""
    return timm.create_model("efficientnet_b0", pretrained=True, num_classes=1)


def train_fold(train_df, valid_df, cache: np.ndarray, fold: int) -> np.ndarray:
    model = build_model().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    # Smooth L1 rather than MSE: it is less dominated by the rare grade-4 cases.
    loss_fn = nn.SmoothL1Loss()
    scaler = torch.amp.GradScaler(DEVICE)

    train_dl = DataLoader(Retinas(train_df, cache, True), batch_size=BATCH, shuffle=True,
                          num_workers=WORKERS, pin_memory=True, drop_last=True,
                          persistent_workers=True)
    valid_dl = DataLoader(Retinas(valid_df, cache, False), batch_size=BATCH,
                          num_workers=WORKERS)

    for epoch in range(EPOCHS):
        started = time.time()
        model.train()
        total = 0.0
        for x, y in train_dl:
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(DEVICE):
                loss = loss_fn(model(x).squeeze(1), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += loss.item() * len(x)
        sched.step()
        print(f"  fold {fold} epoch {epoch + 1}/{EPOCHS}  loss {total / len(train_df):.4f}"
              f"  {time.time() - started:.0f}s", flush=True)

    model.eval()
    preds = []
    with torch.no_grad():
        for x, _ in valid_dl:
            with torch.amp.autocast(DEVICE):
                preds.append(model(x.to(DEVICE)).squeeze(1).float().cpu().numpy())

    torch.save(model.state_dict(), f"{OUT}/fold{fold}.pt")
    return np.concatenate(preds)


def apply_thresholds(raw: np.ndarray, cuts) -> np.ndarray:
    return np.digitize(raw, np.sort(cuts))


def fit_thresholds(raw: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Find the four cut points that maximize quadratic weighted kappa.

    Evenly spaced cuts at 0.5/1.5/2.5/3.5 assume the regression output is
    calibrated to the label scale, which it is not - the model shrinks toward
    the mean because most images are grade 0.
    """
    def negative_qwk(cuts):
        return -cohen_kappa_score(true, apply_thresholds(raw, cuts), weights="quadratic")

    best = minimize(negative_qwk, [0.5, 1.5, 2.5, 3.5], method="nelder-mead")
    return np.sort(best.x)


def fit_referable_threshold(raw: np.ndarray, true: np.ndarray) -> dict:
    """Pick the referable cut that clears TARGET_SENSITIVITY, then maximizes
    specificity. Sensitivity is the binding constraint in screening."""
    referable = (true >= REFERABLE_FROM).astype(int)
    best = None

    for cut in np.linspace(raw.min(), raw.max(), 500):
        pred = (raw >= cut).astype(int)
        tp = int(((pred == 1) & (referable == 1)).sum())
        fn = int(((pred == 0) & (referable == 1)).sum())
        tn = int(((pred == 0) & (referable == 0)).sum())
        fp = int(((pred == 1) & (referable == 0)).sum())

        sens = tp / max(tp + fn, 1)
        spec = tn / max(tn + fp, 1)
        if sens >= TARGET_SENSITIVITY and (best is None or spec > best["specificity"]):
            best = {"threshold": float(cut), "sensitivity": sens, "specificity": spec,
                    "tp": tp, "fp": fp, "tn": tn, "fn": fn}

    if best is None:
        return {"threshold": None,
                "note": f"No cut point reached {TARGET_SENSITIVITY:.0%} sensitivity. "
                        "Report the achieved operating point; do not claim the target."}

    best["roc_auc"] = float(roc_auc_score(referable, raw))
    return best


def export_onnx() -> None:
    """Export fold 0 for MATLAB. Static input, dynamic batch."""
    model = build_model()
    model.load_state_dict(torch.load(f"{OUT}/fold0.pt", map_location="cpu"))
    model.eval()
    torch.onnx.export(
        model,
        torch.randn(1, 3, SIZE, SIZE),
        f"{OUT}/dr_effnetb0_ordinal.onnx",
        input_names=["fundus"],
        output_names=["grade"],
        dynamic_axes={"fundus": {0: "batch"}, "grade": {0: "batch"}},
        opset_version=17,
    )


def main() -> None:
    df = pd.read_csv(f"{DATA}/train.csv")
    print(f"{len(df)} images, grade counts:\n{df.diagnosis.value_counts().sort_index()}")

    # Row into the preprocessed cache. It has to survive the fold split, which
    # hands train_fold a subset with its own index.
    df["cache_row"] = np.arange(len(df))
    cache = build_cache(df.id_code.tolist())

    oof = np.zeros(len(df), dtype=np.float32)
    folds = StratifiedKFold(FOLDS, shuffle=True, random_state=SEED)
    for fold, (tr, va) in enumerate(folds.split(df, df.diagnosis)):
        print(f"fold {fold}", flush=True)
        oof[va] = train_fold(df.iloc[tr], df.iloc[va], cache, fold)
        # Checkpoint after every fold. If the run dies at fold 4 the finished
        # folds are still in the output, not thrown away.
        np.save(f"{OUT}/oof_predictions.npy", oof)

    true = df.diagnosis.values
    cuts = fit_thresholds(oof, true)
    graded = apply_thresholds(oof, cuts)

    card = {
        "model": "efficientnet_b0-ordinal",
        "trained_on": "APTOS 2019 train split, 5-fold CV",
        "image_size": SIZE,
        "preprocessing": "ben-graham circle-crop + gaussian subtraction",
        "grade_thresholds": [float(c) for c in cuts],
        "qwk": float(cohen_kappa_score(true, graded, weights="quadratic")),
        "confusion_matrix": confusion_matrix(true, graded).tolist(),
        "referable": fit_referable_threshold(oof, true),
        "caveat": (
            "Cross-validated on APTOS only. Not clinical validation, and not "
            "evidence of generalization - report IDRiD external validation "
            "separately."
        ),
    }

    with open(f"{OUT}/model_card.json", "w") as f:
        json.dump(card, f, indent=2)

    export_onnx()

    print(json.dumps({k: v for k, v in card.items() if k != "confusion_matrix"}, indent=2))


if __name__ == "__main__":
    main()
