"""Grad-CAM explainability - PS requirement 4.

STUB. Replaced on day 2 by MATLAB's built-in gradCAM() applied to the
ONNX-imported network - a single call in Deep Learning Toolbox.

Wording discipline for anything built on this module: Grad-CAM shows the image
regions that influenced the prediction. It does not prove a lesion is present.
That is why lesion segmentation runs alongside it rather than instead of it.
"""

import hashlib
import io

import numpy as np
from PIL import Image

CAPTION = (
    "Grad-CAM highlights regions that most influenced the model's prediction. "
    "It is not proof that a highlighted region contains a lesion."
)


def gradcam(img: Image.Image, grade: int | None) -> Image.Image | None:
    """Return the original image with an attention heatmap overlaid."""
    if grade is None:
        return None

    base = img.convert("RGB")
    w, h = base.size
    rng = np.random.default_rng(
        int(hashlib.sha256(img.tobytes()[:4096]).hexdigest()[16:24], 16)
    )

    # A few Gaussian blobs standing in for attention peaks.
    yy, xx = np.mgrid[0:h, 0:w]
    heat = np.zeros((h, w), dtype=np.float64)
    for _ in range(rng.integers(2, 5)):
        cy, cx = rng.uniform(0.25, 0.75) * h, rng.uniform(0.25, 0.75) * w
        sigma = rng.uniform(0.06, 0.14) * min(h, w)
        heat += np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma**2))
    heat = heat / max(heat.max(), 1e-9)

    return overlay_heatmap(base, heat)


def overlay_heatmap(base: Image.Image, heat: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """Blend a 0-1 heatmap over an image using a blue-to-red colour ramp."""
    arr = np.asarray(base, dtype=np.float64)
    colour = np.zeros((*heat.shape, 3), dtype=np.float64)
    colour[:, :, 0] = np.clip(heat * 2.0, 0, 1) * 255      # red rises first
    colour[:, :, 1] = np.clip(1.0 - abs(heat - 0.5) * 2, 0, 1) * 255
    colour[:, :, 2] = np.clip((1.0 - heat) * 2.0, 0, 1) * 255

    mix = alpha * heat[:, :, None]
    return Image.fromarray(np.clip(arr * (1 - mix) + colour * mix, 0, 255).astype(np.uint8))


def to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
