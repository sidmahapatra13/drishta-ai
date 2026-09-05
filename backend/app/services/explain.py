"""Grad-CAM explainability - PS requirement 4.

Real, and computed without a backward pass.

The exported head is a global average pool followed by a single linear unit:

    /bn2/act/Mul_output_0  (1280 x 15 x 15)
      -> GlobalAveragePool -> Flatten -> Gemm(classifier.weight (1, 1280))

For that shape Grad-CAM collapses to CAM exactly. The Grad-CAM weight for
channel k is the spatial mean of d(grade)/dA_k, and through a GAP head that
derivative is classifier.weight[k] divided by the map area - the same constant
at every position. So the channel weighting a backward pass would arrive at is
the one the head's own weights already carry, and onnxruntime plus numpy is the
whole implementation. No torch, no MATLAB, no gradients in the request path.

Wording discipline for anything built on this module: Grad-CAM shows the image
regions that influenced the prediction. It does not prove a lesion is present.
That is why lesion segmentation runs alongside it rather than instead of it.
"""

from functools import lru_cache

import cv2
import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper
from PIL import Image

from . import classifier

#: Last activation before the pool. Grad-CAM is conventionally taken at the
#: final convolutional layer, which for this graph is the tensor feeding
#: GlobalAveragePool.
FEATURE_TENSOR = "/bn2/act/Mul_output_0"

HEAD_WEIGHT = "classifier.weight"
HEAD_BIAS = "classifier.bias"

#: The outer ring of the feature map is dropped before the map is shown.
#: EfficientNet-B0 strides the 456 px input by 32, so the grid is 15 cells wide
#: for 14.25 cells of real image and the outermost row and column are mostly
#: zero-padding. Measured across four APTOS images spanning grades 0 to 4, the
#: final column contributed 17-32 units on every one of them - including an
#: image whose entire interior was 0.0 - so left in it decided the maximum and
#: flattened the real attention. Dropping the ring costs the outer 1/15 of the
#: crop on each side, which is vignette and retinal margin.


@lru_cache(maxsize=1)
def _cam_model() -> tuple[ort.InferenceSession, np.ndarray, float] | None:
    """A session that also returns the feature map, plus the head's weight and bias.

    onnxruntime will only hand back tensors named in the graph's outputs, so
    the feature map is promoted to one on an in-memory copy. The weights on
    disk are untouched and the arithmetic is identical - this adds a second
    output to the same graph, it does not alter the one that produces `grade`.
    """
    if not classifier.MODEL_PATH.exists():
        return None

    model = onnx.load(str(classifier.MODEL_PATH))
    model.graph.output.append(
        onnx.helper.make_tensor_value_info(
            FEATURE_TENSOR, onnx.TensorProto.FLOAT, None
        )
    )
    weights = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
    session = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return (
        session,
        weights[HEAD_WEIGHT].reshape(-1).astype(np.float64),
        float(weights[HEAD_BIAS].reshape(-1)[0]),
    )


def severity_map(img: Image.Image) -> np.ndarray | None:
    """The raw contribution map, one cell per feature-map position.

    This is the whole of Grad-CAM for this network: sum_k w_k * A_k, before
    rectification. Its spatial mean plus the head bias reconstructs the exact
    score predict_dr reported - the identity that makes the backward pass
    unnecessary, and the property the tests pin.
    """
    loaded = _cam_model()
    if loaded is None:
        return None
    session, head, _ = loaded

    # The same tensor predict_dr is fed. A Grad-CAM computed on a differently
    # preprocessed image explains a prediction that was never made.
    feats = session.run([FEATURE_TENSOR], {"fundus": classifier.model_input(img)})[0]
    return np.einsum("c,chw->hw", head, feats[0].astype(np.float64))


def attention_map(img: Image.Image) -> np.ndarray | None:
    """A 0-1 attention map over the whole photograph, or None.

    Normalised against its own maximum, so intensity is relative *within one
    eye*. Two patients' maps cannot be compared by brightness, and neither can
    the two eyes of one patient.
    """
    cam = severity_map(img)
    if cam is None:
        return None

    # Only regions that pushed the severity score up are evidence for the grade
    # the model reported; the negative half of the map argues for a lower one.
    cam = np.maximum(cam, 0.0)

    cells = cam.shape[0]
    cam = cam[1:-1, 1:-1]
    if cam.size == 0:
        return None

    # Place the surviving cells back over the region of the photograph they
    # actually came from: the crop the network was fed, inset by the ring.
    arr = np.asarray(img.convert("RGB"))
    box = classifier.retina_bbox(arr)
    top, bottom, left, right = box if box else (0, arr.shape[0], 0, arr.shape[1])
    inset_y = round((bottom - top) / cells)
    inset_x = round((right - left) / cells)
    top, bottom = top + inset_y, bottom - inset_y
    left, right = left + inset_x, right - inset_x
    if bottom <= top or right <= left:
        return None

    full = np.zeros(arr.shape[:2], dtype=np.float64)
    full[top:bottom, left:right] = cv2.resize(
        cam, (right - left, bottom - top), interpolation=cv2.INTER_LINEAR
    )

    # Mask before normalising, not after. Attention landing in the black
    # surround is an artifact, and left in it sets the maximum and scales the
    # real in-retina attention down towards invisibility.
    full *= classifier.retina_mask(arr)
    if full.max() <= 1e-9:
        return None
    return full / full.max()


def gradcam(img: Image.Image, grade: int | None) -> Image.Image | None:
    """Return the original image with an attention heatmap overlaid."""
    if grade is None:
        return None
    heat = attention_map(img)
    if heat is None:
        return None
    return overlay_heatmap(img.convert("RGB"), heat)


def overlay_heatmap(base: Image.Image, heat: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """Blend a 0-1 heatmap over an image using a blue-to-red colour ramp."""
    arr = np.asarray(base, dtype=np.float64)
    colour = np.zeros((*heat.shape, 3), dtype=np.float64)
    colour[:, :, 0] = np.clip(heat * 2.0, 0, 1) * 255      # red rises first
    colour[:, :, 1] = np.clip(1.0 - abs(heat - 0.5) * 2, 0, 1) * 255
    colour[:, :, 2] = np.clip((1.0 - heat) * 2.0, 0, 1) * 255

    mix = alpha * heat[:, :, None]
    return Image.fromarray(np.clip(arr * (1 - mix) + colour * mix, 0, 255).astype(np.uint8))
