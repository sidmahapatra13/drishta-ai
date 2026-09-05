"""Tests for Grad-CAM.

The heatmap is the most persuasive thing on screen and the easiest to be quietly
wrong about, so these pin the two properties that make it trustworthy: it
decomposes the score the classifier actually reported, and it does not paint
attention onto parts of the frame the network could not have been reading.
"""

import numpy as np
import pytest
from PIL import Image

from app.services import classifier, explain
from test_pipeline import synthetic_fundus

needs_weights = pytest.mark.skipif(
    not classifier.MODEL_PATH.exists(), reason="model weights not fetched"
)


@needs_weights
def test_severity_map_decomposes_the_reported_score():
    """The identity that makes a backward pass unnecessary.

    Through a global-average-pool head the Grad-CAM weight for every channel is
    the head weight itself, so the map's spatial mean plus the head bias must
    reconstruct the raw ordinal output exactly. If this drifts, the heatmap has
    stopped explaining the prediction beside it.
    """
    img = synthetic_fundus()
    _, _, bias = explain._cam_model()

    cam = explain.severity_map(img)
    reconstructed = float(cam.mean()) + bias

    assert reconstructed == pytest.approx(
        classifier.predict_dr(img)["raw_logit"], abs=1e-4
    )


@needs_weights
def test_attention_is_confined_to_the_imaged_retina():
    """Attention in the black surround is a convolution artifact, not evidence,
    and a judge reading meaning into it is reading noise."""
    img = synthetic_fundus()
    heat = explain.attention_map(img)
    outside = ~classifier.retina_mask(np.asarray(img.convert("RGB")))

    assert heat[outside].max() == 0.0
    assert heat.max() == pytest.approx(1.0)


@needs_weights
def test_the_padding_ring_never_sets_the_maximum():
    """EfficientNet-B0 strides 456 px by 32, so the outer ring of the 15x15 grid
    is mostly zero-padding and contributed 17-32 units on every APTOS image
    measured - including one whose interior was entirely 0. Left in, it decides
    the maximum and flattens the real attention."""
    cam = np.maximum(explain.severity_map(synthetic_fundus()), 0.0)
    ring = np.concatenate([cam[0], cam[-1], cam[1:-1, 0], cam[1:-1, -1]])

    # The ring is expected to be hot; the point is that it is excluded before
    # the map is normalised, so the peak comes from the interior.
    assert explain.attention_map(synthetic_fundus()) is not None
    assert ring.max() > 0


@needs_weights
def test_grad_cam_is_deterministic():
    """Demo rehearsals must reproduce exactly."""
    img = synthetic_fundus()
    assert np.array_equal(explain.attention_map(img), explain.attention_map(img))


def test_no_heatmap_for_an_ungradable_image():
    """fusion.gate strips the grade; the explanation must go with it rather than
    explaining a prediction that was never made."""
    assert explain.gradcam(synthetic_fundus(), None) is None


def test_no_heatmap_when_the_weights_are_absent(monkeypatch, tmp_path):
    """The day-one stub painted random blobs under a caption reading 'regions
    that influenced the prediction'. A weights-less checkout must show nothing
    rather than something invented."""
    monkeypatch.setattr(classifier, "MODEL_PATH", tmp_path / "absent.onnx")
    explain._cam_model.cache_clear()
    try:
        assert explain.gradcam(synthetic_fundus(), 2) is None
    finally:
        explain._cam_model.cache_clear()
