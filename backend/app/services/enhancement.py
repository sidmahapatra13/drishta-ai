"""Adaptive enhancement for borderline images - PS requirement 1.

Only borderline images are enhanced. Clearly unusable images are rejected, not
enhanced and pushed through the classifier anyway: enhancing a blurred
photograph invents contrast that was never in the retina, which is worse than
rejecting it.

MATLAB (adapthisteq / imreducehaze) owns this in the final pipeline. This
numpy implementation keeps the demo alive if the engine is unavailable and
gives us a reference to compare the MATLAB output against.
"""

import numpy as np
from PIL import Image

from ..contract import QualityReport, QualityStatus

CLAHE_TILES = 8
CLAHE_CLIP = 3.0


def _clahe(channel: np.ndarray, tiles: int = CLAHE_TILES, clip: float = CLAHE_CLIP) -> np.ndarray:
    """Contrast Limited Adaptive Histogram Equalization on a uint8 plane.

    Plain histogram equalization over the whole image washes out the macula to
    boost the bright optic disc. CLAHE equalizes per tile and clips the
    histogram first, so local lesion contrast improves without amplifying noise
    in the dark periphery. Tile results are bilinearly interpolated to avoid
    visible seams.
    """
    h, w = channel.shape
    ty, tx = max(h // tiles, 1), max(w // tiles, 1)
    ny, nx = max(h // ty, 1), max(w // tx, 1)

    # Build a lookup table per tile.
    luts = np.empty((ny, nx, 256), dtype=np.float64)
    for i in range(ny):
        for j in range(nx):
            y0, x0 = i * ty, j * tx
            y1 = h if i == ny - 1 else y0 + ty
            x1 = w if j == nx - 1 else x0 + tx
            tile = channel[y0:y1, x0:x1]
            hist = np.bincount(tile.ravel(), minlength=256).astype(np.float64)

            # Clip tall bins and redistribute the excess uniformly - this is the
            # "contrast limited" part that stops noise being amplified.
            limit = clip * tile.size / 256.0
            excess = np.maximum(hist - limit, 0).sum()
            hist = np.minimum(hist, limit) + excess / 256.0

            cdf = hist.cumsum()
            luts[i, j] = 255.0 * (cdf - cdf[0]) / max(cdf[-1] - cdf[0], 1e-9)

    # Map every pixel through the four nearest tile LUTs and blend.
    yy = np.clip((np.arange(h) - ty / 2) / ty, 0, ny - 1)
    xx = np.clip((np.arange(w) - tx / 2) / tx, 0, nx - 1)
    y0i, x0i = np.floor(yy).astype(int), np.floor(xx).astype(int)
    y1i, x1i = np.minimum(y0i + 1, ny - 1), np.minimum(x0i + 1, nx - 1)
    wy, wx = (yy - y0i)[:, None], (xx - x0i)[None, :]

    idx = channel
    top = luts[y0i[:, None], x0i[None, :], idx] * (1 - wx) + luts[y0i[:, None], x1i[None, :], idx] * wx
    bot = luts[y1i[:, None], x0i[None, :], idx] * (1 - wx) + luts[y1i[:, None], x1i[None, :], idx] * wx
    return np.clip(top * (1 - wy) + bot * wy, 0, 255).astype(np.uint8)


def normalize_illumination(arr: np.ndarray) -> np.ndarray:
    """Flatten the uneven lighting typical of handheld fundus cameras.

    Estimates the illumination field with a heavy box blur and divides it out,
    which removes the bright-centre/dark-edge gradient while preserving lesions,
    since lesions are small relative to the blur kernel.
    """
    gray = arr.mean(axis=2)
    k = max(min(arr.shape[:2]) // 8, 3)

    # Separable box blur via integral image - cheap and adequate for a field estimate.
    pad = np.pad(gray, k, mode="edge")
    cs = pad.cumsum(0).cumsum(1)
    cs = np.pad(cs, ((1, 0), (1, 0)))
    h, w = gray.shape
    ys, xs = np.arange(h) + k, np.arange(w) + k
    y0, y1 = ys - k, ys + k + 1
    x0, x1 = xs - k, xs + k + 1
    field = (
        cs[np.ix_(y1, x1)] - cs[np.ix_(y0, x1)] - cs[np.ix_(y1, x0)] + cs[np.ix_(y0, x0)]
    ) / ((2 * k + 1) ** 2)

    field = np.maximum(field, 1.0)
    corrected = arr / field[:, :, None] * field.mean()
    return np.clip(corrected, 0, 255).astype(np.uint8)


def denoise(arr: np.ndarray) -> np.ndarray:
    """Light 3x3 median filter. Removes sensor speckle without erasing
    microaneurysms, which are only a few pixels across."""
    out = arr.copy()
    stack = np.stack(
        [
            np.roll(np.roll(arr, dy, axis=0), dx, axis=1)
            for dy in (-1, 0, 1)
            for dx in (-1, 0, 1)
        ]
    )
    out[1:-1, 1:-1] = np.median(stack, axis=0)[1:-1, 1:-1]
    return out


def enhance(img: Image.Image) -> Image.Image:
    """CLAHE -> illumination normalization -> denoise, per PS requirement 1."""
    arr = np.asarray(img.convert("RGB"), dtype=np.uint8)
    arr = normalize_illumination(arr)

    # Apply CLAHE to the green channel's contribution only. The green channel
    # carries the most lesion contrast in fundus photography; equalizing all
    # three independently would shift the colour balance.
    ycc = np.asarray(Image.fromarray(arr).convert("YCbCr"), dtype=np.uint8)
    ycc[:, :, 0] = _clahe(ycc[:, :, 0])
    arr = np.asarray(Image.fromarray(ycc, mode="YCbCr").convert("RGB"), dtype=np.uint8)

    return Image.fromarray(denoise(arr))


def enhance_if_borderline(
    img: Image.Image, report: QualityReport, reassess
) -> tuple[Image.Image, QualityReport]:
    """Enhance a borderline image and re-assess it.

    Returns the better of the two outcomes. If enhancement fails to lift the
    image out of borderline we keep the enhanced version anyway when it scored
    higher, but we never promote an ungradable image by enhancing it.
    """
    if report.status is not QualityStatus.BORDERLINE:
        return img, report

    enhanced = enhance(img)
    new_report = reassess(enhanced)
    new_report.enhancement_applied = True

    if new_report.score > report.score:
        return enhanced, new_report
    return img, report
