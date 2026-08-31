"""Lesion evidence - PS requirement 2.

STUB. Replaced on day 3 by classical morphology (bright-lesion detection for
exudates, dark-lesion for hemorrhages, top-hat plus candidate filtering for
microaneurysms), with a multi-class U-Net on the IDRiD segmentation subset as a
stretch goal if day 2 finishes early.

Neovascularization is deliberately absent: no public pixel-level NV masks
exist, so it is scoped out explicitly rather than faked. Say so on the slide.
"""

import hashlib

import numpy as np
from PIL import Image

from ..contract import Lesion, LesionType

#: Lesion classes we can actually support with available annotated data (IDRiD).
SUPPORTED = [LesionType.MICROANEURYSM, LesionType.HEMORRHAGE, LesionType.EXUDATE]


def segment_lesions(img: Image.Image, grade: int | None) -> list[Lesion]:
    """Return per-class lesion evidence for one eye.

    Counts are detector output, not clinically authoritative lesion counts.
    The UI must label them as such.
    """
    if grade is None:
        return []

    rng = np.random.default_rng(
        int(hashlib.sha256(img.tobytes()).hexdigest()[8:16], 16)
    )

    lesions: list[Lesion] = []
    for i, lesion_type in enumerate(SUPPORTED):
        # More severe grades carry more lesions - keeps the stubbed evidence
        # consistent with the stubbed grade so the report never contradicts itself.
        expected = max(grade * 3 - i * 2, 0)
        count = int(rng.poisson(expected)) if expected else 0
        lesions.append(
            Lesion(
                type=lesion_type,
                count=count,
                area_px=int(count * rng.integers(15, 60)) if count else 0,
                mask_url=None,
                confidence=round(float(rng.uniform(0.55, 0.85)), 3) if count else 0.0,
            )
        )
    return lesions
