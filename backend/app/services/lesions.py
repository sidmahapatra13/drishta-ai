"""Lesion evidence - PS requirement 2.

STUB, and deliberately silent. Replaced by classical morphology (bright-lesion
detection for exudates, dark-lesion for hemorrhages, top-hat plus candidate
filtering for microaneurysms), tuned against the IDRiD segmentation masks
rather than by eye.

Until then this returns nothing. The earlier stub invented per-class counts,
which surfaced in two places a judge reads as clinical findings: the lesion
evidence panel in the UI, and the "Detected evidence: ..." clause that
`fusion._summarize` appends to the patient summary. Both callers already guard
on an empty list, so returning one leaves the claim visibly absent instead of
plausibly wrong - which is the rule this project runs on.

Neovascularization is deliberately absent: no public pixel-level NV masks
exist, so it is scoped out explicitly rather than faked. Say so on the slide.
"""

from PIL import Image

from ..contract import Lesion, LesionType

#: Lesion classes we can actually support with available annotated data (IDRiD).
SUPPORTED = [LesionType.MICROANEURYSM, LesionType.HEMORRHAGE, LesionType.EXUDATE]


def segment_lesions(img: Image.Image, grade: int | None) -> list[Lesion]:
    """Return per-class lesion evidence for one eye.

    Counts are detector output, not clinically authoritative lesion counts.
    The UI must label them as such.
    """
    return []
