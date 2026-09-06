"""Draw APTOS against IDRiD as one figure.

Separate from `validate_idrid.py` so the figure can be redrawn without repeating
a ten-minute inference run. It reads only the two committed records, so it
cannot show a number that no run produced.

The form is a dumbbell: one row per metric, the two datasets as two dots joined
by the gap between them. A grouped bar chart would put the emphasis on the
heights; the story here is the *distance* - how much of the APTOS result
survives a different camera, a different clinic and a different country's
screening population.

    python scripts/validate_idrid.py
    python scripts/plot_validation.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CARD = ROOT / "matlab" / "models" / "model_card.json"
RECORD = ROOT / "experiments" / "validation" / "idrid.json"
OUT = ROOT / "experiments" / "validation" / "aptos-vs-idrid.svg"

INK = "#f2efe9"
MUTED = "#8b97a8"
FAINT = "#5b6675"
GROUND = "#0b0e14"
LINE = "#1f2937"

#: Validated as an adjacent pair against the dark surface: CVD delta-E 14.4
#: (protan), 21.7 with normal vision. Neither colour is the palette's alarm
#: red - the external number is new information, not a failure, and colouring it
#: as one would editorialise a result the reader should weigh themselves.
APTOS = "#e8c77a"
IDRID = "#4fa88b"

W, H = 780, 430
LEFT, RIGHT = 250, 690
TOP, ROW = 118, 66

#: Every metric here lives in the upper half, so the track starts at 0.5 rather
#: than compressing all four rows into the right-hand quarter. Legitimate for a
#: dot plot, where position is read against the ticks rather than as a length
#: from zero, and the ticks are labelled so nothing is concealed.
FLOOR = 0.5


def main() -> None:
    for path in (CARD, RECORD):
        if not path.exists():
            sys.exit(f"missing {path} - run scripts/validate_idrid.py first")

    card = json.loads(CARD.read_text())
    record = json.loads(RECORD.read_text())
    external = record["splits"]["all"]

    rows = [
        ("Quadratic weighted kappa", card["qwk"], external["qwk"]),
        (
            "Referable sensitivity",
            card["referable"]["sensitivity"],
            external["referable"]["sensitivity"],
        ),
        (
            "Referable specificity",
            card["referable"]["specificity"],
            external["referable"]["specificity"],
        ),
        (
            "Referable ROC-AUC",
            card["referable"]["roc_auc"],
            external["referable"]["roc_auc"],
        ),
    ]

    def px(v: float) -> float:
        return LEFT + (max(v, FLOOR) - FLOOR) / (1.0 - FLOOR) * (RIGHT - LEFT)

    p = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}" font-family="IBM Plex Sans, system-ui, sans-serif">',
        f'<rect width="{W}" height="{H}" fill="{GROUND}"/>',
        f'<text x="32" y="40" fill="{INK}" font-size="21" font-weight="600">'
        f"What survives a change of dataset</text>",
        f'<text x="32" y="64" fill="{MUTED}" font-size="13">'
        f'Trained on APTOS, tested on {external["n"]} IDRiD images it has never seen '
        f"&#183; the two are never merged</text>",
    ]

    # Legend. Two series, so it is always present, and both ends are also
    # labelled directly with their value.
    for i, (name, colour) in enumerate(
        [("APTOS, 5-fold cross-validated", APTOS), ("IDRiD, external", IDRID)]
    ):
        x = 32 + i * 250
        p.append(f'<circle cx="{x + 6}" cy="88" r="6" fill="{colour}"/>')
        p.append(
            f'<text x="{x + 20}" y="92" fill="{MUTED}" font-size="12">{name}</text>'
        )

    for t in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        x = px(t)
        p.append(
            f'<line x1="{x:.1f}" y1="{TOP - 14}" x2="{x:.1f}" '
            f'y2="{TOP + ROW * len(rows) - 26}" stroke="{LINE}" stroke-width="1"/>'
        )
        p.append(
            f'<text x="{x:.1f}" y="{TOP + ROW * len(rows) - 8}" fill="{FAINT}" '
            f'font-size="11" text-anchor="middle" '
            f'font-family="IBM Plex Mono, ui-monospace, monospace">{t:.1f}</text>'
        )

    for i, (label, a, b) in enumerate(rows):
        y = TOP + i * ROW
        p.append(
            f'<text x="32" y="{y + 5}" fill="{INK}" font-size="13">{label}</text>'
        )
        p.append(
            f'<line x1="{px(a):.1f}" y1="{y}" x2="{px(b):.1f}" y2="{y}" '
            f'stroke="{FAINT}" stroke-width="2"/>'
        )

        # A 2px ground-coloured ring keeps the two dots separable where the
        # metric barely moved and they overlap.
        for value, colour in ((a, APTOS), (b, IDRID)):
            p.append(
                f'<circle cx="{px(value):.1f}" cy="{y}" r="7" fill="{colour}" '
                f'stroke="{GROUND}" stroke-width="2"/>'
            )

        # Direct labels, placed on the outside of each dot so they never collide
        # with the connector between them.
        lo, hi = (a, APTOS, -1), (b, IDRID, 1)
        if a > b:
            lo, hi = (b, IDRID, -1), (a, APTOS, 1)
        for value, colour, side in (lo, hi):
            anchor = "end" if side < 0 else "start"
            p.append(
                f'<text x="{px(value) + side * 14:.1f}" y="{y + 4}" fill="{colour}" '
                f'font-size="12" text-anchor="{anchor}" '
                f'font-family="IBM Plex Mono, ui-monospace, monospace">'
                f"{value:.3f}</text>"
            )

    # Kept to one line that fits the width - the dataset's full name is already
    # in the subtitle, and a caption that runs off the edge is worse than a
    # short one.
    p.append(
        f'<text x="32" y="{H - 20}" fill="{FAINT}" font-size="11">'
        f"IDRiD Disease Grading &#183; CC BY 4.0 &#183; graded through the shipped "
        f"pipeline &#183; external validation, not clinical validation</text>"
    )
    p.append("</svg>")

    OUT.write_text("\n".join(p) + "\n")
    print(f"wrote {OUT}")
    for label, a, b in rows:
        print(f"{label:<28} APTOS {a:.4f}   IDRiD {b:.4f}   {b - a:+.4f}")


if __name__ == "__main__":
    main()
