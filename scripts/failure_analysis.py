"""Where the model fails, and how badly.

"95.5% sensitivity" hides the question a clinician actually asks: *what* is it
missing. Ranked by true severity the answer is not uniform, and the shape of it
matters more than the headline - a missed grade 2 sitting on the referral
boundary and a missed proliferative eye are not the same event.

Two outputs, on two footings for the same licence reason as everywhere else:

  * `experiments/failures/failures.json` and the figure beside it are numbers
    and identifiers. Committed.
  * The contact sheet is APTOS photographs. Written to gitignored
    `local-data/figures/` and never committed - the same rule that keeps the
    dataset out of git. Put it in the deck from there.

The eight captures on the sheet are chosen by a stated rule, not by eye: every
sight-threatening miss (there are few enough to show all of them), then the
remaining misses the model was most confident about. That is the least
flattering selection available, which is the point of showing failures at all.

    python scripts/fetch_aptos_images.py
    python scripts/failure_analysis.py
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.contract import GRADE_LABELS, REFERABLE_FROM_GRADE  # noqa: E402
from app.services import classifier  # noqa: E402

OOF = ROOT / "experiments" / "classification" / "oof_predictions.npy"
LABELS = ROOT / "local-data" / "aptos" / "train.csv"
IMAGES = ROOT / "local-data" / "aptos-full" / "train_images"
OUT_DIR = ROOT / "experiments" / "failures"
SHEET_DIR = ROOT / "local-data" / "figures"

#: Grade 3 and above. A missed eye here is the failure that costs sight.
SIGHT_THREATENING = 3

SHOW = 8

INK = "#f2efe9"
MUTED = "#8b97a8"
FAINT = "#5b6675"
GROUND = "#0b0e14"
TRACK = "#0e131b"
MARK = "#4fa88b"
MISS = "#c9552e"


def load() -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    raw = np.load(OOF).astype(np.float64)
    rows = list(csv.DictReader(LABELS.open()))
    truth = np.array([int(r["diagnosis"]) for r in rows])
    cuts = np.array(json.loads(classifier.CARD_PATH.read_text())["grade_thresholds"])
    return raw, truth, [r["id_code"] for r in rows], cuts


def by_severity(truth: np.ndarray, caught: np.ndarray) -> list[dict]:
    out = []
    for grade in range(REFERABLE_FROM_GRADE, 5):
        sel = truth == grade
        n, hit = int(sel.sum()), int(caught[sel].sum())
        out.append(
            {
                "grade": grade,
                "label": GRADE_LABELS[grade],
                "n": n,
                "caught": hit,
                "missed": n - hit,
                "sensitivity": hit / n,
            }
        )
    return out


def svg(record: dict) -> str:
    """Sensitivity by true severity, drawn as the interface draws its gauges.

    Each bar runs against a full-width track to 100%, so the shortfall is a
    visible gap rather than something the reader has to infer from a number.
    The gap is where the missed patients are, and it is labelled with how many.
    """
    rows = record["by_severity"]
    W, H = 760, 382
    LEFT, RIGHT = 250, 660
    TOP, ROW = 150, 66

    p = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}" font-family="IBM Plex Sans, system-ui, sans-serif">',
        f'<rect width="{W}" height="{H}" fill="{GROUND}"/>',
        f'<text x="32" y="40" fill="{INK}" font-size="21" font-weight="600">'
        f"What the model misses</text>",
        f'<text x="32" y="64" fill="{MUTED}" font-size="13">'
        f"Referral sensitivity by true severity &#183; APTOS out-of-fold, "
        f'{record["referable"]["n"]} referable eyes</text>',
    ]

    hero = record["sight_threatening"]
    p += [
        f'<text x="32" y="104" fill="{MARK}" font-size="27" font-weight="600" '
        f'font-family="IBM Plex Mono, ui-monospace, monospace">'
        f'{hero["sensitivity"] * 100:.1f}%</text>',
        f'<text x="140" y="98" fill="{INK}" font-size="13">'
        f"of sight-threatening disease is caught</text>",
        f'<text x="140" y="116" fill="{FAINT}" font-size="11">'
        f'grade 3 and above &#183; {hero["missed"]} missed of {hero["n"]}</text>',
    ]

    for i, row in enumerate(rows):
        y = TOP + i * ROW
        full = RIGHT - LEFT
        filled = full * row["sensitivity"]

        p.append(
            f'<text x="32" y="{y + 5}" fill="{INK}" font-size="13">'
            f'{row["grade"]} &#183; {row["label"]}</text>'
        )
        p.append(
            f'<text x="32" y="{y + 22}" fill="{FAINT}" font-size="11" '
            f'font-family="IBM Plex Mono, ui-monospace, monospace">'
            f'n = {row["n"]}</text>'
        )
        p.append(
            f'<rect x="{LEFT}" y="{y - 9}" width="{full}" height="18" rx="4" '
            f'fill="{TRACK}"/>'
        )
        p.append(
            f'<rect x="{LEFT}" y="{y - 9}" width="{filled:.1f}" height="18" rx="4" '
            f'fill="{MARK}"/>'
        )
        p.append(
            f'<text x="{RIGHT + 12}" y="{y + 5}" fill="{INK}" font-size="13" '
            f'font-family="IBM Plex Mono, ui-monospace, monospace">'
            f'{row["sensitivity"] * 100:.1f}%</text>'
        )
        # Below the bar, not at its end. At 93-100% sensitivity the shortfall is
        # a few pixels wide, so a label placed in the gap lands on top of the
        # percentage beside it.
        if row["missed"]:
            p.append(
                f'<text x="{LEFT}" y="{y + 28}" fill="{MISS}" font-size="11">'
                f'{row["missed"]} referable {"eye" if row["missed"] == 1 else "eyes"} '
                f"called not referable</text>"
            )

    p.append(
        f'<text x="32" y="{H - 22}" fill="{FAINT}" font-size="11">'
        f"Referable is grade 2 and above &#183; out-of-fold predictions, no image "
        f"scored by a model that trained on it</text>"
    )
    p.append("</svg>")
    return "\n".join(p)


def contact_sheet(chosen: list[dict]) -> Path | None:
    """A grid of the actual failures, for the slide."""
    if not IMAGES.is_dir():
        print(f"skipping contact sheet - no images at {IMAGES.relative_to(ROOT)}")
        return None

    cell, pad, caption = 300, 16, 58
    cols, rowsn = 4, 2
    W = cols * cell + pad * (cols + 1)
    H = rowsn * (cell + caption) + pad * (rowsn + 1) + 70
    sheet = Image.new("RGB", (W, H), "#0b0e14")
    draw = ImageDraw.Draw(sheet)

    def font(path: str, size: int):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            return ImageFont.load_default()

    title = font("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 22)
    body = font("/System/Library/Fonts/Supplemental/Arial.ttf", 14)
    mono = font("/System/Library/Fonts/SFNSMono.ttf", 13)

    draw.text((pad + 4, 20), "Every one of these was called not referable",
              font=title, fill=INK)
    draw.text((pad + 4, 48),
              "APTOS out-of-fold false negatives - all sight-threatening misses, "
              "then the most confident remaining ones",
              font=body, fill=MUTED)

    for i, item in enumerate(chosen):
        cx = pad + (i % cols) * (cell + pad)
        cy = 70 + pad + (i // cols) * (cell + caption + pad)

        img = Image.open(IMAGES / f"{item['id_code']}.png").convert("RGB")
        side = min(img.size)
        img = img.crop(
            ((img.width - side) // 2, (img.height - side) // 2,
             (img.width + side) // 2, (img.height + side) // 2)
        ).resize((cell, cell), Image.LANCZOS)
        sheet.paste(img, (cx, cy))

        colour = MISS if item["true_grade"] >= SIGHT_THREATENING else INK
        draw.text((cx, cy + cell + 8),
                  f"true {item['true_grade']} - {GRADE_LABELS[item['true_grade']]}",
                  font=body, fill=colour)
        draw.text((cx, cy + cell + 28),
                  f"called {item['predicted_grade']}   score {item['raw']:+.2f}   "
                  f"P(ref) {item['referral_probability']:.2f}",
                  font=mono, fill=MUTED)

    SHEET_DIR.mkdir(parents=True, exist_ok=True)
    out = SHEET_DIR / "failure-gallery.png"
    sheet.save(out)
    return out


def main() -> None:
    raw, truth, ids, cuts = load()
    pred = np.digitize(raw, cuts)
    caught = pred >= REFERABLE_FROM_GRADE

    referable = truth >= REFERABLE_FROM_GRADE
    severe = truth >= SIGHT_THREATENING
    misses = np.flatnonzero(referable & ~caught)

    # The probability is derived from the *recorded* score, not the full-precision
    # one, so anyone can reproduce this row from the row itself.
    failures = []
    for i in misses:
        score = round(float(raw[i]), 4)
        failures.append(
            {
                "id_code": ids[i],
                "true_grade": int(truth[i]),
                "predicted_grade": int(pred[i]),
                "raw": score,
                "below_referral_cut_by": round(float(cuts[1]) - score, 4),
                "referral_probability": round(
                    classifier.referral_probability(score), 4
                ),
                "sight_threatening": bool(truth[i] >= SIGHT_THREATENING),
            }
        )

    # Stated rule, applied in order: every sight-threatening miss first, then
    # the lowest-scoring of the rest. Nothing is chosen by how it looks.
    ordered = sorted(
        failures, key=lambda f: (not f["sight_threatening"], f["raw"])
    )

    record = {
        "dataset": "APTOS 2019, 5-fold out-of-fold predictions",
        "referable": {
            "n": int(referable.sum()),
            "missed": int(len(misses)),
            "sensitivity": float(caught[referable].mean()),
        },
        "sight_threatening": {
            "definition": "grade 3 and above",
            "n": int(severe.sum()),
            "missed": int((severe & ~caught).sum()),
            "sensitivity": float(caught[severe].mean()),
        },
        "by_severity": by_severity(truth, caught),
        "selection_rule": (
            "Every sight-threatening miss, then the remaining misses with the "
            "lowest severity score - the ones the model was most confident "
            "about. Not chosen by appearance."
        ),
        "shown": [f["id_code"] for f in ordered[:SHOW]],
        "failures": ordered,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "failures.json").write_text(json.dumps(record, indent=2) + "\n")
    (OUT_DIR / "sensitivity-by-severity.svg").write_text(svg(record) + "\n")

    print(f"{'true severity':<28} {'n':>5} {'missed':>7} {'sensitivity':>12}")
    for row in record["by_severity"]:
        print(f'{row["grade"]} {row["label"]:<26} {row["n"]:>5} {row["missed"]:>7} '
              f'{row["sensitivity"]:>11.1%}')
    s = record["sight_threatening"]
    print(f'\nsight-threatening (3+)  {s["n"]} cases, {s["missed"]} missed '
          f'-> {s["sensitivity"]:.1%}')
    print(f'all referable           {record["referable"]["n"]} cases, '
          f'{record["referable"]["missed"]} missed '
          f'-> {record["referable"]["sensitivity"]:.1%}')

    print(f"\nwrote {OUT_DIR / 'failures.json'}")
    print(f"wrote {OUT_DIR / 'sensitivity-by-severity.svg'}")
    sheet = contact_sheet(ordered[:SHOW])
    if sheet:
        print(f"wrote {sheet}  (gitignored - APTOS images stay out of git)")


if __name__ == "__main__":
    main()
