import type { QualityReport } from "../api";
import "./QualityGauges.css";

/**
 * The three quality signals, each drawn against the gate that judges it.
 *
 * A bare percentage tells a health worker nothing: 40% focus is fine and 40%
 * illumination is a rejection. Marking each metric's own floor on its own
 * track turns three numbers into three verdicts, and makes the one that failed
 * findable at a glance rather than by comparing against remembered thresholds.
 *
 * Floors mirror backend/app/services/quality.py. They are gate constants, not
 * presentation choices.
 */
const GATE = {
  focus: 0.35,
  illumination: 0.35,
  field_of_view: 0.45,
} as const;

const NAMES = {
  focus: "Focus",
  illumination: "Illumination",
  field_of_view: "Field of view",
} as const;

export function QualityGauges({ quality }: { quality: QualityReport }) {
  return (
    <div className="gauges">
      {(Object.keys(GATE) as (keyof typeof GATE)[]).map((key) => {
        const value = quality[key];
        const floor = GATE[key];
        const failed = value < floor;
        return (
          <div className="gauge" key={key} data-failed={failed}>
            <span className="gauge-name">{NAMES[key]}</span>
            <span className="gauge-track">
              <span className="gauge-reject" style={{ width: `${floor * 100}%` }} />
              <span className="gauge-gate" style={{ left: `${floor * 100}%` }} />
              <span className="gauge-fill" style={{ width: `${value * 100}%` }} />
            </span>
            <span className="gauge-value readout">{value.toFixed(2)}</span>
          </div>
        );
      })}
      <p className="gauges-key">
        The mark on each track is the gate. Hatching shows how far a reading
        falls short of it.
      </p>
    </div>
  );
}
