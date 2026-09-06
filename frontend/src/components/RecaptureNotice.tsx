import type { EyeResult } from "../api";
import { recaptureAdvice } from "../lib/grade";
import "./RecaptureNotice.css";

/**
 * The system declining to answer.
 *
 * This is the most defensible thing the product does, so it is rendered at the
 * weight of a verdict rather than as a warning attached to one. A confident
 * "No DR" on a photograph too dark to read is the single most dangerous output
 * this system could produce; refusing is the correct behaviour, and the
 * interface should not apologise for it.
 *
 * `heading` is the gate's dominant failure translated into something a health
 * worker can act on. The full issue list stays visible underneath, because the
 * ophthalmologist reviewing the case later needs the measurement, not the
 * instruction.
 */
export function RecaptureNotice({ eye, prominent }: { eye: EyeResult; prominent?: boolean }) {
  const advice = recaptureAdvice(eye.quality.issues);

  return (
    <div className="recapture" data-prominent={prominent}>
      <p className="recapture-eye readout">{eye.eye} eye</p>
      <h2 className="recapture-title">{advice.title}</h2>
      <p className="recapture-action">{advice.action}</p>

      <p className="recapture-rule">
        No grade is reported for an image the system cannot read.
      </p>

      {eye.quality.issues.length > 0 && (
        <p className="recapture-detail readout">
          measured: {eye.quality.issues.join(" · ")}
        </p>
      )}
    </div>
  );
}
