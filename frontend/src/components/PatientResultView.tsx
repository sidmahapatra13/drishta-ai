import type { PatientResult } from "../api";
import { EyePanel } from "./EyePanel";
import { VerdictPanel } from "./VerdictPanel";
import "./PatientResultView.css";

/**
 * One screening, laid out as an instrument: evidence on the left, decision on
 * the right.
 *
 * The decision column is sticky, so scrolling through both eyes never takes
 * the referral verdict off screen. That is the reading order a clinician
 * actually uses - check the finding, look back at the call, check the next
 * finding - and it is the reason the verdict is not simply stacked on top.
 */
export function PatientResultView({ result }: { result: PatientResult }) {
  const eyes = [result.left, result.right].filter((e) => e !== null);

  return (
    <article className="result">
      <div className="result-evidence">
        {eyes.map((eye) => (
          <EyePanel key={eye.eye} eye={eye} />
        ))}
      </div>

      <div className="result-decision">
        <VerdictPanel result={result} />
        <p className="result-disclaimer">{result.disclaimer}</p>
      </div>
    </article>
  );
}
