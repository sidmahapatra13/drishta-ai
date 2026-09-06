import type { PatientResult } from "../api";
import { colourFor, pct } from "../lib/grade";
import { RecaptureNotice } from "./RecaptureNotice";
import { SeverityLadder } from "./SeverityLadder";
import "./VerdictPanel.css";

/**
 * The referral decision, and the page's centre of gravity.
 *
 * Two outcomes occupy the same position at the same size: a grade, or a
 * refusal to grade. Rendering the refusal smaller, or lower, or as an error
 * would misrepresent what the system did - it did not fail, it declined, and
 * declining is a result.
 */
export function VerdictPanel({ result }: { result: PatientResult }) {
  const ungradable = [result.left, result.right].filter(
    (e) => e && e.recapture_required,
  );

  if (result.grade === null) {
    // No eye survived the gate, so there is no grade to show. The refusal is
    // the verdict.
    const eye = ungradable[0];
    return (
      <section className="verdict">
        {eye ? (
          // RecaptureNotice already states that no grade is reported, so the
          // backend summary would say the same thing twice.
          <RecaptureNotice eye={eye} prominent />
        ) : (
          <>
            <h2 className="verdict-grade">No gradable image</h2>
            <p className="verdict-summary">{result.summary}</p>
          </>
        )}
      </section>
    );
  }

  const priority =
    result.priority === "urgent"
      ? { cls: "urgent", text: "urgent referral" }
      : result.referable
        ? { cls: "referable", text: "referable" }
        : { cls: "clear", text: "not referable" };

  return (
    <section className="verdict">
      <div className="verdict-head">
        <h2 className="verdict-grade" style={{ color: colourFor(result.grade) }}>
          {result.grade_label}
        </h2>
        <p className="verdict-meta readout">
          grade {result.grade}
          <span className={`tag ${priority.cls}`}>{priority.text}</span>
        </p>
      </div>

      <SeverityLadder grade={result.grade} />

      <div className="verdict-next">
        <p>
          {result.referable
            ? "Queued for ophthalmologist review. The specialist confirms or overrides this result."
            : "No referral indicated at this screening. Repeat at the next routine interval."}
        </p>
        {result.referral_probability !== null && (
          <p className="verdict-referral readout">
            {pct(result.referral_probability)} likely referable
            <span className="verdict-provenance">
              {" "}
              calibrated on held-out predictions
            </span>
          </p>
        )}
        {result.confidence !== null && (
          <p className="verdict-likelihood readout">
            grade likelihood (derived) {pct(result.confidence)}
          </p>
        )}
      </div>

      {ungradable.map((eye) => eye && <RecaptureNotice key={eye.eye} eye={eye} />)}
    </section>
  );
}
