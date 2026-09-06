import { useEffect, useState } from "react";
import { getScreening, submitReview, type PatientResult, type ReviewSubmission } from "../api";
import { GRADES } from "../lib/grade";
import { PatientResultView } from "./PatientResultView";
import { ProvenanceBar } from "./ProvenanceBar";
import "./CaseView.css";

const ACTIONS: { action: ReviewSubmission["action"]; label: string }[] = [
  { action: "confirm", label: "Confirm AI result" },
  { action: "override", label: "Override grade" },
  { action: "request_recapture", label: "Request new image" },
  { action: "refer", label: "Refer patient" },
];

/** One case, with the specialist's decision recorded against it. This is the
 *  human-in-the-loop step: the AI result is a recommendation and the
 *  ophthalmologist's verdict is the clinical one. */
export function CaseView({
  screeningId,
  onBack,
}: {
  screeningId: string;
  onBack: () => void;
}) {
  const [result, setResult] = useState<PatientResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [overrideGrade, setOverrideGrade] = useState("2");
  const [notes, setNotes] = useState("");
  const [done, setDone] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    getScreening(screeningId)
      .then(setResult)
      .catch(() => setError("This case could not be loaded."));
  }, [screeningId]);

  async function act(action: ReviewSubmission["action"]) {
    setBusy(true);
    setError(null);
    try {
      await submitReview(screeningId, {
        action,
        reviewer_ref: "DR-DEMO-01",
        override_grade: action === "override" ? Number(overrideGrade) : null,
        notes: notes || null,
      });
      setDone(action);
    } catch {
      setError("The review could not be recorded.");
    } finally {
      setBusy(false);
    }
  }

  if (error && !result) return <p className="case-error">{error}</p>;
  if (!result) return <p className="case-loading">Loading the case…</p>;

  const eye = result.left ?? result.right;

  return (
    <div className="case">
      <div className="case-bar">
        <button className="link" onClick={onBack}>Back to queue</button>
        <ProvenanceBar
          modelVersion={eye?.model_version}
          thresholdVersion={eye?.threshold_version}
        />
      </div>

      <PatientResultView result={result} />

      <section className="review">
        <h2 className="review-title">Specialist review</h2>
        <p className="review-note">
          The AI result is a screening recommendation. Your decision is the clinical
          one and is recorded against this screening.
        </p>

        <div className="review-fields">
          <label className="field">
            <span className="field-name">Override to grade</span>
            <select value={overrideGrade} onChange={(e) => setOverrideGrade(e.target.value)}>
              {GRADES.map((g) => (
                <option key={g.n} value={g.n}>
                  {g.n} — {g.label}
                </option>
              ))}
            </select>
          </label>
          <label className="field">
            <span className="field-name">Notes</span>
            <input type="text" value={notes} onChange={(e) => setNotes(e.target.value)} />
          </label>
        </div>

        <div className="review-actions">
          {ACTIONS.map((a) => (
            <button key={a.action} className="secondary" disabled={busy} onClick={() => act(a.action)}>
              {a.label}
            </button>
          ))}
        </div>

        {done && (
          <p className="review-recorded readout">
            Recorded: {ACTIONS.find((a) => a.action === done)?.label.toLowerCase()}
            {done === "override" && ` to grade ${overrideGrade}`}.
          </p>
        )}
        {error && <p className="case-error">{error}</p>}
      </section>
    </div>
  );
}
