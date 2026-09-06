import { useState } from "react";
import { analyze } from "../api";
import type { ScreeningFormState } from "../lib/screeningForm";
import "./ScreeningForm.css";

/** A file input cannot have its value restored programmatically, so after the
 *  tab remounts the native control reads "No file chosen" even though the File
 *  is still held in state. Name the retained file ourselves rather than let the
 *  control contradict what is actually loaded. */
function EyeInput({
  label,
  file,
  onPick,
}: {
  label: string;
  file: File | null;
  onPick: (f: File | null) => void;
}) {
  return (
    <label className="field">
      <span className="field-name">{label}</span>
      <input
        type="file"
        accept="image/*"
        onChange={(e) => onPick(e.target.files?.[0] ?? null)}
      />
      {file && <span className="field-loaded readout">loaded: {file.name}</span>}
    </label>
  );
}

export function ScreeningForm({
  form,
  setForm,
  onDone,
}: {
  form: ScreeningFormState;
  setForm: (update: Partial<ScreeningFormState>) => void;
  onDone: () => void;
}) {
  const { patientRef, age, years, left, right } = form;
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function submit() {
    setBusy(true);
    setError(null);
    try {
      setForm({
        result: await analyze({ patientRef, age, diabetesYears: years, left, right }),
      });
      onDone();
    } catch (e) {
      setError(e instanceof Error ? e.message : "The screening could not be completed.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="screening-form">
      <div className="form-grid">
        <label className="field">
          <span className="field-name">Patient ID</span>
          <input
            type="text"
            value={patientRef}
            onChange={(e) => setForm({ patientRef: e.target.value })}
          />
        </label>
        <label className="field">
          <span className="field-name">Age</span>
          <input type="number" value={age} onChange={(e) => setForm({ age: e.target.value })} />
        </label>
        <label className="field">
          <span className="field-name">Years since diagnosis</span>
          <input type="number" value={years} onChange={(e) => setForm({ years: e.target.value })} />
        </label>
      </div>

      <div className="form-grid form-eyes">
        <EyeInput label="Left eye" file={left} onPick={(f) => setForm({ left: f })} />
        <EyeInput label="Right eye" file={right} onPick={(f) => setForm({ right: f })} />
      </div>

      <div className="form-submit">
        <button className="primary" onClick={submit} disabled={busy || (!left && !right)}>
          {busy ? "Screening…" : "Start screening"}
        </button>
        {!left && !right && (
          <p className="form-hint">Add a photograph of at least one eye to begin.</p>
        )}
        {error && <p className="form-error">{error}</p>}
      </div>
    </section>
  );
}
