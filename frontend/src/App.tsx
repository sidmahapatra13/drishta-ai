import { useEffect, useState } from "react";
import {
  analyze,
  assetUrl,
  getScreening,
  reviewQueue,
  scenarios,
  submitReview,
  type EyeResult,
  type PatientResult,
  type QueueRow,
  type ReviewSubmission,
  type ScenarioResult,
} from "./api";

const GRADES = [
  { n: 0, label: "No apparent DR" },
  { n: 1, label: "Mild NPDR" },
  { n: 2, label: "Moderate NPDR" },
  { n: 3, label: "Severe NPDR" },
  { n: 4, label: "Proliferative DR" },
];

const colourFor = (grade: number | null) =>
  grade === null ? "var(--muted)" : grade >= 4 ? "var(--urgent)" : grade >= 2 ? "var(--retina)" : "var(--clear)";

const pct = (v: number) => `${Math.round(v * 100)}%`;

/** The signature element: the severity scale drawn as the ordinal instrument it
 *  is, with the referral threshold as a real line between grades 1 and 2. */
function SeverityLadder({ grade }: { grade: number | null }) {
  return (
    <div>
      <ol className="ladder">
        {GRADES.map((g) => (
          <li
            key={g.n}
            data-active={g.n === grade}
            data-threshold={g.n === 2}
            style={{ ["--reading" as string]: colourFor(g.n) }}
          >
            <span className="num readout">{g.n}</span>
            <span className="label">{g.label}</span>
            {g.n === grade && <span className="eyebrow">detected</span>}
          </li>
        ))}
      </ol>
      <p className="threshold-note">Referral threshold — grade 2 and above</p>
    </div>
  );
}

function QualityPanel({ eye }: { eye: EyeResult }) {
  const metrics = [
    ["Focus", eye.quality.focus],
    ["Illumination", eye.quality.illumination],
    ["Field of view", eye.quality.field_of_view],
  ] as const;

  return (
    <div className="panel stack">
      <div>
        <p className="eyebrow">{eye.eye} eye — image quality</p>
        {metrics.map(([name, value]) => (
          <div className="metric" key={name}>
            <span>{name}</span>
            <span className="bar">
              <span
                style={{
                  width: pct(value),
                  ["--reading" as string]: value < 0.5 ? "var(--urgent)" : value < 0.7 ? "var(--disc)" : "var(--clear)",
                }}
              />
            </span>
            <span className="val">{pct(value)}</span>
          </div>
        ))}
      </div>

      {eye.quality.enhancement_applied && (
        <p className="eyebrow">Enhanced — CLAHE, illumination normalization, denoise</p>
      )}

      {eye.recapture_required ? (
        <div className="recapture">
          <h3>Image ungradable</h3>
          <p>{eye.recapture_reason}</p>
          <p style={{ marginTop: 8, color: "var(--muted)", fontSize: 13 }}>
            Reposition the camera and capture this eye again. No grade is reported for an
            ungradable image.
          </p>
        </div>
      ) : (
        <span className="tag clear" style={{ alignSelf: "start" }}>
          gradable
        </span>
      )}

      {eye.image_url && (
        <div className="grid-2">
          <figure>
            <img src={assetUrl(eye.image_url)!} alt={`${eye.eye} fundus`} />
            <figcaption>Fundus image</figcaption>
          </figure>
          {eye.gradcam_url && (
            <figure>
              <img src={assetUrl(eye.gradcam_url)!} alt={`${eye.eye} Grad-CAM`} />
              <figcaption>
                Grad-CAM — regions that influenced the prediction. Not proof of a lesion.
              </figcaption>
            </figure>
          )}
        </div>
      )}

      {eye.lesions.length > 0 && (
        <div>
          <p className="eyebrow">Lesion evidence (detector output)</p>
          <table>
            <tbody>
              {eye.lesions.map((l) => (
                <tr key={l.type}>
                  <td style={{ textTransform: "capitalize" }}>{l.type}</td>
                  <td className="num" style={{ textAlign: "right" }}>
                    {l.count}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

function Result({ result }: { result: PatientResult }) {
  const tag = result.recapture_required && result.grade === null
    ? { cls: "pending", text: "recapture required" }
    : result.priority === "urgent"
      ? { cls: "urgent", text: "urgent referral" }
      : result.referable
        ? { cls: "referable", text: "referable" }
        : { cls: "clear", text: "non-referable" };

  return (
    <div className="stack">
      <div className="panel stack">
        <p className="eyebrow">
          Screening {result.screening_id} — patient {result.patient_ref}
        </p>
        <div className="verdict">
          <span className="grade-label" style={{ color: colourFor(result.grade) }}>
            {result.grade_label ?? "No gradable image"}
          </span>
          <span className={`tag ${tag.cls}`}>{tag.text}</span>
          {result.confidence !== null && (
            <span className="readout" style={{ color: "var(--muted)", fontSize: 14 }}>
              calibrated confidence {pct(result.confidence)}
            </span>
          )}
        </div>
        <p style={{ margin: 0, color: "var(--muted)" }}>{result.summary}</p>
      </div>

      <div className="grid-2">
        <div className="panel">
          <p className="eyebrow" style={{ marginBottom: 12 }}>
            International Clinical DR Severity Scale
          </p>
          <SeverityLadder grade={result.grade} />
        </div>
        <div className="panel stack">
          <p className="eyebrow">Next step</p>
          <p style={{ margin: 0 }}>
            {result.referable
              ? "Queued for ophthalmologist review. The specialist confirms or overrides this result."
              : "No referral indicated at this screening. Repeat at the next routine interval."}
          </p>
        </div>
      </div>

      {result.left && <QualityPanel eye={result.left} />}
      {result.right && <QualityPanel eye={result.right} />}

      <p className="disclaimer">{result.disclaimer}</p>
    </div>
  );
}

export interface ScreeningForm {
  patientRef: string;
  age: string;
  years: string;
  left: File | null;
  right: File | null;
  result: PatientResult | null;
}

export const EMPTY_FORM: ScreeningForm = {
  patientRef: "IND-10291",
  age: "57",
  years: "8",
  left: null,
  right: null,
  result: null,
};

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
    <label>
      <span className="name">{label}</span>
      <input
        type="file"
        accept="image/*"
        onChange={(e) => onPick(e.target.files?.[0] ?? null)}
      />
      {file && <span className="retained readout">loaded: {file.name}</span>}
    </label>
  );
}

function NewScreening({
  form,
  setForm,
  onDone,
}: {
  form: ScreeningForm;
  setForm: (update: Partial<ScreeningForm>) => void;
  onDone: () => void;
}) {
  const { patientRef, age, years, left, right, result } = form;
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function submit() {
    setBusy(true);
    setError(null);
    try {
      const res = await analyze({ patientRef, age, diabetesYears: years, left, right });
      setForm({ result: res });
      onDone();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Screening failed.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="stack">
      <div className="panel stack">
        <p className="eyebrow">New screening</p>
        <div className="grid-2">
          <label>
            <span className="name">Patient ID</span>
            <input type="text" value={patientRef} onChange={(e) => setForm({ patientRef: e.target.value })} />
          </label>
          <label>
            <span className="name">Age</span>
            <input type="number" value={age} onChange={(e) => setForm({ age: e.target.value })} />
          </label>
          <label>
            <span className="name">Years since diabetes diagnosis</span>
            <input type="number" value={years} onChange={(e) => setForm({ years: e.target.value })} />
          </label>
        </div>
        <div className="grid-2">
          <EyeInput label="Left eye" file={left} onPick={(f) => setForm({ left: f })} />
          <EyeInput label="Right eye" file={right} onPick={(f) => setForm({ right: f })} />
        </div>
        <button className="primary" onClick={submit} disabled={busy || (!left && !right)}>
          {busy ? "Analyzing…" : "Start screening"}
        </button>
        {error && <p className="error">{error}</p>}
        {!left && !right && <p className="empty">Add at least one fundus image to begin.</p>}
      </div>
      {result && <Result result={result} />}
    </div>
  );
}

function ReviewQueue({ onOpen }: { onOpen: (id: string) => void }) {
  const [rows, setRows] = useState<QueueRow[] | null>(null);
  useEffect(() => {
    reviewQueue().then(setRows).catch(() => setRows([]));
  }, []);

  if (!rows) return <p className="empty">Loading queue…</p>;
  if (!rows.length) return <p className="empty">No screenings yet. Run one to populate the queue.</p>;

  return (
    <div className="panel">
      <p className="eyebrow" style={{ marginBottom: 12 }}>
        Ophthalmologist review queue — most urgent first
      </p>
      <table>
        <thead>
          <tr>
            <th>Patient</th>
            <th>Grade</th>
            <th>Priority</th>
            <th>Status</th>
            <th />
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.screening_id}>
              <td className="num">{r.patient_ref}</td>
              <td className="num" style={{ color: colourFor(r.grade) }}>
                {r.grade ?? "—"}
              </td>
              <td>
                <span className={`tag ${r.priority === "urgent" ? "urgent" : r.priority === "review" ? "referable" : "clear"}`}>
                  {r.priority ?? "pending"}
                </span>
              </td>
              <td style={{ color: "var(--muted)" }}>{r.review_status}</td>
              <td style={{ textAlign: "right" }}>
                <button className="link" onClick={() => onOpen(r.screening_id)}>
                  Open case
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

const ACTIONS: { action: ReviewSubmission["action"]; label: string }[] = [
  { action: "confirm", label: "Confirm AI result" },
  { action: "override", label: "Override grade" },
  { action: "request_recapture", label: "Request new image" },
  { action: "refer", label: "Refer patient" },
];

function CaseView({ screeningId, onBack }: { screeningId: string; onBack: () => void }) {
  const [result, setResult] = useState<PatientResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [overrideGrade, setOverrideGrade] = useState("2");
  const [notes, setNotes] = useState("");
  const [done, setDone] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    getScreening(screeningId)
      .then(setResult)
      .catch((e) => setError(e instanceof Error ? e.message : "Could not load case."));
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
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not record the review.");
    } finally {
      setBusy(false);
    }
  }

  if (error && !result) return <p className="error">{error}</p>;
  if (!result) return <p className="empty">Loading case…</p>;

  return (
    <div className="stack">
      <button className="link" onClick={onBack}>
        ← Back to queue
      </button>

      <Result result={result} />

      <div className="panel stack">
        <p className="eyebrow">Specialist review</p>
        <p style={{ margin: 0, color: "var(--muted)", fontSize: 14 }}>
          The AI result is a screening recommendation. Your decision is the
          clinical one and is recorded against this screening.
        </p>

        <div className="grid-2">
          <label>
            <span className="name">Override to grade</span>
            <select value={overrideGrade} onChange={(e) => setOverrideGrade(e.target.value)}>
              {GRADES.map((g) => (
                <option key={g.n} value={g.n}>
                  {g.n} — {g.label}
                </option>
              ))}
            </select>
          </label>
          <label>
            <span className="name">Notes</span>
            <input type="text" value={notes} onChange={(e) => setNotes(e.target.value)} />
          </label>
        </div>

        <div className="actions">
          {ACTIONS.map((a) => (
            <button key={a.action} className="secondary" disabled={busy} onClick={() => act(a.action)}>
              {a.label}
            </button>
          ))}
        </div>

        {done && (
          <p className="recorded">
            Recorded: {ACTIONS.find((a) => a.action === done)?.label.toLowerCase()}
            {done === "override" && ` to grade ${overrideGrade}`}.
          </p>
        )}
        {error && <p className="error">{error}</p>}
      </div>
    </div>
  );
}

function Simulation() {
  const [rows, setRows] = useState<ScenarioResult[] | null>(null);
  useEffect(() => {
    scenarios().then(setRows).catch(() => setRows([]));
  }, []);

  if (!rows?.length) return <p className="empty">Loading scenarios…</p>;

  return (
    <div className="stack">
      <div className="panel">
        <p className="eyebrow" style={{ marginBottom: 12 }}>
          District deployment — 100,000 patients per year
        </p>
        <table>
          <thead>
            <tr>
              <th>Scenario</th>
              <th>Bottleneck</th>
              <th>Network</th>
              <th>Doctor</th>
              <th>Meets demand</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.scenario}>
                <td>{r.scenario}</td>
                <td style={{ color: "var(--disc)" }}>{r.bottleneck}</td>
                <td className="num">{pct(r.utilization.network)}</td>
                <td className="num">{pct(r.utilization.doctor)}</td>
                <td>
                  <span className={`tag ${r.meets_demand ? "clear" : "urgent"}`}>
                    {r.meets_demand ? "yes" : "no"}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="disclaimer">
        Source: {rows[0].source}. Replaced by the Simulink/SimEvents model — these figures
        must come from a run, never be typed in.
      </p>
    </div>
  );
}

export default function App() {
  const [tab, setTab] = useState<"screen" | "queue" | "sim">("screen");
  const [version, setVersion] = useState(0);
  // Held here rather than inside NewScreening: switching tabs unmounts the tab
  // body, and a health worker who glances at the queue mid-screening must not
  // lose the images they just captured.
  const [form, setFormState] = useState<ScreeningForm>(EMPTY_FORM);
  const [openCase, setOpenCase] = useState<string | null>(null);
  const setForm = (update: Partial<ScreeningForm>) =>
    setFormState((prev) => ({ ...prev, ...update }));

  return (
    <div className="shell">
      <header className="top">
        <h1>AI-Assisted DR Screening</h1>
        <span className="flow">capture → check → analyze → explain → refer</span>
      </header>

      <nav className="tabs" role="tablist">
        {([
          ["screen", "New screening"],
          ["queue", "Review queue"],
          ["sim", "Deployment simulation"],
        ] as const).map(([key, label]) => (
          <button
            key={key}
            role="tab"
            aria-selected={tab === key}
            onClick={() => {
              if (key !== "queue") setOpenCase(null);
              setTab(key);
            }}
          >
            {label}
          </button>
        ))}
      </nav>

      {tab === "screen" && (
        <NewScreening form={form} setForm={setForm} onDone={() => setVersion((v) => v + 1)} />
      )}
      {tab === "queue" &&
        (openCase ? (
          <CaseView screeningId={openCase} onBack={() => { setOpenCase(null); setVersion((v) => v + 1); }} />
        ) : (
          <ReviewQueue key={version} onOpen={setOpenCase} />
        ))}
      {tab === "sim" && <Simulation />}
    </div>
  );
}
