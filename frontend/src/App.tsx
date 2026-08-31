import { useEffect, useState } from "react";
import {
  analyze,
  assetUrl,
  reviewQueue,
  scenarios,
  type EyeResult,
  type PatientResult,
  type QueueRow,
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

function NewScreening({ onDone }: { onDone: () => void }) {
  const [patientRef, setPatientRef] = useState("IND-10291");
  const [age, setAge] = useState("57");
  const [years, setYears] = useState("8");
  const [left, setLeft] = useState<File | null>(null);
  const [right, setRight] = useState<File | null>(null);
  const [result, setResult] = useState<PatientResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function submit() {
    setBusy(true);
    setError(null);
    try {
      const res = await analyze({ patientRef, age, diabetesYears: years, left, right });
      setResult(res);
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
            <input type="text" value={patientRef} onChange={(e) => setPatientRef(e.target.value)} />
          </label>
          <label>
            <span className="name">Age</span>
            <input type="number" value={age} onChange={(e) => setAge(e.target.value)} />
          </label>
          <label>
            <span className="name">Years since diabetes diagnosis</span>
            <input type="number" value={years} onChange={(e) => setYears(e.target.value)} />
          </label>
        </div>
        <div className="grid-2">
          <label>
            <span className="name">Left eye</span>
            <input type="file" accept="image/*" onChange={(e) => setLeft(e.target.files?.[0] ?? null)} />
          </label>
          <label>
            <span className="name">Right eye</span>
            <input type="file" accept="image/*" onChange={(e) => setRight(e.target.files?.[0] ?? null)} />
          </label>
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

function ReviewQueue() {
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
            </tr>
          ))}
        </tbody>
      </table>
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
            onClick={() => setTab(key)}
          >
            {label}
          </button>
        ))}
      </nav>

      {tab === "screen" && <NewScreening onDone={() => setVersion((v) => v + 1)} />}
      {tab === "queue" && <ReviewQueue key={version} />}
      {tab === "sim" && <Simulation />}
    </div>
  );
}
