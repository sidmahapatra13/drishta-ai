import { useState } from "react";
import { CaseView } from "./components/CaseView";
import { PatientResultView } from "./components/PatientResultView";
import { ProvenanceBar } from "./components/ProvenanceBar";
import { ReviewQueue } from "./components/ReviewQueue";
import { ScreeningForm } from "./components/ScreeningForm";
import { SimulationTable } from "./components/SimulationTable";
import { useRoute, type Tab } from "./lib/route";
import { EMPTY_FORM, type ScreeningFormState } from "./lib/screeningForm";
import "./App.css";

const TABS: [Tab, string][] = [
  ["screen", "New screening"],
  ["queue", "Review queue"],
  ["sim", "Deployment simulation"],
];

export default function App() {
  const [{ tab, caseId }, go] = useRoute();
  const [version, setVersion] = useState(0);
  // Held here rather than inside ScreeningForm: switching tabs unmounts the tab
  // body, and a health worker who glances at the queue mid-screening must not
  // lose the images they just captured.
  const [form, setFormState] = useState<ScreeningFormState>(EMPTY_FORM);
  const setForm = (update: Partial<ScreeningFormState>) =>
    setFormState((prev) => ({ ...prev, ...update }));

  const resultEye = form.result?.left ?? form.result?.right;

  return (
    <div className="shell">
      <header className="masthead">
        <h1>Diabetic retinopathy screening</h1>
        <p className="masthead-flow readout">capture · check · analyse · explain · refer</p>
      </header>

      <nav className="tabs" role="tablist">
        {TABS.map(([key, label]) => (
          <button
            key={key}
            role="tab"
            aria-selected={tab === key}
            onClick={() => go({ tab: key, caseId: null })}
          >
            {label}
          </button>
        ))}
      </nav>

      <main className="main">
        {tab === "screen" && (
          <>
            <ScreeningForm
              form={form}
              setForm={setForm}
              onDone={() => setVersion((v) => v + 1)}
            />
            {form.result && (
              <>
                <ProvenanceBar
                  modelVersion={resultEye?.model_version}
                  thresholdVersion={resultEye?.threshold_version}
                />
                <PatientResultView result={form.result} />
              </>
            )}
          </>
        )}

        {tab === "queue" &&
          (caseId ? (
            <CaseView
              screeningId={caseId}
              onBack={() => {
                go({ tab: "queue", caseId: null });
                setVersion((v) => v + 1);
              }}
            />
          ) : (
            <ReviewQueue
              key={version}
              onOpen={(id) => go({ tab: "queue", caseId: id })}
            />
          ))}

        {tab === "sim" && <SimulationTable />}
      </main>
    </div>
  );
}
