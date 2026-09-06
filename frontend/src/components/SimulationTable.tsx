import { useEffect, useState } from "react";
import { scenarios, type ScenarioResult } from "../api";
import { pct } from "../lib/grade";
import "./SimulationTable.css";

/** District-scale deployment. Every figure comes from a run of the model in
 *  services/simulation.py - the source is printed underneath so nobody can
 *  mistake the analytical estimate for the Simulink/SimEvents result that will
 *  replace it. */
export function SimulationTable() {
  const [rows, setRows] = useState<ScenarioResult[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    scenarios()
      .then(setRows)
      .catch(() => setError("Scenarios could not be loaded. Is the API running?"));
  }, []);

  if (error) return <p className="queue-empty">{error}</p>;
  if (!rows?.length) return <p className="queue-empty">Running scenarios…</p>;

  return (
    <div className="sim">
      <table className="queue">
        <thead>
          <tr>
            <th>Scenario</th>
            <th>Bottleneck</th>
            <th>Network</th>
            <th>Specialist</th>
            <th>Meets demand</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.scenario}>
              <td>{r.scenario}</td>
              <td className="sim-bottleneck">{r.bottleneck}</td>
              <td className="readout">{pct(r.utilization.network)}</td>
              <td className="readout">{pct(r.utilization.doctor)}</td>
              <td>
                <span className={`tag ${r.meets_demand ? "clear" : "urgent"}`}>
                  {r.meets_demand ? "yes" : "no"}
                </span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="sim-source readout">source: {rows[0].source}</p>
      <p className="sim-note">
        Replaced by the Simulink/SimEvents model. These figures come from a run,
        never from an estimate typed in by hand.
      </p>
    </div>
  );
}
