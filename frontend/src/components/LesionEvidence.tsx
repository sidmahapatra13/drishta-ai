import type { Lesion } from "../api";
import "./LesionEvidence.css";

/**
 * Per-class lesion counts.
 *
 * Silent when the list is empty, which is what the detector currently returns.
 * A panel reading "0 microaneurysms" is a clinical finding; an absent panel is
 * an absent claim. The distinction matters because the backend deliberately
 * reports nothing rather than inventing counts, and the UI must not undo that
 * by rendering the absence as a negative result.
 */
export function LesionEvidence({ lesions }: { lesions: Lesion[] }) {
  if (!lesions.length) return null;

  return (
    <div className="lesions">
      <h3 className="lesions-title">Lesion evidence</h3>
      <table>
        <tbody>
          {lesions.map((l) => (
            <tr key={l.type}>
              <td className="lesions-type">{l.type}</td>
              <td className="readout lesions-count">{l.count}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="lesions-caveat">
        Detector output. Not a clinically authoritative lesion count.
      </p>
    </div>
  );
}
