import { useEffect, useState } from "react";
import { reviewQueue, type QueueRow } from "../api";
import { colourFor } from "../lib/grade";
import "./ReviewQueue.css";

/**
 * The ophthalmologist's worklist, most urgent first.
 *
 * The ordering is the product, not a convenience: the premise of the whole
 * system is that a scarce specialist spends their time on the patients most
 * likely to lose sight. Sorting is done in SQL (db.review_queue) so the
 * ordering is a property of the system rather than of this table.
 */
export function ReviewQueue({ onOpen }: { onOpen: (id: string) => void }) {
  const [rows, setRows] = useState<QueueRow[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    reviewQueue()
      .then(setRows)
      .catch(() => setError("The queue could not be loaded. Is the API running?"));
  }, []);

  if (error) return <p className="queue-empty">{error}</p>;
  if (!rows) return <p className="queue-empty">Loading the queue…</p>;
  if (!rows.length)
    return <p className="queue-empty">No screenings yet. Run one and it appears here.</p>;

  return (
    <table className="queue">
      <thead>
        <tr>
          <th>Patient</th>
          <th>Grade</th>
          <th>Priority</th>
          <th>Review</th>
          <th />
        </tr>
      </thead>
      <tbody>
        {rows.map((r) => (
          <tr key={r.screening_id}>
            <td className="readout">{r.patient_ref}</td>
            <td className="readout queue-grade" style={{ color: colourFor(r.grade) }}>
              {r.grade ?? "—"}
            </td>
            <td>
              <span
                className={`tag ${
                  r.priority === "urgent"
                    ? "urgent"
                    : r.priority === "review"
                      ? "referable"
                      : r.priority === "low"
                        ? "clear"
                        : "pending"
                }`}
              >
                {/* No priority means no gradable image, so the case is waiting
                    on a recapture rather than sitting at the bottom of the
                    queue as a low-severity pass. */}
                {r.priority ?? "recapture"}
              </span>
            </td>
            <td className="queue-status">{r.review_status}</td>
            <td className="queue-action">
              <button className="link" onClick={() => onOpen(r.screening_id)}>
                Open case
              </button>
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
