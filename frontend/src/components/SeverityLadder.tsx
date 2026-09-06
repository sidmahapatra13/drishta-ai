import { GRADES, REFERABLE_FROM_GRADE, colourFor } from "../lib/grade";
import "./SeverityLadder.css";

/**
 * The severity scale drawn as the ordinal instrument it is.
 *
 * A 5-class bar chart would imply the grades are independent categories. They
 * are not: they are ordered, and one boundary among them - the line between 1
 * and 2 - decides whether this patient sees an ophthalmologist. Drawing the
 * scale as a ladder with that boundary as a real rule shows both facts at
 * once, and shows them as properties of the scale rather than of our model.
 */
export function SeverityLadder({ grade }: { grade: number | null }) {
  return (
    <ol className="ladder">
      {GRADES.map((g) => (
        <li key={g.n}>
          {g.n === REFERABLE_FROM_GRADE && (
            <p className="ladder-threshold">
              <span>referral threshold</span>
            </p>
          )}
          <span
            className="ladder-row"
            data-active={g.n === grade}
            style={{ ["--reading" as string]: colourFor(g.n) }}
          >
            <span className="ladder-num readout">{g.n}</span>
            <span className="ladder-label">{g.label}</span>
            {g.n === grade && <span className="ladder-here">this patient</span>}
          </span>
        </li>
      ))}
    </ol>
  );
}
