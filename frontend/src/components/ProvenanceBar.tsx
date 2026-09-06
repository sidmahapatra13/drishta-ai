import "./ProvenanceBar.css";

/**
 * Which network produced the numbers on this page.
 *
 * Kept permanently visible rather than buried, because "every number shown to
 * a judge comes from a real run" is only checkable if the run is named. It is
 * also how the deterministic fallback announces itself: when the weights are
 * absent the backend reports `stub-v0`, and that must be impossible to mistake
 * for a real result.
 */
export function ProvenanceBar({
  modelVersion,
  thresholdVersion,
}: {
  modelVersion?: string;
  thresholdVersion?: string;
}) {
  if (!modelVersion) return null;
  const stub = modelVersion.startsWith("stub");

  return (
    <p className="provenance readout" data-stub={stub}>
      {stub ? "STUB PREDICTOR - not a real result" : modelVersion}
      {!stub && thresholdVersion && <span> · {thresholdVersion}</span>}
    </p>
  );
}
