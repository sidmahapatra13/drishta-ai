/**
 * Vocabulary shared by every view that shows a severity grade.
 *
 * The scale, its labels and the referral threshold are clinical facts fixed by
 * the problem statement, not presentation choices, so they live in one place
 * rather than being retyped per component.
 */

export const GRADES = [
  { n: 0, label: "No apparent DR" },
  { n: 1, label: "Mild NPDR" },
  { n: 2, label: "Moderate NPDR" },
  { n: 3, label: "Severe NPDR" },
  { n: 4, label: "Proliferative DR" },
] as const;

/** The International Clinical DR Severity Scale calls grade 2 and above
 *  referable. Fixed by the PS; mirrored from contract.REFERABLE_FROM_GRADE. */
export const REFERABLE_FROM_GRADE = 2;

export const colourFor = (grade: number | null) =>
  grade === null
    ? "var(--muted)"
    : grade >= 4
      ? "var(--urgent)"
      : grade >= REFERABLE_FROM_GRADE
        ? "var(--retina)"
        : "var(--clear)";

export const pct = (v: number) => `${Math.round(v * 100)}%`;

/**
 * Turn the quality gate's issue list into one instruction.
 *
 * The gate ranks its issues by how far each metric sits below its own floor,
 * so the first entry is the dominant failure and the only one worth acting on.
 * A health worker holding a camera needs one thing to do, not a list of five.
 */
const RECAPTURE_ADVICE: Record<string, { title: string; action: string }> = {
  "underexposed": {
    title: "Too dark to grade",
    action: "Increase the flash or move closer, then capture this eye again.",
  },
  "overexposed": {
    title: "Too bright to grade",
    action: "Reduce the flash, then capture this eye again.",
  },
  "blown highlights": {
    title: "Washed out by glare",
    action: "Angle the camera slightly away from the reflection and capture again.",
  },
  "severe defocus": {
    title: "Out of focus",
    action: "Hold the camera steady, refocus, and capture this eye again.",
  },
  "insufficient focus": {
    title: "Not sharp enough to grade",
    action: "Hold the camera steady, refocus, and capture this eye again.",
  },
  "uneven illumination": {
    title: "Lit unevenly",
    action: "Centre the camera on the pupil and capture this eye again.",
  },
  "retina too small in frame": {
    title: "Too far from the eye",
    action: "Move the camera closer until the retina fills the frame.",
  },
  "retina off-centre": {
    title: "Retina not centred",
    action: "Line the camera up with the pupil and capture this eye again.",
  },
  "no retinal area detected": {
    title: "No retina in frame",
    action: "Point the camera into the pupil and capture this eye again.",
  },
};

const FALLBACK = {
  title: "Not gradable",
  action: "Reposition the camera and capture this eye again.",
};

export function recaptureAdvice(issues: string[]) {
  return (issues.length && RECAPTURE_ADVICE[issues[0]]) || FALLBACK;
}
