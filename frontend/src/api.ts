/**
 * Typed client mirroring backend/app/contract.py.
 *
 * These types are the frozen contract. If they drift from the Python models the
 * build breaks here first, which is the point - six people are working against
 * this shape while the model behind it is still a stub.
 */

const BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";

export type QualityStatus = "gradable" | "borderline" | "ungradable";
export type Priority = "urgent" | "review" | "low";
export type Eye = "left" | "right";

export interface QualityReport {
  focus: number;
  illumination: number;
  field_of_view: number;
  score: number;
  status: QualityStatus;
  issues: string[];
  enhancement_applied: boolean;
}

export interface Lesion {
  type: string;
  count: number;
  area_px: number;
  mask_url: string | null;
  confidence: number;
}

export interface EyeResult {
  eye: Eye;
  quality: QualityReport;
  grade: number | null;
  grade_label: string | null;
  referable: boolean | null;
  confidence: number | null;
  calibrated: boolean;
  probabilities: number[] | null;
  lesions: Lesion[];
  gradcam_url: string | null;
  image_url: string | null;
  model_version: string;
  threshold_version: string;
  recapture_required: boolean;
  recapture_reason: string | null;
}

export interface PatientResult {
  screening_id: string;
  patient_ref: string;
  left: EyeResult | null;
  right: EyeResult | null;
  grade: number | null;
  grade_label: string | null;
  referable: boolean | null;
  priority: Priority | null;
  confidence: number | null;
  recapture_required: boolean;
  recapture_eyes: Eye[];
  summary: string;
  disclaimer: string;
}

export interface QueueRow {
  screening_id: string;
  patient_ref: string;
  created_at: string;
  grade: number | null;
  referable: number | null;
  priority: Priority | null;
  review_status: string;
}

export interface ScenarioResult {
  scenario: string;
  patients_per_day: number;
  images_per_day: number;
  referrals_per_day: number;
  utilization: Record<string, number>;
  bottleneck: string;
  avg_doctor_queue: number | null;
  doctor_utilization_without_ai: number;
  meets_demand: boolean;
  source: string;
}

export const assetUrl = (path: string | null) => (path ? `${BASE}${path}` : null);

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const detail = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(detail.detail ?? `Request failed (${res.status})`);
  }
  return res.json() as Promise<T>;
}

export async function analyze(fields: {
  patientRef: string;
  age?: string;
  diabetesYears?: string;
  left?: File | null;
  right?: File | null;
}): Promise<PatientResult> {
  const body = new FormData();
  body.append("patient_ref", fields.patientRef);
  if (fields.age) body.append("age", fields.age);
  if (fields.diabetesYears) body.append("diabetes_duration_years", fields.diabetesYears);
  if (fields.left) body.append("left", fields.left);
  if (fields.right) body.append("right", fields.right);
  return json(await fetch(`${BASE}/screening/analyze`, { method: "POST", body }));
}

export async function reviewQueue(): Promise<QueueRow[]> {
  return json(await fetch(`${BASE}/review/queue`));
}

export async function scenarios(): Promise<ScenarioResult[]> {
  return json(await fetch(`${BASE}/simulation/scenarios`));
}

export async function askClinical(question: string, screeningId?: string) {
  return json<{ answer: string; citations: { title: string; source: string }[] }>(
    await fetch(`${BASE}/clinical/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, screening_id: screeningId ?? null }),
    }),
  );
}

export async function getScreening(screeningId: string): Promise<PatientResult> {
  return json(await fetch(`${BASE}/screening/${screeningId}`));
}

export interface ReviewSubmission {
  action: "confirm" | "override" | "request_recapture" | "refer";
  reviewer_ref: string;
  override_grade?: number | null;
  notes?: string | null;
}

export async function submitReview(screeningId: string, review: ReviewSubmission) {
  return json<{ status: string }>(
    await fetch(`${BASE}/review/${screeningId}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(review),
    }),
  );
}
