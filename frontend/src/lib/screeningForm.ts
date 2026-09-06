import type { PatientResult } from "../api";

/** Screening inputs, held in App so a tab switch does not discard the images
 *  a health worker has already loaded. */
export interface ScreeningFormState {
  patientRef: string;
  age: string;
  years: string;
  left: File | null;
  right: File | null;
  result: PatientResult | null;
}

export const EMPTY_FORM: ScreeningFormState = {
  patientRef: "IND-10291",
  age: "57",
  years: "8",
  left: null,
  right: null,
  result: null,
};
