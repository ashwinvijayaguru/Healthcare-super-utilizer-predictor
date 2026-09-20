import type { Assessment, FormSchema, Intake, ModelInfo } from "./types";

const BASE = import.meta.env.VITE_API_BASE ?? "";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${BASE}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
  });
  if (!response.ok) {
    let detail = `Request failed (${response.status})`;
    try {
      const body = await response.json();
      // FastAPI validation errors arrive as a list of field-level problems;
      // surface the first one rather than "[object Object]".
      if (Array.isArray(body?.detail)) {
        const first = body.detail[0];
        detail = `${first?.loc?.slice(-1)[0] ?? "input"}: ${first?.msg ?? "invalid"}`;
      } else if (typeof body?.detail === "string") {
        detail = body.detail;
      }
    } catch {
      /* response had no JSON body; keep the status-code message */
    }
    throw new Error(detail);
  }
  return response.json() as Promise<T>;
}

export const getFormSchema = () => request<FormSchema>("/api/v1/form-schema");
export const getModelInfo = () => request<ModelInfo>("/api/v1/model");

export const scorePatient = (externalRef: string, intake: Intake) =>
  request<Assessment>("/api/v1/assessments", {
    method: "POST",
    body: JSON.stringify({ patient: { external_ref: externalRef }, intake }),
  });

/** Defaults for a healthy adult, so the form opens in a valid state. */
export const DEFAULT_INTAKE: Intake = {
  age: 45, lives_alone: 0,
  ed_visits_12mo: 0, inpatient_admits_12mo: 0, inpatient_days_12mo: 0,
  days_since_last_discharge: 999, prior_30d_readmission: 0,
  outpatient_visits_12mo: 3, missed_appointments_12mo: 0, has_primary_care: 1,
  chronic_condition_count: 0, charlson_index: 0,
  heart_failure: 0, copd: 0, diabetes_complicated: 0, ckd_stage4_plus: 0,
  cirrhosis: 0, cancer_active: 0, serious_mental_illness: 0,
  substance_use_disorder: 0, cognitive_impairment: 0, mobility_impairment: 0,
  falls_12mo: 0,
  medication_count: 2, high_risk_med_count: 0, medication_adherence_pdc: 0.9, opioid_therapy: 0,
  housing_instability: 0, food_insecurity: 0, transportation_barrier: 0,
  social_isolation: 0, financial_strain: 0, uninsured_or_medicaid: 0,
  caregiver_support: 1, area_deprivation_index: 45, limited_health_literacy: 0,
};

export interface Preset {
  id: string;
  name: string;
  note: string;
  intake: Intake;
}

/**
 * Demo profiles. Composites written to exercise distinct paths through the
 * model, not records of real people.
 */
export const PRESETS: Preset[] = [
  {
    id: "stable",
    name: "Stable, well-supported",
    note: "Nothing flagged. Shows what a low score looks like.",
    intake: { ...DEFAULT_INTAKE, age: 38 },
  },
  {
    id: "clinical",
    name: "Clinically complex, socially stable",
    note: "High medical burden, strong support. Risk is driven by conditions.",
    intake: {
      ...DEFAULT_INTAKE, age: 74, chronic_condition_count: 5, charlson_index: 6,
      heart_failure: 1, copd: 1, diabetes_complicated: 1, mobility_impairment: 1,
      ed_visits_12mo: 2, inpatient_admits_12mo: 1, inpatient_days_12mo: 6,
      days_since_last_discharge: 120, outpatient_visits_12mo: 12,
      medication_count: 12, high_risk_med_count: 3, medication_adherence_pdc: 0.88,
      caregiver_support: 1, area_deprivation_index: 30,
    },
  },
  {
    id: "social",
    name: "Socially driven risk",
    note: "Modest medical burden, heavy unmet social need. The case the model exists for.",
    intake: {
      ...DEFAULT_INTAKE, age: 52, lives_alone: 1, chronic_condition_count: 2,
      charlson_index: 1, diabetes_complicated: 1, serious_mental_illness: 1,
      ed_visits_12mo: 3, outpatient_visits_12mo: 1, missed_appointments_12mo: 5,
      has_primary_care: 0, medication_count: 6, high_risk_med_count: 2,
      medication_adherence_pdc: 0.41, housing_instability: 1, food_insecurity: 1,
      transportation_barrier: 1, social_isolation: 1, financial_strain: 1,
      uninsured_or_medicaid: 1, caregiver_support: 0, area_deprivation_index: 93,
      limited_health_literacy: 1,
    },
  },
  {
    id: "transition",
    name: "Recently discharged",
    note: "Inside the 30-day window, where readmission risk concentrates.",
    intake: {
      ...DEFAULT_INTAKE, age: 68, lives_alone: 1, chronic_condition_count: 4,
      charlson_index: 4, heart_failure: 1, ckd_stage4_plus: 1,
      ed_visits_12mo: 3, inpatient_admits_12mo: 2, inpatient_days_12mo: 14,
      days_since_last_discharge: 9, prior_30d_readmission: 1,
      outpatient_visits_12mo: 5, missed_appointments_12mo: 2,
      medication_count: 14, high_risk_med_count: 4, medication_adherence_pdc: 0.62,
      falls_12mo: 1, caregiver_support: 0, transportation_barrier: 1,
      area_deprivation_index: 70,
    },
  },
];
