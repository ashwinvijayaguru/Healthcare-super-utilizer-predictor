export type Tier = "low" | "rising" | "high" | "very_high";
export type Severity = "critical" | "warning" | "advisory";

export interface FieldSpec {
  name: string;
  label: string;
  kind: "numeric" | "binary" | "ordinal";
  min: number | null;
  max: number | null;
  unit: string;
  higher_is_protective: boolean;
}

export interface FormGroup {
  key: string;
  label: string;
  fields: FieldSpec[];
}

export interface FormSchema {
  groups: FormGroup[];
  tiers: { key: Tier; label: string; action: string; patient_line: string }[];
  outcome_definition: string;
}

export interface Driver {
  feature: string;
  label: string;
  state: string;
  group: string;
  group_label: string;
  value: number;
  contribution: number;
  direction: "increases" | "decreases";
}

export interface RedFlag {
  code: string;
  severity: Severity;
  title: string;
  detail: string;
  patient_detail: string;
  suggested_action: string;
  evidence: Record<string, unknown>;
}

export interface PatientSummary {
  headline: string;
  body: string;
  what_this_means: string[];
  next_steps: { text: string; owner: "care_team" | "patient" }[];
  reassurance: string;
  source: "llm" | "template";
}

export interface Assessment {
  assessment_id: string | null;
  external_ref: string;
  risk_score: number;
  raw_score: number;
  risk_percent: number;
  risk_tier: Tier;
  risk_tier_label: string;
  tier_action: string;
  percentile: number | null;
  baseline_rate: number;
  lift_vs_baseline: number;
  expected_in_100: number;
  drivers: Driver[];
  protective_factors: Driver[];
  group_attribution: Record<string, number>;
  red_flags: RedFlag[];
  clinician_summary: string;
  patient_summary: PatientSummary;
  model_name: string;
  model_version: string;
  model_trained_at: string;
  latency_ms: number;
  disclaimer: string;
}

export interface ModelInfo {
  model_name: string;
  model_version: string;
  estimator_kind: string;
  trained_at: string;
  tier_cuts: Record<string, number>;
  holdout_metrics: Record<string, number>;
  base_rate: number;
  outcome_definition: string;
}

export type Intake = Record<string, number>;
