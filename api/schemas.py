"""Request and response schemas.

``ClinicalIntake`` is written out explicitly rather than generated from
ml.config, because an explicit model gives real IDE completion, per-field
documentation and readable OpenAPI output. The cost of writing it by hand is
that it can drift from the training contract, so ``assert_contract_alignment``
is executed at import time and fails loudly if the two ever disagree.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ml.config import FEATURE_NAMES

Unit = Annotated[int, Field(ge=0, le=1)]


class ClinicalIntake(BaseModel):
    """Everything the model needs about one member. No identifiers."""

    model_config = ConfigDict(extra="forbid")

    # demographics
    age: float = Field(..., ge=18, le=105, description="Age in years")
    lives_alone: Unit = 0

    # prior utilisation
    ed_visits_12mo: int = Field(0, ge=0, le=60)
    inpatient_admits_12mo: int = Field(0, ge=0, le=30)
    inpatient_days_12mo: int = Field(0, ge=0, le=200)
    days_since_last_discharge: int = Field(999, ge=0, le=999,
                                           description="Use 999 when there is no prior admission")
    prior_30d_readmission: Unit = 0
    outpatient_visits_12mo: int = Field(0, ge=0, le=100)
    missed_appointments_12mo: int = Field(0, ge=0, le=40)
    has_primary_care: Unit = 1

    # clinical
    chronic_condition_count: int = Field(0, ge=0, le=20)
    charlson_index: int = Field(0, ge=0, le=20)
    heart_failure: Unit = 0
    copd: Unit = 0
    diabetes_complicated: Unit = 0
    ckd_stage4_plus: Unit = 0
    cirrhosis: Unit = 0
    cancer_active: Unit = 0
    serious_mental_illness: Unit = 0
    substance_use_disorder: Unit = 0
    cognitive_impairment: Unit = 0
    mobility_impairment: Unit = 0
    falls_12mo: int = Field(0, ge=0, le=20)

    # medications
    medication_count: int = Field(0, ge=0, le=40)
    high_risk_med_count: int = Field(0, ge=0, le=15)
    medication_adherence_pdc: float = Field(0.85, ge=0.0, le=1.0,
                                            description="Proportion of days covered, 0-1")
    opioid_therapy: Unit = 0

    # social drivers
    housing_instability: Unit = 0
    food_insecurity: Unit = 0
    transportation_barrier: Unit = 0
    social_isolation: Unit = 0
    financial_strain: Unit = 0
    uninsured_or_medicaid: Unit = 0
    caregiver_support: Unit = 1
    area_deprivation_index: int = Field(50, ge=1, le=100)
    limited_health_literacy: Unit = 0

    @field_validator("high_risk_med_count")
    @classmethod
    def _high_risk_within_total(cls, v: int, info: Any) -> int:
        total = info.data.get("medication_count")
        if total is not None and v > total:
            raise ValueError("high_risk_med_count cannot exceed medication_count")
        return v

    @field_validator("inpatient_days_12mo")
    @classmethod
    def _days_require_admits(cls, v: int, info: Any) -> int:
        admits = info.data.get("inpatient_admits_12mo")
        if admits == 0 and v > 0:
            raise ValueError("inpatient_days_12mo must be 0 when there are no admissions")
        return v


def assert_contract_alignment() -> None:
    """Fail fast if the API schema and the training contract diverge."""
    api_fields = set(ClinicalIntake.model_fields)
    contract = set(FEATURE_NAMES)
    if api_fields != contract:
        raise RuntimeError(
            "intake schema does not match the trained feature contract; "
            f"missing={sorted(contract - api_fields)} unexpected={sorted(api_fields - contract)}"
        )


assert_contract_alignment()


class PatientRef(BaseModel):
    """Pseudonymous reference. Deliberately cannot carry a name or a full DOB."""

    model_config = ConfigDict(extra="forbid")

    external_ref: str = Field(..., min_length=1, max_length=128,
                              description="Opaque key that resolves to a person only in your system")
    birth_year: int | None = Field(None, ge=1900, le=2100)
    sex_at_birth: Literal["female", "male", "other", "unknown"] | None = None
    postal_sector: str | None = Field(None, max_length=8,
                                      description="First 3 ZIP digits only, per HIPAA Safe Harbor")

    @field_validator("external_ref")
    @classmethod
    def _no_obvious_identifiers(cls, v: str) -> str:
        # A weak but useful guard: this demo must not accumulate identifiers by
        # accident because someone pasted a name into the reference field.
        if "@" in v or v.count(" ") >= 2:
            raise ValueError("external_ref must be an opaque key, not a name or email address")
        return v


class AssessmentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    patient: PatientRef
    intake: ClinicalIntake
    generate_narrative: bool = True
    persist: bool = True


class DriverOut(BaseModel):
    feature: str
    label: str
    state: str
    group: str
    group_label: str
    value: float
    contribution: float
    direction: Literal["increases", "decreases"]


class FlagOut(BaseModel):
    code: str
    severity: Literal["critical", "warning", "advisory"]
    title: str
    detail: str
    patient_detail: str
    suggested_action: str
    evidence: dict[str, Any] = {}


class NextStep(BaseModel):
    text: str
    owner: Literal["care_team", "patient"]


class PatientSummary(BaseModel):
    headline: str
    body: str
    what_this_means: list[str]
    next_steps: list[NextStep]
    reassurance: str
    reading_level_note: str = "Written to be read without medical training."
    source: Literal["llm", "template"] = "template"


class AssessmentResponse(BaseModel):
    assessment_id: str | None
    external_ref: str
    risk_score: float = Field(..., description="Calibrated probability, 0-1, bounded to the supported range")
    raw_score: float = Field(..., description="Unbounded calibrated output, for audit")
    risk_percent: float
    risk_tier: Literal["low", "rising", "high", "very_high"]
    risk_tier_label: str
    tier_action: str
    percentile: float | None
    baseline_rate: float = Field(..., description="Outcome rate in the whole reference panel")
    lift_vs_baseline: float
    expected_in_100: int = Field(..., description="Of 100 similar members, how many meet the outcome")
    drivers: list[DriverOut]
    protective_factors: list[DriverOut]
    group_attribution: dict[str, float]
    red_flags: list[FlagOut]
    clinician_summary: str
    patient_summary: PatientSummary
    model_name: str
    model_version: str
    model_trained_at: str
    latency_ms: int
    created_at: datetime | None = None
    disclaimer: str


class AssessmentListItem(BaseModel):
    assessment_id: str
    external_ref: str
    risk_score: float
    risk_tier: str
    model_version: str
    created_at: datetime


class ModelInfo(BaseModel):
    model_name: str
    model_version: str
    estimator_kind: str
    feature_schema_version: str
    trained_at: str
    tier_cuts: dict[str, float]
    holdout_metrics: dict[str, Any]
    base_rate: float
    outcome_definition: str


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    database: bool
    narrative_provider: str
    version: str
