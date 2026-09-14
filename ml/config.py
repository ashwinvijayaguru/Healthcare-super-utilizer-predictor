"""Single source of truth for the feature contract, label definition and risk tiers.

Everything downstream (cohort builder, trainer, API, UI) imports from here so the
training-time and serving-time feature space can never silently drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"

MODEL_NAME = "superutilizer-risk"
MODEL_VERSION = "1.2.0"
FEATURE_SCHEMA_VERSION = "2024.11"

# --------------------------------------------------------------------------
# Outcome definition
# --------------------------------------------------------------------------
# A patient is a "super utilizer" in the 6-12 month follow-up window if ANY of:
#   * >= 4 emergency department visits, OR
#   * >= 2 acute inpatient admissions, OR
#   * total allowed medical cost in the top 5% of the cohort.
# This mirrors the operational definitions used by CMMI's Health Care Innovation
# Awards, the Camden Coalition hotspotting program, and most published
# super-utilizer literature.
LABEL_ED_VISITS = 4
LABEL_ADMISSIONS = 2
LABEL_COST_PERCENTILE = 95
FOLLOW_UP_MONTHS = (6, 12)


@dataclass(frozen=True)
class FeatureSpec:
    """Describes one model input: how it is validated, and how it is explained."""

    name: str
    kind: str  # "numeric" | "binary" | "ordinal"
    label: str  # clinician-facing label
    group: str  # demographics | utilization | clinical | medications | sdoh
    minimum: float | None = None
    maximum: float | None = None
    unit: str = ""
    # Plain-language phrasing used when this feature drives a patient's score up.
    patient_phrase: str = ""
    # Whether a HIGH value is protective rather than risky.
    higher_is_protective: bool = False
    # How to name a binary feature when it is ABSENT. Auto-negation produces
    # nonsense like "No has an established primary care provider", so every
    # binary states its own absent form.
    absent_label: str = ""
    choices: tuple[str, ...] = field(default_factory=tuple)


FEATURES: tuple[FeatureSpec, ...] = (
    # ---------------------------------------------------------------- demographics
    FeatureSpec("age", "numeric", "Age", "demographics", 18, 105, "years",
                "being in an age group that tends to need more hospital care"),
    FeatureSpec("lives_alone", "binary", "Lives alone", "demographics",
                patient_phrase="living alone, with fewer people nearby to help day to day", absent_label="Does not live alone"),
    # ---------------------------------------------------------------- utilization
    FeatureSpec("ed_visits_12mo", "numeric", "ED visits (past 12 months)", "utilization", 0, 60, "visits",
                "how often you have needed the emergency room in the past year"),
    FeatureSpec("inpatient_admits_12mo", "numeric", "Inpatient admissions (past 12 months)", "utilization", 0, 30,
                "admissions", "how many times you have been admitted to hospital in the past year"),
    FeatureSpec("inpatient_days_12mo", "numeric", "Inpatient days (past 12 months)", "utilization", 0, 200, "days",
                "the total number of nights you have spent in hospital"),
    FeatureSpec("days_since_last_discharge", "numeric", "Days since last discharge", "utilization", 0, 999, "days",
                "having been discharged from hospital recently", higher_is_protective=True),
    FeatureSpec("prior_30d_readmission", "binary", "30-day readmission in past year", "utilization",
                patient_phrase="having had to return to hospital within a month of going home", absent_label="No 30-day readmission in past year"),
    FeatureSpec("outpatient_visits_12mo", "numeric", "Outpatient visits (past 12 months)", "utilization", 0, 100,
                "visits", "how much routine outpatient care you have had"),
    FeatureSpec("missed_appointments_12mo", "numeric", "Missed appointments (past 12 months)", "utilization", 0, 40,
                "appointments", "appointments that were missed, which can let problems build up"),
    FeatureSpec("has_primary_care", "binary", "Has an established primary care provider", "utilization",
                patient_phrase="not having a regular family doctor who knows your history",
                higher_is_protective=True, absent_label="No established primary care provider"),
    # ---------------------------------------------------------------- clinical
    FeatureSpec("chronic_condition_count", "numeric", "Chronic condition count", "clinical", 0, 20, "conditions",
                "the number of long-term conditions you are managing at once"),
    FeatureSpec("charlson_index", "numeric", "Charlson comorbidity index", "clinical", 0, 20, "points",
                "the overall burden of your medical conditions"),
    FeatureSpec("heart_failure", "binary", "Heart failure", "clinical",
                patient_phrase="heart failure, which can flare up quickly", absent_label="No heart failure"),
    FeatureSpec("copd", "binary", "COPD", "clinical",
                patient_phrase="COPD, which can flare up and make breathing difficult", absent_label="No COPD"),
    FeatureSpec("diabetes_complicated", "binary", "Diabetes with complications", "clinical",
                patient_phrase="diabetes that has started to affect other organs", absent_label="No diabetes complications"),
    FeatureSpec("ckd_stage4_plus", "binary", "CKD stage 4+ or dialysis", "clinical",
                patient_phrase="advanced kidney disease", absent_label="No advanced kidney disease"),
    FeatureSpec("cirrhosis", "binary", "Cirrhosis / advanced liver disease", "clinical",
                patient_phrase="advanced liver disease", absent_label="No advanced liver disease"),
    FeatureSpec("cancer_active", "binary", "Active cancer treatment", "clinical",
                patient_phrase="cancer treatment that is currently active", absent_label="No active cancer treatment"),
    FeatureSpec("serious_mental_illness", "binary", "Serious mental illness", "clinical",
                patient_phrase="a mental health condition that needs ongoing support", absent_label="No serious mental illness"),
    FeatureSpec("substance_use_disorder", "binary", "Substance use disorder", "clinical",
                patient_phrase="a substance use condition that is affecting your health", absent_label="No substance use disorder"),
    FeatureSpec("cognitive_impairment", "binary", "Dementia / cognitive impairment", "clinical",
                patient_phrase="memory or thinking difficulties that make self-care harder", absent_label="No cognitive impairment"),
    FeatureSpec("mobility_impairment", "binary", "Mobility impairment", "clinical",
                patient_phrase="difficulty moving around safely", absent_label="No mobility impairment"),
    FeatureSpec("falls_12mo", "numeric", "Falls (past 12 months)", "clinical", 0, 20, "falls",
                "recent falls, which often lead to injuries and hospital visits"),
    # ---------------------------------------------------------------- medications
    FeatureSpec("medication_count", "numeric", "Active medication count", "medications", 0, 40, "medications",
                "the number of medicines you take, which is a lot to keep track of"),
    FeatureSpec("high_risk_med_count", "numeric", "High-risk medications", "medications", 0, 15, "medications",
                "medicines that need careful monitoring, such as blood thinners or insulin"),
    FeatureSpec("medication_adherence_pdc", "numeric", "Medication adherence (PDC)", "medications", 0.0, 1.0, "ratio",
                "gaps in taking medicines as prescribed", higher_is_protective=True),
    FeatureSpec("opioid_therapy", "binary", "Long-term opioid therapy", "medications",
                patient_phrase="long-term opioid pain treatment, which needs close follow-up", absent_label="Not on long-term opioid therapy"),
    # ---------------------------------------------------------------- SDOH
    FeatureSpec("housing_instability", "binary", "Housing instability or homelessness", "sdoh",
                patient_phrase="not having stable, secure housing", absent_label="Stable housing"),
    FeatureSpec("food_insecurity", "binary", "Food insecurity", "sdoh",
                patient_phrase="not always having reliable access to enough food", absent_label="Reliable access to food"),
    FeatureSpec("transportation_barrier", "binary", "Transportation barrier", "sdoh",
                patient_phrase="difficulty getting to medical appointments", absent_label="No transportation barrier"),
    FeatureSpec("social_isolation", "binary", "Social isolation", "sdoh",
                patient_phrase="having little day-to-day contact with family or friends", absent_label="Not socially isolated"),
    FeatureSpec("financial_strain", "binary", "Financial strain / cost-related non-adherence", "sdoh",
                patient_phrase="cost getting in the way of care or medicines", absent_label="No financial strain"),
    FeatureSpec("uninsured_or_medicaid", "binary", "Medicaid or uninsured", "sdoh",
                patient_phrase="insurance coverage that can limit your options", absent_label="Commercial or Medicare coverage"),
    FeatureSpec("caregiver_support", "binary", "Reliable caregiver support", "sdoh",
                patient_phrase="not having someone who can reliably help at home",
                higher_is_protective=True, absent_label="No reliable caregiver support"),
    FeatureSpec("area_deprivation_index", "numeric", "Area Deprivation Index (national percentile)", "sdoh", 1, 100,
                "percentile", "living in a neighbourhood with fewer health and support resources"),
    FeatureSpec("limited_health_literacy", "binary", "Limited health literacy", "sdoh",
                patient_phrase="medical information being hard to follow", absent_label="No health literacy concerns"),
)

FEATURE_NAMES: tuple[str, ...] = tuple(f.name for f in FEATURES)
FEATURE_BY_NAME: dict[str, FeatureSpec] = {f.name: f for f in FEATURES}

GROUP_LABELS = {
    "demographics": "Demographics",
    "utilization": "Prior utilization",
    "clinical": "Medical conditions",
    "medications": "Medications",
    "sdoh": "Social drivers of health",
}

# --------------------------------------------------------------------------
# Risk tiers
# --------------------------------------------------------------------------
# Cut points are set on the validation set so that the top tier captures roughly
# the caseload a care-management team can actually absorb (~5% of the panel).
# They are overwritten by train.py and stored in the model bundle.
DEFAULT_TIER_CUTS = {"rising": 0.12, "high": 0.28, "very_high": 0.50}

TIER_META = {
    "low": {
        "label": "Low",
        "action": "Routine care. Re-score at the next scheduled visit.",
        "patient_line": "Your care team does not see anything right now that suggests you are heading for a hospital stay.",
    },
    "rising": {
        "label": "Rising",
        "action": "Watchlist. Confirm primary care follow-up and screen for unmet social needs.",
        "patient_line": "A few things in your history are worth keeping an eye on so that small problems do not turn into big ones.",
    },
    "high": {
        "label": "High",
        "action": "Refer to care management. Outreach within 7 days; medication review and SDOH assessment.",
        "patient_line": "Your care team would like to stay in closer contact with you over the next few months.",
    },
    "very_high": {
        "label": "Very high",
        "action": "Intensive case management. Outreach within 48 hours; assign a care navigator and build a shared care plan.",
        "patient_line": "Your care team wants to work closely with you right away to keep you well and out of hospital.",
    },
}

TIER_ORDER = ("low", "rising", "high", "very_high")
