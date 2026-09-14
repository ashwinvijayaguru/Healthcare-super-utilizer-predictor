"""Feature construction shared by training and serving.

Both the trainer and the API call ``build_feature_frame``. That is the whole
defence against training/serving skew: there is exactly one implementation, and
it is deterministic and row-independent (no fitted state, no aggregates over the
batch), so a single-row request produces the identical vector it would have
produced inside the training set.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml.config import FEATURE_BY_NAME, FEATURE_NAMES

SDOH_RISK_FLAGS = (
    "housing_instability",
    "food_insecurity",
    "transportation_barrier",
    "social_isolation",
    "financial_strain",
    "uninsured_or_medicaid",
    "limited_health_literacy",
)

HIGH_ACUITY_CONDITIONS = (
    "heart_failure",
    "copd",
    "diabetes_complicated",
    "ckd_stage4_plus",
    "cirrhosis",
    "cancer_active",
)

BEHAVIORAL_CONDITIONS = ("serious_mental_illness", "substance_use_disorder")

DERIVED_NAMES: tuple[str, ...] = (
    "acute_utilization_index",
    "recent_discharge_30d",
    "recent_discharge_90d",
    "sdoh_burden_count",
    "sdoh_burden_severe",
    "adherence_gap",
    "medication_complexity",
    "unmanaged_condition_burden",
    "high_acuity_condition_count",
    "behavioral_health_burden",
    "care_continuity_score",
    "ed_to_outpatient_ratio",
    "frail_elderly",
)

MODEL_FEATURE_NAMES: tuple[str, ...] = FEATURE_NAMES + DERIVED_NAMES


def coerce_contract(records: pd.DataFrame | dict | list[dict]) -> pd.DataFrame:
    """Validate/clip incoming rows against the feature contract in ml.config."""
    if isinstance(records, dict):
        frame = pd.DataFrame([records])
    elif isinstance(records, list):
        frame = pd.DataFrame(records)
    else:
        frame = records.copy()

    missing = [name for name in FEATURE_NAMES if name not in frame.columns]
    if missing:
        raise ValueError(f"missing required features: {missing}")

    out = pd.DataFrame(index=frame.index)
    for name in FEATURE_NAMES:
        spec = FEATURE_BY_NAME[name]
        series = pd.to_numeric(frame[name], errors="coerce")
        if spec.kind == "binary":
            series = series.fillna(0).clip(0, 1).round()
        else:
            lo = spec.minimum if spec.minimum is not None else -np.inf
            hi = spec.maximum if spec.maximum is not None else np.inf
            # Median imputation is deliberately avoided: a missing utilization
            # count is materially different from an average one, and the
            # boundary value keeps the behaviour reproducible at serving time.
            series = series.fillna(lo if spec.higher_is_protective is False else hi).clip(lo, hi)
        out[name] = series.astype(float)
    return out


def build_feature_frame(records: pd.DataFrame | dict | list[dict]) -> pd.DataFrame:
    """Return the full model matrix (contract features + derived features)."""
    df = coerce_contract(records)

    df["acute_utilization_index"] = df["ed_visits_12mo"] + 2.0 * df["inpatient_admits_12mo"]
    df["recent_discharge_30d"] = (df["days_since_last_discharge"] <= 30).astype(float)
    df["recent_discharge_90d"] = (df["days_since_last_discharge"] <= 90).astype(float)

    sdoh = df[list(SDOH_RISK_FLAGS)].sum(axis=1) + (1.0 - df["caregiver_support"])
    df["sdoh_burden_count"] = sdoh
    df["sdoh_burden_severe"] = (sdoh >= 3).astype(float)

    df["adherence_gap"] = 1.0 - df["medication_adherence_pdc"]
    df["medication_complexity"] = df["medication_count"] * df["adherence_gap"] + df["high_risk_med_count"]
    df["unmanaged_condition_burden"] = df["chronic_condition_count"] * (1.0 - df["has_primary_care"])
    df["high_acuity_condition_count"] = df[list(HIGH_ACUITY_CONDITIONS)].sum(axis=1)
    df["behavioral_health_burden"] = df[list(BEHAVIORAL_CONDITIONS)].sum(axis=1) + df["opioid_therapy"]

    df["care_continuity_score"] = (
        df["has_primary_care"] * 2.0
        + np.log1p(df["outpatient_visits_12mo"])
        - np.log1p(df["missed_appointments_12mo"])
    )
    df["ed_to_outpatient_ratio"] = df["ed_visits_12mo"] / (df["outpatient_visits_12mo"] + 1.0)
    df["frail_elderly"] = (
        (df["age"] >= 75)
        & ((df["mobility_impairment"] > 0) | (df["cognitive_impairment"] > 0) | (df["falls_12mo"] > 0))
    ).astype(float)

    return df[list(MODEL_FEATURE_NAMES)]
