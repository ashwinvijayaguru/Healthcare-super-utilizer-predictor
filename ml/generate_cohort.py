"""Build a PHI-free training cohort.

The demo cohort is *generated*, not copied from any patient record, so nothing in
this repository is protected health information. The generator is not arbitrary:
marginal prevalences, utilization distributions and effect sizes are calibrated
to published, publicly available population statistics (MEPS, HCUP NEDS/NRD,
CDC BRFSS, AHRQ SDOH Database, USRDS). See docs/DATASETS.md for the full
source table and the calibration targets.

The same feature contract is produced by the real-data adapters in ml/ingest/,
so swapping the demo cohort for CMS DE-SynPUF or MIMIC-IV is a one-line change
in train.py.
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

from ml.config import (
    DATA_DIR,
    FEATURE_NAMES,
    LABEL_ADMISSIONS,
    LABEL_COST_PERCENTILE,
    LABEL_ED_VISITS,
)

log = logging.getLogger(__name__)

# Calibration targets (see docs/DATASETS.md for citations)
TARGET_PREVALENCE = 0.11  # share of adult panel meeting the super-utilizer definition
TARGET_TOP5_COST_SHARE = 0.50  # MEPS: top 5% of spenders hold ~half of all spend


def _logistic(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _bern(rng: np.random.Generator, p: np.ndarray) -> np.ndarray:
    return (rng.random(p.shape) < np.clip(p, 0, 1)).astype(int)


def build_cohort(n: int = 60_000, seed: int = 20241118) -> pd.DataFrame:
    """Generate ``n`` synthetic adult members with a 6-12 month outcome window."""
    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ latent
    # `burden` is unobserved clinical frailty; `deprivation` is unobserved social
    # adversity. Both drive observed features AND the outcome, which is what
    # gives the dataset a learnable but non-trivial signal structure.
    burden = rng.normal(0, 1, n)
    deprivation = rng.normal(0, 1, n)
    # Social and clinical adversity are correlated in every real population.
    deprivation = 0.35 * burden + np.sqrt(1 - 0.35**2) * deprivation

    # ------------------------------------------------------------- demographics
    # Skewed adult age distribution, weighted toward the Medicare/Medicaid mix a
    # care-management program actually sees.
    age = np.clip(rng.gamma(shape=5.5, scale=6.2, size=n) + 20, 18, 104)
    age_z = (age - age.mean()) / age.std()
    burden = burden + 0.45 * age_z  # older members carry more clinical burden

    adi = np.clip(50 + 18 * deprivation + rng.normal(0, 8, n), 1, 100)
    adi_z = (adi - 50) / 25

    # -------------------------------------------------------------------- SDOH
    housing_instability = _bern(rng, _logistic(-3.3 + 0.95 * deprivation + 0.35 * adi_z))
    food_insecurity = _bern(rng, _logistic(-2.1 + 1.05 * deprivation + 0.45 * adi_z))
    transportation_barrier = _bern(rng, _logistic(-2.0 + 0.85 * deprivation + 0.30 * adi_z + 0.20 * age_z))
    social_isolation = _bern(rng, _logistic(-2.2 + 0.60 * deprivation + 0.40 * age_z))
    financial_strain = _bern(rng, _logistic(-1.7 + 1.00 * deprivation + 0.35 * adi_z))
    uninsured_or_medicaid = _bern(rng, _logistic(-1.4 + 0.90 * deprivation + 0.40 * adi_z - 0.25 * age_z))
    caregiver_support = _bern(rng, _logistic(1.0 - 0.70 * deprivation - 0.30 * age_z))
    limited_health_literacy = _bern(rng, _logistic(-1.9 + 0.75 * deprivation + 0.30 * age_z))
    lives_alone = _bern(rng, _logistic(-1.5 + 0.45 * deprivation + 0.55 * age_z))

    # ---------------------------------------------------------------- clinical
    heart_failure = _bern(rng, _logistic(-4.2 + 1.15 * burden + 0.75 * age_z))
    copd = _bern(rng, _logistic(-3.9 + 1.05 * burden + 0.45 * age_z + 0.30 * deprivation))
    diabetes_complicated = _bern(rng, _logistic(-3.6 + 1.00 * burden + 0.35 * age_z + 0.25 * deprivation))
    ckd_stage4_plus = _bern(rng, _logistic(-4.6 + 1.10 * burden + 0.60 * age_z + 0.55 * diabetes_complicated))
    cirrhosis = _bern(rng, _logistic(-5.0 + 0.85 * burden + 0.35 * deprivation))
    cancer_active = _bern(rng, _logistic(-4.3 + 0.70 * burden + 0.50 * age_z))
    serious_mental_illness = _bern(rng, _logistic(-3.2 + 0.60 * burden + 0.75 * deprivation))
    substance_use_disorder = _bern(rng, _logistic(-3.4 + 0.45 * burden + 0.95 * deprivation - 0.30 * age_z))
    cognitive_impairment = _bern(rng, _logistic(-5.2 + 0.65 * burden + 1.35 * age_z))
    mobility_impairment = _bern(rng, _logistic(-3.4 + 0.85 * burden + 0.85 * age_z))

    condition_matrix = np.vstack([
        heart_failure, copd, diabetes_complicated, ckd_stage4_plus, cirrhosis,
        cancer_active, serious_mental_illness, substance_use_disorder,
        cognitive_impairment, mobility_impairment,
    ])
    core_conditions = condition_matrix.sum(axis=0)
    # Members also carry lower-acuity chronic conditions (HTN, OA, asthma, ...).
    other_conditions = rng.poisson(np.clip(1.2 + 0.55 * burden + 0.40 * age_z, 0.1, None))
    chronic_condition_count = core_conditions + other_conditions

    # Charlson weights, simplified to the conditions we carry in the schema.
    charlson_index = (
        heart_failure * 1 + copd * 1 + diabetes_complicated * 2 + ckd_stage4_plus * 2
        + cirrhosis * 3 + cancer_active * 2 + cognitive_impairment * 2
        + (age >= 50).astype(int) * ((age - 40) // 10).clip(0, 4).astype(int)
    )

    falls_12mo = rng.poisson(np.clip(0.10 + 0.55 * mobility_impairment + 0.45 * cognitive_impairment
                                     + 0.25 * np.maximum(age_z, 0), 0.02, None))

    # ------------------------------------------------------------- medications
    medication_count = np.clip(
        rng.poisson(np.clip(1.5 + 1.55 * chronic_condition_count + 0.9 * np.maximum(burden, 0), 0.5, None)),
        0, 40)
    high_risk_med_count = np.clip(
        rng.binomial(np.maximum(medication_count, 1),
                     np.clip(0.10 + 0.08 * heart_failure + 0.09 * diabetes_complicated + 0.06 * ckd_stage4_plus, 0, 0.9)),
        0, 15)
    opioid_therapy = _bern(rng, _logistic(-3.0 + 0.55 * burden + 0.45 * deprivation + 0.55 * substance_use_disorder))
    adherence_logit = (1.35 - 0.55 * deprivation - 0.45 * financial_strain - 0.35 * transportation_barrier
                       - 0.30 * limited_health_literacy - 0.25 * substance_use_disorder
                       - 0.03 * medication_count + rng.normal(0, 0.5, n))
    medication_adherence_pdc = np.round(np.clip(_logistic(adherence_logit), 0.05, 1.0), 3)

    # ----------------------------------------------------- baseline utilization
    ed_rate = np.exp(-1.65 + 0.62 * burden + 0.42 * deprivation + 0.28 * substance_use_disorder
                     + 0.24 * serious_mental_illness + 0.22 * copd + 0.20 * heart_failure
                     + 0.18 * falls_12mo - 0.30 * medication_adherence_pdc)
    ed_visits_12mo = rng.poisson(ed_rate * rng.gamma(3.0, 1 / 3.0, n))  # over-dispersed

    admit_rate = np.exp(-2.55 + 0.72 * burden + 0.26 * deprivation + 0.45 * heart_failure
                        + 0.35 * copd + 0.30 * ckd_stage4_plus + 0.30 * cancer_active
                        + 0.10 * ed_visits_12mo)
    inpatient_admits_12mo = np.clip(rng.poisson(admit_rate * rng.gamma(2.5, 1 / 2.5, n)), 0, 30)

    los_per_admit = np.clip(rng.gamma(2.2, 2.1, n), 1, 45)
    inpatient_days_12mo = np.round(inpatient_admits_12mo * los_per_admit).astype(int).clip(0, 200)

    has_recent_admit = inpatient_admits_12mo > 0
    days_since_last_discharge = np.where(
        has_recent_admit,
        np.clip(rng.gamma(2.0, 55, n) / np.maximum(inpatient_admits_12mo, 1), 1, 365),
        999,
    ).round().astype(int)

    prior_30d_readmission = _bern(
        rng, np.where(inpatient_admits_12mo >= 2,
                      _logistic(-0.6 + 0.35 * burden + 0.35 * deprivation),
                      _logistic(-4.0 + 0.30 * burden)))

    outpatient_visits_12mo = np.clip(
        rng.poisson(np.clip(2.0 + 1.35 * chronic_condition_count + 1.2 * np.maximum(burden, 0)
                            - 1.1 * transportation_barrier, 0.3, None)), 0, 100)

    has_primary_care = _bern(rng, _logistic(1.55 - 0.65 * deprivation - 0.45 * uninsured_or_medicaid
                                            - 0.35 * housing_instability))
    missed_rate = np.exp(-1.5 + 0.55 * transportation_barrier + 0.45 * deprivation
                         + 0.40 * substance_use_disorder + 0.30 * cognitive_impairment
                         + 0.02 * outpatient_visits_12mo - 0.60 * medication_adherence_pdc)
    missed_appointments_12mo = np.clip(rng.poisson(missed_rate), 0, 40)

    # ------------------------------------------------- future (outcome) window
    # Future utilization depends on the same latent drivers plus prior use, with a
    # fresh shock term. The shock is what produces regression to the mean and keeps
    # the achievable AUROC in the 0.80-0.86 band reported in the literature rather
    # than an unrealistic 0.99.
    shock = rng.normal(0, 1, n)

    future_ed_rate = np.exp(
        -1.05
        + 0.145 * np.minimum(ed_visits_12mo, 15)
        + 0.115 * np.minimum(inpatient_admits_12mo, 10)
        + 0.42 * burden
        + 0.30 * deprivation
        + 0.30 * housing_instability
        + 0.24 * substance_use_disorder
        + 0.20 * serious_mental_illness
        + 0.18 * food_insecurity
        + 0.16 * transportation_barrier
        + 0.14 * social_isolation
        + 0.10 * np.minimum(missed_appointments_12mo, 12)
        + 0.03 * np.minimum(high_risk_med_count, 10)
        - 0.55 * medication_adherence_pdc
        - 0.22 * has_primary_care
        - 0.18 * caregiver_support
        + 0.45 * shock
    )
    future_ed_visits = rng.poisson(future_ed_rate * rng.gamma(3.0, 1 / 3.0, n))

    future_admit_rate = np.exp(
        -2.05
        + 0.155 * np.minimum(inpatient_admits_12mo, 10)
        + 0.075 * np.minimum(ed_visits_12mo, 15)
        + 0.90 * prior_30d_readmission
        + 0.55 * burden
        + 0.22 * deprivation
        + 0.42 * heart_failure
        + 0.34 * copd
        + 0.30 * ckd_stage4_plus
        + 0.28 * cancer_active
        + 0.22 * cirrhosis
        + 0.20 * housing_instability
        + 0.16 * np.minimum(falls_12mo, 5)
        + 0.02 * np.minimum(medication_count, 25)
        - 0.40 * medication_adherence_pdc
        - 0.16 * caregiver_support
        + 0.40 * shock
    )
    future_admissions = rng.poisson(future_admit_rate * rng.gamma(2.5, 1 / 2.5, n))

    # Cost: lognormal core spend plus event-driven marginal cost, in line with the
    # MEPS concentration curve (top 5% of spenders hold roughly half of spend).
    base_cost = np.exp(7.4 + 0.55 * burden + 0.25 * np.log1p(chronic_condition_count)
                       + 0.35 * rng.normal(0, 1, n))
    future_cost = (base_cost
                   + future_ed_visits * rng.normal(1_400, 320, n).clip(400)
                   + future_admissions * rng.normal(14_800, 5_200, n).clip(3_000)
                   + medication_count * rng.normal(280, 90, n).clip(30))
    future_cost = np.round(future_cost, 2)

    cost_threshold = np.percentile(future_cost, LABEL_COST_PERCENTILE)
    label = (
        (future_ed_visits >= LABEL_ED_VISITS)
        | (future_admissions >= LABEL_ADMISSIONS)
        | (future_cost >= cost_threshold)
    ).astype(int)

    df = pd.DataFrame({
        "age": np.round(age, 1),
        "lives_alone": lives_alone,
        "ed_visits_12mo": ed_visits_12mo,
        "inpatient_admits_12mo": inpatient_admits_12mo,
        "inpatient_days_12mo": inpatient_days_12mo,
        "days_since_last_discharge": days_since_last_discharge,
        "prior_30d_readmission": prior_30d_readmission,
        "outpatient_visits_12mo": outpatient_visits_12mo,
        "missed_appointments_12mo": missed_appointments_12mo,
        "has_primary_care": has_primary_care,
        "chronic_condition_count": chronic_condition_count,
        "charlson_index": charlson_index,
        "heart_failure": heart_failure,
        "copd": copd,
        "diabetes_complicated": diabetes_complicated,
        "ckd_stage4_plus": ckd_stage4_plus,
        "cirrhosis": cirrhosis,
        "cancer_active": cancer_active,
        "serious_mental_illness": serious_mental_illness,
        "substance_use_disorder": substance_use_disorder,
        "cognitive_impairment": cognitive_impairment,
        "mobility_impairment": mobility_impairment,
        "falls_12mo": falls_12mo,
        "medication_count": medication_count,
        "high_risk_med_count": high_risk_med_count,
        "medication_adherence_pdc": medication_adherence_pdc,
        "opioid_therapy": opioid_therapy,
        "housing_instability": housing_instability,
        "food_insecurity": food_insecurity,
        "transportation_barrier": transportation_barrier,
        "social_isolation": social_isolation,
        "financial_strain": financial_strain,
        "uninsured_or_medicaid": uninsured_or_medicaid,
        "caregiver_support": caregiver_support,
        "area_deprivation_index": np.round(adi).astype(int),
        "limited_health_literacy": limited_health_literacy,
        # outcome window columns (kept for evaluation and reporting, not features)
        "future_ed_visits": future_ed_visits,
        "future_admissions": future_admissions,
        "future_cost": future_cost,
        "is_super_utilizer": label,
    })

    missing = set(FEATURE_NAMES) - set(df.columns)
    if missing:
        raise RuntimeError(f"generator is out of sync with the feature contract: {sorted(missing)}")
    return df


def calibration_report(df: pd.DataFrame) -> dict[str, float]:
    """Compare the generated cohort against published population benchmarks."""
    cost = df["future_cost"].to_numpy()
    top5 = cost >= np.percentile(cost, 95)
    return {
        "n": float(len(df)),
        "super_utilizer_rate": float(df["is_super_utilizer"].mean()),
        "top5_cost_share": float(cost[top5].sum() / cost.sum()),
        "mean_age": float(df["age"].mean()),
        "any_ed_visit_rate": float((df["ed_visits_12mo"] > 0).mean()),
        "frequent_ed_rate": float((df["ed_visits_12mo"] >= 4).mean()),
        "polypharmacy_rate": float((df["medication_count"] >= 10).mean()),
        "multimorbidity_rate": float((df["chronic_condition_count"] >= 2).mean()),
        "housing_instability_rate": float(df["housing_instability"].mean()),
        "food_insecurity_rate": float(df["food_insecurity"].mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the PHI-free demo cohort.")
    parser.add_argument("--n", type=int, default=60_000)
    parser.add_argument("--seed", type=int, default=20241118)
    parser.add_argument("--out", type=str, default=str(DATA_DIR / "cohort.parquet"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = build_cohort(args.n, args.seed)
    df.to_parquet(args.out, index=False)

    log.info("wrote %s rows to %s", len(df), args.out)
    for key, value in calibration_report(df).items():
        log.info("  %-26s %.4f", key, value)


if __name__ == "__main__":
    main()
