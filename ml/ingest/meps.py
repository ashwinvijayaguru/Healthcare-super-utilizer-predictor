"""MEPS Household Component adapter.

MEPS is the only major public US dataset that carries individual-level
utilisation, spending AND social/economic circumstance on the same person, which
makes it the right source for the SDOH half of this model. It is a public use
file with no PHI and no data use agreement.

Download: https://meps.ahrq.gov/mepsweb/data_stats/download_data_files.jsp

Panel structure gives the observation/outcome split for free: each MEPS panel
follows a household for two calendar years, so year 1 supplies the features and
year 2 the outcome. Use the two-year longitudinal file for this.

Column names carry a two-digit year suffix that changes each release, so they
are built from the `year` argument here. Check the codebook for your release.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ml.config import LABEL_ADMISSIONS, LABEL_COST_PERCENTILE, LABEL_ED_VISITS

log = logging.getLogger(__name__)


def load(path: str | Path, observation_year: int = 2021, outcome_year: int = 2022) -> pd.DataFrame:
    obs, out = str(observation_year)[-2:], str(outcome_year)[-2:]
    raw = pd.read_sas(path) if str(path).endswith((".sas7bdat", ".ssp")) else pd.read_csv(path)
    raw.columns = [c.upper() for c in raw.columns]

    def column(name: str, default: float = 0.0) -> pd.Series:
        if name not in raw.columns:
            log.warning("MEPS column %s absent; defaulting to %s", name, default)
            return pd.Series(default, index=raw.index)
        return pd.to_numeric(raw[name], errors="coerce").fillna(default)

    df = pd.DataFrame(index=raw.index)
    df["age"] = column(f"AGE{obs}X", 50).clip(18, 90)

    df["ed_visits_12mo"] = column(f"ERTOT{obs}").clip(0, 60)
    df["inpatient_admits_12mo"] = column(f"IPDIS{obs}").clip(0, 30)
    df["inpatient_days_12mo"] = column(f"IPNGTD{obs}").clip(0, 200)
    df["outpatient_visits_12mo"] = (column(f"OBTOTV{obs}") + column(f"OPTOTV{obs}")).clip(0, 100)
    df["days_since_last_discharge"] = np.where(df["inpatient_admits_12mo"] > 0, 180, 999)
    df["prior_30d_readmission"] = (df["inpatient_admits_12mo"] >= 2).astype(int)
    df["missed_appointments_12mo"] = 0
    # MEPS asks whether the person has a usual source of care.
    df["has_primary_care"] = (column("HAVEUS42", 1) == 1).astype(int)

    # Priority conditions are coded 1 = yes, 2 = no in MEPS.
    yes = lambda name: (column(name, 2) == 1).astype(int)  # noqa: E731
    df["heart_failure"] = yes("CHDDX")
    df["copd"] = yes("EMPHDX")
    df["diabetes_complicated"] = yes("DIABDX_M18")
    df["cancer_active"] = yes("CANCERDX")
    df["serious_mental_illness"] = (column(f"K6SUM{obs}", 0) >= 13).astype(int)  # K6 serious distress
    df["cognitive_impairment"] = yes("COGLIM31")
    df["mobility_impairment"] = yes("WLKLIM31")
    df["ckd_stage4_plus"] = 0
    df["cirrhosis"] = 0
    df["substance_use_disorder"] = 0
    df["falls_12mo"] = 0
    df["chronic_condition_count"] = df[[
        "heart_failure", "copd", "diabetes_complicated", "cancer_active",
        "serious_mental_illness", "cognitive_impairment", "mobility_impairment"]].sum(axis=1)
    df["charlson_index"] = (df["heart_failure"] + df["copd"] + 2 * df["diabetes_complicated"]
                            + 2 * df["cancer_active"] + ((df["age"] - 40) // 10).clip(0, 4))

    df["medication_count"] = column(f"RXTOT{obs}").clip(0, 40)
    df["high_risk_med_count"] = (0.25 * df["medication_count"]).round().clip(0, 15)
    df["medication_adherence_pdc"] = np.where(column("MNHLTH31", 3) >= 4, 0.65, 0.88)
    df["opioid_therapy"] = 0

    # ------------------------------------------------------------------ SDOH
    poverty = column(f"POVCAT{obs}", 3)          # 1 = poor ... 5 = high income
    df["financial_strain"] = (poverty <= 2).astype(int)
    df["uninsured_or_medicaid"] = (column(f"INSURC{obs}", 1).isin([4, 5, 6, 7, 8])).astype(int)
    df["food_insecurity"] = (column("FDSCAT", 1) >= 3).astype(int)
    df["transportation_barrier"] = (column("DLAYCA42", 2) == 1).astype(int)
    df["housing_instability"] = 0
    df["social_isolation"] = (column("MARRY31X", 1).isin([2, 3, 4])).astype(int)
    df["lives_alone"] = (column(f"FAMSZE{obs}", 2) <= 1).astype(int)
    df["caregiver_support"] = (column(f"FAMSZE{obs}", 2) > 1).astype(int)
    df["limited_health_literacy"] = (column("EDUCYR", 12) < 12).astype(int)
    # MEPS has no ADI; percentile-rank poverty category as a stand-in and label
    # it as such wherever it is reported.
    df["area_deprivation_index"] = ((6 - poverty) / 5 * 100).clip(1, 100).round()

    # --------------------------------------------------------------- outcome
    future_ed = column(f"ERTOT{out}")
    future_admits = column(f"IPDIS{out}")
    future_cost = column(f"TOTEXP{out}")
    threshold = np.percentile(future_cost, LABEL_COST_PERCENTILE)
    df["is_super_utilizer"] = (
        (future_ed >= LABEL_ED_VISITS) | (future_admits >= LABEL_ADMISSIONS)
        | (future_cost >= threshold)).astype(int)

    df = df[df["age"] >= 18].reset_index(drop=True)
    log.info("MEPS: %d respondents, %.2f%% positive", len(df), 100 * df["is_super_utilizer"].mean())
    return df
