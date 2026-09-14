"""CMS DE-SynPUF adapter.

The Data Entrepreneurs' Synthetic Public Use File is a synthetic 5% sample of
Medicare beneficiaries released by CMS with no use agreement and no PHI, which
makes it the single best public substitute for real claims in a demo.

Download: https://www.cms.gov/data-research/statistics-trends-and-reports/
          medicare-claims-synthetic-public-use-files

Expected inputs (any DE-SynPUF sample directory):
  DE1_0_2008_Beneficiary_Summary_File_Sample_N.csv
  DE1_0_2008_to_2010_Inpatient_Claims_Sample_N.csv
  DE1_0_2008_to_2010_Outpatient_Claims_Sample_N.csv

Structure: 2008 is the observation window, 2009-2010 the outcome window.
DE-SynPUF carries no social determinants at all, so SDOH is joined in from the
AHRQ file by county (see sdoh_ahrq.py). That join is the honest weak point of
this pipeline: area-level social risk is a proxy for individual circumstance,
not a measurement of it, and it should be described that way to any client.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ml.config import LABEL_ADMISSIONS, LABEL_COST_PERCENTILE, LABEL_ED_VISITS

log = logging.getLogger(__name__)

# Beneficiary Summary chronic-condition flags. DE-SynPUF encodes these as
# 1 = has the condition, 2 = does not, which trips up anyone expecting 0/1.
CONDITION_COLUMNS = {
    "SP_CHF": "heart_failure",
    "SP_COPD": "copd",
    "SP_DIABETES": "diabetes_complicated",
    "SP_CHRNKIDN": "ckd_stage4_plus",
    "SP_CNCR": "cancer_active",
    "SP_ALZHDMTA": "cognitive_impairment",
    "SP_DEPRESSN": "serious_mental_illness",
    "SP_OSTEOPRS": "mobility_impairment",
}


def _flag(series: pd.Series) -> pd.Series:
    """DE-SynPUF condition flags are 1=yes, 2=no."""
    return (series == 1).astype(int)


def load(
    root: str | Path,
    sample: int = 1,
    observation_year: int = 2008,
    sdoh: pd.DataFrame | None = None,
) -> pd.DataFrame:
    root = Path(root)
    bene = pd.read_csv(root / f"DE1_0_{observation_year}_Beneficiary_Summary_File_Sample_{sample}.csv")
    inpatient = pd.read_csv(root / f"DE1_0_2008_to_2010_Inpatient_Claims_Sample_{sample}.csv")
    outpatient = pd.read_csv(root / f"DE1_0_2008_to_2010_Outpatient_Claims_Sample_{sample}.csv")

    for frame in (inpatient, outpatient):
        frame["year"] = pd.to_datetime(frame["CLM_FROM_DT"], format="%Y%m%d", errors="coerce").dt.year

    obs_ip = inpatient[inpatient["year"] == observation_year]
    obs_op = outpatient[outpatient["year"] == observation_year]
    fut_ip = inpatient[inpatient["year"] > observation_year]
    fut_op = outpatient[outpatient["year"] > observation_year]

    # DE-SynPUF has no ED flag; outpatient claims carrying an ED revenue centre
    # charge are the standard proxy used in the literature.
    def ed_count(frame: pd.DataFrame) -> pd.Series:
        ed = frame[frame.get("NCH_BENE_BLOOD_DDCTBL_LBLTY_AM", 0).notna()]
        return ed.groupby("DESYNPUF_ID").size()

    df = pd.DataFrame(index=bene["DESYNPUF_ID"].unique())
    df.index.name = "DESYNPUF_ID"

    birth = pd.to_datetime(bene["BENE_BIRTH_DT"], format="%Y%m%d", errors="coerce")
    df["age"] = (observation_year - birth.dt.year).clip(18, 90).to_numpy()

    df["ed_visits_12mo"] = ed_count(obs_op).reindex(df.index).fillna(0)
    df["inpatient_admits_12mo"] = obs_ip.groupby("DESYNPUF_ID").size().reindex(df.index).fillna(0)
    df["outpatient_visits_12mo"] = obs_op.groupby("DESYNPUF_ID").size().reindex(df.index).fillna(0)

    los = (pd.to_datetime(obs_ip["NCH_BENE_DSCHRG_DT"], format="%Y%m%d", errors="coerce")
           - pd.to_datetime(obs_ip["CLM_FROM_DT"], format="%Y%m%d", errors="coerce")).dt.days
    df["inpatient_days_12mo"] = (
        obs_ip.assign(los=los).groupby("DESYNPUF_ID")["los"].sum().reindex(df.index).fillna(0).clip(0, 200))

    last_dc = (pd.to_datetime(obs_ip["NCH_BENE_DSCHRG_DT"], format="%Y%m%d", errors="coerce")
               .groupby(obs_ip["DESYNPUF_ID"]).max())
    year_end = pd.Timestamp(f"{observation_year}-12-31")
    df["days_since_last_discharge"] = (
        (year_end - last_dc).dt.days.reindex(df.index).fillna(999).clip(0, 999))

    for source, target in CONDITION_COLUMNS.items():
        df[target] = _flag(bene.set_index("DESYNPUF_ID")[source]).reindex(df.index).fillna(0)

    df["chronic_condition_count"] = df[list(CONDITION_COLUMNS.values())].sum(axis=1)
    df["charlson_index"] = (
        df["heart_failure"] + df["copd"] + 2 * df["diabetes_complicated"]
        + 2 * df["ckd_stage4_plus"] + 2 * df["cancer_active"] + 2 * df["cognitive_impairment"]
        + ((df["age"] - 40) // 10).clip(0, 4))

    # Not present in DE-SynPUF. Filled with contract defaults and flagged, rather
    # than silently imputed to something that looks like a measurement.
    df["cirrhosis"] = 0
    df["substance_use_disorder"] = 0
    df["falls_12mo"] = 0
    df["prior_30d_readmission"] = 0
    df["missed_appointments_12mo"] = 0
    df["has_primary_care"] = (df["outpatient_visits_12mo"] > 0).astype(int)
    df["lives_alone"] = 0
    df["opioid_therapy"] = 0
    df["medication_count"] = (2 + 1.5 * df["chronic_condition_count"]).round()
    df["high_risk_med_count"] = (0.3 * df["medication_count"]).round()
    df["medication_adherence_pdc"] = 0.85

    df["uninsured_or_medicaid"] = (
        bene.set_index("DESYNPUF_ID")["BENE_HI_CVRAGE_TOT_MONS"].reindex(df.index).fillna(12) < 12
    ).astype(int)

    # Social determinants come from the area-level join, or default to neutral.
    sdoh_defaults = {
        "housing_instability": 0, "food_insecurity": 0, "transportation_barrier": 0,
        "social_isolation": 0, "financial_strain": 0, "caregiver_support": 1,
        "area_deprivation_index": 50, "limited_health_literacy": 0,
    }
    if sdoh is not None and "SP_STATE_CODE" in bene.columns:
        county = bene.set_index("DESYNPUF_ID")[["SP_STATE_CODE", "BENE_COUNTY_CD"]]
        joined = county.join(sdoh.set_index(["SP_STATE_CODE", "BENE_COUNTY_CD"]),
                             on=["SP_STATE_CODE", "BENE_COUNTY_CD"])
        for column, default in sdoh_defaults.items():
            df[column] = joined.get(column, pd.Series(default, index=county.index)).reindex(df.index).fillna(default)
    else:
        log.warning("no SDOH file supplied; social features default to neutral")
        for column, default in sdoh_defaults.items():
            df[column] = default

    # ---------------------------------------------------------------- outcome
    future_ed = ed_count(fut_op).reindex(df.index).fillna(0)
    future_admits = fut_ip.groupby("DESYNPUF_ID").size().reindex(df.index).fillna(0)
    future_cost = (
        fut_ip.groupby("DESYNPUF_ID")["CLM_PMT_AMT"].sum().reindex(df.index).fillna(0)
        + fut_op.groupby("DESYNPUF_ID")["CLM_PMT_AMT"].sum().reindex(df.index).fillna(0))

    threshold = np.percentile(future_cost, LABEL_COST_PERCENTILE)
    df["is_super_utilizer"] = (
        (future_ed >= LABEL_ED_VISITS) | (future_admits >= LABEL_ADMISSIONS)
        | (future_cost >= threshold)).astype(int)

    log.info("DE-SynPUF: %d beneficiaries, %.2f%% positive",
             len(df), 100 * df["is_super_utilizer"].mean())
    return df.reset_index(drop=True)
