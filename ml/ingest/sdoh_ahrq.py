"""AHRQ Social Determinants of Health Database adapter.

Area-level social measures published by AHRQ at county and census-tract level,
free and unrestricted. Used to attach social context to claims files (such as
DE-SynPUF) that carry none.

Download: https://www.ahrq.gov/sdoh/data-analytics/sdoh-data.html

Important caveat to carry into any client conversation: these are *area* rates.
Assigning a county's food insecurity rate to a person is ecological inference.
It is a reasonable prior when nothing better exists, and it is not the same as
asking the patient. The web form asks the patient directly, which is why the
individual-level answers should always override this join when available.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

# AHRQ column names vary by release year; these are the 2020+ conventions.
DEFAULT_MAPPING = {
    "ACS_PCT_HH_FOOD_STMP": "food_insecurity",
    "ACS_PCT_HU_NO_VEH": "transportation_barrier",
    "ACS_PCT_PERSON_INC_BELOW99": "financial_strain",
    "ACS_PCT_HH_ALONE": "social_isolation",
    "ACS_PCT_RENTER_HU_COST_50PCT": "housing_instability",
    "ACS_PCT_LT_HS": "limited_health_literacy",
}


def load(
    path: str | Path,
    year: int = 2020,
    mapping: dict[str, str] | None = None,
    threshold_percentile: float = 75.0,
) -> pd.DataFrame:
    mapping = mapping or DEFAULT_MAPPING
    raw = pd.read_csv(path, low_memory=False)
    raw.columns = [c.upper() for c in raw.columns]

    out = pd.DataFrame()
    for source, target in mapping.items():
        key = f"{source}_{year}" if f"{source}_{year}" in raw.columns else source
        if key not in raw.columns:
            log.warning("AHRQ column %s absent; %s set to 0", key, target)
            out[target] = 0
            continue
        series = pd.to_numeric(raw[key], errors="coerce")
        # Convert a continuous area rate into the binary the contract expects by
        # marking areas above the chosen percentile. The cut point is explicit
        # rather than hidden, because it materially changes prevalence.
        out[target] = (series >= series.quantile(threshold_percentile / 100)).astype(int)

    deprivation_inputs = [c for c in ("ACS_PCT_PERSON_INC_BELOW99", "ACS_PCT_LT_HS",
                                      "ACS_PCT_HU_NO_VEH") if c in raw.columns]
    if deprivation_inputs:
        composite = raw[deprivation_inputs].apply(pd.to_numeric, errors="coerce").mean(axis=1)
        out["area_deprivation_index"] = (composite.rank(pct=True) * 100).round().clip(1, 100)
    else:
        out["area_deprivation_index"] = 50

    out["caregiver_support"] = 1 - out.get("social_isolation", 0)

    for key in ("STATE_FIPS", "COUNTY_FIPS", "COUNTYFIPS"):
        if key in raw.columns:
            out[key] = raw[key]

    log.info("AHRQ SDOH: %d areas mapped to %d features", len(out), len(out.columns))
    return out
