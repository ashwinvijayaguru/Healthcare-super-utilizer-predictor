"""The scoring service: model bundle loading, prediction and assembly."""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import asdict
from typing import Any

import joblib
import numpy as np
import pandas as pd

from api.config import get_settings
from ml.config import (
    FOLLOW_UP_MONTHS,
    LABEL_ADMISSIONS,
    LABEL_COST_PERCENTILE,
    LABEL_ED_VISITS,
    TIER_META,
)
from ml.explain import attribute, group_attribution, red_flags
from ml.features import build_feature_frame

log = logging.getLogger(__name__)

DISCLAIMER = (
    "This is a statistical estimate to help a care team prioritise outreach. "
    "It is not a diagnosis and does not predict what will happen to any individual. "
    "Clinical judgement always takes precedence."
)

OUTCOME_DEFINITION = (
    f"At least {LABEL_ED_VISITS} emergency department visits, or at least "
    f"{LABEL_ADMISSIONS} inpatient admissions, or total cost in the top "
    f"{100 - LABEL_COST_PERCENTILE}% of the panel, within "
    f"{FOLLOW_UP_MONTHS[0]}-{FOLLOW_UP_MONTHS[1]} months."
)


class ModelNotLoadedError(RuntimeError):
    pass


class RiskScorer:
    """Thread-safe singleton wrapper around the serialised model bundle."""

    _instance: "RiskScorer | None" = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        settings = get_settings()
        if not settings.model_path.exists():
            raise ModelNotLoadedError(
                f"model bundle not found at {settings.model_path}. "
                "Run `python -m ml.generate_cohort && python -m ml.train` first."
            )
        self.bundle: dict[str, Any] = joblib.load(settings.model_path)
        self.model = self.bundle["model"]
        self.tier_cuts: dict[str, float] = self.bundle["tier_cuts"]
        self.base_rate: float = float(self.bundle.get("base_rate", 0.06))
        bounds = self.bundle.get("probability_bounds", {})
        self.prob_floor: float = float(bounds.get("floor", 0.001))
        self.prob_ceiling: float = float(bounds.get("ceiling", 0.95))

        self.metrics: dict[str, Any] = {}
        if settings.metrics_path.exists():
            self.metrics = json.loads(settings.metrics_path.read_text())

        # Score distribution used to convert a probability into a panel
        # percentile. Derived once from the recorded tier cuts so the API does
        # not need the training data at runtime.
        self._percentile_anchors = sorted(
            [(0.0, 0.0), (self.tier_cuts["rising"], 85.0),
             (self.tier_cuts["high"], 95.0), (self.tier_cuts["very_high"], 98.0), (1.0, 100.0)]
        )
        log.info("loaded %s v%s (%s)", self.bundle["model_name"], self.bundle["model_version"],
                 self.bundle.get("estimator_kind", "unknown"))

    @classmethod
    def instance(cls) -> "RiskScorer":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = RiskScorer()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Drop the cached bundle so the next call reloads from disk."""
        with cls._lock:
            cls._instance = None

    # ------------------------------------------------------------------ tiers
    def tier_for(self, probability: float) -> str:
        if probability >= self.tier_cuts["very_high"]:
            return "very_high"
        if probability >= self.tier_cuts["high"]:
            return "high"
        if probability >= self.tier_cuts["rising"]:
            return "rising"
        return "low"

    def percentile_for(self, probability: float) -> float:
        xs = [a for a, _ in self._percentile_anchors]
        ys = [b for _, b in self._percentile_anchors]
        return round(float(np.interp(probability, xs, ys)), 1)

    # ------------------------------------------------------------------ score
    def score(self, intake: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()

        frame = pd.DataFrame([intake])
        X = build_feature_frame(frame)
        raw_probability = float(self.model.predict_proba(X)[:, 1][0])
        # Bounded to what the calibration data supports (see ml.train).
        probability = min(max(raw_probability, self.prob_floor), self.prob_ceiling)

        tier = self.tier_for(probability)
        drivers, protective = attribute(self.bundle, X)
        groups = group_attribution(self.bundle, X)
        flags = red_flags(intake)

        latency_ms = int((time.perf_counter() - started) * 1000)

        return {
            "risk_score": round(probability, 4),
            "raw_score": round(raw_probability, 4),
            "risk_percent": round(probability * 100, 1),
            "risk_tier": tier,
            "risk_tier_label": TIER_META[tier]["label"],
            "tier_action": TIER_META[tier]["action"],
            "percentile": self.percentile_for(probability),
            "baseline_rate": round(self.base_rate, 4),
            "lift_vs_baseline": round(probability / self.base_rate, 2) if self.base_rate else 0.0,
            "expected_in_100": int(round(probability * 100)),
            "drivers": [asdict(d) for d in drivers],
            "protective_factors": [asdict(d) for d in protective],
            "group_attribution": groups,
            "red_flags": [asdict(f) for f in flags],
            "model_name": self.bundle["model_name"],
            "model_version": self.bundle["model_version"],
            "model_trained_at": self.bundle["trained_at"],
            "latency_ms": latency_ms,
            "disclaimer": DISCLAIMER,
        }

    # ------------------------------------------------------- clinician summary
    def clinician_summary(self, intake: dict[str, Any], result: dict[str, Any]) -> str:
        """Dense, factual, one paragraph. Written for someone who reads charts."""
        parts: list[str] = []
        parts.append(
            f"{int(intake['age'])}-year-old member, risk {result['risk_percent']:.1f}% "
            f"({result['risk_tier_label'].lower()} tier, ~{result['percentile']:.0f}th percentile of panel), "
            f"{result['lift_vs_baseline']:.1f}x the panel baseline of {result['baseline_rate']*100:.1f}%."
        )

        util = []
        if intake["ed_visits_12mo"]:
            util.append(f"{int(intake['ed_visits_12mo'])} ED visits")
        if intake["inpatient_admits_12mo"]:
            util.append(f"{int(intake['inpatient_admits_12mo'])} admissions "
                        f"({int(intake['inpatient_days_12mo'])} days)")
        if intake["prior_30d_readmission"]:
            util.append("prior 30-day readmission")
        if intake["days_since_last_discharge"] < 90:
            util.append(f"discharged {int(intake['days_since_last_discharge'])} days ago")
        parts.append("Prior 12 months: " + (", ".join(util) if util else "no acute utilisation") + ".")

        conditions = [
            label for key, label in (
                ("heart_failure", "heart failure"), ("copd", "COPD"),
                ("diabetes_complicated", "complicated diabetes"), ("ckd_stage4_plus", "CKD stage 4+"),
                ("cirrhosis", "cirrhosis"), ("cancer_active", "active cancer"),
                ("serious_mental_illness", "serious mental illness"),
                ("substance_use_disorder", "substance use disorder"),
                ("cognitive_impairment", "cognitive impairment"),
                ("mobility_impairment", "mobility impairment"),
            ) if intake.get(key)
        ]
        parts.append(
            f"{int(intake['chronic_condition_count'])} chronic conditions "
            f"(Charlson {int(intake['charlson_index'])})"
            + (f", including {', '.join(conditions)}." if conditions else ".")
        )
        parts.append(
            f"{int(intake['medication_count'])} active medications "
            f"({int(intake['high_risk_med_count'])} high-risk), PDC {intake['medication_adherence_pdc']:.2f}."
        )

        social = [
            label for key, label in (
                ("housing_instability", "housing instability"), ("food_insecurity", "food insecurity"),
                ("transportation_barrier", "transport barrier"), ("social_isolation", "social isolation"),
                ("financial_strain", "financial strain"), ("limited_health_literacy", "limited health literacy"),
            ) if intake.get(key)
        ]
        if not intake.get("caregiver_support"):
            social.append("no reliable caregiver")
        parts.append("Social drivers: " + (", ".join(social) if social else "none recorded") + ".")

        sdoh_share = result["group_attribution"].get("sdoh", 0.0)
        parts.append(f"Social factors account for {sdoh_share*100:.0f}% of the modelled risk increase.")

        criticals = [f["title"] for f in result["red_flags"] if f["severity"] == "critical"]
        if criticals:
            parts.append("Critical flags: " + "; ".join(criticals) + ".")
        parts.append(result["tier_action"])
        return " ".join(parts)
