"""Per-patient attribution and clinical red flags.

Two independent layers, deliberately kept separate:

1. ``attribute`` - what the *model* used. Exact Shapley values: closed form for
   the linear pipeline, TreeSHAP for the boosted model. Never a post-hoc
   rationalisation invented from feature values.
2. ``red_flags`` - what a *clinician* would want escalated regardless of what the
   model said. These are deterministic rules. A model can be wrong or drift; a
   patient on a blood thinner with a 0.4 adherence ratio still needs a call.

Keeping them separate means a low model score can never suppress a safety signal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from ml.config import FEATURE_BY_NAME, GROUP_LABELS
from ml.features import MODEL_FEATURE_NAMES

# Derived features are real model inputs but meaningless to a care manager.
#
# A derived feature is folded onto a contract feature ONLY when the two point the
# same way clinically. Folding across a sign flip corrupts the explanation: e.g.
# care_continuity_score is dominated by outpatient volume, so folding it onto
# has_primary_care renders "has a primary care provider" as a risk driver, which
# is both wrong and exactly the kind of output that destroys clinician trust.
# Everything that does not align cleanly keeps its own display identity below.
DERIVED_PARENT = {
    "acute_utilization_index": "ed_visits_12mo",
    "ed_to_outpatient_ratio": "ed_visits_12mo",
    "medication_complexity": "medication_count",
    "unmanaged_condition_burden": "chronic_condition_count",
    "high_acuity_condition_count": "chronic_condition_count",
    "frail_elderly": "age",
}


@dataclass(frozen=True)
class DisplaySpec:
    label: str
    group: str
    patient_phrase: str
    kind: str = "numeric"
    unit: str = ""
    absent_label: str = ""
    # Composite indices have no meaningful units, so their raw value is hidden.
    show_value: bool = False


# Derived features that are shown in their own right.
DERIVED_DISPLAY: dict[str, DisplaySpec] = {
    "recent_discharge_30d": DisplaySpec(
        "Discharged within 30 days", "utilization",
        "having left hospital very recently, when problems most often come back",
        kind="binary", absent_label="No discharge in the past 30 days"),
    "recent_discharge_90d": DisplaySpec(
        "Discharged within 90 days", "utilization",
        "having been in hospital in the last few months",
        kind="binary", absent_label="No discharge in the past 90 days"),
    "sdoh_burden_count": DisplaySpec(
        "Cumulative social risk", "sdoh",
        "several things outside of medical care stacking up at once", show_value=True,
        unit="needs"),
    "sdoh_burden_severe": DisplaySpec(
        "Three or more unmet social needs", "sdoh",
        "several unmet everyday needs at the same time",
        kind="binary", absent_label="Fewer than three unmet social needs"),
    "adherence_gap": DisplaySpec(
        "Gap in medication adherence", "medications",
        "gaps in taking medicines as prescribed"),
    "behavioral_health_burden": DisplaySpec(
        "Behavioural health burden", "clinical",
        "mental health or substance use needs that deserve support"),
    "care_continuity_score": DisplaySpec(
        "Continuity of outpatient care", "utilization",
        "how consistently you are able to attend regular appointments"),
}


def _display_for(name: str) -> tuple[str, DisplaySpec | Any] | None:
    """Resolve a model feature to the name and metadata shown to a human."""
    if name in DERIVED_DISPLAY:
        return name, DERIVED_DISPLAY[name]
    parent = DERIVED_PARENT.get(name, name)
    spec = FEATURE_BY_NAME.get(parent)
    return (parent, spec) if spec else None

Severity = Literal["critical", "warning", "advisory"]


def state_label(spec: Any, value: float) -> str:
    """Render a feature as the state the patient is actually in.

    Without this, a binary feature at 0 is shown under its affirmative name and a
    care manager reads "Has an established primary care provider - increases
    risk", which is the opposite of the truth.
    """
    label = spec.label
    kind = getattr(spec, "kind", "numeric")
    if kind == "binary":
        if value >= 0.5:
            return label
        return getattr(spec, "absent_label", "") or f"Not recorded: {label}"
    if not getattr(spec, "show_value", True):
        return label
    unit = getattr(spec, "unit", "")
    rendered = f"{value:.2f}" if unit == "ratio" else str(int(round(value)))
    suffix = f" {unit}" if unit and unit not in {"ratio", "percentile"} else ""
    return f"{label}: {rendered}{suffix}"


@dataclass
class Driver:
    feature: str
    label: str
    state: str
    group: str
    group_label: str
    value: float
    contribution: float
    direction: Literal["increases", "decreases"]
    patient_phrase: str


@dataclass
class RedFlag:
    code: str
    severity: Severity
    title: str
    detail: str
    patient_detail: str
    suggested_action: str
    evidence: dict[str, Any] = field(default_factory=dict)


def _linear_shap(bundle: dict, X: pd.DataFrame) -> np.ndarray:
    """Exact Shapley values for a StandardScaler -> LogisticRegression pipeline.

    For a linear model on standardised inputs the Shapley value of feature j is
    simply coef_j * (x_j - mean_j) / scale_j, i.e. the coefficient times the
    z-score. No sampling, no approximation.
    """
    pipeline = bundle["baseline_pipeline"]
    scaler = pipeline.named_steps["scale"]
    clf = pipeline.named_steps["clf"]
    z = (X[list(MODEL_FEATURE_NAMES)].to_numpy() - scaler.mean_) / scaler.scale_
    return z * clf.coef_[0]


def _tree_shap(bundle: dict, X: pd.DataFrame) -> np.ndarray:
    import shap

    explainer = shap.TreeExplainer(bundle["booster"])
    values = explainer.shap_values(X[list(MODEL_FEATURE_NAMES)])
    if isinstance(values, list):  # older shap returns one array per class
        values = values[1]
    return np.asarray(values)


def contributions(bundle: dict, X: pd.DataFrame) -> np.ndarray:
    """Signed contribution of every model feature, in log-odds space."""
    if bundle.get("estimator_kind") == "lightgbm":
        return _tree_shap(bundle, X)
    return _linear_shap(bundle, X)


# Contributions below this are numerically real but clinically meaningless.
# Showing them fills the panel with noise like "no cognitive impairment: +0.002".
MIN_DISPLAY_CONTRIBUTION = 0.05


def attribute(bundle: dict, X: pd.DataFrame, top_n: int = 6) -> tuple[list[Driver], list[Driver]]:
    """Return (risk drivers, protective factors) for a single-row frame."""
    if len(X) != 1:
        raise ValueError("attribute() expects exactly one row")

    raw = contributions(bundle, X)[0]
    folded: dict[str, float] = {}
    for name, value in zip(MODEL_FEATURE_NAMES, raw):
        resolved = _display_for(name)
        if resolved is None:
            continue
        key, _ = resolved
        folded[key] = folded.get(key, 0.0) + float(value)

    drivers: list[Driver] = []
    protective: list[Driver] = []
    for name, contribution in folded.items():
        resolved = _display_for(name)
        if resolved is None or abs(contribution) < MIN_DISPLAY_CONTRIBUTION:
            continue
        _, spec = resolved
        value = float(X.iloc[0][name]) if name in X.columns else float("nan")
        driver = Driver(
            feature=name,
            label=spec.label,
            state=state_label(spec, value),
            group=spec.group,
            group_label=GROUP_LABELS[spec.group],
            value=value,
            contribution=round(float(contribution), 4),
            direction="increases" if contribution > 0 else "decreases",
            patient_phrase=spec.patient_phrase,
        )
        (drivers if contribution > 0 else protective).append(driver)

    drivers.sort(key=lambda d: -d.contribution)
    protective.sort(key=lambda d: d.contribution)
    return drivers[:top_n], protective[:top_n]


def group_attribution(bundle: dict, X: pd.DataFrame) -> dict[str, float]:
    """Share of total upward push coming from each feature group.

    This is the number that makes the product's case: it shows a care manager how
    much of the risk is clinical versus social, which decides who owns the case.
    """
    raw = contributions(bundle, X)[0]
    totals: dict[str, float] = {key: 0.0 for key in GROUP_LABELS}
    for name, value in zip(MODEL_FEATURE_NAMES, raw):
        resolved = _display_for(name)
        if resolved and value > 0:
            totals[resolved[1].group] += float(value)
    total = sum(totals.values())
    if total <= 0:
        return {key: 0.0 for key in totals}
    return {key: round(value / total, 4) for key, value in totals.items()}


# ---------------------------------------------------------------------------
# Clinical red flags (model-independent)
# ---------------------------------------------------------------------------

def red_flags(p: dict[str, Any]) -> list[RedFlag]:
    """Deterministic clinical rules evaluated on the raw intake record."""
    flags: list[RedFlag] = []

    def add(**kwargs: Any) -> None:
        flags.append(RedFlag(**kwargs))

    if p["ed_visits_12mo"] >= 4:
        add(code="FREQUENT_ED", severity="critical",
            title="Frequent emergency department use",
            detail=f"{int(p['ed_visits_12mo'])} ED visits in the past 12 months meets the frequent-utilizer threshold.",
            patient_detail="You have needed the emergency room several times this year. That is usually a sign that something can be managed better before it becomes urgent.",
            suggested_action="Review what drove each visit and put a same-day access plan in place with primary care.",
            evidence={"ed_visits_12mo": p["ed_visits_12mo"]})

    if p["prior_30d_readmission"] == 1 or p["days_since_last_discharge"] <= 30:
        add(code="POST_DISCHARGE_WINDOW", severity="critical",
            title="Recent discharge or 30-day readmission history",
            detail="The member is inside, or has previously failed, the 30-day post-discharge window.",
            patient_detail="You have been in hospital recently. The first few weeks after going home are when people most often run into trouble.",
            suggested_action="Complete a transitional care management call within 48 hours and reconcile medications.",
            evidence={"days_since_last_discharge": p["days_since_last_discharge"],
                      "prior_30d_readmission": p["prior_30d_readmission"]})

    if p["medication_adherence_pdc"] < 0.6 and p["high_risk_med_count"] >= 1:
        add(code="ADHERENCE_HIGH_RISK_MEDS", severity="critical",
            title="Poor adherence on high-risk medications",
            detail=(f"PDC {p['medication_adherence_pdc']:.2f} with {int(p['high_risk_med_count'])} "
                    "high-risk medication(s) such as anticoagulants, insulin or antiarrhythmics."),
            patient_detail="Some of your medicines need to be taken consistently to be safe, and there seem to be gaps.",
            suggested_action="Pharmacist-led medication review; consider 90-day fills, blister packs or delivery.",
            evidence={"pdc": p["medication_adherence_pdc"], "high_risk_meds": p["high_risk_med_count"]})

    if p["medication_count"] >= 15:
        add(code="SEVERE_POLYPHARMACY", severity="warning",
            title="Severe polypharmacy",
            detail=f"{int(p['medication_count'])} active medications materially raises interaction and error risk.",
            patient_detail="You are taking a lot of different medicines, which is hard to keep track of and can cause side effects.",
            suggested_action="Comprehensive medication review and deprescribing assessment.",
            evidence={"medication_count": p["medication_count"]})

    if p["opioid_therapy"] == 1 and p["substance_use_disorder"] == 1:
        add(code="OPIOID_WITH_SUD", severity="critical",
            title="Long-term opioid therapy with substance use disorder",
            detail="Concurrent long-term opioid therapy and a documented substance use disorder.",
            patient_detail="Your pain treatment needs careful, supportive follow-up so it stays safe for you.",
            suggested_action="Care-plan review with pain management and behavioural health; confirm naloxone access.",
            evidence={})

    if p["housing_instability"] == 1 and (p["heart_failure"] == 1 or p["copd"] == 1 or p["ckd_stage4_plus"] == 1):
        add(code="UNSTABLE_HOUSING_WITH_ACUTE_CONDITION", severity="critical",
            title="Unstable housing with a condition needing controlled conditions",
            detail="Housing instability alongside a condition that depends on refrigeration, electricity, rest or dialysis access.",
            patient_detail="Managing your condition is much harder without stable housing, and that is something your care team can help with.",
            suggested_action="Immediate housing navigation referral; check medication storage and equipment power needs.",
            evidence={})

    social_count = sum(int(p[k]) for k in
                       ("housing_instability", "food_insecurity", "transportation_barrier",
                        "social_isolation", "financial_strain", "limited_health_literacy"))
    if social_count >= 3:
        add(code="COMPOUND_SOCIAL_RISK", severity="warning",
            title="Compound social risk",
            detail=f"{social_count} concurrent unmet social needs recorded.",
            patient_detail="Several things outside of medical care are making it harder to stay well right now.",
            suggested_action="Community health worker referral and closed-loop social service referrals.",
            evidence={"social_risk_count": social_count})

    if p["transportation_barrier"] == 1 and p["missed_appointments_12mo"] >= 3:
        add(code="TRANSPORT_DRIVEN_GAPS", severity="warning",
            title="Transportation barrier driving missed care",
            detail=f"{int(p['missed_appointments_12mo'])} missed appointments alongside a documented transport barrier.",
            patient_detail="Getting to appointments is difficult, and visits are being missed as a result.",
            suggested_action="Enrol in non-emergency medical transport; offer telehealth for suitable visits.",
            evidence={"missed_appointments_12mo": p["missed_appointments_12mo"]})

    if p["has_primary_care"] == 0 and p["chronic_condition_count"] >= 2:
        add(code="UNMANAGED_MULTIMORBIDITY", severity="critical",
            title="Multiple chronic conditions without primary care",
            detail=f"{int(p['chronic_condition_count'])} chronic conditions with no established primary care relationship.",
            patient_detail="You are managing several conditions without one regular doctor who sees the whole picture.",
            suggested_action="Assign a primary care provider and book an initial visit within 14 days.",
            evidence={"chronic_condition_count": p["chronic_condition_count"]})

    if p["falls_12mo"] >= 2 and (p["lives_alone"] == 1 or p["caregiver_support"] == 0):
        add(code="FALL_RISK_ALONE", severity="warning",
            title="Repeat falls with limited support at home",
            detail=f"{int(p['falls_12mo'])} falls in 12 months with limited in-home support.",
            patient_detail="Falls at home are a real risk for you, and there may not be someone nearby to help.",
            suggested_action="Home safety evaluation, PT referral and personal emergency response system.",
            evidence={"falls_12mo": p["falls_12mo"]})

    if p["serious_mental_illness"] == 1 and p["social_isolation"] == 1:
        add(code="SMI_WITH_ISOLATION", severity="warning",
            title="Serious mental illness with social isolation",
            detail="Behavioural health need combined with limited social contact.",
            patient_detail="Staying connected to people matters a lot for your health, and right now that support is thin.",
            suggested_action="Behavioural health outreach and peer support connection.",
            evidence={})

    if p["cognitive_impairment"] == 1 and p["medication_count"] >= 8 and p["caregiver_support"] == 0:
        add(code="COGNITIVE_MED_MANAGEMENT", severity="critical",
            title="Complex medication regimen with cognitive impairment and no caregiver",
            detail="Self-administration of a complex regimen is unlikely to be reliable without support.",
            patient_detail="Keeping track of this many medicines is a lot to manage alone.",
            suggested_action="Arrange supervised medication administration or caregiver support.",
            evidence={"medication_count": p["medication_count"]})

    if p["food_insecurity"] == 1 and (p["diabetes_complicated"] == 1 or p["heart_failure"] == 1):
        add(code="FOOD_INSECURITY_DIET_SENSITIVE", severity="warning",
            title="Food insecurity with a diet-sensitive condition",
            detail="Diabetes or heart failure control depends directly on consistent, appropriate nutrition.",
            patient_detail="Your condition depends a lot on regular, suitable meals, and food has been hard to count on.",
            suggested_action="Medically tailored meals or food pharmacy referral; SNAP enrolment check.",
            evidence={})

    order = {"critical": 0, "warning": 1, "advisory": 2}
    flags.sort(key=lambda f: order[f.severity])
    return flags
