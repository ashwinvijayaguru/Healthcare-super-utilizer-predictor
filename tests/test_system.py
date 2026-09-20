"""Test suite.

The tests that matter most here are not the happy paths. They are:
  * training/serving skew (a single row must score identically inside a batch),
  * probability bounds (the model must never claim certainty it cannot support),
  * red flags firing independently of the score,
  * the schema containing nowhere to put PHI.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.narrative import TemplateNarrator, _parse_json_object, _validate_narrative
from api.schemas import ClinicalIntake, assert_contract_alignment
from api.scoring import RiskScorer
from ml.config import FEATURE_NAMES
from ml.explain import red_flags
from ml.features import build_feature_frame
from ml.generate_cohort import build_cohort

LOW_RISK = {
    "age": 32, "lives_alone": 0, "ed_visits_12mo": 0, "inpatient_admits_12mo": 0,
    "inpatient_days_12mo": 0, "days_since_last_discharge": 999, "prior_30d_readmission": 0,
    "outpatient_visits_12mo": 2, "missed_appointments_12mo": 0, "has_primary_care": 1,
    "chronic_condition_count": 0, "charlson_index": 0, "heart_failure": 0, "copd": 0,
    "diabetes_complicated": 0, "ckd_stage4_plus": 0, "cirrhosis": 0, "cancer_active": 0,
    "serious_mental_illness": 0, "substance_use_disorder": 0, "cognitive_impairment": 0,
    "mobility_impairment": 0, "falls_12mo": 0, "medication_count": 1, "high_risk_med_count": 0,
    "medication_adherence_pdc": 0.95, "opioid_therapy": 0, "housing_instability": 0,
    "food_insecurity": 0, "transportation_barrier": 0, "social_isolation": 0,
    "financial_strain": 0, "uninsured_or_medicaid": 0, "caregiver_support": 1,
    "area_deprivation_index": 20, "limited_health_literacy": 0,
}

HIGH_RISK = {
    **LOW_RISK, "age": 71, "lives_alone": 1, "ed_visits_12mo": 6, "inpatient_admits_12mo": 3,
    "inpatient_days_12mo": 22, "days_since_last_discharge": 12, "prior_30d_readmission": 1,
    "missed_appointments_12mo": 5, "has_primary_care": 0, "chronic_condition_count": 7,
    "charlson_index": 8, "heart_failure": 1, "copd": 1, "diabetes_complicated": 1,
    "mobility_impairment": 1, "falls_12mo": 3, "medication_count": 18, "high_risk_med_count": 5,
    "medication_adherence_pdc": 0.42, "housing_instability": 1, "food_insecurity": 1,
    "transportation_barrier": 1, "social_isolation": 1, "financial_strain": 1,
    "uninsured_or_medicaid": 1, "caregiver_support": 0, "area_deprivation_index": 92,
    "limited_health_literacy": 1,
}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def scorer():
    return RiskScorer.instance()


# ----------------------------------------------------------------- contract
def test_api_schema_matches_training_contract():
    assert_contract_alignment()
    assert set(ClinicalIntake.model_fields) == set(FEATURE_NAMES)


def test_feature_builder_is_row_independent():
    """A row scored alone must produce the identical vector it does in a batch.

    This is the single most common cause of a model that works in a notebook and
    silently misbehaves in production.
    """
    cohort = build_cohort(200, seed=5)
    batch = build_feature_frame(cohort)
    for idx in (0, 17, 199):
        single = build_feature_frame(cohort.iloc[[idx]].reset_index(drop=True))
        np.testing.assert_allclose(single.to_numpy()[0], batch.to_numpy()[idx], rtol=1e-9)


def test_feature_builder_rejects_missing_features():
    with pytest.raises(ValueError, match="missing required features"):
        build_feature_frame(pd.DataFrame([{"age": 50}]))


# ------------------------------------------------------------------ scoring
def test_high_risk_scores_above_low_risk(scorer):
    assert scorer.score(HIGH_RISK)["risk_score"] > scorer.score(LOW_RISK)["risk_score"]


def test_probabilities_never_saturate(scorer):
    """No individual may be told they are a certainty in either direction."""
    for record in (LOW_RISK, HIGH_RISK):
        score = scorer.score(record)["risk_score"]
        assert 0.0 < score < 1.0
        assert scorer.prob_floor <= score <= scorer.prob_ceiling


def test_scoring_is_deterministic(scorer):
    assert scorer.score(HIGH_RISK)["risk_score"] == scorer.score(HIGH_RISK)["risk_score"]


def test_drivers_reflect_direction_correctly(scorer):
    """A protective feature that is present must not be listed as a risk driver."""
    result = scorer.score({**LOW_RISK, "has_primary_care": 1})
    risk_features = {d["feature"] for d in result["drivers"]}
    protective_features = {d["feature"] for d in result["protective_factors"]}
    assert "has_primary_care" not in risk_features
    assert "has_primary_care" in protective_features


def test_group_attribution_sums_to_one(scorer):
    total = sum(scorer.score(HIGH_RISK)["group_attribution"].values())
    assert total == pytest.approx(1.0, abs=1e-3)


def test_social_factors_move_the_score(scorer):
    """SDOH must be load-bearing, not decorative.

    Measured on a mid-risk profile on purpose. The extreme profile saturates
    against the probability ceiling, where no input can move the number further -
    a real property of the bounded output, not a bug, but it makes the extreme
    profile useless for detecting sensitivity.
    """
    moderate = {**LOW_RISK, "age": 58, "ed_visits_12mo": 2, "chronic_condition_count": 3,
                "charlson_index": 3, "copd": 1, "medication_count": 7,
                "high_risk_med_count": 2, "medication_adherence_pdc": 0.72}
    without = scorer.score(moderate)["risk_score"]
    with_sdoh = scorer.score({**moderate, "housing_instability": 1, "food_insecurity": 1,
                              "transportation_barrier": 1, "social_isolation": 1,
                              "financial_strain": 1, "caregiver_support": 0,
                              "area_deprivation_index": 90})["risk_score"]
    assert with_sdoh > without
    assert with_sdoh - without > 0.02, "social drivers should have a material effect"


def test_extreme_profiles_saturate_at_the_ceiling(scorer):
    """Documents the deliberate flat spot at the top of the range."""
    result = scorer.score(HIGH_RISK)
    assert result["risk_score"] == pytest.approx(scorer.prob_ceiling, abs=1e-6)
    assert result["raw_score"] >= result["risk_score"]
    assert result["risk_tier"] == "very_high"


# ---------------------------------------------------------------- red flags
def test_red_flags_are_independent_of_the_model():
    """A low model score must never suppress a clinical safety signal."""
    record = {**LOW_RISK, "medication_adherence_pdc": 0.3, "high_risk_med_count": 2,
              "medication_count": 4}
    codes = {f.code for f in red_flags(record)}
    assert "ADHERENCE_HIGH_RISK_MEDS" in codes


def test_frequent_ed_flag_threshold():
    assert "FREQUENT_ED" not in {f.code for f in red_flags({**LOW_RISK, "ed_visits_12mo": 3})}
    assert "FREQUENT_ED" in {f.code for f in red_flags({**LOW_RISK, "ed_visits_12mo": 4})}


def test_healthy_record_raises_no_flags():
    assert red_flags(LOW_RISK) == []


def test_critical_flags_sort_first():
    flags = red_flags(HIGH_RISK)
    severities = [f.severity for f in flags]
    assert severities == sorted(severities, key=lambda s: {"critical": 0, "warning": 1, "advisory": 2}[s])


# ---------------------------------------------------------------- narrative
def test_template_narrative_is_always_complete(scorer):
    for record in (LOW_RISK, HIGH_RISK):
        summary = TemplateNarrator().generate(record, scorer.score(record))
        assert summary["headline"] and summary["body"] and summary["reassurance"]
        assert summary["what_this_means"] and summary["next_steps"]


def test_patient_narrative_avoids_clinical_jargon(scorer):
    summary = TemplateNarrator().generate(HIGH_RISK, scorer.score(HIGH_RISK))
    blob = " ".join([summary["headline"], summary["body"],
                     " ".join(summary["what_this_means"])]).lower()
    for term in ("super utilizer", "risk score", "algorithm", "comorbidity", "charlson"):
        assert term not in blob


def test_narrative_validator_rejects_banned_terms():
    with pytest.raises(ValueError, match="prohibited term"):
        _validate_narrative({
            "headline": "Your risk score is high", "body": "b",
            "what_this_means": ["x"], "next_steps": ["call us"], "reassurance": "r"})


def test_narrative_validator_rejects_incomplete_output():
    with pytest.raises(ValueError, match="missing fields"):
        _validate_narrative({"headline": "h", "body": "b"})


def test_narrative_parser_tolerates_code_fences():
    assert _parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}


def test_narrative_validator_normalises_string_steps():
    payload = {"headline": "h", "body": "b", "what_this_means": ["x"],
               "next_steps": ["call the clinic"], "reassurance": "r"}
    _validate_narrative(payload)
    assert payload["next_steps"] == [{"text": "call the clinic", "owner": "care_team"}]


# ---------------------------------------------------------------------- API
def test_health_endpoint(client):
    body = client.get("/api/v1/health").json()
    assert body["status"] == "ok" and body["model_loaded"] and body["database"]


def test_assessment_round_trip(client):
    response = client.post("/api/v1/assessments",
                           json={"patient": {"external_ref": "TEST-RT"}, "intake": HIGH_RISK})
    assert response.status_code == 201
    body = response.json()
    assert body["risk_tier"] in {"low", "rising", "high", "very_high"}
    assert body["drivers"] and body["clinician_summary"] and body["patient_summary"]["headline"]

    fetched = client.get(f"/api/v1/assessments/{body['assessment_id']}").json()
    assert fetched["risk_score"] == body["risk_score"]


def test_assessment_can_skip_persistence(client):
    body = client.post("/api/v1/assessments",
                       json={"patient": {"external_ref": "TEST-NP"}, "intake": LOW_RISK,
                             "persist": False}).json()
    assert body["assessment_id"] is None


def test_form_schema_covers_every_feature(client):
    schema = client.get("/api/v1/form-schema").json()
    fields = {f["name"] for group in schema["groups"] for f in group["fields"]}
    assert fields == set(FEATURE_NAMES)


def test_unknown_field_is_rejected(client):
    response = client.post("/api/v1/assessments",
                           json={"patient": {"external_ref": "X"},
                                 "intake": {**LOW_RISK, "patient_name": "Jane Doe"}})
    assert response.status_code == 422


def test_out_of_range_input_is_rejected(client):
    response = client.post("/api/v1/assessments",
                           json={"patient": {"external_ref": "X"}, "intake": {**LOW_RISK, "age": 7}})
    assert response.status_code == 422


def test_inconsistent_input_is_rejected(client):
    response = client.post(
        "/api/v1/assessments",
        json={"patient": {"external_ref": "X"},
              "intake": {**LOW_RISK, "medication_count": 2, "high_risk_med_count": 6}})
    assert response.status_code == 422


# ------------------------------------------------------------------ privacy
def test_external_ref_rejects_obvious_identifiers(client):
    response = client.post("/api/v1/assessments",
                           json={"patient": {"external_ref": "jane.doe@example.com"},
                                 "intake": LOW_RISK})
    assert response.status_code == 422


def test_schema_has_nowhere_to_store_phi():
    """Structural guarantee, not a policy promise."""
    from api.models_orm import Base

    banned = {"name", "first_name", "last_name", "dob", "date_of_birth", "ssn", "mrn",
              "phone", "email", "address", "street"}
    for table in Base.metadata.tables.values():
        for column in table.columns:
            assert column.name.lower() not in banned, f"{table.name}.{column.name} can hold PHI"


def test_age_over_89_is_capped(client):
    """HIPAA Safe Harbor: ages above 89 must not be stored exactly."""
    body = client.post("/api/v1/assessments",
                       json={"patient": {"external_ref": "TEST-AGE"},
                             "intake": {**LOW_RISK, "age": 97}}).json()
    stored = client.get(f"/api/v1/assessments/{body['assessment_id']}").json()
    assert stored["features"]["age"] <= 90


# ------------------------------------------------------------------- cohort
def test_cohort_is_reproducible():
    a = build_cohort(500, seed=11)
    b = build_cohort(500, seed=11)
    pd.testing.assert_frame_equal(a, b)


def test_cohort_matches_published_benchmarks():
    """Guards the calibration targets documented in docs/DATASETS.md."""
    from ml.generate_cohort import calibration_report

    report = calibration_report(build_cohort(20_000, seed=3))
    assert 0.03 <= report["super_utilizer_rate"] <= 0.15
    assert 0.42 <= report["top5_cost_share"] <= 0.62   # MEPS: top 5% hold ~50% of spend
    assert 0.12 <= report["any_ed_visit_rate"] <= 0.28  # NHIS: ~20% of adults
    assert 0.05 <= report["polypharmacy_rate"] <= 0.20
    assert 45 <= report["mean_age"] <= 60
