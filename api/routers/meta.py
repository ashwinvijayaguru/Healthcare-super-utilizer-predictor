"""Model metadata, form schema and health endpoints."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from api.config import get_settings
from api.db import get_session
from api.narrative import narrative_provider_name
from api.schemas import HealthResponse, ModelInfo
from api.scoring import OUTCOME_DEFINITION, RiskScorer
from ml.config import FEATURES, GROUP_LABELS, TIER_META, TIER_ORDER

router = APIRouter(prefix="/api/v1", tags=["meta"])


@router.get("/health", response_model=HealthResponse)
def health(session: Annotated[Session, Depends(get_session)]) -> HealthResponse:
    settings = get_settings()

    database_ok = True
    try:
        session.execute(text("SELECT 1"))
    except Exception:  # noqa: BLE001
        database_ok = False

    model_ok = True
    try:
        RiskScorer.instance()
    except Exception:  # noqa: BLE001
        model_ok = False

    return HealthResponse(
        status="ok" if (database_ok and model_ok) else "degraded",
        model_loaded=model_ok,
        database=database_ok,
        narrative_provider=narrative_provider_name(),
        version=settings.api_version,
    )


@router.get("/model", response_model=ModelInfo)
def model_info() -> ModelInfo:
    scorer = RiskScorer.instance()
    bundle = scorer.bundle
    return ModelInfo(
        model_name=bundle["model_name"],
        model_version=bundle["model_version"],
        estimator_kind=bundle.get("estimator_kind", "unknown"),
        feature_schema_version=bundle["feature_schema_version"],
        trained_at=bundle["trained_at"],
        tier_cuts=bundle["tier_cuts"],
        holdout_metrics=bundle.get("holdout_metrics", {}),
        base_rate=scorer.base_rate,
        outcome_definition=OUTCOME_DEFINITION,
    )


@router.get("/metrics")
def full_metrics() -> dict[str, Any]:
    """Evaluation report, including subgroup performance. Powers the model card."""
    return RiskScorer.instance().metrics


@router.get("/form-schema")
def form_schema() -> dict[str, Any]:
    """The intake form, described by the server.

    The UI renders its fields from this rather than hardcoding them, so adding a
    feature to the model does not require a frontend release.
    """
    groups: dict[str, list[dict[str, Any]]] = {key: [] for key in GROUP_LABELS}
    for spec in FEATURES:
        groups[spec.group].append({
            "name": spec.name,
            "label": spec.label,
            "kind": spec.kind,
            "min": spec.minimum,
            "max": spec.maximum,
            "unit": spec.unit,
            "higher_is_protective": spec.higher_is_protective,
        })
    return {
        "groups": [
            {"key": key, "label": label, "fields": groups[key]}
            for key, label in GROUP_LABELS.items()
        ],
        "tiers": [{"key": key, **TIER_META[key]} for key in TIER_ORDER],
        "outcome_definition": OUTCOME_DEFINITION,
    }
