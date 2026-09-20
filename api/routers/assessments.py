"""Assessment endpoints."""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from api.db import get_session
from api.models_orm import Assessment, AssessmentDriver, AssessmentFlag, AuditLog, Patient
from api.narrative import build_patient_summary
from api.schemas import (
    AssessmentListItem,
    AssessmentRequest,
    AssessmentResponse,
)
from api.scoring import RiskScorer

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["assessments"])

MAX_AGE_STORED = 90  # HIPAA Safe Harbor: ages over 89 must be aggregated


def _get_or_create_patient(session: Session, ref) -> Patient:
    patient = session.scalar(select(Patient).where(Patient.external_ref == ref.external_ref))
    if patient is None:
        patient = Patient(
            external_ref=ref.external_ref,
            birth_year=ref.birth_year,
            sex_at_birth=ref.sex_at_birth,
            postal_sector=ref.postal_sector,
        )
        session.add(patient)
        session.flush()
    else:
        # Late-arriving demographics are allowed to fill gaps but not to churn.
        patient.birth_year = patient.birth_year or ref.birth_year
        patient.sex_at_birth = patient.sex_at_birth or ref.sex_at_birth
        patient.postal_sector = patient.postal_sector or ref.postal_sector
    return patient


@router.post("/assessments", response_model=AssessmentResponse, status_code=status.HTTP_201_CREATED)
def create_assessment(
    payload: AssessmentRequest,
    request: Request,
    session: Annotated[Session, Depends(get_session)],
) -> AssessmentResponse:
    """Score one member and, unless asked not to, persist the result."""
    scorer = RiskScorer.instance()
    intake = payload.intake.model_dump()

    # Safe Harbor: never persist or reason about an exact age above 89.
    if intake["age"] > MAX_AGE_STORED:
        intake["age"] = float(MAX_AGE_STORED)

    result = scorer.score(intake)
    result["clinician_summary"] = scorer.clinician_summary(intake, result)
    result["patient_summary"] = (
        build_patient_summary(intake, result)
        if payload.generate_narrative
        else {"headline": "Summary not requested", "body": "", "what_this_means": [],
              "next_steps": [], "reassurance": "", "source": "template"}
    )

    assessment_id = None
    created_at = None
    if payload.persist:
        patient = _get_or_create_patient(session, payload.patient)
        assessment = Assessment(
            patient_id=patient.id,
            risk_score=result["risk_score"],
            risk_tier=result["risk_tier"],
            percentile=result["percentile"],
            model_name=result["model_name"],
            model_version=result["model_version"],
            feature_schema_version=scorer.bundle["feature_schema_version"],
            features=intake,
            group_attribution=result["group_attribution"],
            narrative_source=result["patient_summary"].get("source", "template"),
            patient_summary=result["patient_summary"].get("body"),
            clinician_summary=result["clinician_summary"],
            latency_ms=result["latency_ms"],
            created_by=request.headers.get("x-actor"),
        )
        session.add(assessment)
        session.flush()

        for rank, driver in enumerate(result["drivers"], start=1):
            session.add(AssessmentDriver(
                assessment_id=assessment.id, feature=driver["feature"], label=driver["label"],
                state=driver["state"], feature_group=driver["group"], value=driver["value"],
                contribution=driver["contribution"], direction=driver["direction"], rank=rank))
        for flag in result["red_flags"]:
            session.add(AssessmentFlag(
                assessment_id=assessment.id, code=flag["code"], severity=flag["severity"],
                title=flag["title"], detail=flag["detail"],
                suggested_action=flag["suggested_action"], evidence=flag["evidence"]))
        session.add(AuditLog(
            action="assessment.created", actor=request.headers.get("x-actor"),
            subject_type="assessment", subject_id=assessment.id,
            request_id=request.headers.get("x-request-id"),
            detail={"risk_tier": result["risk_tier"], "model_version": result["model_version"]}))

        assessment_id = assessment.id
        created_at = assessment.created_at

    return AssessmentResponse(
        assessment_id=assessment_id,
        external_ref=payload.patient.external_ref,
        created_at=created_at,
        **result,
    )


@router.get("/assessments", response_model=list[AssessmentListItem])
def list_assessments(
    session: Annotated[Session, Depends(get_session)],
    external_ref: str | None = None,
    risk_tier: str | None = None,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> list[AssessmentListItem]:
    stmt = select(Assessment, Patient.external_ref).join(Patient, Assessment.patient_id == Patient.id)
    if external_ref:
        stmt = stmt.where(Patient.external_ref == external_ref)
    if risk_tier:
        stmt = stmt.where(Assessment.risk_tier == risk_tier)
    stmt = stmt.order_by(Assessment.created_at.desc()).limit(limit).offset(offset)

    return [
        AssessmentListItem(
            assessment_id=row.Assessment.id, external_ref=row.external_ref,
            risk_score=row.Assessment.risk_score, risk_tier=row.Assessment.risk_tier,
            model_version=row.Assessment.model_version, created_at=row.Assessment.created_at)
        for row in session.execute(stmt)
    ]


@router.get("/assessments/{assessment_id}")
def get_assessment(
    assessment_id: str,
    session: Annotated[Session, Depends(get_session)],
) -> dict:
    assessment = session.get(Assessment, assessment_id)
    if assessment is None:
        raise HTTPException(status_code=404, detail="assessment not found")
    return {
        "assessment_id": assessment.id,
        "external_ref": assessment.patient.external_ref,
        "risk_score": assessment.risk_score,
        "risk_tier": assessment.risk_tier,
        "percentile": assessment.percentile,
        "model_version": assessment.model_version,
        "features": assessment.features,
        "group_attribution": assessment.group_attribution,
        "clinician_summary": assessment.clinician_summary,
        "patient_summary": assessment.patient_summary,
        "narrative_source": assessment.narrative_source,
        "drivers": [
            {"feature": d.feature, "label": d.label, "state": d.state, "group": d.feature_group,
             "value": d.value, "contribution": d.contribution, "direction": d.direction, "rank": d.rank}
            for d in sorted(assessment.drivers, key=lambda d: d.rank)
        ],
        "red_flags": [
            {"code": f.code, "severity": f.severity, "title": f.title,
             "detail": f.detail, "suggested_action": f.suggested_action, "evidence": f.evidence}
            for f in assessment.flags
        ],
        "created_at": assessment.created_at,
    }
