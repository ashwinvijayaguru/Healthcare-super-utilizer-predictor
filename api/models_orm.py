"""Persistence layer.

The schema is deliberately PHI-free. There is no name, address, date of birth,
MRN, phone or email column anywhere, and there is nowhere to put one. Patients
are identified by a caller-supplied opaque ``external_ref`` (a pseudonymous key
that resolves back to a person only inside the client's own system) plus a birth
*year*. Under HIPAA Safe Harbor, year of birth is permitted for anyone under 90;
ages of 90+ are stored capped at 90, which is handled in the service layer.

That constraint is a demo requirement, but it is also good design: the risk
service never needs identity to do its job, so it should never hold it.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def new_id() -> str:
    return str(uuid.uuid4())


class Base(DeclarativeBase):
    pass


class Patient(Base):
    """A pseudonymous subject of assessment. Contains no identifying attributes."""

    __tablename__ = "patients"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    external_ref: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    birth_year: Mapped[int | None] = mapped_column(Integer, nullable=True)
    sex_at_birth: Mapped[str | None] = mapped_column(String(16), nullable=True)
    postal_sector: Mapped[str | None] = mapped_column(String(8), nullable=True)  # 3-digit ZIP max
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    assessments: Mapped[list["Assessment"]] = relationship(
        back_populates="patient", cascade="all, delete-orphan", order_by="Assessment.created_at.desc()")

    __table_args__ = (
        CheckConstraint("birth_year IS NULL OR birth_year BETWEEN 1900 AND 2100", name="ck_birth_year"),
    )


class Assessment(Base):
    """One scored risk assessment, immutable once written."""

    __tablename__ = "assessments"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    patient_id: Mapped[str] = mapped_column(ForeignKey("patients.id", ondelete="CASCADE"), index=True)

    risk_score: Mapped[float] = mapped_column(Float)  # calibrated probability, 0-1
    risk_tier: Mapped[str] = mapped_column(String(16), index=True)
    percentile: Mapped[float | None] = mapped_column(Float, nullable=True)

    model_name: Mapped[str] = mapped_column(String(64))
    model_version: Mapped[str] = mapped_column(String(32), index=True)
    feature_schema_version: Mapped[str] = mapped_column(String(32))

    # Full input snapshot, so a score can always be reproduced and audited.
    features: Mapped[dict] = mapped_column(JSON)
    group_attribution: Mapped[dict] = mapped_column(JSON)
    narrative_source: Mapped[str] = mapped_column(String(16), default="template")
    patient_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    clinician_summary: Mapped[str | None] = mapped_column(Text, nullable=True)

    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    created_by: Mapped[str | None] = mapped_column(String(128), nullable=True)

    patient: Mapped[Patient] = relationship(back_populates="assessments")
    drivers: Mapped[list["AssessmentDriver"]] = relationship(
        back_populates="assessment", cascade="all, delete-orphan")
    flags: Mapped[list["AssessmentFlag"]] = relationship(
        back_populates="assessment", cascade="all, delete-orphan")

    __table_args__ = (
        CheckConstraint("risk_score >= 0 AND risk_score <= 1", name="ck_risk_score_range"),
        CheckConstraint("risk_tier IN ('low','rising','high','very_high')", name="ck_risk_tier"),
        Index("ix_assessment_patient_created", "patient_id", "created_at"),
    )


class AssessmentDriver(Base):
    """One feature's signed contribution to a score (exact Shapley value)."""

    __tablename__ = "assessment_drivers"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    assessment_id: Mapped[str] = mapped_column(
        ForeignKey("assessments.id", ondelete="CASCADE"), index=True)
    feature: Mapped[str] = mapped_column(String(64))
    label: Mapped[str] = mapped_column(String(128))
    state: Mapped[str] = mapped_column(String(160))
    feature_group: Mapped[str] = mapped_column(String(32))
    value: Mapped[float] = mapped_column(Float)
    contribution: Mapped[float] = mapped_column(Float)
    direction: Mapped[str] = mapped_column(String(16))
    rank: Mapped[int] = mapped_column(Integer)

    assessment: Mapped[Assessment] = relationship(back_populates="drivers")

    __table_args__ = (
        CheckConstraint("direction IN ('increases','decreases')", name="ck_driver_direction"),
    )


class AssessmentFlag(Base):
    """A rule-based clinical red flag raised independently of the model score."""

    __tablename__ = "assessment_flags"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    assessment_id: Mapped[str] = mapped_column(
        ForeignKey("assessments.id", ondelete="CASCADE"), index=True)
    code: Mapped[str] = mapped_column(String(64), index=True)
    severity: Mapped[str] = mapped_column(String(16), index=True)
    title: Mapped[str] = mapped_column(String(160))
    detail: Mapped[str] = mapped_column(Text)
    suggested_action: Mapped[str] = mapped_column(Text)
    evidence: Mapped[dict] = mapped_column(JSON, default=dict)

    assessment: Mapped[Assessment] = relationship(back_populates="flags")

    __table_args__ = (
        CheckConstraint("severity IN ('critical','warning','advisory')", name="ck_flag_severity"),
    )


class ModelRegistry(Base):
    """Which model version is live, and what it scored on its holdout."""

    __tablename__ = "model_registry"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    model_name: Mapped[str] = mapped_column(String(64), index=True)
    model_version: Mapped[str] = mapped_column(String(32))
    feature_schema_version: Mapped[str] = mapped_column(String(32))
    estimator_kind: Mapped[str] = mapped_column(String(48))
    trained_at: Mapped[str] = mapped_column(String(48))
    holdout_metrics: Mapped[dict] = mapped_column(JSON)
    tier_cuts: Mapped[dict] = mapped_column(JSON)
    is_active: Mapped[int] = mapped_column(Integer, default=1, index=True)
    registered_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class AuditLog(Base):
    """Append-only record of who scored what and when."""

    __tablename__ = "audit_log"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    action: Mapped[str] = mapped_column(String(48), index=True)
    actor: Mapped[str | None] = mapped_column(String(128), nullable=True)
    subject_type: Mapped[str | None] = mapped_column(String(48), nullable=True)
    subject_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    request_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    detail: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
