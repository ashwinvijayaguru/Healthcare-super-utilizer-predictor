-- =====================================================================
-- Super-Utilizer Risk Service - PostgreSQL schema
--
-- Design rule: this database has nowhere to put PHI.
-- There is no name, address, date of birth, MRN, phone or email column,
-- and adding one should fail code review. Subjects are identified by an
-- opaque external_ref that resolves to a person only inside the client's
-- own system, plus a birth YEAR (permitted under HIPAA Safe Harbor for
-- anyone under 90; the service caps ages at 90 before writing).
--
-- The risk service never needs identity to do its job, so it never holds it.
-- =====================================================================

CREATE EXTENSION IF NOT EXISTS "pgcrypto";   -- gen_random_uuid()

-- ---------------------------------------------------------------------
-- patients
-- ---------------------------------------------------------------------
CREATE TABLE patients (
    id              UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    external_ref    VARCHAR(128) NOT NULL UNIQUE,
    birth_year      INTEGER,
    sex_at_birth    VARCHAR(16),
    postal_sector   VARCHAR(8),           -- first 3 ZIP digits only
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ  NOT NULL DEFAULT now(),

    CONSTRAINT ck_birth_year
        CHECK (birth_year IS NULL OR birth_year BETWEEN 1900 AND 2100),
    CONSTRAINT ck_sex_at_birth
        CHECK (sex_at_birth IS NULL OR sex_at_birth IN ('female','male','other','unknown')),
    -- A weak but useful guard against someone pasting a name or an email into
    -- the pseudonymous key field.
    CONSTRAINT ck_external_ref_opaque
        CHECK (position('@' in external_ref) = 0)
);

COMMENT ON TABLE  patients IS 'Pseudonymous assessment subjects. Contains no identifying attributes by design.';
COMMENT ON COLUMN patients.external_ref IS 'Opaque key supplied by the caller; re-identifiable only in the source system.';
COMMENT ON COLUMN patients.postal_sector IS 'Three-digit ZIP prefix at most, per HIPAA Safe Harbor.';

-- ---------------------------------------------------------------------
-- assessments  (immutable once written)
-- ---------------------------------------------------------------------
CREATE TABLE assessments (
    id                      UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id              UUID         NOT NULL REFERENCES patients(id) ON DELETE CASCADE,

    risk_score              DOUBLE PRECISION NOT NULL,   -- calibrated probability
    risk_tier               VARCHAR(16)  NOT NULL,
    percentile              DOUBLE PRECISION,

    model_name              VARCHAR(64)  NOT NULL,
    model_version           VARCHAR(32)  NOT NULL,
    feature_schema_version  VARCHAR(32)  NOT NULL,

    -- Complete input snapshot. Any score can be reproduced from this row alone,
    -- which is what makes an appeal or an audit answerable rather than a guess.
    features                JSONB        NOT NULL,
    group_attribution       JSONB        NOT NULL DEFAULT '{}'::jsonb,

    narrative_source        VARCHAR(16)  NOT NULL DEFAULT 'template',
    patient_summary         TEXT,
    clinician_summary       TEXT,

    latency_ms              INTEGER,
    created_at              TIMESTAMPTZ  NOT NULL DEFAULT now(),
    created_by              VARCHAR(128),

    CONSTRAINT ck_risk_score_range CHECK (risk_score >= 0 AND risk_score <= 1),
    CONSTRAINT ck_risk_tier        CHECK (risk_tier IN ('low','rising','high','very_high')),
    CONSTRAINT ck_narrative_source CHECK (narrative_source IN ('llm','template'))
);

CREATE INDEX ix_assessments_patient_created ON assessments (patient_id, created_at DESC);
CREATE INDEX ix_assessments_tier_created    ON assessments (risk_tier, created_at DESC);
CREATE INDEX ix_assessments_model_version   ON assessments (model_version);
-- Lets a monitoring job query drift on any single feature without a schema change.
CREATE INDEX ix_assessments_features_gin    ON assessments USING GIN (features jsonb_path_ops);

COMMENT ON COLUMN assessments.features IS 'Full de-identified model input, for reproducibility and drift monitoring.';

-- Assessments are an audit record, so they are append-only.
CREATE OR REPLACE FUNCTION forbid_assessment_mutation() RETURNS trigger AS $$
BEGIN
    RAISE EXCEPTION 'assessments are immutable; write a new assessment instead';
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_assessments_immutable
    BEFORE UPDATE OR DELETE ON assessments
    FOR EACH ROW EXECUTE FUNCTION forbid_assessment_mutation();

-- ---------------------------------------------------------------------
-- assessment_drivers  (per-feature Shapley attribution)
-- ---------------------------------------------------------------------
CREATE TABLE assessment_drivers (
    id             UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    assessment_id  UUID         NOT NULL REFERENCES assessments(id) ON DELETE CASCADE,
    feature        VARCHAR(64)  NOT NULL,
    label          VARCHAR(128) NOT NULL,
    state          VARCHAR(160) NOT NULL,   -- the state the member is actually in
    feature_group  VARCHAR(32)  NOT NULL,
    value          DOUBLE PRECISION NOT NULL,
    contribution   DOUBLE PRECISION NOT NULL,  -- signed, log-odds space
    direction      VARCHAR(16)  NOT NULL,
    rank           INTEGER      NOT NULL,

    CONSTRAINT ck_driver_direction CHECK (direction IN ('increases','decreases')),
    CONSTRAINT ck_driver_group
        CHECK (feature_group IN ('demographics','utilization','clinical','medications','sdoh'))
);

CREATE INDEX ix_drivers_assessment ON assessment_drivers (assessment_id, rank);
CREATE INDEX ix_drivers_feature    ON assessment_drivers (feature);

-- ---------------------------------------------------------------------
-- assessment_flags  (rule-based, independent of the score)
-- ---------------------------------------------------------------------
CREATE TABLE assessment_flags (
    id                UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    assessment_id     UUID         NOT NULL REFERENCES assessments(id) ON DELETE CASCADE,
    code              VARCHAR(64)  NOT NULL,
    severity          VARCHAR(16)  NOT NULL,
    title             VARCHAR(160) NOT NULL,
    detail            TEXT         NOT NULL,
    suggested_action  TEXT         NOT NULL,
    evidence          JSONB        NOT NULL DEFAULT '{}'::jsonb,

    CONSTRAINT ck_flag_severity CHECK (severity IN ('critical','warning','advisory'))
);

CREATE INDEX ix_flags_assessment ON assessment_flags (assessment_id);
CREATE INDEX ix_flags_code       ON assessment_flags (code, severity);

COMMENT ON TABLE assessment_flags IS
    'Deterministic clinical rules. Raised independently of the model score so a low score can never suppress a safety signal.';

-- ---------------------------------------------------------------------
-- model_registry
-- ---------------------------------------------------------------------
CREATE TABLE model_registry (
    id                      UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    model_name              VARCHAR(64)  NOT NULL,
    model_version           VARCHAR(32)  NOT NULL,
    feature_schema_version  VARCHAR(32)  NOT NULL,
    estimator_kind          VARCHAR(48)  NOT NULL,
    trained_at              VARCHAR(48)  NOT NULL,
    holdout_metrics         JSONB        NOT NULL DEFAULT '{}'::jsonb,
    tier_cuts               JSONB        NOT NULL DEFAULT '{}'::jsonb,
    is_active               SMALLINT     NOT NULL DEFAULT 1,
    registered_at           TIMESTAMPTZ  NOT NULL DEFAULT now()
);

-- At most one live model at a time.
CREATE UNIQUE INDEX ux_model_registry_active
    ON model_registry (model_name) WHERE is_active = 1;

-- ---------------------------------------------------------------------
-- audit_log  (append-only)
-- ---------------------------------------------------------------------
CREATE TABLE audit_log (
    id            UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    action        VARCHAR(48)  NOT NULL,
    actor         VARCHAR(128),
    subject_type  VARCHAR(48),
    subject_id    UUID,
    request_id    VARCHAR(64),
    detail        JSONB        NOT NULL DEFAULT '{}'::jsonb,
    created_at    TIMESTAMPTZ  NOT NULL DEFAULT now()
);

CREATE INDEX ix_audit_action_created ON audit_log (action, created_at DESC);
CREATE INDEX ix_audit_subject        ON audit_log (subject_id);

-- =====================================================================
-- Reporting views
-- =====================================================================

-- The working list: latest score per member, highest risk first.
CREATE VIEW v_current_risk AS
SELECT DISTINCT ON (a.patient_id)
       p.external_ref,
       a.id            AS assessment_id,
       a.risk_score,
       a.risk_tier,
       a.percentile,
       a.model_version,
       a.created_at,
       (SELECT count(*) FROM assessment_flags f
         WHERE f.assessment_id = a.id AND f.severity = 'critical') AS critical_flags
FROM assessments a
JOIN patients p ON p.id = a.patient_id
ORDER BY a.patient_id, a.created_at DESC;

-- Caseload sizing: how many members sit in each tier right now.
CREATE VIEW v_panel_distribution AS
SELECT risk_tier,
       count(*)                                        AS members,
       round(100.0 * count(*) / sum(count(*)) OVER (), 2) AS pct_of_panel,
       round(avg(risk_score)::numeric, 4)              AS mean_score
FROM v_current_risk
GROUP BY risk_tier;

-- Which social drivers are actually moving scores across the panel. This is the
-- view that turns individual scores into a population-health intervention list.
CREATE VIEW v_social_driver_frequency AS
SELECT d.feature,
       d.label,
       count(*)                              AS times_flagged,
       round(avg(d.contribution)::numeric, 4) AS mean_contribution
FROM assessment_drivers d
JOIN assessments a ON a.id = d.assessment_id
WHERE d.feature_group = 'sdoh'
  AND d.direction = 'increases'
  AND a.created_at > now() - interval '90 days'
GROUP BY d.feature, d.label
ORDER BY times_flagged DESC;

-- Feature drift: compare recent input distributions against the previous month.
CREATE VIEW v_feature_drift AS
SELECT date_trunc('week', created_at)                    AS week,
       model_version,
       count(*)                                          AS assessments,
       round(avg((features->>'age')::numeric), 1)        AS mean_age,
       round(avg((features->>'ed_visits_12mo')::numeric), 2)  AS mean_ed_visits,
       round(avg((features->>'medication_count')::numeric), 2) AS mean_medications,
       round(avg(risk_score)::numeric, 4)                AS mean_score
FROM assessments
GROUP BY 1, 2
ORDER BY 1 DESC;
