# System Architecture

## What the system does

Given a de-identified clinical and social record for one member, it returns a
calibrated probability that they will become a high healthcare utilizer within
6–12 months, the specific factors driving that probability, any clinical red
flags, and a summary written for the patient to read.

**Outcome definition.** A member is a super utilizer in the follow-up window if
**any** of: ≥4 emergency department visits, **or** ≥2 inpatient admissions,
**or** total allowed cost in the top 5% of the panel. This mirrors the
operational definitions used by CMMI's Health Care Innovation Awards and the
Camden Coalition hotspotting programme. It is defined once, in `ml/config.py`,
and every other component imports it.

## Shape of the system

```
┌──────────────────────────────────────────────────────────────────────┐
│  Browser — React 18 + TypeScript (Vite)                              │
│  Intake flowsheet · risk band · attribution · flags · patient letter │
└─────────────────────────────┬────────────────────────────────────────┘
                              │  JSON over HTTPS
┌─────────────────────────────▼────────────────────────────────────────┐
│  FastAPI                                                             │
│  ┌────────────┐ ┌───────────┐ ┌────────────┐ ┌──────────────────┐    │
│  │ validation │ │  scoring  │ │ explainer  │ │ narrative        │    │
│  │ (Pydantic) │ │ (bundle)  │ │ (Shapley)  │ │ Claude→template  │    │
│  └────────────┘ └───────────┘ └────────────┘ └──────────────────┘    │
│  ┌──────────────────────────────────────────────────────────────┐    │
│  │ red-flag rules — evaluated independently of the model score   │    │
│  └──────────────────────────────────────────────────────────────┘    │
└──────┬────────────────────────────────────────────────┬──────────────┘
       │ SQLAlchemy                                     │ HTTPS (optional)
┌──────▼──────────────────────┐              ┌──────────▼───────────────┐
│ PostgreSQL (SQLite in demo) │              │ Anthropic API            │
│ patients · assessments      │              │ patient-facing narrative │
│ drivers · flags · audit     │              └──────────────────────────┘
│ model_registry              │
└─────────────────────────────┘
       ▲
       │ offline, not in the request path
┌──────┴──────────────────────────────────────────────────────────────┐
│ Training — cohort builder / public-data adapters → features →       │
│ candidate selection → isotonic calibration → evaluation → bundle    │
└─────────────────────────────────────────────────────────────────────┘
```

## The five decisions that shape this design

### 1. One feature contract, imported everywhere

`ml/config.py` holds the feature list, the label definition and the risk tiers.
The cohort builder, the trainer, the API schema and the web form all derive from
it. `api/schemas.py` asserts alignment **at import time** — the process refuses
to boot if the API schema and the training contract have drifted apart.

Training/serving skew is the most common way a model that works in a notebook
misbehaves in production. `ml/features.py` is the only implementation of feature
construction, it holds no fitted state, and it is row-independent. A test scores
a row alone and inside a batch and asserts the vectors are identical.

### 2. Red flags are computed separately from the score

The rule engine in `ml/explain.py` never reads the model output. A patient on an
anticoagulant with a 0.42 adherence ratio gets a critical flag whether the model
says 3% or 80%.

Models drift, are miscalibrated on subgroups, and are sometimes simply wrong. A
system where a low score can silence a safety signal has built a single point of
failure into the part that matters most. Thirteen rules cover frequent ED use,
the post-discharge window, adherence on high-risk medications, severe
polypharmacy, opioid therapy with SUD, unstable housing with a
condition needing controlled conditions, compound social risk, transport-driven
care gaps, unmanaged multimorbidity, repeat falls without support, SMI with
isolation, cognitive impairment with a complex regimen, and food insecurity with
a diet-sensitive condition.

### 3. Explanations are exact, not narrated

Driver attributions are Shapley values — closed-form for the linear pipeline,
TreeSHAP for the boosted model — not a plausible-sounding story assembled after
the fact from feature values. Two consequences the implementation handles:

- **Derived features are only folded onto a parent when they point the same way
  clinically.** An early version folded a composite "care continuity" feature
  onto the primary-care flag and rendered *"has an established primary care
  provider — increases risk"*, which is both wrong and the kind of output that
  ends clinician trust in a tool permanently.
- **Binary features carry explicit absent-state labels.** A feature at 0 shows
  as "No established primary care provider", never as its affirmative name.

### 4. The LLM is the enhancement, the template is the floor

The patient narrative has two providers behind one interface. `ClaudeNarrator`
sends **only the already-computed structured facts** — never free text, never
the raw record — with instructions to rephrase and not to add. The output is
validated field by field, including a banned-term check, before use.

`TemplateNarrator` is deterministic, needs no network and produces a complete
summary. If the key is absent, the call times out, or validation fails, the
service falls back silently and the user still gets a correct summary. A
patient-facing health feature that can go blank because a third-party API had a
bad minute is not a production feature.

### 5. The database has nowhere to put PHI

Not a policy promise — a structural one. There is no name, DOB, MRN, phone,
email or address column in any table, ages above 89 are capped before writing,
`external_ref` rejects anything containing `@`, and a test iterates every column
in the metadata asserting none is named after an identifier. Assessments are
append-only, enforced by a trigger, because they are an audit record.

## Request path

1. **Validate** — Pydantic checks ranges and cross-field consistency
   (`high_risk_med_count ≤ medication_count`; inpatient days require an admission).
2. **Cap age** at 90 for Safe Harbor.
3. **Build features** — 36 contract features plus 13 derived, one deterministic function.
4. **Predict** — calibrated probability, then bounded (see below).
5. **Explain** — Shapley attribution, folded to human-recognisable labels, grouped
   into clinical / social / medication / utilisation shares.
6. **Flag** — 13 rules against the raw record.
7. **Narrate** — Claude if configured, template otherwise.
8. **Persist** — assessment, drivers, flags and an audit row in one transaction.

Measured end-to-end latency is 50–80 ms with the template narrator. With the LLM
narrator, expect 2–5 s dominated by the API call.

## Probability bounds

Isotonic calibration is a step function and emits exactly 0.0 and 1.0 at the
tails. The first working version of this service returned **100.0%** risk and a
patient summary reading *"about 100 out of 100 people need urgent hospital
care."* That is false — even the top calibration bin has a 78% observed event
rate — and it is harmful to show a patient.

Bounds are now derived from what the calibration data supports: the ceiling is
the observed event rate among the highest-scoring 2% of the calibration split
plus Wilson-style padding (0.881 on the current model), the floor is similarly
derived. The unbounded value is still returned as `raw_score` for audit.

A consequence worth stating plainly: at the ceiling the score stops
discriminating. That is correct behaviour, not a defect — anyone at the ceiling
is already in the top tier and the operational decision is the same — but it
means the extreme end cannot be used for fine-grained ranking.

## Deliberately not built

- **Batch panel scoring.** The model is a pure function of a feature row; a nightly
  Spark or dbt job over the member table is the right shape, and the request path
  is the wrong one for a million members.
- **Model monitoring.** `v_feature_drift` and `v_social_driver_frequency` exist as
  views, but nothing schedules them or alerts on them.
- **Real authentication.** A static API key guards `/api/*`. Production needs OIDC
  with per-user identity, since `created_by` is currently whatever the caller puts
  in a header.
- **Distributed rate limiting.** The limiter is in-process and per-worker. Behind
  more than one worker, move it to Redis or the ingress.
