# Super-Utilizer Risk Console

Predicts whether a patient is likely to become a high healthcare utilizer in the
next 6–12 months, explains why in terms a care manager can act on, and writes a
summary the patient can actually read.

Trained on public/synthetic data. **No PHI anywhere in this repository**, by
design and by structural constraint rather than by policy.

```bash
pip install -r requirements.txt
python -m ml.generate_cohort --n 60000
python -m ml.train
cd web && npm install && npm run build && cd ..
uvicorn api.main:app --port 8000     # → http://localhost:8000
```

## What it does

| | |
|---|---|
| **Input** | 36 de-identified features: conditions, prior utilisation, medications, age, social drivers |
| **Output** | Calibrated probability + risk tier, Shapley attribution, clinical red flags, patient-facing summary |
| **Model** | Isotonic-calibrated classifier selected between LightGBM and regularised logistic regression |
| **Holdout** | AUROC 0.883 · AUPRC 0.512 · Brier 0.041 · 9.3× lift in the top 5% |
| **Latency** | 50–80 ms end to end (template narrator) |

## Documentation

| Document | Covers |
|---|---|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | System design and the five decisions behind it |
| [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md) | Performance, calibration, subgroups, limitations |
| [`docs/DATASETS.md`](docs/DATASETS.md) | What trained it, what to swap in, the PHI position |
| [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) | Running it, and the production checklist |
| [`db/schema.sql`](db/schema.sql) | Canonical PostgreSQL DDL with views |

## Project structure

```
superutilizer/
├── ml/                          Training — offline, never in the request path
│   ├── config.py                Feature contract, label definition, risk tiers.
│   │                            Single source of truth; everything imports it.
│   ├── generate_cohort.py       PHI-free cohort calibrated to published statistics
│   ├── features.py              The ONE feature builder, shared by training and serving
│   ├── train.py                 Candidate selection, calibration, evaluation, bundling
│   ├── explain.py               Shapley attribution + 13 model-independent red-flag rules
│   └── ingest/                  Adapters for real public datasets
│       ├── cms_desynpuf.py      CMS synthetic Medicare claims
│       ├── meps.py              MEPS — the only public source with individual SDOH
│       └── sdoh_ahrq.py         AHRQ area-level social measures
│
├── api/                         Serving
│   ├── main.py                  App, middleware, rate limiting, SPA mount
│   ├── config.py                Environment-driven settings
│   ├── schemas.py               Pydantic contracts + import-time drift assertion
│   ├── scoring.py               Model bundle, prediction, clinician summary
│   ├── narrative.py             Patient summary: Claude → validated → template fallback
│   ├── models_orm.py            SQLAlchemy models (no PHI columns exist)
│   ├── db.py                    Engine, sessions, SQLite pragmas
│   └── routers/
│       ├── assessments.py       POST /assessments, GET /assessments[/{id}]
│       └── meta.py              /health, /model, /metrics, /form-schema
│
├── web/                         React 18 + TypeScript (Vite)
│   └── src/
│       ├── App.tsx              Layout and state
│       ├── styles.css           Design tokens and components
│       ├── components/
│       │   ├── IntakeForm.tsx   Rendered from the server's form schema
│       │   ├── RiskBand.tsx     Continuous band with tier cut points as ticks
│       │   ├── DotMatrix.tsx    100 squares — frequency, not just a percentage
│       │   ├── Attribution.tsx  Driver bars + clinical/social split
│       │   ├── RedFlags.tsx     Severity-ranked, with the action to take
│       │   └── PatientLetter.tsx  Serif, warmer stock — a letter, not a chart
│       └── lib/                 API client, types, demo presets
│
├── db/schema.sql                PostgreSQL DDL, constraints, triggers, views
├── tests/test_system.py         32 tests
├── docs/                        Architecture, model card, datasets, deployment
├── Dockerfile                   3 stages: SPA build → training → slim runtime
└── docker-compose.yml           API + PostgreSQL
```

## API

```
GET  /api/v1/health          liveness, model status, narrative provider
GET  /api/v1/model           version, tier cuts, holdout metrics
GET  /api/v1/metrics         full evaluation report incl. subgroups
GET  /api/v1/form-schema     field definitions — the UI renders from this
POST /api/v1/assessments     score a member
GET  /api/v1/assessments     list, filterable by member or tier
GET  /api/v1/assessments/id  full stored assessment
```

Interactive docs at `/docs`.

## Design decisions worth knowing before you read the code

**Red flags never consult the model score.** Thirteen deterministic clinical
rules run against the raw record. A low score cannot suppress a safety signal,
because models drift and are sometimes just wrong.

**The template narrator is the floor, not a stub.** If the LLM key is missing,
the call times out, or output fails validation, the patient still gets a complete
correct summary. A patient-facing health feature that can go blank because a
third-party API had a bad minute is not a production feature.

**Probabilities are bounded to what the data supports.** Isotonic calibration
saturates; an early build returned 100% risk and told a patient "about 100 out of
100 people need urgent hospital care." Bounds are now derived from observed event
rates in the calibration tails.

**LightGBM did not win.** It was trained as a real candidate against regularised
logistic regression. A paired bootstrap put the AUPRC difference at −0.004, CI
[−0.012, +0.005] — indistinguishable — so the selection rule keeps the simpler
model. On real claims data the booster usually wins and the harness will pick it
up automatically.

**The database has nowhere to put PHI.** No name, DOB, MRN, phone, email or
address column exists; ages above 89 are capped; a test walks every column in the
metadata and fails if one is named after an identifier.

## Tests

```bash
pytest tests/ -v     # 32 passing
```

The ones that matter most: training/serving skew, probability bounds, red-flag
independence, driver direction correctness, and the structural PHI assertion.

## Status

This is a demo build, complete and working end to end, but not clinically
validated. `docs/DEPLOYMENT.md` has the production checklist — the short version
is that it needs real authentication, a real dataset, and clinical sign-off on
the rules before it goes anywhere near a patient.
