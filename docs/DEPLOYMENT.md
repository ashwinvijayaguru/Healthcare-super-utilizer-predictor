# Deployment and operations

## Quick start (demo, ~4 minutes)

Requires Python 3.11+ and Node 18+.

```bash
git clone <repo> && cd superutilizer

python -m venv .venv && .venv\Scripts\activate.bat   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m ml.generate_cohort --n 60000    # ~15 s  → data/cohort.parquet
python -m ml.train                        # ~60 s  → artifacts/model.joblib

cd web && npm install && npm run build && cd ..  # removed install && npm as I have installed the npm

uvicorn api.main:app --port 8000
```

Open <http://localhost:8000>. The API serves the built SPA, so this is the whole
demo on one port with no external services.

Verify:

```bash
curl -s localhost:8000/api/v1/health
# {"status":"ok","model_loaded":true,"database":true,
#  "narrative_provider":"template","version":"1.2.0"}
```

`make demo` runs all of the above.

## Development (two processes, hot reload)

```bash
uvicorn api.main:app --reload --port 8000    # terminal 1
cd web && npm run dev                        # terminal 2 → localhost:5173
```

Vite proxies `/api` to port 8000, so the browser sees one origin and CORS never
enters the picture locally.

## Docker

```bash
docker compose up --build
```

Three stages: build the SPA, train the model, assemble a slim runtime. Training
at build time means the container ships with a model and has no first-request
cold start. For a real deployment replace stage 2 with a pull from your model
registry pinned to a version — you do not want a production image whose model
depends on when it happened to be built.

Compose brings up PostgreSQL alongside and applies `db/schema.sql` on first boot.

Two things that bite:
- **`libgomp1` is required by LightGBM** and absent from `python:*-slim`. The
  Dockerfile installs it. Without it you get an opaque import error.
- The container runs as a non-root `app` user; `/app/data` is chowned to it.

## Enabling LLM patient summaries

```bash
export SURISK_ANTHROPIC_API_KEY=sk-ant-...
```

Without it, summaries come from the deterministic template and everything still
works — `/api/v1/health` reports which provider is live. The LLM path adds 2–5 s
of latency; the template path is ~1 ms.

## Production checklist

Do not put this in front of real patients without working through the following.
It is a demo build; several of these are deliberate gaps, not oversights.

**Must do**

- [ ] **Replace the static API key with OIDC/JWT.** `created_by` is currently
      whatever the caller puts in an `x-actor` header, which makes the audit log
      untrustworthy for anything that matters.
- [ ] **Move to PostgreSQL** and manage schema with Alembic rather than
      `create_all`. `db/schema.sql` is the canonical DDL.
- [ ] **Move rate limiting to Redis or the ingress.** The in-process limiter is
      per-worker, so N workers means N× the intended limit.
- [ ] **TLS everywhere**, and set `SURISK_CORS_ORIGINS` to your real origins.
- [ ] **Clinical sign-off on the red-flag rules and the tier actions.** These are
      written from published practice but they encode care decisions and need a
      named clinical owner.
- [ ] **Retrain on real data.** The shipped model is trained on a synthetic
      cohort and its coefficients should not drive real outreach.

**Should do**

- [ ] Schedule `v_feature_drift` and alert when input distributions or the mean
      score move materially — the usual first sign that upstream data changed.
- [ ] Track observed outcome rates per tier against predicted, quarterly. The
      model's honesty claim rests on calibration, and calibration decays.
- [ ] Re-run subgroup metrics on real data (see MODEL_CARD.md). The synthetic
      run already shows the model under-predicting in low-deprivation areas
      (calibration ratio 0.56); that pattern needs checking, not assuming away.
- [ ] Set up model versioning properly — MLflow or S3 with immutable version
      pinning, and a rollback path.
- [ ] Add batch scoring. Scoring a million-member panel through the request path
      is the wrong shape; the model is a pure function of a feature row and
      belongs in a nightly job.

**Consider**

- [ ] Human review queue before any patient summary is actually sent.
- [ ] Recording care-manager overrides as training signal for the next model.
- [ ] Adding a "why was I flagged" endpoint for member-facing appeals — the full
      input snapshot in `assessments.features` already makes this answerable.

## Configuration

All settings are environment variables prefixed `SURISK_`. See `.env.example`.

| Variable | Default | Notes |
|---|---|---|
| `SURISK_DATABASE_URL` | SQLite in `./data` | `postgresql+psycopg://…` in production |
| `SURISK_MODEL_PATH` | `./artifacts/model.joblib` | |
| `SURISK_CORS_ORIGINS` | localhost dev ports | comma-separated |
| `SURISK_ANTHROPIC_API_KEY` | unset | enables LLM summaries |
| `SURISK_API_KEY` | unset | when set, `/api/*` requires `x-api-key` |
| `SURISK_RATE_LIMIT_PER_MINUTE` | 120 | per process |

## Retraining

```bash
python -m ml.generate_cohort --n 100000 --seed 7   # or use an ml/ingest adapter
python -m ml.train
pytest tests/ -v
```

`ml/train.py` retrains both candidates, re-runs the paired bootstrap, re-derives
the tier cut points and probability bounds, and writes `artifacts/metrics.json`.
Bump `MODEL_VERSION` in `ml/config.py`; the service registers the new version in
`model_registry` at boot and deactivates the previous one.

Do not skip the test run. `test_api_schema_matches_training_contract` and
`test_feature_builder_is_row_independent` are the two that catch the failures
which are otherwise silent in production.

## Troubleshooting

| Symptom | Cause |
|---|---|
| `/health` shows `model_loaded: false` | No `artifacts/model.joblib`. Run the training steps. The app boots anyway so health can report it rather than crash-looping. |
| `RuntimeError: intake schema does not match the trained feature contract` | `ml/config.py` and `api/schemas.py` diverged. Intentional — fix the mismatch. |
| SPA 404s at `/` | `web/dist` missing. Run `npm run build`. |
| LightGBM `ImportError` in Docker | `libgomp1` not installed. |
| Every condition inverted after a DE-SynPUF run | Its flags are `1 = has`, `2 = does not`. |
