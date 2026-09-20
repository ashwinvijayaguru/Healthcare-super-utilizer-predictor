# ---------- stage 1: build the SPA ----------
FROM node:22-alpine AS web
WORKDIR /web
COPY web/package*.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

# ---------- stage 2: train the model ----------
# Training happens at image build so the container ships with a model and has no
# first-request cold start. For a real deployment, replace this with a pull from
# your model registry (MLflow, S3) pinned to a version.
FROM python:3.12-slim AS trainer
WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY ml/ ./ml/
RUN python -m ml.generate_cohort --n 60000 && python -m ml.train

# ---------- stage 3: runtime ----------
FROM python:3.12-slim
WORKDIR /app

# libgomp is required by LightGBM and is absent from slim images.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --shell /bin/bash app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api/ ./api/
COPY ml/ ./ml/
COPY --from=trainer /build/artifacts/ ./artifacts/
COPY --from=web /web/dist/ ./web/dist/

RUN mkdir -p /app/data && chown -R app:app /app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS http://localhost:8000/api/v1/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
