"""Application entry point.

    uvicorn api.main:app --reload
"""

from __future__ import annotations

import logging
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from api.config import get_settings
from api.db import init_db, session_scope
from api.models_orm import ModelRegistry
from api.routers import assessments, meta
from api.scoring import RiskScorer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("api")

PUBLIC_PATHS = {"/api/v1/health", "/docs", "/redoc", "/openapi.json"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    try:
        scorer = RiskScorer.instance()
        with session_scope() as session:
            # Record the live model so a score can always be traced to a version.
            existing = session.query(ModelRegistry).filter_by(
                model_version=scorer.bundle["model_version"], is_active=1).first()
            if existing is None:
                session.query(ModelRegistry).update({"is_active": 0})
                session.add(ModelRegistry(
                    model_name=scorer.bundle["model_name"],
                    model_version=scorer.bundle["model_version"],
                    feature_schema_version=scorer.bundle["feature_schema_version"],
                    estimator_kind=scorer.bundle.get("estimator_kind", "unknown"),
                    trained_at=scorer.bundle["trained_at"],
                    holdout_metrics=scorer.bundle.get("holdout_metrics", {}),
                    tier_cuts=scorer.bundle["tier_cuts"],
                    is_active=1))
        log.info("model %s ready", scorer.bundle["model_version"])
    except Exception as exc:  # noqa: BLE001
        # Boot without a model so /health can report the problem instead of the
        # container crash-looping with the reason buried in the logs.
        log.error("model unavailable at startup: %s", exc)
    yield


settings = get_settings()
app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description=(
        "Predicts the probability that a member becomes a high healthcare utilizer "
        "in the next 6-12 months, with per-factor attribution, clinical red flags "
        "and a plain-language patient summary. Operates on de-identified input only."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

_request_log: dict[str, deque[float]] = defaultdict(deque)


@app.middleware("http")
async def observability_and_limits(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    started = time.perf_counter()

    path = request.url.path
    if path.startswith("/api/") and path not in PUBLIC_PATHS:
        # Simple in-process limiter. Behind more than one worker, move this to
        # Redis or the ingress - it is per-process only.
        client = request.client.host if request.client else "unknown"
        window = _request_log[client]
        now = time.monotonic()
        while window and now - window[0] > 60:
            window.popleft()
        if len(window) >= settings.rate_limit_per_minute:
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={"detail": "rate limit exceeded", "request_id": request_id},
            )
        window.append(now)

        if settings.api_key and request.method != "OPTIONS":
            if request.headers.get("x-api-key") != settings.api_key:
                return JSONResponse(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    content={"detail": "invalid or missing API key", "request_id": request_id},
                )

    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - started) * 1000
    response.headers["x-request-id"] = request_id
    response.headers["x-response-time-ms"] = f"{elapsed_ms:.1f}"
    log.info("%s %s -> %s in %.1fms", request.method, path, response.status_code, elapsed_ms)
    return response


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    # Never leak internals to a clinical UI; log the detail, return a stable shape.
    log.exception("unhandled error on %s", request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "internal error", "request_id": request.headers.get("x-request-id")},
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "request_id": request.headers.get("x-request-id")},
    )


app.include_router(meta.router)
app.include_router(assessments.router)

# Serve the built SPA if it is present, so one container can host the whole demo.
_static_dir = Path(__file__).resolve().parents[1] / "web" / "dist"
if _static_dir.exists():
    app.mount("/", StaticFiles(directory=str(_static_dir), html=True), name="web")
