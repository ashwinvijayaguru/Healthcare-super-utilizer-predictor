"""Runtime configuration. Everything is environment-driven; no secrets in code."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="SURISK_", extra="ignore")

    environment: str = "development"
    api_title: str = "Super-Utilizer Risk API"
    api_version: str = "1.2.0"

    # sqlite for the demo; point at postgresql+psycopg://... in any real deployment
    database_url: str = f"sqlite:///{PROJECT_ROOT / 'data' / 'app.db'}"
    model_path: Path = PROJECT_ROOT / "artifacts" / "model.joblib"
    metrics_path: Path = PROJECT_ROOT / "artifacts" / "metrics.json"

    cors_origins: str = "http://localhost:5173,http://localhost:4173,http://localhost:8080"

    # Optional: enables LLM-written patient summaries. Without it the service
    # falls back to a deterministic template - the app never hard-depends on it.
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-sonnet-4-6"
    llm_timeout_seconds: float = 25.0
    llm_max_tokens: int = 1100

    # Static API key for the demo. Replace with OIDC/JWT before production use.
    api_key: str | None = None
    rate_limit_per_minute: int = 120

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
