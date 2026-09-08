"""Application settings loaded from environment variables."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

_SERVER_ROOT = Path(__file__).resolve().parents[2]
_REPO_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    """Env-driven settings. Names match docs/contracts.md."""

    model_config = SettingsConfigDict(
        env_file=(
            str(_REPO_ROOT / ".env"),
            str(_SERVER_ROOT / ".env"),
            ".env",
        ),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        protected_namespaces=(),
    )

    APP_ENV: str = "development"
    APP_VERSION: str = "0.1.0"
    CORS_ORIGINS: str = "http://localhost:3000"
    MAX_UPLOAD_BYTES: int = 8_000_000
    STORE_UPLOADS: bool = False
    DEMO_MODE: str = "auto"
    MODEL_CHECKPOINT: str = ""
    LOG_LEVEL: str = "info"

    @property
    def cors_origin_list(self) -> list[str]:
        return [part.strip() for part in self.CORS_ORIGINS.split(",") if part.strip()]

    @property
    def demo_mode_normalized(self) -> str:
        value = (self.DEMO_MODE or "auto").strip().lower()
        if value in {"true", "1", "yes"}:
            return "true"
        if value in {"false", "0", "no"}:
            return "false"
        return "auto"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
