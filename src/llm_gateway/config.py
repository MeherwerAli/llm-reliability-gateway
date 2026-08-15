from decimal import Decimal
from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="GATEWAY_", env_file=".env", extra="ignore")

    env: Literal["development", "test", "production"] = "development"
    state_backend: Literal["memory", "distributed"] = "memory"
    api_key: str = "portfolio-demo-key-change-me"
    monthly_budget_usd: Decimal = Decimal("10.00")
    requests_per_minute: int = Field(default=60, ge=1)
    tokens_per_minute: int = Field(default=20_000, ge=1)
    provider: Literal["deterministic", "openai-compatible"] = "deterministic"
    provider_base_url: str = "http://localhost:8000"
    provider_api_key: str | None = None
    provider_timeout_seconds: float = Field(default=30.0, gt=0, le=300)
    database_url: str = "postgresql://gateway:gateway@localhost:5432/gateway"
    redis_url: str = "redis://localhost:6379/0"
    allow_http_webhooks: bool = False


@lru_cache
def get_settings() -> Settings:
    return Settings()
