from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Production Qwen Serving Gateway"
    upstream_base_url: str = "http://vllm:8000"
    upstream_api_key: str = "change-me-upstream"
    public_api_keys: str = "change-me-public"
    allowed_models: str = "tool-calling"
    readiness_model: str = "tool-calling"
    max_concurrency: int = 8
    queue_timeout_seconds: float = 0.25
    upstream_timeout_seconds: float = 120.0
    max_request_bytes: int = 1_000_000
    tool_policy_enabled: bool = True
    tool_policy_models: str = "tool-calling"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @property
    def public_api_key_set(self) -> tuple[str, ...]:
        return tuple(item.strip() for item in self.public_api_keys.split(",") if item.strip())

    @property
    def allowed_model_set(self) -> frozenset[str]:
        return frozenset(item.strip() for item in self.allowed_models.split(",") if item.strip())

    @property
    def tool_policy_model_set(self) -> frozenset[str]:
        return frozenset(
            item.strip() for item in self.tool_policy_models.split(",") if item.strip()
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
