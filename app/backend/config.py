"""Infra connection settings, loaded from .env — deliberately separate from
configs/config.yaml, which holds the research pipeline's domain config (vegetables,
model hyperparameters), not deployment-specific connection strings."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str = "postgresql+asyncpg://postgres:pg123@localhost:5432/vegepredict"
    redis_url: str = "redis://localhost:6379/0"
    cache_ttl_seconds: int = 7 * 24 * 3600  # matches the weekly retrain cadence

    model_config = SettingsConfigDict(env_file=".env", env_prefix="VEGEPREDICT_")


settings = Settings()
