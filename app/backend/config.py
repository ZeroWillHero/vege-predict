"""Infra connection settings, loaded from .env — deliberately separate from
configs/config.yaml, which holds the research pipeline's domain config (vegetables,
model hyperparameters), not deployment-specific connection strings."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str = "postgresql+asyncpg://postgres:pg123@localhost:5432/vegepredict"
    redis_url: str = "redis://localhost:6379/0"
    cache_ttl_seconds: int = 7 * 24 * 3600  # matches the weekly retrain cadence

    # Dev-only default so local setup works out of the box — production deployments must
    # override VEGEPREDICT_JWT_SECRET with a real random secret via .env, since this default
    # is checked into source control and provides no real security.
    jwt_secret: str = "dev-only-insecure-secret-override-in-production"
    jwt_expire_minutes: int = 60

    model_config = SettingsConfigDict(env_file=".env", env_prefix="VEGEPREDICT_")


settings = Settings()
