from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration, populated from environment variables / .env."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    environment: str = "development"
    database_url: str = "postgresql+asyncpg://docmind:docmind@postgres:5432/docmind"
    redis_url: str = "redis://redis:6379/0"
    secret_key: str = "change-me-in-production"

    # Will be used starting Phase 1 (auth) and Phase 4 (LLM calls).
    access_token_expire_minutes: int = 60
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None

    # MinIO Storage Settings
    minio_endpoint: str = "minio:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_bucket_name: str = "docmind-files"
    minio_secure: bool = False


@lru_cache
def get_settings() -> Settings:
    """Cached settings instance, so we only parse the environment once."""
    return Settings()
