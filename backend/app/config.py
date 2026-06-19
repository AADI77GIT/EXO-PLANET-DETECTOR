from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    database_url: str = "postgresql+asyncpg://exo:exo@localhost:5432/exodetector"
    redis_url: str = "redis://localhost:6379/0"
    api_key: str = "change-me"
    data_dir: Path = Path("data")
    model_dir: Path = Path("models")
    max_upload_mb: int = 50
    detrending_method: str = "wotan biweight"
    wotan_window_length_days: float = 0.3
    bls_min_period_days: float = 0.5
    bls_max_period_days: float = 15.0
    bls_period_samples: int = 5000

    @property
    def fits_dir(self) -> Path:
        return self.data_dir / "fits"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def results_dir(self) -> Path:
        return self.data_dir / "results"


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    for path in (settings.data_dir, settings.fits_dir, settings.processed_dir, settings.results_dir, settings.model_dir):
        path.mkdir(parents=True, exist_ok=True)
    return settings


settings = get_settings()
