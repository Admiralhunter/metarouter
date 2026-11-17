"""Configuration settings for LM Router."""

from pathlib import Path
from typing import Annotated, Optional

import yaml
from pydantic import BaseModel, Field, NonNegativeInt, PositiveInt, StringConstraints, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ServerSettings(BaseSettings):
    """Server configuration."""

    host: str = "0.0.0.0"
    port: int = 8000


class LMStudioInstanceConfig(BaseModel):
    """Configuration for a single LM Studio instance."""

    name: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)] = "default"
    base_url: str = "http://localhost:1234"
    timeout: PositiveInt = 300
    refresh_interval: NonNegativeInt = 60


class LMStudioSettings(BaseSettings):
    """LM Studio API configuration.

    Supports both single-instance (backward compatible) and multi-instance configs.
    Single instance: set base_url directly.
    Multi-instance: provide a list of instances.
    """

    model_config = SettingsConfigDict(env_prefix="LMSTUDIO_")

    base_url: str = "http://localhost:1234"
    timeout: PositiveInt = 300
    refresh_interval: NonNegativeInt = 60
    instances: list[LMStudioInstanceConfig] = Field(default_factory=list)
    health_check_interval: PositiveInt = 30

    @classmethod
    def settings_customise_sources(
        cls, settings_cls, init_settings, env_settings, dotenv_settings, file_secret_settings
    ):
        return env_settings, init_settings, dotenv_settings, file_secret_settings

    @field_validator("instances")
    @classmethod
    def require_unique_instance_names(cls, instances: list[LMStudioInstanceConfig]):
        if len({instance.name for instance in instances}) != len(instances):
            raise ValueError("LM Studio instance names must be unique")
        return instances

    def get_instances(self) -> list[LMStudioInstanceConfig]:
        """Get resolved list of instances.

        If explicit instances are configured, return those.
        Otherwise, create a single instance from the base_url/timeout/refresh_interval.
        """
        if self.instances:
            return self.instances
        return [
            LMStudioInstanceConfig(
                name="default",
                base_url=self.base_url,
                timeout=self.timeout,
                refresh_interval=self.refresh_interval,
            )
        ]


class RouterSettings(BaseSettings):
    """Router-specific settings."""

    model: str = "microsoft/phi-4"
    prefer_loaded_bonus: int = 50
    auto_load_models: bool = True


class PerformanceTrackingSettings(BaseSettings):
    """Performance tracking configuration."""

    enabled: bool = True
    sample_size: int = 10


class BenchmarkSettings(BaseSettings):
    """Benchmark data configuration."""

    enabled: bool = True  # Enable benchmark data in routing context
    cache_ttl_hours: int = 24  # Hours before cache is considered stale
    auto_fetch_missing: bool = True  # Fetch when encountering unknown models
    api_timeout: int = 30  # Timeout for API requests


class LoggingSettings(BaseSettings):
    """Logging configuration."""

    level: str = "INFO"
    log_routing_decisions: bool = True


class Settings(BaseSettings):
    """Main application settings."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    server: ServerSettings = Field(default_factory=ServerSettings)
    lm_studio: LMStudioSettings = Field(default_factory=LMStudioSettings)
    router: RouterSettings = Field(default_factory=RouterSettings)
    performance_tracking: PerformanceTrackingSettings = Field(
        default_factory=PerformanceTrackingSettings
    )
    benchmarks: BenchmarkSettings = Field(default_factory=BenchmarkSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)

    @classmethod
    def from_yaml(cls, config_path: Path | str) -> "Settings":
        """Load settings from YAML file."""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path) as f:
            config_data = yaml.safe_load(f) or {}

        return cls(
            server=ServerSettings(**config_data.get("server", {})),
            lm_studio=LMStudioSettings(**config_data.get("lm_studio", {})),
            router=RouterSettings(**config_data.get("router", {})),
            performance_tracking=PerformanceTrackingSettings(
                **config_data.get("performance_tracking", {})
            ),
            benchmarks=BenchmarkSettings(**config_data.get("benchmarks", {})),
            logging=LoggingSettings(**config_data.get("logging", {})),
        )


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get or create settings instance."""
    global _settings
    if _settings is None:
        config_path = Path("config/router.yaml")
        if not config_path.is_file():
            config_path = Path(__file__).resolve().parents[3] / config_path
        if config_path.exists():
            _settings = Settings.from_yaml(config_path)
        else:
            _settings = Settings()
    return _settings


def set_settings(settings: Settings) -> None:
    """Set global settings instance."""
    global _settings
    _settings = settings
