"""Configuration settings for LM Router."""

from pathlib import Path
from typing import Optional

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ServerSettings(BaseSettings):
    """Server configuration."""

    host: str = "0.0.0.0"
    port: int = 8000


class LMStudioSettings(BaseSettings):
    """LM Studio API configuration."""

    model_config = SettingsConfigDict(env_prefix="LMSTUDIO_")

    base_url: str = "http://localhost:1234"
    timeout: int = 300
    refresh_interval: int = 60


class RouterSettings(BaseSettings):
    """Router-specific settings."""

    model: str = "microsoft/phi-4"
    prefer_loaded_bonus: int = 50
    auto_load_models: bool = True


class PerformanceTrackingSettings(BaseSettings):
    """Performance tracking configuration."""

    enabled: bool = True
    sample_size: int = 10


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
    logging: LoggingSettings = Field(default_factory=LoggingSettings)

    @classmethod
    def from_yaml(cls, config_path: Path | str) -> "Settings":
        """Load settings from YAML file."""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path) as f:
            config_data = yaml.safe_load(f)

        return cls(
            server=ServerSettings(**config_data.get("server", {})),
            lm_studio=LMStudioSettings(**config_data.get("lm_studio", {})),
            router=RouterSettings(**config_data.get("router", {})),
            performance_tracking=PerformanceTrackingSettings(
                **config_data.get("performance_tracking", {})
            ),
            logging=LoggingSettings(**config_data.get("logging", {})),
        )


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get or create settings instance."""
    global _settings
    if _settings is None:
        # Try to load from default config location
        config_path = Path(__file__).parent.parent.parent.parent / "config" / "router.yaml"
        if config_path.exists():
            _settings = Settings.from_yaml(config_path)
        else:
            _settings = Settings()
    return _settings


def set_settings(settings: Settings) -> None:
    """Set global settings instance."""
    global _settings
    _settings = settings
