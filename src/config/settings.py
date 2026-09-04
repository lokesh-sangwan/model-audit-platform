from importlib import import_module


try:
    _settings_module = import_module("pydantic_settings")
    BaseSettings = _settings_module.BaseSettings
    SettingsConfigDict = _settings_module.SettingsConfigDict
except ImportError:
    from pydantic import BaseSettings

    SettingsConfigDict = dict


class Settings(BaseSettings):
    # Application
    app_name: str = "ML Audit & Deployment Readiness Platform"
    app_env: str = "development"
    log_level: str = "INFO"

    # Data processing
    test_size: float = 0.20
    random_state: int = 42

    # Deployment decision thresholds
    deploy_threshold: float = 0.80
    monitor_threshold: float = 0.60

    # Data drift thresholds
    drift_monitor_threshold: float = 0.20
    drift_block_threshold: float = 0.50

    # Explainability
    shap_sample_size: int = 50

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )


settings = Settings()