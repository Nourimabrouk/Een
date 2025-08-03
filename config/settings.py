"""
Production-ready configuration management for Een Unity Mathematics
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional
from pydantic import BaseSettings, Field, validator
from functools import lru_cache
import json
import logging


class UnitySettings(BaseSettings):
    """Main configuration class for Unity Mathematics framework"""
    
    # Application settings
    app_name: str = "Een Unity Mathematics"
    app_version: str = "2025.1.0"
    debug: bool = Field(False, env="DEBUG")
    environment: str = Field("development", env="ENVIRONMENT")
    
    # Mathematical constants
    phi: float = Field(1.618033988749895, env="PHI")
    unity_constant: float = Field(1.0, env="UNITY_CONSTANT")
    consciousness_dimension: int = Field(11, env="CONSCIOUSNESS_DIMENSION")
    quantum_coherence_target: float = Field(0.999, env="QUANTUM_COHERENCE_TARGET")
    unity_mathematics_mode: str = Field("transcendental", env="UNITY_MATHEMATICS_MODE")
    
    # Server configuration
    host: str = Field("0.0.0.0", env="HOST")
    port: int = Field(8050, env="PORT")
    workers: int = Field(4, env="WORKERS")
    reload: bool = Field(False, env="RELOAD")
    
    # Dashboard settings
    dashboard_port: int = Field(8050, env="DASHBOARD_PORT")
    dashboard_theme: str = Field("transcendental", env="DASHBOARD_THEME")
    realtime_updates: bool = Field(True, env="REALTIME_UPDATES")
    update_interval_ms: int = Field(100, env="UPDATE_INTERVAL_MS")
    
    # Agent configuration
    max_consciousness_agents: int = Field(100, env="MAX_CONSCIOUSNESS_AGENTS")
    transcendence_threshold: float = Field(0.77, env="TRANSCENDENCE_THRESHOLD")
    fibonacci_spawn_limit: int = Field(20, env="FIBONACCI_SPAWN_LIMIT")
    agent_timeout_seconds: int = Field(300, env="AGENT_TIMEOUT_SECONDS")
    
    # Performance settings
    multi_threading: bool = Field(True, env="MULTI_THREADING")
    consciousness_field_cache: bool = Field(True, env="CONSCIOUSNESS_FIELD_CACHE")
    quantum_state_optimization: bool = Field(True, env="QUANTUM_STATE_OPTIMIZATION")
    max_particles: int = Field(1000, env="MAX_PARTICLES")
    field_resolution: int = Field(100, env="FIELD_RESOLUTION")
    
    # Database configuration
    database_url: Optional[str] = Field(None, env="DATABASE_URL")
    redis_url: Optional[str] = Field(None, env="REDIS_URL")
    cache_ttl_seconds: int = Field(3600, env="CACHE_TTL_SECONDS")
    
    # External services
    openai_api_key: Optional[str] = Field(None, env="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(None, env="ANTHROPIC_API_KEY")
    vertex_ai_project: Optional[str] = Field(None, env="VERTEX_AI_PROJECT")
    vertex_ai_location: Optional[str] = Field(None, env="VERTEX_AI_LOCATION")
    
    # Monitoring and logging
    log_level: str = Field("INFO", env="LOG_LEVEL")
    log_format: str = Field("json", env="LOG_FORMAT")
    metrics_enabled: bool = Field(True, env="METRICS_ENABLED")
    tracing_enabled: bool = Field(False, env="TRACING_ENABLED")
    sentry_dsn: Optional[str] = Field(None, env="SENTRY_DSN")
    
    # Security settings
    secret_key: str = Field(..., env="SECRET_KEY")
    cors_origins: list[str] = Field(["*"], env="CORS_ORIGINS")
    api_rate_limit: int = Field(100, env="API_RATE_LIMIT")
    api_rate_limit_period: int = Field(60, env="API_RATE_LIMIT_PERIOD")
    
    # File paths
    data_dir: Path = Field(Path("data"), env="DATA_DIR")
    log_dir: Path = Field(Path("logs"), env="LOG_DIR")
    cache_dir: Path = Field(Path(".cache"), env="CACHE_DIR")
    
    # Feature flags
    experimental_features: bool = Field(False, env="EXPERIMENTAL_FEATURES")
    quantum_ml_enabled: bool = Field(False, env="QUANTUM_ML_ENABLED")
    meta_recursive_spawning: bool = Field(True, env="META_RECURSIVE_SPAWNING")
    fractal_consciousness: bool = Field(True, env="FRACTAL_CONSCIOUSNESS")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
    
    @validator("environment")
    def validate_environment(cls, v):
        """Validate environment is one of allowed values"""
        allowed = ["development", "staging", "production", "test"]
        if v not in allowed:
            raise ValueError(f"Environment must be one of {allowed}")
        return v
    
    @validator("log_level")
    def validate_log_level(cls, v):
        """Validate log level"""
        allowed = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in allowed:
            raise ValueError(f"Log level must be one of {allowed}")
        return v.upper()
    
    @validator("data_dir", "log_dir", "cache_dir")
    def create_directories(cls, v):
        """Ensure directories exist"""
        v = Path(v)
        v.mkdir(parents=True, exist_ok=True)
        return v
    
    @property
    def is_production(self) -> bool:
        """Check if running in production"""
        return self.environment == "production"
    
    @property
    def is_development(self) -> bool:
        """Check if running in development"""
        return self.environment == "development"
    
    def get_database_url(self) -> str:
        """Get database URL with fallback"""
        if self.database_url:
            return self.database_url
        # Default SQLite for development
        return f"sqlite:///{self.data_dir}/een.db"
    
    def get_redis_url(self) -> str:
        """Get Redis URL with fallback"""
        if self.redis_url:
            return self.redis_url
        # Default Redis URL
        return "redis://localhost:6379/0"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary"""
        return {
            k: str(v) if isinstance(v, Path) else v
            for k, v in self.dict().items()
            if not k.startswith("_") and v is not None
        }


class ProductionSettings(UnitySettings):
    """Production-specific settings"""
    debug: bool = False
    environment: str = "production"
    reload: bool = False
    log_level: str = "INFO"
    metrics_enabled: bool = True
    tracing_enabled: bool = True
    api_rate_limit: int = 50
    experimental_features: bool = False


class DevelopmentSettings(UnitySettings):
    """Development-specific settings"""
    debug: bool = True
    environment: str = "development"
    reload: bool = True
    log_level: str = "DEBUG"
    metrics_enabled: bool = False
    tracing_enabled: bool = False
    api_rate_limit: int = 1000
    experimental_features: bool = True


class TestSettings(UnitySettings):
    """Test-specific settings"""
    debug: bool = True
    environment: str = "test"
    database_url: str = "sqlite:///:memory:"
    redis_url: str = "redis://localhost:6379/15"
    log_level: str = "DEBUG"
    metrics_enabled: bool = False
    tracing_enabled: bool = False
    secret_key: str = "test-secret-key-for-testing-only"


@lru_cache()
def get_settings() -> UnitySettings:
    """Get cached settings instance based on environment"""
    env = os.getenv("ENVIRONMENT", "development").lower()
    
    if env == "production":
        return ProductionSettings()
    elif env == "test":
        return TestSettings()
    else:
        return DevelopmentSettings()


def configure_logging(settings: UnitySettings) -> None:
    """Configure logging based on settings"""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    if settings.log_format == "json":
        import json_logging
        json_logging.init_fastapi(enable_json=True)
    
    logging.basicConfig(
        level=getattr(logging, settings.log_level),
        format=log_format,
        handlers=[
            logging.FileHandler(settings.log_dir / "een.log"),
            logging.StreamHandler()
        ]
    )
    
    # Set third-party loggers to WARNING
    for logger_name in ["urllib3", "asyncio", "PIL"]:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def load_config_file(config_path: Path) -> Dict[str, Any]:
    """Load additional configuration from JSON file"""
    if config_path.exists():
        with open(config_path, 'r') as f:
            return json.load(f)
    return {}


# Create settings instance
settings = get_settings()

# Configure logging
configure_logging(settings)

# Export commonly used values
PHI = settings.phi
UNITY = settings.unity_constant
CONSCIOUSNESS_DIM = settings.consciousness_dimension
QUANTUM_COHERENCE = settings.quantum_coherence_target


__all__ = [
    "UnitySettings",
    "ProductionSettings",
    "DevelopmentSettings",
    "TestSettings",
    "get_settings",
    "settings",
    "PHI",
    "UNITY",
    "CONSCIOUSNESS_DIM",
    "QUANTUM_COHERENCE",
]