"""
Configuration Management System for Local Chatbot API
Centralized configuration with environment variable support
"""

import os
import json
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

@dataclass
class CacheConfig:
    """Cache configuration settings."""
    enabled: bool = True
    max_size: int = 1000
    eviction_percentage: float = 0.1
    semantic_threshold: float = 0.55
    fuzzy_match_length: int = 20
    semantic_check_limit: int = 15

@dataclass
class ModelConfig:
    """Model configuration settings."""
    default_model: str = "gemma2:2b"
    default_provider: str = "ollama"
    temperature: float = 0.1
    max_tokens: int = 120
    top_p: float = 0.45
    timeout: int = 8

@dataclass
class PerformanceConfig:
    """Performance configuration settings."""
    performance_mode: str = "speed"  # "speed", "balanced", "quality"
    enable_all_optimizations: bool = True
    max_response_length: int = 10000
    cache_warming_enabled: bool = True
    predictive_queries_count: int = 20

@dataclass
class ServerConfig:
    """Server configuration settings."""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    log_level: str = "INFO"
    cors_enabled: bool = True

@dataclass
class MonitoringConfig:
    """Monitoring configuration settings."""
    analytics_enabled: bool = True
    performance_tracking: bool = True
    metrics_history_size: int = 100
    alert_threshold_response_time: float = 5.0
    alert_threshold_cache_hit_rate: float = 0.3

@dataclass
class AppConfig:
    """Main application configuration."""
    cache: CacheConfig
    model: ModelConfig
    performance: PerformanceConfig
    server: ServerConfig
    monitoring: MonitoringConfig

class ConfigManager:
    """Configuration manager with file and environment variable support."""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or "config.json"
        self.config = self._load_config()
    
    def _load_config(self) -> AppConfig:
        """Load configuration from file and environment variables."""
        # Default configuration
        config = AppConfig(
            cache=CacheConfig(),
            model=ModelConfig(),
            performance=PerformanceConfig(),
            server=ServerConfig(),
            monitoring=MonitoringConfig()
        )
        
        # Load from file if exists
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    file_config = json.load(f)
                    config = self._merge_config(config, file_config)
            except Exception as e:
                print(f"Warning: Could not load config file: {e}")
        
        # Override with environment variables
        config = self._apply_env_variables(config)
        
        return config
    
    def _merge_config(self, base_config: AppConfig, override_config: Dict[str, Any]) -> AppConfig:
        """Merge file configuration with base configuration."""
        if "cache" in override_config:
            base_config.cache = CacheConfig(**{**asdict(base_config.cache), **override_config["cache"]})
        if "model" in override_config:
            base_config.model = ModelConfig(**{**asdict(base_config.model), **override_config["model"]})
        if "performance" in override_config:
            base_config.performance = PerformanceConfig(**{**asdict(base_config.performance), **override_config["performance"]})
        if "server" in override_config:
            base_config.server = ServerConfig(**{**asdict(base_config.server), **override_config["server"]})
        if "monitoring" in override_config:
            base_config.monitoring = MonitoringConfig(**{**asdict(base_config.monitoring), **override_config["monitoring"]})
        
        return base_config
    
    def _apply_env_variables(self, config: AppConfig) -> AppConfig:
        """Apply environment variable overrides."""
        # Cache settings
        config.cache.enabled = os.getenv("CACHE_ENABLED", str(config.cache.enabled)).lower() == "true"
        config.cache.max_size = int(os.getenv("CACHE_MAX_SIZE", str(config.cache.max_size)))
        
        # Model settings
        config.model.default_model = os.getenv("DEFAULT_MODEL", config.model.default_model)
        config.model.temperature = float(os.getenv("MODEL_TEMPERATURE", str(config.model.temperature)))
        config.model.max_tokens = int(os.getenv("MODEL_MAX_TOKENS", str(config.model.max_tokens)))
        
        # Performance settings
        config.performance.performance_mode = os.getenv("PERFORMANCE_MODE", config.performance.performance_mode)
        config.performance.enable_all_optimizations = os.getenv("ENABLE_OPTIMIZATIONS", str(config.performance.enable_all_optimizations)).lower() == "true"
        
        # Server settings
        config.server.host = os.getenv("SERVER_HOST", config.server.host)
        config.server.port = int(os.getenv("SERVER_PORT", str(config.server.port)))
        config.server.log_level = os.getenv("LOG_LEVEL", config.server.log_level)
        
        # Monitoring settings
        config.monitoring.analytics_enabled = os.getenv("ANALYTICS_ENABLED", str(config.monitoring.analytics_enabled)).lower() == "true"
        
        return config
    
    def save_config(self, config: Optional[AppConfig] = None):
        """Save current configuration to file."""
        config_to_save = config or self.config
        with open(self.config_file, 'w') as f:
            json.dump(asdict(config_to_save), f, indent=2)
    
    def get_config(self) -> AppConfig:
        """Get current configuration."""
        return self.config
    
    def update_config(self, updates: Dict[str, Any]):
        """Update configuration with new values."""
        self.config = self._merge_config(self.config, updates)
        self.save_config()

# Global configuration instance
_config_manager = None

def get_config_manager(config_file: Optional[str] = None) -> ConfigManager:
    """Get global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager(config_file)
    return _config_manager

def get_config() -> AppConfig:
    """Get current application configuration."""
    return get_config_manager().get_config()

# Example configuration file
DEFAULT_CONFIG = {
    "cache": {
        "enabled": True,
        "max_size": 1000,
        "eviction_percentage": 0.1,
        "semantic_threshold": 0.55,
        "fuzzy_match_length": 20,
        "semantic_check_limit": 15
    },
    "model": {
        "default_model": "gemma2:2b",
        "default_provider": "ollama",
        "temperature": 0.1,
        "max_tokens": 120,
        "top_p": 0.45,
        "timeout": 8
    },
    "performance": {
        "performance_mode": "balanced",
        "enable_all_optimizations": True,
        "max_response_length": 10000,
        "cache_warming_enabled": True,
        "predictive_queries_count": 20
    },
    "server": {
        "host": "0.0.0.0",
        "port": 8000,
        "workers": 1,
        "log_level": "INFO",
        "cors_enabled": True
    },
    "monitoring": {
        "analytics_enabled": True,
        "performance_tracking": True,
        "metrics_history_size": 100,
        "alert_threshold_response_time": 5.0,
        "alert_threshold_cache_hit_rate": 0.3
    }
}

if __name__ == "__main__":
    # Create default configuration file
    config_manager = ConfigManager()
    config_manager.save_config()
    print(f"Default configuration saved to {config_manager.config_file}")
    print("Current configuration:")
    print(json.dumps(asdict(config_manager.get_config()), indent=2))