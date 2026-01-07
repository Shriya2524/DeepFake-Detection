"""
Configuration management for deepfake detection system.

Handles loading, merging, and accessing configuration from YAML files.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union
import yaml


class Config:
    """
    Configuration class that provides dot-notation access to config values.
    
    Example:
        config = Config({"model": {"backbone": "resnet18"}})
        print(config.model.backbone)  # "resnet18"
    """
    
    def __init__(self, config_dict: Dict[str, Any]):
        """
        Initialize Config from dictionary.
        
        Args:
            config_dict: Configuration dictionary
        """
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)
    
    def __repr__(self) -> str:
        return f"Config({self.to_dict()})"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert Config back to dictionary."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get config value with optional default.
        
        Args:
            key: Configuration key (supports dot notation, e.g., "model.backbone")
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key.split(".")
        value = self
        
        try:
            for k in keys:
                if isinstance(value, Config):
                    value = getattr(value, k)
                elif isinstance(value, dict):
                    value = value[k]
                else:
                    return default
            return value
        except (AttributeError, KeyError):
            return default
    
    def update(self, updates: Dict[str, Any]) -> None:
        """
        Update configuration with new values.
        
        Args:
            updates: Dictionary of updates to apply
        """
        for key, value in updates.items():
            if isinstance(value, dict) and hasattr(self, key):
                existing = getattr(self, key)
                if isinstance(existing, Config):
                    existing.update(value)
                else:
                    setattr(self, key, Config(value))
            else:
                if isinstance(value, dict):
                    setattr(self, key, Config(value))
                else:
                    setattr(self, key, value)


def deep_merge(base: Dict, override: Dict) -> Dict:
    """
    Deep merge two dictionaries.
    
    Args:
        base: Base dictionary
        override: Dictionary with values to override
        
    Returns:
        Merged dictionary
    """
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    
    return result


def load_config(
    config_path: Union[str, Path],
    base_config_path: Optional[Union[str, Path]] = None,
    overrides: Optional[Dict[str, Any]] = None
) -> Config:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration YAML file
        base_config_path: Optional path to base config to merge with
        overrides: Optional dictionary of overrides to apply
        
    Returns:
        Config object with loaded configuration
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid YAML
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    # Load base config if provided
    config_dict = {}
    if base_config_path:
        base_path = Path(base_config_path)
        if base_path.exists():
            with open(base_path, "r", encoding="utf-8") as f:
                config_dict = yaml.safe_load(f) or {}
    
    # Load main config
    with open(config_path, "r", encoding="utf-8") as f:
        main_config = yaml.safe_load(f) or {}
    
    # Merge configs
    config_dict = deep_merge(config_dict, main_config)
    
    # Apply overrides
    if overrides:
        config_dict = deep_merge(config_dict, overrides)
    
    return Config(config_dict)


# Global config instance
_global_config: Optional[Config] = None


def get_config() -> Config:
    """
    Get the global configuration instance.
    
    Returns:
        Global Config object
        
    Raises:
        RuntimeError: If config hasn't been loaded yet
    """
    global _global_config
    
    if _global_config is None:
        # Try to load default config
        default_path = Path(__file__).parent.parent.parent / "config" / "default_config.yaml"
        if default_path.exists():
            _global_config = load_config(default_path)
        else:
            raise RuntimeError(
                "Configuration not loaded. Call load_config() first or ensure "
                "config/default_config.yaml exists."
            )
    
    return _global_config


def set_global_config(config: Config) -> None:
    """
    Set the global configuration instance.
    
    Args:
        config: Config object to set as global
    """
    global _global_config
    _global_config = config


def init_config(
    config_path: Optional[Union[str, Path]] = None,
    overrides: Optional[Dict[str, Any]] = None
) -> Config:
    """
    Initialize global configuration.
    
    Args:
        config_path: Path to config file (uses default if None)
        overrides: Optional overrides to apply
        
    Returns:
        Initialized Config object
    """
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / "config" / "default_config.yaml"
    
    config = load_config(config_path, overrides=overrides)
    set_global_config(config)
    
    return config

