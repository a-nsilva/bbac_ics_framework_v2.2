#!/usr/bin/env python3
"""
BBAC ICS Framework - Configuration Loader

Centralized configuration management using YAML files.
Loads and provides access to all framework configuration.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


logger = logging.getLogger(__name__)


class ConfigLoader:
    """
    Centralized configuration loader.
    
    Loads configuration from YAML files in config/ directory.
    """
    
    def __init__(self, config_dir: Optional[Path] = None):
        """
        Initialize configuration loader.
        
        Args:
            config_dir: Directory containing config YAML files
        """
        if config_dir is None:
            # Default to config/ directory relative to project root
            config_dir = Path(__file__).parent.parent.parent / 'config'
        
        self.config_dir = Path(config_dir)
        
        if not self.config_dir.exists():
            logger.warning(f"Config directory not found: {self.config_dir}")
            self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Configuration containers
        self._config: Dict[str, Any] = {}
        
        # Load all configuration files
        self._load_all_configs()
        
        logger.info(f"ConfigLoader initialized from {self.config_dir}")
    
    def _load_all_configs(self):
        """Load all YAML configuration files."""
        # Define configuration files to load
        config_files = [
            'baseline.yaml',
            'fusion.yaml',
            'thresholds.yaml',
            'ros_params.yaml',
        ]
        
        for filename in config_files:
            filepath = self.config_dir / filename
            
            if filepath.exists():
                try:
                    self._load_yaml_file(filepath)
                    logger.info(f"Loaded config: {filename}")
                except Exception as e:
                    logger.error(f"Error loading {filename}: {e}")
            else:
                logger.warning(f"Config file not found: {filename}")
                # Create default config file
                self._create_default_config(filepath)
    
    def _load_yaml_file(self, filepath: Path):
        """Load a single YAML configuration file."""
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
        
        if data:
            # Merge into main config
            self._config.update(data)
    
    def _create_default_config(self, filepath: Path):
        """Create default configuration file if missing."""
        filename = filepath.name
        
        # Default configurations
        defaults = {
            'baseline.yaml': {
                'baseline': {
                    'window_days': 10,
                    'recent_weight': 0.7,
                    'max_historical_baselines': 10,
                }
            },
            'fusion.yaml': {
                'fusion': {
                    'fusion_method': 'weighted_voting',
                    'weights': {
                        'rule': 0.4,
                        'behavioral': 0.3,
                        'ml': 0.3,
                    },
                    'high_confidence_threshold': 0.9,
                    'decision_threshold': 0.5,
                }
            },
            'thresholds.yaml': {
                'thresholds': {
                    'max_auth_attempts': 3,
                    'trust_threshold': 0.8,
                    'min_confidence_for_update': 0.7,
                    'require_grant_for_update': True,
                    't1_allow': 0.7,
                    't2_mfa': 0.5,
                    't3_review': 0.3,
                }
            },
            'ros_params.yaml': {
                'ros': {
                    'node_name': 'bbac_node',
                    'target_latency_ms': 100.0,
                    'qos_depth': 10,
                },
                'paths': {
                    'data_dir': 'data/100k',
                    'profiles_dir': 'profiles',
                    'logs_dir': 'logs',
                    'quarantine_dir': 'quarantine',
                    'train_file': 'bbac_trainer.csv',
                    'validation_file': 'bbac_validation.csv',
                    'test_file': 'bbac_test.csv',
                    'agents_file': 'agents.json',
                    'anomaly_metadata_file': 'anomaly_metadata.json',
                },
                'ml_params': {
                    'statistical': {
                        'anomaly_threshold': 0.5,
                    },
                    'sequence': {
                        'anomaly_threshold': 0.5,
                    },
                },
                'learning': {
                    'buffer_size': 1000,
                    'min_samples_for_update': 100,
                },
                'policy': {
                    'maintenance_hours': [2, 3],
                },
                'rbac': {},
            },
        }
        
        default_data = defaults.get(filename)
        
        if default_data:
            try:
                with open(filepath, 'w') as f:
                    yaml.dump(default_data, f, default_flow_style=False, indent=2)
                
                logger.info(f"Created default config: {filename}")
                
                # Load the newly created default
                self._config.update(default_data)
                
            except Exception as e:
                logger.error(f"Error creating default config {filename}: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.
        
        Supports nested keys with dot notation (e.g., 'fusion.weights.rule').
        
        Args:
            key: Configuration key (supports dot notation)
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def __getattr__(self, name: str) -> Any:
        """
        Allow attribute-style access to top-level config keys.
        
        Example: config.baseline instead of config.get('baseline')
        """
        if name.startswith('_'):
            # Don't intercept private attributes
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        
        if name in self._config:
            return self._config[name]
        
        raise AttributeError(f"Configuration key '{name}' not found")
    
    def reload(self):
        """Reload all configuration files."""
        self._config.clear()
        self._load_all_configs()
        logger.info("Configuration reloaded")
    
    def save(self):
        """Save current configuration back to YAML files."""
        # Group config by file
        file_configs = {
            'baseline.yaml': {'baseline': self._config.get('baseline', {})},
            'fusion.yaml': {'fusion': self._config.get('fusion', {})},
            'thresholds.yaml': {'thresholds': self._config.get('thresholds', {})},
            'ros_params.yaml': {
                'ros': self._config.get('ros', {}),
                'paths': self._config.get('paths', {}),
                'ml_params': self._config.get('ml_params', {}),
                'learning': self._config.get('learning', {}),
                'policy': self._config.get('policy', {}),
                'rbac': self._config.get('rbac', {}),
            },
        }
        
        for filename, data in file_configs.items():
            filepath = self.config_dir / filename
            
            try:
                with open(filepath, 'w') as f:
                    yaml.dump(data, f, default_flow_style=False, indent=2)
                
                logger.info(f"Saved config: {filename}")
                
            except Exception as e:
                logger.error(f"Error saving {filename}: {e}")


# Global configuration instance
config = ConfigLoader()


__all__ = ['config', 'ConfigLoader']