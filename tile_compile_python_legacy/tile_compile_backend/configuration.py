import yaml
import json
from typing import Dict, Any, Optional
from pathlib import Path
import jsonschema

class ConfigurationManager:
    """
    Manages configuration loading, validation, and processing
    """
    @classmethod
    def load_config(
        cls, 
        config_path: Path, 
        schema_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Load and validate configuration from file
        
        Args:
            config_path: Path to configuration file
            schema_path: Optional path to JSON schema for validation
        
        Returns:
            Validated configuration dictionary
        """
        # Load configuration
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Validate against schema if provided
        if schema_path and schema_path.exists():
            cls.validate_config(config, schema_path)
        
        return config
    
    @classmethod
    def validate_config(
        cls, 
        config: Dict[str, Any], 
        schema_path: Path
    ) -> bool:
        """
        Validate configuration against JSON schema
        
        Args:
            config: Configuration dictionary
            schema_path: Path to JSON schema file
        
        Returns:
            True if valid, raises exception otherwise
        """
        with open(schema_path, 'r') as f:
            schema = json.load(f)
        
        try:
            jsonschema.validate(instance=config, schema=schema)
            return True
        except jsonschema.exceptions.ValidationError as e:
            raise ValueError(f"Configuration validation failed: {e}")
    
    @classmethod
    def generate_default_config(
        cls, 
        output_path: Path, 
        base_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate a default configuration file
        
        Args:
            output_path: Path to save the configuration
            base_config: Optional base configuration to extend
        
        Returns:
            Generated configuration dictionary
        """
        default_config = {
            'project': {
                'name': 'tile_compile_project',
                'version': '3.0.0'
            },
            'data': {
                'frames_min': 50,
                'color_mode': 'OSC',
                'bayer_pattern': 'GBRG'
            },
            'registration': {
                'engine': 'siril',
                'allow_rotation': True,
                'min_star_matches': 10
            },
            'metrics': {
                'global': {
                    'background_level': {'method': 'median'},
                    'noise_level': {'method': 'std'}
                },
                'tile': {
                    'size': 64,
                    'overlap': 0.25
                }
            },
            'reconstruction': {
                'method': 'tile_based',
                'weights': {
                    'global_quality': 0.6,
                    'local_quality': 0.4
                }
            },
            'synthetic': {
                'frames_min': 15,
                'frames_max': 30,
                'method': 'linear_average'
            }
        }
        
        # Merge with base config if provided
        if base_config:
            cls._deep_update(default_config, base_config)
        
        # Write to file
        with open(output_path, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False)
        
        return default_config
    
    @staticmethod
    def _deep_update(base_dict: Dict, update_dict: Dict) -> Dict:
        """
        Recursively update nested dictionaries
        
        Args:
            base_dict: Base dictionary to update
            update_dict: Dictionary with updates
        
        Returns:
            Updated dictionary
        """
        for key, value in update_dict.items():
            if isinstance(value, dict):
                base_dict[key] = base_dict.get(key, {})
                base_dict[key] = ConfigurationManager._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
        return base_dict
    
    @classmethod
    def generate_json_schema(
        cls, 
        output_path: Path
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive JSON schema for configuration validation
        
        Args:
            output_path: Path to save the JSON schema
        
        Returns:
            Generated JSON schema dictionary
        """
        schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {
                "project": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "version": {"type": "string"}
                    },
                    "required": ["name"]
                },
                "data": {
                    "type": "object",
                    "properties": {
                        "frames_min": {"type": "number", "minimum": 10},
                        "color_mode": {
                            "type": "string", 
                            "enum": ["OSC", "RGB", "MONO"]
                        },
                        "bayer_pattern": {
                            "type": "string", 
                            "enum": ["RGGB", "BGGR", "GBRG", "GRBG"]
                        }
                    }
                },
                "registration": {
                    "type": "object",
                    "properties": {
                        "engine": {
                            "type": "string", 
                            "enum": ["siril", "opencv_cfa"]
                        },
                        "allow_rotation": {"type": "boolean"},
                        "min_star_matches": {"type": "number", "minimum": 1}
                    }
                }
            },
            "required": ["project", "data"]
        }
        
        # Write schema to file
        with open(output_path, 'w') as f:
            json.dump(schema, f, indent=2)
        
        return schema

def validate_and_prepare_configuration(
    config_path: Path, 
    schema_path: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Centralized configuration validation and preparation
    
    Args:
        config_path: Path to configuration file
        schema_path: Optional path to JSON schema
    
    Returns:
        Validated and processed configuration
    """
    # Use ConfigurationManager to load and validate
    config = ConfigurationManager.load_config(config_path, schema_path)
    
    # Additional preprocessing or validation can be added here
    
    return config