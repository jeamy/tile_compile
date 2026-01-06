"""
Test Suite: Configuration Management

Tests the configuration management system for Methodik v3:
- Default config generation
- Config loading and validation
- JSON schema generation
- Deep config updates
- Config merging and overrides

These tests ensure the configuration system correctly handles
tile_compile.yaml files and validates them against the schema.
"""

import os
import json
import yaml
import pytest
from pathlib import Path
from tile_compile_backend.configuration import (
    ConfigurationManager, 
    validate_and_prepare_configuration
)

class TestConfigurationManager:
    def setup_method(self):
        self.test_config_dir = Path(__file__).parent / 'config_test'
        self.test_config_dir.mkdir(exist_ok=True)

    def teardown_method(self):
        # Clean up test directory
        for file in self.test_config_dir.iterdir():
            file.unlink()
        self.test_config_dir.rmdir()

    def test_generate_default_config(self):
        config_path = self.test_config_dir / 'default_config.yaml'
        config = ConfigurationManager.generate_default_config(config_path)
        
        assert config_path.exists()
        assert 'project' in config
        assert 'data' in config
        assert 'registration' in config
        assert 'metrics' in config
        assert 'reconstruction' in config
        assert 'synthetic' in config

    def test_load_config(self):
        config_path = self.test_config_dir / 'test_config.yaml'
        test_config = {
            'project': {'name': 'test_project', 'version': '3.0.0'},
            'data': {'frames_min': 50, 'color_mode': 'OSC'}
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(test_config, f)
        
        loaded_config = ConfigurationManager.load_config(config_path)
        
        assert loaded_config == test_config

    def test_generate_json_schema(self):
        schema_path = self.test_config_dir / 'config_schema.json'
        schema = ConfigurationManager.generate_json_schema(schema_path)
        
        assert schema_path.exists()
        assert '$schema' in schema
        assert 'type' in schema
        assert 'properties' in schema

    def test_deep_update(self):
        base = {
            'project': {'name': 'original'},
            'data': {'frames_min': 10}
        }
        update = {
            'project': {'version': '3.0.0'},
            'data': {'frames_min': 50}
        }
        
        updated = ConfigurationManager._deep_update(base, update)
        
        assert updated['project']['name'] == 'original'
        assert updated['project']['version'] == '3.0.0'
        assert updated['data']['frames_min'] == 50

    def test_configuration_validation(self):
        config_path = self.test_config_dir / 'validated_config.yaml'
        schema_path = self.test_config_dir / 'config_schema.json'
        
        # Generate schema
        ConfigurationManager.generate_json_schema(schema_path)
        
        # Create valid config
        valid_config = {
            'project': {'name': 'validated_project'},
            'data': {
                'frames_min': 50, 
                'color_mode': 'OSC', 
                'bayer_pattern': 'GBRG'
            }
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(valid_config, f)
        
        # Validate configuration
        validated_config = validate_and_prepare_configuration(config_path, schema_path)
        
        assert validated_config == valid_config

    def test_invalid_configuration(self):
        config_path = self.test_config_dir / 'invalid_config.yaml'
        schema_path = self.test_config_dir / 'config_schema.json'
        
        # Generate schema
        ConfigurationManager.generate_json_schema(schema_path)
        
        # Create invalid config (missing required fields)
        invalid_config = {
            'project': {}  # Missing name
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(invalid_config, f)
        
        # Expect validation error
        with pytest.raises(ValueError):
            validate_and_prepare_configuration(config_path, schema_path)