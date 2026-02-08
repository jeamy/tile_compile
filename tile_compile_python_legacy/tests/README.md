# Tile Compile Test Suite

## Overview

This test suite validates the core components of the Tile Compile image reconstruction pipeline, following the Methodik v3 specification.

## Test Components

### Policy Validation (`test_policy.py`)
- Linearity checks
- Frame count validation
- Channel separation verification
- Phase progression testing

### Metrics Calculation (`test_metrics.py`)
- Global metrics computation
- Tile-level metrics extraction
- Channel-based metric analysis

## Running Tests

### Prerequisites
- Python 3.8+
- Poetry (dependency management)

### Installation
```bash
poetry install
```

### Execute Tests
```bash
poetry run pytest tests/
```

## Test Configuration
- `test_config.yaml`: Provides a standardized test configuration
- Simulates OSC astronomical image processing workflow

## Key Validation Points
1. Strict linearity enforcement
2. No frame selection
3. Channel-independent processing
4. Robust metrics computation
5. Tile-based reconstruction integrity