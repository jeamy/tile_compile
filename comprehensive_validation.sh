#!/bin/bash

# Exit on any error
set -e

# Color codes for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function for logging
log() {
    echo -e "${GREEN}[VALIDATION]${NC} $1"
}

# Function for error logging
error_log() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Navigate to project directory
cd "$(dirname "$0")"

# Ensure virtual environment is active
if [ -z "$VIRTUAL_ENV" ]; then
    log "Activating virtual environment..."
    source .venv/bin/activate
fi

# Verify Python and pip
log "Python Version: $(python --version)"
log "Pip Version: $(pip --version)"

# Upgrade pip and install requirements
log "Upgrading pip and installing requirements..."
pip install --upgrade pip
pip install -r requirements.txt

# Create necessary directories
mkdir -p synthetic_datasets
mkdir -p validation_results

# Generate Synthetic Datasets
log "Generating Synthetic Astronomical Datasets..."
if ! python generate_datasets.py; then
    error_log "Dataset generation failed!"
    exit 1
fi

# Run Comprehensive Validation
log "Running Comprehensive Validation..."
if ! python run_validation.py; then
    error_log "Validation process failed!"
    exit 1
fi

# Optional: Generate Detailed Report
log "Generating Validation Report..."
python -c "
import json
import os
import glob

validation_dir = 'validation_results'
report_path = os.path.join(validation_dir, 'validation_summary.json')

# Collect validation results
validation_results = {}
for filepath in glob.glob(os.path.join(validation_dir, '*.json')):
    filename = os.path.basename(filepath)
    with open(filepath, 'r') as f:
        validation_results[filename] = json.load(f)

# Write comprehensive report
with open(report_path, 'w') as f:
    json.dump(validation_results, f, indent=2)

print(f'Validation summary saved to {report_path}')
"

log "Validation Complete. Detailed reports are available in validation_results/ directory."