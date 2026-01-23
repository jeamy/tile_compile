#!/bin/bash

# Ensure we're in the project directory
PROJECT_DIR="$(dirname "$(readlink -f "$0")")"
cd "$PROJECT_DIR"

# Function to compare version numbers
version_greater_equal() {
    printf '%s\n%s\n' "$2" "$1" | sort -C -V
}

# Check Python version
PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
MIN_PYTHON_VERSION="3.8"

# Compare Python versions using bash version comparison
if ! version_greater_equal "$PYTHON_VERSION" "$MIN_PYTHON_VERSION"; then
    echo "Error: Python $MIN_PYTHON_VERSION or higher is required. Current version: $PYTHON_VERSION"
    exit 1
fi

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip and setuptools
pip install --upgrade pip setuptools wheel

# Install packaging module first
pip install packaging

# Install requirements
pip install -r requirements.txt

# Optional: Install development tools
pip install black flake8 mypy pytest

# Optional: Install Jupyter (if needed)
pip install jupyter

# Display installed packages
pip list

# Deactivate virtual environment
deactivate

echo "Virtual environment setup complete. Activate with: source .venv/bin/activate"