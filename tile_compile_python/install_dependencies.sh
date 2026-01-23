#!/bin/bash

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Log functions
log() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Ensure we're in the correct directory
cd "$(dirname "$0")"

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    log "Activating virtual environment..."
    source .venv/bin/activate
fi

# Upgrade pip and setuptools
log "Upgrading pip and setuptools..."
pip install --upgrade pip setuptools wheel

# First, install build dependencies
log "Installing build dependencies..."
pip install wheel build setuptools-rust

# Run dependency check script
log "Checking dependencies..."
python check_dependencies.py

# Check the exit status
if [ $? -ne 0 ]; then
    warn "Some dependencies are missing. Attempting to install..."
    
    # Install base requirements
    log "Installing base requirements..."
    pip install -r requirements.txt
    
    # Install scientific computing stack
    log "Installing scientific computing stack..."
    pip install numpy scipy pandas astropy scikit-image scikit-learn

    # Install visualization and image processing
    log "Installing visualization and image processing libraries..."
    pip install matplotlib imageio pillow opencv-python seaborn

    # Install testing and development tools
    log "Installing testing and development tools..."
    pip install pytest pytest-cov memory-profiler

    # Rerun dependency check
    python check_dependencies.py
fi

# Verify critical modules
log "Verifying critical modules..."
python -c "
import sys
print('Python version:', sys.version)

# Check critical modules
modules_to_check = [
    'numpy', 'scipy', 'pandas', 
    'astropy', 'scikit-image', 
    'matplotlib', 'imageio'
]

for module in modules_to_check:
    try:
        __import__(module)
        print(f'✓ {module} imported successfully')
    except ImportError:
        print(f'✗ Failed to import {module}')
        sys.exit(1)
"

# Final success message
log "Dependency installation completed successfully!"