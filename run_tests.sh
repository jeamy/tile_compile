#!/bin/bash
set -e

# Ensure executable
chmod +x run_tests.sh

# Change to script directory
cd "$(dirname "$0")"

# Run tests with coverage
poetry run pytest tests/ \
    --cov=tile_compile_backend \
    --cov-report=term-missing \
    --cov-report=html:coverage_report \
    --cov-fail-under=80

# Optional: Open coverage report
if command -v xdg-open &> /dev/null; then
    xdg-open coverage_report/index.html
elif command -v open &> /dev/null; then
    open coverage_report/index.html
fi