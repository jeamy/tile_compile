#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# pack_tile_compile_snapshot.sh
#
# Creates a clean project snapshot archive for review / analysis.
# Designed for tile_compile experimental reference work (Methodik v4).
# -----------------------------------------------------------------------------

set -euo pipefail

PROJECT_NAME="tile_compile"
DATE=$(date +%Y%m%d)
OUT="${PROJECT_NAME}_snapshot_${DATE}.tar.gz"

# Sanity check
if [[ ! -f "tile_compile.yaml" ]]; then
  echo "ERROR: run this script from the project root (tile_compile.yaml missing)" >&2
  exit 1
fi

# Optional context file
if [[ ! -f README_CONTEXT.md ]]; then
  cat <<'EOF' > README_CONTEXT.md
# Context for ChatGPT / External Review

Project: tile_compile
Purpose: Experimental reference implementation of
         Tile-basierte Qualitätsrekonstruktion für DSO – Methodik v4

Key points:
- Global registration is deprecated
- Tile-local registration (v4) is authoritative
- Performance is irrelevant, correctness is primary
- Iterative tile reference + warp smoothing are in progress

Entry points:
- tile_compile_runner.py
- runner/tile_local_registration_v4.py
- tile_compile.yaml

Open questions / TODO:
- (fill in if desired)
EOF
  echo "Created README_CONTEXT.md"
fi

# Build archive

tar \
  --exclude='**/__pycache__' \
  --exclude='*.pyc' \
  --exclude='.git' \
  --exclude='.venv' \
  --exclude='runs' \
  --exclude='*.log' \
  --exclude='*.tmp' \
  -czf "$OUT" \
  analyze_m45_dataset.py \
  check_dependencies.py \
  comprehensive_validation.sh \
  download_reference_datasets.py \
  generate_artifacts_report.py \
  generate_datasets.py \
  install_dependencies.sh \
  prepare_m45_dataset.py \
  pyproject.toml \
  README.md \
  README_CONTEXT.md \
  requirements.txt \
  run-cli.sh \
  run_tests.sh \
  run_validation.py \
  setup_venv.sh \
  start_gui.sh start_gui.ps1 start_gui.cmd \
  tile_compile.yaml \
  tile_compile.schema.yaml \
  tile_compile.schema.json \
  tile_compile_runner.py \
  tile_compile_backend_cli.py \
  tile_compile_gui_state.json \
  gui \
  runner \
  tile_compile_backend \
  siril_scripts \
  tests \
  validation


echo "Snapshot created: $OUT"
echo "You can now upload this archive directly."
