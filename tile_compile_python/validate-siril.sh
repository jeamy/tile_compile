#!/usr/bin/env bash
set -euo pipefail

python tile_compile_backend_cli.py validate-ssf siril_register_osc.ssf --strict-exit-codes
python tile_compile_backend_cli.py validate-ssf siril_stack_average.ssf --expect-save --strict-exit-codes

# Optional zusätzlich: Config + Defaults prüfen (uses tile_compile.yaml)
python tile_compile_backend_cli.py validate-siril-scripts --path tile_compile.yaml --strict-exit-codes
