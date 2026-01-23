#!/usr/bin/env bash
set -euo pipefail

SRC_DIR="/home/mux/Bilder/DWARF_RAW_M 31_EXP_10_GAIN_80_2024-10-07-20-51-46-987/min"
N_FRAMES="${N_FRAMES:-10}"
PATTERN="${PATTERN:-*.fit*}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUNS_DIR="${RUNS_DIR:-${PROJECT_ROOT}/runs}"

mkdir -p "${RUNS_DIR}"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
BUNDLE_DIR="${RUNS_DIR}/smoke_m31_10frames_${RUN_TS}"
mkdir -p "${BUNDLE_DIR}"
OUT_LOG="${BUNDLE_DIR}/stdout_stderr.log"
exec > >(tee -a "${OUT_LOG}") 2>&1
echo "[smoke] stdout/stderr redirected to ${OUT_LOG}"

if [ ! -d "${SRC_DIR}" ]; then
  echo "Source directory not found: ${SRC_DIR}" >&2
  exit 2
fi

WORK_DIR="${WORK_DIR:-${BUNDLE_DIR}/work}"
INPUT_DIR="${WORK_DIR}/input"
mkdir -p "${INPUT_DIR}"

mapfile -t FRAMES < <(find "${SRC_DIR}" -maxdepth 1 -type f -name "${PATTERN}" | sort | head -n "${N_FRAMES}")
if [ "${#FRAMES[@]}" -lt "${N_FRAMES}" ]; then
  echo "Not enough frames found in ${SRC_DIR} (pattern=${PATTERN}): want=${N_FRAMES} got=${#FRAMES[@]}" >&2
  exit 3
fi

for f in "${FRAMES[@]}"; do
  ln -s "${f}" "${INPUT_DIR}/$(basename "${f}")"
done

printf "%s\n" "${FRAMES[@]}" > "${BUNDLE_DIR}/frames_selected.txt"
cp -f "${BASH_SOURCE[0]}" "${BUNDLE_DIR}/smoke_script.sh" 2>/dev/null || true

CONFIG_IN="${CONFIG_IN:-${PROJECT_ROOT}/tile_compile_python/tile_compile.yaml}"
CONFIG_OUT="${BUNDLE_DIR}/config_smoke.yaml"

export CONFIG_IN
export CONFIG_OUT

python3 - <<'PY'
import os
import yaml

cfg_in = os.environ.get("CONFIG_IN")
cfg_out = os.environ.get("CONFIG_OUT")

with open(cfg_in, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}

if not isinstance(cfg, dict):
    raise SystemExit("config root must be a mapping")

v4 = cfg.setdefault("v4", {})
if not isinstance(v4, dict):
    cfg["v4"] = v4 = {}

v4.setdefault("debug_tile_registration", True)

with open(cfg_out, "w", encoding="utf-8") as f:
    yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
PY

echo "[smoke] work_dir=${WORK_DIR}"
echo "[smoke] input_dir=${INPUT_DIR}"
echo "[smoke] runs_dir=${RUNS_DIR}"
echo "[smoke] bundle_dir=${BUNDLE_DIR}"
echo "[smoke] config=${CONFIG_OUT}"

python3 "${PROJECT_ROOT}/tile_compile_python/tile_compile_runner.py" run \
  --config "${CONFIG_OUT}" \
  --input-dir "${INPUT_DIR}" \
  --runs-dir "${BUNDLE_DIR}" \
  --project-root "${PROJECT_ROOT}" \
  --pattern "${PATTERN}"
