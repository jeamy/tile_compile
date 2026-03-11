#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"

BUILD_DIR_DEFAULT="${PROJECT_ROOT}/web_backend_cpp/build"
BUILD_DIR="${BUILD_DIR:-${BUILD_DIR_DEFAULT}}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
BACKEND_BIN_DEFAULT="${BUILD_DIR}/tile_compile_web_backend"
BACKEND_BIN="${BACKEND_BIN:-${BACKEND_BIN_DEFAULT}}"
CLI_BIN_DEFAULT="${PROJECT_ROOT}/tile_compile_cpp/build/tile_compile_cli"
CLI_BIN="${TILE_COMPILE_CLI:-${CLI_BIN_DEFAULT}}"
RUNNER_BIN_DEFAULT="${PROJECT_ROOT}/tile_compile_cpp/build/tile_compile_runner"
RUNNER_BIN="${TILE_COMPILE_RUNNER:-${RUNNER_BIN_DEFAULT}}"
CONFIG_PATH_DEFAULT="${PROJECT_ROOT}/tile_compile_cpp/tile_compile.yaml"
CONFIG_PATH="${TILE_COMPILE_CONFIG:-${CONFIG_PATH_DEFAULT}}"
SCHEMA_PATH_DEFAULT="${PROJECT_ROOT}/tile_compile_cpp/tile_compile.schema.yaml"
SCHEMA_PATH="${TILE_COMPILE_SCHEMA:-${SCHEMA_PATH_DEFAULT}}"
PRESETS_DIR_DEFAULT="${PROJECT_ROOT}/tile_compile_cpp/examples"
PRESETS_DIR="${TILE_COMPILE_PRESETS_DIR:-${PRESETS_DIR_DEFAULT}}"
UI_DIR_DEFAULT="${PROJECT_ROOT}/web_frontend"
UI_DIR="${TILE_COMPILE_UI_DIR:-${UI_DIR_DEFAULT}}"
RUNS_DIR_DEFAULT="${PROJECT_ROOT}/runs"
RUNS_DIR="${TILE_COMPILE_RUNS_DIR:-${RUNS_DIR_DEFAULT}}"
ALLOWED_ROOTS_DEFAULT="${PROJECT_ROOT}:${RUNS_DIR}:${PROJECT_ROOT}/tmp"
ALLOWED_ROOTS="${TILE_COMPILE_ALLOWED_ROOTS:-${ALLOWED_ROOTS_DEFAULT}}"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
DO_BUILD="1"

usage() {
  cat <<EOF
Usage: $(basename "$0") [options] [-- <extra backend args>]

Options:
  --host <host>         Backend bind host (default: ${HOST})
  --port <port>         Backend port (default: ${PORT})
  --build-dir <path>    CMake build directory (default: ${BUILD_DIR_DEFAULT})
  --backend-bin <path>  Backend binary path (default: ${BACKEND_BIN_DEFAULT})
  --build-type <type>   CMake build type (default: ${BUILD_TYPE})
  --runs-dir <path>     Runs directory (default: ${RUNS_DIR_DEFAULT})
  --no-build            Skip cmake configure/build step
  -h, --help            Show this help

Env overrides:
  HOST, PORT, BUILD_DIR, BUILD_TYPE, BACKEND_BIN,
  TILE_COMPILE_CLI, TILE_COMPILE_RUNNER,
  TILE_COMPILE_CONFIG, TILE_COMPILE_SCHEMA,
  TILE_COMPILE_PRESETS_DIR, TILE_COMPILE_UI_DIR,
  TILE_COMPILE_RUNS_DIR, TILE_COMPILE_ALLOWED_ROOTS
EOF
}

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --host)
      HOST="$2"; shift 2;;
    --port)
      PORT="$2"; shift 2;;
    --build-dir)
      BUILD_DIR="$2"; shift 2;;
    --backend-bin)
      BACKEND_BIN="$2"; shift 2;;
    --build-type)
      BUILD_TYPE="$2"; shift 2;;
    --runs-dir)
      RUNS_DIR="$2"; shift 2;;
    --no-build)
      DO_BUILD="0"; shift;;
    -h|--help)
      usage; exit 0;;
    --)
      shift
      EXTRA_ARGS=("$@")
      break;;
    *)
      EXTRA_ARGS+=("$1")
      shift;;
  esac
done

if [[ -L "${RUNS_DIR}" && ! -e "${RUNS_DIR}" ]]; then
  echo "[backend] Runs directory path is a broken symlink: ${RUNS_DIR}" >&2
  echo "[backend] Fix the symlink target or start with --runs-dir <directory>." >&2
  exit 1
fi

if [[ -e "${RUNS_DIR}" && ! -d "${RUNS_DIR}" ]]; then
  echo "[backend] Runs directory path exists but is not a directory: ${RUNS_DIR}" >&2
  exit 1
fi

mkdir -p "${RUNS_DIR}"

if [[ "${DO_BUILD}" == "1" ]]; then
  echo "[backend] Configuring C++ backend in ${BUILD_DIR}"
  cmake -S "${PROJECT_ROOT}/web_backend_cpp" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"
  echo "[backend] Building tile_compile_web_backend"
  cmake --build "${BUILD_DIR}" --parallel "$(nproc)"
fi

if [[ ! -x "${BACKEND_BIN}" ]]; then
  echo "[backend] Binary not found or not executable: ${BACKEND_BIN}" >&2
  exit 1
fi

export TILE_COMPILE_PROJECT_ROOT="${PROJECT_ROOT}"
export TILE_COMPILE_HOST="${HOST}"
export TILE_COMPILE_PORT="${PORT}"
export TILE_COMPILE_CLI="${CLI_BIN}"
export TILE_COMPILE_RUNNER="${RUNNER_BIN}"
export TILE_COMPILE_CONFIG="${CONFIG_PATH}"
export TILE_COMPILE_SCHEMA="${SCHEMA_PATH}"
export TILE_COMPILE_PRESETS_DIR="${PRESETS_DIR}"
export TILE_COMPILE_UI_DIR="${UI_DIR}"
export TILE_COMPILE_RUNS_DIR="${RUNS_DIR}"
export TILE_COMPILE_ALLOWED_ROOTS="${ALLOWED_ROOTS}"

BACKEND_CMD=(
  "${BACKEND_BIN}"
)

if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  BACKEND_CMD+=("${EXTRA_ARGS[@]}")
fi

echo "[backend] Starting: ${BACKEND_CMD[*]}"
echo "[backend] UI: http://${HOST}:${PORT}/ui/"
exec "${BACKEND_CMD[@]}"
