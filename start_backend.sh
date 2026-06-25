#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"

BUILD_DIR_DEFAULT="${PROJECT_ROOT}/web_backend_cpp/build"
BUILD_DIR="${BUILD_DIR:-${BUILD_DIR_DEFAULT}}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
BACKEND_BIN_DEFAULT="${BUILD_DIR}/tile_compile_web_backend"
BACKEND_BIN="${BACKEND_BIN:-${BACKEND_BIN_DEFAULT}}"
CPP_BUILD_DIR_DEFAULT="${PROJECT_ROOT}/tile_compile_cpp/build"
CPP_BUILD_DIR="${CPP_BUILD_DIR:-${CPP_BUILD_DIR_DEFAULT}}"
DEFAULT_OPENCV_CUDA_DIR="${HOME}/.local/opencv-cuda-4.11/lib64/cmake/opencv4"
DEFAULT_CUDA_ROOT="/usr/local/cuda-12.9"
DEFAULT_CUDA_NVCC="${DEFAULT_CUDA_ROOT}/bin/nvcc"
TILE_COMPILE_OPENCV_DIR="${TILE_COMPILE_OPENCV_DIR:-${OpenCV_DIR:-}}"
if [[ -z "${TILE_COMPILE_OPENCV_DIR}" && -f "${DEFAULT_OPENCV_CUDA_DIR}/OpenCVConfig.cmake" ]]; then
  TILE_COMPILE_OPENCV_DIR="${DEFAULT_OPENCV_CUDA_DIR}"
fi
TILE_COMPILE_CUDA_TOOLKIT_ROOT_DIR="${TILE_COMPILE_CUDA_TOOLKIT_ROOT_DIR:-${CUDA_TOOLKIT_ROOT_DIR:-}}"
if [[ -z "${TILE_COMPILE_CUDA_TOOLKIT_ROOT_DIR}" && -d "${DEFAULT_CUDA_ROOT}" ]]; then
  TILE_COMPILE_CUDA_TOOLKIT_ROOT_DIR="${DEFAULT_CUDA_ROOT}"
fi
TILE_COMPILE_CUDA_NVCC_EXECUTABLE="${TILE_COMPILE_CUDA_NVCC_EXECUTABLE:-${CUDA_NVCC_EXECUTABLE:-}}"
if [[ -z "${TILE_COMPILE_CUDA_NVCC_EXECUTABLE}" && -x "${DEFAULT_CUDA_NVCC}" ]]; then
  TILE_COMPILE_CUDA_NVCC_EXECUTABLE="${DEFAULT_CUDA_NVCC}"
fi
CLI_BIN_DEFAULT="${CPP_BUILD_DIR}/tile_compile_cli"
CLI_BIN="${TILE_COMPILE_CLI:-${CLI_BIN_DEFAULT}}"
RUNNER_BIN_DEFAULT="${CPP_BUILD_DIR}/tile_compile_runner"
RUNNER_BIN="${TILE_COMPILE_RUNNER:-${RUNNER_BIN_DEFAULT}}"
CONFIG_PATH_DEFAULT="${PROJECT_ROOT}/tile_compile_cpp/tile_compile.yaml"
CONFIG_PATH="${TILE_COMPILE_CONFIG:-${CONFIG_PATH_DEFAULT}}"
SCHEMA_PATH_DEFAULT="${PROJECT_ROOT}/tile_compile_cpp/tile_compile.schema.yaml"
SCHEMA_PATH="${TILE_COMPILE_SCHEMA:-${SCHEMA_PATH_DEFAULT}}"
PRESETS_DIR_DEFAULT="${PROJECT_ROOT}/tile_compile_cpp/examples"
PRESETS_DIR="${TILE_COMPILE_PRESETS_DIR:-${PRESETS_DIR_DEFAULT}}"
UI_DIR_DEFAULT="${PROJECT_ROOT}/web_frontend_v3"
UI_DIR="${TILE_COMPILE_UI_DIR:-${UI_DIR_DEFAULT}}"
AGENT_SERVICE_DIR_DEFAULT="${PROJECT_ROOT}/agent_service"
AGENT_SERVICE_DIR="${TILE_COMPILE_AGENT_SERVICE_DIR:-${AGENT_SERVICE_DIR_DEFAULT}}"
RUNS_DIR_DEFAULT="${PROJECT_ROOT}/runs"
RUNS_DIR="${TILE_COMPILE_RUNS_DIR:-${RUNS_DIR_DEFAULT}}"
ALLOWED_ROOTS_DEFAULT="${PROJECT_ROOT}:${RUNS_DIR}:${PROJECT_ROOT}/tmp"
ALLOWED_ROOTS="${TILE_COMPILE_ALLOWED_ROOTS:-${ALLOWED_ROOTS_DEFAULT}}"
AI_AGENT_AUTOSTART="${TILE_COMPILE_AI_AGENT_AUTOSTART:-1}"
AI_AGENT_PID=""

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
  --cpp-build-dir <p>   C++ core build directory (default: ${CPP_BUILD_DIR_DEFAULT})
  --backend-bin <path>  Backend binary path (default: ${BACKEND_BIN_DEFAULT})
  --build-type <type>   CMake build type (default: ${BUILD_TYPE})
  --runs-dir <path>     Runs directory (default: ${RUNS_DIR_DEFAULT})
  --no-build            Skip cmake configure/build step
  -h, --help            Show this help

Env overrides:
  HOST, PORT, BUILD_DIR, CPP_BUILD_DIR, BUILD_TYPE, BACKEND_BIN,
  TILE_COMPILE_CLI, TILE_COMPILE_RUNNER,
  TILE_COMPILE_CONFIG, TILE_COMPILE_SCHEMA,
  TILE_COMPILE_PRESETS_DIR, TILE_COMPILE_UI_DIR,
  TILE_COMPILE_AGENT_SERVICE_DIR, TILE_COMPILE_RUNS_DIR,
  TILE_COMPILE_ALLOWED_ROOTS, TILE_COMPILE_AI_AGENT_AUTOSTART,
  TILE_COMPILE_OPENCV_DIR, TILE_COMPILE_CUDA_TOOLKIT_ROOT_DIR,
  TILE_COMPILE_CUDA_NVCC_EXECUTABLE
EOF
}

cleanup_agent_service() {
  if [[ -n "${AI_AGENT_PID}" ]] && kill -0 "${AI_AGENT_PID}" >/dev/null 2>&1; then
    echo "[backend] Stopping PI AI sidecar pid=${AI_AGENT_PID}"
    kill "${AI_AGENT_PID}" >/dev/null 2>&1 || true
    wait "${AI_AGENT_PID}" >/dev/null 2>&1 || true
  fi
}

start_agent_service() {
  if [[ "${AI_AGENT_AUTOSTART}" == "0" ]]; then
    echo "[backend] PI AI sidecar autostart disabled."
    return 0
  fi
  if [[ ! -f "${AGENT_SERVICE_DIR}/package.json" ]]; then
    echo "[backend] PI AI sidecar not found: ${AGENT_SERVICE_DIR}"
    return 0
  fi
  if ! command -v npm >/dev/null 2>&1; then
    echo "[backend] WARNUNG: npm nicht gefunden; PI AI sidecar wird nicht gestartet."
    return 0
  fi
  local node_major
  node_major=$(node -e 'console.log(process.versions.node.split(".")[0])' 2>/dev/null || echo "0")
  if [[ "${node_major}" -lt 20 ]]; then
    echo "[backend] WARNUNG: Node.js ${node_major} ist zu alt für PI AI sidecar (>= 20 erforderlich, wegen RegExp v-flag in pi-tui)."
    echo "[backend] PI AI sidecar wird nicht gestartet. Bitte Node.js auf >= 20 aktualisieren."
    return 0
  fi
  local needs_install=0
  local needs_build=0
  if [[ ! -d "${AGENT_SERVICE_DIR}/node_modules" ]]; then
    needs_install=1
  elif [[ "${AGENT_SERVICE_DIR}/package.json" -nt "${AGENT_SERVICE_DIR}/node_modules/.package-lock.json" ]]; then
    needs_install=1
  fi
  if [[ ! -f "${AGENT_SERVICE_DIR}/dist/server.js" ]]; then
    needs_build=1
  else
    # Rebuild if any .ts source is newer than the built output
    while IFS= read -r -d '' src; do
      if [[ "${src}" -nt "${AGENT_SERVICE_DIR}/dist/server.js" ]]; then
        needs_build=1
        break
      fi
    done < <(find "${AGENT_SERVICE_DIR}/src" -name '*.ts' -print0 2>/dev/null)
  fi
  if [[ "${needs_install}" == "1" ]]; then
    echo "[backend] PI AI sidecar: npm install"
    if ! npm --prefix "${AGENT_SERVICE_DIR}" install; then
      echo "[backend] WARNUNG: npm install fehlgeschlagen; Backend startet ohne Sidecar."
      return 0
    fi
  fi
  if [[ "${needs_build}" == "1" ]]; then
    echo "[backend] PI AI sidecar: npm run build (Quellen neuer als dist)"
    if ! npm --prefix "${AGENT_SERVICE_DIR}" run build; then
      echo "[backend] WARNUNG: PI AI sidecar build fehlgeschlagen; Backend startet ohne Sidecar."
      return 0
    fi
  fi
  echo "[backend] Starting PI AI sidecar from ${AGENT_SERVICE_DIR}"
  npm --prefix "${AGENT_SERVICE_DIR}" start &
  AI_AGENT_PID=$!
  trap cleanup_agent_service EXIT INT TERM
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
    --cpp-build-dir)
      CPP_BUILD_DIR="$2"; shift 2;;
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
  mkdir -p "${CPP_BUILD_DIR}"
  CPP_CMAKE_ARGS=(
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"
  )
  if [[ "$(uname -s)" == "Darwin" ]] && command -v brew >/dev/null 2>&1; then
    _BREW_PREFIX="$(brew --prefix)"
    CPP_CMAKE_ARGS+=("-DCMAKE_PREFIX_PATH=${_BREW_PREFIX}")
  fi
  if [[ -n "${TILE_COMPILE_OPENCV_DIR}" ]]; then
    CPP_CMAKE_ARGS+=("-DOpenCV_DIR=${TILE_COMPILE_OPENCV_DIR}")
  fi
  if [[ -n "${TILE_COMPILE_CUDA_TOOLKIT_ROOT_DIR}" ]]; then
    CPP_CMAKE_ARGS+=("-DCUDA_TOOLKIT_ROOT_DIR=${TILE_COMPILE_CUDA_TOOLKIT_ROOT_DIR}")
  fi
  if [[ -n "${TILE_COMPILE_CUDA_NVCC_EXECUTABLE}" ]]; then
    CPP_CMAKE_ARGS+=("-DCUDA_NVCC_EXECUTABLE=${TILE_COMPILE_CUDA_NVCC_EXECUTABLE}")
  fi

  echo "[backend] Configuring C++ core in ${CPP_BUILD_DIR}"
  cmake -S "${PROJECT_ROOT}/tile_compile_cpp" -B "${CPP_BUILD_DIR}" "${CPP_CMAKE_ARGS[@]}"
  echo "[backend] Building tile_compile_runner and tile_compile_cli"
  cmake --build "${CPP_BUILD_DIR}" --parallel "$(nproc 2>/dev/null || sysctl -n hw.logicalcpu)" --target tile_compile_runner tile_compile_cli

  echo "[backend] Configuring C++ backend in ${BUILD_DIR}"
  BACKEND_CMAKE_ARGS=(-DCMAKE_BUILD_TYPE="${BUILD_TYPE}")
  if [[ "$(uname -s)" == "Darwin" ]] && command -v brew >/dev/null 2>&1; then
    BACKEND_CMAKE_ARGS+=("-DCMAKE_PREFIX_PATH=${_BREW_PREFIX}")
  fi
  cmake -S "${PROJECT_ROOT}/web_backend_cpp" -B "${BUILD_DIR}" "${BACKEND_CMAKE_ARGS[@]}"
  echo "[backend] Building tile_compile_web_backend"
  cmake --build "${BUILD_DIR}" --parallel "$(nproc 2>/dev/null || sysctl -n hw.logicalcpu)"
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
export TILE_COMPILE_AGENT_SERVICE_DIR="${AGENT_SERVICE_DIR}"
export TILE_COMPILE_RUNS_DIR="${RUNS_DIR}"
export TILE_COMPILE_ALLOWED_ROOTS="${ALLOWED_ROOTS}"
export TILE_COMPILE_OPENCV_DIR="${TILE_COMPILE_OPENCV_DIR}"
export TILE_COMPILE_CUDA_TOOLKIT_ROOT_DIR="${TILE_COMPILE_CUDA_TOOLKIT_ROOT_DIR}"
export TILE_COMPILE_CUDA_NVCC_EXECUTABLE="${TILE_COMPILE_CUDA_NVCC_EXECUTABLE}"

BACKEND_CMD=(
  "${BACKEND_BIN}"
)

if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  BACKEND_CMD+=("${EXTRA_ARGS[@]}")
fi

echo "[backend] Starting: ${BACKEND_CMD[*]}"
echo "[backend] UI: http://${HOST}:${PORT}/ui/"
start_agent_service
"${BACKEND_CMD[@]}"
