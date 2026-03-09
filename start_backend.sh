#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR_DEFAULT="${PROJECT_ROOT}/.venv"
VENV_DIR="${VENV_DIR:-${VENV_DIR_DEFAULT}}"
REQ_FILE="${PROJECT_ROOT}/web_backend/requirements-backend.txt"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
RELOAD="1"
INSTALL_REQS="1"

usage() {
  cat <<EOF
Usage: $(basename "$0") [options] [-- <extra uvicorn args>]

Options:
  --host <host>         Uvicorn host (default: ${HOST})
  --port <port>         Uvicorn port (default: ${PORT})
  --venv <path>         Virtualenv path (default: ${VENV_DIR_DEFAULT})
  --python <binary>     Python binary for venv creation (default: python3)
  --no-reload           Disable uvicorn --reload
  --no-install          Skip pip install -r requirements-backend.txt
  -h, --help            Show this help

Env overrides:
  HOST, PORT, VENV_DIR, PYTHON_BIN
EOF
}

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --host)
      HOST="$2"; shift 2;;
    --port)
      PORT="$2"; shift 2;;
    --venv)
      VENV_DIR="$2"; shift 2;;
    --python)
      PYTHON_BIN="$2"; shift 2;;
    --no-reload)
      RELOAD="0"; shift;;
    --no-install)
      INSTALL_REQS="0"; shift;;
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

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "[backend] Creating venv at ${VENV_DIR}"
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"

if [[ "${INSTALL_REQS}" == "1" ]]; then
  echo "[backend] Installing requirements from ${REQ_FILE}"
  python -m pip install --upgrade pip
  python -m pip install -r "${REQ_FILE}"
fi

UVICORN_CMD=(
  uvicorn
  app.main:app
  --app-dir "${PROJECT_ROOT}/web_backend"
  --host "${HOST}"
  --port "${PORT}"
)

if [[ "${RELOAD}" == "1" ]]; then
  UVICORN_CMD+=(--reload)
fi

if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  UVICORN_CMD+=("${EXTRA_ARGS[@]}")
fi

echo "[backend] Starting: ${UVICORN_CMD[*]}"
exec "${UVICORN_CMD[@]}"
