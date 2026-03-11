#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAYLOAD_DIR="${SCRIPT_DIR}/payload"
INSTALL_ROOT="${HOME}/tilecompile"
LOG_DIR="${INSTALL_ROOT}/logs"
RUNS_DIR="${INSTALL_ROOT}/runs"
PID_FILE="${LOG_DIR}/gui2-backend.pid"
PORT="${TILE_COMPILE_GUI2_PORT:-8080}"
HOST="127.0.0.1"
URL="http://${HOST}:${PORT}/ui/"
BACKEND_BIN="${INSTALL_ROOT}/web_backend_cpp/build/tile_compile_web_backend"

log() {
  printf '[gui2] %s\n' "$*"
}

have_command() {
  command -v "$1" >/dev/null 2>&1
}

copy_payload() {
  mkdir -p "${INSTALL_ROOT}"
  if have_command rsync; then
    rsync -a --delete "${PAYLOAD_DIR}/" "${INSTALL_ROOT}/"
    return
  fi
  cp -a "${PAYLOAD_DIR}/." "${INSTALL_ROOT}/"
}

server_ready() {
  if have_command curl; then
    curl -fsS --max-time 2 "${URL}" >/dev/null 2>&1
    return $?
  fi
  return 1
}

open_browser() {
  if [[ "${TILE_COMPILE_GUI2_NO_BROWSER:-0}" == "1" ]]; then
    return
  fi
  if have_command xdg-open; then
    xdg-open "${URL}" >/dev/null 2>&1 &
    return
  fi
  if have_command open; then
    open "${URL}" >/dev/null 2>&1 &
    return
  fi
  log "Kein Browser-Launcher gefunden. Oeffne ${URL} manuell."
}

run_backend_foreground() {
  local lib_dir="${INSTALL_ROOT}/tile_compile_cpp/lib"
  local backend_pid=""

  if [[ ! -x "${BACKEND_BIN}" ]]; then
    log "Backend-Binary nicht gefunden: ${BACKEND_BIN}"
    exit 1
  fi

  export TILE_COMPILE_PROJECT_ROOT="${INSTALL_ROOT}"
  export TILE_COMPILE_HOST="${HOST}"
  export TILE_COMPILE_PORT="${PORT}"
  export TILE_COMPILE_CLI="${INSTALL_ROOT}/tile_compile_cpp/build/tile_compile_cli"
  export TILE_COMPILE_RUNNER="${INSTALL_ROOT}/tile_compile_cpp/build/tile_compile_runner"
  export TILE_COMPILE_RUNS_DIR="${RUNS_DIR}"
  export TILE_COMPILE_CONFIG="${INSTALL_ROOT}/tile_compile_cpp/tile_compile.yaml"
  export TILE_COMPILE_SCHEMA="${INSTALL_ROOT}/tile_compile_cpp/tile_compile.schema.yaml"
  export TILE_COMPILE_PRESETS_DIR="${INSTALL_ROOT}/tile_compile_cpp/examples"
  export TILE_COMPILE_UI_DIR="${INSTALL_ROOT}/web_frontend"
  export TILE_COMPILE_ALLOWED_ROOTS="${INSTALL_ROOT}:$(printf '%s' "${HOME}"):/tmp:/media"
  if [[ -d "${lib_dir}" ]]; then
    export LD_LIBRARY_PATH="${lib_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    export DYLD_LIBRARY_PATH="${lib_dir}${DYLD_LIBRARY_PATH:+:${DYLD_LIBRARY_PATH}}"
  fi

  cleanup_backend() {
    local exit_code="${1:-0}"
    trap - EXIT INT TERM
    if [[ -n "${backend_pid}" ]] && kill -0 "${backend_pid}" 2>/dev/null; then
      log "Beende Crow-Backend."
      kill "${backend_pid}" 2>/dev/null || true
      wait "${backend_pid}" 2>/dev/null || true
    fi
    return "${exit_code}"
  }

  log "Starte Crow-Backend im Vordergrund auf ${URL} (Ctrl+C zum Beenden)."
  ( sleep 1; open_browser ) &
  "${BACKEND_BIN}" &
  backend_pid="$!"
  log "Crow-Backend laeuft mit PID ${backend_pid}."

  trap 'cleanup_backend $?' EXIT
  trap 'exit 130' INT TERM

  local exit_code=0
  if wait "${backend_pid}"; then
    exit_code=0
  else
    exit_code=$?
  fi
  cleanup_backend "${exit_code}"
  return "${exit_code}"
}

main() {
  if [[ ! -d "${PAYLOAD_DIR}" ]]; then
    log "payload/ nicht gefunden."
    exit 1
  fi

  copy_payload
  mkdir -p "${LOG_DIR}" "${RUNS_DIR}"

  if server_ready; then
    log "GUI2-Backend laeuft bereits."
    open_browser
    exit 0
  fi

  run_backend_foreground
}

main "$@"
