#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAYLOAD_DIR="${SCRIPT_DIR}/payload"
INSTALL_ROOT="${HOME}/tilecompile"
VENV_DIR="${INSTALL_ROOT}/.venv"
LOG_DIR="${INSTALL_ROOT}/logs"
RUNS_DIR="${INSTALL_ROOT}/runs"
PID_FILE="${LOG_DIR}/gui2-backend.pid"
PORT="${TILE_COMPILE_GUI2_PORT:-8080}"
HOST="127.0.0.1"
URL="http://${HOST}:${PORT}/ui/"

log() {
  printf '[gui2] %s\n' "$*"
}

have_command() {
  command -v "$1" >/dev/null 2>&1
}

detect_python() {
  if have_command python3; then
    command -v python3
    return 0
  fi
  if have_command python; then
    command -v python
    return 0
  fi
  return 1
}

confirm_python_install() {
  if [[ ! -t 0 ]]; then
    log "Python 3.11+ fehlt. Ohne Python starten Backend und Reports nicht."
    return 1
  fi
  local answer=""
  printf "Python 3.11+ wurde nicht gefunden. Jetzt installieren? Ohne Python starten GUI2-Backend und Reports nicht. [y/N] "
  read -r answer
  case "${answer}" in
    y|Y|yes|YES) return 0 ;;
    *) return 1 ;;
  esac
}

install_python_linux() {
  log "Python wurde nicht gefunden. Versuche Installation."
  if have_command apt-get; then
    if ! have_command sudo; then
      log "sudo fehlt. Bitte installiere python3 python3-venv python3-pip manuell."
      return 1
    fi
    sudo apt-get update
    sudo apt-get install -y python3 python3-venv python3-pip
    return 0
  fi
  if have_command dnf; then
    if ! have_command sudo; then
      log "sudo fehlt. Bitte installiere python3 python3-virtualenv python3-pip manuell."
      return 1
    fi
    sudo dnf install -y python3 python3-pip
    return 0
  fi
  if have_command pacman; then
    if ! have_command sudo; then
      log "sudo fehlt. Bitte installiere python python-pip manuell."
      return 1
    fi
    sudo pacman -Sy --noconfirm python python-pip
    return 0
  fi
  log "Kein unterstuetzter Paketmanager erkannt. Bitte Python 3.11+ manuell installieren."
  return 1
}

install_python_macos() {
  log "Python wurde nicht gefunden. Versuche Installation."
  if have_command brew; then
    brew install python@3.11
    return 0
  fi
  log "Homebrew wurde nicht gefunden. Bitte Python 3.11+ manuell installieren."
  return 1
}

ensure_python() {
  local py_bin=""
  if py_bin="$(detect_python)"; then
    printf '%s\n' "$py_bin"
    return 0
  fi

  if ! confirm_python_install; then
    log "Python wurde nicht installiert. Die App funktioniert ohne Python nicht."
    return 1
  fi

  case "$(uname -s)" in
    Darwin)
      install_python_macos || return 1
      ;;
    Linux)
      install_python_linux || return 1
      ;;
    *)
      log "Unbekanntes System. Bitte Python 3.11+ manuell installieren."
      return 1
      ;;
  esac

  if py_bin="$(detect_python)"; then
    printf '%s\n' "$py_bin"
    return 0
  fi

  log "Python konnte nicht automatisch installiert werden."
  return 1
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
  local py_bin="$1"
  "${py_bin}" - <<PY >/dev/null 2>&1
import sys
from urllib.request import urlopen

try:
    with urlopen("${URL}", timeout=1.5) as resp:
        sys.exit(0 if resp.status < 500 else 1)
except Exception:
    sys.exit(1)
PY
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

start_backend() {
  local venv_python="$1"
  local log_file="${LOG_DIR}/gui2-backend.log"
  local lib_dir="${INSTALL_ROOT}/tile_compile_cpp/lib"

  export TILE_COMPILE_CLI="${INSTALL_ROOT}/tile_compile_cpp/build/tile_compile_cli"
  export TILE_COMPILE_RUNNER="${INSTALL_ROOT}/tile_compile_cpp/build/tile_compile_runner"
  export TILE_COMPILE_RUNS_DIR="${RUNS_DIR}"
  export TILE_COMPILE_CONFIG_PATH="${INSTALL_ROOT}/tile_compile_cpp/tile_compile.yaml"
  export TILE_COMPILE_STATS_SCRIPT="${INSTALL_ROOT}/tile_compile_cpp/scripts/generate_report.py"
  export TILE_COMPILE_ALLOWED_ROOTS="${INSTALL_ROOT}:$(printf '%s' "${HOME}"):/tmp:/media"
  export PYTHONUNBUFFERED="1"
  if [[ -d "${lib_dir}" ]]; then
    export LD_LIBRARY_PATH="${lib_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    export DYLD_LIBRARY_PATH="${lib_dir}${DYLD_LIBRARY_PATH:+:${DYLD_LIBRARY_PATH}}"
  fi

  nohup "${venv_python}" -m uvicorn app.main:app \
    --app-dir "${INSTALL_ROOT}/web_backend" \
    --host "${HOST}" \
    --port "${PORT}" \
    >"${log_file}" 2>&1 &
  echo "$!" > "${PID_FILE}"
}

main() {
  if [[ ! -d "${PAYLOAD_DIR}" ]]; then
    log "payload/ nicht gefunden."
    exit 1
  fi

  local system_python=""
  if ! system_python="$(ensure_python)"; then
    log "Start abgebrochen: GUI2 funktioniert ohne Python nicht."
    exit 1
  fi

  copy_payload
  mkdir -p "${LOG_DIR}" "${RUNS_DIR}"

  if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    log "Erzeuge virtuelle Umgebung unter ${VENV_DIR}"
    "${system_python}" -m venv "${VENV_DIR}"
  fi

  local venv_python="${VENV_DIR}/bin/python"
  log "Installiere Python-Requirements in ${VENV_DIR}"
  "${venv_python}" -m pip install --upgrade pip
  "${venv_python}" -m pip install -r "${INSTALL_ROOT}/web_backend/requirements-backend.txt"

  if [[ -f "${PID_FILE}" ]]; then
    local existing_pid
    existing_pid="$(cat "${PID_FILE}" 2>/dev/null || true)"
    if [[ -n "${existing_pid}" ]] && kill -0 "${existing_pid}" >/dev/null 2>&1; then
      if server_ready "${venv_python}"; then
        log "GUI2-Backend laeuft bereits."
        open_browser
        exit 0
      fi
    fi
  fi

  if ! server_ready "${venv_python}"; then
    log "Starte FastAPI-Backend auf ${URL}"
    start_backend "${venv_python}"
  fi

  for _ in $(seq 1 20); do
    if server_ready "${venv_python}"; then
      open_browser
      exit 0
    fi
    sleep 1
  done

  log "Backend wurde nicht rechtzeitig erreichbar. Siehe ${LOG_DIR}/gui2-backend.log"
  exit 1
}

main "$@"
