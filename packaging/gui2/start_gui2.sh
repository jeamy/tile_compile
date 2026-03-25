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

has_installed_layout() {
  [[ -x "${BACKEND_BIN}" && -d "${INSTALL_ROOT}/web_frontend" && -d "${INSTALL_ROOT}/tile_compile_cpp" ]]
}

install_launcher_scripts() {
  local src_sh="${SCRIPT_DIR}/start_gui2.sh"
  local src_command="${SCRIPT_DIR}/start_gui2.command"
  local dst_sh="${INSTALL_ROOT}/start_gui2.sh"
  local dst_command="${INSTALL_ROOT}/start_gui2.command"

  if [[ -f "${src_sh}" && ! -f "${dst_sh}" ]]; then
    cp -a "${src_sh}" "${dst_sh}"
    chmod +x "${dst_sh}"
  fi
  if [[ -f "${src_command}" && ! -f "${dst_command}" ]]; then
    cp -a "${src_command}" "${dst_command}"
    chmod +x "${dst_command}"
  fi
}

copy_payload() {
  mkdir -p "${INSTALL_ROOT}"
  
  # Check if this is an update (installation already exists)
  local is_update=false
  if [[ -d "${INSTALL_ROOT}/web_backend_cpp" || -d "${INSTALL_ROOT}/web_frontend" || -d "${INSTALL_ROOT}/tile_compile_cpp" ]]; then
    is_update=true
    log "Existierende Installation gefunden - fuehre selektives Update durch"
  else
    log "Neue Installation - kopiere alle Dateien"
  fi
  
  if [[ "${is_update}" == "true" ]]; then
    # Selective update: only replace app directories, preserve user data
    # Remove old app directories
    rm -rf "${INSTALL_ROOT}/web_frontend" "${INSTALL_ROOT}/web_backend_cpp" "${INSTALL_ROOT}/tile_compile_cpp"
    
    # Copy only app directories from payload
    if [[ -d "${PAYLOAD_DIR}/web_frontend" ]]; then
      cp -a "${PAYLOAD_DIR}/web_frontend" "${INSTALL_ROOT}/"
    fi
    if [[ -d "${PAYLOAD_DIR}/web_backend_cpp" ]]; then
      cp -a "${PAYLOAD_DIR}/web_backend_cpp" "${INSTALL_ROOT}/"
    fi
    if [[ -d "${PAYLOAD_DIR}/tile_compile_cpp" ]]; then
      cp -a "${PAYLOAD_DIR}/tile_compile_cpp" "${INSTALL_ROOT}/"
    fi
    
    log "App-Dateien aktualisiert. User-Daten (configs, runs, astap, pcc) bleiben erhalten."
  else
    # Fresh install: copy everything
    if have_command rsync; then
      rsync -a --delete "${PAYLOAD_DIR}/" "${INSTALL_ROOT}/"
    else
      cp -a "${PAYLOAD_DIR}/." "${INSTALL_ROOT}/"
    fi
  fi
  
  install_launcher_scripts
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

launch_in_terminal() {
  # If already in terminal, just run directly
  if [[ -t 0 ]] && [[ -t 1 ]]; then
    return 1
  fi
  
  # Try to launch in a new terminal window
  local script_path="$0"
  local script_args="$*"
  
  # Try common Linux terminal emulators
  if have_command gnome-terminal; then
    gnome-terminal -- bash -c "cd '${PWD}' && exec '${script_path}' ${script_args}; read -p 'Press Enter to close...'"
    return 0
  elif have_command konsole; then
    konsole -e bash -c "cd '${PWD}' && exec '${script_path}' ${script_args}; read -p 'Press Enter to close...'"
    return 0
  elif have_command xfce4-terminal; then
    xfce4-terminal -e "bash -c \"cd '${PWD}' && exec '${script_path}' ${script_args}; read -p 'Press Enter to close...'\""
    return 0
  elif have_command xterm; then
    xterm -e bash -c "cd '${PWD}' && exec '${script_path}' ${script_args}; read -p 'Press Enter to close...'"
    return 0
  elif have_command x-terminal-emulator; then
    x-terminal-emulator -e bash -c "cd '${PWD}' && exec '${script_path}' ${script_args}; read -p 'Press Enter to close...'"
    return 0
  fi
  
  return 1
}

run_backend_foreground() {
  local lib_dir="${INSTALL_ROOT}/tile_compile_cpp/lib"
  local backend_lib_dir="${INSTALL_ROOT}/web_backend_cpp/lib"
  local backend_pid=""

  if [[ ! -x "${BACKEND_BIN}" ]]; then
    log "Backend-Binary nicht gefunden: ${BACKEND_BIN}"
    exit 1
  fi

  export TILE_COMPILE_PROJECT_ROOT="${INSTALL_ROOT}"
  export TILE_COMPILE_HOST="${HOST}"
  export TILE_COMPILE_PORT="${PORT}"
  
  local cli_bin="${INSTALL_ROOT}/tile_compile_cpp/build/tile_compile_cli"
  local runner_bin="${INSTALL_ROOT}/tile_compile_cpp/build/tile_compile_runner"
  
  if [[ ! -x "${cli_bin}" ]]; then
    log "WARNUNG: tile_compile_cli nicht gefunden oder nicht ausfuehrbar: ${cli_bin}"
  fi
  if [[ ! -x "${runner_bin}" ]]; then
    log "WARNUNG: tile_compile_runner nicht gefunden oder nicht ausfuehrbar: ${runner_bin}"
  fi
  
  export TILE_COMPILE_CLI="${cli_bin}"
  export TILE_COMPILE_RUNNER="${runner_bin}"
  export TILE_COMPILE_RUNS_DIR="${RUNS_DIR}"
  export TILE_COMPILE_CONFIG="${INSTALL_ROOT}/tile_compile_cpp/tile_compile.yaml"
  export TILE_COMPILE_SCHEMA="${INSTALL_ROOT}/tile_compile_cpp/tile_compile.schema.yaml"
  export TILE_COMPILE_PRESETS_DIR="${INSTALL_ROOT}/tile_compile_cpp/examples"
  export TILE_COMPILE_UI_DIR="${INSTALL_ROOT}/web_frontend"
  export TILE_COMPILE_GUI2_INSTALL_ROOT="${INSTALL_ROOT}"
  local allowed_roots=("${INSTALL_ROOT}" "$(printf '%s' "${HOME}")" "/tmp" "/media")
  if [[ -n "${TMPDIR:-}" ]]; then
    allowed_roots+=("${TMPDIR%/}")
  fi
  local allowed_roots_joined
  allowed_roots_joined="$(IFS=:; printf '%s' "${allowed_roots[*]}")"
  export TILE_COMPILE_ALLOWED_ROOTS="${allowed_roots_joined}"
  export TILE_COMPILE_INPUT_SEARCH_ROOTS="${allowed_roots_joined}"
  # Optional backend memory guard overrides:
  # TILE_COMPILE_BACKEND_SUBPROCESS_CAPTURE_BYTES (default 1048576)
  # TILE_COMPILE_BACKEND_JOB_STDIO_STORE_BYTES (default 131072)
  # TILE_COMPILE_BACKEND_SCAN_FRAMES_PREVIEW (default 256)
  # TILE_COMPILE_BACKEND_SCAN_PER_DIR_FRAMES_PREVIEW (default 32)
  # TILE_COMPILE_BACKEND_SCAN_PER_DIR_RESULTS_PREVIEW (default 64)
  # TILE_COMPILE_BACKEND_SCAN_MESSAGES_PREVIEW (default 128)
  # TILE_COMPILE_BACKEND_SCAN_COLOR_CANDIDATES_PREVIEW (default 32)
  # TILE_COMPILE_BACKEND_REPORT_EVENTS_MAX (default 4096)
  # TILE_COMPILE_BACKEND_REPORT_LOG_TAIL (default 128)
  # TILE_COMPILE_BACKEND_REPORT_TEXT_BYTES (default 262144)
  # TILE_COMPILE_BACKEND_REPORT_JSON_FILE_BYTES (default 4194304)
  # TILE_COMPILE_BACKEND_RETAINED_JOBS (default 128)
  local lib_paths=()
  if [[ -d "${lib_dir}" ]]; then
    lib_paths+=("${lib_dir}")
  fi
  if [[ -d "${backend_lib_dir}" ]]; then
    lib_paths+=("${backend_lib_dir}")
  fi
  if [[ "${#lib_paths[@]}" -gt 0 ]]; then
    local joined_libs
    joined_libs="$(IFS=:; printf '%s' "${lib_paths[*]}")"
    export LD_LIBRARY_PATH="${joined_libs}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    export DYLD_LIBRARY_PATH="${joined_libs}${DYLD_LIBRARY_PATH:+:${DYLD_LIBRARY_PATH}}"
  fi

  log "Starte Crow-Backend im Vordergrund auf ${URL} (Ctrl+C zum Beenden)."
  log "Konsole bleibt sichtbar - Backend-Ausgaben werden hier angezeigt."
  log ""
  log "==================================================================="
  log "  Oeffne im Browser: ${URL}"
  log "==================================================================="
  log ""
  
  if [[ "${TILE_COMPILE_GUI2_NO_BROWSER:-0}" != "1" ]]; then
    ( sleep 2; open_browser ) &
  fi
  
  local exit_code=0
  "${BACKEND_BIN}" || exit_code=$?
  
  if [[ "${exit_code}" -ne 0 ]]; then
    log ""
    log "Backend wurde mit Exit-Code ${exit_code} beendet."
    log "Druecke Enter zum Schliessen..."
    read -r
  fi
  
  return "${exit_code}"
}

main() {
  # If not running in a terminal, launch in new terminal window
  if launch_in_terminal "$@"; then
    exit 0
  fi
  
  if [[ -d "${PAYLOAD_DIR}" ]]; then
    copy_payload
  elif ! has_installed_layout; then
    log "payload/ nicht gefunden und Installationslayout unvollstaendig."
    exit 1
  fi

  mkdir -p "${LOG_DIR}" "${RUNS_DIR}"

  if server_ready; then
    log "GUI2-Backend laeuft bereits."
    open_browser
    exit 0
  fi

  run_backend_foreground
}

main "$@"
