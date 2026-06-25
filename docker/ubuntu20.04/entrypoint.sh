#!/bin/bash
set -e

# entrypoint.sh – starts PI AI sidecar (if enabled) then the C++ backend

AGENT_SERVICE_DIR="${TILE_COMPILE_AGENT_SERVICE_DIR:-/opt/tile_compile/agent_service}"
AI_AGENT_AUTOSTART="${TILE_COMPILE_AI_AGENT_AUTOSTART:-1}"
SIDECAR_PID=""

cleanup() {
  if [ -n "${SIDECAR_PID}" ] && kill -0 "${SIDECAR_PID}" >/dev/null 2>&1; then
    echo "[entrypoint] Stopping PI AI sidecar pid=${SIDECAR_PID}"
    kill "${SIDECAR_PID}" >/dev/null 2>&1 || true
    wait "${SIDECAR_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

# Start PI AI sidecar
if [ "${AI_AGENT_AUTOSTART}" = "0" ]; then
  echo "[entrypoint] PI AI sidecar autostart disabled."
elif [ ! -f "${AGENT_SERVICE_DIR}/dist/server.js" ]; then
  echo "[entrypoint] PI AI sidecar not built: ${AGENT_SERVICE_DIR}/dist/server.js not found."
elif ! command -v node >/dev/null 2>&1; then
  echo "[entrypoint] WARNING: node not found; PI AI sidecar will not be started."
else
  echo "[entrypoint] Starting PI AI sidecar from ${AGENT_SERVICE_DIR}"
  node "${AGENT_SERVICE_DIR}/dist/server.js" &
  SIDECAR_PID=$!
  echo "[entrypoint] PI AI sidecar pid=${SIDECAR_PID}"
  # Give sidecar a moment to bind
  sleep 1
fi

# Start C++ backend (foreground)
echo "[entrypoint] Starting tile_compile_web_backend"
exec /opt/tile_compile/web_backend_cpp/build/tile_compile_web_backend
