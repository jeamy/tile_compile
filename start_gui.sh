#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

gui_dir="${repo_root}/gui"

if [[ ! -d "${gui_dir}" ]]; then
  echo "ERROR: gui directory not found: ${gui_dir}" >&2
  exit 1
fi

echo "Info: GUI workflow: Scan input first; if color mode is UNKNOWN you must confirm it before Start. The confirmation is stored as color_mode_confirmed in runs/<run_id>/run_metadata.json" >&2

need_cmd() {
  local cmd="$1"
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    echo "ERROR: missing prerequisite: ${cmd}" >&2
    return 1
  fi
}

need_cmd npm
need_cmd cargo
need_cmd rustc

# Wayland workaround (GTK/GDK protocol error 71 on some setups)
if [[ "${XDG_SESSION_TYPE:-}" == "wayland" ]]; then
  export GDK_BACKEND="${GDK_BACKEND:-x11}"
  export WINIT_UNIX_BACKEND="${WINIT_UNIX_BACKEND:-x11}"
  echo "Info: Wayland session detected -> forcing X11 backend: GDK_BACKEND=${GDK_BACKEND}, WINIT_UNIX_BACKEND=${WINIT_UNIX_BACKEND}" >&2
fi

# WebKitGTK / GBM / dmabuf workaround (can crash on some Linux GPU/driver setups)
# - Disables dmabuf renderer path which can trigger: "Failed to create GBM buffer ... invalid argument"
export WEBKIT_DISABLE_DMABUF_RENDERER="${WEBKIT_DISABLE_DMABUF_RENDERER:-1}"

# Optional hard fallback: force software rendering
# Enable via: TILE_COMPILE_GUI_SOFTWARE=1 ./start_gui.sh
if [[ "${TILE_COMPILE_GUI_SOFTWARE:-}" == "1" ]]; then
  export LIBGL_ALWAYS_SOFTWARE="${LIBGL_ALWAYS_SOFTWARE:-1}"
  export GDK_GL="${GDK_GL:-disable}"
  echo "Info: software rendering enabled (LIBGL_ALWAYS_SOFTWARE=1, GDK_GL=disable)" >&2
fi

cd "${gui_dir}"

# No auto-install: user must run 'npm install' manually.
if [[ ! -d "node_modules" ]]; then
  echo "ERROR: node_modules missing in ${gui_dir}" >&2
  echo "Please run: (cd gui && npm install)" >&2
  exit 1
fi

exec npm run dev
