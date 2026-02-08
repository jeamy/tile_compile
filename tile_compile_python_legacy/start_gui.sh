#!/usr/bin/env bash
set -e

repo_root="$(cd "$(dirname "$0")" && pwd)"
cd "${repo_root}"

echo "Info: starting Qt6 GUI (Python)." >&2

check_pyside6() {
  local py="$1"
  if [ -z "${py}" ]; then return 1; fi
  if ! command -v "${py}" >/dev/null 2>&1; then return 1; fi
  "${py}" -c "import PySide6" >/dev/null 2>&1
}

python_bin=""

if [ -n "${TILE_COMPILE_PYTHON:-}" ]; then
  if check_pyside6 "${TILE_COMPILE_PYTHON}"; then
    python_bin="${TILE_COMPILE_PYTHON}"
  fi
fi

if [ -z "${python_bin}" ] && [ -x ".venv/bin/python" ]; then
  if check_pyside6 ".venv/bin/python"; then
    python_bin=".venv/bin/python"
  fi
fi

if [ -z "${python_bin}" ] && [ -x "venv/bin/python" ]; then
  if check_pyside6 "venv/bin/python"; then
    python_bin="venv/bin/python"
  fi
fi

if [ -z "${python_bin}" ]; then
  for interpreter in "python3.12" "python3" "python" "python3.11" "python3.10" "python3.9"; do
    if check_pyside6 "${interpreter}"; then
      python_bin="${interpreter}"
      break
    fi
  done
fi

if [ -z "${python_bin}" ]; then
  echo "ERROR: Python dependency missing: PySide6" >&2
  echo "Please install it in your environment (e.g. venv) before running the GUI." >&2
  exit 1
fi

if [ ! -f "gui/main.py" ]; then
  echo "ERROR: gui/main.py not found" >&2
  exit 1
fi

echo "Info: using python: ${python_bin}" >&2
exec "${python_bin}" -m gui.main
