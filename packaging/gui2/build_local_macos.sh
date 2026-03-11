#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

TAG="${TAG:-local}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
BUILD_SUFFIX="${BUILD_SUFFIX:-build-gui2-release}"
PORT="${PORT:-8080}"
SKIP_BUILD=0
SKIP_SMOKE=0

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --tag <tag>            Bundle tag suffix (default: ${TAG})
  --build-type <type>    CMake build type (default: ${BUILD_TYPE})
  --build-suffix <name>  Build dir suffix (default: ${BUILD_SUFFIX})
  --port <port>          Smoke-test port (default: ${PORT})
  --skip-build           Skip configure/build
  --skip-smoke           Skip smoke test
  -h, --help             Show this help

Environment overrides:
  TAG, BUILD_TYPE, BUILD_SUFFIX, PORT
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tag) TAG="$2"; shift 2 ;;
    --build-type) BUILD_TYPE="$2"; shift 2 ;;
    --build-suffix) BUILD_SUFFIX="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --skip-build) SKIP_BUILD=1; shift ;;
    --skip-smoke) SKIP_SMOKE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

RUNNER_BUILD_DIR="${PROJECT_ROOT}/tile_compile_cpp/${BUILD_SUFFIX}"
BACKEND_BUILD_DIR="${PROJECT_ROOT}/web_backend_cpp/${BUILD_SUFFIX}"
ARTIFACTS_DIR="${PROJECT_ROOT}/artifacts"
BUNDLE_DIR="${PROJECT_ROOT}/bundle"
ROOT="${BUNDLE_DIR}/tile_compile_gui2-macos-${TAG}"
PAYLOAD="${ROOT}/payload"

require_cmd() {
  local cmd="$1"
  local hint="$2"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "[gui2-package] Missing required command: ${cmd}" >&2
    echo "[gui2-package] ${hint}" >&2
    exit 1
  fi
}

preflight_macos() {
  if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "[gui2-package] This script must be run on macOS." >&2
    exit 1
  fi

  local macos_major
  macos_major="$(sw_vers -productVersion | awk -F. '{print $1}')"

  require_cmd xcode-select "Install Xcode Command Line Tools with: xcode-select --install"
  if ! xcrun --find clang++ >/dev/null 2>&1; then
    echo "[gui2-package] No macOS C++ compiler found." >&2
    echo "[gui2-package] Install Xcode Command Line Tools with: xcode-select --install" >&2
    exit 1
  fi

  require_cmd cmake "Install CMake, for example with: brew install cmake"
  require_cmd ninja "Install Ninja, for example with: brew install ninja"
  require_cmd pkg-config "Install pkg-config, for example with: brew install pkg-config"
  require_cmd otool "Xcode Command Line Tools are required."
  require_cmd install_name_tool "Xcode Command Line Tools are required."
  require_cmd ditto "macOS system tool 'ditto' is required."
  require_cmd python3 "Install Python 3, for example with: brew install python"

  if [[ "${macos_major}" -lt 13 ]]; then
    if ! pkg-config --exists opencv4; then
      echo "[gui2-package] OpenCV is not available via the default Homebrew formula on macOS 12." >&2
      echo "[gui2-package] Homebrew currently requires newer macOS releases for its opencv formula." >&2
      echo "[gui2-package] Use macOS 13+ for the Homebrew-based path, or provide a working local OpenCV installation with pkg-config metadata." >&2
      exit 1
    fi
  fi
}

build_all() {
  cmake -S "${PROJECT_ROOT}/tile_compile_cpp" -B "${RUNNER_BUILD_DIR}" -G Ninja \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -DBUILD_TESTS=OFF
  cmake --build "${RUNNER_BUILD_DIR}" -j"$(sysctl -n hw.ncpu)"

  cmake -S "${PROJECT_ROOT}/web_backend_cpp" -B "${BACKEND_BUILD_DIR}" -G Ninja \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -DBUILD_TESTS=OFF \
    -DTILE_COMPILE_BACKEND_STATIC_STDLIB=OFF
  cmake --build "${BACKEND_BUILD_DIR}" -j"$(sysctl -n hw.ncpu)"
}

collect_macos_libs() {
  local target="$1"
  otool -L "$target" \
    | tail -n +2 \
    | awk '{print $1}' \
    | while read -r dep; do
        local name
        name="$(basename "$dep")"
        case "$dep" in
          /usr/lib/*|/System/*) continue ;;
        esac
        case "$name" in
          libc++.1.dylib|libc++abi.1.dylib|libunwind.1.dylib) continue ;;
        esac
        [[ -f "$dep" ]] || continue
        local dst="${PAYLOAD}/tile_compile_cpp/lib/${name}"
        if [[ ! -f "$dst" ]]; then
          cp "$dep" "$dst"
          chmod u+w "$dst" || true
          collect_macos_libs "$dst"
        fi
      done
}

rewrite_macos_refs() {
  local target="$1"
  local mode="$2"
  chmod u+w "$target" || true
  if [[ "$mode" == "exe" ]]; then
    install_name_tool -add_rpath "@executable_path/../lib" "$target" 2>/dev/null || true
  else
    install_name_tool -id "@loader_path/$(basename "$target")" "$target" || true
  fi
  otool -L "$target" \
    | tail -n +2 \
    | awk '{print $1}' \
    | while read -r dep; do
        local name
        name="$(basename "$dep")"
        case "$dep" in
          /usr/lib/*|/System/*) continue ;;
        esac
        case "$name" in
          libc++.1.dylib|libc++abi.1.dylib|libunwind.1.dylib) continue ;;
        esac
        local bundled="${PAYLOAD}/tile_compile_cpp/lib/${name}"
        [[ -f "$bundled" ]] || continue
        if [[ "$mode" == "exe" ]]; then
          install_name_tool -change "$dep" "@rpath/${name}" "$target" || true
        else
          install_name_tool -change "$dep" "@loader_path/${name}" "$target" || true
        fi
      done
}

assemble_bundle() {
  rm -rf "${ROOT}"
  mkdir -p "${ARTIFACTS_DIR}" "${PAYLOAD}/tile_compile_cpp/build" "${PAYLOAD}/tile_compile_cpp/lib" "${PAYLOAD}/web_backend_cpp/build"
  cp "${PROJECT_ROOT}/packaging/gui2/start_gui2.sh" "${ROOT}/start_gui2.sh"
  cp "${PROJECT_ROOT}/packaging/gui2/start_gui2.command" "${ROOT}/start_gui2.command"
  chmod +x "${ROOT}/start_gui2.sh" "${ROOT}/start_gui2.command"
  cp -a "${PROJECT_ROOT}/web_frontend" "${PAYLOAD}/"
  cp -a "${PROJECT_ROOT}/tile_compile_cpp/examples" "${PAYLOAD}/tile_compile_cpp/"
  cp "${PROJECT_ROOT}/tile_compile_cpp/tile_compile.yaml" "${PAYLOAD}/tile_compile_cpp/"
  cp "${PROJECT_ROOT}/tile_compile_cpp/tile_compile.schema.yaml" "${PAYLOAD}/tile_compile_cpp/"
  cp "${PROJECT_ROOT}/tile_compile_cpp/tile_compile.schema.json" "${PAYLOAD}/tile_compile_cpp/"
  cp "${RUNNER_BUILD_DIR}/tile_compile_runner" "${PAYLOAD}/tile_compile_cpp/build/"
  cp "${RUNNER_BUILD_DIR}/tile_compile_cli" "${PAYLOAD}/tile_compile_cpp/build/"
  cp "${BACKEND_BUILD_DIR}/tile_compile_web_backend" "${PAYLOAD}/web_backend_cpp/build/"
  collect_macos_libs "${PAYLOAD}/tile_compile_cpp/build/tile_compile_runner"
  collect_macos_libs "${PAYLOAD}/tile_compile_cpp/build/tile_compile_cli"
  collect_macos_libs "${PAYLOAD}/web_backend_cpp/build/tile_compile_web_backend"
  rewrite_macos_refs "${PAYLOAD}/tile_compile_cpp/build/tile_compile_runner" exe
  rewrite_macos_refs "${PAYLOAD}/tile_compile_cpp/build/tile_compile_cli" exe
  rewrite_macos_refs "${PAYLOAD}/web_backend_cpp/build/tile_compile_web_backend" exe
  install_name_tool -add_rpath "@executable_path/../../tile_compile_cpp/lib" \
    "${PAYLOAD}/web_backend_cpp/build/tile_compile_web_backend" 2>/dev/null || true
  find "${PAYLOAD}/tile_compile_cpp/lib" -type f \( -name "*.dylib" -o -name "*.so*" \) -print0 \
    | while IFS= read -r -d '' dylib; do
        rewrite_macos_refs "$dylib" dylib
      done
  printf '%s\n' "${TAG}" > "${PAYLOAD}/.gui2-release-tag"
  ditto -c -k --sequesterRsrc --keepParent "${ROOT}" "${ARTIFACTS_DIR}/tile_compile_gui2-macos-${TAG}.zip"
}

smoke_test() {
  local root="${ROOT}"
  export HOME="${PROJECT_ROOT}/smoke-home-macos"
  export TILE_COMPILE_GUI2_NO_BROWSER=1
  export TILE_COMPILE_GUI2_PORT="${PORT}"
  rm -rf "${HOME}"
  mkdir -p "${HOME}"
  bash "${root}/start_gui2.sh" >/tmp/out_gui2_smoke_macos.txt 2>&1 &
  local start_pid=$!
  trap 'kill "${start_pid}" 2>/dev/null || true; wait "${start_pid}" 2>/dev/null || true' EXIT
  python3 - <<'PY'
import json
import os
import time
from urllib.request import urlopen

port = os.environ["TILE_COMPILE_GUI2_PORT"]
url = f"http://127.0.0.1:{port}/api/app/state"
deadline = time.time() + 30
while time.time() < deadline:
    try:
        with urlopen(url, timeout=2) as resp:
            data = json.load(resp)
            assert isinstance(data, dict)
            break
    except Exception:
        time.sleep(1)
else:
    raise SystemExit("backend smoke test failed")
PY
  kill "${start_pid}" 2>/dev/null || true
  wait "${start_pid}" 2>/dev/null || true
  trap - EXIT
}

main() {
  preflight_macos
  [[ "${SKIP_BUILD}" == "1" ]] || build_all
  assemble_bundle
  [[ "${SKIP_SMOKE}" == "1" ]] || smoke_test
  echo "[gui2-package] Created ${ARTIFACTS_DIR}/tile_compile_gui2-macos-${TAG}.zip"
}

main
