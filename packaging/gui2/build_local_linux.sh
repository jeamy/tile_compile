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
ROOT="${BUNDLE_DIR}/tile_compile_gui2-linux-${TAG}"
PAYLOAD="${ROOT}/payload"

build_all() {
  cmake -S "${PROJECT_ROOT}/tile_compile_cpp" -B "${RUNNER_BUILD_DIR}" -G Ninja \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -DBUILD_TESTS=OFF
  cmake --build "${RUNNER_BUILD_DIR}" -j"$(nproc)"

  cmake -S "${PROJECT_ROOT}/web_backend_cpp" -B "${BACKEND_BUILD_DIR}" -G Ninja \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -DBUILD_TESTS=OFF \
    -DTILE_COMPILE_BACKEND_STATIC_STDLIB=OFF
  cmake --build "${BACKEND_BUILD_DIR}" -j"$(nproc)"
}

bundle_linux_libs() {
  local exe="$1"
  ldd "$exe" \
    | awk '/=> \// { print $3 }' \
    | sort -u \
    | while read -r dep; do
        [[ -f "$dep" ]] || continue
        case "$dep" in
          /lib/*|/lib64/*) continue ;;
        esac
        cp -n "$dep" "${PAYLOAD}/tile_compile_cpp/lib/" || true
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
  bundle_linux_libs "${PAYLOAD}/tile_compile_cpp/build/tile_compile_runner"
  bundle_linux_libs "${PAYLOAD}/tile_compile_cpp/build/tile_compile_cli"
  bundle_linux_libs "${PAYLOAD}/web_backend_cpp/build/tile_compile_web_backend"
  printf '%s\n' "${TAG}" > "${PAYLOAD}/.gui2-release-tag"
  (
    cd "${BUNDLE_DIR}"
    zip -r "${ARTIFACTS_DIR}/tile_compile_gui2-linux-${TAG}.zip" "tile_compile_gui2-linux-${TAG}"
  )
}

smoke_test() {
  local root="${ROOT}"
  export HOME="${PROJECT_ROOT}/smoke-home-linux"
  export TILE_COMPILE_GUI2_NO_BROWSER=1
  export TILE_COMPILE_GUI2_PORT="${PORT}"
  rm -rf "${HOME}"
  mkdir -p "${HOME}"
  bash "${root}/start_gui2.sh" >/tmp/out_gui2_smoke_linux.txt 2>&1 &
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
  [[ "${SKIP_BUILD}" == "1" ]] || build_all
  assemble_bundle
  [[ "${SKIP_SMOKE}" == "1" ]] || smoke_test
  echo "[gui2-package] Created ${ARTIFACTS_DIR}/tile_compile_gui2-linux-${TAG}.zip"
}

main
