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

Run this script from an MSYS2 MinGW64 shell on Windows.

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
ROOT="${BUNDLE_DIR}/tile_compile_gui2-windows-${TAG}"
PAYLOAD="${ROOT}/payload"
DIST_BIN="${PAYLOAD}/tile_compile_cpp/build"
BACKEND_BIN_DIR="${PAYLOAD}/web_backend_cpp/build"

build_all() {
  cmake -S "${PROJECT_ROOT}/tile_compile_cpp" -B "${RUNNER_BUILD_DIR}" -G Ninja \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -DBUILD_TESTS=OFF
  cmake --build "${RUNNER_BUILD_DIR}" -j"$(nproc)"

  cmake -S "${PROJECT_ROOT}/web_backend_cpp" -B "${BACKEND_BUILD_DIR}" -G Ninja \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -DTILE_COMPILE_BACKEND_STATIC_STDLIB=OFF
  cmake --build "${BACKEND_BUILD_DIR}" -j"$(nproc)"
}

copy_deps() {
  local exe="$1"
  ntldd -R "$exe" 2>/dev/null \
    | awk '{
        for (i = 1; i <= NF; ++i) {
          path = $i
          gsub(/\r/, "", path)
          if (tolower(path) ~ /\.dll$/ && (path ~ /^[A-Za-z]:/ || path ~ /^\//)) print path;
        }
      }' \
    | sort -u \
    | while read -r dep; do
        [[ -f "$dep" ]] || continue
        cp -n "$dep" "${PAYLOAD}/tile_compile_cpp/lib/" || true
      done
}

assemble_bundle() {
  rm -rf "${ROOT}"
  mkdir -p "${ARTIFACTS_DIR}" "${DIST_BIN}" "${PAYLOAD}/tile_compile_cpp/lib" "${BACKEND_BIN_DIR}"
  cp "${PROJECT_ROOT}/packaging/gui2/start_gui2.bat" "${ROOT}/start_gui2.bat"
  cp "${PROJECT_ROOT}/packaging/gui2/start_gui2.ps1" "${ROOT}/start_gui2.ps1"
  cp -a "${PROJECT_ROOT}/web_frontend" "${PAYLOAD}/"
  cp -a "${PROJECT_ROOT}/tile_compile_cpp/examples" "${PAYLOAD}/tile_compile_cpp/"
  cp "${PROJECT_ROOT}/tile_compile_cpp/tile_compile.yaml" "${PAYLOAD}/tile_compile_cpp/"
  cp "${PROJECT_ROOT}/tile_compile_cpp/tile_compile.schema.yaml" "${PAYLOAD}/tile_compile_cpp/"
  cp "${PROJECT_ROOT}/tile_compile_cpp/tile_compile.schema.json" "${PAYLOAD}/tile_compile_cpp/"
  cp "${RUNNER_BUILD_DIR}/tile_compile_runner.exe" "${DIST_BIN}/"
  cp "${RUNNER_BUILD_DIR}/tile_compile_cli.exe" "${DIST_BIN}/"
  cp "${BACKEND_BUILD_DIR}/tile_compile_web_backend.exe" "${BACKEND_BIN_DIR}/"
  copy_deps "${DIST_BIN}/tile_compile_runner.exe"
  copy_deps "${DIST_BIN}/tile_compile_cli.exe"
  copy_deps "${BACKEND_BIN_DIR}/tile_compile_web_backend.exe"
  printf '%s\n' "${TAG}" > "${PAYLOAD}/.gui2-release-tag"
}

smoke_test() {
  local root_win
  local home_win
  root_win="$(cygpath -w "${ROOT}")"
  home_win="$(cygpath -w "${PROJECT_ROOT}/smoke-home-windows")"
  mkdir -p "${PROJECT_ROOT}/smoke-home-windows"
  powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "
    \$ErrorActionPreference = 'Stop'
    \$Tag = '${TAG}'
    \$root = '${root_win}'
    \$homeDir = '${home_win}'
    New-Item -ItemType Directory -Path \$homeDir -Force | Out-Null
    \$env:USERPROFILE = \$homeDir
    \$env:TILE_COMPILE_GUI2_NO_BROWSER = '1'
    \$env:TILE_COMPILE_GUI2_PORT = '${PORT}'
    \$startScript = Join-Path \$root 'start_gui2.ps1'
    \$launcher = Start-Process -FilePath 'powershell' -ArgumentList @('-ExecutionPolicy','Bypass','-File',\$startScript) -PassThru
    \$deadline = (Get-Date).AddSeconds(45)
    \$ok = \$false
    while ((Get-Date) -lt \$deadline) {
      try {
        \$resp = Invoke-WebRequest -Uri 'http://127.0.0.1:${PORT}/api/app/state' -UseBasicParsing -TimeoutSec 2
        if (\$resp.StatusCode -lt 500) {
          \$ok = \$true
          break
        }
      } catch {}
      Start-Sleep -Seconds 1
    }
    if (-not \$ok) {
      throw 'backend smoke test failed'
    }
    if (\$launcher -and -not \$launcher.HasExited) {
      cmd /c \"taskkill /PID \$((\$launcher).Id) /T /F\" | Out-Null
    }
  "
}

create_zip() {
  local src_win
  local dst_win
  src_win="$(cygpath -w "${ROOT}")"
  dst_win="$(cygpath -w "${ARTIFACTS_DIR}/tile_compile_gui2-windows-${TAG}.zip")"
  powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "
    \$ErrorActionPreference = 'Stop'
    \$src = '${src_win}'
    \$dst = '${dst_win}'
    if (Test-Path \$dst) { Remove-Item -Force \$dst }
    Compress-Archive -Path \$src -DestinationPath \$dst -Force
  "
}

main() {
  [[ "${SKIP_BUILD}" == "1" ]] || build_all
  assemble_bundle
  [[ "${SKIP_SMOKE}" == "1" ]] || smoke_test
  create_zip
  echo "[gui2-package] Created ${ARTIFACTS_DIR}/tile_compile_gui2-windows-${TAG}.zip"
}

main
