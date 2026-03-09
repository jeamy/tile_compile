#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"

IMAGE_TAG="${IMAGE_TAG:-tile-compile-gui2:ubuntu20.04}"
CONTAINER_NAME="${CONTAINER_NAME:-tile-compile-gui2}"
HOST_PORT="${HOST_PORT:-8080}"
INPUT_DIR="${INPUT_DIR:-${PROJECT_ROOT}/tmp/docker-input}"
RUNS_DIR="${RUNS_DIR:-${PROJECT_ROOT}/tmp/docker-runs}"
EXTRA_ALLOWED_ROOTS="${EXTRA_ALLOWED_ROOTS:-}"

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --image-tag <tag>        Docker image tag (default: ${IMAGE_TAG})
  --name <name>            Container name (default: ${CONTAINER_NAME})
  --port <port>            Host port mapped to container 8080 (default: ${HOST_PORT})
  --input-dir <path>       Host input data directory mount (default: ${INPUT_DIR})
  --runs-dir <path>        Host runs output directory mount (default: ${RUNS_DIR})
  --extra-root <path>      Additional allowed root inside container at /data/extra
  --no-build               Skip docker build
  -h, --help               Show this help

Environment overrides:
  IMAGE_TAG, CONTAINER_NAME, HOST_PORT, INPUT_DIR, RUNS_DIR, EXTRA_ALLOWED_ROOTS
EOF
}

DO_BUILD=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --image-tag) IMAGE_TAG="$2"; shift 2 ;;
    --name) CONTAINER_NAME="$2"; shift 2 ;;
    --port) HOST_PORT="$2"; shift 2 ;;
    --input-dir) INPUT_DIR="$2"; shift 2 ;;
    --runs-dir) RUNS_DIR="$2"; shift 2 ;;
    --extra-root) EXTRA_ALLOWED_ROOTS="$2"; shift 2 ;;
    --no-build) DO_BUILD=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

mkdir -p "${INPUT_DIR}" "${RUNS_DIR}"

if [[ "${DO_BUILD}" == "1" ]]; then
  echo "[docker] Building ${IMAGE_TAG}"
  docker build -t "${IMAGE_TAG}" -f "${PROJECT_ROOT}/docker/ubuntu20.04/Dockerfile" "${PROJECT_ROOT}"
fi

if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  echo "[docker] Removing existing container ${CONTAINER_NAME}"
  docker rm -f "${CONTAINER_NAME}" >/dev/null
fi

ALLOWED_ROOTS="/opt/tile_compile:/data/input:/data/runs:/tmp"
MOUNT_EXTRA=()
if [[ -n "${EXTRA_ALLOWED_ROOTS}" ]]; then
  mkdir -p "${EXTRA_ALLOWED_ROOTS}"
  MOUNT_EXTRA=(-v "${EXTRA_ALLOWED_ROOTS}:/data/extra")
  ALLOWED_ROOTS="${ALLOWED_ROOTS}:/data/extra"
fi

echo "[docker] Starting ${CONTAINER_NAME} on http://127.0.0.1:${HOST_PORT}/ui/"
docker run -d \
  --name "${CONTAINER_NAME}" \
  -p "${HOST_PORT}:8080" \
  -v "${INPUT_DIR}:/data/input" \
  -v "${RUNS_DIR}:/data/runs" \
  "${MOUNT_EXTRA[@]}" \
  -e TILE_COMPILE_ALLOWED_ROOTS="${ALLOWED_ROOTS}" \
  -e TILE_COMPILE_RUNS_DIR="/data/runs" \
  "${IMAGE_TAG}" >/dev/null

echo "[docker] Container logs:"
docker logs --tail 30 "${CONTAINER_NAME}" || true
echo "[docker] Open: http://127.0.0.1:${HOST_PORT}/ui/"
