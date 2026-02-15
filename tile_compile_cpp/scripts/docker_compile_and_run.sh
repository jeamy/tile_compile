#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TC_CPP_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${TC_CPP_DIR}/.." && pwd)"

IMAGE_NAME="tile_compile_cpp:dev"
CONTAINER_NAME="tile_compile_cpp_dev"
RUNS_HOST_DIR="${TC_CPP_DIR}/runs"
RUNS_CONTAINER_DIR="/workspace/tile_compile_cpp/runs"

usage() {
  cat <<EOF
Usage:
  $(basename "$0") build-image [--image <name>]
  $(basename "$0") run-shell   [--image <name>] [--name <container>] [--runs-host-dir <path>] [--runs-container-dir <path>]
  $(basename "$0") run-app     [--image <name>] [--runs-host-dir <path>] [--runs-container-dir <path>] -- <runner args>
  $(basename "$0") run-gui     [--image <name>] [--runs-host-dir <path>] [--runs-container-dir <path>] [-- <gui args>]

Description:
  build-image  Builds a Docker image and compiles tile_compile_cpp inside the image.
  run-shell    Starts an interactive container shell with default runs volume mounted.
  run-app      Runs ./tile_compile_runner inside the container.
  run-gui      Runs ./tile_compile_gui inside the container with X11 forwarding.

Defaults:
  image name:         ${IMAGE_NAME}
  container name:     ${CONTAINER_NAME}
  runs host dir:      ${RUNS_HOST_DIR}
  runs container dir: ${RUNS_CONTAINER_DIR}

Examples:
  $(basename "$0") build-image
  $(basename "$0") run-shell
  $(basename "$0") run-app -- run --config /mnt/config/tile_compile.yaml --input-dir /mnt/input --runs-dir ${RUNS_CONTAINER_DIR}
  $(basename "$0") run-gui

Notes:
  - For run-app with host files, mount them explicitly using docker options by extending this script
    or run run-shell and execute commands manually.
  - The default runs volume maps host '${RUNS_HOST_DIR}' to container '${RUNS_CONTAINER_DIR}'.
  - For run-gui, allow local Docker access once on the host: xhost +local:docker
EOF
}

parse_common_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --image)
        IMAGE_NAME="$2"
        shift 2
        ;;
      --name)
        CONTAINER_NAME="$2"
        shift 2
        ;;
      --runs-host-dir)
        RUNS_HOST_DIR="$2"
        shift 2
        ;;
      --runs-container-dir)
        RUNS_CONTAINER_DIR="$2"
        shift 2
        ;;
      --)
        shift
        break
        ;;
      *)
        break
        ;;
    esac
  done

  REM_ARGS=("$@")
}

build_image() {
  mkdir -p "${RUNS_HOST_DIR}"

  docker build \
    --tag "${IMAGE_NAME}" \
    --file - \
    "${REPO_ROOT}" <<'DOCKERFILE'
FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    pkg-config \
    ca-certificates \
    git \
    libgl1-mesa-dev \
    libopengl-dev \
    libopencv-dev \
    libeigen3-dev \
    libcfitsio-dev \
    libyaml-cpp-dev \
    nlohmann-json3-dev \
    libssl-dev \
    qt6-base-dev \
    libspdlog-dev \
    python3 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
RUN mkdir -p /workspace/tile_compile_cpp

# Copy only files needed to build runner/cli/gui (avoid copying runs/, build/, caches, etc.)
COPY tile_compile_cpp/CMakeLists.txt /workspace/tile_compile_cpp/CMakeLists.txt
COPY tile_compile_cpp/tile_compile.yaml /workspace/tile_compile_cpp/
COPY tile_compile_cpp/tile_compile.schema.yaml /workspace/tile_compile_cpp/
COPY tile_compile_cpp/tile_compile.schema.json /workspace/tile_compile_cpp/
COPY tile_compile_cpp/include /workspace/tile_compile_cpp/include
COPY tile_compile_cpp/src /workspace/tile_compile_cpp/src
COPY tile_compile_cpp/apps /workspace/tile_compile_cpp/apps
COPY tile_compile_cpp/gui_cpp /workspace/tile_compile_cpp/gui_cpp

RUN rm -rf /workspace/tile_compile_cpp/build \
 && cmake -S /workspace/tile_compile_cpp -B /workspace/tile_compile_cpp/build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=OFF \
 && cmake --build /workspace/tile_compile_cpp/build -j"$(nproc)"

WORKDIR /workspace/tile_compile_cpp/build
CMD ["bash"]
DOCKERFILE
}

run_shell() {
  mkdir -p "${RUNS_HOST_DIR}"

  docker run --rm -it \
    --name "${CONTAINER_NAME}" \
    -v "${RUNS_HOST_DIR}:${RUNS_CONTAINER_DIR}" \
    "${IMAGE_NAME}" \
    bash
}

run_app() {
  mkdir -p "${RUNS_HOST_DIR}"

  if [[ ${#REM_ARGS[@]} -eq 0 ]]; then
    echo "ERROR: run-app requires runner arguments after '--'" >&2
    echo "Example: $(basename "$0") run-app -- run --config /path/config.yaml --input-dir /path/in --runs-dir ${RUNS_CONTAINER_DIR}" >&2
    exit 2
  fi

  docker run --rm -it \
    -v "${RUNS_HOST_DIR}:${RUNS_CONTAINER_DIR}" \
    -w /workspace/tile_compile_cpp/build \
    "${IMAGE_NAME}" \
    ./tile_compile_runner "${REM_ARGS[@]}"
}

run_gui() {
  mkdir -p "${RUNS_HOST_DIR}"

  if [[ -z "${DISPLAY:-}" ]]; then
    echo "ERROR: DISPLAY is not set. Start from a graphical session." >&2
    exit 2
  fi

  docker run --rm -it \
    -e DISPLAY="${DISPLAY}" \
    -e QT_QPA_PLATFORM=xcb \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v "${RUNS_HOST_DIR}:${RUNS_CONTAINER_DIR}" \
    -w /workspace/tile_compile_cpp/build \
    "${IMAGE_NAME}" \
    ./tile_compile_gui "${REM_ARGS[@]}"
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

COMMAND="$1"
shift

parse_common_args "$@"

case "${COMMAND}" in
  build-image)
    build_image
    ;;
  run-shell)
    run_shell
    ;;
  run-app)
    run_app
    ;;
  run-gui)
    run_gui
    ;;
  --help|-h|help)
    usage
    ;;
  *)
    echo "ERROR: unknown command '${COMMAND}'" >&2
    usage
    exit 2
    ;;
esac
