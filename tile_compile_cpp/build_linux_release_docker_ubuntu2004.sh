#!/bin/bash
# tile_compile_cpp - Linux Release Build (Docker / Ubuntu 20.04)
# Baut den Linux-Release in einer Ubuntu 20.04 Umgebung (glibc 2.31) für maximale Kompatibilität.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

IMAGE_NAME="tile_compile_cpp-build-ubuntu2004"
DOCKERFILE_PATH="$SCRIPT_DIR/docker/ubuntu20.04/Dockerfile"

SKIP_BUILD=0
if [ "${1:-}" = "--skip-build" ]; then
  SKIP_BUILD=1
fi

if ! command -v docker &>/dev/null; then
  echo "FEHLER: docker wurde nicht gefunden. Bitte Docker installieren." >&2
  exit 1
fi

echo "=== tile_compile_cpp - Linux Release Build (Docker / Ubuntu 20.04) ==="

if [ "$SKIP_BUILD" -eq 0 ]; then
  echo "[1/2] Docker-Image bauen..."
  docker build -t "$IMAGE_NAME" -f "$DOCKERFILE_PATH" "$SCRIPT_DIR"
else
  echo "[1/2] Docker-Image Build übersprungen (--skip-build)"
fi

echo "[2/2] Release-Build im Container ausführen..."
USER_ID="$(id -u 2>/dev/null || echo 0)"
GROUP_ID="$(id -g 2>/dev/null || echo 0)"

docker run --rm \
  -u "$USER_ID:$GROUP_ID" \
  -v "$SCRIPT_DIR:/work" \
  -w /work \
  "$IMAGE_NAME" \
  bash -lc "rm -rf build-linux-release && bash build_linux_release.sh" || {
  echo ""
  echo "Falls der Build wegen fehlender Abhängigkeiten fehlschlägt, versuche:"
  echo "  docker run --rm -v '$SCRIPT_DIR:/work' -w /work '$IMAGE_NAME' bash -lc 'apt-get update && apt-get install -y nlohmann-json3-dev && bash build_linux_release.sh'"
  exit 1
}

echo ""
echo "Fertig. Output unter:"
echo "  $SCRIPT_DIR/dist/"
