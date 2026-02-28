#!/bin/bash
# tile_compile_cpp - Linux AppImage Build (Docker / Ubuntu 20.04)
# Erstellt ein portable AppImage in Ubuntu 20.04 für maximale Kompatibilität

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="tile_compile_cpp-build-ubuntu2004"
DOCKERFILE_PATH="$SCRIPT_DIR/docker/ubuntu20.04/Dockerfile"
SKIP_BUILD=0

if [ "${1:-}" = "--skip-build" ]; then
    SKIP_BUILD=1
fi

echo "=== tile_compile_cpp - Linux AppImage Build (Docker) ==="
echo ""

if ! command -v docker &>/dev/null; then
    echo "FEHLER: docker wurde nicht gefunden"
    exit 1
fi

if [ ! -f "$SCRIPT_DIR/build_linux_appimage.sh" ]; then
    echo "FEHLER: build_linux_appimage.sh fehlt unter $SCRIPT_DIR" >&2
    exit 1
fi

#============================================================================
# [1] Docker-Image bauen
#============================================================================
if [ "$SKIP_BUILD" -eq 0 ]; then
    echo "[1/2] Docker-Image bauen..."
    docker build -t "$IMAGE_NAME" -f "$DOCKERFILE_PATH" "$SCRIPT_DIR"
else
    echo "[1/2] Docker-Image Build übersprungen (--skip-build)"
fi

#============================================================================
# [2] AppImage Build im Container
#============================================================================
echo "[2/2] AppImage Build im Container..."

docker run --rm \
  -v "$SCRIPT_DIR:/work" \
  -w /work \
  "$IMAGE_NAME" \
  bash -lc "
    echo 'Installiere AppImage-Tools...'
    apt-get update > /dev/null 2>&1
    apt-get install -y wget file > /dev/null 2>&1
    
    echo 'Starte AppImage Build...'
    export QT6_DIR=/usr/lib/x86_64-linux-gnu/cmake/Qt6
    export PATH=/usr/lib/x86_64-linux-gnu/qt6/bin:/usr/bin:\$PATH
    export QT_SELECT=qt6
    export SKIP_DEPS=1
    
    bash build_linux_appimage.sh
  " || {
  echo ""
  echo "Der AppImage-Build im Container ist fehlgeschlagen."
  exit 1
}

echo ""
if [ ! -f "$SCRIPT_DIR/dist/tile_compile_cpp-x86_64.AppImage" ]; then
    echo "FEHLER: AppImage wurde nicht erzeugt." >&2
    exit 1
fi

echo "Fertig. AppImage unter:"
echo "  $SCRIPT_DIR/dist/tile_compile_cpp-x86_64.AppImage"
