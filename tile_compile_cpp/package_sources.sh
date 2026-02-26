#!/bin/bash
# Script to package all relevant source files for the tile_compile app into a tar.gz archive.
# Includes C++ sources, headers, build system, config, and documentation files.
# Excludes build artifacts, caches, and temporary directories.

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR" && pwd)"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
ARCHIVE_NAME="tile_compile_sources_${TIMESTAMP}.tar.gz"
ARCHIVE_PATH="$PROJECT_ROOT/$ARCHIVE_NAME"

echo "Packaging tile_compile sources from: $PROJECT_ROOT"
echo "Archive will be created at: $ARCHIVE_PATH"

# Create a temporary directory to stage files
STAGE_DIR=$(mktemp -d)
trap 'rm -rf "$STAGE_DIR"' EXIT

# Copy project structure excluding unwanted directories
echo "Staging files..."
rsync -av --progress \
    --exclude='build/' \
    --exclude='build*/' \
    --exclude='*.o' \
    --exclude='*.a' \
    --exclude='*.so' \
    --exclude='*.dylib' \
    --exclude='*.dll' \
    --exclude='*.exe' \
    --exclude='*.app' \
    --exclude='*.dmg' \
    --exclude='*.deb' \
    --exclude='*.rpm' \
    --exclude='.cache/' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='*.pyo' \
    --exclude='.git/' \
    --exclude='.gitignore' \
    --exclude='.DS_Store' \
    --exclude='Thumbs.db' \
    --exclude='*.log' \
    --exclude='texput.log' \
    --exclude='runs/' \
    --exclude='Testing/' \
    --exclude='tests/' \
    --exclude='*.tmp' \
    --exclude='*.temp' \
    --exclude='*.bak' \
    --exclude='*~' \
    --exclude='.vscode/' \
    --exclude='.idea/' \
    --exclude='cmake-build-*/' \
    --exclude='CMakeFiles/' \
    --exclude='CMakeCache.txt' \
    --exclude='cmake_install.cmake' \
    --exclude='Makefile' \
    --exclude='compile_commands.json' \
    --exclude='.clang-format' \
    --exclude='.clang-tidy' \
    --include='*/' \
    "$PROJECT_ROOT/" "$STAGE_DIR/tile_compile/"

# Create the archive
echo "Creating archive..."
cd "$STAGE_DIR"
tar -czf "$ARCHIVE_PATH" tile_compile/

# Verify archive was created
if [[ -f "$ARCHIVE_PATH" ]]; then
    ARCHIVE_SIZE=$(du -h "$ARCHIVE_PATH" | cut -f1)
    echo "✓ Archive created successfully: $ARCHIVE_NAME (size: $ARCHIVE_SIZE)"
    echo ""
    echo "Archive contents preview:"
    tar -tzf "$ARCHIVE_PATH" | head -20
    echo "..."
    echo "Total files in archive: $(tar -tzf "$ARCHIVE_PATH" | wc -l)"
else
    echo "✗ Failed to create archive!"
    exit 1
fi

echo ""
echo "To extract the archive:"
echo "  tar -xzf $ARCHIVE_NAME"
echo ""
echo "Done."
