#!/bin/bash
# tile_compile_cpp - Linux AppImage Build
# Builds a portable AppImage from the Linux release bundle.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"
DIST_ROOT="$PROJECT_DIR/dist"
RELEASE_DIR="$DIST_ROOT/linux"
APPDIR="$DIST_ROOT/AppDir"
APPIMAGE_NAME="tile_compile_cpp-x86_64.AppImage"
APPIMAGE_PATH="$DIST_ROOT/$APPIMAGE_NAME"
TOOLS_DIR="$PROJECT_DIR/.tools"
APPIMAGETOOL_BIN="$TOOLS_DIR/appimagetool"

echo "=== tile_compile_cpp - Linux AppImage Build ==="
echo ""

if ! command -v cmake >/dev/null 2>&1; then
  echo "FEHLER: cmake nicht gefunden." >&2
  exit 1
fi

if ! command -v file >/dev/null 2>&1; then
  echo "FEHLER: file nicht gefunden (fuer AppImage-Tool-Check)." >&2
  exit 1
fi

echo "[1/4] Erzeuge/aktualisiere Linux Release-Bundle..."
SKIP_DEPS="${SKIP_DEPS:-0}" bash "$PROJECT_DIR/build_linux_release.sh"

if [ ! -d "$RELEASE_DIR" ]; then
  echo "FEHLER: Linux release directory fehlt: $RELEASE_DIR" >&2
  exit 1
fi

echo "[2/4] Erzeuge AppDir..."
rm -rf "$APPDIR"
mkdir -p "$APPDIR/usr/bin" \
         "$APPDIR/usr/lib" \
         "$APPDIR/usr/plugins" \
         "$APPDIR/usr/share/applications" \
         "$APPDIR/usr/share/icons/hicolor/scalable/apps"

for exe in tile_compile_gui tile_compile_runner tile_compile_cli; do
  if [ ! -f "$RELEASE_DIR/$exe" ]; then
    echo "FEHLER: Executable fehlt im Release-Bundle: $RELEASE_DIR/$exe" >&2
    exit 1
  fi
  cp -f "$RELEASE_DIR/$exe" "$APPDIR/usr/bin/"
done

if [ -d "$RELEASE_DIR/lib" ]; then
  cp -a "$RELEASE_DIR/lib/." "$APPDIR/usr/lib/"
fi

if [ -d "$RELEASE_DIR/plugins" ]; then
  cp -a "$RELEASE_DIR/plugins/." "$APPDIR/usr/plugins/"
fi

mkdir -p "$APPDIR/usr/bin/gui_cpp"
cp -f "$PROJECT_DIR/gui_cpp/constants.js" "$APPDIR/usr/bin/gui_cpp/"
cp -f "$PROJECT_DIR/gui_cpp/styles.qss" "$APPDIR/usr/bin/gui_cpp/"
for f in tile_compile.yaml tile_compile.schema.yaml tile_compile.schema.json; do
  cp -f "$PROJECT_DIR/$f" "$APPDIR/usr/bin/"
done
if [ -d "$PROJECT_DIR/examples" ]; then
  mkdir -p "$APPDIR/usr/bin/examples"
  cp -a "$PROJECT_DIR/examples/." "$APPDIR/usr/bin/examples/"
fi

cat > "$APPDIR/usr/bin/qt.conf" <<'EOF'
[Paths]
Prefix=..
Plugins=../plugins
Imports=../imports
Qml2Imports=../qml
EOF

cat > "$APPDIR/AppRun" <<'EOF'
#!/bin/bash
set -e
APPDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LD_LIBRARY_PATH="$APPDIR/usr/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export QT_PLUGIN_PATH="$APPDIR/usr/plugins${QT_PLUGIN_PATH:+:$QT_PLUGIN_PATH}"
export QT_QPA_PLATFORM_PLUGIN_PATH="$APPDIR/usr/plugins/platforms"
exec "$APPDIR/usr/bin/tile_compile_gui" "$@"
EOF
chmod +x "$APPDIR/AppRun"

cat > "$APPDIR/usr/share/applications/tile_compile_gui.desktop" <<'EOF'
[Desktop Entry]
Type=Application
Name=tile_compile_gui
Comment=Tile Compile GUI
Exec=tile_compile_gui
Icon=tile_compile_gui
Categories=Science;Graphics;
Terminal=false
EOF

cat > "$APPDIR/usr/share/icons/hicolor/scalable/apps/tile_compile_gui.svg" <<'EOF'
<svg xmlns="http://www.w3.org/2000/svg" width="256" height="256" viewBox="0 0 256 256">
  <rect width="256" height="256" rx="28" fill="#1f2937"/>
  <rect x="36" y="36" width="84" height="84" rx="10" fill="#0ea5e9"/>
  <rect x="136" y="36" width="84" height="84" rx="10" fill="#22c55e"/>
  <rect x="36" y="136" width="84" height="84" rx="10" fill="#f59e0b"/>
  <rect x="136" y="136" width="84" height="84" rx="10" fill="#ef4444"/>
</svg>
EOF
cp -f "$APPDIR/usr/share/icons/hicolor/scalable/apps/tile_compile_gui.svg" \
      "$APPDIR/tile_compile_gui.svg"
cp -f "$APPDIR/usr/share/applications/tile_compile_gui.desktop" \
      "$APPDIR/tile_compile_gui.desktop"

echo "[3/4] Bereite appimagetool vor..."
mkdir -p "$TOOLS_DIR"
if command -v appimagetool >/dev/null 2>&1; then
  APPIMAGETOOL_BIN="$(command -v appimagetool)"
else
  if [ ! -x "$APPIMAGETOOL_BIN" ]; then
    APPIMAGETOOL_APPIMAGE="$TOOLS_DIR/appimagetool-x86_64.AppImage"
    if ! command -v wget >/dev/null 2>&1; then
      echo "FEHLER: wget fehlt zum Download von appimagetool." >&2
      exit 1
    fi
    echo "Lade appimagetool herunter..."
    wget -q -O "$APPIMAGETOOL_APPIMAGE" \
      "https://github.com/AppImage/AppImageKit/releases/download/continuous/appimagetool-x86_64.AppImage"
    chmod +x "$APPIMAGETOOL_APPIMAGE"
    cat > "$APPIMAGETOOL_BIN" <<EOF
#!/bin/bash
exec "$APPIMAGETOOL_APPIMAGE" "\$@"
EOF
    chmod +x "$APPIMAGETOOL_BIN"
  fi
fi

echo "[4/4] Erzeuge AppImage..."
rm -f "$APPIMAGE_PATH"
APPIMAGE_EXTRACT_AND_RUN=1 ARCH=x86_64 "$APPIMAGETOOL_BIN" "$APPDIR" "$APPIMAGE_PATH"

if [ ! -f "$APPIMAGE_PATH" ]; then
  echo "FEHLER: AppImage wurde nicht erzeugt." >&2
  exit 1
fi

echo ""
echo "========================================"
echo "  AppImage-Build fertig!"
echo "========================================"
echo ""
echo "Output:"
echo "  $APPIMAGE_PATH"
