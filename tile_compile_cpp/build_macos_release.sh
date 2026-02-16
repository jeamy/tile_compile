#!/bin/bash
# tile_compile_cpp - macOS Release Build
# Prüft Abhängigkeiten, baut Release und erstellt eine portable Dist.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"
BUILD_DIR="$PROJECT_DIR/build-macos-release"
DIST_DIR="$PROJECT_DIR/dist/macos"
BUILD_TYPE="Release"
APP_BUNDLE="$DIST_DIR/tile_compile_gui.app"
MACOS_MAJOR="$(sw_vers -productVersion 2>/dev/null | cut -d. -f1)"
if ! [[ "$MACOS_MAJOR" =~ ^[0-9]+$ ]]; then
  MACOS_MAJOR=0
fi

echo "=== tile_compile_cpp - macOS Release Build ==="
echo ""

#==============================================================================
# [0] Abhängigkeiten prüfen
#==============================================================================
MISSING_DEPS=""
LIB_CHECK_WARNINGS=""

if ! command -v clang++ &>/dev/null; then
  MISSING_DEPS="$MISSING_DEPS xcode-cli"
fi

if ! command -v cmake &>/dev/null; then
  MISSING_DEPS="$MISSING_DEPS cmake"
fi

if ! command -v pkg-config &>/dev/null; then
  MISSING_DEPS="$MISSING_DEPS pkg-config"
else
  if ! pkg-config --exists eigen3 2>/dev/null; then
    LIB_CHECK_WARNINGS="$LIB_CHECK_WARNINGS eigen"
  fi
  if ! pkg-config --exists opencv4 2>/dev/null; then
    LIB_CHECK_WARNINGS="$LIB_CHECK_WARNINGS opencv"
  fi
  if ! pkg-config --exists cfitsio 2>/dev/null; then
    LIB_CHECK_WARNINGS="$LIB_CHECK_WARNINGS cfitsio"
  fi
  if ! pkg-config --exists yaml-cpp 2>/dev/null; then
    LIB_CHECK_WARNINGS="$LIB_CHECK_WARNINGS yaml-cpp"
  fi
  if ! pkg-config --exists nlohmann_json 2>/dev/null; then
    LIB_CHECK_WARNINGS="$LIB_CHECK_WARNINGS nlohmann-json"
  fi
  if ! pkg-config --exists openssl 2>/dev/null; then
    LIB_CHECK_WARNINGS="$LIB_CHECK_WARNINGS openssl"
  fi
fi

QT6_OK=0
if command -v qmake6 &>/dev/null; then
  QT6_OK=1
elif command -v qtpaths6 &>/dev/null; then
  QT6_OK=1
elif [ -n "$CMAKE_PREFIX_PATH" ] && [ -d "$CMAKE_PREFIX_PATH/lib/cmake/Qt6" ]; then
  QT6_OK=1
else
  for cand in "$HOME"/Qt/6.*/macos; do
    if [ -d "$cand/lib/cmake/Qt6" ]; then
      export CMAKE_PREFIX_PATH="$cand"
      QT6_OK=1
      break
    fi
  done
fi
if [ "$QT6_OK" -eq 0 ]; then
  MISSING_DEPS="$MISSING_DEPS qt6"
fi

if [ -n "$MISSING_DEPS" ]; then
  echo "Fehlende Abhängigkeiten:$MISSING_DEPS"
  echo ""
  if [[ " $MISSING_DEPS " == *" xcode-cli "* ]]; then
    echo "Xcode Command Line Tools fehlen. Starte: xcode-select --install"
    xcode-select --install || true
    echo "Bitte Installation abschliessen und Skript erneut starten."
    exit 1
  fi

  if command -v port &>/dev/null; then
    echo "Versuche automatische Installation via MacPorts..."
    PORT_PKGS=""
    for dep in $MISSING_DEPS; do
      case "$dep" in
        cmake)      PORT_PKGS="$PORT_PKGS cmake" ;;
        pkg-config) PORT_PKGS="$PORT_PKGS pkgconfig" ;;
        qt6)        PORT_PKGS="$PORT_PKGS qt6-qtbase qt6-qttools" ;;
      esac
    done

    if [ -n "$PORT_PKGS" ]; then
      echo "sudo port install$PORT_PKGS"
      sudo port install $PORT_PKGS || {
        echo "FEHLER: MacPorts-Installation fehlgeschlagen."
        exit 1
      }
    fi

    export CMAKE_PREFIX_PATH="/opt/local${CMAKE_PREFIX_PATH:+;$CMAKE_PREFIX_PATH}"
    export PKG_CONFIG_PATH="/opt/local/lib/pkgconfig:/opt/local/share/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
    export PATH="/opt/local/bin:/opt/local/sbin:$PATH"

    if [ -z "$OpenCV_DIR" ]; then
      if [ -f "/opt/local/lib/cmake/opencv4/OpenCVConfig.cmake" ]; then
        export OpenCV_DIR="/opt/local/lib/cmake/opencv4"
      elif [ -f "/opt/local/libexec/opencv4/cmake/OpenCVConfig.cmake" ]; then
        export OpenCV_DIR="/opt/local/libexec/opencv4/cmake"
      elif [ -f "/opt/local/share/opencv4/OpenCVConfig.cmake" ]; then
        export OpenCV_DIR="/opt/local/share/opencv4"
      fi
    fi
  elif command -v brew &>/dev/null && [ "$MACOS_MAJOR" -ge 13 ]; then
    echo "Versuche automatische Installation via Homebrew..."
    BREW_PKGS=""
    for dep in $MISSING_DEPS; do
      case "$dep" in
        cmake)      BREW_PKGS="$BREW_PKGS cmake" ;;
        pkg-config) BREW_PKGS="$BREW_PKGS pkg-config" ;;
        qt6)        BREW_PKGS="$BREW_PKGS qt" ;;
      esac
    done
    if [ -n "$BREW_PKGS" ]; then
      echo "brew install$BREW_PKGS"
      brew install $BREW_PKGS || {
        echo "FEHLER: Homebrew-Installation fehlgeschlagen."
        exit 1
      }
    fi
  else
    echo "Automatische Installation nicht möglich."
    echo "macOS 12 (Monterey): bitte MacPorts installieren (https://www.macports.org/install.php)."
    echo "macOS >=13: Homebrew ist moeglich."
    exit 1
  fi
fi

if [ -z "$CMAKE_PREFIX_PATH" ] && command -v qtpaths6 &>/dev/null; then
  QT_PREFIX_DETECTED="$(qtpaths6 --install-prefix 2>/dev/null || echo "")"
  if [ -n "$QT_PREFIX_DETECTED" ]; then
    export CMAKE_PREFIX_PATH="$QT_PREFIX_DETECTED"
  fi
fi

if [ -z "$CMAKE_PREFIX_PATH" ]; then
  for cand in "$HOME"/Qt/6.*/macos; do
    if [ -d "$cand/lib/cmake/Qt6" ]; then
      export CMAKE_PREFIX_PATH="$cand"
      break
    fi
  done
fi

if [ -n "$CMAKE_PREFIX_PATH" ] && [ -d "$CMAKE_PREFIX_PATH/bin" ]; then
  export PATH="$CMAKE_PREFIX_PATH/bin:$PATH"
fi

if [ -z "$Qt6_DIR" ] && [ -n "$CMAKE_PREFIX_PATH" ] && [ -d "$CMAKE_PREFIX_PATH/lib/cmake/Qt6" ]; then
  export Qt6_DIR="$CMAKE_PREFIX_PATH/lib/cmake/Qt6"
fi

if [ -n "$LIB_CHECK_WARNINGS" ]; then
  echo "Hinweis: pkg-config konnte folgende Module nicht bestaetigen:$LIB_CHECK_WARNINGS"
  echo "Der Build wird trotzdem versucht (CMake-Fallback fuer nicht-Homebrew-Installationen)."
  if command -v port &>/dev/null; then
    PORT_LIBS=""
    for dep in $LIB_CHECK_WARNINGS; do
      case "$dep" in
        eigen)         PORT_LIBS="$PORT_LIBS eigen3" ;;
        opencv)        PORT_LIBS="$PORT_LIBS opencv4" ;;
        cfitsio)       PORT_LIBS="$PORT_LIBS cfitsio" ;;
        yaml-cpp)      PORT_LIBS="$PORT_LIBS yaml-cpp" ;;
        nlohmann-json) PORT_LIBS="$PORT_LIBS nlohmann-json" ;;
        openssl)       PORT_LIBS="$PORT_LIBS openssl" ;;
      esac
    done
    if [ -n "$PORT_LIBS" ]; then
      echo "Versuche fehlende Libraries via MacPorts zu installieren..."
      echo "sudo port install$PORT_LIBS"
      sudo port install $PORT_LIBS || {
        echo "WARNUNG: Library-Installation via MacPorts fehlgeschlagen."
      }
      export CMAKE_PREFIX_PATH="/opt/local${CMAKE_PREFIX_PATH:+;$CMAKE_PREFIX_PATH}"
      export PKG_CONFIG_PATH="/opt/local/lib/pkgconfig:/opt/local/share/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
      export PATH="/opt/local/bin:/opt/local/sbin:$PATH"

      if [ -z "$OpenCV_DIR" ]; then
        if [ -f "/opt/local/lib/cmake/opencv4/OpenCVConfig.cmake" ]; then
          export OpenCV_DIR="/opt/local/lib/cmake/opencv4"
        elif [ -f "/opt/local/libexec/opencv4/cmake/OpenCVConfig.cmake" ]; then
          export OpenCV_DIR="/opt/local/libexec/opencv4/cmake"
        elif [ -f "/opt/local/share/opencv4/OpenCVConfig.cmake" ]; then
          export OpenCV_DIR="/opt/local/share/opencv4"
        fi
      fi
    fi
  fi
  if [[ " $LIB_CHECK_WARNINGS " == *" opencv "* ]] && [ "$MACOS_MAJOR" -gt 0 ] && [ "$MACOS_MAJOR" -lt 13 ]; then
    echo "macOS < 13 erkannt: Homebrew/OpenCV kann wegen Qt/qtmultimedia scheitern."
    echo "Alternative fuer OpenCV/CFITSIO/yaml-cpp/nlohmann-json:"
    echo "  sudo port install opencv cfitsio yaml-cpp nlohmann-json openssl"
    echo "  export CMAKE_PREFIX_PATH=\"/opt/local\""
    echo "  export PKG_CONFIG_PATH=\"/opt/local/lib/pkgconfig:/opt/local/share/pkgconfig:\$PKG_CONFIG_PATH\""
  fi
  echo ""
fi

echo "Alle Abhängigkeiten vorhanden."
echo ""

#==============================================================================
# [1] CMake konfigurieren
#==============================================================================
echo "[1/3] CMake konfigurieren..."
cmake -S "$PROJECT_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE="$BUILD_TYPE" -DBUILD_TESTS=OFF

#==============================================================================
# [2] Bauen
#==============================================================================
echo ""
echo "[2/3] Bauen..."
NPROC=$(sysctl -n hw.ncpu 2>/dev/null || echo 4)
cmake --build "$BUILD_DIR" -j"$NPROC" --config "$BUILD_TYPE"

#==============================================================================
# [3] Dist-Verzeichnis erstellen
#==============================================================================
echo ""
echo "[3/3] Dist-Verzeichnis erstellen..."
rm -rf "$DIST_DIR"
mkdir -p "$DIST_DIR"

for bin in tile_compile_gui tile_compile_runner tile_compile_cli; do
  if [ -f "$BUILD_DIR/$bin" ]; then
    cp "$BUILD_DIR/$bin" "$DIST_DIR/"
  elif [ -f "$BUILD_DIR/$BUILD_TYPE/$bin" ]; then
    cp "$BUILD_DIR/$BUILD_TYPE/$bin" "$DIST_DIR/"
  else
    echo "FEHLER: Binärdatei $bin wurde nicht gefunden." >&2
    exit 1
  fi
done

mkdir -p "$DIST_DIR/gui_cpp"
cp "$PROJECT_DIR/gui_cpp/constants.js" "$DIST_DIR/gui_cpp/"
cp "$PROJECT_DIR/gui_cpp/styles.qss" "$DIST_DIR/gui_cpp/"

for f in tile_compile.yaml tile_compile.schema.yaml tile_compile.schema.json; do
  cp "$PROJECT_DIR/$f" "$DIST_DIR/"
done

# Beispiel-Konfigurationen/Schemas mitliefern
if [ -d "$PROJECT_DIR/examples" ]; then
  mkdir -p "$DIST_DIR/examples"
  cp -R "$PROJECT_DIR/examples/." "$DIST_DIR/examples/"
fi

# Externe Daten (Siril / ASTAP) werden bewusst NICHT eingebündelt.

mkdir -p "$APP_BUNDLE/Contents/MacOS"
mkdir -p "$APP_BUNDLE/Contents/Resources"
mkdir -p "$APP_BUNDLE/Contents/Frameworks"
mkdir -p "$APP_BUNDLE/Contents/PlugIns"

mv "$DIST_DIR/tile_compile_gui" "$APP_BUNDLE/Contents/MacOS/tile_compile_gui"
cp "$DIST_DIR/tile_compile_runner" "$APP_BUNDLE/Contents/MacOS/tile_compile_runner"
cp "$DIST_DIR/tile_compile_cli" "$APP_BUNDLE/Contents/MacOS/tile_compile_cli"

mkdir -p "$APP_BUNDLE/Contents/MacOS/gui_cpp"
cp "$DIST_DIR/gui_cpp/constants.js" "$APP_BUNDLE/Contents/MacOS/gui_cpp/"
cp "$DIST_DIR/gui_cpp/styles.qss" "$APP_BUNDLE/Contents/MacOS/gui_cpp/"

cp "$DIST_DIR/tile_compile.yaml" "$APP_BUNDLE/Contents/MacOS/"
cp "$DIST_DIR/tile_compile.schema.yaml" "$APP_BUNDLE/Contents/MacOS/"
cp "$DIST_DIR/tile_compile.schema.json" "$APP_BUNDLE/Contents/MacOS/"

if [ -d "$DIST_DIR/examples" ]; then
  mkdir -p "$APP_BUNDLE/Contents/MacOS/examples"
  cp -R "$DIST_DIR/examples/." "$APP_BUNDLE/Contents/MacOS/examples/"
fi

cat > "$APP_BUNDLE/Contents/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>tile_compile_gui</string>
  <key>CFBundleDisplayName</key><string>tile_compile_gui</string>
  <key>CFBundleExecutable</key><string>tile_compile_gui</string>
  <key>CFBundleIdentifier</key><string>com.tilecompile.gui</string>
  <key>CFBundleVersion</key><string>1.0</string>
  <key>CFBundleShortVersionString</key><string>1.0</string>
  <key>LSMinimumSystemVersion</key><string>10.15.0</string>
</dict>
</plist>
EOF

if command -v macdeployqt &>/dev/null; then
  echo "Bündele Qt-Frameworks mit macdeployqt..."
  macdeployqt "$APP_BUNDLE" -executable="$APP_BUNDLE/Contents/MacOS/tile_compile_runner" -executable="$APP_BUNDLE/Contents/MacOS/tile_compile_cli" -verbose=1 || {
    echo "WARNUNG: macdeployqt konnte die Qt-Frameworks nicht vollständig bündeln." >&2
  }
elif command -v qtpaths6 &>/dev/null; then
  echo "Bündele Qt-Frameworks manuell (Fallback ohne macdeployqt)..."
  QT_PREFIX="$(qtpaths6 --install-prefix 2>/dev/null || true)"
  QT_PLUGIN_DIR="$(qtpaths6 --plugin-dir 2>/dev/null || true)"

  if [ -d "$QT_PREFIX/lib" ]; then
    cp -R "$QT_PREFIX/lib/Qt"*.framework "$APP_BUNDLE/Contents/Frameworks/" 2>/dev/null || true
  fi

  for sub in platforms styles imageformats iconengines; do
    if [ -d "$QT_PLUGIN_DIR/$sub" ]; then
      mkdir -p "$APP_BUNDLE/Contents/PlugIns/$sub"
      cp -R "$QT_PLUGIN_DIR/$sub"/*.dylib "$APP_BUNDLE/Contents/PlugIns/$sub"/ 2>/dev/null || true
    fi
  done

  cat > "$APP_BUNDLE/Contents/Resources/qt.conf" <<EOF
[Paths]
Prefix=..
Plugins=PlugIns
Imports=.
Qml2Imports=.
EOF

  if command -v install_name_tool &>/dev/null; then
    install_name_tool -add_rpath "@executable_path/../Frameworks" "$APP_BUNDLE/Contents/MacOS/tile_compile_gui" || true
    install_name_tool -add_rpath "@executable_path/../Frameworks" "$APP_BUNDLE/Contents/MacOS/tile_compile_runner" || true
    install_name_tool -add_rpath "@executable_path/../Frameworks" "$APP_BUNDLE/Contents/MacOS/tile_compile_cli" || true
  fi
else
  echo "Hinweis: Weder macdeployqt noch qtpaths6 gefunden – Qt-Frameworks werden nicht mitgeliefert."
fi

rm -f "$DIST_DIR/tile_compile_runner" "$DIST_DIR/tile_compile_cli" 2>/dev/null || true
rm -rf "$DIST_DIR/gui_cpp" 2>/dev/null || true
rm -rf "$DIST_DIR/examples" 2>/dev/null || true
rm -f "$DIST_DIR/tile_compile.yaml" "$DIST_DIR/tile_compile.schema.yaml" "$DIST_DIR/tile_compile.schema.json" 2>/dev/null || true

if command -v hdiutil &>/dev/null; then
  DMG_PATH="$PROJECT_DIR/dist/tile_compile_cpp-macos.dmg"
  echo "Erzeuge DMG-Image unter $DMG_PATH..."
  rm -f "$DMG_PATH"
  hdiutil create -volname "tile_compile_cpp" -srcfolder "$APP_BUNDLE" -ov -format UDZO "$DMG_PATH" || {
    echo "WARNUNG: DMG-Image konnte nicht erzeugt werden." >&2
  }
fi

echo ""
echo "========================================"
echo "  Release-Build fertig!"
echo "========================================"
echo ""
echo "Start:"
echo "  open $APP_BUNDLE"
echo ""
