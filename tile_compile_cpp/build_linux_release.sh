#!/bin/bash
# tile_compile_cpp - Linux Release Build
# Prüft Abhängigkeiten, baut Release und erstellt eine portable Dist.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"
BUILD_DIR="$PROJECT_DIR/build-linux-release"
DIST_DIR="$PROJECT_DIR/dist/linux"
BUILD_TYPE="Release"
ZIP_NAME="tile_compile_cpp-linux-release.zip"

echo "=== tile_compile_cpp - Linux Release Build ==="
echo ""

#==============================================================================
# [0] Abhängigkeiten prüfen
#==============================================================================
if [ "${SKIP_DEPS:-0}" = "1" ]; then
  echo "Ueberspringe Abhaengigkeiten-Check (SKIP_DEPS=1)"
  echo ""
else
MISSING_DEPS=""

if ! command -v cmake &>/dev/null; then
  MISSING_DEPS="$MISSING_DEPS cmake"
fi

if ! command -v g++ &>/dev/null && ! command -v clang++ &>/dev/null; then
  MISSING_DEPS="$MISSING_DEPS g++"
fi

if ! command -v pkg-config &>/dev/null; then
  MISSING_DEPS="$MISSING_DEPS pkg-config"
else
  if ! pkg-config --exists eigen3 2>/dev/null; then
    MISSING_DEPS="$MISSING_DEPS libeigen3-dev"
  fi
  if ! pkg-config --exists opencv4 2>/dev/null; then
    MISSING_DEPS="$MISSING_DEPS libopencv-dev"
  fi
  if ! pkg-config --exists cfitsio 2>/dev/null; then
    MISSING_DEPS="$MISSING_DEPS libcfitsio-dev"
  fi
  if ! pkg-config --exists yaml-cpp 2>/dev/null; then
    MISSING_DEPS="$MISSING_DEPS libyaml-cpp-dev"
  fi
  if ! pkg-config --exists nlohmann_json 2>/dev/null; then
    MISSING_DEPS="$MISSING_DEPS nlohmann-json3-dev"
  fi
  if ! pkg-config --exists openssl 2>/dev/null; then
    MISSING_DEPS="$MISSING_DEPS libssl-dev"
  fi
fi

QT6_OK=0
if command -v qmake6 &>/dev/null; then
  QT6_OK=1
elif pkg-config --exists Qt6Core 2>/dev/null; then
  QT6_OK=1
fi
if [ "$QT6_OK" -eq 0 ]; then
  MISSING_DEPS="$MISSING_DEPS qt6-base-dev"
fi

if [ -n "$MISSING_DEPS" ]; then
  echo "Fehlende Abhängigkeiten:$MISSING_DEPS"
  echo ""
  if command -v apt-get &>/dev/null; then
    echo "Versuche automatische Installation via apt..."
    APT_PKGS=""
    for dep in $MISSING_DEPS; do
      case "$dep" in
        cmake)        APT_PKGS="$APT_PKGS cmake" ;;
        g++)          APT_PKGS="$APT_PKGS g++" ;;
        pkg-config)   APT_PKGS="$APT_PKGS pkg-config" ;;
        qt6-base-dev) APT_PKGS="$APT_PKGS qt6-base-dev qt6-tools-dev libgl1-mesa-dev" ;;
        libeigen3-dev) APT_PKGS="$APT_PKGS libeigen3-dev" ;;
        libopencv-dev) APT_PKGS="$APT_PKGS libopencv-dev" ;;
        libcfitsio-dev) APT_PKGS="$APT_PKGS libcfitsio-dev" ;;
        libyaml-cpp-dev) APT_PKGS="$APT_PKGS libyaml-cpp-dev" ;;
        nlohmann-json3-dev) APT_PKGS="$APT_PKGS nlohmann-json3-dev" ;;
        libssl-dev) APT_PKGS="$APT_PKGS libssl-dev" ;;
      esac
    done

    # Prüfe ob wir root sind (z.B. im Docker-Container)
    if [ "$(id -u)" -eq 0 ]; then
      # Root: sudo weglassen
      echo "apt-get update && apt-get install -y$APT_PKGS"
      apt-get update && apt-get install -y $APT_PKGS || {
        echo ""
        echo "FEHLER: Automatische Installation fehlgeschlagen."
        echo "Bitte installiere manuell:"
        echo "  apt-get install$APT_PKGS"
        exit 1
      }
    else
      # Nicht root: sudo verwenden
      echo "sudo apt-get update && sudo apt-get install -y$APT_PKGS"
      sudo apt-get update && sudo apt-get install -y $APT_PKGS || {
        echo ""
        echo "FEHLER: Automatische Installation fehlgeschlagen."
        echo "Bitte installiere manuell:"
        echo "  sudo apt-get install$APT_PKGS"
        exit 1
      }
    fi
  else
    echo "Automatische Installation nicht möglich (kein apt gefunden)."
    echo "Bitte installiere manuell: cmake, g++/clang++, pkg-config, Qt6 und C++ Libs"
    echo "Benötigte Libs: eigen3, opencv4, cfitsio, yaml-cpp, nlohmann-json, openssl"
    echo "Fedora/RHEL:  sudo dnf install cmake gcc-c++ qt6-qtbase-devel"
    echo "Arch Linux:   sudo pacman -S cmake gcc qt6-base"
    echo "openSUSE:     sudo zypper install cmake gcc-c++ qt6-base-devel"
    exit 1
  fi
fi
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
NPROC=$(nproc 2>/dev/null || echo 4)
cmake --build "$BUILD_DIR" -j"$NPROC"

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

# Projektdateien, die zur Laufzeit benötigt werden
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

# Qt + Non-Qt-Libs für portable App bündeln
if command -v linuxdeployqt &>/dev/null; then
  echo "Bündele Libraries mit linuxdeployqt..."
  DESKTOP_FILE="$DIST_DIR/tile_compile_gui.desktop"
  cat > "$DESKTOP_FILE" <<EOF
[Desktop Entry]
Type=Application
Name=tile_compile_gui
Exec=tile_compile_gui
Categories=Science;
EOF
  linuxdeployqt "$DESKTOP_FILE" \
    -bundle-non-qt-libs \
    -executable="$DIST_DIR/tile_compile_gui" \
    -executable="$DIST_DIR/tile_compile_runner" \
    -executable="$DIST_DIR/tile_compile_cli" || {
      echo "WARNUNG: linuxdeployqt konnte nicht alle Libraries bündeln." >&2
    }
elif (command -v qtpaths6 &>/dev/null || command -v qtpaths &>/dev/null) && command -v patchelf &>/dev/null; then
  echo "Bündele Libraries manuell (Fallback ohne linuxdeployqt)..."
  QT_PATHS_CMD="qtpaths6"
  if ! command -v qtpaths6 &>/dev/null; then
    QT_PATHS_CMD="qtpaths"
  fi

  QT_PLUGIN_DIR="$($QT_PATHS_CMD --plugin-dir 2>/dev/null || true)"
  QT_LIB_DIR="$($QT_PATHS_CMD --library-dir 2>/dev/null || true)"

  if [ -z "$QT_PLUGIN_DIR" ] && command -v qmake6 &>/dev/null; then
    QT_PLUGIN_DIR="$(qmake6 -query QT_INSTALL_PLUGINS 2>/dev/null || true)"
  fi
  if [ -z "$QT_LIB_DIR" ] && command -v qmake6 &>/dev/null; then
    QT_LIB_DIR="$(qmake6 -query QT_INSTALL_LIBS 2>/dev/null || true)"
  fi

  mkdir -p "$DIST_DIR/lib" "$DIST_DIR/plugins"

  copy_lib() {
    local src="$1"
    [ -z "$src" ] && return 0
    [ ! -e "$src" ] && return 0
    local base
    base="$(basename "$src")"
    if [ ! -e "$DIST_DIR/lib/$base" ]; then
      cp -P "$src" "$DIST_DIR/lib/" 2>/dev/null || true
    fi
    if [ -L "$src" ]; then
      local real
      real="$(readlink -f "$src" 2>/dev/null || true)"
      if [ -n "$real" ] && [ -f "$real" ]; then
        cp -P "$real" "$DIST_DIR/lib/" 2>/dev/null || true
      fi
    fi
  }

  copy_deps() {
    local target="$1"
    [ ! -e "$target" ] && return 0
    while IFS= read -r dep; do
      [ -z "$dep" ] && continue
      local base
      base="$(basename "$dep")"
      case "$base" in
        linux-vdso.so.*|ld-linux*.so*|libc.so.*|libm.so.*|libdl.so.*|libpthread.so.*|librt.so.*)
          continue
          ;;
      esac
      copy_lib "$dep"
    done < <(ldd "$target" 2>/dev/null | awk '/=>/ {print $(NF-1)}' | grep -E '^/' || true)
  }

  for exe in "$DIST_DIR/tile_compile_gui" "$DIST_DIR/tile_compile_runner" "$DIST_DIR/tile_compile_cli"; do
    copy_deps "$exe"
  done

  for sub in platforms styles imageformats iconengines; do
    if [ -d "$QT_PLUGIN_DIR/$sub" ]; then
      mkdir -p "$DIST_DIR/plugins/$sub"
      cp -L "$QT_PLUGIN_DIR/$sub"/*.so "$DIST_DIR/plugins/$sub"/ 2>/dev/null || true
    fi
  done

  if ls "$DIST_DIR/lib/"*.so* >/dev/null 2>&1; then
    for lib in "$DIST_DIR/lib/"*.so*; do
      [ -f "$lib" ] && copy_deps "$lib"
    done
  fi

  if [ -d "$DIST_DIR/plugins" ]; then
    while IFS= read -r sofile; do
      copy_deps "$sofile"
    done < <(find "$DIST_DIR/plugins" -type f -name "*.so" 2>/dev/null)
  fi

  cat > "$DIST_DIR/qt.conf" <<EOF
[Paths]
Prefix=.
Plugins=plugins
Imports=imports
Qml2Imports=qml
EOF

  patchelf --set-rpath '$ORIGIN/lib' "$DIST_DIR/tile_compile_gui" || true
  patchelf --set-rpath '$ORIGIN/lib' "$DIST_DIR/tile_compile_runner" || true
  patchelf --set-rpath '$ORIGIN/lib' "$DIST_DIR/tile_compile_cli" || true
  for lib in "$DIST_DIR/lib/"*.so*; do
    patchelf --set-rpath '$ORIGIN' "$lib" || true
  done
else
  echo "Hinweis: Weder linuxdeployqt noch (qtpaths6+patchelf) gefunden – Libraries werden nicht mitgeliefert."
  echo "Installiere linuxdeployqt oder qtpaths6 + patchelf für portable Dist."
fi

cat > "$DIST_DIR/run_tile_compile_gui.sh" <<'EOF'
#!/bin/bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export LD_LIBRARY_PATH="$DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export QT_PLUGIN_PATH="$DIR/plugins${QT_PLUGIN_PATH:+:$QT_PLUGIN_PATH}"
export QT_QPA_PLATFORM_PLUGIN_PATH="$DIR/plugins/platforms"

exec "$DIR/tile_compile_gui" "$@"
EOF
chmod +x "$DIST_DIR/run_tile_compile_gui.sh" 2>/dev/null || true

cat > "$DIST_DIR/run_tile_compile_runner.sh" <<'EOF'
#!/bin/bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export LD_LIBRARY_PATH="$DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export QT_PLUGIN_PATH="$DIR/plugins${QT_PLUGIN_PATH:+:$QT_PLUGIN_PATH}"
export QT_QPA_PLATFORM_PLUGIN_PATH="$DIR/plugins/platforms"

exec "$DIR/tile_compile_runner" "$@"
EOF
chmod +x "$DIST_DIR/run_tile_compile_runner.sh" 2>/dev/null || true

# Verify runtime dependencies are fully resolved in the bundle.
verify_ldd_target() {
  local target="$1"
  local report="$2"
  if [ ! -e "$target" ]; then
    return 0
  fi
  env LD_LIBRARY_PATH="$DIST_DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
      QT_PLUGIN_PATH="$DIST_DIR/plugins${QT_PLUGIN_PATH:+:$QT_PLUGIN_PATH}" \
      QT_QPA_PLATFORM_PLUGIN_PATH="$DIST_DIR/plugins/platforms${QT_QPA_PLATFORM_PLUGIN_PATH:+:$QT_QPA_PLATFORM_PLUGIN_PATH}" \
      ldd "$target" > "$report" 2>&1 || true

  if grep -q "not found" "$report"; then
    echo "FEHLER: Ungeloeste Abhaengigkeiten in $target" >&2
    cat "$report" >&2
    exit 1
  fi
}

echo "Pruefe Runtime-Abhaengigkeiten..."
verify_ldd_target "$DIST_DIR/tile_compile_gui" "$DIST_DIR/ldd_gui.txt"
verify_ldd_target "$DIST_DIR/tile_compile_runner" "$DIST_DIR/ldd_runner.txt"
verify_ldd_target "$DIST_DIR/tile_compile_cli" "$DIST_DIR/ldd_cli.txt"

if [ -d "$DIST_DIR/plugins" ]; then
  while IFS= read -r so; do
    [ -f "$so" ] || continue
    verify_ldd_target "$so" "$DIST_DIR/ldd_$(basename "$so").txt"
  done < <(find "$DIST_DIR/plugins" -type f -name "*.so" 2>/dev/null)
fi

if command -v zip &>/dev/null; then
  echo ""
  echo "Erzeuge Release-Zip: $ZIP_NAME"
  (
    cd "$PROJECT_DIR/dist" || exit 1
    rm -f "$ZIP_NAME"
    zip -r "$ZIP_NAME" "linux" >/dev/null
  )
  echo "Release-Zip erstellt: $PROJECT_DIR/dist/$ZIP_NAME"
else
  echo "Hinweis: 'zip' nicht gefunden, Release-Zip wird nicht erstellt."
fi

echo ""
echo "========================================"
echo "  Release-Build fertig!"
echo "========================================"
echo ""
echo "Start GUI:"
echo "  cd $DIST_DIR"
echo "  ./run_tile_compile_gui.sh"
echo ""
