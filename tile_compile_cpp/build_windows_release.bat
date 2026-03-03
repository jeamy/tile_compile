@echo off
rem tile_compile_cpp - Windows Release Build mit automatischer Dependency-Installation
rem Prueft Abhaengigkeiten, installiert sie bei Bedarf via MSYS2, baut Release und erstellt portable Dist.

setlocal ENABLEDELAYEDEXPANSION

set "SCRIPT_DIR=%~dp0"
if "%SCRIPT_DIR:~-1%"=="\" set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
set "PROJECT_DIR=%SCRIPT_DIR%"
set "BUILD_DIR=C:\windows-tile-compile-build"
set "DIST_DIR=%BUILD_DIR%\dist\windows"
set BUILD_TYPE=Release

echo === tile_compile_cpp - Windows Release Build ===
echo === Mit automatischer MSYS2-Dependency-Installation ===
echo.

rem ===========================================================================
rem [0] Abhaengigkeiten pruefen und installieren
rem ===========================================================================
set MISSING_DEPS=

rem Pruefe MSYS2-Pfade fuer Tools
set "MSYS2_MINGW_BIN="
if exist "C:\msys64\mingw64\bin\g++.exe" (
  set "MSYS2_MINGW_BIN=C:\msys64\mingw64\bin"
) else if exist "C:\msys64\ucrt64\bin\g++.exe" (
  set "MSYS2_MINGW_BIN=C:\msys64\ucrt64\bin"
) else if exist "C:\msys64\clang64\bin\clang++.exe" (
  set "MSYS2_MINGW_BIN=C:\msys64\clang64\bin"
) else if exist "C:\msys2\mingw64\bin\g++.exe" (
  set "MSYS2_MINGW_BIN=C:\msys2\mingw64\bin"
)

if defined MSYS2_MINGW_BIN (
  set "PATH=%MSYS2_MINGW_BIN%;%PATH%"
  echo Erkannt: MSYS2/MinGW unter %MSYS2_MINGW_BIN%
)

where cmake >NUL 2>&1
if errorlevel 1 (
  rem Versuche cmake aus MSYS2-Pfaden
  if exist "C:\msys64\usr\bin\cmake.exe" (
    set "PATH=C:\msys64\usr\bin;%PATH%"
  ) else if exist "C:\msys2\usr\bin\cmake.exe" (
    set "PATH=C:\msys2\usr\bin;%PATH%"
  )
)
where cmake >NUL 2>&1
if errorlevel 1 set MISSING_DEPS=!MISSING_DEPS! cmake

where g++ >NUL 2>&1
if errorlevel 1 set MISSING_DEPS=!MISSING_DEPS! g++

rem Qt6 pruefen - sowohl qmake6 als auch cmake FindQt6
set QT6_FOUND=0
where qmake6 >NUL 2>&1
if not errorlevel 1 set QT6_FOUND=1

if exist "C:\Qt\6.10.1\mingw_64\lib\cmake\Qt6\Qt6Config.cmake" set QT6_FOUND=1
if exist "C:\Qt\6.8.2\mingw_64\lib\cmake\Qt6\Qt6Config.cmake" set QT6_FOUND=1
if exist "C:\msys64\mingw64\lib\cmake\Qt6\Qt6Config.cmake" set QT6_FOUND=1
if exist "C:\msys64\ucrt64\lib\cmake\Qt6\Qt6Config.cmake" set QT6_FOUND=1
if exist "C:\msys64\clang64\lib\cmake\Qt6\Qt6Config.cmake" set QT6_FOUND=1
if exist "C:\msys2\mingw64\lib\cmake\Qt6\Qt6Config.cmake" set QT6_FOUND=1

if "%QT6_FOUND%"=="0" set MISSING_DEPS=!MISSING_DEPS! qt6

rem OpenCV pruefen
set OPENCV_FOUND=0
pkg-config --exists opencv4 2>NUL
if not errorlevel 1 set OPENCV_FOUND=1

if exist "C:\msys64\mingw64\lib\cmake\opencv4\OpenCVConfig.cmake" set OPENCV_FOUND=1
if exist "C:\msys64\ucrt64\lib\cmake\opencv4\OpenCVConfig.cmake" set OPENCV_FOUND=1
if exist "C:\msys64\clang64\lib\cmake\opencv4\OpenCVConfig.cmake" set OPENCV_FOUND=1
if exist "C:\msys2\mingw64\lib\cmake\opencv4\OpenCVConfig.cmake" set OPENCV_FOUND=1

if "%OPENCV_FOUND%"=="0" set MISSING_DEPS=!MISSING_DEPS! opencv

echo [0/4] Pruefe Abhaengigkeiten...
if defined MISSING_DEPS (
  echo Fehlende Abhaengigkeiten:!MISSING_DEPS!
  echo.
) else (
  echo Alle Abhaengigkeiten bereits installiert.
  goto :deps_done
)
rem ===========================================================================
rem [0.5] Automatische Installation via MSYS2
rem ===========================================================================
echo [0.5/4] Versuche automatische Installation via MSYS2...
echo.

set MSYS2_FOUND=0
set MSYS2_PATH=

if exist "C:\msys64\usr\bin\pacman.exe" (
  set MSYS2_FOUND=1
  set "MSYS2_PATH=C:\msys64"
) else if exist "C:\msys2\usr\bin\pacman.exe" (
  set MSYS2_FOUND=1
  set "MSYS2_PATH=C:\msys2"
)

if "%MSYS2_FOUND%"=="0" (
  echo ===========================================================================
  echo MSYS2 NICHT GEFUNDEN - Automatische Installation nicht moeglich.
  echo ===========================================================================
  echo.
  echo Bitte installiere MSYS2 manuell:
  echo   1. Downloade von: https://www.msys2.org/
  echo   2. Fuehre das Installer aus (C:\msys64 empfohlen)
  echo   3. Oeffne "MSYS2 MinGW 64-bit" Terminal
  echo   4. Fuehre aus:
  echo      pacman -Syu
  echo      pacman -S --needed mingw-w64-x86_64-toolchain mingw-w64-x86_64-cmake mingw-w64-x86_64-pkgconf
  echo      pacman -S --needed mingw-w64-x86_64-eigen3 mingw-w64-x86_64-opencv mingw-w64-x86_64-cfitsio
  echo      pacman -S --needed mingw-w64-x86_64-yaml-cpp mingw-w64-x86_64-nlohmann-json mingw-w64-x86_64-openssl
  echo      pacman -S --needed mingw-w64-x86_64-qt6-base mingw-w64-x86_64-qt6-tools
  echo.
  echo Danach dieses Script erneut ausfuehren.
  echo.
  exit /B 1
)

echo MSYS2 gefunden unter: %MSYS2_PATH%
echo.
echo Installiere/Update alle Abhaengigkeiten...
echo Dies kann einige Minuten dauern bei erster Installation.
echo.

rem Fuehre pacman-Befehle aus
set "PACMAN=%MSYS2_PATH%\usr\bin\pacman.exe"

echo [1] Update Package-Datenbank...
"%PACMAN%" -Sy
if errorlevel 1 (
  echo FEHLER: Konnte pacman nicht ausfuehren. Bitte als Administrator erneut versuchen.
  exit /B 1
)

echo.
echo [2] Installiere Toolchain und Build-Tools...
"%PACMAN%" -S --needed --noconfirm mingw-w64-x86_64-toolchain mingw-w64-x86_64-cmake mingw-w64-x86_64-pkgconf mingw-w64-x86_64-make
if errorlevel 1 (
  echo FEHLER: Installation der Toolchain fehlgeschlagen.
  exit /B 1
)

echo.
echo [3] Installiere Bibliotheken (Eigen3, OpenCV, cfitsio, yaml-cpp, json, openssl)...
"%PACMAN%" -S --needed --noconfirm mingw-w64-x86_64-eigen3 mingw-w64-x86_64-opencv mingw-w64-x86_64-cfitsio mingw-w64-x86_64-yaml-cpp mingw-w64-x86_64-nlohmann-json mingw-w64-x86_64-openssl
if errorlevel 1 (
  echo FEHLER: Installation der Bibliotheken fehlgeschlagen.
  exit /B 1
)

echo.
echo [4] Installiere Qt6...
"%PACMAN%" -S --needed --noconfirm mingw-w64-x86_64-qt6-base mingw-w64-x86_64-qt6-tools mingw-w64-x86_64-qt6-svg
if errorlevel 1 (
  echo FEHLER: Installation von Qt6 fehlgeschlagen.
  exit /B 1
)

echo.
echo ===========================================================================
echo Installation abgeschlossen!
echo ===========================================================================
echo.
echo WICHTIG: Du musst das Terminal jetzt NEU STARTEN, damit die neuen
echo          Abhaengigkeiten im PATH gefunden werden.
echo.
echo Moechtest du jetzt neu starten und dann automatisch weitermachen?
set /p RESTART="Neustarten und weitermachen? (j/n): "
if /I "%RESTART%"=="j" (
  echo Starte neu...
  start "" "%~f0"
  exit
) else (
  echo Bitte starte das Terminal neu und fuehre das Script erneut aus.
  exit /B 0
)

:deps_done
echo.
rem ===========================================================================
rem [1] MSYS2-Umgebung erkennen und Pfade setzen
rem ===========================================================================
echo [1/4] Erkenne Build-Umgebung...

set USE_MINGW=1
set MSYS2_PREFIX=

if exist "C:\msys64\mingw64\lib\cmake\opencv4\OpenCVConfig.cmake" (
  set "MSYS2_PREFIX=C:\msys64\mingw64"
) else if exist "C:\msys64\ucrt64\lib\cmake\opencv4\OpenCVConfig.cmake" (
  set "MSYS2_PREFIX=C:\msys64\ucrt64"
) else if exist "C:\msys64\clang64\lib\cmake\opencv4\OpenCVConfig.cmake" (
  set "MSYS2_PREFIX=C:\msys64\clang64"
) else if exist "C:\msys2\mingw64\lib\cmake\opencv4\OpenCVConfig.cmake" (
  set "MSYS2_PREFIX=C:\msys2\mingw64"
)

if not defined MSYS2_PREFIX (
  echo FEHLER: MSYS2 MinGW-Umgebung nicht gefunden.
  echo Bitte installiere MSYS2 wie oben beschrieben.
  exit /B 1
)

echo Erkannt: MSYS2/MinGW unter %MSYS2_PREFIX%

rem Qt6-Pfad setzen (entweder aus MSYS2 oder standalone Qt)
set QT_PREFIX=
if exist "%MSYS2_PREFIX%\lib\cmake\Qt6\Qt6Config.cmake" (
  set "QT_PREFIX=%MSYS2_PREFIX%"
  echo Qt6 gefunden unter MSYS2: %QT_PREFIX%
) else (
  for /d %%D in (C:\Qt\6.*) do (
    if exist "%%D\mingw_64\lib\cmake\Qt6\Qt6Config.cmake" (
      set "QT_PREFIX=%%D\mingw_64"
      echo Qt6 gefunden unter: !QT_PREFIX!
      goto :qt_found
    )
  )
)
:qt_found

if not defined QT_PREFIX (
  echo FEHLER: Qt6 nicht gefunden!
  exit /B 1
)
set "QT_BIN=%QT_PREFIX%\bin"

rem CMAKE_PREFIX_PATH zusammenbauen
set "CMAKE_PREFIX_PATH=%QT_PREFIX%;%MSYS2_PREFIX%"
set "Qt6_DIR=%QT_PREFIX%\lib\cmake\Qt6"

rem PATH fuer DLLs setzen
set "PATH=%QT_PREFIX%\bin;%MSYS2_PREFIX%\bin;%PATH%"

rem OpenCV_DIR setzen falls noetig
if not defined OpenCV_DIR (
  if exist "%MSYS2_PREFIX%\lib\cmake\opencv4\OpenCVConfig.cmake" (
    set "OpenCV_DIR=%MSYS2_PREFIX%\lib\cmake\opencv4"
  )
)

set "OBJDUMP_EXE=%MSYS2_PREFIX%\bin\objdump.exe"
if not exist "%OBJDUMP_EXE%" set "OBJDUMP_EXE=%MSYS2_PREFIX%\bin\x86_64-w64-mingw32-objdump.exe"
if not exist "%OBJDUMP_EXE%" (
  for %%P in (objdump.exe x86_64-w64-mingw32-objdump.exe) do (
    for %%Q in (%%~$PATH:P) do (
      if exist "%%~Q" set "OBJDUMP_EXE=%%~Q"
    )
  )
)

echo.
echo Konfiguration:
echo   CMAKE_PREFIX_PATH=%CMAKE_PREFIX_PATH%
echo   Qt6_DIR=%Qt6_DIR%
echo   OpenCV_DIR=%OpenCV_DIR%
echo.

rem ===========================================================================
rem [2] CMake konfigurieren
rem ===========================================================================
echo [2/4] CMake konfigurieren...

rem Build-Verzeichnis sauber neu anlegen (del /S loescht keine Ordner und laesst Attribute stehen)
if exist "%BUILD_DIR%" (
  echo Entferne altes Build-Verzeichnis...
  attrib -R -S -H "%BUILD_DIR%\*.*" /S /D 2>NUL
  rmdir /S /Q "%BUILD_DIR%" 2>NUL
  timeout /t 1 /nobreak >NUL
)

echo Erstelle Build-Verzeichnis: %BUILD_DIR%
mkdir "%BUILD_DIR%" 2>NUL
if not exist "%BUILD_DIR%" (
  echo FEHLER: Konnte Build-Verzeichnis nicht erstellen.
  echo Bitte pruefe die Berechtigungen oder erstelle es manuell.
  exit /B 1
)

rem Erstelle kritische Unterverzeichnisse vorab
mkdir "%BUILD_DIR%\.qt" 2>NUL
mkdir "%BUILD_DIR%\CMakeFiles" 2>NUL

rem Pruefe Schreibrechte im Build-Verzeichnis
echo Test > "%BUILD_DIR%\test_write.tmp" 2>NUL
if not exist "%BUILD_DIR%\test_write.tmp" (
  echo FEHLER: Keine Schreibrechte im Build-Verzeichnis.
  echo Bitte pruefe die Berechtigungen oder fuehre als Administrator aus.
  exit /B 1
)
del "%BUILD_DIR%\test_write.tmp" 2>NUL

rem CMake konfigurieren mit verbesserten Windows-Settings
echo Konfiguriere CMake...
cmake -S "%PROJECT_DIR%" -B "%BUILD_DIR%" ^
  -G "Ninja" ^
  -DCMAKE_BUILD_TYPE=%BUILD_TYPE% ^
  -DBUILD_TESTS=OFF ^
  -DCMAKE_PREFIX_PATH="%CMAKE_PREFIX_PATH%" ^
  -DQt6_DIR="%Qt6_DIR%" ^
  -DOpenCV_DIR="%OpenCV_DIR%" ^
  -DCMAKE_CXX_STANDARD=17 ^
  -DMINGW_HAS_SECURE_API=1 ^
  -DWIN32_LEAN_AND_MEAN=1 ^
  -DNOMINMAX=1 ^
  -DCMAKE_CXX_FLAGS="-D_USE_MATH_DEFINES -D_CRT_SECURE_NO_WARNINGS" ^
  -DCMAKE_EXE_LINKER_FLAGS="-static-libgcc -static-libstdc++" ^
  -DCMAKE_AUTOGEN_VERBOSE=ON ^
  -DCMAKE_DISABLE_PRECOMPILE_HEADERS=ON

if errorlevel 1 (
  echo.
  echo FEHLER: CMake-Konfiguration mit Ninja fehlgeschlagen. Versuche MinGW Makefiles...
  cmake -S "%PROJECT_DIR%" -B "%BUILD_DIR%" ^
    -G "MinGW Makefiles" ^
    -DCMAKE_BUILD_TYPE=%BUILD_TYPE% ^
    -DBUILD_TESTS=OFF ^
    -DCMAKE_PREFIX_PATH="%CMAKE_PREFIX_PATH%" ^
    -DQt6_DIR="%Qt6_DIR%" ^
    -DOpenCV_DIR="%OpenCV_DIR%" ^
    -DCMAKE_CXX_STANDARD=17 ^
    -DMINGW_HAS_SECURE_API=1 ^
    -DWIN32_LEAN_AND_MEAN=1 ^
    -DNOMINMAX=1 ^
    -DCMAKE_CXX_FLAGS="-D_USE_MATH_DEFINES -D_CRT_SECURE_NO_WARNINGS" ^
    -DCMAKE_EXE_LINKER_FLAGS="-static-libgcc -static-libstdc++" ^
    -DCMAKE_AUTOGEN_VERBOSE=ON ^
    -DCMAKE_DISABLE_PRECOMPILE_HEADERS=ON
  
  if errorlevel 1 (
    echo.
    echo FEHLER: CMake-Konfiguration fehlgeschlagen.
    echo.
    echo Moegliche Loesungen:
    echo   1. Pruefe ob alle Abhaengigkeiten installiert sind:
    echo      pacman -Q ^| findstr mingw-w64-x86_64
    echo   2. Loesche den Build-Ordner und versuche es erneut:
    echo      rmdir /S /Q "%BUILD_DIR%"
    echo.
    exit /B 1
  )
)

rem ===========================================================================
rem [3] Bauen
rem ===========================================================================
echo.
echo [3/4] Bauen... (dies kann mehrere Minuten dauern)
cmake --build "%BUILD_DIR%" --config %BUILD_TYPE% -j%NUMBER_OF_PROCESSORS%
if errorlevel 1 goto :error

rem ===========================================================================
rem [4] Dist-Verzeichnis erstellen
rem ===========================================================================
echo.
echo [4/4] Erstelle Distribution...
if exist "%DIST_DIR%" rmdir /S /Q "%DIST_DIR%"
mkdir "%DIST_DIR%"

echo Installiere Targets und Runtime-Abhaengigkeiten via CMake...
cmake --install "%BUILD_DIR%" --prefix "%DIST_DIR%"
if errorlevel 1 (
  echo FEHLER: cmake --install fehlgeschlagen.
  goto :error
)

set "DIST_BIN=%DIST_DIR%\bin"
if not exist "%DIST_BIN%" mkdir "%DIST_BIN%"

rem Fallback: falls install() in der lokalen CMake-Version keine Runtime-Dateien liefert
for %%B in (tile_compile_gui.exe tile_compile_runner.exe tile_compile_cli.exe) do (
  if not exist "%DIST_BIN%\%%B" (
    if exist "%BUILD_DIR%\%%B" copy /Y "%BUILD_DIR%\%%B" "%DIST_BIN%" >NUL
    if exist "%BUILD_DIR%\%BUILD_TYPE%\%%B" copy /Y "%BUILD_DIR%\%BUILD_TYPE%\%%B" "%DIST_BIN%" >NUL
  )
)

for %%B in (tile_compile_gui.exe tile_compile_runner.exe tile_compile_cli.exe) do (
  if not exist "%DIST_BIN%\%%B" (
    echo FEHLER: Binaerdatei %%B wurde nicht gefunden.
    goto :error
  )
)

mkdir "%DIST_DIR%\gui_cpp" 2>NUL
copy /Y "%PROJECT_DIR%\gui_cpp\constants.js" "%DIST_DIR%\gui_cpp" >NUL
copy /Y "%PROJECT_DIR%\gui_cpp\styles.qss" "%DIST_DIR%\gui_cpp" >NUL

for %%F in (tile_compile.yaml tile_compile.schema.yaml tile_compile.schema.json) do (
  copy /Y "%PROJECT_DIR%\%%F" "%DIST_DIR%" >NUL
)

if exist "%PROJECT_DIR%\examples" (
  mkdir "%DIST_DIR%\examples" 2>NUL
  xcopy "%PROJECT_DIR%\examples\*" "%DIST_DIR%\examples" /E /I /Y >NUL
)

set "WINDEPLOYQT="
if exist "%QT_PREFIX%\bin\windeployqt6.exe" set "WINDEPLOYQT=%QT_PREFIX%\bin\windeployqt6.exe"
if not defined WINDEPLOYQT if exist "%QT_PREFIX%\bin\windeployqt.exe" set "WINDEPLOYQT=%QT_PREFIX%\bin\windeployqt.exe"

if defined WINDEPLOYQT (
  echo Fuehre windeployqt aus...
  "%WINDEPLOYQT%" --no-translations --no-opengl-sw --dir "%DIST_BIN%" "%DIST_BIN%\tile_compile_gui.exe"
) else (
  echo WARNUNG: windeployqt nicht gefunden. Nutze nur CMake-Runtime-Install + Fallback-Copies.
)

if not exist "%DIST_BIN%\qt.conf" (
  > "%DIST_BIN%\qt.conf" (
    echo [Paths]
    echo Plugins=.
  )
)

echo Kopiere notwendige Runtime-DLLs (Fallback) aus %MSYS2_PREFIX%\bin ...
echo   [1/5] Toolchain + Core...
for %%D in (libgcc_s_seh-1.dll libstdc++-6.dll libwinpthread-1.dll) do (
  if exist "%MSYS2_PREFIX%\bin\%%D" copy /Y "%MSYS2_PREFIX%\bin\%%D" "%DIST_BIN%" >NUL
)
for %%D in (libintl-8.dll libiconv-2.dll libunistring-5.dll libpcre2-8-0.dll) do (
  if exist "%MSYS2_PREFIX%\bin\%%D" copy /Y "%MSYS2_PREFIX%\bin\%%D" "%DIST_BIN%" >NUL
)

echo   [2/5] Compression + Crypto...
for %%F in ("%MSYS2_PREFIX%\bin\zlib*.dll") do copy /Y "%%F" "%DIST_BIN%" >NUL 2>NUL
for %%F in ("%MSYS2_PREFIX%\bin\libbz2*.dll") do copy /Y "%%F" "%DIST_BIN%" >NUL 2>NUL
for %%F in ("%MSYS2_PREFIX%\bin\liblzma*.dll") do copy /Y "%%F" "%DIST_BIN%" >NUL 2>NUL
for %%F in ("%MSYS2_PREFIX%\bin\libzstd*.dll") do copy /Y "%%F" "%DIST_BIN%" >NUL 2>NUL
for %%F in ("%MSYS2_PREFIX%\bin\libcrypto*.dll") do copy /Y "%%F" "%DIST_BIN%" >NUL 2>NUL
for %%F in ("%MSYS2_PREFIX%\bin\libssl*.dll") do copy /Y "%%F" "%DIST_BIN%" >NUL 2>NUL
for %%F in ("%MSYS2_PREFIX%\bin\libexpat*.dll") do copy /Y "%%F" "%DIST_BIN%" >NUL 2>NUL
for %%F in ("%MSYS2_PREFIX%\bin\libffi*.dll") do copy /Y "%%F" "%DIST_BIN%" >NUL 2>NUL
for %%F in ("%MSYS2_PREFIX%\bin\libsqlite3*.dll") do copy /Y "%%F" "%DIST_BIN%" >NUL 2>NUL
for %%F in ("%MSYS2_PREFIX%\bin\libcurl*.dll") do copy /Y "%%F" "%DIST_BIN%" >NUL 2>NUL
for %%F in ("%MSYS2_PREFIX%\bin\libssh2*.dll") do copy /Y "%%F" "%DIST_BIN%" >NUL 2>NUL
for %%F in ("%MSYS2_PREFIX%\bin\libnghttp2*.dll") do copy /Y "%%F" "%DIST_BIN%" >NUL 2>NUL
for %%F in ("%MSYS2_PREFIX%\bin\libnghttp3*.dll") do copy /Y "%%F" "%DIST_BIN%" >NUL 2>NUL
for %%F in ("%MSYS2_PREFIX%\bin\libngtcp2*.dll") do copy /Y "%%F" "%DIST_BIN%" >NUL 2>NUL
for %%F in ("%MSYS2_PREFIX%\bin\libidn2*.dll") do copy /Y "%%F" "%DIST_BIN%" >NUL 2>NUL
for %%F in ("%MSYS2_PREFIX%\bin\libpsl*.dll") do copy /Y "%%F" "%DIST_BIN%" >NUL 2>NUL
for %%F in ("%MSYS2_PREFIX%\bin\libb2*.dll") do copy /Y "%%F" "%DIST_BIN%" >NUL 2>NUL

echo   [3/5] Project Dependencies...
for %%F in ("%MSYS2_PREFIX%\bin\libcfitsio*.dll") do copy /Y "%%F" "%DIST_BIN%" >NUL 2>NUL
for %%F in ("%MSYS2_PREFIX%\bin\libyaml-cpp*.dll") do copy /Y "%%F" "%DIST_BIN%" >NUL 2>NUL

echo   [4/5] OpenCV + Image Formats...
for %%F in ("%MSYS2_PREFIX%\bin\libopencv*.dll") do copy /Y "%%F" "%DIST_BIN%" >NUL 2>NUL
for %%F in ("%MSYS2_PREFIX%\bin\libpng*.dll") do copy /Y "%%F" "%DIST_BIN%" >NUL 2>NUL
for %%F in ("%MSYS2_PREFIX%\bin\libjpeg*.dll") do copy /Y "%%F" "%DIST_BIN%" >NUL 2>NUL
for %%F in ("%MSYS2_PREFIX%\bin\libtiff*.dll") do copy /Y "%%F" "%DIST_BIN%" >NUL 2>NUL
for %%F in ("%MSYS2_PREFIX%\bin\libwebp*.dll") do copy /Y "%%F" "%DIST_BIN%" >NUL 2>NUL

echo   [5/5] FFmpeg + Qt Text Rendering...
for %%F in ("%MSYS2_PREFIX%\bin\avcodec*.dll") do copy /Y "%%F" "%DIST_BIN%" >NUL 2>NUL
for %%F in ("%MSYS2_PREFIX%\bin\avformat*.dll") do copy /Y "%%F" "%DIST_BIN%" >NUL 2>NUL
for %%F in ("%MSYS2_PREFIX%\bin\avutil*.dll") do copy /Y "%%F" "%DIST_BIN%" >NUL 2>NUL
for %%F in ("%MSYS2_PREFIX%\bin\swscale*.dll") do copy /Y "%%F" "%DIST_BIN%" >NUL 2>NUL
for %%F in ("%MSYS2_PREFIX%\bin\swresample*.dll") do copy /Y "%%F" "%DIST_BIN%" >NUL 2>NUL
for %%F in ("%MSYS2_PREFIX%\bin\libharfbuzz*.dll") do copy /Y "%%F" "%DIST_BIN%" >NUL 2>NUL
for %%F in ("%MSYS2_PREFIX%\bin\libfreetype*.dll") do copy /Y "%%F" "%DIST_BIN%" >NUL 2>NUL
for %%F in ("%MSYS2_PREFIX%\bin\libgraphite2*.dll") do copy /Y "%%F" "%DIST_BIN%" >NUL 2>NUL
for %%F in ("%MSYS2_PREFIX%\bin\libbrotli*.dll") do copy /Y "%%F" "%DIST_BIN%" >NUL 2>NUL
for %%F in ("%MSYS2_PREFIX%\bin\libicu*.dll") do copy /Y "%%F" "%DIST_BIN%" >NUL 2>NUL
for %%F in ("%MSYS2_PREFIX%\bin\libglib-2.0*.dll") do copy /Y "%%F" "%DIST_BIN%" >NUL 2>NUL
for %%F in ("%MSYS2_PREFIX%\bin\libdouble-conversion*.dll") do copy /Y "%%F" "%DIST_BIN%" >NUL 2>NUL
for %%F in ("%MSYS2_PREFIX%\bin\libmd4c*.dll") do copy /Y "%%F" "%DIST_BIN%" >NUL 2>NUL

if exist "%OBJDUMP_EXE%" (
  echo Starte transitive DLL-Suche (objdump sweep)...
  call :run_dep_sweep
) else (
  echo WARNUNG: objdump nicht gefunden. Transitive DLL-Suche uebersprungen.
)

if exist "%MSYS2_PREFIX%\bin\ntldd.exe" (
  echo Pruefe DLL-Abhaengigkeiten mit ntldd...
  "%MSYS2_PREFIX%\bin\ntldd.exe" -R "%DIST_BIN%\tile_compile_runner.exe" > "%DIST_DIR%\ntldd_runner.txt"
  "%MSYS2_PREFIX%\bin\ntldd.exe" -R "%DIST_BIN%\tile_compile_cli.exe" > "%DIST_DIR%\ntldd_cli.txt"
  "%MSYS2_PREFIX%\bin\ntldd.exe" -R "%DIST_BIN%\tile_compile_gui.exe" > "%DIST_DIR%\ntldd_gui.txt"
  findstr /I /R "not found missing" "%DIST_DIR%\ntldd_runner.txt" >NUL && goto :error
  findstr /I /R "not found missing" "%DIST_DIR%\ntldd_cli.txt" >NUL && goto :error
  findstr /I /R "not found missing" "%DIST_DIR%\ntldd_gui.txt" >NUL && goto :error
) else (
  echo WARNUNG: ntldd.exe nicht gefunden. Verifikation uebersprungen.
)

if not exist "%BUILD_DIR%\dist" mkdir "%BUILD_DIR%\dist"
set ZIP_NAME=tile_compile_cpp-windows-release.zip
set "ZIP_FULL=%BUILD_DIR%\dist\%ZIP_NAME%"
if exist "%ZIP_FULL%" del /F /Q "%ZIP_FULL%"
echo.
echo Erzeuge Release-Zip: %ZIP_NAME%
echo Compress-Archive -Path '%DIST_DIR%' -DestinationPath '%ZIP_FULL%' -Force > "%TEMP%\tc_zip.ps1"
powershell -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%TEMP%\tc_zip.ps1"
if exist "%ZIP_FULL%" (
  echo Release-Zip erstellt: %ZIP_FULL%
) else (
  echo WARNUNG: ZIP-Erstellung fehlgeschlagen. Release-Verzeichnis liegt unter: %DIST_DIR%
)
del "%TEMP%\tc_zip.ps1" 2>NUL

echo.
echo ===========================================================================
echo   Release-Build FERTIG!
echo ===========================================================================
echo.
echo Distribution liegt unter:
echo   %DIST_DIR%
echo.
echo Ausfuehrbare Dateien:
echo   - GUI:     %DIST_BIN%\tile_compile_gui.exe
echo   - Runner:  %DIST_BIN%\tile_compile_runner.exe
echo   - CLI:     %DIST_BIN%\tile_compile_cli.exe
echo.
echo ZIP-Archiv:
echo   %ZIP_FULL%
echo.
echo Zum Starten:
echo   %DIST_BIN%\tile_compile_gui.exe
echo.

exit /B 0

:error
echo.
echo Build fehlgeschlagen.
exit /B 1

:run_dep_sweep
set DEP_PASS=0
for /L %%P in (1,1,10) do (
  set /a DEP_PASS+=1
  set "COPIED_THIS_PASS=0"
  echo   Pass !DEP_PASS!...

  if exist "%DIST_BIN%\tile_compile_gui.exe" call :scan_binary_deps "%DIST_BIN%\tile_compile_gui.exe"
  if exist "%DIST_BIN%\tile_compile_runner.exe" call :scan_binary_deps "%DIST_BIN%\tile_compile_runner.exe"
  if exist "%DIST_BIN%\tile_compile_cli.exe" call :scan_binary_deps "%DIST_BIN%\tile_compile_cli.exe"

  if exist "%DIST_BIN%\platforms\qwindows.dll" call :scan_binary_deps "%DIST_BIN%\platforms\qwindows.dll"
  
  if exist "%DIST_BIN%\imageformats" (
    for %%B in ("%DIST_BIN%\imageformats\*.dll") do call :scan_binary_deps "%%B"
  )
  if exist "%DIST_BIN%\styles" (
    for %%B in ("%DIST_BIN%\styles\*.dll") do call :scan_binary_deps "%%B"
  )

  if "!COPIED_THIS_PASS!"=="0" (
    echo   Keine neuen DLLs in Pass !DEP_PASS!, Sweep beendet.
    goto :run_dep_sweep_done
  )
)

:run_dep_sweep_done
echo   DLL-Sweep abgeschlossen nach !DEP_PASS! Pass(es).
exit /B 0

:scan_binary_deps
set "SCAN_BIN=%~1"
if not exist "%SCAN_BIN%" exit /B 0

set "TEMP_DEPS=%TEMP%\tc_deps_%RANDOM%.txt"
"%OBJDUMP_EXE%" -p "%SCAN_BIN%" 2>NUL | findstr /I /C:"DLL Name:" > "%TEMP_DEPS%" 2>NUL

if exist "%TEMP_DEPS%" (
  for /f "tokens=3" %%D in (%TEMP_DEPS%) do (
    call :copy_dep_dll "%%D"
  )
  del "%TEMP_DEPS%" 2>NUL
)
exit /B 0

:copy_dep_dll
set "DEP_DLL=%~1"
if not defined DEP_DLL exit /B 0

if exist "%DIST_BIN%\%DEP_DLL%" exit /B 0

for %%S in (kernel32.dll user32.dll gdi32.dll advapi32.dll shell32.dll ole32.dll oleaut32.dll comdlg32.dll comctl32.dll ws2_32.dll winmm.dll imm32.dll secur32.dll bcrypt.dll rpcrt4.dll shlwapi.dll uxtheme.dll dwmapi.dll msvcrt.dll d3d9.dll setupapi.dll shcore.dll wtsapi32.dll) do (
  if /I "%DEP_DLL%"=="%%~S" exit /B 0
)

rem Skip api-ms-win-* DLLs (Windows API sets)
echo %DEP_DLL% | findstr /I /C:"api-ms-win-" >NUL
if not errorlevel 1 exit /B 0

if exist "%QT_BIN%\%DEP_DLL%" (
  copy /Y "%QT_BIN%\%DEP_DLL%" "%DIST_BIN%" >NUL
  echo   [auto] Kopiert: %DEP_DLL%  ^(von QT_BIN^)
  set "COPIED_THIS_PASS=1"
  exit /B 0
)

if exist "%MSYS2_PREFIX%\bin\%DEP_DLL%" (
  copy /Y "%MSYS2_PREFIX%\bin\%DEP_DLL%" "%DIST_BIN%" >NUL
  echo   [auto] Kopiert: %DEP_DLL%  ^(von MSYS2 bin^)
  set "COPIED_THIS_PASS=1"
  exit /B 0
)

if /I not "%DEP_DLL%"=="api-ms-win-core-path-l1-1-0.dll" (
  echo   [auto][warn] Nicht gefunden: %DEP_DLL%
)
exit /B 0
