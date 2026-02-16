@echo off
rem tile_compile_cpp - Windows Release Build mit automatischer Dependency-Installation
rem Prueft Abhaengigkeiten, installiert sie bei Bedarf via MSYS2, baut Release und erstellt portable Dist.

setlocal ENABLEDELAYEDEXPANSION

set "SCRIPT_DIR=%~dp0"
if "%SCRIPT_DIR:~-1%"=="\" set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
set "PROJECT_DIR=%SCRIPT_DIR%"
set "BUILD_DIR=%PROJECT_DIR%\build-windows-release"
set "DIST_DIR=%PROJECT_DIR%\dist\windows"
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

if not exist "%BUILD_DIR%" (
  echo Erstelle Build-Verzeichnis: %BUILD_DIR%
  mkdir "%BUILD_DIR%" 2>NUL
  if not exist "%BUILD_DIR%" (
    echo FEHLER: Konnte Build-Verzeichnis nicht erstellen.
    echo Bitte pruefe die Berechtigungen oder erstelle es manuell.
    exit /B 1
  )
)

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

for %%B in (tile_compile_gui.exe tile_compile_runner.exe tile_compile_cli.exe) do (
  if exist "%BUILD_DIR%\%%B" (
    copy /Y "%BUILD_DIR%\%%B" "%DIST_DIR%" >NUL
  ) else (
    if exist "%BUILD_DIR%\%BUILD_TYPE%\%%B" (
      copy /Y "%BUILD_DIR%\%BUILD_TYPE%\%%B" "%DIST_DIR%" >NUL
    ) else (
      echo FEHLER: Binaerdatei %%B wurde nicht gefunden.
      goto :error
    )
  )
)

mkdir "%DIST_DIR%\gui_cpp" 2>NUL
copy /Y "%PROJECT_DIR%\gui_cpp\constants.js" "%DIST_DIR%\gui_cpp" >NUL
copy /Y "%PROJECT_DIR%\gui_cpp\styles.qss" "%DIST_DIR%\gui_cpp" >NUL

for %%F in (tile_compile.yaml tile_compile.schema.yaml tile_compile.schema.json) do (
  copy /Y "%PROJECT_DIR%\%%F" "%DIST_DIR%" >NUL
)

rem Beispiel-Konfigurationen/Schemas mitliefern
if exist "%PROJECT_DIR%\examples" (
  mkdir "%DIST_DIR%\examples" 2>NUL
  xcopy "%PROJECT_DIR%\examples\*" "%DIST_DIR%\examples" /E /I /Y >NUL
)

rem Externe Daten (Siril / ASTAP) werden bewusst NICHT eingebuendelt.

set "QT_PREFIX=%CMAKE_PREFIX_PATH%"
rem Extrahiere ersten Pfad aus CMAKE_PREFIX_PATH (falls mehrere mit ; getrennt)
for /f "tokens=1 delims=;" %%i in ("%CMAKE_PREFIX_PATH%") do set "QT_PREFIX=%%i"

if not defined QT_PREFIX (
  if defined Qt6_DIR (
    set "QT_PREFIX=%Qt6_DIR%\..\..\..”
  )
)

echo Verwende Qt-Pfad: %QT_PREFIX%
set "QT_BIN=%QT_PREFIX%\bin"
if exist "%QT_BIN%\Qt6Core.dll" (
  echo Kopiere Qt6 Runtime-DLLs...
  for %%D in (Qt6Core.dll Qt6Gui.dll Qt6Widgets.dll Qt6Network.dll Qt6Svg.dll Qt6PrintSupport.dll Qt6OpenGL.dll Qt6Sql.dll Qt6Test.dll Qt6Concurrent.dll Qt6Xml.dll) do (
    if exist "%QT_BIN%\%%D" (
      copy /Y "%QT_BIN%\%%D" "%DIST_DIR%" >NUL
      echo   Kopiert: %%D
    ) else (
      echo   Nicht gefunden: %%D
    )
  )

  for %%D in (libgcc_s_seh-1.dll libstdc++-6.dll libwinpthread-1.dll) do (
    if exist "%QT_BIN%\%%D" (
      copy /Y "%QT_BIN%\%%D" "%DIST_DIR%" >NUL
      echo   Kopiert: %%D
    ) else if exist "%MSYS2_PREFIX%\bin\%%D" (
      copy /Y "%MSYS2_PREFIX%\bin\%%D" "%DIST_DIR%" >NUL
      echo   Kopiert: %%D
    )
  )

  rem OpenCV DLLs kopieren (mit korrekten Versionsnummern)
  echo Kopiere OpenCV DLLs...
  for %%D in (opencv_core413.dll opencv_imgproc413.dll opencv_imgcodecs413.dll opencv_features2d413.dll opencv_flann413.dll opencv_calib3d413.dll opencv_videoio413.dll) do (
    if exist "%MSYS2_PREFIX%\bin\%%D" (
      copy /Y "%MSYS2_PREFIX%\bin\%%D" "%DIST_DIR%" >NUL
      echo   Kopiert: %%D
    ) else (
      rem Fallback für andere Versionen
      for %%F in ("%MSYS2_PREFIX%\bin\opencv_core*.dll") do (
        copy /Y "%%F" "%DIST_DIR%" >NUL
        echo   Kopiert: %%~nxF
      )
      for %%F in ("%MSYS2_PREFIX%\bin\opencv_imgproc*.dll") do (
        copy /Y "%%F" "%DIST_DIR%" >NUL
        echo   Kopiert: %%~nxF
      )
      for %%F in ("%MSYS2_PREFIX%\bin\opencv_imgcodecs*.dll") do (
        copy /Y "%%F" "%DIST_DIR%" >NUL
        echo   Kopiert: %%~nxF
      )
    )
  )

  rem Weitere Abhaengigkeiten (inkl. OpenSSL)
  echo Kopiere weitere Abhaengigkeiten...
  for %%D in (libcfitsio.dll libyaml-cpp.dll libssl-3.dll libcrypto-3.dll libzstd.dll libbzip2.dll liblzma.dll zlib1.dll) do (
    if exist "%MSYS2_PREFIX%\bin\%%D" (
      copy /Y "%MSYS2_PREFIX%\bin\%%D" "%DIST_DIR%" >NUL
      echo   Kopiert: %%D
    ) else (
      rem Alternative Namensvarianten
      for %%F in ("%MSYS2_PREFIX%\bin\libcfitsio*.dll") do (
        copy /Y "%%F" "%DIST_DIR%" >NUL
        echo   Kopiert: %%~nxF
      )
      for %%F in ("%MSYS2_PREFIX%\bin\libssl*.dll") do (
        copy /Y "%%F" "%DIST_DIR%" >NUL
        echo   Kopiert: %%~nxF
      )
      for %%F in ("%MSYS2_PREFIX%\bin\libcrypto*.dll") do (
        copy /Y "%%F" "%DIST_DIR%" >NUL
        echo   Kopiert: %%~nxF
      )
    )
  )

  mkdir "%DIST_DIR%\platforms" 2>NUL
  if exist "%QT_PREFIX%\plugins\platforms\qwindows.dll" (
    copy /Y "%QT_PREFIX%\plugins\platforms\qwindows.dll" "%DIST_DIR%\platforms" >NUL
    echo   Kopiert: platforms/qwindows.dll
  ) else (
    echo WARNUNG: qwindows.dll nicht gefunden unter %QT_PREFIX%\plugins\platforms
  )

  if exist "%QT_PREFIX%\plugins\imageformats" (
    mkdir "%DIST_DIR%\imageformats" 2>NUL
    xcopy "%QT_PREFIX%\plugins\imageformats\*.dll" "%DIST_DIR%\imageformats" /Y >NUL
    echo   Kopiert: imageformats/*.dll
  )

  if exist "%QT_PREFIX%\plugins\styles" (
    mkdir "%DIST_DIR%\styles" 2>NUL
    xcopy "%QT_PREFIX%\plugins\styles\*.dll" "%DIST_DIR%\styles" /Y >NUL
    echo   Kopiert: styles/*.dll
  )
) else (
  echo WARNUNG: Qt6Core.dll nicht unter %QT_BIN% gefunden. Bitte Qt-Pfad pruefen.
)

set ZIP_NAME=tile_compile_cpp-windows-release.zip
where powershell >NUL 2>&1
if not errorlevel 1 (
  echo.
  echo Erzeuge Release-Zip: %ZIP_NAME%
  pushd "%PROJECT_DIR%\dist"
  if exist "%ZIP_NAME%" del /F /Q "%ZIP_NAME%"
  powershell -NoLogo -NoProfile -Command "Compress-Archive -Path '*.*' -DestinationPath '%ZIP_NAME%' -Force"
  popd
  echo Release-Zip erstellt: %PROJECT_DIR%\dist\%ZIP_NAME%
) else (
  echo.
  echo Erstelle Release-Verzeichnis (PowerShell nicht verfuegbar)...
  echo Das Release-Verzeichnis liegt unter: %DIST_DIR%
)

echo.
echo ===========================================================================
echo   Release-Build FERTIG!
echo ===========================================================================
echo.
echo Distribution liegt unter:
echo   %DIST_DIR%
echo.
echo Ausfuehrbare Dateien:
echo   - GUI:     %DIST_DIR%\tile_compile_gui.exe
echo   - Runner:  %DIST_DIR%\tile_compile_runner.exe
echo   - CLI:     %DIST_DIR%\tile_compile_cli.exe
echo.
echo ZIP-Archiv:
echo   %PROJECT_DIR%\dist\%ZIP_NAME%
echo.
echo Zum Starten:
echo   %DIST_DIR%\tile_compile_gui.exe
echo.

exit /B 0

:error
echo.
echo Build fehlgeschlagen.
exit /B 1
