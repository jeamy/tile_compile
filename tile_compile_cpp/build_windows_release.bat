@echo off
rem tile_compile_cpp - Windows Release Build
rem Prueft Abhaengigkeiten, baut Release und erstellt eine portable Dist.

setlocal ENABLEDELAYEDEXPANSION

set "SCRIPT_DIR=%~dp0"
if "%SCRIPT_DIR:~-1%"=="\" set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
set "PROJECT_DIR=%SCRIPT_DIR%"
set "BUILD_DIR=%PROJECT_DIR%\build-windows-release"
set "DIST_DIR=%PROJECT_DIR%\dist\windows"
set BUILD_TYPE=Release

echo === tile_compile_cpp - Windows Release Build ===
echo.

rem ===========================================================================
rem [0] Abhaengigkeiten pruefen
rem ===========================================================================
set MISSING_DEPS=

where cmake >NUL 2>&1
if errorlevel 1 set MISSING_DEPS=!MISSING_DEPS! cmake

where cl >NUL 2>&1
if errorlevel 1 (
  where g++ >NUL 2>&1
  if errorlevel 1 set MISSING_DEPS=!MISSING_DEPS! compiler
)

where qmake6 >NUL 2>&1
if errorlevel 1 (
  if not defined Qt6_DIR (
    if not defined CMAKE_PREFIX_PATH (
      set MISSING_DEPS=!MISSING_DEPS! qt6
    )
  )
)

if defined MISSING_DEPS (
  echo Fehlende Abhaengigkeiten:!MISSING_DEPS!
  echo.
  echo Automatische Installation unter Windows nicht moeglich.
  echo Bitte installiere manuell:
  echo   - CMake: https://cmake.org/download/
  echo   - C++ Compiler: MSVC oder MinGW
  echo   - Qt6 mit passendem Compiler-Kit
  echo.
  echo Typische Konfiguration fuer MinGW:
  echo   set CMAKE_PREFIX_PATH=C:\Qt\6.10.1\mingw_64
  echo   set Qt6_DIR=C:\Qt\6.10.1\mingw_64\lib\cmake\Qt6
  echo   set PATH=C:\Qt\Tools\mingw1310_64\bin;%%PATH%%
  exit /B 1
)

echo Alle Abhaengigkeiten vorhanden.
echo.

rem ===========================================================================
rem [1] CMake konfigurieren
rem ===========================================================================
echo [1/3] CMake konfigurieren...

where g++ >NUL 2>&1
if not errorlevel 1 (
  echo Erkannt: MinGW (g++) - verwende Generator "MinGW Makefiles"
  cmake -S "%PROJECT_DIR%" -B "%BUILD_DIR%" -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=%BUILD_TYPE% -DBUILD_TESTS=OFF
) else (
  echo Erkannt: MSVC/Standardgenerator
  cmake -S "%PROJECT_DIR%" -B "%BUILD_DIR%" -DCMAKE_BUILD_TYPE=%BUILD_TYPE% -DBUILD_TESTS=OFF
)
if errorlevel 1 goto :error

rem ===========================================================================
rem [2] Bauen
rem ===========================================================================
echo.
echo [2/3] Bauen...
cmake --build "%BUILD_DIR%" --config %BUILD_TYPE%
if errorlevel 1 goto :error

rem ===========================================================================
rem [3] Dist-Verzeichnis erstellen
rem ===========================================================================
echo.
echo [3/3] Dist-Verzeichnis erstellen...
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
copy /Y "%PROJECT_DIR%gui_cpp\constants.js" "%DIST_DIR%\gui_cpp" >NUL
copy /Y "%PROJECT_DIR%gui_cpp\styles.qss" "%DIST_DIR%\gui_cpp" >NUL

for %%F in (tile_compile.yaml tile_compile.schema.yaml tile_compile.schema.json) do (
  copy /Y "%PROJECT_DIR%%%F" "%DIST_DIR%" >NUL
)

rem Beispiel-Konfigurationen/Schemas mitliefern
if exist "%PROJECT_DIR%examples" (
  mkdir "%DIST_DIR%\examples" 2>NUL
  xcopy "%PROJECT_DIR%examples\*" "%DIST_DIR%\examples" /E /I /Y >NUL
)

rem Externe Daten (Siril / ASTAP) werden bewusst NICHT eingebuendelt.

set "QT_PREFIX=%CMAKE_PREFIX_PATH%"
if not defined QT_PREFIX (
  if defined Qt6_DIR (
    set "QT_PREFIX=%Qt6_DIR%\..\..\.."
  )
)

set "QT_BIN=%QT_PREFIX%\bin"
if exist "%QT_BIN%\Qt6Core.dll" (
  echo Kopiere Qt6 Runtime-DLLs...
  for %%D in (Qt6Core.dll Qt6Gui.dll Qt6Widgets.dll Qt6Network.dll) do (
    if exist "%QT_BIN%\%%D" copy /Y "%QT_BIN%\%%D" "%DIST_DIR%" >NUL
  )

  for %%D in (libgcc_s_seh-1.dll libstdc++-6.dll libwinpthread-1.dll) do (
    if exist "%QT_BIN%\%%D" copy /Y "%QT_BIN%\%%D" "%DIST_DIR%" >NUL
  )

  mkdir "%DIST_DIR%\platforms" 2>NUL
  if exist "%QT_PREFIX%\plugins\platforms\qwindows.dll" (
    copy /Y "%QT_PREFIX%\plugins\platforms\qwindows.dll" "%DIST_DIR%\platforms" >NUL
  ) else (
    echo WARNUNG: qwindows.dll nicht gefunden unter %QT_PREFIX%\plugins\platforms
  )

  if exist "%QT_PREFIX%\plugins\imageformats" (
    mkdir "%DIST_DIR%\imageformats" 2>NUL
    xcopy "%QT_PREFIX%\plugins\imageformats\*.dll" "%DIST_DIR%\imageformats" /Y >NUL
  )

  if exist "%QT_PREFIX%\plugins\styles" (
    mkdir "%DIST_DIR%\styles" 2>NUL
    xcopy "%QT_PREFIX%\plugins\styles\*.dll" "%DIST_DIR%\styles" /Y >NUL
  )
) else (
  echo WARNUNG: Qt6Core.dll nicht unter %QT_BIN% gefunden. Bitte Qt-Pfad pruefen.
)

set ZIP_NAME=tile_compile_cpp-windows-release.zip
where powershell >NUL 2>&1
if errorlevel 1 (
  echo Hinweis: PowerShell nicht gefunden, Release-Zip wird nicht erstellt.
) else (
  echo Erzeuge Release-Zip: %ZIP_NAME%
  pushd "%PROJECT_DIR%dist"
  if exist "%ZIP_NAME%" del /F /Q "%ZIP_NAME%"
  powershell -NoLogo -NoProfile -Command "Compress-Archive -Path 'windows\*' -DestinationPath '%ZIP_NAME%' -Force"
  popd
  echo Release-Zip erstellt: %PROJECT_DIR%dist\%ZIP_NAME%
)

echo.
echo ========================================
echo   Release-Build fertig!
echo ========================================
echo.
echo Start GUI:
echo   %DIST_DIR%\tile_compile_gui.exe
echo.
goto :eof

:error
echo.
echo Build fehlgeschlagen.
exit /B 1
