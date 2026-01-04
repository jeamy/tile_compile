@echo off
setlocal enabledelayedexpansion

set "REPO_ROOT=%~dp0"
set "GUI_DIR=%REPO_ROOT%"

if not exist "%GUI_DIR%\" (
  echo ERROR: gui directory not found: %GUI_DIR% 1>&2
  exit /b 1
)

echo Info: GUI workflow: Scan input first; if color mode is UNKNOWN you must confirm it before Start. The confirmation is stored as color_mode_confirmed in runs\^<run_id^>\run_metadata.json

where npm >nul 2>nul
if errorlevel 1 (
  echo ERROR: missing prerequisite: npm 1>&2
  exit /b 1
)

where cargo >nul 2>nul
if errorlevel 1 (
  echo ERROR: missing prerequisite: cargo 1>&2
  exit /b 1
)

where rustc >nul 2>nul
if errorlevel 1 (
  echo ERROR: missing prerequisite: rustc 1>&2
  exit /b 1
)

if not exist "%GUI_DIR%\node_modules\" (
  echo ERROR: node_modules missing in %GUI_DIR% 1>&2
  echo Please run: cd gui-tauri-legacy ^&^& npm install 1>&2
  exit /b 1
)

pushd "%GUI_DIR%" >nul
npm run dev
set "EXITCODE=%ERRORLEVEL%"
popd >nul
exit /b %EXITCODE%
