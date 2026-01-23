@echo off
setlocal enabledelayedexpansion

set "REPO_ROOT=%~dp0"
set "GUI_DIR=%REPO_ROOT%gui"

if not exist "%GUI_DIR%\" (
  echo ERROR: gui directory not found: %GUI_DIR% 1>&2
  exit /b 1
)

echo Info: starting Qt6 GUI (Python).

where python >nul 2>nul
if errorlevel 1 (
  echo ERROR: missing prerequisite: python 1>&2
  exit /b 1
)

pushd "%REPO_ROOT%" >nul
python -c "import PySide6" >nul 2>nul
if errorlevel 1 (
  echo ERROR: Python dependency missing: PySide6 1>&2
  echo Please install it in your environment ^(e.g. venv^) before running the GUI. 1>&2
  popd >nul
  exit /b 1
)

python -m gui.main
set "EXITCODE=%ERRORLEVEL%"
popd >nul
exit /b %EXITCODE%
