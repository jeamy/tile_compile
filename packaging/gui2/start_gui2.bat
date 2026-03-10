@echo off
setlocal
set "SCRIPT_DIR=%~dp0"
powershell -ExecutionPolicy Bypass -File "%SCRIPT_DIR%start_gui2.ps1" %*
set "EXITCODE=%ERRORLEVEL%"
if not "%EXITCODE%"=="0" (
  echo.
  echo [gui2] Start fehlgeschlagen. Bitte Meldungen oben lesen.
  echo [gui2] Bitte Meldungen oben lesen und Taste druecken, um das Fenster zu schliessen.
  pause
)
endlocal
