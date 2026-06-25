@echo off
setlocal enabledelayedexpansion

rem ============================================================
rem  start_backend.bat  -  Windows equivalent of start_backend.sh
rem  Starts the tile_compile C++ backend with proper env vars.
rem ============================================================

set "SCRIPT_DIR=%~dp0"
if "%SCRIPT_DIR:~-1%"=="\" set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
set "PROJECT_ROOT=%SCRIPT_DIR%"

rem --- Build directories ---
if not defined BUILD_DIR set "BUILD_DIR=%PROJECT_ROOT%\web_backend_cpp\build"
if not defined CPP_BUILD_DIR set "CPP_BUILD_DIR=%PROJECT_ROOT%\tile_compile_cpp\build"
set "BUILD_TYPE=Release"

rem --- Backend binary ---
if not defined BACKEND_BIN set "BACKEND_BIN=%BUILD_DIR%\tile_compile_web_backend.exe"

rem --- C++ runner / CLI ---
if not defined TILE_COMPILE_CLI set "TILE_COMPILE_CLI=%CPP_BUILD_DIR%\tile_compile_cli.exe"
if not defined TILE_COMPILE_RUNNER set "TILE_COMPILE_RUNNER=%CPP_BUILD_DIR%\tile_compile_runner.exe"

rem --- Config / schema / presets ---
if not defined TILE_COMPILE_CONFIG set "TILE_COMPILE_CONFIG=%PROJECT_ROOT%\tile_compile_cpp\tile_compile.yaml"
if not defined TILE_COMPILE_SCHEMA set "TILE_COMPILE_SCHEMA=%PROJECT_ROOT%\tile_compile_cpp\tile_compile.schema.yaml"
if not defined TILE_COMPILE_PRESETS_DIR set "TILE_COMPILE_PRESETS_DIR=%PROJECT_ROOT%\tile_compile_cpp\examples"

rem --- UI ---
if not defined TILE_COMPILE_UI_DIR set "TILE_COMPILE_UI_DIR=%PROJECT_ROOT%\web_frontend_v3"

rem --- Agent service ---
if not defined TILE_COMPILE_AGENT_SERVICE_DIR set "TILE_COMPILE_AGENT_SERVICE_DIR=%PROJECT_ROOT%\agent_service"

rem --- Runs directory ---
if not defined TILE_COMPILE_RUNS_DIR set "TILE_COMPILE_RUNS_DIR=%PROJECT_ROOT%\runs"

rem --- Allowed roots (semicolon-separated on Windows) ---
if not defined TILE_COMPILE_ALLOWED_ROOTS set "TILE_COMPILE_ALLOWED_ROOTS=%PROJECT_ROOT%;%TILE_COMPILE_RUNS_DIR%;%PROJECT_ROOT%\tmp"

rem --- AI agent autostart ---
if not defined TILE_COMPILE_AI_AGENT_AUTOSTART set "TILE_COMPILE_AI_AGENT_AUTOSTART=1"

rem --- Host / port ---
if not defined HOST set "HOST=127.0.0.1"
if not defined PORT set "PORT=8080"

set "DO_BUILD=1"

rem --- Parse command-line args ---
:parse_args
if "%~1"=="" goto args_done
if /i "%~1"=="--host" ( set "HOST=%~2" & shift & shift & goto parse_args )
if /i "%~1"=="--port" ( set "PORT=%~2" & shift & shift & goto parse_args )
if /i "%~1"=="--build-dir" ( set "BUILD_DIR=%~2" & shift & shift & goto parse_args )
if /i "%~1"=="--cpp-build-dir" ( set "CPP_BUILD_DIR=%~2" & shift & shift & goto parse_args )
if /i "%~1"=="--backend-bin" ( set "BACKEND_BIN=%~2" & shift & shift & goto parse_args )
if /i "%~1"=="--build-type" ( set "BUILD_TYPE=%~2" & shift & shift & goto parse_args )
if /i "%~1"=="--runs-dir" ( set "TILE_COMPILE_RUNS_DIR=%~2" & shift & shift & goto parse_args )
if /i "%~1"=="--no-build" ( set "DO_BUILD=0" & shift & goto parse_args )
if /i "%~1"=="-h" ( call :usage & exit /b 0 )
if /i "%~1"=="--help" ( call :usage & exit /b 0 )
echo [backend] Unknown argument: %~1
shift
goto parse_args
:args_done

rem --- Ensure runs directory exists ---
if not exist "%TILE_COMPILE_RUNS_DIR%" mkdir "%TILE_COMPILE_RUNS_DIR%"

rem --- Build step ---
if "%DO_BUILD%"=="1" (
  if not exist "%CPP_BUILD_DIR%" mkdir "%CPP_BUILD_DIR%"

  echo [backend] Configuring C++ core in %CPP_BUILD_DIR%
  cmake -S "%PROJECT_ROOT%\tile_compile_cpp" -B "%CPP_BUILD_DIR%" -DCMAKE_BUILD_TYPE=%BUILD_TYPE%
  if errorlevel 1 (
    echo [backend] ERROR: C++ core cmake configure failed.
    exit /b 1
  )
  echo [backend] Building tile_compile_runner and tile_compile_cli
  cmake --build "%CPP_BUILD_DIR%" --parallel %NUMBER_OF_PROCESSORS% --target tile_compile_runner tile_compile_cli
  if errorlevel 1 (
    echo [backend] ERROR: C++ core build failed.
    exit /b 1
  )

  echo [backend] Configuring C++ backend in %BUILD_DIR%
  cmake -S "%PROJECT_ROOT%\web_backend_cpp" -B "%BUILD_DIR%" -DCMAKE_BUILD_TYPE=%BUILD_TYPE%
  if errorlevel 1 (
    echo [backend] ERROR: Backend cmake configure failed.
    exit /b 1
  )
  echo [backend] Building tile_compile_web_backend
  cmake --build "%BUILD_DIR%" --parallel %NUMBER_OF_PROCESSORS%
  if errorlevel 1 (
    echo [backend] ERROR: Backend build failed.
    exit /b 1
  )
)

rem --- Check backend binary ---
if not exist "%BACKEND_BIN%" (
  echo [backend] ERROR: Backend binary not found: %BACKEND_BIN%
  exit /b 1
)

rem --- Export env vars for the backend process ---
set "TILE_COMPILE_PROJECT_ROOT=%PROJECT_ROOT%"
set "TILE_COMPILE_HOST=%HOST%"
set "TILE_COMPILE_PORT=%PORT%"

rem --- Start PI AI sidecar (optional) ---
if "%TILE_COMPILE_AI_AGENT_AUTOSTART%"=="0" (
  echo [backend] PI AI sidecar autostart disabled.
  goto start_backend
)
if not exist "%TILE_COMPILE_AGENT_SERVICE_DIR%\package.json" (
  echo [backend] PI AI sidecar not found: %TILE_COMPILE_AGENT_SERVICE_DIR%
  goto start_backend
)
where npm >nul 2>&1
if errorlevel 1 (
  echo [backend] WARNING: npm not found; PI AI sidecar will not be started.
  goto start_backend
)
where node >nul 2>&1
if errorlevel 1 (
  echo [backend] WARNING: node not found; PI AI sidecar will not be started.
  goto start_backend
)
for /f "delims=" %%v in ('node -e "console.log(process.versions.node.split('.')[0])" 2^>nul') do set "NODE_MAJOR=%%v"
if not defined NODE_MAJOR set "NODE_MAJOR=0"
if !NODE_MAJOR! LSS 20 (
  echo [backend] WARNING: Node.js !NODE_MAJOR! is too old for PI AI sidecar ^(^>= 20 required^).
  goto start_backend
)
if not exist "%TILE_COMPILE_AGENT_SERVICE_DIR%\node_modules" (
  echo [backend] PI AI sidecar: npm install
  call npm --prefix "%TILE_COMPILE_AGENT_SERVICE_DIR%" install
  if errorlevel 1 (
    echo [backend] WARNING: npm install failed; starting without sidecar.
    goto start_backend
  )
)
if not exist "%TILE_COMPILE_AGENT_SERVICE_DIR%\dist\server.js" (
  echo [backend] PI AI sidecar: npm run build
  call npm --prefix "%TILE_COMPILE_AGENT_SERVICE_DIR%" run build
  if errorlevel 1 (
    echo [backend] WARNING: sidecar build failed; starting without sidecar.
    goto start_backend
  )
)
echo [backend] Starting PI AI sidecar from %TILE_COMPILE_AGENT_SERVICE_DIR%
start "PI AI Sidecar" /b cmd /c "npm --prefix "%TILE_COMPILE_AGENT_SERVICE_DIR%" start"
set "AI_AGENT_STARTED=1"

:start_backend
echo [backend] Starting: %BACKEND_BIN%
echo [backend] UI: http://%HOST%:%PORT%/ui/
echo [backend] Runner: %TILE_COMPILE_RUNNER%
echo [backend] CLI:    %TILE_COMPILE_CLI%
echo [backend] Config: %TILE_COMPILE_CONFIG%
echo [backend] Runs:   %TILE_COMPILE_RUNS_DIR%
"%BACKEND_BIN%"
set "BACKEND_EXIT=%errorlevel%"

rem --- Cleanup sidecar ---
if defined AI_AGENT_STARTED (
  echo [backend] Stopping PI AI sidecar
  taskkill /fi "WINDOWTITLE eq PI AI Sidecar*" /f >nul 2>&1
)

exit /b %BACKEND_EXIT%

:usage
echo Usage: %~nx0 [options]
echo.
echo Options:
echo   --host ^<host^>         Backend bind host (default: %HOST%)
echo   --port ^<port^>         Backend port (default: %PORT%)
echo   --build-dir ^<path^>    CMake build directory (default: %BUILD_DIR%)
echo   --cpp-build-dir ^<p^>   C++ core build directory (default: %CPP_BUILD_DIR%)
echo   --backend-bin ^<path^>  Backend binary path (default: %BACKEND_BIN%)
echo   --build-type ^<type^>   CMake build type (default: %BUILD_TYPE%)
echo   --runs-dir ^<path^>     Runs directory (default: %TILE_COMPILE_RUNS_DIR%)
echo   --no-build            Skip cmake configure/build step
echo   -h, --help            Show this help
echo.
echo Env overrides:
echo   HOST, PORT, BUILD_DIR, CPP_BUILD_DIR, BUILD_TYPE, BACKEND_BIN,
echo   TILE_COMPILE_CLI, TILE_COMPILE_RUNNER,
echo   TILE_COMPILE_CONFIG, TILE_COMPILE_SCHEMA,
echo   TILE_COMPILE_PRESETS_DIR, TILE_COMPILE_UI_DIR,
echo   TILE_COMPILE_AGENT_SERVICE_DIR, TILE_COMPILE_RUNS_DIR,
echo   TILE_COMPILE_ALLOWED_ROOTS, TILE_COMPILE_AI_AGENT_AUTOSTART
goto :eof
