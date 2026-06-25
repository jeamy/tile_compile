@echo off
setlocal enabledelayedexpansion

rem ============================================================
rem  start_gui3_docker.bat  -  Windows equivalent of start_gui3_docker.sh
rem  Starts the tile_compile Docker container with proper mounts.
rem ============================================================

set "SCRIPT_DIR=%~dp0"
if "%SCRIPT_DIR:~-1%"=="\" set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
set "PROJECT_ROOT=%SCRIPT_DIR%"

set "IMAGE_TAG=tile-compile-web-backend:ubuntu24.04"
set "CONTAINER_NAME=tile-compile-web-backend"
set "HOST_PORT=8080"
set "INPUT_DIR=%PROJECT_ROOT%\tmp\docker-input"
set "RUNS_DIR=%PROJECT_ROOT%\tmp\docker-runs"
set "ENV_FILE=%PROJECT_ROOT%\.env"
set "AGENT_ENV_FILE=%PROJECT_ROOT%\agent_service\.env"
set "NO_AGENT=0"
set "DO_BUILD=1"

rem Resolve .env: check root first, then agent_service
if not exist "%ENV_FILE%" (
  if exist "%AGENT_ENV_FILE%" (
    set "ENV_FILE=%AGENT_ENV_FILE%"
  )
)

:parse_args
if "%~1"=="" goto args_done
if /i "%~1"=="--image-tag" ( set "IMAGE_TAG=%~2" & shift & shift & goto parse_args )
if /i "%~1"=="--name" ( set "CONTAINER_NAME=%~2" & shift & shift & goto parse_args )
if /i "%~1"=="--port" ( set "HOST_PORT=%~2" & shift & shift & goto parse_args )
if /i "%~1"=="--input-dir" ( set "INPUT_DIR=%~2" & shift & shift & goto parse_args )
if /i "%~1"=="--runs-dir" ( set "RUNS_DIR=%~2" & shift & shift & goto parse_args )
if /i "%~1"=="--env-file" ( set "ENV_FILE=%~2" & shift & shift & goto parse_args )
if /i "%~1"=="--no-agent" ( set "NO_AGENT=1" & shift & goto parse_args )
if /i "%~1"=="--no-build" ( set "DO_BUILD=0" & shift & goto parse_args )
if /i "%~1"=="-h" ( call :usage & exit /b 0 )
if /i "%~1"=="--help" ( call :usage & exit /b 0 )
echo [docker] Unknown option: %~1
call :usage
exit /b 1
:args_done

if not exist "%INPUT_DIR%" mkdir "%INPUT_DIR%"
if not exist "%RUNS_DIR%" mkdir "%RUNS_DIR%"

if "%DO_BUILD%"=="1" (
  echo [docker] Building %IMAGE_TAG%
  docker build -t "%IMAGE_TAG%" -f "%PROJECT_ROOT%\docker\ubuntu24.04\Dockerfile" "%PROJECT_ROOT%"
  if errorlevel 1 (
    echo [docker] ERROR: Docker build failed.
    exit /b 1
  )
)

docker ps -a --format "{{.Names}}" | findstr /x "%CONTAINER_NAME%" >nul 2>&1
if not errorlevel 1 (
  echo [docker] Removing existing container %CONTAINER_NAME%
  docker rm -f "%CONTAINER_NAME%" >nul 2>&1
)

set "ALLOWED_ROOTS=/opt/tile_compile:/data/input:/data/runs:/tmp"
set "MOUNT_EXTRA="
set "ENV_FILE_FLAGS="
set "AGENT_FLAGS="

rem Check for .env file(s)
set "ENV_FILE_FLAGS="
if exist "%ENV_FILE%" (
  echo [docker] Mounting .env from %ENV_FILE%
  set "ENV_FILE_FLAGS=-v "%ENV_FILE%:/opt/tile_compile/.env:ro""
) else (
  echo [docker] WARNING: .env not found at %ENV_FILE% - sidecar will run without API keys
)
if exist "%AGENT_ENV_FILE%" if not "%AGENT_ENV_FILE%"=="%ENV_FILE%" (
  echo [docker] Mounting agent_service/.env from %AGENT_ENV_FILE%
  set "ENV_FILE_FLAGS=!ENV_FILE_FLAGS! -v "%AGENT_ENV_FILE%:/opt/tile_compile/agent_service/.env:ro""
)

if "%NO_AGENT%"=="1" (
  set "AGENT_FLAGS=-e TILE_COMPILE_AI_AGENT_AUTOSTART=0"
  echo [docker] PI AI sidecar disabled
)

echo [docker] Starting %CONTAINER_NAME% on http://127.0.0.1:%HOST_PORT%/ui/
docker run -d ^
  --name "%CONTAINER_NAME%" ^
  -p "%HOST_PORT%:8080" ^
  -v "%INPUT_DIR%:/data/input" ^
  -v "%RUNS_DIR%:/data/runs" ^
  !ENV_FILE_FLAGS! ^
  !AGENT_FLAGS! ^
  -e TILE_COMPILE_PROJECT_ROOT="/opt/tile_compile" ^
  -e TILE_COMPILE_HOST="0.0.0.0" ^
  -e TILE_COMPILE_PORT="8080" ^
  -e TILE_COMPILE_ALLOWED_ROOTS="%ALLOWED_ROOTS%" ^
  -e TILE_COMPILE_RUNS_DIR="/data/runs" ^
  -e TILE_COMPILE_UI_DIR="/opt/tile_compile/web_frontend_v3" ^
  -e TILE_COMPILE_CONFIG="/opt/tile_compile/tile_compile_cpp/tile_compile.yaml" ^
  -e TILE_COMPILE_SCHEMA="/opt/tile_compile/tile_compile_cpp/tile_compile.schema.yaml" ^
  -e TILE_COMPILE_PRESETS_DIR="/opt/tile_compile/tile_compile_cpp/examples" ^
  -e TILE_COMPILE_CLI="/opt/tile_compile/tile_compile_cpp/build/tile_compile_cli" ^
  -e TILE_COMPILE_RUNNER="/opt/tile_compile/tile_compile_cpp/build/tile_compile_runner" ^
  "%IMAGE_TAG%" >nul

if errorlevel 1 (
  echo [docker] ERROR: docker run failed.
  exit /b 1
)

echo [docker] Container logs:
docker logs --tail 30 "%CONTAINER_NAME%" 2>&1
echo [docker] Open: http://127.0.0.1:%HOST_PORT%/ui/
exit /b 0

:usage
echo Usage: %~nx0 [options]
echo.
echo Options:
echo   --image-tag ^<tag^>        Docker image tag (default: %IMAGE_TAG%)
echo   --name ^<name^>            Container name (default: %CONTAINER_NAME%)
echo   --port ^<port^>            Host port mapped to container 8080 (default: %HOST_PORT%)
echo   --input-dir ^<path^>       Host input data directory mount (default: %INPUT_DIR%)
echo   --runs-dir ^<path^>        Host runs output directory mount (default: %RUNS_DIR%)
echo   --env-file ^<path^>        Path to .env file with API keys (default: %ENV_FILE%)
echo   --no-agent               Disable PI AI sidecar in container
echo   --no-build               Skip docker build
echo   -h, --help               Show this help
goto :eof
