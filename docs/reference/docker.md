# Docker

The Docker image bundles the C++ backend, runner, CLI, web frontend (v3), and the PI AI sidecar (Node.js 22 LTS).
The entrypoint script (`docker/ubuntu24.04/entrypoint.sh`) starts the sidecar and backend automatically.

## Quick start

**Linux / macOS:**
```bash
./start_gui3_docker.sh
```
Open: http://127.0.0.1:8080/ui/

**Windows:**
```cmd
start_gui3_docker.bat
```

## Input data and run output

Host directories are mounted into the container:

| Mount | Host (default) | Container path | Purpose |
|-------|----------------|----------------|---------|
| Input | `./tmp/docker-input` | `/data/input` | FITS light frames |
| Runs | `./tmp/docker-runs` | `/data/runs` | Run output, artifacts, logs |
| Extra | `--extra-root <path>` | `/data/extra` | Additional allowed root |

Place your FITS files in `./tmp/docker-input/` (or specify a custom path with `--input-dir`).
In the GUI, enter `/data/input` as the input directory and `/data/runs` as the runs directory.

```bash
# Custom input directory
./start_gui3_docker.sh --input-dir /path/to/my/fits --runs-dir /path/to/runs
```

## Options

```bash
./start_gui3_docker.sh --help
```

| Option | Description | Default |
|--------|-------------|---------|
| `--image-tag <tag>` | Docker image tag | `tile-compile-web-backend:ubuntu24.04` |
| `--name <name>` | Container name | `tile-compile-web-backend` |
| `--port <port>` | Host port → container 8080 | `8080` |
| `--input-dir <path>` | Host input data mount | `./tmp/docker-input` |
| `--runs-dir <path>` | Host runs output mount | `./tmp/docker-runs` |
| `--env-file <path>` | `.env` file with API keys (mounted read-only) | `./.env` |
| `--no-agent` | Disable PI AI sidecar in container | (enabled by default) |
| `--no-build` | Skip `docker build` | (build by default) |
| `--extra-root <path>` | Additional allowed root at `/data/extra` | — |

## `.env` file

The `.env` file provides API keys, Docker mount paths, and configuration for the PI AI sidecar.
Copy `.env.example` to `.env` in the project root and fill in your settings:

```bash
cp .env.example .env
# Edit .env – set INPUT_DIR, RUNS_DIR, AI_SCAN_ENABLED, model, API keys, etc.
```

The file is mounted read-only into the container at `/opt/tile_compile/.env`.
The sidecar loads it automatically via `dotenv`.

Docker-relevant settings in `.env`:

| Variable | Description | Default if unset |
|----------|-------------|------------------|
| `INPUT_DIR` | Host directory with FITS light frames (mounted to `/data/input`) | `./tmp/docker-input` |
| `RUNS_DIR` | Host directory for run output (mounted to `/data/runs`) | `./tmp/docker-runs` |
| `ASTAP_DATA_DIR` | Host directory with ASTAP binary and star catalog (mounted to `/data/astap`) | `./tmp/docker-astap` |
| `SIRIL_CATALOG_DIR` | Host directory with Siril/PCC Gaia catalog (mounted to `/data/siril`) | `./tmp/docker-siril` |
| `HOST_PORT` | Host port mapped to container 8080 | `8080` |
| `IMAGE_TAG` | Docker image tag | `tile-compile-web-backend:ubuntu24.04` |
| `CONTAINER_NAME` | Container name | `tile-compile-web-backend` |
| `EXTRA_ALLOWED_ROOTS` | Additional host path mounted at `/data/extra` | — |

CLI arguments (`--input-dir`, `--runs-dir`, etc.) override `.env` values.

## Docker Compose (alternative)

Instead of the start scripts, you can use `docker compose`:

```bash
# Run from project root – --env-file is required for volume interpolation
docker compose --env-file .env -f docker/ubuntu24.04/docker-compose.yml up -d --build
docker compose --env-file .env -f docker/ubuntu24.04/docker-compose.yml logs -f
docker compose --env-file .env -f docker/ubuntu24.04/docker-compose.yml down
```

`--env-file .env` enables variable interpolation (`INPUT_DIR`, `RUNS_DIR`, `HOST_PORT`) from the root `.env`.
The `env_file` directive in the compose file additionally injects API keys into the container.

## GUI paths inside the container

| GUI field | Container path | Source |
|-----------|----------------|--------|
| Input directory | `/data/input` | Mount of `INPUT_DIR` (host) |
| Runs directory | `/data/runs` | Auto-filled from backend env (`TILE_COMPILE_RUNS_DIR`) |

The runs directory is auto-populated in the GUI. The input directory must be entered manually (`/data/input`).

## Container architecture

```
entrypoint.sh
  ├── PI AI sidecar (node dist/server.js)  →  127.0.0.1:3001
  └── C++ backend (tile_compile_web_backend)  →  0.0.0.0:8080
        └── connects to sidecar at http://127.0.0.1:3001
```

## Advanced: CLI-only Docker (legacy)

For running just the C++ runner inside a container (without backend/UI), the legacy script is still available:

```bash
./tile_compile_cpp/scripts/docker_compile_and_run.sh build-image
./tile_compile_cpp/scripts/docker_compile_and_run.sh run-app -- run \
  --config /mnt/config/tile_compile.yaml \
  --input-dir /mnt/input \
  --runs-dir /workspace/tile_compile_cpp/runs
```
