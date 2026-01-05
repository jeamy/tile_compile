# Tile Compile: Astronomical Image Reconstruction Toolkit

## Prerequisites
- Python 3.8+
- Poetry (dependency management)

## Prerequisites
- Python 3.8+
- `venv` module (usually comes with Python)
- pip

## Installation Steps

1. Clone the repository:
```bash
git clone https://github.com/yourusername/tile_compile.git
cd tile_compile
```

2. Set up virtual environment and install dependencies:
```bash
# Make setup script executable
chmod +x setup_venv.sh

# Run setup script
./setup_venv.sh
```

3. Activate virtual environment:
```bash
# On Unix/macOS
source .venv/bin/activate

# On Windows
.venv\Scripts\activate
```

## Development Workflow

### Activate Environment
```bash
source .venv/bin/activate
```

### Run Tests
```bash
pytest tests/
```

### Run Validation
```bash
python -m validation.dataset_generator
python -m validation.performance_benchmark
python -m validation.comparative_analysis
```

### Deactivate Environment
```bash
deactivate
```

## Troubleshooting
- Ensure Python 3.8+ is installed
- Check that all system dependencies are met
- Refer to requirements.txt for package details

## Running Tests

```bash
poetry run pytest
```

## Running the Application

```bash
poetry run tile-compile run --config config.yaml --input-dir /path/to/input
```

## Project Structure
- `tile_compile_backend/`: Core backend modules
- `tests/`: Test suite
- `siril_scripts/`: Siril processing scripts
- `docs/`: Documentation

## Methodik v3 Compliance
This toolkit implements the Tile-based Quality Reconstruction Methodology version 3.