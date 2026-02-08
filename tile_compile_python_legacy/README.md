# Tile-Compile Python Version

Dies ist die Python-Implementierung des Tile-Compile Backends.

## Voraussetzungen

- Python 3.9+
- PySide6 (für GUI)
- NumPy, OpenCV, Astropy, scikit-learn, PyYAML

## Installation

```bash
cd tile_compile_python

# Virtual Environment erstellen
python3 -m venv .venv
source .venv/bin/activate

# Abhängigkeiten installieren
pip install -r requirements.txt
```

## GUI starten

```bash
./start_gui.sh
```

## CLI verwenden

```bash
# Pipeline ausführen
python3 tile_compile_runner.py run \
    --config tile_compile.yaml \
    --input-dir /pfad/zu/frames \
    --runs-dir runs

# Run fortsetzen
python3 tile_compile_runner.py resume \
    --run-dir runs/20240119_123456_abc12345 \
    --from-phase 5
```

## Struktur

```
tile_compile_python/
├── gui/                    # PyQt6 GUI
├── runner/                 # Pipeline-Runner Module
├── tile_compile_backend/   # Backend-Berechnungen
├── tests/                  # Unit-Tests
├── validation/             # Validierungs-Tools
├── siril_scripts/          # Siril-Skripte (optional)
├── tile_compile_runner.py  # CLI-Hauptprogramm
├── tile_compile_backend_cli.py  # Backend-CLI
└── requirements.txt        # Python-Abhängigkeiten
```

## Tests

```bash
./run_tests.sh
```
