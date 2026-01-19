# Tile-Compile C++ Portierungsplan - Übersicht

## Ziel

Portierung der Backend-Berechnungen und des Runners von Python nach C++ für:
- **Höhere Performance** bei rechenintensiven Bildverarbeitungsoperationen
- **Bessere Speichereffizienz** bei großen FITS-Dateien
- **Einfachere Deployment** ohne Python-Abhängigkeiten

## Architektur-Entscheidung

### GUI bleibt in Python
Die PyQt6-GUI (`gui/`) bleibt unverändert in Python, da:
1. Sie nur als Orchestrator fungiert und Python-Scripts/Executables aufruft
2. PyQt6 bietet exzellente GUI-Entwicklung
3. Die GUI ist nicht performance-kritisch
4. Einfache Integration mit C++ Backend über:
   - **Subprocess-Aufrufe** (wie bisher mit Python-Scripts)
   - **Shared Libraries** (optional, via ctypes/cffi)

### C++ Backend-Architektur

```
tile_compile_cpp/
├── CMakeLists.txt
├── include/
│   ├── tile_compile/
│   │   ├── core/           # Grundlegende Typen und Utilities
│   │   ├── io/             # FITS I/O, Config-Parsing
│   │   ├── image/          # Bildverarbeitung
│   │   ├── registration/   # Registrierung (ECC, Phase Correlation)
│   │   ├── metrics/        # Qualitätsmetriken
│   │   ├── clustering/     # K-Means, Quantile-Clustering
│   │   ├── reconstruction/ # Tile-basierte Rekonstruktion
│   │   ├── synthetic/      # Synthetische Frame-Generierung
│   │   └── pipeline/       # Pipeline-Orchestrierung
├── src/
│   └── [entsprechende .cpp Dateien]
├── apps/
│   ├── tile_compile_runner.cpp  # CLI-Hauptprogramm
│   └── tile_compile_cli.cpp     # Backend-CLI
└── tests/
    └── [Unit-Tests]
```

## Abhängigkeiten

### Erforderliche C++ Bibliotheken

| Python-Bibliothek | C++ Äquivalent | Zweck |
|-------------------|----------------|-------|
| NumPy | Eigen3 | Lineare Algebra, Array-Operationen |
| OpenCV | OpenCV (C++) | Bildverarbeitung, ECC, Warping |
| Astropy (FITS) | CFITSIO | FITS-Datei I/O |
| scikit-learn | mlpack / Eigen | K-Means Clustering |
| PyWavelets | wavelib / eigen-wavelets | Wavelet-Transformation |
| SciPy | Eigen + custom | Statistische Funktionen |
| YAML | yaml-cpp | Konfigurationsdateien |
| JSON | nlohmann/json | Event-Logging, Artefakte |

### Build-System
- **CMake** (≥3.16) als Build-System
- **vcpkg** oder **Conan** für Dependency-Management
- **C++17** Standard (für std::filesystem, structured bindings, etc.)

## Portierungsphasen

1. **Phase 1**: Core-Infrastruktur (2-3 Wochen)
2. **Phase 2**: I/O und Konfiguration (1-2 Wochen)
3. **Phase 3**: Bildverarbeitung (2-3 Wochen)
4. **Phase 4**: Registrierung (2-3 Wochen)
5. **Phase 5**: Metriken und Clustering (2-3 Wochen)
6. **Phase 6**: Rekonstruktion und Synthetic (2-3 Wochen)
7. **Phase 7**: Pipeline-Integration (2-3 Wochen)
8. **Phase 8**: Testing und Validierung (2-3 Wochen)

**Geschätzte Gesamtdauer**: 16-22 Wochen

## Dateien zur Portierung

### runner/ (12 Dateien, ~230 KB)
- `phases_impl.py` (196 KB) - Hauptpipeline-Implementierung
- `image_processing.py` (14 KB) - CFA/Bayer, Normalisierung
- `opencv_registration.py` (5 KB) - ECC, Phase Correlation
- `calibration.py` (3 KB) - Bias/Dark/Flat-Kalibrierung
- `siril_utils.py` - ENTFÄLLT (Registrierung nativ mit OpenCV)
- `utils.py` (3 KB) - Hilfsfunktionen
- `fits_utils.py` (1 KB) - FITS-Utilities
- `events.py` (2 KB) - Event-Logging
- `phases.py` (2 KB) - Phase-Definitionen
- `assumptions.py` (2 KB) - Annahmen-Konfiguration

### tile_compile_backend/ (17 Dateien, ~100 KB)
- `metrics.py` (15 KB) - Qualitätsmetriken, Wiener-Filter
- `validate.py` (14 KB) - Konfigurationsvalidierung
- `clustering.py` (11 KB) - K-Means, State-Clustering
- `synthetic.py` (12 KB) - Synthetische Frames
- `reconstruction.py` (9 KB) - Tile-Rekonstruktion
- `tile_grid.py` (9 KB) - Tile-Grid-Generierung
- `registration.py` (8 KB) - Registrierungs-Backend
- `sigma_clipping.py` (7 KB) - Sigma-Clipping
- `linearity.py` (7 KB) - Linearitätsvalidierung
- `configuration.py` (7 KB) - Konfigurationsklassen
- `scan.py` (8 KB) - Frame-Scanning
- `runs.py` (3 KB) - Run-Management
- `logs.py` (3 KB) - Logging
- `status.py` (2 KB) - Status-Tracking
- `artifacts.py` (1 KB) - Artefakt-Management
- `config_io.py` (0.3 KB) - Config I/O
- `schema.py` (0.4 KB) - Schema-Loading

### Einstiegspunkt
- `tile_compile_runner.py` (326 Zeilen) - CLI-Hauptprogramm

## Nächste Schritte

Siehe detaillierte Dokumentation in:
- `01_module_mapping.md` - Detaillierte Modul-Zuordnung
- `02_dependencies.md` - Abhängigkeiten und Bibliotheken
- `03_phase1_core.md` - Phase 1: Core-Infrastruktur
- `04_phase2_io.md` - Phase 2: I/O und Konfiguration
- `05_phase3_image.md` - Phase 3: Bildverarbeitung
- `06_phase4_registration.md` - Phase 4: Registrierung
- `07_phase5_metrics.md` - Phase 5: Metriken und Clustering
- `08_phase6_reconstruction.md` - Phase 6: Rekonstruktion
- `09_phase7_pipeline.md` - Phase 7: Pipeline-Integration
- `10_testing.md` - Testing-Strategie
