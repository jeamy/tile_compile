# Tile-Compile

Tile-Compile ist ein Toolkit für **tile-basierte Qualitätsrekonstruktion** astronomischer Bildstapel (Methodik v3).

Given a directory of FITS lights, it:

- optionally **calibrates** lights (bias/dark/flat)
- **registers** frames via a robust 6-stage cascade (star matching, trail detection, feature matching, phase+ECC)
- estimates **global + local (tile) quality metrics**
- reconstructs an improved image based on tile-weighted overlap-add
- optionally clusters frame "states" and generates synthetic frames
- **sigma-clip stacks** the result
- **debayers** OSC data (nearest-neighbor demosaic)
- produces a final stacked output plus **diagnostic artifacts** (JSON)

## Versions

| Version | Directory | Status | Backend |
|---------|-----------|--------|---------|
| **C++** | `tile_compile_cpp/` | **Active** (v3) | C++17 + Eigen + OpenCV + cfitsio + yaml-cpp |
| **Python** | `tile_compile_python/` | Legacy (v4) | Python + NumPy + OpenCV |

Die **C++ Version** ist die aktive Implementierung. Die Python-Version ist als Referenz vorhanden.

## Methodik v3 — Kernfeatures

### Pipeline-Phasen

| Phase | Name | Beschreibung |
|-------|------|-------------|
| 0 | SCAN_INPUT | Frame-Erkennung, FITS-Header, Bayer-Pattern, Linearitätsprüfung |
| 1 | REGISTRATION | 6-stufige Kaskade + CFA-aware Pre-Warping |
| 2 | NORMALIZATION | Hintergrund-basierte Normalisierung (kanalweise) |
| 3 | GLOBAL_METRICS | B/σ/E → gewichtete globale Qualitäts-Scores |
| 4 | TILE_GRID | FWHM-adaptive Tile-Geometrie |
| 5 | LOCAL_METRICS | Stern- und Struktur-basierte Tile-Qualität |
| 6 | TILE_RECONSTRUCTION | Gewichtete Overlap-Add mit Hanning-Fenster |
| 7 | STATE_CLUSTERING | K-Means Clustering + synthetische Frames |
| 8 | STACKING | Sigma-Clip oder Mean-Stacking |
| 9 | DEBAYER | Nearest-Neighbor Demosaic (OSC → RGB) |
| 10 | ASTROMETRY | Plate Solving via ASTAP (WCS-Koordinaten) |
| 11 | PCC | Photometric Color Calibration (Farbkalibrierung) |
| 12 | DONE | Validierung + Report |

Detaillierte Dokumentation: `doc/v3/process_flow/`

### Registrierung — 6-stufige Kaskade

Die Registrierung ist robust gegen schwierige Bedingungen:

| Stufe | Methode | Funktioniert bei |
|-------|---------|-----------------|
| 1 | **Triangle Star Matching** | Punktsterne, ≥6 Sterne, Feldrotation |
| 2 | **Trail Endpoint Detection** | Star Trails (Alt/Az Feldrotation) |
| 3 | **AKAZE Feature Matching** | Allgemeine Bildfeatures |
| 4 | **Robust Phase+ECC** | Nebel/Wolken + große Rotation (Multi-Scale + LoG) |
| 5 | **Hybrid Phase+ECC** | Fallback ohne Sterne |
| 6 | **Identity Fallback** | Letzte Rettung (CC=0, Frame bleibt) |

### Konfiguration

Alle Einstellungen in `tile_compile.yaml`. Schema-Validierung via `tile_compile.schema.json` / `.yaml`.

Referenz: `doc/v3/configuration_reference.md`

## Quickstart (C++ Version)

### Build-Voraussetzungen

- **CMake** ≥ 3.20
- **C++17** Compiler (GCC 11+, Clang 14+)
- **OpenCV** ≥ 4.5
- **Eigen3**
- **cfitsio**
- **yaml-cpp**
- **nlohmann-json**

### Build

```bash
cd tile_compile_cpp
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)
```

### CLI Runner

```bash
./tile_compile_runner \
  --config ../tile_compile.yaml \
  --input-dir /path/to/lights \
  --runs-dir /path/to/runs
```

### CLI Scan (Frame-Erkennung)

```bash
./tile_compile_cli scan \
  --input-dir /path/to/lights \
  --pattern "*.fits"
```

### GUI (Qt6)

```bash
./tile_compile_gui
```

## Quickstart (Python-Version, Legacy)

```bash
cd tile_compile_python
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
./start_gui.sh
```

## GUI Bedienungsanleitung

Die GUI ist in Tabs organisiert:

### 1) Scan

- **Input dir:** Verzeichnis mit FITS Lights
- **Pattern:** `*.fit*` (Standard)
- Klicke **Scan** → zeigt erkannte Frames, Farbmodus (OSC/MONO), Bayer-Pattern

### 2) Kalibrierung (optional)

- **use_bias / use_dark / use_flat** aktivieren
- Master-Frames oder Verzeichnisse auswählen
- Kalibrierung läuft vor der Registrierung

### 3) Konfiguration

- Lade `tile_compile.yaml` oder bearbeite direkt im GUI
- **Validate config** prüft Schema-Konsistenz
- **Wichtige Felder:**
  - `registration.engine`: `triangle_star_matching` (Default)
  - `normalization.mode`: `background` oder `median`
  - `global_metrics.weights`: α=0.4, β=0.3, γ=0.3 (Summe = 1.0)
  - `tile.size_factor`: FWHM-basierte Tile-Größe (Default: 32)
  - `stacking.method`: `rej` (Sigma-Clip) oder `average`

### 4) Run starten

- **Runs dir:** Zielverzeichnis für Outputs (`runs/<run_id>/`)
- Klicke **Start** → Pipeline läuft im Hintergrund
- **Pipeline Progress** zeigt aktuelle Phase + Fortschritt

### 5) Astrometrie (Plate Solving)

Der **Astrometry-Tab** löst die Himmelskoordinaten des gestackten Bildes:

- **FITS-Datei:** Das gestackte RGB-FITS auswählen (`stacked_rgb.fits`)
- **Star Database:** ASTAP-Sterndatenbank herunterladen (D50 empfohlen, ~200 MB)
  - D05 = Tycho2 (hell, schnell)
  - D20 = Gaia DR2 bis Mag 20
  - D50 = Gaia DR2 bis Mag 50 (empfohlen für Deep-Sky)
- **Solve** → ASTAP Plate Solve (blind, Vollhimmel)
- **Ergebnis:** RA/Dec, Pixel-Skala (arcsec/px), Rotation, FOV
- **Save Solved** → speichert die FITS-Datei **als RGB-Cube mit WCS-Headern**
  - Die `_solved.fits` enthält die originalen Farbdaten + WCS-Koordinaten
  - Eine `.wcs`-Datei wird daneben kopiert

> **Hinweis:** ASTAP konvertiert intern zu Mono für das Solving. Die "Save Solved"-Funktion liest das Original-RGB und schreibt es mit den gelösten WCS-Headern neu. Das Ergebnis ist ein RGB-Farbbild mit Koordinaten.

### 6) Photometric Color Calibration (PCC)

Der **PCC-Tab** kalibriert die Farbbalance anhand von Sternkatalog-Photometrie:

- **FITS-Datei:** Die `_solved.fits` aus dem Astrometry-Tab auswählen (RGB + WCS)
- **WCS-Datei:** Wird automatisch erkannt (`_solved.wcs`)
- **Catalog source:** Katalogquelle auswählen (siehe unten)

#### Katalogquellen

| Quelle | Typ | Beschreibung |
|--------|-----|-------------|
| **siril** | Lokal | Siril Gaia DR3 XP-Sampled Katalog (~21 GB, 48 HEALPix-Chunks) |
| **vizier_gaia** | Online | VizieR Gaia DR3 Cone Search (RA, Dec, Gmag, Teff) |
| **vizier_apass** | Online | VizieR APASS DR9 Cone Search (B-V → Teff via Ballesteros 2012) |

##### Siril Gaia DR3 Katalog (empfohlen)

Der **Siril Gaia DR3 XP-Sampled Katalog** liefert die besten PCC-Ergebnisse, da er vollständige XP-Spektren (343 Wellenlängen-Bins, 336–1020 nm) pro Stern enthält. Die Daten werden als HEALPix Level-8 Binärdateien gespeichert:

- **Speicherort:** `~/.local/share/siril/siril_cat1_healpix8_xpsamp/`
- **Download:** Direkt aus der GUI über den Button **"Download Missing Chunks"** — die 48 `.dat.bz2`-Dateien werden von [Zenodo](https://zenodo.org/records/14738271) heruntergeladen und automatisch mit `bzip2` entpackt
- **Format:** Binäres Siril-Katalogformat mit Half-Float (IEEE 754 binary16) XP-Spektren, skalierter RA/Dec/Mag, HEALPix NESTED Indexierung
- **Cone Search:** HEALPix Disc-Query → nur relevante Chunks werden gelesen

> **Hinweis:** Dieser Katalog ist identisch mit dem, den Siril für seine eigene PCC verwendet. Falls Siril bereits installiert ist und den Katalog heruntergeladen hat, erkennt Tile-Compile die vorhandenen Dateien automatisch.

##### VizieR Online-Quellen (schneller Start)

Für schnelle Tests ohne lokalen Katalog-Download:

- **vizier_gaia:** Fragt VizieR Gaia DR3 (`I/355/gaiadr3`) ab — liefert RA, Dec, Gmag und Teff direkt. Nur Sterne mit bekannter Teff werden verwendet.
- **vizier_apass:** Fragt VizieR APASS DR9 (`II/336/apass9`) ab — liefert B- und V-Magnitude. Teff wird über die Ballesteros (2012) Formel aus B-V berechnet.

Beide Online-Quellen sind auf 10.000 Sterne pro Abfrage limitiert.

#### PCC-Einstellungen

| Parameter | Default | Beschreibung |
|-----------|---------|-------------|
| Aperture radius | 8 px | Apertur-Radius für Sternphotometrie |
| Annulus inner | 12 px | Innerer Radius des Himmelsrings |
| Annulus outer | 18 px | Äußerer Radius des Himmelsrings |
| Mag limit (faint) | 14.0 | Schwächster Katalogstern |
| Mag limit (bright) | 6.0 | Hellster Katalogstern (Sättigung vermeiden) |
| Min stars | 10 | Minimum Sterne für zuverlässigen Fit |
| Sigma clip | 2.5 | Sigma-Clipping für Ausreißer |

#### PCC-Algorithmus

PCC berechnet **diagonale Skalierungsfaktoren** (R, G, B) — keine Kanalmischung:

1. Katalogsterne werden per WCS auf Pixelkoordinaten projiziert
2. Aperturphotometrie misst instrumentelle Flüsse (R/G/B) pro Stern (Apertur + Sky-Annulus)
3. Erwartete Sternfarbe wird aus **Teff** berechnet (Planck-Schwarzkörper → lineare sRGB-Konversion). Bei Siril-Katalog alternativ aus XP-Spektren via Filterkurven-Integration.
4. Pro Stern: `correction_R = (cat_R/cat_G) / (inst_R/inst_G)`
5. Sigma-clipped Median → `scale_R`, `scale_B` (Green = Referenz, scale_G = 1.0)
6. Ergebnis: Diagonale Farbkorrekturmatrix `diag(scale_R, 1.0, scale_B)`

#### PCC-Ergebnis

- **Save Corrected** → speichert das farbkalibrierte RGB-FITS am gewählten Pfad
- Zusätzlich werden Einzelkanal-Dateien gespeichert (`_R.fit`, `_G.fit`, `_B.fit`)
- Die Korrekturmatrix und Statistiken werden im Log angezeigt

### 7) Ergebnisse

Nach erfolgreichem Lauf unter `runs/<run_id>/`:

- `outputs/`
  - `registered/` — registrierte Frames (falls `write_registered_frames: true`)
  - `synthetic_*.fit` — synthetische Frames
  - `stacked.fit` — finaler Stack (Mono)
  - `stacked_rgb.fits` — finaler Stack (RGB, nach Debayer)
  - `stacked_rgb_solved.fits` — RGB mit WCS-Headern (nach Astrometrie)
  - `stacked_rgb_pcc.fits` — farbkalibriertes RGB (nach PCC)
- `artifacts/`
  - `global_metrics.json` — globale Qualitätsmetriken
  - `global_registration.json` — Warp-Matrizen + Correlation-Scores
  - `tile_grid.json` — Tile-Geometrie
  - `local_metrics.json` — lokale Tile-Metriken
  - `clustering.json` — Cluster-Zuordnungen
  - `synthetic_frames.json` — synthetische Frame-Info
  - `validation.json` — FWHM-Verbesserung, Tile-Pattern-Check

## Calibration (Bias/Darks/Flats)

- Master-Frames (`bias_master`, `dark_master`, `flat_master`) werden direkt verwendet
- Verzeichnisse (`bias_dir`, `darks_dir`, `flats_dir`) → Master wird automatisch erzeugt
- `dark_auto_select: true` → automatische Dark-Zuordnung nach Belichtungszeit (±5%)

## Projektstruktur

```
tile-compile/
├── tile_compile_cpp/           # C++ Implementierung (aktiv)
│   ├── apps/                   # Runner + CLI
│   ├── include/tile_compile/   # Header (config, core, image, metrics, registration, reconstruction)
│   ├── src/                    # Implementierung
│   ├── gui_cpp/                # Qt6 GUI
│   ├── tests/                  # Unit-Tests
│   ├── tile_compile.yaml       # Default-Konfiguration
│   ├── tile_compile.schema.json
│   └── tile_compile.schema.yaml
├── tile_compile_python/        # Python Implementierung (Legacy)
│   ├── gui/                    # PyQt6 GUI
│   ├── runner/                 # Pipeline-Runner
│   └── tile_compile_backend/   # Backend-Module
├── doc/
│   ├── v3/
│   │   ├── process_flow/       # Pipeline-Phasen-Dokumentation
│   │   └── configuration_reference.md
│   └── ...
├── runs/                       # Run-Outputs
└── README.md
```

## Tests

```bash
cd tile_compile_cpp/build
ctest --output-on-failure
```