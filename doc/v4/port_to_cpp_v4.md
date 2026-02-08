# Portierung: Python/Qt -> C++/Qt (Methodik v4)

## Ziel

- **GUI**: von Python/Qt (aktuelles `tile_compile_python/gui/`) auf **C++/Qt6 (Widgets)** umstellen.
- **Backend**: v4-Methodik (Runner + Algorithmen) vollständig nach C++ portieren.
- **Repo-Struktur**: klare Trennung von Python-Referenz und C++-Port (Parallelbetrieb).
- **Scope**: **Full Port** für v4 (GUI + Backend/Runner/Algorithmen) – Python bleibt temporär als Referenz.

## Nicht-Ziele / Annahmen

- Keine funktionalen Änderungen/Refactorings außerhalb der Portierung.
- Keine API-/Schema-Änderungen ohne separaten Design-Doc.
- Keine lokalen Installationsschritte ohne explizite Aufforderung.
- **Methodik v4** ist die einzige Ziel-Methodik (kein v3/v2 Support im Port).

## Ausgangslage (v4)

Relevante Python-Module (Referenz-Implementation):

- **Runner/Orchestrierung**
  - `tile_compile_python/tile_compile_runner.py`
  - `tile_compile_python/runner/tile_processor_v4.py`
  - `tile_compile_python/runner/tile_local_registration_v4.py`
  - `tile_compile_python/runner/v4_parallel.py`
- **Backend/Kern**
  - `tile_compile_python/tile_compile_backend/` (IO, Pipeline, Stacking, Diagnostics)
  - `tile_compile_python/runner/adaptive_tile_grid.py` (adaptive Tiles, Probe)
- **Konfiguration & Methodik**
  - `tile_compile_python/tile_compile.yaml`
  - `doc/configuration_reference_v4.md`
  - `doc/tile_based_quality_reconstruction_methodology_v4.md`
  - `doc/v_4_tests_and_diagnostics.md`
  - `doc/v_4_parallel_tile_reconstruction_production.md`

## Ziel-Repo-Struktur (Vorschlag)

Da bereits `tile_compile_cpp/` existiert, wird diese Struktur als Ziel angenommen und ergänzt:

```
/ (repo root)
  tile_compile_python/
    gui/
    runner/
    tile_compile_backend/
    tests/
    scripts/

  tile_compile_cpp/
    CMakeLists.txt
    include/
      tile_compile/
    src/
      core/
      image/
      clustering/
      io/
      runner/
      gui/
    apps/
      cli_main.cpp
      runner_main.cpp
    tests/

  doc/
    port_to_cpp_v4.md
```

## Architektur-Entscheidung: Was wird portiert?

Festgelegt:

1. **Qt**: **Qt6**
2. **UI**: **Widgets**
3. **Scope**: **Full Port** (GUI + Backend/Runner/Algorithmen v4)

## Übergangsstrategie (Python als Referenz)

- Python bleibt als **Regression-Orakel** (Golden Master) während der Portierung.
- Jede C++-Etappe wird gegen Python-Ausgabe verglichen (Toleranzbasiert, Pixel-Statistiken).
- Fallback-Strategie: CLI-Interop (C++ ruft Python subprocess) ist **nur optional** und nur temporär.

## Migrationsphasen (Portierungsplan v4)

### Phase 0: Analyse & Stabilisierung (v4-baseline)

**Ziele:** V4 in Python stabilisieren, genaue Schnittstellen definieren.

- **Pipeline-Inventar**
  - Phasenmodell der v4-Pipeline dokumentieren (vgl. `v4_parallel.py`).
  - Datenflüsse & Artefakte: Eingabe-Frames, Zwischenartefakte, Endprodukte.
- **Konfigurationsvertrag**
  - `tile_compile.yaml` und `doc/configuration_reference_v4.md` als Source of Truth.
  - Mapping „Konfig-Key -> C++ Modul/Owner“ definieren.
- **Artefakt-Matrix**
  - Liste aller v4-Diagnoseoutputs (z.B. warp fields, tile maps, histograms).
- **Testdaten-Baseline**
  - Mind. 2–3 kleine Datensets (M31/M45 etc.) als Regression-Set.

**Artefakte**:
- `doc/port_to_cpp_v4.md` (dieses Dokument)
- optional: `doc/cpp_backend_interface_v4.md`

### Phase 1: Repo-Readiness (ohne funktionale Änderungen)

- Verzeichnisstruktur prüfen/vereinheitlichen (Python vs. C++).
- Einheitliche Namensräume und Include-Konventionen für C++ definieren.
- Build-Baseline: CMake + Qt6 + OpenCV + FITS + FFT/Linalg (soweit erforderlich).

### Phase 2: C++ Core Skeleton (v4-fähig)

**Ziele:** Minimaler C++ Core mit klaren Module-Interfaces.

- `tile_compile_cpp/src/core/`:
  - Config Loader (YAML -> struct)
  - Logging/Progress Event API
  - Fehlercodes/Result-Typen
- `tile_compile_cpp/src/io/`:
  - FITS I/O (Read/Write, memmap optional)
- `tile_compile_cpp/apps/`:
  - `cli_main.cpp` (dummy flow, nur config validation)

### Phase 3: Algorithmus-Portierung v4 (Kern zuerst)

**Empfohlene Reihenfolge & Mapping:**

1. **Config & Validation**
   - YAML Loader, Schema-Checks (strict) – Basis für alle Stufen.
2. **Normalization & Global Metrics**
   - `global_coarse_normalize()`
   - `compute_global_weights()`
3. **Tile Grid / Adaptive Tiles**
   - `build_initial_tile_grid()`
   - Adaptive refinement (hierarchisch + variance-based)
4. **Tile-Local Registration**
   - ECC-Reg (cv2) → C++ (OpenCV) Port
   - Warp-Variance & Consistency Checks
5. **TileProcessor v4**
   - Iterative Refinement, temporal smoothing
   - Gewichtete Rekonstruktion (per tile)
6. **Overlap-Add & Reconstruction**
   - Windowing + variance weighting
7. **Clustering & Synthetic Frames**
   - Clustering und Synthese gemäß v4-Methodik
8. **Final Stacking & Debayer**
   - Stacking Methoden + optional Debayer

**Qualitätssicherung:**
- Jede Stufe gegen Python referenzieren (Metriken: MSE, PSNR, Δhistogramm).
- Exportiere Debug-Artefakte (warp maps, tile invalid map) identisch.

### Phase 4: Runner/Orchestrierung in C++

- Pipeline-Phasenmodell (State Machine) nachbilden.
- Parallelisierung gemäß `v4_parallel.py`.
- Globales Progress- und Cancellation-System.
- Fehlerpropagation und Retry-Strategien.

### Phase 5: CLI/Headless Betrieb

- Vollständige CLI mit identischen Flags/Config-Pfad wie Python.
- Ausgabe-Pfade und Artefaktstruktur 1:1.
- Regression-Tests im CI (CLI = primärer Testtreiber).

### Phase 6: GUI-Portierung (Qt6 Widgets)

- UI-Flow aus `tile_compile_python/gui/` nach C++ übertragen.
- Job-Status, Progress, Cancel, Logs (UI ↔ Backend Event Bus).
- Validierung/Fehlermeldungen konsistent mit Python-GUI.

### Phase 7: Parallelbetrieb & Umschalten

- Python-GUI bleibt parallel lauffähig (Fallback).
- C++ wird Default, sobald Feature-Parität erreicht ist.
- Regression-Suite läuft in CI gegen beide Implementierungen.

### Phase 8: Python-Deprecation (optional)

- Python behalten als Referenz (Debug/Regression) oder auslagern.
- Cleanup der Python-Dependencies, sobald C++ stabil ist.

## Modul-Mapping (Python -> C++)

| Python | C++ Zielmodul | Funktion |
|--------|---------------|----------|
| `runner/tile_processor_v4.py` | `src/runner/tile_processor_v4.cpp` | Tile-Processing v4, iterative refinement |
| `runner/tile_local_registration_v4.py` | `src/runner/tile_local_registration_v4.cpp` | ECC Registration, warp stats |
| `runner/adaptive_tile_grid.py` | `src/runner/adaptive_tile_grid.cpp` | Adaptive Tiles, probe | 
| `tile_compile_backend/` | `src/core/` + `src/image/` | Core pipeline, IO, stacking, diagnostics |
| `runner/v4_parallel.py` | `src/runner/v4_parallel.cpp` | Parallel tile processing |

## Build/Dependencies (C++)

- **Qt6 Widgets** (GUI)
- **OpenCV** (ECC, warp)
- **CFITSIO / Astropy-Äquivalent** (FITS I/O)
- **Eigen** (linear algebra)
- Optional: **OpenMP/TBB** (Parallelisierung)

## Tests & Regression

- **Unit-Tests**: pro Modul (config, registration, overlap-add)
- **Integration**: minimaler v4-Pipeline-Run (small dataset)
- **Golden Master**: Python-Ergebnisse als Referenz

## Risiken & Stolpersteine

- **Numerische Abweichungen** (OpenCV/FFT/Linalg Unterschiede)
- **Threading/GUI** (UI darf nicht blockieren)
- **FITS I/O** (BZERO/BSCALE, memmap handling)
- **Performance** (Tile-wise parallelization, memory limits)

## Empfohlene Commit-Strategie

1. Struktur & CMake-Skeleton
2. Config/IO
3. Module in Scheiben (Tile-Grid → Registration → TileProcessor → Reconstruction)
4. Runner + CLI
5. GUI

## Nächste Schritte (konkret)

1. Phase 0 Inventar & v4-Pipeline-Dokumentation abschließen.
2. C++ Skeleton in `tile_compile_cpp/` erweitern.
3. Config Loader + IO implementieren.
4. Tile-Grid & Registration portieren.
5. Runner/Parallelisierung + CLI.
6. GUI Port + Umschalten.
