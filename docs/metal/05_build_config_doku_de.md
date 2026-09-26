# 05 - Build, Konfiguration, GUI, Doku, CI

## 1. CMake

Gemeinsames `tile_compile_cpp/CMakeLists.txt` erhält nur:

```cmake
option(TILE_COMPILE_ENABLE_METAL "Apple Metal GPU backend" ON)   # nur wirksam bei APPLE
set(TILE_COMPILE_WITH_METAL OFF)
if(APPLE AND TILE_COMPILE_ENABLE_METAL AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/metal/CMakeLists.txt")
    add_subdirectory(metal)            # definiert Target tile_compile_metal
    set(TILE_COMPILE_WITH_METAL ON)
endif()
target_compile_definitions(tile_compile_lib PUBLIC TILE_COMPILE_WITH_METAL=$<BOOL:${TILE_COMPILE_WITH_METAL}>)
if(TILE_COMPILE_WITH_METAL)
    target_link_libraries(tile_compile_lib PUBLIC tile_compile_metal)
endif()
```

- `metal/CMakeLists.txt`: `enable_language(OBJCXX)`, Frameworks `Metal`,
  `Foundation` (ggf. `MetalPerformanceShaders`), Shader-Kompilierung
  (`xcrun -sdk macosx metal -c ... ; xcrun metallib`), Ergebnis als
  generiertes Byte-Array **in die Binärdatei eingebettet** (keine
  Beilagedatei, siehe 06). Fehlt der Metal-Compiler, wird der
  Shader-Quelltext eingebettet und zur Laufzeit übersetzt.
- `TILE_COMPILE_ENABLE_METAL` ist nur bei `APPLE` **und** arm64 wirksam
  (Default ON); auf Intel-macOS aus.
- Shader-Optionen: keine Fast-Math (`-fno-fast-math`,
  `MTLMathModeSafe`), `-std=metal3.0` oder höher nach Feature-Bedarf
  (Float-Atomics; zu verifizieren).
- `metal/emulation` wird plattformunabhängig als Test-Target gebaut
  (`BUILD_TESTS=ON`), damit MP0 und die DF-Konsistenzprüfung auf Linux
  laufen; der Device-Teil nur unter APPLE.
- Referenzquellen-Liste ("Plan 19.6", `-ffp-contract=off`) auf
  AppleClang prüfen und per Konfigurationstest absichern.
- `build_info` (`src/core/build_info.cpp`, Hash-Zeile `cuda=...|openmp=...`):
  `metal=<ON|OFF>` ergänzen, damit Reports das Backend erkennen.
- OpenMP auf macOS (`libomp`) unverändert; nicht Teil dieses Plans.

## 2. Konfiguration

`acceleration_backend` (heute `auto`; `runtime_limits.acceleration_backend`):
Wert `metal` ergänzen. Überall konsistent halten (AGENTS.md):

- C++-Konfigurationsstruct (`configuration.hpp`, Default unverändert `auto`),
  Parser-Serialisierung und Validierung (`src/io/config.cpp`),
- `tile_compile_cpp/tile_compile.schema.json` und `.schema.yaml`
  (Enum-Erweiterung), `tile_compile.yaml` nur falls der Wert dort explizit
  steht,
- Tests (`test_acceleration_backend.cpp`, Config-Tests), Beispielprofile
  unter `tile_compile_cpp/examples/` und deren `README.md`.

Semantik (für die Parameterdoku): `auto` = `cuda > metal > opencl > cpu`
je Phase, nur wenn buildbar und Gerät vorhanden; `metal` erzwingt Metal,
ohne verfügbares Backend Fallback auf CPU mit `fallback_reason` (kein
Fehler); Wirkung auf FORWARD_DRIZZLE (v2-Kernel) und Prewarp/Quality-Map.
Kein Methodenselektor, keine neuen Phasenartefakte.

Nach Änderungen JSON/YAML validieren (Parser + Schema-Test).

## 3. GUI und Report

- `web_frontend_v3`: Backend-Auswahl um `metal` erweitern (bestehende
  Komponenten wiederverwenden, keine Inline-Stile);
  `i18n/de.json` **und** `en.json`; falls Report-Beschriftungen betroffen:
  `report_de.json`/`report_en.json` synchron.
- Run-Monitor liest `backend_used` (`metal_v2`) und `gpu`-Telemetrie aus den
  Run-Artefakten (Zustand nach Reload aus Artefakten, nicht nur aus dem
  Browser).
- Verifikation Desktop und Mobil mit statischen Fixtures oder einem
  bereits laufenden Dienst; **kein** Start von Backend/Sidecar durch diese
  Arbeit.
- `web_frontend/` (legacy) unverändert.

## 4. Zu aktualisierende Dokumente

Gemäß `.devin/skills/update-param-doc/SKILL.md`:

- `docs/configuration_reference.md`, `configuration_reference_en.md`
  (Wert `metal`: Einheiten n/a, Bereich, Default `auto`, Wechselwirkungen,
  Fallback, Nicht-Anwendbarkeit auf Nicht-Mac),
- `docs/configuration_examples_practical_de.md` / `_en.md`,
- `tile_compile_cpp/examples/` + `README.md`,
- `docs/getting_started/installation.md` / `installation_de.md` (macOS:
  Xcode Command Line Tools, Metal-Gerät, Homebrew-libomp),
- `docs/reference/build.md` (Option `TILE_COMPILE_ENABLE_METAL`),
- `docs/process_flow/phase_4_forward_drizzle.md` (Backend Metal, Fallback,
  Vertragsklasse),
- Methodikdokumente nur, wenn Semantik/Invarianten geändert wurden
  (Metal-Toleranzvertrag als Anhang, nicht als Methodikänderung).

Attic-Dokumente und explizit versionierte ältere Methodiken bleiben
unverändert.

## 5. CI und Release

Ausführlich in `06_github_ci_macos_de.md`. Kurzfassung: Shader werden in
die Binärdateien eingebettet (Offline-`metallib` oder Laufzeitkompilierung
als Rückfall), Metal nur bei arm64, Intel-Job unverändert, Hardwaretests
ohne Device als `SKIPPED (no Metal device)`.

## 6. Abschlusscheckliste je Phase (aus AGENTS.md)

- Code, Schemas, Beispiele und Doku stimmen überein.
- Syntax-/Formatprüfungen bestehen; relevante Tests laufen; benötigte
  Targets (`tile_compile_runner`, `tests`, `tests_metal` unter macOS) bauen.
- Kein Backend und kein Lauf ohne ausdrückliche Anforderung gestartet.
- Endbericht: geändertes Verhalten, Verifikation, Restrisiko;
  Umgebungs- und Codefehler getrennt.
