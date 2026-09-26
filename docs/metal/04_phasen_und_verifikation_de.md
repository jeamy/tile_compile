# 04 - Phasen und Verifikation

Reihenfolge nach Abhängigkeit, nicht nach Dauer. Jede Phase hat
Eingangsbedingung, Umfang, Verifikation und Ausgangskriterium. Phasen mit
"(Mac)" brauchen Apple-Silicon-Hardware und macOS-Toolchain.

**CI-Gate ab MP1:** Jede Phase muss zusätzlich den GitHub-Release-Build
(`release-tile-compile-gui3.yml`, alle vier Plattform-Jobs, insbesondere
`macos-14` Apple Silicon und `macos-15-intel`) grün halten; konkrete
Kriterien je Phase in `06_github_ci_macos_de.md`, Abschnitt 4.

Terminalregel (`AGENTS.md`): jedes Kommando nach `/tmp/out_<zweck>.txt`
umleiten und in getrenntem Schritt lesen. Kein Backend/Sidecar/Lauf ohne
ausdrückliche Anforderung.

## MP0 - Präzisionsmessung und Vertragsfreeze (Linux)

- **Eingang:** keine.
- **Umfang:** `metal/emulation/` (DF-Algebra, Stufen T0/T1/T2, Harness) und
  Messung nach 02, Abschnitt 4 gegen `forward_drizzle_v2_cpu`;
  Entscheidung Scatter-Variante S1/S2/S3; Klärung offener Frage 1 (Resume
  über Backends) durch Lesen von `forward_drizzle_v2_store` und
  Gate-7-Spezifikation.
- **Verifikation:** Gate-4-Sequenzen und 21 Adversarial-Fälle; Kipprate und
  Endbilddifferenz auf mindestens einem OSC-Datensatz mit affinen Frames und
  einem mit lokalen Warps, gemessen aus vorhandenen Artefakten (read-only).
  Ergebnisse als JSONL/JSON im Stil `docs/forward_drizzle_v2_gates/`.
- **Ausgang:** (a) gewählte Stufe T0/T1/T2 + Scatter-Variante,
  (b) eingefrorener Paritätsvertrag Tier A-D mit Zahlen,
  (c) Go/No-Go für FD-v2 auf Metal. Bei No-Go: nur MP1-MP3.

## MP1 - Gemeinsame Abstraktion (Linux, keine Verhaltensänderung)

- **Eingang:** MP0 nicht zwingend, aber Registry-Form hängt an MP0
  (Fähigkeitsabfrage) nur schwach.
- **Umfang:** 01, Abschnitte 3.1-3.4: Typen nach
  `forward_drizzle_v2_types.hpp`; `gpu_backend.hpp` + Registry;
  `AccelerationBackend::metal` (nicht buildbar -> honoured=false + Grund);
  `make_kernel(GpuBackend)`; Fehlerklassen-Alias; neutrale
  Fallback-Felder; Config-Wert `metal` (C++-Struct/Parser/Validierung,
  JSON-/YAML-Schema, Beispielprofile), GUI-Auswahl + `de.json`/`en.json`,
  Doku laut 05.
- **Verifikation:** vollständige Catch2-Suite unverändert grün (CUDA-
  Parity-Tests laufen weiter), neue Tests für Auswahl/Fallback
  (`metal` angefordert, nicht buildbar -> CPU + Grund), Schema-/YAML-
  Validierung, `tile_compile_runner` und `tests` bauen. Kein Diff im
  Verhalten des CUDA-Pfads.
- **Ausgang:** `metal` ist auswählbar, fällt sauber auf CPU zurück.

## MP2 - Metal-Grundgerüst (Mac)

- **Eingang:** MP1.
- **Umfang:** `metal/CMakeLists.txt`, `TILE_COMPILE_ENABLE_METAL`,
  Shader-Build (`metal`/`metallib`), `metal_device.mm`
  (Probe, Familien-Gates, Working-Set-Budget), Buffer-Wrapper,
  `metal_register.cpp`, `build_info`-Eintrag, Test-Target
  `tests_metal` mit `SKIP` ohne Device.
- **Verifikation:** Build auf macOS arm64; Hello-Kernel (Buffer
  Round-Trip, Layout-`static_assert`s, DF-Emulation == Shader); Probe
  meldet Gerät/Familie/Budget; Build auf Linux unverändert (Teilbaum wird
  nicht konfiguriert).
- **Ausgang:** Gerät ansprechbar, Budget bekannt, Struktur-Layouts fixiert.

## MP3 - FP32-Bildoperationen (Mac)

- **Eingang:** MP2. Unabhängig von MP0-Go.
- **Umfang:** `warp_affine.metal` (Mono/RGB/CFA), `box_filter.metal`,
  Anbindung in `AccelerationOps` und `source_quality_map`.
- **Verifikation:** Toleranzvergleich gegen CPU auf Zufalls-/Realbildern
  und gegen `opencv_cuda` (auf NVIDIA-Referenz), Randfälle (`valid_mask`,
  `has_data`, Ränder, Extremwerte), Multithread-Aufruf mit einer Queue je
  Worker. Kein Wirkungsvergleich des Endbilds nötig, solange Toleranzen
  eingehalten sind; Report führt `acceleration.backend = metal`.
- **Ausgang:** Prewarp und Quality-Map laufen auf Metal.

## MP4 - Forward-Drizzle v2 auf Metal (Mac; nur nach MP0-Go)

Teilschritte, jeweils erst nach grüner Parität des vorherigen:

- **MP4a Bausteine + Affine-Scatter:** Polygonfläche, Affine-Leaf-Ecken,
  `scatter_affine` mit Fenster/Tile-Klemmung.
- **MP4b Fold + Finalize (uniform/robust):** `fold_accumulate`,
  `finalize`, Support-Ebenen, Reservoir, Konfidenz; persistenter Workspace,
  `hotpath_allocations==0`, `device_global_synchronizations==0`.
- **MP4c Pieces + Samples:** Piece-Lebenszyklus (Tranche 7), ragged
  Sample-Pfad (Tranche 8) inklusive Quality-Streams (float und packed).
- **MP4d SFR:** `sfr_build/_vote/_reduce`, Konsensus.
- **MP4e Local Warp + Cache:** `scatter_local` (Inversion, Subdivision,
  Diskardierung), `scatter_cached`.
- **MP4f Full-Frame-Estimator:** Pilot + `full_seed`/`full_apply`, Bimodal-
  Veto, Diagnosezähler.
- **Verifikation je Teilschritt:** Metal-Gegenstück des CUDA-Tests
  (03, Abschnitt 4) gegen die CPU-Referenz mit dem eingefrorenen Vertrag;
  Entscheidungs-Kipprate-Bericht; Vergleich mit dem CUDA-Ergebnis derselben
  Eingabe nur als Zusatz (CUDA und Metal dürfen sich innerhalb der
  jeweiligen Toleranz unterscheiden).
- **Ausgang:** alle vom Treiber genutzten `ForwardDrizzleV2Kernel`-Methoden
  auf Metal implementiert und parität-getestet.

## MP5 - Treiberintegration und Fehlerpfade (Mac)

- **Eingang:** MP4.
- **Umfang:** `make_kernel(GpuBackend::metal)`, `backend_used = "metal_v2"`,
  Telemetrie (Gerät, upload/kernel/download-Sekunden, Amplifikation,
  Workspace-Reservierungen), Budget-/Chunking gegen Working-Set,
  Fault-Injection, Device-Loss, Command-Buffer-Fehler, Resume-Verhalten
  nach Klärung offener Frage 1; optional Coverage-Gather für
  SAMPLING_GEOMETRY.
- **Verifikation:** Metal-Fassung von "gate10 cuda fault restarts on cpu"
  (Fehler in Band n -> gesamte Phase auf CPU, nur vollständig validiertes
  CPU-Ergebnis committet); Kill/Resume-Tests (Gate 7) gegen das
  Metal-Backend; Konfig `full_frame_estimator` auf Metal ohne Warnung.
- **Ausgang:** ein synthetischer Ende-zu-Ende-Lauf über
  `run_forward_drizzle_v2` mit Metal committet reproduzierbar innerhalb des
  Vertrags; Fallback bewiesen.

## MP6 - Reale Läufe und Messung (Mac; nur auf ausdrückliche Anforderung)

- **Eingang:** MP5, ausdrückliche Anforderung des Laufs.
- **Umfang:** CPU-v2 vs. Metal-v2 auf derselben Hardware und Eingabe
  (Affin und lokale Warps, `full_frame_estimator` an/aus).
- **Bewertung:** Effektive Config, Phasen-Events, Metriken,
  Validierungsartefakte und Endausgabe-Messung; Auswahlgates (raw/uniform/
  multiband) einzeln benennen, nicht aus FWHM allein schließen. Aussagen
  klar als "aus Artefakten abgeleitet" oder "durch neuen Lauf verifiziert"
  markieren.
- **Performance-Gate (Vorschlag, analog Gate 10):** Metal-v2 strikt
  schneller als CPU-v2 auf derselben Apple-Hardware; sonst `auto` wählt
  Metal nicht (Regel nur nach Messung festlegen).
- **Ausgang:** dokumentierte Messung; `auto`-Regel für Metal.

## MP7 - Release, CI und Dokumentation

- **Umfang:** Workflow `release-tile-compile-gui3.yml` (Apple-Artefakt mit
  Metal-Build), Installations-/Referenzdoku (05), Methodikdokumente
  nur bei geänderter Semantik, Changelog.
- **Verifikation:** macOS-Build im CI, `tests_metal` auf verfügbarem
  Device oder als SKIP (dann Hardwaretests manuell dokumentiert),
  Dokumentenkonsistenz nach `update-param-doc`.

## Abhängigkeitsdiagramm

```
MP0 --------------------------------┐ (Go/No-Go, Vertrag)
MP1 -> MP2 -> MP3 (unabhängig)      │
          └--------> MP4 (a..f) <---┘ -> MP5 -> MP6 -> MP7
```

## Risiken

| Risiko | Wirkung | Gegenmaßnahme |
| --- | --- | --- |
| DF/FP32 verfehlt Vertrag Tier B-D | FD-v2 nicht auf Metal | MP0 vorab; T2 als Rückfall; No-Go zulässig, MP3 bleibt nutzbar |
| Kipprate an Support/Clip-Schwellen | sichtbare Unterschiede | Kipper-Lokalität nachweisen; Vertrag Tier C; keine Lockerung ohne Beleg |
| Float-Atomics nicht in allen Familien | S1 unbrauchbar | Familien-Gate, S2/S3 |
| Unified-Memory-Druck | RAM-Budget des Systems | Working-Set im Budgetplan, Chunking |
| Command-Buffer-Zeitlimits | Abbruch langer Kernel | Teil-Dispatches, Grenzwert zu messen |
| Backend-übergreifendes Resume | inkonsistente Generationen | offene Frage 1 vor MP5 klären |
| Plattformabhängigkeit vorhandener Tests | Fehlzuordnung | 02, Abschnitt 7 |
| Nur Linux-Entwicklung | Hardwarefehler spät | MP0/MP1 auf Linux, Emulation == Shader-Test |
