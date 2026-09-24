# Process-Flow-Dokumentation — CFA Forward Drizzle + Multiband Pipeline (tile_compile_cpp)

## Überblick

Dieses Dokument beschreibt den **tatsächlichen Ausführungsfluss** der
C++-Implementierung (`tile_compile_cpp/apps/runner_pipeline.cpp` →
`runner_forward_drizzle.cpp`).

Die Pipeline hat genau **eine Rekonstruktionsmethode**: **CFA Forward Drizzle +
Multiband**. Sie verarbeitet **FITS-Frames** (Mono oder OSC/CFA) und
rekonstruiert den Stack, indem jedes kalibrierte Quellsample über die gemessene
Registrierungsgeometrie in ein super-aufgelöstes Ausgaberaster abgebildet wird —
pro CFA-Kanal bei OSC-Daten, ohne Debayering-Schritt. Die entfernten Methoden
AQMH und Classic Tile-Compile existieren im Code nicht mehr; ein Top-Level-
`method:`-Key in einer Config wird fail-closed abgelehnt.

**Implementierung:** C++20 mit Eigen, OpenCV, cfitsio, nlohmann/json, YAML-cpp.

**GUI3-Integration:** Der produktive GUI-Pfad nutzt das Web-Frontend plus das
Crow/C++-Backend. Crow orchestriert die C++-Pipeline über `tile_compile_cli`
und `tile_compile_runner`; die Verarbeitungslogik wird nicht neu implementiert.

Der Resume-Einstiegspunkt (`resume-reconstruction`) und seine Artefakt-/Cache-
Abhängigkeiten sind in [Resume-Abhängigkeiten](resume_dependencies_de.md)
dokumentiert. Resume ist ab `GLOBAL_QUALITY` und `FORWARD_DRIZZLE`
(Rekonstruktion) sowie ab `ASTROMETRY`, `BGE`, `PCC` und
`HYPERMETRIC_STRETCH` (Downstream-Kette aus den persistierten Outputs)
unterstützt.

## Kanonische Phasen (C++-Implementierung)

Von `tile_compile_runner reconstruct` emittierte Phasenfolge (kanonische
Reihenfolge in `web_backend_cpp/include/services/run_inspector.hpp`):

| # | Phase | Kurzbeschreibung |
|---|-------|------------------|
| 1 | `SCAN_INPUT` | FITS-Aufzählung, Header-/Modus-Erkennung (MONO/OSC + Bayer-Pattern), Linearitätsprüfung, Disk-Space-Precheck |
| 2 | `CHANNEL_SPLIT` | Nur Metadaten: Farbmodus, Kanäle und Bayer-Pattern werden festgehalten. Es werden keine Dateien gesplittet — die CFA-Kanalzuordnung erfolgt pro Pixel in FORWARD_DRIZZLE |
| 3 | `NORMALIZATION` | Additiver Hintergrund `B_f` und photometrischer Faktor `P_f` pro Frame (pro CFA-Kanal bei OSC); schreibt `cache/normalized_frames/` und `artifacts/normalization.json` |
| 4 | `REGISTRATION` | Globale Registrierung (kaskadierte Fallbacks) auf Registrierungs-Proxys; affine + optional glatter lokaler Warp pro Frame; schreibt `global_registration.json` und den Plan `registration_sampling.json` |
| 5 | `NORMALIZED_CACHE` | Versiegelt den Normalisierungs-Cache gegen den Sampling-Plan; Voraussetzung für alle Folgephasen |
| 6 | `SAMPLING_GEOMETRY` | Ausgaberaster, Coverage-Masken und der Local-Warp-Geometrie-Cache (`artifacts/forward_drizzle_geometry/`); schreibt `sampling_geometry.json` |
| 7 | `COMMON_OVERLAP` | Pixelweise gemeinsame gültige Abdeckung aller Frames; schreibt `forward_common_overlap.json` und die Analysis-/Reconstruction-Support-Masken |
| 8 | `SOURCE_QUALITY_MAPS` | Pixelweise Quell-Qualitätskarten pro Frame unter `cache/source_quality_maps/`; schreibt `source_quality_plan.json` |
| 9 | `GLOBAL_QUALITY` | Globale Qualitäts-Gate: aggregiert die Quellkarten zu Frame-Gewichten und dem finalen Gewichtsplan; Resume-Einstiegspunkt |
| 10 | `FORWARD_DRIZZLE` | Kern der Rekonstruktion: gechunkter Forward-Drizzle-Gather aller Quellsamples in den gebänderten v2-Store (`artifacts/forward_drizzle_v2/`); schreibt `forward_drizzle.json` + Checkpoint |
| 11 | `MULTIBAND` | Multiband-Rekonstruktion (Nieder-/Hochfrequenz) aus dem Uniform-Store und den Band-Pässen; schreibt `reconstruction_multiband.fits` |
| 12 | `ASTROMETRY` | Optional: Plate Solving / WCS (ASTAP, lokaler Katalog-Fallback) |
| 13 | `BGE` | Optional: Background-Gradient-Extraction vor PCC |
| 14 | `PCC` | Optional: Photometric Color Calibration |
| 15 | `HYPERMETRIC_STRETCH` | Optional: VeraLux HyperMetric Stretch (explizite nichtlineare Streckung, läuft zuletzt) |

`STACKING` wird vor den Downstream-Phasen als Pass-Through-Marker emittiert;
es gibt keine klassische Stack-Phase — der Drizzle-Store *ist* der Stack.

## Ablaufdiagramm

```
┌─────────────────────────────────────────────────────────────┐
│               INPUT: MONO / OSC RAW FITS FRAMES             │
└────────────────────────────┬────────────────────────────────┘
                             │
              ┌──────────────▼───────────────┐
              │  SCAN_INPUT                  │
              │  • FITS-Dims + Header        │
              │  • MONO/OSC + Bayer          │
              │  • Linearität + Disk-Check   │
              └──────────────┬───────────────┘
                             │
              ┌──────────────▼───────────────┐
              │  CHANNEL_SPLIT (Metadaten)   │
              │  • Modus + Bayer-Pattern     │
              │  • deferred zum Drizzle      │
              └──────────────┬───────────────┘
                             │
              ┌──────────────▼───────────────┐
              │  NORMALIZATION               │
              │  • B_f / P_f pro Frame       │
              │  • pro CFA-Kanal (OSC)       │
              │  • cache/normalized_frames/  │
              └──────────────┬───────────────┘
                             │
              ┌──────────────▼───────────────┐
              │  REGISTRATION                │
              │  • kaskadierte Ausrichtung   │
              │  • affine + lokaler Warp     │
              │  • registration_sampling.json│
              └──────────────┬───────────────┘
                             │
              ┌──────────────▼───────────────┐
              │  NORMALIZED_CACHE            │
              │  • Cache-Seal gg. Plan       │
              └──────────────┬───────────────┘
                             │
              ┌──────────────▼───────────────┐
              │  SAMPLING_GEOMETRY           │
              │  • Ausgaberaster + Coverage  │
              │  • Local-Warp-Geom.-Cache    │
              └──────────────┬───────────────┘
                             │
              ┌──────────────▼───────────────┐
              │  COMMON_OVERLAP              │
              │  • Common-Valid-Masken       │
              └──────────────┬───────────────┘
                             │
              ┌──────────────▼───────────────┐
              │  SOURCE_QUALITY_MAPS         │
              │  • pixelweise Quellqualität  │
              │  • cache/source_quality_maps │
              └──────────────┬───────────────┘
                             │
              ┌──────────────▼───────────────┐
              │  GLOBAL_QUALITY              │
              │  • Frame-Gewichte + Gate     │
              │  • Resume-Einstieg           │
              └──────────────┬───────────────┘
                             │
              ┌──────────────▼───────────────┐
              │  FORWARD_DRIZZLE             │
              │  • gechunkter CFA-Gather     │
              │  • gebänderter v2-Store      │
              │  • Resume-Einstieg           │
              └──────────────┬───────────────┘
                             │
              ┌──────────────▼───────────────┐
              │  MULTIBAND                   │
              │  • Band-Fusion               │
              │  • reconstruction_*.fits     │
              └──────────────┬───────────────┘
                             │
              ┌──────────────▼───────────────┐
              │  ASTROMETRY → BGE → PCC →    │
              │  HYPERMETRIC_STRETCH         │
              │  (jeweils optional)          │
              └──────────────┬───────────────┘
                             │
              ┌──────────────▼───────────────┐
              │  OUTPUTS:                    │
              │  • rekonstruierter Stack     │
              │  • artifacts/*.json          │
              │  • logs/run_events.jsonl     │
              └──────────────────────────────┘
```

## Kernprinzipien

1. **Single Method:** Es gibt keinen Methoden-Selektor. Configs mit `method:`
   oder `reconstruction.engine:` werden fail-closed abgelehnt; entfernte
   Strukturblöcke (`aqmh`, `pipeline`, `tile`, `local_metrics`, `synthetic`, …)
   werden mit Migrations-Warnung gestrippt.
2. **Kein Debayering:** Bei OSC-Input wird jeder Quellpixel im Drizzle-Gather
   pro Pixel seinem CFA-Kanal zugeordnet (`cfa_channel_for_source_pixel` mit
   `cfa_origin`-Offsets aus dem Sampling-Plan). Die R/G/B-Ausgabeebenen
   entstehen direkt.
3. **Forward-Mapping:** Quellsamples werden ins Ausgaberaster gesplattet
   (subpixel-genau, `pixfrac` steuert die Drop-Größe), statt Ausgabepixel
   invers abzubilden — das ermöglicht die CFA-sichere Superauflösung.
4. **Begrenzter Speicher:** FORWARD_DRIZZLE läuft in Zeilen-Chunks mit
   konfigurierbarem `reconstruction.drizzle.memory_budget_mb`; der gebänderte
   Store wird inkrementell geschrieben, die Input-Amplifikation bleibt niedrig.
5. **Linearität:** Alle Phasen bis einschließlich PCC bleiben linear;
   HYPERMETRIC_STRETCH ist der explizite finale nichtlineare Schritt.
6. **Resume-Vertrag:** `resume-reconstruction --from-phase` unterstützt nur
   `GLOBAL_QUALITY` und `FORWARD_DRIZZLE`; alle Vorgänger-Artefakte plus
   `run_provenance.json` (Config-sha256) müssen validieren.

## Dokumentstruktur

| Datei | Inhalt |
|-------|--------|
| [phase_0_overview.md](phase_0_overview.md) | Phasentabelle, Artefakt-Map, Konfigurationsoberfläche |
| [phase_1_scan_normalization.md](phase_1_scan_normalization.md) | SCAN_INPUT, CHANNEL_SPLIT, NORMALIZATION |
| [phase_2_registration_geometry.md](phase_2_registration_geometry.md) | REGISTRATION, NORMALIZED_CACHE, SAMPLING_GEOMETRY, COMMON_OVERLAP |
| [phase_3_quality.md](phase_3_quality.md) | SOURCE_QUALITY_MAPS, GLOBAL_QUALITY |
| [phase_4_forward_drizzle.md](phase_4_forward_drizzle.md) | FORWARD_DRIZZLE Gather, Chunking, Store, Checkpoint |
| [phase_5_multiband.md](phase_5_multiband.md) | MULTIBAND Band-Fusion |
| [phase_6_postprocessing.md](phase_6_postprocessing.md) | ASTROMETRY, BGE, PCC, HYPERMETRIC_STRETCH |
| [data_flow_user_description_en.md](data_flow_user_description_en.md) | Datenzentrierte Beschreibung (EN) |
| [data_flow_user_description_de.md](data_flow_user_description_de.md) | Datenzentrierte Beschreibung (DE) |
| [resume_dependencies_en.md](resume_dependencies_en.md) | Resume-Vertrag (EN) |
| [resume_dependencies_de.md](resume_dependencies_de.md) | Resume-Vertrag (DE) |
| [flow_analysis.md](flow_analysis.md) | Implementierungsnotizen |

### Normative Spezifikation

- `/docs/forward_drizzle_v2_zielarchitektur_2026-09-12_de.md`
- `/docs/AQMH/aqmh_cfa_forward_drizzle_multiband_implementierungsplan_de.md`
  (der Implementierungsplan, dem der Single-Method-Cutover folgt)

### C++-Implementierung

- `/tile_compile_cpp/apps/runner_pipeline.cpp` — Scan + frühe Phasen
- `/tile_compile_cpp/apps/runner_phase_metrics.cpp` — CHANNEL_SPLIT, NORMALIZATION
- `/tile_compile_cpp/apps/runner_phase_registration.cpp` — REGISTRATION
- `/tile_compile_cpp/apps/runner_forward_drizzle.cpp` — NORMALIZED_CACHE … MULTIBAND + Downstream-Orchestrierung
- `/tile_compile_cpp/apps/runner_downstream.cpp` — ASTROMETRY, BGE, PCC, HMS
