# Pipeline-Übersicht — CFA Forward Drizzle + Multiband

> **C++-Implementierung:** `runner_pipeline.cpp`, `runner_forward_drizzle.cpp`,
> `runner_downstream.cpp`
> **Methode:** genau eine — `CFA Forward Drizzle + Multiband` (kein Selektor)

## Übersicht

tile_compile besitzt seit dem Single-Method-Cutover genau eine
Rekonstruktionsmethode. Es gibt keinen `method:`-Konfigurationsschlüssel mehr:

- Top-Level `method:` und `reconstruction.engine:` werden beim Laden
  **fail-closed** abgelehnt (`UNKNOWN_LEGACY_KEY`).
- Entfernte Strukturblöcke (`aqmh`, `pipeline`, `assumptions`, `tile`,
  `tile_denoise`, `local_metrics`, `synthetic`, `validation`) sowie die
  entfernten `stacking.*`/`runtime_limits.*`-Subkeys werden beim Laden
  **mit Warnung gestrippt** (`migrate-config` schreibt die bereinigte Datei).
- `stacking.common_overlap_required_fraction` wird automatisch nach
  `reconstruction.common_overlap_required_fraction` umbenannt.

## Phasenfolge

Emittierte Phasen eines `reconstruct`-Laufs (kanonisch in
`web_backend_cpp/include/services/run_inspector.hpp`):

| Reihenfolge | Phase | Emitting Code | Zweck |
|-------------|-------|---------------|-------|
| 1 | `SCAN_INPUT` | `runner_pipeline.cpp` | FITS-Enumeration, Modus-/Bayer-Erkennung, Linearität, Disk-Precheck |
| 2 | `CHANNEL_SPLIT` | `runner_phase_metrics.cpp` | Metadaten (Modus, Kanäle, Bayer-Pattern) — keine Datei-Splits |
| 3 | `NORMALIZATION` | `runner_phase_metrics.cpp` | `B_f`/`P_f` pro Frame (pro CFA-Kanal), Normalized-Frame-Cache |
| 4 | `REGISTRATION` | `runner_phase_registration.cpp` | Globale Ausrichtung, Sampling-Plan |
| 5 | `NORMALIZED_CACHE` | `runner_forward_drizzle.cpp` | Seal des Caches gegen den Plan |
| 6 | `SAMPLING_GEOMETRY` | `runner_forward_drizzle.cpp` | Ausgaberaster, Coverage, Geometrie-Cache |
| 7 | `COMMON_OVERLAP` | `runner_forward_drizzle.cpp` | Gemeinsame gültige Abdeckung |
| 8 | `SOURCE_QUALITY_MAPS` | `runner_forward_drizzle.cpp` | Pixelweise Quell-Qualitätskarten |
| 9 | `GLOBAL_QUALITY` | `runner_forward_drizzle.cpp` | Frame-Gewichte + Gate (Resume-Einstieg) |
| 10 | `FORWARD_DRIZZLE` | `runner_forward_drizzle.cpp` | Gechunkter CFA-Gather in den v2-Store (Resume-Einstieg) |
| 11 | `MULTIBAND` | `runner_forward_drizzle.cpp` | Band-Fusion → `reconstruction_multiband.fits` |
| 12 | `ASTROMETRY` | `runner_downstream.cpp` | Optional: Plate Solving |
| 13 | `BGE` | `runner_downstream.cpp` | Optional: Hintergrund-Extraktion |
| 14 | `PCC` | `runner_downstream.cpp` | Optional: Photometric Color Calibration |
| 15 | `HYPERMETRIC_STRETCH` | `runner_downstream.cpp` | Optional: finale Streckung |

`STACKING` wird als Pass-Through-Marker vor den Downstream-Phasen emittiert;
`PREWARP`-Events gehören intern zur REGISTRATION (kein eigener Eintrag in der
kanonischen UI-Reihenfolge).

## Zentrale Artefakte

| Datei | Phase | Inhalt |
|-------|-------|--------|
| `artifacts/normalization.json` | NORMALIZATION | Modus, Bayer, `B_*`/`P_*` pro Frame |
| `artifacts/global_metrics.json` | NORMALIZATION | Globale Frame-Metriken/Gewichte |
| `artifacts/global_registration.json` | REGISTRATION | Warp-Matrizen, CC, Dithering-Diagnostik |
| `artifacts/registration_sampling.json` | REGISTRATION | Sampling-Plan (affine + lokale Warps, CFA-Origin) |
| `cache/normalized_frames/` | NORMALIZED_CACHE | Versiegelte normalisierte Frames |
| `artifacts/sampling_geometry.json` | SAMPLING_GEOMETRY | Ausgaberaster, Coverage-Statistik |
| `artifacts/forward_drizzle_geometry/` | SAMPLING_GEOMETRY | Local-Warp-Geometrie-Cache (`.leaves`, `manifest.json`) |
| `artifacts/forward_common_overlap.json` | COMMON_OVERLAP | Common-Overlap-Statistik |
| `cache/source_quality_maps/` | SOURCE_QUALITY_MAPS | Persistente Quell-Qualitätskarten |
| `artifacts/source_quality_plan.json` | SOURCE_QUALITY_MAPS/GLOBAL_QUALITY | Gewichtsplan |
| `artifacts/forward_drizzle_v2/` | FORWARD_DRIZZLE | Gebänderter Uniform-Store (wx/w/w² pro Kanal/Band) |
| `artifacts/forward_drizzle_checkpoint.json` | FORWARD_DRIZZLE | Resume-Checkpoint (Geometrie-Hash, Artefaktgrößen) |
| `artifacts/forward_drizzle.json` | FORWARD_DRIZZLE | Lauf-Zusammenfassung |
| `artifacts/forward_drizzle_geometry_profile.json` | FORWARD_DRIZZLE | Geometrie-Instrumentierung |
| `artifacts/reconstruction_multiband.fits` | MULTIBAND | Fusioniertes Ergebnis |
| `artifacts/run_provenance.json` | (Run-Start) | Config-sha256 etc. — Resume-Vergleichsanker |
| `config.yaml` | (Run-Start) | Eingefrorene Lauf-Konfiguration |
| `outputs/` | Downstream | Finale FITS-Ausgaben |
| `logs/run_events.jsonl` | alle | Event-Stream (JSONL) |

## Konfigurationsoberfläche

Die `reconstruction.*`-Sektion steuert den Rekonstruktionskern
(42 Leaf-Parameter, siehe `tile_compile.schema.yaml` /
`docs/configuration_reference.md`):

- `reconstruction.drizzle.*` — `pixfrac`, `kernel`, `internal_scale`,
  `min_clip_contributors`, Chunking (`chunk_rows`, `chunk_halo_rows`),
  `memory_budget_mb`
- `reconstruction.clipping.*` — `clip_sigma_low/high`, `min_n_eff`,
  `robust_passes`, `guard_fallback`
- `reconstruction.coverage_gate.*` — Coverage-Mindestanforderungen
- `reconstruction.multiband.*` — Bandaufteilung und Fusion
- `reconstruction.quality.pyramid.*` — Quell-Qualitätskarten-Pyramide
- `reconstruction.diagnostics.*` — Diagnose-Umfang, Uniform-Store-Persistenz,
  Preview
- `reconstruction.common_overlap_required_fraction` — Overlap-Mindestanteil
- `reconstruction.delete_source_cache_after_run`,
  `reconstruction.keep_profile_cache_after_run` — Cache-Lebenszyklus

## Verzeichnisstruktur eines Laufs

```
runs/<run_id>/
├── config.yaml                  # eingefrorene Run-Konfiguration
├── logs/run_events.jsonl        # Event-Stream
├── artifacts/
│   ├── run_provenance.json      # Resume-Anker (config sha256)
│   ├── normalization.json
│   ├── global_metrics.json
│   ├── global_registration.json
│   ├── registration_sampling.json
│   ├── sampling_geometry.json
│   ├── forward_common_overlap.json
│   ├── forward_drizzle_geometry/   # Local-Warp-Geometrie-Cache
│   ├── source_quality_plan.json
│   ├── forward_drizzle_v2/         # gebänderter Uniform-Store
│   ├── forward_drizzle_checkpoint.json
│   ├── forward_drizzle.json
│   ├── reconstruction_multiband.fits
│   └── report.html                 # via POST /api/runs/<id>/stats
├── cache/
│   ├── normalized_frames/
│   └── source_quality_maps/
└── outputs/                        # finale FITS
```
