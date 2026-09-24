# Resume-Abhängigkeiten

Dieses Dokument beschreibt den implementierten Resume-Vertrag des aktuellen
Single-Method-Runners (`tile_compile_runner resume-reconstruction`).
Legacy-Run-Layouts und Legacy-Cache-Pfade liegen außerhalb dieses
Vertrags.

## Grundregel

Ein Resume ist nur sicher, wenn jeder Input der Zielphase als valides
Artefakt oder Cache vorliegt. `config.yaml` im Run-Verzeichnis ist immer
erforderlich, und `artifacts/run_provenance.json` muss zum Run passen.

Der Runner validiert **fail-closed**: fehlende Artefakte,
Hash-Mismatches oder nicht unterstützte Phasen brechen mit einem Fehler
ab — es gibt keinen stillen Teil-Fallback.

## Unterstützte Einstiegspunkte

`resume-reconstruction --from-phase` akzeptiert zwei Gruppen von Phasen
(Groß-/Kleinschreibung egal; `HMS` ist ein Alias für
`HYPERMETRIC_STRETCH`):

### Rekonstruktions-Resume

| Angeforderte Phase | Mechanismus | Mindest-Abhängigkeiten | Was läuft |
|---|---|---|---|
| `GLOBAL_QUALITY` | Direktes Resume | `config.yaml` byte-identisch zur Run-Start-Config (sha256 in `run_provenance.json`), `registration_sampling.json`, `sampling_geometry.json`, `forward_common_overlap.json`, Geometrie-Dateien + `forward_drizzle_geometry/`-Cache (bei lokalen Warps), `cache/source_quality_maps/`, `source_quality_plan.json`, `cache/normalized_frames/` | GLOBAL_QUALITY, FORWARD_DRIZZLE, MULTIBAND, STACKING-Marker, alle aktivierten Downstream-Phasen |
| `FORWARD_DRIZZLE` | Direktes Resume | alles obige plus `forward_drizzle_checkpoint.json` (Geometrie-Hash + protokollierte Artefaktgrößen müssen stimmen) und konsistenter `forward_drizzle_v2/`-Store | FORWARD_DRIZZLE (ab Checkpoint), MULTIBAND, alle aktivierten Downstream-Phasen |

### Downstream-Resume

Diese Einstiegspunkte nutzen ausschließlich die persistierten
Rekonstruktions-Outputs und lassen die M1–M3-Vorläufer (Sampling-Geometrie,
Forward Drizzle) unangetastet:

| Angeforderte Phase | Mindest-Abhängigkeiten | Änderbare Config-Sektionen | Was läuft |
|---|---|---|---|
| `ASTROMETRY` | `outputs/stacked_rgb.fits` oder `stacked_rgb_solve.fits` | `astrometry`, `bge`, `pcc`, `chroma_denoise`, `hypermetric_stretch`, `runtime_limits` | ASTROMETRY, BGE, PCC, HYPERMETRIC_STRETCH (soweit aktiviert) |
| `BGE` | wie oben | `bge`, `pcc`, `chroma_denoise`, `hypermetric_stretch`, `runtime_limits` | BGE, PCC, HYPERMETRIC_STRETCH |
| `PCC` | wie oben; bei `bge.method != "none"` zusätzlich konsistentes `stacked_rgb_bge_linear.fits` | `pcc`, `chroma_denoise`, `hypermetric_stretch`, `runtime_limits` | PCC, HYPERMETRIC_STRETCH |
| `HYPERMETRIC_STRETCH` | wie oben plus `outputs/stacked_rgb_pcc.fits` (lineares HMS-Input) | `hypermetric_stretch`, `runtime_limits` | HYPERMETRIC_STRETCH |

Für Downstream-Resume wird die Config **abschnittsweise** gegen die
Run-Start-Config verglichen (die Revision aus
`artifacts/config_revisions/`, sonst `config.yaml` bei passendem sha256):
nur die in der Tabelle erlaubten Sektionen dürfen abweichen. Jede andere
Änderung wird mit `FORWARD_STAGE_CONFIG_SCOPE_MISMATCH` abgelehnt — so
kann z.B. der HMS-Editor `hypermetric_stretch.*` ändern, ohne die
persistierten PCC-/Rekonstruktions-Artefakte zu invalidieren.

`MULTIBAND` ist kein Resume-Einstiegspunkt; es läuft immer zusammen mit
FORWARD_DRIZZLE erneut. Jeder andere Phasenname wird mit
`FORWARD_STAGE_UNSUPPORTED_RESUME_PHASE` abgelehnt.

## Cache-Abhängigkeiten

| Cache | Erzeugt von | Benötigt für Resume |
|---|---|---|
| `cache/normalized_frames` | NORMALIZATION (+ NORMALIZED_CACHE-Seal) | Rekonstruktions-Einstiegspunkte (Quell-LRU speist den Gather) |
| `cache/source_quality_maps` | SOURCE_QUALITY_MAPS | Rekonstruktions-Einstiegspunkte (Sample-Gewichte) |
| `artifacts/forward_drizzle_geometry` | SAMPLING_GEOMETRY | Rekonstruktions-Einstiegspunkte, wenn lokale Warp-Modelle existieren; bei rein-affinen Läufen komplett übersprungen |

`reconstruction.delete_source_cache_after_run: true` löscht
`cache/source_quality_maps` und `cache/normalized_frames` am Run-Ende —
ein **Rekonstruktions-Resume** ist dann unmöglich; Downstream-Resume aus
den persistierten Outputs funktioniert weiterhin.
`reconstruction.keep_profile_cache_after_run` steuert die Vorhaltung der
Profil-Caches unabhängig davon.

## Beim Resume beachtete Environment-Overrides

- `TC_FORWARD_DRIZZLE_MEMORY_BUDGET_MB` — überschreibt
  `reconstruction.drizzle.memory_budget_mb`, ohne den Checkpoint zu
  invalidieren (der committed Store ist budget-invariant und auf dem
  CPU-Pfad bit-exakt).

## Unsichere Annahmen

- Eine nicht unterstützte Phase anzufordern degradiert **nicht** zu einem
  Vollrerun; sie schlägt fehl.
- Ein Phase-Event ersetzt kein benötigtes Artefakt — der Checkpoint
  verifiziert Artefakt-Bytegrößen und den Coverage-Geometrie-Hash.
- `STACKING` und `MULTIBAND` sind keine Resume-Einstiegspunkte.
  `ASTROMETRY`, `BGE`, `PCC`, `HYPERMETRIC_STRETCH` sind Downstream-
  Einstiegspunkte: sie benötigen die persistierten Rekonstruktions-Outputs
  und erlauben nur Änderungen in den Downstream-Config-Sektionen.
- Eine `config.yaml`, die außerhalb des erlaubten Scopes von der
  Run-Start-Config abweicht (bzw. deren sha256 bei Rekonstruktions-Resume
  nicht mehr zu `run_provenance.json` passt), bricht das Resume ab.

## Quellen

- `tile_compile_cpp/apps/runner_forward_drizzle.cpp`
  (`run_forward_drizzle_stages`, Resume-Validierung)
- `tile_compile_cpp/apps/runner_pipeline.cpp`
- `web_backend_cpp/include/services/run_inspector.hpp`
  (`RESUME_FROM_PHASES`)
