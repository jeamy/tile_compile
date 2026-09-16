# Konfiguration

Die Pipeline wird über `tile_compile.yaml` konfiguriert.

## Wichtige Abschnitte

- `data` — Eingabehandling, Frame-Limits
- `normalization` — Hintergrund-/Skalierungsschätzung
- `registration` — Ausrichtungs-Engine, Anker-Strategie, Prewarp-Verfeinerungen
- `dithering` — Dithering-Diagnose-Gate
- `global_metrics` — globale Frame-Gewichtung
- `reconstruction` — der Rekonstruktionskern: `drizzle.*` (Kernel, Pixfrac,
  Chunking, Speicherbudget), `clipping.*`, `coverage_gate.*`, `multiband.*`,
  `quality.pyramid.*`, `diagnostics.*`
- `astrometry` — Plate Solving
- `bge` — Hintergrund-Gradienten-Extraktion
- `pcc` — Photometric Color Calibration
- `hypermetric_stretch` — finale nichtlineare Streckung
- `stacking` — Per-Frame-Kosmetik-Korrektur
- `runtime_limits` — Speicher, Zeit, Worker-Limits, Acceleration-Backend

Es gibt keinen `method`-Abschnitt — CFA Forward Drizzle + Multiband ist die
einzige Methode. Ein Top-Level-`method:`-Key wird fail-closed abgelehnt;
Legacy-Blöcke (`aqmh`, `pipeline`, `tile`, `local_metrics`, `synthetic`, …)
werden mit Migrations-Warnung gestrippt. `tile_compile_cli migrate-config
<in> <out>` schreibt eine bereinigte Datei.

## GPU-Backend

```yaml
runtime_limits:
  acceleration_backend: auto  # auto | opencv_cuda | opencv_opencl | cpu
  parallel_workers: 8
  memory_budget: 2048
```

`auto` bevorzugt CUDA, dann OpenCL, dann CPU. FORWARD_DRIZZLE hat
CUDA-Beschleunigung für die Geometrie-/Gather-Hotpaths; der CPU-Pfad ist
die Bit-Exaktheits-Referenz und bleibt immer als Fallback verfügbar.
REGISTRATION bleibt CPU-only. Die effektive Wahl wird in
`artifacts/acceleration_context.json` geschrieben und in Live-Fortschritts-
Logs angezeigt.

## Schema

Validieren mit:

```bash
./tile_compile_cli validate-config --path tile_compile.yaml
```

Schema abrufen:

```bash
./tile_compile_cli get-schema
```

## Beispiele

Siehe `tile_compile_cpp/examples/` für szenariospezifische Konfigurationen:

- `m104.example.yaml` — Alt/Az, starke Rotation, schlechtes Seeing
- `large_n.example.yaml` / `medium_n.example.yaml` / `small_n.example.yaml` — Framezahl-Stufen
- `mono.example.yaml` — Mono-Workflow
- `smart_telescope_dwarf_seestar.example.yaml` — Smart Telescope
- `canon_equatorial_balanced.example.yaml` — Ausgewogene DSLR
- `reconstruction_tuning.example.yaml` — reconstruction.*-Tuning-Referenz

## Parameter Studio (GUI3)

Das Web-Frontend bietet einen geführten Parameter-Editor mit:

- Szenario-Presets (Alt/Az, Rotation, helle Sterne, wenige Frames, Gradient)
- Situations-Assistent — automatische Parametervorschläge
- Echtzeit-Validierung gegen das Schema
