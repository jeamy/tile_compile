# Konfiguration

Die Pipeline wird über `tile_compile.yaml` konfiguriert.

## Wichtige Abschnitte

- `input` — Quellverzeichnis, Frame-Limits
- `registration` — Ausrichtungs-Engine, Anker-Strategie, Astrometrie
- `method` / `aqmh` — AQMH-Qualitätskarten, Rekonstruktion, Storage, Cherry-Pick
- `tile` — Klassische Tile-Geometrie
- `stacking` — Gewichtung, Rejection, Ausgabeformat
- `background_extraction` — BGE-Parameter
- `photometric_color_calibration` — PCC-Katalog, Referenz
- `runtime_limits` — Speicher, Zeit, Worker-Limits, Acceleration-Backend

## GPU-Backend

```yaml
runtime_limits:
  acceleration_backend: auto  # auto | opencv_cuda | opencv_opencl | cpu
  parallel_workers: 8
  memory_budget: 2048
```

`auto` bevorzugt CUDA, dann OpenCL, dann CPU. Die Backend-Unterstützung ist
phasenspezifisch: PREWARP, AQMH-Maps, klassische Tile-Rekonstruktion,
synthetische Rekonstruktion und STACKING unterstützen CUDA/OpenCL; Streaming-
AQMH-Rekonstruktion unterstützt derzeit CUDA; REGISTRATION bleibt CPU-only.
AQMH Cherry-Pick verwendet bewusst CPU. Die effektive Wahl wird in
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
- `full_mode.example.yaml` — Hohe Qualität, äquatorial
- `emergency_mode.example.yaml` — Minimaler Laufzeitmodus
- `smart_telescope_dwarf_seestar.example.yaml` — Smart Telescope
- `canon_equatorial_balanced.example.yaml` — Ausgewogene DSLR

## Parameter Studio (GUI3)

Das Web-Frontend bietet einen geführten Parameter-Editor mit:

- Szenario-Presets (Alt/Az, Rotation, helle Sterne, wenige Frames, Gradient)
- Situations-Assistent — automatische Parametervorschläge
- Echtzeit-Validierung gegen das Schema
