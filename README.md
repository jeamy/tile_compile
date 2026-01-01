# tile_compile

Pipeline für **tile-basierte Qualitätsrekonstruktion** von DSO-Serien (OSC, **linear**) mit:

- **Registrierung** (derzeit über **Siril CLI** als Backend)
- **globale Frame-Metriken** (Transparenz/Rauschen/Struktur)
- **lokale Tile-Gewichtung** (Sterne oder Struktur)
- Rekonstruktion von **15–30 synthetischen Frames**
- **finales Stacking** der synthetischen Frames (derzeit über **Siril CLI**, `average`)

Die Methodik ist in `tile_basierte_qualitatsrekonstruktion_methodik.md` beschrieben.

## Grundannahmen

- Daten sind **linear** (kein Stretch / Asinh / Log)
- Registrierung ist vollständig (Translation + Rotation)
- Viele Frames (Default-Gate: `frames_min: 800`)

## Konfiguration & Pipeline-Steuerung

- **`tile_compile.yaml`**
  - Zentrale Parameter (Geometry/Weights/Validation/Runtime)
  - Auswahl der Backends für Registrierung/Stacking
- **`tile_compile.proc`**
  - Prozessbeschreibung (Phasen 1–7)
  - Definiert die Reihenfolge der Schritte und welche Verzeichnisse verwendet werden

## Verzeichnislayout (konzeptionell)

- **Input frames**: beliebiges Input-Verzeichnis (projektabhängig)
- **Registrierte Frames**: `registration.output_dir` (Default: `registered/`)
- **Synthetische Frames**: `stacking.input_dir` (Default: `synthetic/`)
- **Finaler Stack**: `stacking.output_file` (Default: `stacked.fit`)

## Registrierung (modulare API)

Die Registrierung ist als austauschbarer Schritt modelliert:

- Konfig:
  - `registration.engine`: `siril | relative | wcs_anchor`
  - `registration.reference`: `auto | frame_index | path`
  - `registration.output_dir`: z. B. `registered`
  - `registration.registered_filename_pattern`: z. B. `"[abcde]reg_{index:05d}.fit"`

In `tile_compile.proc` wird danach auf registrierte Frames umgeschaltet:

- `LOAD_FRAMES_FROM_DIR registration.output_dir`

## Synthetische Frames

Die Rekonstruktion erzeugt **15–30** synthetische Frames (siehe Methodik-Datei für die formale Definition/Clustering).

In `tile_compile.proc`:

- `ENSURE_DIR stacking.input_dir`
- `WRITE stacking.input_dir/syn_XX.fits`

## Finales Stacking (modulare API)

Auch das Stacking ist als austauschbarer Schritt modelliert (derzeit Siril):

- Konfig:
  - `stacking.engine`: `siril`
  - `stacking.method`: `average`
  - `stacking.input_dir`: `synthetic`
  - `stacking.input_pattern`: `syn_*.fits`
  - `stacking.output_file`: `stacked.fit`

In `tile_compile.proc`:

- `CALL stack_frames(engine=stacking.engine, method=stacking.method, ...)`

## Validierung

Die Pipeline enthält Abbruchregeln/Validierung (z. B. minimale FWHM-Verbesserung, keine Hintergrundverschlechterung, keine Tile-Artefakte). Details siehe `tile_basierte_qualitatsrekonstruktion_methodik.md`.
