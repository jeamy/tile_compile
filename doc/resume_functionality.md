# Resume-Funktionalität für abgebrochene Runs

## Übersicht

Abgebrochene oder fehlgeschlagene Runs können ab einer bestimmten Phase neu gestartet werden. Alle Phasen vor der angegebenen Phase werden übersprungen, alle Phasen ab der angegebenen Phase werden neu berechnet.

## CLI-Nutzung

### Direkt über Runner

```bash
python3 tile_compile_runner.py resume \
  --run-dir /path/to/runs/20260108_220541_bf8ccabf \
  --from-phase 10
```

### Über Backend-CLI (für GUI)

```bash
python3 tile_compile_backend_cli.py resume-run \
  /path/to/runs/20260108_220541_bf8ccabf \
  --from-phase 10
```

## Parameter

- `--run-dir`: Pfad zum bestehenden Run-Verzeichnis
- `--from-phase`: Phase-Nummer, ab der neu gestartet werden soll (0-11)
- `--project-root`: (Optional) Projekt-Root-Verzeichnis

## Phasen-Übersicht

| Phase | Name | Beschreibung |
|-------|------|--------------|
| 0 | SCAN_INPUT | Input-Frames scannen |
| 1 | REGISTRATION | Frame-Registrierung |
| 2 | CHANNEL_SPLIT | RGB/CFA-Kanäle trennen |
| 3 | NORMALIZATION | Normalisierung |
| 4 | GLOBAL_METRICS | Globale Metriken berechnen |
| 5 | TILE_GRID | Tile-Grid generieren |
| 6 | LOCAL_METRICS | Lokale Metriken berechnen |
| 7 | TILE_RECONSTRUCTION | Tile-Rekonstruktion |
| 8 | STATE_CLUSTERING | State-Clustering |
| 9 | SYNTHETIC_FRAMES | Synthetische Frames generieren |
| 10 | STACKING | Finales Stacking |

## Verhalten

### Übersprungene Phasen

Phasen mit `phase_id < from_phase` werden übersprungen:
- `phase_start` Event wird emittiert
- `phase_end` Event mit `status: "skipped"` wird emittiert
- Keine Berechnung findet statt

### Ausgeführte Phasen

Phasen mit `phase_id >= from_phase` werden normal ausgeführt:
- Alle Berechnungen werden neu durchgeführt
- Bestehende Outputs werden überschrieben
- Normale Event-Logs werden geschrieben

## Voraussetzungen

Für einen erfolgreichen Resume müssen folgende Artefakte vorhanden sein:

1. **Run-Verzeichnis** mit:
   - `config.yaml` (Original-Konfiguration)
   - `logs/run_events.jsonl` (wird erweitert)
   - `outputs/registered/` (registrierte Frames aus Phase 1)
   - `work/channels/` (Channel-Splits aus Phase 2, falls Phase >= 2)

2. **Abhängigkeiten**:
   - Phase 2-11 benötigen registrierte Frames aus Phase 1
   - Phase 3-11 benötigen Channel-Splits aus Phase 2
   - Phase 4-11 benötigen normalisierte Frames aus Phase 3

## Log-Events

### Resume-Start

```json
{
  "type": "run_resume",
  "run_id": "20260108_220541_bf8ccabf",
  "from_phase": 10,
  "ts": "2026-01-09T05:30:00.000000+00:00",
  "siril_exe": "/usr/bin/siril-cli",
  "siril_source": "system"
}
```

### Übersprungene Phase

```json
{
  "type": "phase_end",
  "phase": 9,
  "phase_name": "SYNTHETIC_FRAMES",
  "status": "skipped",
  "reason": "resume_from_phase",
  "resume_from": 10,
  "ts": "2026-01-09T05:30:00.100000+00:00"
}
```

## GUI-Integration

Die GUI kann die Resume-Funktionalität wie folgt nutzen:

1. **Run-Status abrufen**:
   ```bash
   python3 tile_compile_backend_cli.py get-run-status /path/to/run
   ```

2. **Phase anklicken** → Button "Restart" anzeigen

3. **Resume ausführen**:
   ```bash
   python3 tile_compile_backend_cli.py resume-run /path/to/run --from-phase <N>
   ```

4. **Fortschritt überwachen**:
   - `logs/run_events.jsonl` lesen
   - `phase_start`, `phase_progress`, `phase_end` Events verarbeiten

## Beispiel-Workflow

### Szenario: Phase 10 (STACKING) ist fehlgeschlagen

```bash
# 1. Run-Status prüfen
python3 tile_compile_backend_cli.py get-run-status runs/20260108_220541_bf8ccabf
# → phase: 10, status: "FAILED"

# 2. Ab Phase 10 neu starten
python3 tile_compile_runner.py resume \
  --run-dir runs/20260108_220541_bf8ccabf \
  --from-phase 10

# 3. Ergebnis prüfen
ls -lh runs/20260108_220541_bf8ccabf/outputs/stacked.fit
```

## Implementierungsdetails

### `tile_compile_runner.py`

- Neuer Subcommand: `resume`
- Funktion `resume_run(args)` lädt Config aus Run-Dir und startet Pipeline

### `runner/phases_impl.py`

- Neuer Parameter: `resume_from_phase: Optional[int] = None`
- Hilfsfunktion: `should_skip_phase(phase_num: int) -> bool`
- Alle Phasen 0-10 prüfen Skip-Bedingung

### `tile_compile_backend_cli.py`

- Neuer Command: `resume-run`
- Funktion `cmd_resume_run(args)` ruft `tile_compile_runner.py resume` auf

## Fehlerbehandlung

- **Run-Dir nicht gefunden**: Exit-Code 1, JSON-Error
- **Config fehlt**: Exit-Code 1, JSON-Error
- **Registered-Frames fehlen**: Exit-Code 1, JSON-Error
- **Phase-Nummer ungültig**: Wird von argparse validiert (0-11)

## Performance

Resume ist **deutlich schneller** als kompletter Neustart:

| Szenario | Normale Laufzeit | Resume ab Phase 10 |
|----------|------------------|-------------------|
| 422 Frames | ~45 Minuten | ~3 Sekunden |

Grund: Phasen 0-9 werden übersprungen, nur Phase 10 (Stacking) wird ausgeführt.
