# AQMH_RECONSTRUCTION / TILE_RECONSTRUCTION — Rekonstruktion

> **Aktueller Code:** `runner_phase_aqmh_reconstruction.cpp`, `runner_pipeline.cpp`
> **AQMH-Phase:** `Phase::AQMH_RECONSTRUCTION` (Enum 21)
> **Classic-Phase:** `Phase::TILE_RECONSTRUCTION` (Enum 9)

## Methodische Verzweigung

```text
AQMH:    AQMH_MAPS -> AQMH_GLOBAL_QUALITY -> AQMH_RECONSTRUCTION
Classic: LOCAL_METRICS -> TILE_RECONSTRUCTION
```

Die Phase `TILE_RECONSTRUCTION` wird bei AQMH nicht emittiert. AQMH ist eine
pixelweise Rekonstruktion und verwendet keine Classic-Tile-Gewichte.

## AQMH_RECONSTRUCTION

Die Phase rekonstruiert jeden gültigen Canvas-Pixel aus den vorgewarp­ten Frames
und den Quality-Maps. Die globalen Frame-Gewichte aus
`AQMH_GLOBAL_QUALITY` werden mit den pixelweisen Map-Werten kombiniert.

Der aktuelle Ablauf ist:

1. Quality-Map- und Frame-Cache validieren.
2. Optional den Registrierungs-Gewichtsschutz auf die globalen AQMH-Gewichte
   anwenden.
3. Pixelweise gewichtete Rekonstruktion mit Support-/Gültigkeitsmasken und
   deterministischem Sigma-Clipping ausführen.
4. Optional Cherry-Pick nur verwenden, wenn es explizit aktiviert und das
   Mindest-Kriterium erfüllt ist.
5. Uniform Control ausschließlich als Validierungsreferenz berechnen.
6. Nachbearbeitungskandidaten gegen Uniform Control und Raw AQMH validieren.
   Besteht ein Kandidat nicht, bleibt Raw AQMH einschließlich Gewichtssumme
   erhalten. Uniform Control darf Raw AQMH nicht ersetzen oder abschwächen.
7. Das unveränderliche Raw-Ergebnis als Downstream-Eingang persistieren.

Die Berechnung verwendet native CUDA, OpenCL oder CPU entsprechend der
Acceleration-Auswahl und fällt bei einem Backend-Fehler kontrolliert auf CPU
zurück. Die Ausführung ist region-/chunkweise speicherbegrenzt.

### Ergebnisse

```text
outputs/aqmh_reconstructed_raw.fit   # unveränderliche Raw-AQMH-Baseline
outputs/reconstructed_L.fit          # ausgewähltes Luminanz-Ergebnis
artifacts/aqmh_reconstruction.json   # Gate-, Cache- und Backend-Diagnostik
```

Wichtige Diagnostikfelder sind `uniform_control_gate_triggered`,
`raw_aqmh_preserved_by_guard`, `raw_aqmh_validation`,
`final_vs_raw_aqmh_validation` und `selected_candidate`.

## TILE_RECONSTRUCTION (nur Classic)

Classic verwendet die lokalen Gewichte `L_f,t` und die globalen Gewichte `G_f`:

```text
W_f,t = G_f * L_f,t
```

Die Tiles werden support-aware per Overlap-Add in den Canvas geschrieben.
Danach können `STATE_CLUSTERING` und `SYNTHETIC_FRAMES` den Classic-Zweig
weiterverarbeiten.

## Übergang

- AQMH: `AQMH_RECONSTRUCTION` -> `AQMH_DIAGNOSTICS` -> `STACKING`
- Classic: `TILE_RECONSTRUCTION` -> `STATE_CLUSTERING` ->
  `SYNTHETIC_FRAMES` -> `STACKING`
