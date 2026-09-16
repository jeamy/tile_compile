# Phase 4 — FORWARD_DRIZZLE

> **C++-Implementierung:** `runner_forward_drizzle.cpp` (Stage-Orchestrierung),
> `src/reconstruction/forward_drizzle_v2*.cpp` (Gather/Store),
> `src/reconstruction/sampling_geometry.cpp` (Geometrie)

Der Kern der Pipeline. Forward Drizzle bildet **jedes Quellsample vorwärts**
in das super-aufgelöste Ausgaberaster ab — statt wie klassische Stacker
Ausgabepixel invers in Quellframes zu sampeln.

## Funktionsweise

1. **Geometrie:** Für jeden Ausgabepixel(-Chunk) wird die inverse affine
   Abbildung (`map_to_source`) — plus optional das lokale Warp-Modell aus dem
   Geometrie-Cache — ausgewertet, um zu bestimmen, welche Quellsamples in
   den Zielbereich fallen.
2. **CFA-Kanalzuordnung:** Bei OSC bestimmt
   `cfa_channel_for_source_pixel(sx, sy, bayer_pattern, cfa_origin_x,
   cfa_origin_y)` pro Quellpixel die Zielebene (R=0, G=1, B=2). Es gibt
   **kein Debayering** — die Bayer-Struktur wird direkt gedrizzlet
   ("drizzle before debayer"), wodurch Dithering die volle Sub-Pixel-
   Auflösung pro Kanal liefert.
3. **Drizzle-Kernel:** Jedes Sample wird mit dem konfigurierten Kernel
   (`reconstruction.drizzle.kernel`) und Drop-Größe
   (`reconstruction.drizzle.pixfrac`) gewichtet in die Zielzellen
   akkumuliert: `wx` (gewichtete Summe), `w` (Gewichte), `w²` (Varianz-
   Terme) pro Kanal.
4. **Qualitätsgewichtung:** Sample-Gewicht = Quell-Qualitätskarte
   (SOURCE_QUALITY_MAPS) × Frame-Gewicht `G_f` (GLOBAL_QUALITY) ×
   Kernel-Beitrag × Support-Maske.
5. **Robuste Reduktion:** `reconstruction.clipping.*` steuert
   Sigma-Clipping (`clip_sigma_low/high`), Mindestbeiträge
   (`min_clip_contributors`, `min_n_eff`), Iterationen (`robust_passes`)
   und das `guard_fallback`-Verhalten bei unzureichender n_eff.

## Chunking und Speicher

- Verarbeitung in **Zeilen-Chunks** (`reconstruction.drizzle.chunk_rows`
  mit `chunk_halo_rows` Overlap für Kernel-Überhang)
- `reconstruction.drizzle.memory_budget_mb` begrenzt die
  Quell-Frame-LRU und den Band-Planner (Default:
  `runtime_limits.memory_budget`)
- `TC_FORWARD_DRIZZLE_MEMORY_BUDGET_MB` erlaubt Budget-Sweeps ohne
  Config-Edit, ohne den Checkpoint zu invalidieren (CPU-Pfad bleibt
  budget-invariant und bit-exakt)

## Der gebänderte v2-Store

`artifacts/forward_drizzle_v2/` enthält den inkrementell geschriebenen
Uniform-Store (Akkumulatoren `wx`/`w`/`w²` pro Kanal und Band). Er ist die
alleinige Quelle für MULTIBAND und die Exporte — ein separater
Stacking-Durchlauf existiert nicht (`STACKING` ist nur ein
Pass-Through-Marker).

## Artefakte

| Datei | Inhalt |
|-------|--------|
| `artifacts/forward_drizzle_v2/` | Gebänderter Uniform-Store |
| `artifacts/forward_drizzle_checkpoint.json` | Resume-Checkpoint (Geometrie-Hash, Artefakt-Bytegrößen) |
| `artifacts/forward_drizzle.json` | Lauf-Zusammenfassung (Chunk-Statistiken, Timings) |
| `artifacts/forward_drizzle_geometry_profile.json` | Geometrie-Instrumentierung (Counter je Variante) |
| `artifacts/forward_drizzle_uniform_diagnostic.json` | Diagnose (je nach `diagnostics.*`) |

## Diagnose-Schalter

`reconstruction.diagnostics.*` steuert ausschließlich Diagnoseausgaben
(`level`, `preview_forward_drizzle_uniform`,
`persist_forward_drizzle_uniform_store`) — ohne Einfluss auf das
Rekonstruktionsergebnis.

## Resume

`resume-reconstruction --from-phase FORWARD_DRIZZLE` validiert Checkpoint,
Geometrie-Hash, Provenance (config sha256) und Store-Konsistenz, setzt dann
den Gather fort und führt MULTIBAND + Downstream erneut aus. Frühere
Phasen werden nicht wiederholt.
