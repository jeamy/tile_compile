# Resume-Abhängigkeiten

Dieses Dokument beschreibt den tatsächlich implementierten Resume-Vertrag des
aktuellen Runners. Alte Run-Layouts und alte Cache-Pfade sind nicht Bestandteil
des Vertrags.

## Grundregel

Ein Resume ist nur sicher, wenn alle Eingaben der Zielphase als gültige
Artefakte oder Caches vorhanden sind. `config.yaml` im Run-Verzeichnis ist
immer erforderlich.

Die CLI akzeptiert mehrere Phasen, aber sie haben nicht alle dieselbe
Semantik:

- **In-Place-Vollständigkeitslauf:** Die frühen Phasen werden nicht ab der
  ausgewählten Phase fortgesetzt. Der Runner liest `input_dir` aus dem letzten
  `run_start`-Event und startet den vollständigen Pipeline-Lauf erneut im selben
  Run-Verzeichnis.
- **Direktes Resume:** Die Phase lädt persistierte Artefakte und startet nur
  den passenden Downstream-Pfad.

## Unterstützte Einstiege

| Angeforderte Phase | Mechanismus | Tatsächlicher Einstieg | Mindestabhängigkeiten | Ergebnis |
|---|---|---|---|---|
| `SCAN_INPUT`, `CHANNEL_SPLIT`, `NORMALIZATION`, `GLOBAL_METRICS`, `TILE_GRID`, `REGISTRATION`, `PREWARP`, `COMMON_OVERLAP`, `LOCAL_METRICS`, `TILE_RECONSTRUCTION`, `STATE_CLUSTERING`, `SYNTHETIC_FRAMES`, `DEBAYER` | In-Place-Vollständigkeitslauf | neuer vollständiger `run` | `config.yaml`, gültiges letztes `run_start`-Event mit `input_dir`, lesbare Eingabeframes | alle Phasen werden neu erzeugt; vorhandene Artefakte werden nicht als Resume-Eingabe garantiert |
| `AQMH_MAPS`, `AQMH_GLOBAL_QUALITY`, `AQMH_RECONSTRUCTION`, `AQMH_DIAGNOSTICS` | Direktes AQMH-Resume | `AQMH_RECONSTRUCTION` | `artifacts/aqmh_metrics.json`, `cache/aqmh/aqmh_cache.json`, gültige `cache/prewarped_frames`, `outputs/canvas_mask.fits`; bei abweichender Maskengröße zusätzlich rekonstruierbare `cache/aqmh_masks` | neue AQMH-Rekonstruktion, Diagnostik und anschließend `STACKING`/`DEBAYER` |
| `STACKING` bei AQMH | Direktes Resume oder AQMH-Rekonstruktion | `STACKING` bei vorhandenem Raw-Artefakt, sonst `AQMH_RECONSTRUCTION` | bevorzugt `outputs/aqmh_reconstructed_raw.fit`; falls es fehlt zusätzlich alle AQMH-Abhängigkeiten der vorherigen Zeile | Stack, Debayer und weitere Folgephasen |
| `STACKING` bei Classic | Direktes Resume | `STACKING` | mindestens ein gültiges `outputs/synthetic_*.fit`; optional `artifacts/synthetic_frames.json`, `artifacts/global_registration.json`, Masken | Stack, Debayer und weitere Folgephasen |
| `ASTROMETRY` | Direktes Post-Processing-Resume | `ASTROMETRY` | `outputs/stacked_rgb_solve.fits` oder `outputs/stacked_rgb.fits`; optional vorhandenes `artifacts/stacked_rgb.wcs` als Fallback | Astrometrie, danach BGE und PCC |
| `BGE` | Direktes Post-Processing-Resume | `BGE` | RGB-Output wie bei `ASTROMETRY`, `outputs/canvas_mask.fits`; für Classic zusätzlich passende `artifacts/local_metrics.json` und `artifacts/tile_grid.json`, für AQMH kann BGE Tile-Daten aus dem RGB-Output ableiten | BGE, danach PCC |
| `PCC` | Direktes Post-Processing-Resume | `PCC` | RGB-Output; optional `outputs/stacked_rgb_bge_linear.fits`; für aktiviertes PCC ein gültiger WCS oder ein vorhandenes WCS-Artefakt | PCC |
| `HYPERMETRIC_STRETCH`/`HMS` | Direktes Post-Processing-Resume | `HYPERMETRIC_STRETCH` | bevorzugt `outputs/pcc_R.fit`, `pcc_G.fit`, `pcc_B.fit` oder `outputs/stacked_rgb_pcc.fits`; bei `require_successful_pcc: false` alternativ BGE-/Solve-RGB | HMS-Ausgabe |

## Cache-Abhängigkeiten

Alle aktuellen Caches liegen unter `cache/`:

| Cache | Erzeugt durch | Wird direkt benötigt von |
|---|---|---|
| `cache/normalized_frames` | Normalisierung | aktuelle Vollständigkeitsläufe und nachfolgende Pipeline-Phasen desselben Laufs |
| `cache/prewarped_frames` | Registration/Prewarp | `AQMH_RECONSTRUCTION` und alle AQMH-Resume-Einstiege, die dorthin abgebildet werden |
| `cache/aqmh` | AQMH-Quality-Maps | AQMH-Rekonstruktion und AQMH-Diagnostik |
| `cache/aqmh_masks` | AQMH-/Maskenphasen | AQMH-Resume, wenn `outputs/canvas_mask.fits` nicht in voller Canvas-Größe vorliegt |
| `cache/phase9_osc_rgb` | RGB-/Phase-9-Verarbeitung | nur wenn der jeweilige Downstream-Code es für den konkreten Lauf verwendet |

`aqmh.reconstruction.delete_prewarped_cache_after_run: true` löscht den
Prewarp-Cache nach dem Lauf. Dann ist ein direktes AQMH-Resume ab
`AQMH_RECONSTRUCTION` nicht möglich; ein In-Place-Vollständigkeitslauf bleibt
mit gültigem Input-Log möglich.

## Nicht sichere Annahmen

- Eine angeforderte frühe Phase bedeutet **nicht**, dass nur diese Phase läuft.
- Ein vorhandenes Phase-Event ersetzt kein benötigtes Artefakt.
- Caches außerhalb von `cache/prewarped_frames`, `cache/normalized_frames`,
  `cache/aqmh` und `cache/aqmh_masks` werden nicht gelesen.
- `STACKING` ist bei AQMH ohne `aqmh_reconstructed_raw.fit` kein reines
  Stack-Resume, sondern benötigt zuerst eine gültige AQMH-Rekonstruktion.

## Quellen

- `tile_compile_cpp/apps/runner_resume.cpp`
- `tile_compile_cpp/apps/runner_pipeline.cpp`
- `web_backend_cpp/include/services/run_inspector.hpp`
