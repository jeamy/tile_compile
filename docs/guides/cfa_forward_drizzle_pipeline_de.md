# CFA-Forward-Drizzle + Mehrband-Rekonstruktion

Dies ist die **einzige Rekonstruktionsmethode** der aktuellen
`tile_compile`-Pipeline. Es gibt keine Methodenauswahl: jeder Lauf verwendet
CFA-bewusstes Forward-Drizzle, gefolgt von Mehrband-Rekonstruktion und einer
Drei-Wege-Kandidatenauswahl. Die älteren Classic-/AQMH-Tile-Methoden werden im
Produkt nicht mehr angeboten (ihre Dokumente unter *Professional & Technical*
beschreiben historisches Verhalten).

## Was es tut

Jedes kalibrierte, registrierte Light-Frame wird **direkt aus seinen
Bayer-(CFA-)Samples** mit einem exakten quadratischen Droplet-Kernel auf das
Ausgaberaster gedrizzelt. Kein Frame wird vor seinem Beitrag debayert oder
interpoliert — Farb- und Auflösungsinformation gelangen unverändert vom Sensor
in die Rekonstruktion.

Drei Kandidatenbilder werden auf derselben Geometrie gebaut, die Pipeline wählt
automatisch das beste:

| Kandidat | Aufbau | Rolle |
|---|---|---|
| **`drizzle_uniform`** | Gleiches Gewicht je Sample | Robuste Kontrolle; die Sicherheitsuntergrenze |
| **`drizzle_raw`** | Qualitätsgewichtet je Quell-Sample | Schärfer, wo die Daten es tragen |
| **`drizzle_multiband`** | À-trous-Mehrband-Fusion der Raw-Schätzung mit bandweiser Konfidenz (`alpha`) | Beste Detailerhaltung bei hoher Konfidenz |

Die Auswahl entscheidet sich an festen Validierungsmetriken (Hintergrund-RMS,
Seam-Score, FWHM, Elongation, Tail). Verschlechtert sich der schärfere Kandidat
auf einer Pflicht-Sicherheitsmetrik, fällt die Pipeline zurück —
`drizzle_multiband` → `drizzle_raw` → `drizzle_uniform` — und protokolliert den
Grund im Run-Report.

## Pipeline-Phasen

Die aktiven Rekonstruktionsphasen in Laufreihenfolge:

| Phase | Beschreibung |
|---|---|
| `NORMALIZED_CACHE` | Normalisierten CFA-Quellcache bauen/wiederverwenden (mit Metadaten) |
| `SAMPLING_GEOMETRY` | Registrierungs-Sampling-Plan; direkt rasterisierte Kanal-Coverage, `n_eff` und Loch-Abdeckung; das **Coverage-Gate** |
| `COMMON_OVERLAP` | Gemeinsamer gültiger Daten-Overlap über die akzeptierten Frames |
| `SOURCE_QUALITY_MAPS` | Frame-weise Quell-Qualitätskarten (Pyramide) für die Raw-/Multiband-Gewichte |
| `GLOBAL_QUALITY` | Globale frame-weise Qualitätsgewichte |
| `FORWARD_DRIZZLE` | CFA-Forward-Drizzle → der transaktionale U/R/F/M-Profilspeicher (CPU oder CUDA, byte-identisch) |
| `MULTIBAND` | À-trous-Fusion zu `X_out`, Drei-Wege-Kandidatenauswahl, Auslieferung je Kanal |

Die nachgelagerten Phasen (`STACKING`-Pass-through, `DEBAYER`, `ASTROMETRY`,
`BGE`, `PCC`, `HYPERMETRIC_STRETCH`) sind unverändert.

### Coverage-Gate

`SAMPLING_GEOMETRY` führt vor jeder Rekonstruktion ein hartes, fail-closed Gate
aus: gültiger Kanalanteil ≥ 0,995, p10-`n_eff` ≥ `max(3.0, 0.15·N)` je Kanal,
≥ 1024 Analysepixel und keine interne ungestützte Kanalinsel. Bei dünner
CFA-Abdeckung (wenige Frames, geringes Dither, starke Feldrotation) deckt ein
R- oder B-Sample pro Frame nur einen kleinen Rasteranteil ab — unzureichend
geditherte Sätze scheitern hier, statt Farbsäume oder Kammartefakte zu
erzeugen. Bei grenzwertiger Abdeckung sind die dokumentierten Abhilfen global
`pixfrac = 1.0` oder `internal_scale = 1`.

## Scale-Modi

`internal_scale` / `output_scale` in `reconstruction.drizzle`:

- **1 / 1** — Rekonstruktion in nativer Auflösung.
- **2 / 2** — 2× übersampelte Rekonstruktion und Ausgabe.
- **2 / 1** (Produktionsstandard) — Rekonstruktion intern in 2×, dann ein
  einzelner deterministischer 2×2-Flächenmittelwert auf native Ausgabe. Der
  empfohlene Modus für kritisch abgetastete Sternfelder.

## Ausgaben

In `runs/<run_id>/`:

| Datei | Inhalt |
|---|---|
| `outputs/reconstructed_<ch>.fit` | Der **gewählte** Kandidat, je Kanal, in Output-Scale-Geometrie |
| `outputs/forward_drizzle_raw_<ch>.fit` | Die unveränderliche Raw-Baseline, immer geschrieben |
| `artifacts/reconstruction_multiband.fits` | Das fusionierte Mehrband-`X_out` (Fuse-Commit-Ziel) |
| `artifacts/forward_drizzle.json` | Der Rekonstruktionsvertrag: Geometrie, Coverage-Gate, Kandidatenauswahl + Validierungs-Gates, Clipping, Fluxraum, bandweise Alpha-Konfidenz, Ressourcen, Durchsatz und die Referenzmaschine |

Mit `diagnostics.level: full` werden zusätzlich die Uniform- und
Multiband-Control-FITS ausgeliefert. `diagnostics.level` und die
Profilcache-Retention **verändern das Rechenergebnis nie** — nur, welche
Zusatzdateien geschrieben und welche Caches behalten werden.

Alle Ebenen liegen im **normalisierten linearen Arbeitsraum** (derselbe Raum wie
`reconstruction_multiband.fits`). Die STACKING-17.4-Normalisierungs-Rücknahme
(`scale_r/g/b`, Pedestal) erfolgt nachgelagert, nicht hier.

## Caches und Resume

| Einstellung | Standard | Wirkung |
|---|---|---|
| `reconstruction.keep_profile_cache_after_run` | `false` | Der interne U/R/F/M-Profilspeicher ist ein Rekonstruktionscache und wird nach einem fertigen Bild gelöscht. `true` behält ihn (gehasht), damit eine erneute Fusion den Forward-Drizzle-Durchlauf überspringt — kostet Plattenplatz. Ohne Einfluss auf das Ergebnis. |
| `reconstruction.delete_source_cache_after_run` | `false` | Behält den normalisierten CFA-Quellcache und die Qualitätskarten, damit die Rekonstruktions-Wiederaufnahme möglich bleibt. `true` gibt sofort Plattenplatz frei, **deaktiviert aber die Rekonstruktions-Wiederaufnahme** für den Lauf; der Report weist das als `resume_reconstruction_disabled` aus. |

Der Abschnitt **CFA Forward Drizzle / Multiband** im Run-Report zeigt das
Coverage-Gate, die Kandidatenentscheidung mit ihren kandidatenweisen
Validierungsmetriken, den Fluxraum, die Rauschdiagnostik und den Ressourcen-/
Durchsatzblock — jeder Wert stimmt mit dem entsprechenden
`forward_drizzle.json`-/`sampling_geometry.json`-Artefakt überein.

## Verwandt

- [Typischer Workflow (GUI3)](workflow_de.md)
- [GUI3 Benutzerhandbuch](../gui3_user_guide_de.md)
- [Outputs & Artifacts](../reference/outputs.md)
