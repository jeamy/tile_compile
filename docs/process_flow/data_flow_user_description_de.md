# Process Flow – Technischer Datenfluss des Systems

## Ziel der Pipeline

Das System verwandelt einen Satz kalibrierter astronomischer Einzelframes
in ein reproduzierbares Endprodukt in einem gemeinsamen geometrischen und
photometrischen Referenzraum — mit genau einer Rekonstruktionsmethode:
**CFA Forward Drizzle + Multiband**.

Technisch ist die Pipeline in drei große Blöcke gegliedert:

- **Vorbereitung und Normalisierung**
  - Eingaben validieren
  - Intensitätsniveaus vereinheitlichen (pro CFA-Kanal bei OSC)
  - Registrierungsgeometrie messen
- **Qualitätsmodellierung und Forward-Rekonstruktion**
  - Normalisierungs-Cache versiegeln
  - Ausgaberaster, Coverage-Masken und Geometrie-Caches ableiten
  - pixelweise Quell-Qualitätskarten und globale Frame-Gewichte berechnen
  - jedes Quellsample per Forward-Drizzle in einen gebänderten
    Super-Resolution-Store abbilden
  - Multiband-Ergebnis fusionieren
- **Nachbearbeitung und Kalibrierung**
  - Astrometrie / WCS
  - optional BGE
  - optional PCC
  - optional HyperMetric Stretch

Primärprodukt ist ein lineares rekonstruiertes Bild (Mono bzw. R/G/B
getrennt bei OSC). Es gibt keinen Methoden-Selektor: AQMH und Classic
Tile-Compile existieren nicht mehr, und ein `method:`-Key in einer Config
wird fail-closed abgelehnt.

## Kernbegriffe

- **Run** — eine vollständige Pipeline-Ausführung mit eigenem
  Verzeichnis unter `runs/<run_id>/`.
- **Phase** — ein klar abgegrenzter Verarbeitungsschritt wie
  `REGISTRATION`, `FORWARD_DRIZZLE` oder `PCC`.
- **Artefakt** — persistierte Diagnose- oder Zwischendaten, typischerweise
  unter `artifacts/`.
- **Event-Timeline** — chronologische Ausführungs-Events in
  `logs/run_events.jsonl`.
- **Sampling-Plan** — `artifacts/registration_sampling.json`: der
  eingefrorene geometrische Vertrag (Ausgaberaster, affine + lokale
  Warps, CFA-Origin), dem alle Folgephasen folgen.
- **Uniform-Store** — der gebänderte Akkumulator-Satz unter
  `artifacts/forward_drizzle_v2/` aus dem Gather; einzige Quelle für
  Multiband-Fusion und Exporte.
- **Resume** — `resume-reconstruction` setzt einen Lauf nur ab
  `GLOBAL_QUALITY` oder `FORWARD_DRIZZLE` fort.

## Gesamtablauf

```text
Eingabe-Frames (FITS, MONO oder OSC/CFA)
   -> SCAN_INPUT
   -> CHANNEL_SPLIT            (nur Metadaten — keine Datei-Splits)
   -> NORMALIZATION            (B_f/P_f pro Frame & CFA-Kanal,
                                cache/normalized_frames/)
   -> REGISTRATION             (affine + optional lokaler Warp,
                                registration_sampling.json)
   -> NORMALIZED_CACHE         (Cache-Seal gegen den Plan)
   -> SAMPLING_GEOMETRY        (Ausgaberaster, Coverage-Masken,
                                forward_drizzle_geometry/-Cache)
   -> COMMON_OVERLAP           (Common-Valid-Support-Masken)
   -> SOURCE_QUALITY_MAPS      (cache/source_quality_maps/)
   -> GLOBAL_QUALITY           (Frame-Gewichte + Gates)
   -> FORWARD_DRIZZLE          (gechunkter CFA-Gather,
                                artifacts/forward_drizzle_v2/)
   -> MULTIBAND                (Band-Fusion,
                                reconstruction_multiband.fits)
   -> STACKING                 (Pass-Through-Marker)
   -> ASTROMETRY (optional)
   -> BGE        (optional)
   -> PCC        (optional)
   -> HYPERMETRIC_STRETCH (optional)
```

## Warum Forward Drizzle statt klassischem Stacking

- **Subpixel-Genauigkeit:** Jedes Quellsample wird über die gemessene
  Frame-Geometrie (affine + optional glattes lokales Warp) ins
  Ausgaberaster gesplattet — gedithered Daten ergeben echte
  Superauflösung.
- **CFA-nativ:** Bei OSC wird das Bayer-Pattern nie interpoliert —
  `cfa_channel_for_source_pixel` ordnet jeden Quellpixel der passenden
  R/G/B-Akkumulatorebene zu. Keine Debayer-Artefakte, voller
  Dither-Gewinn pro Kanal.
- **Qualitätsgewichtet:** Pixelweise Quell-Qualitätskarten × globale
  Frame-Gewichte ersetzen jede globale Frame-Ablehnung; schlechte Pixel
  verlieren Gewicht, nicht ganze Frames.
- **Begrenzter Speicher:** Zeilen-gechunkter Gather mit konfigurierbarem
  Speicherbudget; die Input-Amplifikation bleibt auch bei großen Stacks
  gering.

## Phasen im Detail

### SCAN_INPUT

- Enumeriert FITS-Eingaben, liest Dimensionen und Header (EXPTIME,
  Filter, Bayer-Keys).
- Erkennt MONO vs. OSC und das Bayer-Pattern; validiert Linearität und
  freien Plattenplatz (`scandir × 4` als Konservativschätzung).
- Modus-Inkonsistenzen oder unlesbare Eingaben sind hier terminal.

### CHANNEL_SPLIT (Metadaten)

- Hält Farbmodus und Bayer-Pattern im Event-Stream fest
  (`note: deferred_to_forward_drizzle_cfa`).
- Schreibt **keine Dateien**: die eigentliche Kanalzuordnung pro Pixel
  erfolgt im Drizzle-Gather.

### NORMALIZATION

- Berechnet additiven Hintergrund `B_f` und photometrischen Faktor `P_f`
  pro Frame — pro CFA-Kanal bei OSC (`B_r/g/b`, `P_r/g/b`).
- Photometrische Skalierung über `exposure_ratio`, wenn alle
  EXPTIME-Header valide sind, sonst `identity_fallback`.
- Persistiert normalisierte Frames nach `cache/normalized_frames/` und
  schreibt `normalization.json` + `global_metrics.json` (frühe
  Frame-Gewichte `G_f`).
- `normalization.enabled: false` ist terminal.

### REGISTRATION

- Kaskadierte globale Registrierung auf downsampleten
  Registrierungs-Proxys (CFA-sicher bei OSC).
- Liefert affine Transformation und optional ein glattes lokales
  Warp-Modell pro Frame; Fehlschläge degradieren auf Identitäts-Warp mit
  CC=0 — keine harte Frame-Ablehnung.
- Schreibt `global_registration.json` und den eingefrorenen Plan
  `registration_sampling.json` (Ausgabedimensionen, `internal_scale`,
  `cfa_origin`, Transformationen pro Frame).
- `run_provenance.json` verankert den Config-sha256 für die spätere
  Resume-Validierung.

### NORMALIZED_CACHE

- Versiegelt `cache/normalized_frames/` gegen den Sampling-Plan. Ab
  hier ist der Cache schreibgeschützte Eingabe für alle Konsumenten.

### SAMPLING_GEOMETRY

- Erzeugt das Ausgaberaster und die Coverage-Masken
  (`analysis_common_mask`, `reconstruction_support_mask`).
- Materialisiert den Local-Warp-Geometrie-Cache
  `artifacts/forward_drizzle_geometry/` für Frames mit lokalem Modell —
  rein affine Läufe überspringen ihn komplett.
- Schreibt `sampling_geometry.json`; der Coverage-Geometrie-Hash fließt
  in den Resume-Checkpoint ein.

### COMMON_OVERLAP

- Berechnet die pixelweise gemeinsame gültige Abdeckung aller Frames.
- Erzwingt `reconstruction.common_overlap_required_fraction`; schreibt
  `forward_common_overlap.json` und fixiert die Support-Masken für den
  Rest des Laufs.

### SOURCE_QUALITY_MAPS

- Pixelweise Qualitätskarten pro Frame aus der
  `reconstruction.quality.pyramid.*`-Konfiguration.
- Persistiert unter `cache/source_quality_maps/`; der Plan wird in
  `source_quality_plan.json` zusammengefasst.

### GLOBAL_QUALITY

- Aggregiert die Quellkarten zu finalen Frame-Gewichten `G_f` und wendet
  die Coverage-/Clipping-Gates an (`coverage_gate.*`,
  `common_overlap_required_fraction`, `clipping.min_n_eff`).
- Erster unterstützter Resume-Einstiegspunkt.

### FORWARD_DRIZZLE

- Der Kern-Gather: pro Zeilen-Chunk (`drizzle.chunk_rows` +
  `chunk_halo_rows`) wird jedes beitragende Quellsample mit
  `drizzle.kernel`/`drizzle.pixfrac` in die `wx`/`w`/`w²`-Akkumulatoren
  des gebänderten v2-Stores gesplattet — pro CFA-Kanal bei OSC.
- Robuste Reduktion nach `clipping.*` (`clip_sigma_low/high`,
  `min_clip_contributors`, `robust_passes`, `guard_fallback`).
- Speicher begrenzt durch `drizzle.memory_budget_mb`; der Checkpoint
  (`forward_drizzle_checkpoint.json`) protokolliert Geometrie-Hash und
  Artefaktgrößen für das Resume.

### MULTIBAND

- Fusioniert den gebänderten Store nach `reconstruction.multiband.*` zu
  `reconstruction_multiband.fits`; Kandidaten-Zwischendaten laufen über
  `artifacts/multiband_candidate_spool/`.
- Läuft bei Resume immer erneut (kein Einstiegspunkt).

### STACKING (Pass-Through-Marker)

- Wird vor dem Downstream für Status-/Report-Kontinuität emittiert. Der
  Drizzle-Store *ist* bereits der Stack — eine klassische Stack-Phase
  existiert nicht.

### ASTROMETRY / BGE / PCC / HYPERMETRIC_STRETCH

- Optionale Downstream-Phasen in fester Reihenfolge, jede einzeln
  überspringbar.
- Bis einschließlich PCC bleibt alles linear; HMS ist die explizite
  nichtlineare Endstufe.

## Typische Run-Struktur

```text
runs/<run_id>/
├── config.yaml                     # eingefrorene Run-Config
├── logs/run_events.jsonl           # Event-Stream
├── artifacts/
│   ├── run_provenance.json         # Resume-Anker (config sha256)
│   ├── normalization.json
│   ├── global_metrics.json
│   ├── global_registration.json
│   ├── registration_sampling.json
│   ├── sampling_geometry.json
│   ├── forward_common_overlap.json
│   ├── forward_drizzle_geometry/
│   ├── source_quality_plan.json
│   ├── forward_drizzle_v2/
│   ├── forward_drizzle_checkpoint.json
│   ├── forward_drizzle.json
│   ├── reconstruction_multiband.fits
│   └── report.html                 # via POST /api/runs/<id>/stats
├── cache/
│   ├── normalized_frames/
│   └── source_quality_maps/
└── outputs/                        # finale FITS-Produkte
```

## Resume

Es existieren nur zwei Einstiegspunkte — `GLOBAL_QUALITY` und
`FORWARD_DRIZZLE`. Beide validieren Provenance (Config-sha256), Checkpoint
(Geometrie-Hash, Artefaktgrößen) und alle benötigten Caches fail-closed.
Downstream-Phasen laufen immer erneut. Siehe
[resume_dependencies_de.md](resume_dependencies_de.md).

## Auswertung mit dem integrierten Report-Generator

```text
POST /api/runs/<run_id>/stats  (GUI: Stats erstellen)
```

erzeugt `artifacts/report.html` (self-contained, inline SVG) und
`artifacts/stats.json` aus den Artefakt-JSONs und `run_events.jsonl` — Normalisierungsverläufe,
Registrierungsbewertung, Coverage-/Support-Heatmaps, Drizzle- und
Multiband-Diagnostik, Downstream-Ergebnisse (BGE/PCC) und die
Pipeline-Timeline.

## Hinweise zur Interpretation

- Eine `skipped` Downstream-Phase (z.B. keine astrometrische Lösung) ist
  kein Fehler; der Grund steht im Phase-End-Payload.
- `channels` in CHANNEL_SPLIT beschreibt nur das Kanalmodell — die
  tatsächliche Kanalabdeckung steht in `forward_drizzle.json`.
- Ein sofort fertiges `STACKING` ist erwartbar: es ist ein Marker, die
  eigentliche Arbeit geschah in FORWARD_DRIZZLE/MULTIBAND.

## Kurzfazit

Die Pipeline ist ein einziger deterministischer
CFA-Forward-Drizzle-Rekonstruktionspfad: normalisieren, registrieren,
pixelweise Qualität messen, jedes Sample in einen gebänderten
Super-Resolution-Store splatten, Bänder fusionieren — danach optional
solven, korrigieren, kalibrieren und stretchen, mit einem strikten
fail-closed Resume-Vertrag an den beiden teuren Grenzen.
