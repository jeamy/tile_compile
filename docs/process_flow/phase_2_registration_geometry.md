# Phase 2 — REGISTRATION, NORMALIZED_CACHE, SAMPLING_GEOMETRY, COMMON_OVERLAP

> **C++-Implementierung:** `runner_phase_registration.cpp`
> (`run_phase_registration_prewarp`), `runner_forward_drizzle.cpp`
> (Cache-Seal, SAMPLING_GEOMETRY, COMMON_OVERLAP)

## REGISTRATION

Misst die Geometrie, die später jeden Quellpixel ins Ausgaberaster abbildet.

**Ablauf:**

- Arbeitet auf den **Registrierungs-Proxys** aus NORMALIZATION
  (downsampled, CFA-sicher für OSC)
- Kaskadierte Fallbacks; fehlgeschlagene Frames fallen auf Identitäts-Warp
  mit CC=0 zurück (keine harte Frame-Selektion)
- Fehlgeschlagene/verworfene Frames bekommen einen modellvorhergesagten
  Warp (Feldrotations-Polynom, Blending, Interpolation, Nearest-Copy) nur,
  wenn das immer aktive Plausibilitäts-Gate die Vorhersage gegen die
  benachbarten gemessenen Anker akzeptiert (Winkel: `1 deg + 2x` lokale/
  globale Rate, gedeckelt 5 deg; Shift: `60 px + 2x` Rate, gedeckelt
  400 px; Extrapolation max. 3 Frames). Unplausible Vorhersagen bleiben
  `unresolved` und fallen aus Canvas/Prewarp/Drizzle heraus; Zähler unter
  `diag.reg_model_predicted_implausible{,_reasons,_frames}` in
  `global_registration.json`
- Ergebnis pro Frame: **affine Transformation** + optional **glattes lokales
  Warp-Modell** (`has_smooth_local_model`)
- Dithering-Diagnostik (`dithering.min_shift_px` als Gate)

**Artefakte:**

- `artifacts/global_registration.json` — Warp-Matrizen
  `(a00, a01, tx, a10, a11, ty)`, Korrelation/CC, Dithering-Diagnose
- `artifacts/registration_sampling.json` — der **Sampling-Plan**
  (`RegistrationSamplingPlan`): Referenzframe, Ausgabedimensionen,
  `internal_scale`, CFA-Origin (`cfa_origin_x/y`), pro Frame affine Inverse +
  optionales lokales Modell
- `artifacts/run_provenance.json` — Config-sha256 als Resume-Anker

`PREWARP`-Events gehören intern zu dieser Phase; ein materialisierter
Prewarp-Canvas existiert nicht — Forward Drizzle nutzt die geometrische
Beschreibung direkt statt vorgewarpter Bilder.

## NORMALIZED_CACHE

Kurze Phase: versiegelt `cache/normalized_frames/` gegen den Sampling-Plan
(`seal_normalized_cache`). Jeder Downstream-Konsument darf den Cache nur noch
lesen; ein frischer Lauf ohne Cache bricht mit
`FORWARD_STAGE_NORMALIZED_CACHE_REQUIRED` ab. Resume-Läufe validieren den
Cache anhand des Checkpoints.

## SAMPLING_GEOMETRY

Übersetzt den Sampling-Plan in die produktive Rastergeometrie.

- Ausgabedimensionen aus `internal_scale` × Referenzgeometrie
- Coverage-Masken (`analysis_common_mask`, `reconstruction_support_mask`)
  als FITS-Rows
- **Local-Warp-Geometrie-Cache** unter
  `artifacts/forward_drizzle_geometry/`: Für Frames mit lokalem Modell werden
  die Sample-Leaves einmalig materialisiert (`.leaves` + `manifest.json` +
  `.rows`-Index). Affine-only-Läufe überspringen den Cache komplett
- Pro Geometrie-Variante (`pixfrac` und ggf. `pixfrac=1.0` für die
  Uniform-Kontrolle) eine Cache-Identität (`make_geometry_cache_identity`)
- `artifacts/sampling_geometry.json` — Raster, Coverage-Statistik,
  Geometrie-Hash (`compute_coverage_geometry_hash`, prüft Config+Plan für
  den Resume-Checkpoint)

## COMMON_OVERLAP

- Pixelweise Schnittmenge gültiger Quellpixel aller Frames über die
  Registrierungsgeometrie
- `reconstruction.common_overlap_required_fraction` als Mindestanteil;
  Unterschreiten ist terminal
- `artifacts/forward_common_overlap.json` — Overlap-Statistik
- Die Analysis- und Reconstruction-Support-Masken werden hier fixiert und
  fließen in den Checkpoint ein — Support-Masken-Semantik ist damit über
  Resume stabil
