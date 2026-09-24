# Phase 3 — SOURCE_QUALITY_MAPS, GLOBAL_QUALITY

> **C++-Implementierung:** `runner_forward_drizzle.cpp`
> (`run_forward_drizzle_stages`, Stages ab `Phase::SOURCE_QUALITY_MAPS`)

## SOURCE_QUALITY_MAPS

Erzeugt die pixelweise Quell-Qualitätsbewertung, die FORWARD_DRIZZLE als
Sample-Gewicht nutzt.

**Was:**

- Pro Frame eine pixelweise Qualitätskarte, berechnet über die
  `reconstruction.quality.pyramid.*`-Konfiguration (Multi-Scale-Sharpness/
  SNR über Pyramiden-Level, `score_scale` vor der Sigmoid-Abbildung,
  `k_artifact`/`frac_artifact_max` für Artefakt-Abwertung)
- Persistenz unter `cache/source_quality_maps/` — wiederverwendbar bei
  Resume und steuerbar über
  `reconstruction.delete_source_cache_after_run` /
  `reconstruction.keep_profile_cache_after_run`
- `artifacts/source_quality_plan.json` — der resultierende Gewichtsplan

## GLOBAL_QUALITY

Das Gate zwischen Analyse und Produktion — und ein **Resume-Einstiegspunkt**.

**Aufgaben:**

- Aggregation der Quellkarten zu globalen Frame-Gewichten `G_f`
  (Kombination mit den frühen global_metrics-Gewichten)
- Coverage-/Qualitäts-Gates (`reconstruction.coverage_gate.*`,
  `reconstruction.common_overlap_required_fraction`,
  `reconstruction.clipping.*`-Mindestanforderungen an den effektiven
  Beitrag `n_eff`)
- Finaler Gewichtsplan, der FORWARD_DRIZZLE exakt vorgibt, welche Samples
  mit welchem Gewicht eingehen

**Resume-Bedeutung:** `resume-reconstruction --from-phase GLOBAL_QUALITY`
setzt voraus, dass alle Vorgänger-Artefakte (Sampling-Plan, Geometrie-Cache,
Overlap-Masken, Source-Quality-Maps, Checkpoints) vorhanden sind und der
Geometrie-Hash zum Checkpoint passt. Ab hier werden GLOBAL_QUALITY,
FORWARD_DRIZZLE, MULTIBAND und alle Downstream-Phasen neu ausgeführt.

**Support-Masken-Semantik:** Die in COMMON_OVERLAP fixierten Masken werden
hier nicht verändert — Resume kann die Gewichtung, nie die räumliche
Support-Definition ändern.
