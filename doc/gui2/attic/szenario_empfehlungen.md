# Szenario-Empfehlungen (Linie B)

Diese Empfehlungen sind als Preset-Deltas fuer den `Situation Assistant` gedacht.

## Pflicht-Szenarien

## 1) Alt/Az

Typisches Ziel: stabile Registrierung trotz Feldrotation.

| Parameter | Empfehlung | Grund |
|---|---|---|
| `registration.allow_rotation` | `true` | Rotation muss im Modell erlaubt sein |
| `registration.star_topk` | `150..220` | mehr Sternkandidaten fuer robuste Matches |
| `registration.reject_shift_px_min` | `>= 100` | grosse natuerliche Shifts nicht falsch verwerfen |
| `registration.reject_shift_median_multiplier` | `4.0..6.0` | breite Shift-Verteilung tolerieren |
| `dithering.enabled` | `true` | hilft gegen Pattern-Artefakte |

## 2) Starke Rotation

Typisches Ziel: Robustheit gegen Rotationsausreisser.

| Parameter | Empfehlung | Grund |
|---|---|---|
| `registration.engine` | `robust_phase_ecc` | robustere Registrierung bei schwierigen Feldern |
| `registration.allow_rotation` | `true` | zwingend bei deutlicher Feldrotation |
| `registration.star_inlier_tol_px` | `3.5..5.0` | tolerantere Inlier-Bedingung |
| `registration.reject_cc_min_abs` | `0.25..0.35` | harte CC-Grenzen vermeiden |

## 3) Helle Sterne im Feld

Typisches Ziel: weniger Halos/Farb-Ausreisser in PCC/BGE.

| Parameter | Empfehlung | Grund |
|---|---|---|
| `pcc.mag_bright_limit` | `5.0..7.0` | sehr helle Sterne begrenzen |
| `pcc.k_max` | `2.0..2.8` | zu starke Farbgains vermeiden |
| `pcc.sigma_clip` | `2.3..3.0` | robustere Ausreisserunterdrueckung |
| `bge.mask.star_dilate_px` | `5..8` | Sternumgebung in BGE staerker maskieren |

## 4) Wenige Frames / kurze Session

Typisches Ziel: stabile Resultate trotz geringem N.

| Parameter | Empfehlung | Grund |
|---|---|---|
| `assumptions.frames_reduced_threshold` | `150..220` | frueher in Reduced-Mode wechseln |
| `assumptions.reduced_mode_skip_clustering` | `true` | instabile Clusterbildung vermeiden |
| `synthetic.frames_min` | `3..5` | minimale Synthetic-Basis absichern |
| `synthetic.clustering.cluster_count_range` | `[3, 10]` | kleinere Clusterzahl fuer kleine Datensaetze |

## 5) Starker Hintergrundgradient / Lichtverschmutzung

Typisches Ziel: Hintergrundmodell stabilisieren.

| Parameter | Empfehlung | Grund |
|---|---|---|
| `bge.enabled` | `true` | Gradient aktiv modellieren |
| `bge.fit.method` | `rbf` | flexible Modellierung fuer komplexe Gradienten |
| `bge.fit.rbf_lambda` | `5e-3..2e-2` | Regularisierung gegen Ueberschwingen |
| `bge.sample_quantile` | `0.12..0.25` | robuste Hintergrundsamples |
| `bge.structure_thresh_percentile` | `0.75..0.90` | Struktur von Hintergrund trennen |

## Objektprofile (optional)

| Objekt | Tendenz |
|---|---|
| Galaxie | konservativer Denoise, Fokus auf feine Struktur (`tile_denoise` moderat) |
| Emissionsnebel | BGE/PCC wichtiger, Gradient- und Farbstabilitaet priorisieren |
| Sternhaufen | Sternmetrik und Saturationskontrolle priorisieren |

## Hinweis zur GUI

- Empfehlungen werden als Delta angezeigt, nicht still angewendet.
- Benutzer sieht vor Uebernahme immer `vorher -> nachher`.
- Mehrere Szenarien werden priorisiert zusammengefuehrt, Konflikte werden markiert.

## MONO-Beispielprofile aus dem Repository

Diese Profile sind als `scenario_profile` in den YAML-Beispielen hinterlegt und koennen in GUI2 direkt als Startpunkt genutzt werden.

| Profil-ID | Datei | Fokus | GUI2-Szenario-Tags |
|---|---|---|---|
| `mono_full_mode` | `tile_compile_cpp/examples/tile_compile.mono_full_mode.example.yaml` | Vollmodus mit robuster Alt/Az-Registrierung | `mono`, `full_mode`, `alt_az`, `rotation_robust` |
| `mono_small_n_anti_grid` | `tile_compile_cpp/examples/tile_compile.mono_small_n_anti_grid.example.yaml` | Kleines N, Tile-Naehte und Pattern reduzieren | `mono`, `small_n`, `anti_grid`, `alt_az`, `rotation_robust` |
| `mono_small_n_ultra_conservative` | `tile_compile_cpp/examples/tile_compile.mono_small_n_ultra_conservative.example.yaml` | Sehr kleines N, maximal konservative Stabilitaet | `mono`, `very_small_n`, `ultra_conservative`, `alt_az`, `rotation_robust` |

### Mapping-Regel fuer GUI2

- Wenn `scenario_profile.id` vorhanden ist, wird das passende Profil im Situation Assistant vorausgewaehlt.
- `scenario_profile.gui2_scenarios` wird als aktive Tag-Liste uebernommen.
- Der Benutzer sieht vor der Uebernahme weiterhin die Delta-Ansicht (`vorher -> nachher`).
