# SlopCodeBench-Analyse: tile_compile (Stand 2026-09-16)

Analyse nach den drei Metriken des SlopCodeBench-Papers
([arXiv:2603.24755](https://arxiv.org/html/2603.24755v1),
[Earendil-Blog](https://earendil.com/posts/measuring-code-sloppiness/)):

1. **LOC-Aenderung** — Netto produzierte Zeilen im Zeitfenster.
2. **Verbosity** = |AST-Grep-flagged ∪ Clone-Lines| / LOC.
3. **Erosion** = Σ mass(f) fuer CC(f)>10 / Σ mass(f),
   mass(f) = CC(f) × √SLOC(f).

Scope: `tile_compile_cpp/src`, `tile_compile_cpp/apps`, `web_backend_cpp/src`
(+ `include`), C++-Quellen ohne Build-Artefakte und Third-Party
(`_deps` ausgeschlossen). Werkzeuge: lizard 1.24.0, jscpd 5.2.1.

## Benchmark-Referenzwerte (SlopCodeBench Tabelle 2)

| Gruppe | n | Verbosity | Erosion |
|---|---|---|---|
| Human (alle) | 48 | 0.15 ± 0.06 | 0.31 ± 0.17 |
| Agent | 990 | 0.33 ± 0.10 | 0.68 ± 0.20 |

Einzelreferenzen: scikit-learn Erosion 0.411, scipy 0.457.

## Messergebnisse

| Metrik | Wert | Human | Agent | Rating |
|---|---|---|---|---|
| LOC (30 Tage, netto) | +6.134 (+45.361 / −39.227) | — | — | FD-Cutover dominiert |
| Verbosity | **0.056** | 0.15 ± 0.06 | 0.33 ± 0.10 | **unter Human-Niveau (sehr gut)** |
| Erosion (CC>10) | **0.900** | 0.31 ± 0.17 | 0.68 ± 0.20 | **ueber Agent-Niveau (kritisch)** |

Erosion auch bei fuer C++ faireren Schwellen deutlich kritisch:
CC>15 → 0.846, CC>20 → 0.795.

Codebasis: 81.849 Zeilen C++ (jscpd), 1.946 Funktionen, AvgCCN 9.7.

### Verbosity-Details

- Clone: 350 exakte Klone, 3.137 duplizierte Zeilen (**3.83 %** — unter
  Human-Referenz 6 %)
- Boilerplate (Doxygen-`@brief`/`@details`/`Implements`-Muster): 1.414 Zeilen
- Toter Code (`#if 0`): 0 Zeilen
- AST-Grep-Proxy-Erfassung ist fuer C++ nur naeherungsweise; semantische
  Duplikate (gleiche Logik, andere Typen/Namen) erfasst jscpd nicht.

## Breakdown nach Komponente (Erosion, CC>10)

| Komponente | Funktionen | Erosion |
|---|---|---|
| `apps/` (Runner-Phasen) | 243 | **0.961** |
| `src/io/` (Config/YAML) | 54 | **0.949** |
| `src/astrometry/` (PCC) | 66 | **0.913** |
| `src/image/` (BGE, HMS, Deband) | 161 | **0.889** |
| `src/reconstruction/` (FD/AQMH) | 385 | **0.863** |
| `web_backend_cpp/src/` (Routes) | 783 | **0.863** |
| `src/metrics/` | 61 | 0.847 |
| `src/registration/` | 84 | 0.824 |
| **`src/core/`** | 111 | **0.299** (human-Range, gesund) |

## Top Offenders (nach mass = CC × √SLOC)

Gesamtmasse 240.284; die Top 25 halten **55,3 %** der gesamten
Komplexitaetsmasse. `run_phase_registration_prewarp` allein: **20,4 %**.

| CC | SLOC | mass | Funktion |
|---|---|---|---|
| 777 | 3991 | 49.087 | `runner::run_phase_registration_prewarp` (apps/runner_phase_registration.cpp) |
| 376 | 1368 | 13.907 | `routes::register_pi_routes` (web_backend_cpp/src/routes/pi_routes.cpp) |
| 291 | 692 | 7.655 | `config::Config::from_yaml` (src/io/config.cpp) |
| 224 | 457 | 4.789 | `config::Config::validate` (src/io/config.cpp) |
| 201 | 807 | 5.710 | `register_tools_routes` (web_backend_cpp/src/routes/tools_routes.cpp) |
| 190 | 844 | 5.320 | `register_runs_routes` (web_backend_cpp/src/routes/runs_routes.cpp) |
| 180 | 877 | 5.331 | `routes::register_ai_routes` (web_backend_cpp/src/routes/ai_routes.cpp) |
| 165 | 884 | 4.906 | `image::apply_background_extraction` (src/image/background_extraction.cpp) |
| 161 | 1061 | 5.244 | `runner::run_rgb_downstream` (apps/runner_downstream.cpp) |
| 141 | 900 | 4.230 | `runner::run_forward_drizzle_stages` (apps/runner_forward_drizzle.cpp) |
| 126 | 596 | 3.076 | `runner::run_phase_channel_split_normalization_global_metrics` |
| 118 | 333 | 2.153 | `ForwardDrizzleV2CpuKernel::accumulate_frame_impl` |
| 109 | 359 | 2.065 | `register_scan_routes` (web_backend_cpp) |
| 104 | 622 | 2.594 | `run_pipeline_command` (apps/runner_pipeline.cpp) |
| 103 | 201 | 1.460 | `read_run_status` (web_backend_cpp/src/services/run_inspector.cpp) |
| 98 | 286 | 1.657 | `reconstruction::attempt_backend` (forward_drizzle_v2_dispatch) |
| 97 | 386 | 1.906 | `astrometry::run_pcc` (src/astrometry/photometric_color_cal.cpp) |
| 93 | 246 | 1.459 | `register_config_routes` (web_backend_cpp) |
| 91 | 345 | 1.690 | `image::finalize_bge_from_channel_models` (src/image/autobge.cpp) |
| 90 | 306 | 1.574 | `register_preprocessing_routes` (web_backend_cpp) |
| 89 | 222 | 1.326 | `ForwardDrizzleV2CpuKernel::accumulate_affine_piece` |
| 76 | 449 | 1.610 | `runner::run_preprocess_pipeline` |
| 71 | 350 | 1.328 | `compute_geometric_coverage` (src/registration/sampling_geometry.cpp) |
| 67 | 408 | 1.353 | `persist_forward_drizzle_v2_from_predecessors` |
| 57 | 466 | 1.231 | `run_preprocess_postprocess` (apps/runner_preprocess.cpp) |

Verteilung: 44 Funktionen mit CC>50, davon 16 mit CC>100.

## Interpretation

**Verbosity (0.056):** Unter dem Human-Median — der Code ist nicht
aufgeblaeht. Clone-Ratio 3.83 % ist niedrig; die grossen Refactorings des
Forward-Drizzle-Cutovers haben Duplikate offenbar mit entfernt.

**Erosion (0.900):** Kritisch — 90 % der gewichteten Komplexitaet liegt in
Funktionen mit CC>10. Das Muster ist konsistent ueber fast alle
Komponenten: grosse monolithische Funktionen, die Parsing, Validierung,
Orchestrierung und Fachlogik in einem Block mischen. Typische Treiber:

- **Runner-Phasen** (`apps/`): `run_phase_registration_prewarp` mit CC 777
  / 3991 SLOC ist eine 4.000-Zeilen-Funktion und der groesste einzelne
  Komplexitaetsblock des Repos (20 % Gesamtmasse).
- **Route-Registrierung** (`web_backend_cpp`): `register_*_routes`-Funktionen
  mit CC 90–380 — pro Endpunkt inline-Handler mit eigener
  Fehlerbehandlung/Validierung.
- **Config** (`src/io/config.cpp`): `from_yaml` (CC 291) + `validate`
  (CC 224) — flache Schluessellisten als lange if-Ketten statt
  tabellengetriebener Parser.
- **Fachlogik**: `apply_background_extraction` (CC 165), `run_pcc` (CC 97),
  `finalize_bge_from_channel_models` (CC 91) — Mehrpass-Verarbeitung mit
  Guards/Fallbacks inline.

Ausnahme `src/core` (0.299): kleine, gut verteilte Hilfsfunktionen —
zeigt, dass die Basisabstraktionen gesund sind.

## Empfehlungen (Refactoring-Prioritaeten)

1. **`run_phase_registration_prewarp` zerlegen** (CC 777 → Ziel <50 je
   Teilfunktion): Frame-Loop, CUDA/CPU-Backend-Wege, Cache-Persistenz,
   Fortschritts-Reporting und Fehlerpfade in benannte Teilfunktionen.
   Groesster einzelner Hebel: allein diese Funktion reduziert Erosion um
   ~0,15–0,20.
2. **Route-Registrierung tabellengetrieben machen**: Handler-Signatur
   vereinheitlichen, Endpunkte als Tabelle (Pfad, Methode, Handler,
   Validierung) registrieren — betrifft ~10 Funktionen mit je CC 90–380
   im web_backend.
3. **Config-Parser tabellengetrieben**: `from_yaml`/`validate` auf eine
   Key→{Getter,Setter,Range,Default}-Tabelle umstellen; reduziert auch
   das Risiko divergenter Defaults (vgl. `use_dark`-Fall).
4. **Fachfunktionen nach Passes zerlegen**: `apply_background_extraction`,
   `run_pcc`, `finalize_bge_from_channel_models`, `accumulate_frame_impl`
   — Guards und Kandidaten-Passes als eigene Funktionen.
5. **Leitplanke**: neuen Code auf CC≤15 je Funktion begrenzen (lizard
   kann via `-C 15` in CI warnen), damit Erosion nicht weiter steigt.

Geschaetzte Wirkung: Punkte 1–3 allein (Top-5-Funktionen) binden ~40 %
der Gesamtmasse und wuerden Erosion von 0.90 Richtung 0.55–0.65 bringen —
immer noch ueber Human-Niveau, aber im Agent-Band statt darueber.

## Einschraenkungen

- AST-Grep-Regeln des Papers sind Python-spezifisch; hier via
  grep/jscpd-Proxies — tatsaechliche Verbosity kann hoeher liegen.
- CC>10 ist auf Python kalibriert; fuer C++ wurden zusaetzlich CC>15/20
  berichtet (Ergebnis bleibt kritisch).
- Bildverarbeitungspipelines haben inhärent viele Branches (Config-Pfade,
  Format-Checks); der Vergleich mit generischen Python-Repos ist nur
  bedingt fair — die interne Verteilung (Top 25 = 55 % Masse) ist das
  belastbarere Signal.
