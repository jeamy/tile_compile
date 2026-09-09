# CFA-Forward-Drizzle: Entwicklungsprotokoll

Historische Revisionen und Implementierungsnachweise vom 2026-09-01 bis
2026-09-07. Aus dem Implementierungsplan am 2026-09-07 ausgelagert.

**Gültigkeit:** Dieses Protokoll dokumentiert damalige Befunde, Entscheidungen,
Testzahlen und Interpretationen. Spätere Einträge können frühere widerlegen.
Abschlussbehauptungen und offene Punkte sind keine aktuelle Statusauskunft.
Maßgeblich ist ausschließlich der [aktuelle Implementierungsplan](aqmh_cfa_forward_drizzle_multiband_implementierungsplan_de.md#plan-0).
Die Eintragstexte sind unverändert erhalten; historische Abschnittsnummern
dienen als Beleg-IDs und werden nicht neu vergeben.

**Lesereihenfolge:** Revision 01.09. → Grundentscheidungen 02.09. →
Implementierung 03.–04.09. → Audit 05.09. → Fortschritt 05.–07.09.
Innerhalb des 05.09. ist die Reihenfolge thematisch, keine stundengenaue Chronik.
Die frühere Empfehlung zum ersten Implementierungsschnitt steht am Ende als
historischer Ausgangsplan.

| Zeitraum / Thema | Einträge |
|---|---|
| Konsistenzprüfung 01.09. | [§29](#historie-29) |
| Korrekturen und Nachweisregeln 02.09. | [§30.2](#historie-30-2), [§30.3](#historie-30-3) |
| Grundentscheidungen 02.09. | [§31](#historie-31) |
| Grundlagen und Geometrie 03.–04.09. | [§30.4–30.11](#historie-30-4) |
| Audit und Store-/Runner-Verträge 05.09. | [§0.1–0.5](#historie-0-1) |
| CPU, Q-Maps, Mehrband und CUDA 05.–07.09.; M8-Start 08.09.; M9-Start 08.09.; §11.14 P0–P2 + P3 Teil 1 + P4-Analyse 08.09.; P3 Teil 2 + Runner-Scheduler + P5-Profil 09.09. | [§30.12–30.69](#historie-30-12) |
| Ursprünglicher erster Implementierungsschnitt | [§28](#historie-28) |

Historische Querverweise auf §30.1 meinen die damalige Statustabelle;
deren aktuelle Fassung steht jetzt in [Plan §0.1](aqmh_cfa_forward_drizzle_multiband_implementierungsplan_de.md#status).
Die frühere Entscheidungsrevision §32 ist in die Fachkapitel integriert:
[CUDA §19.6](aqmh_cfa_forward_drizzle_multiband_implementierungsplan_de.md#entscheidung-cuda),
[Evidenz §15.5](aqmh_cfa_forward_drizzle_multiband_implementierungsplan_de.md#entscheidung-evidenz),
[Rauschen §15.6](aqmh_cfa_forward_drizzle_multiband_implementierungsplan_de.md#entscheidung-rauschen),
[Abnahme §23.1](aqmh_cfa_forward_drizzle_multiband_implementierungsplan_de.md#entscheidung-abnahme),
[Ressourcen §11.13](aqmh_cfa_forward_drizzle_multiband_implementierungsplan_de.md#ressourcen-restarbeit).

---

<a id="historie-29"></a>

## 29. Revisionsnotiz (Konsistenzprüfung 2026-09-01)

Folgende Fehler, Lücken und Widersprüche der Erstfassung wurden korrigiert.
Die Zielarchitektur ist unverändert; geändert wurden Zahlen, Definitionen und
Vertragsdetails.

**Zahlen (Abschnitt 2.1):**

- M31-Verhältnis 4,28 / 3,06 ist 1,40, nicht 1,38; die Aussage „28–38 %“ war
  mit keiner der beiden Messungen vereinbar und lautet jetzt 40–50 %.
- Der Unterschied 3,97 px (Bisektion) vs. 4,04 px (Objektübersicht) für M16
  ist als Methodenunterschied gekennzeichnet.

**Mathematik/Geometrie:**

- 11.6: `J_f` ist die Jacobi-Matrix der *nativen* Abbildung; die Formel
  `pixfrac^2 * internal_scale^2 * |det J_f|` hätte mit einer „internen“
  Jacobi-Matrix `internal_scale^2` doppelt gezählt.
- 11.11/6.3: Halo `ceil(drop_size) + 1` gilt nur für Translation; bei
  Rotation wächst die Bounding-Box bis Faktor `sqrt(2)`, bei Skalierung
  entsprechend. Neue Untergrenze `ceil(drop_size * sqrt(2)) + 1`, `auto` aus
  dem Sampling-Plan; 6.3 hatte zudem eine andere Untergrenze als 11.11.
- 14.3: Die Regel für `levels = 1, 2` widersprach der Basisregel; die
  Aussage „`alpha = 0` ergibt Raw“ (14.3, 20.4) war falsch, weil der Grobrest
  aus `U` stammt. Korrekt: `X = R - C_R,L + C_U,L`.
- 6.3: Kopplung `levels <= pyramid.scales` durch die tatsächliche
  Abhängigkeit (`levels >= 2` braucht Scale 1) ersetzt.

**Verträge/Lücken:**

- `coverage_gate.min_channel_n_eff` sollte vor der Rekonstruktion prüfen,
  `n_eff` war aber erst nach der Rekonstruktion definiert. Jetzt: geometrisches
  Uniform-`n_eff` aus der Coverage-Passage (9.2, 9.5).
- `min_dither_spread_px` als RMS absoluter Offsets war untauglich (große
  ganzzahlige Offsets ohne Subpixeldiversität hätten bestanden). Jetzt modulo
  CFA-Periode.
- 6.2 hatte keine Semantik für `min_clip_contributors`, `chunk_halo_rows`,
  `clipping.*`, `coverage_gate.*`, `quality.pyramid.scales`,
  `diagnostics.level`; ergänzt.
- 11.2 `ForwardDrizzleConfig` bildete 6.1 nur teilweise ab; vervollständigt.
- 11.8 Schritt 8 definierte kein Ergebnis bei Verletzung von
  `min_fraction`/`min_n_eff`; ergänzt.
- 7.4/18.3: `internal_scale`/`output_scale` im `plan_hash` hätten bei
  Skalenwechsel die skalenunabhängigen Q-Maps invalidiert; Hashdomäne
  präzisiert. 18.3 übersah, dass `internal_scale`/`pixfrac` auch Coverage und
  Masken invalidieren.
- 17.1 nannte den Gewichtsplan *vor* den Source-Q-Maps, obwohl `G_quality`
  den Green-Proxy konsumiert; Reihenfolge korrigiert und mit 5.2 abgeglichen.
- 5.2 verwendete `CFA_FORWARD_DRIZZLE` statt `FORWARD_DRIZZLE`, nannte
  `SAMPLING_GEOMETRY` nicht und führte `MULTIBAND_FUSION`/Kandidatengates wie
  eigene Phasen, obwohl sie kein Resume-Einstieg sind.
- 18.1/18.2 fehlten `common_overlap.json`, `global_quality.json` sowie die
  Einstiege `SAMPLING_GEOMETRY`/`GLOBAL_QUALITY`.
- M3 benötigte source-space Q-Maps, die erst M5 einführte; Minimalumfang in M3
  definiert.
- 1.1/M10/25.11: M10 sollte die test-only Hülle löschen, 25.11 sie einen
  Zyklus behalten; Targetnamen differierten. Vereinheitlicht, M11 ergänzt.

**Namen/Codebasis:**

- `G_aqmh` → `G_quality`, `drizzle_raw_aqmh` → `drizzle_raw` (Namensvertrag
  1.2); `forward_drizzle_raw.fit` → `forward_drizzle_raw_L.fit`; Multiband-RGB
  und `n_eff` pro Profil in 16.2 ergänzt; `stacked_rgb.fits` als bestehender
  Downstream-Eingang benannt.
- `registration::SmoothLocalModel` existiert nicht; der bestehende Typ heißt
  `SmoothLocalWarpModel`.
- 22.3 listete `source_quality_map*`, `reconstruction_validation*` als
  „zu ändernde“ Dateien, die es nicht gibt; als Umbenennungen der
  vorhandenen `aqmh_*`-Dateien gekennzeichnet; fehlende Runnerdateien für
  `SAMPLING_GEOMETRY`, `GLOBAL_QUALITY`, `RECONSTRUCTION_DIAGNOSTICS` ergänzt.
- 25.2: R/B liegen nicht auf einem Quincunx (das ist G), sondern auf dem
  2×2-Blockraster.

**Zum Stand der Revision 2026-09-01 noch offen:**

- Ableitungsregel für `model_prediction_factor(f)` und
  `registration_residual_factor(f)` aus dem bestehenden Registrierungsergebnis
  (Wertebereich `[0, 1]` festgelegt, Formel offen; vor M3).
- Konkrete Statistik für `A_registration` (14.4) und `A_artifact`.
- Rechen-/Peak-RSS-Obergrenze für den Referenzdatensatz (11.11; vor M9).

Diese historischen Restpunkte wurden zunächst in Abschnitt 26 übernommen und
mit der Entscheidungsrevision 31 verbindlich aufgelöst. Offen bleibt ihre
Umsetzung und empirische Verifikation.

---

<a id="historie-30-2"></a>

### 30.2 In dieser Revision korrigierte Fehler und Lücken

- M0/M1 besitzen keinen Rekonstruktionspfad; ab M2 existiert technisch Uniform
  1x. Die frühere Aussage „bis M3 kein Pfad“ widersprach M2.
- OSC und MONO sind der verbindliche Umfang dieses Cutovers; bereits debayertes
  RGB wird fail-closed abgelehnt. MONO-Ausgaben und -Gates sind separat
  festgelegt.
- `RegistrationSamplingPlan` enthält stabile Frameidentität, CFA-Ursprung sowie
  Koordinatenskalierung/-offsets des bestehenden lokalen Warpmodells.
  `model_prediction_factor` und `registration_residual_factor` werden im Plan
  persistiert; der `QualityFrameWeightPlan` ist an denselben Sampling-Hash
  gebunden.
- Affine Droplets sind Parallelogramme; lokale Warps benötigen adaptive
  Subdivision. Die frühere pauschale Parallelogramm-/Jacobianformel war für
  nichtlineare lokale Modelle falsch.
- Oberflächenhelligkeit, physischer Aperturflux, 2x→1x-Flächenmittel und das
  Vorzeichen von Canvas/Crop in CRPIX sind getrennt definiert.
- `chunk_halo_rows` verwendet konsistent den ganzzahligen Sentinel `-1`; ein
  String-`auto` in YAML passte nicht zum C++-Entwurf.
- `common_overlap_required_fraction` fehlte trotz behaupteter vollständiger
  Configliste und ist jetzt Bestandteil des Reconstruction-Roots.
- `SAMPLING_GEOMETRY` schreibt ein eigenes `sampling_geometry.json`. Damit kann
  ein Gatefehler vor `COMMON_OVERLAP` protokolliert werden, ohne ein Artefakt
  einer nie gestarteten Phase vorzutäuschen.
- Modulo-Dither wird zirkulär an mehreren Feldpositionen gemessen. Die spätere
  Entscheidungsrevision 31 stuft es zugunsten direkter Coverage-/`n_eff`-Gates
  zur reinen Diagnose herab.
- Source-Identity-, Sampling-, Coverage-, Q-Map-, Reconstruction- und
  Multibandhashes sind getrennt. Eine Neuregistrierung invalidiert keine
  inhaltlich unveränderten Source-Q-Maps; Frameumsortierung oder CFA-Phasenwechsel
  invalidieren dagegen korrekt.
- `n_eff`, Coverage und Quality-Separation bleiben kanalbezogen; das gemeinsame
  OSC-Alpha verwendet konservativ die Kanalminima, damit G dünne R/B-Abdeckung
  nicht verdeckt.
- Maskierte À-trous-Faltung propagiert level- und profilspezifischen Support.
- U/R/F/M werden über transaktionale Stores und streifenweise Fusion verarbeitet;
  der frühere Vollmatrix-Entwurf widersprach dem RAM-Ziel.
- CUDA-Fehler starten die gesamte uncommitted Phase auf CPU neu. Gemischte
  Backendbilder und sichtbar partielle Outputs sind ausgeschlossen.
- Runtime-Sicherheitsgates und M9-Promotionsgates behandeln N/A explizit:
  eine verpflichtende Sicherheitsmetrik mit N/A macht den Kandidaten ungültig;
  eine Sternmetrik mit N/A erfüllt nie positive Multiband-Promotionsevidenz.
- M9 besitzt jetzt eine verbindliche Datenklassenmatrix und einen reproduzierbaren
  Vertrag für den isolierten PREWARP-Referenzvergleich.
- M10 isoliert die Legacy-Quellen aus dem Produkt; erst M11 löscht sie physisch.
  Dadurch stimmen Titel, Grace-Zyklus und Löschabnahme überein.

<a id="historie-30-3"></a>

### 30.3 Umgang mit verbleibenden empirischen Nachweisen

1. Die algorithmischen Entscheidungen und Ausgangswerte sind in Abschnitt 26
   und Revision 31 festgeschrieben; Implementierer dürfen sie nicht lokal oder
   objektspezifisch umdeuten.
2. Jeder Vertrag erhält mindestens einen synthetischen, zunächst fehlschlagenden
   Test. Widerlegt ein Fixture einen Ausgangswert, wird zuerst der Plan samt
   Hash-/Artefaktversion revidiert und erst danach der Parserdefault geändert.
3. Ein stilles Clampen, eine automatische Scale-/Pixfrac-Umschaltung oder ein
   Zurückfallen auf PREWARP bleibt unzulässig.
4. M9 startet keinen Lauf automatisch. Ohne ausdrückliche Benutzeranforderung
   bleiben reale Pflichtmatrix, Ressourcenbestätigung und M10 gesperrt.
5. Nach jedem Meilenstein werden Status, tatsächlich bestandene Tests und noch
   offene empirische Nachweise in 30.1 aktualisiert; Dateiexistenz oder ein
   kompilierbarer Stub allein gilt nicht als Implementierung.

<a id="historie-31"></a>

## 31. Historische Entscheidungsfixierung (2026-09-02)

Mit dieser Revision sind die zuvor in Abschnitt 26 als offen geführten
Algorithmus- und Produktentscheidungen verbindlich festgelegt:

- direkte Coverage-/`n_eff`-Gates statt eines harten Dither-Proxys;
- feste lokale Subdivisionstoleranzen und 4/4-Support beim 2x→1x-Flächenmittel;
- explizite Scale-Modi ohne Auto, Produktionsdefault `2/1`, gemeinsames
  `pixfrac=0.8` ohne per-Kanal-Variante;
- konkrete Registrierungsfaktoren und ein gleichfarbiger Green-/MONO-Proxy;
- robuste Formeln für `A_artifact`, `A_registration`, abwärtsbegrenzte
  Alpha-Glättung und ein Energieguard ohne Sternkonzentrationsausnahme;
- feste N/A-/Mindeststern- und Promotionsregeln;
- CPU-À-trous in M7; CUDA-À-trous nur als spätere, nachweislich rentable
  Erweiterung;
- getrennte öffentliche Diagnostik und interne Profilcache-Retention;
- messbare RAM-, Temporärdisk- und Durchsatzgates;
- dauerhafter Produktionsdefault `delete_source_cache_after_run=false`;
- M66, IC5070 und ein realer MONO-Datensatz als zusätzliche M9-Nachweise.

Die Werte sind keine objektspezifischen Tuningvorschläge, sondern der
versionierte Ausgangsvertrag. Diese Aussage galt für die damaligen Fragen. Die späteren Auditfragen werden
in §32 entschieden; der aktuelle Implementierungsstatus steht in §23/§30.1.


---

<a id="historie-30-4"></a>

### 30.4 Implementierungsfortschritt (2026-09-03)

**Erledigt und verifiziert — Abschnitt 7 (`RegistrationSamplingPlan`-Datenmodell):**

| Element | Datei | Status |
|---|---|---|
| Typen `SamplingWarpConvention`, `FrameSamplingTransform`, `RegistrationSamplingPlan` (7.1) | `include/tile_compile/registration/registration_sampling_plan.hpp` (neu) | fertig |
| Geprüfte 2×3-Affininversion `invert_affine_2x3` (7.2) | `src/registration/registration_sampling_plan.cpp` (neu) | fertig |
| Geguardete lokale Source→Canvas-Fixpunktinversion `invert_local_source_to_canvas` (7.3) | dito | fertig (M1-Diagnostik/-Ratenzählung folgt in M1) |
| Verlustfreie Serialisierung `serialize_/parse_from_json_string` (7.4) | dito | fertig |
| Kanonischer `compute_plan_hash` (feste Feldreihenfolge, LE, bit-exakte IEEE-754) (7.4) | dito | fertig |
| `evaluate_smooth_local_displacement` aus `global_registration.cpp` exportiert (nötig für 7.3, statt Reimplementierung) | `src/registration/global_registration.cpp`, `.hpp` | fertig |
| Tests: affiner Round-trip, singuläre/reflektierende/OOB-Matrix abgelehnt, verlustfreie Serialisierung, Hashstabilität ggü. Diagnostik, Hashsensitivität ggü. Semantik, 2x-Skalen­invarianz, lokale Inversion konvergiert / bricht deterministisch ab (20.1) | `tests/test_registration_sampling_plan.cpp` (neu, 8 Fälle) | grün |

**Erledigt und verifiziert — `pipeline_contract_version` (M0, 17.1 / 18.1):**

| Element | Datei | Status |
|---|---|---|
| Konstanten + Semantik (`kPipelineContractVersionActive=0` = Legacy/Cutover-in-Arbeit, `…SingleMethod=1` = Zielvertrag) | `include/tile_compile/core/pipeline_contract.hpp` (neu) | fertig |
| `pipeline_contract_version` + `_label` in `run_provenance.json` und im `run_start`-Event | `apps/runner_pipeline.cpp` | fertig |
| Resume-Guard: Legacy-Run wird abgelehnt, sobald das Binary den Single-Method-Vertrag spricht — derzeit **schlafend** (aktive Version 0), aktiviert sich automatisch mit M10 | `apps/runner_resume.cpp` | fertig |

Der Wert ist bewusst `0`, weil die aktive Pipeline noch der Legacy-PREWARP-AQMH-Pfad
ist (kein Fake-`1` auf einem Nicht-Single-Method-Run, §30.3.5).

**Erledigt und verifiziert — Runner-Sperre und Legacy-Referenztarget (M0):**

| Element | Datei | Status |
|---|---|---|
| CMake-Target `tile_compile_legacy_reference` = Runner mit `-DTILE_COMPILE_LEGACY_REFERENCE`, nicht installiert, nur für reproduzierbare Regressions-/Bisektionsläufe (M11 gelöscht) | `CMakeLists.txt` | fertig, baut |
| `PIPELINE_UNAVAILABLE_DURING_CUTOVER`: `tile_compile_runner run` **und** `resume` brechen vor jeder Run-Mutation ab; `tile_compile_legacy_reference` ist ausgenommen; aktiviert sich automatisch bei M10 (Kopplung an `kPipelineContractVersionActive`) | `include/tile_compile/core/pipeline_contract.hpp`, `apps/runner_pipeline.cpp`, `apps/runner_resume.cpp` | fertig, verifiziert (Normal-Runner gesperrt, Legacy-Ref läuft) |
| `preprocess` bleibt bewusst nutzbar (gemeinsame Kalibrations-Infrastruktur, §4.1) | — | Entscheidung dokumentiert |

**Erledigt und verifiziert — Config-Migrationsmodul (M0, §6.5):**

| Element | Datei | Status |
|---|---|---|
| `migrate_legacy_config_node()`: `method`/`*.engine` → `UNKNOWN_LEGACY_KEY` (fail-closed, nie gestrippt); `tile`/`tile_denoise`/`local_metrics`/`synthetic` + entfernte Sub-Keys (`dithering.min_shift_px`, `stacking.method/sigma_clip/…`, `validation.min_tile_weight_variance/require_no_tile_pattern`) → Strip + `WARN` + Report | `include/tile_compile/config/legacy_config_migration.hpp`, `src/io/legacy_config_migration.cpp` (neu) | fertig |
| `ConfigMigrationReport::to_json_string()` → `artifacts/config_migration.json`-Payload | dito | fertig |
| Tests | `tests/test_legacy_config_migration.cpp` (neu, 6 Fälle) | grün |
| `--force-classic` (CLI11 + Fallback-Parser) und `FORCE_CLASSIC`-env entfernt; `getEffectiveMethod()` liefert konstant die einzige Methode | `apps/runner_main.cpp`, `src/io/config.cpp` | fertig |

**Erledigt und verifiziert — Eingabeklassen-Policy und `migrate-config` (M0, §3.1.1 / §6.5):**

| Element | Datei | Status |
|---|---|---|
| `classify_input_for_single_method()`: MONO → accept; OSC + bekanntes Bayer → accept; OSC ohne Bayer → `UNSUPPORTED_INPUT`; bereits debayertes RGB → `UNSUPPORTED_INPUT`; sonst → reject | `include/tile_compile/core/input_class_policy.hpp` (neu) | fertig |
| Verdrahtung im SCAN_INPUT von `run_pipeline_command` (fail-closed, `#ifndef TILE_COMPILE_LEGACY_REFERENCE`) | `apps/runner_pipeline.cpp` | fertig |
| Tests (MONO/alle Bayer/OSC-ohne-Bayer/RGB) | `tests/test_input_class_policy.cpp` (neu, 4 Fälle) | grün |
| `tile_compile_cli migrate-config <in> <out>`: `method`/engine → Fehler `UNKNOWN_LEGACY_KEY` + rc≠0; sonst Strip + Report + bereinigte YAML | `apps/cli_main.cpp` | fertig, verifiziert |

**Erledigt und verifiziert — `reconstruction:`-Konfigurationsvertrag und Legacy-Testtarget (M0, §6.1–6.3):**

| Element | Datei | Status |
|---|---|---|
| `ReconstructionConfig` (+ `Drizzle`/`Clipping`/`CoverageGate`/`Quality`/`Multiband`-Substrukturen) exakt nach §6.1 | `include/tile_compile/config/configuration.hpp` | fertig |
| Parser `node["reconstruction"]` + Serializer `to_yaml` + `ReconstructionConfig::validate()` mit allen Regeln aus §6.3 (in `Config::validate()` verdrahtet) | `src/io/config.cpp` | fertig |
| Fließt durch `tile_compile_cli dump-default-config` und `validate-config` → also auch durch den Backend-Proxy | verifiziert | fertig |
| Tests: Defaults gültig, voller Block round-trip (Parse + `to_yaml`→Parse), 22 einzelne Contract-Verletzungen abgelehnt, `delete_prewarped_cache_after_run` leckt nicht in den neuen Vertrag | `tests/test_reconstruction_config.cpp` (neu, 4 Fälle / 50 Assertions) | grün |
| `tile_compile_legacy_reference_tests`-Target: `test_aqmh_reconstruction.cpp` dorthin verschoben (PREWARP-Kernel-only, §22.3) | `CMakeLists.txt` | fertig |

Der neue Block liegt **parallel** zu `aqmh:`; keine aktive Phase konsumiert ihn
(die Forward-Drizzle-Pipeline entsteht ab M2). Der alte `aqmh:`-Block treibt die
aktive Pipeline weiter bis M10.

Build aller Targets grün. **Hauptsuite: 294 Fälle, 293 bestanden, nur noch 1
vorbestehender Fehlschlag** (`test_acceleration_backend.cpp:254`, Sandbox-GPU-
Artefakt); der zweite vorbestehende Fehlschlag ist ins Legacy-Testtarget
gewandert (18 Fälle, 17 bestanden). +22 neue M0-Fälle gesamt.

**Erledigt und verifiziert — Migrations-Verdrahtung und Schema-Fläche (M0, §6.4/§6.5):**

| Element | Datei | Status |
|---|---|---|
| `Config::from_yaml_text_migrated(text, report)` — sanitize → Load → `migrate_legacy_config_node` (kann `ConfigError` werfen) → `from_yaml` | `src/io/config.cpp`, `configuration.hpp` | fertig |
| `run_pipeline_command` verwendet es (`#ifndef TILE_COMPILE_LEGACY_REFERENCE`; Legacy-Ref lädt verbatim) und schreibt `artifacts/config_migration.json`, wenn die Migration etwas geändert hat | `apps/runner_pipeline.cpp` | fertig (hinter der Runner-Sperre; aktiviert sich mit M10) |
| `reconstruction:`-Property in `tile_compile.schema.json` **und** `tile_compile.schema.yaml` ergänzt (Backend liest `.schema.yaml`); `get-schema` liefert jetzt beide Roots | Schema-Dateien | fertig, verifiziert |

Verifiziert: `tile_compile_runner run` bleibt gesperrt; `tile_compile_legacy_reference run`
lädt `method: aqmh` verbatim und produziert einen Run; `cli migrate-config`
lehnt `method:` weiter ab; `get-schema` ist valides JSON mit `reconstruction`
**und** `aqmh`.

**Erledigt — Backend, M0-Anteil (§17.5):**

| Element | Datei | Status |
|---|---|---|
| Run-Create-Contract prüfen: **kein Methoden-/Engine-Parameter** (der Backend-Run-Create übergibt nur `--config <path>` an den Runner; keine eigene Methodenwahl) | `web_backend_cpp/src/routes/runs_routes.cpp` | bestätigt |
| `read_run_method_local()` löst bei fehlendem `method:`-Schlüssel (neuer Configstil) auf die einzige Methode auf, statt `""` zurückzugeben — hält Phasen-/Resume-Mapping für neue Runs intakt; expliziter Legacy-Wert wird für die read-only History durchgereicht | dito | fertig, Build grün |

Der Backend-Test-Suite-Stand ist **unverändert** (4 vorbestehende Fehlschläge:
`contract`, `memory_guards`, `report_phase_issues` + ein flakiger; nicht
methoden-/rekonstruktionsbezogen, durch die Änderung nicht verschoben).

**Der tiefere §17.5-Anteil ist M8-gekoppelt** und wird dort umgesetzt:
`pi_context_v2.cpp` Method-Fact auf `pipeline_contract_version` umstellen;
`run_inspector.cpp` `normalizePhaseEvent`/`getPhaseOrderForMethod` auf die neue
Phasenfolge (`SAMPLING_GEOMETRY`/`FORWARD_DRIZZLE`) — die es erst ab M2 gibt;
Test-Fakes + `web_backend_cpp_contract` an den Single-Method-Vertrag anpassen.

**Noch offen in M0:**

- `tile_compile.yaml`-Default-Datei (aktuell ein nicht von diesem Vorhaben
  stammender ungespeicherter Stand) und `examples/` um den `reconstruction:`-Block
  ergänzen;
- Renames `aqmh`→`reconstruction` (Restrukturierung, gekoppelt an die
  M2/M3/M6-Substrukturen — kein reiner Schlüsseltausch), `global_metrics`→
  `reconstruction.quality.frame_weights`, Cosmetic-Keys→`calibration.frame_cleanup`.

**Beim Implementieren von Abschnitt 7 gefundene Plan-Lücken (in den Vertrag zu übernehmen):**

1. **7.4 Hashdomäne — abgeleitete `source_to_canvas`-Matrix.** „affine Matrizen"
   (Plural) ist mehrdeutig. Umgesetzt: sowohl `canvas_to_source` (Eingabe) als
   auch die abgeleitete Inverse `source_to_canvas` plus
   `source_to_canvas_affine_valid` fließen in den Hash. Folge: eine Änderung der
   Inversionsroutine invalidiert Caches auch bei semantisch identischem Plan —
   das widerspricht dem Build-ID-Prinzip aus 10.2. 7.4 muss explizit festlegen,
   ob abgeleitete Matrizen gehasht werden.
2. **7.4 Hashdomäne — `chain_depth` / `model_predicted`.** 7.1 speichert sie,
   7.4 listet sie nicht. Umgesetzt: **nicht** gehasht (Wirkung steckt in
   `model_prediction_factor`). Sollte in 7.4 ausdrücklich als diagnostisch
   ausgenommen stehen.
3. **7.3 — Nullband des lokalen Modells.** Der 4×4-Gauss-Basiskern liefert im
   8-%-Randtaper und außerhalb `[0, image-1]` die Verschiebung 0; dort ist die
   lokale Inversion exakt gleich der affinen. Das ist Modellauswertung, kein
   stiller Fallback (7.3 verbietet stillen Fallback). 7.3 sollte das Nullband
   benennen, damit die M1-Diagnostik nicht jeden Randsample als „lokales Modell
   umgangen" meldet.
4. **7.2/7.3 — Determinantengrenzen.** 7.2 sagt „Determinante außerhalb der
   vorhandenen Registrierungsgrenzen"; die vorhandenen Grenzen
   (`reject_scale_min/max`) sind **lineare Skalen**, keine Determinanten.
   Umgesetzt: `det ∈ [reject_scale_min², reject_scale_max²]`, `det ≤ 0`
   (Spiegelung) abgelehnt. 7.2 muss die Quadrat-Abbildung und die
   Spiegelungsablehnung ausdrücklich nennen.
5. **7.1 — Feld `residual_applicable`.** 11.9 braucht es (`false` → Faktor 0,55),
   die Struktur in 7.1 listet es nicht. Umgesetzt: Feld ergänzt, wird
   persistiert und gehasht (semantisch: bestimmt die Faktorherleitung). 7.1 in
   die Struktur aufnehmen.

<a id="historie-30-5"></a>

### 30.5 M1 begonnen: `RegistrationSamplingPlan` real befüllt (2026-09-04)

**Erledigt und real verifiziert:** `write_registration_sampling_plan()` in
`apps/runner_phase_registration.cpp`, additiv unmittelbar nach
Canvas-/Offsetberechnung und vor jeder PREWARP-Bildverzerrung aufgerufen (kein
Verhaltensunterschied für den bestehenden Pfad). Befüllt den Plan aus bereits
vorhandenen Registrierungsergebnissen:

- `canvas_to_source` = die finalen, offsetkorrigierten Warps (`global_frame_warps`);
  `source_to_canvas` via `invert_affine_2x3` mit
  `det ∈ [reject_scale_min², reject_scale_max²]`;
- `provenance`/`chain_depth` aus dem bestehenden `RegistrationProvenance`/
  `reg_chain_depth`; `model_prediction_factor` exakt nach der 11.9-Fallunterscheidung
  (direkt → 1,0; `model_interpolated`/`_blended`/`_global_poly`/`_local_poly` →
  `clamp(1/(1+0,4·depth), 0.5, 0.9)`; `model_nearest_copy` → zusätzlich `min(…, 0.5)`);
- `registration_residual_factor`/`residual_applicable` aus dem bestehenden
  `reg_residual_stats` — `registration_residual_weight_factor()` im bestehenden
  Code entspricht bereits **exakt** der 11.9-Formel (0,18/0,70/0,45/1,40/0,75/0,45,
  Clamp 0,55–1,0); wiederverwendet, nicht neu implementiert;
- `smooth_local_model`/`model_coordinate_scale`/`model_offset_x/y` aus
  `local_refinement_stats[fi]` bzw. dem bestehenden `local_model_coordinate_scale`
  und dem Canvas-Offset.

**Real verifiziert** (Legacy-Referenzbinary, M31, 20 Frames,
`registration_sampling.json`, 31 KB): Canvas 3840×2160→3866×2174, Offset
[12, 8] — deckt sich exakt mit der unabhängig geloggten
„Field rotation detected"-Meldung; `registration_residual_factor` von Frame 0
(0,8821) deckt sich exakt mit dem unabhängig geloggten
`reg_residual_weight_factor_median` aus dem `REGISTRATION`-Phase-Event. Build
(`tile_compile_runner`, `tile_compile_legacy_reference`) und volle Suite grün
(293/294, unverändert).

**Beim Verdrahten gefundene Plan-Lücken:**

1. **7.1 — Herleitung von `cfa_origin_x/y` nicht spezifiziert.** Umgesetzt als
   `0, 0` (keine ROI-/Crop-Verschiebung vor der Normalisierung im bestehenden
   Pfad). Falls ein künftiger Crop-Schritt vor `normalized_frames` eingeführt
   wird, muss diese Stelle die reale Parität liefern — 7.1 sollte die Quelle
   benennen.
2. **7.1 — Herleitung von `frame_id` nicht spezifiziert** über „stabil aus
   Inputmanifest + Inhaltsidentität" hinaus. Umgesetzt als Dateiname (stabil
   innerhalb eines Laufs, da `frames` dieselbe sortierte Reihenfolge wie
   `run_provenance.json` hat) statt Content-Hash, um nicht jedes Rohbild ein
   zweites Mal zu hashen (der Hash existiert bereits in
   `run_provenance.json`). 7.1 sollte festlegen, ob Dateiname ausreicht oder
   Content-Identität über einen Lauf hinweg verbindlich ist.

Noch nicht abgedeckt in diesem M1-Schnitt: `SAMPLING_GEOMETRY` als eigene
Phase, geometrische Coverage ohne Bild-PREWARP, `analysis_common_mask`,
`coverage_gate`-Auswertung, `sampling_geometry.json`. PREWARP läuft
unverändert weiter.

<a id="historie-30-6"></a>

### 30.6 M1 fortgesetzt: geometrische Coverage ohne Bild-PREWARP (2026-09-04)

**Erledigt und mit synthetischen Fixtures getestet** — als eigenständiges,
noch **nicht in den Runner verdrahtetes** Modul (analog zum Vorgehen bei
Abschnitt 7: erst Datenmodell/Algorithmus + Tests, Verdrahtung folgt separat):

| Element | Datei | Status |
|---|---|---|
| `compute_geometric_coverage()`: Vorwärtsabbildung Quelle→Canvas (affin oder über die bestehende `invert_local_source_to_canvas`-Fixpunktiteration), achsparalleles Droplet-Footprint, Kanalzuordnung über das bestehende `get_bayer_offsets()` (keine Duplikation der CFA-Logik) | `include/.../sampling_geometry.hpp`, `src/registration/sampling_geometry.cpp` (neu) | fertig |
| `analysis_common_mask` / `reconstruction_support_mask`, `coverage_gate`-Auswertung (`min_frames`, `min_analysis_pixels`, `min_supported_fraction`, `min_channel_n_eff` p10) | dito | fertig |
| `serialize_sampling_geometry_json()` — Artefaktschema aus 9.4 | dito | fertig |
| Tests: MONO Identität volle Abdeckung; zwei Frames mit Teilüberlappung (Masken unterscheiden „gemeinsam" von „irgendein Frame"); OSC RGGB R/G/B-Dichteverhältnis exakt (64/128/64 auf 16×16); R und B nie am selben Pixel; Gate lehnt zu wenige Frames ab; Lochprüfung meldet sich explizit als nicht implementiert statt still zu bestehen; 2x-Skalierung verdoppelt die interne Canvas; JSON-Serialisierung | `tests/test_sampling_geometry.cpp` (neu, 7 Fälle / 1492 Assertions) | grün |

Build (alle Targets) und volle Suite grün (300/301, der eine vorbestehende
GPU-Fehlschlag unverändert; +7 neue Fälle).

**Bewusste Vereinfachungen/Annahmen dieses Schnitts (im Code kommentiert):**

1. Das Coverage-Droplet ist **achsparallel** im internen Raster — ein
   konservativer Abdeckungstest, nicht das exakte rotierte Parallelogramm aus
   11.6. Die photometrische Wertakkumulation mit exakter Fläche ist eine
   separate M2-Aufgabe mit eigenem Kernel.
2. **Kein Chunking** (9.2 „nur betroffene Zielchunks besuchen" ist nicht
   umgesetzt) — volle Quellpixel-Iteration pro Frame. Für die Tests
   unproblematisch; vor einer Runner-Verdrahtung auf realen Framegrößen zu
   profilieren.
3. `analysis_common_mask`/`reconstruction_support_mask` werden als
   **Minimum über die aktiven Kanäle** gebildet (dichtes G darf dünnes R/B
   nicht verdecken) — eine dokumentierte Annahme in Analogie zu 14.4, keine
   wörtliche Plan-Vorgabe für diese Stelle.
4. `min_supported_fraction`/`min_channel_n_eff_p10` verwenden die
   **Vereinigung der Kanal-Supportflächen** als Nenner der Analyseregion —
   ebenfalls eine dokumentierte Annahme, da der Plan diesen Nenner auf
   Implementierungsebene nicht festlegt.
5. `max_internal_hole_area_px` (topologische Lochprüfung) ist **nicht
   implementiert**; das Gate-Ergebnis meldet dies explizit
   (`hole_check_implemented=false`) statt fälschlich „keine Löcher" zu
   melden.

**Noch offen:** Verdrahtung als eigene Phase `SAMPLING_GEOMETRY` im Runner
(ersetzt die COMMON_OVERLAP-Coverage-Ableitung aus PREWARP-Pixeln), Chunking
für reale Framegrößen, `sampling_geometry.json` als echtes Laufartefakt,
Lochprüfung.

<a id="historie-30-7"></a>

### 30.7 M1 fortgesetzt: Coverage-Diagnostik real auf dem Runner verdrahtet (2026-09-04)

**Erledigt und auf einem echten Lauf verifiziert** (weiterhin additiv, noch
nicht gate-wirksam): `write_sampling_geometry_diagnostic()` ruft
`compute_geometric_coverage()` mit den echten `reconstruction.drizzle.*`- und
`reconstruction.coverage_gate`-Werten aus der M0-Config auf und schreibt
`artifacts/sampling_geometry.json` unmittelbar nach dem
`registration_sampling.json`-Schritt. Ein fehlschlagendes Gate wird geloggt,
**nicht** erzwungen — COMMON_OVERLAP und PREWARP laufen unverändert weiter.

**Realer Lauf** (Legacy-Referenzbinary, M31, 20 Frames, Produktionsdefault
`internal_scale=2`): internes Raster 7732×4348, 20 gültige Frames,
Analysefläche 33,57 Mio. Px, `min_supported_fraction=0,9985`,
`min_channel_n_eff_p10=5,0`, Gate **bestanden**. Build (`tile_compile_runner`,
`tile_compile_legacy_reference`) und volle Suite grün (300/301, unverändert).

**Gefundene Grenze (kein Fehler, aber ein Datenpunkt für die spätere
Verdrahtung als scharfes Gate):** Laufzeit **~7 s für 20 Frames** bei 2x/OSC,
einzelnstreifig, ohne Chunking/Parallelisierung (30.6, Punkt 2). Hochgerechnet
auf einen 100–600-Frame-Produktionslauf sind das grob 35 s–3,5 min als
**zusätzlicher** Diagnoseschritt oben auf die bestehende PREWARP-Zeit. Für den
reinen Diagnosezweck akzeptabel; vor einer Verdrahtung als scharfes,
run-blockierendes Gate sollte das parallelisiert werden (derselbe
`compute_adaptive_worker_count`-Mechanismus, den PREWARP bereits nutzt).

<a id="historie-30-8"></a>

### 30.8 M1: `SAMPLING_GEOMETRY` als echte, scharf getrennte neue Phase (2026-09-04)

Auf ausdrücklichen Wunsch vorrangig die **neue** Version fertigzustellen, klar
getrennt von der alten: `run_phase_sampling_geometry()` ist jetzt keine
Diagnose mehr, sondern die reale Phase aus §8.2/§9.5. Der `coverage_gate`
**beendet den Lauf fail-closed**, exakt wie §9.5 verlangt — aber ausschließlich
auf dem neuen Pfad. Umsetzung: der Abbruchzweig steht unter
`#ifndef TILE_COMPILE_LEGACY_REFERENCE`; das eingefrorene Legacy-Binary läuft
bei fehlgeschlagenem Gate unverändert weiter (reine Protokollierung, keine
Verhaltensänderung) — die beiden Pipelines sind damit nicht nur logisch,
sondern **compile-zeitlich** getrennt.

**Verifiziert — dreifach:**

1. **Compile-Trennung nachgewiesen auf Binärebene:** `strings` auf beiden
   Binaries — die Abbruchmeldung „no silent fallback" ist **ausschließlich**
   in `tile_compile_runner` vorhanden, im `tile_compile_legacy_reference`
   korrekt wegoptimiert.
2. **Legacy-Pfad unverändert:** echter Lauf (M31, 20 Frames) mit absichtlich
   zu strengem `coverage_gate.min_frames: 999` über
   `tile_compile_legacy_reference` — Gate meldet korrekt
   `gate_passed=no … min_frames: 20 < 999`, PREWARP läuft danach unbeirrt
   weiter („Field rotation detected …", „Using 6 parallel workers …").
3. **Build (alle Targets) und volle Suite grün** (300/301, unverändert).

Die entsprechende Abbruchlogik auf dem neuen Pfad (`tile_compile_runner`) ist
bis M2 nicht end-to-end auslösbar, weil `PIPELINE_UNAVAILABLE_DURING_CUTOVER`
den Lauf schon vorher stoppt — genau die beabsichtigte Reihenfolge. Der
Codepfad selbst ist damit auf zwei Arten abgesichert: Unit-Tests auf
`compute_geometric_coverage`/`GeometricCoverageResult` (30.6) und der
bitidentische, nur im Präprozessor-Zweig unterschiedliche Quellcode zum
verifiziert korrekten Legacy-Zweig.

<a id="historie-30-9"></a>

### 30.9 M1: Lochprüfung + Parallelisierung implementiert und verifiziert (2026-09-04)

Beide zuvor offenen Punkte aus 30.6 sind jetzt umgesetzt:

**Lochprüfung (`max_internal_hole_area_px`).** 4-Wege-Flutfüllung vom
Canvasrand über die unbelegten Pixel von `reconstruction_support_mask`
(„außen"); nicht erreichte unbelegte Pixel sind innere Löcher,
zusammenhangskomponenten-markiert, größte Fläche gemeldet. `gate.passed`
berücksichtigt jetzt `largest_internal_hole_area_px > max_internal_hole_area_px`.

**Parallelisierung.** Die Pro-Frame-Rasterisierung ist über Worker-Threads
partitioniert (Frames in zusammenhängende Teillisten, private Akkumulatoren
pro Worker, anschließende elementweise Summation). Mathematisch unbedenklich:
Ganzzahladdition über Frames ist kommutativ/assoziativ, das Ergebnis ist
**bitidentisch unabhängig von Workerzahl und Ablaufplanung** — durch einen
dedizierten Test (1 vs. 4 vs. 7 Worker auf demselben 5-Frame-Fixture,
identische Support-Arrays, Masken und Gate-Werte) verifiziert.

**Tests (6 neue Fälle):** Ring-Loch erkannt, randberührende unbelegte Fläche
korrekt **nicht** als Loch gezählt, größte von mehreren Löchern ausgewählt,
voll belegte/unbelegte Maske liefert 0, Gate-Verletzung bei Lochüberschreitung,
Parallel-Determinismus.

**Realer Fund auf M31 (20 Frames, Produktionsdefault 2x/`pixfrac=0.8`):** Die
Lochprüfung findet auf echten Daten ein **reales 3-Pixel-Loch**
(`largest_internal_hole_area_px: 3`), der Default-Gate
(`max_internal_hole_area_px: 0`) lehnt entsprechend ab. Das ist keine
Fehlfunktion, sondern die empirische Bestätigung von Abschnitt 25.2/9.5: bei
nur 20 Frames und `pixfrac=0.8` auf 2x-Raster entstehen im dünn belegten
R/B-Kanal echte Sub-Pixel-Lücken. Für die spätere M9-Kalibrierung der
Gate-Defaults ist das ein konkreter Datenpunkt (mehr Frames oder größeres
`pixfrac` nötig, um dieses Gate bei kleinen Framezahlen zu bestehen). Laufzeit
mit Lochprüfung + Parallelisierung: ~6,0 s für 20 Frames (vorher ~7,0 s ohne
Lochprüfung, einzelstreifig) — die zusätzliche Lochprüfung kostet also real
weniger, als die Parallelisierung einspart.

Build (alle Targets) und volle Suite grün (305/306, der eine vorbestehende
GPU-Fehlschlag unverändert; +5 netto neue Fälle, siehe oben).

---

<a id="historie-30-10"></a>

### 30.10 M1: Masken als FITS persistiert + `analysis_common_mask`-Befund geklärt (2026-09-04)

> **Überholt (Audit 2026-09-05, siehe §0/§0.1, Befund A3):** Die hier
> beschriebene `analysis_common_mask`-Semantik (kanalweiser CFA-Droplet-Schnitt
> mit `common_overlap_required_fraction`) existiert im aktuellen Code nicht
> mehr. `analysis_common_mask` ist jetzt eine dichte, CFA-unabhängige
> Frame-Footprint-Überlappung (`sampling_geometry.cpp`, `pixfrac=1.0`). Die
> untenstehende Empfehlung „Default absenken" ist **zurückgenommen**: bei einer
> echten Schnittmenge aller Frame-Footprints verkleinert jeder zusätzliche
> Frame die Fläche monoton, mehr Frames „heilen" sie nicht (§0). Abschnitt zur
> historischen Nachvollziehbarkeit belassen, nicht mehr maßgeblich.

Der letzte offene Punkt aus 30.6/30.7 („`analysis_common_mask`/
`reconstruction_support_mask` noch nicht als FITS persistiert") ist umgesetzt:
`run_phase_sampling_geometry()` schreibt beide Masken jetzt als
0/1-wertige FITS-Float-Bilder in Internal-Canvas-Auflösung nach
`artifacts/sampling_geometry_analysis_common_mask.fits` und
`artifacts/sampling_geometry_reconstruction_support_mask.fits` (eigener
lokaler Writer `write_sampling_geometry_mask_fits()`, spiegelt die
bestehende `write_canvas_mask_fits()`-Konvention aus `runner_pipeline.cpp`).

**Realer Befund und Verifikation.** Ein frischer Lauf auf den gleichen 20
M31-Frames (Produktionsdefault, `common_overlap_required_fraction` nicht im
Config gesetzt → Default `1.0`) zeigt: `reconstruction_support_mask` ist wie
erwartet zu ~99,6 % belegt (0/1-wertig, Teilmenge korrekt größer als
`analysis_common_mask`), aber `analysis_common_mask` ist **über die gesamte
33,5-Mio.-Pixel-Canvas exakt null** (`unique values: [0.]`).

Das wurde nicht ungeprüft als Bug hingenommen, sondern mathematisch
gegengerechnet: `analysis_common_mask[i]=1` verlangt
`support_count ≥ ceil(common_overlap_required_fraction · valid_frame_count)`
**gleichzeitig auf jedem aktiven Kanal**. Bei `common_overlap_required_fraction
= 1.0` und `valid_frame_count = 20` ist das `required_common = 20` — jedes
einzelne Pixel müsste von **allen 20** Frames auf R **und** G **und** B
getroffen werden. Bei einer OSC-Bayer-CFA mit ~16 % Flächendichte pro Kanal
und Frame (R/B je 1 von 4 Sites) und echter Dither-/Rotations-Variation
zwischen den 20 Frames ist es praktisch ausgeschlossen, dass irgendein
Internal-Pixel von allen 20 R-Drop­lets gleichzeitig getroffen wird — ein
exakt-null-Ergebnis ist also die **korrekte** Konsequenz der strengen
Default-Schwelle, kein Rechenfehler.

Zur Gegenprobe wurde derselbe Lauf mit `reconstruction.common_overlap_required_fraction:
0.5` wiederholt (`required_common = 10` statt `20`, alle anderen Parameter
identisch): `analysis_common_mask` wird dabei zu **5,2 %** belegt
(0/1-wertig, weiterhin korrekt). Das bestätigt zwei Dinge zugleich: (1) die
Config wird korrekt bis in `compute_geometric_coverage()` durchgereicht, kein
Parsing-Bug; (2) die Maske reagiert **monoton** auf die Schwelle (strenger →
weniger Fläche, lockerer → mehr Fläche) wie mathematisch gefordert.

**Einordnung für M9-Kalibrierung.** Das ist ein weiterer konkreter
Datenpunkt neben dem 3-Pixel-Loch aus 30.9: der Default
`common_overlap_required_fraction: 1.0` ist für reale OSC-Datensätze mit
wenigen Frames (20) faktisch nie erfüllbar und macht `analysis_common_mask`
für Analysezwecke, die auf „mindestens ein Frame" (`reconstruction_support_mask`)
statt „alle Frames" abzielen, unbrauchbar leer. Für produktive Nutzung von
`analysis_common_mask` (z. B. als konservative Referenzfläche für
Qualitätsvergleiche) muss der Default entweder gesenkt werden (z. B.
`0.5`–`0.7`) oder die Semantik im Plan (§9.3) präzisiert werden: „alle
Frames" ist nur bei sehr großen Framezahlen (Hunderte) und/oder MONO-Daten
(volle Flächendichte pro Frame) praktisch erreichbar. Der bestehende
Plantext in §9.3 bleibt fachlich korrekt (er beschreibt exakt dieses
Verhalten), verdient aber eine Ergänzung, dass der Default für kleine
OSC-Runs bewusst restriktiv ist und in M9 kalibriert werden muss — ergänzt.

Testverzeichnisse (`/tmp/m1fits*`) nach Verifikation entfernt.

Build (alle Targets) unverändert grün; keine Code-Änderung nötig, nur
FITS-Persistenz ergänzt und der Befund empirisch verifiziert (kein Bug).

---

<a id="historie-30-11"></a>

### 30.11 M1: zirkuläre Ditherdiagnostik implementiert und verifiziert (2026-09-04)

Die in §9.3/8.x beschriebene, aber bislang nicht implementierte zirkuläre
Dither-Streuungsdiagnostik ist jetzt umgesetzt: `compute_dither_spread_circular_diagnostic()`
in `sampling_geometry.cpp/hpp` wertet an 5 nativen Canvas-Stellen (Mitte + 4
Ecken) über alle validen Frames `canvas_to_source` direkt aus (keine
Inversion nötig, `canvas_to_source` ist per Konvention schon auf
Native-Canvas-Koordinaten definiert, §7). Pro Stelle und Achse: `theta = pi *
(s mod 2)`, mittlerer Resultantenvektor `R = |mean(e^{i·theta})|` über die
Frames, `sigma_circ_px = sqrt(-2·ln(R)) / pi` (Rayleigh-Schätzer für
zirkuläre Streuung, `R` auf `[1e-12, 1-1e-12]` geklemmt gegen `ln(0)`/`ln(1)`).
`x_p10`/`y_p10` sind das 10.-Perzentil (konservativstes = am wenigsten
diverses der 5 Stellen) je Achse. Reines Diagnosefeld, geht **nicht** in
`gate.passed` ein — exakt wie im Plan gefordert (ein Dither-Mod-2-Proxy kann
bei Rotation/lokalen Warps falsch entscheiden, die direkt rasterisierte
Kanalcoverage bleibt maßgeblich).

**Tests (3 neue Fälle, mathematisch exakt verifizierbar statt nur
plausibilisiert):**
- identische Frames (kein Dither) → `R=1` exakt → `sigma≈0` (bis auf die
  `1e-12`-Klemmung gegen `ln(1)`, Toleranz entsprechend gesetzt);
- 4 Frames mit Phasen 0/0,5/1,0/1,5 px → `theta` exakt auf 0/π/2/π/3π/2
  verteilt → perfekt uniform auf dem Kreis → `R=0` exakt → `sigma` erreicht
  die (klemmungsbedingte) Obergrenze `sqrt(-2·ln(1e-12))/π`;
- keine validen Frames → `0.0`, nicht `NaN`.

**Realer Lauf (M31, 20 Frames, Produktionsdefault):**
`dither_spread_circular_px_p10=(0.484, 0.597)` — plausibler Zwischenwert
zwischen den beiden Extremfällen der Tests (echtes Dither ist weder perfekt
entartet noch perfekt uniform), im JSON-Artefakt
(`dither_spread_circular_px_diagnostic`) und im Log sichtbar, ohne den
Gate-Ausgang zu beeinflussen (`gate_passed` unverändert `no`, weiterhin
wegen des 3-Px-Lochs aus 30.9, nicht wegen der Ditherdiagnostik).

Damit ist von den in 30.9/30.10 offen gelassenen M1-Punkten nur noch
„COMMON_OVERLAP von den geometrischen Masken speisen" übrig — das wird
**bewusst zurückgestellt**: `COMMON_OVERLAP` in `runner_pipeline.cpp` ist
noch der geteilte Legacy-Code (M2 „CPU Forward-Drizzle 1x Uniform-Control"
ist noch nicht gebaut, es gibt also noch keine neue, getrennte
COMMON_OVERLAP-Entsprechung im neuen Pfad). Diesen Punkt jetzt zu erzwingen
würde bedeuten, gemeinsam genutzten Legacy-Code zu verändern — das
widerspricht der expliziten Vorgabe „die neue Version wird getrennt von der
alten, nur die neue Version ist wichtig". Richtig ist, dass die neue
COMMON_OVERLAP-Entsprechung die geometrischen Masken konsumiert, sobald sie
als eigene M2-Phase existiert; bis dahin bleibt der Punkt offen und wird in
§23 entsprechend vermerkt.

> **Überholt (Audit 2026-09-05, siehe §0/§0.1, Befund B3):** Die folgende
> „gegenstandslos"-Einschätzung galt für den damaligen Voll-Canvas-Pro-Worker-
> Algorithmus und war in dieser Form ein echtes RAM-Risiko (vom Audit korrekt
> benannt), nicht bloß eine Fehleinschätzung des Hinweistexts. Der aktuelle
> Code hat das Problem inzwischen anders und vollständig gelöst: `sampling_geometry.cpp`
> und `forward_drizzle.cpp` verarbeiten beide zeilenweise gestreamte Chunks
> (`plan_drizzle_memory`/`stream_forward_drizzle_uniform`) mit geprüftem
> Speicherbudget, nicht mehr vollbildgroße Puffer pro Worker. Der Text unten
> bleibt zur Nachvollziehbarkeit stehen, ist aber durch die reale
> Streaming-Implementierung ersetzt, nicht bestätigt.

Die als offen dokumentierte „canvas-chunk-beschränkte Rasterisierung" aus dem
Implementierungshinweis am Kopf von `sampling_geometry.hpp` wurde ebenfalls
geprüft: Der bestehende Algorithmus iteriert bereits pro Frame nur über
Quellpixel (`O(source_w · source_h)` pro Frame), nicht über die Canvas
(`O(internal_w · internal_h)`) — die im Plan (§9.2) angemahnte Beschränkung
auf „betroffene Zielchunks" ist für dieses Coverage-Modul also bereits
gegenstandslos (die reale Laufzeit von ~6 s für 20 Frames auf einer
7732×4348-Canvas bestätigt das). Der Hinweis bezieht sich fachlich auf die
spätere M2-Wertakkumulation (große Float-Puffer beim eigentlichen
Forward-Drizzle), nicht auf dieses M1-Coverage-Modul; der Hinweistext in
`sampling_geometry.hpp` wurde entsprechend präzisiert.

Build (alle Targets) und volle Suite grün (308/309, der eine
vorbestehende GPU-Fehlschlag unverändert; +3 netto neue Fälle).
Testverzeichnis (`/tmp/m1dither`) nach Verifikation entfernt.

---

<a id="historie-0-1"></a>

### 0.1 Verifikationstabelle zum Audit (2026-09-05, gegen aktuellen Code geprüft)

Jeder Befund des [Audits](aqmh_cfa_forward_drizzle_audit_2026-09-05_de.md) wurde
gegen den tatsächlichen, aktuellen Codestand nachgeprüft (nicht nur gegen den
Audittext übernommen). Belegstellen sind Datei:Zeile zum Prüfzeitpunkt.

| Befund | Verdikt | Beleg |
|---|---|---|
| A1 Coverage nutzt nicht den Rekonstruktionskernel | **behoben** | `sampling_geometry.cpp` ruft jetzt `rasterize_drizzle_stripe()` (`forward_drizzle.cpp`) für Coverage **und** Wertakkumulation auf; derselbe exakte Polygon-Rechteck-Schnitt für beide |
| A2 gemeldetes `n_eff` ist Frame-Anzahl | **behoben** | `compute_geometric_coverage()`: `quantiles[c]->add(w[c][i]*w[c][i]/w2[c][i])` — echtes `(ΣB)²/Σ(B²)` aus den geometrischen Flächengewichten, nicht `support_count` |
| A3 Gate und Analysemaske haben verschiedene Bezugsflächen | **behoben, mit geänderter Semantik** | `analysis_common_mask` ist jetzt eine dichte, CFA-unabhängige Frame-Footprint-Überlappung (`pixfrac=1.0`-Rasterisierung, `footprint_count>=required`); `gate.analysis_pixels`/Supportanteile/`n_eff`-Perzentile werden **in derselben** Fläche ausgewertet (`sampling_geometry.cpp:374-394`). Ersetzt die in §30.10 beschriebene Semantik vollständig, siehe 0. |
| A4 lokale Subdivision erfüllt Abnahme nicht | **behoben** | `subdivide_local()`: Positions- **und** Flächenkriterium gelten jetzt unconditioniert auch an Maximaltiefe (`forward_drizzle.cpp:233-238`); `frame_total`/`frame_discarded` zählen Quellsamples, nicht Subdroplets (`prepare_drizzle_frames`); ein fehlgeschlagenes Kind verwirft das ganze Sample (`leaves.resize(before); return false;`). Getestet inkl. echter Krümmung und „Position besteht, Fläche verletzt" (`test_forward_drizzle.cpp`, „local rejection counts source samples once") |
| A5 Pixelzentrum-Adapter fehlt | **behoben** | `opencv_to_edge_sampling_map()` (`registration_sampling_plan.cpp:38-42`): `t_edge = t_cv + (0.5,0.5) - A·(0.5,0.5)`, exakt die im Audit geforderte Form; verdrahtet in `runner_phase_registration.cpp` vor `invert_affine_2x3` |
| A6 Persistenz/Resume kein belastbarer Vertrag | **teilweise behoben, weiter fortgeschritten seit 0.1** | `AtomicOutput` (stage-dir + fsync + rename) ersetzt direktes Trunkieren für `core::write_text`, FITS-Schreiben (inkl. `write_fits_float()`, dort erst in 30.14 nachträglich gefixt — siehe dort) und den Coverage-Quantil-Spool; `coverage_geometry_hash` bindet jetzt Kernel/Pixfrac/Scale/`common_fraction`/Subdivisionsparameter (`compute_coverage_geometry_hash`); `frame_id` ist jetzt `source_identity_hash:index` mit `source_identity_hash = sha256(input_manifest.sha256 + config.sha256)` statt Dateibasisname. Neu (30.14): erster echter Schritt zum transaktionalen M2-Profilstore (`write_forward_drizzle_uniform_store()`, atomare Ebenendateien, real auf M31 verifiziert). **Weiterhin offen:** Store-weite (nicht nur Datei-weise) Transaktionalität über alle Ebenendateien, eigene Phasen-/Store-Resume-Einstiege, strikter Ablehnungsvertrag für beschädigte Artefakte über den reinen Parse-Erfolg hinaus, `write_fits_rgb()`/`write_fits_rgb_u32()` weiterhin nicht-atomar. Die im Audit zitierte Überakzeptanz `pipeline_contract_is_single_method(v>=1)` ist **gegen den aktuell gelesenen Code nicht reproduzierbar**: `pipeline_contract.hpp:39-41` prüft exakte Gleichheit (`v == kPipelineContractVersionSingleMethod`), keine Ungleichung — möglicherweise bezog sich der Auditbefund auf einen anderen Codestand |
| B1 „mehr Frames heilen leere Schnittmaske" | **zurückgenommen** (Audit korrekt) | Direkt in §0 zurückgenommen; durch die A3-Neusemantik ohnehin gegenstandslos, da `analysis_common_mask` keine reine CFA-Kanal-Schnittmenge mehr ist |
| B2 Ursachenanalyse zu absolut formuliert | **präzisiert** | §0 stuft die 19–25-%-Prognose explizit als Hypothese ein, nennt die fehlende Isotropie-/Bias-Abgrenzung |
| B3 M1 hat RAM-Problem | **behoben** | `plan_drizzle_memory()` + zeilenweises Streaming (`stream_forward_drizzle_uniform`, `DiskQuantile`, `StripeHoles`) ersetzen die vollbildgroßen Pro-Worker-Puffer; Speicherbudget wird vor jedem Store-Write geprüft (`DRIZZLE_MEMORY_BUDGET`-Fehler), Host-/cgroup-Headroom einbezogen |
| B4 Legacy-Isolation/Phasenstatus überzeichnet | **teilweise behoben** | `TILE_COMPILE_BUILD_LEGACY_REFERENCE` jetzt Default `OFF` (`CMakeLists.txt:564-565`). **Weiterhin offen, geprüft und bewusst nicht oberflächlich gefixt:** `Phase`-Enum (`core/types.hpp:193ff`) hat noch keinen `SAMPLING_GEOMETRY`-Wert; ein fehlschlagendes Gate meldet weiterhin `Phase::PREWARP`. Ein Enum-Wert allein wäre trivial anzuhängen (am Ende, um bestehende Ganzzahlwerte nicht zu verschieben), aber `emitter.phase_start(run_id, Phase::PREWARP, ...)` läuft bereits **vor** dem SAMPLING_GEOMETRY-Block (`runner_phase_registration.cpp:4687` vs. Coverage-Aufruf ~4796) — ein `phase_end`/`error` mit einem Phasenwert zu melden, für den nie `phase_start` lief, wäre ein inkonsistenter Eventstrom und potenziell schlimmer als die jetzige Fehlbezeichnung. Die korrekte Lösung verschiebt den `PREWARP`-`phase_start` hinter den SAMPLING_GEOMETRY-Block (echte sequenzielle Phase statt verschachtelt) und erfordert eine Durchsicht aller dazwischenliegenden Log-/Progress-Aufrufe sowie der Backend-Phasenreihenfolgelogik — bewusst nicht in dieser Revision gemacht, um keinen Halbfix mit neuem Eventstrom-Fehler zu erzeugen. |
| B5 Abnahmebehauptungen reichen weiter als Tests | **weitgehend behoben** | Neue Tests: Apertur-Fluss/Zentroid bei 1x/2x (`test_forward_drizzle.cpp`, „aperture flux and centroid survive fractional shifts"), Streifen-Determinismus (`compute_forward_drizzle_uniform` chunk_rows=1 vs. voll), Speicherbudget-Ablehnung vor Quell-I/O |
| B6 Dokumentations-/Terminwidersprüche | **in Arbeit** | Diese Revision (§0.1) plus punktuelle Korrekturen an §23/§30; historische §30-Notizen bleiben als datierter Verlauf stehen, sind aber laut §0 explizit nicht mehr maßgeblich |

<a id="historie-0-2"></a>

### 0.2 Nachprüfung der M3-Ergänzungen und OOM-Korrekturen

Der aktuelle [Audit-Nachtrag, Abschnitt 6](aqmh_cfa_forward_drizzle_audit_2026-09-05_de.md#6-umsetzung-und-nachprüfung-der-zwischenzeitlichen-ergänzungen-2026-09-05)
überschreibt die Speicher- und Vollständigkeitsbehauptungen in §30.16–30.22.
Insbesondere war die nachträgliche Kandidatenprüfung aus §30.17 kein OOM-Schutz.
Der gemeinsame Uniform/Raw-Pfad besitzt jetzt eine Streaming-API mit vorab
budgetierten, flachen Kandidatenpuffern. Kandidatenspeicher wird im Worst Case
mit allen Frames je Pixel/Kanal angesetzt; beide materialisierten Ergebnisse
werden zusätzlich angerechnet. Explizite zu große Chunks werden vor Quell-I/O
abgewiesen. Qualitätsproxies werden Frame-weise verarbeitet; FITS-Export erzeugt
keine zusätzlichen Vollbildmatrizen mehr.

Qualitätsplan und Profilmanifest haben zusätzliche semantische Prüfungen.
Dateiintegrität allein ist keine Store-Vollständigkeit oder Resume-Freigabe.
M3 bleibt teilweise implementiert: keine vollständige neue Runnerphasenfolge,
kein transaktionaler Gesamtstore, `Q_composite=1`. Die Library-Funktionen dürfen
nicht als bereits ausgelieferter Mehrband-Rekonstruktionspfad bezeichnet werden.
Die nächste Reihenfolge ist Store-/Resume-Vertrag, Runner-Anbindung,
verbleibende Einzelbild-/Extremfallbudgets, danach Q-Maps und Mehrbandfusion.


**Verifikation dieser Nachprüfung:** `tests` und `tile_compile_runner` gebaut;
382/382 Testfälle der Hauptsuite und 24/24 gezielte Auditregressionen bestanden.
Keine neuen realen Bildverarbeitungsläufe gestartet.

<a id="historie-0-3"></a>

### 0.3 Transaktionaler Streaming-Profilstore (2026-09-05)

Der nächste Store-Arbeitsschritt aus dem Audit ist umgesetzt:

- `drizzle_profile_store.hpp/cpp` persistiert ungeclipptes Uniform sowie das
  gemeinsam geclippte Uniform/Raw-Paar direkt aus Streifen in FITS. Der bisherige
  optionale Uniform-Diagnoseexport im Runner verwendet diesen Streaming-Store.
- Jede Veröffentlichung erhält einen eigenen `generation-…`-Ordner. Erst nach
  vollständigen Ebenen, Schließen/fsync, Prüfsummen und FITS-Prüfung wird
  `current.json` atomar ersetzt. Die unveränderliche Generation enthält zusätzlich
  `commit.json`. Ein Fehler vor Veröffentlichung erhält den bisherigen Commit;
  ein nach dem Rename gemeldeter Sync-Fehler löscht die möglicherweise bereits
  referenzierte Generation ausdrücklich nicht.
- Leser prüfen die **exakte** Kanal-/Feldmenge, positive Dimensionen,
  Float-FITS/ROWORDER, Prüfsummen, Commit-Version und caller-seitig erwartete
  Quell-, Sampling- und Rekonstruktionsidentität. Die Identität unterscheidet
  ungeclipptes M2-Uniform von geclipptem M3-Uniform/Raw und bindet die tatsächlich
  übergebenen `G_eff`-Werte. Chunkhöhe/Budget verändern die Identität nicht.
- Der RAM-Plan rechnet 8 MiB FITS-/Metadatenreserve plus eine float-Zeile zusätzlich
  an. Kein vollständiges Ausgabeprofil wird im Storepfad materialisiert. Vor
  Ebenenschreibbeginn wird freie Disk für die vollständige neue Generation plus
  Reserve geprüft; konkurrierende Disk-Nutzung kann dennoch spätere I/O-Fehler
  auslösen, die keinen unvollständigen Store veröffentlichen.
- Alte erfolgreiche Generationen und bei Prozessabbruch verbliebene Orphans
  werden nicht automatisch gelöscht. Leser nutzen ausschließlich den Commit.
  Der alte flache Diagnose-Store wird nicht als gültige neue Generation behandelt.

**Noch keine komplette M2/M3-Abnahme:** Der Store ist transaktional und bietet
einen budgetierten Region-Reader (`read_drizzle_profile_region`). Dieser prüft
vor jedem Ausschnitt erneut die vollständige Generation; das begrenzt den RAM,
kostet aber vollständige Prüfsummen-I/O. Ein wiederverwendbarer verifizierter
Reader sowie der allgemeine Q-Cache fehlen noch. Eigene neue Runnerphasen,
`QualityFrameWeightPlan` als validiertes Vorgängerartefakt, vollständige
Quellcache-/Resume-Prüfung und Uniform-Fallback im Produktionspfad bleiben offen.
Der gemeinsame Uniform/Raw-Store ist als Bibliotheks-API vorhanden; der Runner
persistiert weiterhin ausschließlich ungeclipptes Diagnose-Uniform. Es wurden
keine Benutzerruns gestartet oder bestehende Runartefakte umgeschrieben.

**Nächste Reihenfolge:** Qualitätsplan und normalisierten Quellcache als
verifizierte Vorgängerartefakte bereitstellen; danach neue Runnerphasen und
Resume/Fallback zusammen integrieren; anschließend wiederverwendbare verifizierte
Region-Reader und verbleibende Einzelbild-/Extremfallbudgets, erst danach Q-Maps
und Mehrbandfusion.


**Verifikation von §0.3:** `tests` und `tile_compile_runner` gebaut; sieben
Store-/Region-Regressionen mit 135 Assertions bestanden; vollständige Hauptsuite
**389/389 Testfälle**, 575.030 Assertions. JSON/YAML und übereinstimmende
Schemafelder geprüft. Die Tests verwenden synthetische Quellen und temporäre
Stores; keine wissenschaftliche Bildqualitäts- oder Powerloss-Abnahme.

<a id="historie-0-4"></a>

### 0.4 Geprüfte Vorgängerartefakte und Bibliotheksintegration (2026-09-05)

Der nächste Schritt aus §0.3 ist auf Bibliotheksebene umgesetzt:

- `normalized_source_cache.hpp/cpp` verwendet das bestehende Runner-Cacheformat
  `<source_index>.raw` (IEEE-754 float32, Little Endian, Row-Major). Ein atomar
  veröffentlichtes `normalized_source_manifest.json` bindet Quellidentität,
  Dimensionen, Farb-/CFA-Vertrag, vollständige Frame-IDs/Indizes, Bytezahl und
  SHA256 jeder Datei. Das Manifest bestätigt vorhandene normalisierte Dateien;
  es ersetzt oder rekonstruiert keine Kalibrierung/Normalisierung. Der Erzeuger
  muss deren Herkunft bereits kennen. Es gibt kein stilles Reparieren alter
  oder unvollständiger Caches.
- `VerifiedNormalizedSourceCache` hält höchstens ein Quellbild. Er prüft beim
  Laden die tatsächlich gelesenen Bildbytes statt die Datei in einem zweiten
  Lesevorgang zu hashen. Eine nach Manifestveröffentlichung veränderte,
  vertauschte oder gekürzte Datei scheitert vor ihrer Verwendung. Der Cache ist
  damit auf Inhaltsintegrität geprüft, aber noch kein dauerhaft aufbewahrter
  neuer Runner-Cache: Dessen Lebensdauer-/Cleanup-Vertrag bleibt umzubauen.
- `source_quality_artifact.hpp/cpp` berechnet die globale Quellqualität Frame für
  Frame, persistiert `QualityFrameWeightPlan` atomar und bindet ihn zusätzlich
  an den konkreten Cache-Manifesthash. Loader prüfen Sampling-Plan samt Hash,
  Qualitätskonfiguration, vollständige Frame-ID-Menge und exakte Übernahme der
  Registrierungsfaktoren. `resolve_quality_frame_weights` ordnet anhand der
  Frame-IDs den tatsächlichen Quellindizes zu; die Reihenfolge im Qualitätsartefakt
  darf nicht versehentlich zur Gewichtszuordnung werden.
- `persist_forward_drizzle_from_predecessors` verbindet geprüften Cache,
  gespeicherten Qualitätsplan und gemeinsames Uniform/Raw-Streaming als
  Bibliothekskette. Beide Vorgängerhashes sind Bestandteil der Store-Identität.
  **Store-Commit-Schema ist jetzt 2**; Schema-1-Commits werden nicht automatisch
  als passende neue Artefakte übernommen. Die direkte Diagnose-/Referenz-API
  bleibt ohne Vorgängerbindung verwendbar, erfüllt aber ausdrücklich nicht den
  Vertrag dieses geprüften Bibliothekseinstiegs.
- Die Qualitätsphase prüft vor Bild-I/O einen konservativen CPU-Arbeitssatz von
  128 Byte je Quellpixel plus Quell-/Ladepuffer, 8 MiB Reserve, Frame-Metadaten
  und Sternmessungs-Scratch gegen ihr explizites MiB-Budget und verfügbaren
  Host-/cgroup-Spielraum. Große Einzelbilder können deshalb früh abgelehnt
  werden. Dies ist eine Vorabplanung, keine harte RSS-Obergrenze für sämtliche
  Fremdbibliotheken. Der dichte `G_eff`-Vektor wird vor Allokation begrenzt und
  zusätzlich im Drizzle-Budget berücksichtigt; extreme Quellindizes dürfen
  keine unbeschränkten Vektorallokationen auslösen.

**Abgrenzung:** Die neue Kette ist synthetisch als Bibliotheksintegration
verifiziert. Es wurden keine produktiven Runnerphasen freigeschaltet, keine
Phase als resumierbar markiert und keine Benutzerruns gestartet. Der bisherige
Runner-Cache bleibt ohne expliziten neuen Lebensdauervertrag temporär. Die
Artefakt-APIs dürfen deshalb nicht als fertige Resume-Freigabe ausgelegt werden.

**Nächste Arbeitspakete:** Cache-Erzeugung und Aufbewahrung im neuen Runnerpfad
mit eindeutiger Normalisierungsprovenienz integrieren; danach neue Phasen,
Vorgängerprüfung und Uniform-Fallback gemeinsam verdrahten. Für native große
Sensorbilder die globale Qualitätsmessung speichersparender gestalten, statt
ihre konservative Schranke zu umgehen. Anschließend die wiederholte
Region-Read-Prüfsummen-I/O optimieren; Q-Maps und Mehrbandfusion bleiben danach.


**Verifikation von §0.4:** `tests` und `tile_compile_runner` gebaut. Die
13 gezielten Vorgänger-/Store-Tests bestehen (160 Assertions), ebenso die
gesamte Hauptsuite mit **395/395 Testfällen**, 575.055 Assertions. Nachgewiesen
sind Byte-/Provenienzprüfung, Frame-ID-Zuordnung, alte Commits bei Fehlern,
frühe Budgetablehnung und die vollständige synthetische Bibliothekskette.
Keine neuen Bildverarbeitungsläufe oder Backendprozesse gestartet.

<a id="historie-0-5"></a>

### 0.5 Neue Runner-Phasenfolge verdrahtet und real end-to-end verifiziert (2026-09-05)

Die in §0.4 als „nächstes Arbeitspaket" benannte gemeinsame Verdrahtung von
neuen Phasen, Vorgängerprüfung und Uniform-Fallback ist umgesetzt (Details
und Testliste: §30.23). Die §0.4-Abgrenzung „keine produktiven Runnerphasen
freigeschaltet, keine Phase als resumierbar markiert" ist damit **überholt**:

- `Phase`-Enum um `NORMALIZED_CACHE=24`/`SAMPLING_GEOMETRY=25`/`GLOBAL_QUALITY=26`/
  `FORWARD_DRIZZLE=27` erweitert (ans Ende angehängt, bestehende Werte
  unverschoben — der B4-Restpunkt aus §0.1 ist erledigt);
- `apps/runner_forward_drizzle.{hpp,cpp}` + CLI-Subcommands `reconstruct`
  und `resume-reconstruction`; der Pfad läuft einsträngig, ohne
  PREWARP-Nutzsignal, ohne Backendstart (`execution_scope =
  "forward_drizzle_m1_m3"`);
- **real auf M31 verifiziert**: 6 Frames laufen sauber durch
  `SCAN_INPUT → … → REGISTRATION → NORMALIZED_CACHE → SAMPLING_GEOMETRY →
  COMMON_OVERLAP → GLOBAL_QUALITY → FORWARD_DRIZZLE` bis
  `status=reconstruction_ready`; bei zu wenigen Frames scheitert
  `SAMPLING_GEOMETRY` korrekt fail-closed (`FORWARD_STAGE_COVERAGE_GATE_FAILED`);
  `source_quality_plan.json` und `forward_drizzle_profiles/generation-*/`
  (24 FITS-Ebenen uniform+raw, `current.json` mit Identitätskette +
  checksummiertem Manifest) werden real geschrieben; Raw weicht durch die
  echte per-Frame-`G_eff`-Gewichtung messbar von Uniform ab, bei
  bitidentischer Support-Maske.

Noch **nicht** freigegeben: der Cache-Lebensdauer-/Cleanup-Vertrag des neuen
Runner-Pfads (der normalisierte Cache wird versiegelt/verifiziert, aber sein
Aufbewahrungsvertrag ist unverändert temporär), sowie der Produktions-Cutover
selbst (M10 — der `PIPELINE_UNAVAILABLE_DURING_CUTOVER`-Lock des
Voll-Runners bleibt; nur der eigene `reconstruct`-Einstieg umgeht ihn
bewusst für den M1–M3-Scope).

<a id="historie-30-12"></a>

### 30.12 M2 begonnen: CFA-Forward-Drizzle-Kern, nur Uniform-Control (2026-09-05)

Mit dem vorhandenen M1-Core beginnt M2; seine vollständige Phasenabnahme bleibt offen. M2 (§23 „CPU Forward-Drizzle 1x
Uniform"). Neue Dateien: `include/tile_compile/reconstruction/forward_drizzle.hpp`,
`src/reconstruction/forward_drizzle.cpp`, `tests/test_forward_drizzle.cpp`.

**Umgesetzt (Kerngeometrie und -mathematik, real verifiziert statt nur
plausibilisiert):**

- **Exaktes Square-Droplet-Kernel für affine Frames** (§11.6): das Quellpixel
  wird als Quadrat mit Eckpunkten `s ± pixfrac/2` behandelt, die vier Ecken
  werden über die geprüfte affine `source_to_canvas` exakt abgebildet, und
  jedes überlappte Zielpixel erhält `K` als **echten**
  Polygon-Rechteck-Schnitt (Sutherland-Hodgman-Clipping + Shoelace-Fläche) —
  **kein** achsparalleler Bounding-Box-Ersatz, wie in §11.6 explizit
  gefordert ("Ein achsparalleles Droplet ohne Mitdrehung mit dem Frame ist
  nicht zulässig"). Das ist eine andere, strengere Geometrie als M1s
  `sampling_geometry.cpp`, das für das reine Coverage-Gate nur "berührt oder
  nicht" braucht.
- **Flächenidentität exakt verifiziert**: `sum_q K(q,s) = pixfrac^2 *
  internal_scale^2 * |det J_f|` wird für reine Translation, Rotation (30°),
  Skalierung (1,5x) und Rotation+Skalierung kombiniert, je bei
  `internal_scale ∈ {1,2}`, auf `1e-5` genau nachgewiesen (8 Testfälle). Das
  bestätigt zugleich, dass `|det J|` korrekt aus der **nativen** 2×2-Matrix
  von `source_to_canvas` genommen wird und `internal_scale^2` genau einmal
  separat multipliziert wird — keine doppelte Skalierung über eine
  fälschlich "interne" Jacobi-Matrix (ein Fehler, der bei 2x sofort einen
  Faktor-4-Fehler erzeugen würde und durch genau diesen Test aufgedeckt
  worden wäre).
- **CFA-Farbzuordnung** (§11.4): nutzt jetzt die aus `sampling_geometry.cpp`
  herausgelöste gemeinsame Implementierung (`core::cfa_channel_for_source_pixel`,
  neu in `types.hpp`) statt einer zweiten Kopie der Paritätslogik — vermeidet
  das Risiko, dass eine künftige Korrektur nur an einer Stelle ankommt. Test
  bestätigt korrekte R/G/B-Trennung inklusive gemeinsamer Akkumulation von G1
  und G2 in denselben Grünkanal.
- **MONO-Pfad**: befüllt ausschließlich `L`; R/G/B bleiben nachweislich
  komplett leer (nicht künstlich mit Kopien von `L` gefüllt, §11.4 explizit
  verboten) — per Test verifiziert.
- **Frame-lokale Aggregation** (§11.7) und **Uniform-Control** (§11.9) in der
  im Plan vorgesehenen zweistufigen Form implementiert (`A_f,c`/`B_f,c` pro
  Frame, `x_f,c = A/B`, anschließend `w_uniform = B_f,c(q)`-gewichtete
  Kombination über alle Frames) statt der algebraisch äquivalenten, aber für
  M3 nicht wiederverwendbaren Kurzform `sum_f A / sum_f B` — bewusst so
  gebaut, weil M3s robustes Clipping (§11.8) exakt auf `x_f,c(q)` als
  statistischer Einheit operiert. Test mit zwei identisch registrierten
  Frames und unterschiedlichen Werten bestätigt den erwarteten arithmetischen
  Mittelwert sowie `n_eff = 2` für zwei gleich gewichtete Beiträge (§11.10).
- **Feste Frame-Reihenfolge** (§11.12): Frames werden vor der Akkumulation
  nach `source_index` aufsteigend sortiert; die Kombination in die laufenden
  Summen erfolgt strikt sequenziell in dieser Reihenfolge (Fließkomma-Addition
  ist nicht assoziativ — das ist eine echte Determinismus-Anforderung, keine
  Kosmetik).
- **Adaptive Subdivision für lokale Warps** (§11.6), **real mit
  Konvergenzprüfung**, nicht nur approximativ: pro Quad werden Kantenmitten
  und Zentrum mit der geguardeten Inversion aus §7.3 abgebildet und gegen die
  bilineare Näherung aus den Eckpunkten geprüft (Positionskriterium
  `<= 0,05` interne Pixel); zusätzlich wird das Quad probeweise in vier
  Kinder geteilt und deren Flächensumme gegen die Elternfläche geprüft
  (Flächenkriterium `<= 0,5 %`). Ein Blatt wird erst akzeptiert, wenn **beide**
  Kriterien erfüllt sind; sonst wird bis `max_subdivision_depth = 2` weiter
  unterteilt, danach verworfen und gezählt. Ein Frame, dessen Verwurfsquote
  `per_frame_inversion_error_rate_max = 0,1 %` überschreitet, wird
  **vollständig ausgeschlossen** (kein Teilausfall, kein stiller Fallback).
  Zwei Tests bestätigen beide Enden: ein lokales Modell mit Nullverschiebung
  reproduziert die affine Fläche exakt (Konvergenz sofort, keine Verwürfe);
  ein absichtlich nicht invertierbares lokales Modell (`model_coordinate_scale
  = 0`) führt zum vollständigen, korrekt gezählten Frameausschluss statt zu
  stillem Datenverlust.
- Quellwerte werden ausschließlich über den injizierten `SourceImageProvider`
  gelesen (Vertrag: normalisierter CFA-Cache, §10.1) — die Produktionsverdrahtung
  an den echten Cache und `prewarped_frames`-Verbot (§23 M2-Abnahme) folgt mit
  der Runner-Anbindung (`runner_phase_forward_drizzle.cpp`, noch offen, siehe
  unten).

**Bewusst noch nicht umgesetzt (dokumentierte, nachverfolgte Vereinfachung,
kein stiller Scope-Verlust):**

- **Transaktionale Profilstores** (§11.3): M2 nutzt einen einfachen
  In-Memory-`ProfilePlane` statt des disk-/mmap-gestützten, crash-sicheren
  `DrizzleProfileStore` mit `read_region`/`write_region`. Store-/Chunk-/
  Crash-Commit-Tests aus der M2-Abnahme sind damit **noch nicht erfüllt**.
- **Chunking mit Rand-Halo** (§11.11): Ein Frame wird komplett in einem
  Durchlauf mit einem vollbild-großen `A`/`B`-Puffer verarbeitet (ein Frame
  gleichzeitig, kein Vollbild-Puffer für alle Frames — das Teilkriterium aus
  §11.3 ist erfüllt), aber ohne Zeilen-Chunking, Rand-Halo oder
  Speicherbudget-Gate. Für reale 24-MP-Datensätze ist das ein
  Performance-/Speicherrisiko, kein Korrektheitsproblem für die hier
  getesteten synthetischen Fixtures.
- **Parallelisierung**: der Rekonstruktionskern läuft aktuell einsträngig.
  Anders als bei M1 (wo die Parallelisierung in einem Folgeschritt sicher
  nachgerüstet wurde, 30.9) ist hier noch offen, welche Partitionierung
  (Zeilenbänder statt Frames — Frames können wegen der
  Frame-lokalen-Aggregation nicht einfach wie in M1 über Worker verteilt
  werden, ohne pro Worker vollbildgroße frame-lokale Puffer zu duplizieren)
  determinismuserhaltend ist; als Folgeschritt vorgesehen.
- ~~**Runner-Anbindung**: noch nicht geschrieben~~ — **inzwischen erledigt
  als Diagnose-Preview, siehe 30.13.** (Eine eigenständige
  `apps/runner_phase_forward_drizzle.{hpp,cpp}`-Datei mit echter
  Pipeline-Phase inkl. Resume-Vertrag §10.3 ist weiterhin M2/M3-Restarbeit;
  die Preview lebt vorerst als Funktion in `runner_phase_registration.cpp`.)
- Das reservierte Diagnosefeld `max_affine_area_relative_error` wird bewusst
  nicht aus der heißen Schleife heraus befüllt (Performance); die
  Flächenidentität ist stattdessen direkt durch dedizierte Unit-Tests
  verifiziert (siehe oben), nicht durch Laufzeit-Instrumentierung.

Build (alle Targets: `tile_compile_lib`, `tests`, `tile_compile_runner`,
`tile_compile_legacy_reference`, `tile_compile_legacy_reference_tests`) grün.
Volle Suite 317/318 (der eine vorbestehende GPU-Fehlschlag unverändert; +9
neue Fälle für den Forward-Drizzle-Kern).

**Nachträglich in derselben Revision gefundener und behobener Performance-Defekt
(kein mathematischer Fehler):** `source_of(f.source_index)` wurde ursprünglich
innerhalb der inneren Sample-Schleife aufgerufen (einmal pro Quellpixel statt
einmal pro Frame). Für den in-Memory-Test unschädlich, aber bei der
vorgesehenen Produktionsanbindung an `RunnerFrameCache::load_normalized()`
(Disk-Cache-Zugriff) wäre das ein handfestes Performance-Problem gewesen.
Behoben durch Hoisting vor die Sample-Schleife; volle Suite erneut grün
verifiziert.

M2 ist damit in der Kerngeometrie/-mathematik weit fortgeschritten, aber in
der M2-Abnahme (§23) noch nicht vollständig: Store-/Chunk-/Crash-Commit-Tests
und die reale Runner-Verdrahtung auf `prewarped_frames`-freien Daten stehen
noch aus.

---

<a id="historie-30-13"></a>

### 30.13 M2: Diagnose-Preview auf echten M31-Daten verifiziert (2026-09-05)

Der Forward-Drizzle-Kern wurde als opt-in, standardmäßig **deaktivierte**
Diagnosefunktion `run_forward_drizzle_uniform_preview()` in
`runner_phase_registration.cpp` verdrahtet, gesteuert über die neue
Konfigurationsoption `reconstruction.diagnostics.preview_forward_drizzle_uniform`
(Default `false`; Schema, Parser, Serializer und 3 Tests aktualisiert).

**Bewusstes Design:** die Preview läuft **nicht** als eigene Pipeline-Phase
mit Resume-Vertrag (§10.3 ist eine eigene, größere Restarbeit), sondern als
zusätzlicher, niemals abbrechender Aufruf direkt nach `SAMPLING_GEOMETRY` —
auf **beiden** Binaries (neu und Legacy-Referenz) gleichermaßen, weil sie
reine additive Diagnostik ist, keine Verhaltensänderung. Jede Ausnahme wird
abgefangen und nur als Warnung geloggt; der Lauf wird davon nie beeinflusst.
Quellwerte kommen über dieselbe `load_frame_normalized`-Closure, die
`REGISTRATION`/`PREWARP` bereits für den normalisierten Cache verwenden
(lazy: Cache-Hit oder FITS-Neuladen+Normalisierung) — **nicht**
`prewarped_frames` (§23 M2-Abnahme strukturell erfüllt für den Kern, siehe
30.12).

**Realer Verifikationslauf** (M31, 4 Frames, `tile_compile_legacy_reference`,
Produktionsdefaults 2x/`pixfrac=0.8`):

```json
{
  "coverage_fraction_R": 0.9619, "coverage_fraction_G": 0.9796, "coverage_fraction_B": 0.9619,
  "internal_width": 7696, "internal_height": 4328,
  "local_model_samples_total": 0, "local_model_samples_discarded": 0,
  "frames_excluded_subdivision_error_rate": [],
  "elapsed_s": 30.86
}
```

Beobachtungen: (1) die Preview lief erfolgreich **trotz** eines
fehlschlagenden `SAMPLING_GEOMETRY`-Gates (4 Frames sind zu wenig,
`min_supported_fraction`/`min_channel_n_eff`/Loch-Kriterium alle verletzt) —
bestätigt empirisch, dass die Diagnose wie vorgesehen vom Gate-Ergebnis
entkoppelt ist; (2) G-Coverage (0,98) liegt wie erwartet über R/B (0,96) —
konsistent mit der dichteren Bayer-Grünabtastung; (3) keine lokalen Modelle
aktiv bei diesen 4 Frames (`local_attempted=1, local_corr=0, local_rejected=1`
im Registrierungslog), daher `local_model_samples_total=0` — die
Subdivisionslogik wurde hier nicht durchlaufen, ihre Korrektheit ist
weiterhin ausschließlich durch die synthetischen Tests aus 30.12 belegt,
nicht durch diesen Lauf; (4) **~31 s für 4 Frames** bei
7696×4328-Internalraster bestätigt real die in 30.12 dokumentierte, erwartete
Performance-Schwäche des einsträngigen, ungechunkten Referenzkerns mit
exaktem Polygon-Clipping — für 20+ Frames wäre das mehrere Minuten, was die
Priorität von Chunking/Parallelisierung für M2s Abschluss unterstreicht statt
sie zu widerlegen.

Testverzeichnis (`/tmp/m2preview`) nach Verifikation entfernt. Build (alle
Targets) und volle Suite grün (317/318, der eine vorbestehende
GPU-Fehlschlag unverändert).

---

<a id="historie-30-14"></a>

### 30.14 M2: erster echter Schritt zum transaktionalen Profilstore + Fund und Fix eines echten Atomizitätslecks (2026-09-05)

Nach der Auditintegration (§0/§0.1) war von den M2-Abnahmepunkten noch
„transaktionale Profilstores" offen. Umgesetzt wurde ein erster, ehrlich
begrenzter Schritt: `write_forward_drizzle_uniform_store()`
(`runner_phase_registration.cpp`) persistiert die materialisierten
Uniform-Control-Ebenen (`value`/`weight_sum`/`n_eff`/`support` je aktivem
Kanal) als FITS-Dateien unter `artifacts/forward_drizzle_uniform_store/`,
gesteuert über die neue, unabhängig von der Preview schaltbare Option
`reconstruction.diagnostics.persist_forward_drizzle_uniform_store` (Default
`false`; Schema, Parser, Serializer, 4 Config-Tests).

**Echter Fund beim Bauen der Persistenz, nicht nur beim Nachdenken darüber:**
Die ursprüngliche Absicht war, mich auf die Atomizität von `io::write_fits_float()`
zu verlassen — analog zur bereits atomaren `io::write_fits_mask_rows()`. Beim
Nachschauen im aktuellen Code stellte sich heraus, dass das **nicht stimmt**:
`write_fits_float()` schrieb bislang über cfitsios `"!"`-Präfix-Konvention
(Datei löschen, neu anlegen), **nicht** über `core::AtomicOutput` — ein reales,
unentdecktes Atomizitätsloch genau in der Funktion, auf die sich der neue
M2-Store verlassen sollte, und mit Auswirkung auf **alle** bestehenden Aufrufer
dieser Funktion im gesamten Codebase (Ausgabebilder, Q-Maps, etc.), nicht nur
auf den neuen Store. Behoben durch Umbau von `write_fits_float()` auf exakt
dasselbe Stage-fsync-rename-Muster wie `write_fits_mask_rows()`
(`fits_io.cpp`), mit `try`/`catch` um den cfitsio-Handle-Lebenszyklus, damit
ein fehlgeschlagener Schreibversuch den gemappten Handle sauber schließt, bevor
die Exception weitergereicht wird. **Nicht** mit angefasst (bewusst begrenzter
Scope, dokumentiert statt stillschweigend offengelassen): `write_fits_rgb()`
und `write_fits_rgb_u32()` im selben File nutzen weiterhin die alte,
nicht-atomare `"!"`-Konvention.

Ein neuer Test (`test_fits.cpp`, „FITS audit: write_fits_float is atomic")
verifiziert das direkt: erster committeter Schreibvorgang lesbar, ein
simulierter Absturz mitten im zweiten Schreibvorgang (abgebrochene
Staging-Datei, nie committet) lässt den ersten Commit unverändert lesbar,
kein verwaistes `.stage-*`-Verzeichnis bleibt zurück, ein darauffolgender
gültiger zweiter Schreibvorgang ersetzt den Inhalt korrekt.

**Explizit dokumentierte Grenze der neuen Store-Funktion selbst:** jede
Plandatei ist einzeln atomar (nie trunkiert beobachtbar), aber es ist **keine
Store-weite Transaktion** über alle Plandateien hinweg — ein Absturz zwischen
zwei Dateischreibvorgängen kann einen Satz aus Dateien unterschiedlicher
Generation hinterlassen. Der volle §11.3-Vertrag (mmap-gestützt,
`read_region()`/`write_region()`, Einzeltransaktion über alle Ebenen) bleibt
offene M2-Arbeit; das ist ein echter, aber kleinerer Schritt dorthin, kein
vollständiger Ersatz.

Reale Verifikation auf M31 (4 Frames, `tile_compile_legacy_reference`,
Produktionsdefaults 2x/`pixfrac=0.8`): 12 Ebenendateien (R/G/B ×
`value`/`weight_sum`/`n_eff`/`support`) plus `manifest.json` erfolgreich
geschrieben; Inhalt via Python/astropy geprüft — `R_support`-Mittelwert
(0,962) stimmt mit der unabhängig berechneten `coverage_fraction_R` aus der
Preview (30.13, 0,9619) überein (interne Konsistenzprüfung zwischen zwei
unterschiedlichen Berechnungspfaden bestanden); `n_eff`-Maximum ≈3,0 bei 4
gültigen Frames, `weight_sum`-Wertebereich plausibel für `pixfrac=0,8` bei
`internal_scale=2`. Nach Verifikation mit dem jetzt tatsächlich atomaren
`write_fits_float()` erneut real gegengeprüft (nicht nur mit dem alten,
nicht-atomaren Schreibpfad getestet). Testverzeichnisse entfernt.

Build (alle Targets) und volle Suite grün (333/334; der eine
Fehlschlag weiterhin `test_acceleration_backend.cpp:254`, unverändert
umgebungsbedingt).

---

<a id="historie-30-15"></a>

### 30.15 M3 begonnen: gemeinsames robustes Clipping als geprüfter, eigenständiger Algorithmus (2026-09-05)

M3 (§23 „Robustes Clipping und Raw-Forward-Drizzle") ist ein großer
Meilenstein (gemeinsame Akzeptanzmaske, minimale Source-Space-Q-Versorgung
mit Quad-Green/Highpass-MAD, `GLOBAL_QUALITY` mit `G_quality(f)`,
`QualityFrameWeightPlan`, Raw-Baseline-Persistenz). Begonnen wurde bewusst mit
dem Kernstück, das unabhängig von der noch nicht existierenden Q-Map-Infrastruktur
(§13) vollständig implementier- und prüfbar ist: der eigentliche
Clipping-Algorithmus aus §11.8.

**Umgesetzt:** `apply_robust_clipping()` (`forward_drizzle.hpp/cpp`) setzt
die verbindlichen 8 Schritte aus §11.8 exakt um — deterministische Sortierung
nach Wert mit Frameindex-Tie-Break, geometrisch (`B_f,c(q)`) gewichteter
Median und MAD, asymmetrische `clip_sigma_low`/`clip_sigma_high`-Grenzen,
Wiederholung bis `robust_passes` oder unveränderte Maske, `min_clip_contributors`-
Bypass für dünn belegte Kanäle, und die `min_fraction`/`min_n_eff`-Pixelveto-
Prüfung gegen den geometrisch möglichen Frame-Support (nicht gegen
Q-Gewichte). Degenerierte-MAD-Guards exakt nach Plantext: identische Werte
bleiben gültig, kein erfundenes Epsilon für konstante Hintergründe.

**8 neue Tests, mit von Hand nachgerechneten Erwartungswerten (nicht nur
Plausibilisierung):**
- unterhalb `min_clip_contributors` wird trotz offensichtlichem Ausreißer
  nichts geclippt (Schutz dünner R/B-Kanäle);
- ein klarer Ausreißer wird über den degenerierten-MAD-Fall exakt erkannt
  (vier identische Werte + ein Ausreißer → `MAD=0` → Ausreißer fällt sofort
  heraus);
- identische Werte bleiben vollständig gültig (kein Epsilon-Wegclippen);
- asymmetrische Sigma-Grenzen clippen nachweisbar nur auf der konfigurierten
  Seite (von Hand nachgerechnet: Median 12, MAD 1, Grenzen `[7, 12.5]` bei
  `sigma_low=5, sigma_high=0.5`);
- ein **echter Mehr-Pass-Fall**: eine Gruppe weit gestreuter Ausreißer bläht
  das MAD in Durchgang 1 auf, sodass ein moderater Ausreißer (18) zunächst
  **nicht** erkannt wird; erst Durchgang 2 rechnet MAD ohne die entfernten
  Extremwerte neu und erkennt ihn — mit `robust_passes=1` bleibt er gültig,
  mit `robust_passes=2` wird er korrekt entfernt (von Hand über drei
  Iterationsstufen nachgerechnet, nicht nur am Testergebnis abgelesen);
- das Pixelveto (`min_fraction`/`min_n_eff`) verwirft das gesamte Pixel bei
  zu starkem Datenverlust, auch wenn die Clipping-Maske selbst korrekt wäre;
- **Determinismus**: dieselbe Kandidatenmenge in unterschiedlicher
  Eingabereihenfolge liefert pro Frameindex identische Akzeptanzentscheidungen;
- leere Eingabe wird sauber als Pixelveto behandelt, kein Absturz.

**Bewusst noch nicht umgesetzt (dokumentierter, nicht stillschweigender
Umfang):** der Algorithmus ist **noch nicht** in
`compute_forward_drizzle_uniform()`/`stream_forward_drizzle_uniform()`
verdrahtet. §11.8 verlangt ausdrücklich, dass dieselbe Akzeptanzmaske für
Uniform, Raw-Forward-Drizzle **und** alle Detailprofile gilt — das erfordert,
pro Streifen und Pixel die einzelnen `x_f,c(q)`-Beiträge **aller** Frames
vorzuhalten (nicht nur die bisherigen laufenden Summen), was das
Speicherbudget-Modell um den in §11.11 bereits vorgesehenen Term
`per_frame_sample_bytes * active_frames_in_band` erweitert. Das ist die
nächste, größere Integrationsarbeit, ebenso wie die minimale
Source-Space-Q-Versorgung und `GLOBAL_QUALITY` — beides eigenständige,
noch nicht begonnene Teile von M3.

Build (alle Targets) und volle Suite grün (341/342; der eine Fehlschlag
weiterhin `test_acceleration_backend.cpp:254`, unverändert umgebungsbedingt).

---

<a id="historie-30-16"></a>

### 30.16 M3 fortgesetzt: gemeinsame Akzeptanzmaske für Uniform und Raw verdrahtet (2026-09-05)

Der in 30.15 als offen benannte nächste Schritt ist umgesetzt:
`compute_forward_drizzle_uniform_and_raw()` (`forward_drizzle.hpp/cpp`)
berechnet Uniform-Control und Raw-Forward-Drizzle **im selben Durchlauf**
mit **derselben** Clipping-Entscheidung pro Pixel/Kanal, exakt wie §11.8
verlangt („Die resultierende Akzeptanzmaske wird unverändert für Uniform,
Raw-Forward-Drizzle und alle Detailprofile verwendet").

**Architektur:** anders als `stream_forward_drizzle_uniform()`s
O(1)-pro-Pixel-Laufsummen hält diese Funktion pro Streifen und Pixel die
einzelnen `(x_f,c(q), B_f,c(q))`-Beiträge **aller** Frames vor (eine
`std::vector<ClipCandidate>` je Pixel/Kanal), ruft darauf
`apply_robust_clipping()` auf, und aggregiert danach getrennt für Uniform
(`w=B`) und Raw (`w=B·G_eff·Q_composite`).

**Ehrlich benannter Stand von Raw:** `G_eff` und `Q_composite` erfordern die
noch nicht existierende `GLOBAL_QUALITY`-Phase und Q-Map-Infrastruktur
(§13) — beide sind vorerst auf den neutralen Wert `1.0` fixiert. Raw ist
damit **numerisch identisch** mit dem geclippten Uniform, bis diese
Gewichte real ankommen. Das ist kein Fehler, sondern der einzige ehrliche
Zwischenstand: die Funktion existiert bereits jetzt mit der richtigen
Struktur, damit die spätere Q-Gewicht-Verdrahtung additiv bleibt statt einer
zweiten Umschreibung des Streaming-Kerns.

**4 neue Integrationstests** (zusätzlich zu den 8 reinen Algorithmus-Tests
aus 30.15): ein Ausreißer wird aus Uniform **und** Raw identisch entfernt
(Restwert exakt 10 bei vier 10ern + einem 100er-Ausreißer, Raw==Uniform auf
1e-9 genau); unterhalb `min_clip_contributors` bleibt der Ausreißer in
beiden Profilen erhalten (Mittelwert aller fünf Werte = 28, entspricht
exakt M2s bisherigem ungeclipptem Verhalten für diesen Grenzfall);
`min_fraction`-Veto verwirft das Pixel in **beiden** Profilen identisch,
keine Teilbefüllung; MONO füllt nur `L` in beiden Profilen.

**Bewusst noch nicht getan (dokumentierte Grenze, kein stiller
Vollständigkeitsanspruch):**
- **Kein Speicherbudget für die neue Kandidatenliste.** `plan_drizzle_memory()`
  kennt weiterhin nur den in `stream_forward_drizzle_uniform()` bereits
  budgetierten Term; der zusätzliche `per_frame_sample_bytes *
  active_frames_in_band`-Term aus §11.11 ist für diese Funktion **nicht**
  eingerechnet. Bei realistisch geditherten Daten bleibt die Kandidatenzahl
  pro Pixel klein (durch die geometrische Überlappungsgrenze begrenzt), bei
  pathologisch deckungsgleichen Frames (kein/kaum Dither) könnte das
  Speicherbudget ungeprüft überschritten werden.
- **Keine Streaming-Sink-Variante.** Anders als `stream_forward_drizzle_uniform()`
  materialisiert diese Funktion das volle Ergebnis; eine
  `stream_forward_drizzle_uniform_and_raw()`-Variante mit Sink-Callback
  existiert noch nicht.
- **Keine Verifikation auf echten M31-Daten in dieser Revision.** Der
  Algorithmus ist rein pixelstatistisch (unabhängig von der konkreten
  Ditherverteilung) und durch die Unit-Tests bereits hart über Hand­rechnung
  verifiziert; was durch einen echten Lauf zusätzlich geprüft würde, ist
  primär das Speicher-/Laufzeitverhalten bei realer Frame-/Pixelzahl, nicht
  die Rechenlogik. Das ist bewusst zurückgestellt statt einer weiteren,
  diesmal weniger aussagekräftigen M31-Bestätigung — echte Verifikation
  dieser Speichergrenze braucht ohnehin zuerst das Budget-Accounting oben.
- **Keine Runner-Anbindung.** Diese Funktion ist bislang nur über die
  Testsuite mit synthetischen Plänen aufgerufen worden.

Build (alle Targets) und volle Suite grün (345/346; der eine Fehlschlag
weiterhin `test_acceleration_backend.cpp:254`, unverändert umgebungsbedingt).

---

<a id="historie-30-17"></a>

### 30.17 M3: Laufzeit-Sicherheitsnetz für das Kandidatenlisten-Speicherbudget (2026-09-05)

Der in 30.16 als offen benannte Punkt „kein Speicherbudget für die
Pro-Pixel-Kandidatenliste" ist teilweise geschlossen — mit einer bewusst
ehrlich benannten Einschränkung, keiner Übertreibung.

**Umgesetzt:** Nach dem Aufbau der Kandidatenlisten für einen Streifen wird
deren tatsächlicher Speicherbedarf gemessen (`capacity() * sizeof(ClipCandidate)`
über alle Kanäle/Pixel) und gegen das verbleibende Budget geprüft. Wird das
Budget überschritten, bricht die Funktion mit einer klaren
`DRIZZLE_MEMORY_BUDGET`-Exception ab, **bevor** die Clipping-Auswertung
beginnt — fail-closed statt eines unkontrollierten Anstiegs des
Speicherverbrauchs.

**Bewusst weiterhin offen (kein A-priori-Bound, nur ein reaktives Netz):**
Anders als beim reinen Uniform-Streaming-Pfad wählt `plan_drizzle_memory()`
die Streifenhöhe weiterhin **ohne** diesen Term — sie kann also nicht im
Voraus kleiner gewählt werden, um innerhalb des Budgets zu bleiben. Das
Sicherheitsnetz erkennt die Überschreitung erst, nachdem die Kandidaten für
den (dann zu großen) Streifen bereits aufgebaut wurden, verhindert also eine
denkbare Speicherspitze während des Aufbaus nicht rückwirkend, sondern nur
die anschließende Weiterverarbeitung. Ein echter A-priori-Bound (Streifenhöhe
vorab anhand einer Schätzung der aktiven Framezahl pro Band verkleinern,
§11.11) bleibt offene Arbeit.

**2 neue Tests:** 50 deckungsgleich registrierte Frames auf einem 50×50-Raster
(jedes der 2500 internen Pixel von allen 50 Frames getroffen, ≈3 MB
Kandidatenspeicher in einem einzigen Streifen) gegen ein bewusst zu kleines
Budget (2 MiB) löst zuverlässig `DRIZZLE_MEMORY_BUDGET` aus; dieselbe Eingabe
mit realistischem Budget (64 MiB) läuft durch.

Build (alle Targets) und volle Suite grün (346/347; der eine Fehlschlag
weiterhin `test_acceleration_backend.cpp:254`, unverändert umgebungsbedingt).

---

<a id="historie-30-18"></a>

### 30.18 M3 fortgesetzt: CFA-aware Analyseproxy (§13.2) implementiert (2026-09-05)

Nach den in sich abgeschlossenen M3-Teilstücken (Clipping-Algorithmus,
gemeinsame Maske, Speicherbudget-Netz) wurde mit dem größeren,
eigenständigen M3-Baustein begonnen, der die minimale
Source-Space-Q-Versorgung trägt: dem CFA-aware Analyseproxy aus §13.2. Neue
Dateien: `include/tile_compile/reconstruction/source_quality_proxy.hpp`,
`src/reconstruction/source_quality_proxy.cpp`, `tests/test_source_quality_proxy.cpp`.

**Umgesetzt, `proxy_version=1` exakt nach Plantext:**
- **Quad-Green-Gitter** (Schritt 1): `G_quad = 0.5*(G1+G2)` pro 2×2-Bayer-Quad,
  über die gemeinsame `cfa_channel_for_source_pixel()`-Klassifikation (keine
  zweite CFA-Paritätslogik);
- **Globales `sigma_green`** (Schritt 2): `hp = quad_green - B3_blur(quad_green)`,
  `sigma_green = 1,4826 * median(|hp - median(hp)|)` — die **exakte**
  median-basierte MAD-Formel aus dem Plantext, bewusst **nicht** die
  Local-Mean-Näherung, die der bestehende `compute_aqmh_quality_map()`-Pfad
  aus Performancegründen verwendet (unterschiedliche Formel für einen
  unterschiedlichen Zweck: globaler Skalar statt Fenster-Karte);
- **Separable B3-Spline-Unschärfe** (`[1,4,6,4,1]/16`, Rand geklemmt) als
  eigenständige, direkt testbare Funktion;
- **Vollauflösender Edge-aware-Grünproxy** (Schritt 3): native Grünwerte an
  G-Positionen unverändert übernommen; an R/B-Positionen wird zwischen
  horizontalem und vertikalem Nachbarpaar-Mittel **die Richtung mit dem
  geringeren lokalen Gradienten** gewählt (verhindert Mittelung über eine
  reale Kante hinweg) — reine Analysegröße, verändert nie das Nutzsignal;
- **MONO-Pfad**: verwendet die normalisierte L-Ebene direkt, keine
  CFA-Interpolation, kein Quad-Gitter.

**9 neue Tests, alle mit exakt von Hand nachgerechneten Erwartungswerten:**
B3-Unschärfe erhält eine Konstante exakt (DC-Verstärkung 1) und verteilt
einen Einzelspike exakt nach den Kernelgewichten `1/4/6/4/1÷16`; die
MAD-Sigma-Formel auf `{1,2,3,4,5}` ergibt exakt `1,4826`, auf einem
konstanten Bild exakt `0`; das Quad-Grün-Gitter auf einem hand-konstruierten
4×4-RGGB-Schachbrett ergibt an allen vier Quads die exakt erwarteten Werte;
der Edge-aware-Proxy wählt nachweislich die glattere Richtung sowohl bei
horizontaler als auch bei vertikaler Kante (zwei komplementäre Fälle);
native Grünwerte bleiben unverändert; MONO liefert `proxy_full` exakt gleich
der Eingabe, kein Quad-Gitter; `sigma_green` ist auf einem perfekt
gleichmäßigen CFA-Bild exakt `0`.

**Bewusst noch nicht umgesetzt (M5-Scope, im Plan selbst so vorgesehen, kein
Versehen):** skalenspezifische `ScaleQualityMap`-Karten, der
Sink/Callback-Streaming-Vertrag (§13.3), das Cache-Layout und Region-Reads
(§13.4/13.5). Ebenfalls noch offen (M3-intern): `G_quality(f)` selbst — die
Kombination dieses Proxys mit der bestehenden globalen
SNR-/Schärfe-/Sternstatistik-Formel (`calculate_global_weights_with_stars()`)
in einer neuen `GLOBAL_QUALITY`-Phase ist der nächste Schritt, hier noch
nicht angefasst (inzwischen erledigt: `G_quality(f)` in 30.19, `GLOBAL_QUALITY`-
Phase in 30.23; die §13.2-Tests gegen Bayer-Checkerboard/farbige Sterne/
schmalbandiges MONO in 30.24). Offen bleibt nur die Zero-Veto-Maskenweiterleitung
(Schritt 4 aus §13.2) und deren Leckage-Test — die Zero-Veto-Maske selbst
ist in `proxy_version=1` noch nicht implementiert.

Build (alle Targets) und volle Suite grün (355/356; der eine Fehlschlag
weiterhin `test_acceleration_backend.cpp:254`, unverändert umgebungsbedingt).

**Nachträglich real auf echten M31-Rohdaten verifiziert (Nutzerfrage: sollten
Zwischenschritte grundsätzlich auch gegen Echtdaten geprüft werden, nicht nur
synthetisch?).** Antwort dazu: Ja für alles, was einen plausiblen
Echtdatenpfad hat, auch ohne volle Runner-Anbindung — hier per schlankem
Wegwerf-Treiber direkt gegen die Bibliotheksfunktion, nicht nur diskutiert.
4 reale M31-Rohframes (`raw_M 31_10s_80_0000..0003.fits`, GBRG,
3840×2160, 12-Bit-ADU in 16-Bit-Containern, `BAYERPAT`-Header ausgelesen):

```text
elapsed_s≈0.115-0.126 (Vollauflösungsproxy, 8,3 MP, pro Frame)
sigma_green≈11.32-11.35 (über 4 Frames, plausibler, stabiler Rauschmaßstab)
proxy_full: kein NaN/Inf, min=0, max=4095 (exakt der 12-Bit-ADU-Bereich), mean≈216
quad_green: kein NaN/Inf, 1920×1080 (exakt halbe Auflösung von 3840×2160)
```

Bestätigt: keine NaN/Inf/Abstürze auf echten Sensordaten, Rauschmaßstab über
mehrere Frames der gleichen Serie eng stabil (11,32-11,35, keine Ausreißer),
Laufzeit für den Vollauflösungspfad klein genug, um pro Frame in
`GLOBAL_QUALITY` unproblematisch zu sein. Wegwerf-Treiber nach Verifikation
entfernt (kein Bestandteil der committeten Suite — echte Sensordateien
liegen außerhalb des Repos und sind nicht portabel; die synthetischen Tests
mit exakter Ground Truth bleiben die primäre, dauerhaft laufende
Korrektheitsprüfung, echte Daten ergänzen sie punktuell bei jedem Schritt,
der einen plausiblen Echtdatenpfad hat).

---

<a id="historie-30-19"></a>

### 30.19 M3: `G_quality(f)` in `GLOBAL_QUALITY` — echter Vertragskonflikt gefunden und mit minimalem Fix geschlossen (2026-09-05)

Weiter mit dem in 30.18 benannten nächsten Schritt: `G_quality(f)` aus dem
neuen Analyseproxy, gemäß §11.9 unter Wiederverwendung „derselben
mathematischen Definition wie bisher". Neue Dateien:
`include/tile_compile/reconstruction/global_quality.hpp`,
`src/reconstruction/global_quality.cpp`, `tests/test_global_quality.cpp`.

**Echter, beim Implementieren entdeckter Vertragskonflikt (nicht im Plantext
benannt):** Die vorhandene, wiederzuverwendende Funktion
`metrics::calculate_global_weights_with_stars()` liefert
`w = exp(k · clamp(Q, lo, hi))` — ein **unbeschränktes** positives Gewicht,
im bestehenden Code ausdrücklich **nicht** auf Summe 1 normiert, weil die
absolute, framezahl-unabhängige Skala „meaningful" ist (Kommentar im
bestehenden Code). Mit den Default-Clamp-Grenzen `[-3,3]` reicht der
Wertebereich bis `exp(3)≈20`. Das widerspricht direkt §11.9s Vertrag
„`G_quality(f)` ... liegen in `[0,1]`" — und dieser Vertrag ist nicht nur
kosmetisch: `A_coverage,c = clamp(w_profile/w_uniform, 0, 1)` (§14.4) ist nur
dann sinnvoll, wenn `w_profile <= w_uniform` punktweise gilt, was
`G_eff <= 1` voraussetzt. Ein `G_quality(f) > 1` für auch nur ein Frame würde
diese Invariante still brechen, ohne dass irgendwo ein Fehler auftritt.

**Minimaler, algebraisch exakter Fix statt Neuerfindung:** Die bestehende
Formel bleibt **unverändert** (`dieselbe mathematische Definition wie
bisher" wörtlich erfüllt); obenauf wird die logistische Stauchung
`G_quality = w / (1 + w) = sigmoid(k·Q)` angewendet. Das erhält die
framezahl-unabhängige absolute Skala (weiterhin monoton in demselben
zugrunde liegenden Q-Score) und landet garantiert im offenen Intervall
`(0,1)` — bewusst **nie exakt 0 oder 1**, damit es nie mit dem separaten,
expliziten Q=0-Veto (§11.9) verwechselt werden kann.

**Architektur:** pro Frame wird der §13.2-Analyseproxy berechnet, daraus
`FrameMetrics` (bestehende `calculate_frame_metrics()`) und
`FrameStarMetrics` (bestehende `measure_frame_stars()`, Referenz-Sternzahl
vom ersten Frame) — beide bereits bestehende, ungeänderte Funktionen, jetzt
nur mit dem neuen Proxy statt einem vorgewarpten Bild gefüttert, exakt wie
§11.9 es vorsieht.

**4 neue Tests:** Ausgabe garantiert echt im offenen Intervall `(0,1)`;
ein isoliert stark verrauschter Frame erhält nachweislich ein niedrigeres
`G_quality` (bewusst mit `w_grad=0` isoliert getestet — reines
Pixelrauschen erhöht in der wiederverwendeten Formel auch
`gradient_energy`, die die Formel als „mehr Detail = gut" belohnt; das ist
ein Verhalten der **bestehenden, unveränderten** Formel, kein Verhalten
dieser Revision, und wird hier bewusst nicht mitgetestet, um keine
Behauptung über eine fremde Formel aufzustellen, die diese Revision nicht
geändert hat); die Sigmoid-Transformation ist exakt nachvollziehbar
(Rücktransformation `w=g/(1-g)` reproduziert `g` exakt); MONO läuft ohne
Bayer-Pattern durch.

**Real auf 4 echten M31-Rohframes verifiziert** (GBRG, 3840×2160, per
Wegwerf-Treiber direkt gegen die Funktion, danach entfernt):

```text
elapsed_s≈1.97 für 4 Frames (≈0,49 s/Frame, inkl. Sternerkennung)
G_quality = [0.569, 0.709, 0.336, 0.504]
```

Alle vier Werte echt im offenen Intervall `(0,1)`, keine NaN/Abstürze, reale
Differenzierung zwischen Frames sichtbar (0,336-0,709) — plausibel für real
unterschiedliche Seeing-/Tracking-Qualität zwischen Subs. Laufzeit
hochgerechnet auf einen typischen 20-100-Frame-Lauf (≈10-50 s für
`GLOBAL_QUALITY`) unproblematisch.

**Bewusst noch nicht umgesetzt:** `QualityFrameWeightPlan`-Persistenz
(Hash-Stabilität über Frame-IDs), Verdrahtung von `G_quality(f)` in
`G_eff(f)` zusammen mit `model_prediction_factor`/`registration_residual_factor`
(bereits in `RegistrationSamplingPlan` vorhanden, aber noch nicht mit
`G_quality` multipliziert), Zero-Veto-Maskenweiterleitung aus dem Proxy,
Runner-Anbindung als echte `GLOBAL_QUALITY`-Phase.

Build (alle Targets) und volle Suite grün (359/360; der eine Fehlschlag
weiterhin `test_acceleration_backend.cpp:254`, unverändert umgebungsbedingt).

---

<a id="historie-30-20"></a>

### 30.20 M3: `QualityFrameWeightPlan` mit kanonischem Hash und fail-closed-Loader (2026-09-05)

Weiter mit dem in 30.19 benannten nächsten Schritt: die
`QualityFrameWeightPlan`-Struktur aus §11.9 und die einmalige Berechnung von
`G_eff(f)`. Neue Dateien:
`include/tile_compile/reconstruction/quality_frame_weight_plan.hpp`,
`src/reconstruction/quality_frame_weight_plan.cpp`,
`tests/test_quality_frame_weight_plan.cpp`.

**Umgesetzt, exakt nach §11.9:**
- `G_eff(f) = G_quality(f) · model_prediction_factor(f) ·
  registration_residual_factor(f)`, genau **einmal** vor der
  Pixelrekonstruktion berechnet;
- die beiden Registrierungsfaktoren werden **wörtlich** aus dem
  `RegistrationSamplingPlan` übernommen (wo `runner_phase_registration` sie
  bereits berechnet und persistiert hat) — hier nur gelesen, nie neu
  berechnet: „Eine doppelte Anwendung der Registrierungsfaktoren in Pipeline
  und Rekonstruktor ist ausgeschlossen" (§11.9) ist damit strukturell
  garantiert, nicht nur beabsichtigt;
- kanonischer, byte-exakter Hash über alle Felder (feste Feldreihenfolge,
  Little-Endian, IEEE-754-Bitmuster, NaN-Payload normalisiert) — **dasselbe
  Schema** wie `compute_plan_hash()` für den `RegistrationSamplingPlan`,
  bewusst ein zweiter `ByteSink` im neuen Modul statt einer geteilten
  Abhängigkeit (wie schon beim FITS-Maskenschreiber begründet: kleine, exakt
  gespiegelte Kopie statt Cross-TU-Export einer Anonymous-Namespace-Hilfe);
- `compute_source_quality_config_hash()`: kanonischer Hash über
  `proxy_version=1` plus jeden numerischen `GlobalQualityConfig`-Parameter,
  der `G_quality(f)` beeinflusst — gehört zur Source-Quality-Hashdomäne
  (§18.3), eine Änderung invalidiert `G_quality` und alle Q-Profile;
- **fail-closed-Loader:** `parse_quality_frame_weight_plan()` lehnt ein
  Artefakt ab, dessen gespeicherter `plan_hash` nicht zu einem
  Frisch-Recompute passt, **und** eines, dessen `g_eff` nicht exakt das
  Produkt seiner drei Faktoren ist (getrennte Prüfung, greift auch wenn der
  Hash zum manipulierten Wert passt) — ein Serialisierungs-Round-trip allein
  beweist keine sichere Ablehnung beschädigter Artefakte (Auditbefund A6).

**7 neue Tests:** `g_eff` exakt als Produkt (`0,5·0,8·0,9 = 0,36`);
Registrierungsfaktoren wörtlich übernommen; Größen-Mismatch zwischen
`g_quality`-Vektor und Sampling-Plan abgelehnt; verlustfreier JSON-Round-trip
mit Hash-Revalidierung; manipuliertes Artefakt (ein `g_quality`-Digit
geändert, `plan_hash` unberührt) fail-closed abgelehnt; **inkonsistentes
`g_eff`** (nicht das Produkt, aber mit dazu passendem Hash) trotzdem
abgelehnt; Config-Hash stabil bei gleichem Config, ändert sich bei jeder
Gewichts-/Clamp-/Stern-Parameteränderung; `plan_hash` stabil bei gleicher
Eingabe, ändert sich bei minimaler `g_quality`-Änderung (`0,750 → 0,751`).

**Verifikationsmethodik:** Dies ist ein reines Daten-/Hashing-Modul ohne
Bildverarbeitung — deterministisch mit bekannter Ground Truth. Die
synthetischen Tests mit exakten Hash-Stabilitäts- und
Fail-closed-Loader-Prüfungen **sind** hier die passende und vollständige
Verifikation; ein Echtdatenlauf würde nichts zusätzlich zeigen (kein
plausibler Echtdatenpfad für eine Hash-/Serialisierungsschicht).

**Bewusst noch nicht umgesetzt:** Persistenz als echtes Laufartefakt
(`artifacts/quality_frame_weight_plan.json`), Verdrahtung von `G_eff(f)` in
`compute_forward_drizzle_uniform_and_raw()` (dort stehen `G_eff`/`Q_composite`
weiterhin auf `1.0`, 30.16), Runner-Anbindung als echte `GLOBAL_QUALITY`-Phase,
Zero-Veto-Maskenweiterleitung aus dem Proxy.

Build (alle Targets) und volle Suite grün (366/367; der eine Fehlschlag
weiterhin `test_acceleration_backend.cpp:254`, unverändert umgebungsbedingt).

---

<a id="historie-30-21"></a>

### 30.21 M3: `G_eff(f)` in `compute_forward_drizzle_uniform_and_raw()` verdrahtet — Raw weicht jetzt echt von Uniform ab (2026-09-05)

Der in 30.16/30.20 benannte nächste Schritt ist umgesetzt: `w_raw = B_f,c(q) ·
G_eff(f) · Q_composite` verwendet jetzt das echte, pro Frame skalare
`G_eff(f)`. Neuer optionaler Parameter `g_eff_by_source_index` (indiziert
über `FrameSamplingTransform::source_index`, Werte in `(0,1]` aus einem
`QualityFrameWeightPlan`).

**Verbindliches Verhalten:**
- Leerer `g_eff`-Vektor → `G_eff = 1.0` für alle Frames (rückwärtskompatibel,
  Raw bitidentisch zum geclippten Uniform — per Test verifiziert);
- gefüllter Vektor → Raw und Uniform teilen **dieselbe** Clipping-Maske
  (die Clipping-Entscheidung nutzt weiterhin ausschließlich das geometrische
  Gewicht `B`, §11.8), unterscheiden sich danach aber exakt um den
  Pro-Frame-Faktor `G_eff(f)` in der Aggregation;
- der Support ist in beiden Profilen identisch (`G_eff > 0` setzt nichts auf
  0);
- `Q_composite` bleibt `1.0` — es ist frame-lokal pro Pixel und braucht die
  Q-Map-Infrastruktur (§13, M5);
- Größen-Mismatch (`g_eff`-Vektor ≠ Framezahl) wird fail-closed abgelehnt.

**4 neue Tests, mit von Hand nachgerechneten Werten:** ohne `g_eff` Raw ==
geclipptes Uniform (Support, Weight-Sum, Wert an belegten Pixeln); mit
`g_eff = [1,0; 0,25]` auf zwei identisch registrierten Frames mit Werten
`10`/`30` bleibt Uniform der schlichte Mittelwert `20`, Raw wird zu
`(1·10 + 0,25·30)/1,25 = 14` (zum höher gewichteten Frame gezogen); Support
in beiden Profilen gleich; falsch dimensionierter `g_eff`-Vektor abgelehnt.

**Nebenbefund beim Testen (kein Bug, dokumentiert):** Die geclippte
`compute_forward_drizzle_uniform_and_raw()` unterliegt — anders als die
ungeclippte M2-`compute_forward_drizzle_uniform()` — dem `min_n_eff`-Pixelveto
aus §11.8 Schritt 8. Mit dem Default `min_n_eff = 3.0` und nur zwei
gleichgewichteten Frames ist `n_eff = (2B)²/(2B²) = 2.0 < 3.0`, also wird
jedes Pixel korrekt verworfen. Die Zwei-Frame-Tests setzen deshalb bewusst
`min_n_eff = 1.0`, um das `G_eff`-Verhalten isoliert zu prüfen — das
Default-Veto selbst wird separat getestet (30.16).

**Bewusst noch offen:** Persistenz des `QualityFrameWeightPlan` als
Laufartefakt, die tatsächliche Übergabe eines aus einem realen
`GLOBAL_QUALITY`-Lauf gewonnenen `g_eff`-Vektors an diese Funktion im Runner,
Zero-Veto-Maskenweiterleitung, Raw-Baseline-Persistenz.

Build (alle Targets) und volle Suite grün (369/370; der eine Fehlschlag
weiterhin `test_acceleration_backend.cpp:254`, unverändert umgebungsbedingt).

---

<a id="historie-30-22"></a>

### 30.22 M3: Checksummierter Profilstore-Manifest + fail-closed-Verifikation, real verifiziert (2026-09-05)

M3-Abnahmekriterium „Raw wird atomar mit Checksumme persistiert;
Uniform-Fallback funktioniert" — erster Teil umgesetzt. Neues Modul:
`include/tile_compile/reconstruction/profile_store_manifest.hpp`,
`src/reconstruction/profile_store_manifest.cpp`,
`tests/test_profile_store_manifest.cpp`.

**Umgesetzt:**
- `ProfileStoreManifest`: pro persistierter Ebenendatei Name, Dimensionen und
  `sha256` der Dateibytes (kanonische, nach Name sortierte Reihenfolge) plus
  ein kanonischer Hash über das gesamte Manifest (gleiches `ByteSink`-Schema
  wie `RegistrationSamplingPlan`/`QualityFrameWeightPlan`);
- `build_profile_store_manifest()` liest die bereits **atomar** geschriebenen
  FITS-Ebenen (`io::write_fits_float` staged+fsync+rename, 30.14) und hasht
  sie;
- `verify_profile_store()` re-hasht jede Ebenendatei im Verzeichnis gegen das
  Manifest und meldet `missing`/`corrupt` getrennt; `usable` ist nur dann
  `true`, wenn Manifest-Hash re-validiert **und** jede Ebene vorhanden **und**
  jede Prüfsumme passt — genau die „Raw benutzen"-Bedingung, sonst Fallback
  auf Uniform;
- `parse_profile_store_manifest()` lehnt ein Manifest mit nicht passendem
  `manifest_hash` fail-closed ab (Auditbefund A6).
- In `write_forward_drizzle_uniform_store()` verdrahtet: schreibt jetzt
  zusätzlich `store_manifest.json` (kanonische, atomare Textausgabe) neben
  den bestehenden diagnostischen `manifest.json`.

**5 neue synthetische Tests** (deterministisches Daten-/Hashing-Modul,
bekannte Ground Truth): vollständiger Store verifiziert `usable`; eine
manipulierte Ebenendatei wird als `corrupt` erkannt, Store nicht `usable`
(der Uniform-Fallback-Pfad); eine gelöschte Ebenendatei als `missing`;
JSON-Round-trip mit Hash-Revalidierung, manipuliertes Manifest abgelehnt;
Manifest-Hash stabil bei gleicher Eingabe, ändert sich bei
Höhe/Profil-/Dateiinhaltsänderung.

**Real auf echten M31-Daten verifiziert** (4 Frames,
`tile_compile_legacy_reference`, `persist_forward_drizzle_uniform_store: true`,
Verifikation per Wegwerf-Treiber, danach entfernt):

```text
store_manifest.json geschrieben: 12 Ebenen (R/G/B × value/weight_sum/n_eff/support),
  je 7696x4328, sha256 vorhanden, kanonisch sortiert, manifest_hash gesetzt
verify_profile_store (unveränderter Store): usable=1, manifest_hash_ok=1, missing=0, corrupt=0
verify_profile_store (1 Byte an R_value.fits angehängt): usable=0, corrupt=1 (R_value)
```

Der Fail-closed-Pfad greift also nachweislich auf echten Daten, nicht nur im
synthetischen Test.

**Bewusst noch offen:** Persistenz eines eigenen **Raw**-Stores (aktuell
persistiert nur der M2-Uniform-Preview-Store; ein Raw-Store aus
`compute_forward_drizzle_uniform_and_raw()` braucht die Runner-Anbindung
dieser Funktion), Uniform-Fallback-Logik im Resume-Pfad
(`verify_profile_store` liefert das Signal, der Resume-Einstieg konsumiert es
noch nicht), Whole-Store-Transaktionalität über alle Ebenendateien
(einzeln atomar, nicht als eine Transaktion — 30.14).

Build (alle Targets) und volle Suite grün (374/375; der eine Fehlschlag
weiterhin `test_acceleration_backend.cpp:254`, unverändert umgebungsbedingt).

---

<a id="historie-30-23"></a>

### 30.23 M3 durchgehend integriert: neue Phasen, `reconstruct`-Einstieg, transaktionaler Profilstore, real end-to-end auf M31 verifiziert (2026-09-05)

Zwischen 30.22 und dieser Notiz ist die M0–M3-Durchgehend-Integration
(Audit §4, Arbeitspaket 4) fertiggestellt worden — mehrere zusammengehörige
Module, die die bis dahin einzeln implementierten M3-Bausteine zu einer
laufenden, resumefähigen Pipeline verbinden:

**Neue `Phase`-Enum-Werte** (`core/types.hpp`, ans Ende angehängt, bestehende
Ganzzahlwerte unverschoben — der in §0.1/B4 als „bewusst zurückgestellt"
markierte Punkt ist damit erledigt):
`NORMALIZED_CACHE=24`, `SAMPLING_GEOMETRY=25`, `GLOBAL_QUALITY=26`,
`FORWARD_DRIZZLE=27`. Ein fehlschlagender Coverage-Gate meldet jetzt echt
`Phase::SAMPLING_GEOMETRY`, nicht mehr `Phase::PREWARP`.

**Neue Module:**
- `reconstruction/normalized_source_cache.hpp/.cpp`:
  `VerifiedNormalizedSourceCache` (§10.1) liest das bestehende
  `<source_index>.raw`-Cacheformat, hasht jede Datei, verifiziert gegen ein
  publiziertes Manifest und lädt speicherbudgetiert; eine später
  veränderte/abgeschnittene Datei scheitert fail-closed beim Laden ihrer
  Bytes. `publish_normalized_source_manifest()` registriert vorhandene
  Dateien, normalisiert/repariert/kopiert nichts.
- `reconstruction/drizzle_profile_store.hpp/.cpp`: transaktionaler
  Profilstore (§11.3). Unveränderliche `generation-*`-Verzeichnisse,
  `current.json` als einziger Commit-Punkt; ein unterbrochener Schreiber
  erhält den vorherigen Commit; keine automatische GC alter Generationen.
  `DrizzleStoreIdentity` bindet Source-/Sampling-/Reconstruction-/Cache-/
  Quality-Hash plus `mode` (`uniform_unclipped` bzw. `uniform_raw_clipped`).
  `persist_forward_drizzle_uniform()` und `persist_forward_drizzle_uniform_and_raw()`
  streamen die Ebenen budgetiert; bounded Region-Reads.
- `reconstruction/source_quality_artifact.hpp/.cpp`:
  `persist_source_quality_artifact()`/`load_source_quality_artifact()` für
  das `QualityFrameWeightPlan`-Laufartefakt (§13.4), `resolve_quality_frame_weights()`
  mit Pflicht-Identitätsprüfung und vor der Allokation begrenzter
  Ausgabegröße, sowie `persist_forward_drizzle_from_predecessors()` als
  Bibliotheks-Orchestrierung mit verpflichtenden Vorgänger-Checks.
- `apps/runner_forward_drizzle.hpp/.cpp`: `run_forward_drizzle_stages()`
  (fresh: Cache versiegeln → Geometrie → Common-Overlap → Quality →
  gepaarter Profilstore; resume: Checkpoint/Vorgänger vor jeder Phase oder
  jedem Artefaktschreiben validieren) und `resume_forward_drizzle_command()`.
- CLI: neue Subcommands `reconstruct` („Run M1-M3 to checked internal
  profiles") und `resume-reconstruction`, verdrahtet über
  `run_pipeline_command(..., forward_drizzle_only=true)` in
  `runner_pipeline.cpp` — dieser Pfad erzwingt `parallel_workers=1`,
  verwendet **kein** PREWARP-Nutzsignal und startet **kein** Backend
  (`run_provenance.execution_scope = "forward_drizzle_m1_m3"`).

**Tests (17 neue Fälle in 3 Dateien):** Store-Round-trip inkl. Kontext;
unterbrochene nächste Generation erhält vorherigen Commit; unvollständige/
gefälschte Ebenen scheitern trotz neu gehashtem Manifest; OSC-Paar streamt
im Budget, das Vollbilder ablehnt; Speicherablehnung vor I/O und Publikation;
bounded Region-Reads; kein unchecked-Generation-Select bei malformiertem
Commit. Quellcache: inhaltsgebundenes Framela­den lehnt Ersetzung/Truncation
ab; Provenienz-Mismatch/unvollständige Publikation fail-closed. Quality-
Artefakt: Frame-Identitätsbindung unabhängig von der Artefaktreihenfolge;
persist/load/raw-Rekonstruktion verlangen passende Vorgänger; Memory-Preflight
vor Cache-Reads und erhält altes Artefakt; extreme Source-Indizes können keine
unbegrenzte Gewichts-Allokation auslösen. Forward-Runner: geordnete Phasen
behalten Cache und erzeugen nie PREWARP-Frames; Geometrie-Veto beendet vor
Overlap/Rekonstruktion; Resume validiert Vorgänger vor Phasenstart; geänderte
Config/Geometrie-Artefakt lehnt Resume ab.

**Real end-to-end auf M31 verifiziert** (`tile_compile_runner reconstruct`,
6 Frames, GBRG, `internal_scale=1`, bewusst gelockerter Coverage-Gate um den
vollen Pfad statt eines Fail-closed-Abbruchs zu prüfen):

```text
Phasenfolge: SCAN_INPUT → CHANNEL_SPLIT → NORMALIZATION → REGISTRATION
  → NORMALIZED_CACHE(24) → SAMPLING_GEOMETRY(25, ~35 s) → COMMON_OVERLAP(7)
  → GLOBAL_QUALITY(26, ~2,5 s) → FORWARD_DRIZZLE(27, ~32 s, peak ~649 MB)
run_end: status=reconstruction_ready, success=true, final_image_available=false
```

Artefakte real geschrieben und geprüft: `source_quality_plan.json` mit echten
per-Frame `g_quality` (0,17–0,95, alle im offenen Intervall (0,1)),
`registration_residual_factor` aus dem Sampling-Plan übernommen, `g_eff` als
Produkt (Frame 1: `g_quality=0,168` → `g_eff=0,111`, stark abgewertet).
`forward_drizzle_profiles/generation-*/` mit 24 FITS-Ebenen (uniform + raw,
je R/G/B × value/weight_sum/n_eff/support), `current.json` mit vollständiger
Identitätskette, `commit_hash` und checksummiertem Ebenen-Manifest.
**Raw ≠ Uniform auf echten Daten**: `mean|u−r|` ≈ 2,1–2,6, `max` bis ~590 —
die per-Frame-`G_eff`-Gewichtung verschiebt Raw real gegenüber Uniform;
die Support-Masken sind bitidentisch (`sha256(raw_R_support) ==
sha256(uniform_R_support)`), wie §11.8 verlangt (geteilte Akzeptanzmaske).
Bei nur 8 Frames und Produktionsdefault (`internal_scale=2`,
`pixfrac` default) scheitert `SAMPLING_GEOMETRY` korrekt mit
`FORWARD_STAGE_COVERAGE_GATE_FAILED` — fail-closed wie §9.5 gefordert, kein
stiller Fallback.

Damit ist M3 auf Code- und Integrationsebene im Wesentlichen abgeschlossen.
Verbleibend (nicht mehr blockierend für M4): der frame-lokale
`Q_composite`-Stream (§13.4, überlappt mit M5), die A-priori-Streifengrößen-
Schranke für die Kandidatenliste, und die synthetischen Proxy-Tests gegen
farbige Sterne/schmalbandiges MONO/Veto-Leckage aus §13.2.

Volle Hauptsuite **398/399** (der eine Fehlschlag weiterhin
`test_acceleration_backend.cpp:254`, unverändert umgebungsbedingt);
`tile_compile_runner`/`_legacy_reference`/`_cli` bauen grün.

---

<a id="historie-30-24"></a>

### 30.24 M4 begonnen: 2x→1x-Flächenmittel-Operator und WCS-Skalierung (§12) (2026-09-05)

Mit M3 durchgehend integriert (30.23) beginnt M4 (internes 2x-Raster →
Ausgabegeometrie). Neues Modul:
`include/tile_compile/reconstruction/output_scale.hpp`,
`src/reconstruction/output_scale.cpp`, `tests/test_output_scale.cpp`.

**Umgesetzt, exakt nach §12:**
- **`downsample_profile_plane_2x2()`** — der deterministische 2×2→1x-Flächenmittel-
  Operator aus §12.1 mit dem strengen 4/4-Support-Vertrag:
  `valid_out = valid_00 && valid_01 && valid_10 && valid_11`,
  `value_out = 0,25·(v_00+v_01+v_10+v_11)`,
  `n_eff_out = min(n_eff_00, …, n_eff_11)`. Ein ungültiges Subpixel geht
  **nie** als 0 oder teilnormalisierter Mittelwert ein — das 1x-Pixel wird
  ungültig. `weight_sum` wird wie `value` flächengemittelt (dokumentierte
  Konvention; der Plan fixiert nur `value` und `n_eff`). Ungerade
  Internal-Dimension verwirft die letzte Zeile/Spalte (dieselbe Regel wie
  das Quad-Grün-Gitter), sodass der Operator immer exakt 2×2 ist.
- **`downsample_uniform_and_raw_2x2()`** — wendet den Operator auf jede
  vorhandene Ebene eines Uniform+Raw-Ergebnisses an; Support-Maske bleibt
  über beide Profile geteilt.
- **`scale_wcs_to_output()`** — §12.2 komponentenweise, wörtlich:
  `CRPIX_canvas_native = CRPIX_in + canvas_offset_native`;
  `CRPIX_out = S·(CRPIX_canvas_native − 0,5) + 0,5 − crop_origin_out`
  (das ist die Standard-FITS-Rebin-Form `S·CRPIX − (S−1)/2` plus
  Canvas-Offset und explizites Minus für den Crop); `CD_out = CD_in / S`.
- **`OutputScaleMode`** — explizite Modi `1/1`, `2/1`, `2/2` ohne Auto;
  `valid()` lehnt `output_scale > internal_scale` ab; `needs_2x2_downsample()`
  ist nur für `2/1` wahr.

**8 neue Tests, mit von Hand nachgerechneten Werten:** voll gültiges Quad
mittelt Wert/Gewicht und nimmt das Minimum-`n_eff` (`0,25·(10+20+30+40)=25`,
`min(8,3,5,2)=2`); ein einziges ungültiges Subpixel macht das 1x-Pixel
ungültig (kein Teilmittel, keine 0); ungerade Dimension verwirft
Zeile/Spalte; konstantes Feld → gleiche Konstante (Oberflächenhelligkeit
erhalten); Modus-Validierung; WCS bei `S=1` mit Canvas-Offset und
nativem Crop gegen die Handrechnung; WCS bei `S=2` (Rebin `2·CRPIX − 0,5`,
CD halbiert); 2×2 auf einem Uniform+Raw-Ergebnis (beide Profile landen bei
1x, Support geteilt).

**Bewusst noch nicht umgesetzt (nächster Schritt):** die Verdrahtung des
`2/1`-Modus in die `FORWARD_DRIZZLE`-Runner-Phase. Der Store
(`persist_forward_drizzle_from_predecessors` →
`persist_forward_drizzle_uniform_and_raw`) **streamt** die Ebenen
speicherbudgetiert; ein streamender 2×2-Downsample braucht ein
2-Internal-Zeilen-Fenster pro Ausgabezeile und berührt die
Streaming-Interna des transaktionalen Stores. Da dieser Bereich parallel
aktiv weiterentwickelt wird (§0.3/0.4/30.23), wird der M4-Kernoperator hier
als geprüftes, in sich abgeschlossenes Modul bereitgestellt und die
Store-Integration als koordinierter Folgeschritt gehalten, statt jetzt in
die Streaming-Interna einzugreifen. Ebenfalls offen: die
Downstream-2x-Fähigkeit von BGE/PCC/HMS (§12.3).

**Nachgezogen im selben Durchgang:**

- **Kernel-Autokorrelations-Korrekturfaktor (§12.4)** — von Grund auf
  hergeleitet, keine gefittete Konstante: für weißes Eingangsrauschen ist die
  Varianz einer Groß-Apertursumme erhalten (`= σ²·N_out`), die naive
  Unabhängigkeitsschätzung dagegen `N_out·σ²·S0/W²` mit `W = d = pixfrac·internal_scale`
  und `S0 = Σ_j overlap([j+0,5−d/2, j+0,5+d/2], [0,1])²`. Der
  Sigma-Korrekturfaktor ist damit `f = W/√S0` (`≥ 1`, exakt `1` bei `d = 1`).
  `kernel_noise_correlation_sigma_factor()` + `kernel_noise_autocorrelation_1d()`
  (Lag-Profil `ρ_Δ`). 4 Tests mit exakter Handrechnung: `d=1` → `f=1`, keine
  Korrelation; `d=2` → `f = 2/√1,5`, `ρ_1 = 1,0/1,5`, und
  `√(Σ_alle_Lags ρ_Δ) == f` als Konsistenzprüfung; Produktionsdefault `d=1,6`
  → `f = 1,6/√1,18 ≈ 1,47`; ungültige Argumente abgelehnt. Dies ist der
  Dichte-Eingang-Referenzfall (ein Sample pro Internal-Pixel); die
  per-Kanal-dünnbesetzte Verfeinerung (R/B alle 2 nativen Pixel, mehr
  Korrelation) ist separat und noch nicht gerechnet. Ausweisung in
  `forward_drizzle.json` noch offen.
- **§13.2-Pflichttests für den Analyseproxy** (in 30.18 als offen
  vermerkt): Bayer-Checkerboard (reine R/B-Chroma-Extreme bei flachem
  Grün-Gitter → `sigma_green` bleibt exakt `0`, kein Chroma-Leck ins
  Quad-Grün-Gitter); farbiger (roter) Stern (`R=9000`, `G=400`, `B=30`) →
  Quad-Grün-Wert folgt dem echten Grünfluss (`150 < g < 1500`), kategorisch
  nicht dem 9000er-Rot; schmalbandiges MONO (Rampe + scharfer Spike) →
  `proxy_full` exakt gleich der Eingabe, `sigma_green` exakt gleich
  `median_absolute_deviation_sigma(L − B3_blur(L))`. Veto-Leckage bleibt
  offen, da die Zero-Veto-Maske in `proxy_version=1` noch nicht implementiert
  ist.

Build (alle Targets inkl. `tile_compile_legacy_reference_tests`) grün, volle
Hauptsuite **413/414** (der eine Fehlschlag weiterhin
`test_acceleration_backend.cpp:254`, unverändert umgebungsbedingt; +15 neue
Fälle: 8 Output-Scale/WCS + 4 Kernel-Noise + 3 Proxy-§13.2).

---

<a id="historie-30-25"></a>

### 30.25 M4: speicher-begrenzter `2/1`-Streaming-Downsample, bit-identisch zur Referenz (2026-09-05)

Der in 30.24 als „koordinierter Folgeschritt" gehaltene Punkt (Verdrahtung
des `2/1`-Modus in die **streamende** Store-Phase) ist als geprüftes
Bibliotheks-Primitiv umgesetzt, ohne in die Store-Interna einzugreifen:

**`stream_forward_drizzle_uniform_and_raw_2x2()`** (`output_scale.hpp/cpp`)
umschließt das inzwischen vorhandene `stream_forward_drizzle_uniform_and_raw()`
mit einem zeilengepufferten 2×2→1x-Adapter (`Downsample2x2Adapter`): Er
sammelt ganze Internal-Zeilen je Ebenenfeld, gibt für jedes vollständige,
gerade ausgerichtete Zeilenpaar **eine** Ausgabezeile aus und hält damit
höchstens eine Internal-Zeile Übertrag — der Aufrufer (später der
transaktionale Store) hält nie ein Vollbild in Internal-Auflösung.
Nicht-zusammenhängende oder ungerade Streifen scheitern fail-closed
(`DRIZZLE_2X2_NONCONTIGUOUS_STRIPE` / `_ODD_INTERNAL_HEIGHT`); verlangt
`internal_scale == 2`.

**Bit-Identität nachgewiesen:** Ein Test vergleicht die Streaming-Ausgabe
gegen `downsample_uniform_and_raw_2x2(compute_forward_drizzle_uniform_and_raw(...))`
(die nicht-streamende Referenz) auf einem 4-Frame-OSC-Fixture mit Rotation
und echtem `g_eff`-Vektor — für **jede** Internal-Chunkhöhe `{1, 3, 7, 1000}`
sind `support`, `weight_sum`, `n_eff` exakt gleich und `value` an belegten
Pixeln exakt gleich (3529 Assertions). Der Fix, der das ermöglichte:
Doppelt-Zwischensummen + Reihenfolge `(00,01,10,11)` im Adapter, exakt wie
im nicht-streamenden `downsample_profile_plane_2x2()` — reine
`float`-Akkumulation im Adapter war zunächst nicht bitgleich.

**Verbleibend für den vollen `2/1`-Store:** `make_drizzle_store_identity()`
muss `output_scale` in den `reconstruction_hash` aufnehmen und
`i.width/height` auf die Ausgabegeometrie setzen; danach tauscht
`persist_forward_drizzle_uniform_and_raw()` bei `2/1` den Sink gegen diesen
Adapter. Das berührt den Identitätsvertrag des parallel entwickelten Stores
und wird koordiniert gemacht; das speicher-begrenzte, bit-exakte Primitiv
liegt jetzt bereit.

Build (alle Targets) grün, Hauptsuite **415/416** (der eine Fehlschlag
weiterhin `test_acceleration_backend.cpp:254`; +2 neue Fälle).

---

<a id="historie-30-26"></a>

### 30.26 M4: `2/1`-Modus im transaktionalen Store aktiviert und real verdrahtet (2026-09-05)

Der `2/1`-Modus (interner 2x-Kern, 1x-Ausgabe, Produktionsdefault) ist jetzt
im Store und in der Runner-Phase durchgängig aktiv:

- **`make_drizzle_store_identity()`** validiert `OutputScaleMode{internal_scale,
  output_scale}` (`output_scale ≤ internal_scale`, kein Auto), nimmt
  `output_scale` in den `reconstruction_hash` auf und setzt `i.width/height`
  bei `2/1` auf die halbierte Ausgabegeometrie (`1/1` und `2/2` speichern in
  Internal-Auflösung). Der `2/1`-Identitätshash unterscheidet sich damit
  nachweislich vom `2/2`-Hash derselben Config.
- **`persist_forward_drizzle_uniform_and_raw()`** wählt bei `2/1` den
  `stream_forward_drizzle_uniform_and_raw_2x2()`-Streaming-Pfad (30.25) — der
  Store bekommt Streifen bereits in 1x-Auflösung, nie ein Vollbild in
  Internal-Auflösung. Der reine `persist_forward_drizzle_uniform()`-
  Diagnosepfad lehnt `2/1` mit klarer Meldung ab (der gepaarte Pfad trägt
  den Downsample).
- **`run_forward_drizzle_stages()`** meldet im `FORWARD_DRIZZLE`-`phase_end`
  jetzt `internal_scale`, `output_scale`, `output_scale_applied` (statt
  hartkodiert `false`) und den **Kernel-Rausch-Sigma-Faktor**
  `kernel_noise_correlation_sigma_factor(pixfrac, internal_scale)` (§12.4),
  damit Downstream-Schätzer die effektive Rauschbandbreite kennen.

**Neuer Test** (`[drizzle-store]`): `2/1`-Lauf auf einem 2-Frame-OSC-Fixture
persistiert Ebenen in `12×12` (= `canvas_native`, nicht `24×24` internal),
`verify_drizzle_profile_store` ist `usable`, und die gespeicherten Werte sind
an belegten Pixeln **bit-identisch** zu
`downsample_uniform_and_raw_2x2(compute_forward_drizzle_uniform_and_raw(...))`.

Build (alle Targets) grün, Hauptsuite **416/417** (der eine Fehlschlag
weiterhin `test_acceleration_backend.cpp:254`; +1 neuer Fall).

**Verbleibend für M4 (nicht mehr Algorithmus-, sondern Cutover-Arbeit):** die
Downstream-Phasen (STACKING/DEBAYER/ASTROMETRY/BGE/PCC/HMS) laufen im
`reconstruct`-Einstieg noch **nicht** — dieser endet bei
`reconstruction_ready`. Die 2x-Fähigkeit von BGE/PCC/HMS und die Ausweisung
von `f` in einem `forward_drizzle.json` (statt nur im `phase_end`-Event)
gehören zur Verdrahtung der Ausgabe-Pipeline auf den Profilstore, die mit M6
(Mehrbandfusion → finales Bild) und dem M10-Cutover kommt.

<a id="historie-30-27"></a>

### 30.27 M5 begonnen: skalenspezifische Source-Q-Maps — Per-Scale-Hook + MONO-Bit-Identität (2026-09-05)

Erster M5-Schritt nach der in §4 fixierten Reihenfolge: die **skalenspezifische
Q-Map-Berechnung mit Streaming-Sink**, bevor der Multistream-Cache und die
Region-Reads gebaut werden. Grundsatzentscheidung (bestätigt durch §13.1 und
die M5-Abnahmezeile „Composite bleibt mit der übernommenen, dokumentierten
Quality-Semantik vergleichbar"): **kein neues `psi`**. Die bestehende
Per-Scale-Mathematik von `metrics::compute_aqmh_quality_map()`
(`compute_psi` = `clamp(sigmoid(score_scale·(w_sharp·z_sharp + w_snr·z_snr))·artifact, 0, 1)`
mit `robust_zscore` auf `sharp`/`snr`, Composite = `exp(Σ log psi / computed_scales)`
mit Veto-Propagation) wird **wörtlich wiederverwendet**. Geändert wird nur, was
exponiert wird.

**Chirurgischer Eingriff statt 600 Zeilen Spiegelcode.**
`metrics::compute_aqmh_quality_map()` bekommt einen optionalen, per Default
null-`std::function`-Hook `PerScaleQualityHook(scale_index, downsample_factor,
psi, artifact)`, aufgerufen unmittelbar nach `compute_psi` und **vor**
`accumulate_upsampled_log_psi`. Null-Hook ⇒ byte-für-byte identisches Verhalten;
der Legacy-PREWARP-Q-Map-Pfad übergibt keinen Hook. Bewiesen: die 15
Legacy-`aqmh_quality_map`-Fälle bleiben unverändert grün.

**Neues Modul `reconstruction/source_quality_maps.{hpp,cpp}}`:**
`compute_source_quality_maps(analysis_proxy, source_valid_mask, w, h,
AqmhPyramidConfig, sink=null)`.

- `analysis_proxy` ist der **source-aufgelöste** Analyseproxy: MONO = normalisierte
  L-Ebene unverändert, OSC = edge-aware Voll-Auflösungs-Grünproxy
  (`compute_source_quality_proxy_v1(...).proxy_full`, §13.2 Schritt 3 erlaubt
  ihn ausdrücklich für positionsbezogene Analysefunktionen — Schärfe/SNR/Artefakt
  sind alle fensterbasiert-positionsbezogen). Damit ist die Composite-Geometrie
  identisch mit der Source-Geometrie, ohne Zwischengitter-Umtastung; die
  reine Quad-Gitter-`sigma_green`-Skalar wird bereits von M3s `G_quality`
  konsumiert. Der Quad-Gitter-only-Schärfevariante bleibt einer künftigen
  `proxy_version`-Anhebung vorbehalten.
- Ergebnis: `q_map` (Composite in Source-Geometrie), `scale_maps`
  (`ScaleQualityMap{scale_index, downsample_factor = 1<<(2·scale_index),
  psi in Source-Geometrie}` — nur befüllt, wenn **kein** Sink),
  `artifact_confidence` (`phi_artifact` der **feinsten** berechneten Skala,
  auf Source-Geometrie hochgetastet, `1 = sauber`), Diagnostik
  (`computed_scales`, `omitted_scales`, `composite_p50`,
  `peak_resident_scale_maps`).
- **Residenz** (§13.3 „Nur der laufende Composite-Accumulator bleibt
  resident"): mit Sink wird jede Skala nach der Hochtastung sofort an
  `QualityScaleMapSink(scale_index, downsample_factor, psi_source_geom)`
  gereicht und freigegeben — nachweislich `peak_resident_scale_maps == 1`
  (der Composite-Accumulator ist der `double`-Log-Sum in
  `compute_aqmh_quality_map`). Ohne Sink hält `scale_maps` alle → gleich
  `computed_scales`.
- **Hochtastung** `upsample_to_source()` spiegelt exakt die Interpolation aus
  `accumulate_upsampled_log_psi` (halbpixelzentrierte Position, geklemmter
  2×2-Stencil, support-gewichtete Normierung), damit die gestreamten
  Skalenkarten mit dem Composite konsistent sind. Harte Null / nichtfinit /
  `≤ 0` ⇒ `NaN` (keine Null wird positiv).

**Verifikation — synthetische Unit-Tests mit von Hand geprüften Invarianten
(`tests/test_source_quality_maps.cpp`, 5 Fälle, ~19,6 k Assertions):**

1. **MONO-Bit-Identität (die M5-Abnahme):** ein synthetisches 96×96-Bild durch
   `metrics::compute_aqmh_quality_map()` und `compute_source_quality_maps()` mit
   Voll-Maske → `q_map` **byte-identisch** (Bitmuster-Vergleich, beide
   nichtfinit ⇒ ok); `computed_scales ≥ 2`.
2. **Sink streamt jede berechnete Skala** mit `scale_index = 0,1,…` und
   `downsample_factor = 1<<(2·s)`; `scale_maps` leer; nie mehr als eine
   Vollkarte gleichzeitig resident (`max_concurrent == 1`,
   `peak_resident_scale_maps == 1`).
3. **Ohne Sink** behält `scale_maps` alle Skalen in Source-Geometrie
   (`h×w`), `downsample_factor` konsistent.
4. **Harte Maske** wird auf den Composite re-appliziert (exakt `0` über den
   gesamten maskierten Block); jenseits der Fensterreichweite ist selbst die
   feinste Skala `NaN`. Fensterrand-Leckage in einem dünnen Band direkt
   innerhalb der Maskenkante ist der Multiskalen-Statistik inhärent und
   deckungsgleich mit dem Legacy-Pfad — die §13.5-Garantie „Null-Veto
   überlebt Umtastung" wird über einen **expliziten Veto-Stream im
   Cache/Region-Read-Layer** durchgesetzt (dort getestet, nächster Schritt).
5. **`artifact_confidence`** ist eine Source-Geometrie-Karte in `[0,1]` auf
   ihrem gültigen Support.

Build grün; Hauptsuite **421/422** (der eine Fehlschlag weiterhin
`test_acceleration_backend.cpp:254`, CUDA-Backend-Auswahl, unabhängig;
+5 neue Fälle).

**Nächste M5-Schritte** (§4-Reihenfolge): Multistream-Cache-Writer
(`cache/source_quality_maps/{composite,scale_0..3,artifact}/` + `metadata.json`,
`uint16`-Quantisierung mit reserviertem Null-Veto-Sentinel, `storage_divisor`
räumlich) mit **getrennt berechnetem** `source_identity_hash` (geordnete
Frame-IDs, Inhaltsidentität, Quellmaße, Farbmodus, Sensororientierung,
Bayer/CFA-Ursprung, Normalized-Cache-Hash — **ohne** Registrierung, Canvas,
`internal_scale`, `output_scale`; der aktuelle
`RegistrationSamplingPlan.source_identity_hash` taugt dafür **nicht**, weil er
`config.sha256` mitführt und damit jede Neuregistrierung invalidieren würde)
und `source_quality_config_hash` (Proxy-Version, Pyramiden-Parameter,
Storage-Divisor, dtype); danach Region-Reads-API + Null-Veto-Leckagetest;
zuletzt `SOURCE_QUALITY_MAPS` als angehängte `Phase = 28` **vor**
`GLOBAL_QUALITY` (§11.9). À-trous / `A_artifact` / Alpha bleiben M6.

<a id="historie-30-28"></a>

### 30.28 M5 fortgesetzt: Multistream-Cache + getrennte Identity-/Config-Hashes + Region-Reads (2026-09-05)

Zweiter M5-Schritt nach §4: der **Multistream-Cache-Writer/Reader**, die
**zwei getrennten Hashes** und die **`uint16`-Quantisierung mit reserviertem
Null-Veto-Sentinel**. Neues Modul
`reconstruction/source_quality_map_cache.{hpp,cpp}`.

**Quantisierung (§13.5).** `quantize_quality(v)`: `0.0f` und jeder nichtfinite
Wert ⇒ Code `0` (Veto); `v ∈ (0,1] ⇒ max(1, round(v·65535)) ∈ [1,65535]`.
`dequantize_quality(0) = NaN`, sonst `q/65535` (für `q ≥ 1` **immer > 0**).
Damit gilt bewiesen: eine exakte Null wird nie positiv, und kein
Nicht-Null-Code dekodiert zu `0`.

**`storage_divisor` = räumlich.** Der einzige Vorkommen im Plan (§13.4-JSON)
trägt `source_width`/`source_height` separat; zusammen mit der §13.5-Warnung
vor „Umtastung macht Null-Veto positiv" ist die räumliche Lesart eindeutig
(Legacy hat dazu `AqmhStorageConfig::resolution_divisor = 2`). Karten werden
auf `ceil(source/divisor)` gespeichert. **Konservative Downtastung:** eine
Speicherzelle, die **irgendein** Veto-Quellpixel überdeckt, wird als Veto
(`0`) gespeichert — die einzige saubere Art, §13.5 ohne zweiten Maskenstream
zu garantieren; near-Maskenkante leicht konservativere Gewichtung, dokumentiert.
Region-Reads tasten per **Nearest** über dieselbe Partition hoch (trivially
veto-sicher; bilineare Variante mit Veto-Guard ist eine spätere Verfeinerung).

**Zwei getrennte Hashes (§13.4).**

- `compute_source_quality_identity_hash(plan, normalized_cache_hash)` —
  geordnete `frame_id` + `source_index`, Quellmaße, Farbmodus, Bayer-Pattern,
  CFA-Ursprung, Normalized-Cache-Hash. **Schließt aus:** `internal_scale`,
  `output_scale`, Canvas-Geometrie, `plan_hash` und den
  config-gebundenen `RegistrationSamplingPlan.source_identity_hash` selbst.
  Der Runner berechnet letzteren als `sha256(input_manifest ":" config.sha256)`
  — eine reine Neuregistrierung oder Config-Änderung würde ihn ändern; §13.4
  verbietet das ausdrücklich. Test: alle Registrierungs-/Canvas-/Scale-Felder
  mutieren ⇒ Hash **unverändert**; Quellmaß / CFA-Ursprung / Farbmodus /
  Bayer-Pattern / `frame_id` / Normalized-Cache-Hash mutieren ⇒ Hash ändert.
- `compute_scale_quality_config_hash(AqmhPyramidConfig, cache_cfg)` —
  Proxy-Version, `storage_divisor`, dtype, Pyramiden-Parameter
  (`scales`, `base_window_px`, `w_sharp`, `w_snr`, `score_scale`,
  `k_artifact`, `frac_artifact_max`).

**Layout & Commit.** `cache/source_quality_maps/{composite,scale_0..3,artifact}/…bin`
(kleiner ByteSink-Header + row-major LE `uint16`) je atomar
(`core::AtomicOutput`); `metadata.json` wird **atomar zuletzt** geschrieben
und ist der **einzige Commit-Punkt** (Crash davor ⇒ keine nutzbare Cache).
`streams` listet nur committете Streams. `source_quality_cache_hash` hasht
das kanonische Manifest inkl. Datei-`sha256` und ist aus seiner eigenen
Berechnung ausgeschlossen.

**Fail-closed-Reader.** `usable()` nur wenn `metadata.json` parst,
Schema = 1, dtype = `uint16`, erwartete Identity-/Config-Hashes passen, der
deklarierte `source_quality_cache_hash` **neu berechnet** und **jede**
gelistete `.bin`-Prüfsumme stimmt. Sonst gesetzter `error()`-Code
(`SQM_CACHE_IDENTITY_MISMATCH`, `SQM_CACHE_FILE_CORRUPT: …`,
`SQM_CACHE_MANIFEST_HASH_MISMATCH`, …).

**Verifikation (`tests/test_source_quality_map_cache.cpp`, 4 Fälle,
306 Assertions):** Quantisierungs-Invarianten (0/NaN⇒0, kein
Nicht-Null-Code⇒0, Round-Trip ≤ `1/65535`); Identity-Hash ignoriert
Registrierung/Canvas/Scale, verfolgt Quellinhalt; Config-Hash verfolgt
Pyramiden-Parameter + Divisor + dtype; Writer→Reader-Round-Trip auf einem
8×8-Bild mit 2×2-blockkonstanten Werten und einem vetoisierten 2×2-Block:
`read_full` liefert eine blockige Karte, der vetoisierte Speicherzellenbereich
ist **überall NaN**, positive Pixel innerhalb `1,5/65535` des Blockwerts;
`read_region(2,6)` deckt sich zeilenweise mit `read_full`; falscher
Identity-Hash / manipulierte `.bin` / manipulierte `metadata.json` ⇒
`!usable()`.

Build grün; Hauptsuite **425/426** (weiterhin nur
`test_acceleration_backend.cpp:254`; +4 neue Fälle).

**Nächste M5-Schritte:** Orchestrator, der aus dem Analyseproxy je Frame den
Streaming-Sink in diesen Cache schreibt, dann `SOURCE_QUALITY_MAPS` als
angehängte `Phase = 28` **vor** `GLOBAL_QUALITY` im `reconstruct`-Runner,
Region-Reads-Konsum im Forward-Drizzle-Rekonstruktor (`Q_composite` je
Quellsample, §11.7 geometrisch `K`-gemittelt — **kein** Pixel-Veto), M31-Verifikation.

<a id="historie-30-29"></a>

### 30.29 M5 fortgesetzt: `SOURCE_QUALITY_MAPS` als echte Phase 28 im `reconstruct`-Runner + Orchestrator (2026-09-05)

Dritter M5-Schritt: der **Orchestrator** und die **neue Runner-Phase**.

**`build_source_quality_map_cache(cache_root, plan, cache, pyramid, cache_cfg)`**
(`source_quality_map_cache.cpp`). Für jedes `valid` Frame:
`cache.load(source_index)` → `compute_source_quality_proxy_v1(...)` →
`proxy_full` als source-aufgelöste Analyseeingabe →
`compute_source_quality_maps(..., sink)` mit einem Sink, der jede Skala sofort
als `writer.put("scale_"+k, source_index, psi)` in den Cache streamt; danach
`writer.put("composite", …, maps.q_map)` und
`writer.put("artifact", …, maps.artifact_confidence)`. `writer.commit()`
schreibt `metadata.json` als einzigen Commit-Punkt. Rückgabe:
`source_identity_hash`, `source_quality_config_hash`, `source_quality_cache_hash`,
`streams`, `frames`, `computed_scales`. Keine gleichzeitige Vollmap-Residenz
über die Skalen hinaus (Sink, §13.3).

**Neue Phase `SOURCE_QUALITY_MAPS = 28`** (angehängt, keine bestehenden
Integer-Werte verschoben; `phase_to_string`/Test aktualisiert). In
`run_forward_drizzle_stages()` **nach** `COMMON_OVERLAP` und **vor**
`GLOBAL_QUALITY` (§11.9: `GLOBAL_QUALITY` konsumiert denselben Proxy und liegt
zwingend danach). Frischer Lauf: Phase läuft, schreibt die drei
`source_quality_*`-Hashes in den `forward_drizzle_checkpoint.json`, meldet
`frames`/`computed_scales`/`streams`/`source_quality_cache_hash` im
`phase_end`. Resume: `SourceQualityMapCacheReader` gegen die
Checkpoint-Hashes; `!usable()` oder geänderter `source_quality_cache_hash` ⇒
`FORWARD_STAGE_SOURCE_QUALITY_CACHE_UNUSABLE` / `_CHANGED` (fail-closed).

**Verifikation:**

- **Synthetischer End-to-End-Integrationstest** (`[forward-runner]`-Fixture,
  `tests/test_runner_forward_drizzle.cpp`): `run_forward_drizzle_stages()`
  läuft die vollständige Phasenfolge
  `NORMALIZED_CACHE → SAMPLING_GEOMETRY → COMMON_OVERLAP → SOURCE_QUALITY_MAPS
  → GLOBAL_QUALITY → FORWARD_DRIZZLE`, jede `phase_end` `ok`, der Orchestrator
  produziert real den Cache; `phase_to_int(SOURCE_QUALITY_MAPS) == 28` und
  `FORWARD_DRIZZLE == 27` (Append-Invariante).
- **Orchestrator-Unit-Test** (`tests/test_source_quality_map_cache.cpp`):
  echte `VerifiedNormalizedSourceCache` aus on-disk `.raw`-Frames (MONO,
  96×72), `build_source_quality_map_cache()` → `frames == 2`,
  `computed_scales ≥ 2`, `streams ⊇ {composite, artifact, scale_0}`; ein
  `SourceQualityMapCacheReader` mit den zurückgegebenen Hashes ist `usable()`,
  `source_quality_cache_hash` deckt sich, `read_full("composite", f)` je Frame
  ist `h×w` mit finiten Werten in `(0,1]`.

Build (Lib + Runner + Tests) grün; Hauptsuite **426/427** (weiterhin nur
`test_acceleration_backend.cpp:254`; +1 Orchestrator-Fall, `[forward-runner]`
um die neue Phase erweitert).

**Verbleibend für M5:** Konsum der Region-Reads im Forward-Drizzle-Rekonstruktor
als geometrisch `K`-gemitteltes `Q_composite_f,c(q)` je akzeptiertem
Frame-Beitrag (§11.7 — **kein** Pixel-Veto; ein einzelnes Null-Q-Sample zahlt
null Gewicht, vetoisiert aber nicht das Ausgabepixel), Verdrahtung von
`w_raw = B · G_eff · Q_composite`, und M31-End-to-End-Verifikation der
tatsächlichen Q-Map-Wirkung auf Raw≠Uniform.

<a id="historie-30-30"></a>

### 30.30 M5 Review-Korrekturen: Wert-Stream + expliziter Hard-Veto-Stream, keine stille Null-Map, Artefakt-Nicht-Anwendbarkeit (2026-09-05)

Review von 30.27–30.29 fand drei Punkte, die vor dem `Q_composite`-Konsum
geschlossen sein müssen (sonst treibt „halbe Pixel fehlen" ein
plausibel aussehendes Raw≠Uniform):

1. **Konservative Downtastung dezimierte den Cache.** Die 30.28-Regel „jede
   Veto-Quelle ⇒ Zelle Veto" tötet bei realen, **nicht** partitionsscharfen
   psi-Karten (breite NaN-Bänder der groben Skalen nach Upsampling) jede
   straddelnde 2×2-Zelle. **Fix:** getrennte Streams im `.bin` (Schema 2):
   - **Wert-Zelle** = Valid-Mean über die **strikt positiven** überdeckten
     Quellpixel ⇒ gute Daten am NaN-Rand überleben;
   - **Hard-Veto-Zelle** (`uint8`) = 1, wenn **irgendein** überdecktes
     Quellpixel ein exaktes `Q=0` (finit, `≤ 0`) ist. Der Lesepfad erzwingt
     dort `NaN` unabhängig vom Wert — ein exaktes Null-Veto wird nie positiv
     (§13.5), ohne gute Nachbardaten zu verwerfen. NaN-Quellpixel (kein
     Support, kein Hard-Veto) zählen weder als Wert noch als Veto.
   Neuer Test: gemischte Zelle mit einem exakten `0.0` **und** positiven
   Samples ⇒ Lesen liefert `NaN`; Zelle mit einem `NaN` **und** positiven
   Samples ⇒ Lesen liefert den positiven Mittelwert. Orchestrator-Test prüft
   jetzt zusätzlich: cached Composite behält **> 90 %** der quellgültigen
   Pixel (keine partitionsscharfe Dezimierung).
2. **`compute_source_quality_maps` gab bei Geometrie-Fehlpaarung still eine
   All-Null-`q_map` zurück** — ein Veto für **jeden** Pixel. Jetzt
   `throw std::invalid_argument("SOURCE_QUALITY_MAPS_GEOMETRY_MISMATCH")`.
3. **`artifact_confidence` trug die Legacy-`phi_artifact`-Semantik wörtlich:
   `1.0f` (volles Vertrauen) bei `< 3` gültigen Highpass-Samples** — genau
   das Verhalten, das §14.4 benennt und **ablehnt** („nicht wie der alte
   Diagnosepfad mit `1`, sondern nichtanwendbar"). Fix: die feinste
   Artefaktkarte wird **vor** der Hochtastung dort auf `NaN` maskiert, wo das
   feinste `psi` `NaN` ist (deckt sich mit „unzureichender Support"), sodass
   nur echt saubere, gut gestützte Pixel ihren Wert behalten. Die
   `uint16`-Quantisierung bildet ohnehin sowohl Artefakt-`0` („voll
   artefaktbehaftet") als auch `NaN` auf das Veto-Sentinel ab; §14.4 behandelt
   beide korrekt gleich („nichtfinit, fehlend oder unzureichend gestützt ⇒
   kein volles Vertrauen").

Build grün; Hauptsuite **427/428** (weiterhin nur
`test_acceleration_backend.cpp:254`; +1 Hard-Veto-Fall).

<a id="historie-30-31"></a>

### 30.31 M5 abgeschlossen (Code): `Q_composite` je Quellsample im Rekonstruktor verdrahtet (2026-09-05)

Letzter M5-Codeschritt: `w_raw = B · G_eff · Q_composite_f,c(q)` (§11.9) mit dem
frame-lokalen `Q_composite_f,c(q)` aus §11.7.

**`forward_drizzle`:**

- neuer `SourceQualityProvider = function<const Matrix2Df&(size_t source_index)>`
  — liefert je Frame die **composite Source-Q-Map in Source-Geometrie**
  (M5-Cache, `read_full("composite", …)`), Wert in `(0,1]` bzw. `NaN`/`≤0` bei
  Hard-Veto/kein-Daten;
- `ClipCandidate` bekommt `double q = 1.0` — das **geometrisch `K`-gemittelte**
  `Q_composite_f,c(q) = Σ_s K(q,s)·Q_composite_f(s) / B_f,c(q)`. Ein `NaN`/`≤0`
  Quellsample zahlt **0** in diesen Mittelwert (§11.9: fehlende Q-Map ist kein
  ungewichteter Fallback; `Q=0` ist ein expliziter Sample-Veto).
  `apply_robust_clipping` liest `.q` **nicht** — die Clippingmaske bleibt rein
  geometrisch (§11.8);
- pro Streifen ein dritter `double`-Akkumulator `QA[c][i]` parallel zu `A`/`B`,
  gefüllt in derselben `isfinite(v)`-Bedingung wie `B`; Speicherbudget um den
  `QA`-Akkumulator und einen zweiten quellgroßen Float-Puffer (die Q-Map)
  erweitert;
- Raw-Gewicht: `wr = cand.b · G_eff(f) · cand.q`. **Kein Pixel-Veto:** ein
  `Q_composite_f,c = 0` entfernt nur diesen Frame aus Raw an diesem Pixel;
  Uniform, andere Frames und der Pixel-Support bleiben unberührt (§11.7).
- Provider `null` ⇒ `cand.q = 1.0` überall ⇒ Raw wie zuvor (rückwärtskompatibel).

**Weiterleitung:** `stream_forward_drizzle_uniform_and_raw[_2x2]` und
`compute_forward_drizzle_uniform_and_raw` bekommen den optionalen
`quality_of`-Parameter; `persist_forward_drizzle_uniform_and_raw` und
`persist_forward_drizzle_from_predecessors` einen optionalen
`source_quality_cache_root` — letzterer öffnet den `SourceQualityMapCacheReader`
fail-closed gegen `compute_source_quality_identity_hash(sampling, normalized_cache_hash)`
und baut einen 1-Eintrag-memoisierten `quality_of` (`read_full("composite", idx)`).
`DrizzleStorePredecessors::source_quality_cache_hash` geht in den
`reconstruction_hash` **nur** ein, wenn Q-Maps konsumiert wurden (ältere Stores
bleiben vergleichbar). `runner_forward_drizzle` übergibt `dir/cache/source_quality_maps`.

**Verifikation — synthetische Unit-Tests mit von Hand nachgerechneten Werten
(`tests/test_forward_drizzle.cpp`, neuer Fall, 4 Sektionen):** zwei
identisch registrierte MONO-Frames (Werte 10/20, gleicher geometrischer
Support, kein Clipping):

- **kein Provider** ⇒ Raw == Uniform == 15 (Klarmittel);
- **`Q` = 1,0 vs 0,25** ⇒ Uniform bleibt 15, Raw = `(10·1 + 20·0,25)/1,25 = 12`
  (zieht zum hoch-Q-Frame);
- **`Q` = 0 für Frame 1** ⇒ Uniform 15, Raw = 10 (nur Frame 0), Raw-Pixel
  bleibt **belegt** (`support == 1`), `pixel_channel_rejected == 0` — kein
  Pixel-Veto;
- **`Q` = NaN für Frame 1** ⇒ Raw = 10 (NaN wirkt wie 0).

`[forward-runner]`-Fixture (30.29) exerziert die vollständige Kette
`SOURCE_QUALITY_MAPS`-Cache → Reader → `quality_of` → `K`-gemitteltes
`Q_composite` in das Raw-Gewicht synthetisch end-to-end mit.

Build (alle Targets) grün; Hauptsuite **428/429** (weiterhin nur
`test_acceleration_backend.cpp:254`; +1 Fall).

**Einziger verbleibender M5-Punkt:** M31-End-to-End-Verifikation der
tatsächlichen Q-Map-Wirkung — `reconstruct` auf echten M31-Frames laufen
lassen und `mean|u−r|` **plus** die Finit-Pixel-Fraktion des gecachten
Composite berichten (die beiden zusammen trennen „Qualitätsgewichtung wirkt"
von „halbe Pixel fallen aus"). Erfordert einen frischen Registrierungslauf
(kein resumebarer Lauf-Baum vorhanden).

<a id="historie-30-32"></a>

### 30.32 M5-Kontrolle vor M6: die gecachte Composite-Q-Verteilung ist nicht entartet (2026-09-05)

Vor dem Aufsetzen von M6 auf `Q_composite` / `pow(Q_scale*, exp)` die
diskriminierende Frage geprüft: Falls die realen `psi`-Werte
(`sigmoid(score_scale·(w_sharp·z_sharp + w_snr·z_snr))·artifact`) in einem zu
engen Band liegen, wäre `Q_composite` ein nahezu konstanter Multiplikator, der
sich in `sum_wx_r/sum_w_r` weghebt — Raw sähe korrekt gewichtet aus, trüge
aber praktisch keine ortsaufgelöste Qualitätsinformation, und
`pow(Q_scale0, 4)` für das Fine-Profil wäre potenziertes Rauschen.

Messung über den bestehenden Orchestrator-Test (96×72 MONO, echter
`VerifiedNormalizedSourceCache` aus on-disk-`.raw`-Frames →
`compute_source_quality_proxy_v1` → `compute_source_quality_maps` → Cache →
`read_full("composite")`):

| | p05 | p50 | p95 | p95/p05 | p95−p05 |
|---|---|---|---|---|---|
| Frame 0 | 0,248 | 0,476 | 0,852 | 3,44 | 0,60 |
| Frame 1 | 0,229 | 0,473 | 0,823 | 3,60 | 0,59 |

Die Verteilung spannt einen nutzbaren Bereich (Test-Assertions
`p95/p05 > 1,5` und `p95−p05 > 0,10`). Über die Fine-Bandbreite bedeutet das
ein Gewichtsverhältnis `0,85^4 / 0,25^4 ≈ 130` — genau die aggressive
Feindetailselektivität, die der Entwurf will. **M6 kann darauf aufsetzen.**
Der reale M31-Lauf bleibt die finale Bestätigung, ist aber nicht mehr
blockierend.

<a id="historie-30-33"></a>

### 30.33 M6 begonnen: maskierte supportpropagierende À-trous-Zerlegung (§14.2) (2026-09-05)

Erster M6-Schritt nach der Reihenfolge des Plans und dem Advisor-Rat: die
**À-trous-Zerlegung als eigenständige reine Funktion** — der einzige
M6-Baustein mit vollständig von Hand nachrechenbaren Abnahmekriterien; Blend
und Energieguard sind ohne ihn bedeutungslos. Kein Legacy-Code zum Spiegeln
(der `b3_spline_blur` aus dem Proxy ist der undilatierte Level-0-Blur mit
Clamp-Rand — hier gilt maskierte Renormierung ohne Clamp).

**`reconstruction/atrous_decomposition.{hpp,cpp}`:**

- `atrous_decompose(value, mask, w, h, levels∈[1,4]) -> AtrousDecomposition`
  mit `bands[j].detail` (`D_j`, NaN off `M_(j-1)&&M_j`), `bands[j].support`,
  `coarse` (`C_levels`), `coarse_support` (`M_levels`);
- shift-invariante À-trous mit separierbarem `h = [1,4,6,4,1]/16`; Dilatation
  = `2^(j-1)-1` Nullen zwischen den Taps ⇒ **Level 1 undilatiert**, Level 2
  ein Nullloch, Level 3 drei — direkt gegen die Off-by-one geprüft;
- maskierte 2D-Faltung als zwei separierbare Pässe **desselben** Kerns
  (Linearität ⇒ separierbare Anwendung == die 2D-Faltung des Plans) auf
  `M_(j-1)` und `C_(j-1)·M_(j-1)`, dann eine Division: `den_j = conv(M)`,
  `C_j = conv(C·M)/den_j`;
- `M_j = M_(j-1) && (den_j >= den_min)`; **`den_min = kAtrousDenMinFraction = 0.5`**
  ist eine feste, versionierte Konstante in der **Multiband-Hash-Domäne** —
  bewusst **kein** Config-Feld (§14.2: nicht ohne `multiband_config_hash`-Bump
  tunbar). Der voll gestützte separierbare Kern summiert in jeder Achse zu 1,
  also ist das voll gestützte 2D-Gewicht für **jedes** Level exakt 1 und
  `den_min` levelunabhängig;
- `D_j = C_(j-1) - C_j`, gültig **nur** auf `M_(j-1) && M_j` (nicht nur `M_j`
  — sonst wird der Detailsupport um eine Level-Erosion zu weit und speist
  falsche `alpha_j=0`);
- Randbehandlung: außerhalb liegende Taps tragen nichts bei (wie Maske 0),
  nie ein geklemmter Randwert (§14.7);
- `atrous_reconstruction_max_error()` prüft die Identität nur auf dem
  engsten gemeinsamen Support (`M_levels`).

**Verifikation — synthetische Unit-Tests mit von Hand nachgerechneten Werten
(`tests/test_atrous_decomposition.cpp`, 7 Fälle, ~37 k Assertions):**

1. **Konstantbild** ⇒ `|D_j| < 1e-5` auf dem Support, `C_L == 7,25` (DC-Gain
   1); Randpixel fallen legitim unter `den_min` (out-of-image-Taps tragen
   nichts) und fallen weg — die tiefe Innenregion bleibt voll gestützt.
2. **Rekonstruktionsidentität** `C_L + Σ_j D_j == input` (`max err < 2e-3`)
   für Level 1, 2, 3, 4.
3. **Level-1-Spike-Antwort** = exakter separierbarer Kern:
   `D_1(0,0) = 1 − (6/16)² = 220/256`, `D_1(±1,0) = −24/256`,
   `D_1(±1,±1) = −16/256`, `D_1(±2,0) = −6/256`, `D_1(±3,0) = 0`.
4. **Level-2-Dilatation** reicht weiter als Level 1: `C_2` nichtnull bei
   Offset 5 und 6, null bei 7.
5. **Support schrumpft monoton** mit einem Maskenloch
   (`M_1 ≥ M_2 ≥ M_3`, `coarse_support == M_3`, `M_3 < M_0`); wo ein Band-
   Support 0 ist, ist `D_j` dort `NaN` (Vertrag `M_(j-1) && M_j`).
6. `levels ∉ [1,4]` wird abgelehnt.

Build (alle Targets) grün; Hauptsuite **435/436** (weiterhin nur
`test_acceleration_backend.cpp:254`; +7 neue Fälle).

**Nächste M6-Schritte:** Fine/Medium-Profile im Drizzle (baut auf 30.31),
dann Bandzuordnung + Blend (`D_out,j = D_R,j + alpha_j·(D_profile,j − D_R,j)`,
`X_out = C_U,L + Σ D_out,j`; Identität `alpha≡0 ⇒ X_out = R − C_R,L + C_U,L`),
dann adaptives Alpha (`A_neff`/`A_coverage`/`A_separation`/`A_artifact`/
`A_registration`), Energieguard (MAD-Fenster ≤ 1,30, 6-Schritt-Bisektion),
B3-Alpha-Glättung mit `min`-Kappe, zuletzt Streifenpfad gegen den
In-Memory-Referenzpfad (§14.7) und Dreiwegvalidation (§15).

<a id="historie-30-34"></a>

### 30.34 M6 fortgesetzt: Fine/Medium-Profile im Drizzle (§11.9/§14.1) (2026-09-05)

Zweiter M6-Schritt: `w_fine = B·G_eff·pow(Q_scale0, fine_quality_exponent)`,
`w_medium = B·G_eff·pow(Q_scale1, medium_quality_exponent)`, **geteilte**
Clippingmaske mit Uniform/Raw (§11.8). Baut direkt auf der `Q_composite`-
Verdrahtung (30.31) auf.

- `SourceQualityProvider` → **`FrameQualityProvider`** (`FrameQualityMaps`
  mit drei optionalen Zeigern `composite`/`scale0`/`scale1`); `ClipCandidate`
  bekommt `q0`/`q1` neben `q` (alle von `apply_robust_clipping` ignoriert);
  `ForwardDrizzleUniformAndRawResult` bekommt `fine`/`medium`
  (`ForwardDrizzleUniformResult`, leer wenn nicht angefordert);
  `MultibandProfileParams{emit_fine, emit_medium, fine_quality_exponent=4,
  medium_quality_exponent=2}`.
- Ein dritter Streifenakkumulator je Q-Stream (`QA`/`QA0`/`QA1`), gefüllt in
  derselben `isfinite(v)`-Bedingung wie `B`; `pow`-Gewichtung erst nach dem
  Clipping. `need_qc` per Pre-Scan (nur Null-Check der Zeiger): Raw wendet
  `Q_composite` nur an, wenn der Provider für ≥ 1 Frame eine Composite-Map
  liefert — ein Aufrufer, der nur `scale0`/`scale1` liefert, bekommt
  Fine/Medium gewichtet und Raw = `B·G_eff`.
- `emit_fine`/`emit_medium` **ohne** Provider ⇒
  `DRIZZLE_MULTIBAND_REQUIRES_QUALITY_PROVIDER`. Der `2/1`-Streaming-Pfad
  (`stream_forward_drizzle_uniform_and_raw_2x2`) lehnt `emit_fine`/
  `emit_medium` vorerst ab (`DRIZZLE_2X2_MULTIBAND_STREAMING_UNSUPPORTED`,
  spätere Batch). `persist_forward_drizzle_from_predecessors` konsumiert
  weiterhin nur den `composite`-Stream (F/M-Store-Verdrahtung später).
- Speicherbudget: bis zu 5 Doubles/Pixel/Kanal (`A`/`B`/`QA`/`QA0`/`QA1`),
  bis zu 3 quellgroße Q-Map-Puffer, bis zu 4 Ausgabe-Profilebenen.

**Verifikation (`tests/test_forward_drizzle.cpp`, Fine/Medium-Sektionen):**
zwei identische MONO-Frames (10/20), `scale0` 1,0/0,5, `scale1` 1,0/0,25 —
Uniform/Raw bleiben 15; **Fine = (10·1 + 20·0,25)/1,25 = 12** bei Exponent 2;
Medium = 12 bei Exponent 1; Default-Exponent 4 selektiver:
`Fine = (10 + 20·0,0625)/1,0625`; `medium.L.value` leer wenn nicht
angefordert; `emit_fine` ohne Provider wirft. Refaktor verhaltensneutral:
Hauptsuite **435/436** (weiterhin nur `test_acceleration_backend.cpp:254`).

**Nächste M6-Schritte:** Mehrband-Fusionsmodul (In-Memory-Referenzpfad
§14.7): À-trous je Profil/Kanal → Bandzuordnung §14.3 → Blend
`D_out,j = D_R,j + alpha_j·(D_profile,j − D_R,j)` mit **vorgegebenem** Alpha
(Identitäten `alpha≡0 ⇒ X_out = R − C_R,L + C_U,L`, `U==R==F==M ⇒ X_out == R`);
danach adaptives Alpha (§14.4), Energieguard (§14.5), Alpha-B3-Glättung
(§14.7), Streifenpfad, Dreiwegvalidation (§15).

<a id="historie-30-35"></a>

### 30.35 M6 fortgesetzt: In-Memory-Referenz-Mehrbandpipeline — Fusion, adaptives Alpha, Energieguard, Alpha-Glättung (2026-09-05)

Dritter bis fünfter M6-Schritt in einem Zug, alle als eigenständige,
handgeprüfte reine Funktionen; der **In-Memory-Referenzpfad** (§14.7
ausdrücklich für Tests erlaubt). Streaming/Halo und die frame-daten-
abhängigen Alpha-Faktoren kommen in späteren Batches.

**`reconstruction/multiband_fusion` — `fuse_multiband_channel()` (§14.2/§14.3).**
Ein Kanal: À-trous-Zerlegung von U/R/F/M → Bandzuordnung (`D1←F`, `D2←M`,
`D3..DL←R`, Grobrest `C(L)←U`) → Blend
`D_out,j = D_R,j + alpha_j·(D_profile,j − D_R,j)`,
`X_out = C_U,L + Σ D_out,j`. Ein ungültiges Detailprofil erzwingt lokal
`alpha_j=0`; ein fehlendes Raw-Band (oder `C_R,L`/`C_U,L`) macht den
Mehrbandpixel ungültig (§14.2). Verifiziert (5 Fälle): **`alpha≡0` mit
`U==R` ⇒ `X_out == R`** (Identität `X_out = C_U,L + R − C_R,L`);
**`U==R==F==M` ⇒ `X_out == R` für jedes Alpha** (§14.3); `alpha≡1`
injiziert das Fine-Band-1-Detail (Level 1); Maskenloch in `R` ⇒ Pixel
ungültig, Loch nur in `F` ⇒ `alpha=0` erzwungen, Pixel bleibt gültig ≈ Basis;
Eingabevalidierung.

**`reconstruction/adaptive_alpha` — `compute_adaptive_alpha()` (§14.4, Teil 1).**
Die zwei allein aus den Profilebenen berechenbaren Faktoren:
`A_neff,c = smoothstep(min_effective_samples, full_effective_samples,
n_eff_profile,c)`, `A_coverage,c = clamp(profile_weight_c / uniform_weight_c,
0, 1)`, beide **min über aktive Kanäle** (§14.6); dazu `alpha_cap` und die
extern gelieferten, bereits kanal-reduzierten
`A_separation`/`A_artifact`/`A_registration` (Default 1). Ein Faktor 0 ⇒
`alpha_j = 0`. Ein alpha_j-Map wird **geteilt** über R/G/B; nur Fine-/
Medium-Bänder bekommen ein Map, Raw-Bänder ein leeres (Alpha ignoriert).
Hermite-`alpha_smoothstep` exponiert. Verifiziert (7 Fälle, Handwerte):
smoothstep-Mittelpunkt 0,5 / `t=0,25 ⇒ 0,15625`; `n_eff < min ⇒ 0`,
`n_eff ≥ full & volle Coverage ⇒ alpha_cap`; Coverage `30/100 ⇒ 0,3`;
OSC-Minimum `0,5·0,8`; externe Faktoren multiplizieren, einer 0 ⇒ 0;
unsupported Profilpixel ⇒ 0; Raw-Band ⇒ leeres Map; Eingabevalidierung.

**`reconstruction/alpha_guard` — `apply_energy_guard()` (§14.5) +
`smooth_alpha_b3()` (§14.7).**
- Energieguard je Band auf der Arbeitsluminanz (`0,25R+0,5G+0,25B`, MONO
  `D_L`), `window_radius_j = max(3, 2^(j+1))`,
  `scale_raw = max(MAD_window(D_raw,luma), background_band_floor_j)`,
  `energy_ratio(alpha) = MAD_window(D_mixed,luma(alpha)) / scale_raw ≤ 1,30`.
  Bei Überschreitung mit `alpha_pre`: **deterministische 6-Schritt-
  Bisektion** auf `[0, alpha_pre]` für das größte zulässige `alpha_guarded`
  (`lo` ist wegen `ratio(0) = mad_r/scale_raw ≤ 1` stets zulässig). **Keine**
  Sternkonzentrationsausnahme, **kein** Hartclipping von Pixeln/Raw-Bändern.
  `< 25` gültige Fensterpixel ⇒ `alpha = 0`. `1,4826·MAD` als `mad_sigma`
  exponiert. Verifiziert (6 Fälle): Fensterradius `4/8/16`; `mad_sigma`
  Handwerte; `D_profile == D_R ⇒ Alpha unverändert`; verrauschtes Detail ⇒
  `0 < alpha_guarded < 1` **und** nachgerechnetes `energy_ratio ≤ 1,30`;
  `< 25` Pixel ⇒ 0; `background_floor` setzt die Rauschskala.
- Alpha-Glättung: separierbarer B3 `[1,4,6,4,1]/16` **nur innerhalb der
  eigenen 4-zusammenhängenden Supportkomponente** (Flood-Fill-Labeling),
  `alpha_blur = conv(alpha_guarded·support, B3)/conv(support, B3)`,
  `alpha_final = min(alpha_guarded, alpha_blur)`. Die **`min`-Kappe ist
  verbindlich** (Glättung reduziert nur, hebt nie an). `alpha_guarded == 0`
  bleibt exakt 0. Verifiziert (2 Fälle): exakte Null bleibt Null; die
  `min`-Kappe hält einen 0,2-Nachbarn eines 1,0-Spikes auf 0,2, während der
  Spike selbst durch die Glättung < 1 fällt; **keine Leckage** über getrennte
  Support-Inseln (1-px-Lücke ⇒ getrennte Komponenten).

**`fuse_multiband()` — Orchestrator (In-Memory-Referenz, §14).**
Ganzes Frame: `compute_adaptive_alpha` → je F/M-Band Luma-`D_R`/`D_profile`
bilden → `apply_energy_guard` → `smooth_alpha_b3` → `fuse_multiband_channel`
je Kanal mit dem **geteilten** `alpha_final`. Verifiziert (2 Fälle):
**`U==R==F==M` ⇒ `X_out == R`** durch die volle Pipeline; ein verrauschtes
Fine-Profil ⇒ geguardetes Interior-Alpha `< 0,98` gegen effektiv 1,0 ohne
Guard.

Build (alle Targets) grün; Hauptsuite **457/458** (weiterhin nur
`test_acceleration_backend.cpp:254`; +22 neue Fälle über die fünf Module).

**Verbleibend für M6:** `A_separation`/`A_artifact`/`A_registration` aus
frame-lokalen Streifenstatistiken (gewichtete Quantile der Pro-Frame-Q-Werte,
`artifact_confidence`, Registrierungsfaktoren — erfordert, dass der Drizzle
pro Frame/Pixel Quantil-Eingaben ausgibt); Streifen-À-trous gegen den
Referenzpfad (§14.7); Dreiwegvalidation/Gating (§15); F/M-Persistenz im
transaktionalen Store + Runner-Verdrahtung; M31-End-to-End.

<a id="historie-30-36"></a>

### 30.36 M6 fortgesetzt: `A_separation`/`A_artifact`/`A_registration` als geprüftes Primitiv (§14.4) (2026-09-06)

Sechster M6-Schritt: die drei Alpha-Faktoren, die frame-lokale
Streifenstatistiken brauchen, als **reviewte, getestete reine Funktion** —
dieselbe „Primitiv zuerst, Verdrahtung später"-Strategie wie bei
`apply_robust_clipping`.

**`reconstruction/alpha_confidence`.**

- `weighted_percentile(values, weights, p)` — Hazen-Plotting-Position
  (`CDF_k = (cum_k − w_k/2)/total`), lineare Interpolation zwischen den
  Klammersamples. Handwerte: gleiche Gewichte reduzieren auf den gewöhnlichen
  Perzentil (`p50 = 3`, `p90 = 5` für `[1..5]`); Gewichte `[3,1]` auf
  `[10,20]` ⇒ `p50 = 12,5`.
- `compute_alpha_confidence_channel(accepted, params)` je Kanal/Zielpixel aus
  `AlphaFactorContribution{b, q_composite, artifact_conf, is_direct,
  residual_factor}`:
  - `A_separation = smoothstep(min_quality_separation, full_quality_separation,
    weighted_p90(q) − weighted_p50(q))` (Gewicht `B`);
  - `A_artifact = smoothstep(0,25, 0,75, weighted_p10(artifact_conf))`;
    **< `min_artifact_contributors` (8) endliche `a_f` ⇒ nicht anwendbar ⇒ 0**
    (nicht volles Vertrauen, §14.4); nichtfinite `artifact_conf` zählen nicht
    mit;
  - `A_registration = min(smoothstep(0,50, 0,85, direct_fraction),
    smoothstep(0,55, 0,90, weighted_p20(residual_factor)))` mit
    `direct_fraction = Σ B·is_direct / Σ B`.

**Verifikation (`tests/test_alpha_confidence.cpp`, 9 Fälle, Handwerte):**
leere Menge ⇒ alles 0; `A_separation` = 0 für uniforme Q-Population,
saturiert (> 0,99) für breite Q-Streuung; `A_artifact` braucht ≥ 8 endliche
`a_f`, ein Low-Artefakt-Tail zieht `weighted_p10` und damit `A_artifact`
herunter; nichtfinite `artifact_conf` reduzieren die Zählung; `A_registration`
= `min` beider Gates (all-direct/resid 1 ⇒ > 0,99; direct_fraction 0,5 ⇒ 0;
residual 0,55 ⇒ 0); ein sehr schweres `B` verschiebt die Perzentile.

Build grün; Hauptsuite **466/467** (weiterhin nur
`test_acceleration_backend.cpp:254`; +9 Fälle).

**M6-Algorithmikkern ist damit vollständig** (À-trous, Fine/Medium,
Bandblend + alle Identitäten, `A_neff`/`A_coverage`, Energieguard,
Alpha-Glättung, `A_separation`/`A_artifact`/`A_registration`,
`fuse_multiband`-Orchestrator — 6 Module, 31 Testfälle, jede
Plan-Identität nachgerechnet). **Verbleibend ist reine Integrationsarbeit**
(Audit §4 Workpackages 4–7): die drei Confidence-Faktoren im Drizzle-Streifen
als Pro-Pixel-Maps ausgeben (braucht den `artifact`-Cache-Stream als
`cand.qa` und die Pro-Frame-Registrierungsflags), Streifen-À-trous gegen den
Referenzpfad (§14.7), Dreiwegvalidation/Gating (§15), F/M-Persistenz im
transaktionalen Store + Runner-Verdrahtung zum finalen `X_out`-Bild,
M31-End-to-End.

<a id="historie-30-37"></a>

### 30.37 M6: Alpha-Confidence im Drizzle verdrahtet + End-to-End-Referenzpfad (2026-09-06)

Siebter M6-Schritt: die drei Confidence-Faktoren (30.36) im
Drizzle-Streifen als Pro-Pixel-Maps ausgeben und den **vollständigen
In-Memory-Referenzpfad** schließen.

**Drizzle-Verdrahtung (`stream_/compute_forward_drizzle_uniform_and_raw`).**

- `FrameQualityMaps` bekommt einen vierten optionalen Zeiger `artifact`;
  `MultibandProfileParams` bekommt `emit_alpha_confidence` +
  `AlphaConfidenceParams`; `ClipCandidate` bekommt `qa` (K-Average von
  `artifact_confidence`, wie `q`/`q0`/`q1` von `apply_robust_clipping`
  ignoriert).
- `emit_alpha_confidence` ⇒ Pre-Scan-Pflicht: Composite- **und**
  Artefakt-Stream müssen vorhanden sein (sonst
  `DRIZZLE_ALPHA_CONFIDENCE_REQUIRES_COMPOSITE_MAP` /
  `_ARTIFACT_MAP`). `reg_by_source` einmal aus dem Plan:
  `is_direct = (!model_predicted && model_prediction_factor == 1.0f)` (§11.9;
  Provenienzflag **und** ungefalteter Einheitsfaktor, in 30.38 verschärft),
  `residual = registration_residual_factor`.
- Vierter K-Average-Akkumulator `QAA`. Nach dem Clipping je Kanal
  `AlphaFactorContribution`-Liste aus den akzeptierten Kandidaten →
  `compute_alpha_confidence_channel` → **konservatives Minimum über die
  aktiven Kanäle** in Streifen-Maps `a_separation`/`a_artifact`/
  `a_registration` + `alpha_confidence_support`.
- Speicherbudget auf 6 Doubles/Pixel/Kanal + 4 Q-Map-Puffer + die drei
  Confidence-Maps erweitert.

Verifiziert (`tests/test_forward_drizzle.cpp`): `emit_alpha_confidence`
erzeugt drei `[0,1]`-Maps mit Support am belegten Pixel; **2 Frames < 8 ⇒
`A_artifact == 0`** (nicht anwendbar); `emit_alpha_confidence` ohne
Composite/Artefakt wirft.

**`reconstruct_multiband_reference()` — End-to-End (In-Memory, §14.7).**
Ein Ganzframe-Drizzle mit `emit_fine`/`emit_medium` (aus `levels`) +
`emit_alpha_confidence` → `fuse_multiband(dz.uniform, dz.raw, dz.fine,
dz.medium, …, dz.a_separation, dz.a_artifact, dz.a_registration, …)`. Die
kanal-minimierten Drizzle-Maps passen unverändert auf die externen
Faktor-Parameter von `compute_adaptive_alpha`. Verifiziert: `nf=4` identische
MONO-Frames (40×36), konstante Q-Maps ⇒ **`X_out == R`** (U==R==F==M im Wert)
gegen einen reinen Raw-Drizzle auf dem gemeinsamen Support; `alpha_final`
hat `levels` Einträge.

**Damit ist die M6-Referenzpipeline vollständig und end-to-end getestet:**
À-trous → Fine/Medium → Bandblend → adaptives Alpha
(`A_neff`/`A_coverage`/`A_separation`/`A_artifact`/`A_registration`,
geteiltes RGB-Alpha) → Energieguard → B3-Glättung → `X_out`. 7 Module,
~35 Testfälle, jede Plan-Identität nachgerechnet.

Build (alle Targets) grün; Hauptsuite **467/468** (weiterhin nur
`test_acceleration_backend.cpp:254`).

**Verbleibend für M6 (reine Produktionspfad-Integration, Audit §4 WP 5–7):**
Streifen-À-trous mit Fusionshalo gegen den Referenzpfad (§14.7);
Dreiwegvalidation/Gating (§15, 20/30-Stern, Bootstrap-CI, N/A, Promotion);
F/M-Persistenz im transaktionalen Store + Runner-Verdrahtung, die
`reconstruct` bis zu einem finalen `X_out`-Bild bringt; M31-End-to-End.

---

<a id="historie-30-38"></a>

### 30.38 M6 Review-Korrektur: `A_artifact`-Schwelle zählt echte Artefaktdaten; `is_direct` verschärft; Exponenten entdoppelt (2026-09-06)

Review von 30.37 fand drei Punkte:

1. **`A_artifact`-Nichtanwendbarkeit falsch verdrahtet.** §14.4 fordert
   `A_artifact=0` bei *weniger als acht gültigen Framebeiträgen für die
   robuste Statistik* — „gültig" qualifiziert die **Artefaktdaten**, nicht
   die bloße Frameanwesenheit. Die 30.37-Vereinfachung ließ
   `compute_alpha_confidence_channel` effektiv die akzeptierten
   Kanalbeiträge zählen: `ClipCandidate.qa` ist als K-Average (NaN→0 wie
   jeder Q-Strom) **immer endlich**, also konnte „hier keine Artefaktmap"
   nicht von „`artifact_conf == 0`" unterschieden werden. Fix: neuer
   Akkumulator `QAF` (K-Gewicht der Beiträge mit **endlichem**
   Artefaktsample) parallel zu `QAA`; `ClipCandidate.qa_has_data =
   (QAF > 0)`; die Drizzle-Schleife übergibt `artifact_conf = NaN` statt
   `cd.qa`, wenn `!qa_has_data` — `compute_alpha_confidence_channel`
   schließt Nichtfinite bereits aus `art_v`/`art_w` aus und zählt
   `art_v.size()`. Damit zählt die Schwelle exakt die Beiträge mit echtem
   Artefaktdatum (§14.4: „Nichtfinite, fehlende … nichtanwendbar").
2. **`is_direct` war ein nackter Float-Vergleich** auf
   `model_prediction_factor == 1.0f`. Jetzt `!model_predicted &&
   model_prediction_factor == 1.0f` — Provenienzflag und ungefalteter
   Einheitsfaktor müssen beide zutreffen; robust, falls die §11.9-Ableitung
   je aufhört, für direkte Frames ein literales `1.0` zu liefern.
3. **Fine/Medium-Exponenten doppelt.** `MultibandReconstructionParams`
   trug eigene `fine_/medium_quality_exponent`, während
   `params.multiband` (`config::ReconstructionMultibandConfig`) dieselbe
   Plankonstante hält. Die eigenen Felder entfernt;
   `reconstruct_multiband_reference` liest jetzt
   `params.multiband.fine_/medium_quality_exponent` als einzige Quelle.

Verifiziert (`tests/test_forward_drizzle.cpp`, neuer `TEST_CASE`): 10
identisch registrierte Frames, alle mit Composite-Map, **9 mit** Artefaktmap
(0.95), **1 ohne** — korrekte Regel ⇒ `A_artifact` anwendbar (9 ≥ 8) und
sättigt auf `1`; die frühere frameanwesenheits-zählende Regel hätte den
map-losen Frame als `artifact_conf == 0` gewertet, das gewichtete p10 unter
0.75 gezogen und `A_artifact` deutlich unter `1` gemeldet. Bestehende
Tests (2 Frames < 8 ⇒ `A_artifact == 0`; End-to-End `X_out == R`) unverändert
grün. Hauptsuite **468/469** (weiterhin nur `test_acceleration_backend.cpp:254`).

**Follow-up vor §15:** die §14.4-Regel `A_artifact,c=0` bei `< 8` Beiträgen ist
**pro Kanal** mit `min_c`. Ob 30er-OSC-Stacks das bei Produktionsgeometrie
(`internal_scale 2`, `pixfrac 0.8`, Bayer) auf R/B überhaupt erreichen, ist
Arithmetik, keine Vermutung — vor dem Bau des §15-Gates ein synthetisches
30-Frame-OSC-Histogramm von `res.a_artifact` über `alpha_confidence_support`
erstellen. Kollabiert `a_artifact` dort auf ~0, reduziert sich der
Mehrband-Kandidat auf `R − C_R,L + C_U,L`. Ergebnis und Auflösung: **30.40**
(§15 ist sternbasiert, daher nicht blockiert).

---

<a id="historie-30-39"></a>

### 30.39 M6: streifenweise Mehrbandfusion, bit-identisch zur Vollbildreferenz (§14.7) (2026-09-06)

Achter M6-Schritt (Audit §4 WP 5): `fuse_multiband_streamed()` +
`multiband_fusion_halo_rows()` in `reconstruction/multiband_fusion`. Verarbeitet
den internen Frame in Zeilenstreifen von `chunk_rows` **Kernzeilen**; jeder
Streifen ruft intern `fuse_multiband` auf seinem Kern ± Halo auf und committet
nur die Kernzeilen.

**Halo** (`multiband_fusion_halo_rows(L)`): kumulative vertikale À-trous-Reichweite
über alle Level `2·(2^L − 1)` + breitestes Energieguard-MAD-Fenster
`energy_guard_window_radius(L)` + B3-Glättungsreichweite `2`. Für `L=3`: `14 + 16
+ 2 = 32`. Bewusst konservativ — der Guard läuft nur auf Band 1–2, die strikte
Untergrenze ist `max(atrous_reach(L), radius(2)+atrous_reach(2)+2)`; Überpolstern
kostet nur Arbeit, keine Korrektheit. `den_min=0,5`-Supporterosion ist durch
`atrous_reach` bereits abgedeckt.

**Verdrahtung:** `slice_profiles`/`slice_plane`/`slice_vec` schneiden Zeilen
`[ys,ye)` aus U/R/F/M und den `a_*`-Maps; `background_band_floor` ist bandweise,
nicht räumlich, und geht unverändert durch. Raw-Bänder behalten ein **leeres**
`alpha_final` (nicht null-gefüllt), damit auch der Strukturvergleich passt.
`chunk_rows ≤ 0 || ≥ height` ⇒ ein Streifen (identisch zu `fuse_multiband`).

**Verifiziert** (`tests/test_multiband_fusion.cpp`, 4 neue Fälle):

- MONO `64×176` (`h > 2·halo + chunk` ⇒ echte innenliegende Streifen), U/R/F/M
  mit glattem Feld + Fine-Ripple + Band-2-Energie + mildem Raw-Rauschen + zwei
  maskierten Löchern auf verschiedenen Höhen, nichtkonstante
  `a_separation`/`a_registration`. `chunk ∈ {13, 32, 64}` ⇒ **byte-identisch**
  (gleiches NaN-Muster, `==` auf endlichen Werten) für `X_out`, `support`, jedes
  `alpha_final`-Band und `pixels_supported`.
- **OSC** `40×150`, `levels=2`, per-Kanal-Löcher (R/G/B-Support unterschiedlich):
  deckt den R/G/B-Slicing-Zweig ab; `chunk ∈ {11, 32}` byte-identisch, und
  `pixels_supported` stimmt über beide Zählwege überein (Vollbild summiert
  `fuse_multiband_channel`s Pro-Kanal-Zähler, Streifen zählt Supports direkt).
- **Adversarieller Hufeisen-Support**: schmale vertikale Kerbe von oben bis zu
  einer Biegung `halo` Zeilen tiefer als die oberen Arme — ein Streifen über den
  Armen kann nicht sehen, dass beide Arme eine 4-verbundene Komponente sind (der
  B3-Flood-Fill ist der eine nichtlokale Schritt). Ergebnis: `diverged = 0` —
  empirisch reicht der Halo auch hier; der Test dokumentiert die theoretische
  Schranke (falls je > 0: `< 2 %` der Pixel, `max_abs < 0.05` bei Amplitude ~10,
  ≤ `halo` von der Kerbe entfernt).
- `chunk ≥ height` bit-identisch zu `fuse_multiband`.

Build grün; Hauptsuite **473/474** (weiterhin nur `test_acceleration_backend.cpp:254`;
inkl. 30.40-CHARACTERISATION).

**Verbleibend für M6:** Dreiwegvalidation/Gating §15 (sternbasiert, nicht
blockiert — 30.40); F/M-Persistenz im transaktionalen Store +
Runner-Verdrahtung zum finalen `X_out`; M31-End-to-End.

---

<a id="historie-30-40"></a>

### 30.40 M6-Befund + Auflösung: die `A_artifact`-`< 8`-Regel unterdrückt Fine/Medium-Alpha auf einem Teil des OSC-Innenbereichs; §15 ist sternbasiert und damit nicht blockiert (2026-09-06)

Das in 30.38 verlangte Vorabhistogramm ist gemessen
(`tests/test_forward_drizzle.cpp`, CHARACTERISATION-Fall). Synthetischer
30-Frame-OSC-Stack bei **Produktionsgeometrie** (`internal_scale 2`,
`pixfrac 0.8`, RGGB, deterministischer Subpixel-Dither auf 6×6-Gitter,
Composite-Q pro Frame `[0.35, 0.9]`, Artefaktmap sauber `0.9`), kein Clipping:

| Frames | Innen-Pixel mit `A_artifact > 0` |
|---|---|
| 30 | **≈ 60 %** |
| 60 | ≈ 84 % |

Der Grund ist die §14.4-Regel wörtlich: `A_artifact,c = 0` bei weniger als
acht gültigen Framebeiträgen **pro Kanal**, dann `A_artifact = min_c`. Auf
Bayer-OSC ist die R/B-Abdeckung ~¼ der Green-Dichte; bei
`internal_scale 2` deckt ein Droplet ~`pixfrac²·scale²` Zielpixel. An ~40 %
der Innenpixel (bei 30 Frames) bleiben R oder B unter acht akzeptierten
Beiträgen ⇒ `A_artifact = 0` ⇒ **`alpha_j = 0` für die Fine/Medium-Bänder
dort** ⇒ der Mehrband-Kandidat ist dort exakt `R − C_R,L + C_U,L`.

**Auflösung (nach Lesen von §15).** §15 ist **sternbasiert**, nicht
pixelbasiert: `prepare_validation_samples` detektiert einen festen Sternsatz
**einmal** auf dem Uniform-Control; Uniform/Raw/Multiband werden an exakt
diesen Positionen gemessen, die gebootstrappte 95-%-CI liegt auf dem
FWHM-**Median über ≥ 20 gematchte Sterne** (§15.2/§15.3.5). Validationsterne
sind hell, kompakt, hoch-SNR — genau die Pixel mit voller Mehrkanalabdeckung,
an denen `A_artifact` anwendbar ist. Der 30.40-Kollaps liegt im
abdeckungsarmen Hintergrund, wo keine Validationsterne sitzen. Damit ist §15
**nicht blockiert**.

Die `min_c`-Lockerung von `A_artifact` (§14.4) wäre ohnehin ein
**Planamendment** — die Alpha-Confidence-Konstanten liegen im
`multiband_config_hash` (§16.4: „…Alpha-, Energie-, Support-, …,
Validationvertrag"), ein Bump wäre nötig — und wird hier nicht gemacht.

Verbleibende §15-Implementierungsauflage (kein Amendment, deckt sich mit
§15.3.6 „kleine Stichproben nicht als impliziter Pass"): ein Validationstern,
dessen lokale `alpha_final` über **alle** Bänder `≡ 0` ist, misst „Multiband"
identisch zu Raw und darf **nicht** als positive Multiband-Evidenz zählen —
in `prepare_validation_samples`/der Sternmetrik als `multiband_effective`-Flag
je Stern führen.

Bis dahin ist der CHARACTERISATION-Test die Regressionsschwelle
(`0.45 < live_frac < 0.80` bei nf=30).

---

<a id="historie-30-41"></a>

### 30.41 M6: Mehrband-Profilstore + Runner-`MULTIBAND`-Phase → finales `X_out` (Audit §4 WP 6–7) (2026-09-06)

Neunter M6-Schritt: die durable Persistenz und die Runner-Verdrahtung bis zum
Bild.

**Store (`drizzle_profile_store`).** Neuer Modus
`uniform_raw_multiband_clipped`; `DrizzleStoreIdentity.multiband_levels`
(0 ⇒ Nicht-Mehrband, byte-identische Vor-M6-Identität) trägt die Bandzahl, aus
der `plane_names()` den Ebenensatz reproduziert: `uniform`/`raw`/`fine`
(+ `medium` bei `levels≥2`) je `value`/`weight_sum`/`n_eff`/`support` plus vier
Einzelfeld-Pseudoebenen `alpha_{separation,artifact,registration,support}_X_value`
für die kanal-minimierten Confidence-Maps. `multiband_config_hash`-Inhalt
(§16.4) **additiv** unter `algorithm["multiband"]` nur im Mehrbandstore:
Levels, F/M-Exponenten, `kAtrousDenMinFraction`/`kAtrousDecompositionVersion`
(versionierte Hash-Domänen-Konstanten), `AdaptiveAlphaParams`,
`EnergyGuardParams`, `AlphaConfidenceParams`. `persist_forward_drizzle_multiband`
streamt `uniform+raw+fine+(medium)` + die vier Maps über
`multiband_stripe()`; `DrizzleStoreResult.identity` liefert **die tatsächlich
geschriebene Identität** (kein zweites `make_drizzle_store_identity` beim
Zurücklesen). `read_drizzle_profile_region` liest die Pseudoebenen über den
unveränderten `base+field`-Pfad (`channel="X"`, nur `value`).

**Orchestrierung (`source_quality_artifact`).**
`persist_multiband_store_from_predecessors` baut den Store aus den
M5-Prädekessoren — der Q-Map-Cache liefert `composite` + `scale_0`/`scale_1` +
`artifact`; fehlende feinere Streams (kleines Bild, weniger Pyramidenskalen)
sind ein Nullzeiger (Gewicht degradiert), kein Hard-Fail.
`fuse_multiband_store_to_image` liest den Store **streifenweise** zurück
(Kern ± `multiband_fusion_halo_rows`, je Streifen `fuse_multiband`) und
akkumuliert nur das finale Bild (1 Ebene MONO / 3 OSC) → MONO-Float- bzw.
OSC-RGB-FITS. Peak-Eingabespeicher ist `O(chunk + 2·halo)` Zeilen,
**unabhängig von der vollen Framegröße**. Die Generation wird **einmal**
verifiziert (`verify_drizzle_profile_store` → `generation_dir`), danach lesen
alle Streifen über `read_drizzle_profile_region_preverified` **ohne
Neu-Hashing** — sonst wäre die I/O-Last quadratisch in der Framegröße
(`O(H/chunk)` volle Store-Rehashes). Damit skaliert der Pfad in Speicher
**und** I/O auf große Mosaike (M31/M42 in Vollauflösung).

**`2/1`-Produktionsgeometrie (Plan 12.1).** `Downsample2x2Adapter` +
nicht-streamendes `downsample_uniform_and_raw_2x2` erweitert: Fine/Medium über
dieselbe 2×2-Flächenmittelung wie Uniform/Raw; die kanal-minimierten
Confidence-Maps über **2×2-`min` + `AND`-Support** — `2x2-mean` würde nicht mit
dem bereits kodierten Kanalminimum kommutieren, `min` erhält die im Plan
durchgängige „Confidence wird nie angehoben"-Richtung (dieselbe `min`-Regel wie
`n_eff` in derselben Funktion). Die `emit_fine`/`emit_medium`- und
`DRIZZLE_STORE_MULTIBAND_2_1_UNSUPPORTED`-Sperren entfallen;
`persist_forward_drizzle_multiband` leitet `2/1` durch
`stream_forward_drizzle_uniform_and_raw_2x2`.

**Runner.** Neue Phase `Phase::MULTIBAND = 29` (angehängt). Bei
`reconstruction.multiband.enabled` (Default an) baut `FORWARD_DRIZZLE` direkt
den Mehrbandstore (halbierte Geometrie bei `2/1`), danach fusioniert
`MULTIBAND` zu `artifacts/reconstruction_multiband.fits`; `run_end` meldet
`final_image_ready` / `final_image_available: true`. `fuse_multiband_store_to_image`
prüft `identity.multiband_levels == cfg.levels` (Struktur-Guard; Alpha-/Guard-
Kanten werden aus der Aufrufer-Config vertraut, da im Runner Schreiben und
Fusion dieselbe Config teilen).

**Verifiziert.** `tests/test_drizzle_profile_store.cpp`: (1) Mehrbandstore
`1/1` round-trip + `fuse_multiband_streamed` auf den zurückgelesenen Ebenen
**bit-identisch** zur In-Memory-Referenz; Nicht-Mehrband-Erwartung und falsche
Bandzahl validieren den Store nicht. (2) **Mehrband `2/1`**: gespeicherte
Uniform/Raw/Fine/Medium- **und** `alpha_{separation,artifact,registration}`-
Ebenen bit-identisch zu `downsample_uniform_and_raw_2x2(compute_…mb…)` (die
Referenz-Confidence-Maps werden zuerst als nicht-leer geprüft, sonst wäre der
Vergleich vakuös); die fünf Detailebenen sind über `chunk_rows ∈ {2,4,16}`
`sha256`-identisch. Bestehende Nicht-Mehrband-Store-Tests unverändert
(Hash-Stabilität). (3) **OSC** (`plan_for(osc=true)`, 8 Frames, HDR-Feld:
heller kompakter Kern über schwachem Gradient — das M42-Regime, nicht M31s
glatte ausgedehnte Struktur): `persist_forward_drizzle_multiband` +
`fuse_multiband_store_to_image` schreiben ein RGB-`X_out`, das pro Kanal
bit-identisch zur In-Memory-Referenz (`compute_… + fuse_multiband`, OSC) ist,
und über `chunk_rows ∈ {2, 7, H}` `sha256`-identisch (Streifen-/Seam-Logik des
Store-Pfads). `[forward-runner]`: Phasenfolge endet auf `MULTIBAND`,
`reconstruction_multiband.fits` lesbar (32×32, endliche Innenpixel).
**Hinweis:** die Runner-Fixture hat 2 konstante Frames + all-NaN-Artefaktstream
⇒ `A_artifact≡0` ⇒ `alpha≡0` ⇒ `X_out = R − C_R,L + C_U,L`; der Runner-Test
prüft die **Verdrahtung**, nicht die Fusionsmathematik (die deckt der
bit-exakte Store-Test eine Ebene tiefer ab). Hauptsuite **476/477** (weiterhin
nur `test_acceleration_backend.cpp:254`).

**Objekt-Generalität.** Der gesamte M6-Pfad enthält **keine objekt- oder
datensatzspezifische Abstimmung**: alle Schwellen (`den_min`, `energy_limit`,
`min_artifact_contributors`, Smoothstep-Kanten) sind planversionierte
Konstanten in der `multiband_config_hash`-Domäne, keine pro-Objekt-Parameter.
Die Regimeunterschiede zwischen z. B. M31 (glatte ausgedehnte
Low-Surface-Brightness-Struktur) und M42 (hoher Dynamikumfang, gesättigte
Trapez-Sterne) wirken nur über die **Eingabedaten** auf die planmäßig
konservativen Faktoren: der Energieguard (§14.5, **keine**
Sternkonzentrationsausnahme) drückt Alpha nahe hellen Kernen, `A_registration`
drückt Alpha bei vielen modellierten Frames, `A_artifact` bei < 8 gültigen
Kanalbeiträgen (30.40). Das ist plankonformes Verhalten, keine
objektspezifische Bruchstelle; der Kandidat degradiert dort sauber auf
`R − C_R,L + C_U,L`. Test­abdeckung deckt MONO **und** OSC, `1/1`/`2/1`/`2/2`,
kleines Bild bis speicher­begrenzten Streifenlauf ab. Verbleibende empirische
Prüfung: **eine reale Registrierungs­runde** (M31 klärt gleichzeitig den
offenen M5-Echtdatenpunkt; M42/OSC als zweites Objekt für das HDR-Regime).

**Damit ist die Runner-Verdrahtung bis zum finalen `X_out` für alle
Output-Scales (`1/1`, `2/1`, `2/2`) und beide Farbmodi (MONO/OSC) vollständig
und synthetisch objektunabhängig verifiziert.** Offen für M6: nur noch die
reale Registrierungs­runde (M31 + M42/OSC).

<a id="historie-30-42"></a>

### 30.42 M6: Dreiwegvalidation §15 als geprüftes Auswahlmodul (`multiband_validation`) (2026-09-06)

Zehnter M6-Schritt: der §15-Auswahlvertrag als eigenständiges, getestetes
Modul (`include`/`src/reconstruction/multiband_validation`). Dieses Modul
besitzt nur den **Auswahlvertrag** plus die Pro-Stern-FWHM-Statistik, die §15
neu einführt; das Schreiben von `selected_candidate` (§16.3) übernimmt die
Runner-`MULTIBAND`-Phase (**30.43**).

**Drei feste Kandidaten, eine feste Sternpopulation.** `drizzle_uniform`
(sichere Kontrolle), `drizzle_raw` (`B·G_eff·Q_composite`, **nie**
nachbearbeitet, §15.1), `drizzle_multiband` (§14-Fusion). Sterne werden
**einmal** auf `drizzle_uniform` detektiert (`prepare_validation_samples`
über `prepare_aqmh_validation_reference`, §15.2); `candidate_vs_raw`
detektiert **nicht** neu.

**`multiband_effective` je Stern (30.40-Follow-up).** Ist der fusionierte
`alpha_final` in ±1 px um das Sternzentrum über **alle** Bänder ≡ 0
(`|alpha| ≤ alpha_effective_eps`), misst Multiband dort identisch zu Raw ⇒
der Stern trägt **keine** Multiband-Evidenz und fällt aus der
FWHM-Vergleichsteilmenge. Die Pro-Stern-FWHM (raw vs. multiband) läuft nur
auf Sternen, an denen **beide** Seiten einen endlichen Patch-Fit liefern
**und** `multiband_effective` gilt.

**Deterministischer Bootstrap-CI.** `bootstrap_median_ci`: 2000 Resamples
(`kMultibandValidationBootstrapResamples`), `SplitMix64` aus festem
`kMultibandValidationBootstrapSeed = 0x9E3779B97F4A7C15`, sortierte
Resample-Mediane, `ci_low/ci_high` = 2,5-/97,5-Perzentil,
`relative_width = (ci_high − ci_low) / median`. Reproduzierbar und testbar.
`fwhm_ci_ok := (n ≥ min_stars_fwhm) ∧ (relative_width ≤ 0,10)` (§15.3.5).

**Seam-Metrik — bewusste Abweichung vom Legacy-Proxy, Planbestätigung
ausstehend.** §15.3.4 fordert `seam_score ≤ 1,05·seam_score_uniform` und
§14.7 „keine Seam-Stufe an der Maskenkante", **definiert aber keine Formel**.
Der Legacy-`compare_aqmh_to_reference`-`seam_score` (globale
Gradientenenergie `mean|∇|/σ`) ist für §15 **unbrauchbar**: er bestraft
**genau** die PSF-Schärfung, die die Methode erzeugen soll — empirisch
`seam_raw/seam_uniform > 1,05` allein aus der geringeren PSF-Breite, ohne
jede echte Diskontinuität. Ersatz: `boundary_seam_score` =
mittleres `|Laplace|` auf der **inneren Kante** der Support-Maskenkontur
**geteilt durch** mittleres `|Laplace|` des **eigenen Innenbereichs** des
Kandidaten (gestridet). Selbstnormierend: eine uniforme PSF-Änderung kürzt
sich im Quotienten (~1), eine echte Maskenkanten-Stufe hebt **nur** den
Zähler. **Die exakte Form ist eine Implementierungswahl und erwartet
Planbestätigung**; die selbst eingeführten Konstanten
(`kMultibandValidationSeamMinBoundaryPixels`, Innen-Stride-Ziel) sind ebenfalls
noch nicht plan-fixiert.

> **Update 30.47 (2026-09-06):** Die erste Fassung sampelte die **Randpixel
> selbst** und kollabierte damit auf jedem realen Maskenfeld zum 0-Sentinel
> (`ratio(0,0)=∞` hätte Raw verworfen). Behoben: jetzt **Interior-Edge**
> (Pixel einen Schritt innerhalb, voll on-support), nicht messbar ⇒ N/A statt
> 0. Aber der **M42-Resume** zeigt: auf realem OSC-Luma mit ~5 % verstreuten
> Ein-Pixel-Dropouts ist die Metrik *inert* (U/R/M `seam_score` 1,030/1,026/
> 1,023, < 0,4 % auseinander). **Die Seam-Form-Frage ist damit nicht mehr nur
> „Formel bestätigen", sondern: welcher Locus?** Wahrscheinliche Reparatur —
> morphologisches Öffnen der Stützmaske vor der Randableitung, damit isolierte
> Dropouts nicht beitragen. Bis zur Plan-Antwort **nicht weiter gepatcht**;
> blockiert das Umschalten der ausgelieferten Datei (30.43-Folgepunkt 1).

**Hash-Domäne:** ~~falls das Auswahlergebnis je `selected_candidate` speist,
gehören `kMultibandValidationVersion` **und** die Seam-Konstanten in die
`multiband_config_hash`-Domäne~~ — **zurückgenommen in 30.46:** eigener
`validation_config_hash` in `forward_drizzle.json` (der Store-Hash ist von
diesen Werten unabhängig); die Seam-Konstanten sind dort mitgehasht.

**Auswahllogik (`select_reconstruction_candidate`).** Feld- und
Tail-Metriken werden am festen Sternset via `compare_aqmh_to_reference`
**gemessen** (nur Messung, keine Legacy-Entscheidungslogik);
`background_rms` gegen **Uniform**; Seam via `boundary_seam_score` (nicht der
Legacy-Global-Gradient-Proxy). **Raw vs. Uniform:** Support, Numerik
(Inf verboten, NaN = Off-Support erlaubt), `background_rms`
(anwendbar + Verhältnis), Seam-Verhältnis nur bei `has_boundary` — ein
verletztes **oder N/A** anwendbares Pflicht-Gate ⇒ **Uniform**. **Multiband
vs. Raw (§15.3.4):** jede der sechs Ungleichungen über `need(applicable,
pass, na_msg, fail_msg)`; **N/A setzt `mb_fail`** (keine positive Evidenz) ⇒
**Raw** bleibt („kleine Stichproben sind nie ein impliziter Pass",
§15.3.6). Alle sechs bestanden + Support + Numerik ⇒ **Multiband**. Alle
Pro-Metrik-Felder (`applicable`/`value`/`reason_if_not_applicable`) werden
**vor** dem Gating befüllt, sodass der Report-Vertrag (§15.3.6) auch für
Metriken hinter dem ersten Fehlschlag von `need()` erfüllt bleibt.

**Verifiziert.** `tests/test_multiband_validation.cpp`, 8 handgebaute
synthetische Fälle (alle grün): (1) Bootstrap-CI deterministisch +
klammert den Median; (2) `multiband ≡ raw` (alpha 0) ⇒ **Raw** (0,95×-FWHM-
Gate scheitert bei Gleichheit); (3) echte flusserhaltende FWHM-Verbesserung
(`a0·(σ_r/σ)²`, 260×240, 81 Sterne) ⇒ **Multiband** promoviert; (4)
Raw-Hintergrund-Regression (4× Rauschen) ⇒ **Uniform**; (5) < 20 effektive
Sterne ⇒ FWHM **N/A** ⇒ **Raw** bleibt; (6) `prepare_validation_samples`
setzt `multiband_effective` korrekt aus Pro-Band-Alpha-Maps (nur linke
Hälfte aktiv); (7) rauschfreier (entarteter) Hintergrund ⇒ `background_rms`
**nicht anwendbar** ⇒ Pflicht-Safety-N/A ⇒ **Uniform** (der `ratio()`-Pfad
mit ~0-Nenner wird nie erreicht, weil die Anwendbarkeitsprüfung vorausgeht);
(8) echte Seam-Stufe an der Support-Innenkante blockiert Multiband **trotz**
besserer FWHM ⇒ **Raw**, Begründung enthält „seam". Hauptsuite unverändert
grün bis auf das bekannte `test_acceleration_backend.cpp:254` (CUDA-
Backend-Wahl, unrelated).

**Offen für M6 nach diesem Schritt:** die Runner-`selected_candidate`-
Verdrahtung folgt in **30.43**; danach bleibt nur die **reale
Registrierungs­runde** (M31 + M42/OSC).

<a id="historie-30-43"></a>

### 30.43 M6: `selected_candidate` in der Runner-`MULTIBAND`-Phase verdrahtet (§16.3) (2026-09-06)

Elfter M6-Schritt: die §15-Auswahl wird jetzt im Runner ausgeführt und als
**Entscheidung protokolliert**. Sie ändert **noch nicht**, welche Datei
ausgeliefert wird (siehe „Bewusst offen" unten).

**Kandidatenbildung im selben Fusionsdurchlauf.**
`fuse_multiband_store_to_image` nimmt einen optionalen
`MultibandCandidateLuma *candidates_out`. Im bereits existierenden
Streifen­loop (der `U`/`R` pro Streifen ohnehin liest und bisher verwarf)
werden zusätzlich drei Vollbild-Arbeitsluminanz­ebenen
(`uniform_luma`/`raw_luma`/`multiband_luma`), die Uniform-Luma-Supportmaske
und die fusionierten `alpha_final`-Pro-Band-Maps akkumuliert — **ohne
zusätzliche Store-I/O**. Der **Peak-Eingangs-Residency** bleibt
`O(chunk + 2·halo)` Zeilen; die **Vollbild-Residency** wächst um die drei
Kandidatenebenen (`3·4N` Byte) plus die aktiven Alpha-Maps
(`≤ levels·4N`) plus die Byte-Maske (`N`) — für ein Vollauflösungs-Mosaik
der dominante Term, relevant für die M9-RAM-Gates (§31). Die Arbeitsluminanz
ist die **eine** feste Definition
`kWorkingLumaDefinition = 0.25R+0.50G+0.25B` (MONO: `L` direkt), neu als
einzige Quelle in `multiband_fusion.hpp` — `luma_band` (Energieguard) **und**
der `luma_definition`-String im Artefakt lesen dieselbe Konstante, können
also nicht divergieren. Luma-Support nur, wo **alle** aktiven Kanäle
co-präsent sind (Bayer-Geometrie kann das dünn lassen — im synthetischen
OSC-Store-Test ist die Prüfung deshalb beschränkt, nicht `> 0`).

**Auswahl im Runner.** `prepare_validation_samples(uniform_luma, …,
uniform_support, alpha_final_by_band)` detektiert die feste Sternpopulation
auf der Uniform-Kontrolle und setzt `multiband_effective` aus den
Alpha-Maps; `select_reconstruction_candidate(uniform, raw, multiband, …,
uniform_support)` liefert `SelectedCandidate` + `reason`. **Dieselbe**
Uniform-Supportmaske geht an **beide** Aufrufe (sonst wäre das Seam-Gate
still über `has_boundary=false` deaktiviert, während Sterne trotzdem
maskiert würden).

**Artefakt `artifacts/forward_drizzle.json`.** Neu geschrieben mit **nur den
tatsächlich befüllbaren** §16.3-Feldern: `schema_version`,
`pipeline_method`, `pipeline_contract_version`, `sampling_plan_hash`,
`coverage_geometry_hash`, `multiband_reconstruction_hash`,
`multiband_levels`, `luma_definition`, `validation` (Version, `stars_total`
/ `stars_multiband_effective` / `multiband_star_sample_count`, je Kandidat
die sechs `ValidationMetric` mit `value`/`applicable`/`sample_count`/
`ci_low`/`ci_high`/`reason_if_not_applicable` + `support_ok`/`numerics_ok`),
`selected_candidate`, `selection_reason`, `fallback_reason`, `outputs[]` (die
geschriebene `reconstruction_multiband.fits` + sha256), `commit_complete`.
`fallback_reason` ist `null` **nur** bei Multiband-Auswahl und trägt für
**jede** Nicht-Multiband-Auswahl den `reason`-String — auch wenn Raw der
normale konservative Ausgang ist (Multiband hat das 0,95×-FWHM-Gate schlicht
nicht geräumt), nicht nur bei echten Gate-Verletzungen; die vollständige
Begründung steht zusätzlich unbedingt in `selection_reason`. **Nicht**
gestubbt werden `coverage`/`profiles`/`clipping`/`acceleration`/
`timing_seconds` — ein leeres Objekt läse sich als „gemessen, nichts zu
berichten"; §16.4 listet dort echte Pflichtdiagnostik, die dieser Schritt
noch nicht liefert. `selected_candidate` + `selection_reason` stehen auch im
`MULTIBAND`-`phase_end`-Event und im Checkpoint. **Kein** Checkpoint-Hash-
Guard für `forward_drizzle.json`: die `MULTIBAND`-Phase regeneriert es bei
jedem (Wieder-)Lauf vollständig, es gibt beim Resume nichts zu verifizieren
(anders als die unveränderlichen Geometrie-Prädekessoren).

**Verifiziert.** `tests/test_drizzle_profile_store.cpp`: MONO-Store —
`fuse_multiband_store_to_image(&cand)` liefert `multiband_luma` **bit-exakt**
zum fusionierten `X_out` (NaN außerhalb), `uniform_support` = Uniform-eigener
Support, `alpha_final_by_band` Größe 3 mit **D3 leer** (Raw-Quelle) / D1
belegt, alle Felder chunk-unabhängig (`chunk_rows ∈ {7,2}`). OSC-Store — die
nicht-ditherte CFA-Fixture ko-lokalisiert **nie** alle drei Farben in
derselben Ausgabezelle, Luma-Support ist also legitim überall leer; was
nicht-vakuös bleibt: `combine_luma` **überschätzt nicht** — `uniform_support`
== tri-Kanal-Prädikat exakt (= all-Null), `multiband_luma` all-NaN,
`alpha_final_by_band` Größe 3 / D3 leer, chunk-unabhängig. Die bit-exakte
OSC-Luma-Kombination braucht eine ditherte Geometrie und ist dem realen
M42/OSC-Lauf vorbehalten; die Akkumulations-/Streifen­mathematik deckt der
MONO-Store-Test bit-exakt ab. `tests/test_runner_forward_drizzle.cpp` `[forward-runner]`:
`artifacts/forward_drizzle.json` existiert, `selected_candidate` ∈
{`drizzle_uniform`,`drizzle_raw`,`drizzle_multiband`}, das `phase_end`-Event
trägt denselben Wert; die Fixture (2 konstante Frames + all-NaN-Artefakt ⇒
`alpha≡0` ⇒ Multiband≡Raw; nahezu konstante Kontrolle ⇒ `background_rms` ~0
⇒ **nicht anwendbar** ⇒ Pflicht-Safety-N/A) wählt **`drizzle_uniform`** mit
`fallback_reason` gesetzt — genau der 30.42-Test-7-Pfad, end-to-end im
Runner. Hauptsuite **484/485** (weiterhin nur
`test_acceleration_backend.cpp:254`).

**Bewusst offen (eigener Batch).**
1. **Auslieferung.** Der Runner schreibt weiter `reconstruction_multiband.fits`
   und keyt Checkpoint/Resume darauf; die Auswahl ist eine **protokollierte
   Entscheidung**, kein Umschalten der ausgelieferten Datei. §16.3 trennt
   „welcher Kandidat gewann" (`selected_candidate`) von „welche Dateien
   existieren" (`outputs[]`) selbst. Das Umhängen der ausgelieferten Ebene
   ändert Resume-Vertrag, Checkpoint-Bedeutung und den STACKING-Input (§17.2)
   und gehört mit eigenen Tests in einen späteren Schritt.
2. **Hash-Domäne.** ~~Da das Auswahlergebnis nun `selected_candidate` speist,
   gehören `kMultibandValidationVersion` **und** die
   `boundary_seam_score`-Konstanten in die `multiband_config_hash`-Domäne.~~
   **Zurückgenommen in 30.46:** `multiband_config_hash` hasht die Store-Bytes,
   die von den Auswahlkonstanten unabhängig sind. Umgesetzt wurde ein eigener
   `validation_config_hash` in `forward_drizzle.json` — kein Store-Hash
   berührt, keine Migration.
3. **§16.4 teilweise bedient:** Feld-/Backend-/RSS-/Cache-Retention-/
   Timing-Diagnostik im `forward_drizzle.json` fehlt noch. (`cache_retention`
   ist seit 30.45 im `run_end`-Event, nicht im Artefakt.)
4. Die **reale Registrierungs­runde** (M31 + M42/OSC). **Vorhersage für den
   OSC/M42-Lauf, die dort zu bestätigen/widerlegen ist:** die
   OSC-Arbeitsluminanz braucht R∧G∧B-Ko-Support in derselben Ausgabezelle —
   wo eine ditherte Geometrie das in den Feldrändern dünn lässt, sieht
   `prepare_validation_samples` dort weniger nutzbare Sterne, und
   `min_stars_fwhm = 20` wird genau dort schwerer erreichbar. Das wirkt in
   dieselbe Richtung wie der 30.40-Befund (`A_artifact < 8` unterdrückt Alpha
   bereits auf ~40 % des OSC-Innenbereichs bei nf=30): beide drücken OSC
   Richtung „keine positive Multiband-Evidenz ⇒ Raw bleibt". Das ist
   plankonform und konservativ, kein Fehler — aber der reale Lauf soll es
   messen, nicht überraschen.

<a id="historie-30-44"></a>

### 30.44 M7 begonnen (Slice 1): transaktionaler CPU-Neustartvertrag für die FORWARD_DRIZZLE-CUDA-Phase (§19.4) (2026-09-06)

M7 zerfällt in Slices. **Dieser Slice liefert ausschließlich den
Transaktionsvertrag aus §19.4** — die Droplet-/Clipping-/Profil-Kernel
(§19.2 Stufen 3–7) und ihre Paritätsmatrix (§19.5) sind Slice 2.

**Warum zuerst der Vertrag.** Er ist eine Korrektheitsgarantie **unabhängig
von jedem Kernel** und mit Fault-Injection **ohne GPU** vollständig testbar:
scheitert die CUDA-Phase in irgendeinem Chunk, wird die **gesamte**
`FORWARD_DRIZZLE`-Phase auf dem CPU-Referenzpfad neu gestartet und nur ein
vollständig berechnetes, validiertes, gehashtes CPU-Ergebnis committed — nie
ein gemischtes CPU-/CUDA-Bild oder ein halb akkumuliertes Pixel.

**Neu.** `include`/`src/reconstruction/forward_drizzle_cuda.{hpp,cpp}`
(immer gebaut, kein `.cu` in Slice 1):
- `ForwardDrizzleCudaError` — signalisiert dem Aufrufer „uncommittete
  Generation verwerfen, ganze Phase auf CPU neu starten";
- `forward_drizzle_cuda_runtime_available()` — Slice 1 **immer `false`**
  (keine Kernel); Slice 2 macht daraus einen echten Device-Probe;
- `set_/forward_drizzle_cuda_fault_after_chunks(n)` — Test-only Fault-
  Injection (Prozess-global, auch aus
  `TILE_COMPILE_FORWARD_DRIZZLE_CUDA_FAULT_AFTER_CHUNKS`): eine CUDA-
  Persistenz wirft `ForwardDrizzleCudaError` nach `n` committeten Stripes
  (`n=0` ⇒ vor dem ersten Stripe);
- `ForwardDrizzleCudaOptions{bool attempt}` — durchgereicht in
  `persist_forward_drizzle_multiband`.

**Store-Seite.** `persist_forward_drizzle_multiband` bekommt `cuda`-Optionen.
Bei `attempt` und **ohne** armierte Fault-Injection wirft es **sofort**
(Slice 1 hat keinen Kernelpfad); mit Fault-Injection wirft es aus dem
Stripe-Sink nach `n` Stripes. Der Sink-Wurf propagiert durch
`stream_forward_drizzle_uniform_and_raw` (kein `try/catch`, keine
OpenMP-Region über dem Sink) bis zum `StoreWriter`, dessen Destruktor die
**nicht publizierte** Generation via `fs::remove_all` entfernt — `current.json`
bleibt unangetastet, es gibt keine leere Restgeneration.

**Orchestrierung.** `persist_multiband_store_from_predecessors` bekommt einen
`acceleration_backend`-String (`"cpu"` | `"cuda"`). Bei `"cuda"` **und**
etwas Attemptbarem (echter Device-Pfad ab Slice 2 **oder** armierte Fault-
Injection) wird der CUDA-Weg versucht; auf `ForwardDrizzleCudaError` wird der
**gesamte** Build erneut aufgerufen — mit **identischen** Argumenten, daher
deterministisch bit-identisch — auf dem CPU-Pfad. Der Retry ist **nicht
rekursiv**: scheitert der CPU-Neustart selbst, propagiert das. Ergebnis trägt
`backend_used` (`"cuda"` nur bei vollständig durchgelaufenem CUDA-Versuch;
`"cpu"` auch nach Fallback) + `cuda_fallback_reason`.

**Runner.** `AccelerationPhase::forward_drizzle` neu angehängt;
`phase_supports_backend(forward_drizzle, cuda) = true` (OpenCV-CUDA/OpenCL
`false` — der Pfad ist ein **eigener** Kernel, keine OpenCV-Operation).
`run_forward_drizzle_stages` löst über `select_acceleration_backend` die
**Absicht** auf (`"cuda"` wenn baubar) und reicht sie an
`persist_multiband_store_from_predecessors`; die Entscheidung „gibt es ein
nutzbares Gerät" trifft die Persistenzfunktion, nicht der Runner. Das
`FORWARD_DRIZZLE`-`phase_end`-Event + der Checkpoint tragen
`acceleration_backend` (Slice 1: immer `"cpu"`) und ggf.
`cuda_fallback_reason`. **Kein** `acceleration`-JSON-Block in
`forward_drizzle.json` in diesem Slice — bis es einen echten Kernel gibt,
wäre der einzige Wert eine Konstante (gehört mit Slice 2 + §16.4).

**Verifiziert (ohne GPU).** `tests/test_drizzle_profile_store.cpp`
(`[drizzle-store]`, MONO Levels-3, 48×48, `chunk_rows=8` ⇒ 6 Stripes):
(1) `attempt` ohne Fault ⇒ sofortiger `ForwardDrizzleCudaError`, **keine**
`current.json`, **keine** Generation angelegt; (2) Fault nach 3 Stripes ⇒
Wurf, die angefangene Generation ist vom `StoreWriter`-Destruktor entfernt;
(3) Fault nach 2 Stripes **im selben Root**, dann CPU-Neustart ⇒ das
committete Store ist über alle 19 Ebenen (`uniform`/`raw`/`fine`/`medium` ×
`value`/`weight_sum`/`n_eff`/`support` + 4 Alpha-Maps) **`sha256`-identisch**
zum reinen CPU-Build, `verify_drizzle_profile_store` grün.
Die 19-Ebenen-`sha256`-Gleichheit dieses Same-Root-Neustarts (ohne zweite
Fixture) ist die **tragende** Garantie. `tests/test_runner_forward_drizzle.cpp`
(`[forward-runner]`) ergänzt sie **bestätigend**: die
`FORWARD_DRIZZLE`-`phase_end` meldet `acceleration_backend="cpu"`; ein
injizierter Fault (nach 1 Stripe) über den Runner ⇒ Phase startet auf CPU
neu, `cuda_fallback_reason` enthält „injected fault", `status`
`final_image_ready`, `reconstruction_multiband.fits` `sha256`-identisch zu
einem zweiten fault-freien Fixture-Lauf (setzt voraus, dass der FITS-Header
keine pfad-/zeitabhängigen Felder trägt — falls das je hinzukommt, ist der
Store-Test die maßgebliche Prüfung). Fault-Injection ist prozess-global und
in beiden Testdateien über einen RAII-`CudaFaultGuard` gekapselt (Disarm im
Destruktor), damit ein fehlschlagendes `REQUIRE` den Wert nicht in
Folgetests leckt. (Der CUDA-Build ist hier aktiv — GTX 1660 Ti, nvcc 13.0,
`CMAKE_CUDA_ARCHITECTURES` enthält 75 —, daher läuft der Runner-Fault-Test
echt statt zu skippen.) Hauptsuite **486/487** (weiterhin **nur**
`test_acceleration_backend.cpp:254`, s. u.).

**Vorbestehender roter Test / Befund.**
`acceleration_context_keeps_aqmh_maps_cpu_only` schlägt auf einer Maschine
mit funktionsfähigem OpenCV-CUDA fehl: der Test verlangt, dass `opencv_cuda`
für `aqmh_maps` auf CPU zurückfällt, aber `phase_supports_backend` erlaubt
`opencv_cuda` für `aqmh_maps` (Zeile in `acceleration.cpp`). Das ist eine
**Legacy-AQMH**-Erwartungsdiskrepanz (die Regel griff früher nur, weil keine
Testumgebung OpenCV-CUDA hatte), **kein** M7-Regressor: der Mechanismus
`phase_supports_backend` **ist** die von §19.2/§20.8 geforderte Per-Phase-
Datentabelle und liefert für `forward_drizzle` das korrekte Ergebnis
(`cuda` erlaubt, alles OpenCV/OpenCL `false`). Der Fix (entweder `aqmh_maps`
aus der `opencv_cuda`-Zeile nehmen oder den Test an die Tabelle anpassen)
ist eine Legacy-Entscheidung außerhalb von M7 und wurde bewusst nicht
nebenbei gemacht.

**Slice 2 (mit GPU) — die zwei Dinge, die den Slice freischalten.**
`forward_drizzle_cuda.cu` mit der quadratkern-exakten Droplet-Akkumulation
(frame-lokale Atomics + feste profilweise Reduktion, §19.3), und
`forward_drizzle_cuda_runtime_available()` als echter Device-Probe statt des
hartkodierten `false`. Abnahme bleibt die Paritätsmatrix §19.5 an ihrem Ort;
Memory-Auto-Sizing, Timing und der `acceleration`-Block (§16.4) folgen im
selben Slice.

<a id="historie-30-45"></a>

### 30.45 M1–M6: Statustabelle nachgezogen, erster realer M31-Lauf, `star_support_ok`-Defekt gefunden und behoben (2026-09-06)

**Statustabelle (§30.1) auf den tatsächlichen Stand gebracht.** Sie war auf
2026-09-02 datiert („kein begonnener Meilenstein") und lief den seit M0
geleisteten Arbeiten weit hinterher. Nachgezogen: **M1 abgeschlossen**
(COMMON_OVERLAP aus den Geometriemasken ist mit der M3-Integration erledigt);
**M2 korrektheits-vollständig** — die zuvor als offen geführte store-weite
Transaktionalität und die eigene Phase mit Resume sind durch die
M3-Integration (30.23) geschlossen, allein die **Parallelisierung** des
Drizzle-Rasterisierers bleibt (reine Performance, kein Vertrag; sinnvoll
gebündelt mit M7-Slice-2); **M3**: `Q_composite`-Stream und
Zero-Veto-Weiterleitung waren bereits in M5 (30.30/30.31) erledigt, der
„automatische Uniform-Fallback im Resume-Pfad" ist **kein Defizit** (der Plan
verlangt fail-closed, der qualitative Rückfall ist §15) — wirklich verbleibend
sind die A-priori-Streifengrößenschranke (reaktives Netz existiert, 30.17) und
der **Cache-Lebensdauervertrag**; **M4**-Rest ist plan-seitig nach M6/M10
gelegte Ausgabe-Cutover-Arbeit.

**Cache-Lebensdauervertrag (§16.2) verdrahtet.** `keep_profile_cache_after_run`
/ `delete_source_cache_after_run` wurden geparst, aber nicht konsumiert. Jetzt:
nach einem **committeten** finalen Bild löscht `run_forward_drizzle_stages` den
internen U/R/F/M-Profilstore per Default (er ist ein Rekonstruktionscache,
**nie** ein Downstream-Resume-Prädekessor) und behält ihn nur bei
`keep_profile_cache_after_run=true` als gehashten Cache; die Quellcaches
(`normalized_frames`, `source_quality_maps`) werden per Default behalten und
nur bei `delete_source_cache_after_run=true` gelöscht — dann meldet das
`run_end`-Event `cache_retention.resume_reconstruction_disabled=true`. Alles im
`run_end`-Event + Checkpoint. `[forward-runner]`-Tests: Default löscht den
Profilstore und behält den Normalized-Cache; die Resume-Tests wählen
`keep_profile_cache_after_run=true`, um den Store über Läufe zu inspizieren.

**Erster realer M31-Lauf (`reconstruct`, 40 Frames).** Voller Pfad SCAN →
NORMALIZATION → REGISTRATION → NORMALIZED_CACHE → SAMPLING_GEOMETRY →
COMMON_OVERLAP → SOURCE_QUALITY_MAPS → GLOBAL_QUALITY → FORWARD_DRIZZLE →
MULTIBAND → `final_image_ready`, `success=true`. **Echte 2/1-Produktions­geometrie**
(`internal_scale=2`, `output_scale=1`, `output_scale_applied=true`,
`analysis_pixels=32 603 432` intern; Stack-Sampling bestätigte
`stream_forward_drizzle_uniform_and_raw_2x2`) — der Downsample-Pfad **wurde**
real ausgeführt. `acceleration_backend="cpu"` mit
`cuda_fallback_reason="forward_drizzle_cuda_unavailable"` (M7-Slice-1 greift
real). **M5-Echtdatenpunkt erfüllt:** `SOURCE_QUALITY_MAPS` schrieb
`composite` + `scale_0..3` + `artifact` (`computed_scales=4`) für 40 reale
Frames; die drei Kandidaten unterscheiden sich real
(`background_rms` Uniform **2,210** vs. Raw **2,405** vs. Multiband **2,409**
am selben Sternset), d. h. `G_eff·Q_composite` verschiebt Raw messbar
gegenüber Uniform (deckt sich mit `mean|u−r| ≈ 2,1–2,6` aus 30.23).

**Befund: `star_support_ok` war für echte Daten unbrauchbar — behoben.** Der
erste Lauf detektierte **250** Sterne auf der Uniform-Kontrolle, aber die
Auswahl fiel auf `drizzle_uniform` mit Grund „raw star support invalid" — und
`support_ok` war für **alle drei** Kandidaten `false`, **auch für die
Uniform-Kontrolle selbst**. Ursache: `star_support_ok` **und**
`per_star_fwhm_aligned` verlangten den **gesamten** 15×15-Patch (225 Pixel)
endlich. Reale OSC-Arbeitsluminanz hat verstreute Off-Support-NaN (Luma
braucht R∧G∧B-Ko-Support je Pixel); bei ~88 % Pixel-Support überlebt ein
225-Pixel-Patch mit Wahrscheinlichkeit `0,88²²⁵ ≈ 10⁻¹²` — **null** von 250
Sternen bestehen, garantiert, nicht datenabhängig. Ein Gate, das die sichere
Referenz ablehnt, misst das Falsche. **Fix:** gemeinsamer
`extract_star_patch()` — Mittelpunkt endlich **und** ≥ 75 %
(`kMultibandValidationStarPatchMinFiniteFraction`, versioniert im Header) des
Patches endlich; die verbleibenden
spärlichen Löcher werden mit dem Endlich-Pixel-**Median** des Patches gefüllt
(Hintergrundschätzung, kann keinen Peak erzeugen). `star_support_ok` und die
FWHM-Extraktion nutzen jetzt denselben Pfad — die beiden Prüfungen sind
konsistent statt dass eine die andere überstimmt. Neuer synthetischer Test:
ein Feld mit ~6 % verstreuten NaN (nie auf einem Sternzentrum) liefert
weiterhin nutzbare Sterne, `support_ok` für alle drei Kandidaten, FWHM
anwendbar, Multiband promoviert. Hauptsuite **487/488** (weiterhin nur
`test_acceleration_backend.cpp:254`).

**M31-Neulauf** (`resume-reconstruction --from-phase FORWARD_DRIZZLE`, reale
Predecessors wiederverwendet) mit dem Fix — **plankonformes Ergebnis**:
- `support_ok` jetzt `true` für **alle drei** Kandidaten (Fix greift real);
- `stars_total = 250`, aber `stars_multiband_effective = 0`: `alpha_final` ist
  an **allen** 250 Sternzentren ≈ 0. Das ist die vorhergesagte Interaktion
  von 30.40 mit dem **Energieguard §14.5**, der Alpha gerade an hellen Kernen
  (= Sterne) drückt — die sternbasierte §15-Validation sieht dort Multiband ≡
  Raw. FWHM daher N/A für Raw und Multiband;
- Auswahl: **`drizzle_uniform`**, Grund
  `raw rejected -> uniform: raw background_rms regression vs uniform`
  (`bg_RMS` Raw 2,405 / Uniform 2,210 = **1,088 > 1,05**). Das strikte
  §15.3.2-Sicherheitsgate greift korrekt und **bevor** Multiband bewertet wird
  (§15.3 Schritt 2). Der Forward-Drizzle-Regressionswert **0,088** liegt weit
  unter dem in §15.3 „Erwartete Auswahlverteilung" für PREWARP-AQMH
  dokumentierten `≈ 0,56` — der neue Pfad ist hier deutlich rauschärmer, aber
  das 5-%-Gate ist bewusst streng. Der Plan hält genau dieses Ergebnis fest:
  „Es ist … ein plausibles und zulässiges Ergebnis, dass Raw und insbesondere
  Multiband auf einem Teil der Datensätze **nicht** promoted werden … eine
  niedrige Multiband-Trefferquote ist kein Fehlschlag der Methode."
- `cache_retention: {profile_cache: deleted, source_cache: retained}` im
  `run_end` — der **Cache-Lebensdauervertrag** ist damit auf einem realen Lauf
  verifiziert.

**Damit ist der M6-Echtdatenpunkt für M31 erfüllt:** der Pfad lief end-to-end
auf realer 2/1-Produktions­geometrie, die Dreiwegvalidation griff mit 250
realen Sternen, evaluierte die Sicherheitsgates korrekt und lieferte das
plan-antizipierte Uniform-Control mit exakt dem dokumentierten Gate-Grund. Der
`star_support_ok`-Defekt (der den echten Auswahlpfad zuvor verdeckte) ist
behoben und synthetisch gepinnt.

---

<a id="historie-30-46"></a>

### 30.46 M6: `validation_config_hash` statt Einbindung in `multiband_config_hash`; zwei plan-seitige Fragen aus dem M31-Lauf (2026-09-06)

**Auswahl-Reproduzierbarkeit über einen eigenen Hash — nicht über den
Store-Identity-Hash.** Die 30.42/30.43-Folgenotiz sah vor,
`kMultibandValidationVersion` und die selbst eingeführten Konstanten in
`multiband_config_hash` zu falten. Das ist **falsch** und wird hiermit
zurückgenommen. `multiband_config_hash` ist Teil von `DrizzleStoreIdentity`
und bestimmt die **Store-Bytes** (U/R/F/M-Ebenen + die vier Alpha-Maps). Die
Validierungskonstanten — Patch-Radius, Mindest-Endlich-Anteil, Bootstrap-Seed/
-Resamples, die Seam-Konstanten, die `MultibandValidationConfig`-Schwellen —
ändern **kein** einziges dieser Bytes; sie bestimmen ausschließlich das
**Auswahlergebnis**, das in `artifacts/forward_drizzle.json` liegt. Sie
einzufalten würde zwei byte-identische Stores unterschiedlich hashen, jeden
existierenden Store invalidieren und die 30.41-Hash-Stabilitätstests brechen —
ohne Gegenwert.

Stattdessen: `multiband_validation_config_hash()` (SHA-256 über die
versionierten Konstanten + die effektive `MultibandValidationConfig`) wird als
`validation.validation_config_hash` in `forward_drizzle.json` geschrieben. Der
`selected_candidate` ist damit reproduzierbar/auditierbar, `DrizzleStoreIdentity`
bleibt unangetastet, keine Migration. Die zuvor inline in
`boundary_seam_score` liegenden Magic Numbers (`>= 8`, Interior-Stride-Ziel
`40000`) sind jetzt benannte, in den Hash aufgenommene Header-Konstanten
(`kMultibandValidationSeam*`). Tests: neuer Stabilitäts-/Sensitivitäts-Test
(64 Hex, deterministisch, reagiert auf eine echte Schwellenänderung, ignoriert
Neu-Angabe der Defaults); `[forward-runner]` prüft den Hash im Artefakt gegen
`multiband_validation_config_hash()`. Hauptsuite **488/489** (weiterhin nur
`test_acceleration_backend.cpp:254`).

Der `seam_score`-Locus + die Konstanten `>= 8` / Interior-Stride sind ein
eigener offener Punkt — **§30.47** (Sentinel-Defekt behoben, Metrik auf realem
OSC aber *inert*; Form plan-seitig unbestätigt, 30.42).

`metric_json` im Runner
serialisiert eine nicht-anwendbare Metrik jetzt einheitlich als `value: null`
(vorher `0` für den Default-Zweig, `null` für den NaN-Zweig — ein Konsument
konnte „nicht berechnet" und „als NaN berechnet" nicht unterscheiden). Der
Auswahl-Config (`MultibandValidationConfig`) wird im Runner **einmal**
instanziiert und speist Auswahl **und** `validation_config_hash` — der Hash ist
damit strukturell der tatsächlich verwendete Config, nicht nur „zufällig auch
Default". Hauptsuite **490/491** (weiterhin nur der vorbestehende
`test_acceleration_backend.cpp:254`).

**Plan-Frage 1 — `stars_multiband_effective = 0` auf realen M31-Daten.** §15
misst Multiband-Evidenz **an Sternen**; der Energieguard §14.5 drückt Alpha
gerade an hellen Kernen, also genau dort, wo Sterne sind. Auf M31 war der
Schnitt über **alle 250** Sterne leer — intern konsistent mit §14.5, §15 und
der §15.3-Notiz „niedrige Multiband-Trefferquote ist kein Fehlschlag". Aber:
der sternbasierte Promotionspfad kann die Multiband-Gewinne, die in
**ausgedehnter Struktur** liegen, strukturell nicht sehen. Das ist **keine**
Code-Frage — es ist eine Frage an den Plan-Eigner, ob §15.3.4 eine
nicht-sternbasierte Evidenzkomponente braucht (z. B. Struktur-SNR auf einer
Maske ohne Sternkerne) oder ob die reine Sternmessung bewusst so bleibt.

**Plan-Frage 2 — das 5-%-`background_rms`-Gate vs. Forward-Drizzle-Rauschen.**
M31 verwarf Raw mit `bg_RMS`-Verhältnis 1,088 > 1,05. §12.4 hält fest, dass
Forward-Drizzle bei `pixfrac < 1` / 2x korreliertes Rauschen hinzufügt; §15.3
dokumentiert für PREWARP-AQMH `≈ 0,56`. Der neue Pfad liegt mit 0,088 weit
darunter, verfehlt das strikte Gate aber trotzdem. Frage an den Plan-Eigner:
ist das 5-%-Gate gegen den (in §12.4 erwarteten und benannten) Drizzle-Rausch­
beitrag als bewusst-streng gewollt, oder soll die Schranke die
`pixfrac`/Scale-Konfiguration berücksichtigen?

**M42-Echtdatenlauf (OSC, 40 Frames) — zweiter unabhängiger M6-Datenpunkt.**
Voller `reconstruct`-Pfad → `final_image_ready`, `success=true`, reale
2/1-Produktions­geometrie (`internal_scale=2`, `output_scale=1`,
`output_scale_applied=true`, `estimated_peak_bytes ≈ 4,23 GB`,
`kernel_noise_sigma_factor = 1,473`), `acceleration_backend=cpu`
(`forward_drizzle_cuda_unavailable`), `pixels_supported = 21 079 282`.
Dreiwegvalidation:
- `stars_total = 105` (Nebelfeld, punktquellenärmer als M31s sternreiches
  Feld — plausibel);
- **`support_ok = true` für alle drei Kandidaten** — der
  `extract_star_patch`-Fix (30.45) greift auf einem **zweiten, unabhängigen**
  realen OSC-Datensatz;
- `stars_multiband_effective = 0` erneut (`alpha_final ≈ 0` an allen 105
  Sternzentren) — dieselbe strukturelle Interaktion mit §14.5 wie M31;
- FWHM/p90/Tail/Elongation N/A („fewer than 20 effective stars (0 of 105)");
- `background_rms`: Uniform **0,9919** / Raw **1,0797** / Multiband **1,0807**;
  **Verhältnis Raw/Uniform = 1,0886 > 1,05** ⇒ Raw wird vom strikten
  §15.3.2-Gate verworfen, **bevor** Multiband bewertet wird ⇒
  **`drizzle_uniform`**, Grund
  `raw rejected -> uniform: raw background_rms regression vs uniform`;
- `cache_retention: {profile_cache: deleted, source_cache: retained}` — Vertrag
  auf einem zweiten realen Lauf verifiziert.

**Bemerkenswert:** M42s Raw/Uniform-`bg_RMS`-Verhältnis (1,0886) ist praktisch
identisch mit M31s (1,0882). Der Forward-Drizzle-Rauschbeitrag bei
`pixfrac<1` / 2x ist damit ein **systematischer, reproduzierbarer** ~8,8-%-
Offset über zwei verschiedene Objekte — kein Datensatzrauschen. Das
**verstärkt Plan-Frage 2**: das strikte 5-%-Gate schließt den neuen Pfad
konsistent vom Raw-/Multiband-Zweig aus, obwohl der Regressionswert (0,088)
weit unter dem §15.3-Referenzwert für PREWARP-AQMH (`≈ 0,56`) liegt.
`validation_config_hash` fehlte im M31-Artefakt und im ersten M42-Artefakt
(beide liefen mit dem Binary **vor** dieser Änderung); der M42-Resume ab
FORWARD_DRIZZLE (s. u.) schrieb ihn dann real
(`082f44803b3fbbbb0a7ad9a3264aa8a5f3e1db3a4705c4d9cdf5411fa3322292`); der Wert
ist deterministisch und synthetisch + im `[forward-runner]`-Test gepinnt.

**Damit ist der M6-Echtdatenpunkt doppelt erfüllt (M31 sternreich, M42
Nebel):** der Pfad läuft end-to-end auf realer 2/1-Geometrie, die
Dreiwegvalidation detektiert reale Sterne (250 bzw. 105), der
`star_support_ok`-Fix trägt auf beiden, die Sicherheitsgates greifen korrekt
und liefern reproduzierbar das plan-antizipierte Uniform-Control mit exakt dem
dokumentierten Gate-Grund.

---

<a id="historie-30-47"></a>

### 30.47 M6: `seam_score`-Sentinel-Defekt behoben — auf realem OSC-Luma aber *inert*; die Seam-Form bleibt plan-seitig unbestätigt (2026-09-06)

**Defekt (dritter Fall derselben Klasse). `seam_score` war auf jedem realen
Maskenfeld der 0-Sentinel.** Aufgedeckt durch Review des ersten M42-Artefakts:
`seam_score val=0 appl=True` für **alle drei** Kandidaten (`sample_count =
4 137 234`). `boundary_seam_score` sampelte die **Randpixel selbst** — ein
Randpixel hat per Definition einen Off-Support-(NaN-)Nachbarn, `laplacian_abs`
gibt dort NaN zurück, `b_n` bleibt 0, die Funktion fällt auf `return 0.0`.
Exakt die `0,88²²⁵`-Situation aus 30.45 (`star_support_ok`) und dieselbe Klasse
wie die Ganz-Patch-endlich-Anforderung von `per_star_fwhm_aligned`: **drei
Metriken desselben Moduls, deren Stencil-/Endlichkeitsanforderung garantiert,
dass sie auf realen maskierten Daten nie messen.** Folge hier:
`ratio(seam_r, seam_u) = ratio(0,0) = ∞ > 1,05` — Raw würde in §15.3.2 als
„seam_score regression" verworfen, sobald `background_rms` das Gate passieren
lässt (auf M31/M42 nur unsichtbar, weil `bg_RMS` Raw zuerst verwarf).

**Fix.** `boundary_seam_score` sampelt jetzt die **Interior-Edge** — vollständig
gestützte Pixel *einen Schritt innerhalb* des Randes, deren 5-Punkt-Stern
komplett on-support ist, also messbar — normiert weiter auf das tiefe
Interieur. Nicht messbar ⇒ **NaN** (nicht 0); die Metrik wird dann
`applicable=false`. §15.3.2: kein Support-Rand ⇒ Seam-Constraint greift nicht
(kein Reject); Rand vorhanden aber unmessbar ⇒ wie ein verfehltes
Mandatory-Safety-Gate (Reject auf Uniform), inkl. neuem Reject-Pfad für ein
Interior-Edge-Set mit 1–7 Pixeln
(`< kMultibandValidationSeamMinBoundaryPixels`); §15.3.4-Promotion: kein Rand ⇒
Ungleichung vakuum erfüllt, unmessbar ⇒ keine positive Multiband-Evidenz ⇒
Raw. Regressionstest: Maskenloch mit **NaN** im Off-Support, **ohne**
Seam-Sprung ⇒ `applicable` für alle drei, `value` endlich `> 0`, Raw **nicht**
mit „seam"-Grund verworfen. Der bestehende Seam-Sprung-Test bleibt grün.

**Echtdaten-Nachlauf (M42-Resume ab FORWARD_DRIZZLE mit dem Fix-Binary): der
Sentinel ist weg, die Metrik ist auf realem OSC-Luma aber *inert*.** Post-Fix
aus `forward_drizzle.json`: `seam_score.sample_count = 1 077 246` (~5,1 % der
21 079 282 gestützten Pixel), `seam_score.value` Uniform **1,0296** / Raw
**1,0257** / Multiband **1,0233**; Verhältnisse **Raw/Uniform 0,9962**,
**Multiband/Uniform 0,9938** — die drei stimmen auf **< 0,4 %** überein.
Ursache: die reale OSC-Arbeitsluminanz-Stützmaske ist von ~5 % verstreuten
Ein-Pixel-Dropouts durchsetzt (die Vor-Fix-Randmenge war 4 137 234 Pixel ≈
19,6 % der gestützten Fläche — kein Rand, sondern Löcher überall). Die
Interior-Edge-Menge wird davon dominiert; Zähler (Kanten-Laplace) und Nenner
(Tiefen-Interieur-Laplace) laufen gegen denselben Wert. Der Fix beseitigt das
`ratio(0,0)=∞`-Fehlverhalten, aber das Gate kann einen **echten** Sprung an der
wahren Rekonstruktions-Stützgrenze nicht mehr sehen — das Signal ist stark von
Dropout-Kanten verdünnt.

**Das ist eine Frage der Seam-*Form* (welcher Locus gemessen wird), nicht des
Sentinels.** Wahrscheinliche Reparatur: morphologisches Öffnen der Stützmaske
vor der Randableitung, sodass isolierte Dropouts nicht beitragen, nur die
äußere Grenze + große zusammenhängende Löcher. Diese Form — inkl. `>= 8` und
Interior-Stride — ist plan-seitig unbestätigt (30.42) und wird dort als
offener Punkt geführt; **nicht** ein drittes Mal nach eigenem Ermessen
gepatcht. Ein Charakterisierungstest pinnt das aktuelle Verhalten: verstreute
Dropouts + ein echter +200-Schritt entlang der Interior-Edge-Spalten der
zusammenhängenden Lücke bewegen `seam_multiband/seam_uniform` um **< 5 %** (real
gemessen ~0,99) — eine Seam-Form-Reparatur, die den Locus auf die echte Grenze
einschränkt, bricht diesen Test bewusst und erzwingt eine sichtbare
Aktualisierung.

**M6-Auslieferungsumschaltung (§16.3-Folgepunkt „ausgelieferte Datei = gewählter
Kandidat", 30.43) bleibt blockiert**, bis der Plan-Eigner die Seam-Form
beantwortet: die Umschaltung darf nicht erfolgen, solange ein verpflichtendes
§15.3.2-Safety-Gate inert bzw. unbestätigt ist.

> **Beides in 30.48 aufgelöst (2026-09-06, „vervollständige M6"):** die
> Seam-Form ist per Auftrag entschieden (morphologisches Schließen der
> Stützmaske vor der Randableitung, `kMultibandValidationSeamMaskCloseRadius`)
> und die Auslieferungsumschaltung ist umgesetzt (§16.1-Layout).

**Muster-Beobachtung.** `star_support_ok`, `per_star_fwhm_aligned` (30.45) und
`seam_score` (hier) hatten alle dieselbe Wurzel: eine Mess­definition mit einer
Stencil-/Ganz-Nachbarschaft-endlich-Anforderung, die auf realen maskierten
OSC-Daten garantiert scheitert. Die p90-/Tail-/Elongations-Metriken laufen über
den **geteilten Legacy-Helfer** `compare_aqmh_to_reference` (nur Messung); der
lieferte auf M31/M42 plausible `star_count` (N/A kam von „< 30 effektive
Sterne", nicht von NaN-Nachbarschaft), ist aber bei einer künftigen
Modul-Revision mit demselben Blick zu prüfen — er wird mit dem Legacy-AQMH-Pfad
geteilt, Änderungen dort haben größeren Radius.

---

<a id="historie-30-48"></a>

### 30.48 M6 abgeschlossen: Seam-Form entschieden, §16.1-Auslieferung, §16.4-Diagnostik (2026-09-06)

Auf die Anweisung „vervollständige M6" wurden die drei in 30.46/30.47
offenen Punkte umgesetzt (der Plan-Eigner = Auftraggeber hat damit die zwei
Entscheidungsfragen implizit beantwortet). Reihenfolge A → B → C, jeder Schritt
mit Tests; Hauptsuite **490/491** (weiterhin nur der vorbestehende
`test_acceleration_backend.cpp:254`).

**A-Echtdaten-Bestätigung (M42-Resume mit dem Fix-Binary).** Der geschlossene
Locus schrumpft die Interior-Edge-Menge von **1 077 246** (30.47, Dropout-
dominiert) auf **2 723** Pixel — die echte Footprint-Grenze. `seam_score`
Uniform **1,0599** / Raw **1,0652** / Multiband **1,1311**; Verhältnisse
**Raw/Uniform 1,0053** (innerhalb des 5-%-Gates) / **Multiband/Uniform 1,0672**
(**über** dem Gate). Die Metrik **diskriminiert jetzt real**: sie sieht, dass
Multiband an der Stützgrenze mehr Krümmung einträgt als Uniform, während Raw
sauber bleibt. Die Auswahl bleibt `drizzle_uniform` (Raw fällt schon am
`bg_RMS`-Gate 1,0886), aber das §15.3.2-Seam-Gate ist von „inert ≈ 1" zu einem
echten Mandatory-Safety-Kriterium geworden. `validation_config_hash`
`f1bf5607…` real bestätigt.

**A — Seam-Form: morphologisches Schließen der Stützmaske.**
`boundary_seam_score` leitet den Interior-Edge-Locus jetzt aus einer
**geschlossenen** Kopie der Stützmaske ab (`close_support_mask` = Dilatation
dann Erosion, quadratisches SE, `kMultibandValidationSeamMaskCloseRadius = 1`,
separable min/max). Verstreute Ein-Pixel-Dropouts (bis 2·Radius breit)
verschwinden, die wahre äußere Grenze und zusammenhängende Löcher bleiben. Der
Laplace wird **weiterhin auf dem Originalfeld** mit der
On-Support-Stencil-Anforderung ausgewertet (Schließen wählt nur den Locus, es
füllt keine Daten). Die Radius-Konstante ist in `validation_config_hash`
aufgenommen (neuer Wert
`f1bf5607a94e245404ad6b30d547aa6762e9d0e7e9c44644be636ee381fd343f`, im
Unit-Test als Literal gepinnt, damit eine stille Regression nicht am
Runner-Vergleich gegen den frisch berechneten Wert vorbeirutscht). Der
Charakterisierungstest ist von „Gate ist blind" (`|s_m/s_u − 1| < 0,05`) auf
die **positive** Aussage gedreht: verstreute Dropouts lassen Uniform/Raw
gleich (`|s_r/s_u − 1| < 0,02`), ein echter +200-Schritt an der
zusammenhängenden Grenze treibt `s_m/s_u` deutlich über 1,05 — d. h. das Gate
sieht den echten Sprung und ignoriert das Dropout-Rauschen. Der bestehende
Seam-Sprung-Test bleibt grün.

**B — §16.1-Auslieferungslayout.**
`fuse_multiband_store_to_image` bekam einen opt-in-Parameter
`MultibandCandidateChannels *channels_out`: im **selben** Fusionsdurchlauf
werden die drei Kandidaten zusätzlich in **voller Kanalauflösung** (nicht nur
Luma) erfasst — Uniform = `U.{R,G,B|L}.value`, Raw = `R.*.value`, Multiband =
das fusionierte `X_out` pro Kanal. Der Runner schreibt danach nach `outputs/`:
- `forward_drizzle_raw_{L|R,G,B}.fit` — **unveränderliche Raw-Baseline, immer**
  geschrieben, unabhängig davon welcher Kandidat gewinnt;
- `reconstructed_{L|R,G,B}.fit` — der **gewählte** Kandidat;
- bei `diagnostics.level = full` zusätzlich
  `forward_drizzle_uniform_*` + `forward_drizzle_multiband_*`.
`forward_drizzle.json` `outputs[]` listet jede Datei mit `path`/`size`/
`sha256`; `checkpoint["outputs"]` hält dieselbe Menge (informativ — die
`MULTIBAND`-Phase läuft bei Resume **immer** neu und vertraut keinem früheren
Output, daher kein Resume-Größe/Hash-Guard wie bei den Geometrie-Predecessors).
`artifacts/reconstruction_multiband.fits` bleibt als **internes** Artefakt
(Fuse-Commit-Ziel, Debug). **Bewusst NICHT geschrieben: `outputs/stacked.fits`
/ `outputs/stacked_rgb.fits`** — das sind die **kanonischen** Downstream-
Eingänge (Astrometrie, BGE, PCC, HMS; `runner_pipeline.cpp`,
`runner_phase_post_stack_output.cpp`, `web_backend`), die die §17.4-Photometrie-
Rücknahme (`scale_r/g/b`, Pedestal) erwarten. Diese Rücknahme ist M10-Cutover-
Arbeit; eine Datei mit kanonischem Namen im falschen (normalisiert-linearen)
Raum wäre eine Falle. `reconstructed_*` / `forward_drizzle_raw_*` sind **neue**
Namen ohne Consumer und daher unbedenklich. Alle geschriebenen `outputs/`-
Dateien liegen im normalisiert-linearen Arbeitsraum, `output_scale`-Geometrie.
Tests: MONO-Runner end-to-end (zwei Dateien bei `summary`, Checkpoint,
`outputs[]` mit verifizierten Hashes, `stacked*` **nicht** vorhanden),
OSC-`channels_out` bit-exakt gegen die Nicht-Streaming-Referenz. **Echtdaten
(M42-Resume, 2026-09-07):** OSC-Pfad real bestätigt — sechs Dateien
`forward_drizzle_raw_{R,G,B}.fit` + `reconstructed_{R,G,B}.fit` (je 33,8 MB,
3858×2190 float), **alle sechs `outputs[]`-Einträge: `sha256` + `size` = Datei
auf Platte**, kein `stacked*`, `final_image = outputs/reconstructed_R.fit`,
`outputs_count = 6`.

**C — §16.4-Pflichtdiagnostik in `forward_drizzle.json`.** Neu befüllt:
`geometry` (Quell-/Canvas-/Rekonstruktionsmaße, `internal/output_scale`,
`kernel`, `pixfrac`), `clipping` (`pixel_channel_evaluations`/`_rejected`,
`candidate_contributions_clipped`), `local_warp`
(`local_model_samples_total`/`_discarded` + ausgeschlossene Frames), `pixels_supported`,
`acceleration` (`forward_drizzle_backend`, `cuda_fallback_reason`,
`workers_used`, `resolved_chunk_rows`), `resources` (`estimated_peak_bytes`,
`rss_baseline/peak/growth_kib` via `getrusage(RUSAGE_SELF).ru_maxrss`,
`memory_budget_mb`), `timing_seconds` (Wall-Clock je abgeschlossener Phase).
**Bewusst absent statt `{}`:** Per-Kernel-Timing, Retries, Backend-Chunkgrößen
der GPU — das gehört M7-Slice-2 (ein leeres `{}` würde als „gemessen, nichts
zu berichten" fehllesen, 30.43). **Echtdaten (M42-Resume):** alle sechs Blöcke
befüllt; `resources.rss_peak_kib = 4 527 156` (~4,32 GB) gegen
`estimated_peak_bytes` 4,23 GB und `memory_budget_mb` 4096 — die zusätzliche
Pro-Kanal-`chans`-Erfassung (3 Kandidaten × 3 Kanäle voll resident im
Fusionsdurchlauf) passt in die bestehende Hülle, Peak nur knapp über der
Schätzung, kein Blow-up.

**M6 ist abgeschlossen — Code + synthetisch (inkl. der synthetischen
MONO-Pfade: `[forward-runner]`-Fixture 32×32, MONO-Store-Round-Trip,
`fuse_multiband` MONO-Identitäten) + Echtdaten OSC: M31 (sternreich) und M42
(Nebel). Beide Echtdatensätze sind OSC — es gibt keinen realen MONO-Datensatz
auf dem Rechner; MONO ist ausschließlich synthetisch abgedeckt.** Hauptsuite
**491/492** (nur der vorbestehende `test_acceleration_backend.cpp:254`).
Der M42-Resume mit dem A+B+C-Binary hat die geschlossene Seam-Metrik (A,
`sample_count` 1 077 246 → 2 723, Ratios raw/uni 1,0053 vs. mb/uni 1,0672 —
diskriminiert real), die §16.1-OSC-Auslieferung (B, sechs Dateien,
`outputs[]`-Hashes = Platte) und die §16.4-Diagnostik (C, alle Blöcke, RSS im
Rahmen) alle bestätigt. **Offen bleiben nur die beiden nicht-Code-Fragen aus
30.46 als Plan-Anmerkungen** (sternbasierte Multiband-Evidenz;
5-%-`bg_RMS`-Gate vs. §12.4-Rauschen) — das aktuelle Verhalten ist in beiden
Fällen plankonform (§15.3 „Erwartete Auswahlverteilung").

---

<a id="historie-30-49"></a>

### 30.49 M7-Slice-2 begonnen: Auto-Chunking-Planer (§19.4) (2026-09-06)

Auf „danach beginne mit m7" wurde M7-Slice-2 mit dem risikoärmsten,
GPU-freien Baustein begonnen: `plan_cuda_chunking()` — reine Arithmetik, ohne
Device-Aufruf, unit-getestet. Aus einer Free-Memory-Zahl und einem
Working-Set-Schätzwert pro Ausgabe-Zeile bestimmt sie (§19.4):
Sicherheitsmarge (`reserve_fraction` von Free, mind. `reserve_floor_bytes` =
256 MiB — für Treiber/OpenCV), nutzbare Bytes, initiale Chunkhöhe (memory-
bound oder image-bound oder Config-Deckel), und die **Retry-Leiter**
(Halbierungen von der Chunkhöhe bis `min_chunk_rows = 1`). `feasible = false`,
wenn nicht mal eine Zeile passt ⇒ Aufrufer bleibt auf dem CPU-Referenzpfad.
Tests: Reserve/Fit/Deckel/Floor/Degenerate-Inputs/Retry-Leiter.

**Device-Probe (`.cu`).** `src/reconstruction/forward_drizzle_cuda_device.cu` —
nur unter `TILE_COMPILE_WITH_CUDA` kompiliert (CMake wie
`aqmh_reconstruction_cuda.cu`), gegen die CUDA-Runtime:
`forward_drizzle_cuda_device_memory()` = `cudaGetDeviceCount` +
`cudaMemGetInfo` (`{0,0}` bei keinem/unbrauchbarem Device, sticky error wird
geleert). Der CUDA-freie Build definiert dieselben zwei Funktionen in der
`.cpp` unter `#if !TILE_COMPILE_WITH_CUDA`. **`forward_drizzle_cuda_runtime_available()`
bleibt bewusst `false`** — ein vorhandenes Device ohne Kernel würde nur einen
CPU-Neustart pro Lauf kosten (die Persist-Ebene wirft weiterhin „not
implemented" ⇒ §19.4-Neustart). Auf dieser Maschine (GTX 1660 Ti) liefert der
Probe real `free ≤ total > 0`, verifiziert im Test; `plan_cuda_chunking` mit
dem echten Free-Wert ist `feasible`.

**Chunk-Treiber + Retry-Leiter (`run_cuda_chunked`, host-only).** Vor den
Kerneln verdrahtet, damit die §19.4-Leiter bewiesen ist, bevor real etwas auf
einem Device fehlschlagen kann. Läuft das Bild in Bändern von
`plan.chunk_rows` ab; wirft ein Chunk-Prozessor `CudaAllocFailure`, wird die
**aktuelle** Bandhöhe halbiert und dasselbe Band erneut versucht, bis
`min_chunk_rows`. Bleibt es auch dort erfolglos ⇒ `ForwardDrizzleCudaError`
(⇒ Aufrufer verwirft alle temporären CUDA-Stores und startet die ganze
FORWARD_DRIZZLE-Phase auf CPU neu, §19.4). Eine **nicht**-`CudaAllocFailure`-
Exception ist ein harter Fehler und propagiert unverändert (nicht in einen
Retry umgedeutet). Mit einem Mock-Prozessor getestet: Happy-Path
(lückenlose, das Bild exakt deckende Bänder), OOM-zweimal-dann-Erfolg
(100→50→25, Rest bei 25), Dauer-OOM am Floor ⇒ `ForwardDrizzleCudaError`,
harter Fehler propagiert, infeasible Plan wird vorab abgelehnt.

**Zwei echte Kernel + zwei §19.5-Paritätseinträge — der geometrische Kern.**
Tag `[forward-drizzle][cuda-parity]`, beide skippen sauber ohne CUDA-Device:

1. **`forward_drizzle_cuda_polygon_rect_area_batch()`** — `__device__`-**1:1-Port**
   von `polygon_rectangle_intersection_area` (Sutherland-Hodgman-Konvex-Clip
   gegen 4 achsenparallele Halbebenen + Shoelace, statische Schranke 8), ein
   Thread pro (Quad, Zelle)-Paar. Test: 6 Hand-Fälle (exakte Zelle,
   Verschiebung, disjunkt→0, Ecke, Parallelogramm, degenerierter Punkt) + 4000
   pseudozufällige rotiert-skalierte Droplets gegen Ganzzahl-Zellen. **3582/4006
   bit-identisch**, schlechtester relativer Unterschied **4,7·10⁻¹¹**, harte
   Schranke `5·10⁻¹⁰` relativ.

2. **`forward_drizzle_cuda_affine_leaf_corners_batch()`** — `__device__`-1:1-Port
   von `build_affine_leaf` + `to_internal`: das native-Source-Quadrat
   `[sx±h]×[sy±h]` durch eine 2×3-Affine gemappt, dann `×internal_scale`, 4
   CCW-Ecken je Sample. Test: 5000 Samples über eine realistische
   Sensor-Ausdehnung unter einer Affine mit Rotation + anisotroper Skala + Shear
   + Translation. **~57 % bit-identisch** (die 3-Term-Dot-Produkt-Affine wird von
   nvcc per FMA kontrahiert, von g++ nicht), harte Schranke `1·10⁻⁹` relativ
   (die Ecken speisen Flächen, die in `float32`-Ebenen landen).

Beide zusammen sind der geometrische Kern, den der Rasterisierer komponiert.
Hauptsuite **494/495**.

**Slice-2 Restplan** — die zwei geometrischen Kerne stehen; verbleibend: die
Rasterisierung selbst (§19.2 Stufen 3–7: frame-lokale Droplet-Akkumulation
mit diesem Flächenkern → pixel-major-Transpose → Pro-Pixel-Clipping →
Akzeptanzmaske → Profilakkumulation), frame-lokale
Atomics + feste Reduktion (§19.3), Verdrahtung von `run_cuda_chunked` +
`plan_cuda_chunking` in `persist_forward_drizzle_multiband` hinter einem
`runtime_available()`, das erst **mit** den Kerneln `true` wird, Paritätsmatrix
§19.5 (CPU↔CUDA bit-exakt bzw. dokumentierte Toleranz), Timing-/`acceleration`-
Per-Kernel-Felder in `forward_drizzle.json`. À-trous bleibt CPU (§19.2).

**Offene Entwurfsfrage für den Rasterisierer:** die CPU-Referenz akkumuliert
je Zelle in fester Reihenfolge (Source-Pixel-major: `sy` außen, `sx` innen,
dann Zellen `y`/`x`). Ein naiver Scatter-`atomicAdd` von vielen Threads ist
reihenfolge-nichtdeterministisch ⇒ nicht bit-exakt. §19.3 erlaubt: (a)
frame-lokale Atomics in isoliertem Buffer + feste profilweise Reduktion, (b)
deterministische Sort-/Segmented-Reduction, (c) getestete Toleranz. Zu
entscheiden: Gather-Kernel (Thread pro Ausgabezelle, deterministische
Source-Schleife — matcht CPU-Reihenfolge exakt, braucht die inverse Abbildung
+ begrenzte Suchfenster) vs. Scatter + Toleranz. Das ist ein eigener
fokussierter Block; §19.1 verlangt eine vollständig getestete CPU-Referenz als
Voraussetzung, die mit M6 jetzt vorliegt.

---

<a id="historie-30-50"></a>

### 30.50 M6-Ressourcenabnahme (§11.13): Kandidaten-Spooling, Vor-Plan, phasen-lokale RSS (2026-09-07)

Die fünf §11.13-Punkte für die MULTIBAND-Phase (Fusion/Validierung/Export)
umgesetzt; M6 damit von „funktional implementiert" auf „Ressourcenvertrag
erfüllt" gehoben.

1. **Kein stilles Anheben eines expliziten Budgets.** `fuse_multiband_store_to_image`
   nutzte `budget = max(memory_budget_mb, 256)` — ein explizit kleines Budget
   wurde stillschweigend auf 256 MiB gehoben. Jetzt: `memory_budget_mb == 0`
   ⇒ interner 256-MiB-Boden (unset); jeder Wert `> 0` ist ein **expliziter**
   Auftrag und wird wörtlich übernommen.

2. **Geteilter, überlaufsicherer Working-Set-Vor-Plan.**
   `plan_multiband_fusion_memory(W,H,nch,levels,chunk,halo, with_luma,
   with_spool, budget_bytes)` → `MultibandFusionMemoryPlan` mit sechs
   Einzeltermen (`final_image`, `stripe_working`, `candidate_luma`,
   `spool_stripe`, `delivery_readback`, `margin`), sättigende `sat_add`/`sat_mul`,
   `fits = estimated_peak_bytes <= budget_bytes`. Der Plan wird **vor** der
   ersten großen Allokation berechnet; `!fits` ⇒ `throw "MULTIBAND_MEMORY_BUDGET"`.
   Zu diesem Zeitpunkt wurde nur der (bereits committete) Store gelesen — die
   vorherige gültige Generation und alle bestehenden Ausgaben bleiben unberührt
   (fail-closed).

3. **Kandidatenkanäle streifenweise in temporäre Stores.** Die 3·nch vollen
   Kandidatenebenen (`MultibandCandidateChannels`) wurden komplett resident
   gehalten (~9·N·float bei OSC). Ersetzt durch `MultibandCandidateSpool`: je
   (Kandidat, Kanal) ein Append-Stream in ein Scratch-Verzeichnis; die
   Streifenschleife schreibt exakt die `core`-Zeilen in streng steigender
   y-Reihenfolge ⇒ jede Datei ist eine zeilen-majore `W·H`-Ebene.
   `read_candidate_spool_plane` liest je eine Ebene zurück (Größenprüfung
   `== N·4`). Der Runner exportiert nur die gelieferten Kandidaten (Raw-Basis +
   ausgewählter + bei `diagnostics.level=full` die Kontrollen), Ebene für Ebene
   → FITS, dann freigegeben (Spitzen-Liefer-RAM = eine Ebene). Spool-Dir unter
   `artifacts/multiband_candidate_spool`, nach erfolgreicher Lieferung entfernt.

4. **Getrennte Ausweisung von Schätzung und gemessener RSS.**
   `read_maxrss_kb()` (`ru_maxrss`) ist ein Prozess-**Lebenszeit**-Maximum; die
   Differenz zweier solcher Maxima ist kein phasen-lokales Wachstum. `resources`
   in `forward_drizzle.json` jetzt:
   - `multiband_estimated_working_set_bytes` + `multiband_working_set_breakdown`
     (die sechs Terme) + `multiband_working_set_fits_budget`;
   - `rss_process_peak_kib` (`VmHWM`), `rss_maxrss_kib` (`ru_maxrss`) — beide
     ausdrücklich als Lebenszeit-Maxima benannt;
   - `multiband_phase_rss_start_kib` / `_now_kib` / `_growth_kib` — `VmRSS` aus
     `/proc/self/status`, gemessen gegen den Wert bei MULTIBAND-Phasenbeginn;
   - `multiband_phase_rss_envelope_kib` = `budget·1024·1.05 + 256·1024` (§11.11)
     und `multiband_phase_rss_within_envelope` (bool);
   - `phase_rss` — pro Phase `{start,end,growth}` aus den begin/end-Lambdas.

5. **Regressionsfixtures** (`[drizzle-store]`, `[forward-runner]`):
   - `plan-11.13 multiband working-set planner` — exakte Terme
     (`final_image == nch·N·4`, `candidate_luma == N·(13+levels·4)`,
     `estimated == Σ der sechs Terme`), Monotonie in N, OSC > MONO,
     Fail-Closed-Grenze (`estimated_peak_bytes-1` ⇒ `!fits`, `== estimated` ⇒
     `fits`), Spool-Term nur bei `with_spool`.
   - `plan-11.13(1)(3): explizit kleines Budget` — 200×200-Fixture, `mb=1` ⇒
     `MULTIBAND_MEMORY_BUDGET`-Wurf, `!plan.fits`, `budget_bytes == 1 MiB`,
     kein Zielbild geschrieben, `current.json` unverändert, Store weiter
     `usable`; danach `mb=256` ⇒ `NOTHROW`, `spool.populated`, Zielbild da.
   - `plan-11.13(5): injizierter Spool-Fehler` — nicht existierendes Spool-Dir
     ⇒ `FUSE_STORE_SPOOL_DIR_MISSING`, kein Zielbild, `current.json`
     unverändert, Store `usable`.
   - `plan-11.13(5): Chunkhöhen-Variation` — Fusion bei `chunk ∈ {1,3,H}` in
     getrennte Spools; identische `selected`-Kandidat, identische
     `applicable`-Bitsets aller Metriken/Kandidaten, identische SHA-256 jeder
     gespoolten Ebene über alle Chunkhöhen.
   - `test_runner_forward_drizzle` — `resources.rss_process_peak_kib > 0`,
     `multiband_estimated_working_set_bytes > 0`,
     `multiband_working_set_fits_budget == true`,
     `multiband_phase_rss_within_envelope == true`, Breakdown/Growth-Felder
     vorhanden.

6. **Temp-Space-Vertrag (§11.11) und Spool-Aufräumen.**
   - `plan_multiband_fusion_memory` weist jetzt `spool_temp_bytes`
     (= 3·nch·N·4) aus; `fuse_multiband_store_to_image` verfeinert
     `required_free_temp_bytes = spool_temp_bytes·1.20 + max(2 GiB,
     5 %·Kapazität)` mit der realen Kapazität des Spool-Dateisystems
     (`std::filesystem::space`), setzt `available_temp_bytes`/`temp_space_ok`
     und wirft fail-closed `MULTIBAND_TEMP_SPACE` **vor** der ersten großen
     Allokation, wenn der freie Platz nicht reicht. Die Spool-Dir-Prüfung
     wanderte in dasselbe Vor-Allokations-Fenster.
   - Runner: `SpoolGuard` (RAII) entfernt `artifacts/multiband_candidate_spool`
     bei **jedem** Verlassen der MULTIBAND-Phase — Erfolg wie Wurf —, damit ein
     fehlgeschlagener Lauf keine bis zu 3·nch vollen `.f32`-Ebenen auf dem
     Temp-Dateisystem hinterlässt. `resources` trägt `multiband_spool_temp_bytes`,
     `multiband_required_free_temp_bytes`, `multiband_available_temp_bytes`,
     `multiband_temp_space_ok`.
   - `/proc/self/status`-Parser bricht am ersten Token-Ende nach der Zahl ab
     (statt alle Ziffern der Zeile zu sammeln).
   - `test_runner_forward_drizzle` prüft `multiband_temp_space_ok` und dass die
     Spool-Dir nach dem Lauf verschwunden ist; der Planer-Test prüft
     `spool_temp_bytes` und die `required_free_temp`-Untergrenze.

Verifikation: Code + synthetisch (Planer-Arithmetik bit-genau, Spool-Round-Trip
bit-exakt gegen die nicht-streamende `fuse_multiband`-Referenz, Chunk-Invarianz
der Auswahl, Fail-Closed für RAM- und Temp-Budget, Commit-Vertrag bei
injiziertem Fehler, Spool-Aufräumen, Runner-Integrationstest mit den neuen
`resources`-Feldern). Kein realer Reconstruction-Lauf angestoßen (der Plan
erteilt keinen Run-Auftrag); Hauptsuite **498/499** (die eine Abweichung ist der
vorbestehende, unabhängige `test_acceleration_backend.cpp:254`
Legacy-AQMH-OpenCV-CUDA-Fall).

---

<a id="historie-30-51"></a>

### 30.51 M7: Paritäts-Fehlergrenzen fixiert (§19.5.1) + deterministischer Vorwärts-Listenrasterisierer als CPU-Referenz (2026-09-07)

**§19.5.1 (Plan) — Fehlergrenzen der Paritätsmatrix festgeschrieben.** Vor
jeder Rasterisierer-Zeile, weil §19.6 nachträgliches Erweitern verbietet.
Tabelle mit Spalten CPU↔CUDA / Wiederholung / Chunkvariation: diskrete
Entscheidungen bit-exakt; Profilfelder `D_*`/`X_out` rel. ≤ `1·10⁻⁹` +
abs. ≤ `1·10⁻⁶`·Median; nahezu nullwertige Felder nur absolut; `weight_sum`/
`coverage`/`n_eff` rel. ≤ `1·10⁻⁹`; Aperturflux je Stern rel. ≤ `1·10⁻³`
(weit unter dem 0,5 %-Produktgate); Zentroid ≤ `2·10⁻³` px; Wiederholung und
Chunkvariation überall bit-exakt. Die `1·10⁻⁹`-Feldgrenze ist die
Fortpflanzung der beiden gemessenen Geometriekern-Abweichungen, nicht die
Kern-Toleranz selbst.

**`DrizzleAreaSink` um `int leaf` erweitert.** Der Sink liefert jetzt
`(sx, sy, channel, leaf, index, area)`; `leaf` ist die 0-basierte Ordnung des
transformierten Quellpixel-Leafs innerhalb des `(sx,sy)`-Samples (immer 0 im
affinen Pfad, 0..n-1 bei subdividiertem lokalem Warp) und Teil des
§19.6-Schlüssels. Vier Aufrufstellen mechanisch nachgezogen
(`forward_drizzle.cpp` ×2, `sampling_geometry.cpp` ×2).

**`forward_drizzle_contrib_list.{hpp,cpp}` (neu) — CPU-Referenz für §19.6.**
- `DrizzleContribKey` = `(frame_order, channel, target_y, target_x, source_y,
  source_x, leaf_order)`; `DrizzleContrib` = Key + `area` (k, immer > 0) +
  `value` (v, immer endlich). `contrib_key_less` = kanonische Ordnung in
  Plan-Feldreihenfolge.
- `build_uniform_contrib_list(plan, source_of, cfg, y_begin, rows, sub,
  mem_budget_bytes)`: Zwei-Pass (Vorzählen mit Endlich-v-Filter → exakt
  reservieren → materialisieren), überlaufsichere Zählung, dann **eine**
  kanonische Sortierung. `records_bytes > mem_budget_bytes` ⇒
  `DRIZZLE_CONTRIB_LIST_BUDGET` (Aufrufer halbiert Chunkhöhe gemäß §19.4).
- `reduce_uniform_contrib_list`: pro Segment `(frame, channel, Zielzelle)` ein
  Thread, Double-Akkumulatoren `A=Σk·v`, `B=Σk` in Datensatzreihenfolge, dann
  `wx += A`, `w += B`, `w2 += B·B`. Kein paralleler Reduktionsbaum.
- **Bit-Exaktheit per Konstruktion:** für eine feste Zielzelle ist die
  Streaming-Emissionsreihenfolge von `rasterize_drizzle_stripe` genau
  `(source_y, source_x, leaf_order)` aufsteigend = die kanonische
  Schlüsselordnung innerhalb eines Segments; Segmente werden in
  Frame-dann-Zell-Reihenfolge besucht = die Streaming-`wx += A`-Akkumulation
  über Frames.

**Produktionsform `accumulate_uniform_by_frame` (Advisor-Hinweis).** Da
`frame_order` der führende Schlüsselteil ist, überspannt kein Segment zwei
Frames: die Liste wird **frameweise** gebaut/sortiert/reduziert und in
gemeinsame `wx`/`w`/`w2` akkumuliert — bit-identisch zum
Gesamt-Streifen-`build+sort+reduce` (dieselben `wx[cell] += A_segment` in
derselben Frame-dann-Zell-Reihenfolge), aber Spitzenspeicher = ein Frame
statt aller Frames. Größenordnung an M42-Realgeometrie (7716×4380 intern,
40 Frames, Chunk 64): Gesamt-Streifen ≈ 12,6 M Records ≈ 0,6 GB/Streifen (auf
6 GiB knapp, mit allen Profilen unhaltbar); frameweise ≈ 15 MB/Frame. Die
`q`-Werte für Raw/Detail/Alpha kommen bei der Reduktion aus den Q-Maps über
`(source_y, source_x)`, nicht je Record gespeichert — `DrizzleContrib` bleibt
`area + value`. `build_uniform_contrib_list` (Gesamtform) bleibt als
Debug-/Spezifikations-Artefakt.

**§19.5.1 verfeinert (Advisor-Hinweis):** im normalisierten linearen Raum ist
`X_out` ≪ 1, also wirkt `rel. ≤ 1·10⁻⁹` mit `max(1,·)` effektiv absolut — für
`X_out` fest genug, für die à-trous-Detailbänder zu locker. Neue Zeile: `D_j`
relativ zur je Band gemessenen robusten Skala `s_j = p99.5(|D_j|)` mit
abs. Boden `1·10⁻⁶·s_j`; die Methode ist jetzt festgeschrieben, `s_j` wird
beim Matrixlauf aus Banddaten bestimmt und literal gepinnt (datengetriebene
Instanziierung, kein nachträgliches Erweitern). Der tote `abs. ≤ 1·10⁻⁶·Median`
Zusatz entfernt.

**Tests `[forward-drizzle][contrib-list]` (3 Fälle, ~91 000 Assertions):**
1. bit-identisch zur inline gespiegelten Streaming-Uniform-Referenz
   (`wx`/`w`/`w2`, exakter `double`-Vergleich) über {affin + Subpixel +
   Rotation}-Frames × {MONO; OSC in **allen vier** Bayer-Pattern} ×
   {mit/ohne lokalen Warp} × {innen; an den Canvas-Rändern geclippt} — plus
   `records.size() == predicted_count`, `std::is_sorted` nach `contrib_key_less`,
   jeder Record `area > 0` und `isfinite(value)`, und
   `accumulate_uniform_by_frame` bit-identisch zur Gesamtform.
2. Canvas in drei ungleiche Streifen `{[0,5),[5,12),[12,16)}` zerlegt →
   zusammengesetzte Akkumulatoren bit-identisch zum Ein-Streifen-Ergebnis
   (§19.6 „unterschiedliche Chunkhöhen → identische Profile" auf Referenzebene).
3. `mem_budget_bytes=64` ⇒ `DRIZZLE_CONTRIB_LIST_BUDGET` vor der
   Materialisierung; großzügiges Budget ⇒ `NOTHROW`.

Hauptsuite **501/502** (die eine Abweichung bleibt der vorbestehende,
unabhängige `test_acceleration_backend.cpp:254`). Kein realer Lauf.

**Noch offen für M7** (§19.6/§19.5): Raw-/Detailprofil- und
Alpha-Beitragslisten (derselbe Schlüssel, zusätzliche `q`-Werte je Record),
Clipping mit gemeinsamer Akzeptanzmaske und fester Framefolge-Profilreduktion,
`.cu`-Kernel, die die zwei Geometriekerne zu dieser Listenform komponieren,
`run_cuda_chunked`/`plan_cuda_chunking` in den produktiven Storepfad,
`forward_drizzle_cuda_runtime_available()` erst mit bestandener §19.5-Matrix
auf `true`, Per-Kernel-Timing in `forward_drizzle.json`.

---

<a id="historie-30-52"></a>

### 30.52 M7: CPU-Referenz für den vollständigen §19.6-Pfad — Clipping/Profile faktorisiert + Raw/Detail/Alpha-Beitragslisten (2026-09-07)

Schließt die zwei **algorithmischen** M7-Lücken auf der CPU-Referenzseite
(die `.cu`-Kernel + produktive Verdrahtung + native §19.5-Matrix bleiben).

**Schnitt A — reiner Refactor (berührt M6-verifizierten Produktionscode).**
Der Pro-`(Kanal, Zielzelle)`-Block „Clipping → 4 Profile → Alpha" (vormals
inline in `stream_forward_drizzle_uniform_and_raw`, ~80 Zeilen) ist als
`reduce_pixel_profiles(pixel, DrizzleProfileReduceConfig, g_eff_for,
reg_by_source, gi, uniform/raw/fine/medium-Plane*, ac_sep/art/reg*, diag&)`
herausgezogen. `stream_...` ruft ihn jetzt in derselben `for c / for i`-Schleife
auf. Verhalten unverändert: Hauptsuite **501/502** vor und nach dem Schnitt,
alle `[forward-drizzle]`/`[drizzle-store]`/`multiband*`/`[output-scale]`-Fälle
grün. `candidates`-Slice ist bereits frame-geordnet (Push in
`prepared.frames`-Reihenfolge) → §19.6 Schritt 4 „feste Framefolge".

**Schnitt B — `accumulate_pair_by_frame` (neu, in `forward_drizzle_contrib_list`).**
Ein Streifen, voller Pfad Uniform(geklippt)+Raw+Fine+Medium+Alpha über die
deterministische frameweise Beitragsliste:
- je Frame: `build_frame_records` → kanonische Sortierung → Segmentreduktion je
  `(Kanal, Zielzelle)`: `A=Σk·v`, `B=Σk` **und** die Q-K-Mittel `QA/QA0/QA1/QAA`
  = `Σ k·fold(qv)` mit `fold(qv)=(isfinite∧>0)?qv:0` **je Record in
  Recordreihenfolge** aus den Frame-Q-Maps über `(source_y, source_x)` gelesen
  (nicht je Record gespeichert); `QAF += k` wenn Artefakt-Sample endlich.
- danach je Segment mit `B>0` ein `ClipCandidate{ f.source_index /* nicht fo!,
  für Tie-Break + g_eff + reg_by_source */, A/B, B, QA/B…, QAF>0 }` in den
  frame-major `cand[c][i*frame_count + counts++]`-Puffer.
- `reduce_pixel_profiles` (identisch mit dem Streaming-Pfad) je `(c, Zelle)`.
- Spitzenspeicher: ein Frame Records + der `ClipCandidate`-Streifenpuffer,
  nicht die ganze Streifen-Liste. Der flache `ClipCandidate`-Puffer
  (`channels·W·rows·frame_count·sizeof`) begrenzt `rows` **nicht** selbst (anders
  als `stream_...` über `plan_drizzle_memory`) — die Funktion ist ausdrücklich
  **streifen-scoped**: der Aufrufer wählt `rows` (bei der produktiven
  Verdrahtung aus `plan_cuda_chunking`/dem §11.13-Vorplan). `mem_budget_bytes`
  begrenzt Recordvektor **und** Kandidatenpuffer, überlaufsicher, Wurf
  `DRIZZLE_CONTRIB_LIST_BUDGET` **vor** der Allokation.

**Bit-Exaktheit per Konstruktion:** die Segment-/Recordreihenfolge entspricht
der Streaming-Emissionsreihenfolge je Zelle; `reduce_pixel_profiles` ist
buchstäblich dieselbe Funktion. `frame_index` im Kandidat ist `source_index`
(für den Clip-Tie-Break §11.8 Schritt 3 und `g_eff`/`reg_by_source`), die
Push-Reihenfolge ist `fo` (prepared-frame) — beide getrennt und korrekt.

**Tests `[forward-drizzle][contrib-list]` (+3 Fälle, ~346k Assertions gesamt):**
1. `accumulate_pair_by_frame` bit-identisch zu
   `compute_forward_drizzle_uniform_and_raw` — `value`/`weight_sum`/`n_eff`/
   `support` aller vier Profile (exakter `float`-Vergleich, NaN-behandelt),
   `a_separation`/`a_artifact`/`a_registration` + Support, **und alle drei
   `clipping`-Zähler** — über {MONO; OSC in allen 4 Bayer-Pattern} × {±lokaler
   Warp} × {innen; randgeclippt}, mit **aktivem Clipping**
   (`candidate_contributions_clipped > 0`, Frame 3 heller Ausreißer, Frame 2
   ohne Artefakt-Map).
2. **Streifenzerlegung** `{[0,5),[5,12),[12,16)}` → zusammengesetzte Profile,
   Alpha-Maps **und Clipping-Zähler** bit-identisch zum Ein-Streifen-Ergebnis
   (Chunk-Invarianz der Clip-Entscheidungen — §19.5.1-Nulltoleranzzeile).
3. `mem_budget_bytes=512` ⇒ `DRIZZLE_CONTRIB_LIST_BUDGET` vor der
   Kandidatenpuffer-Allokation; großzügiges Budget ⇒ `NOTHROW`.

Hauptsuite **504/505** (die 1 Abweichung bleibt `test_acceleration_backend.cpp:254`).

**Noch offen für M7:** die `.cu`-Kernel (frame-lokale Droplet-Akkumulation mit
den zwei Geometriekernen → Segmentreduktion → Kandidat → Clip → Profil),
`accumulate_pair_by_frame`/`accumulate_uniform_by_frame` +
`run_cuda_chunked`/`plan_cuda_chunking` in `persist_forward_drizzle_multiband`
verdrahten, `forward_drizzle_cuda_runtime_available()` erst mit bestandener
nativer §19.5-Matrix (inkl. `s_j`-Pinning je Detailband, §19.5.1) auf `true`,
Per-Kernel-Timing in `forward_drizzle.json`.

---

<a id="historie-30-53"></a>

### 30.53 M7: FP-Kontraktionspolitik fixiert + affiner CUDA-Rasterisierer, bit-identisch auf echter Hardware (2026-09-07)

**FP-Kontraktion (§19.6 „FMA-/Compilerpolitik explizit fixieren").** Getestete
Hypothese: `-ffp-contract=off` (CPU) + `--fmad=false` (CUDA) auf dem
Referenzpfad ⇒ keine Fusion von `a*b+c` auf beiden Seiten ⇒ die zwei
Geometriekerne werden **100 % bit-identisch** (vorher 3582/4006 bzw. ~57 %).
In `CMakeLists.txt` als `set_source_files_properties` für
`forward_drizzle.cpp`, `forward_drizzle_contrib_list.cpp`,
`forward_drizzle_cuda.cpp` + die beiden Paritätstest-TUs (die die
CPU-Referenz inline nachbilden); `--fmad=false` für
`forward_drizzle_cuda_device.cu`. **Korrektheitstragend, kein
Optimierungsschalter** — ein neuer TU auf dem Pfad ohne diesen Eintrag
regressiert die Bit-Exaktheit still (Kommentar an Ort und Stelle).
Die alten `[cuda-parity]`-Toleranztests (`5e-10`/`1e-9` relativ) sind auf
`REQUIRE(cpu == gpu)` verschärft.

**`k_affine_frame_contribs` (neu, `.cu`) — der affine Droplet-Rasterisierer.**
1:1-Port von `build_affine_leaf` + der `rasterize_drizzle_stripe`-BBox/Flächen-
Schleife: ein Thread je Quellpixel des Streifenbands, affine Leaf-Ecken,
BBox-Clamp `[0,W)×[y_begin,y_begin+rows)`, je Zelle `d_polygon_rect_area`,
`k>0` ⇒ Record an dichter atomarer Position (Reihenfolge beliebig — der Host
sortiert nach dem eindeutigen kanonischen Schlüssel, daher deterministisch).
Host-Wrapper `forward_drizzle_cuda_affine_frame_contributions` kopiert nur das
**bandlokale** Quell-Sub-Bild (nicht das ganze Bild je Frame), führt eine
Per-Pixel-Zellobergrenze (`max_cells_per_pixel`, Default 32) und eine globale
Kapazitätsgrenze; jede Überschreitung / jeder CUDA-Fehler ⇒ `false` ⇒
CPU-Fallback für den ganzen Streifen (§19.4, kein CPU/CUDA-Mix im Commit).

**`accumulate_pair_by_frame_cuda` (neu).** `accumulate_pair_by_frame` in
`accumulate_pair_impl(..., PairFrameRecordProducer)` faktorisiert; nur die
Recordproduktion unterscheidet CPU (`build_frame_records`) und CUDA
(`cuda_pair_producer` → Device-Rasterisierer, Bandberechnung per
`invert_affine_2x3` exakt wie die CPU). Sortierung, Q-Fold, `reduce_pixel_profiles`,
Ergebnisaufbau sind **geteilt** ⇒ bit-identisch, wenn die Records passen.
Lokale-Warp-Frames: `ForwardDrizzleCudaError` (harte Ablehnung, kein
Durchrutschen).

**Native Parität `[forward-drizzle][contrib-list][cuda-parity]` (GTX 1660 Ti):**
`accumulate_pair_by_frame_cuda` bit-identisch zu `accumulate_pair_by_frame` —
`value`/`weight_sum`/`n_eff`/`support` aller vier Profile, `a_*` + Support,
**und alle drei `clipping`-Zähler** (`evaluations`/`rejected`/
`candidate_contributions_clipped`, letzterer > 0) — über {MONO; OSC alle 4
Bayer} × {innen; randgeclippt}, 10 Varianten, ~120k Assertions. Plus:
Lokale-Warp-Frame ⇒ `ForwardDrizzleCudaError`.

Hauptsuite **506/507** (die 1 Abweichung bleibt `test_acceleration_backend.cpp:254`).

**Noch offen für M7 — die §19.5-Matrix ist konstruktiv unvollständig:**
- **Kernel-Fähigkeitslücke:** lokale Warps (Subdivision) laufen NICHT auf der
  GPU. §19.5 nennt „lokale Warps" ausdrücklich → die Matrix ist erst mit dem
  Device-Subdivisionspfad vollständig. 10 bestandene affine Varianten sind
  nicht „§19.5-Matrix bestanden".
- Produktive Verdrahtung von `accumulate_pair_by_frame_cuda` +
  `run_cuda_chunked`/`plan_cuda_chunking` in `persist_forward_drizzle_multiband`
  (der bandlokale `src_buf` wird noch je Frame gebaut — für die Verdrahtung zu
  hoisten; eine echte Zeitmessung erst danach).
- `forward_drizzle_cuda_runtime_available()` bleibt `false` bis die
  vollständige Matrix (inkl. lokaler Warps + `s_j`-Pinning je Detailband)
  besteht. Der Flag-Flip ist der letzte Schritt, nicht einer zum Test-Aktivieren.
- Per-Kernel-Timing in `forward_drizzle.json` (hängt an der Verdrahtung).

### 30.54 M7: lokale Warps sind CPU-only (§19.6.1) + produktive CUDA-Verdrahtung + Timing (2026-09-07)

**Vertragsentscheidung §19.6.1 — lokale Warps bleiben CPU-only, solange §19.5.1 gilt.**
`smooth_local_basis` wertet die 4×4-Gauß-Basis über `std::exp` aus; `expf`
glibc vs. CUDA-libdevice unterscheiden sich ~1 ULP. Dieses Ergebnis fließt in
**diskrete** Entscheidungen (Fixpunkt-Konvergenz `step < tol_px`,
`out_of_bounds`, akzeptierte Leaf-Menge) — §19.6/§19.5.1 lassen dafür keine
Toleranz zu, also gibt es aktuell keinen zulässigen GPU-Pfad. Eine
Wiederaufnahme bräuchte eine §19.5.1-Änderung (eine vorab festgelegte Toleranz
für das *kontinuierliche* Verschiebungsfeld plus Nachweis exakt gleicher
diskreter Entscheidungen), die §19.5.1 derzeit untersagt. Bis dahin: die
§19.5-Zeilen mit lokalen Warps sind CPU-Referenz-only (kein GPU-Vergleich, weil
es keinen GPU-Pfad gibt); die Matrix ist für den GPU-Teil vollständig, sobald
alle **affinen** Zeilen bestehen. Im Plan als §19.6.1 festgeschrieben.

**Produktive Verdrahtung in `persist_forward_drizzle_multiband`.** Vor jedem
CUDA-Versuch (nur bei `cuda.attempt && fault_after < 0`) drei Gates, alle in
der Funktion geprüft: kein Frame mit `has_smooth_local_model`, nicht Modus 2/1,
ein nutzbares Device (`forward_drizzle_cuda_device_memory().free_bytes > 0`).
Verletzt eines das Gate ⇒ `ForwardDrizzleCudaError` **vor** dem
`StoreWriter` ⇒ der Aufrufer
(`persist_multiband_store_from_predecessors`) fängt es, setzt
`cuda_fallback_reason` und baut den ganzen Store auf dem CPU-Referenzpfad
(§19.4, kein CPU/CUDA-Mix; der „Neustart" ist hier ein No-Op-Relabel, da noch
keine Generation offen war). Bei erfüllten Gates: `plan_cuda_chunking`
(Working-Set-Schätzung je interner Zeile = Host-`ClipCandidate`-Puffer +
Device-Beitragsvektor + 8 Double-Akkumulatoren/Kanal) → `run_cuda_chunked`
treibt die interne Leinwand in gerätegroßen Bändern, jedes Band ein
`accumulate_pair_by_frame_cuda`-Lauf in **denselben** `writer.multiband_stripe`-
Sink. `DRIZZLE_CONTRIB_LIST_BUDGET` aus einem zu hohen Band wird in
`CudaAllocFailure` übersetzt (Band halbieren, §19.4-Leiter); jeder andere
`ForwardDrizzleCudaError` propagiert (CPU-Neustart).

**Store-Level-Parität `[drizzle-store][cuda-parity]` (GTX 1660 Ti):** ein via
CUDA-Streifenpfad gebauter Multiband-Store ist **byte-identisch** zum
CPU-Streaming-Build — alle Planes (`uniform`/`raw`/`fine`/`medium` ×
`value`/`weight_sum`/`n_eff`/`support` + die vier Alpha-Maps), plus identische
`clipping`-Zähler — über {MONO; OSC} bei subpixel-rotierten affinen Frames mit
engagiertem Clipping; zusätzlich byte-identisch zu einem Ganzleinwand-CPU-Build
(die 8-Zeilen-CUDA-Bänder erzeugen keinen eigenen Seam). `commit.json` und der
Generationsverzeichnisname (Zeitstempel/Zähler) sind wie bei den bestehenden
Chunk-Invarianz-Tests **nicht** Teil der Bit-Gleichheit; verglichen werden nur
die Plane-FITS. Neue Gates auch getestet: Lokale-Warp-Frame ⇒
`ForwardDrizzleCudaError`; Modus 2/1 ⇒ `ForwardDrizzleCudaError`. Der frühere
Slice-1-Test „attempt ohne Fault ⇒ sofortiger Wurf" wurde auf „CUDA-Pfad
committet einen bit-identischen Store" umgestellt (der Wurf-Zweig bleibt für
den Fall „kein Device").

**Timing.** `DrizzleStoreResult::cuda_timing` (`used`, `bands`,
`resolved_chunk_rows`, `min_chunk_rows`, `bytes_per_row`, `device_free_bytes`,
`stripe_seconds`, `total_seconds`); im Runner unter
`acceleration.cuda_stripe_path` in `forward_drizzle.json` emittiert (nur wenn
der Pfad tatsächlich lief, sonst `null`).

**`forward_drizzle_cuda_runtime_available()` bleibt `false` (in §30.54).** Die
Verdrahtung existiert und ist store-level bit-verifiziert, aber der Flag-Flip
wartet auf einen realen Großbild-Ressourcenlauf (braucht einen ausdrücklichen
Run-Auftrag). → In **§30.55** auf Run-Auftrag ausgeführt und die Probe-Form
aktiviert.

Hauptsuite **508/509** (die 1 Abweichung bleibt `test_acceleration_backend.cpp:254`).

**`forward_drizzle_cuda_runtime_available()` bleibt `false`** (bis §30.55).

**Noch offen für M7 (Stand §30.54):**
- GPU-Subdivision für lokale Warps: **ausgesetzt, solange §19.5.1 gilt**
  (§19.6.1). Ohne eine §19.5.1-Änderung kein GPU-Pfad; die §19.5-Matrix ist für
  den GPU-Teil mit den affinen Zeilen komplett.
- Realer Großbild-Lauf (M31/M42, affin) als Voraussetzung für den
  `runtime_available()`-Flip — braucht einen Run-Auftrag → in §30.55 erledigt.
- `s_j`-Pinning je Detailband (§19.5.1) beim Matrix-Lauf.
- `bytes_per_row` ist bewusst Host+Device-Summe (nicht max) — die aufgelöste
  Bandhöhe ist damit konservativ kleiner als reines VRAM erlaubte; der
  Host-`ClipCandidate`-Puffer wird separat durch eine absolute Obergrenze
  (`cfg.memory_budget_mb`, sonst 2 GiB) begrenzt, damit die §19.4-Halbierung
  überhaupt greifen kann.

---

### 30.55 M6/M7: reale M31/M42-Läufe — CUDA store-byte-identisch, §11.13 bestätigt, Runtime aktiviert (2026-09-07)

Auf ausdrücklichen Run-Auftrag ausgeführt. Binary `git=51d3e851 dirty`,
GTX 1660 Ti (6 GiB, ~3,7–4,4 GiB frei), CUDA 13.0. Alle Läufe frische
`reconstruct`-Vollläufe (40 Frames, `--max-frames 40`), Configs unterscheiden
sich **nur** in `runtime_limits.acceleration_backend` und
(top-level) `reconstruction.keep_profile_cache_after_run: true`.
Verglichen wird der **Plane-FITS-Digest** des committeten
`forward_drizzle_profiles/generation-*` (52 Planes: uniform/raw/fine/medium ×
value/weight_sum/n_eff/support + 4 Alpha-Maps); `commit.json` und
Generationsname (Zeitstempel) sind ausgenommen. Die Läufe lagen unter
`verify_m6m7/` (gitignoriert, Wegwerf-Artefakte).

**M7 — affiner CUDA-Pfad bit-identisch auf echten Daten (M31), zwei Geometrien:**
| Lauf | CPU Plane-Digest / `.fits` | CUDA Plane-Digest / `.fits` | CUDA `cuda_stripe_path` |
|---|---|---|---|
| `internal_scale=1` | `da60a400…` / `eb16598f…` | **gleich** | `bands=21, chunk_rows=106, stripe_s≈797` |
| `internal_scale=2, output_scale=2` (Produktions-Oversampling) | `0ead57b8…` / `35575fdc…` | **gleich** | `bands=92, chunk_rows=48, stripe_s≈1983` |

Real: 40 OSC-Frames, native Leinwand ≈ 5760×5664 (bei 2/2 interne Leinwand 2×,
≈ 96 MP Ausgabe), echte Triangle-Star-Registrierung (0 lokale Modelle, 40
affin), echte Q-Maps, aktives Robust-Clipping (`candidate_contributions_clipped`
99 M bei 1/1, 301 M bei 2/1). **Der 2/2-Lauf ist der belastbarste Nachweis:**
bei `internal_scale=2` überdeckt jedes 0,8-Droplet ≈ 1,6 interne Zellen, d. h.
der Sutherland-Hodgman-Mehrzellen-Clip läuft für praktisch jedes Droplet —
über 92 Device-Bänder byte-identisch. `forward_drizzle_backend=cuda`,
`cuda_fallback_reason=null` in beiden Läufen.

Durchsatz: CUDA-FORWARD_DRIZZLE ≈ 1,6× langsamer als CPU (1/1: ~13 vs ~8 min;
2/2: ~33 vs ~21 min) — host-gebundene deterministische Sortier-/Segment-
reduktion dominiert, GPU-Auslastung niedrig, kein Halbieren nötig. Bit-Exaktheit
ist der Vertrag, nicht der Durchsatz (§19.6). Nebenbefund: der CUDA-Bandpfad
hat einen **kleineren Host-RSS-Peak** (2/2: `rss_process_peak_kib` 3,6 GB CUDA
vs 14,7 GB CPU) — 48-Zeilen-Device-Bänder statt 229-Zeilen-CPU-Chunks.

**M7 — Modus 2/1 (Produktionsconfig, `internal_scale=2, output_scale=1`):** die
CUDA-Anfrage wird korrekt abgelehnt —
`cuda_fallback_reason: "forward_drizzle CUDA: mode 2/1 is CPU-only (no device 2x2 average)"`,
`backend=cpu`, Lauf sauber zu Ende. Auf der Pipeline **wie normal konfiguriert**
ist der Device-Pfad damit inert; ein Device-2×2-Downsample ist ein eigener
Slice (offen).

**M7 — lokale-Warp-Ablehnung auf echten Daten (M42, `internal_scale=1`, 1 Frame
mit `has_smooth_local_model`):**
`cuda_fallback_reason: "forward_drizzle CUDA: local-warp frame present; device path is affine-only (plan 19.6.1)"`,
`backend=cpu`. Der Fallback-Store ist **byte-identisch** (`d869f53b…`) zu einem
reinen CPU-Lauf derselben Daten; `reconstruction_multiband.fits` gleich
(`07f09104…`).

**CPU-Referenz unverändert durch die §30.52/§30.53-Refaktorierung.** Der
M31-Lauf in Produktionsconfig (2/1, CPU) liefert `reconstruction_multiband.fits`
`sha256=49ca284c…` — **byte-identisch** zum Vor-§30.54-Binary (`git=9bedc775`).
Das deckt die gesamte Kette ab: `reduce_pixel_profiles`-Extraktion aus dem
Streaming-Pfad, `-ffp-contract=off` auf vier TUs, `--fmad=false` auf dem `.cu` —
alles ein No-Op auf dem realen Produktionsergebnis. Breitere Regressionsfläche
als die CUDA-Parität selbst.

**M6 §11.13 — auf beiden realen Datensätzen bestätigt** (M31 2/1 und 1/1, M42
1/1): `multiband_working_set_fits_budget=true`, `multiband_temp_space_ok=true`,
`multiband_phase_rss_within_envelope=true` (MULTIBAND-RSS-Wachstum 17–128 MB),
`multiband_candidate_spool` nach dem Lauf entfernt, `phase_rss` je Phase
protokolliert.

**`forward_drizzle_cuda_runtime_available()` → aktiviert.** Von `return false` auf
die Probe-Form `forward_drizzle_cuda_device_memory().free_bytes > 0`. Jeder
Versuch bleibt in `persist_forward_drizzle_multiband` durch die drei Gates
(affin-only, nicht Modus 2/1, Device vorhanden) abgesichert; alles andere fällt
mit `cuda_fallback_reason` auf den CPU-Referenzpfad zurück. Zwei veraltete
Testannahmen angepasst: `test_drizzle_profile_store.cpp` („CUDA-Pfad bleibt
deaktiviert" → „aktiv genau bei vorhandenem Device") und
`test_runner_forward_drizzle.cpp` (`backend=="cpu"` → `cpu|cuda`, bei `cuda`
kein Fallback-Grund). Hauptsuite **508/509** (unverändert nur
`test_acceleration_backend.cpp:254`, ohne Bezug).

**Noch offen für M7:** Device-2×2-Downsample (Modus 2/1) als eigener Slice —
sonst ist der CUDA-Pfad auf der Produktionsconfig inert. `s_j`-Pinning je
Detailband (§19.5.1). Durchsatz-Optimierung (aktuell langsamer als CPU).

---

<a id="historie-30-56"></a>

### 30.56 M7: hybrider Pfad §19.6.2 — lokale-Warp-Geometrie auf CPU, Rasterisierung auf GPU (2026-09-07)

Statt lokale Warps auf dem Device ganz auszulassen (§19.6.1: `std::exp` in
`smooth_local_basis` macht CPU↔GPU-Bitidentität der diskreten Entscheidungen
unerreichbar), wird die **Geometrie** vollständig auf der CPU-Referenz
berechnet und nur die **Rasterisierung** auf die GPU gegeben. Plan §19.6.2
lässt diesen hybriden Backend-Pfad jetzt ausdrücklich zu (kein versteckter
Fehlerfallback).

**Gemeinsame Zell-Enumeration herausgezogen.** `rasterize_drizzle_stripe`
delegiert an `enumerate_drizzle_stripe_leaf_cells(plan, f, scale, pixfrac,
y_begin, rows, DrizzleLeafCellSink, sub)`: identische Quellzeilen-Band-
Ableitung, `sample_leaves`, `floor`/`ceil`-Bounding-Box je Leaf, aber statt der
Fläche werden je (Leaf, Zelle) die **vier exakten Ecken** und der Zell-Ursprung
emittiert (keine Vorfilterung auf Fläche — der Konsument entscheidet).
`rasterize_drizzle_stripe` ist danach ein dünner Wrapper, der pro Zelle
`polygon_rectangle_intersection_area` aufruft; Verhalten byte-identisch, alle
bestehenden Tests decken es transitiv ab. `DrizzleAreaSink` trägt jetzt den
`leaf`-Index (kanonischer §19.6-Schlüssel).

**`build_frame_records_hybrid_local`** (neben `build_frame_records`): für einen
lokale-Warp-Frame streamt `enumerate_drizzle_stripe_leaf_cells` (Leaf, Zelle)-
Arbeitspakete in **budgetierte Batches** (`max_batch_items`, Default 2^20). Je
Batch berechnet `forward_drizzle_cuda_polygon_rect_area_batch` die exakten
Polygon-Zell-Flächen auf dem Device (derselbe bit-exakte Kernel wie der affine
Pfad, `--fmad=false`, §30.49/§30.53). `leaf_order`, Quellindex und Kanal werden
host-seitig neben dem Batch getragen; `DrizzleContrib`-Records entstehen auf dem
Host, der Downstream (Sortierung nach `contrib_key_less`, Q-Faltung, Clipping,
`reduce_pixel_profiles`) ist unveränderter gemeinsamer Hostcode. Kein
CPU↔GPU-`exp`-Vergleich, keine neue Toleranz, §19.5.1 unverändert. Bei
Device-Alloc-Druck halbiert sich `batch_cap` bis zu einer Untergrenze (4096),
darunter `ForwardDrizzleCudaError` → §19.4-CPU-Neustart. Host-Budget
weiterhin über `DRIZZLE_CONTRIB_LIST_BUDGET` durchgesetzt (inkrementell statt
Vorzählung).

**Verdrahtung.** `cuda_pair_producer` verzweigt jetzt: `has_smooth_local_model`
→ Hybridpfad, sonst affiner Device-Rasterisierer. `accumulate_pair_by_frame_cuda`
nimmt `subdivision` (nach `rows`, wie die CPU-Variante) und `max_batch_items`.
`persist_forward_drizzle_multiband` lehnt lokale Warps **nicht mehr** ab; es
bleiben nur die zwei Gates Modus-2/1 und Device-vorhanden. Committet ein Lauf
mit ≥ 1 lokale-Warp-Frame über CUDA, meldet
`MultibandStoreBuildResult::backend_used = "cuda_hybrid"` (sonst `"cuda"`);
`DrizzleCudaStoreTiming::hybrid_local_frames` zählt sie,
`acceleration.cuda_stripe_path.hybrid_local_frames` im `forward_drizzle.json`.

**Timing-Aufschlüsselung.** `HybridPathStats` (`cpu_seconds`,
`gpu_raster_seconds`, `gpu_batch_calls`, `leaf_cells`, `records`) wird von
`build_frame_records_hybrid_local` gemessen: eine Gesamtuhr um
`enumerate_drizzle_stripe_leaf_cells` + finalen `flush()`, eine innere Uhr nur
um die `forward_drizzle_cuda_polygon_rect_area_batch`-Schleife; `cpu_seconds =
gesamt − gpu`. Über `accumulate_pair_by_frame_cuda(..., HybridPathStats*)` (neuer
optionaler letzter Parameter, wird addiert, nicht zurückgesetzt) akkumuliert
`persist_forward_drizzle_multiband` je Band und schreibt
`DrizzleCudaStoreTiming::hybrid_cpu_seconds` / `hybrid_gpu_raster_seconds` /
`hybrid_leaf_cells` → `acceleration.cuda_stripe_path`. `cpu_seconds` umfasst
CPU-Geometrie, (Leaf,Zelle)-Marshalling und Host-Record-Assembly; die Rasterzeit
lumpt H2D + Kernel + D2H (feiner nur mit `.cu`-Instrumentierung). Bei einem
Wurf mitten im Frame (Budget/Alloc-Boden) bleibt `stats` für diesen Frame
unberührt — kein Doppelzählen auf dem §19.4-Neustartpfad.

**Parität.**
- `test_forward_drizzle_contrib_list.cpp` `[cuda-parity]`: der frühere
  „CUDA-Pfad lehnt lokalen Warp ab"-Test ist jetzt eine **Bitidentitäts**-
  Prüfung — MONO+OSC, Rand/kein Rand, `accumulate_pair_by_frame_cuda` gegen die
  CPU-`accumulate_pair_by_frame` (alle vier Profile, Alpha-Maps, alle drei
  `clipping`-Zähler). Zusätzlich mit **winzigen** Batch-Grenzen (1, 7, 64 →
  Flush mitten im Frame/Leaf) und einem **ungleichen Streifen-Split** mit
  `max_batch_items=5`: die summierten Clipping-Zähler bleiben gleich der
  Ganzstreifen-CPU-Referenz. 16 Ganzstreifen-Varianten geprüft.
- `test_drizzle_profile_store.cpp` `[drizzle-store][cuda-parity]`: der frühere
  „lehnt lokale Warps ab"-Abschnitt prüft jetzt, dass ein Store mit einem
  lokale-Warp-Frame über CUDA (`hybrid_local_frames == 1`, `hybrid_leaf_cells
  > 0`) **byte-identisch** (Plane-FITS-Digest) zum reinen CPU-Build ist —
  Fixture mit Subpixel-Rotation, nicht-uniformer Quelle und aktivem Clipping;
  der Modus-2/1-Abschnitt lehnt weiterhin ab.
- `test_forward_drizzle_contrib_list.cpp`: zusätzlicher `[cuda-parity]`-Fall
  „subdivided local-warp (leaf_order > 0)" — Gauß-Koeffizienten mit
  Zweiter-Ordnung-Variation über das 4×4-Gitter (`ux²−uy²`, `ux·uy`) erzwingen
  echte Subdivision (`REQUIRE(max_leaf > 0)`, `REQUIRE(local_seen)`); bit-identisch
  CPU↔Hybrid bei Batch-Grenzen 1/9.

Keine neue Übersetzungseinheit — `enumerate_drizzle_stripe_leaf_cells` liegt in
`forward_drizzle.cpp`, der Hybrid-Producer in `forward_drizzle_contrib_list.cpp`,
beide schon in der `-ffp-contract=off`-Liste der `CMakeLists.txt`.

**Reihenfolge (§19.6.2):** zuerst dieser Schnitt mit unveränderten
Paritätsgrenzen, danach profilieren. Dominiert die lokale
Inversion/Subdivision den Durchsatz, folgt eine eigenständige, versionierte
Numerikrevision (`std::exp` → FMA-freie Minimax-Approximation, bit-identisch,
ohne §19.5.1-Änderung, mit neuem Registrierungs-Hash) — **nicht** eine
feldbezogene Toleranz.

**Suite:** 526/528 (`ctest`). Die zwei roten Tests liegen im **legacy-AQMH-Pfad**,
den keine dieser Änderungen berührt (`git diff` betrifft nur `forward_drizzle*`,
`drizzle_profile_store*`, `source_quality_artifact*`, `runner_forward_drizzle*`;
`test_aqmh_reconstruction.cpp` und die AQMH-CUDA-Rekonstruktion sind unverändert):
`test_acceleration_backend.cpp:254` (`acceleration_context_keeps_aqmh_maps_cpu_only`,
AQMH_MAPS-Selektion) und `legacy_reference` `aqmh_native_cuda_reconstruction_matches_cpu_reference`
(legacy AQMH-CUDA `weight_sum` überschreitet `margin(2e-4)` um ~2,4e-4 auf der
GTX 1660 Ti — GPU-Reduktionsreihenfolge, zu enge Marge). Der Versuch, ihren
Vorbestand aus einem sauberen HEAD-Worktree zu bestätigen, scheiterte an einer
CUDA-Toolchain-Fehlkonfiguration des frischen `cmake`-Configs (13.0 statt exakt
12.9 durch OpenCV-CUDA); die Zuordnung stützt sich daher auf die
Code-Trennung, nicht auf einen Vergleichslauf. Realer Großbild-Hybrid-Lauf (M42)
noch offen (braucht Run-Auftrag).

---

<a id="historie-30-57"></a>

### 30.57 M7 finalisiert: CUDA auf der Produktionsconfig (Modus 2/1) + reale Läufe (2026-09-07)

Auf ausdrücklichen Auftrag „M7 finalisieren" mit Freigabe realer Volllläufe.

**Kernbefund: „Device-2×2-Downsample" war kein fehlender Kernel.** Der
Device-Pfad erzeugt bereits korrekte interne-2×-Streifen (belegt durch den
realen M31-2/2-Lauf, §30.55); Modus 2/1 fiel nur zurück, weil der CUDA-Zweig in
`persist_forward_drizzle_multiband` die internen 2×-Streifen direkt an den mit
1×-`identity` erzeugten `StoreWriter` gab. Behoben durch Wiederverwendung der
vorhandenen 2×2→1×-Faltung.

**Implementierung.**
- `Downsample2x2Adapter` (bisher anonym in `output_scale.cpp`) als
  `Downsample2x2StripeAdapter` (pImpl) in `output_scale.hpp` exportiert:
  `feed(y_internal, internal_stripe)` puffert interne Zeilen und emittiert je
  gerades Zeilenpaar einen 1×-Ausgabestreifen; `finish()` wirft bei ungerader
  Gesamthöhe. Byte-identisch zu `downsample_uniform_and_raw_2x2` und zu
  `stream_forward_drizzle_uniform_and_raw_2x2`, unabhängig von der Bandhöhe.
- CUDA-Zweig: das Modus-2/1-Gate entfällt. Bei Modus 2/1 wird **genau ein**
  `Downsample2x2StripeAdapter` gebaut; jedes `run_cuda_chunked`-Band
  (interne 2×-Höhe) geht durch `adapter.feed(y0, stripe)` statt direkt an
  `sink`, danach `adapter.finish()`. Die `run_cuda_chunked`-Halbierungs-Retries
  füttern den Adapter nie doppelt (der `sink`/`feed`-Aufruf steht **nach** dem
  erfolgreichen `accumulate_pair_by_frame_cuda`). Ein `ForwardDrizzleCudaError`
  verwirft die Generation; `persist_multiband_store_from_predecessors` startet
  auf einem frischen CPU-Pfad mit **eigenem** Adapter neu — nie geschachtelt.
- Wächter vor `feed`: `stripe.uniform.internal_height == rows` und (bei
  `emit_fine`/`emit_medium`) die Detailebenen ebenso — ein Band mit
  abweichendem Ebenensatz desynchronisiert die zeilengepufferte Faltung
  sonst still (`DRIZZLE_STORE_CUDA_BAND_PLANE_HEIGHT_MISMATCH`).
- Gate vor dem CUDA-Versuch: **nur noch Device vorhanden**. Modus 2/1 läuft
  jetzt über die Host-Faltung, lokale Warps über den Hybridpfad §19.6.2 — von
  den ehemals drei Gates (§30.54) bleibt keines mehr als Ablehnungsgrund außer
  „kein Device".

**Parität (synthetisch).** `test_drizzle_profile_store.cpp`
`[drizzle-store][cuda-parity]`: der frühere „Modus 2/1 lehnt ab"-Abschnitt
prüft jetzt **byte-identisch** (Plane-FITS-Digest) zum CPU-Modus-2/1-Build, bei
Chunkhöhe **3** (Bandgrenze auf ungerader interner Zeile — genau der Fall, in
dem eine fehlausgerichtete Faltung verschöbe) **und** Ganzleinwand. Neuer
Abschnitt: Modus 2/1 **mit** einem lokale-Warp-Frame — Hybridpfad §19.6.2 **und**
2×2-Faltung in **einem** Build, `hybrid_local_frames == 1`, Digest identisch zum
CPU-Build.

**Reale Volllläufe (40 Frames, GTX 1660 Ti, `git=<dirty>`).**
`verify_m6m7/run_m7_final.sh`.

**M31 Produktionsconfig — Modus 2/1 (`internal_scale=2, output_scale=1`), affin.**
`reconstruction_multiband.fits` **byte-identisch** CPU↔CUDA:
`sha256=49ca284c3306fbe1a8dcbd961f47447efabfd9148b92d82e7befb6b5d3eaee18` —
**dieselbe Zahl wie der Vor-§30.54-Baustand** (§30.55), d. h. die gesamte
Kette §30.52–§30.57 inklusive der Host-2×2-Faltung ist ein No-Op auf dem realen
Produktionsergebnis. CUDA: `forward_drizzle_backend=cuda`,
`cuda_fallback_reason=null`, `cuda_stripe_path.bands=92`,
`resolved_chunk_rows=48`, Ausgabegeometrie 3870×2188 (`output_scale_applied`).
`hybrid_local_frames=0` (rein affin). §11.13 grün auf beiden;
`rss_process_peak` 3,85 GB CUDA vs 4,69 GB CPU. FORWARD_DRIZZLE-Zeit CPU **1899 s**
vs CUDA **2038 s** (≈ 1,07× — für Modus 2/1 praktisch gleichauf; die Host-Faltung
ist billig). _(Der `verify_m6m7/m31_*`-Config hält den Profil-Cache nicht — der
Plane-FITS-Digest fehlt auf diesem realen Lauf; der Plane-Digest für Modus 2/1
ist synthetisch abgedeckt (`[drizzle-store][cuda-parity]`, Abschnitt „Modus 2/1
… byte-identisch", Chunkhöhe 3 + Ganzleinwand), real trägt der fusionierte-Bild-
Hash — er liegt hinter dem Store und deckt die Kette bis zur Fusion ab.)_

**M42 lokaler Warp — 1/1 (`internal_scale=1`), Hybridpfad §19.6.2.**
Plane-FITS-Digest (52 Planes) **identisch** CPU↔CUDA
(`d869f53bd57fd89a090332170f6262a2ca75e5d3fdcd4412858bc1341bc7101c`) **und**
`reconstruction_multiband.fits` identisch
(`07f091043d02a97ce6121aa4736cebc1f54d9366cbf6fa9e2a82f2ee3f28a78f`) — beides
**dieselben Zahlen wie der CPU-Referenzlauf in §30.55**, d. h. der Hybridpfad
reproduziert bit-genau, was die CPU-Referenz vor §30.56 lieferte. CUDA:
`forward_drizzle_backend=cuda_hybrid` (Label greift auf einem echten Lauf),
`cuda_fallback_reason=null`, `cuda_stripe_path.hybrid_local_frames=1`,
`hybrid_leaf_cells=26 992 705`, 24 Bänder. §11.13 grün; `rss_process_peak`
3,96 GB CUDA vs 4,65 GB CPU. FORWARD_DRIZZLE-Zeit CPU **878 s** vs
CUDA-Hybrid **1832 s** (≈ 2,1×).

**Profiling-Befund (der eigentliche Zweck des Timing-Splits).** Auf dem
M42-Hybridlauf:
`hybrid_cpu_seconds = 444,2` gegen `hybrid_gpu_raster_seconds = 0,60` —
die CPU-Geometrie (Fixpunkt-Inversion + adaptive Subdivision + Marshalling +
Record-Assembly) des **einen** lokale-Warp-Frames dominiert die GPU-Polygon-
Rasterisierung um **~740×**. Die reine Rasterisierungs-Auslagerung bringt für
lokale Warps also praktisch nichts — die Geometrie ist die gesamte Kosten.
Konsequenz nach §19.6.2: wenn der Hybrid-Durchsatz relevant wird, ist der
**nächste Schritt die versionierte Numerikrevision** (`std::exp` →
FMA-freie Minimax-Approximation, damit die **gesamte** Geometrie bit-identisch
auf der GPU laufen kann), **nicht** eine §19.5.1-Toleranz. Bis dahin bleibt der
Hybridpfad korrekt und byte-identisch, aber nicht schneller.

**Nebenbefund (kosmetisch):** `local_warp.local_model_samples_total` ist auf dem
CPU-Lauf 8 294 400, auf dem Hybridlauf 0 — nur der **Zähler** bleibt ungemeldet,
die Frame-/Sample-Ausschlussentscheidung läuft unverändert. `accumulate_pair_impl`
(gemeinsam für `cpu_pair_producer` und `cuda_pair_producer`) ruft weiterhin
`prepare_drizzle_frames` auf; die `per_frame_inversion_error_rate_max`-Prüfung und
`frames_excluded_subdivision_error_rate` werden dort für **beide** Pfade
angewandt, der Producer sieht nur `prepared.frames` (bereits gefiltert). Auf dem
realen M42-Lauf ist `frames_excluded_subdivision_error_rate=[]` und
`local_model_samples_discarded=0` auf CPU **und** Hybrid identisch — nur
`local_model_samples_total` wird auf dem Hybridpfad nicht getallt (er geht nicht
durch `build_frame_records`, das diesen Zähler führt). Kein Korrektheits- oder
Frame-Set-Unterschied (Plane-Digests identisch).

**Fazit M7.** CUDA-Vorwärtsdrizzle ist auf **allen** produktiv relevanten
Konfigurationen aktiv und byte-identisch zur CPU-Referenz: affin 1/1 und 2/2
(§30.55), **Produktionsconfig Modus 2/1** (§30.57, Host-Faltung) und
**lokale Warps** (§30.56/§30.57, Hybridpfad). Vor dem CUDA-Versuch bleibt nur
„Device vorhanden". Offen ist ausschließlich Durchsatz-Optimierung (kein
Korrektheits- oder Freigabe-Blocker; §19.6 stellt Bit-Exaktheit voran) und, davon
abhängig, die optionale Numerikrevision für einen vollen GPU-Lokalpfad.

**`s_j`-Pinning (§19.5.1): nicht anwendbar im aktuellen Zustand.** Die
`D_fine`/`D_medium`-Grenze ist eine Toleranz **relativ zu** `s_j`; mit
`-ffp-contract=off` + `--fmad=false` sind die `[cuda-parity]`-Tests auf
`cpu == gpu` exakt, die Toleranz also ungenutzt und `s_j` hat nichts zu
kalibrieren. Das Pinnen wird **erst** erforderlich, wenn eine nicht-bit-exakte
Numerikrevision (Option D, `std::exp` → Minimax) landet — dann wird `s_j` beim
Matrixlauf je Band gemessen und im Fixture gepinnt. Bis dahin: kein offener
Punkt.

**Durchsatz.** Keine Optimierungsarbeit (§19.6: Bit-Exaktheit hat Vorrang). Die
gemessene CPU↔CUDA-Zeit wird als Befund eingetragen, nicht als Blocker.

---

<a id="historie-30-58"></a>

### 30.58 M8 begonnen: Reihenfolge festgelegt + Durchsatz-Baseline eingefroren (2026-09-08)

M7 abgeschlossen (§30.55–§30.57), Start von M8 (GUI, Report, Doku,
Cache-Kommunikation).

**Reihenfolge (Benutzerbeschluss):** Report → GUI/Cache → Legacy-Entfernung →
Doku. Begründung: die Report-Abnahme („Anzeige = tatsächliche FITS-/Runartefakte",
§23.1) ist gegen die vier realen §30.57-Läufe sofort prüfbar, ohne neuen Lauf;
die Doku beschreibt zuletzt das Ergebnis der ersten drei Schritte.

**Durchsatz-Baseline eingefroren (Benutzerbeschluss).** Die realen
`FORWARD_DRIZZLE`-Wandzeiten der §30.57-Läufe werden die eingefrorene
M8-Baseline für das 20-%-Gate aus §11.11 — festgeschrieben in **§11.11.1** des
Implementierungsplans mit Referenzumgebung (AMD Ryzen 7 3700X 8C/16T, GTX 1660 Ti,
Release + `-ffp-contract=off`/`--fmad=false`, `reconstruct`, `--max-frames 40`,
`workers_used=1`, warmer SSD-Cache):

| Config | Backend | `FORWARD_DRIZZLE` s | `pixels_supported` |
|---|---|---|---|
| M31 Modus 2/1 (`internal_scale=2, output_scale=1`, affin) | `cpu` | 1899,13 | 21 353 918 |
| M31 Modus 2/1 | `cuda` | 2037,65 | 21 353 918 |
| M42 lokaler Warp 1/1, Hybrid §19.6.2 | `cpu` | 878,19 | 24 879 983 |
| M42 1/1 Hybrid | `cuda_hybrid` | 1831,95 | 24 879 983 |

M9 vergleicht je Config/Backend auf derselben Referenzumgebung (Median aus drei
Wiederholungen ≥ 0,8 ×). Der normierte Bruch
`throughput = processed_source_samples / forward_drizzle_wall_seconds` wird mit
§30.59 verbindlich — `processed_source_samples` = nominale Eingangs-Samples
`frames · W · H` in `forward_drizzle.json` v2.

**Nächster Schritt:** Report — `report_generator` + `report_{de,en}.json`:
Coverage/`n_eff`/Alpha/Candidate-Gates, M4-Übernahme (Pixelmaßstab, Fluxraum,
Rausch-/Korrelationsvertrag OSC/MONO × Scale-Modi), plus die neuen
`forward_drizzle.json`-Felder. Verifikation gegen die §30.57-Artefakte.

**Noch nichts committet.**

---

<a id="historie-30-59"></a>

### 30.59 M8 Report-Schritt: `forward_drizzle.json` v2 + Report-Sektion (A + B, 2026-09-08)

Erster M8-Änderungsschritt aus §30.58 („Report zuerst"). **A = Report-Generator,
B = Core-Emit-Lücken**, in einem Zug.

**B — `forward_drizzle.json` `schema_version` 1 → 2 (additiv, keine bestehende
Form geändert), in `apps/runner_forward_drizzle.cpp`:**

- `throughput`: `frames_used`, `source_width/height`, `processed_source_samples`
  (= `frames · W · H`, nominale Eingangs-CFA-Samples — stabiler, reproduzierbarer
  Nenner ohne Hot-Loop-Zähler; eine Post-Masken-Verfeinerung wäre ein NEUES
  Feld, keine Neudefinition), `forward_drizzle_wall_seconds`,
  `source_samples_per_second`. Das ist der Nenner für das §11.11/§11.11.1-Gate.
- `runtime_environment`: `build` (= `core::build_info_json(false)`), `hardware`
  (`cpu_model` aus `/proc/cpuinfo`, `logical_cores`, `gpu`), `threads`
  (`parallel_workers_config`, `workers_used`). Beantwortet „gleiche Maschine wie
  die Baseline?" ohne zweites Artefakt. Die `gpu`-Probe
  (`AccelerationContext`-Ctor → `cv::cuda::setDevice`) läuft **vor**
  `begin(Phase::FORWARD_DRIZZLE)` und nur wenn `fd_accel.using_gpu` — nie
  mitten in MULTIBAND, wo sie nach dem Erfassen von `phase_rss_start` läge und
  die §11.13-Phasen-RSS-Buchhaltung verzerren würde; der CUDA-Stripe-Pfad
  wählt das Gerät ohnehin selbst, also keine zusätzliche Störung des Computes.
  Ins JSON kommt `gpu` nur, wenn ein CUDA-Pfad tatsächlich committet hat
  (`fd_backend_used` beginnt mit `cuda`), sonst `null`.
- `flux_space`: `space = "normalised_linear_working"`, `luma_definition`,
  `same_space_as = "reconstruction_multiband.fits"`,
  `stacking_normalisation_undo_applied = false`, Hinweis auf die 17.4-Rücknahme
  in M10. §23.1-M4-Übernahme („Fluxraum im Report") als eindeutige Aussage.
- `alpha_confidence_summary`: je Band `support_px`, `alpha_below_one_px`,
  `alpha_below_one_fraction`, `mean_alpha_on_support`, `min_alpha_on_support`,
  berechnet vor der Freigabe von `cand` (`alpha_final_by_band`), Nenner = Luma-
  Support. `alpha_final_by_band[b]` ist entweder leer (alpha ≡ 1) oder
  `assign(W·H)` — dasselbe Vollraster wie `uniform_support`; ein
  `af.size() != uniform_support.size()` wirft `FORWARD_STAGE_ALPHA_GRID_MISMATCH`,
  damit eine spätere Rasteränderung laut scheitert statt eine sinnlose Zahl zu
  melden. Die Frame-/Sample-Ausschlussentscheidung ist davon unberührt (läuft in
  `prepare_drizzle_frames`, §30.57).
- **`n_eff`/Coverage werden NICHT neu emittiert** — `sampling_geometry.json`
  enthält bereits die vollständige Coverage-Gate-Zusammenfassung
  (`geometric_uniform_neff_p10` je Kanal, `min_channel_n_eff_p10`,
  `supported_fraction`, Löcher, `violations`, Dither-Streuung). Der Report liest
  sie von dort.
- `tests/test_runner_forward_drizzle.cpp`: Assertions für alle vier neuen Blöcke
  + `schema_version == 2` + `processed_source_samples == frames·W·H`. `tests`-
  Target in `CMakeLists.txt` bekommt jetzt eine eigene generierte Build-Info-TU
  (`tile_compile_add_build_info(tests …)`), weil das in `tests` kompilierte
  `runner_forward_drizzle.cpp` nun `core::build_info_json()` referenziert.

**A — Report-Sektion `gen_forward_drizzle(fd, sg)` in
`web_backend_cpp/src/services/report_generator.cpp`:**

Neue Sektion „CFA Forward Drizzle / Multiband" mit sechs Karten, aus
`forward_drizzle.json` + `sampling_geometry.json`:

1. **Abdeckung & Geometrie** — Pipeline-Methode/Vertrag, Quelle/Rekon px,
   internal/output scale, Kernel, pixfrac, `pixels_supported`, Coverage-Gate
   (passed, `min_supported_fraction`, `min_channel_n_eff_p10`, `supported_fraction`
   je Kanal, `geometric_uniform_neff_p10` je Kanal, Löcher, `violations`). Status
   `bad` bei nicht bestandenem Gate.
2. **Kandidatenauswahl & Gates** — `selected_candidate`/`selection_reason`/
   `fallback_reason` + 3×6-Metrikmatrix (`median_fwhm`, `p90_fwhm`, `tail`,
   `elongation`, `background_rms`, `seam_score`) je Kandidat mit Wert oder
   „n/a (Grund)". Status `warn`, wenn nicht `drizzle_multiband` gewählt wurde.
3. **Fluxraum & Pixelmaßstab** — `flux_space`-Block, Pixelmaßstab
   (Rekon/Quelle-Verhältnis, internal×output); Himmels-Pixelmaßstab ausdrücklich
   „auf M10 verschoben".
4. **Rauschdiagnostik** — `background_rms` + `seam_score` je Kandidat, mit dem
   Hinweis, dass `background_rms` im normalisierten linearen Arbeitsraum liegt.
5. **Ressourcen & Durchsatz** — Backend, `throughput`-Block, `runtime_environment`
   (CPU-Modell, Kerne, GPU, Worker, Build), Phase-RSS-Hülle/Temp-OK/Working-Set,
   `cuda_stripe_path`-Kennzahlen inkl. `hybrid_*`. Status `bad`, wenn eine der
   Ressourcengrenzen nicht ok ist.
6. **Alpha-Konfidenz** — je Band `support_px`, `alpha < 1`-Anteil, Mittel-/Min-
   Alpha.

Registriert nach `gen_reconstruction`; `build_report_html` bekommt zwei neue
Parameter (`fwd_drizzle`, `sampling_geometry`), `generate_run_report` liest die
beiden Artefakte. Basis-Strings **englisch**; 39 DE-Übersetzungen in
`web_frontend_v3/i18n/report_de.json` (Titel, Kartenüberschriften, Zeilenlabels,
zwei Erklärsätze). Bewusst **nicht** übersetzt: kurze Ein-Wort-Labels, die
Präfixe anderer Report-Wörter sind (`Metric`→`Metrics`, `Candidate`→`Candidates`,
`Note`, `Band`) — `apply_replacements` matcht greedy längster-Key-zuerst über das
ganze Dokument, ein Teiltreffer würde fremden Text verstümmeln. Alte
`forward_drizzle.json` ohne v2-Felder rendern die Sektion weiter (Karten zeigen
„n/a"/„-").

**Verifikation.** Neuer `web_backend_cpp/tests/test_report_forward_drizzle.cpp`
(`BackendHarness`, Fixture in M42-Form, `schema_version:2`) rendert den Report
mit `TILE_COMPILE_REPORT_LOCALE=en` und prüft, dass die englischen Basis-Strings
(Sektions- + alle sechs Kartentitel, `cfa_forward_drizzle_multiband`,
`selection_reason`, `processed_source_samples`, CPU-Modell, Coverage-Gate-`n_eff`,
`normalised_linear_working`) den EN-Übersetzungslauf (DE→EN, greedy) **verbatim**
überstehen — der Test für die Teiltreffer-Verstümmelung aus dem Review. Grün.
`[forward-runner]` (tile_compile) grün mit den v2-Assertions (inkl.
`processed_source_samples == frames·W·H`).

**Vorbestehende rote Tests (unverändert, nicht durch diesen Schritt):**
`tile_compile` `acceleration_context_keeps_aqmh_maps_cpu_only` (Umgebung: CUDA-
Gerät vorhanden) + die zwei Legacy-AQMH-Fälle; `web_backend_cpp_contract`
(`raw stack ui` 404) + `web_backend_cpp_report_phase_issues`
(`tileCompileReportSetLanguage`/`const templates=` existieren nirgends im Baum) —
alle vier scheitern auch auf sauberem HEAD.

**Offen für M8:** GUI-Configfelder + Cache-Kommunikation, toter `method`-Zweig,
DE/EN-Doku. **Noch nichts committet.**

---

<a id="historie-30-60"></a>

### 30.60 M8: GUI-Methodenwahl entfernt, Cache-Kommunikation, Rechen-Invarianz-Test (2026-09-08)

Nach §30.59 die restlichen M8-Punkte, in der §30.58-Reihenfolge (GUI/Cache →
Legacy → Doku).

**Plan-Korrektur zuerst (M8/M10-Grenze).** Die frühere M8-Formulierung „toter
Code: `Config::method`, `getEffectiveMethod()` …" war falsch: `Config::method`
ist **nicht** tot — `normalizeMethod` leitet `aqmh.enabled` daraus ab,
`config.cpp:252` parst ihn, `to_yaml` serialisiert ihn, `runner_pipeline.cpp:1234`
weist ihn zu, `tile_compile.schema.json:6` dokumentiert `aqmh`/`classic_tile_compile`.
Das trägt den Legacy-AQMH-Rekonstruktionspfad, den die M9-10-%-Gegenüberstellung
braucht. **M8 = nur der GUI-Auswahlmechanismus**; `Config::method` + Schema +
`normalizeMethod` + `AqmhConfig` sind **nach M10 verschoben** (dessen
Änderungsliste nennt „Methodenschlüssel aus Schema … entfernen" ohnehin schon).
§23.1 „kein offener Punkt verschwindet" — der verschobene Teil ist explizit in
M10 ergänzt, die M8-Abnahmezeile entsprechend eng gefasst.

**[Legacy] GUI-Methodenwahl entfernt.**
- `web_frontend_v3/js/pages/parameter.js`: `draft?.method`, `reconMethod`,
  `methodHiddenCats`-Ternär, `isMethodParamVisible`, `AQMH_ONLY_CATEGORIES` weg.
  `CLASSIC_ONLY_CATEGORIES` → `LEGACY_HIDDEN_CATEGORIES`, die Legacy-Kategorien
  (`synthetic`, `tile`, `tile_denoise`, `local_metrics`, `global_metrics`)
  werden jetzt **bedingungslos** ausgeblendet (waren es unter Single-Method
  ohnehin immer) statt method-abhängig — kein Verhaltensunterschied, nur der
  Auswahl-Zweig verschwindet. `isLegacyCategoryHidden(path)` ersetzt
  `isMethodParamVisible(path, method)`. **`aqmh` bleibt sichtbar** —
  `cfg.aqmh.pyramid` speist die aktive SOURCE_QUALITY_MAPS-Phase
  (`runner_forward_drizzle.cpp:212`); die Umbenennung nach
  `reconstruction.quality.*` ist M10.
- `web_frontend_v3/js/components/phase-list.js`: `getPhasesForConfig` ist nicht
  mehr config-abhängig (der `method === "classic_tile_compile"`-Zweig weg);
  `CLASSIC_PHASES` bleibt nur für `CLICKABLE_PHASES`, bis die Legacy-Phasen-IDs
  in M10 aus UI/Resume fallen.

**[GUI] Phasenliste des Run-Monitors auf die aktive Pipeline korrigiert.**
Beim Prüfen fiel auf: die bisherige `AQMH_PHASES`-Liste war für die
Single-Method-Pipeline schlicht falsch — sie listete `AQMH_MAPS`,
`AQMH_GLOBAL_QUALITY`, `AQMH_RECONSTRUCTION`, `AQMH_DIAGNOSTICS` (feuern im
`reconstruct`-Pfad nie) sowie `STACKING`/`DEBAYER`/`ASTROMETRY`/`BGE`/`PCC`/
`HYPERMETRIC_STRETCH` (nur Legacy-`run`), und **keine** der sechs
Forward-Drizzle-Phasen. `updatePhaseState` hängt unbekannte Phasen nicht an,
also war der Live-Fortschritt von `FORWARD_DRIZZLE`/`MULTIBAND` unsichtbar und
vier AQMH-Zeilen standen dauerhaft auf „pending". Neu `RECONSTRUCT_PHASES` =
exakt die vom `forward_drizzle_only`-Pfad emittierte Reihenfolge, gegen die
Event-Logs zweier realer Läufe (M31/M42 `*_m6verify_*`) verifiziert:
`SCAN_INPUT, CHANNEL_SPLIT, NORMALIZATION, REGISTRATION, NORMALIZED_CACHE,
SAMPLING_GEOMETRY, COMMON_OVERLAP, SOURCE_QUALITY_MAPS, GLOBAL_QUALITY,
FORWARD_DRIZZLE, MULTIBAND` (11 Zeilen). `PREWARP` entfällt (der Forward-Pfad
ruft `run_phase_registration_prewarp(..., registration_only=true)`).
`GLOBAL_METRICS` **auch nicht** in der Liste: `runner_phase_metrics.cpp:541`
exponiert es nur als Stage wenn `aqmh.enabled==false`, aber `normalizeMethod`
(`io/config.cpp:87`) setzt `Config::method` per Default auf `"aqmh"` →
`aqmh.enabled==true` auf dem `reconstruct`-Pfad by construction (beide realen
Läufe emittieren kein `GLOBAL_METRICS`-Event). Es käme sonst als dauerhaft
„pending"-Zeile — genau der Defekt, der hier für die AQMH-Zeilen behoben wurde.
Kehrt in die Liste zurück, wenn M10 `Config::method` entfernt. Im Browser gegen
einen realen M42-Lauf verifiziert: Run-Monitor zeigt die 11 Zeilen, keine
Konsolenfehler. Das M10-Item „PREWARP-/DEBAYER-Scheinphasen aus der aktiven
Phasen-ID-Liste entfernen" bleibt bestehen (betrifft die *Legacy*-IDs in
`CLASSIC_PHASES`/`CLICKABLE_PHASES`).
- **Nicht angefasst** (Anzeige von Lauf-Metadaten, keine Auswahl):
  `run-history.js`/`run-monitor.js` zeigen `status.method` historischer Läufe.

**[GUI] Cache-Kommunikation.** `keep_profile_cache_after_run` und
`delete_source_cache_after_run` sind schon im Schema + Struct (Default je
`false`) und werden vom schemagetriebenen Parameter-Editor automatisch
gerendert (nach Wegfall des Methodenfilters sichtbar). Neu: **Erklärungstext**
— `description` für beide Felder in `tile_compile.schema.json` **und**
`tile_compile.schema.yaml` (Resume-Warnung: `delete_source_cache_after_run=true`
deaktiviert die Rekonstruktions-Wiederaufnahme, Report weist
`resume_reconstruction_disabled` aus) plus lokalisierte
`param.reconstruction.*.short_help` in `web_frontend_v3/i18n/{de,en}.json`.

**[GUI] Rechen-Invarianz-Nachweis (die einzige M8-Abnahme mit echtem Test).**
Neuer `[forward-runner]`-Fall in `tests/test_runner_forward_drizzle.cpp`: derselbe
synthetische Lauf viermal (`diagnostics.level` `summary`/`full` ×
`keep_profile_cache_after_run` `false`/`true`) liefert **byte-identisch**
`final_image_sha256`, `reconstruction_multiband.fits`, `reconstructed_L.fit`,
`forward_drizzle_raw_L.fit` und denselben `selected_candidate`; `full` schreibt
nur die zwei zusätzlichen Control-FITS-Sätze. Fünfter Lauf mit
`delete_source_cache_after_run=true`: gleiches Rechenergebnis,
`cache_retention.source_cache=="deleted"` und
`resume_reconstruction_disabled==true`. Grün.

**[GUI] `method`-Feld ausgeblendet + Browser-Abnahme.** Der reine
Branching-Wegfall in `parameter.js` ließ das Schema-Feld `method` (noch bis M10
im Schema) weiterhin als editierbares `<select>` rendern — also doch ein
Auswahlmechanismus. Fix: `"method"` in `LEGACY_HIDDEN_CATEGORIES` (deckt Feld +
„Method"-Kategorie über `entry.category` bzw. `path.split(".")[0]`).
**Browser-Abnahme** (lokaler `tile_compile_web_backend` auf :8080 — `client.js`
zwingt alle API-Calls auf Port 8080, daher nicht 8091):
- Kategorienliste: kein `method`, kein `synthetic`/`tile`/`tile_denoise`/
  `local_metrics`/`global_metrics`; `aqmh` + `reconstruction` vorhanden.
- Suche „method": nur `bge.method`, `registration.engine`,
  `aqmh.reconstruction.pre_debayer_method` — kein Top-Level-`method`, keine
  „Method"-Kategorie (12 statt 13 Treffer).
- `reconstruction.drizzle.*` (9 Felder), `coverage_gate.*`, `multiband.*` und
  beide Cache-Felder rendern; die DE-`short_help` steht als `title=`-Tooltip am
  Label **und** am `<select>` (im DOM verifiziert), in `de.json` **und**
  `en.json` vorhanden.
- Keine Konsolenfehler.
- Report-Sektion gegen einen realen (Vor-v2-)M42-Lauf: Sektionstitel +
  fünf Karten voll DE-übersetzt, fehlende v2-Felder degradieren sauber zu
  „n/a" (die Alpha-Konfidenz-Karte fehlt mangels `alpha_confidence_summary`).
- **Bug gefunden + behoben:** Kartentitel mit `&` (`Coverage & geometry`, …)
  wurden von `make_plain_card_html`→`html_escape` zu `&amp;` und matchten die
  `report_de.json`-Schlüssel nicht mehr → blieben englisch. Titel auf „and"
  umgestellt (`Coverage and geometry` …), Schlüssel + EN-Test nachgezogen.
- **Test verschärft:** `test_report_forward_drizzle.cpp` rendert dieselbe
  Fixture jetzt **zweimal** — Locale `en` (Basis-Strings überleben den
  Ersetzungslauf) *und* Locale `de` (die deutschen Kartentitel erscheinen
  tatsächlich, die englischen Formen sind weg). Der `en`-Durchlauf allein
  konnte den `&amp;`-Bug nicht fangen (englische Basis braucht keine
  Übersetzung); der `de`-Durchlauf ist die diskriminierende Prüfung. Die
  Harness bootet das Backend pro Durchlauf neu (`start()` liest
  `TILE_COMPILE_REPORT_LOCALE` frisch). Grün.

**[Doku] DE/EN-Methodikdokumentation.** Neu:
`docs/guides/cfa_forward_drizzle_pipeline_{en,de}.md` — die aktive
Single-Method-Pipeline: Was-es-tut, Drei-Wege-Kandidaten (`drizzle_uniform`/
`_raw`/`_multiband`) + Fallback, aktive Phasen (`NORMALIZED_CACHE` …
`MULTIBAND`), Coverage-Gate, Scale-Modi (1/1, 2/2, 2/1), Ausgaben, Caches &
Resume (die zwei Flags), Report-Sektion. In `mkdocs.yml` unter „Workflows &
Tools" verlinkt. `docs/guides/workflow{,_de}.md`: Schritt 2 + Phasentabelle
auf die aktive Pipeline umgestellt, Verweis auf die neue Seite, Hinweis dass
die *Process-Flow*-Docs die historische Classic-/AQMH-Pipeline beschreiben.
`mkdocs build` grün (die 7 `--strict`-Warnungen sind vorbestehend im
Implementierungsplan-Doc, nicht in den neuen Seiten).

**Verbleibend für M8:**
- Vollständige Neuschreibung des `process_flow/`-Baums (per-Phase) auf die neue
  Pipeline — das ist M10-Ära-Arbeit (wenn Legacy entfernt wird); die neue
  Guide-Seite deckt die aktive Methodik ab.
- `docs/configuration_reference{,_en}.md` und
  `docs/configuration_examples_practical_{de,en}.md` beschreiben durchgehend den
  Legacy-Classic-/AQMH-Pfad (`method: classic`, `aqmh:`-Blöcke, „AQMH
  deaktivieren", `debayer_first`). Sie sind nicht *falsch* — der Pfad existiert
  bis M10 (M9 braucht ihn) — aber sie erklären nicht den Single-Method-
  `reconstruct`-Pfad und die jetzt in der GUI sichtbaren
  `reconstruction.*`-Felder. Angleich = mit der Legacy-Entfernung in M10; die
  neue Guide-Seite trägt die aktive Methodik bis dahin.
- `report_en.json`: die Report-Strings sind nur DE übersetzt; EN läuft über die
  englischen Basis-Strings (per EN-Locale-Test verifiziert, kein Shadowing).
- Explizite Legacy-Lauf-„read-only"-Badges: die Resume-Machbarkeitsprüfung
  (Backend-Dry-Run + Frontend-Grundanzeige) trägt die Substanz; ein
  Badge-Audit steht aus.

**Teststand:** `tile_compile` 510/511 (nur `acceleration_context_keeps_aqmh_maps_cpu_only`
rot, vorbestehend), inkl. neuem Rechen-Invarianz-Fall.
`web_backend_cpp_report_forward_drizzle` grün (`web_backend_cpp_contract` +
`web_backend_cpp_report_phase_issues` vorbestehend rot). Frontend `node --check`
grün, im Browser verifiziert. Schema-JSON/-YAML + i18n valide. `mkdocs build`
grün. **Noch nichts committet.** `pi_models/live_edit/*/shadow_predictions.jsonl`
sind Harness-Nebeneffekt, nicht Teil der Arbeit.

Nachtrag: §30.60 wurde committet — `27ba6e82 M8` (Autor Jeamy, 2026-09-08),
Arbeitsbaum sauber, inkl. der Run-Monitor-Phasenliste (`RECONSTRUCT_PHASES`,
11 Zeilen) und `test_report_forward_drizzle.cpp` EN+DE-Doppellauf.

---

### 30.61 M9 begonnen: synthetisches Qualitätsfixture + M66-100-Frame-Gate (2026-09-08)

Nutzerentscheidung: M9-Umfang jetzt = **Schritt 1 (synthetisch) + 100-Frame-M66-Gate
parallel**; PREWARP-AQMH-Referenz für den späteren 10-%-Vergleich = **Legacy-Testharness**.

**[§21] Unabhängiges synthetisches Qualitätsfixture.** Neu
`tile_compile_cpp/tests/test_forward_drizzle_synthetic_quality.cpp`
(`[synthetic-quality]`, ~13 s). Treibt den **ausgelieferten** Pfad
(`run_forward_drizzle_stages`: NORMALIZED_CACHE .. MULTIBAND) mit Frames aus einer
analytischen Ground-Truth-Szene und prüft die Plan-§3.2-Synthetikgates:
- **Erzeuger ist kernel-unabhängig** (Plan 21.2): analytische Gauß-Sterne, PSF per
  Quadratur (`sigma_eff² = sigma_intrinsic² + sigma_psf²`), bekannte Per-Frame-Affine
  (Rotation ±0,4° + Sub-Pixel-Dither), 3×3-Box-Integration je Quellpixel, GBRG-Bayer,
  Poisson- + Ausleserauschen mit festem Seed. Der Polygon-Rasterizer wird auf der
  Erzeugerseite **nie** aufgerufen.
- **Ergebnisse** (28–36 Frames, 176×176 Canvas, FWHM ~6,9 px): Zentroidfehler-Median
  **0,012 Ausgabepixel** (Gate < 0,1); Fluxstreuung (MAD der Per-Stern-Ratio) auf der
  **G/Luma-Ebene ~0,3 %** (Gate < 0,5 % — die Ebene, die die Aperturphotometrie/PCC
  nutzt), R/B **~1–2 %** (Gate < 2 %, sampling-limitiert: die R/B-CFA-Ebenen liegen
  auf einem 2-px-Untergitter, das ist eine Abtast- und keine Rekonstruktionsgrenze);
  kein systematischer Flux-Bias > 5 %; Multiband keine FWHM-Regression gegen Raw.
- **Bewusst zunächst eng** (erster Schnitt, nicht die ganze §21-Liste). **Offen in
  §21:** Moffat-Profile, bekannte WCS, diffuse Struktur + linearer/gekrümmter
  Himmel (gehören zu den RMS-/Seam-Gates), lokales Warp-Feld, ortsvariable PSF,
  Hotpix/Cosmics, bekannte Schlechtframes, RGGB/BGGR/GRBG, MONO-Ein-Ebenen-Fixture,
  MTF50, Farbdifferenz, Seam-/Support-Fehler, Bootstrap-CIs. Ein Coverage-Kommentar
  im Testkopf listet Abgedecktes vs. Offenes.

**[Gate 2] M66-100-Frame-Lauf.** `reconstruct --max-frames 100` auf
`/media/tc_ssd/M66_lights_min` (277 Frames, 3840×2160 OSC GBRG, DWARF II 5 s VIS),
Config = m6verify-Config (`runs/m42_m6verify_.../config.yaml`) mit **einer** Änderung:
`runtime_limits.memory_budget 4096 → 8192`. Run-ID `m66_m9gate_20260908`, in `runs/`,
`git_sha 27ba6e82`, `git_dirty false`, `input_manifest_sha256 674533a6…` (für den
Legacy-Vergleich replaybar).
- **Befund: der eingefrorene 4-GB-Umschlag lässt 100 Frames bei 3840×2160 nicht zu.**
  Der Admission-Guard vor NORMALIZATION (`runner_pipeline.cpp:1815`,
  `plan_drizzle_memory` mit `chunk_rows = height`) wirft
  `DRIZZLE_MEMORY_BUDGET: explicit chunk_rows exceeds budget`:
  `retained = 100·3840·2160·4 = 3,32 GB`; `budget − retained − source − fixed ≈ 900 MB`;
  `max_rows ≈ 1832 < height 2160`. Bei 40 Frames (m6verify) passt es. Das Gate läuft
  daher gegen einen **8-GB-Umschlag**, nicht den §11.11.1-4-GB — bei der Bewertung
  von „11.11-Formeln eingehalten" als Eingabe protokollieren, nicht als bestandenes
  4-GB-Resultat lesen. (Nebenfrage, nicht jetzt: der Guard nutzt `chunk_rows = height`
  für die Vorabschätzung, der Lauf selbst `chunk_rows = 0`/auto ≤ 256 — der Guard
  weist damit Konfigurationen ab, die der Lauf per Chunking bewältigt hätte.)
- **Phasenzeiten:** SCAN_INPUT 1,6 s, CHANNEL_SPLIT ~0 s, NORMALIZATION 65 s (1 Worker),
  REGISTRATION 80 s, NORMALIZED_CACHE 2 s, **SAMPLING_GEOMETRY 36 min** (09:47:57→10:23:55),
  COMMON_OVERLAP 0,2 s, SOURCE_QUALITY_MAPS 4 min 39 s, GLOBAL_QUALITY 37 s.
  `compute_geometric_coverage` ist ausdrücklich „deterministic bounded reference, no
  per-worker canvases" (`sampling_geometry.cpp:274` `(void)num_workers`) — **einkernig**;
  je 256-Zeilen-Stripe wird jeder Frame **zweimal** voll rasterisiert (CFA-Droplet +
  Full-Footprint), also O(Quellpixel·Frames·Stripes). Bei 100 Frames / vollem Sensor
  die dominante Wandzeit. Eigener M9-Datenpunkt.
- **Als Hintergrund-Task gestartete FORWARD_DRIZZLE-Läufe wurden wiederholt vom
  Claude-Code-Hintergrund-Task-Supervisor gekillt** (Meldung „system running low
  on memory"). **Wichtige Korrektur:** das ist eine Heuristik des Supervisors,
  **nicht** der Kernel — `free` zeigte durchgehend ~30 GB `available`, kein
  dmesg-OOM, der Runner selbst meldete nie einen Fehler. Auch triviale
  Watcher-`until`-Schleifen (~3 MB) wurden in Sekunden gekillt. Die frühere
  Formulierung „die Box hat keinen Spielraum / Umgebungsblocker" war falsch —
  Ursache ist der Supervisor-Schwellwert, der die ~30 GB freien RAM nicht
  widerspiegelt. Der Vorgänger-Checkpoint (bis GLOBAL_QUALITY) bleibt jeweils
  intakt; `resume-reconstruction --from-phase FORWARD_DRIZZLE` überspringt die
  36 min SAMPLING_GEOMETRY + 4,6 min SOURCE_QUALITY_MAPS.
- **Sackgasse: `memory_budget` lässt sich nicht nachträglich am Lauf senken.** Ein
  Versuch, die Config-Snapshot von 8192 auf 4096 zu editieren (plus `config.sha256`
  in `run_provenance.json` und `config_sha256` im Checkpoint), scheiterte an
  `FORWARD_STAGE_SOURCE_IDENTITY_MISMATCH`: `plan.source_identity_hash` in
  `registration_sampling.json` ist über `sha256_bytes(input_manifest ":" config_sha)`
  gebildet — die Config-SHA ist in die Sampling-Geometrie-Identität eingewoben.
  `memory_budget` ist also **Teil der Lauf-Identität**, obwohl es das Rechenergebnis
  nicht ändert. Config-Snapshot + Hashes zurückgesetzt; der Checkpoint ist wieder
  konsistent für einen 8192-Resume.
- **Lösung: der Lauf muss außerhalb des CC-Hintergrund-Supervisors laufen.** Der
  Nutzer startete `resume-reconstruction --from-phase FORWARD_DRIZZLE` per
  `nohup … &` im eigenen Terminal (Stand: läuft, > 60 min CPU-Zeit einkernig in
  FORWARD_DRIZZLE, RSS stabil ~3,7 GB, keine Fehler, ~30 GB frei). M66 ist ein
  rotierendes Feld → §19.6.2-Local-Warp-Pfad (CPU-Geometrie + GPU-Polygonfläche),
  laut Memory ~1–2× langsamer als reine CPU und ohne Beschleunigung lokaler
  Warps. **Eigener M9-Datenpunkt: FORWARD_DRIZZLE auf 100 rotierenden Realframes
  ist einkernig > 1 h.**
- **Stand M66-Realdaten-Gate:** Ressourcen/Zeiten des 100-Frame-Laufs bis
  GLOBAL_QUALITY vollständig charakterisiert (der primäre „Speicher/Runtime"-Zweck
  von Gate 2). FORWARD_DRIZZLE läuft (Nutzer-Terminal); MULTIBAND +
  `forward_drizzle.json`-v2 + gematchte Metriken auf M66 folgen nach Abschluss.

**[Legacy-Referenz] Harness identifiziert.** `tile_compile_legacy_reference`
(`CMakeLists.txt:617`, Option `TILE_COMPILE_BUILD_LEGACY_REFERENCE`) — identische
Quellen wie `tile_compile_runner`, kompiliert mit `TILE_COMPILE_LEGACY_REFERENCE`,
überspringt den `PIPELINE_UNAVAILABLE_DURING_CUTOVER`-Guard; nicht installiert, in
M11 entfernt. Der 10-%-Vergleich (frisches isoliertes Verzeichnis außerhalb `runs/`,
gleiches Manifest/Frameauswahl/Normalisierung/Crop/`output_scale`, primär
`internal_scale=2 output_scale=1` wo der Altpfad nur 1× kann, plus `G_quality:=1`-
Kontrolllauf) ist ein eigener Schritt, noch nicht begonnen.

**Strukturelle M9-Blocker (Abnahme wird später schließen als der Laufplan nahelegt):**
1. Die §21-Truth-Fixture-Menge ist erst angefangen (siehe Offen-Liste oben).
2. Kein realer MONO-/Schmalbanddatensatz vorhanden — die Pflichtmatrix-Zeile „realer
   MONO-Datensatz" und die MONO-Bisektion sind hier nicht ausführbar.
3. ASTROMETRY/BGE/PCC/HMS laufen im `reconstruct`-Pfad nicht (STACKING-Pass-through
   erst M10), daher ist die Abnahmezeile „keine schwere Regression in … Astrometrie
   oder Downstream-Kompatibilität" auf diesem Codestand **nicht bewertbar**.

**Teststand:** neues `[synthetic-quality]` grün (73 Assertions, 3× deterministisch);
`[forward-runner]`+`[synthetic-quality]` 7 Fälle / 233 Assertions grün. M66-Gate läuft.
**Noch nichts committet** (M9-Arbeit).

---

### 30.62 §11.14 P0: Forward-Drizzle-Geometrie-Instrumentierung + K-Faktor-Messung (2026-09-08)

Erster verbindlicher Schritt der §11.14-Reihenfolge (P0: „Geometrieaufrufe, I/O
und Teilzeiten messen"). Nicht-invasive Diagnosezähler + grobe Teilzeit-Timer
über den kompletten Geometriepfad; anschließend eine synthetische Lokal-Warp-
Fixture, die den heutigen K-Faktor (Stripe-Anzahl) bei Chunkhöhen 1/16/64/256/
Vollhöhe charakterisiert.

**Neu:**
- `tile_compile/reconstruction/drizzle_geometry_stats.hpp/.cpp` — Prozess-globale
  `Registry` (einfache `uint64_t`, **nicht atomar** — der Referenzpfad ist heute
  einkernig; deterministische Parallelität ist P3 und besitzt die Atomics).
  `ScopedEnable` (Facility pro Phase an/aus + Reset), `ScopedVariant`
  (Consumer×pixfrac-Kontext, RAII), `ScopedGeometryTimer` (Wall + `CLOCK_PROCESS_
  CPUTIME_ID` pro `enumerate`-Aufruf), `stamp_context`, `to_json`.
- Zähler pro **Variante** (`prepare_exclusion_scan`, `coverage_cfa`,
  `coverage_footprint`, `production_uniform_raw`, `uniform_diagnostic`,
  `contrib_count`, `contrib_fill`, `hybrid_cpu_geometry`): `enumerate_calls`,
  `source_rows_scanned`, `source_samples_visited`,
  `top_level_sample_leaves_calls` (**nur** am Kopf von `sample_leaves`, nicht in
  der `subdivide_local`-Rekursion — das ist die P2-Abnahmezahl aus §11.14.4),
  `sample_leaves_discarded`, `leaves_generated`, `subdivide_local_calls`,
  `local_forward_calls`, `invert_calls`, `invert_iterations`
  (== `basis_evaluations` == `smooth_local_basis`-Aufrufe; jeder Newton-Schritt
  ruft genau ein `evaluate_smooth_local_displacement` → 16 `std::exp`, daher
  `exp_calls == 16·invert_iterations`), `leaf_cells_emitted`, `geometry_wall_s`,
  `geometry_cpu_s`.
- `tests/test_forward_drizzle_geometry_scaling.cpp` (`[forward-drizzle-p0]`,
  2 Fälle / 93 Assertions, deterministisch): Fall 1 `stream_forward_drizzle_uniform`
  über Chunkhöhen {1,16,64,256,0}, Fall 2 `compute_geometric_coverage`
  (= SAMPLING_GEOMETRY, die 36-min-Phase aus §30.61) über {1,16,64,0}.
- `tests/test_runner_forward_drizzle.cpp` `[forward-runner]`: prüft, dass
  `forward_drizzle_geometry_profile.json` bei erfolgreichem Stage geschrieben
  wird und nicht-leere `variants` mit emittierten `leaf_cells` trägt.
- Runner: `run_forward_drizzle_stages` läuft komplett unter `ScopedEnable(true)`;
  neues Artefakt `artifacts/forward_drizzle_geometry_profile.json` nach
  FORWARD_DRIZZLE (kein Checkpoint-Hash-Guard, wie `forward_drizzle.json`).

**Verdrahtung (guard: `registry().enabled`, ein gut vorhergesagter Branch):**
`forward_drizzle.cpp` (`local_forward`, `subdivide_local`, `sample_leaves`,
`prepare_drizzle_frames`-Sweep, `enumerate_drizzle_stripe_leaf_cells`, +
`ScopedVariant` an `stream_forward_drizzle_uniform` und
`stream_forward_drizzle_uniform_and_raw`), `registration_sampling_plan.cpp`
(`invert_local_source_to_canvas` — Aufruf + tatsächliche Newton-Schritte),
`sampling_geometry.cpp` (beide `rasterize_drizzle_stripe`-Aufrufe im
Stripe-Loop + `stamp_context`), `forward_drizzle_contrib_list.cpp`
(Count-/Fill-Pass + Hybrid-`enumerate`). Neue TU auf der `-ffp-contract=off`-
Liste (nur Integer/Timer, aber auf dem Pfad → identisch gepinnt).

**Messergebnis (Fixture: 1 Lokal-Warp-Frame, Quelle 24×24 = 576 px,
Canvas 48×48, `internal_scale=1`, pixfrac 0.8). `top_level_sample_leaves` /
`invert_iterations` (= `basis_evaluations`) je Consumer-Variante:**

| chunk_rows | K | `uniform_diagnostic` | `coverage_cfa` (pf 0.8) | `coverage_footprint` (pf 1.0) | wall (uniform) |
|-----------:|--:|---------------------:|------------------------:|-----------------------------:|---------------:|
| 1          | 48| 27 648 / 497 664     | 27 648 / 497 664       | 27 648 / 497 664            | 0.062 s        |
| 16         | 3 | 1 728 / 31 104       | 1 728 / 31 104         | 1 728 / 31 104             | 0.004 s        |
| 64 / 256 / 0 (auto) | 1 | 576 / 10 368 | 576 / 10 368     | 576 / 10 368              | 0.0014 s       |

`prepare_drizzle_frames` **zusätzlich konstant** `576` / `10 368` (ein voller
Quell-Sweep, K-unabhängig, Blätter verworfen). SAMPLING_GEOMETRY-Geometrie pro
Lokal-Warp-Frame = `coverage_cfa` + `coverage_footprint` = **2·K·source_pixels**
Top-Level-`sample_leaves` (zwei pixfrac-getrennte Varianten, §11.14.3 —
Fixture prüft `cfa.pixfrac == 0.8f ≠ foot.pixfrac == 1.0` explizit). Genau diese
zwei Passes verursachten die 36 min im realen 100-Frame-M66-Lauf (§30.61); die
Fixture zeigt denselben K-Faktor auf beiden.

**Damit ist der O(N·P·K·consumers)-Term aus §11.14.1 bestätigt, nicht widerlegt:**
- Für einen Lokal-Warp-Frame setzt `enumerate_drizzle_stripe_leaf_cells`
  `source_y0=0, source_y1=source_height` (kein Stripe-Bounding) und ruft
  `sample_leaves` **pro Quellpixel pro Stripe** — also `K·source_pixels`
  Top-Level-Aufrufe je Consumer-Variante statt `source_pixels`.
- Der Blow-up-Faktor **ist exakt K, je Consumer** (chunk_rows=1 ⇒ 48× je
  `uniform_diagnostic`/`coverage_cfa`/`coverage_footprint`, 44× Wall-Zeit
  gegenüber K=1). SAMPLING_GEOMETRY zahlt ihn zweimal (beide Coverage-Passes).
- `prepare_drizzle_frames` fügt einen weiteren vollen Sweep hinzu, dessen
  Blätter sofort verworfen werden.
- Ausgabe **bitidentisch** über alle fünf Chunkhöhen (Profil-Digest-Assertion).
- Auto-Chunking wählt für diese Mini-Szene K=1; der Blow-up beißt erst, wenn
  `chunk_rows` klein erzwungen wird **oder** das Budget bei großem Canvas viele
  Stripes erzwingt — genau das 100-Frame-M66- / 600-Frame-Produktionsregime
  (§30.61: SAMPLING_GEOMETRY 36 min einkernig, FORWARD_DRIZZLE > 1 h).

**P2-Zielzahl (§11.14.4), heute vs. Soll:** `production_uniform_raw` /
`uniform_diagnostic` `top_level_sample_leaves_calls` muss nach P1/P2
`eligible_local_frames · source_pixels` erreichen und **K-invariant** sein.
Heute: `1.0×` bei K=1, `3.0×` bei K=3, `48.0×` bei K=48. Die Fixture protokolliert
dieses Verhältnis, damit P2 eine konkrete Vorher/Nachher-Zahl hat.

**Bit-Exaktheit:** `[cuda-parity]` + `[forward-runner]` (329 727 Assertions /
13 Fälle) **vor der Verdrahtung identisch zu nach der Verdrahtung**, plus
`[forward-drizzle-p0]` 93 (2 Fälle, 3× deterministisch) + `[synthetic-quality]`
73. Der `sample_leaves`-Umbau ist logikerhaltend (Diff geprüft; Parity-Gate
beweist es). **Vorbestehender, unabhängiger** Fehlschlag im Gesamtlauf:
`test_acceleration_backend.cpp:254` (`acceleration_context_keeps_aqmh_maps_cpu_only`
— GPU-Umgebungs-abhängige Backend-Wahl in `core/acceleration.cpp`; per
`git stash` verifiziert, dass er auf dem unveränderten Baseline **identisch**
fehlschlägt — von P0 nicht verursacht).

**Noch nichts committet.** Nächster Schritt: P1 (autoritative Geometrie einmal,
`reconstruction/drizzle_geometry_cache.hpp/cpp`).

---

### 30.63 §11.14 P1+P2: autoritative Geometrie einmal, räumlicher Index, Bibliotheks-Verdrahtung (2026-09-08)

P1 und P2 als eine bit-exakt gegatete Änderung (untrennbar: P1s Store ist ohne
P2s Index/Verdrahtung nicht beobachtbar, das Abnahmekriterium steht nur in P2).

**Datenvolumen (aus P0-Zahlen):** empirisch 1 Leaf/Sample → ~0,6 GB
Record-Bytes pro Frame @ 3840×2160 (`LeafRecord` 72 B, 8,29 M Samples). Mit
Subdivision (`max_subdivision_depth=2`) theoretisch bis 16 Leaves/Sample,
**angenommene** 2–3 Leaves/Sample im Mittel ⇒ ~1,2–1,8 GB/Frame ⇒ 600 Frames
~360 GB–1,1 TB (Obergrenze abhängig von der genannten Leaf/Sample-Annahme,
kein Messwert). Geometrie-*Rechnung* fällt von O(N·P·K·consumers) auf O(V·N·P),
einmalig.

**Architekturwahl (nicht „mathematisch erzwungen", Korrektur nach Review):**
Ein Disk-Generation-Store ist die gewählte Lösung, weil er (a) phasenübergreifende
Wiederverwendung erlaubt (SAMPLING_GEOMETRY baut, FORWARD_DRIZZLE liest, statt
neu zu rechnen) und (b) den RAM-Bedarf unabhängig von Frame- **und** Zeilenzahl
hält. Eine reine „pro Frame bauen–lesen–verwerfen"-Variante wäre RAM-seitig bei
jedem N tragbar (der frühere „≤ 100 Frames"-Satz war falsch), verliert aber die
phasenübergreifende Wiederverwendung und zahlt O(N·P) je geometrieberührender
Phase erneut. **Offener Risikopunkt (P4):** Schreiben + vollständiges
Hash-Lesen + spätere Reads von 360 GB–1,1 TB müssen ausdrücklich in das
2400-s-Budget passen; sonst ersetzt I/O den Rechenengpass. Bis das gemessen ist,
gilt der Disk-Store als **nicht** für 600 Frames belegt.

**Neu — `tile_compile/reconstruction/drizzle_geometry_cache.hpp/.cpp`:**
- `make_geometry_cache_identity` — Hash pro Variante über
  `compute_coverage_geometry_hash` (bindet plan_hash, kernel, internal_scale,
  pixfrac, Subdivision) + Inversionsparameter + Canvas + CFA-Vertrag +
  color_mode + Algorithmusversion. **Dichter Footprint (pixfrac=1) und
  Drizzle-Coverage (pixfrac=0.8) = verschiedene Identitäten**; gleiche
  dedupliziert.
- `build_drizzle_geometry_cache` — pro (Variante, **lokalem** Frame) direkter
  `sample_leaves`-Sweep (jetzt aus `forward_drizzle.hpp` exportiert; `Leaf`
  ebenso). **Streaming:** jede fertige Quellzeile wird als ein
  zusammenhängender Block an die `.leaves`-Datei angehängt und **vor** der
  nächsten Zeile freigegeben (begrenzter Batch ≤ `source_width · 16 · 72 B`,
  unabhängig von Frame- und Canvashöhe; `memory_budget_bytes` = Untergrenze-
  Prüfung). **Ausschlusszählung direkt aus dem `sample_leaves`-Rückgabewert** —
  **nie** aus `geomstats` (Review-Punkt 3; Test `[geometry-cache]` weist
  identische Ausschlüsse mit Zählern an/aus nach). Serialisierung: `.rows`
  (`source_height` × `RowEntry` = {canvas_ymin, canvas_ymax, offset, count},
  32 B), `.leaves` (`LeafRecord` = {source_x:u32, channel:u16, leaf_order:u16,
  x[4]:f64, y[4]:f64}, 72 B), Records je Zeile geordnet nach `(sx, leaf_order)`.
  Streaming-SHA-256 der `.leaves` (kein zweiter Lesepass). **Crash-sicherer
  Commit** (Review-Punkt 4): eindeutige Generation- **und** Staging-Namen
  (`<hash>-<uid>`, `<uid>` aus rd()/pid), alte Generation **unangetastet**
  (nie gelöscht), alle Payload-Dateien `fsync`, dann `manifest.json` +
  `fsync`, dann `rename(staging→gen)` + Verzeichnis-`fsync`, zuletzt
  `current.json` per Temp-Datei-`rename` + `fsync`. Abbruch an jedem Punkt →
  vorherige committete Generation voll nutzbar.
- `DrizzleGeometryCacheReader` — lädt/​hasht beim Öffnen **nur** den
  Zeilenindex (`.rows`, ~35 KB/Frame) und prüft (Review-Punkt 5):
  Schema-/Algorithmusversion; Identität pro Variante (geometry_hash, Canvas,
  internal_scale, pixfrac, color_mode); **exakte erwartete lokale
  Framepopulation** (kein Fehlen/Extra/Duplikat, `expected_local_source_indices`-
  Argument); Dimensions-, Ausschlussstatistik- und Ratenkonsistenz; **jeder
  Row-Offset/-Count gegen die echte `.leaves`-Dateilänge** (lückenlose,
  ausgerichtete Kachelung, Summe == Dateigröße). `verify_record_bytes`
  (Default aus) streamt zusätzlich die `.leaves`-SHA — O(Store-Bytes) I/O,
  vom Aufrufer budgetiert; sonst tragen `.rows`-Hash + Strukturprüfung +
  Commit-Kette die Integrität. Leaf-**Records nie vollständig resident**:
  `enumerate_stripe` seekt/liest nur die den Streifen schneidenden Row-Blöcke
  (Test prüft `resident_bytes()` gegen die Leaf-Volumen-Schranke). Replay =
  **derselbe bbox-Clamp + Zell-Loop** in kanonischer
  `(sy, sx, leaf_order, y, x)`-Reihenfolge → bit-identisch. TU auf der
  `-ffp-contract=off`-Liste.
- `ScopedActiveGeometryCache` — thread-lokaler aktiver Reader. Das
  Delegations-*Mechanik* ist P3-tauglich (const-Reads, ein Guard pro Worker);
  **noch offen für P3**: ein Reader wird geteilt, nicht pro Worker neu
  geöffnet/geprüft. Ohne Guard = Status quo.

**Verdrahtung (`forward_drizzle.cpp`):**
- `enumerate_drizzle_stripe_leaf_cells` — Cache-Kurzschluss oben für lokale
  Frames; kein `sample_leaves`, kein `subdivide_local`, keine Inversion.
- `prepare_drizzle_frames` — bei aktivem Cache liefert `frame_stats` Total,
  Discarded, Ausschluss-Entscheidung; der Voll-Sweep entfällt (§11.14.3:
  Ausschlusszählung in den Build gefaltet). `local_model_samples_total/_discarded`
  und `frames_excluded_subdivision_error_rate` bit-gleich (dieselbe
  `double(discarded)/double(total)`-Rechnung, dieselbe Schranke).
- `rasterize_drizzle_stripe` → nutzt `enumerate_drizzle_stripe_leaf_cells`,
  daher sind Coverage (`compute_geometric_coverage`), Uniform/Raw/Fine/Medium
  (`stream_forward_drizzle_uniform_and_raw`) und die Beitragslisten
  (`build_frame_records`, `build_frame_records_hybrid_local`) **automatisch**
  mitgedeckt, sobald ein Cache aktiv ist.

**Neue Tests:**
- `tests/test_drizzle_geometry_cache.cpp` (`[geometry-cache]`, 7 Fälle,
  ~8 100 Assertions): Build→Read **zellenweise bit-identisch** (sx, sy, c,
  leaf_order, x, y + 8 Eck-`double` per `memcmp`) bei Chunkhöhen
  {1,7,16,Vollhöhe}; `resident_bytes()` skaliert **nicht** mit dem
  Leaf-Volumen; Reader = **null** `invert_iterations`/`subdivide_local`;
  **Ausschlüsse identisch mit `geomstats` an und aus**; strikter
  Readervertrag (Identität, Population {fehlend/extra/Duplikat}, Schema,
  Row-Offsets); Record-Korruption gefangen mit `verify_record_bytes` **und**
  strukturell (Truncation); Rebuild lässt die alte Generation intakt,
  `current.json` atomar, kein `.staging-`-Rest.
- `tests/test_forward_drizzle_geometry_scaling.cpp` neuer Fall
  `[forward-drizzle-p0][geometry-scaling][geometry-cache]`: mit publiziertem
  Cache ist die Uniform-/Coverage-Ausgabe **bit-identisch zum No-Cache-Lauf**
  (Digest == Baseline bei jeder Chunkhöhe); `uniform_diagnostic`,
  `coverage_cfa`, `coverage_footprint`, `prepare_exclusion_scan` machen je
  **null** `sample_leaves`/`invert_iterations`/`subdivide_local`; der Build
  macht genau `V · eligible_local_frames · source_pixels` Top-Level-Aufrufe.
  **⇒ K-Faktor für alle Consumer eliminiert, §11.14.4-Abnahme
  bibliotheksseitig belegt.**

**Parität:** Gesamtlauf 520/521 Fälle grün, **1 373 375 Assertions**
(inkl. `[cuda-parity]`); einziger Fehlschlag = der vorbestehende,
baseline-verifizierte `test_acceleration_backend.cpp:254`. Cache-Zweige sind
No-ops ohne aktiven Reader → Parität trivial erhalten. `sample_leaves`/`Leaf`-
Export ist logikerhaltend (Definition nur aus der anonymen Namespace
verschoben).

**Review-Nachtrag (2026-09-08), fünf behobene Lücken der ersten Fassung:**
1. Reader lud vorher jede `.leaves`-Datei komplett in den RAM → jetzt
   Range-Reads, nur `.rows` resident, `resident_bytes()` beweisbar. 
2. Builder sammelte den ganzen Frame vor dem Schreiben → jetzt zeilenweises
   Streaming mit Freigabe.
3. Ausschluss hing an `geomstats.enabled` (sonst `discarded=0`) → jetzt direkt
   aus `sample_leaves`; Test an/aus.
4. Generationswechsel löschte die alte Generation vor dem Rename und nutzte
   `flush()` statt `fsync` → jetzt eindeutige Namen, alte unangetastet,
   `fsync` von Dateien + Verzeichnissen, atomarer Pointer-Swap, kollisionsfreies
   Staging.
5. Readervertrag prüfte nur den Geometriehash → jetzt Schema/Algo, Population,
   Dimensions-/Ausschluss-/Offsetkonsistenz, Datei-Längen, optionale
   Record-Byte-Verifikation.

**Runner-Integration (erledigt, `runner_forward_drizzle.cpp`):**
- **Frischlauf:** in SAMPLING_GEOMETRY, **vor** `compute_geometric_coverage`,
  bei mindestens einem lokalen Frame: Disk-Preflight (konservativ
  4 Leaves/Sample · Variantenzahl · 1,5 Marge gegen `fs::space(dir)`), dann
  `build_drizzle_geometry_cache` nach `dir/artifacts/forward_drizzle_geometry/`,
  Reader öffnen, `ScopedActiveGeometryCache` publizieren (funktionsweit → deckt
  Coverage, GLOBAL_QUALITY **und** FORWARD_DRIZZLE, single-thread). Phase-Ende
  meldet `geometry_cache_seconds` / `geometry_cache_leaves`.
- **Checkpoint** `forward_drizzle_checkpoint.json` → neues Objekt
  `geometry_cache`: `generation`, `manifest_sha256`, `local_source_indices`,
  `variants[]` ({pixfrac, geometry_hash}), `total_leaves`,
  `total_record_bytes`.
- **Resume** (`--from-phase GLOBAL_QUALITY`/`FORWARD_DRIZZLE`): rekonstruiert
  die Varianten-Identitäten, prüft Anzahl + `geometry_hash` + sortierte
  `local_source_indices` gegen den Checkpoint,
  `FORWARD_STAGE_GEOMETRY_CACHE_PRESENCE_MISMATCH` wenn Lokal-Flag ↔ Checkpoint
  divergieren, öffnet den Reader **mit `verify_record_bytes=true`** (voller
  `.leaves`-SHA — einmalig, nicht der Per-Phase-Pfad), prüft
  `manifest_sha256`. Jeder Fehlschlag wirft **vor** jedem `phase_start`.
- **Affin-only-Läufe**: kein lokaler Frame → Cache komplett übersprungen,
  Verhalten unverändert (die 6 bestehenden `[forward-runner]`-Fälle grün).
- Neuer Test `test_runner_forward_drizzle.cpp`
  `[forward-runner][geometry-cache]` (`LocalWarpFixture`, 3 lokale Frames,
  48² Canvas, `chunk_rows=3` ⇒ K≈16): Frischlauf baut/publiziert/​checkpointet
  den Cache; `forward_drizzle_geometry_profile.json` zeigt für
  `production_uniform_raw`/`coverage_cfa`/`coverage_footprint`/
  `prepare_exclusion_scan` je **null** `invert_iterations` /
  `top_level_sample_leaves_calls`; Resume ab FORWARD_DRIZZLE verifiziert den
  Cache neu (`FORWARD_DRIZZLE`,`MULTIBAND` starten); ein mittiger Byte-Flip in
  `.leaves` lässt den Resume **vor** jedem `phase_start` scheitern.

**Cache × CUDA-Hybrid-Parität (Nachtrag Review):** neuer Fall
`[cuda-parity][geometry-cache]` in `test_forward_drizzle_contrib_list.cpp` —
für einen Lokal-Warp-Frame (MONO + OSC) gilt mit publiziertem Cache:
`accumulate_pair_by_frame` (CPU) unverändert gegenüber No-Cache **und**
`accumulate_pair_by_frame_cuda` (Hybrid) unverändert gegenüber No-Cache
**und** CPU == Hybrid — das Quadrat kommutiert byte-identisch (57 738
Assertions). Vorher war `[cuda-parity]` × Cache nicht getestet.

**Messwert Leaf/Sample:** der `LocalWarpFixture`-Lauf meldet
`geometry_cache_leaves = 6144` bei 3 Frames × 32² = 3072 Samples ⇒
**2 Leaves/Sample** (stärkerer Warp als die P0-Fixture mit 1). Erster
gemessener Beleg für die „2–3 Leaves/Sample"-Annahme, auf der die
P4-I/O-Frage ruht; der Disk-Preflight (Faktor 4) hat damit nur ~2×
Reserve, nicht 4×.

**Parität nach Runner-Integration:** Gesamtlauf **522/523, 1 431 129
Assertions**; einziger Fehlschlag weiterhin `test_acceleration_backend.cpp:254`.

**Damit ist P1+P2 abgeschlossen** (Bibliothek + Runner + Resume + Tests +
CUDA-Parität).

---

### 30.64 §11.14 P3/P4-Vorbereitung: Skalierungsleiter + gemessene Raten (2026-09-08)

Nutzerentscheidung: „beides vorbereiten" — die synthetische Skalierungsleiter
(gemeinsame Infrastruktur für P3-Worker-Bit-Exaktheit und P4-I/O/RAM-Messung)
zuerst bauen, dann aus den ersten Zahlen über die Reihenfolge P3↔P4 entscheiden.

**Neu:**
- `build_drizzle_geometry_cache` liefert jetzt die Wall-Zeit-Aufteilung
  `sample_leaves_seconds` (der O(V·N·P)-Rechenterm, einmalig, per-Frame
  parallelisierbar) vs. `write_seconds` (Record-/Index-Schreiben + `fsync`).
- `tests/test_geometry_scaling_ladder.cpp` (`[geometry-ladder]`): parametrische
  Lokal-Warp-Szene über eine Frame-Leiter. Default-Sprossen {4,8,16} @ 96²
  (schnell, CI); `TC_LADDER_FULL=1` → {8,40,100,200} @ 192². Misst pro Sprosse
  Build-Compute/Write, Disk-Bytes, Reader-`open`/`open+verify`,
  `resident_bytes`, Leaves/Sample, Forward-Drizzle mit/ohne Cache. Prüft:
  Cache-an-Ausgabe **byte-identisch** zu Cache-aus bei jeder Sprosse;
  `resident_bytes` < `total_record_bytes/4` (Zeilenindex-gebunden);
  Leaves/Sample über die Leiter ~konstant. Heavy-Sweep bricht bei kleinem
  `/tmp` sauber ab (kein Defekt).

**Messung (Default-Sprossen, diese Maschine):**

| N | compute_s | write_s | disk_MiB | wr_MiB/s | resident_KiB | Leaves/Sample | drz_cache_s | drz_nocache_s | Speedup |
|--:|----------:|--------:|---------:|---------:|-------------:|--------------:|------------:|--------------:|--------:|
| 4 | 0,077 | 0,005 | 2,53 | ~500 | 12 | **1,00** | 0,035 | 1,72 | **49×** |
| 8 | 0,155 | 0,008 | 5,06 | ~620 | 24 | 1,00 | 0,056 | 3,45 | **61×** |
| 16 | 0,318 | 0,017 | 10,12 | ~600 | 49 | 1,00 | 0,112 | 6,84 | **61×** |

- **Leaves/Sample = 1,00 exakt und konstant** (dieser Warp-Klasse; der stärkere
  `LocalWarpFixture`-Warp gab 2 — die Spanne 1–3 steht).
- **`resident_bytes` = nur Zeilenindex**, ~3 KiB/Frame, entkoppelt vom
  Record-Volumen (2,5–10 MiB). Der „kein Vollleaf-Cache"-Vertrag hält bei
  Skalierung.
- **Write ≈ 500–680 MiB/s** (Sprosse rechengebunden ⇒ Untergrenze).
- **Forward-Drizzle mit Cache 49–61× schneller** als ohne, bit-identisch.

**Projektion 600 Frames @ 3840×2160 (Extrapolation aus der 16-Frame/96²-Sprosse,
kein Lauf):** ~334 GiB Records auf Disk; `sample_leaves`-Build ~10 700 s
einkernig (**~670 s / 16 Kerne** — per-Frame trivial parallel); Record-Write
~570 s bei ≥ 600 MiB/s.

**Entscheidung P3↔P4:** Die I/O widerlegt die Architektur **nicht** — 334 GiB,
~570 s Write und ~1,8 MiB residenter Index sind unkritisch. Der einmalige
O(V·N·P)-`sample_leaves`-Rechenterm (~3 h einkernig) ist die Kostenstelle, und
der ist **per-Frame trivial parallelisierbar** ⇒ **P3 (deterministische
Parallelität) ist der richtige nächste Schritt**, die verbindliche Planreihenfolge
wird durch die Messung bestätigt (die Advisor-Sorge „I/O kippt die Architektur"
ist gemessen widerlegt). P4 behält den realen 600-Frame-End-to-End-Nachweis bei
echter Canvasgröße.

**Parität:** Gesamtlauf **523/524, 1 431 142 Assertions**; einziger Fehlschlag
weiterhin `test_acceleration_backend.cpp:254`.

**Noch nichts committet.**

---

### 30.65 §11.14 P3 Teil 1: paralleler Geometrie-Cache-Build (2026-09-08)

Die Messung (§30.64) zeigt den **Geometrie-Build** als dominante Kostenstelle
(~3 h einkernig für 600 Frames) — und er ist per-Frame trivial parallel.
Deshalb zuerst P3s ersten Bulletpoint: „Geometrie-Batches unabhängig parallel
erzeugen. Ergebnisse nach stabiler Frame-/Quell-/Leaf-ID veröffentlichen, nie
nach Task-Fertigstellungsreihenfolge."

**`build_drizzle_geometry_cache(..., int max_workers = 1)`:**
- Jede `(Variante, lokaler Frame)`-Kombination ist eine unabhängige Task
  (`build_one_frame`): eigener `sample_leaves`-Sweep, eigene `.rows`/`.leaves`-
  Dateien, eigene SHA-Kontexte, **kein geteilter veränderlicher Zustand**.
- OpenMP-Team (`#pragma omp parallel for schedule(dynamic,1) if(workers>1)`) über
  die flache Taskliste; Exception-Weiterleitung über `std::exception_ptr` +
  `#pragma omp critical` (OpenMP-Regionen lassen keine Exceptions heraus).
- Die `geomstats`-Zähler (prozess-global) werden über die Region deaktiviert und
  danach restauriert — sonst Data-Race auf dem Diagnosezähler; der Build ist
  ohnehin kein Consumer.
- **Manifest-Assemblierung seriell in deterministischer
  `(Variante, source_index)`-Reihenfolge** aus den `outs[k]`-Slots (Tasks in
  eben dieser Reihenfolge gepusht) → der committete Store ist
  **byte-identisch** zur 1-Worker-Referenz, unabhängig von Workerzahl und
  Scheduling.
- Ergebnis meldet `workers_used`, `wall_seconds`, `sample_leaves_seconds`
  (Summe über Frames = Einkern-Äquivalent), `write_seconds`.

**Runner:** `runner_forward_drizzle.cpp` leitet den Build-Workercount aus
`hardware_concurrency()` ab, gedeckelt auf die Taskzahl;
`TC_GEOMETRY_CACHE_WORKERS` überschreibt (`=1` = Referenzmodus). `workers_used`
+ Build-Teilzeiten landen im Checkpoint `geometry_cache` (informativ, **nicht**
Teil der Resume-Validierung — der Store ist workerzahl-unabhängig identisch).
Die Reduktion bleibt einkernig (`parallel_workers=1` unangetastet).

**Tests:**
- `test_drizzle_geometry_cache.cpp` `[geometry-cache][geometry-parallel]`:
  Build mit 1/2/4 Workern (6 lokale Frames, 2 Varianten) → identische
  per-Frame `rows_sha256`/`leaves_sha256`, `total_leaves`, `total_record_bytes`,
  Ausschlusszahl **und** identischer Forward-Drizzle-Digest.
- `test_geometry_scaling_ladder.cpp`: P3-Worker-Sweep (1/2/4/8) auf der größten
  Sprosse — per-Frame-SHAs byte-identisch über alle Workerzahlen; gemessene
  Wall-Speedups (16-Frame-Sprosse, diese Maschine):

| Workers | wall_s | Speedup vs. 1 |
|--------:|-------:|--------------:|
| 1 | 0,373 | 1,0× |
| 2 | 0,192 | 1,9× |
| 4 | 0,103 | 3,6× |
| 8 | 0,069 | 5,4× |

  `sample_leaves_seconds` (Gesamtarbeit) bleibt ~0,34–0,42 s über alle
  Workerzahlen. ⇒ Der ~11 900-s-Einkern-Build-Projektionswert skaliert real
  (≈ 3,6× bei 4 Kernen, 5,4× bei 8), macht die Disk-Store-Architektur
  laufzeitseitig tragbar.

**Parität:** Gesamtlauf **524/525, 1 431 157 Assertions**; einziger Fehlschlag
weiterhin `test_acceleration_backend.cpp:254`.

**Noch offen (P3 Teil 2, gekoppelt an P4):** Reduktionsarbeit nach unabhängigen
Zielregionen partitionieren (kanonische Reduktionsfolge pro Zelle, kein freier
Summenbaum), Workerzahl aus **gemeinsamem** Budget (Quellen, Q-Maps,
Geometrieindex, Kandidaten, Halo, Reader, Scratch) statt nur
`hardware_concurrency`, **geteilter** Reader statt pro Worker, dann Runner-Zwang
`parallel_workers=1` durch budgetierten Scheduler ersetzen, Tests 1/2/4/max mit
Store-/Masken-/Auswahl-Bit-Exaktheit + Peak-RSS im Vertrag. Diese Zahlen hängen
von P4s Speicher-/I-O-Messung bei echter Canvasgröße ab.

**Noch nichts committet.**

---

### 30.66 §11.14 P4: Speicher-/I-O-Residenz — Analyse, Admission-Term-Korrektur, Messung (2026-09-08)

**Kern-P4-Aufgabe: den behaupteten `3840*2160*4*N`-Admission-Term lokalisieren
oder als unbestätigt zurückweisen.**

**Lokalisiert:** `apps/runner_pipeline.cpp` Zeile 1817, `forward_drizzle_only`-
Preflight, direkt nach SCAN_INPUT. Übergab
`retained_bytes = pixels·sizeof(float)·frames.size()` an `plan_drizzle_memory`
(Kommentar: „bound existing registration-proxy retention"). **Der Term war
4× zu hoch:** der Registration-Proxy ist ein **2×2-Downsample**
(`build_registration_proxy` → `cfa_green_proxy_downsample2x2` /
`downsample2x2_mean`, `apps/runner_shared.cpp:1499`), also `pixels/4` Floats,
**keine** Vollauflösungsebene. Korrigiert zu
`(pixels/4)·sizeof(float)·frames.size()`. Realer frameskalierender
RAM-Term: **`N·pixels` Bytes** (100 Frames @ 3840×2160 = **~830 MB**, nicht
3,3 GB) — die §30.61-Blockade (100 Frames @ 4 GB abgelehnt) ist damit weg.
Keine pauschalen „25 GB".

**Residenztabelle (alle frameskalierenden Buffer):**

| Buffer | Eigentümer | Pro Frame | Ort | Anmerkung |
|---|---|---|---|---|
| Registration-Proxies | `RunnerFrameCache::registration_proxies_` | `pixels/4 · 4 B` | **RAM** | **Einziger** N-RAM-Term; **vor** FORWARD_DRIZZLE freigegeben (`runner_pipeline.cpp:1861`). Admission-Guard lädt ihn jetzt korrekt. |
| Normalisierte Vollframes | `DiskCacheFrameStore` | `pixels · 4 B` | **Disk** | LRU-Read via `VerifiedNormalizedSourceCache(memory_budget_mb)` → begrenztes Arbeitsfenster, nicht N-resident. |
| Source-Quality-Maps | `SourceQualityMapCacheReader` | ≤ 4 Streams · `pixels · 4 B` | **Disk** | „valid until next call" → 1 Frame gleichzeitig resident. |
| Geometrie-Cache-Leaves | `.leaves`-Dateien | `pixels · Leaves/Sample · 72 B` | **Disk** | `enumerate_stripe` seekt nur schneidende Row-Blöcke; nie voll resident. |
| Geometrie-Cache-Zeilenindex | `DrizzleGeometryCacheReader` | `source_height · 32 B` | **RAM** | Gemessen **3 KiB/Frame**, O(N), entkoppelt vom Record-Volumen. |
| Kandidaten-Buffer | `stream_forward_drizzle_uniform_and_raw` | `stripe_rows · width · N · sizeof(ClipCandidate)` | **RAM** | N-abhängig, aber **streifenbegrenzt** (chunk_rows), nicht vollhöhe; von `plan_drizzle_memory` budgetiert. |
| Streifen-Akkumulatoren (A,B,QA*) | dito | `stripe_rows · width · ~48 B · Kanäle` | **RAM** | **N-unabhängig**, chunk-begrenzt. |

⇒ **Nach der Korrektur gibt es keine O(N·P)-RAM-Pflicht.** Alles ist entweder
disk-backed mit begrenztem Fenster oder streifenbegrenzt. Der einzige echte
N-RAM-Term (Proxies) ist ¼-Auflösung und vor der Rekonstruktion frei.

**Skalierungsleiter-Messung (`test_geometry_scaling_ladder.cpp`, neuer
P4-Residenzblock):**

| N | Disk-Records MiB | Reader resident KiB | resident/record | Prozess-VmHWM MiB |
|--:|-----------------:|--------------------:|----------------:|-----------------:|
| 4 | 2,53 | 12 | 0,0048 | 232 |
| 8 | 5,06 | 24 | 0,0048 | 232 |
| 16 | 10,12 | 49 | 0,0048 | 232 |

- Reader-Residenz **linear in N** (3 KiB/Frame), **nicht** in N·P. Bei echter
  3840×2160-Canvas wäre der Zeilenindex `2160·32 B/Frame` ≈ 0,01 % des
  Record-Volumens.
- Prozess-`VmHWM` **konstant über N** — der Build mit seinem begrenzten
  Ein-Zeilen-Staging wächst nicht mit der Framezahl.
- Assertion: resident/Frame ~konstant, resident/record-Verhältnis wächst nicht
  mit N.

**Parität:** Gesamtlauf **524/525, 1 431 159 Assertions**; einziger Fehlschlag
weiterhin `test_acceleration_backend.cpp:254`. Die Admission-Guard-Korrektur
in `runner_pipeline.cpp` hat keinen Unit-Test (nur im vollen `reconstruct`-Lauf
erreicht); sie **lockert** einen Guard mit belegter Begründung (Proxy =
2×2-Downsample).

**Noch offen:** die reale 40/100/200/600-Frame-Skalierungsleiter bei **echter
Canvasgröße** (synthetisch für Lastskalierung zulässig; braucht aber Platz +
Zeit — die `TC_LADDER_FULL`-Sprossen sind 192², nicht 3840²) und der
600-Frame-End-to-End-Nachweis gehören zu **P6** (Benutzerlauf, nicht
autorisiert). P4-seitig ist der Speicherpfad analysiert und der eine falsche
Guard-Term korrigiert.

**Als Nächstes:** P3 Teil 2 (Reduktion nach Zielregionen partitionieren,
gemeinsames Worker-Budget aus der Residenztabelle oben, geteilter Reader,
`parallel_workers=1` durch Scheduler ersetzen). Dann P5 (Resthotspot). **P6
ist ein Benutzerlauf** — nicht autorisiert.

**Noch nichts committet.**

---

### 30.67 §11.14 P3 Teil 2: streifeninterne Zeilenband-Parallelität der Reduktion (2026-09-09)

**Aufgabe:** die Rekonstruktions-Reduktion nach unabhängigen Zielregionen
partitionieren, ohne die kanonische Reduktionsreihenfolge pro Zelle zu ändern
(kein freier Summenbaum, keine ungeordneten Float-Atomics), gemeinsames
Worker-Budget, geteilter Reader.

**Partitionierungsachse — nicht die naheliegende.** Erste Idee war
Streifen-Parallelität (jeder `chunk_rows`-Streifen ein Worker). Verworfen, weil:
1. `VerifiedNormalizedSourceCache` hält **einen** `Matrix2Df image_`; `load()`
   überschreibt ihn und gibt eine Referenz darauf zurück. W Streifen-Worker =
   W gleichzeitige Schreiber in denselben Puffer + hängende Referenzen. Die
   Frame-Schleife ist **innerhalb** der Streifenschleife, also würde jeder
   Streifen-Worker alle Frames erneut laden.
2. Der Profile-Store erzwingt strikte y-Reihenfolge (`drizzle_profile_store.cpp`
   `y != next_y_` → Wurf). Streifen-Parallelität bräuchte eine geordnete
   Drain-Schlange.

**Gewählt: streifeninterne Zeilenband-Partition** (`stream_forward_drizzle_uniform_and_raw`,
neuer Trailing-Parameter `int workers = 1`). Pro Streifen wird `[0, rows)` in
`nb = min(workers, rows)` lückenlose, disjunkte Canvas-Zeilenbänder geteilt.
Die Frame-Schleife bleibt **außen + seriell** (ein `source_of`-Load pro Frame
auf dem Aufruferthread); pro Frame läuft eine `#pragma omp parallel num_threads(nb)`-
Region, in der jeder Worker sein Band rastert (`rasterize_drizzle_stripe(plan,
*f, scale, pixfrac, y + r0, r1 - r0, …)` — der `i`-Callbackwert ist
fensterrelativ und wird per `gi = i + r0·width` in die streifenweiten
Akkumulatoren umbasiert), die Kandidaten seines Bandes sammelt und danach in
einer zweiten Band-Region `reduce_pixel_profiles` über dieselben `i`-Bereiche
aufruft. Jede Canvas-Zelle wird von **genau einem** Worker geschrieben, und die
Quell-Iterationsreihenfolge pro `i` ist unverändert → **bit-identisch** zu
`workers == 1`.

Warum das alle Klippen umgeht: **kein Provider-Concurrency** (ein `load()` pro
Frame, Aufruferthread), **keine Sink-Umsortierung** (ein `sink(y, result)` pro
Streifen, y-Reihenfolge, Store unangetastet), **keine Budget-Division** (A/B/QA*/
candidates sind dieselben Puffer, nur andere Indexbereiche → RAM identisch zur
seriellen Variante, keine `budget/W`-Verkleinerung von `chunk_rows`), **keine
Reduktionsbaum-Atomics**. Die 2/1-Produktionsvariante
(`stream_forward_drizzle_uniform_and_raw_2x2`) reicht `workers` nur durch — der
2×2→1×-Fold liegt hinter dem (weiterhin geordneten, ein-pro-Streifen) inneren
Sink.

**Geometrie-Cache-Weitergabe.** OpenMP-Worker erben den `thread_local`-Guard
`g_active_geometry_cache` nicht. `stream_*` fängt `active_geometry_cache()` auf
dem Aufruferthread ab und jeder Band-Worker setzt einen eigenen
`ScopedActiveGeometryCache` — ohne den würden Worker still auf volle
Re-Enumeration zurückfallen (numerisch **korrekt**, aber P1/P2 entwertet, von
Byte-Identität allein nicht erkennbar). Absicherung: neuer
`std::atomic<uint64_t> DrizzleGeometryCacheReader::enumerate_call_count()`;
der Test verlangt, dass er bei **jedem** W pro Lauf strikt wächst.

**Geometrie-Statistik.** Die prozessglobale, nicht-concurrency-sichere
`geomstats::Registry` wird bei `workers > 1` für die Streifenschleife
deaktiviert (RAII-Restore). Bei `workers == 1` (Default, Runner-Pfad
unverändert) ist die `ScopedVariant`/`ScopedGeometryTimer`-Instrumentierung
byte-identisch zu vorher.

**Band-Dispatch.** `#pragma omp for schedule(static, 1)` über `b ∈ [0, nb)`,
**nicht** Schlüsselung über `omp_get_thread_num()` — `num_threads(nb)` ist nur
eine Obergrenze, die der Laufzeitkern (z. B. unter `OMP_THREAD_LIMIT` oder
Nesting) senken darf; die Worksharing-Schleife garantiert die Bandabdeckung
auch dann, wenn weniger als `nb` Threads vergeben werden. Verifiziert:
`[geometry-parallel]` grün bei `OMP_NUM_THREADS=1` **und** `=6`.

**Ausnahmen.** Ein aus einer OpenMP-Region entkommender Wurf ist UB; der
Schleifenkörper fängt pro Iteration und rethrowt nach der Region via
`std::exception_ptr` + `#pragma omp critical` (dasselbe Muster wie der
P3-Teil-1-Build).

**Reichweite dieses Schnitts (bewusst eng):** nur `stream_forward_drizzle_uniform_and_raw`
(+ 2x2-Wrapper). `compute_geometric_coverage` bleibt seriell (eigener
Aufrufort, eigene Gate-Config; laut §30.64 nicht mehr der Engpass, seit
cache-gespeist). Der Runner-Scheduler folgt in **§30.68**; in diesem Schnitt
bleiben alle Bestandsaufrufer/-Tests durch `workers = 1` auf dem seriellen
Referenzpfad.

**Tests** (`test_drizzle_geometry_cache.cpp`, `[geometry-cache][geometry-parallel]`):
1. **Band-Tiling-Invariante:** Vereinigung der Leaf-Zellen aus 2/3/5 disjunkten
   Canvas-Zeilenfenstern == Ganzstreifen-Zellen (als sortierte Multimenge —
   die Band-Zerlegung ändert die *Sequenz*, nicht den Inhalt), für einen
   cache-gespeisten Local-Frame **und** einen Affin-Frame.
2. **Reduktions-Bit-Identität:** `compute_forward_drizzle_uniform_and_raw` mit
   vollem Mehrband + Quality-Provider (Raw/Fine/Medium + alle drei
   Alpha-Confidence-Faktoren), `workers ∈ {1,2,4}` × `chunk_rows ∈ {3,7}`:
   alle Profilebenen-Bytes (value/weight_sum/n_eff/support), alle Alpha-Maps,
   `alpha_confidence_support` und die drei Clipping-Zähler byte-identisch zu
   `workers == 1`; `enumerate_call_count()` bei jedem W > 0.
`-fopenmp` ist für `tile_compile_lib` aktiv (`flags.make`), die W=2/4-Läufe
sind also echte Parallelität.

**Parität:** Gesamtlauf **526/527, 1 431 204 Assertions** (+2 Fälle, +45
Assertions ggü. §30.66); einziger Fehlschlag weiterhin
`test_acceleration_backend.cpp:254` (GPU-umgebungsabhängig, vorbestehend).

**Noch offen (P3-seitig):** Wall-Speedup-Messung der Reduktion bei realer
Canvasgröße (die Unit-Fixtures sind zu klein für aussagekräftige Zeiten).
Runner-Scheduler → **§30.68**. Es gibt **keine** Budget-Division:
`plan_drizzle_memory` wird einmal mit dem vollen Budget aufgerufen, die Bänder
teilen dieselben Puffer. Die echte Worker-Zahl-Schranke ist `nb ≤ rows` — bei
Produktions-`chunk_rows` (Auto ≤ 256) kann eine 16-Kern-Box alle 16 Bänder
nutzen, ein klein erzwungenes `chunk_rows` deckelt die Parallelität still; das
ist ein Scheduler-Input, kein Speicherproblem. P5 (Resthotspot) danach. **P6
ist ein Benutzerlauf** — nicht autorisiert.

**Noch nichts committet.**

---

### 30.68 §11.14 P3 Teil 2 — Runner-Scheduler für die Reduktions-Worker (2026-09-09)

**Aufgabe:** die §30.67-Fähigkeit (`workers`-Parameter) als produktiven
Scheduler in den Runner integrieren; „gemeinsames Budget".

**„Gemeinsames Budget" — ehrliche Fassung.** Die §30.66-Residenztabelle plus
das Band-Design (§30.67) beantworten die Budgetfrage bereits, und die Antwort
ist: fast nichts zu budgetieren. Bänder teilen `A/B/QA*/candidates` → **null**
frameskalierender RAM pro Worker. Einziger Pro-Worker-Term: ein `std::ifstream`
+ ein `block`-Puffer in `enumerate_stripe`, begrenzt durch
`max_row_record_count · sizeof(LeafRecord)` (neuer Accessor
`DrizzleGeometryCacheReader::max_row_record_count()` liefert die Zahl direkt aus
dem Zeilenindex). Die Schranke ist damit **CPU**:
`min(parallel_workers, hardware_concurrency)`, danach pro Streifen `nb ≤ rows`.
Ein Solver über nachweislich-null-Terme wäre Theater.

**Verkabelung.** `int workers = 1` (überall Default) durch die Aufrufkette
gefädelt: `persist_multiband_store_from_predecessors` →
`persist_forward_drizzle_multiband` → `stream_forward_drizzle_uniform_and_raw[_2x2]`;
und `persist_forward_drizzle_from_predecessors` →
`persist_forward_drizzle_uniform_and_raw` → dito. **Kein Feld in
`config::ReconstructionDrizzleConfig`** — die Struct speist
`make_drizzle_store_identity`; eine Threadzahl im Identitätshash würde die
Byte-Identitätsprüfung über Worker-Zahlen sinnlos machen (vgl. §30.61:
`memory_budget` in `source_identity_hash`). Nur explizite Parameter.

**CUDA-Pfad.** `persist_forward_drizzle_multiband`s `cuda_stripe_path` hat
eigene Device-Band-Chunkung und erreicht `stream_*` nie → `workers` wirkt dort
**nicht**; bei einem CUDA→CPU-Neustart greift der CPU-Zweig ihn auf. Kommentar
im Code, plus `forward_drizzle_reduction_workers` im Checkpoint neben
`forward_drizzle_backend`, damit das Profil keine nicht-stattgefundene
Parallelität behauptet.

**Runner-Auflösung** (`run_forward_drizzle_stages`, vor `Phase::FORWARD_DRIZZLE`):
`fd_workers = clamp(cfg.runtime_limits.parallel_workers, 1, hardware_concurrency)`,
`TC_FORWARD_DRIZZLE_WORKERS` überschreibt (=1 = serielle Referenz; spiegelt
`TC_GEOMETRY_CACHE_WORKERS`). **`runner_pipeline.cpp:1237` bleibt unangetastet:**
es zwingt `parallel_workers=1` nur für *frische* `forward-drizzle-only`-Läufe
(Dev-Scope `forward_drizzle_m1_m3`) — dort auch NORM/REG seriell, ein
unabhängiges Verhalten, das hier nicht blind mitgeändert wird. Der Pfad, den
der Benutzer real fährt (`resume-reconstruction --from-phase FORWARD_DRIZZLE`,
`resume_forward_drizzle_command`), geht **nicht** durch `run_pipeline_command`
und kommt mit dem Konfigwert (Default 4) an → bekommt die Parallelität. `end`-
Extra bekommt `reduction_workers`.

**Profil-JSON-Ehrlichkeit.** Bei `fd_workers > 1` ist die geomstats-Registry
für die FORWARD_DRIZZLE-Streifenschleife deaktiviert → die
`production_uniform_raw` / `contrib_*` / `hybrid_*`-Zähler sind **null, weil
nicht aufgezeichnet**, nicht weil der Cache alles bedient hätte. Damit niemand
das als P1/P2-Beleg fehlliest, trägt
`forward_drizzle_geometry_profile.json` in dem Fall
`forward_drizzle_stage_stats_suppressed_reduction_workers: N` (plus
`geometry_cache_max_row_record_count` = der Pro-Worker-Lesepuffer-Budgetterm).
Der bestehende `[forward-runner][geometry-cache]`-P1/P2-Test pinnt jetzt
`TC_FORWARD_DRIZZLE_WORKERS=1`, damit seine Zähler-Assertions gültig bleiben.

**Test** (`test_runner_forward_drizzle.cpp`,
`[forward-runner][geometry-cache][geometry-parallel]`): `LocalWarpFixture`
(MONO, 3 Local-Frames, Mehrband) bei `TC_FORWARD_DRIZZLE_WORKERS ∈ {1,2,4}` —
die committeten Profilebenen-`.fits` (nicht `current.json`, dessen
uhrbasierter Generationenname pro Lauf variiert) byte-identisch über alle
Worker-Zahlen; Checkpoint führt `forward_drizzle_reduction_workers`;
Suppression-Notiz fehlt bei W=1, vorhanden bei W>1. Grün bei
`OMP_NUM_THREADS ∈ {1,2,4,8}`.

**Parität:** Gesamtlauf **527/528, 1 431 221 Assertions** (+1 Fall ggü.
§30.67); einziger Fehlschlag weiterhin `test_acceleration_backend.cpp:254`
(GPU-umgebungsabhängig, vorbestehend).

**Commit-Stand:** §30.62–§30.67 (P0–P4 + P3 Teil 1/2) wurden vom Benutzer als
`1f9b5306` + `33ee3288` committet; dieser §30.68-Schnitt (Runner-Scheduler,
9 Dateien) ist noch uncommittet.

**Noch offen (P3-seitig):** nur noch Wall-Speedup-Messung bei realer Canvas —
gehört zum §11.14-P6-Benutzerlauf. Damit ist der P3-Code (Teil 1
Build-Parallelität + Teil 2 Reduktions-Bänder + Scheduler) vollständig; P5
(Resthotspot) als Nächstes. **P6 ist ein Benutzerlauf** — nicht autorisiert.

**Noch nichts committet.**

---

### 30.69 §11.14 P5 (Teil 1) — Resthotspot-Profil nach P2/P3 (2026-09-09)

**Aufgabe (Plan §11.14.7):** den verbleibenden Hotspot messen. **Nur
profilieren** — dominiert die einmalige lokale Basisauswertung weiterhin, dann
*untersuchen*: exakte Wiederverwendung identischer Prüfpunkte + SIMD **vor**
Minimax/GPU. Minimax/GPU braucht eine eigene Numerik-/Modellidentitätsrevision.
Kein Lockern von Gate-/Profilgrenzen.

**Messwerkzeug:** neuer `tests/test_forward_drizzle_local_basis_profile.cpp`
(`[geometry-p5]`, reine Messung, nur Sanity-`REQUIRE`s). Voller
`sample_leaves`-Sweep über eine 96²-Quelle für einen Local-Warp-Frame, je
Variante (`pixfrac 0.8` = cfa / `pixfrac 1.0` = footprint) und je
Krümmungsregime (`curv` 0/3.5/12 skaliert die Höherordnungs-RBF-Koeffizienten).
Mikrobenchmarks (Min aus 5): `std::exp(float)` / `expf()` / `std::exp(double)`,
ein voller `evaluate_smooth_local_displacement`. **`perf` ist in dieser
Umgebung nicht verfügbar** → die Term-Aufteilung ist mikrobenchmark-abgeleitet
(Heißschleifen-Untergrenze je Aufruf), **kein Profiler-Self-Time-Split**.

**Kette:** `sample_leaves` → `subdivide_local` (Tiefe 0: **3×3-Gitter** =
9 `local_forward`) → `invert_local_source_to_canvas` (Newton, je Schritt ein
`smooth_local_basis`) → `smooth_local_basis` = **16 `std::exp`**
(`kSmoothLocalGridSize=4`, `Coefficients = Eigen::Matrix<float,16,1>`).

**Ergebnisse (dieser Box):**

| Regime | mean Newton-Iter | local_forward/Sample | exp/Quell-px | ns/`sample_leaves` (cfa / fp) | leaves/Sample | subdiv/Sample |
|---|--:|--:|--:|--:|--:|--:|
| near-linear (curv 0) | 1,93 | 9,00 | 277 | 2170 / 2166 | 1,000 | 1,000 |
| curved (curv 3,5) | 2,68 | 9,00 | 386 | 3721 / 2982 | 1,000 | 1,000 |
| strongly-curved (curv 12) | 3,65 | 9,00 | 526 | 4123 / 4110 | 1,000 | 1,000 |

- **Keine Rekursion in irgendeinem Regime.** `subdiv/Sample = 1,000`,
  `leaves/Sample = 1,000`, `local_forward/Sample = 9,00` **exakt**, selbst bei
  curv 12. Grund: das 4×4-RBF hat `kSigma = 0,28` in **normierten**
  Gitterkoordinaten → das Feld ist über *jedes* einzelne Quellpixel
  quasi-linear, unabhängig von der Koeffizientengröße; die Krümmung über 1 px
  bleibt unter `position_epsilon_internal_px = 0,05`. Das ist **kein
  Kleinstufen-Artefakt**, sondern strukturell — und deckt sich mit **jeder**
  bisherigen Messung (§30.55 M42, §30.64-Leiter: alle leaves/Sample = 1).
- ⇒ **Exakte Prüfpunkt-Wiederverwendung bringt fast nichts.** cfa-Variante:
  Tiefe-0-Proben sind pixel-**intern** (`sx+0,1 / sx+0,5 / sx+0,9` bei
  pixfrac 0,8) → **null** Nachbar-Sharing; Parent↔Child-Reuse nur bei
  Rekursion, die nie eintritt → Ceiling **1,0×**. footprint-Variante:
  Tiefe-0-Gitter kachelt das Halbinteger-Lattice → Ceiling **2,25×**, aber die
  kleinere Variante und ein Lattice-Memo nötig. Für den Produktionspfad (cfa):
  nichts zu holen.
- **`std::exp` läuft bereits über den schnellen `float`-Pfad:**
  `std::exp(float)` = **2,97 ns**, `expf()` = 2,98 ns, `std::exp(double)` =
  5,31 ns → keine Promotion-Regression, das war eine mögliche Erklärung und ist
  **ausgeschlossen**.
- **`std::exp` ist NICHT der dominante Einzelterm, aber auch nicht klein.**
  Attribution (mikrobenchmark-abgeleitet, Untergrenzen): `std::exp` **≥ 31–38 %**
  der `sample_leaves`-Zeit; voller `smooth_local_basis` (16 exp +
  `Coefficients::Zero()` + 16 Stores/`sum+=` + Taper + `basis*=taper/sum` +
  Rückgabe per Wert + 2 `.dot()` im Aufrufer) **≥ 64–80 %**; **Residuum**
  (9× `local_forward`-Aufruf-Overhead + Newton-Steuerung + `subdivide_local`
  Bilinear-Fehlerprobe + 5× `shoelace_area` + Leaf-Bookkeeping) **~20–36 %**.
  Ein einzelner Basis-Eval = **~98–99 ns** (stabil über alle Regime), bare-exp-
  Anteil daran **48 %** → die anderen ~52 ns sind Eigen-/`.dot()`-/Taper-/
  Rückgabe-Overhead **um** die 16 exp herum.
- **`ns/sample_leaves` ≈ 2170** (near-linear) deckt sich mit der
  §30.64-Leiterprojektion (~2150 ns/Sample). Bei 600 f @ 3840×2160 × 2
  Varianten ⇒ ~2,3·10¹³ ns ≈ **~23 000 s einthreadig**, mit P3 (~16 Kerne)
  **~1440 s** — der echte einmalige Bauaufwand, ~75 % des §11.14-P6-Ziels von
  ≤1920 s für die *gesamte* Kette. Realer Angriffspunkt.

**P5-Teil-1-Befund / Empfehlung:**
1. **„Identische Prüfpunkte" ist erledigt-durch-Messung** — kein Nutzen für
   den Produktionspfad (kein Sharing ohne Rekursion, Rekursion tritt beim
   glatten RBF nicht auf).
2. **Bit-exakte Chance = der Nicht-exp-Anteil des Basis-Evals + die
   Aufrufkette.** ~52 ns von 99 ns pro Basis-Eval sind Eigen-`Matrix<float,16,1>`-
   Verkehr (`Coefficients::Zero()`, Rückgabe von 16 Floats per Wert,
   `basis*=taper/sum`, 2× `.dot()` im Aufrufer) plus die 5-stufige nicht-inlinte
   Kette `local_forward → invert_local_source_to_canvas →
   local_displacement_render_units → evaluate_smooth_local_displacement →
   smooth_local_basis` (`cv::Point2f`-Rückgabe, wiederholte `isfinite`-Checks).
   Angriff: `float basis[16]` statt Eigen, Rückgabe per Out-Param, `.dot()` als
   einfache Schleife, Kette abflachen/inlinen — **mit sequentieller
   `sum += value`-Reduktion** (die Summe teilt jeden Koeffizienten → umsortierte
   Reduktion ist **nicht** bit-exakt). Realistische Erwartung **~1,3–1,5×**
   (nicht 1,5–2× — das Residuum + die 38 % `exp` bleiben). Byte-Identität via
   Manifest-`rows_sha256`/`leaves_sha256` der Cache-Build-Tests.
3. **Ein schnelleres `exp`** (Minimax/Vektor-`expf`) trifft die ~38 %, ändert
   aber Bits → gehört in die eigene Numerikrevision (Plan: „Minimax oder voller
   GPU-Lokalpfad braucht eine eigene … Revision"), **nicht** in P5 Teil 1.
4. `smooth_local_basis` lebt in `src/registration/global_registration.cpp` —
   **nicht** auf der `-ffp-contract=off`-Liste. Vor jeder Änderung dort muss die
   TU auf die Liste (CMakeLists), sonst prüft das Byte-Identitäts-Gate ein
   bewegliches Ziel.

**P5 Teil 2 — erster Schritt VOR jedem Quellumbau:** Die 5-stufige Kette
kreuzt eine TU-Grenze (`local_forward` / `invert_local_source_to_canvas` in
`registration_sampling_plan.cpp` → `evaluate_smooth_local_displacement` gibt
`cv::Point2f` über die Grenze zu `global_registration.cpp` zurück). Zuerst mit
`-flto` bauen (oder die Kette temporär als `inline` in einen Header ziehen) und
`[geometry-p5]` erneut messen. Fällt `ns/sample_leaves` deutlich → der Gewinn
ist reines Inlining/LTO, **kein Quellumbau nötig**. Bleibt es stehen → die
Kosten liegen wirklich im Eigen-/`.dot()`-Verkehr und der Umbau (Punkt 2) ist
gerechtfertigt. ~10 min, kann den Rewrite überflüssig machen.

**Noch nichts committet.** Kein Produktionscode geändert (nur neuer Test +
CMake-Eintrag).

---

<a id="historie-28"></a>

## 28. Empfohlener erster Implementierungsschnitt

Der erste konkrete Code-Schnitt umfasst ausschließlich M0 bis M2:

1. Single-Method-Konfigurationsvertrag ohne Engine-Branch und mit
   `pipeline_contract_version`;
2. `RegistrationSamplingPlan` inklusive Artefakt und Tests;
3. Entkopplung der Registration vom PREWARP;
4. direkte geometrische Support-/`n_eff`-/Loch-Coverage und Ditherdiagnostik;
5. normalisierte CFA-Quelle mit Cachemetadaten;
6. CPU-Forward-Drizzle bei `internal_scale=1` nur als Uniform-Control;
7. Flux-, Zentroid-, Bayer-, Masken-, Chunk- und Determinismustests.

Dieser Schnitt enthält noch keine Q-Gewichtung, keine Mehrbandfusion, keine
CUDA-Implementierung, keinen realen Run und noch keine physische Löschung der
test-only Vergleichsimplementierungen. Er etabliert die geometrische und
speichertechnische Basis des späteren verlustarmen Produktpfads, ist aber noch
kein freigabefähiger vollständiger Produktpfad. Erst wenn dieser Vertrag korrekt
und regressionsfrei ist, folgen Raw-Forward-Drizzle, 2x, Mehrband,
Qualitätsnachweis und abschließend M10/M11.

---
