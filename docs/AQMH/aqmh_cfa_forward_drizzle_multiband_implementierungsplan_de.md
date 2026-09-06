# CFA-Forward-Drizzle und Mehrband-Rekonstruktion als einzige Methodik

## Detaillierter Implementierungsplan

**Status:** Implementierung begonnen; M0/M1 teilweise umgesetzt, nicht abgenommen; M2-Uniform-Prototyp und M3-Bausteine vorhanden; Runner-/Store-Integration und weitere Meilensteine offen.  
**Datum:** Statusaudit 2026-09-05; Entwurf und Entscheidungsrevisionen ab 2026-09-02 (Abschnitte 29 bis 31).  
**Implementierungsaudit:** Der [Code- und Vertragsaudit vom 2026-09-05](aqmh_cfa_forward_drizzle_audit_2026-09-05_de.md) korrigiert die bisherigen Abnahmebehauptungen und legt die nächsten Arbeitspakete fest. Die nachfolgende Umsetzung korrigiert Coverage-Geometrie, geometrisches `n_eff`, Maskenbezug und lokale Subdivision und führt begrenztes Streaming ein; aktueller Vertrag siehe Abschnitt 0. Historische Fortschrittsnotizen und Häkchen unten sind zusammen mit diesen Korrekturen zu lesen; sie ersetzen keine aktuelle Abnahme.  
**Zielcodebasis:** `tile_compile_cpp` (C++20, OpenCV, Eigen, optionale CUDA-Beschleunigung)  
**Zielzustand:** ausschließlich CFA-Forward-Drizzle mit kontrollierter Mehrband-Rekonstruktion; Classic Tile Compile und das bisherige PREWARP-AQMH werden entfernt  
**Primäres Ziel:** den gegen Siril gemessenen Schärfeverlust im linearen Stack durch eine einzige verlustarme Rekonstruktion aus den normalisierten CFA-Quelldaten beseitigen  

---

## 0. Korrekturvertrag und Streaming-Implementierung (2026-09-05)

Dieser Abschnitt ersetzt widersprechende Aussagen und Abnahmebehauptungen in
älteren Fortschrittsnotizen (§30); die grundlegenden M3–M11-Arbeitspakete bleiben
bestehen. Er ist der aktuelle Geometrie-/Speichervertrag für M1/M2.

- Coverage und Uniform rasterisieren denselben exakt transformierten Square-
  Droplet-Kernel. Geometrisches `n_eff` basiert auf den Frame-Flächensummen.
- Die Analysefläche stammt aus dichten, mit `pixfrac=1` und unabhängig vom
  CFA-Kanal rasterisierten Frame-Footprints. Der konfigurierte Überlappungsanteil
  bezieht sich auf diese Footprints. Kanalsupport und Löcher werden anschließend
  darin geprüft; sie bestimmen nicht selbst ihren Prüfbereich.
- Alle lokalen Blätter müssen auch an Maximaltiefe beide Konvergenzkriterien
  erfüllen. Eine zusätzliche Prüfunterteilung liefert den Flächenvergleich.
  Ein fehlgeschlagenes Subdroplet verwirft das ganze Quellsample; die Framequote
  zählt jedes betroffene Quellsample genau einmal. Frameausschlüsse stehen vor
  der ersten Ausgabe fest und sind von der Chunkhöhe unabhängig.
- OpenCV-Zentren werden mit `t_edge=t_cv+(0.5,0.5)-A*(0.5,0.5)` adaptiert;
  lokale Modelloffsets erhalten entsprechend den halben Pixeloffset.
- Arbeitsdaten sind Zielstreifen; Auto wählt maximal 256 Zeilen innerhalb des
  Budgets. Exakte Quellfootprint-Aufzählung erfasst sämtliche Beiträge über
  Streifengrenzen, sodass kein duplizierter Ausgabe-Halo erforderlich ist.
- Das Budget umfasst zurückgehaltene Ergebnisse, Quell-/Ladepuffer, Streifen und
  Reserve. Explizite zu große Chunks werden abgelehnt. Der CPU-Referenzpfad ist
  einsträngig; keine Vollbildpuffer pro Worker. Host-/cgroup-Headroom begrenzt
  zusätzlich die Allokation, ersetzt aber keine Gesamtprozess-RSS-Messung.
- Die Uniform-Preview nutzt einen Streaming-Summensink. Coverage behält nur zwei
  Bytemasken, schreibt exakte Perzentildaten auf temporäre Disk und sucht Löcher
  mit zwei Scanlines. FITS-Maskenexport konvertiert nur eine Zeile auf einmal.
- Neue Geometrieartefakte verwenden Schema 2, kanonische Hashes und atomare
  Veröffentlichung. Unvollständige/beschädigte Sampling-Pläne werden abgelehnt.
  Das allein erlaubt noch kein Resume: normalisierte Cache-Manifeste,
  Profilstores und die vollständige neue Runnerphasenfolge bleiben nötig.
- Mehr Frames verbessern eine Vereinigungsdeckung, können aber den Schnitt
  aller Frame-Supportmasken nicht vergrößern. Die frühere Empfehlung, eine leere
  All-Frames-CFA-Maske durch mehr Frames zu heilen, ist zurückgenommen.
- Die 19–25-%-Schärfeprognose ist eine Hypothese. Gleiche Elongation schließt
  isotrope Registrierungsresiduen nicht aus; quadratische FWHM-Differenzen und
  Quincunx-Fits liefern ohne Bias-/PSF-Nachweis keine gesicherten Einzelbeiträge.
  Abnahme benötigt unabhängige pixelintegrierte Truth-Fixtures und später die
  ausdrücklich angeforderte M9-Matrix auch für den ausgelieferten 2/1-Pfad.
- Das Legacy-Referenzbinary wird standardmäßig nicht gebaut. Explizit gebaute
  Referenzrunner dürfen nur neue temporäre Ausgabebäume erstellen und kein
  Resume ausführen. Gemeinsam kompilierte Quellen sind nicht „eingefroren“.

M0/M1/M2 werden dadurch nicht pauschal als vollständig abgenommen markiert.
Insbesondere ist die Geometriefunktion weiterhin in die alte Runnerstruktur
integriert; ein Funktionsname ist kein fertiger Event-/Resumevertrag.

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

## 1. Verbindliche Entscheidung

Die weitere Entwicklung konzentriert sich auf zwei zusammengehörige Änderungen,
die nach dem Cutover die einzige Rekonstruktionsmethodik bilden:

1. **CFA-aware Forward-Drizzle:** normalisierte Bayer-Samples werden ohne
   Debayer- und PREWARP-Interpolation direkt auf das gemeinsame Zielraster
   projiziert.
2. **Kontrollierte Mehrband-Rekonstruktion:** niedrige Frequenzen stammen aus einer robusten
   Uniform-Rekonstruktion; mittlere und hohe Frequenzen werden aus
   qualitätsselektiven Drizzle-Profilen übernommen.

Nicht Bestandteil dieses Plans sind:

- weitere Varianten von linearer, kubischer oder Lanczos-PREWARP-Interpolation;
- Unsharp Mask, Sternkernschärfung oder andere klassische
  Nachschärfungsalgorithmen;
- ein weiterer Siril-Registrierungs-Kreuztest zur Ursachenbestimmung;
- eine Übernahme oder Vermischung von Classic-Tile-Gewichten mit der neuen
  Rekonstruktion;
- gelockerte Sicherheitsgates, um ein nominell schärferes Ergebnis zu
  erzwingen;
- KI-basierte oder generative Detailrekonstruktion.

Für neue Runs gibt es nach dem Cutover keinen Methoden- oder Engine-Schalter.
Der Runner führt ausschließlich den CFA-Forward-Drizzle-Pfad aus. Classic Tile
Compile und das bisherige PREWARP-AQMH dürfen weder als auswählbare Methode noch
als stiller Qualitätsfallback erhalten bleiben.

Während der Implementierungs- und Beweisphase dürfen die alten Pfade in einem
klar abgegrenzten Übergangsstand noch kompilierbar sein, damit objektive
Regressionstests möglich bleiben. Diese Koexistenz ist zeitlich begrenzt und
endet mit dem verbindlichen Löschmeilenstein. Sie ist kein Bestandteil der
Zielarchitektur.

Interne Sicherheitsvarianten der neuen Methodik bleiben verpflichtend:

- Uniform-Control aus denselben Forward-Drizzle-Samples;
- unveränderliche Raw-Forward-Drizzle-Qualitätsrekonstruktion;
- gegateter Mehrbandkandidat;
- CPU-Referenzpfad als semantischer Fallback für CUDA-Fehler.

Diese Varianten sind keine eigenständigen Benutzermethoden und erzeugen keine
Engine-Auswahl.

### 1.1 Transaktionaler Rollout

M0 bis M9 sind ein nicht freizugebender Implementierungs- und Beweisstand. Die
alten Rekonstruktionen dürfen dort ausschließlich über ein separates,
standardmäßig deaktiviertes Testtarget wie
`tile_compile_legacy_reference_tests` erreichbar sein. Dieses Target wird nicht
installiert, nicht vom Backend aufgerufen und besitzt keinen Resume- oder
Schreibzugriff auf Benutzer-Runs. Es dient nur zur Erzeugung reproduzierbarer
Vergleichswerte.

Kein Zwischenstand wird als neuer Produktionsrunner ausgeliefert. M10 ist ein
atomarer Release-Cutover: neue Methodik aktivieren, alte Produktpfade aus allen
Produkt-Targets entfernen, vollständige Suite ausführen und erst danach
freigeben. Damit gibt es weder eine dauerhaft doppelte Architektur noch einen
Zeitraum, in dem eine unvollständige neue Pipeline Benutzer-Runs verarbeitet.

Ausdrücklich in Kauf genommen wird, dass der aktive Runner auf dem
Entwicklungszweig in M0 und M1 keinen lauffähigen Rekonstruktionspfad besitzt;
er muss in diesem Zustand vor der ersten Run-Mutation mit einem stabilen
`PIPELINE_UNAVAILABLE_DURING_CUTOVER`-Fehler abbrechen. Ab M2 ist der
Uniform-1x-Pfad technisch lauffähig, ab M3 zusätzlich Raw, beide bleiben aber
nicht freigegebene Beweisstände. Bis einschließlich M8 werden ausschließlich
Tests und das Legacy-Referenztarget ausgeführt; reale Qualitätsläufe finden nur
mit ausdrücklicher Benutzeranforderung in M9 statt.

Das test-only Legacy-Target heißt einheitlich
`tile_compile_legacy_reference_tests` (auch in Abschnitt 25.11). Sein
physisches Löschdatum ist nicht M10, sondern der darauf folgende Release-Zyklus
(Abschnitt 25.11); M10 entfernt es aus allen Produkt-Targets und aus der
Default-Konfiguration des Builds.

### 1.2 Namens- und Namespace-Vertrag

„Bisheriges AQMH“ bezeichnet in diesem Dokument den aktuellen
PREWARP-basierten Rekonstruktionspfad. Dessen Runnerphase, Cachevertrag,
Postprocessing und auswählbare Methodik werden entfernt. Mathematisch weiterhin
benötigte lokale Qualitätsgrößen werden als internes Quality-Subsystem der
neuen Rekonstruktion übernommen; ihre Herkunft macht sie nicht zu einer zweiten
Methode.

Der öffentliche Konfigurationsroot heißt nach dem Cutover `reconstruction`.
Die bisherigen benutzersichtbaren Roots und Auswahlwerte für `classic` und
`aqmh` werden entfernt. Neue Phase-, Artefakt- und Reportnamen verwenden
`quality`, `forward_drizzle` und `multiband`, sofern keine zwingende historische
Lesekompatibilität besteht. Verbleibende `aqmh_*`-Dateinamen dürfen nur als
explizit dokumentierte Artefaktnamenmigration mit Schema-Version existieren und
müssen vor M10 entweder neutral umbenannt oder als read-only Legacyname
klassifiziert werden.

---

## 2. Ausgangsbefund und technische Problemdefinition

Beim untersuchten M31-Datensatz wurden identische Sternpositionen über die WCS
beider Ergebnisse verglichen. Der Median der gematchten linearen Grünkanal-FWHM
lag bei etwa 3,15 px für Siril und 4,74 px für AQMH. Der Unterschied ist bereits
in `stacked_rgb.fits` vorhanden; Plate Solve, BGE und PCC verändern ihn praktisch
nicht. HMS verbreitert die Sterne zusätzlich, verursacht aber nicht die
ursprüngliche lineare Differenz.

Der aktuelle AQMH-Datenpfad ist für OSC im Wesentlichen:

```text
normalisiertes CFA
  -> Debayer pro Frame
  -> geometrischer PREWARP pro RGB-Kanal
  -> Luminanzbildung für Q-Maps
  -> pixelweise AQMH-Rekonstruktion
  -> optionale Detail-/Schärfungskandidaten
```

Damit entstehen vor der eigentlichen AQMH-Rekonstruktion mindestens zwei
detailverändernde Operationen:

1. die räumliche Interpolation des Bayer-Musters beim Debayer;
2. die geometrische Interpolation jedes debayerten Kanals beim PREWARP.

AQMH gewichtet anschließend bereits interpolierte Werte. Frequenzen, die durch
Debayer und PREWARP abgeschwächt wurden, können durch eine andere Gewichtung
nicht zuverlässig rekonstruiert werden. Die bisherigen Validierungsartefakte
zeigen außerdem, dass die bisherige Raw-AQMH-Ausgabe gegenüber dem
Uniform-Control nur einen kleinen FWHM-Vorteil erreicht. Nachträgliche
Sternkernschärfung erhöhte dagegen
Hintergrund, Seam-Werte und Star-Tails deutlich und wurde deshalb korrekt
abgelehnt.

Die neue Rekonstruktion muss daher vor der ersten geometrischen Interpolation
ansetzen.

### 2.1 Empirische Reproduktion und Stufen-Bisektion (2026-09-01)

Der Ausgangsbefund wurde unabhängig nachgemessen (Grünkanal, Stern-Detektion,
elliptischer 2D-Gauß-Fit, WCS-gematchte Sternpaare). Ergebnisse:

**M31, linearer Stack, gematchte Paare (n = 468):**

| Größe | tile_compile | Siril | Verhältnis |
|---|---:|---:|---:|
| FWHM Median | 4,28 px (12,8″) | 3,06 px (9,1″) | **1,40** |
| Elongation Median (maj/min) | 1,19 | 1,18 | ~1,0 |

Die Differenz aus dem Ursprungsbefund ist damit reproduziert (Dokumentzahl
4,74 / 3,15 = 1,50; hier 4,28 / 3,06 = 1,40 — anderer Lauf und Fitter, gleiche
Größenordnung). Angegeben ist jeweils der Quotient der Mediane; der Median der
paarweisen Quotienten kann geringfügig abweichen.
Im HMS-Stadium schrumpft das Verhältnis auf ~1,16; die von HMS gelieferten
Dateien unterschätzen das Problem, der lineare Vergleich ist maßgeblich.

Die ähnliche Elongation ist mit einem isotropen Resamplingverlust vereinbar.
Sie schließt isotrope Registrierungsresiduen jedoch nicht aus. Die folgende
Stufen-Bisektion begrenzt einen möglichen zusätzlichen Stackbeitrag empirisch;
sie ist kein Beweis einer fehlerfreien Registrierung.

**Stufen-Bisektion (zunächst M16, `prewarped_rgb`-Cache vorhanden, identisches
Zielraster, n = 606 bzw. 12 Einzelframes; später auf M31 und M42 wiederholt,
siehe Tabelle weiter unten):**

| Stufe | Grünkanal-FWHM Median | Delta |
|---|---:|---:|
| einzelner prewarpter Frame (Median über 12 Frames) | 3,79 px | — |
| einfacher Mittelwert-Stack aller 222 prewarpten Frames | 3,78 px | +0 % ggü. Einzelframe |
| AQMH-Rekonstruktion (`aqmh_reconstructed_raw.fit`) | 3,97 px | +5 % ggü. Mittelwert |

Interpretation:

- Das Stapeln selbst verbreitert **nicht** (3,79 → 3,78 px). Ein dominanter zusätzlicher Stackbeitrag ist in diesem Medianvergleich nicht
  sichtbar; kleine oder isotrope Registrierungsresiduen bleiben möglich.
- Die AQMH-Qualitätsgewichtung/Rekonstruktion trägt nur **rund 5 %** bei
  (3,79 → 3,97 px).
- Der gesamte Rest — der ~3,8-px-Boden — steckt bereits im einzelnen prewarpten
  Frame, also in **Debayer-Interpolation + geometrischer Warp-Interpolation +
  physikalischem Seeing-/Optik-Boden**.

**Aufschlüsselung des Per-Frame-Bodens — drei Objekte (Grünkanal,
Verteilungsmedian; Stufe (1) fittet nur die nativen Grün-Pixel des
Bayer-Quincunx ohne jede Interpolation; M31 und M42 mit neuen Läufen bei
`delete_prewarped_cache_after_run: false`, 2026-09-03):**

| Stufe | M16 | M31 | M42 |
|---|---:|---:|---:|
| (1) native Grün-Samples, **keine Interpolation** | 3,00 px | 3,13 px | 3,77 px |
| (2) bilinear debayert, **ungewarpt** | 3,27 px | 3,51 px | 4,05 px |
| (3) prewarpt (edge-aware Debayer + Warp), Einzelframe | 3,79 px | 4,25 px | 4,59 px |
| (3) prewarpt, Mittelwert-Stack aller Frames | 3,78 px | 4,17 px | 4,63 px |
| (final) AQMH-Rekonstruktion (`aqmh_reconstructed_raw.fit`) | 3,97 px | 4,24 px | 5,02 px |

Heuristische quadratische Zusatzbeiträge (nur unter passenden PSF-/Faltungsannahmen als getrennte Beiträge interpretierbar):

| Beitrag | M16 | M31 | M42 |
|---|---:|---:|---:|
| Debayer-Interpolation ((2) vs (1)) | ≈ 1,30 px | ≈ 1,59 px | ≈ 1,48 px |
| Warp-Interpolation ((3) vs (2)) | ≈ 1,92 px | ≈ 2,24 px | ≈ 2,24 px |
| AQMH-Gewichtung ((final) vs (3)) | ≈ 1,18 px | ≈ 0,80 px | ≈ 1,95 px |
| **Stapeln (Einzelframe → Mittelwert)** | **≈ 0** | **≈ 0** | **≈ 0** |

Durchgängige Befunde:

- **Das Stapeln verbreitert nicht** (Einzelframe ≈ Mittelwert-Stack auf allen
  drei Objekten). Die Gleichheit der Mediane liefert keinen Hinweis auf einen dominanten
  zusätzlichen Stackbeitrag, widerlegt aber nicht jede Art von Registrierungsresiduum.
- **Die Warp-Interpolation ist der größte Einzelbeitrag** (~1,9–2,2 px
  quadratisch), gefolgt von der Debayer-Interpolation (~1,3–1,6 px). Genau diese
  beiden Stufen entfallen bei CFA-Forward-Drizzle.
- Die AQMH-Gewichtung trägt objektabhängig 0,8–2,0 px bei (M42 am stärksten).
- **Hypothese zum Potenzial von CFA-Forward-Drizzle: ~0,8–1,0 px
  linear, also ~19–25 % der aktuellen Per-Frame-FWHM** (M16 3,79 → ~3,0;
  M31 4,17 → ~3,1; M42 4,63 → ~3,8), plus der Gewichtungsanteil.
- Konsistent mit dem Siril-Endpunktvergleich weiter unten: tile_compile liegt
  auf 4 von 5 Objekten 31–44 % über Siril, das die doppelte Resampling-Kette
  ebenfalls vermeidet.

**Vorbehalte:** Stufe (2) verwendet einfaches Bilinear statt des im Produkt
genutzten edge-aware Verfahrens (realer Debayer-Anteil eher etwas kleiner,
Warp-Anteil entsprechend größer). Stufe (1) fittet auf dem halbdichten Grün-Quincunx. Ohne unabhängigen Biasnachweis
ist dies keine garantierte Obergrenze des interpolationsfreien Bodens; das
genannte Potenzial ist noch keine konservativ abgesicherte Gewinnzusage. Forward-Drizzle mit
`pixfrac < 1` bei 2x führt einen eigenen kleinen Kernel und korreliertes
Rauschen ein; der native Boden wird nicht vollständig erreicht.

**Endpunktvergleich gegen Siril auf weiteren Objekten (2026-09-02).** Für M42,
IC434, M66 und IC5070 wurden mit Siril 1.4.4 lineare Referenzstacks aus
denselben Rohframes erzeugt (Rezept in Abschnitt 2.2), per lokalem Gaia-DR3-
Solver plate-solved und WCS-gematcht mit dem jeweiligen tile_compile-
`stacked_rgb.fits` verglichen. Grünkanal, gematchte Sternpaare:

| Objekt | gematchte Paare | tile_compile | Siril | Verhältnis tc/Siril | Elong. tc | Elong. Siril |
|---|---:|---:|---:|---:|---:|---:|
| M31    | 468 | 4,28 px | 3,06 px | **1,40** | 1,19 | 1,18 |
| M42    | 345 | 5,04 px | 3,49 px | **1,44** | 1,06 | 1,11 |
| M66    | 189 | 4,22 px | 3,13 px | **1,34** | 1,10 | 1,04 |
| IC434  | 274 | 4,90 px | 3,73 px | **1,31** | 1,08 | 1,06 |
| IC5070 | 399 | 5,63 px | 6,29 px | **0,90** | 1,27 | 1,29 |

**4 von 5 Objekten** zeigen tile_compile-Sterne 31–44 % breiter als Siril,
weitgehend isotrop (Elongation beidseitig ~1,05–1,20). Das ist objektübergreifend
konsistent und stützt die Prämisse deutlich über M31 hinaus.

**Ausnahme IC5070:** Hier ist Siril 10 % *schlechter* als tile_compile, und
**beide** Stacks sind stark elongiert (~1,28). Dieser Datensatz hat ein echtes
anisotropes Problem (Feldrotation/Drift/Tracking); Sirils globale Registrierung
kam damit schlechter zurecht als tile_compile (das zusätzlich Cherry-Pick
anwendet und Siril nur 398 von 466 Subs verwendete). Forward-Drizzle behebt den
anisotropen Anteil nicht — solche Läufe gehören in die M9-Matrix, aber mit
realistischer Erwartung, und IC5070 taugt nicht als Schärfe-Referenzfall.

Hinweis: die Siril-Stacks sind ohne Farbkalibrierung (kein `pcc`/`spcc` —
offline, VizieR nicht erreichbar, lokaler Photometriekatalog nicht installiert)
und zeigen daher einen sensor-nativen Magenta-Stich (R/G ≈ B/G ≈ 1,37). Das ist
rein kosmetisch; die Grünkanal-FWHM ist davon unberührt.

**Schlussfolgerung für den Plan:** Die Kernprämisse hält, und der Gewinn ist
beziffert. Der FWHM-Verlust entsteht **vor** der AQMH-Rekonstruktion, in der
Per-Frame-Interpolation (~1,9 px Warp-, ~1,3 px Debayer-Beitrag in Quadratur),
und genau diese Stufe entfällt bei CFA-Forward-Drizzle. Die früher geäußerte
Vermutung „per-Frame-Registrierungsresiduum" ist für M16 widerlegt (Stapeln
verbreitert nicht, Sterne nicht elongiert).

**Stand der Absicherung:** Der Endpunktvergleich gegen Siril liegt für fünf
Objekte vor (4 bestätigen, IC5070 ist ein anisotroper Sonderfall). Die
**Stufen-Aufschlüsselung** (native → bilinear → prewarpt → AQMH) ist jetzt für
**drei** Objekte durchgeführt (M16, M31, M42) — mit durchgängig gleichem Muster:
Stapeln ≈ 0, Warp größter Beitrag, ~19–25 % der Per-Frame-FWHM
interpolationsbedingt rückgewinnbar. Vor M10 verbleibt: Bestätigung über die
synthetischen Fixtures aus Abschnitt 21 (Interpolationsanteil per Konstruktion
bekannt) und mindestens ein untersampelter Datensatz (native FWHM nahe 2 px),
bei dem der 2x-Drizzle-Gewinn am größten sein sollte. Das ist ein
M9-Releasegate, kein M0–M2-Blocker.

**Konsequenz für das Go/No-Go-Gate (Abschnitt 3.2, Zeile 1):** Der
PREWARP-AQMH-Referenzstack und der neue Forward-Drizzle-Stack müssen für den
10-%-Vergleich bei **identischem `output_scale`** gerendert werden. Der bisher
gemessene tile_compile-Lauf liegt bei 1x; ein 2x-Forward-Drizzle-Ergebnis gegen
eine 1x-Baseline zu vergleichen würde das Gate allein durch die verdoppelte
Ausgabeauflösung bestehen lassen.

### 2.2 Reproduzierbarkeit der Siril-Referenzstacks

Die Siril-Vergleichsstacks für M31, M42, IC434, M66 und IC5070 wurden mit
**Siril 1.4.4** (`siril-cli`) aus denselben Dwarf-II-Rohframes erzeugt, die auch
der jeweilige tile_compile-Lauf konsumiert hat. Keine Dark-, Flat- oder
Bias-Master (die tile_compile-Läufe verwenden ebenfalls keine). Kein Drizzle,
lineare 32-bit-Ausgabe in nativer Sensorgeometrie (3840×2160), Bayer `GBRG` aus
dem Header.

Aufruf pro Objekt:

```bash
siril-cli -d "<quellverzeichnis_mit_rohframes>" -s "<objekt>.ssf"
```

Skript `<objekt>.ssf` (Platzhalter `<workdir>` und `<basis>` beim Lauf
ersetzt):

```text
requires 1.2.0
link light -out=<workdir>/process
cd <workdir>/process
calibrate light -debayer
register pp_light
stack r_pp_light rej 3 3 -norm=addscale -output_norm -out=<basis>/<objekt>_siril
close
```

Damit ergibt sich:

- `link` — symlink-Sequenz `light_*` aus allen FITS des Quellverzeichnisses
  (keine Kopie, kein zusätzlicher Speicher);
- `calibrate light -debayer` — reines Debayern nach RGB (GBRG), 32-bit;
  ohne Master nur Formatkonvertierung;
- `register pp_light` — globale Sternregistrierung (Standard, Homographie);
- `stack r_pp_light rej 3 3 -norm=addscale -output_norm` — Winsorized Sigma
  Clipping mit `low = high = 3.0`, additiv-skalierende Eingabe-Normalisierung,
  aktivierte Ausgabe-Normalisierung, Durchschnitts-Integration, **alle**
  registrierten Frames (Sirils eigene Pixel-Rejection, keine vorgeschaltete
  Frame-Auswahl).

Tatsächlich gestapelte Frames (Siril verwirft in der Registrierung Frames mit zu
wenigen Sternen):

| Objekt | Rohframes | von Siril gestackt | Solver |
|---|---:|---:|---|
| M42    | 610 | 610 | lokaler Gaia DR3, plate-solve ok |
| IC434  | 359 | 359 | lokaler Gaia DR3, plate-solve ok |
| M66    | 975 | 837 | lokaler Gaia DR3, plate-solve ok |
| IC5070 | 466 | 398 | lokaler Gaia DR3, plate-solve ok |

Nach dem Stacking wurde jeder Stack mit
`platesolve <RA>,<DEC> -focal=100.355 -pixelsize=1.45` gelöst (TAN-SIP, WCS im
Header). `pcc`/`spcc` schlug offline fehl (VizieR HTTP 403, kein lokaler
Photometriekatalog) — die Referenzstacks haben daher einen sensor-nativen
Magenta-Stich (R/G ≈ B/G ≈ 1,37), der die Grünkanal-FWHM nicht beeinflusst.

Unterschiede zum tile_compile-Lauf, die beim Vergleich zu berücksichtigen sind:
tile_compile wendet eine eigene Frame-Qualitätsauswahl (Cherry-Pick) an, Siril
nur die Pixel-Rejection beim Stacken; die effektiv gestapelte Framezahl weicht
daher ab (siehe Tabelle). Für einen Effekt in der Größenordnung 1,3–1,4× ist das
vertretbar, für Feinvergleiche unter ~5 % nicht.

Die Skripte (`run_all.sh`, `status.sh`) und diese Notiz liegen unter
`docs/AQMH/attic/siril_reference/` (nur Aufzeichnung, nicht Teil des Builds).
Neue Referenzstacks für weitere Objekte folgen exakt diesem Rezept.

---

## 3. Ziele, Nichtziele und Erfolgsdefinition

### 3.1 Funktionale Ziele

- Direkte Projektion normalisierter CFA-Samples auf ein gemeinsames Raster.
- Unterstützung aller vier Bayer-Pattern `RGGB`, `BGGR`, `GRBG`, `GBRG`.
- Unterstützung monochromer Rohframes durch denselben Forward-Drizzle-Kern mit
  genau einer Sampleebene, sofern Monochromdaten weiterhin zum verbindlichen
  Produktumfang gehören.
- Erhaltung der Oberflächenhelligkeit und des WCS-pixelflächenkorrigierten
  Aperturflusses bei Translation, Rotation, affinem und geguardetem lokalem Warp.
- Unterstützung des bestehenden geguardeten lokalen Registrierungsmodells.
- Robustes, deterministisches Clipping auf Frame-Beiträgen.
- Uniform-Control, Raw-Forward-Drizzle und Mehrbandkandidat mit identischem geometrischem
  Support und identischen Clippingentscheidungen.
- Streaming- und Chunk-Verarbeitung mit begrenztem RAM-Verbrauch.
- CPU-Referenzimplementierung und semantisch äquivalenter CUDA-Pfad.
- Vollständige Resume- und Cache-Validierung.
- Korrekte WCS-, Masken-, Crop- und Offset-Behandlung bei internem 2x-Raster.
- Unveränderliche Raw-Forward-Drizzle-Baseline sowie geprüfter Kandidatenfallback.

### 3.1.1 Eingabevertrag vor Entfernung der Altmethoden

Die Entfernung von Classic darf keine bisher zugesicherte Eingabeklasse
unbemerkt in einen falschen Rechenpfad zwingen:

- OSC-Rohdaten mit bekanntem Bayer-Pattern sind der verbindliche Primärpfad.
- Monochrom-Rohdaten gehören für diesen Plan verbindlich zum Produktumfang und
  verwenden denselben Forward-Projektionskern mit genau einer Sampleebene. Sie
  müssen vor M10 dieselben Flux-, Warp-, Clipping- und Resumeverträge bestehen;
  kanalbezogene Gates werden dabei auf `L` statt auf `R/G/B` ausgewertet.
- Bereits debayerte RGB-Frames gehören nicht zum Umfang dieses Cutovers. Sie
  werden beim Scan vor jeder Run-Mutation mit dem stabilen Fehler
  `UNSUPPORTED_INPUT_RGB_FORWARD_ADAPTER_REQUIRED` abgelehnt. Ein späterer
  RGB-Forward-Sampling-Adapter ist eine eigene, erneut zu validierende
  Erweiterung und kein stiller CFA-Ersatz.
- Alle Frames eines Runs müssen dieselben Abmessungen, denselben Farbmodus,
  dieselbe Sensororientierung und bei OSC dieselbe effektive Bayer-Phase
  besitzen. Gemischte Modi, unbekannte Pattern, ungerade Crop-/Flip-Änderungen
  ohne aktualisierten CFA-Anker und widersprüchliche FITS-/Config-Metadaten
  führen fail-closed zum Runabbruch vor der Rekonstruktion.

Damit ist der Produktumfang vor M1 festgelegt: OSC und MONO werden unterstützt,
bereits debayertes RGB nicht. Diese Einschränkung muss mit M0 in GUI, CLI,
Schema und aktiver Dokumentation sichtbar sein und ist selbst ein
M10-Releasegate.

### 3.2 Qualitätsziele

Für eine spätere Aktivierung des neuen Pfads gelten mindestens folgende
Go/No-Go-Kriterien:

| Kriterium | Mindestanforderung |
|---|---:|
| Gematchte lineare FWHM Forward-Drizzle gegen bisherigen PREWARP-AQMH-Stack | mindestens 10 % besser |
| Gematchte lineare FWHM Mehrband gegen Raw-Forward-Drizzle | zusätzlich mindestens 5 % besser |
| **Absolute Hintergrund-RMS Forward-Drizzle gegen PREWARP-AQMH-Stack** (gleiche Ausgabeskala, gleiche Fläche, korrelationskorrigiert) | **höchstens 15 % schlechter** |
| Hintergrund-RMS-Regression gegen Uniform-Control | höchstens 5 % |
| Seam-Score-Regression gegen Uniform-Control | höchstens 5 % |
| Star-Tail-Regression gegen Raw-Forward-Drizzle | höchstens 10 % |
| Elongations-Regression gegen Raw-Forward-Drizzle | höchstens 8 % |
| Photometrischer Fluxfehler bei synthetischen Tests | unter 0,5 % |
| Sternzentroidfehler bei synthetischen Warps | unter 0,1 Ausgabepixel |
| CPU-/CUDA-Abweichung | innerhalb explizit dokumentierter numerischer Toleranzen |

Der photometrische Fluxfehler bezieht sich auf Aperturphotometrie in
Weltkoordinaten beziehungsweise auf die mit der WCS-Pixelfläche gewichtete
Oberflächenhelligkeit, nicht auf die rohe Summe einer bei 2x vervierfachten
Pixelzahl (11.6/12.1).

**Verbindliche Vergleichsbedingungen für die FWHM- und RMS-Zeilen:**

- Der PREWARP-AQMH-Referenzstack und der Forward-Drizzle-Stack werden bei
  **identischem `output_scale`** gerendert. Ein 2x-Ergebnis gegen eine
  1x-Baseline zu messen ist unzulässig — die verdoppelte Ausgabeauflösung
  bestünde das FWHM-Gate für sich allein.
- Zusätzlich zum vollen Gewichtsplan wird ein Kontroll­lauf mit `G_quality(f) := 1`
  gemessen (Abschnitt 11.9), damit der Geometrie-Effekt getrennt vom Effekt der
  geänderten Frame-Gewichtseingabe sichtbar ist.
- Die absolute-RMS-Zeile vergleicht neuen Pfad gegen **alten Pfad**, nicht
  Drizzle gegen Drizzle. Drizzle mit `pixfrac < 1` erzeugt pixelkorreliertes
  Rauschen; die naive Pixelstatistik unterschätzt die Varianz und wird über die
  bekannte Kernel-Autokorrelation korrigiert, bevor verglichen wird.

Die FWHM-Anforderungen sind Releasekriterien, keine Garantie dafür, dass ein
einzelner Testlauf automatisch zum Cutover führt. Für das Single-Method-Release
sind mehrere Datensätze mit unterschiedlichen Sternfeldern, Hintergründen,
Rotationen und Framezahlen erforderlich — darunter mindestens je ein
untersampelter Datensatz (native FWHM nahe 2 px), ein Monochrom-Datensatz,
ein Datensatz mit kleinem Dither und niedriger Framezahl sowie einer mit starker
Feldrotation.

### 3.3 Nichtziele

- Kein Versuch, Beugungsgrenzen oder nicht vorhandene Bildinformation durch
  Überschwingen zu simulieren.
- Keine PSF-Deconvolution in der ersten Implementierung.
- Kein dynamisches Aufweichen der Gates für bestimmte Objekte.
- Keine dauerhafte Parallelarchitektur mit Classic oder PREWARP-AQMH.
- Kein automatischer Wechsel zu Siril oder einem externen Stacker.

---

## 4. Verbindliche Verträge der Single-Method-Architektur

### 4.1 Eine öffentliche Rekonstruktionsmethode

Nach dem Cutover existiert genau eine öffentliche Rekonstruktionsmethode. Es
gibt weder `classic`/`aqmh` als Benutzerauswahl noch einen
`aqmh.reconstruction.engine`-Schalter. Classic-Metriken, Classic-Tile-Gewichte,
PREWARP-Nutzsignalframes und alte AQMH-Kandidaten dürfen nicht in die neue
Rekonstruktion einfließen. Gemeinsame Infrastruktur für Kalibration,
Normalisierung, Registrierung, WCS, Masken, Logging und Run-Management wird
weiterverwendet, sofern sie keinen alten Methodenpfad erzwingt.

### 4.2 Kandidaten- und Kontrollvertrag

Jeder neue Mehrbandkandidat wird geprüft gegen:

1. das Uniform-Control mit identischen Samples, Masken und Clippingentscheidungen;
2. die unveränderliche Raw-Forward-Drizzle-Baseline desselben Sampleplans.

Wenn der Mehrbandkandidat scheitert, bleibt Raw-Forward-Drizzle. Wenn
Raw-Forward-Drizzle gegenüber
Uniform-Control scheitert, wird Uniform-Control verwendet. Ein Kandidat darf
nicht dadurch bestehen, dass Kandidat und Kontrolle unterschiedliche
Sternpopulationen messen.

### 4.3 GPU-Vertrag

Der CPU-Pfad definiert die Semantik. CUDA muss diese Semantik innerhalb
dokumentierter Toleranzen erhalten und einen getesteten CPU-Fallback besitzen.
GPU-Verfügbarkeit darf nicht aus einem sandboxbedingten Gerätefehler abgeleitet
werden.

### 4.4 Resume-Vertrag

Eine Phase darf nur als direkt resumierbar gelten, wenn alle erforderlichen
Vorgängerartefakte und Caches vorhanden und anhand von Metadaten validiert sind.
Ein vorhandenes Phase-Event allein reicht nicht.

### 4.5 Historische Run-Daten

Bestehende Runs sind Benutzerdaten. Die Implementierung darf deren Artefakte,
Caches oder Outputs nicht migrieren oder überschreiben. Historische Outputs,
Reports, Logs und Konfigurationen bleiben read-only sichtbar und exportierbar.

Ein historischer Classic- oder PREWARP-AQMH-Run ist mit dem Single-Method-Runner
nicht resumierbar. Ein Resumeversuch endet vor jeder Mutation mit einem
eindeutigen Fehlercode, der alte Methodik, erkannte Schema-/Pipelineversion und
zulässige Alternativen nennt. Es gibt keine automatische Konvertierung alter
Caches und keinen stillen Neustart als neue Methodik. Ein vollständiger neuer
Run aus den unveränderten Quelldaten ist die einzige unterstützte Fortsetzung.

---

## 5. Zielpipeline

### 5.1 Zu entfernender historischer Pfad

```text
REGISTRATION
  -> PREWARP RGB/CFA
  -> COMMON_OVERLAP
  -> AQMH_MAPS auf Canvas
  -> AQMH_GLOBAL_QUALITY
  -> AQMH_RECONSTRUCTION aus prewarped_frames
  -> AQMH_DIAGNOSTICS
  -> STACKING/DEBAYER/Downstream
```

Der dargestellte Pfad dient während der Beweisphase ausschließlich als
Vergleichsreferenz. Er ist nicht Teil des ausgelieferten Zielsystems und wird
im Löschmeilenstein aus Runner, Bibliothek, Schema, GUI, Tests und aktiver
Dokumentation entfernt.

### 5.2 Einziger Zielpfad

```text
CHANNEL_SPLIT/NORMALIZATION/GLOBAL_METRICS
  -> normalized_frames (unveränderte CFA-Geometrie)
  -> REGISTRATION
       - RegistrationSamplingPlan (artifacts/registration_sampling.json)
  -> SAMPLING_GEOMETRY
       - geometrische Drizzle-Coverage, kein Bild-PREWARP
       - coverage_gate: direkter Kanalsupport, p10-n_eff, interne Löcher
  -> COMMON_OVERLAP
  -> SOURCE_QUALITY_MAPS auf CFA-Quellkoordinaten
  -> GLOBAL_QUALITY (G_quality(f) aus dem CFA-Green-Proxy, QualityFrameWeightPlan)
  -> FORWARD_DRIZZLE (eine Phase-ID, ein Artefakt forward_drizzle.json)
       - Uniform-Control
       - Raw-Forward-Drizzle
       - skalenspezifische Detailprofile
       - Mehrbandfusion
       - Dreiwegvalidation und Kandidatengates
  -> RECONSTRUCTION_DIAGNOSTICS
  -> STACKING-Pass-through
  -> ASTROMETRY/BGE/PCC/HMS
```

Mehrbandfusion, Validation und Kandidatenauswahl sind Teilschritte der Phase
`FORWARD_DRIZZLE` und keine eigenen Resume-Einstiege (Abschnitt 18.2). Die
Phasen-IDs in diesem Dokument sind verbindlich: `SAMPLING_GEOMETRY`,
`COMMON_OVERLAP`, `SOURCE_QUALITY_MAPS`, `GLOBAL_QUALITY`, `FORWARD_DRIZZLE`,
`RECONSTRUCTION_DIAGNOSTICS`.

`PREWARP` wird nicht als erfolgreiche Scheinphase weitergeführt. Es wird durch
die semantisch korrekte Phase `SAMPLING_GEOMETRY` ersetzt. Ebenso wird
`DEBAYER` aus der aktiven OSC-Phasenfolge entfernt, weil die Rekonstruktion
bereits R/G/B erzeugt. Frontend, History, Report und Resume routen ausschließlich
über stabile Phase-IDs der neuen Pipeline. Historische Phase-IDs werden nur im
read-only History-Parser verstanden.

---

## 6. Konfigurationsvertrag

### 6.1 Vorgeschlagene Konfiguration

```yaml
reconstruction:
  # Bleibt auch nach M10 standardmäßig false. true ist ausschließlich eine
  # explizite Benutzerentscheidung nach vollständig erfolgreichem Run.
  delete_source_cache_after_run: false
  keep_profile_cache_after_run: false
  common_overlap_required_fraction: 1.0

  diagnostics:
    level: summary            # summary | full

  drizzle:
    internal_scale: 2
    output_scale: 1            # Produktionsdefault; 2 für expliziten 2x-Output
    kernel: square
    pixfrac: 0.8
    robust_passes: 2
    min_clip_contributors: 5  # unterhalb dieser Zahl kein Sigma/MAD-Clipping
    chunk_rows: 0
    chunk_halo_rows: -1       # -1 = auto; sonst explizite Randzeilen je Chunk
    memory_budget_mb: 0

  clipping:
    clip_sigma_low: 3.0
    clip_sigma_high: 3.0
    min_fraction: 0.4
    min_n_eff: 3.0

  coverage_gate:
    min_frames: 2
    min_supported_fraction: 0.995
    min_channel_n_eff_floor: 3.0
    min_channel_n_eff_fraction: 0.15
    min_analysis_pixels: 1024
    max_internal_hole_area_px: 0

  quality:
    pyramid:
      scales: 4              # Anzahl Analyse-Skalen der Source-Q-Pyramide

  multiband:
    enabled: true
    levels: 3
    alpha_cap: 1.0
    fine_quality_exponent: 4.0
    medium_quality_exponent: 2.0
    min_quality_separation: 0.05
    full_quality_separation: 0.20
    min_effective_samples: 8.0
    full_effective_samples: 24.0
```

Alle referenzierten Schlüssel sind hier vollständig aufgeführt. Werte, die in
späteren Abschnitten genannt werden (`clip_sigma_low/high`, `min_fraction`,
`min_n_eff`, `min_clip_contributors` in 11.8; `chunk_halo_rows` in 11.11;
`coverage_gate` in 9.5 und 25.2; `alpha_cap` in 14.4; `diagnostics.level` in
16.2; `reconstruction.quality.pyramid.scales` in 13.3/13.4) beziehen sich exakt
auf diese Struktur.

Nicht öffentlich konfigurierbar (interne, versionierte Konstanten, im Artefakt
protokolliert): Storage-Divisor und Datentyp des Q-Map-Caches (13.4), lokaler
Bandenergieguard, Alpha-Glättung und À-trous-Denominatorschwellen (14.2/14.5),
Iterations- und Subdivisionsgrenzen der lokalen Inversion/Dropletgeometrie
(7.3/11.6), 2x→1x-Supportregel (12.1) sowie die CPU-/CUDA-Toleranzen (19.5).
Keine dieser Konstanten darf ohne
Versions-/Hashänderung geändert werden.

### 6.2 Semantik

#### `reconstruction.delete_source_cache_after_run`

Steuert die Löschung von `cache/normalized_frames` und der source-space
Quality-Map-Caches nach einem erfolgreichen Lauf. Bei `true` ist ein direktes
Resume ab `FORWARD_DRIZZLE` nicht möglich. Der Produktionsdefault bleibt auch
nach M10 `false`, weil `normalized_frames` die einzige Rekonstruktionsquelle ist
und der Single-Method-Cutover keinen Qualitätsfallback besitzt. `true` ist nur
als explizite Benutzerwahl für abgeschlossene Archiv-/Batch-Runs zulässig.
Gelöscht wird ausschließlich nach erfolgreichem Abschluss aller konfigurierten
Downstream-Phasen, atomarem Commit der finalen Outputs und verifizierten
Artefaktchecksummen. Report und Resume-UI müssen danach ausdrücklich anzeigen,
dass eine erneute Rekonstruktion nur aus den ursprünglichen Quelldaten möglich
ist. Zusätzlich wird eine explizite GUI-/CLI-Aktion zur späteren
Cachebereinigung vorgesehen; eine automatische Löschung allein wegen M10 ist
unzulässig. Der alte Parameter `delete_prewarped_cache_after_run` wird aus aktiven
Schemas, Beispielen, Parsern und Serialisierung entfernt und bei neuen
Konfigurationen als unbekannter Legacy-Schlüssel abgelehnt.

#### `reconstruction.keep_profile_cache_after_run`

Steuert ausschließlich die transaktionalen internen U/R/F/M-Profilstores.
Default `false`: Die Stores existieren während `FORWARD_DRIZZLE`, werden nach
erfolgreichem Output-/Artefaktcommit und Checksumprüfung aber gelöscht. Raw,
ausgewähltes Ergebnis, Support, Validation und kompakte Diagnostik bleiben
persistiert; Downstream-Resume benötigt keine internen Profile. Bei `true`
bleiben die gehashten Stores für explizites Rekonstruktions-Tuning erhalten.
Dieser Schalter ist unabhängig von `diagnostics.level`: `full` steuert
öffentliche Kontroll-FITS, nicht die Cachelebensdauer. Ändert sich eine
Profil-/Clippingkonfiguration, darf ein Store nur bei passendem Hash
wiederverwendet werden.

#### `reconstruction.common_overlap_required_fraction`

Anteil der gültigen Sampling-Transforms, der an einem Zielpixel geometrischen
Support besitzen muss, damit es zur `analysis_common_mask` gehört. Bereich
`(0, 1]`. Die Maske wird bereits in `SAMPLING_GEOMETRY` gebildet, damit das
Coverage-Gate seinen p10-Bezugsbereich ohne Zirkelschluss auswerten kann;
`COMMON_OVERLAP` persistiert und veröffentlicht anschließend diese bereits
festgelegte Maske.

#### `reconstruction.drizzle.internal_scale`

Ganzzahliger interner Drizzle-Faktor. Zulässige Werte im ersten Release sind
explizit `1` oder `2`; es gibt **keinen** Auto-Modus. `2` ist der
Subpixel-Rekonstruktionsmodus für ausreichend abgedeckte, kritisch bis
untersampelte Daten und der Produktionsdefault nach bestandenem M9.
`1` ist ein unterstützter Produktionsmodus für übersampelte oder bewusst
ressourcenschonende Runs. Eine diagnostische Empfehlung darf anhand nativer
FWHM und Coverage ausgegeben werden, ändert den konfigurierten Wert aber nie.
Ein späterer Auto-Modus ist eine eigene Vertragsänderung mit persistierter
Preflight-Entscheidung und kein Fehlerfallback.

#### `reconstruction.drizzle.output_scale`

Zulässige Werte sind explizit `1` oder `2`, mit
`output_scale <= internal_scale`; auch hier gibt es keinen Auto-Modus. Der
Produktionsdefault ist `internal_scale=2, output_scale=1`, damit die
Subpixelintegration bei kompatibler nativer Ausgabegeometrie genutzt wird.
`2/2` bleibt der explizite Modus für kritisch/unterabgetastete Daten und wird im
M4-/M9-Schärfenachweis verwendet, damit kein Downsampling den Nachweis
verfälscht. `1/1` bleibt für übersampelte oder ressourcenbegrenzte Daten
unterstützt. Der 2x→1x-Schritt ist der einmalige Flächenoperator aus 12.1.

#### `reconstruction.drizzle.kernel`

Im MVP ausschließlich `square`. Der Kernel repräsentiert eine
oberflächenhelligkeitserhaltende Pixel-Droplet-Fläche; physischer Aperturflux
folgt dem Vertrag aus 11.6. Weitere Kernel werden erst nach dem korrekten
Square-Pfad zugelassen.

#### `reconstruction.drizzle.pixfrac`

Lineare Kantenlänge des Droplets relativ zu einem Quellpixel. Bereich
`(0, 1]`, Produktionsdefault `0.8`. Alle Uniform-, Raw- und Detailprofile sowie
alle aktiven Farbkanäle verwenden im ersten Release denselben Wert, damit
Transferfunktion, Support und Clippingpopulation vergleichbar bleiben.
Per-Kanal-`pixfrac` ist bis nach M10 ausgeschlossen. Scheitert Coverage mit
`0.8`, werden als explizite neue Konfiguration global `1.0` oder
`internal_scale=1` geprüft; es gibt keine stillen oder kanalabhängigen
Anpassungen.

#### `reconstruction.drizzle.robust_passes`

Anzahl der deterministischen Sigma-/MAD-Clipping-Iterationen. Bereich `1..6`.
Alle Rekonstruktionsprofile verwenden dieselben resultierenden
Akzeptanzentscheidungen.

#### `reconstruction.drizzle.min_clip_contributors`

Mindestzahl endlicher Frame-Beiträge `x_f,c(q)` pro Zielpixel und Kanal, ab
der Sigma-/MAD-Clipping überhaupt ausgeführt wird (11.8, Schritt 2). Darunter
bleiben alle endlichen Beiträge gültig. Schützt die dünn belegten R/B-Kanäle
bei kleinen Framezahlen. Bereich `>= 2`; typischer Wert oberhalb von
`clipping.min_n_eff`.

#### `reconstruction.drizzle.chunk_halo_rows`

Zusätzliche Zielzeilen ober- und unterhalb jedes Chunk-Kernbereichs, damit
Droplets, die eine Chunkgrenze überdecken, vollständig akkumuliert werden
(11.11). Der ganzzahlige Sentinel `-1` bedeutet `auto` und leitet den Wert aus
der maximalen projizierten vertikalen Droplet-Ausdehnung aller gültigen Frames
im Sampling-Plan ab (Rotation, Skalierung und lokale Subdivision inklusive).
Ein expliziter nichtnegativer Wert muss mindestens dem aus dem Plan berechneten
konservativen Mindesthalo entsprechen; die einfache Schranke
`ceil(drop_size * sqrt(2)) + 1` genügt nur ohne zusätzliche Skalierung oder
lokale Krümmung. Kleinere Werte sind ein Konfigurationsfehler, kein stiller
Clamp.

#### `reconstruction.clipping.*`

`clip_sigma_low`/`clip_sigma_high`: asymmetrische Grenzen in MAD-skalierten
Sigma-Einheiten um den Median der Frame-Beiträge. `min_fraction`: minimaler
Anteil akzeptierter Beiträge relativ zum geometrisch möglichen Frame-Support
des Pixels/Kanals. `min_n_eff`: minimale effektive Samplezahl (11.10) des
Uniform-Profils. Unterschreitet ein Pixel/Kanal eine der beiden Grenzen, wird
es im Kanalsupport als nicht belegt markiert (9.3, 11.8 Schritt 8).

#### `reconstruction.coverage_gate.*`

Fail-closed-Vorprüfung in `SAMPLING_GEOMETRY` (9.5), bevor Q-Maps oder
Rekonstruktion gerechnet werden:

- `min_frames`: technische Untergrenze gültiger Sampling-Transforms. Der Wert
  `2` ersetzt nicht die strengere effektive-`n_eff`-Prüfung.
- `min_supported_fraction`: Mindestanteil der `analysis_common_mask`, der pro
  aktivem Kanal exakten geometrischen Support besitzt; verbindlich `0.995`.
- effektive Kanalgrenze:

  ```text
  required_channel_n_eff(N) =
      max(min_channel_n_eff_floor,
          min_channel_n_eff_fraction * N)
  ```

  mit `min_channel_n_eff_floor=3.0`,
  `min_channel_n_eff_fraction=0.15` und `N` gültigen
  Sampling-Transforms. Geprüft wird p10 des geometrischen Uniform-`n_eff` pro
  aktivem Kanal über die bereits in `SAMPLING_GEOMETRY` gebildete
  `analysis_common_mask`. MONO wertet nur `L` aus.
- `min_analysis_pixels=1024`: kleinere oder leere Analysemasken scheitern mit
  `insufficient_analysis_support`.
- `max_internal_hole_area_px=0`: nach Ausschluss des geometrischen Randes darf
  kein vollständig ungestütztes zusammenhängendes Kanal-Loch innerhalb der
  Analysemaskenfläche verbleiben.

Die zirkuläre Dither-Streuung modulo zwei Quellpixel wird weiterhin an Mitte und
vier Ecken diagnostisch berechnet (`theta = pi * (offset mod 2)`,
`sigma_circ_px = sqrt(-2 * ln(R)) / pi`), ist aber **kein hartes Gate**. Die
direkt rasterisierte Kanalcoverage und ihr `n_eff` sind maßgeblich und erfassen
auch Rotation oder lokale Warps, bei denen ein Dither-Proxy falsch entscheiden
könnte.

Eine Unterschreitung eines direkten Coveragewerts bricht den Run mit Nennung des
verletzten Schlüssels, Kanals und Ist-Werts ab. Es gibt keinen Fallback auf
`internal_scale = 1`. Die Zahlenwerte sind der verbindliche M1-Ausgangsvertrag;
sollten die synthetischen M1-Fixtures ihn widerlegen, wird zuerst dieser Plan
versioniert und erst danach der Parserdefault geändert.

#### `reconstruction.quality.pyramid.scales`

Anzahl der Analyse-Skalen der Source-Q-Pyramide (13.3). Skala 0 ist die
feinste. Das Mehrbandprofil `F` benötigt Skala 0, `M` benötigt Skala 1; der
Composite ist das geometrische Mittel aller gültigen Skalen.

#### `reconstruction.diagnostics.level`

`summary`: nur JSON-Kennzahlen und kompakte Heatmaps. `full`: zusätzlich alle
in 16.2 gelisteten FITS-Kontrollausgaben. Beeinflusst keine Rechenergebnisse.

#### Mehrbandparameter

Die Exponenten steuern ausschließlich, wie stark vorhandene Q-Unterschiede in
Fine- und Medium-Profilen wirken. Sie sind keine Schärfungsbeträge.
`min_quality_separation` und `full_quality_separation` steuern den adaptiven
Band-Mix. Wenn die Q-Werte der Frames lokal kaum getrennt sind, wird kein
Detailprofil zugemischt.

### 6.3 Validierung

- `internal_scale` und `output_scale` müssen in `{1, 2}` liegen.
- `output_scale <= internal_scale`.
- `kernel == square` im MVP.
- `pixfrac > 0 && pixfrac <= 1`.
- `robust_passes` in `[1, 6]`.
- `min_clip_contributors >= 2`.
- `chunk_rows >= 0`, `memory_budget_mb >= 0`.
- `chunk_halo_rows == -1` oder ganzzahlig `>=` dem aus Sampling-Plan und
  Droplet-Subdivision berechneten konservativen Mindesthalo.
- `0 < common_overlap_required_fraction <= 1`.
- `clip_sigma_low > 0`, `clip_sigma_high > 0`.
- `0 < min_fraction <= 1`, `min_n_eff >= 1`.
- `diagnostics.level` in `{summary, full}`.
- `reconstruction.quality.pyramid.scales` in `[1, 4]`.
- `coverage_gate.min_frames >= 2`,
  `0 < min_supported_fraction <= 1`,
  `min_channel_n_eff_floor >= 1`,
  `0 < min_channel_n_eff_fraction <= 1`,
  `min_analysis_pixels >= 1` und `max_internal_hole_area_px >= 0`.
- `levels` in `[1, 4]`; `levels >= 2` erfordert
  `reconstruction.quality.pyramid.scales >= 2` (Scale-1-Q für das
  Medium-Profil), `levels >= 1` erfordert `scales >= 1`. Weitere Levels
  benötigen keine zusätzlichen Q-Skalen (14.3).
- `alpha_cap` in `[0, 1]`.
- Exponenten `>= 0`.
- `0 <= min_quality_separation < full_quality_separation <= 1`.
- `1 <= min_effective_samples < full_effective_samples`.
- OSC erfordert ein bekanntes Bayer-Pattern; ein unbekanntes Pattern ist ein
  Konfigurations-/Inputfehler und besitzt keinen Methodenfallback.
- Altschlüssel zur Methoden- oder Engine-Auswahl werden nach dem Cutover
  fail-closed abgelehnt; sie werden nicht still ignoriert oder übersetzt.
- Entfernte *strukturelle* Legacy-Blöcke aus 6.5 (z. B. `tile`, `tile_denoise`,
  `local_metrics`, `synthetic`) werden dagegen mit `WARN`-Meldung und Eintrag in
  `artifacts/config_migration.json` gestrippt; Umbenennungen aus 6.5 werden
  automatisch übernommen.

### 6.4 Zu aktualisierende Konfigurationsquellen

- `tile_compile_cpp/include/tile_compile/config/configuration.hpp`
- `tile_compile_cpp/src/io/config.cpp`
- `tile_compile_cpp/tile_compile.schema.yaml`
- `tile_compile_cpp/tile_compile.schema.json`
- `tile_compile_cpp/tile_compile.yaml`
- aktive Dateien unter `tile_compile_cpp/examples/`
- `tile_compile_cpp/examples/README.md`
- `docs/configuration_reference.md`
- `docs/configuration_reference_en.md`
- `docs/configuration_examples_practical_de.md`
- `docs/configuration_examples_practical_en.md`

Bei der M0-Implementierung wird für alle neuen, geänderten und entfernten
Parameter verbindlich `.devin/skills/update-param-doc/SKILL.md` angewendet;
Parser, beide Schemas, Default-YAML, Beispiele und deutsche/englische
Konfigurationsdokumentation werden in demselben Änderungsschnitt aktualisiert.
`tile_compile.schema.json`/`.schema.yaml` werden aus `configuration.hpp`
generiert (`tile_compile_cli get-schema`); sie werden nicht von Hand editiert,
sondern nach der Parseränderung regeneriert und im selben Commit eingecheckt.
Historische Dateien unter `attic/` werden nicht auf neue Defaults umgeschrieben.

### 6.5 Schicksal der Legacy-Top-Level-Blöcke

Die aktive `config.yaml` enthält Blöcke aus der Classic-/Tile-Ära, die der
Single-Method-Pfad nicht mehr kennt. Verbindliche Behandlung je Block:

| Block | Behandlung im Single-Method-Pfad |
|---|---|
| `pipeline`, `output`, `data`, `calibration`, `normalization`, `linearity`, `registration`, `astrometry`, `bge`, `pcc`, `hypermetric_stretch` | unverändert behalten |
| `runtime_limits` | behalten; `tile_analysis_max_factor_vs_stack`, `tile_reconstruction_diagnostics` neutral umbenennen |
| `method` | **entfernt**; als benannter Fehler `UNKNOWN_LEGACY_KEY` behandelt (semantischer Methodenschlüssel, kein bloßer Strukturballast — bleibt fail-closed, wird nicht stillschweigend gestrippt) |
| `aqmh` | Root-Umbenennung → `reconstruction`; Unterstruktur nach 6.1 |
| `assumptions` | behalten; `reduced_mode_*` (Tile-Clustering) entfernt |
| `dithering` | auf Diagnose reduziert; `min_shift_px`-Gate entfernt (Dither ist nur Diagnose, §26) |
| `global_metrics` | behalten als Gewichtskonfiguration für `G_quality(f)`; verschoben nach `reconstruction.quality.frame_weights` |
| `local_metrics` | entfernt (kein Tiling) |
| `tile` | entfernt; UI-Referenzen in `preprocessing_service.cpp` mit umgestellt |
| `tile_denoise` | entfernt |
| `synthetic` | entfernt |
| `chroma_denoise` | **behalten** und auf dem Drizzle-Output neu validiert (siehe unten) |
| `stacking` | `common_overlap_required_fraction` bereits nach `reconstruction` verschoben (§30.2); `method`, `sigma_clip`, `cluster_quality_weighting`, `output_stretch`, `tile_common_valid_min_fraction` entfernt; **`per_frame_cosmetic_correction(_sigma)` und `cosmetic_correction(_sigma)` bleiben erhalten** und werden nach `calibration.frame_cleanup` verschoben — Hotpixel-/Cosmic-Ray-Entfernung läuft pro Frame vor der Registrierung (`runner_phase_registration.cpp`) und ist für Forward-Drizzle weiterhin nötig |
| `validation` | `min_tile_weight_variance`, `require_no_tile_pattern` entfernt; `min_fwhm_improvement_percent`, `max_background_rms_increase_percent` mit den Gates aus 3.2 und `aqmh.validation` zu einem `reconstruction.validation`-Block konsolidiert |

**Migrationssemantik (Entscheidung 2026-09-03): strippen mit Warnung.** Trifft
der Parser auf einen entfernten *strukturellen* Block oder Schlüssel aus obiger
Tabelle (nicht `method`/Engine — die bleiben fail-closed), entfernt er ihn,
protokolliert eine `WARN`-Meldung mit Pfad und Wert und schreibt die Liste der
gestrippten Schlüssel nach `artifacts/config_migration.json`. Der Run läuft mit
der bereinigten Konfiguration weiter. Damit ist der Vorgang auditierbar und
kein *stilles* Übersetzen im Sinne von §30.3.3; §30.3.3 bezieht sich weiterhin
verbindlich auf semantische Umdeutung, Clamping und Methoden-/Scale-Fallbacks.
Umbenennungen (`aqmh`→`reconstruction`, `global_metrics`→
`reconstruction.quality.frame_weights`, Cosmetic-Keys→`calibration.frame_cleanup`)
werden dabei automatisch übernommen und ebenfalls in
`config_migration.json` vermerkt. Ein optionales
`tile_compile_cli migrate-config <in> <out>` schreibt dieselbe Transformation
explizit in eine neue Datei, ohne einen Run zu starten.

**`chroma_denoise` auf dem Drizzle-Output.** Der Block bleibt aktiv und
operiert unverändert `post_stack_linear` auf `reconstructed_R/G/B` **nach** der
Kandidatenauswahl und **vor** BGE. Er berührt den U/R/M-Vergleich und die
Runtime-Gates nicht. Wegen des durch `pixfrac < 1` und 2x eingeführten
korrelierten Rauschens (§12.4) werden `chroma_wavelet`- und
`chroma_bilateral`-Parameter in M8 auf synthetischen Drizzle-Fixtures neu
kalibriert; die bisherigen Defaults gelten bis dahin als vorläufig und werden im
Report als solche gekennzeichnet.

---

## 7. Datenmodell: `RegistrationSamplingPlan`

### 7.1 Neue Typen

Neue Datei:

```text
tile_compile_cpp/include/tile_compile/registration/registration_sampling_plan.hpp
tile_compile_cpp/src/registration/registration_sampling_plan.cpp
```

Vorgeschlagene öffentliche Typen:

```cpp
enum class SamplingWarpConvention {
  canvas_to_source
};

struct FrameSamplingTransform {
  std::string frame_id;              // stabil aus Inputmanifest + Inhaltsidentität
  size_t source_index = 0;
  bool valid = false;
  WarpMatrix canvas_to_source = WarpMatrix::Identity();
  WarpMatrix source_to_canvas = WarpMatrix::Identity();
  bool source_to_canvas_affine_valid = false;
  bool has_smooth_local_model = false;
  registration::SmoothLocalWarpModel smooth_local_model;  // bestehender Typ in global_registration.hpp
  float model_coordinate_scale = 1.0f;
  float model_offset_x = 0.0f;
  float model_offset_y = 0.0f;
  float registration_residual_factor = 1.0f;
  float model_prediction_factor = 1.0f;
  bool model_predicted = false;       // Provenienz; Gewicht steht im Faktor
  int chain_depth = 0;
  std::string provenance;
};

struct RegistrationSamplingPlan {
  int source_width = 0;
  int source_height = 0;
  int canvas_width_native = 0;
  int canvas_height_native = 0;
  int canvas_offset_x_native = 0;
  int canvas_offset_y_native = 0;
  int internal_scale = 1;
  int output_scale = 1;
  ColorMode color_mode = ColorMode::MONO;
  BayerPattern bayer_pattern = BayerPattern::UNKNOWN;
  int cfa_origin_x = 0;              // Sensorparität der Cachekoordinate (0,0)
  int cfa_origin_y = 0;
  SamplingWarpConvention convention =
      SamplingWarpConvention::canvas_to_source;
  std::vector<FrameSamplingTransform> frames;
  std::string plan_hash;
};
```

### 7.2 Warp-Konvention

Der bestehende Warp wird als inverse Sampling-Map behandelt:

```text
s = W(q)
```

mit:

- `q`: Ziel-/Canvas-Koordinate in nativer Geometrie;
- `s`: Quellkoordinate im normalisierten Frame.

Forward-Drizzle benötigt:

```text
q = W^-1(s)
```

Für rein affine Frames wird `source_to_canvas` einmalig mit einer geprüften
2x3-Affininversion berechnet und `source_to_canvas_affine_valid=true` gesetzt.
Bei einem lokalen Modell ist diese Matrix nur der affine Startwert; die
nichtlineare Source→Canvas-Abbildung wird ausschließlich über die geguardete
Inversion aus 7.3 ausgewertet und darf die Matrix nicht als fertige Inverse
verwenden. Singularität, nichtfinite Koeffizienten oder eine Determinante
außerhalb der vorhandenen Registrierungsgrenzen machen den Frame für
Forward-Drizzle ungültig.

### 7.3 Lokale Korrektur

Das aktuelle lokale Modell wird in inverser Samplingrichtung verwendet:

```text
s = W_global(q + d(q))
```

Für einen Quellpunkt `s` wird zunächst

```text
u = inverse(W_global)(s)
```

berechnet. Danach wird

```text
q + d(q) = u
```

mit einer beschränkten Fixpunktiteration gelöst:

```text
q_0 = u
q_(n+1) = u - d(q_n)
```

`d(q)` wird mit exakt den aus dem bestehenden Remapvertrag übernommenen und im
Sampling-Plan persistierten Größen `model_coordinate_scale`, `model_offset_x`
und `model_offset_y` ausgewertet. Ohne diese Größen ist ein lokales Modell nicht
resumierbar. Für Differential- und Flächenrechnungen gilt am konvergierten Punkt

```text
J_source_to_canvas = (I + Dd(q))^-1 * J_global^-1
```

und nicht nur die affine inverse Jacobi-Matrix. Nichtfinite oder nichtinvertierbare
lokale Jacobians verwerfen das Sample und speisen die Framefehlergrenze.

Verbindliche Grenzen:

- höchstens 6 Iterationen;
- Konvergenztoleranz `1e-3` native Pixel;
- Abbruch bei nichtfiniten Werten;
- Abbruch außerhalb einer konfigurationsunabhängigen Sicherheitsmarge;
- Nutzung nur bei bereits akzeptiertem Jacobian-/Singularwert-Gate des
  lokalen Modells.

Fehlgeschlagene Inversionen werden pro Frame gezählt. Ein einzelnes Sample darf
verworfen werden; überschreitet die Fehlerrate eines Frames 0,1 %, wird der
gesamte Frame aus der Drizzle-Rekonstruktion ausgeschlossen und im Artefakt
begründet. Das lokale Modell darf nicht stillschweigend durch den globalen Warp
ersetzt werden.

### 7.4 Serialisierung

Neues Artefakt:

```text
artifacts/registration_sampling.json
```

Pflichtfelder:

```json
{
  "schema_version": 1,
  "warp_convention": "canvas_to_source",
  "source_width": 3840,
  "source_height": 2160,
  "canvas_width_native": 3926,
  "canvas_height_native": 2312,
  "canvas_offset_x_native": 42,
  "canvas_offset_y_native": 76,
  "internal_scale": 2,
  "output_scale": 1,
  "color_mode": "OSC",
  "bayer_pattern": "RGGB",
  "cfa_origin_x": 0,
  "cfa_origin_y": 0,
  "plan_hash": "...",
  "frames": []
}
```

Pro Frame werden `frame_id`/`source_index`, affine Matrizen, Gültigkeit,
Provenienz, `model_prediction_factor`, Residualfaktor, Chain-Tiefe und eine
serialisierte lokale Modellbeschreibung einschließlich Koordinatenskalierung und
-offset gespeichert.

**Hashdomäne von `plan_hash`.** Der Hash umfasst in Schema 2 zusätzlich `source_identity_hash` (Inputmanifest
und effektive Konfiguration) sowie die native Sampling-Geometrie: Quellabmessungen, Canvasabmessungen und -offsets in nativen
Pixeln, Warp-Konvention, Farbmodus, Bayer-Pattern und CFA-Ursprung sowie pro
Frame `frame_id`, `source_index`, Gültigkeit, affine Matrizen, lokales Modell,
Modell-Koordinatenskalierung/-offsets, `model_prediction_factor` und
Residualfaktor. `internal_scale` und
`output_scale` werden im Artefakt mitgeschrieben, gehen aber **nicht** in
`plan_hash` ein; sie gehören zur Drizzle-Geometriehashdomäne (18.3). Reine
Diagnostikfelder (`provenance`, Zeitstempel) sind ebenfalls nicht Teil des
Hashes (Test in 20.1).

Der Hash wird nicht aus implementationsabhängig formatierten JSON-Floats
gebildet, sondern aus einer kanonischen Bytekodierung mit fester Feldreihenfolge,
festgelegter Endianness und bitweiser IEEE-754-Repräsentation. Beim Laden werden
`internal_scale` und `output_scale` trotz ihrer Nichtaufnahme in `plan_hash`
separat gegen die aktive Konfiguration geprüft.

Source-Q-Maps verwenden **nicht** `plan_hash` als Gültigkeitsschlüssel, weil
Registrierungs- und Canvasänderungen ihre source-space Werte nicht ändern. Ihre
eigene Hashdomäne ist in 13.4 und 18.3 definiert.

Das Artefakt wird erst nach Canvas-/Offsetberechnung geschrieben. Das bisherige
`global_registration.json` wird weiterhin erzeugt und bleibt Diagnostikquelle
für die Registrierung selbst.

---

## 8. Ersetzung von PREWARP durch `SAMPLING_GEOMETRY`

### 8.1 Bestehende Funktion

`run_phase_registration_prewarp()` übernimmt heute Registrierung,
Canvasberechnung und vollständigen PREWARP. Diese Kopplung wird aufgelöst. Die
Funktion und alle reinen Nutzsignal-PREWARP-Aufrufer werden im Löschmeilenstein
entfernt.

### 8.2 Neue interne Aufteilung

```cpp
bool run_phase_registration(..., PhaseRegistrationContext& out);

bool run_phase_sampling_geometry(
    const RegistrationSamplingPlan& plan,
    ...,
    PhaseRegistrationContext& out);
```

Der aktive Runner ruft ausschließlich auf:

```cpp
run_phase_registration(...);
run_phase_sampling_geometry(...);
```

Während M0 bis M9 darf eine test-only Vergleichshülle den alten Pfad noch aus
dedizierten Regressionstests aufrufen. Produktionskonfiguration, CLI, Backend
und GUI erhalten keinen Zugriff darauf. Mit M10 wird sie vollständig aus den
Produkt-Targets entfernt und bleibt nur im standardmäßig deaktivierten
`tile_compile_legacy_reference_tests`; physisch gelöscht wird sie in M11.

### 8.3 Erweiterter `PhaseRegistrationContext`

Zusätzliche Felder:

```cpp
RegistrationSamplingPlan sampling_plan;
std::shared_ptr<RunnerFrameCache> source_frame_cache;
std::string reconstruction_source;
```

Es gilt verbindlich:

- `source_frame_cache` verweist auf die normalisierten Quellframes;
- `frame_has_data[fi]` bedeutet: normalisierter Frame vorhanden und
  Sampling-Transform gültig;
- Canvasgröße und Offsets stammen aus `sampling_plan`;
- `overlap_coverage_count` wird geometrisch erzeugt.

`prewarped_frames`, `prewarp_performed` und PREWARP-spezifische Contextfelder
werden vollständig aus dem aktiven Context entfernt. Allgemeine Warpmodelle
bleiben Bestandteil des `RegistrationSamplingPlan`; entfernt wird die
Interpolation des Nutzsignals, nicht die geometrische Registrierung.

---

## 9. Geometrische Coverage und Masken

### 9.1 Zweck

`COMMON_OVERLAP` wird vor der eigentlichen Rekonstruktion benötigt. Der aktive
Pfad kann Coverage nicht mehr aus finiten PREWARP-Pixeln ableiten.

### 9.2 Geometrische Supportberechnung

Coverage verwendet denselben Polygonrasterisierer einschließlich lokaler
Subdivision und Sampleverwerfung wie Uniform. Pro Zielstreifen und akzeptiertem
Frame werden die positiven Schnittflächen zunächst zu `B_f,c(q)` summiert.
Framecount, `sum B_f,c` und `sum B_f,c²` bleiben getrennte Größen. Das geometrische
`n_eff` entspricht damit dem Uniform-`n_eff` bei vollständig finiten Quellen.

Die Framepopulation wird vor der ersten Streifenausgabe durch eine lokale
Geometrievorprüfung festgelegt. Ein ausgeschlossener Frame kann keine bereits
publizierten Streifen teilweise beeinflussen. Affine Quellzeilen werden durch
Inverseabbildung des Zielstreifens eingeschränkt; lokale Modelle werden ohne
unbewiesene Bounding-Box-Annahme vollständig besucht.

### 9.3 Masken und unabhängiger Gatebereich

`analysis_common_mask` basiert auf dichten Frame-Footprints: Alle Quellpixel
werden mit `pixfrac=1` und ohne CFA-Kanalunterscheidung geometrisch rasterisiert.
Ein Zielpixel gehört dazu, wenn mindestens
`ceil(common_overlap_required_fraction * akzeptierte_Frames)` dieser Footprints
beitragen. Dies ist unabhängig vom dünnen CFA-Rekonstruktionssupport.

`reconstruction_support_mask` verlangt mindestens einen tatsächlichen
Drizzlebeitrag in jedem aktiven Kanal (MONO nur L). Supportanteile, gewichtete
`n_eff`-Perzentile und kanalspezifische Löcher werden innerhalb der unabhängigen
Analysefläche geprüft. Fehlender Kanalsupport zählt dort als null. Komponenten,
die an die Außenseite der Analysefläche anschließen, gelten als Randverlust;
vollständig innenliegende Komponenten als Löcher. Der Flächenanteil erfasst auch
Randverlust. Eine leere Analysemaske scheitert am Mindestpixelgate.

Der Default `common_overlap_required_fraction=1` verlangt den Überlapp aller
dichten Frame-Footprints, nicht Beiträge aller Frames in jedem dünnen CFA-Kanal.
Die anderslautende Interpretation und Defaultkalibrierung aus §30.10 ist
überholt. Mehr Frames können eine Schnittmenge nicht vergrößern.

Die spätere COMMON_OVERLAP-Integration muss dieselben Masken übernehmen.
Nichtfinite Quellwerte und Clipping dürfen den Rekonstruktionssupport nur
verkleinern; vor einer Ausgabe ist dessen erforderliche Mindestdeckung erneut
zu prüfen. Der aktuelle Diagnosepfad beansprucht noch keinen vollständigen
COMMON_OVERLAP-/Resume-Vertrag.

### 9.4 Artefaktfelder

`SAMPLING_GEOMETRY` schreibt unabhängig vom Gate-Ergebnis atomar:

```text
artifacts/sampling_geometry.json
```

Hauptfelder:

```json
{
  "schema_version": 1,
  "coverage_source": "forward_drizzle_geometry",
  "kernel": "square",
  "pixfrac": 0.8,
  "internal_scale": 2,
  "sampling_plan_hash": "...",
  "coverage_geometry_hash": "...",
  "coverage_gate": {
    "passed": true,
    "valid_frames": 0,
    "dither_spread_circular_px_diagnostic": {"x_p10": 0.0, "y_p10": 0.0},
    "supported_fraction": {"R": 0.0, "G": 0.0, "B": 0.0},
    "geometric_uniform_neff_p10": {"R": 0.0, "G": 0.0, "B": 0.0},
    "required_channel_neff": 0.0,
    "largest_internal_hole_area_px": {"R": 0, "G": 0, "B": 0},
    "analysis_pixels": 0,
    "violations": []
  }
}
```

`common_overlap.json` verweist bei bestandenem Gate auf
`sampling_geometry.json` und dessen `coverage_geometry_hash`; bei einem
Gatefehler wird `COMMON_OVERLAP` nicht gestartet und folglich auch kein
irreführendes `common_overlap.json` erzeugt.

### 9.5 Coverage-Gate

Das `coverage_gate` (6.1/6.2) wird am Ende von `SAMPLING_GEOMETRY` auf den
Ergebnissen der geometrischen Coverage und der dort bereits gebildeten
`analysis_common_mask` ausgewertet, bevor `COMMON_OVERLAP` Masken schreibt und
bevor Q-Maps oder Rekonstruktion laufen. Ein verletztes Gate beendet den Run
fail-closed mit dem verletzten Schlüssel, dem betroffenen Kanal und dem Ist-Wert
im Phase-Event und in `sampling_geometry.json`. Es gibt keinen stillen Wechsel
auf `internal_scale = 1` und keinen Methodenfallback; der Benutzer kann
`internal_scale` explizit ändern und den Run neu starten.

---

## 10. Normalisierte CFA-Quelle und Cache-Vertrag

### 10.1 Wiederverwendung des vorhandenen Caches

`RunnerFrameCache` speichert bereits normalisierte Vollframes unter
`cache/normalized_frames`. Dieser Cache wird im neuen Pfad zur verbindlichen
Rekonstruktionsquelle.

Erweiterungen am API:

```cpp
bool extract_normalized_region(
    size_t frame_index, int x0, int y0, int width, int height,
    Matrix2Df& out) const;

void set_preserve_normalized_files(bool preserve);
std::filesystem::path normalized_cache_dir() const;
```

### 10.2 Cache-Metadaten

Neue Datei:

```text
cache/normalized_frames/metadata.json
```

Pflichtfelder:

- Schema-Version;
- Framezahl und Frameabmessungen;
- Datentyp und Byte-Reihenfolge;
- Input-Manifest-Hash;
- Normalisierungsartefakt-Hash;
- Kalibrationskonfigurations-Hash;
- Farbmodus und Bayer-Pattern;
- pro Frame stabiler `frame_id`, kanonischer `source_index`, `has_data`,
  Dateigröße und verpflichtende Inhaltschecksumme der Cachedatei;
- CFA-Ursprung beziehungsweise MONO-Kanalvertrag und Sensororientierung;
- Erzeugungszeit und Build-ID. Die Build-ID ist nur Provenienz; die
  Gültigkeitsentscheidung erfolgt über Schema-, Format- und Inhaltshashes, damit
  ein reiner Rebuild keinen semantisch identischen Cache invalidiert.

### 10.3 Resume-Validierung

Direktes Resume ab `FORWARD_DRIZZLE` ist nur erlaubt, wenn:

1. `registration_sampling.json` vorhanden und parsebar ist;
2. dessen `frame_id`-/`source_index`-Folge exakt zum Normalized-Cache und zum
   Inputmanifest passt;
3. die Source-Identity-Hashes von Normalized- und Q-Map-Cache übereinstimmen;
   der registrierungsabhängige `plan_hash` wird bewusst nicht als Q-Map-Hash
   missbraucht;
4. alle als gültig markierten Frames mit passender Inhaltschecksumme im Cache
   vorhanden sind;
5. Frameabmessungen, Farbmodus, Sensororientierung, Bayer-Pattern und CFA-Ursprung
   übereinstimmen;
6. der jeweilige domänenspezifische Konfigurationshash aller semantisch
   relevanten Parameter passt;
7. `sampling_geometry.json`, Masken und Outputs dieselbe Ausgabegeometrie und
   denselben `coverage_geometry_hash` besitzen.

Fehlt eine Abhängigkeit, endet das direkte Resume mit einem präzisen
Fehlergrund. Der Runner darf keine alte PREWARP-Datei als Ersatz lesen.

---

## 11. CPU-Referenz: CFA-Forward-Drizzle

### 11.1 Neue Dateien

```text
tile_compile_cpp/include/tile_compile/reconstruction/forward_drizzle.hpp
tile_compile_cpp/src/reconstruction/forward_drizzle.cpp
tile_compile_cpp/apps/runner_phase_forward_drizzle.hpp
tile_compile_cpp/apps/runner_phase_forward_drizzle.cpp
tile_compile_cpp/tests/test_forward_drizzle.cpp
```

### 11.2 Öffentliche Konfiguration

```cpp
enum class DrizzleKernel { square };

struct ForwardDrizzleClippingConfig {
  float clip_sigma_low = 3.0f;
  float clip_sigma_high = 3.0f;
  float min_fraction = 0.4f;
  float min_n_eff = 3.0f;
};

struct ForwardDrizzleConfig {
  int internal_scale = 2;
  int output_scale = 1;
  DrizzleKernel kernel = DrizzleKernel::square;
  float pixfrac = 0.8f;
  int robust_passes = 2;
  int min_clip_contributors = 5;
  int chunk_rows = 0;              // 0 = auto
  int chunk_halo_rows = -1;        // -1 = auto
  size_t memory_budget_mb = 0;     // 0 = auto
  ForwardDrizzleClippingConfig clipping;
};
```

Die Struktur bildet `reconstruction.drizzle` und `reconstruction.clipping`
aus 6.1 vollständig ab. Der öffentliche Root wird als
`config::ReconstructionConfig` mit expliziten Unterstrukturen
`ForwardDrizzleConfig`, `CoverageGateConfig`, `QualityPyramidConfig`,
`MultibandConfig` und `ReconstructionDiagnosticsConfig` modelliert; zusätzlich
enthält er `delete_source_cache_after_run`, `keep_profile_cache_after_run` und
`common_overlap_required_fraction`. Es gibt keinen `AqmhConfig`-Alias im
aktiven Vertrag. Parser, Serialisierung und beide Schemas müssen exakt dieselben
Defaults und Grenzen verwenden.

### 11.3 Ergebnisstruktur

```cpp
struct DrizzleProfileStore {
  // Disk-/mmap-gestützte, transaktionale Ebenen mit read_region/write_region;
  // Tests dürfen dafür eine kleine In-Memory-Implementierung verwenden.
  ProfilePlaneStore R;
  ProfilePlaneStore G;
  ProfilePlaneStore B;
  ProfilePlaneStore luma;
  ProfilePlaneStore weight_sum_R;
  ProfilePlaneStore weight_sum_G;
  ProfilePlaneStore weight_sum_B;
  ProfilePlaneStore weight_sum_L;
  ProfilePlaneStore n_eff_R;
  ProfilePlaneStore n_eff_G;
  ProfilePlaneStore n_eff_B;
  ProfilePlaneStore n_eff_L;
};

struct ForwardDrizzleResult {
  ColorMode color_mode = ColorMode::MONO;
  DrizzleProfileStore uniform_control;
  DrizzleProfileStore raw_quality;
  std::vector<DrizzleProfileStore> detail_profiles;
  ProfileMaskStore support_R;
  ProfileMaskStore support_G;
  ProfileMaskStore support_B;
  ProfileMaskStore support_L;
  ProfileMaskStore combined_support;
  ForwardDrizzleDiagnostics diagnostics;
};
```

Die Store-Typen sind logische Ergebnisreferenzen, keine Zusage vollresidenter
Matrizen. Der Produktionspfad darf U/R/F/M bei 2x nie gleichzeitig als
Vollbilder im RAM halten: Rekonstruktion schreibt transaktionale Profilstores,
Mehrband liest sie mit dem in 14.7 definierten Halo streifenweise und löscht
nicht mehr benötigte temporäre Profile nach atomarem Phasencommit. Während eines
Chunks werden außerdem keine Vollbild-Frame-Buffers für alle Frames gleichzeitig
gehalten. Das Host-Speicherbudget umfasst Rekonstruktion **und** Mehrbandfusion.

### 11.4 CFA-Farbzuordnung

Die Farbe eines Samples wird ausschließlich aus seinen ganzzahligen
Cache-Quellkoordinaten, dem ursprünglichen Bayer-Pattern und dem persistierten
CFA-Ursprung bestimmt:

```text
x_sensor_mod2 = (x_cache + cfa_origin_x) mod 2
y_sensor_mod2 = (y_cache + cfa_origin_y) mod 2
```

Canvasoffset, Rotation und Dither dürfen die Farbzuteilung nicht ändern. Ein
Crop mit ungeradem Ursprung aktualisiert den CFA-Ursprung; Spiegelung,
Transposition oder sonstige Orientierungsänderung muss Pattern und Ursprung
explizit transformieren oder wird vor M1 fail-closed abgelehnt. Alle Frames
eines Runs müssen denselben effektiven Vertrag besitzen.

Beispiel `RGGB`:

```text
(x even, y even) -> R
(x odd,  y even) -> G
(x even, y odd)  -> G
(x odd,  y odd)  -> B
```

G1 und G2 akkumulieren in denselben Grünkanal. Diagnostisch werden ihre
Coveragewerte getrennt gezählt, um CFA-Paritätsfehler sichtbar zu machen. MONO
überspringt die CFA-Zuordnung vollständig und akkumuliert jedes Sample in `L`;
R/G/B-Stores und -Outputs bleiben in diesem Modus absent, nicht als künstliche
Kopien von L gefüllt.

Für OSC wird die Arbeitsluminanz einheitlich und vor der finalen
Outputskalierung als `L = 0.25 * R + 0.50 * G + 0.25 * B` gebildet. Diese feste,
Green-betonte Definition wird für Q-bezogene Diagnostik und Runtime-Validation
verwendet und im Artefakt versioniert; PCC oder spätere Farbfaktoren ändern sie
nicht rückwirkend.

### 11.5 Koordinaten

Für ein Quellsample mit Pixelzentrum

```text
s = (x + 0.5, y + 0.5)
```

wird die native Canvasposition `q_native` über den Sampling-Plan berechnet.
Die interne Position ist:

```text
q_internal = internal_scale * q_native
```

Die FITS-/OpenCV-Konventionen werden an einer einzigen Adaptergrenze
konvertiert. Im Rekonstruktionskern gilt ausschließlich Pixelzentrumgeometrie;
Mischungen aus ganzzahliger Pixelkante und Pixelzentrum sind verboten.

### 11.6 Square-Droplet-Kernel

Ein Quellpixel wird als am Pixelzentrum `s = (x + 0.5, y + 0.5)` zentriertes
Quadrat mit Quellkantenlänge `pixfrac` behandelt; seine Ecken liegen bei
`s +/- pixfrac / 2`. **Verbindliche Geometrie:**

- bei rein affiner Abbildung ist das geometrische Abbild dieses Quadrats ein
  Parallelogramm im internen Zielraster (bei reiner Translation ein
  achsparalleles Quadrat der Kantenlänge `drop_size = pixfrac * internal_scale`);
- bei aktivem lokalen Modell ist das Abbild im Allgemeinen gekrümmt und darf
  nicht als exakt affines Parallelogramm ausgegeben werden. Das Quadrat wird
  adaptiv in Quads unterteilt. Pro Quad werden Ecken, Kantenmittelpunkte und
  Zentrum mit der geguardeten Source→Canvas-Inversion aus 7.3 abgebildet und
  gegen die bilineare Quad-Näherung geprüft. Verbindlich gelten:

  ```text
  subdivision_position_epsilon_internal_px = 0.05
  max_subdivision_depth = 2
  subdivision_area_relative_epsilon = 0.005
  per_frame_inversion_error_rate_max = 0.001
  ```

  Die Flächenkonvergenz wird zwischen zwei aufeinanderfolgenden
  Subdivisionsstufen geprüft, nicht nur gegen die Center-Jacobi-Näherung. Ein
  Blatt wird erst akzeptiert, wenn Positions- und Flächenkriterium erfüllt
  sind. Ist dies bei Maximaltiefe nicht der Fall, wird das Subdroplet verworfen
  und als Inversionsfehler gezählt. Überschreitet ein Frame 0,1 % verworfene
  Samples, wird er vollständig ausgeschlossen. Parameter, maximale beobachtete
  Fehler und verworfene Subdroplets werden im Artefakt protokolliert.

Für jedes überlappte Zielpixel wird `K` als Polygon-Rechteck-Schnitt der affinen
Dropletfläche beziehungsweise als Summe der adaptiven Subdroplet-Schnitte
berechnet. Ein achsparalleles Droplet ohne Mitdrehung mit dem Frame ist
**nicht** zulässig.

Für affine Abbildungen gilt exakt:

```text
K >= 0
sum_q K(q, s) = pixfrac^2 * internal_scale^2 * |det J_f|
```

Bei lokalen Abbildungen wird die rechte Seite durch das Flächenintegral von
`|det J_f(s')|` über das Droplet ersetzt und numerisch gegen die Summe der
Subdropletflächen geprüft. `J_f` ist die Jacobi-Matrix der **nativen** Abbildung
`q_native = W^-1(s)` einschließlich der lokalen Ableitung aus 7.3; der Faktor
`internal_scale^2` darf nicht ein zweites Mal über eine „interne" Jacobi-Matrix
eingehen. Für rein translatorische oder rotatorische Frames ist
`|det J_f| = 1`. Testfälle mit Skalierungs- und lokalem Warp prüfen die jeweils
korrekte Flächenform, nicht pauschal eine Konstante.

Vor der Frame-Aggregation darf das Sample entweder mit der absoluten
Überdeckungsfläche oder mit einer auf die Dropletfläche normierten Fläche
eingezahlt werden. Der Code verwendet eine einzige dokumentierte Variante für
alle Profile. Der Quotient aus Wertsumme und Geometriesumme (`A_f,c / B_f,c`,
siehe 11.7) dividiert `|det J_f|` heraus und erhält damit konstante
**Oberflächenhelligkeit** unabhängig von lokaler Flächenverzerrung. Das ist
nicht gleichbedeutend mit unveränderter roher Pixelsumme bei geändertem
`output_scale`: integrierter photometrischer Flux wird als apertursummierte
Oberflächenhelligkeit multipliziert mit der WCS-Pixelfläche gemessen. Die
synthetischen Fluxgates verwenden genau diese skaleninvariante Definition.
Coverage, Weight-Sum und physischer Aperturflux bleiben getrennte Größen.

### 11.7 Frame-lokale Aggregation

Mehrere CFA-Samples desselben Frames können dasselbe Zielpixel und denselben
Kanal überlappen. Sie werden zunächst zu genau einem Frame-Beitrag aggregiert:

```text
A_f,c(q) = sum_s K(q,s) * v_f(s)
B_f,c(q) = sum_s K(q,s)
x_f,c(q) = A_f,c(q) / B_f,c(q), falls B_f,c(q) > 0
```

`x_f,c(q)` ist die Einheit für robustes Clipping. Einzelne Quellsamples dürfen
nicht als statistisch unabhängige Frames behandelt werden.

Zusätzlich werden frame-lokal akkumuliert:

```text
Q_composite_f,c(q)
Q_scale0_f,c(q)
Q_scale1_f,c(q)
artifact_confidence_f,c(q)
```

jeweils geometrisch mit demselben `K` gemittelt.

### 11.8 Gemeinsames robustes Clipping

Clipping erfolgt pro Zielpixel und Farbkanal über die vorhandenen
`x_f,c(q)`-Beiträge.

Verbindlicher Ablauf:

1. endliche Frame-Beiträge sammeln;
2. wenn die Zahl endlicher Beiträge `< min_clip_contributors` ist: **kein
   Clipping**, alle endlichen Beiträge bleiben gültig (weiter bei Schritt 7).
   Das schützt die dünn belegten R/B-Kanäle bei kleinen Framezahlen vor
   MAD-Instabilität;
3. deterministisch nach Wert und bei Gleichstand nach Frameindex sortieren;
4. geometrisch mit `B_f,c(q)` gewichteten Median und MAD bestimmen; Q-, globale
   Qualitäts- oder Profilgewichte dürfen die Clippingmaske nicht beeinflussen;
5. unteren und oberen Grenzwert aus den vorhandenen asymmetrischen
   `clip_sigma_low`/`clip_sigma_high` bilden;
6. Beiträge außerhalb der Grenzen markieren;
7. bis `robust_passes` wiederholen oder bei unveränderter Maske abbrechen;
8. `min_fraction` und `min_n_eff` gegen den geometrisch möglichen
   Frame-Support prüfen. Scheitert eine der beiden Prüfungen, wird das
   Pixel für diesen Kanal in **allen** Profilen als nicht belegt markiert
   (`channel_support_c(q) = 0`, Wert nichtfinit) und in
   `clipping.rejected_pixels_per_channel` gezählt. Es gibt keine Auffüllung
   aus Nachbarpixeln oder aus dem Uniform-Profil.

`min_fraction` ist dabei die Zahl akzeptierter Frame-Beiträge geteilt durch die
Zahl geometrisch möglicher Frame-Beiträge mit `B_f,c(q) > 0`, nicht ein Quotient
von Q-Gewichten. Die resultierende Akzeptanzmaske wird unverändert für Uniform,
Raw-Forward-Drizzle und alle Detailprofile verwendet. Q-Gewichte dürfen nicht
bestimmen, ob ein Sample als Ausreißer gilt.

Bei degenerierter MAD wird die vorhandene numerische Guard-Semantik übernommen:

- identische Werte bleiben gültig;
- einzelne nichtfinite Werte werden ausgeschlossen;
- kein willkürliches epsilonbasiertes Wegclippen von konstanten Hintergründen.

### 11.9 Gewichtsprofile

Der globale effektive Framefaktor wird genau einmal vor der
Pixelrekonstruktion berechnet:

```text
G_eff(f) = G_quality(f)
         * model_prediction_factor(f)
         * registration_residual_factor(f)
```

Die Pipeline erhält dafür eine explizite Struktur

```cpp
struct QualityFrameWeight {
  std::string frame_id;
  float g_quality = 0.0f;
  float model_prediction_factor = 0.0f;
  float registration_residual_factor = 0.0f;
  float g_eff = 0.0f;
};
struct QualityFrameWeightPlan {
  std::string source_identity_hash;
  std::string sampling_plan_hash;
  std::string source_quality_config_hash;
  std::vector<QualityFrameWeight> frames;
};
```

Eine doppelte Anwendung der Registrierungsfaktoren in Pipeline und Rekonstruktor
ist ausgeschlossen. Beide Werte werden in `runner_phase_registration` berechnet,
im `RegistrationSamplingPlan` persistiert und von `GLOBAL_QUALITY` nur gelesen.

`registration_residual_factor(f)` übernimmt zunächst unverändert die bestehende
geguardete Funktion in deren dokumentierten Proxy-Pixeleinheiten:

```text
median_penalty = clamp((median_px - 0.18) / (0.70 - 0.18), 0, 1)
p90_penalty    = clamp((p90_px - 0.45) / (1.40 - 0.45), 0, 1)
penalty        = max(median_penalty, 0.75 * p90_penalty)
registration_residual_factor = clamp(1 - 0.45 * penalty, 0.55, 1)
```

Das Referenzframe erhält ohne Residualmessung `1.0`. Ein anderes gültiges Frame
ohne anwendbare Residualmessung erhält konservativ `0.55` und
`residual_applicable=false`; ein geometrisch ungültiges Frame wird verworfen.
Proxy-Skalierung, Median, p90, Anwendbarkeit und Faktor werden persistiert.

`model_prediction_factor(f)` ist verbindlich:

```text
direkt gemessen oder direkt astrometrisch gerettet -> 1.0
modelliert/interpoliert/blended -> clamp(1 / (1 + 0.4 * chain_depth), 0.5, 0.9)
nearest-copy                   -> min(obiger Wert, 0.5)
unresolved                     -> Frame ungültig
```

Eine spätere kovarianzbasierte Unsicherheitsfunktion ist eine versionierte
Methodikänderung. Im ersten Release gelten ausschließlich die obigen Formeln.
`model_prediction_factor` und `registration_residual_factor` werden genau einmal
in `G_eff` multipliziert; `A_registration` steuert später nur Alpha und ist kein
zweites Profilgewicht.

**Herkunft von `G_quality(f)` nach PREWARP-Entfernung.** Der bisherige globale
Frame-Qualitätsfaktor (bisher `G_aqmh`) wurde in der alten
`AQMH_GLOBAL_QUALITY`-Phase auf der prewarpten Luminanz berechnet. Diese
Eingabe existiert im Zielpfad nicht mehr. Im Single-Method-Pfad wird
`G_quality(f)` in der Phase `GLOBAL_QUALITY` (Abschnitt 5.2) aus dem
**source-space CFA-Green-Proxy** (Abschnitt 13.2) desselben Frames berechnet,
mit derselben mathematischen Definition wie bisher (globale
SNR-/Schärfe-/Sternstatistik pro Frame), jedoch ohne geometrischen PREWARP. Der
Proxy darf fehlende Grünpositionen ausschließlich für die Analyse
interpolieren; das Nutzsignal bleibt unberührt. Die Phase liegt deshalb zwingend
**nach** `SOURCE_QUALITY_MAPS`, weil
sie denselben Proxy konsumiert. Konsequenzen:

- Die Definition der Kennzahl bleibt gleich, ihre Eingabe ändert sich. Der
  Vergleich „neuer Pfad vs. PREWARP-AQMH" im Go/No-Go-Gate mischt damit eine
  geänderte Gewichts­eingabe mit der geänderten Geometrie. Das Gate wird deshalb
  zusätzlich mit `G_quality(f) := 1` (reines Uniform-Gewicht) als Kontroll­lauf
  gemessen, um den Geometrie-Effekt isoliert zu zeigen.
- Der Proxy-/Pyramiden-Hash geht in die Q-Map-Hashdomäne ein (Abschnitt 18.3);
  eine Änderung der Proxy-Version invalidiert `G_quality` und alle Q-Profile.
- `G_quality` ist ein globaler Skalar pro Frame und darf nicht mit den
  frame-lokalen `Q_*_f,c(q)` verwechselt werden.
- Wertebereich: `G_quality(f)`, `Q_composite`, `Q_scale*` und beide
  Registrierungsfaktoren liegen in `[0, 1]`; damit ist
  `w_profile <= w_uniform` und `A_coverage` (14.4) wohldefiniert.
  **Präzisierung (2026-09-05, beim Implementieren entdeckt, siehe §30.19):**
  Die wiederzuverwendende Bestandsformel
  `metrics::calculate_global_weights_with_stars()` liefert
  `exp(k·clamp(Q, lo, hi))` — **unbeschränkt** positiv, nicht in `[0,1]`.
  Der `[0,1]`-Vertrag wird erfüllt, indem obenauf die logistische Stauchung
  `G_quality(f) = w/(1+w) = sigmoid(k·Q)` angewendet wird (Bestandsformel
  bleibt wörtlich unverändert). Ergebnis liegt im **offenen** Intervall
  `(0,1)` — nie exakt 0 (kein Konflikt mit dem separaten Q=0-Veto) und nie
  exakt 1.

Für akzeptierte Beiträge gelten:

```text
w_uniform = B_f,c(q)

w_raw = B_f,c(q)
      * G_eff(f)
      * Q_composite_f,c(q)

w_fine = B_f,c(q)
       * G_eff(f)
       * pow(Q_scale0_f,c(q), fine_quality_exponent)

w_medium = B_f,c(q)
         * G_eff(f)
         * pow(Q_scale1_f,c(q), medium_quality_exponent)
```

Q=0 bleibt ein explizites Veto. Fehlende Q-Maps führen nicht zu einem
ungewichteten Fallback innerhalb eines Qualitätsprofils. Das Uniform-Control
bleibt davon unabhängig.

### 11.10 Effektive Samplezahl

Für jedes Profil und jeden aktiven Kanal:

```text
n_eff_profile,c(q) = (sum_f w_profile,f,c(q))^2
                     / sum_f w_profile,f,c(q)^2
```

OSC speichert `n_eff_R/G/B` separat; MONO speichert `n_eff_L`. Eine
luminanzbasierte Zusammenfassung darf nur zusätzliche Diagnostik sein und nie
die dünnere R/B-Coverage verdecken. Zusätzlich werden geometrische Framezahl,
akzeptierte Framezahl und kanalspezifische Coverage gespeichert. Diese Größen
dürfen nicht miteinander verwechselt werden.

### 11.11 Chunking, Rand-Halo und Speicherbudget

Der Rekonstruktor arbeitet auf zusammenhängenden Zielzeilen.

**Streifengrenzen (aktueller CPU-Vertrag).** Jeder Zielstreifen enumeriert
sämtliche Quellfootprints, die seine Kernzeilen schneiden. Frame-lokale Beiträge
werden erst danach kombiniert. Damit benötigt die pixelweise Coverage-/Uniform-
Akkumulation keinen duplizierten Ausgabe-Halo. `chunk_halo_rows` bleibt als
Kompatibilitätsfeld erhalten; seine Werte verändern diesen CPU-Operator nicht.
Spätere räumliche Mehrbandfilter benötigen weiterhin ihren eigenen Filterhalo.
Tests vergleichen Kernhöhen 1 und Vollbild bei Rotation, Skalierung und Scherung.

Auto wählt höchstens 256 Zeilen innerhalb des Budgets. Der aktuelle CPU-
Referenzpfad verwendet einen Worker. Jede spätere Parallelisierung muss ihre
zusätzlichen Puffer im Budget erfassen, bevor weitere Worker gestartet werden.
Die Convenience-API mit Vollbildresultat budgetiert auch dieses vor der ersten
Allokation; produktive Uniform-Diagnostik nutzt den Streaming-Sink.

Automatische Chunkhöhe:

```text
bytes_per_row =
  output_width * (
      thread_count * (
          profile_accumulator_bytes      # pro Profil und Kanal
        + framelocal_contribution_bytes   # A_f,c, B_f,c, Q_* frame-lokal
        + clipping_state_bytes)
    + per_frame_sample_bytes * active_frames_in_band
    + shared_readonly_bytes
  )
  + safety_margin

chunk_rows = floor(memory_budget / bytes_per_row)
```

Der Faktor `thread_count` ist zwingend: der deterministische CPU-Pfad hält je
Worker eigene frame-lokale und Profil-Akkumulatoren (Abschnitt 11.12). Bei 4
Profilen × 3 Kanälen × N Threads ist das der dominierende Term.

Grenzen:

- mindestens 1 Kernzeile (plus Halo);
- höchstens Ausgabebildhöhe;
- ein konfigurierter `chunk_rows`-Wert überschreibt Auto-Sizing nur, wenn die
  geschätzte Speichernutzung inklusive `thread_count`-Faktor das harte
  Prozessbudget nicht verletzt;
- bei CUDA wird separat gegen freien Device-Speicher und Host-Pinned-Speicher
  geplant;
- Allokationsfehler halbieren die Chunkhöhe begrenzt und protokolliert;
- nach Ausschöpfung der Retries werden alle temporären CUDA-Ergebnisse der
  Phase verworfen und `FORWARD_DRIZZLE` vollständig auf CPU neu gestartet; es
  gibt weder gemischte CPU-/CUDA-Outputs noch einen Methodenfallback.

**Rechen-, RAM- und Temporärdiskvertrag.** Die Fixpunkt-Inversion des lokalen
Modells (Abschnitt 7.3) läuft pro Quellsample und Frame mit bis zu 6 Iterationen;
bei 2x-Raster vervierfacht sich zusätzlich die Zielpixelzahl. Verbindliche harte
Runtime-Gates sind:

```text
rss_growth = peak_rss - rss_at_phase_start
rss_growth <= resolved_memory_budget * 1.05 + 256 MiB

required_free_temp = estimated_temp_peak * 1.20
                   + max(2 GiB, 0.05 * filesystem_capacity)
available_temp >= required_free_temp
```

`resolved_memory_budget` ist der explizite Phasenwert oder das vorhandene globale
Runnerbudget. Der Peak wird durch kleinere Chunks eingehalten; ist selbst eine
Kernzeile nicht möglich, endet die Phase fail-closed. Temporärspeicher wird vor
Phasenstart getrennt geschätzt für frameabhängige Normalized-/Q-Caches und
outputabhängige U/R/F/M-Stores. Unterschreitung bricht vor dem ersten Store-Write
ab.

Das Performance-Releasegate verwendet keine absolute Zeit über verschiedene
Hardwareklassen, sondern

```text
throughput = processed_source_samples / forward_drizzle_wall_seconds
```

auf eingefrorener Referenzhardware und identischem Buildprofil. Der Median aus
drei kleinen deterministischen Benchmarks darf gegenüber der eingefrorenen
M8-Baseline höchstens 20 % sinken. Der reale Referenzdatensatz mit 100 Frames und
~24 MP bestätigt Peak-RSS, Temporärdisk und Durchsatz, ist aber nicht alleinige
Timingbasis. Hardware, Build, kalter/warmer Cache, Threadzahl, Backend und alle
Ist-/Grenzwerte werden in `forward_drizzle.json` protokolliert. Eine Verletzung
ist ein M9-Release-No-Go; RAM-/Diskverletzungen sind zusätzlich Runtimefehler.

### 11.12 Determinismus

- feste Frame-Reihenfolge;
- feste Sortier-/Tie-Break-Regel;
- keine atomare Floating-Point-Akkumulation mit nichtdeterministischer
  Reihenfolge im CPU-Referenzpfad;
- parallele Worker erzeugen frame-lokale Buffers und reduzieren in fester
  Reihenfolge;
- Chunkhöhe und Threadzahl dürfen Ergebnisse nur innerhalb der dokumentierten
  Floating-Point-Toleranz verändern.

---

## 12. Internes 2x-Raster und Downstream-Geometrie

### 12.1 Abmessungen

```text
canvas_width_internal  = canvas_width_native  * internal_scale
canvas_height_internal = canvas_height_native * internal_scale
```

Canvasoffsets werden ebenfalls skaliert. Crop-Rechtecke werden erst in nativer
Geometrie bestimmt und danach exakt skaliert, damit keine unterschiedlichen
Rundungsregeln zwischen Masken, RGB und WCS entstehen.

Ist `internal_scale=2` und `output_scale=1`, wird **nach** Auswahl des Kandidaten
ein einziger deterministischer 2x2-Flächenmittelwert auf Bild, Masken und
relevante Gewichtsebenen angewendet. Im Qualitäts- und Ausgabesupport gilt
verbindlich:

```text
valid_out = valid_00 && valid_01 && valid_10 && valid_11
value_out = 0.25 * (v_00 + v_01 + v_10 + v_11)
n_eff_out = min(n_eff_00, n_eff_01, n_eff_10, n_eff_11)
```

Ungültige Subpixel gehen nicht als Null oder teilnormalisierter Mittelwert ein;
der 1x-Pixel wird ungültig und über Maske/Crop entfernt. Dadurch bleibt die
Transferfunktion räumlich konstant. Uniform, Raw und Multiband werden vor
Runtime- und M9-Vergleichen mit demselben Operator in dieselbe
`output_scale`-Geometrie gebracht. Die Flächenmittelung erhält die in 11.6
definierte Oberflächenhelligkeit; Aperturflux wird mit der WCS-Pixelfläche
verglichen. Operator, Vierersupport und Randregel gehören zum
`multiband_config_hash`.

### 12.2 WCS

Bei einer Ausgabeskalierung `S = output_scale` sei
`canvas_offset_native` die Verschiebung der ursprünglichen Referenzgeometrie in
den nativen Canvas und `crop_origin_out` die obere/linke, entfernte Cropkante in
**Ausgabepixeln**. Komponentenweise gilt:

```text
CRPIX_canvas_native = CRPIX_in + canvas_offset_native
CRPIX_out = S * (CRPIX_canvas_native - 0.5) + 0.5 - crop_origin_out
CD_out    = CD_in / S
CDELT_out = CDELT_in / S
```

Der Cropterm hat damit ein explizites Minuszeichen; bei einem in nativen Pixeln
bestimmten Crop ist `crop_origin_out = S * crop_origin_native`. Tests prüfen
Pixelzentren, positive/negative Canvasoffsets und einen nichtnulligen Crop gegen
bekannte Weltkoordinaten. Wenn eine CD-Matrix vorhanden ist, werden nicht
zusätzlich widersprüchliche CDELT-Werte erzeugt.

### 12.3 Downstream

Folgende Komponenten müssen 2x-Geometrie verarbeiten:

- `canvas_mask.fits` und `common_overlap_mask.fits`;
- Crop und Output-Offset;
- `reconstructed_R/G/B.fit` und `reconstructed_L.fit`;
- STACKING-Pass-through;
- Astrometrie-/WCS-Schreiben;
- BGE-Sampling und Masken;
- PCC-Sternradien/FWHM-Automatik;
- HMS;
- Report-/Preview-Skalierung.

Pixelbezogene Konfigurationsparameter werden nicht pauschal verdoppelt. Jede
Downstream-Komponente muss kennzeichnen, ob ein Parameter in Quellpixeln,
nativen Canvaspixeln oder aktuellen Ausgabepixeln definiert ist.

### 12.4 Korreliertes Rauschen

Forward-Drizzle mit `pixfrac < 1` und `internal_scale = 2` erzeugt
**pixel-zu-pixel korreliertes Rauschen** — benachbarte Ausgabepixel sind nicht
mehr statistisch unabhängig. Jede Downstream-Komponente, die unabhängiges
Pixelrauschen annimmt, muss geprüft und ggf. angepasst werden:

- BGE-Hintergrund-RMS und Ausreißerschwellen;
- PCC-Sterndetektionsschwelle und SNR-basierte Sternablehnung;
- HMS-Rauschschätzung;
- jede SNR-Karte oder Fehlerfortpflanzung, deren Konstanten auf den alten,
  unkorrelierten Ausgabestatistiken kalibriert wurden.

Die Kernel-Autokorrelation ist aus `kernel`, `pixfrac` und `internal_scale`
analytisch bekannt und wird als Korrekturfaktor in `forward_drizzle.json`
ausgewiesen, damit Downstream-Schätzer die effektive Rauschbandbreite verwenden
können.

---

## 13. Source-space Qualitätskarten

### 13.1 Problem des aktuellen Q-Map-Pfads

`compute_aqmh_quality_map()` arbeitet derzeit auf einem bereits vorgewarpten
Bild und kombiniert alle berechneten Skalen über das geometrische Mittel zu
einer einzelnen Q-Map. Für die Mehrband-Rekonstruktion werden skalenspezifische Karten in
Quellkoordinaten benötigt.

### 13.2 CFA-aware Analyseproxy

Der Analyseproxy wird aus dem normalisierten CFA erstellt, ohne das spätere
Nutzsignal zu verändern.

Verbindliches Verfahren `proxy_version=1`:

1. Die beiden nativen Grünpositionen jedes 2x2-Bayer-Quads bleiben getrennt
   adressierbar und bilden auf dem Quad-Gitter

   ```text
   G_quad = 0.5 * (G1 + G2)
   ```

2. Schärfe-, Stern- und Rauschstatistik werden auf diesem gleichfarbigen
   Green-Gitter berechnet. Lokales Rauschen stammt aus einem robusten
   Green-Highpass, nicht aus Unterschieden zwischen R, G und B:

   ```text
   hp = G_quad - B3_blur(G_quad)
   sigma_green = 1.4826 * median(|hp - median(hp)|)
   ```

   Die lokale Variante verwendet dieselbe MAD-Definition in einem
   maskierten Fenster mit mindestens neun gültigen Green-Samples.
3. Nur für positionsbezogene Analysefunktionen darf ein full-resolution
   Green-Proxy edge-aware aus horizontalen/vertikalen Grünnachbarn ergänzt
   werden. Diese Interpolation verändert nie das Nutzsignal.
4. Die auf dem Quad-Gitter bestimmten positiven Quality-Werte werden
   deterministisch auf Source-Geometrie interpoliert. Die harte Zero-Veto-Maske
   wird separat mit konservativer Maskensemantik übertragen, sodass ein Veto
   niemals positiv interpoliert wird.
5. R/B-Chroma geht weder in `sigma_green` noch dominant in die lokale
   Schärferangfolge ein. Dadurch wird reale Objektfarbe nicht als Rauschen
   fehlklassifiziert.

MONO verwendet direkt die normalisierte L-Ebene als Proxy und dieselbe
Highpass-/MAD-Definition ohne CFA-Interpolation. Proxy-Version, CFA-Ursprung,
Orientierungsvertrag, B3-Kernel, Fensterregeln und sämtliche numerischen
Parameter gehören zum Source-Quality-Hash. Das Proxyverfahren erhält Tests gegen
Bayer-Checkerboard, farbige Sterne, schmalbandige MONO-Daten und Veto-Leckage.

### 13.3 Erweiterung des Ergebnisses

```cpp
struct ScaleQualityMap {
  int scale_index = 0;
  int downsample_factor = 1;
  Matrix2Df psi;
};

struct SourceQualityMapResult {
  Matrix2Df q_map;
  std::vector<ScaleQualityMap> scale_maps;
  Matrix2Df artifact_confidence;
  SourceQualityMapDiagnostics diagnostics;
};
```

Da mehrere Vollmaps pro Frame zu viel RAM benötigen, erhält die produktive
Funktion zusätzlich einen optionalen Sink/Callback:

```cpp
using QualityScaleMapSink =
    std::function<void(size_t scale_index, const Matrix2Df& map)>;
```

Jede skalenspezifische Map wird direkt nach Berechnung in den Cache geschrieben
und freigegeben. Nur der laufende Composite-Accumulator bleibt resident.

### 13.4 Cache-Layout

```text
cache/source_quality_maps/
  metadata.json
  composite/
    source_quality_composite_000000.bin
    ...
  scale_0/
    source_quality_s0_000000.bin
    ...
  scale_1/
  scale_2/
  scale_3/
  artifact/
```

Pflichtmetadaten:

```json
{
  "schema_version": 1,
  "coordinate_space": "source_cfa",
  "source_width": 3840,
  "source_height": 2160,
  "storage_divisor": 2,
  "dtype": "uint16",
  "source_identity_hash": "...",
  "normalized_cache_hash": "...",
  "source_quality_config_hash": "...",
  "source_quality_cache_hash": "...",
  "proxy_version": 1,
  "cfa_origin_x": 0,
  "cfa_origin_y": 0,
  "streams": ["composite", "scale_0", "scale_1", "scale_2", "scale_3", "artifact"]
}
```

`streams` listet nur tatsächlich vollständig committed Streams: in M3 zunächst
`["composite"]`, ab M5 die benötigten Skalen und `artifact`. Der
`source_identity_hash` umfasst geordnete Frame-IDs, Inhaltsidentität,
Quellabmessungen, Farbmodus, Sensororientierung, Bayer-Pattern/CFA-Ursprung und
den Normalized-Cache-Hash, aber **keine** Registrierung, Canvasgeometrie,
`internal_scale` oder `output_scale`. `source_quality_config_hash` umfasst
Proxy-Version, Pyramiden-/Q-Parameter, Storage-Divisor und Datentyp.
`source_quality_cache_hash` hasht das kanonische Streammanifest samt
Dateichecksummen; sein eigenes Metadatenfeld ist aus dieser Berechnung
selbstverständlich ausgeschlossen. Dadurch invalidiert eine reine
Neuregistrierung keine unveränderten Source-Q-Maps.

### 13.5 Region Reads

Der Forward-Drizzle-Rekonstruktor benötigt Q-Werte an Quellsamplepositionen.
Der Cache erhält deshalb ein Source-Region-API. Für einen Zielchunk wird pro
Frame die affine oder lokal gekrümmte Quell-Bounding-Box mit Sicherheitsmarge
bestimmt. Nur diese Zeilen werden dekodiert.

Die bestehende Zero-Veto-Semantik bleibt erhalten. Down-/Upsampling des
Map-Caches darf ein exaktes Null-Veto nicht in einen positiven Wert verwandeln.

---

## 14. Kontrollierte Mehrband-Rekonstruktion

### 14.1 Grundprinzip

Die Mehrband-Rekonstruktion schärft kein fertiges Bild. Sie kombiniert
Frequenzbänder mehrerer
Rekonstruktionen, die aus denselben akzeptierten CFA-Samples, aber
unterschiedlichen Qualitätsgewichten stammen.

Profile:

- `U`: Uniform-Control;
- `R`: Raw-Forward-Drizzle mit zusammengesetzter Q-Map;
- `F`: Fine-Profil mit Scale-0-Q;
- `M`: Medium-Profil mit Scale-1-Q.

### 14.2 À-trous-Zerlegung

Verwendet wird eine shift-invariante À-trous-Zerlegung mit separierbarem
B3-Spline-Kernel:

```text
h = [1, 4, 6, 4, 1] / 16
```

Für Level `j` werden zwischen den Koeffizienten `2^(j-1)-1` Nullen eingefügt.
Maskierte Faltung propagiert einen level-spezifischen Support:

```text
den_j = convolve(M_(j-1), h_j)
C_j   = convolve(C_(j-1) * M_(j-1), h_j) / den_j
M_j   = M_(j-1) && (den_j >= den_min_j)
D_j   = C_(j-1) - C_j, gültig nur auf M_(j-1) && M_j
```

`den_min_j` ist aus dem vollständig unterstützten Kernelgewicht als feste,
versionierte Schwelle abgeleitet und gehört zum Mehrbandhash. Jede Profil- und
Kanalkombination propagiert ihren eigenen Support. Pixel mit unzureichendem
Faltungsdenominator bleiben ungültig und werden nicht mit Nullwerten
aufgefüllt; im Mix setzt ein ungültiges Detailprofil `alpha_j=0`, während ein
fehlendes Raw-Band den gesamten Mehrbandpixel ungültig macht. Die
Rekonstruktionsidentität wird nur auf dem gemeinsamen gültigen Support des
jeweiligen Profils geprüft.

### 14.3 Bandzuordnung

Bei `levels = 3` (Default):

| Band | Quelle |
|---|---|
| `D1` fein | Fine-Profil `F` |
| `D2` mittel | Medium-Profil `M` |
| `D3` grob | Raw-Forward-Drizzle `R` |
| Rest `C3` | Uniform-Control `U` |

Allgemeine Regel für `levels = L`:

| `L` | `D1` | `D2` | `D3 … D(L)` | Grobrest `C(L)` |
|---|---|---|---|---|
| 1 | `F` | — | — | `U` |
| 2 | `F` | `M` | — | `U` |
| ≥ 3 | `F` | `M` | `R` | `U` |

Zusätzliche Levels über 3 verfeinern also nur die Skalen­trennung des
`R`-gestützten Anteils und führen keine neue Profilquelle ein. Bei `L <= 2`
liefert `R` kein eigenes Band, bleibt aber die Blendbasis jeder Banddifferenz
(siehe unten). `L = 1` bedeutet: Fine-Detail über Uniform-Grobrest.

Die tatsächliche Ausgabe wird nicht hart umgeschaltet, sondern gegen
Raw-Forward-Drizzle
geblendet:

```text
D_out,j(q) = D_R,j(q)
           + alpha_j(q) * (D_profile,j(q) - D_R,j(q))

X_out(q) = C_U,L(q) + sum_j D_out,j(q)
```

Bei `alpha_j = 0` ist das Band `j` exakt das Raw-Band. Für Bänder mit
`profile = R` ist `alpha_j` wirkungslos. **Die Gesamtausgabe bei `alpha ≡ 0`
ist nicht `R`, sondern**

```text
X_out = R - C_R,L + C_U,L
```

also Raw-Forward-Drizzle mit dem Uniform-Grobrest. Das ist die beabsichtigte
Semantik (Abschnitt 1: niedrige Frequenzen aus Uniform) und der minimale
Mehrbandkandidat; der Kandidat wird trotzdem vollständig gegen `R` und `U`
gegatet (15.3). Nur wenn `U`, `R`, `F`, `M` identisch sind, ist die Ausgabe
unabhängig von `alpha` exakt `R`.

### 14.4 Adaptives Alpha

`alpha_j` ist das Produkt mehrerer Vertrauensfaktoren und dadurch höchstens so
groß wie sein kleinster Faktor:

```text
alpha_j = alpha_cap
        * A_neff
        * A_coverage
        * A_separation
        * A_artifact
        * A_registration
```

Alle Faktoren liegen in `[0,1]`.

#### Effektive Samplezahl

Für OSC wird zuerst je Kanal gerechnet und anschließend konservativ das Minimum
verwendet; MONO verwendet nur L:

```text
A_neff,c = smoothstep(min_effective_samples,
                      full_effective_samples,
                      n_eff_profile,c)
A_neff   = min_c A_neff,c
```

#### Coverage

```text
A_coverage,c = clamp(profile_support_weight_c / uniform_support_weight_c, 0, 1)
A_coverage   = min_c A_coverage,c
```

Damit kann die dichtere G-Coverage kein gemeinsames Alpha freigeben, wenn R oder
B unzureichend belegt ist. Nicht aktive Kanäle gehen nicht in das Minimum ein.

#### Qualitätstrennung

Aus den vorhandenen frame-lokalen Q-Werten werden pro Zielpixel robuste
Quantile bestimmt:

```text
separation = Q_p90 - Q_p50
A_separation = smoothstep(min_quality_separation,
                          full_quality_separation,
                          separation)
```

Die Quantile werden mit geometrischem `B_f,c` auf der gemeinsamen akzeptierten
Framepopulation je aktivem Kanal bestimmt; für das gemeinsame OSC-Alpha gilt
wieder das Minimum der kanalspezifischen Separationsfaktoren. Sind alle Frames
lokal ähnlich bewertet, bleibt `A_separation=0`; Mehrband erzeugt dann keinen
künstlichen Kontrast.

#### Artefaktvertrauen

`artifact_confidence_f(s)` liegt in `[0,1]` mit `1 = sauber` und
`0 = Artefakt/Veto`. Pro Frame und Kanal wird mit demselben Dropletkern
aggregiert:

```text
a_f,c(q) = sum_s K(q,s) * artifact_confidence_f(s) / B_f,c(q)
a_p10,c(q) = weighted_p10_f(a_f,c(q), weight=B_f,c(q))
A_artifact,c(q) = smoothstep(0.25, 0.75, a_p10,c(q))
A_artifact(q) = min_c A_artifact,c(q)
```

Nichtfinite, fehlende oder lokal unzureichend gestützte Artefaktdaten sind kein
volles Vertrauen; die neue Source-Map markiert unzureichenden Support daher
nicht wie der alte Diagnosepfad mit `1`, sondern nichtanwendbar. Sind weniger
als acht gültige Framebeiträge für die robuste Statistik vorhanden, gilt
`A_artifact=0`. Einzelne vetoisierte Samples erhalten bereits Q-Gewicht null;
das gewichtete untere Perzentil verhindert zusätzlich, dass ein relevanter
Artefakttail von vielen guten Frames verdeckt wird. `A_artifact` steuert nur
Alpha und ist kein weiteres Profilgewicht.

#### Registrierungsvertrauen

`A_registration` verwendet geometrische Supportgewichte, nicht die bereits in
`G_eff` enthaltenen Profilgewichte:

```text
direct_fraction_c(q) =
    sum_f B_f,c(q) * is_direct_registration_f / sum_f B_f,c(q)

residual_p20_c(q) =
    weighted_p20_f(registration_residual_factor_f, weight=B_f,c(q))

A_registration,c(q) = min(
    smoothstep(0.50, 0.85, direct_fraction_c(q)),
    smoothstep(0.55, 0.90, residual_p20_c(q)))
A_registration(q) = min_c A_registration,c(q)
```

`is_direct_registration` ist nur für direkt gemessene oder direkt astrometrisch
gerettete Frames eins. Modellierte, interpolierte und nearest-copy Frames bleiben
null. Sie können zum Grundsignal beitragen, aber kein maximales Fine-Band-
Vertrauen erzeugen. Diese Verwendung der bereits persistierten
Residualinformation begrenzt ausschließlich Alpha und multipliziert den
Residualfaktor nicht ein zweites Mal in ein Profilgewicht.

### 14.5 Lokaler Energieguard

Der Guard arbeitet pro Band auf der festen Arbeitsluminanz und besitzt im ersten
Release **keine Sternkonzentrationsausnahme**:

```text
D_luma,j = 0.25 * D_R,j + 0.50 * D_G,j + 0.25 * D_B,j
window_radius_j = max(3, 2^(j+1)) interne Pixel
scale_raw = max(MAD_window(D_raw,luma,j), background_band_floor_j)
energy_ratio(alpha) = MAD_window(D_mixed,luma,j(alpha)) / scale_raw
```

Für MONO ist `D_luma,j = D_L,j`. `background_band_floor_j` wird robust aus dem
Uniform-Band auf der `analysis_common_mask` bestimmt. Mindestens 25 gültige
Fensterpixel sind erforderlich; andernfalls gilt für dieses Band `alpha=0`.
Verbindliche Energiegrenze ist zunächst

```text
energy_ratio <= 1.30
```

Wird sie mit `alpha_pre` überschritten, bestimmt eine deterministische
Bisektion mit sechs Iterationen das größte
`alpha_guarded in [0, alpha_pre]`, das die Grenze erfüllt. Pixelwerte und Raw-
Bänder werden nie hart geclippt oder verändert. Die frühere Idee, bei höherer
Sternkonzentration mehr Energie zuzulassen, entfällt: Gerade an Sternkernen wären
Ringing und chromatische Säume am gefährlichsten. Fensterstatistik,
Hintergrundfloor, Verhältnis, Iterationszahl und resultierende Alpha-Reduktion
werden im Artefakt protokolliert. `1.30` ist der verbindliche M6-Ausgangswert;
eine spätere Änderung erfordert neue Fixtures und eine
`multiband_config_hash`-Version.

### 14.6 RGB-Semantik

- `alpha_j` wird aus den konservativen Minima der kanalbezogenen Faktoren
  `A_neff`, `A_coverage` und `A_separation` sowie den gemeinsamen Faktoren
  `A_artifact` und `A_registration` berechnet. Luminanz/Grün liefert die
  Qualitätsinformation, darf aber R/B-Minima nicht ersetzen.
- Dasselbe `alpha_j` wird auf R-, G- und B-Banddifferenzen angewendet.
- Jeder Kanal behält seinen eigenen Support und seine eigene Raw-Bandbasis.
- Bei fehlendem Support eines Farbkanals gilt dort `alpha=0`.
- Es gibt keinen multiplikativen Luminanzratio-Transfer.
- Chroma und Hintergrundrest stammen in den niedrigsten Frequenzen immer aus
  dem Uniform-Control.

Damit werden Farbsäume durch unterschiedliche per-channel Detailmasken
vermieden.

### 14.7 Rand- und Seam-Behandlung

- normalisierte maskierte À-trous-Faltung;
- nach dem Energieguard wird Alpha mit demselben separierbaren B3-Kern nur
  innerhalb der jeweiligen 4-zusammenhängenden Supportkomponente geglättet:

  ```text
  alpha_blur = convolve(alpha_guarded * support, B3)
               / convolve(support, B3)
  alpha_final = min(alpha_guarded, alpha_blur)
  ```

  Die `min`-Kappe ist verbindlich: Glättung darf lokale Evidenz nur reduzieren,
  nie Alpha in ein unsicheres oder vetoisiertes Pixel hineinheben;
- keine Faltung über ungültige Canvasbereiche oder getrennte Supportinseln;
- `alpha_guarded=0` bleibt exakt null und Alpha fällt vor Supportkanten weich
  auf null;
- Bandrekonstruktion prüft, dass die Summe aller Bänder plus Rest das
  Eingangsprofil innerhalb numerischer Toleranz reproduziert;
- separate Seam-Diagnostik pro Band und für die finale Summe;
- die À-trous-Fusion arbeitet streifenweise aus den transaktionalen
  Profilstores. Der Fusionshalo ist mindestens der kumulative B3-Spline-Radius
  `2 * (2^levels - 1)` plus Alpha-Glättungsradius und wird nur im Kernbereich
  committed;
- Vollbild-U/R/F/M-Matrizen dürfen im Produktionspfad nicht gleichzeitig
  resident sein. Ein kleiner In-Memory-Referenzpfad ist ausschließlich für
  Tests zulässig; Streifen- und Referenzpfad müssen innerhalb dokumentierter
  Toleranz übereinstimmen.

---

## 15. Kandidaten-, Baseline- und Gate-Logik

### 15.1 Unveränderliche Stufen

```text
drizzle_uniform
drizzle_raw
drizzle_multiband
```

Die Bezeichner folgen dem Namensvertrag aus 1.2 (kein `aqmh` in neuen
Artefakt-/Reportnamen) und werden so in `selected_candidate` (16.3), Reports
und GUI verwendet. Nach Erzeugung darf `drizzle_raw` nicht durch nachfolgende
Neutralisierung, Schärfung oder Strukturmischung verändert werden.

### 15.2 Gemeinsame Sternpopulation

Die vorhandene paarweise Validation wird zu einem festen Dreifachvertrag
erweitert:

```cpp
struct ValidationSampleSet {
  std::vector<ValidationStarSample> stars;
  int width = 0;
  int height = 0;
};

ValidationSampleSet prepare_validation_samples(
    const Matrix2Df& uniform_control,
    const std::vector<uint8_t>& validation_mask);
```

Uniform, Raw und Multiband werden an exakt diesen Positionen gemessen. Für
`candidate_vs_raw` werden nicht erneut Sterne auf Raw-Forward-Drizzle
detektiert.

### 15.3 Auswahl

1. Uniform-Referenz und fester Sternsatz werden vorbereitet.
2. Raw-Forward-Drizzle wird gegen Uniform gemessen. Sind Support, Numerik,
   Hintergrund-RMS oder Seam-Metrik nicht anwendbar oder verletzt, wird Uniform
   gewählt. Nichtanwendbare Sternmetriken allein verwerfen Raw nicht.
3. Multiband wird gegen Uniform und Raw an demselben Sternsatz gemessen.
4. Multiband wird nur gewählt, wenn gleichzeitig gilt:

   ```text
   median_FWHM_multiband <= 0.95 * median_FWHM_raw
   p90_FWHM_multiband    <= 1.00 * p90_FWHM_raw
   tail_multiband        <= 1.10 * tail_raw
   elongation_multiband  <= 1.08 * elongation_raw
   background_RMS        <= 1.05 * background_RMS_uniform
   seam_score            <= 1.05 * seam_score_uniform
   ```

   Support und numerische Diagnostik müssen ebenfalls vollständig gültig sein.
5. FWHM ist ab 20, p90-FWHM, Tail und Elongation sind ab 30 unsaturierten,
   isolierten, erfolgreich gematchten Sternen anwendbar. Zusätzlich darf die
   relative Breite des gebootstrappten 95-%-Konfidenzintervalls des
   FWHM-Medians höchstens 10 % betragen. Rand-, Sättigungs- oder Fitfehler
   reduzieren die Stichprobenzahl.
6. Jede Metrik besitzt explizit `applicable`, `value`, Stichprobenzahl,
   Konfidenzintervall und `reason_if_not_applicable`. Eine nichtanwendbare
   Pflicht-Sicherheitsmetrik macht den jeweiligen Kandidaten ungültig. Eine
   nichtanwendbare Sternmetrik erfüllt nie positive Multiband-Evidenz; in diesem
   Fall bleibt Raw. Damit werden kleine Stichproben nicht als impliziter Pass
   behandelt.
7. Scheitert Multiband, wird Raw gewählt. Scheitert Raw an einem anwendbaren oder
   verpflichtend nichtanwendbaren Sicherheitsgate, wird Uniform gewählt.

Für die **M9-Releasepromotion** ist N/A bei einem in der Datensatzzeile
geforderten Kriterium ein nicht bestandenes Releasegate. Die kontrollierte
Matrix muss sämtliche Promotionskriterien auf den dafür vorgesehenen
Datensätzen anwendbar machen.

**Erwartete Auswahlverteilung.** Das Gate `background_RMS <= 1.05 *
background_RMS_uniform` ist streng. Zur Einordnung: mit dem aktuellen
PREWARP-AQMH überschreiten reale M31-/M42-Läufe die 5-%-Grenze um mehr als das
Zehnfache (`background_rms_regression ≈ 0.56`), weshalb dort der Uniform-Fallback
greift. Der Forward-Drizzle führt bei `pixfrac < 1` und 2x zusätzlich
korreliertes Rauschen ein (12.4). Es ist deshalb ein plausibles und zulässiges
Ergebnis, dass Raw und insbesondere Multiband auf einem Teil der Datensätze
**nicht** promoted werden und der Pfad dort das Uniform-Control liefert. Die
M9-Diagnostik protokolliert je Datensatz den gewählten Kandidaten und den
Gate-Grund; eine niedrige Multiband-Trefferquote ist kein Fehlschlag der
Methode, sondern Teil des erwarteten Verhaltens. Ein Fehlschlag liegt erst vor,
wenn der neue Pfad das jeweilige Uniform-Control gegenüber dem alten Pfad
regressiv verschlechtert.

### 15.4 Entferntes Postprocessing

Folgende bisherigen Kandidaten werden aus dem aktiven Code entfernt und nicht
ausgeführt:

- `star_core_sharpening`;
- `structure_masked_detail`;
- Low-Frequency-Neutralisierung als Detailkandidat.

Nur allgemein verwendbare Validierungshelfer bleiben erhalten. Kandidaten-
spezifischer Code, Konfiguration, Artefakte, GUI-Felder und Tests werden
gelöscht. Eine spätere Wiedereinführung wäre eine neue Methodikänderung mit
eigenem Nachweis und ist nicht Teil dieses Plans.

---

## 16. Outputs, Artefakte und Diagnostik

### 16.1 Immer persistierte Outputs

OSC:

```text
outputs/forward_drizzle_raw_L.fit
outputs/forward_drizzle_raw_R.fit
outputs/forward_drizzle_raw_G.fit
outputs/forward_drizzle_raw_B.fit
outputs/reconstructed_L.fit
outputs/reconstructed_R.fit
outputs/reconstructed_G.fit
outputs/reconstructed_B.fit
outputs/stacked_rgb.fits           # STACKING-Pass-through von reconstructed_R/G/B
```

MONO:

```text
outputs/forward_drizzle_raw_L.fit
outputs/reconstructed_L.fit
outputs/stacked.fits               # bestehender MONO-Downstreamname
```

MONO erzeugt keine fingierten R/G/B-Dateien. `stacked.fits` bleibt der
bestehende kanonische Mono-Downstreamname.

`forward_drizzle_raw_*` ist die unveränderliche Raw-Forward-Drizzle-Baseline.
`reconstructed_*` enthält den ausgewählten Kandidaten. `stacked_rgb.fits`
bleibt für OSC der bestehende Downstream-Eingang (Astrometrie, BGE, PCC, HMS)
und wird von STACKING unverändert aus `reconstructed_R/G/B` gebildet (17.2);
alle Outputs liegen in `output_scale`-Geometrie.

Alle Dateien werden zunächst unter einem phasenlokalen temporären Namen
geschrieben, vollständig geschlossen, gehasht und erst danach atomar
umbenannt. `forward_drizzle.json` und anschließend `reconstruction.json` werden
als Commitmarker zuletzt geschrieben. Ein abgebrochener Lauf hinterlässt keine
als gültig interpretierbaren Teiloutputs; Resume validiert Größe und Checksumme.

### 16.2 Diagnostische Outputs

Bei `reconstruction.diagnostics.level: full` zusätzlich:

```text
outputs/forward_drizzle_uniform_L.fit
outputs/forward_drizzle_uniform_R.fit
outputs/forward_drizzle_uniform_G.fit
outputs/forward_drizzle_uniform_B.fit
outputs/forward_drizzle_multiband_L.fit
outputs/forward_drizzle_multiband_R.fit
outputs/forward_drizzle_multiband_G.fit
outputs/forward_drizzle_multiband_B.fit
outputs/multiband_alpha_<j>.fit        # nur für Bänder mit profile != R (Default: j = 1, 2)
outputs/forward_drizzle_neff_uniform_R.fit   # OSC; analog G/B
outputs/forward_drizzle_neff_raw_R.fit       # OSC; analog G/B
outputs/forward_drizzle_neff_uniform_L.fit   # MONO
outputs/forward_drizzle_neff_raw_L.fit       # MONO
outputs/forward_drizzle_coverage.fit
outputs/forward_drizzle_channel_support.fit  # R/G/B- oder L-Support
```

Bei `summary` werden keine zusätzlichen öffentlichen Kontroll-FITS geschrieben;
persistiert bleiben Raw, ausgewähltes Ergebnis, Supportmasken, Validation,
Hashes, Checksummen und kompakte Heatmaps im JSON. Uniform-/Multiband-RGB-
Kontrollen und vollständige Alpha-/`n_eff`-Ebenen werden ausschließlich bei
`full` als öffentliche FITS persistiert.

Die internen transaktionalen U/R/F/M-Stores sind davon unabhängig. Sie werden
bei `keep_profile_cache_after_run=false` nach erfolgreichem Phasencommit gelöscht
und bei `true` als gehashter Rekonstruktionscache behalten. Downstream-Resume
hängt nie von ihnen ab; ein Resume mitten in der Mehrbandfusion existiert nicht.

### 16.3 Neues Artefakt

```text
artifacts/forward_drizzle.json
```

Hauptfelder:

```json
{
  "schema_version": 1,
  "pipeline_method": "cfa_forward_drizzle_multiband",
  "pipeline_contract_version": 1,
  "sampling_plan_hash": "...",
  "coverage_geometry_hash": "...",
  "source_identity_hash": "...",
  "normalized_cache_hash": "...",
  "source_quality_config_hash": "...",
  "source_quality_cache_hash": "...",
  "reconstruction_config_hash": "...",
  "multiband_config_hash": "...",
  "luma_definition": "0.25R+0.50G+0.25B",
  "geometry": {
    "source_width": 3840,
    "source_height": 2160,
    "canvas_width_native": 3926,
    "canvas_height_native": 2312,
    "internal_scale": 2,
    "output_scale": 1,
    "kernel": "square",
    "pixfrac": 0.8
  },
  "clipping": {},
  "coverage": {},
  "profiles": {},
  "multiband": {},
  "validation": {},
  "selected_candidate": "drizzle_multiband",
  "fallback_reason": null,
  "acceleration": {},
  "timing_seconds": {},
  "outputs": [{"path": "...", "size": 0, "sha256": "..."}],
  "commit_complete": true
}
```

### 16.4 Pflichtdiagnostik

- Quell- und Zielabmessungen;
- Sampling-Plan-Hash;
- gültige/ausgeschlossene Frames mit Gründen;
- lokale Warp-Inversionsfehler pro Frame;
- Supportanteil, p10/p50/p90-`n_eff`, größte interne Lochfläche und
  Ditherdiagnostik pro Kanal;
- angewandte Coveragegrenzen und exakter Gate-Grund;
- akzeptierte Samplezahl p10/p50/p90;
- `n_eff` p10/p50/p90 pro Profil und aktivem Kanal;
- Zero-Veto- und Missing-Map-Zähler;
- Clippinganteil pro Kanal;
- Q-Separation, `A_artifact`, `A_registration` und deren Teilstatistiken pro Band;
- Alpha vor Guard, nach Energieguard und final nach B3-Kappe einschließlich
  p10/p50/p90 und Anteil `alpha=0`/`alpha=1`;
- Bandenergie Raw/Detail/Final, lokale Energieverhältnisse und Bisektionszähler;
- CPU-/CUDA-Backend, Chunkgrößen, Retries und Fallbackgründe;
- RSS-Start/Peak/Wachstum/Budget, Temporärdiskschätzung/-peak/-reserve und
  normierter Durchsatz;
- Profilcache-/Source-Cache-Retention sowie gelöschte und committed Stores;
- vollständige Uniform-/Raw-/Multiband-Validation;
- ausgewählter Kandidat und exakter Gate-Grund.

`artifacts/reconstruction.json` ist das übergeordnete Rekonstruktionsartefakt
und verweist auf `artifacts/forward_drizzle.json`. Das bisherige
`aqmh_reconstruction.json` ist ausschließlich ein read-only Legacyartefakt.

---

## 17. Runner-Integration und Methoden-Cutover

### 17.1 Linearer Pfad in `runner_pipeline.cpp`

Nach den unveränderten Vorphasen (Scan, Kalibration, Channel-Split,
Normalisierung, globale Metriken):

```cpp
run_phase_registration(...);          // erzeugt RegistrationSamplingPlan
run_phase_sampling_geometry(...);     // geometrische Coverage + coverage_gate
run_phase_common_overlap(...);
run_phase_source_quality_maps(...);   // CFA-Green-Proxy, Composite, Scale-Maps
run_phase_global_quality(...);        // G_quality(f) -> QualityFrameWeightPlan
run_phase_forward_drizzle(...);       // Profile, Mehrband, Validation, Auswahl
run_phase_reconstruction_diagnostics(...);
```

Der Gewichtsplan entsteht **nach** den Source-Q-Maps, weil `G_quality(f)`
denselben Proxy konsumiert (11.9). Es gibt keinen Methodenbranch. Runner, CLI, Backend und GUI können keine andere
Rekonstruktionsmethode auswählen. Aufrufe der alten AQMH-Map-/Rekonstruktions-
phasen und sämtliche Zugriffe auf `prewarped_frames` werden entfernt.

### 17.2 STACKING

STACKING ist ein Pass-through für das bereits rekonstruierte lineare Ergebnis.
`weight_sum` ist ausschließlich die diagnostische Forward-Drizzle-Gewichtssumme
und wird nicht als historische OLA-Gewichtssumme interpretiert.

### 17.3 Kein nachgelagertes DEBAYER

Forward-Drizzle erzeugt bei OSC direkt R/G/B. Die nachgelagerte DEBAYER-Phase
und ihr Pass-through-Ereignis werden aus der aktiven Pipeline entfernt. Ein
erneutes Debayern der rekonstruierten Luminanz- oder Farbebenen ist verboten.
Historische DEBAYER-Ereignisse bleiben ausschließlich im read-only
History-Parser bekannt.

### 17.4 Output-Scaling

Die vorhandenen Normalisierungs-Outputskalen `scale_r/g/b`, Hintergründe und
Pedestal werden nach Auswahl des Kandidaten exakt einmal angewendet. Uniform,
Raw und Multiband werden für Runtime-Gates im selben normalisierten linearen
Arbeitsraum und nach demselben `internal_scale -> output_scale`-Operator
gemessen. M9-Vergleiche gegen den alten Pfad verwenden zusätzlich dieselbe
finale Outputskalierung, denselben Crop und dieselben physikalischen
Maßeinheiten; insbesondere wird absolutes Hintergrund-RMS nicht zwischen
unterschiedlich skalierten ADU-Räumen verglichen.

### 17.5 Backend-Anpassungen (`web_backend_cpp`)

Das Backend parst und validiert Konfiguration nicht selbst, sondern proxyt an
`tile_compile_cli` (`/api/config/schema` → `get-schema`, `/defaults` →
`dump-default-config`, `/validate` → `validate-config`). Schema- und
Validierungsänderungen aus M0 wirken dadurch ohne zusätzliche Backendarbeit.
Der Runner — nicht das Backend — schreibt `pipeline_contract_version` beim
Run-Start in `run_provenance.json` und die Run-Metadaten; das Backend liest sie
nur für Anzeige und Resume-Gating.

Verbindlich in M0/M8 zu ändern:

| Datei | Änderung |
|---|---|
| `src/services/pi/pi_context_v2.cpp` | Ableitung `pipeline.method` aus `base_config["method"]`/`aqmh.enabled` entfernen bzw. auf `reconstruction`/`pipeline_contract_version` umstellen; Fact-Reader für `aqmh.cherry_pick_*`, `aqmh_reconstruction.json` und `AQMH_RECONSTRUCTION`-Phase-Events in den read-only History-Parser verschieben |
| `src/services/run_inspector.cpp` | `normalizePhaseEvent(event, method)` mit `if (method == "aqmh")`-Zweig auf einen einzigen Pfad reduzieren; historische Phasennamen nur im History-Parser |
| `src/services/preprocessing_service.cpp` | Config-Pfadgruppen der Schritt-3-Oberfläche (`stacking.*`, `tile.*`, `rejection.*`, …) an die neue Konfigurationsfläche und die Entscheidungen aus 6.5 angleichen |
| `tests/fixtures/fake_tile_compile_runner.cpp`, `fake_tile_compile_cli.cpp` | den neuen Pipelinevertrag und `pipeline_contract_version` emittieren |
| `tests/test_backend_contract.cpp`, `test_run_status_resume_progress.cpp`, `test_runs_queue.cpp`, `test_run_start_naming.cpp` | Erwartungen auf Single-Method-Vertrag umstellen; Legacy-Run zeigt read-only, Resume-Button deaktiviert |

Das Backend erhält keinen Methoden- oder Engine-Parameter im Run-Create-Contract.
Ein Run-Create mit Legacy-Methodenfeld wird mit demselben `UNKNOWN_LEGACY_KEY`
abgelehnt wie in CLI und Parser.

---

## 18. Resume-Implementierung und Legacy-Ablehnung

### 18.1 Pipelinevertrags-Erkennung

`runner_resume.cpp` liest zuerst Run-Metadaten und
`pipeline_contract_version`. Fehlt die Version oder identifiziert sie Classic
beziehungsweise PREWARP-AQMH, wird der Resumeversuch fail-closed beendet. Die
Prüfung erfolgt vor dem Öffnen eines Artefakts im Schreibmodus.

Für den einzigen aktiven Vertrag erforderlich (für ein Resume ab
`FORWARD_DRIZZLE`):

- `cache/normalized_frames` mit gültigen Metadaten und Inhaltschecksummen;
- `artifacts/registration_sampling.json`;
- `artifacts/sampling_geometry.json` mit bestandenem `coverage_gate`;
- `artifacts/common_overlap.json` mit Verweis auf denselben
  `coverage_geometry_hash`;
- `cache/source_quality_maps` mit gültigem `source_quality_cache_hash`;
- `artifacts/global_quality.json` (`QualityFrameWeightPlan`) mit demselben
  `source_identity_hash`, `sampling_plan_hash` und
  `source_quality_config_hash`;
- `artifacts/normalization.json`;
- `outputs/common_overlap_mask.fits`;
- `outputs/canvas_mask.fits`;
- identische geordnete `frame_id`-/`source_index`-Folgen und kompatible
  domänenspezifische Config-/Cache-Hashes (10.3/18.3).

Eine vorhandene `cache/prewarped_frames`-Struktur erfüllt keine Abhängigkeit des
neuen Vertrags und darf weder gelesen noch konvertiert werden.

### 18.2 Resume-Einstiege

- `SAMPLING_GEOMETRY`: Normalized-Cache-Metadaten und Sampling-Plan vorhanden;
  Bildwerte müssen für die reine Geometrie nicht geöffnet werden;
- `COMMON_OVERLAP`: `sampling_geometry.json` mit bestandenem Gate und passendem
  Coverage-Hash vorhanden;
- `SOURCE_QUALITY_MAPS`: direkter Neueinstieg in Source-Map-Berechnung nur, wenn
  normalisierte Frames, Sampling-Plan, `sampling_geometry.json` mit bestandenem
  `coverage_gate` und die festgelegte Common-Maskengeometrie vorhanden sind;
- `GLOBAL_QUALITY`: zusätzlich vollständiger Source-Q-Map-Cache;
- `FORWARD_DRIZZLE`: Source-Maps, Gewichtsplan und alle übrigen
  Abhängigkeiten (18.1) müssen vollständig sein; Sampling-, Source-Quality-,
  Reconstruction- und Multibandhash müssen zur aktiven Konfiguration passen.
  Mehrband, Validation und Auswahl sind Teil dieser Phase und keine eigenen
  Einstiege;
- `STACKING`: verwendet das persistierte Raw-/Selected-Artefakt; fehlt es,
  fällt der Einstieg nur dann auf `FORWARD_DRIZZLE` zurück, wenn dessen
  vollständiger Vertrag erfüllt ist.

Historische Phasen wie `PREWARP`, alte `AQMH_MAPS`-Artefakte oder Classic-
Stackingzustände werden in der Resume-UI als „historisch – nicht resumierbar“
markiert. Die UI darf für diese Zeilen keinen aktiven Resume-Button anbieten.

### 18.3 Hashinvalidierung

Die Hashdomänen sind getrennt und in jedem Artefakt namentlich gespeichert:

- `normalized_cache_hash`: kanonischer Hash des committed
  Normalized-Cache-Manifests einschließlich Frame-IDs, Cachedateigrößen und
  Inhaltschecksummen;
- `source_identity_hash`: geordnete Frame-IDs/Inhaltsidentitäten,
  Quellgeometrie/-orientierung, Farbmodus, Bayer-Pattern/CFA-Ursprung,
  Kalibration, Normalisierung und `normalized_cache_hash`;
- `sampling_plan_hash`: `source_identity_hash` plus native Registration,
  Canvas, lokales Modell, Modell-/Residualfaktoren, jedoch ohne
  interne/Ausgabeskala;
- `source_quality_config_hash`: Proxy-Version, Quality-Pyramide/-Gewichte,
  Storage-Divisor und Datentyp;
- `source_quality_cache_hash`: `source_identity_hash` plus
  `source_quality_config_hash` und kanonisches Manifest aller vollständig
  committed Streams mit Größen/Inhaltschecksummen;
- `coverage_geometry_hash`: `sampling_plan_hash` plus `internal_scale`, Kernel,
  `pixfrac`, Droplet-Subdivision und
  `common_overlap_required_fraction`; `output_scale` gehört nur dann hinein,
  wenn die persistierten Masken bereits auf Ausgabegeometrie reduziert sind;
- `reconstruction_config_hash`: Coverage-Hash plus Clipping-, Profil- und
  Gewichtsplanvertrag;
- `multiband_config_hash`: Profilhashes plus Levels, Alpha-, Energie-,
  Support-, Downsample- und Validationvertrag.

Daraus folgen die Invalidierungen:

- `internal_scale`, Kernel, `pixfrac` oder Droplet-Subdivision invalidieren
  Coverage, Gate, Masken, Rekonstruktion und Mehrband, aber nicht Source-Q-Maps
  oder den nativen Sampling-Plan;
- `output_scale` invalidiert Outputkonvertierung, WCS, Validation und alle
  publizierten Outputs; interne Profile dürfen nur wiederverwendet werden, wenn
  ihr separater Internal-Scale-Hash passt;
- Clippingparameter und `min_clip_contributors` invalidieren alle Profile und
  Mehrband;
- reine Mehrbandparameter invalidieren Fusion/Auswahl. U/R/F/M dürfen nur dann
  wiederverwendet werden, wenn vollständig committed Profilstores mit passenden
  Checksummen vorhanden sind; andernfalls wird `FORWARD_DRIZZLE` aus den
  Source-Caches wiederholt statt auf fehlende Zwischenstände zu vertrauen;
- `keep_profile_cache_after_run`, `delete_source_cache_after_run` und
  `diagnostics.level` verändern keine Pixelwerte und gehören nicht in
  Rechenhashes; sie steuern nur Persistenz und werden separat im Artefakt
  protokolliert;
- Änderungen nur an Gategrenzen erlauben eine Neuauswertung aus vollständig
  persistierten Coverage-Statistiken; ändern sie den Maskenbezugsbereich, wird
  Coverage neu berechnet;
- Quality-/Proxy-/Storageänderungen invalidieren Source-Q-Maps, Gewichtsplan,
  Profile und Mehrband;
- Registration oder Canvas invalidieren Sampling-Plan, Coverage, Masken,
  Profile und Mehrband, **nicht** jedoch Source-Q-Maps mit identischem
  `source_identity_hash`;
- Normalisierung, Kalibration, Framefolge, Sensororientierung, Bayer-Pattern oder
  CFA-Ursprung invalidieren sämtliche nachgelagerten Domänen.

---

## 19. CUDA-Implementierung

### 19.1 Reihenfolge

CUDA beginnt erst, wenn CPU-Referenz, Artefakte, Resume und Mehrbandsemantik
vollständig getestet sind.

Neue Dateien:

```text
tile_compile_cpp/include/tile_compile/reconstruction/forward_drizzle_cuda.hpp
tile_compile_cpp/src/reconstruction/forward_drizzle_cuda.cu
```

### 19.2 Kernelaufteilung

Vorgeschlagene Stufen pro Chunk:

1. Source-Regionen und Q-Map-Regionen hostseitig laden.
2. Source-Samples und Transformdaten H2D übertragen.
3. Frame-lokale Droplet-Akkumulation in getrennte Buffers.
4. Transposition in pixel-major Layout.
5. deterministische oder semantisch tolerierte Sortierung/Clipping pro Pixel;
6. gemeinsame Akzeptanzmaske;
7. parallele Profilakkumulation;
8. Ergebnis und Diagnostik D2H;
9. À-trous-Fusion im CPU-Referenzpfad.

M7 implementiert CUDA für Droplet, Clipping und Profilakkumulation, nicht für
À-trous. Eine spätere CUDA-À-trous-Erweiterung ist nur zulässig, wenn Profiling
zeigt, dass die CPU-Fusion mindestens 20 % der gesamten `FORWARD_DRIZZLE`-Zeit
beansprucht und die GPU-Variante inklusive Transfers/Store-I/O die komplette
Phase um mindestens 15 % beschleunigt. Sie benötigt dann dieselben Paritäts- und
Phasen-Neustarttests; ein schnellerer Mikro-Kernel allein genügt nicht.

### 19.3 Atomics

Globale Float-Atomics über alle Frames sind nicht die Referenzsemantik und
führen zu nichtdeterministischer Summenreihenfolge. Zulässig sind:

- frame-lokale Atomics innerhalb eines isolierten Framebuffers, gefolgt von
  fester profilweiser Reduktion;
- deterministische Sort-/Segmented-Reduction-Verfahren;
- Abweichungen nur innerhalb explizit getesteter Toleranz.

### 19.4 Speicher und Fallback

- Auto-Chunking aus tatsächlich freiem Device-Speicher;
- reservierte Sicherheitsmarge für Treiber/OpenCV;
- begrenzte Allokationsretries mit halbierter Chunkhöhe;
- vollständiger Fehlergrund im Artefakt;
- Chunks schreiben ausschließlich in phasenlokale temporäre Stores;
- schlägt CUDA nach Ausschöpfung der Retries in einem beliebigen Chunk fehl,
  werden **alle** temporären CUDA-Stores verworfen und die komplette
  `FORWARD_DRIZZLE`-Phase auf CPU neu gestartet;
- erst ein vollständig berechneter, validierter und gehashter Backendlauf wird
  atomar committed. Gemischte CPU-/CUDA-Bilder und halb akkumulierte Pixel sind
  verboten.

### 19.5 Paritätstests

- Identitätswarp;
- Subpixeltranslation;
- Rotation;
- alle Bayer-Pattern;
- asymmetrisches Clipping;
- Zero-Veto;
- fehlende Map;
- kleine und große Chunkhöhen;
- Rand- und Maskenfälle;
- Uniform, Raw und jedes Detailprofil;
- À-trous-Bänder und Alpha.

---

## 20. Testspezifikation

### 20.1 Registration-Sampling-Tests

Neue Datei:

```text
tests/test_registration_sampling_plan.cpp
```

Testfälle:

- affine Identität round-trip;
- Translation round-trip;
- Rotation/Skalierung round-trip;
- singuläre Matrix wird abgelehnt;
- Canvasoffset korrekt komponiert;
- 2x-Skalierung verändert keine native Warpsemantik;
- lokales Modell konvergiert für gültigen Jacobian;
- Modell-Koordinatenskalierung und Canvasoffset entsprechen dem bestehenden
  Smooth-Local-Remap auf einem Kontrollgitter;
- lokales Modell bricht deterministisch bei ungültigem Feld ab;
- `source_to_canvas` wird bei lokalem Modell nicht als fertige affine Inverse
  missbraucht;
- Serialisierung/Deserialisierung ist verlustarm;
- Frame-Reihenfolge oder `frame_id`-Änderung ändert den Plan-Hash;
- CFA-Ursprung und Orientierungsänderung ändern den Plan-Hash;
- Plan-Hash ändert sich bei semantischen Änderungen;
- Diagnostikänderungen ohne Semantik ändern den Plan-Hash nicht;
- `internal_scale`/`output_scale` ändern den Plan-Hash nicht (7.4).

### 20.2 Drizzle-Kerntests

Neue Datei:

```text
tests/test_forward_drizzle.cpp
```

Pflichtfälle:

1. konstantes Mono-/CFA-Bild bleibt konstant;
2. Einzelimpuls bei Identitätswarp hat erwarteten Droplet-Support;
3. physischer Aperturflux (Oberflächenhelligkeit mal WCS-Pixelfläche) bleibt
   bei mehreren Subpixelphasen und bei 1x/2x erhalten;
4. Translation um 0,5 Pixel verteilt korrekt;
5. Rotation erzeugt keine nichtfiniten Innenpixel;
6. jeder Bayer-Typ ordnet R/G/B korrekt zu;
7. ungerader Source-Crop aktualisiert den CFA-Ursprung korrekt; Canvasoffset,
   Dither und Rotation verändern die Source-CFA-Farbe nicht;
8. G1/G2-Coverage ist plausibel und symmetrisch;
9. mehrere Samples desselben Frames zählen als ein Frame-Beitrag;
10. Ausreißer wird in allen Profilen identisch abgelehnt;
11. fehlende Q-Map beeinflusst Uniform nicht, vetoisiert aber Qualitätsprofile;
12. Q=0 bleibt nach Cache-Decode ein Veto;
13. `min_fraction` und `min_n_eff` arbeiten auf der korrekten Population;
14. Chunkhöhen 1, Auto und Vollbild liefern äquivalente Ergebnisse, auch bei
    Rotation um 45° und Skalierung 1,2 (Halo-Vertrag 11.11);
15. Threadzahlen 1 und N liefern äquivalente Ergebnisse;
16. Outputmasken entsprechen der Coverage;
17. negative normalisierte Hintergrundwerte bleiben gültig;
18. NaN/Inf-Samples werden ausgeschlossen und gezählt;
19. pixfrac-Grenzwerte werden validiert;
20. identische Gewichte ergeben identische Uniform-/Raw-Ausgabe;
21. Flächensumme `sum_q K` folgt der Jacobi-Form aus 11.6 bei Skalierungswarp;
22. Pixel unter `min_fraction`/`min_n_eff` sind in allen Profilen und im
    Kanalsupport identisch als unbelegt markiert;
23. `coverage_gate`: ganzzahlige Offsets ohne Subpixeldiversität werden durch
    direkten Kanalsupport/`n_eff` abgelehnt; Werte nahe 0/2 werden in der
    Ditherdiagnostik zirkulär als benachbart behandelt, lösen allein aber weder
    Pass noch Fail aus; geometrisches Kanal-`n_eff` entspricht dem Uniform-
    `n_eff` der Rekonstruktion innerhalb FP-Toleranz;
24. lokaler Warp mit räumlich variablem Jacobian erfüllt Subdivisionstoleranz,
    Flächenintegral und Zentroidgrenze; Überschreitung wird diagnostiziert;
25. `internal_scale=2, output_scale=1` erfordert 4/4 gültige Subpixel,
    verwendet den festen Flächenmittelwert sowie das minimale Subpixel-`n_eff`
    und dieselbe Geometrie für U/R/Multiband;
26. `n_eff_R/G/B` bleiben getrennt und dünne R/B-Coverage wird nicht durch G
    verdeckt;
27. MONO erzeugt nur L-Stores und besteht dieselben Flux-/Clippingverträge;
28. ein abgebrochener Chunk hinterlässt keinen gültigen Output oder Commitmarker.

### 20.3 Q-Map-Tests

Erweiterung von:

```text
tests/test_source_quality_map.cpp
tests/test_source_quality_map_cache.cpp
```

- `proxy_version=1` bildet `G_quad=0.5*(G1+G2)` korrekt und erzeugt kein
  Bayer-Checkerboard;
- Green-Highpass/MAD verwechselt farbige R/B-Sterne nicht mit Rauschen;
- MONO verwendet direkt L ohne CFA-Interpolation;
- skalenspezifische Maps haben korrekte Geometrie;
- Composite entspricht dem geometrischen Mittel der gültigen Skalen;
- Scale-Sink hält nicht alle Maps gleichzeitig resident;
- Source-Region-Read entspricht Full-Map-Ausschnitt;
- Cache-Metadaten unterscheiden Source- und Canvas-Koordinaten;
- Hashinvalidierung bei Pyramid-/Proxy-Änderung;
- exakte Zero-Veto-Erhaltung bei `uint16` und konservative getrennte
  Veto-Übertragung auf Source-Geometrie.

### 20.4 Mehrbandtests

Neue Datei:

```text
tests/test_multiband_reconstruction.cpp
```

- À-trous-Rekonstruktion reproduziert das Eingangssignal;
- konstantes Bild hat Null-Detailbänder;
- linearer Gradient bleibt im Grobrest;
- identische Profile (`U = R = F = M`) ergeben exakt Raw-Ausgabe;
- `alpha=0` ergibt `R - C_R,L + C_U,L` (Raw-Detail über Uniform-Grobrest,
  14.3); bei `U = R` exakt Raw;
- Bandzuordnung für `levels = 1, 2, 3, 4` entspricht der Tabelle in 14.3;
- fehlende Q-Separation ergibt `alpha=0`;
- unzureichendes `n_eff` ergibt `alpha=0`;
- weighted-p10-`A_artifact` folgt den 0,25/0,75-Grenzen; fehlende, lokal
  unzureichend gestützte oder weniger als acht gültige Artefaktbeiträge ergeben
  `alpha=0` und nie implizites Vertrauen 1;
- `A_registration` folgt direct-fraction/residual-p20 und wendet keinen
  Registrierungsfaktor als Profilgewicht doppelt an;
- schärferer synthetischer Frame liefert nur bei ausreichender Sicherheit Fine-
  Detail;
- reines Rauschen wird nicht als Fine-Detail promoted;
- Maskenkante erzeugt keinen Seam-Sprung und level-spezifischer Support wird
  korrekt propagiert;
- RGB-Kanäle verwenden dieselbe Alpha-Geometrie;
- gemeinsames Alpha verwendet das Minimum aus kanalbezogenem `n_eff`, Coverage
  und Separation;
- neutraler Stern bleibt neutral;
- fehlender oder dünner R-/B-Support produziert keine Farbsäume und setzt Alpha
  konservativ auf null;
- Energieguard hält `energy_ratio<=1.30`, findet Alpha deterministisch in sechs
  Bisektionsschritten und besitzt keine Sternkonzentrationsausnahme;
- B3-Glättung erfüllt `alpha_final<=alpha_guarded`; Veto-Nullen und getrennte
  Supportinseln bleiben unverändert;
- streifenweise Fusion mit minimaler und größerer Halobreite entspricht dem
  kleinen Vollbild-Referenzpfad und hält das konfigurierte Peak-RSS-Budget ein.

### 20.5 Validation-Tests

Erweiterung von:

```text
tests/test_reconstruction_validation.cpp
```

- Uniform, Raw und Kandidat verwenden denselben Sternsatz;
- keine unabhängige Kandidatendetektion;
- Multiband-Fail wählt Raw;
- Raw-Fail wählt Uniform;
- nichtanwendbare Pflicht-Sicherheitsmetrik macht den Kandidaten ungültig;
  nichtanwendbare Sternmetriken blockieren Multiband-Promotion, bei M9 gilt ein
  gefordertes N/A als nicht bestanden;
- FWHM benötigt 20, p90/Tail/Elongation benötigen 30 gültige Sterne und der
  FWHM-Median eine relative 95-%-Bootstrap-CI-Breite von höchstens 10 %;
- Star-Tail-/Elongationsgates arbeiten an identischen Positionen;
- ausgewählter Kandidat und Fallbackgrund werden korrekt serialisiert.

### 20.6 Resume-Tests

- gültiger Single-Method-Cache resumiert `FORWARD_DRIZZLE`;
- fehlender Sampling-Plan oder fehlendes `sampling_geometry.json` wird abgelehnt;
- falscher Plan-/Coverage-Hash wird abgelehnt;
- vertauschte Frame-IDs, beschädigte Cachechecksumme, falsches Bayer-Pattern oder
  falscher CFA-Ursprung werden abgelehnt;
- geänderte Drizzle-Parameter invalidieren nur Rekonstruktion;
- geänderte Q-Parameter invalidieren Source-Maps;
- `keep_profile_cache_after_run=false` löscht interne Profile erst nach Commit,
  ohne Downstream-Resume zu beeinträchtigen; `true` validiert alle Storehashes;
- `delete_source_cache_after_run` bleibt standardmäßig `false`; explizite
  Löschung wird erst nach Gesamtcommit ausgeführt und korrekt im Resume gemeldet;
- Classic- und PREWARP-AQMH-Runs werden vor jeder Mutation mit stabilem
  Legacy-Fehlercode abgelehnt;
- historische Outputs und Reports bleiben read-only darstellbar;
- der aktive Runner liest niemals `prewarped_frames` als Ersatz;
- fehlendes Raw-Artefakt bei STACKING führt nur bei vollständigem
  Rekonstruktionsvertrag zurück zu `FORWARD_DRIZZLE`.

### 20.7 Downstream-Tests

- 2x-WCS mit bekannten Sternkoordinaten;
- Crop und Canvasoffset verschieben CRPIX mit der Vorzeichenkonvention aus 12.2
  korrekt;
- Masken-/RGB- beziehungsweise MONO-Dimensionen stimmen überein;
- STACKING verändert Forward-Drizzle-Ergebnis nicht;
- aktive OSC-Phasenfolge enthält kein nachgelagertes DEBAYER;
- BGE/PCC/HMS akzeptieren 2x-Geometrie;
- PCC-Autoaperturen verwenden aktuelle Ausgabepixel;
- Reports deklarieren `output_scale` und vergleichen keine FWHM unterschiedlicher
  Pixelmaßstäbe ohne Umrechnung.

### 20.8 Ressourcen- und Backendtests

- RSS-Wachstum bleibt unter `budget * 1.05 + 256 MiB`; kleinere Chunks werden
  deterministisch gewählt, eine unmögliche Kernzeile scheitert fail-closed;
- Temporärdisk-Preflight trennt Framecache und Profilstores und verlangt
  Schätzung mal 1,20 plus Reserve; bei Unterschreitung wird nichts geschrieben;
- abgebrochene Disk-/CUDA-Stores besitzen keinen gültigen Commitmarker;
- CUDA-Fehler verwerfen die gesamte Phase und der CPU-Neustart entspricht der
  reinen CPU-Referenz;
- CPU-À-trous ist M7-Referenz; ein CUDA-Mikrobenchmark allein aktiviert keinen
  alternativen Fusionspfad;
- Durchsatzbenchmark protokolliert drei Wiederholungen und deren Median auf
  eingefrorener Referenzhardware.

---

## 21. Synthetische Qualitätsfixtures

Vor realen Runs werden reproduzierbare Fixtures erzeugt:

### 21.1 Hochauflösende Ground Truth

- analytische Gauß-/Moffat-Sterne;
- unterschiedliche Helligkeiten, Farben und Subpixelpositionen;
- glatte Galaxien-/Nebelfrequenzen;
- linearer und gekrümmter Hintergrund;
- bekannte WCS.

### 21.2 Frameerzeugung

- pro Frame bekannte affine Transformation;
- optional gültiges lokales Verzerrungsfeld;
- zufällige Dither-Phasen;
- räumlich variable PSF;
- unabhängige PSF-Faltung und Integration über die Sensorpixelflächen, nicht
  Wiederverwendung des Rekonstruktionskernels als Ground-Truth-Generator;
- Poisson- und Ausleserauschen;
- Hotpixel und Cosmic Rays;
- Bayer-Sampling für alle Pattern;
- bekannte schlechte Frames und lokale Unschärfebereiche.

### 21.3 Metriken

- Fluxfehler;
- Zentroidfehler;
- gematchte FWHM/Elongation/Tails;
- MTF50 oder äquivalente Kantenantwort;
- Hintergrund-RMS;
- Rekonstruktionsfehler gegen Ground Truth;
- Farbdifferenz neutraler und farbiger Sterne;
- Seam- und Supportfehler.

Fixtures und erwartete Kennwerte werden klein gehalten und in den Catch2-Tests
generiert; große Binärfixtures werden vermieden.

**Determinismus und CI-Kosten.** Die Fixture-Erzeugung verwendet einen festen,
im Test benannten RNG-Seed und `double`-Arithmetik mit expliziten Toleranzen für die bei Gauß-/Moffat-Profilen
und Rotation erforderlichen mathematischen Funktionen; erwartete Kennwerte
werden mit derselben expliziten Toleranz geprüft wie die CPU-/CUDA-Parität
(11.12). Weil §30.3.2 pro Vertrag mindestens ein zunächst fehlschlagendes
Fixture verlangt, wird die dadurch entstehende zusätzliche Suite-Laufzeit pro
Meilenstein gemessen und in §30.1 protokolliert; überschreitet sie 90 s auf der
Referenzhardware, werden die teuersten Fixtures hinter ein separates
`tile_compile_slow_fixture_tests`-Target gezogen, das im Standard-CI-Lauf
enthalten bleibt, aber getrennt zeitlich ausgewiesen wird.

---

## 22. Dateibasierter Implementierungsplan

### 22.1 Neue Core-Dateien

| Datei | Inhalt |
|---|---|
| `include/tile_compile/registration/registration_sampling_plan.hpp` | Sampling-Typen, Warp-Konvention, Serialisierungs-API |
| `src/registration/registration_sampling_plan.cpp` | Warp-Inversion, lokale Inversion, Hashing, JSON |
| `include/tile_compile/reconstruction/forward_drizzle.hpp` | CPU-API, Konfiguration, Ergebnis-/Diagnostiktypen |
| `src/reconstruction/forward_drizzle.cpp` | CPU-Referenz, Droplet, Clipping, Profilakkumulation |
| `include/tile_compile/reconstruction/profile_plane_store.hpp` | transaktionale Plane-/Mask-Store- und Region-API für U/R/F/M |
| `src/reconstruction/profile_plane_store.cpp` | disk-/mmap-gestützte Stores, Checksummen und atomarer Commit |
| `include/tile_compile/reconstruction/multiband_reconstruction.hpp` | Mehrband-API und Ergebnisdiagnostik |
| `src/reconstruction/multiband_reconstruction.cpp` | maskierte À-trous-Zerlegung, Alpha, Energieguard |
| `include/tile_compile/reconstruction/forward_drizzle_cuda.hpp` | CUDA-API |
| `src/reconstruction/forward_drizzle_cuda.cu` | CUDA-Kernel und Fallbackdaten |

### 22.2 Neue Runner-Dateien

| Datei | Inhalt |
|---|---|
| `apps/runner_phase_sampling_geometry.hpp/.cpp` | geometrische Coverage, Maskenvorbereitung, `coverage_gate`, `sampling_geometry.json` |
| `apps/runner_phase_source_quality_maps.hpp/.cpp` | Source-CFA-Proxys, Scale-Map-Sink, Cache und Artefakte |
| `apps/runner_phase_global_quality.hpp/.cpp` | `G_quality(f)` aus dem Green-Proxy, `QualityFrameWeightPlan`, `global_quality.json` (ersetzt `runner_phase_aqmh_global_quality.*`) |
| `apps/runner_phase_forward_drizzle.hpp/.cpp` | Phase-Orchestrierung, Controls, Mehrband, Validation, Outputs |
| `apps/runner_phase_reconstruction_diagnostics.hpp/.cpp` | Diagnostikphase (ersetzt `runner_phase_aqmh_diagnostics.*`) |

### 22.3 Wesentlich zu ändernde Dateien

| Datei | Änderung |
|---|---|
| `apps/runner_phase_registration.hpp/.cpp` | RegistrationSamplingPlan erzeugen; Nutzsignal-PREWARP entfernen |
| `apps/runner_phase_metrics.hpp/.cpp` | Normalized-Cache-Metadaten und Preservation |
| `apps/runner_shared.hpp/.cpp` | Source-Region-Reads und Cache-Metadaten |
| `apps/runner_pipeline.cpp` | einzigen linearen Forward-Drizzle-Pfad orchestrieren; Methodenbranch entfernen |
| `apps/runner_resume.cpp` | Single-Method-Resume und fail-closed Legacy-Ablehnung |
| `apps/runner_phase_aqmh_reconstruction.cpp`, `runner_phase_aqmh_maps.cpp`, `runner_aqmh_pipeline.cpp` | allgemein nutzbare Kandidaten-/Validierungsteile extrahieren und in neue Module überführen; Dateien in M10 löschen |
| `include/tile_compile/metrics/source_quality_map.hpp` (Umbenennung von `aqmh_quality_map.hpp`) | skalenspezifische Maps und Sink |
| `src/metrics/source_quality_map.cpp` (Umbenennung von `aqmh_quality_map.cpp`) | Source-CFA-Proxy und Scale-Ausgabe |
| `include/tile_compile/metrics/source_quality_map_cache.hpp` (Umbenennung von `aqmh_quality_map_cache.hpp`) | Multi-Stream-/Source-Region-API |
| `src/metrics/source_quality_map_cache.cpp` (Umbenennung von `aqmh_quality_map_cache.cpp`) | neues Cachelayout und Metadaten |
| `include/tile_compile/metrics/global_quality.hpp` (Umbenennung von `aqmh_global_quality.hpp`) | `G_quality(f)` auf Green-Proxy |
| `include/tile_compile/reconstruction/reconstruction_validation.hpp` (Umbenennung von `aqmh_validation.hpp`) | fester Mehrbild-Sternsatz |
| `src/reconstruction/reconstruction_validation.cpp` (Umbenennung von `aqmh_validation.cpp`) | Dreiwegmessung an identischen Positionen |
| `src/reconstruction/aqmh_sigma_clip.*` | Clipping-Guard-Semantik (11.8) als neutral benanntes Modul übernehmen |
| `include/tile_compile/config/configuration.hpp` | Drizzle-/Multiband-Konfiguration |
| `src/io/config.cpp` | Parsing, Serialisierung, Validierung |
| `CMakeLists.txt` | neue Quellen und Tests |

Die Umbenennungen erfolgen in M0/M5 als reine Verschiebung mit Namespace-
Anpassung; die bisherigen `aqmh_*`-Tests werden dabei mitgezogen
(`test_aqmh_quality_map*.cpp` → `test_source_quality_map*.cpp`,
`test_aqmh_validation.cpp` → `test_reconstruction_validation.cpp`).
Bestehende Tests, die nur den PREWARP-Rekonstruktionskern absichern
(`test_aqmh_reconstruction.cpp`), wandern in das Legacy-Referenztarget.

### 22.4 Tests

| Datei | Zweck |
|---|---|
| `tests/test_registration_sampling_plan.cpp` | Warp-/Planvertrag |
| `tests/test_forward_drizzle.cpp` | CPU-Drizzlekern |
| `tests/test_profile_plane_store.cpp` | Region-I/O, Checksumme, Crash-/Commitvertrag und Budget |
| `tests/test_multiband_reconstruction.cpp` | Bänder, Alpha, Support und Streaming |
| `tests/test_reconstruction.cpp` | Single-Method-Integration und Controls |
| `tests/test_reconstruction_validation.cpp` | feste gematchte Population |
| `tests/test_source_quality_map.cpp` (aus `test_aqmh_quality_map.cpp`) | source-space Skalenmaps |
| `tests/test_source_quality_map_cache.cpp` (aus `test_aqmh_quality_map_cache.cpp`) | Cache-/Region-/Veto-Vertrag |
| `tests/test_sampling_geometry.cpp` | geometrische Coverage, `coverage_gate` |
| `tests/test_runner_resume.cpp` oder bestehende Runner-Contract-Tests | Resume und Hashinvalidierung |

### 22.5 Verbindlich zu entfernende Legacy-Bestände

Vor Abschluss des Cutovers wird mit `rg`, Build-Graph- und Schema-Prüfungen ein
Löschinventar erzeugt. Mindestens folgende Kategorien müssen vollständig aus
dem aktiven Produkt verschwinden:

- Classic-Rekonstruktionsquellen, OLA-/Tile-Gewichtungsorchestrierung und deren
  methodenspezifische Metriken;
- die in 6.5 als „entfernt" markierten Config-Blöcke samt Parser, Schema-Feldern,
  Serialisierung, Beispielen und GUI-Feldern; die dort als „behalten/verschoben"
  markierte Per-Frame-Cosmetic-Correction bleibt funktional erhalten und wird
  nur nach `calibration.frame_cleanup` verschoben;
- alter AQMH-PREWARP-Rekonstruktionskern und Nutzsignal-PREWARP-Cache;
- Methoden-/Engine-Enums, Parserwerte, CLI-Optionen, Backendfelder und
  Frontend-Auswahl;
- PREWARP- und nachgelagerte DEBAYER-Phase aus der aktiven Phasenfolge;
- Legacy-Postprocessingkandidaten aus Abschnitt 15.4;
- methodenspezifische Schemas, Beispiele, Übersetzungen, Reports und aktive
  Dokumentation;
- Tests, die ausschließlich das entfernte Verhalten absichern.

Historische Parser dürfen in einem klar getrennten read-only Modul verbleiben.
Dieses Modul darf keine Rekonstruktionsbibliothek linken und keine Resume- oder
Schreiboperation anbieten. Historische Methodikdokumente werden unter `attic/`
als unveränderte Aufzeichnungen aufbewahrt.

---

## 23. Implementierungsmeilensteine

Jeder Meilenstein muss separat bauen und seine Tests bestehen. Kein
Meilenstein startet automatisch einen realen Bildverarbeitungslauf.

### M0 — Vertragsbaseline und Konfiguration

**Änderungen:**

- [x] **`pipeline_contract_version` und Single-Method-Runmetadaten einführen**
  (2026-09-03, siehe 30.4);
- [~] Methoden-/Engineauswahl entfernen: **CLI + Backend-Run-Create + Runner
  erledigt** (`--force-classic`, `FORCE_CLASSIC`, `getEffectiveMethod`,
  `read_run_method_local`); `aqmh`→`reconstruction`-Restrukturierung (an
  M2/M3/M6 gekoppelt) und tiefer §17.5-Backend-Anteil (M8) offen;
- [~] Legacy-Methoden-/Engineschlüssel fail-closed + strukturelle Blöcke
  strippen: **Modul + Tests + `cli migrate-config` + Runner-Verdrahtung
  (`from_yaml_text_migrated` + `config_migration.json`) erledigt**; die
  key-Renames offen (mit der Restrukturierung);
- [x] **OSC/MONO als Umfang fixieren und bereits debayertes RGB fail-closed
  ablehnen** (2026-09-04, `input_class_policy` + SCAN_INPUT-Verdrahtung + Tests);
- [x] **temporären test-only Referenzzugriff isolieren**: CMake-Targets
  `tile_compile_legacy_reference` + `tile_compile_legacy_reference_tests`
  (`test_aqmh_reconstruction.cpp` verschoben) (2026-09-03/04, siehe 30.4);
- [~] **`reconstruction:`-Konfigurationsvertrag §6.1–6.3**: Struct + Parser +
  `validate()` + Serializer + `schema.json` + `schema.yaml` + 4 Tests erledigt
  (2026-09-04, siehe 30.4); offen: `tile_compile.yaml`-Default, `examples/`;
- [x] **`RegistrationSamplingPlan`-Typen, Frame-/CFA-Vertrag und affine
  Round-trip-Tests** (2026-09-03, siehe 30.4); die lokale Inversion aus 7.3 ist
  ebenfalls implementiert, ihre M1-Diagnostik/Ratenzählung folgt in M1;
- [x] **Entwicklungsrunner bis M2 vor Run-Mutation mit
  `PIPELINE_UNAVAILABLE_DURING_CUTOVER` sperren** (2026-09-03, siehe 30.4;
  `preprocess` bewusst ausgenommen).

**Abnahme:**

- aktive Konfiguration besitzt keinen Methodenschalter;
- neue Runs schreiben den neuen Pipelinevertrag;
- alte Konfigurationen werden fail-closed und ohne Seiteneffekt abgelehnt;
- alte Rekonstruktion ist nur aus dem Testtarget, nicht aus dem Runner
  erreichbar;
- OSC/MONO/RGB-Scanvertrag ist eindeutig und getestet;
- der Entwicklungsrunner bricht in M0/M1 vor jeder Run-Mutation kontrolliert ab;
- vollständige Test-Suite grün.

### M1 — RegistrationSamplingPlan und geometrische Coverage

**Aktueller Abschlussstatus:** M1–M3 sind nicht abgenommen. Die Bibliotheks- und
Artefaktarbeit aus §0.2–0.4 ersetzt keine produktive Runnerintegration. Vorrang
haben jetzt (1) Aufbewahrung und Provenienz des normalisierten Quellcaches,
(2) neue Phasenfolge ohne Nutzsignal-PREWARP mit geometrischem COMMON_OVERLAP,
(3) Missing-/Zero-Veto sowie gemeinsamer Resume-/Fallback-Vertrag und
(4) Abnahme einschließlich großer nativer Bilder und Abbruch-/Wiederanlauftests.


**Änderungen:**

- [ ] Registration und Nutzsignal-PREWARP im aktiven Runner trennen; eigener Event-/Vorgänger-/Resume-Vertrag fehlt weiterhin;
- [x] **Sampling-Plan mit stabilen Frame-IDs, CFA-Ursprung und lokalen
  Modellkoordinaten nach Canvasberechnung persistieren** (2026-09-04, siehe
  30.4) — additiv in `run_phase_registration_prewarp` verdrahtet: schreibt
  `artifacts/registration_sampling.json`, ohne PREWARP zu verändern; real
  gegen einen M31-Lauf verifiziert;
- [x] **affine und lokale Source→Canvas-Abbildung** (Teil von 30.4/30.5);
- [x] **geometrische Coverage ohne Bild-PREWARP**: Algorithmus + Tests
  (30.6), diagnostische Runner-Verdrahtung (30.7), fail-closed
  Geometrieprüfung innerhalb der vorhandenen PREWARP-Runnerphase verdrahtet
  (eigener Event-/Resumevertrag weiterhin offen) und auf einem
  echten M31-Lauf gegen einen absichtlich fehlschlagenden Gate verifiziert
  — Legacy-Pfad nachweislich unverändert (30.8);
- [x] `analysis_common_mask`, direkte Support-/`n_eff`-Gates erledigt, mit
  **seit dem Audit vom 2026-09-05 geänderter, korrigierter Semantik**
  (§0/§0.1, Befunde A2/A3; die Darstellung in 30.6/30.7/30.10 beschreibt die
  inzwischen ersetzte Vorstufe): `analysis_common_mask` ist jetzt eine dichte,
  CFA-unabhängige Frame-Footprint-Überlappung, `n_eff` ein echtes
  flächengewichtetes `(ΣB)²/Σ(B²)` statt einer Frame-Anzahl, beides
  zeilenweise gestreamt statt vollbildgroß gepuffert (behebt zugleich das
  RAM-Risiko aus B3). Loch-Gate (`max_internal_hole_area_px`, jetzt
  streifenweise über `StripeHoles` statt Vollbild-Flutfüllung) real
  verifiziert — 3-Px-Lochfund auf M31 (30.9, Semantik seither umgebaut).
  Beide Masken als FITS persistiert. `sampling_geometry.json` (Schema 2)
  wird als echtes Laufartefakt geschrieben. Zirkuläre Ditherdiagnostik
  (Rayleigh-Schätzer) unverändert gültig und real auf M31 verifiziert,
  beeinflusst `gate.passed` nachweislich nicht. Noch offen (bewusst
  zurückgestellt, betrifft geteilten Legacy-Code): COMMON_OVERLAP von den
  geometrischen Masken speisen statt PREWARP-Pixeln — erst sinnvoll, sobald
  M2 eine eigene, getrennte COMMON_OVERLAP-Entsprechung im neuen Pfad hat;
  ebenso weiterhin offen: eigener `Phase`-Enum-Wert (derzeit meldet ein
  fehlschlagendes Gate `Phase::PREWARP`, Befund B4).

**Abnahme:**

- [ ] aktiver Runner erzeugt keine `prewarped_frames` (PREWARP läuft noch;
  erst mit dem M2-Forward-Drizzle-Kernel ablösbar);
- [x] **test-only Referenzpfad erzeugt für Vergleichstests weiterhin das
  bekannte Ergebnis** — auf Binärebene nachgewiesen (30.8): `strings`-Diff der
  beiden Targets zeigt den Abbruchzweig ausschließlich im neuen Pfad; realer
  Lauf mit absichtlich scheiterndem Gate bestätigt unverändertes
  Legacy-Verhalten;
- [x] **Coverage stimmt bei synthetischen affinen Warps mit der erwarteten
  Referenzmaske überein**; Supportanteil, p10-`n_eff`, Mindestpixelzahl und
  interne Lochfläche werden ausgewertet und bestehen an/unter ihren
  Grenzwerten (13 Tests, 30.6/30.9); real auf M31 verifiziert — die
  Lochprüfung fand dort ein echtes 3-Pixel-Loch bei 20 Frames/`pixfrac=0.8`
  und lehnte korrekt ab (30.9);
- [x] **Ditherdiagnostik ist zirkulär korrekt, beeinflusst die Gateentscheidung
  nicht** — Rayleigh-Schätzer implementiert, mathematisch an zwei Extremfällen
  exakt verifiziert (identische Frames → `R=1`/`sigma≈0`; perfekte
  4-Phasen-Quadratur → `R=0`/`sigma` maximal), real auf M31 im
  plausiblen Zwischenbereich (`0.48`/`0.60` px), `gate.passed` nachweislich
  unverändert durch das Diagnosefeld (30.11);
- [x] **Gateverletzungen stehen in `sampling_geometry.json`, ohne
  `COMMON_OVERLAP` fälschlich als ausgeführt zu markieren** — das Artefakt
  wird unabhängig vom Gate-Ergebnis geschrieben, ein scheiterndes Gate beendet
  den Lauf vor `COMMON_OVERLAP` (30.7/30.8);
- [x] **lokale Inversion ist einschließlich Modellskalierung/-offset
  geguardet und diagnostiziert** (30.4/30.5).

### M2 — CPU Forward-Drizzle 1x Uniform

**Änderungen:**

- [x] **affine Square-Droplets** (echter Polygon-Rechteck-Schnitt, kein
  Bounding-Box-Ersatz) **und adaptive Subdivision für lokale Warps** (echte
  Tiefe-2-Konvergenzprüfung, Positions- und Flächenkriterium) — 30.12;
- [x] **CFA-Farbzuordnung mit CFA-Ursprung sowie MONO-L-Pfad** — gemeinsame
  Implementierung mit M1, getestet (30.12);
- [x] **Frame-lokale Aggregation** — zweistufige Form (`A/B` → `x_f,c` →
  gewichtete Kombination), M3-kompatibel gebaut (30.12);
- [x] **Uniform-Control** — inkl. `n_eff`, feste Frame-Reihenfolge (30.12);
- [x] transaktionaler Streaming-Profilstore mit budgetierten Streifen, Supportebenen, atomarem Generationen-Commit und geprüften Region-Reads (§0.3/0.4). Die In-Memory-Komfortfunktion ist zusätzlich verfügbar. Allgemeiner Q-Cache und produktiver Resume-Vertrag bleiben getrennte offene Punkte.

**Abnahme:**

- [~] **Oberflächenhelligkeits-, Aperturflux-, Zentroid-, CFA-Ursprungs-, MONO-
  und Determinismustests bestanden** — Flächenidentität (separater geometrischer Nachweis, kein vollständiger Photometrienachweis), CFA-Ursprung und MONO real verifiziert (8+3 Tests, 30.12);
  synthetische Aperturflux-/Zentroidtests bei 1x/2x und Streifengrenzen-Determinismus sind inzwischen vorhanden (`test_forward_drizzle.cpp`, §0.2). Offen bleiben umfassendere unabhängige Truth-/PSF-Abnahme und Parallel-Determinismus, falls ein paralleler Pfad eingeführt wird; der aktuelle Referenzpfad ist seriell;
- [x] **lokaler Warp erfüllt 0,05-Pixel-Positions-, 0,5-%-Flächen- und
  Tiefe-2-Subdivisionstoleranzen sowie die 0,1-%-Framefehlergrenze** — real
  implementiert und an beiden Enden getestet (Nullverschiebung konvergiert
  exakt; nicht invertierbares Modell führt zum korrekten Frameausschluss,
  30.12);
- [~] Store-/Chunk-/Commit-Fehlertests bestanden (§0.3): Streifenexport, Fehler nach erstem Streifen, unveränderter vorheriger Commit, Prüfsummen/Shape/Identität und Budget. Echter Prozess-Kill/Powerloss-Test und vollständiger Pipeline-Resume bleiben offen;
- [x] noch keine Q-Maps und keine Mehrbandfusion — zutreffend, M2 berechnet
  ausschließlich Uniform-Control;
- [~] keine Nutzung von `prewarped_frames` — im Kern strukturell erfüllt
  (Quellwerte kommen ausschließlich über den injizierten
  `SourceImageProvider`, Vertrag §10.1). Der optionale Runner-Diagnoseexport lädt normalisierte Quellen; die geprüfte Cache-/Qualitäts-/Store-Kette ist als Bibliotheksintegration vorhanden (§0.4). Offen ist deren produktive Runner-Anbindung einschließlich Cache-Lebensdauer und Resume.

### M3 — Robustes Clipping und Raw-Forward-Drizzle

**Änderungen:**

- [x] **gemeinsame Akzeptanzmaske** — `apply_robust_clipping()` (8 Schritte
  aus §11.8, 30.15) und ihre Verdrahtung als tatsächlich gemeinsame,
  identische Maske über Uniform und Raw (`compute_forward_drizzle_uniform_and_raw()`,
  30.16) sind implementiert und getestet; Detailprofile (Fine/Medium) folgen
  erst mit der Mehrbandfusion (M6);
- [x] minimale source-space Q-Versorgung: `proxy_version=1` mit Quad-Green,
  Green-Highpass/MAD beziehungsweise direktem MONO-L implementiert und exakt
  getestet (`source_quality_proxy.hpp/cpp`, 30.18); **`G_quality(f)` in
  `GLOBAL_QUALITY` als echte Runner-Phase** (`global_quality.hpp/cpp` +
  `runner_forward_drizzle`, 30.19/30.23), real auf M31 end-to-end verifiziert,
  `[0,1]`-Vertragskonflikt der Bestandsformel gefunden und per
  Sigmoid-Transform geschlossen. Noch offen: Zero-Veto-Maskenweiterleitung,
  frame-lokaler `composite`-Q-Stream (§13.4, überlappt mit M5);
  skalenspezifische Streams, Sink und Region-Reads folgen erst in M5;
- [x] effektiver Gewichtsplan genau einmal — `QualityFrameWeightPlan` mit
  einmaliger `G_eff`-Berechnung, Registrierungsfaktoren wörtlich aus dem
  Sampling-Plan (keine Doppelgewichtung strukturell garantiert), kanonischem
  Hash, fail-closed-Loader, atomarer Artefakt-Persistenz mit
  Vorgängerbindung (§0.4) **und Runner-Verdrahtung als `GLOBAL_QUALITY`-Phase**
  (`source_quality_plan.json` real auf M31 geschrieben, 30.20/30.23).
  `Q_composite` bleibt Q-Map-abhängig (M5);
- [x] **Uniform und Raw im selben Durchlauf** (30.16), mit **echtem `G_eff(f)`
  aus `GLOBAL_QUALITY`** verdrahtet (30.21/30.23): Raw weicht auf echten
  M31-Daten messbar von Uniform ab (`mean|u−r|` ≈ 2–3), geteilte,
  bitidentische Clipping-/Support-Maske. `Q_composite` weiterhin `1.0` (M5);
- [x] Raw-Baseline transaktional per Streaming persistieren — transaktionaler
  Store mit unveränderlichen `generation-*`-Verzeichnissen und `current.json`
  als Commit-Punkt (§0.3), auf M31 real geschrieben (24 FITS-Ebenen
  uniform+raw, 30.23).

**Abnahme:**

- [x] identische Samples/Masken und ausschließlich geometrisch gewichtete
  gemeinsame Clippingentscheidungen nachgewiesen — Algorithmus **und**
  Uniform/Raw-Verdrahtung mit von Hand nachgerechneten Erwartungswerten
  verifiziert (12 Tests, 30.15/30.16), **und real auf M31 end-to-end**
  (30.23): `sha256(raw_R_support) == sha256(uniform_R_support)`, d. h. die
  Akzeptanzmaske ist bitidentisch über beide Profile. Das Kandidatenbudget
  ist reaktiv begrenzt (30.17) und synthetisch getestet (§0.2/0.3);
  Speicher-/Laufzeitverhalten real: `SAMPLING_GEOMETRY` ~35 s,
  `FORWARD_DRIZZLE` ~32 s, Peak ~649 MB für 6 Frames bei `internal_scale=1`;
- [x] `QualityFrameWeightPlan`, bestehende Residualfunktion, Missing-Floor 0,55 und
  chain-depth-Modellfaktor sind formel-, frame-ID- und hashstabil; keine
  Doppelgewichtung — Struktur, einmalige `G_eff`-Berechnung, wörtliche
  Übernahme der Registrierungsfaktoren, kanonischer Hash, fail-closed-Loader,
  atomare Artefakt-Persistenz mit Vorgängerbindung (§0.4) und
  Runner-Verdrahtung (30.20/30.23); auf M31 real geschrieben und geprüft
  (`registration_residual_factor` aus dem Sampling-Plan übernommen, `g_eff`
  als exaktes Produkt);
- [ ] Missing-/Zero-Veto-Semantik getestet;
- [x] Raw und zugehöriges geclipptes Uniform werden als **eine** geprüfte
  Generation atomar veröffentlicht (§0.3), inkl. Runner-Verdrahtung
  (`run_forward_drizzle_stages`) und Resume-Einstieg
  (`resume_forward_drizzle_command`), der die Vorgänger vor jedem
  Phasen-/Artefaktschreiben validiert (30.23). Quell-/Sampling-/
  Algorithmusidentität, exakte Ebenenmenge, FITS-Dimensionen und Prüfsummen
  werden validiert; Resume lehnt bei geänderter Config/Geometrie ab.
  Uniform-Fallback: `verify_profile_store` liefert das Signal; der
  automatische Fallback-Zweig im Resume-Pfad ist noch nicht verdrahtet.

### M4 — Internes 2x-Raster

**Änderungen:**

- [~] skalierte Geometrie, Masken, Crop und WCS mit expliziter
  Canvas-/Crop-Vorzeichenkonvention — `scale_wcs_to_output()` (§12.2)
  implementiert und getestet (30.24); die Store-Geometrie ist bei `2/1`
  korrekt halbiert (30.26); Anwendung auf `canvas_mask`/`common_overlap_mask`/
  Crop/WCS-Schreiben gehört zur Ausgabe-Pipeline (M6/M10);
- [x] OSC-RGB-/Luminanz- und MONO-L-Outputs in `output_scale` — der Store
  persistiert Uniform **und** Raw je Kanal in 1x (`2/1`) bzw. 2x (`2/2`),
  real auf M31 verifiziert (30.23/30.26);
- [x] deterministischer 4/4-Support- und 2x→1x-Flächenmittelvertrag mit
  minimalem Subpixel-`n_eff` — `downsample_profile_plane_2x2()` /
  `downsample_uniform_and_raw_2x2()` implementiert und mit Handrechnung
  getestet (30.24);
- [x] explizite Modi `1/1`, `2/1`, `2/2` ohne Auto (`OutputScaleMode`,
  `valid()` lehnt `output > internal` ab); Produktionsdefault `2/1`
  (Config-Default);
- [ ] Downstream-Pass-through.

**Abnahme:**

- [~] synthetische WCS-/Zentroid-/Aperturfluxtests für 1x und 2x bestanden —
  WCS-Skalierung `S=1`/`S=2` gegen Handrechnung getestet (30.24);
  Zentroid-/Aperturfluxtests auf realen 2x/1x-Bildern noch offen;
- [x] U/Raw werden identisch in `output_scale` überführt; 4/4-Support und
  minimales Subpixel-`n_eff` sind nachgewiesen — Operator, 4/4-Vertrag, der
  speicher-begrenzte `2/1`-Streaming-Downsample (bit-identisch zur Referenz,
  jede Chunkhöhe) **und die Aktivierung im transaktionalen Store**
  (`output_scale` im `reconstruction_hash`, halbierte Geometrie, gespeicherte
  Werte bit-identisch zur Referenz) getestet (30.24/30.25/30.26);
- [ ] BGE/PCC/HMS können OSC- und MONO-Ausgaben in 1x und 2x verarbeiten —
  **blockiert auf der Ausgabe-Pipeline-Verdrahtung (M6/M10)**, nicht auf
  M4-Algorithmik: der Kernel-Rauschkorrelations-Faktor `f = W/√S0` liegt vor
  (§12.4/30.24), wird im `FORWARD_DRIZZLE`-`phase_end` gemeldet (30.26) und
  ist als Bandbreiten-Korrektur für BGE-RMS/PCC-SNR/HMS bereit; der
  `reconstruct`-Einstieg endet aber bei `reconstruction_ready` und ruft
  keine Downstream-Phase auf;
- [ ] Reports deklarieren korrekten Pixelmaßstab und Fluxvertrag.

### M5 — Skalenspezifische Source-Q-Maps

**Änderungen:**

- [x] CFA-Green-/Luma-Proxy (MONO: L unverändert; OSC: edge-aware
  `proxy_full`) als source-aufgelöste Analyseeingabe der Pyramide (30.27);
- [x] Scale-Sink (`QualityScaleMapSink`) + null-Default-Per-Scale-Hook in
  `metrics::compute_aqmh_quality_map()`, Legacy byte-identisch (30.27);
- [x] Multi-Stream-Cache (`cache/source_quality_maps/{composite,scale_0..3,artifact}/`
  + `metadata.json`, `uint16`-Wert-Stream + separater `uint8`-Hard-Veto-Stream
  je `.bin` (Schema 2): Wert = Valid-Mean über positive Quellpixel, Veto = 1
  falls **irgendein** exaktes `Q=0` überdeckt ⇒ Lesen erzwingt dort `NaN`;
  `storage_divisor` räumlich, atomarer Commit über `metadata.json`,
  fail-closed-Reader) (30.28/30.30);
- [x] Source-Region-Reads (`read_region(y0,y1)`, Nearest-Upsample, deckt sich
  zeilenweise mit `read_full`) (30.28);
- [x] getrennte Source-Identity-/Quality-Config-Hashes (30.28); Resume ohne
  unnötige Registrierungsabhängigkeit: Hash-Domäne getrennt, Resume-Pfad
  prüft den Cache fail-closed gegen die Checkpoint-Hashes (30.29);
- [x] `SOURCE_QUALITY_MAPS` als angehängte `Phase = 28` vor `GLOBAL_QUALITY`
  im `reconstruct`-Runner + `build_source_quality_map_cache()`-Orchestrator,
  synthetisch end-to-end verifiziert (30.29);
- [x] `Q_composite`-Konsum je Quellsample im Rekonstruktor (§11.7 geometrisch
  `K`-gemittelt über `QA`-Akkumulator, kein Pixel-Veto) +
  `w_raw = B·G_eff·Q_composite`-Verdrahtung durch alle Persist-/Stream-Ebenen,
  `SourceQualityProvider`, fail-closed-Reader in
  `persist_forward_drizzle_from_predecessors`, `source_quality_cache_hash` im
  `reconstruction_hash` (30.31); synthetisch nachgerechnet (15→12, 15→10,
  NaN→0, kein Pixel-Veto);
- [ ] **M31-End-to-End-Verifikation** der Q-Map-Wirkung (`mean|u−r|` +
  Finit-Pixel-Fraktion des gecachten Composite) — einziger offener M5-Punkt.

**Abnahme:**

- [x] keine gleichzeitige Vollmap-Residenthaltung aller Skalen
  (`peak_resident_scale_maps == 1` mit Sink, 30.27);
- [x] Veto- und Regionslesetests bestanden: Composite-harte-Maske (30.27),
  exakter `Q=0`-Hard-Veto überlebt die Speicher-Umtastung auch in gemischten
  Zellen (30.30), `read_region` deckt sich zeilenweise mit `read_full`
  (30.28), `Q_composite=0` erzeugt kein Pixel-Veto im Rekonstruktor (30.31);
- [x] Composite bleibt mit der übernommenen, dokumentierten Quality-Semantik
  vergleichbar (MONO-**Bit-Identität** zur Legacy-Q-Map, 30.27).

### M6 — Mehrprofil-Drizzle und Mehrbandfusion

**Änderungen:**

- [x] Fine-/Medium-Profile (`w_fine = B·G_eff·pow(Q_scale0, fine_quality_exponent)`,
  `w_medium = B·G_eff·pow(Q_scale1, medium_quality_exponent)`, geteilte
  Clippingmaske, `FrameQualityProvider`/`MultibandProfileParams`, 30.34);
  F/M-Persistenz im transaktionalen Store + `2/1`-Streaming noch offen;
- [x] maskierte, supportpropagierende À-trous-Zerlegung (§14.2,
  `reconstruction/atrous_decomposition`, 30.33); Bandzuordnung + Blend
  (`fuse_multiband_channel`, 30.35); **streifenweise** `fuse_multiband_streamed`
  bit-identisch zur Vollbildreferenz für jede Chunkhöhe (30.39, §14.7);
- [x] abwärtsbegrenzte B3-Alpha-Glättung (`smooth_alpha_b3`, `min`-Kappe,
  komponentenweise) und Luma-/MONO-Energieguard mit Grenze 1,30, sechs
  Bisektionsschritten, ohne Sternkonzentrationsausnahme (`apply_energy_guard`,
  30.35);
- [x] gemeinsame RGB-Alpha-Semantik aus konservativen Kanalminima
  (`compute_adaptive_alpha` — `A_neff`/`A_coverage` min-über-Kanal, geteiltes
  alpha_j; `fuse_multiband`-Orchestrator, 30.35);
- [x] `A_separation`, weighted-p10-`A_artifact`,
  direct-fraction/residual-p20-`A_registration` (`reconstruction/alpha_confidence`,
  `weighted_percentile` + Hazen, 30.36) **im Drizzle-Streifen als
  kanal-minimierte Pro-Pixel-Maps verdrahtet** + `reconstruct_multiband_reference()`
  End-to-End (30.37);
- [x] streifenweise À-trous-Fusion mit Halo gegen den In-Memory-Referenzpfad
  (`fuse_multiband_streamed` + `multiband_fusion_halo_rows`, 30.39, §14.7);
- [x] Dreiwegvalidation mit 20/30-Stern-, Bootstrap-CI-, N/A- und
  Promotionsvertrag (§15) — sternbasiert, **nicht blockiert** (30.40); je
  Validationstern ein `multiband_effective`-Flag (lokal `alpha_final ≡ 0`
  über alle Bänder ⇒ keine positive Multiband-Evidenz)
  (`reconstruction/multiband_validation`, deterministischer geseedeter
  Bootstrap-CI, Pro-Stern-FWHM auf der gematchten effektiven Sternteilmenge,
  selbstnormierender `boundary_seam_score`, „N/A ist nie ein impliziter
  Pass"; 8 synthetische Fälle, 30.42). **Im Runner verdrahtet** (30.43):
  `fuse_multiband_store_to_image` bildet die drei Arbeitsluminanz-Kandidaten
  + `alpha_final`-Maps im selben Fusionsdurchlauf, `MULTIBAND` führt die
  Auswahl aus und schreibt `artifacts/forward_drizzle.json` mit
  `selected_candidate` / `selection_reason` / `fallback_reason` / `validation`
  (§16.3, Teilmenge). Separat offen: das Umschalten der **ausgelieferten
  Datei** auf den gewählten Kandidaten (Resume-/Checkpoint-/STACKING-Vertrag)
  und `kMultibandValidationVersion` + Seam-Konstanten in
  `multiband_config_hash`;
- [x] F/M-Persistenz im transaktionalen Store (`uniform_raw_multiband_clipped`,
  Fine/Medium + vier Confidence-Pseudoebenen, `multiband_config_hash` §16.4) +
  Runner-`MULTIBAND`-Phase → `reconstruction_multiband.fits`
  (`persist_multiband_store_from_predecessors` / `fuse_multiband_store_to_image`,
  30.41) — **für alle Output-Scales** (`1/1`, `2/1`, `2/2`); `2/1` mittelt
  Fine/Medium per 2×2-Mean und die kanal-minimierten Confidence-Maps per
  2×2-`min` + `AND`-Support, bit-exakt zur nicht-streamenden Referenz und
  chunk-unabhängig; MONO **und** OSC (`fuse_multiband_store_to_image`
  streifenweise, speicherbegrenzt) synthetisch bit-exakt verifiziert;
- [ ] reale Registrierungsrunde: M31 (klärt zugleich den M5-Echtdatenpunkt) +
  M42/OSC.

**Abnahme:**

- Identitäts-/Nullwirkungs- und level-spezifische Supportverträge bestanden;
- keine Seams/Farbsäume bei sparser R/B-Coverage in synthetischen Tests;
- Streaming entspricht Vollbildreferenz und hält das Speicherbudget;
- Kandidatenfallback, Pflicht-N/A, Mindeststernzahlen und CI-Grenze vollständig
  getestet; Alpha-Glättung hebt weder Veto noch lokale Evidenz an.

### M7 — CUDA

**Änderungen:**

- [ ] CUDA-Droplet, Clipping und Profile; À-trous bleibt im M7-Referenzpfad CPU
  (Slice 2 — Kernel + Paritätsmatrix §19.5);
- [ ] Memory-Auto-Sizing (Slice 2);
- [x] transaktionaler vollständiger CPU-Neustart der Phase bei CUDA-Fehler
  (`forward_drizzle_cuda` + `ForwardDrizzleCudaError`;
  `persist_multiband_store_from_predecessors` verwirft die uncommittete
  Generation und startet den gesamten Build auf dem CPU-Referenzpfad neu;
  committetes Ergebnis bit-identisch zum reinen CPU-Build; Fault-Injection-
  Test ohne GPU, 30.44);
- [ ] detailliertes Timing (Slice 2).

**Abnahme:**

- native GPU-Tests auf tatsächlicher CUDA-Hardware;
- CPU-/CUDA-Parität für Droplet, Clipping und Profile; À-trous bleibt CPU;
- Timing weist Phasenanteile inklusive Transfer- und Store-I/O separat aus;
- kein laufender Test-/Runnerprozess bleibt zurück.

### M8 — GUI, Report und vollständige Dokumentation

**Änderungen:**

- neue Konfigurationsfelder in der aktiven v3-Oberfläche;
- deutsche und englische Texte;
- Run-Report für Coverage, `n_eff`, Alpha und Candidate Gates;
- aktive Methodik- und Prozessdokumentation;
- Resume-Abhängigkeiten;
- historische Classic-/PREWARP-Runs read-only anzeigen und Resume deaktivieren;
- OSC-/MONO-Umfang, RGB-Ablehnung, Coveragefehler und fehlende
  Rekonstruktions-Caches verständlich anzeigen;
- `keep_profile_cache_after_run`, dauerhaften Source-Cache-Default `false` und
  explizite Cachebereinigung samt Resume-Warnung integrieren;
- alle Methodenwahlfelder und Legacy-Konfigurationsvorschläge entfernen.

**Abnahme:**

- Desktop-/Mobile-Prüfung der Frontendfelder einschließlich Cachelöschwarnung;
- `summary`/`full` und Profilcache-Retention verändern keine Rechenergebnisse;
- JSON/YAML valide;
- deutsche/englische Dokumentation konsistent.

### M9 — Kontrollierte Qualitätsläufe

Dieser Meilenstein benötigt eine ausdrückliche Benutzeranforderung zum Starten
von Bildverarbeitungsläufen.

Reihenfolge:

1. kleine synthetische/Fixture-Ausführung;
2. maximal 100 reale Frames für Speicher, Runtime und erste gematchte Metriken;
3. vollständiger Datensatz erst nach bestandenen 100-Frame-Gates;
4. vollständige Pflichtmatrix vor dem Single-Method-Release.

Pflichtmatrix (ein Datensatz darf mehrere Zeilen abdecken, jede Zeile benötigt
aber ein eigenes protokolliertes Ergebnis):

| Klasse | Mindestzweck | Erwartetes Ergebnis |
|---|---|---|
| synthetisch OSC, alle Bayer-Pattern | bekannte PSF, Flux, Dither, affine und lokale Warps | alle numerischen Gates anwendbar und bestanden |
| synthetisch MONO | Ein-Ebenen-Kern und mode-spezifische Outputs | alle MONO-Verträge bestanden |
| kritisch/unterabgetastetes Sternfeld | native FWHM nahe 2 px, gute Ditherabdeckung | Schärfe-/Flux-/Farbpromotion bestanden |
| übersampeltes Sternfeld | 1x gegen 2x, kein künstlicher 2x-Vorteil | korrekte Scale-Entscheidung und keine Regression |
| strukturreiches Nebel-/Galaxiefeld | Hintergrund, schwache Strukturen, Farbe | RMS-, Seam-, Flux- und Farbgates bestanden |
| starke Feldrotation/lokale Korrektur | Warp, Halo, Zentroid, Laufzeit | Geometrie- und Ressourcenverträge bestanden |
| niedrige Framezahl/kleiner Dither | Coverage-Grenzfall | entweder alle Qualitätsgates bestanden oder erwarteter fail-closed Coveragefehler; keine stille Qualitätsdegradation |
| realer MONO-/Schmalbanddatensatz | Produktumfang MONO | Runtime-, Speicher- und Downstreamverträge bestanden |

Für die reale Interpolations-/Seeing-Bisektion sind zusätzlich zu M16
verbindlich vorgesehen:

- M66 als sauberes, weitgehend isotropes Stern-/Galaxiefeld;
- IC5070 als anisotrope Feldrotations-/Driftklasse;
- ein realer MONO-/Schmalbanddatensatz.

Je Datensatz werden native CFA-/MONO-Samples, debayerte ungewarpte Stufe,
PREWARP, altes Raw-AQMH, neues Uniform, neues Raw und Multiband an denselben
Sternpositionen verglichen. Effekte werden mit Bootstrap-Konfidenzintervallen
berichtet. Es gibt bewusst kein pauschales Gate „mindestens 0,5 px pro Objekt“,
weil der physikalische Seeinganteil objekt- und aufnahmeabhängig ist. Verbindlich
sind die relativen Promotionsgates aus 3.2/15.3 und bei synthetischen Fixtures
die bekannte Ground Truth.

Für den 10-%-Vergleich gegen PREWARP-AQMH werden Alt- und Neupfad aus exakt
demselben geordneten Frame-Manifest, derselben Frameauswahl, Normalisierung,
Cropfläche und `output_scale` erzeugt. Da der Altpfad keine Benutzer-Runs
schreiben darf, stammt die Referenz entweder aus vor M0 eingefrorenen,
gehashten Outputs oder aus dem explizit angeforderten Legacy-Testharness in
einem frischen isolierten Vergleichsverzeichnis außerhalb von `runs/`.
Commit-/Build-ID, Config und Outputchecksummen werden protokolliert. Wo der
Altpfad nur 1x ausgeben kann, erfolgt der primäre Altvergleich mit
`internal_scale=2, output_scale=1`; ein bloß hochskaliertes 1x-Altbild ist keine
zulässige 2x-Referenz.

Keine Variante darf ausschließlich nach visueller Schärfe oder finaler FWHM
bewertet werden. Verwendet werden effektive Config, Phase-Events, Coverage,
Q-Maps, Candidate Gates und gematchte Sternpositionen. Ein erwarteter
Coveragefehler zählt nur für die negative Grenzfallzeile als bestanden, nicht
als Ersatz für positive Qualitätsnachweise.

**Abnahme:**

- sämtliche Promotionskriterien aus Abschnitt 3.2 auf der vereinbarten
  Datensatzmatrix bestanden;
- RSS- und Temporärdiskformeln aus 11.11 eingehalten; Durchsatzmedian höchstens
  20 % unter der eingefrorenen M8-Baseline;
- M66-, IC5070- und MONO-Bisektion mit Bootstrap-Konfidenzintervallen
  dokumentiert;
- keine schwere Regression in Runtime, Speicher, Farbe, Astrometrie oder
  Downstream-Kompatibilität;
- dokumentierter Go/No-Go-Entscheid für den endgültigen Cutover.

### M10 — Endgültiger Produkt-Cutover und Legacy-Isolation

M10 wird nur nach bestandenem M9-Go ausgeführt. Ein No-Go führt zur Korrektur
der neuen Methodik, nicht zur dauerhaften Reaktivierung der alten Methoden im
Produkt. „Endgültig“ bezieht sich auf Produktpfad und öffentliche Verträge; die
physische Löschung der ausschließlich test-only verbliebenen Referenzquellen
erfolgt widerspruchsfrei erst in M11.

**Änderungen:**

- Classic- und PREWARP-AQMH-Quellen aus allen Produkt-Targets entfernen; sie
  verbleiben ausschließlich im standardmäßig deaktivierten Target
  `tile_compile_legacy_reference_tests` (25.11);
- Nutzsignal-PREWARP-Cache-Schreibpfade, alte Q-Map-Canvaspfade und
  Legacy-Kandidaten aus dem Produkt löschen;
- alte Methodenfixtures als statische, methodenunabhängige Goldwerte
  konservieren, sofern Paritätstests sie weiterhin benötigen;
- PREWARP-/DEBAYER-Scheinphasen aus aktiver Phase-ID-Liste, Resume und UI
  entfernen;
- `delete_source_cache_after_run` bleibt standardmäßig `false`; eine explizite
  Cachebereinigung wird in GUI/CLI angeboten und weist auf den Verlust des
  Rekonstruktions-Resume hin;
- verbliebene **Rekonstruktions**-Engine-/Methodenschlüssel aus Schema,
  Beispielen, Übersetzungen, Reports und Dokumentation entfernen;
  `registration.engine` und der allgemeine Acceleration-Backendvertrag bleiben
  davon ausdrücklich unberührt;
- historische Methodikdokumente unverändert nach `attic/` verschieben, sofern
  sie noch in aktiven Dokumentationsbereichen liegen;
- read-only Legacy-Run-Parser als getrennte, nicht schreibende Komponente
  absichern.

**Abnahme:**

- `rg`-Inventar findet keine aktive Classic-, PREWARP-AQMH-,
  **Rekonstruktions**-Engine-Branch- oder `prewarped_frames`-Referenz außerhalb
  des erlaubten read-only Legacy-Parsers und historischer Dokumente;
  gleichnamige Registration-/Acceleration-Begriffe werden über Pfad und Typ
  bewusst vom Löschinventar ausgeschlossen;
- Produkt-Build linkt keine alte Rekonstruktionsquelle;
- CLI/API/GUI bieten exakt eine Rekonstruktionsmethodik an und zeigen dafür
  keinen Auswahlmechanismus;
- neue Runs und alle zulässigen Resume-Einstiege verwenden ausschließlich den
  Pipelinevertrag der neuen Methodik;
- Legacy-Runs bleiben sichtbar, aber jeder Resumeversuch wird vor Mutation
  reproduzierbar abgelehnt;
- vollständige CPU-, CUDA-, Runner-, Backend-, Frontend-, Schema- und
  Dokumentationsprüfungen bestehen.

### M11 — Endgültige Löschung des Legacy-Referenztargets

Ein Release-Zyklus nach M10 (25.11): `tile_compile_legacy_reference_tests`
samt exklusiv dafür benötigten Quellen, Fixtures und CMake-Optionen entfernen.
Abnahme: `rg`-Inventar findet keine Classic-/PREWARP-AQMH-Rekonstruktionsquelle
mehr im Repository außerhalb von `attic/`; read-only Legacy-Run-Parser bleibt.

---

## 24. Build- und Prüfstrategie

Nach jedem Core-Meilenstein:

```bash
cmake -S . -B build -DBUILD_TESTS=ON
cmake --build build --target tile_compile_runner tests -j2
./build/tests "relevanter Filter"
./build/tests
```

Zusätzlich:

- `tile_compile_runner` bauen, weil Runnerphasen geändert werden;
- JSON- und YAML-Schemas validieren;
- CPU-Sanitizer-/Bounds-Tests für neue Rasterisierung;
- CUDA-Tests nur mit tatsächlichem GPU-Zugriff;
- Frontendänderungen ohne Start eines zusätzlichen Backends prüfen;
- bestehende Services nur verwenden, nicht neu starten.

Alle Terminalausgaben folgen der Repository-Regel und werden nach
`/tmp/out_*.txt` umgeleitet und separat gelesen.

---

## 25. Risiken und verbindliche Gegenmaßnahmen

### 25.1 Speicherexplosion bei 2x

**Risiko:** vierfache Pixelzahl plus mehrere Profile.  
**Maßnahmen:** Zielzeilen-Chunking, sequenzielle Kanal-/Profilverarbeitung,
transaktionale disk-/mmap-gestützte U/R/F/M-Profilstores, streifenweise
À-trous-Fusion mit kumulativem Halo, frühes Freigeben diagnostischer
Vollbilder, getrenntes Host-/Device-/Temporärdisk-Budget, keine gleichzeitige
Vollbildresidenz aller Profile und keine Vollbild-Frame-Matrix für alle Frames.
Vor M9 werden zusätzlich benötigter freier Temporärspeicher und Verhalten bei
Erschöpfung fail-closed geprüft.

### 25.2 Sparse CFA-Coverage (zentrale Sampling-Randbedingung, kein Randfall)

**Risiko:** Bei CFA-Forward-Drizzle auf 2x deckt ein R- oder B-Sample pro Frame
nur einen kleinen Flächenanteil des Zielrasters ab. Bei `pixfrac = 0,8` liegt
die Einzel-Frame-Flächenüberdeckung für R und B bei rund **16 %** (Droplet
`0,8 × 0,8` Quellpixel je 2×2-Bayer-Block; G liegt auf dem Quincunx und
erreicht ~32 %). Nominell einfache Coverage
erfordert damit ~6+ gut geditherte Frames, gleichmäßige Uniform-Gewichtung
deutlich mehr. Bei kleinen Framezahlen, geringem Dither oder starker
Feldrotation entstehen systematische Löcher und Gewichtsschwankungen in R/B →
Farbsäume, Kammartefakte, kanalabhängige FWHM.

**Maßnahmen:**

- **Hartes direktes `coverage_gate`** (Abschnitte 6.2, 9.5): gültiger
  Kanalanteil mindestens 0,995, p10-`n_eff` mindestens
  `max(3.0, 0.15 * N)`, mindestens 1024 Analysepixel und keine interne
  ungestützte Kanalinsel. Unterschreitung bricht den Run in
  `SAMPLING_GEOMETRY` fail-closed mit Kanal und Ist-Wert ab.
- Zirkuläre Dither-Streuung bleibt eine Diagnose, kein Gate; direkt
  rasterisierte Coverage ist bei Rotation und lokalen Warps maßgeblich.
- Die synthetischen Fixtures decken Framezahlen an und unter den effektiven
  Coverage-/`n_eff`-Grenzen ab.
- kanalspezifischer Support und Mindest-`n_eff` in der Rekonstruktion;
- keine erfundenen Farbwerte; Alpha-Fallback auf Raw bzw. Uniform;
- ein gemeinsames `pixfrac=0.8` bleibt für alle Kanäle verbindlich. Bei
  unzureichender Coverage werden explizit global `pixfrac=1.0` oder
  `internal_scale=1` geprüft. Per-Kanal-`pixfrac` ist im ersten Release und bis
  nach M10 ausgeschlossen, weil es kanalabhängige PSFs erzeugen würde.

### 25.3 Registrierungsrichtung oder Halbpixelversatz

**Risiko:** systematische Unschärfe oder Farbsäume trotz korrekter Matrixwerte.  
**Maßnahmen:** eine einzige Pixelzentrumkonvention, Round-trip-Tests,
synthetische Zentroidtests, explizite Warp-Konvention im Artefakt.

### 25.4 Lokales Warpmodell nicht direkt invertierbar

**Risiko:** einzelne Samples/Frames können nicht forward gemappt werden.  
**Maßnahmen:** beschränkte Iteration, Jacobian-Voraussetzung,
Framefehlergrenze, kein stiller global-affiner Ersatz.

### 25.5 Detailprofil verstärkt Rauschen

**Risiko:** Q-Map interpretiert Noise als Schärfe.  
**Maßnahmen:** SNR-/Artefaktkomponente, Q-Separationsgate, `n_eff`,
Bandenergieguard, niedrige Frequenzen ausschließlich aus Uniform.

### 25.6 Bandseams an Supportgrenzen

**Risiko:** normaler Wavelet-Mix faltet Nullen in gültige Bereiche.  
**Maßnahmen:** maskierte normalisierte Faltung, Supporttaper, Alpha→0 am Rand,
separate Band-Seam-Metriken.

### 25.7 Farbsäume

**Risiko:** unterschiedliche Kanalgewichte erzeugen unterschiedliche
Detailpositionen.  
**Maßnahmen:** gemeinsames Alpha, Farbzuordnung in Source-CFA-Koordinaten,
kanalspezifischer Support, keine unabhängigen per-channel Kandidatenmasken.

### 25.8 Vergleich mit unterschiedlichem Pixelmaßstab

**Risiko:** 2x-FWHM wird fälschlich direkt mit nativer FWHM verglichen.  
**Maßnahmen:** Metriken zusätzlich in nativen Pixeln und Winkelmaß ausgeben;
Reports deklarieren `output_scale`; Gate-Code normalisiert Maßeinheiten.

### 25.9 Resume verwendet veraltete Caches

**Risiko:** technisch erfolgreicher, aber semantisch falscher Lauf.  
**Maßnahmen:** getrennte Hashdomänen für Registration, Normalisierung, Q-Maps,
Drizzle und Mehrband; fail-closed bei Mismatch.

### 25.10 GPU ändert Semantik

**Risiko:** andere Atomics-/Sortierreihenfolge verschiebt Clippingentscheidungen.  
**Maßnahmen:** CPU als Referenz, feste Grenzfalltests, fp32-Q-Option,
gleiche Tie-Break-Regel, dokumentierte Entscheidungstoleranz.

### 25.11 Untestbare Datenklasse regrediert nach der M10-Produktentfernung

**Risiko:** M9 validiert auf einer endlichen Objektmatrix. Nach der Entfernung
von Classic und PREWARP-AQMH aus dem Produkt in M10 gibt es keinen
Produkt-Fallback, obwohl die test-only Quellen erst in M11 physisch gelöscht
werden. Eine reale, in M9 nicht abgedeckte Datenklasse (z. B. stark übersampelt,
Monochrom-Schmalband, extrem kurze Serie) kann mit dem neuen Pfad dennoch
schlechter laufen.

**Maßnahmen:**

- Die M9-Matrix wird verbindlich um die in Abschnitt 3.2 genannten Grenzfälle
  erweitert (untersampelt, Monochrom, kleiner Dither / niedrige Framezahl,
  starke Feldrotation).
- Nach M10 bleibt der alte Pfad **einen Release-Zyklus lang** als separates,
  nicht beworbenes Build-Target (`tile_compile_legacy_reference_tests`,
  Abschnitt 1.1) erhalten — nicht installiert, nicht vom Backend aufgerufen,
  ohne Schreib-/Resume-Zugriff auf Benutzer-Runs, standardmäßig nicht
  konfiguriert, aber baubar. Erst der auf M10 folgende Release löscht die
  Quellen endgültig (M11).
- Ein nach M10 gemeldeter Regressionsfall dieser Art führt zur Korrektur des
  neuen Pfads; die Reaktivierung des alten Pfads im Produkt bleibt
  ausgeschlossen. Das test-only Target dient nur der reproduzierbaren
  Ursachenanalyse.

---

## 26. Verbindlich entschiedene Punkte und verbleibende Nachweise

Die algorithmischen Entscheidungen sind mit der Revision 2026-09-02
festgeschrieben. Änderungen benötigen eine neue Planrevision, angepasste
Hash-/Artefaktversionen und erneut bestandene Fixtures. Offen bleiben nur
empirische Nachweise, die ohne Implementierung beziehungsweise ausdrücklich
autorisierte M9-Läufe nicht vorweggenommen werden können.

| Thema | Verbindliche Festlegung | Verbleibender Nachweis |
|---|---|---|
| Produktumfang | OSC und MONO unterstützt; bereits debayertes RGB fail-closed abgelehnt | Scan-/Contracttests in M0/M1 |
| Coverage-Gate | direkter Kanalsupport ≥0,995; p10-`n_eff >= max(3.0, 0.15*N)`; ≥1024 Analysepixel; keine interne Kanalinsel; Dither nur Diagnose | synthetische Grenzfixtures in M1 |
| lokale Droplet-Subdivision | 0,05 internes Pixel, relative Flächenkonvergenz 0,5 %, Tiefe 2, Framefehlergrenze 0,1 % | Flächen-/Zentroidfixtures in M2 |
| Registrierungsfaktoren | bestehende Residualfunktion mit konservativem Missing-Floor 0,55; Modellfaktor `clamp(1/(1+0.4*depth),0.5,0.9)`, nearest-copy höchstens 0,5 | M3-Fixtures und Persistenztest |
| Source-Proxy | `proxy_version=1`: gleichfarbiges Quad-Green, Green-Highpass/MAD; MONO direkt; keine R/G/B-Quad-Rauschstatistik | CFA-Farb-/Checkerboard-/MONO-Tests in M3/M5 |
| Scale-Modi | kein Auto; `1/1`, `2/1`, `2/2`; Produktionsdefault `2/1`, M4-/M9-Schärfenachweis `2/2` | Coverage-, Downstream- und Ressourcenmatrix |
| 2x→1x | 4/4-Subpixelsupport, festes Flächenmittel, `n_eff_out=min(subpixels)` | WCS-/Flux-/Maskentests in M4 |
| `pixfrac` | ein gemeinsamer Wert für alle Kanäle, Default 0,8; global 1,0 nur als explizite Konfiguration; kein per-Kanal-Wert bis nach M10 | Coveragevergleich 0,8/1,0 in M9 |
| Alpha-Vertrauen | `A_artifact` aus geometrischem weighted-p10; `A_registration` aus direktem Anteil und residual-p20; gemeinsame OSC-Kanalminima | deterministische M6-Fixtures |
| Alpha-Glättung | maskierte B3-Glättung innerhalb einer Supportkomponente und `alpha_final=min(alpha_guarded,alpha_blur)` | Null-/Kanten-/Seamtests in M6 |
| Bandenergieguard | Luma-/MONO-MAD, bandabhängiges Fenster, Grenze 1,30, sechs Bisektionsschritte; keine Sternkonzentrationsausnahme | Rausch-/Stern-/Farbsäumefixture in M6 |
| Validation/N/A | FWHM ab 20, p90/Tail/Elongation ab 30 Sternen; FWHM-CI-Breite ≤10 %; N/A erfüllt nie Promotion | M6-Validationtests und M9-Anwendbarkeit |
| À-trous-Backend | CPU in M7; CUDA erst später bei ≥20 % Phasenanteil und ≥15 % End-to-End-Gewinn | CPU-Parität und Profiling |
| Kontrollpersistenz | Raw/Selected/Support/Validation immer; Kontroll-FITS nur `full`; interne Profile nur bei `keep_profile_cache_after_run=true` | Cache-/Resume-/Checksummentests in M8 |
| Ressourcen | RSS-Wachstum ≤Budget×1,05+256 MiB; Temp-Preflight ≥Schätzung×1,20+Reserve; Durchsatzregression höchstens 20 % auf Referenzhardware | M8-Baseline und M9-100-Frame-Bestätigung |
| Source-Cache | `delete_source_cache_after_run=false` bleibt Produktionsdefault; Löschung nur explizit nach vollständigem Commit | GUI-/CLI-/Resumeprüfung in M8/M10 |
| Interpolations-/Seeing-Aufschlüsselung | M16 plus M66, IC5070 und realer MONO-Datensatz; Bootstrap-CIs, kein fixer 0,5-px-Zwang | autorisierte M9-Läufe |
| Legacy-Grace-Zyklus | `tile_compile_legacy_reference_tests` ab M0 isoliert, in M10 nur test-only, Löschung in M11 | Buildinventar M0/M10/M11 |
| Single-Method-Cutover | nur nach vollständiger M9-Pflichtmatrix und allen Gates | dokumentiertes M9-Go |

Parameterwerte dürfen nicht ausschließlich anhand eines einzelnen M31-Laufs
objektspezifisch festgelegt werden.

---

## 27. Definition of Done

Die Implementierung gilt erst als abgeschlossen, wenn alle folgenden Punkte
erfüllt sind:

- aktive Pipeline verwendet keine vorgewarpten Nutzsignalframes;
- CFA-Samples werden unter Berücksichtigung von CFA-Ursprung und
  Sensororientierung source-space farbkorrekt forward projiziert; MONO verwendet
  ausschließlich L und bereits debayertes RGB wird fail-closed abgelehnt;
- RegistrationSamplingPlan ist mit stabilen Frame-IDs, Modellkoordinaten,
  kanonischem Hash und verlustfreier Serialisierung resumierbar;
- `sampling_geometry.json`, direkte Support-/`n_eff`-/Loch-Gates und exakte
  Supportmasken sind konsistent; Dither ist nur Diagnose und es besteht kein
  Gate-/Phasenzirkelschluss;
- Uniform, Raw und Mehrband verwenden identische Samples und Clippingmasken;
- Raw-Forward-Drizzle ist mit Outputchecksummen atomar und unveränderlich
  persistiert; Teiloutputs werden nie als gültig erkannt;
- Mehrband nutzt Uniform für niedrige Frequenzen und gegatete Profile für
  Details;
- alle Kandidaten verwenden denselben gematchten Sternsatz; 20/30-
  Mindeststernzahlen und Bootstrap-CI gelten, N/A kann keine positive Promotion
  erfüllen;
- CPU-Referenz und CUDA-Pfad bestehen Paritäts- und vollständige
  Phasen-Neustart-Fallbacktests;
- 1x- und 2x-WCS/Crop/Masken, Oberflächenhelligkeit und physischer Aperturflux
  sind korrekt;
- U/R/F/M-Rekonstruktion und Mehrbandfusion halten RAM-/Temporärdiskbudgets ein,
  ohne alle Vollbildprofile gleichzeitig resident zu halten;
- Resume lehnt unvollständige, umsortierte, checksumdefekte oder inkompatible
  Caches ab und verwendet die getrennten Hashdomänen korrekt;
- Source-Caches bleiben standardmäßig erhalten; Profilcache- und
  Diagnostikpersistenz folgen den getrennten Verträgen aus 6.2/16.2;
- Classic Tile Compile und PREWARP-AQMH sind aus allen Produkt-Targets,
  Konfigurationsoberflächen und aktiven Verträgen entfernt;
- historische Runs sind ausschließlich read-only sichtbar und werden vom
  Resume fail-closed abgelehnt;
- C++-Tests, Runner-Build, JSON/YAML-Validierung und relevante Frontendchecks
  bestehen;
- aktive deutsche und englische Dokumentation, Schemas und Beispiele stimmen
  überein;
- kein Backend und kein Bildverarbeitungslauf wurde ohne ausdrückliche
  Benutzeranforderung gestartet;
- die kontrollierte Qualitätsmatrix erreicht die Promotionskriterien;
- das Löschinventar aus M10 enthält keine unerlaubte aktive Legacy-Referenz.

---

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

## 30. Revisionsnotiz und Implementierungsaudit (2026-09-02)

### 30.1 Tatsächlicher Implementierungsstand

Die ursprüngliche Prüfung (2026-09-02) ergab keinen begonnenen Meilenstein;
seither sind M0–M7 begonnen. Die Tabelle ist am **2026-09-06** auf den
tatsächlichen Stand nachgezogen (30.45/30.46).

| Meilenstein | Status (2026-09-06) | Begründung |
|---|---|---|
| M0 | **~90 %** | erledigt & verifiziert (siehe 30.4): Datenmodell Abschnitt 7; `pipeline_contract_version` + Resume-Guard; `tile_compile_legacy_reference`(-`_tests`)-Targets; `PIPELINE_UNAVAILABLE_DURING_CUTOVER`; Config-Migration (Modul + `cli migrate-config` + Runner-Verdrahtung + `config_migration.json`); `--force-classic`/`FORCE_CLASSIC` entfernt; OSC/MONO/RGB-Eingabeklassen-Policy; **`reconstruction:`-Vertrag §6.1–6.3 inkl. `schema.json`/`schema.yaml`**; Backend-M0-Anteil (§17.5). **+22 M0-Tests; Hauptsuite 293/294 (1 vorbestehender GPU-Fehlschlag).** Offen: `tile_compile.yaml`/`examples/`-Fläche; `aqmh`→`reconstruction`-Restrukturierung (an M2/M3/M6 gekoppelt); tiefer §17.5-Anteil (M8) |
| M1 | **abgeschlossen** | `RegistrationSamplingPlan` real befüllt (30.4); `SAMPLING_GEOMETRY`-Phase mit **scharf fail-closed** `coverage_gate` inkl. Lochprüfung, auf echten Daten verifiziert (inkl. eines realen 3-Px-Lochfunds), Legacy-Pfad nachweislich getrennt/unverändert (30.6–30.9); beide geometrischen Masken als FITS persistiert, `analysis_common_mask`-Default empirisch gegengerechnet und in §9.3 dokumentiert (30.10); zirkuläre Ditherdiagnostik implementiert und mathematisch exakt verifiziert (30.11). Der zuvor zurückgestellte Punkt (COMMON_OVERLAP aus den Geometriemasken speisen) ist mit der M3-Integration erledigt: `COMMON_OVERLAP` ist eine eigene Phase im `reconstruct`-Runner, gespeist aus `sampling_geometry_*`-Masken, vom Legacy-Pfad getrennt (30.23, real auf M31). Kein offener M1-Punkt |
| M2 | **korrektheits-vollständig; nur noch Parallelisierung offen (perf, nicht blockierend)** | Kerngeometrie/-mathematik implementiert und real gegen den Audit vom 2026-09-05 verifiziert (§0/§0.1): gemeinsamer Rasterisierer mit Coverage, echtes flächengewichtetes `n_eff`, dichte CFA-unabhängige Analysefläche, echte Tiefe-2-Subdivision mit beidseitiger Konvergenzprüfung auch an Maximaltiefe, Pixelzentrum-Adapter, zeilenweises Speicher-budgetiertes Streaming (`stream_forward_drizzle_uniform`). Diagnose-Preview am Runner, auf echten M31-Daten verifiziert (30.13). Atomizitätsloch in `write_fits_float()` gefunden und behoben (30.14). **Die zuvor hier als offen geführten Punkte sind durch die M3-Integration (30.23) geschlossen:** store-weite Transaktionalität über **alle** Ebenendateien (`StoreWriter` schreibt alle Planes, committet ausschließlich über `current.json`, verwirft eine nicht publizierte Generation im Destruktor — bit-exakt in `[drizzle-store]` getestet, u. a. der M7-30.44-Neustarttest); eigene Pipeline-Phase `Phase::SAMPLING_GEOMETRY` mit fail-closed Resume-Vertrag existiert und ist im `reconstruct`-Runner + `[forward-runner]`-Tests verdrahtet. **Verbleibend:** Parallelisierung des Drizzle-Rasterisierers (der Streaming-Pfad ist bewusst single-threaded, `parallel_workers=1` im `forward_drizzle_only`-Pfad) — reine Performance, keine Korrektheits- oder Vertragslücke; sinnvoll gebündelt mit M7-Slice-2 (CUDA) |
| M3 | **weitgehend abgeschlossen (Code + Integration + M31-Verifikation); Restpunkte nicht blockierend** | der Clipping-Algorithmus (§11.8) implementiert und mit von Hand nachgerechneten Erwartungswerten getestet (30.15); **gemeinsame Akzeptanzmaske für Uniform und Raw im selben Durchlauf verdrahtet** (`compute_forward_drizzle_uniform_and_raw()`, 30.16) — Raw ist bewusst noch numerisch identisch mit Uniform, da `G_eff`/`Q_composite` bis zur `GLOBAL_QUALITY`/Q-Map-Infrastruktur auf `1.0` stehen; reaktives Speicherbudget-Sicherheitsnetz für die Kandidatenliste ergänzt und getestet (30.17, kein A-priori-Bound); **CFA-aware Analyseproxy `proxy_version=1`** (Quad-Grün, exakte MAD-Sigma, Edge-aware-Vollproxy, B3-Unschärfe) implementiert und mit exakt nachgerechneten Erwartungswerten getestet (30.18); **`G_quality(f)`** aus dem Proxy implementiert und real auf 4 M31-Frames verifiziert, dabei einen echten `[0,1]`-Vertragskonflikt der Bestandsformel gefunden und mit minimalem Sigmoid-Fix geschlossen (30.19); **`QualityFrameWeightPlan`** (`G_eff`-Berechnung genau einmal, Registrierungsfaktoren wörtlich aus dem Sampling-Plan, kanonischer Hash, fail-closed-Loader inkl. `g_eff`-Produktprüfung) implementiert und getestet (30.20); **`G_eff(f)` in `compute_forward_drizzle_uniform_and_raw()` verdrahtet** (30.21) — Raw weicht jetzt bei übergebenem `g_eff`-Vektor echt von Uniform ab (geteilte Clipping-Maske, danach Pro-Frame-`G_eff`-Faktor), ohne Vektor rückwärtskompatibel bitidentisch (30.21); **checksummierter Profilstore-Manifest** mit fail-closed-Verifikation (`ProfileStoreManifest`/`verify_profile_store()`), real auf M31 verifiziert (30.22). **Durchgehend integriert (30.23):** neue `Phase`-Enum-Werte (`NORMALIZED_CACHE`/`SAMPLING_GEOMETRY`/`GLOBAL_QUALITY`/`FORWARD_DRIZZLE`), `VerifiedNormalizedSourceCache`, transaktionaler `DrizzleProfileStore` (unveränderliche Generationen + `current.json`), `source_quality_artifact` mit Vorgängerbindung, `apps/runner_forward_drizzle` + CLI-Subcommand `reconstruct`/`resume-reconstruction`. **Real auf M31 end-to-end verifiziert**: alle vier neuen Phasen laufen bis `reconstruction_ready`, `source_quality_plan.json` + `forward_drizzle_profiles/generation-*/` (24 FITS-Ebenen uniform+raw) real geschrieben, Raw ≠ Uniform durch echtes `G_eff`, Support-Maske bitidentisch, Coverage-Gate scheitert fail-closed bei zu wenigen Frames. **Nachgezogen 2026-09-06 (30.45):** frame-lokaler `Q_composite`-Stream **erledigt** in M5 (§13.4, 30.31); Zero-Veto-Maskenweiterleitung **erledigt** in M5 (separater `uint8` Hard-Veto-Stream, 30.30); „automatischer Uniform-Fallback im Resume-Pfad" ist **kein Defizit** — der Plan verlangt fail-closed ohne stillen Fallback (§4.4/§9.5), der *qualitative* Uniform-Rückfall ist §15 (`select_reconstruction_candidate` → `kDrizzleUniform`, 30.42). **Wirklich verbleibend, nicht blockierend:** A-priori-Streifengrößen-Schranke für die Kandidatenliste (reaktives Netz existiert, 30.17), und der **Cache-Lebensdauervertrag** (`keep_profile_cache_after_run` / `delete_source_cache_after_run` werden geparst, aber noch nicht konsumiert — Löschung/Behalt nach erfolgreichem Phasencommit ist noch nicht verdrahtet) |
| M4 | **Algorithmik + Store-Integration fertig; Rest ist Cutover-Arbeit** | §12-Kernoperatoren implementiert und mit Handrechnung getestet (30.24): 2×2→1x-Flächenmittel mit striktem 4/4-Vertrag, `scale_wcs_to_output()` (§12.2 wörtlich), explizite Modi `1/1`/`2/1`/`2/2` ohne Auto, Kernel-Autokorrelations-Faktor `f = W/√S0` (§12.4) hergeleitet; speicher-begrenzter `2/1`-Streaming-Downsample bit-identisch zur Referenz für jede Chunkhöhe (30.25); **`2/1` im transaktionalen Store aktiviert** — `output_scale` im `reconstruction_hash`, halbierte Store-Geometrie bei `2/1`, `phase_end` meldet `output_scale_applied` + `f` (30.26), Bit-Identität zur Referenz getestet. **Nachgezogen 2026-09-06:** der `reconstruct`-Einstieg endet nicht mehr bei `reconstruction_ready` — `MULTIBAND` liefert das finale `X_out` (30.41/30.43). Verbleibend ist **reine Cutover-Arbeit, die der Plan selbst nach M6/M10 legt** (Ausgabe-Pipeline-Verdrahtung): BGE/PCC/HMS auf 2×-Geometrie, `forward_drizzle.json`-Ausweisung des `f`-Faktors im Report, per-Kanal-Sparse-Verfeinerung von `f`. Kein M4-Algorithmik- oder Vertragsdefizit |
| M5 | **abgeschlossen — Code + synthetisch + M31-Echtdaten** (30.45) | `metrics::compute_aqmh_quality_map()` bekam einen per Default null-Per-Scale-Hook (Legacy byte-identisch, 15 Fälle grün); `reconstruction/source_quality_maps` exponiert die **wörtlich wiederverwendete** Legacy-`psi`-Mathematik als skalenspezifische Karten in Source-Geometrie + `artifact_confidence` + geometrische-Mittel-`q_map`; `QualityScaleMapSink` hält **nachweislich nur eine** Vollkarte resident (§13.3); **MONO-Bit-Identität** des Composite zur Legacy-Q-Map (30.27, 5 Fälle). `reconstruction/source_quality_map_cache`: `uint16`-Quantisierung mit reserviertem Null-Veto-Sentinel (0/NaN⇒0, kein Nicht-Null-Code⇒0), räumlicher `storage_divisor`, **getrennter Wert-Stream (Valid-Mean über positive Quellpixel) + expliziter `uint8`-Hard-Veto-Stream** (jedes exakte `Q=0` ⇒ Lesen erzwingt `NaN`, gute Nachbardaten bleiben; §13.5 erfüllt ohne Dezimierung, 30.30), Multistream-Layout mit atomarem `metadata.json`-Commit, fail-closed-Reader (Manifest-Hash + je Datei-`sha256`), `read_region`; **getrennter** `source_quality_identity_hash` (ignoriert Registrierung/Canvas/Scale + den config-gebundenen Plan-Hash) und `scale_quality_config_hash` (30.28, 4 Fälle). `build_source_quality_map_cache()`-Orchestrator + neue Phase `SOURCE_QUALITY_MAPS = 28` (angehängt) **vor** `GLOBAL_QUALITY` im `reconstruct`-Runner, Resume prüft den Cache fail-closed; synthetisch end-to-end über die `[forward-runner]`-Fixture + Orchestrator-Unit-Test verifiziert (30.29). **`Q_composite`-Konsum** im Rekonstruktor: `SourceQualityProvider` + `QA`-Streifenakkumulator ⇒ geometrisch `K`-gemitteltes `Q_composite_f,c(q)` (NaN/`≤0` ⇒ 0), `w_raw = B·G_eff·Q_composite`, `ClipCandidate.q` (von `apply_robust_clipping` **nicht** gelesen), Weiterleitung durch alle Persist-/Stream-Ebenen, fail-closed-Reader in `persist_forward_drizzle_from_predecessors`, `source_quality_cache_hash` bedingt im `reconstruction_hash`; synthetisch nachgerechnet (15→12 bei Q 1,0/0,25; 15→10 bei Q=0 **ohne** Pixel-Veto; NaN⇒0) (30.31). **M31-Echtdaten-Verifikation erfüllt (30.45):** realer 40-Frame-`reconstruct`-Lauf, `SOURCE_QUALITY_MAPS` schrieb `composite`+`scale_0..3`+`artifact` (`computed_scales=4`), die drei Kandidaten unterscheiden sich real (`background_rms` 2,210/2,405/2,409 am selben Sternset ⇒ `G_eff·Q_composite` verschiebt Raw messbar; deckt sich mit `mean|u−r| ≈ 2,1–2,6` aus 30.23) |
| M6 | **In-Memory-Referenzpipeline + Streifenpfad vollständig und end-to-end getestet: `reconstruct_multiband_reference` (À-trous §14.2 + Fine/Medium + Bandblend §14.3 + adaptives Alpha `A_neff`/`A_coverage`/`A_separation`/`A_artifact`/`A_registration` §14.4 im Drizzle verdrahtet + Energieguard §14.5 + B3-Glättung §14.7); `fuse_multiband_streamed` bit-identisch zur Vollbildreferenz für jede Chunkhöhe (30.39). `A_artifact`-Schwelle zählt echte Artefaktdaten (`qa_has_data`, 30.38). Mehrband-Profilstore `uniform_raw_multiband_clipped` (Fine/Medium + 4 Confidence-Pseudoebenen, `multiband_config_hash` §16.4) + Runner-`MULTIBAND`-Phase → `reconstruction_multiband.fits`, `final_image_available` **für alle Output-Scales** `1/1`/`2/1`/`2/2` (30.41; `2/1`: Fine/Medium 2×2-Mean, Confidence-Maps 2×2-`min`+`AND`, bit-exakt + chunk-unabhängig); Store-Round-Trip MONO **und OSC** (`fuse_multiband_store_to_image` streifenweise/speicherbegrenzt, HDR-Feld) bit-exakt zur In-Memory-Referenz; kein objektspezifisches Tuning (alle Schwellen planversionierte Hash-Konstanten). 30.40: `A_artifact`-`<8`-Regel unterdrückt Fine/Medium-Alpha auf ~40 % des OSC-Innenbereichs (nf=30) — §15 sternbasiert, **nicht blockiert**. Dreiwegvalidation §15 als geprüftes Modul `multiband_validation` (30.42): drei feste Kandidaten, eine feste Sternpopulation auf Uniform, `multiband_effective` je Stern, deterministischer geseedeter Bootstrap-CI der FWHM-Mediane, selbstnormierender `boundary_seam_score` (Legacy-Global-Gradient-Proxy bestraft PSF-Schärfung → ersetzt; exakte Seam-Form + Konstanten Planbestätigung ausstehend), „N/A ist nie ein Pass" (§15.3.6); 8 synthetische Fälle grün. **Im Runner verdrahtet** (30.43): `fuse_multiband_store_to_image(&cand)` bildet die drei Arbeitsluminanz-Kandidaten + `alpha_final`-Maps im selben Durchlauf (keine Zusatz-I/O, `kWorkingLumaDefinition` als einzige Luma-Quelle), `MULTIBAND` führt die Auswahl aus und schreibt `artifacts/forward_drizzle.json` (`selected_candidate`/`selection_reason`/`fallback_reason`/`validation`, §16.3-Teilmenge) + `phase_end`-Event + Checkpoint; MONO/OSC-Store-Tests bit-exakt & chunk-unabhängig, Runner-Test end-to-end. **M31-Echtdatenlauf (30.45):** dabei ein realer Defekt in `star_support_ok` gefunden (verlangte den **gesamten** 15×15-Patch endlich ⇒ 0 von 250 Sternen, auch die Uniform-Kontrolle) und behoben (`extract_star_patch`, ≥75 % + Median-Fill); mit Fix wählt die Auswahl **plankonform** `drizzle_uniform` (`raw background_rms regression`, Ratio 1,088 > strikte 1,05 — §15.3 „Erwartete Auswahlverteilung"), `stars_multiband_effective=0` (Energieguard §14.5 drückt Alpha an Sternkernen). **M42-Echtdatenlauf (30.46): zweiter unabhängiger OSC-Datenpunkt** (Nebelfeld, 105 Sterne, `support_ok` für alle drei, gleiches `drizzle_uniform`-Ergebnis; Raw/Uniform-`bg_RMS` 1,0886 ≈ M31s 1,0882 ⇒ ~8,8-%-Drizzle-Rauschoffset ist systematisch, nicht datensatzabhängig). **`validation_config_hash` (30.46):** eigener SHA-256 über die versionierten Auswahlkonstanten + `MultibandValidationConfig` in `forward_drizzle.json` — **nicht** in `multiband_config_hash` (der Store-Bytes hasht, die von diesen Werten unabhängig sind); Seam-Magic-Numbers benannt & mitgehasht; zwei Plan-Fragen dokumentiert (`stars_multiband_effective=0` strukturell, 5-%-`bg_RMS`-Gate vs. §12.4-Drizzle-Rauschen). **`seam_score`-Sentinel-Defekt behoben + Echtdaten-Grenze gefunden (30.47):** sampelte die Randpixel selbst ⇒ NaN-Nachbar per Definition ⇒ 0-Sentinel auf jedem realen Maskenfeld ⇒ `ratio(0,0)=∞` hätte Raw fälschlich verworfen (dritter Fall der `0,88²²⁵`-Klasse: `star_support_ok`, `per_star_fwhm_aligned`, `seam_score`); Fix sampelt die Interior-Edge, nicht-messbar ⇒ N/A statt 0; `metric_json` serialisiert N/A einheitlich als `value: null`. **M42-Resume mit Fix:** Sentinel weg, aber auf realem OSC-Luma ist die Metrik *inert* (`seam_score` 1,030/1,026/1,023 für U/R/M, Ratios < 0,4 % auseinander) — die Stützmaske ist von ~5 % verstreuten Ein-Pixel-Dropouts durchsetzt, die Interior-Edge wird davon dominiert. Frage der Seam-**Form** (Locus; wahrscheinl. morphologisches Maskenöffnen), plan-seitig unbestätigt (30.42) — **nicht** ein drittes Mal selbst gepatcht; Charakterisierungstest pinnt das Verhalten. **Offen (alle drei brauchen den Plan-Eigner bzw. M7-Slice-2):** (1) Seam-Form bestätigen/korrigieren — blockiert (2) das Umschalten der ausgelieferten Datei auf den §15-Kandidaten (Checkpoint/Resume/STACKING-Vertrag, eigener Testsatz), solange ein Mandatory-§15.3.2-Gate inert ist; (3) §16.4-Restdiagnostik (Feld/Backend/RSS/Timing)** | `atrous_decomposition` (30.33): shift-invariante À-trous, Dilatation `2^(j-1)-1`, maskierte renormierte Faltung, `den_min=0,5` versionierte Hash-Domänen-Konstante, `D_j` auf `M_(j-1)&&M_j`, Identität `C_L+ΣD_j==input`, Level-1-Spike = exakter Kern. Fine/Medium-Profile im Drizzle (30.34): `w=B·G_eff·pow(Q_scale*, e)`, `FrameQualityProvider`, geteilte Clippingmaske. `multiband_fusion`/`adaptive_alpha`/`alpha_guard` + `fuse_multiband` (30.35, 22 Fälle): Blend-Identitäten (`alpha≡0,U==R⇒X_out==R`; `U==R==F==M⇒X_out==R`), `A_neff`/`A_coverage` min-über-Kanal + `alpha_cap` + externe Faktoren, 6-Schritt-Bisektions-Energieguard ≤ 1,30 ohne Hartclipping, komponentenweise B3-Glättung mit verbindlicher `min`-Kappe (keine Insel-Leckage) |
| M7 | **Slice 1 (Transaktionsvertrag) fertig** | 30.44: `forward_drizzle_cuda` + `ForwardDrizzleCudaError`; `persist_multiband_store_from_predecessors` verwirft die uncommittete Generation und startet die ganze FORWARD_DRIZZLE-Phase auf dem CPU-Referenzpfad neu (§19.4); committetes Store/Bild bit-identisch zum reinen CPU-Build; `AccelerationPhase::forward_drizzle` + `phase_supports_backend`-Zeile; Fault-Injection-Tests ohne GPU (Store 19-Ebenen-`sha256`, Runner end-to-end). **Slice 2 offen**: `.cu`-Kernel (Droplet/Clipping/Profil), Memory-Auto-Sizing, Device-Probe, Paritätsmatrix §19.5, Timing/`acceleration`-Block §16.4. Vorbestehender roter Legacy-Test `acceleration_context_keeps_aqmh_maps_cpu_only` (aqmh_maps↔opencv_cuda, kein M7-Regressor). Vorhandenes AQMH-CUDA implementiert die neue Semantik weiterhin nicht |
| M8 | offen | GUI, Report, Schemas, Beispiele und aktive Dokumentation verwenden weiterhin die Altverträge |
| M9 | offen | keine autorisierten Qualitätsläufe oder vollständige Pflichtmatrix für den neuen Pfad |
| M10 | offen | kein Produkt-Cutover; Altpfade sind weiterhin Produktbestand |
| M11 | offen | setzt M10 und den Grace-Zyklus voraus |

Vorhandene AQMH-Komponenten für Quality-Maps, Region-Reads, Clipping,
Uniform-Control, Validation und CPU-/GPU-Fallback sind mögliche
**Extraktionsquellen**. Sie erfüllen wegen PREWARP-Eingabe, anderer Hash-/Support-
und Kandidatensemantik keinen Meilenstein dieses Plans ohne Anpassung und neue
Tests. Insbesondere darf ihre Existenz nicht als „teilweise implementierter
Forward-Drizzle“ berichtet werden.

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

## 31. Entscheidungsfixierung der offenen Punkte (2026-09-02)

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
versionierte Ausgangsvertrag. Offen sind ausschließlich Tests und reale
Messungen, nicht mehr die Implementierungssemantik.
