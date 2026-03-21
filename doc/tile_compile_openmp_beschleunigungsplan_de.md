# Tile-Compile OpenMP-Beschleunigungsplan

Stand: 2026-03-20

Dieses Dokument analysiert, an welchen Stellen der aktuellen C++-Pipeline von
`tile_compile_cpp/` OpenMP sinnvoll waere, welchen Nutzen man realistisch
erwarten kann und wo OpenMP trotz vorhandener Parallelisierungsmoeglichkeiten
eher nicht die richtige Wahl ist.

Wichtige Einschraenkung: Die Aussagen basieren auf statischer Codeanalyse der
aktuellen Implementierung. In diesem Schritt wurde kein Laufzeit-Profiling auf
Referenzdatensaetzen durchgefuehrt.

## Kurzfazit

OpenMP lohnt sich in diesem Repo deutlich selektiver als GPU-Offload.

Der Hauptgrund:

- grosse Teile der Pipeline sind bereits auf Frame- oder Tile-Ebene mit
  `std::thread` parallelisiert
- OpenMP kann dort nicht einfach "zusaetzlich" draufgesetzt werden, ohne
  Oversubscription und Cache-Konflikte zu riskieren

Die besten OpenMP-Kandidaten sind daher vor allem:

1. serielle per-Pixel Reduktionskerne in `STACKING`
2. serielle Vollbild-Nachlaeufe nach `TILE_RECONSTRUCTION`
3. BGE-Vollbild-Rendering und aehnliche serielle Flaechenpassagen
4. einige serielle Maskierungs- und Skalierungspfade

Weniger geeignet sind:

- I/O-lastige Phasen
- bereits parallelisierte Frame-/Tile-Dispatcher
- innere Schleifen, die in bereits laufenden Worker-Threads erneut parallel
  werden wuerden
- branchige Registrierungsheuristiken

Realistische Gesamtwirkung von OpenMP ist kleiner als bei einem guten
GPU-Offload:

- nur die heute schon vorhandenen `#pragma omp`: praktisch `0%` bis `15%`
- gezielte OpenMP-Erweiterung auf sinnvolle serielle Hotspots:
  grob `1.05x` bis `1.35x` gesamt
- in guenstigen CPU-lastigen Faellen mit grossem BGE-/Stacking-Anteil:
  etwa `1.2x` bis `1.6x`

OpenMP ist hier also vor allem ein gutes Werkzeug fuer CPU-Hardening und
gezielte Kernoptimierung, nicht der groesste einzelne Durchsatzhebel der
gesamten Pipeline.

## Aktueller Befund im Code

### 1. OpenMP ist derzeit praktisch nicht aktiv

- Es gibt im Quellcode nur zwei OpenMP-Stellen:
  - `tile_compile_cpp/src/image/background_extraction.cpp:2313`
  - `tile_compile_cpp/src/image/background_extraction.cpp:2483`
- Im CMake von `tile_compile_cpp/` gibt es aktuell aber kein
  `find_package(OpenMP)` und keinen Link gegen `OpenMP::OpenMP_CXX`:
  `tile_compile_cpp/CMakeLists.txt:341-382`
- In der aktuellen `compile_commands.json` des Builds fehlt fuer
  `background_extraction.cpp` ein `-fopenmp`-Flag:
  `tile_compile_cpp/build/compile_commands.json`

Das bedeutet:

- die vorhandenen OpenMP-Pragmas sind im aktuellen Build nicht wirksam
- faktisch laeuft die Pipeline derzeit ohne OpenMP-Beschleunigung

### 2. Die Pipeline nutzt bereits manuelle CPU-Parallelisierung

Mehrere Hauptphasen sind schon mit `std::thread` parallelisiert:

- Registrierung:
  `tile_compile_cpp/apps/runner_phase_registration.cpp:654-667`
- Normalisierung:
  `tile_compile_cpp/apps/runner_phase_metrics.cpp:288-301`
- globale Metriken:
  `tile_compile_cpp/apps/runner_phase_metrics.cpp:441-453`
- lokale Metriken:
  `tile_compile_cpp/apps/runner_phase_local_metrics.cpp:170-183`
- Tile-Rekonstruktion:
  `tile_compile_cpp/apps/runner_pipeline.cpp:1527-1558`
- synthetische Rekonstruktion:
  `tile_compile_cpp/apps/runner_pipeline.cpp:2786-2798`

Konsequenz:

- OpenMP sollte hier nicht blind als zusaetzliche innere Parallelisierung
  eingeschaltet werden
- sonst entstehen schnell mehr Threads als CPU-Kerne
- dadurch drohen schlechtere Skalierung, mehr Kontextwechsel und schlechtere
  Cache-Lokalitaet

### 3. OpenCV-Threads werden bereits stellenweise bewusst begrenzt

In `TILE_RECONSTRUCTION` wird OpenCV-Threading explizit auf `1` gesetzt:

- `tile_compile_cpp/apps/runner_pipeline.cpp:995-996`
- Ruecksetzen bei:
  `tile_compile_cpp/apps/runner_pipeline.cpp:1859`

Das ist ein wichtiger Hinweis:

- die Implementierung versucht bereits aktiv, Nested-Parallelism zu vermeiden
- OpenMP muss sich in diese Logik einfuegen, nicht parallel dazu arbeiten

## Bewertungslogik fuer OpenMP

Ein Prozess ist ein guter OpenMP-Kandidat, wenn moeglichst viele dieser Punkte
zutreffen:

- grosse serielle CPU-Schleife
- unabhaengige Iterationen
- wenig oder keine gemeinsamen Schreibzugriffe
- einfacher Thread-Local-Scratch moeglich
- nicht bereits in einer aeusseren Worker-Parallelisierung eingebettet

Ein Prozess ist ein schlechter OpenMP-Kandidat, wenn:

- er bereits auf aeusserer Ebene parallel laeuft
- er vor allem I/O macht
- viele kleine kritische Abschnitte oder Locks noetig waeren
- er stark branchig oder heuristisch ist

## Geeignete OpenMP-Kandidaten

## A. Hohe Prioritaet

### A1. `STACKING`: `sigma_clip_stack()`

Codebasis:

- `tile_compile_cpp/apps/runner_pipeline.cpp:3085-3099`
- `tile_compile_cpp/apps/runner_pipeline.cpp:3122-3126`
- `tile_compile_cpp/src/reconstruction/reconstruction.cpp:393-489`

Warum geeignet:

- `sigma_clip_stack()` laeuft pixelweise ueber komplette Bilder
- jede Pixelposition ist unabhaengig
- die Funktion wird in `STACKING` derzeit seriell aufgerufen
- hier gibt es keine aeussere Tile-Worker-Ebene mehr, die mit OpenMP in
  Konflikt geraten wuerde

OpenMP-Strategie:

- Parallelisierung ueber `idx` in
  `tile_compile_cpp/src/reconstruction/reconstruction.cpp:413-485`
- pro Thread eigener Scratch fuer `values` und `keep`

Erwarteter Nutzen:

- kernelbezogen grob `2x` bis `6x`
- phasenbezogen gut sichtbar, besonders bei vielen synthetischen Frames

### A2. Serielle Nachlaeufe nach `TILE_RECONSTRUCTION`

Codebasis:

- OLA-Nachlauf:
  `tile_compile_cpp/apps/runner_pipeline.cpp:1765-1857`
- finale Division durch `weight_sum`:
  `tile_compile_cpp/apps/runner_pipeline.cpp:1865-1929`
- globales Background-Restore:
  `tile_compile_cpp/apps/runner_pipeline.cpp:1907-1913`
  und
  `tile_compile_cpp/apps/runner_pipeline.cpp:1945-1948`

Warum geeignet:

- diese Schleifen laufen nach Abschluss der Tile-Worker wieder seriell
- sie bearbeiten grosse lineare Arrays
- die Iterationen sind weitgehend unabhaengig

Wichtiger Unterschied:

- der eigentliche OLA-Block schreibt in gemeinsame Zielbilder und ist nicht
  trivial mit einem simplen `parallel for` absicherbar
- die nachfolgenden Vollbild-Paesse sind dagegen sehr gute OpenMP-Kandidaten

Empfehlung:

- zuerst die linearen Vollbild-Nachlaeufe parallelisieren
- fuer OLA nur mit thread-lokalen Akkumulatoren oder Stripe-Aufteilung, nicht
  mit naiver gemeinsamer Addition

Erwarteter Nutzen:

- fuer die linearen Passes `1.5x` bis `4x`
- fuer den ganzen Rekonstruktionsblock eher moderat, aber sauber erreichbar

### A3. BGE-Flaechenrendering

Codebasis:

- `tile_compile_cpp/src/image/background_extraction.cpp:2310-2320`
- `tile_compile_cpp/src/image/background_extraction.cpp:2456-2503`

Warum geeignet:

- das sind klassische unabhaengige 2D-Gitter-Schleifen
- keinerlei I/O
- keine relevante Synchronisation
- OpenMP-Pragmas existieren bereits

Erwarteter Nutzen:

- lokal oft `2x` bis `6x`
- bei grossen RGB-BGE-Runs auch hoeher
- fuer die Gesamtpipeline aber nur relevant, wenn `BGE` ueberhaupt aktiv ist

### A4. Serielle Common-Overlap- und Vollbild-Maskierungspfade

Codebasis:

- `tile_compile_cpp/apps/runner_shared.hpp:125-153`
- `tile_compile_cpp/apps/runner_shared.hpp:156-192`
- `tile_compile_cpp/apps/runner_phase_registration.cpp:1604-1617`

Warum geeignet:

- lineare Scans ueber Vollbilder oder Canvas-Masken
- sehr einfache Datenabhaengigkeiten
- in mehreren spaeten Phasen wiederkehrende Vollbild-Paesse

Erwarteter Nutzen:

- einzeln eher klein bis mittel
- zusammengenommen sinnvoll, wenn man mehrere dieser Passes parallelisiert

## B. Mittlere Prioritaet

### B1. Prewarp-Nachlauf fuer Overlap-Reduktion

Codebasis:

- per-Frame Coverage-Scan im Worker:
  `tile_compile_cpp/apps/runner_phase_registration.cpp:1530-1535`
- Merge der Worker-Coverage:
  `tile_compile_cpp/apps/runner_phase_registration.cpp:1600-1610`
- Aufbau von `common_valid_mask`:
  `tile_compile_cpp/apps/runner_phase_registration.cpp:1612-1617`

Bewertung:

- der aeussere Prewarp ist bereits ueber Frames parallelisiert
- die innere per-Frame-Pixelzaehlung sollte man nicht zusaetzlich mit OpenMP
  parallelisieren
- der nachgelagerte Merge und Maskenaufbau sind dagegen gute OpenMP-Kandidaten

Erwarteter Nutzen:

- moderat
- besonders bei grossen Canvas-Flaechen

### B2. `PCC`: Hintergrund-Scan und Neutralisierung

Codebasis:

- Hintergrund-Median-Samples:
  `tile_compile_cpp/src/astrometry/photometric_color_cal.cpp:1563-1580`
- Hintergrundsneutralisierung:
  `tile_compile_cpp/src/astrometry/photometric_color_cal.cpp:1606-1629`

Warum nur mittel:

- die Loops selbst sind gut parallelisierbar
- die PCC-Phase als Ganzes ist aber nicht von diesen Vollbildschleifen
  dominiert
- die schweren Teile von PCC liegen eher in Sternkatalog-, Matching- und
  Robustheitslogik

Erwarteter Nutzen:

- lokal mittel
- gesamt fuer PCC meist eher klein

### B3. `apply_output_scaling_inplace()` und aehnliche lineare Bildpaesse

Codebasis:

- `tile_compile_cpp/src/image/normalization.cpp:40-70`

Warum mittel:

- technisch leicht parallelisierbar
- grosse lineare Arrays
- aber nicht die dominante Laufzeit der Pipeline

Erwarteter Nutzen:

- klein bis mittel
- als "mitnehmen" sinnvoll, nicht als erstes Ziel

## C. Bedingt sinnvoll, aber nur mit Refactoring

### C1. `LOCAL_METRICS`

Codebasis:

- Frame-Worker:
  `tile_compile_cpp/apps/runner_phase_local_metrics.cpp:79-138`
- Tile-Metrik-Kern:
  `tile_compile_cpp/src/metrics/tile_metrics.cpp:166-281`

Warum nur bedingt:

- `LOCAL_METRICS` ist bereits frame-parallel mit `std::thread`
- `calculate_tile_metrics()` wird pro Tile im Worker aufgerufen
- ein inneres OpenMP auf den Tile-Schleifen wuerde nested parallel laufen

Wann es trotzdem lohnen koennte:

- bei kleinen Frame-Zahlen und sehr vielen Tiles
- wenn man die Parallelisierung von "pro Frame" auf "pro Tile" umstellt oder
  OpenMP als Ersatz statt als Zusatz nutzt

Was nicht zu empfehlen ist:

- direkt `#pragma omp parallel for` innerhalb von
  `tile_compile_cpp/src/metrics/tile_metrics.cpp:177-185`
  oder
  `tile_compile_cpp/src/metrics/tile_metrics.cpp:230-234`
  waehrend die aeussere Phase schon Worker-Threads verwendet

Erwarteter Nutzen:

- ohne Refactoring gering oder negativ
- mit sauberer Neuaufteilung potentiell mittel

### C2. `sigma_clip_weighted_tile()`

Codebasis:

- `tile_compile_cpp/src/reconstruction/reconstruction.cpp:491-625`

Warum nur bedingt:

- der Pixel-Loop ist eigentlich ein guter OpenMP-Kandidat
- aber die Funktion wird hauptsaechlich innerhalb bereits parallelisierter
  Tile-Worker verwendet:
  - `tile_compile_cpp/apps/runner_pipeline.cpp:1422-1426`
  - `tile_compile_cpp/apps/runner_pipeline.cpp:1362-1366`
  - `tile_compile_cpp/apps/runner_pipeline.cpp:2733-2738`

Konsequenz:

- als zusaetzliche innere OpenMP-Schleife problematisch
- sinnvoll nur, wenn man die aeussere Threading-Strategie neu ausrichtet

### C3. `NORMALIZATION` und `GLOBAL_METRICS`

Codebasis:

- Frame-Worker fuer Normalisierung:
  `tile_compile_cpp/apps/runner_phase_metrics.cpp:115-301`
- Frame-Worker fuer globale Metriken:
  `tile_compile_cpp/apps/runner_phase_metrics.cpp:369-453`
- innerer Metrik-Kern:
  `tile_compile_cpp/src/metrics/metrics.cpp:106-164`

Bewertung:

- die Phase ist bereits frame-parallel
- viele innere Schleifen sammeln Samples in `std::vector`
- zusaetzliches OpenMP in diesen inneren Schleifen waere nur nach Umbau auf
  thread-lokale Puffer sinnvoll

Erwarteter Nutzen:

- eher begrenzt
- meist kleiner als bei `STACKING`, BGE oder seriellen Vollbild-Nachlaeufen

## Schlechte OpenMP-Kandidaten

Die folgenden Bereiche wuerde ich nicht priorisieren:

- FITS-I/O und Disk-Cache-Pfade:
  - `tile_compile_cpp/apps/runner_shared.cpp:640-748`
  - `tile_compile_cpp/apps/runner_phase_metrics.cpp:134`
- heuristische Registrierungslogik und Matching-Kaskade
- Event-/JSON-/Artefakt-Schreiben
- kleine Hotspots, die in bereits aktiven Worker-Threads liegen
- naive Nested-OpenMP-Strategien in Kombination mit `std::thread`

## Empfohlene technische Richtung

## 1. OpenMP nur fuer leaf kernels oder serielle Vollbildpaesse

Das ist die wichtigste Regel in dieser Codebasis.

Gut:

- serielle Vollbild-Paesse nach Abschluss einer Worker-Phase
- echte leaf kernels ohne aeussere Parallelisierung
- pro-Thread Scratch moeglich

Schlecht:

- OpenMP innerhalb laufender `std::thread`-Tile- oder Frame-Worker

## 2. OpenMP nicht als Zusatz, sondern als bewusste Parallelisierungsebene

Es gibt zwei sinnvolle Modelle:

1. `std::thread` fuer aeussere Dispatcher, OpenMP nur fuer serielle Nachlaeufe
2. einzelne Kerne komplett auf OpenMP umstellen und aeussere Worker reduzieren

Nicht sinnvoll:

- beides gleichzeitig ungeplant aktivieren

## 3. Threadzahl an `runtime_limits.parallel_workers` koppeln

Wenn OpenMP eingefuehrt wird, sollte die Threadzahl nicht separat und unkontrolliert
leben.

Empfehlung:

- `omp_set_num_threads(cfg.runtime_limits.parallel_workers)`
- Nested-Parallelism standardmaessig deaktivieren
- bei aeusseren `std::thread`-Workern OpenMP-Leaf-Kerne nur dort aktivieren, wo
  keine Ueberschneidung entsteht

## 4. OpenMP zuerst optional in CMake aktivieren

Sinnvolle CMake-Richtung:

- `find_package(OpenMP QUIET)`
- optionale Build-Flag `TILE_COMPILE_ENABLE_OPENMP`
- nur bei erfolgreichem Fund gegen `OpenMP::OpenMP_CXX` linken
- Makro wie `TILE_COMPILE_HAVE_OPENMP` setzen

So bleibt der Build portabel und OpenMP ein optionales CPU-Backend.

## Realistisch erwarteter Nutzen

Die folgenden Schaetzungen beziehen sich auf sinnvolle, nicht-nested Nutzung.

| Bereich | Erwartete Beschleunigung | Bemerkung |
|---|---:|---|
| BGE-Rendering | `2x-6x` | nur wenn BGE aktiv ist |
| `sigma_clip_stack()` | `2x-6x` | guter OpenMP-Kandidat |
| lineare Vollbild-Nachlaeufe | `1.5x-4x` | technisch leicht erreichbar |
| Prewarp-Merge/Maskenbau | `1.2x-2x` | moderater Zusatzhebel |
| PCC-Hintergrundpassagen | `1.2x-2x` | nur kleiner Einfluss auf Gesamtphase |

Gesamtpipeline:

- nur heutige zwei `omp`-Pragmas aktivieren: meist `0%` bis `15%`
- gezielte Erweiterung auf serielle Hauptkerne: etwa `1.05x` bis `1.35x`
- grosse RGB/BGE/Stacking-Faelle: eventuell `1.2x` bis `1.6x`

Groesserer Gesamtgewinn ist moeglich, aber dann eher durch:

- groessere Refactorings der Parallelisierungsebene
- bessere I/O-Pfade
- oder GPU-Offload

## Priorisierter Umsetzungsplan

## Phase 0. OpenMP sauber aktivierbar machen

Ziel:

- reproduzierbar messen koennen

Massnahmen:

- optionales OpenMP in CMake einfuehren
- Compiler-/Link-Flags verifizieren
- Build ohne OpenMP als Default beibehalten

Akzeptanz:

- identischer Codepfad mit und ohne OpenMP moeglich

## Phase 1. Bestehende BGE-Pragmas aktivieren

Ziel:

- niedriges Risiko, sofort messbar

Massnahmen:

- OpenMP in CMake aktivieren
- BGE-Referenzruns messen

Akzeptanz:

- gleiche Ergebnisse
- messbarer Gewinn in BGE-Runs

## Phase 2. `STACKING` und serielle Vollbildpaesse

Ziel:

- die klarsten OpenMP-Hotspots abdecken

Massnahmen:

- `sigma_clip_stack()` mit thread-lokalem Scratch parallelisieren
- finale Rekonstruktions-Division und Background-Restore parallelisieren
- optionale Output-Scaling- und Maskierungsloops parallelisieren

Akzeptanz:

- keine numerischen Regressionsprobleme
- sichtbarer Gesamtgewinn auch ohne BGE

## Phase 3. Prewarp-Nachlaeufe und weitere serielle Array-Paesse

Ziel:

- weitere einfache CPU-Kerne mitnehmen

Massnahmen:

- Merge von `overlap_coverage_count`
- Aufbau von `common_valid_mask`
- serielle RGB-/Frame-Maskierungshelfer

Akzeptanz:

- stabiler kleiner Zusatzgewinn

## Phase 4. Nur bei Bedarf: Refactoring fuer tieferes OpenMP

Ziel:

- pruefen, ob `LOCAL_METRICS` oder tileweise Reduktionskerne neu zugeschnitten
  werden muessen

Massnahmen:

- entscheiden, ob aeussere `std::thread`-Ebene oder innere OpenMP-Ebene die
  bessere Granularitaet hat
- niemals blind nested parallelisieren

Akzeptanz:

- keine Oversubscription
- klarer Gewinn gegen die bisherige `std::thread`-Variante

## Konkrete Empfehlung

Wenn du OpenMP in diesem Projekt einsetzen willst, wuerde ich in genau dieser
Reihenfolge vorgehen:

1. OpenMP optional in CMake aktivieren
2. bestehende BGE-Pragmas tatsaechlich nutzbar machen
3. `sigma_clip_stack()` parallelisieren
4. serielle Vollbild-Nachlaeufe nach `TILE_RECONSTRUCTION` parallelisieren
5. erst danach ueber tiefere Refactorings in `LOCAL_METRICS` oder
   tileweisen Reduktionskernen nachdenken

Der Kernpunkt ist:

- OpenMP ist hier am staerksten als gezielte CPU-Kernoptimierung
- nicht als pauschale zweite Parallelisierungsschicht ueber bereits laufenden
  Worker-Threads
