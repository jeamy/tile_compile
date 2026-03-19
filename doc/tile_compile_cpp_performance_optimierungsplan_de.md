# Tile-Compile C++ Performance-Optimierungsplan

Stand: 2026-03-18

Dieses Dokument fasst die wichtigsten Performance-Hebel in `tile_compile_cpp/`
zusammen und priorisiert sie nach Nutzen, Risiko und Implementierungsaufwand.
Die Empfehlungen basieren auf statischer Codeanalyse der aktuellen
Runner-/Pipeline-Implementierung. Es wurde in diesem Schritt kein
Laufzeit-Profiling auf Referenzdatensaetzen durchgefuehrt.

## Ziel

Ziel ist eine spuerbar schnellere Pipeline fuer typische Runs mit vielen FITS-
Frames, ohne die methodische Logik der TBQR-Pipeline zu veraendern.

Die groessten Bremsen im aktuellen Stand sind:

- mehrfaches Lesen derselben FITS-Frames in mehreren Phasen
- mehrfaches Berechnen derselben normalisierten/proxy-basierten Zwischenformen
- Vollbild-Scans ueber die komplette Canvas in mehreren Phasen
- teure tileweise OSC-Debayer-Schritte in Phase 9
- doppelte Rekonstruktionsarbeit fuer Synthetic Frames

## Kurzfassung der Prioritaeten

1. Gemeinsamen Frame-/Proxy-Cache einfuehren
2. FITS-I/O-Hotpaths verschlanken
3. `COMMON_OVERLAP` in `PREWARP` integrieren
4. OSC-Debayer-Strategie in `TILE_RECONSTRUCTION` umbauen
5. `SYNTHETIC_FRAMES` von doppelter Rekonstruktion entkoppeln
6. Lokale Tile-Metriken und Diagnosepfade abspecken

## Stufe 1: Quick Wins mit hohem ROI

### 1. Gemeinsamen Cache fuer `raw`, `normalized`, `proxy-2x` einfuehren

Problem:

- `NORMALIZATION` liest jedes Frame vollstaendig in
  [runner_phase_metrics.cpp](/media/data/programming/tile_compile/tile_compile_cpp/apps/runner_phase_metrics.cpp#L78)
- `GLOBAL_METRICS` liest dieselben Frames erneut in
  [runner_phase_metrics.cpp](/media/data/programming/tile_compile/tile_compile_cpp/apps/runner_phase_metrics.cpp#L330)
- `REGISTRATION` liest und normalisiert erneut in
  [runner_phase_registration.cpp](/media/data/programming/tile_compile/tile_compile_cpp/apps/runner_phase_registration.cpp#L470)
- spaetere Phasen laden wieder normalisierte Frames in
  [runner_pipeline.cpp](/media/data/programming/tile_compile/tile_compile_cpp/apps/runner_pipeline.cpp#L594)

Massnahme:

- `FrameCache` oder `FrameProductsCache` einfuehren
- pro Frame optional speichern:
  - rohe Pixel
  - normalisierte Pixel
  - Registrierungs-Proxy
  - Global-Metrics-Proxy
- policy-basiert:
  - RAM fuer kleine Datensaetze
  - mmap/raw-cache fuer grosse Datensaetze

Erwarteter Nutzen:

- reduziert Disk-I/O deutlich
- reduziert CPU-Zeit fuer wiederholte Normalisierung
- verbessert besonders `NORMALIZATION + GLOBAL_METRICS + REGISTRATION`

Risiko:

- mittel, weil Lebensdauer und Speicherbudget sauber gesteuert werden muessen

### 2. FITS-Reader fuer Hotpaths auf "Pixel only" aufspalten

Problem:

- [fits_io.cpp](/media/data/programming/tile_compile/tile_compile_cpp/src/io/fits_io.cpp#L177)
  liest immer Header und Pixel
- danach werden Pixel noch einmal elementweise in `Matrix2Df` kopiert in
  [fits_io.cpp](/media/data/programming/tile_compile/tile_compile_cpp/src/io/fits_io.cpp#L284)

Massnahme:

- neuen Hotpath-Reader einfuehren:
  - `read_fits_pixels_float()`
  - optional `read_fits_pixels_and_selected_header_keys()`
- Header nur dort lesen, wo er wirklich benoetigt wird
- Bulk-Copy statt doppelter Schleife verwenden, da `Matrix2Df` Row-Major ist:
  [types.hpp](/media/data/programming/tile_compile/tile_compile_cpp/include/tile_compile/core/types.hpp#L14)

Erwarteter Nutzen:

- schnelleres Laden pro Frame
- weniger Overhead in allen frameweisen Phasen

Risiko:

- niedrig bis mittel

### 3. `COMMON_OVERLAP` direkt waehrend `PREWARP` aufbauen

Problem:

- nach `PREWARP` wird die komplette Canvas ueber alle Frames noch einmal
  abgescannt in
  [runner_pipeline.cpp](/media/data/programming/tile_compile/tile_compile_cpp/apps/runner_pipeline.cpp#L707)
- das ist ein weiterer voller `n_frames * canvas_pixels`-Pass

Massnahme:

- `overlap_coverage_count` schon im `PREWARP`-Worker mitfuehren
- pro gewarptem Frame direkt Coverage aktualisieren
- `common_valid_mask` danach ohne zweiten Frame-Vollscan erzeugen

Erwarteter Nutzen:

- grosse Einsparung bei grossen Canvas-Flaechen
- besonders wichtig bei Alt/Az-Runs mit expandierter Canvas

Risiko:

- niedrig

### 4. `output.write_registered_frames` klar als Debug-Pfad behandeln

Problem:

- [runner_phase_registration.cpp](/media/data/programming/tile_compile/tile_compile_cpp/apps/runner_phase_registration.cpp#L705)
  rereadet und rewritet alle registrierten Frames
- das ist bei grossen Runs teuer und oft nicht noetig

Massnahme:

- Default fuer Produktivlaeufe auf `false` lassen
- Dokumentation und GUI klar als Debug-/Diagnoseoption markieren
- optional Export aus Prewarp-Cache statt erneutem FITS-Read

Erwarteter Nutzen:

- weniger I/O und weniger Runzeit in Diagnose-lastigen Runs

Risiko:

- niedrig

## Stufe 2: Mittlere Umbauten mit sehr gutem ROI

### 5. `NORMALIZATION` und `GLOBAL_METRICS` auf gemeinsamen Proxy zusammenziehen

Problem:

- `GLOBAL_METRICS` normalisiert jedes Bild erneut in
  [runner_phase_metrics.cpp](/media/data/programming/tile_compile/tile_compile_cpp/apps/runner_phase_metrics.cpp#L356)
- anschliessend werden Frame-Metriken und Sternmetriken auf diesem Ergebnis
  berechnet

Massnahme:

- nach `NORMALIZATION` direkt einen Metrics-Proxy erzeugen und cachen
- `calculate_frame_metrics()` und `measure_frame_stars()` auf diesem Proxy
  laufen lassen
- wenn fuer Ranking ausreichend:
  - FWHM-/Sternmessung ebenfalls auf dem Proxy halten

Erwarteter Nutzen:

- reduziert CPU und I/O deutlich
- weniger Mehrfacharbeit in den fruehen Phasen

Risiko:

- mittel, da die Metrik-Stabilitaet gegenueber Full-Resolution verifiziert
  werden muss

### 6. Tile-Metriken auf Thread-Local-Scratch umstellen

Problem:

- [tile_metrics.cpp](/media/data/programming/tile_compile/tile_compile_cpp/src/metrics/tile_metrics.cpp#L158)
  erzeugt pro Tile mehrere grosse Vektoren
- Blur, Sobel, Normalisierung und `goodFeaturesToTrack` laufen pro Tile

Massnahme:

- Thread-Local-Buffers fuer:
  - `px`
  - `resid_px`
  - `bg_vals`
  - `grad_vals`
- optional Fast-Path:
  - fuer klare Struktur-Tiles keine Sternsuche
  - fuer klar leere Tiles frueher Ausstieg

Erwarteter Nutzen:

- weniger Heap-Allocation
- besseres CPU-Cache-Verhalten in `LOCAL_METRICS`

Risiko:

- niedrig bis mittel

### 7. Post-Warp-Diagnostik in Phase 9 ausduennen

Problem:

- `compute_post_warp_metrics()` macht pro Tile wieder Laplacian, Pixel-Scans
  und Verteilungsberechnungen in
  [runner_pipeline.cpp](/media/data/programming/tile_compile/tile_compile_cpp/apps/runner_pipeline.cpp#L909)

Massnahme:

- Diagnosemodus von Produktionsmodus trennen
- nur Stichprobe der Tiles vermessen oder nur bei aktivierter Validierung
- alternativ billigere Metriken auf bereits vorhandenen Tile-Statistiken

Erwarteter Nutzen:

- spuerbar kuerzere Phase 9 bei vielen Tiles

Risiko:

- niedrig

### 8. OLA-Akkumulation parallelisieren

Problem:

- nach dem Tile-Processing wird das overlap-add noch seriell ueber alle Tiles
  ausgefuehrt in
  [runner_pipeline.cpp](/media/data/programming/tile_compile/tile_compile_cpp/apps/runner_pipeline.cpp#L1600)

Massnahme:

- worker-lokale Akkumulatoren pro Stripe oder Block
- anschliessend Reduktion in finale `recon_*` und `weight_sum`
- fuer OSC getrennt fuer `R/G/B`

Erwarteter Nutzen:

- gute Skalierung auf Mehrkernsystemen bei grossen Tile-Zahlen

Risiko:

- mittel wegen Synchronisation und RAM-Bedarf

## Stufe 3: Groessere Umbauten mit maximalem Hebel

### 9. OSC-Tiles nicht pro Tile/frame neu debayern

Problem:

- in [runner_pipeline.cpp](/media/data/programming/tile_compile/tile_compile_cpp/apps/runner_pipeline.cpp#L1112)
  wird fuer jedes Tile und jedes gueltige Frame zuerst das Tile extrahiert und
  danach debayert
- das skaliert schlecht mit vielen Frames und Tiles

Massnahme:

- pro prewarped Frame einmal CFA-Subplanes oder debayerte RGB-Zwischenformen
  erzeugen
- Tiles danach aus den bereits vorbereiteten Kanalbildern schneiden
- alternativ:
  - gruenen Proxy fuer manche Metriken
  - RGB nur fuer Stacking-Teil

Erwarteter Nutzen:

- wahrscheinlich groesster einzelner Speedup in Phase 9 fuer OSC-Runs

Risiko:

- mittel bis hoch, weil Speicherlayout und CFA-Korrektheit sauber bleiben
  muessen

### 10. `SYNTHETIC_FRAMES` inkrementell statt rekonstruktiv erzeugen

Problem:

- [runner_pipeline.cpp](/media/data/programming/tile_compile/tile_compile_cpp/apps/runner_pipeline.cpp#L2487)
  rekonstruiert Clusterbilder tileweise praktisch noch einmal

Massnahme:

- waehrend Phase 9 bereits clusterweise Akkumulatoren oder vorbereitete
  Tile-Pools aufbauen
- Synthetic Frames dann aus vorhandenen Zwischenergebnissen ableiten
- keine zweite fast vollstaendige Tile-Rekonstruktion

Erwarteter Nutzen:

- grosse Einsparung bei aktivierten Synthetic Frames

Risiko:

- hoch, weil Datenfluss der spaeten Pipeline angepasst werden muss

### 11. Registrierungs-Proxy persistent halten

Problem:

- direkte Registrierung und temporale Rescue laden/erzeugen Proxies mehrfach in
  [runner_phase_registration.cpp](/media/data/programming/tile_compile/tile_compile_cpp/apps/runner_phase_registration.cpp#L587)

Massnahme:

- pro Frame Registrierungs-Proxy einmal berechnen
- fuer direkte Registrierung, Outlier-Analyse, temporale Rescue und Modellfit
  wiederverwenden

Erwarteter Nutzen:

- spuerbar schnelleres `REGISTRATION`
- weniger I/O in problematischen Runs

Risiko:

- mittel

## Konkreter Umsetzungsplan fuer Prioritaet 1 bis 3

Die folgenden Punkte beschreiben nicht nur die Zielrichtung, sondern den
empfohlenen technischen Zuschnitt fuer die ersten drei Optimierungen. Die
Abschnitte fokussieren auf Zielbild, technische Schnittstellen, Validierung und
Rollout.

### Prioritaet 1: Gemeinsamer Frame-/Proxy-Cache

Zielbild:

- `NORMALIZATION`, `GLOBAL_METRICS`, `REGISTRATION` und spaetere Loaderpfade
  sollen nicht mehr direkt `io::read_fits_float(...)` aufrufen.
- Stattdessen sollen alle framebasierten Phasen ueber einen gemeinsamen
  Zugriffspfad laufen, der rohe Pixel, normalisierte Bilder und Proxies
  wiederverwendet.
- Der erste Zuschnitt sollte runner-lokal bleiben, damit die Aenderung klein
  und kontrollierbar bleibt.

Empfohlene neue Bausteine:

- `tile_compile_cpp/apps/runner_frame_cache.hpp`
- `tile_compile_cpp/apps/runner_frame_cache.cpp`
- ein kleiner Typ `FrameProductKind` mit mindestens:
  - `Raw`
  - `Normalized`
  - `RegistrationProxy`
  - `MetricsProxy`
- ein `FrameProducts`-Container pro Frame mit optionalen Produkten
- ein `FrameProductsCache`, das pro Frame lazy laedt und Resultate wiederverwendet

Empfohlene API:

- `const Matrix2Df& get_raw(size_t frame_index)`
- `const Matrix2Df& get_normalized(size_t frame_index)`
- `const Matrix2Df& get_registration_proxy(size_t frame_index)`
- `const Matrix2Df& get_metrics_proxy(size_t frame_index)`

Wichtige Designentscheidungen:

- Normalisierung darf nicht mehrfach gerechnet werden. `GLOBAL_METRICS` und
  `REGISTRATION` sollen dasselbe normalisierte Produkt sehen.
- Proxies sollen aus dem bereits geladenen Basisprodukt entstehen, nicht aus
  einem zweiten FITS-Read.
- Thread-Sicherheit soll pro Frame-Entry erfolgen, nicht ueber einen globalen
  Mutex. Sonst verschiebt sich das Bottleneck nur.
- Die erste Version sollte eine einfache RAM-Budget-Grenze haben:
  - `raw` und `normalized` duerfen evicted werden
  - `registration_proxy` und `metrics_proxy` sollten laenger gehalten werden,
    weil sie klein sind und oft erneut benoetigt werden

Empfohlene Einfuehrung in zwei Schritten:

- Schritt A:
  - Cache nur fuer `REGISTRATION` und die dortige temporale Rescue einfuehren
  - Ziel ist, die Proxy-Mehrfachberechnung sofort zu stoppen
- Schritt B:
  - `NORMALIZATION` und `GLOBAL_METRICS` auf denselben Cache umstellen
  - dann den allgemeinen Loader in [runner_pipeline.cpp](/media/data/programming/tile_compile/tile_compile_cpp/apps/runner_pipeline.cpp#L594)
    ebenfalls an den Cache anbinden

Konkreter Umbau in den bestehenden Dateien:

- [runner_phase_metrics.cpp](/media/data/programming/tile_compile/tile_compile_cpp/apps/runner_phase_metrics.cpp#L78)
  soll nicht mehr selbst FITS laden und normalisieren, sondern `get_normalized`
  oder einen expliziten Cache-Populate-Pfad verwenden.
- [runner_phase_metrics.cpp](/media/data/programming/tile_compile/tile_compile_cpp/apps/runner_phase_metrics.cpp#L330)
  soll auf `get_metrics_proxy` umgestellt werden.
- [runner_phase_registration.cpp](/media/data/programming/tile_compile/tile_compile_cpp/apps/runner_phase_registration.cpp#L470)
  soll `get_registration_proxy` verwenden.
- [runner_phase_registration.cpp](/media/data/programming/tile_compile/tile_compile_cpp/apps/runner_phase_registration.cpp#L587)
  soll fuer die Rescue keine Proxies mehr selbst erzeugen.

Validierung:

- Anzahl physischer FITS-Reads pro Phase loggen
- Cache-Hit-Rate pro Produkttyp loggen
- Peak-RAM vor und nach der Aenderung vergleichen
- sicherstellen, dass Referenzframe, Ranking und Registrierungsresultat
  numerisch unveraendert oder nur im Rundungsrauschen veraendert bleiben

Konservative Speedup-Schaetzung:

- `REGISTRATION`: 1.2x bis 1.6x
- `NORMALIZATION + GLOBAL_METRICS`: 1.2x bis 1.5x
- Gesamtpipeline: typischerweise 10 bis 25 Prozent, bei I/O-lastigen Runs mehr

### Prioritaet 2: FITS-I/O-Hotpath verschlanken

Zielbild:

- Header und Pixel sollen getrennt abrufbar sein.
- Die Hotpaths sollen keine Headerarbeit bezahlen, wenn nur Pixel gebraucht
  werden.
- Die Ueberfuehrung der gelesenen Pixel in `Matrix2Df` soll als Bulk-Copy
  erfolgen, nicht als elementweise Schleife.

Empfohlene API-Erweiterung:

- in [fits_io.hpp](/media/data/programming/tile_compile/tile_compile_cpp/include/tile_compile/io/fits_io.hpp)
  neue Funktionen einfuehren:
  - `FitsImageInfo read_fits_info(...)`
  - `Matrix2Df read_fits_pixels_float(...)`
  - `std::pair<Matrix2Df, FitsHeader> read_fits_float_full(...)`
- den bisherigen Aufruf `read_fits_float(...)` zunaechst als kompatiblen Wrapper
  behalten, damit der Umbau schrittweise erfolgen kann

Konkrete Eingriffe:

- [fits_io.cpp](/media/data/programming/tile_compile/tile_compile_cpp/src/io/fits_io.cpp#L177)
  in einen internen gemeinsamen Loader aufteilen
- [fits_io.cpp](/media/data/programming/tile_compile/tile_compile_cpp/src/io/fits_io.cpp#L284)
  von elementweiser Kopie auf `std::copy_n` oder `std::memcpy` umstellen, wenn
  die Speicherform von `Matrix2Df` dort garantiert contiguous ist
- alle Hotcaller in:
  - [runner_phase_metrics.cpp](/media/data/programming/tile_compile/tile_compile_cpp/apps/runner_phase_metrics.cpp#L86)
  - [runner_phase_metrics.cpp](/media/data/programming/tile_compile/tile_compile_cpp/apps/runner_phase_metrics.cpp#L338)
  - [runner_phase_registration.cpp](/media/data/programming/tile_compile/tile_compile_cpp/apps/runner_phase_registration.cpp#L277)
  - [runner_pipeline.cpp](/media/data/programming/tile_compile/tile_compile_cpp/apps/runner_pipeline.cpp#L596)
  schrittweise auf Pixel-only umstellen

Wichtige Guardrails:

- Der neue API-Schnitt darf nicht stillschweigend Headerdaten verlieren, die
  spaetere Phasen wirklich benoetigen.
- Die erste Migration soll nur die klaren Hotpaths umstellen. Diagnose- und
  Exportpfade koennen vorerst beim Full-Read bleiben.
- Vor dem Bulk-Copy muss explizit geprueft werden, dass `Matrix2Df` wirklich in
  der erwarteten Row-Major-Form und ohne Padding vorliegt.

Validierung:

- alter und neuer Loader muessen fuer dieselben Dateien pixelgenau dasselbe
  Bild liefern
- Header-Felder, die heute verwendet werden, muessen in Regressionstests oder
  Vergleichslogs abgesichert werden
- pro 1000 Frame-Loads die mittlere Ladezeit messen

Konservative Speedup-Schaetzung:

- Frame-Load-Hotpath: 15 bis 35 Prozent schneller
- Gesamtpipeline: meist 5 bis 15 Prozent, auf langsamer Disk mehr

### Prioritaet 3: `COMMON_OVERLAP` in `PREWARP` integrieren

Zielbild:

- Die Information "welche Canvas-Pixel waren in wie vielen Frames valide" soll
  bereits im `PREWARP`-Schritt entstehen.
- Die bisherige `COMMON_OVERLAP`-Phase soll danach nur noch aus einer
  Finalisierung bestehen:
  - Coverage-Map reduzieren
  - Schwellwert anwenden
  - FITS-Maske und JSON-Artefakte schreiben

Empfohlene technische Umsetzung:

- in [runner_phase_registration.cpp](/media/data/programming/tile_compile/tile_compile_cpp/apps/runner_phase_registration.cpp#L1324)
  je Worker eine lokale Coverage-Map anlegen
- Datentyp:
  - `uint16_t`, solange die maximale Framezahl sicher darunter bleibt
  - sonst `uint32_t`
- jeder Worker zaehlt nur die Pixel hoch, die durch das gewarpte Frame wirklich
  belegt werden
- nach Abschluss von `PREWARP` werden die Worker-Maps in eine globale
  Coverage-Map reduziert
- die bisherige Logik aus
  [runner_pipeline.cpp](/media/data/programming/tile_compile/tile_compile_cpp/apps/runner_pipeline.cpp#L707)
  wird auf einen reinen Finalisierungspfad umgebaut

Wichtige Implementierungsdetails:

- Keine atomischen Pixel-Inkremente im Hotpath. Das waere auf grossen Canvas
  sofort wieder ein Bottleneck.
- Der Update-Pfad soll nur die reale Bounding-Box des gewarpten Frames
  bearbeiten, nicht die volle Canvas.
- Wenn `PREWARP` ohnehin eine Valid-Maske oder Zeilenbegrenzung kennt, sollte
  genau diese Information fuer die Coverage uebernommen werden.
- Die Crop-/Offset-Logik nach `PREWARP` muss mit der Coverage-Map synchron
  bleiben; sonst verschiebt sich spaeter die Canvas-Maske.

Validierung:

- Anzahl gueltiger Pixel in `common_valid_mask` alt gegen neu vergleichen
- identische Maskengroesse und identische Bounding-Box auf Referenzruns
- Laufzeit von `PREWARP + COMMON_OVERLAP` gemeinsam messen, nicht nur einzeln

Konservative Speedup-Schaetzung:

- `COMMON_OVERLAP`: 2x bis 10x, weil der teure Vollscan verschwindet
- `PREWARP + COMMON_OVERLAP` zusammen: 15 bis 40 Prozent
- Gesamtpipeline: stark canvas-abhaengig, oft 5 bis 20 Prozent

### Empfohlene Umsetzungsreihenfolge fuer einen Sprint

Fuer geringes Risiko und schnelle Rueckmeldung ist diese Reihenfolge sinnvoll:

1. FITS-Hotpath aufspalten, aber alte API als Wrapper belassen
2. `COMMON_OVERLAP` in `PREWARP` integrieren
3. runner-lokalen Frame-/Proxy-Cache zuerst nur in `REGISTRATION` einfuehren
4. Cache auf `NORMALIZATION` und `GLOBAL_METRICS` erweitern
5. danach erst den allgemeinen Loaderpfad in spaeteren Phasen umhaengen

Begruendung:

- Schritt 1 ist methodisch neutral und leicht benchmarkbar.
- Schritt 2 entfernt einen klar isolierten Vollscan.
- Schritt 3 und 4 bringen den groessten Gesamtnutzen, veraendern aber mehr
  Kontrollfluss und sollten daher auf bereits stabilisiertem I/O aufsetzen.

### Mess- und Abnahmekriterien fuer die ersten drei Punkte

Ein Sprint fuer Prioritaet 1 bis 3 sollte erst als abgeschlossen gelten, wenn
die folgenden Kriterien zusammen erfuellt sind:

- mindestens ein kleiner MONO-Run und ein grosser OSC-Run laufen ohne
  Regression durch
- die Phasenzeiten fuer `NORMALIZATION`, `GLOBAL_METRICS`, `REGISTRATION`,
  `PREWARP` und `COMMON_OVERLAP` sind separat vor/nachher dokumentiert
- die Zahl der echten FITS-Reads sinkt nachweisbar
- Referenzwahl, Registrierungsquote und finale Canvas-Maske bleiben fachlich
  stabil
- Peak-RAM bleibt innerhalb eines bewusst gesetzten Budgets

## Konkrete Roadmap

### Schritt 1

- FITS-Pixel-Hotpath einfuehren
- `COMMON_OVERLAP` in `PREWARP` integrieren
- `write_registered_frames` als Debug-Pfad absichern

Ziel:

- sofort messbarer Gewinn bei geringem Risiko

### Schritt 2

- gemeinsamen Frame-/Proxy-Cache einfuehren
- `NORMALIZATION`, `GLOBAL_METRICS`, `REGISTRATION` auf denselben Cache
  umstellen

Ziel:

- grossen Teil des mehrfachen FITS-I/O entfernen

### Schritt 3

- `LOCAL_METRICS` und Post-Warp-Diagnostik abspecken
- OLA-Akkumulation parallelisieren

Ziel:

- Phase 8/9 auf Mehrkernsystemen besser skalieren

### Schritt 4

- OSC-Debayer-Pipeline umbauen
- `SYNTHETIC_FRAMES` ohne doppelte Rekonstruktion implementieren

Ziel:

- maximale Beschleunigung fuer grosse OSC-Datensaetze

## Messplan vor der Umsetzung

Vor jeder groesseren Aenderung sollte ein einfacher Phasen-Benchmark mit
identischem Datensatz gefahren werden.

Zu messen:

- Gesamtzeit pro Phase
- Anzahl gelesener FITS-Dateien pro Phase
- Peak-RAM
- Schreibvolumen auf Disk
- Tile/s in Phase 9
- Frames/s in `NORMALIZATION`, `GLOBAL_METRICS`, `REGISTRATION`, `PREWARP`

Empfohlene Testklassen:

- kleiner MONO-Datensatz
- grosser OSC-Datensatz ohne Canvas-Expansion
- grosser OSC-Datensatz mit Canvas-Expansion / Feldrotation
- Run mit aktivierten Synthetic Frames

## Erwartete Reihenfolge des Gesamtnutzens

Wenn nur die wahrscheinlich wirkungsvollsten Punkte umgesetzt werden sollen,
ist diese Reihenfolge sinnvoll:

1. gemeinsamer Frame-/Proxy-Cache
2. FITS-I/O-Hotpath
3. `COMMON_OVERLAP`-Vollscan entfernen
4. OSC-Debayer-Strategie in Phase 9
5. `SYNTHETIC_FRAMES`-Doppelarbeit entfernen

## Offene Punkte

- Die exakten Zeitanteile pro Phase sollten per Laufzeit-Profiling bestaetigt
  werden.
- Vor allem bei Proxy-basierten Optimierungen muss geprueft werden, ob sich
  Referenzwahl, Frame-Ranking oder Registrierung messbar veraendern.
- Fuer OSC muss jede Optimierung CFA-sicher bleiben; Performance darf nicht zu
  Bayer-Phasenfehlern fuehren.
