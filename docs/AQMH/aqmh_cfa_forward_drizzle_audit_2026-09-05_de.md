# Audit: CFA-Forward-Drizzle und Mehrband-Implementierungsplan

**Stand:** 2026-09-05, aktueller Worktree einschließlich vorhandener uncommitteter Änderungen.  
**Prüfgegenstand:** [Implementierungsplan](aqmh_cfa_forward_drizzle_multiband_implementierungsplan_de.md), neue Core-Module, Runneranbindung, Konfigurations- und Testverträge.  
**Ergebnis:** M0 und M1 sind teilweise implementiert, aber nicht abgenommen. M2 besitzt einen brauchbaren Uniform-Prototyp; seine geometrische und infrastrukturelle Abnahme ist offen. Vor M3 müssen die unten genannten Korrektheitsprobleme behoben werden.

Dieses Audit ändert keine Algorithmen, Defaults oder Run-Daten. Reale Qualitätsmessungen aus dem Plan wurden nicht erneut ausgeführt. Codebefunde sind durch statische Prüfung belegt; neu ausgeführte Tests sind am Ende separat aufgeführt. Historische Erfolgsbehauptungen des Plans gelten nicht als unabhängiger Nachweis.

## 1. Kritische Codebefunde

### A1 — Coverage verwendet nicht den Rekonstruktionskernel (P0)

**Beleg:** `tile_compile_cpp/src/registration/sampling_geometry.cpp:103–164`, dagegen `src/reconstruction/forward_drizzle.cpp`, Funktionen `build_affine_leaf`, `subdivide_local` und `accumulate_leaf`. Vertrag: Plan §§9.2, 11.6, 20.2/23.

Coverage mappt nur das Samplezentrum und markiert ein achsparalleles Quadrat mit fester Breite `pixfrac * internal_scale`. Rotation, Skalierung und lokale Formänderung des Droplets fehlen. M2 mappt dagegen die Ecken und berechnet Polygon-Rechteck-Schnittflächen.

Die Begründung in §30.12, Coverage brauche nur „berührt oder nicht“, reicht nicht: Gerade die berührten Pixel ändern sich mit der Dropletform. Die M1-Maske ist weder generell identisch mit noch eine garantierte Obermenge des M2-Supports. Gateentscheidungen und die geplante Prüfung „exakter Support ist Teilmenge der geometrischen Maske“ sind dadurch unzuverlässig.

**Nächster Schritt:** Einen gemeinsamen geometrischen Rasterisierer für Coverage und Wertakkumulation verwenden; bildwertfreie Coverage muss dieselben positiven Schnittflächen erhalten. Lokale Ausfälle und Frameausschlüsse müssen dieselbe Geometriepopulation erzeugen. Vergleichsfixtures für Rotation 45°, Skalierung, Scherung und gekrümmte lokale Warps müssen beide Pfade pixelweise vergleichen.

### A2 — Das gemeldete geometrische `n_eff` ist eine Frame-Anzahl (P0)

**Beleg:** `sampling_geometry.cpp:455–481`. Dort wird das Perzentil von `support_count_*` berechnet und als `min_channel_n_eff_p10` ausgegeben. Vertrag: §§9.2/6, 11.10.

Gefordert ist `(sum B_f)^2 / sum(B_f^2)` mit den tatsächlichen geometrischen Framegewichten. Die Anzahl berührender Frames entspricht dem nur bei gleichen Gewichten. Beispiel: `B=(1, 0.01, 0.01)` liefert drei berührende Frames, aber nur `n_eff≈1.04`. Das aktuelle Gate kann solche Pixel fälschlich als ausreichend gestützt bewerten.

**Nächster Schritt:** Pro Frame und Kanal zuerst sämtliche Schnittflächen zu `B_f,c` aggregieren, dann `sum B`, `sum B²` und getrennt den Framecount führen. Ein Fixture mit stark ungleichen Überdeckungsflächen muss Coverage-`n_eff` und Uniform-`n_eff` vergleichen und den bisherigen Fehler sichtbar machen.

### A3 — Gate und Analysemaske haben verschiedene Bezugsflächen (P0)

**Beleg:** `sampling_geometry.cpp:412–481`. `analysis_common_mask` ist ein kanalweiser Schnitt mit Mindestframeanteil; `gate.analysis_pixels`, Supportanteile und Perzentile verwenden stattdessen `union_support`. Plan §§6.2 und 9.3 verlangen die Analysemaske als Bezugsbereich.

Das Gate kann damit genug Analysepixel melden, obwohl die tatsächlich persistierte Analysemaske leer ist. Die Änderung von `common_overlap_required_fraction` verändert die Maske, aber nicht diese Gatewerte. §30.10 wertet das leere Ergebnis als unproblematischen Defaultbefund; tatsächlich ist es zusammen mit der abweichenden Gatepopulation ein Vertragsfehler.

**Auch der Plan selbst braucht eine Korrektur:** Wird der Analysebereich bereits aus „mindestens ein Beitrag in jedem Kanal“ gebildet, beträgt der Kanalsupport darin definitionsgemäß 100 %. Das anschließende 99,5-%-Gate prüft dann keine Löcher. Ein bloßes Austauschen von `union_support` gegen `analysis_common_mask` löst die konzeptionelle Lücke nicht.

**Festzulegende Semantik:** Eine geometrische Referenzfläche aus den transformierten vollständigen Frame-Footprints bestimmen, unabhängig vom dünnen CFA-Droplet-Support. Darin je Kanal Supportanteil, echte `n_eff`-Perzentile und Löcher prüfen. Die tatsächlich rekonstruierbare Maske und die spätere Validierungsmaske davon getrennt ableiten. Randbehandlung und erlaubter Flächenverlust gehören ausdrücklich in diesen Vertrag; keine versteckte Auswahl ausschließlich gut belegter Pixel.

### A4 — Lokale Subdivision erfüllt die behauptete Abnahme nicht (P0)

**Beleg:** `forward_drizzle.cpp:324–370`, Tests `test_forward_drizzle.cpp:203–255`. Vertrag: §11.6 und die als bestanden markierte M2-Abnahme.

An der Maximaltiefe akzeptiert der Code ein Blatt allein aufgrund des Positionsfehlers. Die Flächenprüfung wird in diesem Zweig übersprungen. Außerdem zählt `frame_discarded` fehlgeschlagene rekursive Subdroplets, während `frame_total` Quellsamples zählt. Mehrere Fehler desselben Samples können deshalb mehrfach in die als Samplequote bezeichnete Frameausschlussrate eingehen. Teilweise erfolgreiche Kinder werden mit `c0 || c1 || c2 || c3` weitergereicht.

Die bisherigen Tests verwenden ein lokales Nullmodell und ein vollständig nichtinvertierbares Modell. Sie belegen weder Konvergenz bei tatsächlicher Krümmung noch den Grenzfall „Position bestanden, Fläche verletzt“, noch die korrekte 0,1-%-Samplequote.

**Nächster Schritt:** Flächenkonvergenz auch an der letzten akzeptierbaren Stufe nachweisen, etwa durch eine zusätzliche reine Prüfstufe ohne weitere akzeptierte Unterteilung. Verwerfungszähler für Subdroplets und betroffene Quellsamples trennen. Verbindlich entscheiden, ob ein partiell defektes Sample vollständig verworfen wird; für einen klaren Samplequotenvertrag ist vollständiges Verwerfen die einfachere Referenz. Gekrümmte positive und negative Grenzfixtures ergänzen, dann erst diesen Abnahmepunkt schließen.

### A5 — Pixelzentrum-Adapter fehlt an der Registrierungsschnittstelle (P1)

**Beleg:** `runner_phase_registration.cpp:393–397` übernimmt die vorhandene OpenCV-Inverse unverändert. Der neue Kern wertet sie dagegen mit Zentren `x+0.5, y+0.5` aus. Die bestehende Registrierung verwendet OpenCV-Indexkoordinaten; siehe `src/registration/global_registration.cpp`, insbesondere die `WARP_INVERSE_MAP`-Verträge. Plan §11.5 verlangt eine einzige explizite Adaptergrenze.

Für eine Inverse `s_cv = A*q_cv + t` muss bei kantenbasierten Koordinaten gelten:

```text
h = (0.5, 0.5)
s_edge = A*q_edge + t + h - A*h
```

Diese Translation fehlt in der geprüften Übernahme. Reine Translation und Identität verdecken den Fehler, Rotation und Skalierung nicht. Bei lokalen Modellen müssen auch Auswertekoordinaten und Offsets konsistent konvertiert werden. Ein Round-trip mit derselben falsch interpretierten Matrix entdeckt den Fehler ebenfalls nicht.

**Nächster Schritt:** Adapter explizit implementieren und gegen bekannte OpenCV-Kontrollpunkte mit Rotation, Skalierung und Canvasoffset testen. Der integrierte Zentroidnachweis ist offen; eine reale Größenordnung des Bildfehlers wurde in diesem Audit nicht gemessen.

### A6 — Persistenz und Resume sind noch keine belastbaren Verträge (P1)

**Belege:** `runner_phase_registration.cpp:454–464, 667–688`; `src/core/utils.cpp:177`; `src/registration/registration_sampling_plan.cpp:275–346, 381–430`; `apps/runner_resume.cpp:796–832`; `core/pipeline_contract.hpp:39–40`.

- Die als atomar beschriebenen JSON-Writes verwenden `core::write_text`: direktes Öffnen/Trunkieren, kein temporärer Commit und keine Prüfung des finalen Schreibzustands.
- Der Coveragehash umfasst derzeit Planhash, Kernel, gerundeten `std::to_string(pixfrac)` und interne Skala. Überlappungsanteil, Subdivisionsvertrag und dessen Version fehlen trotz §18.3.
- Frame-IDs sind Dateibasennamen; Quellinhaltsidentität und Normalized-Manifesthash sind nicht in den Sampling-Plan eingebunden. Gleichnamige oder inhaltlich ersetzte Quellen sind damit nicht über diesen Vertrag abgesichert.
- Der JSON-Lader übernimmt Hash und redundante Inversen ohne vollständige semantische Verifikation; fehlende Werte erhalten zum Teil Defaults. Ein Serialisierungs-Round-trip beweist keine sichere Ablehnung beschädigter Artefakte.
- Die neuen Phasen-/Store-Resume-Einstiege fehlen. `pipeline_contract_is_single_method(version)` akzeptiert außerdem jede zukünftige Version `>=1`, nicht nur explizit unterstützte Versionen.

**Planinterne Unstimmigkeit:** §7.4 definiert einen reinen Geometriehash, §18.3 einen `sampling_plan_hash` mit `source_identity_hash`. Entweder zwei Hashes mit verschiedenen Namen definieren oder die Domäne vereinheitlichen.

**Nächster Schritt:** Hash-DAG und Versionskompatibilität vor Storeimplementierung festziehen; manifestgebundene Quellidentitäten, strikten Loader und transaktionale Artefaktwrites ergänzen. Beschädigung, Schreibabbruch, veränderte Quellen und unbekannte Versionen müssen fail-closed getestet werden.

## 2. Weitere Lücken in Annahmen und Planung

### B1 — Mehr Frames heilen keine „alle Frames“-Schnittmaske

§30.10 vermutet, der Default `1.0` werde bei Hunderten OSC-Frames eher erreichbar. Für einen festen Pixelbereich gilt das Gegenteil: Beim Hinzufügen eines Frames kann eine Schnittmenge aller Frame-Supportmasken nur gleich bleiben oder schrumpfen. Mehr Frames verbessern die Vereinigungsdeckung und meist die statistische Stützung, nicht den Schnitt aller Masken.

Die genannten 16 % sind zudem `0.25 * pixfrac²` bei `pixfrac=0.8`, also kontinuierliche R/B-Dropletfläche für flächentreue Warps. Das ist nicht der Anteil diskreter Zielpixel, die irgendeine positive Schnittfläche erhalten. Diese Größen dürfen nicht gleichgesetzt werden. Ein neuer Default `0.5–0.7` ist deshalb aus den bisherigen Befunden nicht allgemein ableitbar. Zuerst A1–A3 lösen, anschließend Defaults anhand synthetischer OSC-/MONO-Fixtures begründen; M9 bestätigt sie auf realen Daten.

### B2 — Die Ursachenanalyse ist plausibel, aber zu absolut formuliert

§2.1 erklärt Registrierungsresiduen für widerlegt und 19–25 % Schärfegewinn für rückgewinnbar. Gleiche Elongation schließt isotrope Registrierungsfehler nicht aus: Eine isotrope Streuung der Positionsfehler verbreitert eine PSF in beiden Achsen. Auch der Vergleich von Median-FWHM verschiedener Einzelframes mit einer Stack-FWHM ist kein Identitätsbeweis für perfekte Registrierung.

Quadratische FWHM-Differenzen sind nur unter passenden PSF-/Faltungsannahmen als unabhängige Beiträge interpretierbar. Hier ändern sich unter anderem Fitterpopulation, bilineares gegenüber edge-aware Debayer und teils der Lauf. Der native Quincunx-Fit ist ohne Biasnachweis keine garantierte Obergrenze. Forward-Drizzle bringt selbst einen Dropkernel und beim Default 2/1 einen weiteren Integrationsschritt mit.

**Konsequenz:** Die Zahlen als hypothesengestützte Zielspanne kennzeichnen, nicht als zugesicherten Gewinn. Vorhandene Datenbefunde weiter nutzen; ein neuer Siril-Kreuztest ist dafür nicht nötig. Synthetische Truth-Fixtures mit bekannten Positionsfehlern und einheitlicher Sternpopulation sind der nächste belastbare Nachweis. Die reale M9-Abnahme muss den ausgelieferten 2/1-Pfad zusätzlich zum internen 2/2-Schärfenachweis bewerten.

### B3 — M1 hat bereits ein erhebliches RAM-Problem

`sampling_geometry.cpp:353–407` erzeugt pro Worker vollbildgroße Kanalcounts und Touched-Masken; Workerzahl standardmäßig bis zur Hardwareparallelität bzw. Framezahl. Bei 33,5 Millionen internen Pixeln benötigt OSC allein dafür etwa `3*(2+1)*33.5e6 ≈ 302 MB` pro Worker, also rund 6 GB bei 20 Workern, vor gemeinsamen Arrays und Auswertung.

Im Uniform-Kern entstehen bei OSC neun Double-Summenebenen, sechs frame-lokale Double-Ebenen und drei Ergebnisplanes mit je 13 Byte/Pixel. Das sind ungefähr 159 Byte je internem Pixel: bei 24 MP nativ und 2x etwa 15,3 GB, ohne Quellbild und übrigen Runner. Das ist eine statische Speicherabschätzung, keine RSS-Messung.

Die Behauptung in §30.11, Chunking sei für M1 wegen der Iteration über Quellpixel gegenstandslos, ignoriert diese Zielpuffer, deren vollständiges Rücksetzen und deren Reduktion. Auch M1 benötigt ein Budget und eine begrenzte Partitionierung. Der Durchsatzvergleich nur gegen eine erst in M8 eingefrorene Baseline verhindert zudem Regressionen, garantiert aber keine akzeptable anfängliche Laufzeit.

### B4 — Legacy-Isolation und Phasenstatus sind überzeichnet

Plan §1.1 verlangt ein standardmäßig deaktiviertes Testtarget ohne Resume-/Run-Schreibzugriff. Tatsächlich ist `TILE_COMPILE_BUILD_LEGACY_REFERENCE` in `CMakeLists.txt:564` standardmäßig `ON`; `tile_compile_legacy_reference` kompiliert sämtliche Runnersourcen mit umgangenem Cutover-Lock. Das ist ein ausführbarer Runner inklusive Run-/Resume-Funktionen. Er ist nicht installiert, aber auch kein technisch auf Testfixtures beschränkter Vergleichskern. „Frozen“ bezeichnet hier keine eingefrorenen Quellen: Beide Binaries verwenden dieselben veränderlichen Dateien und dieselbe Bibliothek.

`Phase` enthält weiterhin die Altphasen, aber keine `SAMPLING_GEOMETRY`-/`FORWARD_DRIZZLE`-/`MULTIBAND`-Werte. Die Geometriefunktion liegt in der PREWARP-Phase; ihr Fehler wird als `Phase::PREWARP` gemeldet. Deshalb ist „eigene Phase implementiert“ in §30.8 nicht als Event-/Resumevertrag erfüllt.

Die Preview in §30.13 schreibt nur Diagnostikzahlen, keinen fertigen Uniform-Output und keinen Phasecommit. Sie ist ein Integrations-Smoke-Test. Ihr bestandener Legacy-Lauf trotz Gatefehler beweist nicht die Ausführung im aktiven Runner: Dieser bleibt aktuell gesperrt, und sein Gatefehlerzweig liegt vor dem Previewaufruf.

**Konsequenz:** Vergleichstarget standardmäßig deaktivieren und Testzugriff technisch auf neue temporäre Fixtures beschränken, oder den Rolloutvertrag ausdrücklich ändern. Die Cutover-Sperre bis zur fertigen neuen Pipeline beibehalten. Neue Phasen im Modell, Eventstrom, Backend und Resume einführen; keinen bloßen Funktionsnamen als Phasenabnahme zählen.

### B5 — Abnahmebehauptungen reichen weiter als die Tests

Der affine Flächentest prüft `sum(weight_sum)` eines einzelnen Samples. Das beweist die geometrische Flächenidentität, aber keine allgemeine Erhaltung von Bildoberflächenhelligkeit, Aperturflux, Zentroid oder Farbsymmetrie. Der lokale Nullmodelltest beweist keine adaptive Konvergenz unter Krümmung. Die bestehende grüne Suite kann A1–A5 daher gleichzeitig enthalten.

§21 sollte für die synthetische Frameerzeugung ausdrücklich Sensor-Pixelintegration, PSF-Faltung und eine vom Drizzlekernel unabhängige Referenz verlangen. Sonst kann ein zu ähnlicher Hin-/Rückoperator unrealistisch gute Resultate erzeugen. Die Vorgabe „ohne plattformabhängige Transzendenten“ passt außerdem nicht ohne Präzisierung zu analytischen Gauß-/Moffat-Profilen und beliebigen Rotationen; referenzierte Daten oder klar tolerierte Mathematik sind praktikabler.

### B6 — Dokumentations- und Terminwidersprüche

- Kopfzeile: „Implementierung noch nicht begonnen“; §30.12: „M1 vollständig abgeschlossen“; §23: wesentliche M1-Abnahmepunkte offen. Maßgeblich muss eine einzige evidenzgebundene Statusmatrix sein.
- §1.1 verschiebt reale Läufe auf M9, §30 dokumentiert solche Läufe bereits in M1/M2. Diese historischen Läufe wurden hier nicht nachgeprüft. Frühe ausdrücklich angeforderte Diagnoseversuche brauchen eine klar benannte Ausnahme und ersetzen keine M9-Promotion.
- §6.1 bezeichnet seine Parameterliste als vollständig, enthält aber die später ergänzte Previewoption nicht. Default-YAML, Beispiele und öffentliche Konfigurationsreferenzen sind noch nicht auf den neuen Vertrag umgestellt.
- „317/318 grün“ ist keine vollständig grüne Suite. Ein bekannter Fehler bleibt ein dokumentierter Fehler oder eine explizite Umgebungsbeschränkung. Im aktuellen Audit besteht die Hauptsuite tatsächlich vollständig; siehe unten.

## 3. Tatsächlicher Implementierungsstatus

| Meilenstein | Stand | Für die Abnahme fehlend |
|---|---|---|
| M0 | Teilweise umgesetzt | vollständige Konfigurations-/Dokumentationsumstellung, strikte Legacy-Isolation, präziser Versions-/Identitätsvertrag |
| M1 | Geometrieprototyp und Artefaktintegration vorhanden; nicht abgenommen | gemeinsamer Kernel, echtes `n_eff`, widerspruchsfreie Masken, Koordinatenadapter, Budget, echte Phase/COMMON_OVERLAP-Anbindung |
| M2 | Affiner Uniform-Kern, CFA/MONO-Zuordnung, Frameaggregation und Preview vorhanden | lokale Konvergenzkorrektur, photometrische und Zentroidfixtures, Stores, Chunking, Commit, neuer Runnerpfad |
| M3 | Neuer Source-Quality-/Clipping-/Raw-Pfad noch nicht umgesetzt | bestehende PREWARP-AQMH-Bausteine sind nur mögliche Extraktionsquellen |
| M4 | `internal_scale=2` rechnerisch im Kern vorhanden | 2/1-Downsampling, 4/4-Support, WCS und Downstreamabnahme fehlen |
| M5–M7 | Neue Source-Skalenstreams, Mehrbandpipeline und neue CUDA-Implementierung noch offen | vorhandenes AQMH/CUDA ist keine Abnahme der neuen Methode |
| M8 | Einzelne CLI-/Backendvorarbeiten vorhanden | durchgängige GUI-, Report-, Phasen- und Dokumentationsumstellung |
| M9 | Historische Diagnostik vorhanden; keine neue Gesamtpipeline abgenommen | kontrollierte Qualitäts-/Ressourcenmatrix nach Core- und Integrationsabnahme |
| M10–M11 | Offen | Cutover beziehungsweise spätere endgültige Legacy-Löschung |

## 4. Verbindliche Reihenfolge der nächsten Arbeitspakete

Die folgende Reihenfolge ist die aus diesem Audit abgeleitete Umsetzungsvorgabe. Sie autorisiert für sich keine realen Bildverarbeitungsläufe.

| Reihenfolge | Arbeitspaket | Prüffähiges Abschlusskriterium |
|---|---|---|
| 1 | **M1/M2-Korrektheitsvertrag reparieren:** A1–A5, Maskenreferenzfläche und Fehlerquoten festlegen; Status korrigieren | Zunächst fehlschlagende Gegenbeispiele für Kernelabweichung, ungleiche Gewichte, leere Maske, gekrümmten Warp und Koordinatenadapter; danach dieselben Tests bestanden |
| 2 | **Uniform-Referenz beweisen:** unabhängige photometrische OSC-/MONO-Fixtures | Konstante Flächen, Aperturflux, Zentroid, alle Bayer-Pattern/Ursprünge, negative Werte, NaN/Inf und Frameausfälle korrekt; Coverage und Uniform stimmen geometrisch überein |
| 3 | **Persistenz und Ressourcen fertigstellen:** A6/B3, transaktionale Profilstores und Chunking für Geometrie und Uniform | identische Resultate bei Chunkhöhe 1/Auto/Vollbild und 1/N Workern innerhalb festgelegter Toleranz; Budgetprüfung; Crash-/Disk-full-/Hashinvalidierungsfixtures ohne gültigen Teilcommit |
| 4 | **M0–M2 durchgehend integrieren:** neue Phasen, normalisierte Quelle, COMMON_OVERLAP, Legacy-Grenze | temporärer Integrationstest ohne PREWARP-Nutzsignal und ohne Backendstart; vollständiger Uniform-Output; Event-/Artefakt-/Resume-Vertrag stimmig; Produktlock bis zum freigegebenen Stand erhalten |
| 5 | **M3: gemeinsames Clipping und Raw** | gleiche Ausreißermaske für alle Profile; source-space Composite-Q ohne Classic-Metriken; Raw unveränderlich; Uniform als geometrische Kontrolle |
| 6 | **M4–M6: Ausgabegeometrie, Skalenprofile und Mehrband** | 2/1 und 2/2 samt WCS/Masken geprüft; gemeinsame Sternpositionen; Promotion gegen Uniform und Raw, N/A korrekt behandelt |
| 7 | **M7/M8 und anschließend M9** | neue CPU-/CUDA-Parität mit echter GPU, vollständige Produktdokumentation und UI; danach ausschließlich ausdrücklich angeforderte reale Qualitätsläufe |

**Nächster konkreter Entwicklungsschritt ist Arbeitspaket 1, nicht Parallelisierung der aktuellen Näherung und nicht Mehrbandfusion.** Performancearbeit folgt auf einem korrigierten geometrischen Referenzkern; das Speicherdesign wird dabei bereits berücksichtigt. M0/M1 bleiben offen, bis ihre tatsächlichen Abnahmekriterien erfüllt oder ausdrücklich neu zugeordnet sind.

## 5. Neu ausgeführte Verifikation

- Bestehender Release-Build: `tile_compile_runner` und `tests` erfolgreich gebaut.
- Fokussierte Core-/Coverage-/Konfigurationstests: **24/24 Testfälle, 1.927 Assertions bestanden**.
- Vollständige Hauptsuite: **318/318 Testfälle, 32.106 Assertions bestanden**.
- Separates `tile_compile_legacy_reference_tests`-Target gebaut und ausgeführt: **16 bestanden, 2 native CUDA-Tests übersprungen; 226 Assertions bestanden**. Die Testumgebung meldet keinen zugänglichen CUDA-fähigen Device; daraus folgt keine Aussage über die physisch vorhandene Hardware.
- Logs: `/tmp/out_aqmh_audit_build.txt`, `/tmp/out_aqmh_audit_focused.txt`, `/tmp/out_aqmh_audit_full.txt`, `/tmp/out_aqmh_audit_legacy_build.txt`, `/tmp/out_aqmh_audit_legacy_tests.txt`.

Die erfolgreiche Hauptsuite reproduziert den im Plan erwähnten einzelnen GPU-Fehler in dieser Umgebung nicht. Zusammen mit den übersprungenen Legacy-CUDA-Tests ist das ausdrücklich kein GPU-Paritätsnachweis. Eine neue CUDA-Drizzle-Implementierung fehlt ohnehin noch. Es wurden keine Backenddienste und keine realen Bildverarbeitungsläufe gestartet. Die beschriebenen Fehlerkorrekturen sind nächste Arbeitspakete, keine in diesem Audit bereits vorgenommenen Codeänderungen.


## 6. Umsetzung und Nachprüfung der zwischenzeitlichen Ergänzungen (2026-09-05)

Abschnitte 1–5 dokumentieren den ursprünglichen Befundstand. Die dort genannten
„nächsten Schritte“ sind durch die folgende Umsetzung teilweise erledigt.
Der aktuelle Vorrang bleibt OOM-Vermeidung; es wurde kein Bildverarbeitungslauf
oder Backend gestartet. Frühere reale Läufe in Fortschrittsnotizen wurden hier
nicht wiederholt.

### Erledigte ursprüngliche Befunde

A1–A5 sind im Referenzkern korrigiert: gemeinsamer Polygonrasterisierer, echtes
geometrisches `n_eff`, unabhängige dichte Analysefläche, Sample-weite lokale
Konvergenzprüfung und Pixelzentrum-Adapter. Coverage und Uniform arbeiten in
budgetierten Streifen. Coverage verwendet temporäre Dateien für exakte
Perzentile und Scanlines für die Lochsuche. Geometrieartefakte haben strikte
Loader, Versionsprüfung und atomare Einzeldatei-Veröffentlichung. A6 bleibt für
vollständige Store-/Resume-Verträge offen; M0–M3 sind nicht pauschal abgenommen.

### Neue Befunde und Korrekturen

| Priorität | Zwischenzeitlich eingeführter Fehler | Umsetzung |
|---|---|---|
| P0 | Gemeinsames Uniform/Raw-Clipping allokierte zwei Vollausgaben ohne Budgetanrechnung; Kandidatenlisten wurden erst nach dem Aufbau geprüft. Dithering begrenzt die Zahl beitragender Frames nicht zuverlässig. | `stream_forward_drizzle_uniform_and_raw` mit flachem, vorab bemessenem Kandidatenpuffer; Worst Case aller Frames pro Pixel/Kanal. Auto-Streifenhöhe berücksichtigt Kandidaten, zwei Ausgabestreifen, Akkumulatoren und Clipping-Scratch. Die Komfortfunktion rechnet beide Vollausgaben zusätzlich an. Zu große explizite Chunks scheitern vor Quell-I/O. |
| P0 | `GLOBAL_QUALITY` hielt alle Vollbild-Proxys zugleich. | Ein Proxy pro Frame; nur skalare Metriken bleiben erhalten. Provider-Überladung erlaubt Quellen nacheinander aus dem Cache zu laden. Die bestehende Vektor-Überladung bleibt verfügbar. MAD nutzt seinen Wertebuffer erneut. |
| P0 | Der neue Uniform-FITS-Store erzeugte beim Export zusätzliche Vollbildmatrizen, für Support sogar zwei float-Kopien. | Zeilenweiser float-/Bytemaskenexport; Quellcache wird vor dem Schreiben freigegeben. Der Store selbst materialisiert weiterhin ein budgetiertes Uniform-Ergebnis und lehnt zu große Bilder ab. |
| P1 | `G_eff` konnte durch ungültige Werte oder ungültige Quellindizes undefinierten Zugriff bzw. unbrauchbare Raw-Gewichte erzeugen. | Indexbereich und endliche Faktoren in `[0,1]` werden vor Quell-I/O geprüft; Nullgewichte liefern keinen Raw-Support. Clipping lehnt ungültige Parameter und Kandidaten ab. |
| P1 | Ein neu berechneter Hash legitimierte semantisch ungültige Qualitätspläne. | Builder und Loader prüfen eindeutige nichtleere Frame-IDs, Provenienzfelder, endliche Faktoren in `[0,1]` und das exakte gespeicherte Produkt. |
| P1 | Leere, doppelte, dimensionswidrige oder pfadüberschreitende Manifest-Einträge konnten als verwendbar gelten. | Strukturprüfung vor Datei-I/O, eindeutige lokale Dateistämme, konsistente positive Dimensionen, gültige Prüfsummen; Symlinks/Spezialdateien und Lesefehler gelten nicht als nutzbare Ebenen. |

Die Clipping-Speicherobergrenze ist konservativ: `O(Breite × Streifenzeilen ×
Kanäle × Framezahl)`. Die Framezahl ist **kein** konstanter Faktor. Passt selbst
eine Zeile nicht, wird vor Allokation abgebrochen. Die API braucht dann einen
zukünftigen Spaltenblock-/Disk-Spill-Pfad, keine nachträgliche Budgetwarnung.
Das Phasenbudget ist keine harte Gesamtprozess-RSS-Garantie; Provider und Sinks
müssen den dokumentierten Lebensdauervertrag einhalten. Die globale
Qualitätsbewertung ist jetzt Frame-weise, hat aber noch kein eigenständiges
Budgetmodell für große Einzelbilder und die Scratch-Puffer der Metrikfunktionen.

### Verbindliche nächste Schritte nach dieser Revision

1. Streaming-Sinks in einen vollständigen transaktionalen Profilstore integrieren:
   generationengebundene Ebenen, Manifest zuletzt veröffentlichen, benötigte
   Kanal-/Feldmenge, echte FITS-Dimensionen und Provenienz gegen Erwartungen
   prüfen. Der aktuelle generische Manifestprüfer bestätigt Dateiintegrität,
   keine vollständige Rekonstruktion und keine Resume-Fähigkeit.
2. `GLOBAL_QUALITY`, Qualitätsplan und gemeinsames Uniform/Raw-Clipping als echte
   Runnerphasen anbinden; die alte Uniform-Diagnose bleibt ungeclippt. Nicht
   ungeclipptes M2-Uniform mit geclipptem M3-Raw vergleichen. Normalisierte
   Quellcache-Verträge und Ereignis-/Resume-Reihenfolge dabei vollständig prüfen.
3. Einzelbildbudget für Qualitätsmetriken und sehr breite/framereiche Jobs
   schließen; bei Bedarf Spaltenblöcke oder Disk-Spill einführen. Spitzen-RSS und
   I/O mit synthetischen großen Jobs messen, ohne Benutzerruns zu verändern.
4. Erst danach Q-Maps/Detailprofile und Mehrbandfusion erweitern. `Q_composite`
   ist weiterhin 1; die vorhandenen Module stellen keinen vollständigen
   AQMH-Mehrbandpfad dar. Unabhängige Truth-Fixtures und die autorisierte
   wissenschaftliche Vergleichsmatrix bleiben Voraussetzung der Bildqualitätsabnahme.

### Verifikation dieser Nachprüfung

- `tests` und `tile_compile_runner` erfolgreich gebaut.
- Gezielte Auditregressionen: **24/24 Testfälle**, 539.407 Assertions.
- Gesamte Hauptsuite: **382/382 Testfälle**, 574.895 Assertions.
- Nachgewiesen: budgetabhängige automatische Streifenhöhe bei 50 überlappenden
  Frames; Ablehnung expliziter zu großer Chunks vor Quell-I/O; 8-MiB-Streaming
  bei gleichzeitiger Ablehnung der Vollmaterialisierung; pixelgleiche geclippte
  OSC-Resultate bei verschiedenen Streifenhöhen; Provider-/Vektorgleichheit
  globaler Qualitätsgewichte; ungültige Qualitäts-/Manifestartefakte;
  werterhaltender atomarer Zeilenexport.
- Whitespace-Prüfung der bearbeiteten getrackten Dateien ohne Befund.
- Keine neuen realen Bildverarbeitungsläufe, keine CUDA-Änderungen und kein
  gestarteter Backenddienst. Die Tests belegen keine wissenschaftliche
  Bildqualitätsabnahme oder harte Gesamtprozess-Speicherobergrenze.


## 7. Nächster umgesetzter Arbeitsschritt: Streaming-Gesamtstore

Der erste Schritt aus Abschnitt 6 ist für die Profilpersistenz umgesetzt.
`drizzle_profile_store` veröffentlicht vollständige unveränderliche Generationen
über einen atomaren `current.json`-Commit. Exakte Ebenenmenge, FITS-Shape/Typ,
Prüfsummen und vom Aufrufer vorgegebene Provenienz-/Algorithmusidentität werden
vor Veröffentlichung und beim Laden geprüft. Der Runner-Diagnoseexport nutzt
direktes Uniform-Streaming; die geclippte Uniform/Raw-Bibliotheks-API kann beide
Profile gemeinsam transaktional persistieren. FITS-Scratch wird vorab budgetiert.
Details und verbleibende Schritte stehen in §0.3 des Implementierungsplans.

Die Aussage aus Abschnitt 6, der Diagnose-Store materialisiere noch Uniform,
ist damit überholt. Vollständige neue Runnerphasen, validierte Quellcache- und
Qualitätsplan-Vorgänger sowie Resume/Fallback bleiben offen. Es wurden keine
Bildverarbeitungsläufe gestartet und keine bestehenden Runartefakte verändert.

`read_drizzle_profile_region` liest zusätzlich budgetierte Ausschnitte aus einer
gegen die erwartete Identität geprüften Generation. Zurzeit wird pro Aufruf die
ganze Generation erneut gehasht (begrenzter RAM, zusätzliche vollständige I/O).
Ein wiederverwendbarer Reader ohne wiederholtes Hashen braucht einen eigenen
Lebensdauervertrag und bleibt eine Performancefolgearbeit.

Verifikation: `tests` und `tile_compile_runner` erfolgreich gebaut; sieben neue
Store-/Region-Tests mit 135 Assertions bestanden. Gesamte Hauptsuite:
**389/389 Testfälle**, 575.030 Assertions. JSON/YAML und Schemaübereinstimmung
geprüft. Alle Ausgaben stammen aus synthetischen Testfixtures in temporären
Verzeichnissen; keine neuen Benutzerruns.


## 8. Quellcache- und Qualitätsvorgänger geprüft

Die Bibliothekskette `persist_forward_drizzle_from_predecessors` verbindet nun
inhaltsgeprüfte normalisierte Quellen, atomaren Qualitätsplan und den gemeinsam
geclippten Uniform/Raw-Store. Das Cachemanifest bindet die bestehenden rohen
Floatdateien an Dimensionen, CFA-Vertrag, Frame-IDs/Indizes und SHA256. Der Reader
hasht die tatsächlich geladenen Bytes und hält nur ein Quellbild. Der
Qualitätsartefakt-Loader prüft Cachehash, Sampling-Plan, Qualitätskonfiguration,
Frame-Zuordnung und unveränderte Registrierungsfaktoren. Store-Commit-Schema 2
bindet zusätzlich Cache- und Qualitätsplanhash.

Ein weiterer OOM-Fall ist geschlossen: extreme Quellindizes konnten sehr große
dichte Gewichtsvektoren verlangen. Die Zuordnung ist nun vorab begrenzt; der
übergebene Gewichtsvektor wird auch im Rekonstruktionsbudget angerechnet.
Die Qualitätsberechnung selbst besitzt eine konservative Einzelbildplanung
(128 Byte/Quellpixel zuzüglich Puffer, Metadaten und Scratch). Zu große Bilder
scheitern vor dem Cachelesen. Das schließt noch nicht die wissenschaftlich
korrekte, speicherarme Qualitätsmessung großer nativer Sensorbilder ab.

Offen bleiben produktive Runnerphasen, ein dauerhaft aufbewahrter Cache mit
verifizierter Normalisierungsherkunft und die gemeinsame Resume-/Fallback-
Integration. Eine Manifestveröffentlichung allein beweist nicht, dass beliebige
vorhandene Rohdateien korrekt normalisiert wurden. Einzelheiten und Reihenfolge
stehen in §0.4 des Implementierungsplans. Keine Benutzerruns gestartet.

Verifikation dieser Ergänzung: Runner und Tests gebaut; 13 gezielte
Vorgänger-/Store-Tests mit 160 Assertions sowie die vollständige Hauptsuite
mit **395/395 Testfällen**, 575.055 Assertions bestanden. Keine Benutzerruns
oder Backendprozesse gestartet.
