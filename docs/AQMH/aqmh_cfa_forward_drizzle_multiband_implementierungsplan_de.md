# CFA-Forward-Drizzle und Mehrband-Rekonstruktion als einzige Methodik

## Detaillierter Implementierungsplan

**Status:** CPU-Rekonstruktionspfad M1–M5 funktional integriert; M4-Produktintegration separat M10 zugeordnet. M6-Fusion, Kandidatenauswahl, Ausgabe **und §11.13-Ressourcenabnahme auf realen M31/M42-Läufen** abgeschlossen. M7 CUDA-Vorwärtsdrizzle **abgeschlossen** (§30.55–§30.57): auf allen produktiv relevanten Configs aktiv und **byte-identisch** zur CPU-Referenz — affin 1/1 + 2/2, **Produktionsconfig Modus 2/1** (Host-2×2→1×-Faltung der internen Device-Bänder, kein neuer Kernel), **lokale Warps** (Hybridpfad §19.6.2, `backend=cuda_hybrid`). Belegt real (M31 Modus 2/1, M42 Hybrid, §30.57) und synthetisch (`[cuda-parity]`). Vor dem CUDA-Versuch bleibt nur „Device vorhanden". Offen nur Durchsatz-Optimierung (kein Blocker). Keine pauschale Abnahme M1–M6.

**Datum:** Status- und Entscheidungsrevision 2026-09-07.

**Leseregel:** §0 enthält Status und nächste Schritte, §1–19 die verbindlichen Fachverträge, §20–27 Implementierung und Abnahme, §28 die Belege und §29 die Quellen. Die Entscheidungen vom 2026-09-07 sind in §11.13, §15.5/15.6, §19.6 und §23.1 integriert und präzisieren dort ältere Grundfestlegungen. Historische Fortschrittsnotizen stehen im [Entwicklungsprotokoll](aqmh_cfa_forward_drizzle_entwicklungsprotokoll_de.md); dessen Aussagen ersetzen keine aktuelle Abnahme.

**Prüfumfang:** aktueller Arbeitsbaum einschließlich bestehender uncommitteter Änderungen. Für §19.6.2 (Hybridpfad + Timing-Split, §30.56) Neubuild + gezielte `[cuda-parity]`/`[drizzle-store]`/`[contrib-list]`-Tests **und** volle `ctest`-Suite (526/528; die zwei roten Tests liegen im legacy-AQMH-Pfad ohne gemeinsamen Code). Keine neue vollständige wissenschaftliche M9- oder Powerloss-Abnahme; kein realer Großbild-Hybrid-Lauf. Echtdatenangaben aus früheren Notizen (§30.55) wurden nicht erneut gemessen.

**Zielcodebasis:** `tile_compile_cpp` (C++20, OpenCV, Eigen, optionale CUDA-Beschleunigung)
**Zielzustand:** ausschließlich CFA-Forward-Drizzle mit kontrollierter Mehrband-Rekonstruktion; Classic Tile Compile und das bisherige PREWARP-AQMH werden entfernt
**Primäres Ziel:** den gegen Siril gemessenen Schärfeverlust im linearen Stack durch eine einzige verlustarme Rekonstruktion aus den normalisierten CFA-Quelldaten beseitigen

---

## Wegweiser

- [Status und nächste Schritte](#plan-0)
- [Ziele und Architektur](#plan-1) · [Konfiguration](#plan-6)
- [Geometrie und CPU-Rekonstruktion](#plan-7) · [Ressourcenarbeit](#ressourcen-restarbeit)
- [Q-Maps und Mehrband](#plan-13) · [Kandidatenauswahl](#plan-15)
- [Artefakte, Runner und Resume](#plan-16) · [CUDA](#plan-19)
- [Tests und Fixtures](#plan-20) · [Meilensteine](#plan-23)
- [Verbindliche Festlegungen](#plan-26) · [Definition of Done](#plan-27)
- [Evidenz](#plan-28) · [Quellen und Historie](#plan-29)

Die Kapitelnummern 1–27 bleiben für bestehende Fachverweise erhalten.

---

<a id="plan-0"></a>

## 0. Status, nächste Schritte und verbindliche Invarianten

<a id="status"></a>

### 0.1 Implementierungsstatus (2026-09-07)

Aktuelle Übersicht; detaillierte Checklisten stehen in §23. Implementiert,
getestet und für den Produktrelease abgenommen sind getrennte Aussagen.

| Meilenstein | Aktueller Status | Fehlende Abnahme / Zuordnung |
|---|---|---|
| M0 | Grundlage teilweise umgesetzt | Produktweite Migration, Beispiele und Altvertragsentfernung bleiben M8/M10; keine belastbare Prozentangabe |
| M1 | Neuer Runnerpfad funktional integriert | Produkt-Cutover M10, breitere wissenschaftliche Abnahme M9 |
| M2 | CPU-Kern und transaktionaler Store implementiert | Unabhängige Truth-/PSF-, Prozess-Kill-/Wiederanlaufnachweise; CPU-Parallelisierung optional |
| M3 | Clipping, Qualitätsgewichte und geprüfte Vorgänger integriert | Cache-Produktpolitik M8/M10; Kandidaten-Vorabplanung bereits vorhanden |
| M4 | Kern und Store implementiert | Downstream/WCS/Photometrie M10, Report M8; ursprüngliche Gesamtabnahme dadurch nicht erledigt |
| M5 | Source-Q-Maps samt Konsum implementiert | Reales MONO und objektoffene Matrix M9 |
| M6 | Fusion, Auswahl, Ausgabe und §11.13-Ressourcenvertrag (Vorabplan, Fail-Closed, Kandidaten-Spool, phasen-lokale RSS) implementiert und synthetisch abgenommen | Reale Ressourcenabnahme mit großen Bildern braucht einen Run-Auftrag (§11.13) |
| M7 | **CUDA-Vorwärtsdrizzle abgeschlossen** (§30.55–§30.57): auf **allen** produktiv relevanten Configs aktiv und byte-identisch zur CPU-Referenz — affin 1/1 + 2/2, **Produktionsconfig Modus 2/1** (Host-2×2-Faltung der internen Device-Bänder, kein fehlender Kernel), **lokale Warps** (Hybridpfad §19.6.2, `backend=cuda_hybrid`). Vor dem CUDA-Versuch bleibt nur „Device vorhanden". Reale M31-Modus-2/1- und M42-Hybrid-Vollläufe byte-identisch (§30.57); `[cuda-parity]` synthetisch vollständig; §11.13 auf M31+M42; FP-Kontraktion fixiert; `s_j`-Pinning n/a solange bit-exakt. | Nur Durchsatz-Optimierung (CUDA ≈ 1,1–2,1× langsamer je nach Config; **kein Blocker**, §19.6) und — davon abhängig — die optionale Numerikrevision (`std::exp` → Minimax) für einen vollen GPU-Lokalpfad. Profiling-Befund: `hybrid_cpu_seconds` ≫ GPU-Raster (§30.57). |
| M8 | offen | GUI, Report, vollständige aktive Dokumentation und Cache-Kommunikation |
| M9 | offen | Vollständige unabhängige Pflichtmatrix fehlt; einzelne historische M31/M42-Läufe ersetzen sie nicht |
| M10 | offen | Kanonische Downstream-Ausgabe, Photometrie-/WCS-Vertrag, Retention und Produkt-Cutover |
| M11 | offen | M10 und Grace-Zyklus vorausgesetzt |

**Prüfung dieser Statusrevision:** Neubuild + vollständige `build/tests`-Suite
**508/509** bestanden (die eine Abweichung bleibt das vorbestehende
`test_acceleration_backend.cpp:254`, ohne Bezug zu diesem Pfad). Store-Level-
CUDA-Parität nativ auf der GTX 1660 Ti; zusätzlich reale `reconstruct`-Vollläufe
M31/M42 auf ausdrücklichem Run-Auftrag mit byte-identischem CPU↔CUDA-Store
([§30.55](aqmh_cfa_forward_drizzle_entwicklungsprotokoll_de.md#historie-30-55)).
Diese Verifikationsläufe sind Wegwerf-Artefakte, keine kanonischen Benutzerruns.

### 0.2 Nächste Arbeitsschritte

1. M6-Ressourcenvertrag §11.13 in Code, synthetischen Fixtures **und auf realen
   M31/M42-Vollläufen** bestätigt
   ([§30.50](aqmh_cfa_forward_drizzle_entwicklungsprotokoll_de.md#historie-30-50),
   [§30.55](aqmh_cfa_forward_drizzle_entwicklungsprotokoll_de.md#historie-30-55)):
   Working-Set fits budget, Temp-Space ok, Phase-RSS im Envelope, Spool entfernt.
2. M7 CUDA-Vorwärtsdrizzle **abgeschlossen**
   ([§30.51](aqmh_cfa_forward_drizzle_entwicklungsprotokoll_de.md#historie-30-51)–[§30.57](aqmh_cfa_forward_drizzle_entwicklungsprotokoll_de.md#historie-30-57)):
   auf allen produktiv relevanten Configs aktiv und **byte-identisch** zur
   CPU-Referenz — affin 1/1 + 2/2 (§30.55), **Produktionsconfig Modus 2/1**
   (Host-2×2→1×-Faltung der internen Device-2×-Bänder, kein fehlender Kernel),
   **lokale Warps** (Hybridpfad §19.6.2, `backend=cuda_hybrid`). Real belegt
   (M31 Modus 2/1: `reconstruction_multiband.fits` identisch = Vor-§30.54-Wert;
   M42 Hybrid: 52 Plane-FITS + fusioniertes Bild identisch = §30.55-CPU-Referenz;
   §30.57) und synthetisch (`[cuda-parity]`). Vor dem CUDA-Versuch bleibt nur
   „Device vorhanden". `s_j`-Pinning: nicht anwendbar solange die Kernel
   bit-exakt sind. Profiling-Befund: der Hybrid-Timing-Split zeigt
   `hybrid_cpu_seconds` (444 s CPU-Geometrie) ≫ `hybrid_gpu_raster_seconds`
   (0,6 s) — die Rasterisierungs-Auslagerung allein beschleunigt lokale Warps
   nicht.
   **Offen (kein Korrektheits- oder Freigabe-Blocker):** Durchsatz-Optimierung
   (CUDA ≈ 1,1–2,1× langsamer je Config, §19.6 stellt Bit-Exaktheit voran) und,
   davon abhängig, die optionale Numerikrevision (`std::exp` → Minimax) für
   einen vollen GPU-Lokalpfad.
3. M8-/M10-Integration mit kanonischen WCS-/Photometrie-Ausgaben, Report und
   Cachepolitik abschließen; übertragene Pflichten bleiben offen (§23.1).
4. M9-Pflichtmatrix einschließlich realem MONO und unabhängigen Truth-Fixtures
   vervollständigen. Reale Läufe nur nach ausdrücklichem Auftrag.

Produktfreigabe setzt die vollständigen M8/M9/M10-Nachweise voraus. Die
Ressourcen- und GPU-Freigabekriterien dürfen nicht durch bloßen Fortschritt
einzelner Komponenten ersetzt werden.

### 0.3 Geometrie- und Speicherinvarianten

Die folgenden weiterhin gültigen Korrekturen aus dem Audit vom 2026-09-05
bleiben Teil des aktiven Vertrags. Die damaligen Implementierungsstände
stehen separat im Entwicklungsprotokoll. Ressourcenpräzisierung: §11.13.

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

---

<a id="plan-1"></a>

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

<a id="plan-2"></a>

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

<a id="plan-3"></a>

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

<a id="plan-4"></a>

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

<a id="plan-5"></a>

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

<a id="plan-6"></a>

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
| `stacking` | `common_overlap_required_fraction` bereits nach `reconstruction` verschoben ([§30.2](aqmh_cfa_forward_drizzle_entwicklungsprotokoll_de.md#historie-30-2)); `method`, `sigma_clip`, `cluster_quality_weighting`, `output_stretch`, `tile_common_valid_min_fraction` entfernt; **`per_frame_cosmetic_correction(_sigma)` und `cosmetic_correction(_sigma)` bleiben erhalten** und werden nach `calibration.frame_cleanup` verschoben — Hotpixel-/Cosmic-Ray-Entfernung läuft pro Frame vor der Registrierung (`runner_phase_registration.cpp`) und ist für Forward-Drizzle weiterhin nötig |
| `validation` | `min_tile_weight_variance`, `require_no_tile_pattern` entfernt; `min_fwhm_improvement_percent`, `max_background_rms_increase_percent` mit den Gates aus 3.2 und `aqmh.validation` zu einem `reconstruction.validation`-Block konsolidiert |

**Migrationssemantik (Entscheidung 2026-09-03): strippen mit Warnung.** Trifft
der Parser auf einen entfernten *strukturellen* Block oder Schlüssel aus obiger
Tabelle (nicht `method`/Engine — die bleiben fail-closed), entfernt er ihn,
protokolliert eine `WARN`-Meldung mit Pfad und Wert und schreibt die Liste der
gestrippten Schlüssel nach `artifacts/config_migration.json`. Der Run läuft mit
der bereinigten Konfiguration weiter. Damit ist der Vorgang auditierbar und
kein *stilles* Übersetzen im Sinne von [§30.3.3](aqmh_cfa_forward_drizzle_entwicklungsprotokoll_de.md#historie-30-3); [§30.3.3](aqmh_cfa_forward_drizzle_entwicklungsprotokoll_de.md#historie-30-3) bezieht sich weiterhin
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

<a id="plan-7"></a>

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

<a id="plan-8"></a>

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

<a id="plan-9"></a>

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
Die anderslautende Interpretation und Defaultkalibrierung aus [§30.10](aqmh_cfa_forward_drizzle_entwicklungsprotokoll_de.md#historie-30-10) ist
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

<a id="plan-10"></a>

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

<a id="plan-11"></a>

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
  **Präzisierung (2026-09-05, beim Implementieren entdeckt, siehe [§30.19](aqmh_cfa_forward_drizzle_entwicklungsprotokoll_de.md#historie-30-19)):**
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

<a id="ressourcen-restarbeit"></a>

### 11.13 Offene Ressourcenarbeit: Fusion, Validierung und Export

**Festlegung: 2026-09-07.**

**Codebefund:** `fuse_multiband_store_to_image()` hält vollständige Ausgabe-
kanäle, Kandidatenluminanzen, Alpha-Maps und optional neun OSC-Kandidatenkanäle;
der Runner fordert diese an und erzeugt zusätzliche Validierungsmatrizen.
`max(memory_budget_mb,256)` hebt kleine Budgets an. Einzelne Region-Reader
prüfen ihre eigene Allokation, nicht die Summe aller gleichzeitig lebenden
Objekte. `estimated_peak_bytes` stammt bisher aus dem Drizzle-Schritt und
deckt Fusion/Validierung/Export nicht als gemeinsamen Plan ab.

**Korrektur der Auditinterpretation:** §11.11 erlaubt
`rss_growth <= budget*1.05 + 256 MiB`. Der historische Peak 4.527.156 KiB
(ca. 4,32 GiB) bei 4096 MiB liegt bereits unter der Hülle von 4556,8 MiB,
selbst ohne Abzug einer Baseline. Er beweist keine Budgetüberschreitung, aber
auch keine korrekte Vorabplanung. `ru_maxrss` ist ein Prozess-Lebenszeitmaximum;
die Differenz zweier Lebenszeitmaxima ist kein zuverlässiges phasenlokales
RSS-Wachstum. Historische „passt in die Hülle“-Aussagen sind kein Ersatz für
eine phasenbezogene Messung gegen tatsächliches RSS zu Phasenbeginn.

**Festgelegte Umsetzung und Abnahme, noch offen:**

1. Einen gemeinsamen, überlaufsicheren Host-Arbeitssatz vor großen Allokationen
   planen: vorhandene residente Daten, alle Profil-/Halo-/Q-Puffer, Kandidaten,
   Alpha, Validierungsscratch, FITS-/Exportpuffer und Reserve. Kein stilles
   Anheben eines expliziten Budgets. Host-/cgroup-Headroom zusätzlich beachten.
2. Kandidatenkanäle streifenweise in temporäre Stores schreiben; nach der
   Auswahl den gewählten Kandidaten und immutable Raw atomar ausliefern.
   Validierung mittels begrenzter Regions-/Sternpatch-Reads und gestreamter
   Hintergrundstatistik durchführen. Exakte Statistik kann Disk-Spooling
   verwenden; keine still geänderte Metrik für weniger RAM.
3. Chunkhöhe einschließlich Halo und aller gleichzeitig lebenden Daten wählen.
   Passt selbst die Mindestregion nicht, vor Veröffentlichung fail-closed.
   Vorherige gültige Generation bleibt bei Fehlern erhalten.
4. Geschätzten Arbeitssatz und aktuelle/phasenspezifische RSS-Spitze getrennt
   protokollieren. Prozess-Lebenszeitpeak darf ergänzend bleiben, muss so
   bezeichnet sein. Vorab-Allokationsplan ≤Budget; gemessenes RSS-Wachstum
   erfüllt zusätzlich die bestehende §11.11-Hülle, keine neue Lockerung.
5. Regressionsfixtures: explizites kleines Budget wird vor großen Allokationen
   abgelehnt; Chunkvariation ändert keine Kandidaten-/Masken-/Auswahlsemantik;
   große OSC/MONO-Felder, alle Scale-Modi, Export bei `summary`/`full` und
   injizierter Fehler bewahren den Commitvertrag. Native Ressourcenabnahme
   separat dokumentieren. Erst danach M6-Abnahme und M7-Aktivierung.

**Reihenfolge:** M6-Gesamtbudget schließen → M7-Listenrasterisierer und
Clipping/Profile → vollständige native Paritäts-/Neustartmatrix und Timing →
Produktfreigabe erst mit M8/M9/M10. Reale Benutzerruns benötigen weiterhin
einen ausdrücklichen Auftrag; dieser Plan erteilt keinen Run-Auftrag.

---

<a id="plan-12"></a>

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

<a id="plan-13"></a>

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

<a id="plan-14"></a>

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

<a id="plan-15"></a>

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

**Erwartete Auswahlverteilung und Entscheidung 2026-09-07.** Das 5-%-Gate
bleibt unverändert (§15.6). Uniform und Raw verwenden dieselbe Drizzle-Geometrie,
Scale-Konfiguration und gemeinsame Clippingpopulation. Drizzle-Korrelation
betrifft deshalb beide; sie begründet keinen pauschalen Bonus ausschließlich
für Raw/Multiband. Qualitätsgewichte können Varianz und Kovarianz ändern.

Die in [§30.45](aqmh_cfa_forward_drizzle_entwicklungsprotokoll_de.md#historie-30-45)/30.46 dokumentierten Raw/Uniform-RMS-Verhältnisse um 1,088
beschreiben zwei historische Datensätze, keinen nachgewiesenen universellen
Drizzle-Offset. Kausalität ist mit diesen Zahlen allein nicht geklärt.

Eine niedrige Multiband-Promotionsquote ist zulässig. Sie beweist weder einen
Qualitätsgewinn noch das Bestehen der M9-Matrix. Bei fehlender positiver
Sternevidenz bleibt Raw, sofern dessen Safety-Gates bestehen, sonst Uniform.
Für den ersten Release wird keine zusätzliche Struktur-Promotion freigegeben
(§15.5); Energieguard, feste Sternpopulation und N/A-Vertrag bleiben erhalten.

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

<a id="entscheidung-evidenz"></a>

### 15.5 Entscheidung: Sternevidenz bleibt Freigabekriterium

**Festlegung: 2026-09-07.**

**Gewählt:** Der erste Release behält §15.3 unverändert. Keine zusätzliche
OR-Promotion durch Struktur-SNR, lokale Varianz oder Wavelet-Energie. Der
Energieguard bekommt keine Sternausnahme und wird nicht gelockert, um die
Promotionsquote zu erhöhen. `stars_multiband_effective=0` bleibt fehlende
positive Evidenz, keine Ausnahme vom Mindeststernvertrag.

**Begründung:** Mehr Hochfrequenzenergie kann Signal, Rauschen oder Ringing
sein. Eine am Kandidaten optimierte Strukturmaske kann dieselben Schwankungen
als Gewinn werten, die der Kandidat verstärkt. Die beiden bisherigen OSC-
Datensätze rechtfertigen weder eine neue Metrik noch deren Schwellenwert.
Sterne sind für die erklärte Schärfezielsetzung ein prüfbares PSF-Signal;
fehlende Verbesserung wird nicht in einen Erfolg umdefiniert.

Diese konservative Freigabe kann reale Vorteile in diffuser Struktur übersehen.
Das wird ausdrücklich akzeptiert und im Report als Grenze der Evidenz genannt.
Auch `alpha=0` macht den gesamten Multibandkandidaten wegen des Uniform-Grobrests
nicht automatisch identisch mit Raw (§14.3); die vollständigen Safety-Gates
bleiben deshalb erforderlich.

**Festgelegter Forschungsweg, ohne Freigabewirkung:** unabhängige pixelintegrierte
Truth-Fixtures mit diffusen Strukturen, Sternen, realistischen PSFs, CFA und
Noise-Realisierungen; Bias/Fluxfehler und Fehlerleistung pro räumlicher Skala
gegen Truth messen. Ergänzend können vorab festgelegte, disjunkte Frame-Hälften
reproduzierbare Struktur prüfen. Bei unabhängigen Rauschanteilen und festen
Operatoren gilt `E[(S+n1)(S+n2)] = S²`; adaptive Gewichte, gemeinsame Kalibration
und datenabhängige Masken können diese Voraussetzung verletzen und müssen
gesondert getestet werden. Keine nominelle Unabhängigkeit behaupten.

Eine spätere Promotion erfordert eine eigene validierte Methodikrevision mit
vorab festgelegten Masken/Skalen/Schwellen, Tests auf unabhängigen Daten und
Prüfung gegen **Uniform und immutable Raw**. Bestehende Safety-Gates dürfen
dadurch nicht ersetzt werden. Für M6/M7 ist dies keine offene Entscheidung.

<a id="entscheidung-rauschen"></a>

### 15.6 Entscheidung: 5-%-RMS-Gate bleibt unverändert

**Festlegung: 2026-09-07.**

**Gewählt:** `background_RMS_candidate <= 1.05 * background_RMS_uniform`
bleibt verbindlich, ohne `pixfrac`-/Scale-Bonus und ohne objektspezifisches Tuning.
Die 5 % sind ein konservativer versionierter Engineering-Grenzwert, keine aus
der Drizzle-Theorie ableitbare universelle Naturkonstante.

Für feste akzeptierte Beiträge und feste Gewichte gilt mit
`a_i = w_i / sum(w)` für einen Schätzer `X = sum(a_i x_i)`:

```text
Var(X) = a^T Sigma a
       = sum_i a_i^2 sigma_i^2    # nur bei unabhängigen Eingangsbeiträgen
Cov(X_q, X_r) = a(q)^T Sigma a(r)
Var(sum_q h_q X_q) = h^T Cov(X) h
```

Uniform und Raw teilen den geometrischen Kern und die Clippingpopulation,
haben jedoch durch `G_eff` und Q unterschiedliche normierte Gewichte. Schon
bei gleicher unabhängiger Einzelvarianz ist `Var(X)=sigma²/n_eff`;
Gewichtskonzentration kann die Varianz erhöhen. Datenabhängige Gewichte und
Clipping benötigen darüber hinaus eine empirische Bias-/Varianzprüfung.
Der beobachtete RMS-Unterschied ist deshalb nicht allein dem gemeinsamen
Drizzlekern zuzuschreiben. Ein für beide gleicher Korrekturfaktor kürzt sich
im Kandidat/Uniform-Verhältnis heraus.

[STScI beschreibt Gewichtskarten und korreliertes Drizzle-Rauschen](https://hst-docs.stsci.edu/drizzpac/chapter-3-description-of-the-drizzle-algorithm/3-3-weight-maps-and-correlated-noise).
Die Anwendung auf den hier gemeinsam gerasterten Uniform/Raw-Vergleich ist
unsere Schlussfolgerung, keine dort empfohlene 5-%-Schwelle.

**M9-Nachweise:** dieselben Hintergrundregionen, Masken, Einheiten und
Ausgabeskalen verwenden; neben Pixel-RMS räumliche Kovarianz beziehungsweise
Blank-Sky-Apertur-/Bandsummenvarianz erfassen. Den Effekt von `G_eff`, Q und
Clipping in kontrollierten Fixtures isolieren; Aperturfluss, PSF und
Farberhaltung zugleich prüfen. Pixel-RMS allein belegt keine optimale
Apertur-SNR. Diese zusätzlichen Nachweise ändern das produktive Gate nicht.
Ein Gate-Reject bleibt gültig, bis eine eigenständige Methodikrevision eine
bessere Entscheidungsregel mit unabhängiger Evidenz begründet.

---

<a id="plan-16"></a>

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

> **Implementierungsstand (30.48):** `forward_drizzle_raw_*` und
> `reconstructed_*` werden geschrieben. `stacked[_rgb].fits` **noch nicht** —
> diese kanonischen Namen werden von Astrometrie/BGE/PCC/HMS im
> **post-§17.4**-Photometrieraum erwartet, und die §17.4-Rücknahme ist
> M10-Cutover-Arbeit. Bis dahin würde eine `stacked*`-Datei im
> normalisiert-linearen Raum Downstream täuschen; sie entsteht mit M10.

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

<a id="plan-17"></a>

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

<a id="plan-18"></a>

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

<a id="plan-19"></a>

## 19. CUDA-Implementierung

### 19.1 Reihenfolge

CUDA-Komponentenarbeit darf gegen die getestete CPU-Referenz erfolgen.
Die produktive Aktivierung setzt die M6-Ressourcenabnahme (§11.13), geprüfte
Artefakt-/Resume-Verträge und die vollständige §19.5-Paritätsmatrix voraus.

Neue Dateien:

```text
tile_compile_cpp/include/tile_compile/reconstruction/forward_drizzle_cuda.hpp
tile_compile_cpp/src/reconstruction/forward_drizzle_cuda.cpp
tile_compile_cpp/src/reconstruction/forward_drizzle_cuda_device.cu
```

### 19.2 Kernelaufteilung

Verbindliche Stufen pro Chunk (Reduktionsdetails §19.6):

1. Source-Regionen und Q-Map-Regionen hostseitig laden.
2. Source-Samples und Transformdaten H2D übertragen.
3. Frame-lokale Droplet-Akkumulation in getrennte Buffers.
4. Transposition in pixel-major Layout.
5. deterministische Sortierung und Clipping pro Pixel mit stabilen Tie-Breaks;
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

### 19.3 Deterministische Reduktion

**Verbindliche Entscheidung 2026-09-07:** deterministisch geordnete
Vorwärts-Beitragslisten mit segmentweiser Akkumulation (§19.6). Die frühere
Wahlfreiheit zwischen Float-Atomics, Gather und toleriertem Scatter entfällt
für den ersten produktiven M7-Pfad.

Globale und frame-lokale Float-Atomics für wissenschaftliche Summen sind dort
nicht zulässig. Integer-Zähler für Listenaufbau sind zulässig, sofern danach
eine vollständige kanonische Ordnung hergestellt wird. Parallelität liegt
zwischen Ausgabezellen/Segmenten, nicht in einer beliebigen Summenreihenfolge
innerhalb derselben Zelle. CPU-Parität und deterministische Wiederholungen
sind getrennt nachzuweisen; gleiche Reihenfolge allein garantiert wegen FMA
und Bibliotheksfunktionen noch keine CPU-/GPU-Bitidentität.

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

<a id="paritaets-fehlergrenzen"></a>

#### 19.5.1 Fehlergrenzen der Paritätsmatrix (festgeschrieben 2026-09-07)

Diese Grenzen erfüllt §19.6 „vor Aktivierung"; sie dürfen nicht nachträglich
zum Bestehen erweitert werden. Feldbezogen, nicht als Gesamtprofil-Toleranz.

| Größe | CPU↔CUDA | Wiederholung gleiche HW/Binary | Chunkhöhen-Variation |
| --- | --- | --- | --- |
| Diskrete Entscheidungen (Frameausschluss, Clipping accept/reject, Support/Veto, N/A, Kandidatenauswahl, Akzeptanzmaske, Tie-Breaks) | identisch (bit-exakt) | identisch | identisch |
| `X_out` je Zelle (normalisierter linearer Arbeitsraum) | rel. ≤ `1·10⁻⁹` mit abs. Boden `1·10⁻¹²` | bit-exakt | bit-exakt |
| À-trous-Detailfelder `D_fine` / `D_medium` je Zelle | rel. ≤ `1·10⁻⁹` **relativ zur robusten Bandskala** `s_j` (nicht zu `max(1,·)`) mit abs. Boden `1·10⁻⁶·s_j`; `s_j` = p99.5(|D_j|) auf Support, beim Matrixlauf je Band gemessen und im Testfixture literal gepinnt | bit-exakt | bit-exakt |
| `weight_sum` / `coverage` je Zelle | rel. ≤ `1·10⁻⁹` | bit-exakt | bit-exakt |
| Integrierter Aperturflux je synthetischem Stern (§19.5-Fixtures) | rel. ≤ `1·10⁻³` | rel. ≤ `1·10⁻⁶` | rel. ≤ `1·10⁻⁶` |
| Zentroid je synthetischem Stern | ≤ `2·10⁻³` native Pixel | ≤ `1·10⁻⁴` px | ≤ `1·10⁻⁴` px |
| `n_eff` der Rekonstruktion | rel. ≤ `1·10⁻⁹` | bit-exakt | bit-exakt |

`X_out` liegt im normalisierten linearen Arbeitsraum bei ≪ 1, daher greift
`max(1,|cpu|)` **nicht** und die Grenze wirkt effektiv absolut `1·10⁻⁹` — für
`X_out` fest genug. Die à-trous-Detailbänder haben die kleinsten Beträge; eine
feste absolute `1·10⁻⁹`-Grenze wäre dort ein zweistelliger Relativfehler und
könnte eine echt fehlsortierte Reduktion durchlassen. Deshalb: **die Methode**
(rel. zur je Band gemessenen robusten Skala `s_j`, abs. Boden `1·10⁻⁶·s_j`) ist
hiermit festgeschrieben; die Zahl `s_j` selbst wird beim Matrixlauf aus den
Banddaten bestimmt und im Fixture literal gepinnt — das ist kein
nachträgliches Erweitern, sondern die datengetriebene Instanziierung der hier
fixierten Methode. Die `1·10⁻⁹`-Feldgrenze ist die lineare Fortpflanzung der
zwei gemessenen Geometriekern-Abweichungen (Polygon-Rechteck-Fläche
`4,7·10⁻¹¹`, affine Ecken `<1·10⁻⁹`) in die akkumulierten Gewichte —
ausdrücklich **nicht** dieselbe Zahl wie die Kern-Toleranzen, sondern deren
gebündelte Feldwirkung. Der Aperturflux-Paritätswert `1·10⁻³` liegt weit
innerhalb des Produktgates `< 0,5 %` (§32). Ein Masken-/Gate-Unterschied ist
nie ein tolerierter Feldfehler (§19.6).

<a id="entscheidung-cuda"></a>

### 19.6 Entscheidung: deterministische Vorwärtslisten und Segmentreduktion

**Festlegung: 2026-09-07.**

**Gewählt:** begrenzter Vorwärts-Listenaufbau, kanonische Sortierung und
deterministische Akkumulation je Frame/Zielzelle/Kanal. Kein ungeordneter
Float-Scatter; kein inverses Suchfenster als allgemeiner Produktionsalgorithmus.

Die vorhandene CPU-Subdivision liefert die zulässigen transformierten Leaves
einschließlich ganzer Sample-/Frameausschlüsse. Für jeden Chunk werden alle
schneidenden Leaves konservativ erfasst. Eine inverse lokale Näherung darf
keine Beiträge abschneiden: für gekrümmte Modelle genügt ein affines inverses
Bounding-Window ohne bewiesene Schranke nicht. Ein einheitlicher Vorwärtspfad
deckt affine und lokale Modelle mit derselben Flächengeometrie ab.

Verbindlicher Ablauf:

1. Beiträge vorab zählen, überlaufsicher prefix-summieren und vollständigen
   Speicher für Records, Indizes, Sortierscratch und Akkumulatoren budgetieren.
2. Beitragsschlüssel: `(frame_order, channel, target_y, target_x, source_y,
   source_x, leaf_order)`. Frame-/Leaf-Ordnung stammt aus der CPU-Referenz;
   jeder Beitrag hat einen eindeutigen Schlüssel. Integer-Atomics dürfen
   unbewertete Slots reservieren; ihre Ausführungsreihenfolge bestimmt nie Summen.
3. Records kanonisch sortieren. Je Frame/Zielzelle/Kanal reduziert ein Thread
   die Beiträge in CPU-Quell-/Leaf-Reihenfolge mit Double-Akkumulatoren.
   Ein frei gewählter paralleler Reduktionsbaum ist für diese erste Version
   ausgeschlossen. Zwischen Segmenten wird parallel gearbeitet.
4. Clipping mit identischer Definition, stabilen Tie-Breaks und gemeinsamer
   Akzeptanzmaske; Profilreduktion in fester Framefolge. Keine GPU-eigene
   Q-, Veto- oder Maskeninterpretation.
5. Zu große Listen halbieren die Chunkhöhe gemäß §19.4; passt eine Zeile
   einschließlich Listen/Scratch nicht, vollständiger CPU-Neustart.

**Numerischer Vertrag:** Wiederholungen auf derselben Hardware/Binary und
unterschiedliche Chunkhöhen müssen identische gespeicherte Profile liefern.
CPU↔GPU verlangt exakte diskrete Entscheidungen (Frameausschluss, Clipping,
Support/Veto, N/A und Kandidatenauswahl); kontinuierliche Felder dürfen nur
vorab festgelegte, feldbezogene absolute/relative Toleranzen nutzen. Die zwei
vorhandenen Geometriekernel-Toleranzen sind keine Toleranz für ganze Profile.
Vor Aktivierung muss die Paritätsmatrix die Profil-/Flux-/Zentroidfehlergrenzen
festschreiben und bestehen, einschließlich nahezu nullwertiger Felder und
Schwellenfixtures. Die konkreten Grenzen sind in [§19.5.1](#paritaets-fehlergrenzen)
festgeschrieben (2026-09-07); Toleranzen dürfen nicht nachträglich zum Bestehen
erweitert werden. FMA-/Compilerpolitik explizit fixieren, kein Fast-Math im Referenzpfad.
Ein Masken- oder Gate-Unterschied ist kein tolerierter kleiner Pixelfehler.

**Begründung:** Floating-Point-Addition ist nicht assoziativ; auch FMA verändert
Rundung. Feste Reihenfolge reduziert die Ursachen für Abweichungen, garantiert
allein aber keine plattformübergreifende Bitidentität. Diese Eigenschaften
sind in [NVIDIAs Floating-Point-Dokumentation](https://docs.nvidia.com/cuda/archive/11.6.1/floating-point/index.html)
beschrieben. Die konkrete Listenstrategie ist unsere Entwurfsentscheidung für
lokale Warps und diskontinuierliche Clipping-/Gateentscheidungen. Durchsatzgewinn
bleibt zu messen; Geschwindigkeit hat keinen Vorrang vor dem Paritätsvertrag.

#### 19.6.1 Lokale-Warp-Geometrie bleibt CPU-only, solange §19.5.1 gilt

**Festlegung: 2026-09-07.** (Rasterisierung: siehe §19.6.2.)

Der **Subdivisions- und Verschiebungsfeldpfad** für lokale Warps kann auf der GPU
**nicht** bit-identisch zur CPU-Referenz laufen. Unter §19.5.1 in der geltenden
Fassung (exakte diskrete Entscheidungen, keine Toleranz für Masken/Gates) ist
damit **kein** GPU-Pfad zulässig, der das Verschiebungsfeld selbst auswertet; er
wird nicht auf das Device abgebildet. (Die reine **Rasterisierung** fertiger
CPU-Leaf-Ecken ist davon unberührt und läuft über den hybriden Pfad §19.6.2 —
die GPU wertet dort kein Feld aus.) Eine Wiederaufnahme des vollen GPU-Lokalpfads
setzt eine Änderung von §19.5.1 voraus — konkret eine
vorab festgelegte, feldbezogene Toleranz für das *kontinuierliche*
Verschiebungsfeld (die §19.6 für kontinuierliche Felder grundsätzlich
zulässt), verbunden mit dem Nachweis, dass die davon abhängigen diskreten
Entscheidungen dennoch exakt gleich bleiben. Diese Änderung ist derzeit nicht
gemacht und durch §19.5.1 ausdrücklich untersagt.

Grund für die Bit-Diskrepanz: `smooth_local_basis` (`src/registration/global_registration.cpp`)
wertet eine 4×4-Gauß-Basis über `std::exp` aus. `expf`/`std::exp` sind
transzendent; die glibc- und die CUDA-libdevice-Implementierung unterscheiden
sich um ~1 ULP. Dieses Ergebnis fließt über `evaluate_smooth_local_displacement`
→ das Verschiebungsfeld → den Fixpunkt-Iterator `invert_local_source_to_canvas`
in **diskrete** Entscheidungen: die Konvergenzprüfung `step < tol_px`, den
`out_of_bounds`-Frühausstieg und damit die Menge der akzeptierten Leaves und
ihrer Ecken. §19.6 verlangt für diskrete Entscheidungen exakte CPU↔GPU-Gleichheit
(„Ein Masken- oder Gate-Unterschied ist kein tolerierter kleiner Pixelfehler"),
§19.5.1 lässt hierfür keine Toleranz zu. Beides ist mit `expf`-Parität nicht
erfüllbar.

Konsequenz für die Paritätsmatrix §19.5: die **Geometrie** lokaler Warps
(Verschiebungsfeld, Fixpunkt-Inversion, Bounds, adaptive Subdivision, Leaf-Ecken)
bleibt CPU-Referenz-only — für sie gibt es keinen GPU-Vergleich, weil es keine
GPU-Auswertung des Verschiebungsfelds gibt. Die reine **Rasterisierung** (Polygon-
Zell-Fläche aus fertigen Leaf-Ecken) darf per §19.6.2 auf das Device abgebildet
werden; dafür gelten die vorhandenen affinen Paritätszeilen (Geometriekernel
bit-identisch). Die Matrix ist für den GPU-Teil vollständig, sobald alle
**affinen** Zeilen und die §19.6.2-Rasterzeile bestehen.

Konsequenz für die produktive Verdrahtung (§19.4): `persist_forward_drizzle_multiband`
prüft vor jedem CUDA-Versuch, dass nicht Modus 2/1 vorliegt und ein Device
verfügbar ist. Frames mit lokalem Modell (`has_smooth_local_model`) laufen über
den hybriden Pfad §19.6.2 (CPU-Geometrie → GPU-Rasterisierung), affine Frames
über den reinen Device-Pfad; beide teilen sich denselben kanonischen Downstream.
Trifft eine der beiden verbleibenden Bedingungen (Modus, Device) nicht zu oder
schlägt ein Device-Schritt fehl, läuft die Rekonstruktion vollständig auf dem
CPU-Referenzpfad; der Grund steht in `cuda_fallback_reason`. Es gibt keinen
gemischten CPU/CUDA-Commit.

#### 19.6.2 Hybrider Pfad für lokale Warps: CPU-Geometrie → GPU-Rasterisierung

**Festlegung: 2026-09-07.**

§19.6.1 schließt eine **eigenständige** GPU-Auswertung des Verschiebungsfelds aus.
Es ist aber zulässig und vom Plan ausdrücklich erlaubt, die Geometrie vollständig
auf der CPU-Referenz zu berechnen und der GPU nur die **Rasterisierung** zu
übergeben. Dieser hybride Pfad ist kein versteckter Fehlerfallback, sondern ein
regulärer Backend-Pfad.

**Ablauf (verbindlich):**

1. Die vorhandene CPU-Referenz (`sample_leaves` → `subdivide_local`,
   `invert_local_source_to_canvas`) berechnet je Quell-Sample lokale
   Verschiebung, Fixpunkt-Inversion, Bounds-Prüfung und adaptive Subdivision.
2. Sie legt Sample- und Frame-Ausschlüsse sowie die endgültigen Leaf-Ecken
   **verbindlich** fest. Alle Ausschlüsse werden vor der Veröffentlichung des
   Beitrags angewandt.
3. Die CPU zählt zellweise die Bounding-Box jedes Leaves auf (identisch zu
   `rasterize_drizzle_stripe`: `floor`/`ceil` über die vier exakten Ecken, keine
   Vorfilterung) und überträgt je (Leaf, Zelle) die vier Ecken (8 Doubles,
   unquantisiert), das Zellrechteck `[x, y, x+1, y+1]`, den Quellindex
   `(source_y, source_x)`, den Kanal und die Leaf-Reihenfolge `leaf_order`.
4. Die GPU berechnet ausschließlich die Polygon-Zell-Fläche über denselben
   bit-exakten Kernel wie der affine Pfad (`d_polygon_rect_area`,
   Sutherland-Hodgman + Shoelace, `--fmad=false`). Sie wertet **kein**
   Verschiebungsfeld, **keine** Subdivision und **keine** diskrete Entscheidung
   aus; `floor`/`ceil`/Bounds sind bereits CPU-seitig fixiert.
5. Der kanonische Schlüssel
   `(frame_order, channel, target_y, target_x, source_y, source_x, leaf_order)`
   und der gesamte Downstream (Sortierung, Q-Faltung, Clipping,
   Profilakkumulation in `accumulate_pair_impl`) bleiben unverändert gemeinsamer
   Hostcode.

**Numerischer Vertrag:** Weil die GPU das Verschiebungsfeld nicht unabhängig
auswertet, entfällt der CPU↔GPU-`exp`-Vergleich vollständig. Es wird **keine
neue Toleranz** eingeführt; §19.5.1 bleibt unverändert. Die einzige GPU-Rechnung
ist die Polygon-Fläche, die bereits als bit-identisch nachgewiesen ist
(§30.49/§30.53). Ergebnis: der hybride Pfad muss byte-identische gespeicherte
Profile liefern wie der reine CPU-Referenzpfad — für MONO und OSC, für alle
Chunkhöhen und für beliebige Leaf-Batch-Grenzen.

**Ressourcen und Restart:** Leaf/Zell-Arbeitspakete werden in **budgetierten
Batches** an die GPU gegeben (`max_batch_items`), nicht die gesamte Geometrie
aller Frames im RAM. Der §19.4-Restart-Vertrag gilt weiter: jeder Device-Fehler
(Alloc, Kernel) führt zu vollständigem CPU-Neustart, kein gemischter Commit. Der
Batch schrumpft bei Alloc-Druck (Halbierung bis zu einer festen Untergrenze),
darunter greift der CPU-Neustart.

**Kennzeichnung:** Sind lokale-Warp-Frames beteiligt, meldet der committende
Store `backend_used = "cuda_hybrid"` (sonst `"cuda"`);
`DrizzleCudaStoreTiming::hybrid_local_frames` bzw.
`acceleration.cuda_stripe_path.hybrid_local_frames` nennt die Anzahl. Die
Timing-Telemetrie schlüsselt den Hybridpfad auf: `hybrid_cpu_seconds`
(CPU-Geometrie + Marshalling + Record-Assembly) gegen `hybrid_gpu_raster_seconds`
(Device-Polygon-Kernel inkl. Transfer), plus `hybrid_leaf_cells`. Eine feinere
Trennung von H2D/D2H und Kernel setzt `.cu`-Instrumentierung voraus.

**Reihenfolge:** Zuerst CPU-Leaf-Erzeugung mit GPU-Rasterisierung und
unveränderten Paritätsgrenzen; danach profilieren. **Durch Messung entschieden
(§30.57):** auf dem realen M42-Hybridlauf steht `hybrid_cpu_seconds = 444,2`
gegen `hybrid_gpu_raster_seconds = 0,60` (~740×) — die lokale Geometrie
(Inversion/Subdivision) *ist* die gesamte Kosten, die reine
Rasterisierungs-Auslagerung bringt für lokale Warps nichts. Nur wenn der
Hybrid-Durchsatz danach relevant wird, wird eine
eigenständige, versionierte Numerikrevision verfolgt, die das Transzendente aus
`smooth_local_basis` durch eine FMA-freie Minimax-Approximation ersetzt
(bit-identisch, ohne §19.5.1-Änderung, aber mit neuem Registrierungs-Hash und
Re-Validierung). Ein voller GPU-Lokalpfad über eine feldbezogene Toleranz
(§19.6.1) bleibt davon unberührt und weiterhin durch §19.5.1 untersagt.

---

<a id="plan-20"></a>

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

<a id="plan-21"></a>

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
(11.12). Weil [§30.3.2](aqmh_cfa_forward_drizzle_entwicklungsprotokoll_de.md#historie-30-3) pro Vertrag mindestens ein zunächst fehlschlagendes
Fixture verlangt, wird die dadurch entstehende zusätzliche Suite-Laufzeit pro
Meilenstein gemessen und in §28 protokolliert; überschreitet sie 90 s auf der
Referenzhardware, werden die teuersten Fixtures hinter ein separates
`tile_compile_slow_fixture_tests`-Target gezogen, das im Standard-CI-Lauf
enthalten bleibt, aber getrennt zeitlich ausgewiesen wird.

---

<a id="plan-22"></a>

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

<a id="plan-23"></a>

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

**Status: im neuen `reconstruct`-Pfad funktional implementiert.**

- [x] Sampling-Plan mit stabilen Frame-IDs, CFA-Ursprung, affiner und lokaler
  Source→Canvas-Abbildung, Hashes und Pixelzentrum-Adapter.
- [x] Eigene `SAMPLING_GEOMETRY`-Phase, CFA-unabhängige Footprint-Analysemaske,
  gemeinsamer Dropletkern für Coverage und Rekonstruktion, flächengewichtetes
  `n_eff`, Support-/Loch-Gates und geometrisches `COMMON_OVERLAP`.
- [x] Normalisierte Quellen statt Nutzsignal-PREWARP im neuen Pfad;
  Vorgängerprüfung vor Resume-Phasenstart und fail-closed bei Beschädigung.
- [x] Synthetische Geometrie-/Gate-/Phasenregressionen vorhanden; frühere
  M31-Verifikation dokumentiert ([§30.23](aqmh_cfa_forward_drizzle_entwicklungsprotokoll_de.md#historie-30-23)). Ditherdiagnostik beeinflusst kein Gate.

**Abgrenzung:** Die Entfernung des alten Produktpfads ist M10. Ein funktionaler
M1-Abschluss ist keine Behauptung, dass bereits alle Produkteinstiege umgestellt
sind; unabhängige Truth- und Echtdaten-Gesamtabnahme gehört M9.

### M2 — CPU Forward-Drizzle 1x Uniform

**Status: CPU-Kern und Store funktional implementiert; breitere Abnahme offen.**

- [x] Exakter Polygon-Rechteck-Schnitt, adaptive lokale Subdivision mit
  Positions-/Flächenguard, CFA-/MONO-Zuordnung und Frame-lokale Aggregation.
- [x] Deterministischer serieller Uniform-Referenzpfad und budgetierte Streifen.
- [x] Transaktionaler Gesamtstore: unveränderliche Generationen, `current.json`
  als Commit, exakte Ebenenmenge, Dimensionen, Identität und Prüfsummen geprüft.
- [x] Synthetische Aperturflux-/Zentroid-/CFA-/Chunk-Tests und injizierte
  Storefehler; Runner- und Vorgängerintegration mit M3 vorhanden.
- [ ] Unabhängige pixelintegrierte Truth-/PSF-Abnahme, echte Prozess-Kill- und
  Wiederanlaufnachweise; Powerloss-Aussagen benötigen gesonderte I/O-Nachweise.

CPU-Parallelisierung ist eine optionale Performancearbeit, kein fehlender
mathematischer Operator. Ein künftiger paralleler Pfad muss Determinismus neu
nachweisen. Injizierte Exceptions ersetzen keinen Prozessabbruchtest.

### M3 — Robustes Clipping und Raw-Forward-Drizzle

**Status: funktional integriert; Cache-Produktpolitik bleibt M8/M10.**

- [x] Geometrisch bestimmtes robustes Clipping und gemeinsame Akzeptanzmaske
  für Uniform, Raw sowie die mit M6 hinzugekommenen Detailprofile.
- [x] Source-Proxy, `GLOBAL_QUALITY`, persistierter `QualityFrameWeightPlan`,
  einmaliges `G_eff` und geprüfte Quell-/Sampling-/Cache-Provenienz.
- [x] `Q_composite` und Missing-/Zero-Veto-Versorgung mit M5 integriert;
  Q-Nullgewicht ändert nicht die gemeinsame geometrische Clippingmaske.
- [x] Uniform/Raw in einer atomaren Generation; geprüfter Resume-Einstieg.
- [x] Kandidatenlisten vor Allokation anhand aller möglichen Framebeiträge
  budgetiert (`forward_drizzle.cpp`, `per_channel`/`plan_drizzle_memory`).
  Die ältere Aussage „nur reaktives Netz“ aus [§30.17](aqmh_cfa_forward_drizzle_entwicklungsprotokoll_de.md#historie-30-17) ist überholt.
- [ ] Produktweite Cache-Retention/Bereinigung und zugehörige GUI-/Resume-
  Kommunikation nach M8/M10 abschließen und testen.

Beschädigte Vorgänger führen zum Abbruch, nicht zum stillen Uniform-Ersatz.
Qualitativer Uniform-Fallback ist ausschließlich die Auswahlentscheidung §15.

### M4 — Internes 2x-Raster und Ausgabegeometrie

**Status: M4-Kern implementiert; ursprünglicher Gesamtumfang noch nicht erfüllt.**

- [x] Explizite Modi `1/1`, `2/1`, `2/2`, Default `2/1`, kein Auto.
- [x] Deterministisches 2×2-Flächenmittel, 4/4-Support und minimales
  Subpixel-`n_eff`; Uniform/Raw/Fine/Medium im Store in `output_scale`.
- [x] WCS-Skalierungsoperator synthetisch geprüft; Downsample-Streaming gegen
  Referenz und verschiedene Chunkhöhen geprüft.
- [ ] Unabhängige Flux-/Zentroid-/PSF-Nachweise für die ausgelieferte
  `2/1`-Geometrie und `2/2` in M9.

**Explizit übertragen, nicht erledigt:** M10 übernimmt die Anwendung der
WCS-/Crop-/Maskengeometrie auf kanonische Ausgaben, §17.4-Photometrie-Rücknahme,
STACKING-Pass-through und BGE/PCC/HMS für OSC/MONO in 1x/2x. M8 übernimmt
Pixelmaßstab-/Flux-/Rauschvertragsanzeige im Report. Die ursprüngliche
M4-Gesamtabnahme bleibt von diesen Nachweisen abhängig (§23.1).

### M5 — Skalenspezifische Source-Q-Maps

**Status: funktional implementiert.**

- [x] CFA-Green-/MONO-Proxy, skalenspezifischer Sink, Legacy-kompatibler
  Composite und jeweils eine residente Skalenkarte während der Erzeugung.
- [x] Multistream-Cache mit separatem Wert-/Hard-Veto-Stream, atomarem
  Metadatencommit, getrennten Identitäts-/Konfigurationshashes und Region-Reads.
- [x] `SOURCE_QUALITY_MAPS`-Phase und fail-closed Cacheprüfung beim Resume.
- [x] `Q_composite` je Quellsample im Rekonstruktor und in der Store-Identität;
  Missing-/Veto-/Quantisierungs-/Regionsregressionen vorhanden.
- [x] M31-Q-Map-/Kandidatenwirkung in [§30.45](aqmh_cfa_forward_drizzle_entwicklungsprotokoll_de.md#historie-30-45) dokumentiert; kein weiterhin
  offener M31-Eintrag aus der früheren Checkliste.

Der letzte Punkt ist historischer Echtdatennachweis, keine neue Messung dieser
Revision. Reales MONO und vollständige objektoffene Qualitätsmatrix bleiben M9.

### M6 — Mehrprofil-Drizzle und Mehrbandfusion

**Status: Funktionalität implementiert; Ressourcenvertrag §11.13 erfüllt (Code + synthetisch). Reale Ressourcenabnahme mit großen Bildern steht noch aus.**

- [x] Fine-/Medium-Profile mit gemeinsamer Clippingmaske, transaktionaler
  Mehrprofilstore und Confidence-Ebenen für `1/1`, `2/1`, `2/2`.
- [x] Maskierte À-trous-Fusion, Bandzuordnung, gemeinsame RGB-Alphas,
  Vertrauensfaktoren, Energieguard und abwärtsbegrenzte B3-Glättung.
- [x] Streifen-/Vollbild-Identitäten und MONO-/OSC-Store-Round-Trips getestet.
- [x] Dreiwegvalidation am festen Sternsatz, N/A-/Mindeststern-/CI-Vertrag,
  Maskenschließen für den Seam-Locus und gehashte Auswahlkonstanten.
- [x] `MULTIBAND`-Phase, Auswahl, unveränderliche `forward_drizzle_raw_*`,
  gewählte `reconstructed_*`, optionale Kontrollen sowie Output-Hashes.
- [x] Geometrie-/Clipping-/Warp-/Phasen-/Ressourcendiagnostik vorhanden;
  GPU-Detailtiming bleibt M7.
- [x] Gemeinsame, überlaufsichere Vorabplanung (`plan_multiband_fusion_memory`,
  sechs Einzelterme) für Fusion, Kandidaten, Validierung und Export; Fail-Closed
  `MULTIBAND_MEMORY_BUDGET` vor der ersten großen Allokation; kein stilles
  Anheben eines expliziten Budgets; Kandidatenkanäle streifenweise in einen
  Spool statt 3·nch voller Ebenen; phasen-lokale `VmRSS`-Messung gegen den Wert
  bei Phasenbeginn, getrennt vom Lebenszeit-Peak ([§30.50](aqmh_cfa_forward_drizzle_entwicklungsprotokoll_de.md#historie-30-50)).
- [x] Synthetische Ressourcenfixtures: Planer-Arithmetik, Fail-Closed-Grenze,
  Chunkhöhen-Invarianz von Kandidaten/Masken/**Auswahl**, injizierter Fehler
  bewahrt den Commitvertrag, Runner-Integration mit den neuen `resources`-Feldern.
- [ ] Reale Ressourcenabnahme mit großen Bildern (M31/M42 Full-Res), kleinen
  Budgets, mehreren Chunkhöhen, OSC/MONO und allen Scale-Modi — benötigt einen
  ausdrücklichen Run-Auftrag; erst danach M6 vollständig abnehmen.

M31/M42-OSC-Ergebnisse sind in [§30.45](aqmh_cfa_forward_drizzle_entwicklungsprotokoll_de.md#historie-30-45)–30.50 dokumentiert; reales MONO fehlt.
Die Erfolgsmeldung „M6 abgeschlossen“ aus [§30.48](aqmh_cfa_forward_drizzle_entwicklungsprotokoll_de.md#historie-30-48) wird auf funktionale Umsetzung
eingeschränkt. Die zwei wissenschaftlichen Auswahlfragen sind in §15.5/15.6
entschieden. Sie erlauben keine nachträgliche Lockerung der Safety-Gates.

### M7 — CUDA

**Status: CUDA-Vorwärtsdrizzle abgeschlossen ([§30.55](aqmh_cfa_forward_drizzle_entwicklungsprotokoll_de.md#historie-30-55)–[§30.57](aqmh_cfa_forward_drizzle_entwicklungsprotokoll_de.md#historie-30-57)).** Auf **allen** produktiv relevanten Configs aktiv und **byte-identisch** zur CPU-Referenz: affin 1/1 + 2/2 (§30.55); **Produktionsconfig Modus 2/1** (`internal_scale=2, output_scale=1`) — die internen 2×-Device-Bänder werden host-seitig 2×2→1× gefaltet (`Downsample2x2StripeAdapter`), kein fehlender Kernel; **lokale-Warp-Frames** über den Hybridpfad §19.6.2 (`backend=cuda_hybrid`, [§30.56](aqmh_cfa_forward_drizzle_entwicklungsprotokoll_de.md#historie-30-56)). **Real belegt** (§30.57): M31 Modus 2/1 `reconstruction_multiband.fits` identisch (= Vor-§30.54-Wert); M42 Hybrid 52 Plane-FITS + fusioniertes Bild identisch (= §30.55-CPU-Referenz). `[cuda-parity]` synthetisch vollständig (Modus 2/1 bei ungerader Bandgrenze + Ganzleinwand, Hybrid inkl. `leaf_order > 0`, Hybrid+Faltung kombiniert). **Vor dem CUDA-Versuch bleibt nur „Device vorhanden".** `s_j`-Pinning: nicht anwendbar solange die Kernel bit-exakt sind. Offen: nur Durchsatz-Optimierung (kein Blocker) und die optionale Numerikrevision für einen vollen GPU-Lokalpfad — Profiling-Befund `hybrid_cpu_seconds` ≫ GPU-Raster (§30.57).**

- [x] Transaktionaler vollständiger CPU-Neustart nach CUDA-Fehler, synthetische
  Fault-Injection einschließlich Store-/Runner-Vergleich ([§30.44](aqmh_cfa_forward_drizzle_entwicklungsprotokoll_de.md#historie-30-44)).
- [x] Device-/Speicherprobe, arithmetischer Auto-Chunking-Planer und
  Host-Chunk-Treiber mit Halbierungs-Retries ([§30.49](aqmh_cfa_forward_drizzle_entwicklungsprotokoll_de.md#historie-30-49)).
- [x] CUDA-Polygon-Rechteck-Fläche und affine Leaf-Ecken mit Komponenten-
  Paritätstests; frühere native Hardwareergebnisse in [§30.49](aqmh_cfa_forward_drizzle_entwicklungsprotokoll_de.md#historie-30-49) dokumentiert.
- [x] Paritäts-Fehlergrenzen festgeschrieben ([§19.5.1](#paritaets-fehlergrenzen)); nachträgliches
  Erweitern durch §19.6 ausgeschlossen.
- [x] Deterministischer Vorwärts-Listenrasterisierer (Uniform) als CPU-Referenz:
  `build_uniform_contrib_list` (Vorzählen + überlaufsichere Reservierung +
  kanonische Sortierung nach dem 7-Tupel-Schlüssel), `reduce_uniform_contrib_list`
  und die Produktionsform `accumulate_uniform_by_frame` (frameweise, ein Frame
  Records Spitzenspeicher) — bit-identisch zur Streaming-Uniform-Referenz über
  affin/Subpixel/Rotation × MONO/OSC (alle Bayer) × lokalem Warp × Randfälle
  und über Streifenzerlegung
  ([§30.51](aqmh_cfa_forward_drizzle_entwicklungsprotokoll_de.md#historie-30-51)).
- [x] Raw-/Detailprofil-/Alpha-Beitragslisten und Clipping mit gemeinsamer
  Akzeptanzmaske und fester Framefolge-Profilreduktion gemäß §19.6:
  `reduce_pixel_profiles` (aus dem Streaming-Pfad herausgezogen, dort
  weiterverwendet) + `accumulate_pair_by_frame` — bit-identisch zu
  `compute_forward_drizzle_uniform_and_raw` inkl. aller `clipping`-Zähler bei
  aktivem Clipping, über MONO/OSC (alle Bayer) × lokalem Warp × Randfälle
  ([§30.52](aqmh_cfa_forward_drizzle_entwicklungsprotokoll_de.md#historie-30-52)).
- [x] FP-Kontraktionspolitik fixiert (`-ffp-contract=off` CPU, `--fmad=false`
  CUDA) → beide Geometriekerne **100 % bit-identisch**; die alten
  `[cuda-parity]`-Toleranzen auf `cpu == gpu` verschärft
  ([§30.53](aqmh_cfa_forward_drizzle_entwicklungsprotokoll_de.md#historie-30-53)).
- [x] Affiner Device-Droplet-Rasterisierer `k_affine_frame_contribs` +
  `forward_drizzle_cuda_affine_frame_contributions` (bandlokale Quellkopie,
  atomarer Dense-Append, Zell-/Kapazitätsgrenzen → CPU-Fallback) und
  `accumulate_pair_by_frame_cuda` — auf der GTX 1660 Ti **bit-identisch** zu
  `accumulate_pair_by_frame` (alle vier Profile, Alpha-Maps, alle drei
  `clipping`-Zähler) über MONO/OSC alle Bayer × Randfälle; lokale-Warp-Frames
  hart abgelehnt ([§30.53](aqmh_cfa_forward_drizzle_entwicklungsprotokoll_de.md#historie-30-53)).
- [x] Lokale-Warp-**Geometrie** auf der GPU **ausgesetzt, solange §19.5.1 gilt**
  ([§19.6.1](#plan-19-6)): die Gauß-Basis (`std::exp`) macht diskrete
  CPU↔GPU-Gleichheit des Verschiebungsfelds unerreichbar; ohne eine (derzeit
  untersagte) §19.5.1-Toleranz gibt es keinen vollen GPU-Lokalpfad. Die
  **Rasterisierung** lokaler Warps läuft dagegen über den hybriden Pfad
  §19.6.2 (CPU-Geometrie → GPU-Fläche,
  [§30.56](aqmh_cfa_forward_drizzle_entwicklungsprotokoll_de.md#historie-30-56)):
  die GPU wertet das Feld nicht aus, §19.5.1 unverändert. Die §19.5-Matrix ist
  für den GPU-Teil mit den affinen Zeilen + der §19.6.2-Rasterzeile komplett
  ([§30.54](aqmh_cfa_forward_drizzle_entwicklungsprotokoll_de.md#historie-30-54)).
- [x] `accumulate_pair_by_frame_cuda` + `plan_cuda_chunking`/`run_cuda_chunked`
  in `persist_forward_drizzle_multiband` verdrahtet: Gate-Prüfung vor dem
  `StoreWriter` (§30.54: drei Gates affin-only/nicht Modus 2/1/Device; **seit
  §30.56 nur noch zwei** — nicht Modus 2/1 und Device, lokale Warps nehmen den
  Hybridpfad §19.6.2), sonst voller CPU-Referenz-Build mit
  `cuda_fallback_reason`; gerätegroße Bänder in denselben
  `multiband_stripe`-Sink. Store **byte-identisch** zum CPU-Streaming- und zum
  Ganzleinwand-CPU-Build (MONO + OSC, subpixel-rotiert, Clipping aktiv) —
  `[drizzle-store][cuda-parity]`
  ([§30.54](aqmh_cfa_forward_drizzle_entwicklungsprotokoll_de.md#historie-30-54)).
- [x] Timing: `DrizzleStoreResult::cuda_timing` → `acceleration.cuda_stripe_path`
  in `forward_drizzle.json` (Bänder, aufgelöste Chunkhöhe, Retryboden,
  `bytes_per_row`, freier VRAM, Streifen-/Gesamtsekunden).
- [x] Realer Großbild-Lauf auf Run-Auftrag ([§30.55](aqmh_cfa_forward_drizzle_entwicklungsprotokoll_de.md#historie-30-55)):
  M31 (40 OSC-Frames, native Leinwand ≈ 5760×5664) in **zwei** Geometrien —
  `internal_scale=1` (21 Bänder) und `internal_scale=2, output_scale=2`
  (Produktions-Oversampling, 92 Bänder, Mehrzellen-Clip voll belastet). In
  beiden ist der committete Profil-Store (52 Plane-FITS) **und**
  `reconstruction_multiband.fits` byte-identisch CPU↔CUDA. M42 (1 lokaler-Warp-
  Frame) lehnt korrekt mit `cuda_fallback_reason` ab, Store byte-identisch zum
  CPU-Lauf. §11.13 auf beiden Datensätzen und allen drei Geometrien bestätigt.
- [x] `forward_drizzle_cuda_runtime_available()` aktiviert — Probe-Form
  `forward_drizzle_cuda_device_memory().free_bytes > 0`; jeder Versuch bleibt
  durch die Drei-Gate-Prüfung abgesichert. Zwei veraltete Testannahmen
  angepasst.
- [x] **Hybrider Pfad §19.6.2 (CPU-Geometrie → GPU-Rasterisierung) für lokale
  Warps** ([§30.56](aqmh_cfa_forward_drizzle_entwicklungsprotokoll_de.md#historie-30-56)).
  `enumerate_drizzle_stripe_leaf_cells` als gemeinsame Zell-Enumeration aus
  `rasterize_drizzle_stripe` herausgezogen (`DrizzleLeafCellSink`, `leaf`-Index
  im `DrizzleAreaSink`-Schlüssel); `cuda_pair_producer` verzweigt bei
  `has_smooth_local_model` in `build_frame_records_hybrid_local`: CPU baut die
  verbindliche Leaf-Liste, GPU rasterisiert je (Leaf, Zelle) über
  `forward_drizzle_cuda_polygon_rect_area_batch` (nur Fläche; `leaf_order`,
  Quellindex und Kanal host-seitig getragen); budgetierte Leaf-Batches
  (`max_batch_items`, Halbierung bei Alloc-Druck bis Untergrenze 4096 →
  `ForwardDrizzleCudaError`/§19.4-CPU-Neustart). `accumulate_pair_by_frame_cuda`
  nimmt `subdivision` + `max_batch_items`; `backend_used = "cuda_hybrid"` bei
  ≥ 1 lokale-Warp-Frame, `DrizzleCudaStoreTiming::hybrid_local_frames` +
  `acceleration.cuda_stripe_path.hybrid_local_frames`. Parität `[cuda-parity]`:
  lokale-Warp-Fixture byte-identisch zum CPU-Pair-Pfad (MONO+OSC), auch bei
  Batch-Grenzen 1/7/64 und ungleichem Streifen-Split; Store-Digest byte-identisch
  CPU↔CUDA.
- [x] **Getrennte Hybrid-Timing-Felder** ([§30.56](aqmh_cfa_forward_drizzle_entwicklungsprotokoll_de.md#historie-30-56)):
  `HybridPathStats` (`cpu_seconds`, `gpu_raster_seconds`, `gpu_batch_calls`,
  `leaf_cells`, `records`) wird von `build_frame_records_hybrid_local` je Band
  akkumuliert und über `accumulate_pair_by_frame_cuda(..., HybridPathStats*)` an
  `persist_forward_drizzle_multiband` gereicht → `DrizzleCudaStoreTiming::hybrid_cpu_seconds`
  / `hybrid_gpu_raster_seconds` / `hybrid_leaf_cells` → `acceleration.cuda_stripe_path`.
  `cpu_seconds` = CPU-Geometrie + Marshalling + Record-Assembly; die Rasterzeit
  ist Transfer + Kernel zusammen (H2D/D2H nicht ohne `.cu`-Instrumentierung
  trennbar).
- [x] **CUDA auf der Produktionsconfig (Modus 2/1)** ([§30.57](aqmh_cfa_forward_drizzle_entwicklungsprotokoll_de.md#historie-30-57)):
  kein fehlender Kernel — der Device-Pfad erzeugt bereits interne 2×-Streifen
  (M31-2/2-Lauf, §30.55). `Downsample2x2Adapter` als `Downsample2x2StripeAdapter`
  (pImpl) aus `output_scale.cpp` exportiert; der CUDA-Zweig faltet jedes
  interne 2×-Band host-seitig 2×2→1× vor dem `StoreWriter`, mit Höhen-Wächter
  gegen stille Desynchronisation. Modus-2/1-Gate entfällt — vor dem CUDA-Versuch
  bleibt nur „Device vorhanden". Parität `[drizzle-store][cuda-parity]`:
  byte-identisch zum CPU-Modus-2/1-Build bei ungerader Bandgrenze (Chunkhöhe 3)
  **und** Ganzleinwand; zusätzlicher Abschnitt Modus 2/1 **mit** lokalem Warp
  (Hybrid §19.6.2 + Faltung in einem Build).
- [x] **`s_j`-Pinning: nicht anwendbar** ([§30.57](aqmh_cfa_forward_drizzle_entwicklungsprotokoll_de.md#historie-30-57)).
  Die `D_fine`/`D_medium`-Grenze ist eine Toleranz relativ zu `s_j`; mit
  `-ffp-contract=off` + `--fmad=false` sind die Paritätstests exakt
  `cpu == gpu`, die Toleranz ungenutzt, `s_j` hat nichts zu kalibrieren. Wird
  erst mit einer nicht-bit-exakten Numerikrevision (Option D) erforderlich.
- [x] **Reale Volllläufe (§30.57, auf Auftrag):** M31 Produktionsconfig (Modus
  2/1) — `reconstruction_multiband.fits` **byte-identisch** CPU↔CUDA
  (`49ca284c…`, = Vor-§30.54-Baustand), `backend=cuda`, keine Ablehnung.
  M42 (lokaler Warp, 1/1) — 52 Plane-FITS **und** fusioniertes Bild
  byte-identisch CPU↔CUDA (`d869f53b…` / `07f09104…`, = §30.55-CPU-Referenz),
  `backend=cuda_hybrid`, `hybrid_local_frames=1`. §11.13 grün auf beiden.
- [x] **Profiling mit dem Timing-Split** ([§30.57](aqmh_cfa_forward_drizzle_entwicklungsprotokoll_de.md#historie-30-57)):
  auf M42 dominiert `hybrid_cpu_seconds` (444 s, CPU-Inversion/Subdivision) die
  GPU-Rasterzeit (`hybrid_gpu_raster_seconds` 0,60 s) um **~740×**. Die
  Rasterisierungs-Auslagerung allein bringt für lokale Warps nichts. → **Wenn**
  der Hybrid-Durchsatz relevant wird, ist der nächste Schritt die versionierte
  Numerikrevision (`std::exp` → FMA-freie Minimax-Approximation, damit die
  ganze Geometrie bit-identisch auf der GPU läuft), **nicht** eine
  §19.5.1-Toleranz.
- [ ] **Durchsatz-Optimierung** (kein M7-Blocker, §19.6 stellt Bit-Exaktheit
  voran): CUDA-FORWARD_DRIZZLE ≈ 1,1× (M31 Modus 2/1) bis ≈ 2,1× (M42 1/1
  Hybrid) langsamer als CPU — host-gebundene Sortier-/Reduktionsphase, bei
  Hybrid zusätzlich die CPU-Geometrie. Optional und nachgelagert; siehe die
  Numerikrevision oben.

À-trous bleibt CPU. Komponentenfortschritt ist kein Nachweis einer beschleunigten
Gesamtrekonstruktion. Der M6-Ressourcenvertrag (§11.13) ist auf realen
M31/M42-Läufen geschlossen (§30.55) und wurde auf beiden realen M7-Läufen
(§30.57) erneut grün bestätigt — working-set im Budget, temp-space ok,
Phase-RSS im Envelope, kein Spool. „M7 abgeschlossen" heißt: die CUDA-Arbeit ist
fertig und byte-identisch; die **vollständige** wissenschaftliche Abnahme
(Paritätsmatrix §19.5, Powerloss) bleibt M9.

### M8 — GUI, Report und vollständige Dokumentation

**Übernahme aus M4 (§23.1):** korrekter Pixelmaßstab, Fluxraum und
Rausch-/Korrelationsvertrag im Report für OSC/MONO und alle Scale-Modi.

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

**Übernahme aus M4/M3 (§23.1):** kanonische WCS-/Crop-/Maskenausgabe,
§17.4-Photometrie-Rücknahme, STACKING-Pass-through, BGE/PCC/HMS in 1x/2x
für OSC/MONO sowie umgesetzte Cache-Retention/Bereinigung. Diese Punkte sind
Releaseblocker; ihre Umordnung bedeutet keine Abnahme.

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

<a id="entscheidung-abnahme"></a>

### 23.1 Abnahmegrenzen und Zuordnung übertragener Pflichten

**Festlegung: 2026-09-07.**

**Gewählt:** vorhandene M-Nummern behalten; algorithmischen Kern, Produkt-
Integration und Releaseabnahme ausdrücklich getrennt ausweisen. Kein offener
Punkt verschwindet durch Umbenennung oder Verschiebung.

| Ursprünglicher Punkt | Verantwortlicher Meilenstein | Verbindlicher Nachweis |
|---|---|---|
| 2x-Raster, 2×2-Operator, Storegeometrie, WCS-Hilfsoperator | M4-Kern | Handrechnung, Chunkparität, OSC/MONO und alle Scale-Modi |
| WCS/Crop/Masken in kanonischen Dateien, Photometrie-Rücknahme | M10 | Pixel-/Himmelskoordinaten, Flux-/Einheitenvertrag und Hash-/Outputprüfung |
| STACKING-Pass-through, BGE/PCC/HMS | M10 | Durchgängiger OSC-/MONO-Test in 1x/2x, kein doppeltes Downsampling |
| Pixelmaßstab, Fluxraum, Rauschdiagnostik im Report | M8 | Anzeige stimmt mit tatsächlichen FITS-/Runartefakten überein |
| Cache-Retention und Bereinigung | M10, Oberfläche/Erklärung M8 | Explizites Behalten/Löschen und danach korrekte Resume-Freigabe/Ablehnung |
| CPU-Fusion/Validierung/Export im Gesamtbudget | M6 | Vorabplanung und große Ressourcenfixtures gemäß §11.13 |
| Wissenschaftliche Qualitäts-/Truth-/Datensatzmatrix | M9 | Alle vorgeschriebenen Datentypen und Releasegates, N/A nie als Pass |

„M4-Kern implementiert“ ist zulässig. „Ursprüngliches M4 vollständig abgenommen“
erst nach den übertragenen Nachweisen. „M1–M6 vollständig abgenommen“ bleibt
bis zu ihren offenen Vertragsnachweisen unzulässig. M7-Komponentenarbeit darf
fortgesetzt werden; M6-Ressourcenlücke vor produktiver GPU-Freigabe schließen.
M8/M10-Produktarbeit kann parallel zum wissenschaftlichen Nachweis vorbereitet
werden; der Release bleibt von der vollständigen M9-Matrix abhängig.

---

<a id="plan-24"></a>

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

<a id="plan-25"></a>

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

<a id="plan-26"></a>

## 26. Verbindlich entschiedene Punkte und verbleibende Nachweise

Die Grundentscheidungen vom 2026-09-02 werden durch die Fachentscheidungen
in §15.5/15.6, §19.6 und §23.1 sowie den Ressourcenvertrag §11.13
(2026-09-07) präzisiert. Die vier Auditfragen sind dort entschieden; Implementierungs- und
Abnahmenachweise bleiben offen. Geänderte Berechnungssemantik benötigt vor
Aktivierung angepasste Hash-/Artefaktdomänen und erneut bestandene Fixtures.
Reine Dokumentationskorrekturen ändern keine bestehenden Artefakte oder Hashes.
Die Festlegungen sind anwendungsspezifische Entwurfsentscheidungen, kein
Beweis eines universell optimalen Rekonstruktors. Noch fehlender Code wird
nicht durch eine Planentscheidung als implementiert markiert.

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

<a id="plan-27"></a>

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

<a id="plan-28"></a>

## 28. Evidenz und verbleibende Nachweise

Diese Tabelle trennt Implementierungsbelege, ausgeführte Tests und historische
Echtdatenangaben. Sie enthält keine neue wissenschaftliche oder native GPU-Abnahme.
Codepfade beziehen sich auf den geprüften Arbeitsbaum; ältere Zeilenangaben
und Testergebnisse im Entwicklungsprotokoll können seitdem verändert sein.

| Aussage | Beleg | Aussagegrenze / verbleibende Abnahme |
|---|---|---|
| Neuer Runner nutzt eigene Geometrie-/Qualitäts-/Rekonstruktionsphasen | [Runner](../../tile_compile_cpp/apps/runner_forward_drizzle.cpp), [Regressionen](../../tile_compile_cpp/tests/test_runner_forward_drizzle.cpp) | Produkt-Cutover und Downstream bleiben M10 |
| Transaktionaler Profilstore und CPU-Neustartvertrag vorhanden | [Store](../../tile_compile_cpp/src/reconstruction/drizzle_profile_store.cpp), [Tests](../../tile_compile_cpp/tests/test_drizzle_profile_store.cpp) | Injizierte Exceptions ersetzen keinen Prozess-Kill-/Powerloss-Nachweis |
| Fusion und Dreiwegauswahl implementiert | [Fusion](../../tile_compile_cpp/src/reconstruction/multiband_fusion.cpp), [Validation](../../tile_compile_cpp/src/reconstruction/multiband_validation.cpp) | Gemeinsamer Ressourcenvertrag offen (§11.13) |
| Gezielte Testausführung bestanden | 21 Drizzle-/Store-/Runner-Fälle, 5.426 Assertions; 24 `multiband*`-Fälle, 19.542 Assertions | Vorhandenes Binary; kein Neubuild, keine vollständige Suite oder neue Hardwareabnahme |
| M31/M42: Auswahl, Seam-Defekte und Ausgabedateien historisch untersucht | [§30.45](aqmh_cfa_forward_drizzle_entwicklungsprotokoll_de.md#historie-30-45), [§30.46](aqmh_cfa_forward_drizzle_entwicklungsprotokoll_de.md#historie-30-46), [§30.47](aqmh_cfa_forward_drizzle_entwicklungsprotokoll_de.md#historie-30-47), [§30.48](aqmh_cfa_forward_drizzle_entwicklungsprotokoll_de.md#historie-30-48) | In dieser Revision nicht neu gemessen; beide OSC, reales MONO fehlt |
| Zwei CUDA-Geometriekerne mit früherer nativer Komponentenparität | [§30.49](aqmh_cfa_forward_drizzle_entwicklungsprotokoll_de.md#historie-30-49), [CUDA-Code](../../tile_compile_cpp/src/reconstruction/forward_drizzle_cuda_device.cu) | Kein produktiver GPU-Rekonstruktionspfad; vollständige Paritätsmatrix offen |
| Qualitätsgewinn / Releasefähigkeit der Gesamtmethode | Pflichtmatrix M9, Definition of Done §27 | Noch nicht vollständig nachgewiesen; N/A ist kein bestandenes Releasegate |

Ein reproduzierbarer künftiger Abnahmebericht nennt Code-/Binary-Identität,
effektive Config, Fixture-/Datensatzidentität, Testfilter, Hardware, Ergebnisse
und Grenzen. Historische Kennzahlen werden nicht ungeprüft als aktueller
Status übernommen.

---

<a id="plan-29"></a>

## 29. Quellen und Entwicklungsprotokoll

### 29.1 Externe fachliche Quellen

- **NVIDIA, Floating Point and IEEE 754, CUDA Toolkit 11.6.1.**
  [Dokumentation](https://docs.nvidia.com/cuda/archive/11.6.1/floating-point/index.html).
  Grundlage zu Nichtassoziativität, FMA und CPU-/GPU-Rundungsunterschieden;
  verwendet in §19.6. Die konkrete Listenstrategie ist eine eigene
  Entwurfsentscheidung, keine Vorgabe dieser Quelle.
- **STScI, DrizzlePac: 3.3 Weight Maps and Correlated Noise.**
  [Fachkapitel](https://hst-docs.stsci.edu/drizzpac/chapter-3-description-of-the-drizzle-algorithm/3-3-weight-maps-and-correlated-noise).
  Grundlage zu Gewichtskarten und räumlich korreliertem Drizzle-Rauschen;
  verwendet in §15.6. Die Quelle begründet keinen universellen 5-%-Grenzwert.

Beide Quellen wurden für die Entscheidungsrevision vom 2026-09-07 konsultiert.
Die wissenschaftlichen Grenzen und eigenen Schlussfolgerungen stehen direkt
bei den jeweiligen Fachentscheidungen.

### 29.2 Interne Belege und historische Entwicklung

- [Code- und Vertragsaudit 2026-09-05](aqmh_cfa_forward_drizzle_audit_2026-09-05_de.md).
- [Entwicklungsprotokoll 2026-09-01 bis 2026-09-07](aqmh_cfa_forward_drizzle_entwicklungsprotokoll_de.md):
  frühere §0.1–0.5, §28/29, [§30.2](aqmh_cfa_forward_drizzle_entwicklungsprotokoll_de.md#historie-30-2)–30.49 und §31 mit ihren ursprünglichen
  Abschnittsnummern und unveränderten Eintragstexten.

Historische Beleg-IDs bleiben absichtlich stabil. Die frühere Statusübersicht
§30.1 steht jetzt in §0.1. Die frühere Entscheidungsrevision §32 wurde wie folgt
in die Fachkapitel integriert:

| Bisher | Aktueller Ort |
|---|---|
| §32.1 CUDA | [§19.6](#entscheidung-cuda) |
| §32.2 Sternevidenz | [§15.5](#entscheidung-evidenz) |
| §32.3 Rauschgate | [§15.6](#entscheidung-rauschen) |
| §32.4 Meilensteinabnahme | [§23.1](#entscheidung-abnahme) |
| §32.5 Ressourcenarbeit | [§11.13](#ressourcen-restarbeit) |
