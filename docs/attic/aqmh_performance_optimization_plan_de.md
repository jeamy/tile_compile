# AQMH-Performance-Optimierungsplan

Stand: 24. Juli 2026

Ausgangspunkt ist der Run
`M31-default-afix1_20260722_111345` mit 645 Frames und einem Canvas von
3924 × 2310 Pixeln. `AQMH_MAPS` benötigte 1191,9 Sekunden,
`AQMH_RECONSTRUCTION` 2622,0 Sekunden. Zusammen entsprechen beide Phasen
79,3 % der gesamten Run-Laufzeit.

## Statuslegende

- `offen`: noch nicht implementiert
- `in Arbeit`: Implementierung oder Verifikation läuft
- `erledigt`: implementiert und mit den im Abschnitt genannten Tests geprüft
- `teilweise erledigt`: nutzbarer Teil umgesetzt; ausdrücklich genannter Rest
  bleibt offen

## Unveränderliche Anforderungen

- AQMH und Classic Tile Compile bleiben unabhängige Rekonstruktionsmethoden.
- Die Roh-AQMH-Ausgabe und der Uniform-Control bleiben als unveränderliche
  Validierungsreferenzen erhalten.
- Star-Tail- und Elongationsvergleiche verwenden identische Sternpositionen.
- CPU-, OpenCL- und CUDA-Pfade behalten dieselben Masken-, NaN-, Zero-Veto-,
  Sigma-Clip- und Resume-Verträge.
- Optimierungen dürfen die vorhandenen Qualitäts-Gates nicht umgehen.

## Phase 1: Messung und korrekte Telemetrie

Status: `erledigt`

- Teilzeiten für Rekonstruktionskern, Uniform-Control und
  Postprocessing/Validierung erfassen.
- `uniform_control_same_pass` nur dann melden, wenn das Control tatsächlich im
  selben Datenpass erzeugt wurde.
- den tatsächlich verwendeten Selection-/Sortieralgorithmus melden.
- den AQMH-Laufzeitvergleich mit Classic-Stacking als `nicht anwendbar`
  kennzeichnen, statt `ratio_ok=true` zu melden.

Abnahme:

- Artefakte unterscheiden anwendbar, bestanden und nicht anwendbar.
- Telemetrie behauptet weder Quickselect noch Same-Pass, wenn der ausgeführte
  Pfad etwas anderes verwendet.

Umgesetzt:

- `timing_seconds` trennt Rekonstruktionskern, optionalen
  Uniform-Control-Fallback und Postprocessing/Validierung.
- `uniform_control_same_pass`, `uniform_control_mode`,
  `weighted_selection` und `prefetch_strategy` beschreiben den tatsächlich
  ausgeführten Pfad.
- Der Classic-Laufzeitvergleich erhält für AQMH
  `tile_analysis_ratio_applicable=false` und
  `tile_analysis_ratio_ok=null`.

## Phase 2: Uniform-Control nur einmal berechnen

Status: `erledigt`

- Im CUDA-Datenpass Summe und Anzahl aller gültigen Framewerte je Pixel
  mitführen.
- daraus den ungewichteten Uniform-Control und seine Valid-Mask erzeugen.
- den separaten sigma-geclippten CUDA-Control-Kernel nicht mehr starten.
- den nachgeschalteten CPU-Komplettdurchlauf nur noch als Fallback verwenden,
  wenn ein Backend keinen gültigen Uniform-Control geliefert hat.

Abnahme:

- Uniform-Control entspricht der bisherigen CPU-Referenz innerhalb der
  dokumentierten Float-Toleranz.
- gültige echte Nullpixel bleiben über die separate Valid-Mask gültig.
- CUDA liest die Prewarp-Frames nicht erneut für einen CPU-Control-Durchlauf.

Umgesetzt:

- Der gewichtete CUDA-Kernel berechnet den ungewichteten Mittelwert aus allen
  gültigen Samples im selben Frame-Datenpass.
- Eine separate Byte-Maske unterscheidet gültige Nullwerte von nicht
  unterstützten Pixeln.
- Der frühere separate CUDA-Control-Kernel wurde entfernt.
- Der Runner startet den CPU-Regionendurchlauf nur, wenn das gewählte Backend
  keinen vollständigen Control samt Valid-Maske geliefert hat.

## Phase 3: Q-Map-Prefetch und Region-Streaming

Status: `erledigt`

- den Full-Map-Prefetch entfernen, wenn die Rekonstruktion Regionen streamt.
- keine 645 vollständigen Maps in einen LRU mit nur wenigen residenten Maps
  laden.
- Regionen weiterhin parallel und bedarfsgesteuert pro Chunk laden.

Abnahme:

- `AQMH_MAPS` liest nicht mehr jede gerade geschriebene Map vollständig zurück.
- Rekonstruktion und Resume funktionieren weiterhin mit dem vorhandenen
  diskbasierten Cache.

Umgesetzt:

- `AQMH_MAPS` erzeugt keinen Full-Map-Prefetch-Coordinator mehr und publiziert
  keine frisch geschriebenen Maps in den kleinen LRU.
- Die bestehende parallele, bedarfsgesteuerte Regionsbeladung der
  Rekonstruktionschunks bleibt unverändert aktiv.

## Phase 4: Diagnostik nur bei Anforderung berechnen

Status: `erledigt`

- Regionen nur bei aktivierter Full-Diagnostik und `regions=true` extrahieren.
- reine Diagnosequantile bei deaktivierter Diagnostik überspringen.
- für globale Gewichte erforderliche Sharpness-, SNR- und
  Hintergrundzusammenfassungen unabhängig davon weiter berechnen.

Abnahme:

- `regions=false` führt zu keinem Connected-Component-/Region-Durchlauf.
- AQMH-Globalgewichte bleiben vorhanden und endlich.

Umgesetzt:

- Map-Quantile werden nur bei aktivierter Diagnostik mit einem Level ungleich
  `none` berechnet.
- Regionen werden nur für aktivierte Full-Diagnostik mit `regions=true`
  extrahiert und geschrieben.
- Sharpness-, SNR- und Hintergrundwerte für die globalen Gewichte werden davon
  unabhängig weiterhin erzeugt.

## Phase 5: Validierungsreferenzen wiederverwenden

Status: `erledigt`

- Sternpositionen und Metriken für unveränderlichen Raw-AQMH- und
  Uniform-Control einmal vorbereiten.
- bei Kandidaten nur die Kandidatenseite neu messen.
- dieselben vorbereiteten Sternpositionen für alle Vergleiche gegen dieselbe
  Referenz verwenden.
- die zwölfstufige Abschwächungssuche behält eine abschließende vollständige
  Qualitätsprüfung.

Abnahme:

- keine unabhängigen Sternpopulationen in Vergleichsmetriken.
- identische Gate-Entscheidungen in Regressionstests.
- Baseline-Metriken werden nicht in jeder Suchiteration erneut berechnet.

Umgesetzt:

- `AqmhValidationReference` speichert Referenzmetriken und die einmal
  detektierten Sternpositionen.
- Uniform-Control und Raw-AQMH werden je einmal vorbereitet; alle Kandidaten
  einschließlich der Abschwächungssuche verwenden diese Referenzen erneut.
- Ein Regressionstest vergleicht den gecachten und den bisherigen direkten
  Vergleich über alle Metriken, Anwendbarkeitsflags und Regressionen.

## Phase 6: Lineare lokale Varianz in AQMH_MAPS

Status: `erledigt`

- die naive Fensteriteration durch separierbare Summen, Quadratsummen und
  Support-Counts ersetzen.
- NaN- und Maskenverhalten sowie die Sonderbehandlung für weniger als drei
  gültige Samples erhalten.
- vorhandene Mittelwert-/Count-Zwischenergebnisse nach Möglichkeit
  wiederverwenden.

Abnahme:

- Vergleichstests gegen die bisherige Referenz für Innenbereich, Ränder, NaNs
  und kleine Supports.
- asymptotische Laufzeit `O(Breite × Höhe)` unabhängig von der Fensterfläche.

Umgesetzt:

- Horizontale und vertikale Sliding-Window-Summen berechnen Summe,
  Quadratsumme und gültigen Support in linearer Zeit.
- Ein direkter Vergleich gegen die naive Fensterreferenz prüft mehrere Radien,
  Ränder, NaNs und kleine Supports.

## Phase 7: CUDA-Sigma-Clip und Selection

Status: `teilweise erledigt`

- bereits sortierte Ordnungsinformationen innerhalb einer Iteration
  wiederverwenden.
- redundante Sortierungen für Noise-Floor und MAD entfernen.
- den großen threadlokalen Scratch-Bedarf reduzieren.
- als weiterführenden Schritt die threadserielle Sortierung durch eine
  deterministische Warp-/Block-Selection ersetzen, sofern CUDA-Regressionstests
  die CPU-Semantik bestätigen.

Abnahme:

- CUDA- und CPU-Ergebnisse stimmen in den bestehenden AQMH-Toleranzen überein.
- weniger vollständige Sortierungen je Sigma-Clip-Iteration.
- keine falsche Quickselect-Angabe im Artefakt.

Umgesetzt:

- Die bereits wertsortierte Ordnung liefert den ungewichteten Median für den
  Noise-Floor, bevor der Weighted-MAD-Schritt sie überschreibt.
- Zwei `MaxFrames` große threadlokale Float-Scratch-Arrays wurden entfernt;
  Small-N-Mediane verwenden nun ebenfalls den gemeinsamen Indexpuffer.
- Die Artefaktangabe benennt die aktuelle threadserielle Sortierung korrekt.

Offen:

- Die vollständige Umstellung auf Warp-/Block-Selection bleibt ausstehend.
  Sie verändert den zentralen numerischen CUDA-Pfad und darf gemäß
  GPU-Semantikvertrag erst mit nativen CPU/CUDA-Vergleichstests auf realer
  Hardware abgeschlossen werden. Der neue M31-Lauf bestätigt zwar die native
  CUDA-Ausführung ohne Fallback, ersetzt aber keinen kontrollierten
  CPU-/CUDA-Vergleich mit identischer Eingabe.

## Phase 8: Build, Tests und Laufzeitvergleich

Status: `erledigt`

- relevante AQMH-Unit-Tests ausführen.
- `tile_compile_runner` und vollständige C++-Tests bauen.
- vollständige Testsuite ausführen.
- CUDA-Tests nur mit tatsächlich verfügbarem GPU-Zugriff als native
  CUDA-Verifikation werten.
- einen neuen Bildverarbeitungsrun ausschließlich nach ausdrücklicher Freigabe
  starten; vorhandene Runs bleiben unverändert.

Abnahme:

- Build und relevante Tests sind erfolgreich.
- verbleibende Risiken und nicht nativ geprüfte GPU-Pfade sind dokumentiert.

Ergebnis:

- `tile_compile_runner` und das C++-Testtarget wurden einschließlich der
  CUDA-Übersetzung erfolgreich gebaut.
- AQMH: 73 Testfälle, davon 72 erfolgreich und ein nativer CUDA-Test wegen
  fehlendem CUDA-Gerät übersprungen; 12.515 von 12.515 Assertions erfolgreich.
- Gesamtsuite: 240 Testfälle, davon 239 erfolgreich und derselbe native
  CUDA-Test übersprungen; 27.830 von 27.830 Assertions erfolgreich.
- Der übersprungene Test meldet
  `no CUDA-capable device is detected`. Das ist als ausstehende
  Hardwareverifikation und nicht als bestandener CUDA-Lauf dokumentiert.
- Es wurde kein neuer Bildverarbeitungsrun gestartet und kein vorhandener Run
  verändert.

## Phase 9: Auswertung des nativen GPU-Laufs

Status: `erledigt`

Ausgewertet wurde der vom Benutzer bereitgestellte Run
`m31-default-gpu-opti1_20260724_124627`. Direkter Konfigurationsvergleich ist
`M31-default-afix-picking1_20260722_132406`; der einzige relevante Unterschied
ist das im neuen Lauf deaktivierte AQMH-Cherry-Picking.

Ergebnis:

- Gesamtlaufzeit: 3247 statt 4025 Sekunden, also 19,3 % schneller.
- `AQMH_RECONSTRUCTION`: 1221 statt 2049 Sekunden, also 40,4 % schneller.
- Gegenüber dem ursprünglichen Ausgangslauf sank die Gesamtlaufzeit um 32,5 %
  und die Rekonstruktionszeit um 53,4 %.
- Der native Pfad meldet `cuda_native_v0_2`, 28 Chunks,
  keinen Beschleunigungs-Fallback und keine CUDA-Allokationswiederholung.
- Der Uniform-Control wurde im selben GPU-Datenpass erzeugt; sein
  nachgeschalteter Fallback benötigte 0 Sekunden.
- `AQMH_MAPS` schrieb 2,923 GB Cache-Daten, las sie aber nicht mehr unmittelbar
  vollständig zurück. Die Phase blieb mit 1110 Sekunden dennoch nahezu
  unverändert und ist damit der größte verbleibende Einzelengpass.
- Das globale Hintergrund-RMS verbesserte sich gegenüber dem
  Cherry-Pick-Vergleich um 26,4 %. Die globale FWHM änderte sich nur um
  +0,069 % und ist damit praktisch stabil.
- Cherry-Picking bleibt für diesen Datensatz deaktiviert: Der Vergleichslauf
  lag beim AQMH-Hintergrund 71,4 % über dem Uniform-Control, der neue Lauf bei
  20,2 %.

Qualitätsentscheidung:

- Der unveränderte Raw-AQMH-Kandidat verfehlt weiterhin ausschließlich das
  Hintergrund-Gate gegenüber dem Uniform-Control.
- Der vollständige strukturmaskierte Kandidat besteht alle Gates gegenüber dem
  Uniform-Control und verbessert dort FWHM und Hintergrund.
- Gegenüber der unveränderlichen Raw-AQMH-Referenz verschlechtert er den
  Seam-Score jedoch um 6,29 % bei erlaubten 5 %. Die Auswahl von Raw-AQMH war
  deshalb korrekt; der Grenzwert wird nicht gelockert.
- Alle drei betrachteten Runs enden technisch vollständig, werden aber wegen
  dieses AQMH-Qualitäts-Gates als `failed` markiert.

## Phase 10: Kandidatensuche und Ablehnungsdiagnostik

Status: `erledigt`

- Den Vergleich des vollständigen strukturmaskierten Kandidaten gegen die
  unveränderliche Raw-AQMH-Referenz im Rekonstruktionsartefakt ausgeben.
- Strategie, Zahl der Prüfungen, Zahl zulässiger Kandidaten und bestes Alpha
  der Abschwächungssuche protokollieren.
- Die bisherige binäre Suche ersetzen: Die kombinierte Gate-Entscheidung ist
  über Alpha nicht zwingend monoton, weil Uniform-Control und vollständiger
  Detailkandidat an unterschiedlichen Gates scheitern können.
- Alpha zunächst in absteigenden Achtelschritten prüfen und nur den höchsten
  zulässigen Bereich vier Schritte lokal verfeinern.

Abnahme:

- Jeder Kandidat muss weiterhin sowohl gegen Uniform-Control als auch gegen
  immutable Raw-AQMH bestehen.
- Kein Qualitätsgrenzwert wird gelockert.
- Ein Lauf ohne zulässigen Grobkandidaten benötigt sieben statt zwölf
  Alpha-Prüfungen; unveränderliche Referenzmetriken und Sternpositionen werden
  weiterhin wiederverwendet.
- Das Artefakt erklärt direkt, welches Raw-AQMH-Gate den vollständigen
  Strukturkandidaten abgelehnt hat.

Verifikation:

- `tile_compile_runner` und das vollständige C++-Testtarget bauen erfolgreich.
- AQMH-Auswahl: 74 Testfälle, davon 73 erfolgreich und der native CUDA-Test
  mangels sichtbarem CUDA-Gerät übersprungen; 12.519 Assertions erfolgreich.
- Gesamtsuite: 240 Testfälle, davon 239 erfolgreich und derselbe CUDA-Test
  übersprungen; 27.830 Assertions erfolgreich.

## Phase 11: Verbleibender AQMH_MAPS-Engpass

Status: `teilweise erledigt`

- Teilzeiten pro Frame für lokale Varianz, Strukturtensor, Schärfe, SNR,
  Hintergrund und Cache-Schreiben erfassen.
- Die Speicherplanung prüfen: Der neue Lauf reduzierte bei acht angeforderten
  Workern auf sieben effektive Worker. Eine Erhöhung darf nur erfolgen, wenn
  gemessener Spitzenbedarf und bestehendes Speicherbudget sie sicher erlauben.
- Erst anhand dieser Teilzeiten den dominanten Rechenkern optimieren.
- Keine GPU-Portierung beginnen, bevor CPU-Profil und Transferkosten einen
  belastbaren Nutzen zeigen.

Umgesetzt und nativ gemessen:

- robuste Mediane verwenden lineare Ordnungsstatistik statt vollständiger
  Sortierung; NaN-Filterung erfolgt ohne eine zusätzliche Vollkopie.
- Signal- und Noise-Mean in `phi_snr` teilen sich einen fusionierten
  separierbaren Fensterdurchlauf und eine gemeinsame Support-Count-Map.
- Der vollständige 645-Frame-Test benötigte 628,7 statt 1110 Sekunden:
  43,4 % weniger Laufzeit.
- Der Test konnte acht statt sieben Worker verwenden. Der gemessene Gewinn
  enthält deshalb sowohl die Codeoptimierung als auch den Wegfall der früheren
  speicherbedingten Workerbegrenzung.

Offen:

- Teilzeiten innerhalb einer einzelnen Map-Berechnung ergänzen, damit
  Codegewinn und Worker-Skalierung getrennt quantifiziert werden können.
- Spitzen-RSS pro Worker messen und die Schätzung von 887 MB gegen den
  tatsächlichen Bedarf validieren.

Abnahme:

- Die Summe der Teilzeiten erklärt den Großteil der `AQMH_MAPS`-Laufzeit.
- Workerplanung dokumentiert Budget, Schätzung und tatsächlich beobachteten
  Spitzenbedarf konsistent.
- NaN-, Masken-, Rand- und Resume-Semantik bleiben durch Vergleichstests
  unverändert.

## Phase 12: Registrierungs-Guard prüfen

Status: `offen`

Der neue Lauf dämpfte 44 Frames über den AQMH-Registrierungs-Guard; im direkten
Vergleichslauf waren es 8. Gleichzeitig sank die Registrierungszeit von etwa
215 auf 195 Sekunden. Die Laufzeitverbesserung allein erklärt die deutlich
größere gedämpfte Population nicht.

- die 44 betroffenen Frame-IDs, Registrierungsquellen, Chain-Depths,
  Korrelationswerte und Dämpfungsfaktoren gegen den Vergleichslauf stellen.
- unterscheiden, ob sich die Registrierungsergebnisse geändert haben oder nur
  die Guard-Auswertung beziehungsweise Telemetrie.
- prüfen, ob die gedämpften Frames räumlich oder zeitlich gehäuft sind und ob
  sie die AQMH-Hintergrundregression beeinflussen.
- Schwellenwerte erst nach dieser Ursachenanalyse ändern.

Abnahme:

- Jede zusätzliche Dämpfung ist auf eine konkrete Änderung der
  Registrierungsmetrik oder des Quellpfads zurückgeführt.
- Unveränderte Eingaben ergeben deterministische Guard-Entscheidungen.
- Eine eventuelle Korrektur wird gegen globale Ausgabe, Raw-AQMH,
  Uniform-Control und Registrierungsartefakte geprüft.

## Phase 13: Zweite Rekonstruktionsoptimierung

Status: `teilweise erledigt`

Umgesetzt:

- Vollbilddaten für Seam-Score und First-Difference-Hintergrundrauschen in
  einem gemeinsamen Durchlauf erfassen.
- Median, MAD und Epsilon-Skala ohne wiederholte vollständige Sortierungen und
  Zwischenkopien berechnen.
- die Alpha-Suche bei vollständig unzulässigen Kandidaten von zwölf auf sieben
  Vollbildprüfungen begrenzen.
- Referenzmetriken und identische Sternpositionen weiterhin unverändert
  wiederverwenden.

Native Vollmessung:

- Im 64-Frame-GPU-Smoke-Test benötigte
  `postprocessing_and_validation` 18,6 Sekunden.
- Im vollständigen 645-Frame-Resume sank derselbe Abschnitt von 224,0 auf
  14,68 Sekunden: 93,4 % schneller beziehungsweise 209,3 Sekunden eingespart.
- Raw-AQMH-, Uniform-Control- und Strukturkandidaten-Metriken reproduzieren den
  Referenzlauf; `raw_aqmh` bleibt aus demselben Seam-Guard-Grund ausgewählt.
- Das Raw-FITS ist nicht bitidentisch: 1,14 % der Pixel unterscheiden sich,
  bei einer mittleren absoluten Differenz von 0,0148. Der Referenzlauf
  verwendete 84, der Resume 86 Chunk-Zeilen. Vor weiteren Kerneländerungen ist
  deshalb ein kontrollierter 84/86-Chunk-Invarianzvergleich erforderlich.
- Bei unveränderter Referenz-Kernzeit ergäbe das 1011,6 statt 1221 Sekunden für
  die gesamte Rekonstruktionsphase, also 17,2 % weniger.
- Der gemessene Resume-Kern benötigte wegen deutlich langsamerer
  Region-I/O 1308,1 statt 996,9 Sekunden. Die tatsächliche Resume-Phase lag
  deshalb bei 1323,0 Sekunden. Dieser I/O-beeinflusste Wert ist kein
  Kernel-Speedupvergleich.

Verworfene Kandidaten:

- Thread-serieller CUDA-Quickselect: bei 645 Frames nur 89 von 2310 Zeilen
  nach rund 164 Sekunden und damit klar langsamer als Shellsort. Nicht
  übernommen.
- Frame-major-Eingabelayout: anfangs schnellere Chunks, über 65 % des Canvas
  jedoch keine Verbesserung gegenüber pixel-major. Nicht übernommen.
- Qualitätsgrenzen und numerische Semantik wurden für keinen
  Performanceversuch verändert.

Verifikation:

- Runner und Testtarget bauen einschließlich CUDA-Übersetzung erfolgreich.
- AQMH: 75 Testfälle, 74 erfolgreich, nativer CUDA-Unit-Test mangels im
  Sandbox-Prozess sichtbarem Gerät übersprungen; 12.522 Assertions erfolgreich.
- Gesamtsuite: 241 Testfälle, 240 erfolgreich, derselbe CUDA-Test
  übersprungen; 27.833 Assertions erfolgreich.
- Der native 645-Frame-Resume lief ohne CUDA-Fallback oder
  Allokationswiederholung bis `resume_end: ok`.

Nächster Rekonstruktionsschritt:

- Chunk-Zeilen 84 und 86 bei identischen Caches fest vorgeben und die
  verbleibende Ausgabedifferenz auf Cache-Region-Decode oder CUDA-Kern
  zurückführen.
- CUDA-Events für Region-I/O/Host-Packing, H2D, Kernel und D2H getrennt
  erfassen.
- danach Host-Packing und GPU-Kernel mit zwei Chunk-Puffern überlappen.
- Warp-/Block-Selection nur als kooperative Auswahl implementieren; ein
  weiterer thread-serieller Auswahlalgorithmus ist durch den nativen Test
  widerlegt.

## Phase 14: Opti2-Vollrun

Status: `erledigt`

Direkt verglichen wurden
`m31-default-gpu-opti1_20260724_124627` und
`m31-default-gpu-opti2_20260724_171239`. Beide verarbeiten dieselben 645
Frames. Die gespeicherten YAML-Dateien unterscheiden sich nur bei
Defaultfeldern beziehungsweise ihrer Serialisierung; die wirksamen
Verarbeitungsparameter sind gleich.

Ergebnis:

- Gesamtlaufzeit: 3247 auf 2445 Sekunden, also 24,7 % schneller.
- `AQMH_MAPS`: 1110 auf 601 Sekunden, also 45,9 % schneller.
- `AQMH_RECONSTRUCTION`: 1221 auf 981 Sekunden, also 19,7 % schneller.
- Rekonstruktionskern: 996,89 auf 966,00 Sekunden, also 3,1 % schneller.
- Postprocessing/Validierung: 224,00 auf 14,29 Sekunden, also 93,6 %
  schneller.
- MAPS verwendete in Opti2 acht statt sieben Worker. Der Phasengewinn ist
  deshalb nicht vollständig als algorithmischer Speedup zu interpretieren.
- Opti2 verwendete wegen der zum Start geringeren freien GPU-Speichermenge 80
  statt 84 Zeilen pro Chunk und 29 statt 28 Chunks.
- Globale Ausgabe- und AQMH-Gate-Metriken sind unverändert. Beide Runs wählen
  korrekt `raw_aqmh`; es gab keinen CUDA-Fallback und keine
  Allokationswiederholung.

## Phase 15: CUDA-Pipelineprofil

Status: `erledigt`

Umgesetzt:

- Das Rekonstruktionsartefakt enthält unter
  `cuda_pipeline_timing_seconds` getrennte Zeiten für
  `host_region_load_and_pack`, H2D, Kernel, D2H und `result_commit`.
- CUDA-Events messen die drei GPU-Abschnitte auf demselben Stream.
- Host-Regionen/Packen und Ergebnisübernahme werden mit monotonic wall time
  gemessen.

Native Messung:

- Ein frischer M31-Testlauf mit 64 Frames, drei Chunks und
  `cuda_native_v0_2` benötigte 37,92 Sekunden im Rekonstruktionskern.
- Davon entfielen 21,64 Sekunden beziehungsweise 57,1 % auf
  Host-Regionen/Packen und 14,11 Sekunden beziehungsweise 37,2 % auf den
  Kernel.
- H2D benötigte 0,677 Sekunden, D2H 0,016 Sekunden und die
  Ergebnisübernahme 0,026 Sekunden. Transfers sind damit aktuell kein
  primärer Engpass.
- Postprocessing/Validierung benötigte 15,56 Sekunden. `raw_aqmh` blieb
  ausgewählt, es gab weder fehlende Maps noch einen CUDA-Fallback.

Verworfener Kandidat:

- Eine statische OpenMP-Zuteilung in 16-Frame-Blöcken erhöhte im kontrollierten
  64-Frame-Test Host-Regionen/Packen von 21,64 auf 22,81 Sekunden und den Kern
  von 37,92 auf 39,04 Sekunden. Ursache waren nur vier Arbeitspakete für acht
  Worker. Die Änderung wurde zurückgenommen.
- Eine feinere 8-Frame-Variante wurde nicht übernommen, weil der erforderliche
  native A/B-Test wegen des externen Ausführungslimits nicht mehr gestartet
  werden konnte.

## Phase 16: Reconstruction-Host-Unterprofil

Status: `erledigt`

Umgesetzt:

- `cuda_pipeline_timing_seconds.host_region_load_and_pack` bleibt die reale
  Wall-Time des vollständigen Host-Abschnitts.
- `cuda_pipeline_timing_seconds.host_chunk_setup` misst als Teilmenge davon
  Masken-Initialisierung und Canvas-Mask-Slice.
- `cuda_host_worker_timing_seconds` trennt Frame-Region-Read, Q-Map-Read,
  Valid-Mask-Read und pixel-major Packing.
- Die Unterzeiten sind ausdrücklich
  `sum_of_parallel_worker_wall_times`. Sie beschreiben die Verteilung der
  Workerarbeit, nicht die serielle Phasenlaufzeit.

Verifikation:

- CUDA-Übersetzung, Runner und vollständiges Testtarget bauen erfolgreich.
- Der native CUDA-Regressionstest prüft nichtnegative Unterzeiten und die
  Beziehung zwischen Host-Gesamt- und Setup-Zeit.
- Die native Ausführung dieses Tests bleibt im Sandbox-Prozess mangels
  sichtbarem CUDA-Gerät übersprungen.
- Ein nativer 64-Frame-Lauf mit drei Chunks benötigte vor der
  OpenMP-Korrektur 30,33 Sekunden für Host-Regionen/Packen. Davon waren
  15,60 Worker-Sekunden pixel-major Packing, 9,33 Q-Map-Read, 4,15
  Frame-Read und 1,21 Valid-Mask-Read.
- Die Worker-Summen lagen unerwartet in der Größenordnung der Host-Wall-Time.
  Das Unterprofil hat damit belegt, dass die OpenMP-Pragmas in der
  CUDA-Übersetzung nicht aktiv waren.

## Phase 17: AQMH_MAPS-Unterprofil

Status: `erledigt`

Umgesetzt:

- `worker_timing_seconds` im MAPS-Artefakt trennt Frame-Load,
  Normalisierung, Valid-Maske, Quality-Map-Berechnung, Cache-Write,
  Map-Zusammenfassung, globale Frame-Metriken und Regionsdiagnostik.
- `quality_map_stage_worker_timing_seconds` trennt innerhalb der
  Quality-Map Source-Mask, Pyramid-Vorbereitung, Sharpness, lokalen
  Hintergrund, SNR, Artifact, Summary, Psi-Upsample/Akkumulation und
  Finalisierung.
- Beide Gruppen verwenden summierte parallele Frame-Worker-Wall-Times; die
  Semantik wird im Artefakt explizit angegeben.

Verifikation:

- Der Quality-Map-Test prüft alle Stage-Zeiten auf nichtnegative Werte und
  stellt sicher, dass ihre Summe die gemessene Gesamtzeit nicht überschreitet.
- AQMH: 75 Testfälle, 74 erfolgreich, ein nativer CUDA-Test übersprungen;
  12.533 Assertions erfolgreich.
- Gesamtsuite: 241 Testfälle, 240 erfolgreich, derselbe CUDA-Test
  übersprungen; 27.844 Assertions erfolgreich.
- Im nativen 64-Frame-Profil benötigte die Quality-Map-Berechnung summiert
  390,20 Worker-Sekunden. Der größte Einzelabschnitt war
  Psi-Upsample/Akkumulation mit 137,92 Sekunden beziehungsweise 35,3 %,
  gefolgt von Artifact mit 96,17 und SNR mit 68,10 Sekunden.

Optimierung:

- Bilineares Psi-Upsampling und logarithmische Akkumulation laufen nun in
  einem fusionierten Durchlauf; die vollständige Upsample-Zwischenmatrix
  entfällt.
- Im nativen A/B-Test sank Psi-Upsample/Akkumulation von 137,92 auf
  111,00 Worker-Sekunden, also um 19,5 %. Die gesamte
  Quality-Map-Berechnung sank von 390,20 auf 364,07 Worker-Sekunden
  beziehungsweise um 6,7 %.
- Die MAPS-Phase sank im groben Sekundenartefakt von 63 auf 61 Sekunden.
- Alle 64 binären Q-Map-Cachedateien sind gegenüber der Referenz bitidentisch.

## Phase 18: OpenMP für CUDA-Hostvorbereitung

Status: `erledigt`

Ursache:

- `OpenMP_CUDA` wurde von CMake erkannt, aber nicht an
  `tile_compile_lib` gebunden. Der NVCC-Kompilierungsbefehl enthielt deshalb
  kein `-fopenmp`; die OpenMP-Schleifen in
  `aqmh_reconstruction_cuda.cu` liefen seriell.

Umsetzung:

- Bei aktivem CUDA und gefundenem `OpenMP_CUDA` wird nun zusätzlich
  `OpenMP::OpenMP_CUDA` gelinkt.
- Der erzeugte NVCC-Kompilierungsbefehl enthält nach der Neukonfiguration
  `-fopenmp`.

Native A/B-Messung:

- Bei automatischer Chunk-Planung sank der Rekonstruktionskern von 47,31 auf
  31,50 Sekunden, also um 33,4 %. Die Host-Wall-Time sank von 30,33 auf
  14,78 Sekunden beziehungsweise um 51,3 %.
- Bei kontrolliert identischen `chunk_rows=790` sank der Kern von 47,31 auf
  31,70 Sekunden, also um 33,0 %. Host-Regionen/Packen benötigten 14,46,
  der CUDA-Kernel 14,69 Sekunden.
- Das kontrollierte Raw-AQMH-FITS ist bitidentisch: 0 von 8.507.400 Pixeln
  unterscheiden sich. Es gab keinen CUDA-Fallback, keine fehlenden
  Map-Samples und dieselbe Raw-AQMH-Auswahl.
- Der vorherige Vergleich mit automatisch gewählten 860 statt 790
  Chunk-Zeilen unterschied 0,207 % der Pixel. Der feste
  Chunk-Geometrievergleich weist diese Abweichung der Geometrie und nicht
  einer OpenMP-Race-Condition zu.

Verifikation:

- Runner und vollständiges Testtarget bauen einschließlich der
  CUDA-Übersetzung erfolgreich.
- Gesamtsuite: 241 Testfälle, 240 erfolgreich, ein nativer CUDA-Unit-Test im
  Sandbox-Prozess übersprungen; 27.844 Assertions erfolgreich.

## Phase 19: Opti3-Vollrun und MAPS-Speicherverifikation

Status: `erledigt`

Vergleich:

- `m31-default-gpu-opti3_20260724_201231` verarbeitet dieselben 645 Frames und
  dieselben wirksamen Parameter wie Opti2. Unterschiede in der YAML-Datei
  betreffen nur Defaultfelder und ihre Reihenfolge.
- Die technische Gesamtlaufzeit stieg dadurch von 2445,1 auf 2485,9 Sekunden
  (+1,7 %). Der Reconstruction-Gewinn wurde durch die langsamere MAPS-Phase
  und längere Vorphasen überkompensiert; die nächste Optimierung muss daher
  MAPS beziehungsweise die Speicher-/Worker-Situation adressieren.
- AQMH_MAPS benötigte in Opti3 686,75 statt 600,43 Sekunden. Ursache ist die
  Speicherbegrenzung: verfügbar waren 8,75 GiB, deshalb wurden sechs statt
  acht Worker zugelassen (`capped=true`).
- AQMH_RECONSTRUCTION sank trotz kleinerer automatisch gewählter Chunks von
  980,57 auf 842,30 Sekunden (14,1 %). Der Reconstruction-Core sank von
  966,00 auf 826,58 Sekunden (14,4 %).
- Alle Q-Map-Caches sind bitidentisch. Das Raw-FITS unterscheidet sich nur in
  280 von 9.059.200 Pixeln (0,0031 %; mittlere absolute Differenz
  0,0000308); die Abweichung stammt aus der unterschiedlichen Chunk-Geometrie
  `80` beziehungsweise `67`.
- Die Qualitätsmetriken und Gate-Entscheidungen bleiben praktisch gleich:
  beide Läufe wählen `raw_aqmh`, bestehen die gleichen globalen Gates und
  melden keine fehlenden Maps.

Speicher-A/B-Test:

- Ein isolierter 64-Frame-Lauf mit 28,05 GiB verfügbarer Speicherplanung ließ
  acht Worker zu und meldete `capped=false`.
- Die maximale Prozess-RSS betrug 8,75 GiB. Damit ist die derzeitige
  Schätzung von rund 891 MiB pro MAPS-Worker konservativ, aber plausibel.
- Die Worker-Cap-Berechnung wird deshalb nicht pauschal gelockert. Eine
  Lockerung bei nur rund 9 GiB verfügbarem RAM könnte den Prozess durch
  parallele OpenCV-Zwischenmatrizen in Speicherdruck bringen.
- Der isolierte Test hatte keinen sichtbaren CUDA-Device-Zugriff und fiel nur
  in Reconstruction auf CPU zurück; die MAPS-Speichermessung selbst war
  vollständig und erfolgreich.
- Die OpenMP-Steuerung wurde im MAPS-Worker-Pool zentralisiert: Die Threadzahl
  wird einmal vor dem Start der äußeren Worker gesetzt und nach dem Join wieder
  auf den vorherigen Prozesswert zurückgestellt. Ein neuer 64-Frame-Lauf lief
  mit acht Workern vollständig durch; MAPS und Reconstruction endeten mit
  `status=ok`, Reconstruction nutzte mangels sichtbarem CUDA-Gerät den CPU-
  Fallback.

## Geplanter Reconstruction-Overlap-Kandidat

Der Kandidat wird erst nach der nächsten nativen Unterprofil-Messung
implementiert:

- zwei Host-Eingabepuffer mit jeweils ungefähr halber bisheriger Chunk-Höhe;
- nur ein Device-Puffer und ein Output-Staging-Puffer;
- Vorbereitung von Chunk `n+1` parallel zum Kernel von Chunk `n`;
- dadurch bleibt der Host-Eingabepufferbedarf ungefähr auf dem heutigen
  Niveau, statt sich durch einen vollständigen Doppelpuffer zu verdoppeln;
- H2D, Kernel und D2H bleiben auf demselben Stream geordnet;
- der vorbereitende Worker liefert Statistik und Fehlerzustand zurück und
  verändert das Reconstruction-Result nicht nebenläufig.

Abnahme:

- gleicher Cache, feste Chunk-Geometrie und identische Qualitäts-Gates für
  seriellen und überlappten Pfad;
- kein höherer beobachteter Host-Spitzenbedarf als beim bisherigen
  Auto-Chunk-Pfad;
- messbarer Core-Speedup; andernfalls bleibt der serielle Pfad Standard;
- bitweiser beziehungsweise toleranzbasierter Vergleich von Raw-AQMH,
  Weight-Sum, Uniform-Control und Valid-Maske.

## Nächste Priorität

1. die nächste Overlap-Stufe für Q-Map-/Maskenlesen prüfen; die erste sichere
   Stufe lädt den nächsten Frame-Region-Chunk parallel zur laufenden
   Geräteverarbeitung vor. Der native 64-Frame-CUDA-Test mit fester
   Chunk-Höhe 790 sank von 64,75 auf 53,91 Sekunden (-16,7 %). Das
   rekonstruierte FITS ist bitidentisch zur Referenz. Mit automatischer
   Chunk-Höhe 781 wurden 47,35 Sekunden erreicht (-26,9 %), dieser Wert
   enthält jedoch zusätzlich den kleinen Geometrieunterschied.
2. den Artifact-Abschnitt in AQMH_MAPS unterprofilieren und optimieren
3. die zentrale OpenMP-Threadsteuerung im nativen CUDA-Lauf messen und gegen
   Oversubscription absichern
4. kooperative Warp-/Block-Selection mit CPU-/CUDA-Vergleich
5. Chunk-Geometrieabhängigkeit separat untersuchen; sie ist klein, aber bei
   bitidentischen Anforderungen relevant
6. Registrierungs-Guard-Differenz der 44 Frames erklären

Die bereits umgesetzten Schritte haben den größten belegten
Validierungsanteil und die MAPS-Phase deutlich reduziert. Die nächste
Rekonstruktionsstufe konzentriert sich auf die nun gleich großen Host- und
Kernelabschnitte, damit auf messbare I/O-/Kernel-Überlappung und anschließend
kooperative GPU-Auswahl; die
wissenschaftlichen Qualitäts-Gates bleiben unverändert.
