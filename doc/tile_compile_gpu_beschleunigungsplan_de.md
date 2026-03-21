# Tile-Compile GPU-Beschleunigungsplan

Stand: 2026-03-20

Dieses Dokument analysiert, welche Teile der aktuellen C++-Pipeline von
`tile_compile_cpp/` fuer GPU-Offload geeignet sind, wie ein sinnvoller Ausbau
aussehen kann und welchen Nutzen man realistisch erwarten darf.

Wichtige Einschraenkung: Die Aussagen basieren auf statischer Codeanalyse der
aktuellen Implementierung. In diesem Schritt wurde kein Laufzeit-Profiling auf
Referenzdatensaetzen durchgefuehrt.

## Kurzfazit

Die besten GPU-Kandidaten sind nicht die gesamte Pipeline, sondern die Phasen
mit dichtem Pixel-Durchsatz und wenig Verzweigungslogik:

1. `PREWARP`
2. `TILE_RECONSTRUCTION`
3. `SYNTHETIC_FRAMES` und `STACKING`
4. Teile von `LOCAL_METRICS`
5. Teile von `BGE`

Am wenigsten geeignet sind FITS-I/O, Event-/JSON-Logik, ASTAP/WCS, weite Teile
der heuristischen Registrierungsentscheidung sowie der katalog- und
kontrollflusslastige PCC-Pfad.

Wenn GPU-Offload sauber umgesetzt wird und Frames/Tiles nicht fuer jeden kleinen
Schritt neu zwischen Host und Device kopiert werden, ist fuer grosse Datensaetze
eine grobe Gesamtbeschleunigung in dieser Groessenordnung plausibel:

- konservativ: `1.3x` bis `2.0x`
- realistisch bei vielen Frames und grosser Canvas: `2x` bis `4x`
- unguenstig bei I/O-Limit oder kleinen Runs: oft unter `1.5x`

Der groesste Hebel liegt in `PREWARP` plus `TILE_RECONSTRUCTION`. GPU allein
ersetzt aber keine I/O-Optimierung. Die bereits dokumentierten CPU/I/O-Themen in
`doc/tile_compile_cpp_performance_optimierungsplan_de.md` bleiben relevant.

## Aktueller Befund im Code

### 1. Es gibt derzeit keinen GPU-Backend-Pfad

- Im CMake von `tile_compile_cpp/` gibt es aktuell keine explizite CUDA-, OpenCL-
  oder OpenMP-Integration, sondern nur CPU-Bibliotheken wie Eigen, OpenCV,
  cfitsio und yaml-cpp:
  `tile_compile_cpp/CMakeLists.txt:341-382`
- Im produktiven Code gibt es keine Nutzung von `cv::cuda`, `GpuMat`, `UMat`
  oder aehnlichen GPU-APIs.

### 2. Es gibt bereits CPU-Parallelisierung, aber nicht als GPU-Ersatz

- Registrierung, Metriken, lokale Metriken, Prewarp und Teile der Rekonstruktion
  nutzen `std::thread`, z. B.:
  - `tile_compile_cpp/apps/runner_phase_registration.cpp:654-667`
  - `tile_compile_cpp/apps/runner_phase_metrics.cpp:288-301`
  - `tile_compile_cpp/apps/runner_phase_local_metrics.cpp:170-183`
  - `tile_compile_cpp/apps/runner_pipeline.cpp:1527-1558`
- In `background_extraction.cpp` existieren OpenMP-Pragmas:
  - `tile_compile_cpp/src/image/background_extraction.cpp:2313`
  - `tile_compile_cpp/src/image/background_extraction.cpp:2483`
- Da `tile_compile_cpp/CMakeLists.txt` OpenMP nicht explizit aktiviert, ist der
  Effekt dieser Pragmas aktuell toolchain-abhaengig. Vor jeder GPU-Bewertung
  sollte die CPU-Baseline sauber verifiziert werden.

### 3. Die Datenhaltung ist fuer naive GPU-Offloads unguenstig

- Prewarped Frames werden disk-basiert gespeichert:
  `tile_compile_cpp/apps/runner_phase_registration.cpp:1451-1525`
- Tiles werden spaeter aus einem mmap-/disk-gestuetzten Cache ausgeschnitten:
  `tile_compile_cpp/apps/runner_shared.cpp:692-748`
- Das ist fuer CPU-Skalierung sinnvoll, bedeutet fuer GPU aber:
  Ein Offload pro Tile oder pro Mini-Operation wuerde schnell an PCIe- und
  Host-Device-Transferkosten scheitern.

Konsequenz: GPU lohnt sich nur, wenn ganze Phasen oder groessere Batches auf dem
Device verbleiben.

## Bewertungslogik

Ein Prozess ist ein guter GPU-Kandidat, wenn moeglichst viele dieser Punkte
zutreffen:

- viele unabhaengige Pixel-/Tile-Operationen
- hohe arithmetische Dichte pro geladenem Byte
- wenig globale Synchronisation
- wenig irregulaere Verzweigung
- geringe Abhaengigkeit von Dateisystem und externen Tools

Ein Prozess ist ein schlechter GPU-Kandidat, wenn er hauptsaechlich aus:

- Dateilesen/-schreiben
- kleinen robusten Median-/MAD-Schritten mit viel Host-Logik
- externer Prozessorchestrierung
- branchiger Heuristik oder Graph-/Matching-Logik

besteht.

## Geeignete GPU-Kandidaten

## A. Hohe Prioritaet

### A1. `PREWARP`

Codebasis:

- `tile_compile_cpp/apps/runner_phase_registration.cpp:1388-1615`
- `tile_compile_cpp/src/image/normalization.cpp:72-100`
- `tile_compile_cpp/src/image/cfa_processing.cpp:205-299`

Warum geeignet:

- Pro Frame wird ein Vollbild-Warp ueber die gesamte Canvas gerechnet.
- In OSC werden sogar vier Subplane-Warps pro Frame ausgefuehrt.
- `cv::warpAffine` ist ein klassischer GPU-Workload.
- Danach folgt noch ein weiterer voller Pixelpass fuer den Overlap-Zaehler.

Was man auf die GPU legen kann:

- MONO/RGB: `warpAffine`
- OSC: Subplane-Warps fuer CFA
- Aufbau von `frame_has_data`
- Aufbau bzw. Reduktion der `overlap_coverage_count`

Erwarteter Nutzen:

- Phasenbezogen grob `3x` bis `8x`
- Besonders stark bei grossen Canvas-Flaechen, Alt/Az-Runs und vielen Frames

Besonderheit:

- Diese Phase ist ein sehr guter erster GPU-Einstieg, weil sie mathematisch
  relativ klar ist und wenig heikle Methodenlogik enthaelt.

### A2. `TILE_RECONSTRUCTION`

Codebasis:

- `tile_compile_cpp/apps/runner_pipeline.cpp:929-1952`
- `tile_compile_cpp/src/reconstruction/reconstruction.cpp:491-655`
- `tile_compile_cpp/apps/runner_pipeline.cpp:1765-1859`

Warum geeignet:

- Pro Tile werden viele Frames gesammelt, maskiert und per gewichteter
  Sigma-Clip-Reduktion kombiniert.
- Anschliessend folgt Overlap-Add mit Hanning-Fenstern ueber grosse Mengen an
  Pixeln.
- In OSC wird derselbe Kern dreifach fuer `R/G/B` durchlaufen.

Was man auf die GPU legen kann:

- COMMON_OVERLAP-Maskierung
- gewichtete per-Pixel Sigma-Clip-Reduktion
- tileweises Normalisieren
- Hanning-gewichtete OLA-Akkumulation
- finale Division durch `weight_sum`

Erwarteter Nutzen:

- Kernel-bezogen grob `4x` bis `12x`
- Phasenbezogen realistisch eher `2x` bis `5x`, je nach Tile-Zahl,
  Frame-Zahl, OSC/Mono und Host-Transferstrategie

Wichtiger Hinweis:

- Ein rein tile-lokaler Offload ist nicht genug. Wenn jedes Tile erst auf der
  CPU aus `DiskCacheFrameStore` kopiert und dann einzeln auf die GPU gesendet
  wird, geht ein grosser Teil des Vorteils verloren.

### A3. `SYNTHETIC_FRAMES` und `STACKING`

Codebasis:

- `tile_compile_cpp/apps/runner_pipeline.cpp:2618-2846`
- `tile_compile_cpp/apps/runner_pipeline.cpp:3001-3137`
- `tile_compile_cpp/src/reconstruction/reconstruction.cpp:393-489`
- `tile_compile_cpp/src/reconstruction/reconstruction.cpp:491-655`

Warum geeignet:

- Die Rekonstruktion von Subsets wiederholt grosse Teile der tileweisen
  Reduktion und OLA.
- Das finale `sigma_clip_stack()` ist ein reiner dichte-Pixel-Reduktionskern
  ueber mehrere Vollbilder.

Was man auf die GPU legen kann:

- `sigma_clip_stack()`
- `sigma_clip_weighted_tile_with_fallback()`
- clusterweise Rekonstruktion
- qualitaetsgewichtetes Summieren ueber synthetische Frames

Erwarteter Nutzen:

- Kernel-bezogen `3x` bis `10x`
- Gesamtnutzen haengt stark davon ab, ob `SYNTHETIC_FRAMES` im Run ueberhaupt
  aktiv ist

## B. Mittlere Prioritaet

### B1. `LOCAL_METRICS`

Codebasis:

- `tile_compile_cpp/apps/runner_phase_local_metrics.cpp:96-137`
- `tile_compile_cpp/src/metrics/tile_metrics.cpp:166-281`

Warum teilweise geeignet:

- blur, Residual, Sobel, Gradientenenergie und Teile der Schwellenlogik sind
  dichte Bildoperationen.
- Die Sternsuche via `goodFeaturesToTrack()` und die Patch-Messung sind dagegen
  deutlich irregulaerer.

Was man auf die GPU legen kann:

- Box-Blur / Background-Schatzung
- Residualbildung
- Sobel / Gradient / quadrierte Magnitude
- einfache Schwellwert- und Reduktionsoperationen

Was eher CPU bleiben sollte:

- `goodFeaturesToTrack()`
- patchweise Sternmetrik mit kleiner, branchiger Logik
- tileweite robuste Median-/MAD-Steuerlogik, falls nur wenige Werte anfallen

Erwarteter Nutzen:

- fuer den dichten Teil `2x` bis `6x`
- fuer die Phase insgesamt eher `1.5x` bis `3x`, solange die Sternlogik CPU
  bleibt

### B2. `NORMALIZATION` und `GLOBAL_METRICS`

Codebasis:

- `tile_compile_cpp/apps/runner_phase_metrics.cpp:115-256`
- `tile_compile_cpp/apps/runner_phase_metrics.cpp:369-407`
- `tile_compile_cpp/src/metrics/metrics.cpp:12-164`

Warum nur bedingt geeignet:

- Hintergrundmasken, Sobel und Gradientenenergie sind GPU-freundlich.
- Gleichzeitig ist die Phase von FITS-I/O, Vektorfuellungen und robuster
  Statistik gepraegt.

Erwarteter Nutzen:

- eher moderat, typischerweise `1.2x` bis `2.0x` fuer die gesamte Phase
- hoher Nutzen nur dann, wenn die Daten bereits im GPU-Speicher liegen

### B3. Teile von `BGE`

Codebasis:

- `tile_compile_cpp/src/image/background_extraction.cpp:2126-2320`
- `tile_compile_cpp/src/image/background_extraction.cpp:2456-2503`

Warum nur selektiv geeignet:

- Das Rendern der RBF- bzw. Polynomialflaeche ist massiv datenparallel.
- Das Aufbauen der robusten Modelle ist dagegen ein Mix aus Eigen-Linearalgebra,
  IRLS und relativ kleinen Matrizen.

Was man auf die GPU legen kann:

- Rendern der Flaechen
- punktweise Modell-Evaluation
- eventuell Bildmasken/Morphologie davor, falls dieser Pfad spaeter erweitert
  wird

Was eher CPU bleiben sollte:

- Steuerlogik der Modellwahl
- kleine bis mittlere IRLS-Solver, solange keine grossen Gridmatrizen vorliegen

Erwarteter Nutzen:

- fuer die Render-Kerne hoch (`5x` bis `20x`)
- fuer die Gesamtphase meist begrenzt, solange andere BGE-Schritte dominieren

## C. Niedrige Prioritaet oder nur teilweise sinnvoll

### C1. Registrierung: nur die dichten Primitive, nicht die ganze Kaskade

Codebasis:

- `tile_compile_cpp/src/registration/registration.cpp:7-64`
- `tile_compile_cpp/src/registration/global_registration.cpp:71-216`
- `tile_compile_cpp/apps/runner_phase_registration.cpp:674-756`

Geeignet:

- `GaussianBlur`
- `phaseCorrelate`
- `findTransformECC`
- `warpAffine`

Weniger geeignet:

- AKAZE-basierte Feature-Pfade
- Trail-/Triangle-Matching
- temporale Rescue-Logik
- robuste Outlier-Heuristiken

Bewertung:

- Die Registrierung als Gesamtphase ist kein sauberer GPU-Kandidat.
- Einzelne Primitive koennen beschleunigt werden, aber der Nutzen fuer die
  ganze Kaskade ist deutlich kleiner als bei `PREWARP` oder Rekonstruktion.

## Schlechte GPU-Kandidaten

Die folgenden Bereiche wuerde ich nicht priorisieren:

- FITS-I/O und Dateisystempfade:
  - `tile_compile_cpp/apps/runner_shared.cpp:640-760`
  - `tile_compile_cpp/apps/runner_phase_metrics.cpp:82-99`
- JSON-/Event-/Artefakt-Schreiben
- ASTAP/WCS-Aufrufe und externe Tool-Orchestrierung
- PCC als Gesamtphase:
  - viel Katalog-/Matching-/Robustheitslogik
  - vergleichsweise wenig dichte Pixelarbeit
- kleine robuste Hilfsstatistik auf Host-Vektoren, wenn sie nicht im Hotpath
  gross skaliert

## Empfohlene technische Richtung

## 1. Kein "GPU pro Funktion", sondern phasenweises Offload

Die wichtigste Architekturentscheidung ist:

- nicht pro Tile/Funktion offloaden
- sondern pro Phase oder grossen Batches

Sonst dominieren:

- Host-Device-Kopien
- Device-Synchronisation
- CPU-Seitenlogik fuer Cache und Tile-Zuschnitt

## 2. Backend-Abstraktion einfuehren

Empfohlene minimale API:

- `AccelerationBackend = cpu | opencv_cuda | cuda`
- `DeviceFrame`
- `DeviceFrameBatch`
- `DeviceTileBatch`
- `GpuOps::warp_affine_batch(...)`
- `GpuOps::sigma_clip_reduce(...)`
- `GpuOps::overlap_add(...)`
- `GpuOps::sobel_gradients(...)`

So bleibt der CPU-Pfad unangetastet und der GPU-Pfad optional.

## 3. Nicht auf `UMat` als Hauptstrategie setzen

`UMat`/transparentes OpenCL klingt verlockend, ist fuer diese Codebasis aber
keine gute Hauptstrategie:

- begrenzte Kontrolle ueber Speicherlebensdauer
- unklare Portabilitaet und Performance
- viele benoetigte Spezialkerne sind ohnehin custom

Pragmatischer ist:

- OpenCV CUDA dort, wo es stabile Primitive gibt (`warpAffine`, Filter, DFT)
- eigene CUDA-Kerne fuer sigma-clipping, OLA, Maskierung und Reduktionen

## 4. GPU-Speicher budgetieren wie heute RAM

Die CPU-Pipeline steuert bereits RAM und Disk-Cache recht bewusst, z. B. im
OSC-RGB-Cache:

- `tile_compile_cpp/apps/runner_pipeline.cpp:1115-1229`

Fuer GPU braucht es analog:

- VRAM-Budget
- Batchgroesse in Frames oder Tiles
- Streaming bei Oversubscription
- CPU-Fallback bei zu kleinem Device

## Realistisch erwarteter Nutzen

Die folgenden Schaetzungen sind bewusst vorsichtig und setzen eine sinnvolle
Batch-Strategie voraus.

| Bereich | Erwartete Beschleunigung | Bemerkung |
|---|---:|---|
| `PREWARP` | `3x-8x` | sehr guter erster GPU-Hebel |
| `TILE_RECONSTRUCTION` | `2x-5x` phaseweit | groesster Gesamthebel neben Prewarp |
| `SYNTHETIC_FRAMES` | `2x-4x` | nur relevant im Full-Mode |
| `STACKING` | `2x-6x` | besonders gut bei vielen Synth-Frames |
| `LOCAL_METRICS` | `1.5x-3x` | nur wenn dichter Teil ausgelagert wird |
| `BGE` | `1.1x-2x` gesamt | Renderkerne stark, Gesamtphase oft weniger |

Gesamtpipeline:

- kleine Runs / SSD / geringe Tile-Zahl: oft nur kleiner Gewinn
- grosse Runs / viele Frames / OSC / grosse Canvas: deutlicher Gewinn
- I/O-limitierte Systeme: GPU verbessert die Rechenphasen, aber nicht das
  Dateisystem

## Priorisierter Umsetzungsplan

## Phase 0. Baseline und Messbarkeit

Ziel:

- erst messen, dann offloaden

Massnahmen:

- Phasenlaufzeiten und Unterkern-Laufzeiten loggen:
  - `REGISTRATION`
  - `PREWARP`
  - `LOCAL_METRICS`
  - `TILE_RECONSTRUCTION`
  - `SYNTHETIC_FRAMES`
  - `STACKING`
  - `BGE`
- CPU-Build mit und ohne aktivem OpenMP sauber vergleichen
- Referenzdatensaetze definieren:
  - kleines MONO
  - grosses MONO
  - grosses OSC
  - Alt/Az mit grosser Canvas

Akzeptanz:

- reproduzierbare Zeitbasis pro Phase
- Hotspot-Ranking aus Messwerten statt Vermutung

## Phase 1. GPU-Backend-Grundlage

Ziel:

- optionale GPU-Infrastruktur ohne Funktionsaenderung

Massnahmen:

- Build-Option `TILE_COMPILE_ENABLE_CUDA`
- Runtime-Option `runtime_limits.acceleration_backend`
- Basistypen fuer Device-Speicher und Batch-Transfers
- CPU/GPU-Konsistenztests auf kleinen Testbildern

Akzeptanz:

- CPU bleibt Default
- GPU kann fuer einzelne Testpfade zugeschaltet werden

## Phase 2. `PREWARP` auf GPU

Ziel:

- ersten echten Produktions-Hotspot portieren

Massnahmen:

- GPU-Batch-Warp fuer MONO
- GPU-Batch-Warp fuer OSC-Subplanes
- Overlap-Coverage direkt im GPU-Pfad mitfuehren
- nur einmal pro Frame Host->Device, danach Device->Host fuer das finale
  prewarped Ergebnis oder fuer groessere Batches

Akzeptanz:

- numerisch stabile Warp-Ergebnisse gegen CPU-Referenz
- messbarer Durchsatzgewinn auf grossen Runs

## Phase 3. `TILE_RECONSTRUCTION` plus `STACKING`

Ziel:

- groessten Gesamt-Hotspot beschleunigen

Massnahmen:

- gemeinsame Device-Repraesentation fuer Tile-Stapel
- GPU-Kerne fuer:
  - COMMON_OVERLAP-Maskierung
  - gewichtete Sigma-Clip-Reduktion
  - Hanning-gewichtete OLA
  - finale `weight_sum`-Division
- optional danach `sigma_clip_stack()` fuer Phase `STACKING`

Akzeptanz:

- CPU/GPU-Ausgaben innerhalb definierter numerischer Toleranzen
- keine sichtbaren neuen Tile-Artefakte
- deutlicher End-to-End-Gewinn bei grossen Datensaetzen

## Phase 4. `LOCAL_METRICS`

Ziel:

- dichte Filterarbeit auslagern, branchige Sternlogik vorerst CPU lassen

Massnahmen:

- GPU fuer Blur/Residual/Sobel/Gradientenenergie
- CPU fuer `goodFeaturesToTrack()` und Sternpatch-Metriken
- spaeter pruefen, ob Corner-Detektion ersetzt oder vereinfacht werden kann

Akzeptanz:

- keine Aenderung der Tile-Klassifikation ausser in numerisch erwartbaren
  Kleinstabweichungen

## Phase 5. Optionale Folgearbeiten

Reihenfolge:

1. BGE-Surface-Rendering
2. selektive Registrierungsprimitive
3. globale Metrikpfade
4. evtl. PCC-Hilfsschritte, aber nicht die ganze Phase

## Risiken und Gegenmassnahmen

### 1. Transferkosten fressen den Gewinn auf

Gegenmassnahme:

- phasenweise Offloads
- Batching
- Daten so lange wie moeglich auf dem Device halten

### 2. Numerische Abweichungen beeinflussen Qualitaetsheuristiken

Gegenmassnahme:

- GPU nur fuer klar definierte Kerne
- Referenztests mit Toleranzen
- Sichtpruefung auf Tile-Grenzen, SNR, FWHM und PCC-Folgewirkung

### 3. VRAM reicht nicht fuer grosse OSC-Runs

Gegenmassnahme:

- Streaming-Batches
- adaptive Batchgroesse
- CPU-Fallback

### 4. Portabilitaet leidet

Gegenmassnahme:

- CPU-Pfad als produktiver Fallback
- GPU als optionale Beschleunigung, nicht als harte Voraussetzung

## Konkrete Empfehlung

Wenn nur ein begrenztes Budget verfuegbar ist, wuerde ich in genau dieser
Reihenfolge vorgehen:

1. Messbare Baseline herstellen
2. `PREWARP` auf GPU portieren
3. `TILE_RECONSTRUCTION` und danach `STACKING` portieren
4. erst dann `LOCAL_METRICS`
5. `BGE` und Registrierungsprimitive nur als spaetere Optimierung

Der Grund ist einfach:

- `PREWARP` und Rekonstruktion haben die beste Mischung aus hoher Last,
  sauberem Datenparallelismus und ueberschaubarem methodischem Risiko.
- Diese beiden Bereiche koennen realistisch den groessten Teil des gesamten
  GPU-Nutzens liefern.
