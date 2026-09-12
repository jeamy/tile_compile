# Forward Drizzle v2: Zielarchitektur als AQMH-Ersatz

**Stand:** 2026-09-12  
**Status:** Architekturvorschlag  
**Scope:** Vollständiger Ersatz der aktuellen Record-/Sort-/Clip-Pipeline; keine Zwischenoptimierung des bestehenden Pfads

## 1. Ziel

Forward Drizzle v2 soll AQMH als produktive Rekonstruktionsmethode vollständig ersetzen. Dafür muss die Rekonstruktion gleichzeitig:

- CFA-Samples ohne vorheriges Debayering flusserhaltend auf das Zielraster abbilden;
- helle Sterne, Nebelgradienten, negative normalisierte Hintergründe und sparse R/B-CFA-Abdeckung korrekt behandeln;
- Cosmic Rays, Hotpixel, Satellitenspuren und einzelne fehlerhafte Frames robust unterdrücken;
- innerhalb geometrisch vorhandener Abdeckung keine schwarzen Löcher oder NaN-Inseln erzeugen;
- mit der Framezahl asymptotisch linear skalieren;
- den verfügbaren RAM und VRAM strikt begrenzen;
- GPU und CPU nach demselben mathematischen Vertrag ausführen;
- Coverage, Rekonstruktion, Confidence und Multiband ohne redundante Geometriesweeps erzeugen.

Die bestehende Architektur wird nicht durch zusätzliche lokale Caches, größere Tiles oder weitere CPU-Threads konserviert. Contribution-Records, hostseitige Frame-Sortierungen pro Pixel und die framezahlabhängige Kandidatenmatrix entfallen vollständig.

## 2. Empirischer Ausgangspunkt: `m42-test1`

Der 60-Frame-Run `/media/tc_500/m42-test1` auf 3840×2160-OSC-Daten zeigte:

| Messwert | Wert |
|---|---:|
| Gesamtdauer erfolgreicher Phasen | 1.928,6 s |
| `FORWARD_DRIZZLE` | 1.361,2 s (70,6 %) |
| `SAMPLING_GEOMETRY` | 237,5 s (12,3 %) |
| `SOURCE_QUALITY_MAPS` | 95,2 s (4,9 %) |
| Forward-Drizzle-Stripe-Zeit | 1.224,5 s (90,0 % der FD-Zeit) |
| Source-Samples | 497.664.000 |
| Durchsatz | 365.614 Source-Samples/s |
| interne Pixel/Kanal-Auswertungen | 103.381.848 |
| weggeclippte Framebeiträge | 350.191.643 |
| vollständig verworfene interne Pixel/Kanäle | 2.276.618 |
| CUDA-Y-Bands | 6 |
| maximale X-Tiles je Band | 40 |
| maximale CUDA-Producer-Aufrufe | 14.400 |

### 2.1 Nachgewiesener Bildfehler

Die geometrische Coverage meldete innerhalb der Analysefläche für R, G und B jeweils 100 % Support und keine interne Lücke. Nach dem statistischen Clipping fehlten in derselben Fläche jedoch:

| Kanal | fehlende Pixel | Anteil |
|---|---:|---:|
| R | 406.143 | 5,10 % |
| G | 5.572 | 0,07 % |
| B | 400.013 | 5,02 % |
| mindestens ein Kanal | 793.828 | 9,96 % |

Von 487.513 Lochkomponenten waren 66,8 % Einzelpixel und 96,6 % höchstens vier Pixel groß. Im hellsten Prozent der Bildfläche stieg die Lochrate auf 13,2 %. Das entspricht dem sichtbaren Salz-und-Pfeffer-Muster.

Die Downstream-Maske war exakt gleich `finite(R) && finite(G) && finite(B)`. Ein fehlender Kanal wurde anschließend für alle drei Kanäle auf Schwarz gesetzt. Die Eingangsframes und Qualitätskarten waren endlich; fehlende Darks waren nicht die Ursache der Supportlöcher.

### 2.2 Nachgewiesener Strukturfehler

Der aktuelle CUDA-Pfad führt nur die Polygonrasterisierung auf der GPU aus. Alle folgenden Operationen laufen auf dem Host:

1. Contribution-Records vom Device herunterladen;
2. Records pro Frame/Tile sortieren;
3. Kandidatenmatrix `Kanal × Pixel × Frame` befüllen;
4. pro Pixel/Kanal Median und MAD wiederholt sortieren;
5. Alpha-Quantile mit weiteren Sortierungen berechnen;
6. Uniform, Raw, Fine und Medium seriell reduzieren.

Jeder Frame/Tile-Aufruf allokiert und löscht Devicebuffer, synchronisiert das gesamte Device und lädt Records zurück. Die Reduktion verwendete während des Runs effektiv ungefähr einen CPU-Kern.

## 3. Nichtziele

Folgende Maßnahmen sind ausdrücklich **nicht** die Zielarchitektur:

- nur `min_fraction` oder `min_n_eff` absenken;
- Supportlöcher nachträglich interpolieren;
- fehlende Kanäle erst im Downstream inpainten;
- den seriellen Record-Reducer lediglich mit OpenMP versehen;
- bestehende per-Pixel-Sortierungen nur durch schnelleres Scratch-Reuse beschleunigen;
- größere Kandidatentiles durch mehr RAM oder VRAM erzwingen;
- Contribution-Records weiterhin GPU→CPU übertragen;
- einen separaten vollständigen `SAMPLING_GEOMETRY`-Sweep beibehalten;
- Multiband unverändert mit `Q_p90-Q_p50` als zwingender Confidence-Bedingung betreiben.

Diese Änderungen könnten Symptome reduzieren, würden aber die falsche Skalierung und die fachliche Vermischung von Support und Statistik erhalten.

## 4. Verbindliche Invarianten

### 4.1 Support

Geometrischer Support und statistische Confidence sind getrennte Größen:

```text
geometric_support(pixel, channel)
    := mindestens ein geometrisch gültiger CFA-Beitrag existiert

statistical_confidence(pixel, channel)
    := Zuverlässigkeit des robusten Schätzers

profile_confidence(pixel)
    := zulässige Detailstärke für Fine/Medium/Multiband
```

Es muss immer gelten:

```text
geometric_support(pixel, channel)
    => isfinite(uniform(pixel, channel))
```

Statistisches Clipping oder Confidence dürfen `geometric_support` niemals löschen.

### 4.2 Robuster Fallback

Wenn der robuste Schätzer keine belastbare Teilmenge bestimmen kann, wird nicht NaN ausgegeben. Stattdessen wird aus allen geometrisch gültigen Gruppen ein begrenzter robuster Fallback gebildet. Der Fehler reduziert Confidence, nicht Support.

### 4.3 Flusserhaltung

Für konstante Quellen und affine Transformationen muss die Summe aus Droplet-Overlap und Zielwerten dem definierten quadratischen Kernel entsprechen. `internal_scale=2, output_scale=1` bleibt ein deterministisches 2×2-Flächenintegral.

### 4.4 Skalierung

Für `N` Frames, `P` Zielpixel und eine feste robuste Gruppenzahl `K` gilt:

```text
Laufzeit: O(N × P)
Speicher: O(K × Bandpixel), unabhängig von N
```

Es darf keinen `Pixel × Frame`-Kandidatenspeicher und keine `N log N`-Sortierung pro Pixel geben.

## 5. Neuer Gesamtdatenfluss

```text
SCAN / NORMALIZATION / REGISTRATION
                  │
                  ▼
SOURCE QUALITY PACK
  pro Frame genau einmal berechnen und persistieren
                  │
                  ▼
FUSED COVERAGE + FORWARD DRIZZLE
  pro Y-Band:
    Frames in stabiler Reihenfolge
    Source-/Quality-Rect asynchron laden
    deterministischer Target-Gather
    robuste Gruppenakkumulatoren aktualisieren
    Coverage + Profile + Confidence reduzieren
    internen 2×2-Fold auf GPU ausführen
                  │
                  ▼
STREAMING GPU MULTIBAND
  Band N+1 rekonstruieren
  Band N fusionieren
  Band N-1 schreiben
                  │
                  ▼
VALIDATION
  Support, Numerik, Flux, matched stars, Background, Seam
                  │
                  ▼
ATOMIC COMMIT
```

## 6. Deterministischer Target-Gather statt Contribution-Records

### 6.1 Ownership

Ein GPU-Thread besitzt ein Paar:

```text
(frame, target_pixel)
```

Er bestimmt über die inverse Abbildung eine konservative kleine Source-Nachbarschaft. Für diese Source-Samples berechnet er in fester Reihenfolge den exakten quadratischen Droplet-/Zellüberlapp.

Pro Farbkanal entstehen direkt die hinreichenden Statistiken:

```cpp
struct FrameSufficientStats {
  float value_sum;        // sum(K * source_value)
  float geometry_sum;     // sum(K)
  float q_composite_sum;  // sum(K * Q_composite)
  float q_fine_sum;       // sum(K * Q_scale0)
  float q_medium_sum;     // sum(K * Q_scale1)
  float artifact_sum;     // sum(K * artifact_confidence)
};
```

Ein Framekandidat ist nur gültig, wenn `geometry_sum > 0`. Sein Wert ist `value_sum / geometry_sum`.

### 6.2 Affine Frames

Für affine Transformationen wird die Zielzelle über die gecachte inverse 2×3-Matrix in Source-Geometrie abgebildet. Die konservative Source-Bounding-Box erhält einen mathematisch begründeten Droplet-Rand. Kandidaten werden in aufsteigender `(source_y, source_x)`-Reihenfolge verarbeitet.

### 6.3 Lokale Warp-Modelle

Für lokale Modelle wird pro Frame ein inverses Deformationsgitter mit konservativen Jacobian-Grenzen erzeugt. Der Target-Gather verwendet:

1. inverse Gitterinterpolation als Startwert;
2. begrenzte Newton-Verfeinerung;
3. konservative Source-Nachbarschaft;
4. exakte Vorwärtsprüfung jedes Kandidatenblatts gegen die Zielzelle.

Kann die inverse Begrenzung nicht garantiert werden, wird der Frame fail-closed zurückgewiesen. Es gibt keinen dauerhaften Rückfall auf den alten Recordpfad.

### 6.4 Determinismus

- ein Zielpixel wird nur von einem Thread geschrieben;
- Source-Nachbarn haben feste Reihenfolge;
- Frames werden in stabiler Planreihenfolge akkumuliert;
- keine atomaren Scatter-Summen;
- CPU und GPU verwenden denselben Nachbarschafts- und Overlap-Vertrag.

## 7. Speicherbegrenzte robuste Gruppenstatistik

### 7.1 Gruppenzuordnung

Für große Datensätze werden Frames deterministisch auf `K=15` robuste Gruppen verteilt:

```text
group = stable_hash(frame_id) mod 15
```

Der stabile Hash verhindert, dass zeitlich benachbarte Frames oder Ditherblöcke systematisch in derselben Gruppe landen. Für Datensätze mit weniger als 15 Frames wird eine ungerade Gruppenzahl gewählt, sodass jede Gruppe mindestens einen Frame enthält; sehr kleine Datensätze werden direkt als feste Kandidatenmenge auf dem Device reduziert.

### 7.2 Akkumulatoren

Pro Bandpixel, Kanal und Gruppe werden ausschließlich hinreichende Summen gehalten:

```cpp
struct GroupAccumulator {
  Accum uniform;
  Accum raw;
  Accum fine;
  Accum medium;

  float geometry_w;
  float geometry_w2;
  float q_wx;
  float artifact_wx;
  float registration_wx;
  uint16_t contributors;
};

struct Accum {
  float wx;
  float w;
};
```

Die GPU-Version verwendet eine dokumentierte, deterministische Akkumulationsreihenfolge. Die CPU-Referenz verwendet FP64; die GPU verwendet mindestens kompensierte FP32-Summen oder FP64, falls der definierte Fehlervertrag sonst nicht eingehalten wird. Die Abnahme erfolgt über Flux- und ULP-/Relativfehlerschranken, nicht über eine künstliche Byte-Identitätsforderung, sofern die Hardwarearithmetik diese nicht garantiert.

### 7.3 Robuste Reduktion

Nach dem letzten Frame:

1. Gruppenmittel `mu_g = uniform.wx / uniform.w` bilden;
2. gewichteten Gruppenmedian bestimmen;
3. gewichtete Gruppen-MAD bestimmen;
4. erwartete Streuungsuntergrenze aus Messrauschen und lokaler Bildgeometrie berechnen;
5. Gruppenresiduen mit einer begrenzten Einflussfunktion gewichten;
6. dieselben Robustgewichte auf Uniform, Raw, Fine und Medium anwenden.

Die robuste Skala erhält eine gradientenabhängige Untergrenze:

```text
sigma_model² = sigma_noise²
             + gradientᵀ × registration_covariance × gradient
             + sigma_sampling²
```

Damit werden reale Unterschiede an Sternkanten und Nebelgradienten nicht als Cosmic Rays behandelt.

### 7.4 Kein Supportverlust

Sind MAD oder robuste Gruppengewichte degeneriert:

- bei identischen Gruppenwerten werden alle Gruppen verwendet;
- bei unzureichender Stabilität werden Werte winsorisiert oder auf den Gruppenmedian begrenzt;
- bei zu wenigen belastbaren Gruppen wird auf alle geometrisch gültigen Gruppen zurückgefallen;
- `statistical_confidence` wird reduziert;
- `geometric_support` und Uniform bleiben erhalten.

## 8. Coverage in die Rekonstruktion integrieren

Der separate vollständige `SAMPLING_GEOMETRY`-Rasterlauf entfällt.

Der Target-Gather akkumuliert gleichzeitig:

```text
sum(B)
sum(B²)
contributors
channel_support
frame_footprint_support
```

Daraus entstehen:

```text
n_eff = sum(B)² / sum(B²)
```

sowie die vorhandenen Coverage-Gate-Metriken. Die vollständige Rekonstruktionsgeneration bleibt bis zum Abschluss aller Bänder unveröffentlicht. Erst dann werden geprüft:

- unterstützter Anteil je Kanal;
- p10-`n_eff` je Kanal;
- Analysis-Pixelzahl;
- interne geometrische Lochkomponenten;
- numerische Invarianten.

Bei Gate-Fehler wird die gesamte temporäre Generation verworfen. Damit bleibt der Fail-Closed-Vertrag erhalten, ohne die Geometrie zweimal zu berechnen.

## 9. Persistenter GPU-Workspace

### 9.1 Lebensdauer

Device- und pinned Hostbuffer werden einmal pro Phase auf die maximale geplante Bandgröße allokiert und über alle Bänder und Frames wiederverwendet:

```cpp
struct ForwardDrizzleGpuWorkspace {
  DeviceBuffer source[2];
  DeviceBuffer quality[2];
  DeviceBuffer inverse_geometry;
  DeviceBuffer group_accumulators;
  DeviceBuffer profile_output[2];
  DeviceBuffer confidence_output[2];

  PinnedBuffer host_source[2];
  PinnedBuffer host_quality[2];
  PinnedBuffer host_output[2];

  CudaStream upload;
  CudaStream compute;
  CudaStream download;
  CudaEvent slots_ready[2];
};
```

Innerhalb des Frame-/Band-Loops sind `cudaMalloc`, `cudaFree` und globales `cudaDeviceSynchronize` verboten.

### 9.2 Überlappung

Double Buffering:

```text
Slot 0: GPU berechnet Frame N
Slot 1: Host liest und lädt Frame N+1
Host:   schreibt fertiges Band N-1
```

Abhängigkeiten werden ausschließlich über Stream-Events synchronisiert. Ein Device-weites Synchronisieren ist nur am Phasenabschluss oder bei einem harten Fehler zulässig.

### 9.3 Keine X-Tiles

Die Gruppenspeichergröße ist unabhängig von `frame_count`. Deshalb wird über volle Breite und budgetierte Y-Bänder gearbeitet. Die jetzigen bis zu 40 X-Tiles pro Band entfallen.

## 10. 2×2→1×-Fold auf dem Device

Bei `internal_scale=2, output_scale=1` werden vier interne Subpixel vollständig rekonstruiert. Direkt nach der robusten Reduktion führt ein GPU-Kernel aus:

- 2×2-Flächenmittel der Profilwerte;
- konservative Kombination von `n_eff` und Confidence;
- geometrisches Support-AND über die vier internen Zellen;
- Ausgabe genau eines nativen Pixels.

Nur Output-Scale-Daten werden zum Host übertragen und persistiert. Interne 2×-Profilbilder werden nicht als vollständige Hostbuffer materialisiert.

Da robustes Clipping den geometrischen Support nicht mehr löschen darf, verstärkt das konservative 2×2-Support-AND keine statistischen Einzelpixel mehr zu schwarzen Löchern.

## 11. Neues Confidence-Modell

### 11.1 Fehler des bisherigen Modells

Das bisherige `A_separation` basiert auf:

```text
Q_p90 - Q_p50
```

Sind alle Frames ähnlich gut, ist die Differenz klein und Confidence fällt auf null. Im Testlauf lagen die mittleren Detail-Alphas bei `2,98e-10` und `7,32e-8`; Fine/Medium wurden damit praktisch vollständig deaktiviert.

### 11.2 Neue Faktoren

Confidence wird aus unabhängigen, bereits in den Gruppenakkumulatoren vorhandenen Größen gebildet:

```text
C_quality       = Funktion des robusten absoluten Q-Niveaus
C_stability     = Funktion der normierten Gruppenstreuung
C_samples       = Funktion von n_eff
C_artifact      = robuste Artifact-Confidence
C_registration  = robuste Registration-Confidence
```

Verbindliche Eigenschaft:

```text
alle Frames ähnlich gut => hohe C_quality und hohe C_stability
```

Eine mögliche konservative Kombination ist:

```text
confidence = C_samples
           × C_stability
           × min(C_quality, C_artifact, C_registration)
```

Die genaue Kalibrierung wird mit synthetischen und realen Referenzfällen festgelegt. `Q_p90-Q_p50` darf höchstens ein Zusatzsignal sein, nie eine zwingende Voraussetzung für Detailübernahme.

### 11.3 Keine per-Pixel-Allokationen

Alle Quantile werden auf höchstens 15 Gruppenwerten mit einem festen Sorting-Network oder einer festen lokalen Arraystruktur auf dem Device berechnet. Es gibt keine `std::vector`-Allokation und keine mehrfachen Sortierungen derselben Wertereihe.

## 12. Streaming-Multiband

Forward Drizzle und Multiband werden als Producer-/Consumer-Pipeline verbunden:

```text
Forward rekonstruiert Band N+1
Multiband fusioniert Band N inklusive Halo
Host persistiert Band N-1
```

Die À-trous-Fusion verwendet:

- Uniform als unveränderliche Sicherheitsreferenz;
- Raw/Fine/Medium aus denselben Robustgruppen;
- das neue Confidence-Modell;
- geometrischen Support, nicht statistischen Clip-Support.

Ein vollständiger U/R/F/M-Profilspeicher wird nur erzeugt, wenn die Resume-Konfiguration ihn verlangt. Ohne Resume-Anforderung werden ausschließlich der transaktionale finale Output und kompakte Validierungs-/Diagnoseartefakte persistiert.

## 13. CPU-Referenz

Der CPU-Pfad implementiert denselben Target-Gather- und Robustgruppenalgorithmus. Der alte Contribution-Record-Pfad bleibt nicht als dauerhafter Fallback erhalten.

Parallelisierung:

- Y-Bänder oder Pixelzeilen statisch über Worker verteilen;
- ein Pixel gehört genau einem Worker;
- Frames innerhalb eines Pixels in stabiler Reihenfolge;
- keine parallele Reduktion desselben Pixels;
- Scratch pro Worker oder feste Stackarrays.

CPU und GPU teilen:

- inverse Bounding-Regeln;
- Source-Nachbarreihenfolge;
- Polygon-/Zelloverlap;
- Gruppenzuordnung;
- robuste Einflussfunktion;
- Confidence-Formeln;
- Support- und Fallback-Invarianten.

## 14. Source-Quality-Phase

Source Quality bleibt ein eigener frameweiser Vorlauf, da alle Output-Bänder dieselben Qualitätskarten benötigen. Sie wird jedoch bereinigt:

- Source-Proxy pro Frame genau einmal;
- jede Pyramidenskala genau einmal;
- Composite während desselben Scale-Laufs akkumulieren;
- nur tatsächlich konsumierte Einzel-Skalen persistieren;
- Artifact und Registration zusammen mit ihrem Gültigkeitsstatus speichern;
- keine NaN→JSON-null-Deserialisierungsfehler;
- Accelerator-Telemetrie muss dem tatsächlich verwendeten Backend entsprechen;
- Quality-Rect-Reads erfolgen bandweise und werden mit Source-Reads überlappt.

## 15. Pflichttelemetrie

Performanceinstrumentierung ist nicht optional und hängt nicht von einer Environment-Variable ab. Pro Run werden mindestens ausgegeben:

```text
source_io_seconds
quality_io_seconds
host_to_device_seconds
gather_kernel_seconds
group_reduce_seconds
confidence_seconds
fold_2x2_seconds
multiband_seconds
device_to_host_seconds
store_write_seconds
frames_processed
bands_processed
source_bytes_read
quality_bytes_read
kernel_launches
fallback_pixels_by_channel
geometric_support_pixels_by_channel
nonfinite_pixels_inside_geometric_support
```

Zusätzlich:

- effektive GPU-Auslastung;
- CPU-Worker-Auslastung;
- Peak-RAM und Peak-VRAM;
- Durchsatz in Source-Samples/s und Output-Pixel-Frames/s;
- getrennte Zähler für robuste Downweightings und echte geometrische Nichtabdeckung.

Ein erfolgreicher Commit mit `nonfinite_pixels_inside_geometric_support > 0` ist unzulässig.

## 16. Validierungsmatrix

### 16.1 Geometrie und Flux

- Identity-Affine;
- ganzzahlige und gebrochene Translation;
- Rotation, Skalierung und Shear;
- lokale Warp-Modelle;
- alle Bayer-Patterns und CFA-Origins;
- `pixfrac`-Grenzwerte;
- `internal/output` 1/1, 2/2 und 2/1;
- konstantes Feld;
- linearer Gradient;
- Punktquelle mit analytischem Flux.

### 16.2 Robustheit

- einzelner Hotpixel;
- wiederkehrender Sensor-Hotpixel bei Dither;
- Cosmic Ray in einem Frame;
- Satellitenspur über mehrere Pixel;
- einzelne defokussierte Frames;
- Registrierungs-Ausreißer;
- negative normalisierte Hintergrundwerte;
- gesättigte Sterne;
- helle Nebelgradienten;
- sparse R/B-CFA-Abdeckung.

### 16.3 Support

Für alle Tests:

```text
geometric support => finite Uniform
geometric support + ausreichende Profileingabe => finite Raw/Fine/Medium
kein statistisch erzeugtes Loch
R/G/B-Ausgabesupport konsistent
2×2-Fold erzeugt keine Einzelpixelmaske aus Statistik
```

### 16.4 CPU/GPU

- identische Gruppenzuordnung;
- identische Supportmasken;
- identische Fallbackentscheidungen;
- Profilwerte innerhalb dokumentierter absoluter/relativer Toleranzen;
- deterministische Wiederholung auf demselben Backend;
- GPU-Fehler verwirft die gesamte uncommittete Generation;
- CPU-Neustart erzeugt denselben fachlichen Outputvertrag.

### 16.5 Skalierung

Die Messmatrix umfasst verschiedene Framezahlen bis einschließlich des vollständigen M42-Datensatzes. Nachgewiesen werden muss:

- linearer Verlauf der Gather-Arbeit mit `N`;
- konstante robuste Gruppenspeichergröße;
- keine `N log N`-Pixelreduktion;
- keine mit `N` schrumpfende X-Tile-Breite;
- keine per-Frame-/per-Tile-Deviceallokation;
- keine separate vollständige Geometrierasterisierung.

## 17. Implementierungsreihenfolge der Zielarchitektur

Die Reihenfolge bildet eine neue vertikale Pipeline; der alte Pfad wird nicht schrittweise zur Zielarchitektur erklärt.

### Phase A: mathematischer Vertrag

1. Target-Gather-Nachbarschaft für affine Transformationen spezifizieren.
2. Flux-, Support- und Determinismusinvarianten festlegen.
3. Robustgruppenverfahren einschließlich Gradient-/Registrierungs-Skala spezifizieren.
4. neues Confidence-Modell festlegen.
5. CPU/GPU-Toleranzvertrag definieren.

### Phase B: unabhängiger Referenzkern

1. neuen CPU-Target-Gather implementieren;
2. Gruppenakkumulatoren und robuste Reduktion implementieren;
3. integrierte Coverage implementieren;
4. 2×2-Fold und Supportvertrag implementieren;
5. synthetische Validierungsmatrix vollständig grün stellen.

Dieser Kern verwendet keine alten Contribution-Records und keine alte per-Frame-Kandidatenmatrix.

### Phase C: vollständiger GPU-Kern

1. persistenten Workspace implementieren;
2. affinen Gather-Kernel implementieren;
3. Gruppenakkumulation auf Device implementieren;
4. robuste Gruppenreduktion und Confidence auf Device implementieren;
5. 2×2-Fold auf Device implementieren;
6. async Source-/Quality-Pipeline implementieren;
7. CPU/GPU-Parität verifizieren.

### Phase D: lokale Warps

1. inverses Deformationsgitter spezifizieren;
2. konservative Bounds beweisen und testen;
3. lokalen Gather implementieren;
4. affine und lokale Frames durch denselben Gruppenreducer führen.

### Phase E: Multiband und Transaktion

1. Streaming-GPU-Multiband anbinden;
2. integriertes Coverage-Gate vor Commit anbinden;
3. Resume-Artefakte und Profilspeichervertrag aktualisieren;
4. Pflichttelemetrie und Report aktualisieren;
5. End-to-End-Abnahme auf realen Datensätzen durchführen.

### Phase F: Umschaltung

Der neue Pfad wird erst produktiv, wenn alle Korrektheits-, Robustheits-, Support-, Paritäts- und Skalierungs-Gates erfüllt sind. Danach wird der alte Record-/Sort-Pfad entfernt; er bleibt nicht als langfristiger Alternativmodus bestehen.

## 18. Entscheidende Architekturänderungen im Überblick

| Heute | Forward Drizzle v2 |
|---|---|
| Source-Scatter erzeugt Records | Target-Gather erzeugt direkte Statistik |
| Records GPU→CPU | nur fertige Output-Bänder GPU→CPU |
| Sortierung pro Frame/Tile | keine Recordsortierung |
| Sortierung über alle Frames pro Pixel | feste Reduktion über 15 Gruppen |
| Kandidatenmatrix skaliert mit Framezahl | Speicher skaliert mit Bandpixeln × K |
| bis zu 40 X-Tiles | volle Breite, budgetierte Y-Bänder |
| `cudaMalloc/free` pro Kleinstjob | persistenter Workspace |
| `cudaDeviceSynchronize` pro Job | Stream-Events und Double Buffering |
| Coverage als zweiter Geometriesweep | Coverage im Gather integriert |
| Clipping kann Support löschen | Statistik beeinflusst nur Confidence |
| `Q_p90-Q_p50` kann Alpha nullen | absolute Qualität + Stabilität + n_eff |
| Multiband nach vollständigem Store | Streaming-Multiband-Pipeline |
| Laufzeit enthält `N log N` pro Pixel | Laufzeit linear in N |

## 19. Abschlusskriterium

Forward Drizzle v2 gilt als AQMH-Ersatz, wenn:

1. der vollständige mathematische und transaktionale Vertrag implementiert ist;
2. innerhalb geometrischer Coverage keine schwarzen Löcher entstehen;
3. helle Sterne und Nebelgradienten robust bleiben;
4. Ausreißer ohne Supportverlust unterdrückt werden;
5. CPU und GPU denselben fachlichen Output liefern;
6. die Laufzeit linear mit der Framezahl skaliert;
7. Coverage-Geometrie nicht doppelt berechnet wird;
8. keine Contribution-Records oder framezahlabhängigen Kandidatenmatrizen mehr existieren;
9. Multiband eine fachlich sinnvolle Confidence erhält und nicht global auf null fällt;
10. der alte Record-/Sort-Pfad anschließend entfernt wird.

Die Kernentscheidung lautet:

> Forward Drizzle v2 wird als deterministische, bandweise Target-Gather-Pipeline mit robuster Gruppenstatistik, integrierter Coverage, persistentem GPU-Workspace und GPU-seitiger Multibandfusion neu aufgebaut. Die bestehende Record-/Sort-/Clip-Architektur wird nicht weiterentwickelt, sondern ersetzt.
