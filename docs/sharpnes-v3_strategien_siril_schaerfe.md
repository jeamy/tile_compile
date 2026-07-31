# Strategien, um die Siril-Sternschärfe zu erreichen

**Datum:** 2026-07-31
**Branch:** `sharpnes-v3`
**Referenzen:**

- `docs/m31_20260729_stacked_rgb_vergleich.md`
- `docs/sharpnes-v3_code_review_und_naechste_schritte.md`
- Run `M31-s1_guardfix_20260731_1`
- Run `M31-s1_lanczos4_20260731_1`
- Siril-Referenz `result.fit` / `resultHMS.fit`

---

## 1. Ausgangslage

Das Siril-Bild ist nicht nur durch einen besseren Schärfefilter schärfer. Die entscheidende Differenz liegt in der Pipeline-Domäne:

```text
Siril:
Debayering pro Frame
→ Registration auf voll aufgelöstem RGB
→ Lanczos auf RGB
→ Mean/Sigma-Clip-Stack
```

```text
tile_compile AQMH:
CFA-Mosaik
→ Registration/Prewarp auf CFA-Subplanes
→ AQMH auf CFA-Luminanz
→ Debayering nach der Reconstruction
```

Beim aktuellen `M31-s1_lanczos4_20260731_1` wurden die Sternkerne positionsgematcht ungefähr 17 % schmaler und der Peak/Flux ungefähr 20 % höher als beim Guard-Fix. Gleichzeitig stieg die AQMH-Background-Regression auf etwa 26,65 % und der Lauf wurde deshalb verworfen.

Die sichtbaren Blockstrukturen entstehen bereits im CFA-Domain-Processing. Ein nachträgliches Debayering kann diese Rasterstruktur nur abmildern, aber verlorene oder falsch interpolierte CFA-Information nicht vollständig rekonstruieren.

---

## 2. Vergleich der Strategien

| Strategie | Schärfe-Potenzial | Wahrscheinlichkeit, Siril zu erreichen | Risiko | Empfehlung |
|---|---:|---:|---:|---|
| Debayer-First-AQMH | hoch | hoch | hoch | wichtigster kontrollierter Architekturweg |
| Debayerte Luminanz für Registration | mittel | mittel | mittel | sinnvoller Zwischenversuch |
| CFA-aware Drizzle/Dither-AQMH | sehr hoch | unklar, aber höchste Decke | sehr hoch | bester Kandidat für maximale Schärfe |
| Maskierte RGB-Deconvolution | mittel | niedrig bis mittel | mittel | erst nach Rasterproblem |
| Besseres Demosaicing allein | niedrig bis mittel | niedrig | niedrig | notwendig, aber nicht ausreichend |

---

## 3. Ist Strategie 3 der beste Kandidat?

### Kurze Antwort

**Strategie 3 hat wahrscheinlich das höchste theoretische Schärfe-Potenzial.** Sie ist aber nicht automatisch der wahrscheinlichste oder schnellste Weg, mindestens Siril-Niveau zu erreichen.

Die präzisere Bewertung lautet:

- **höchste theoretische Schärfe-Decke:** Strategie 3, CFA-aware Drizzle/Dither-Reconstruction;
- **höchste Wahrscheinlichkeit bei kontrollierbarem Aufwand:** Debayer-First-AQMH;
- **niedrigstes Risiko für einen Zwischengewinn:** Debayerte Luminanz als Registration-Proxy.

Strategie 3 ist besonders interessant, weil der Guard-Fix-Run starke Dither-Bewegungen zeigt:

```text
detected dithers: 632 / 645 Frames
fraction:         97.98 %
minimum shift:    0.7 px
```

Das bedeutet, dass die Daten grundsätzlich Subpixel-Information über mehrere Bayer-Phasen enthalten können. Ob diese Information ausreichend gleichmäßig verteilt ist, muss aber noch geprüft werden. Eine reine Schätzung „98 % gedithert = Siril-Schärfe garantiert“ wäre falsch.

---

## 4. Strategie 1: Debayer-First-AQMH

### Ziel

Die CFA-Subsampling-Verluste sollen vor dem Warp vermieden werden, während der AQMH-Per-Pixel-Vorteil erhalten bleibt.

### Zielablauf

```text
Frame normalisieren
→ CFA-korrektes Edge-Aware/VNG-Debayering
→ Registration auf RGB oder debayerter Luminanz
→ R/G/B separat auf Canvas warpen
→ gemeinsame Luminanz-Q-Maps berechnen
→ AQMH-Reconstruction je Kanal
→ RGB-Ausgabe
```

### Wichtig

Der alte D+-Versuch war dafür kein vollständiger Beweis. Er verwendete Frame-Level-Global-Quality-Gewichte beziehungsweise einen gewichteten RGB-Mean-Stack, nicht drei echte Per-Pixel-AQMH-Reconstruction-Pfade.

Ein gültiger Test muss daher:

- Q-Maps auf debayerter Luminanz berechnen;
- dieselben Q-Maps auf R, G und B anwenden;
- pro Kanal eigene Valid-Masks führen;
- die kanalweise Sigma-Clip-/Gewichtssumme korrekt erhalten;
- RGB erst nach der Reconstruction zusammenführen.

### Vorteile

- entspricht der Siril-Domäne wesentlich besser;
- reduziert den CFA-Prewarp-Blur;
- kontrollierbarer als ein komplett neues Drizzle-Verfahren;
- gute Vergleichbarkeit mit dem vorhandenen AQMH-CFA-Pfad.

### Risiken

- ungefähr dreifacher Kanal-Compute;
- höherer Speicherbedarf;
- debayerte Kanäle sind korreliert;
- gemeinsame Luminanz-Q-Maps können für R/G/B unterschiedlich optimal sein.

### Empfehlung

Das ist der beste erste echte Architekturtest, wenn das Ziel eine hohe Wahrscheinlichkeit für Siril-ähnliche Sternprofile ist.

---

## 5. Strategie 2: Debayerte Luminanz nur für Registration

### Zielablauf

```text
CFA normalisieren
→ temporäre debayerte Luminanz erzeugen
→ Registration auf der Luminanz
→ AQMH-Reconstruction weiterhin im CFA
→ finales Debayering
```

Das ist kein vollständiger Siril-Pfad, kann aber Registrierungsfehler und Subpixel-Alignment verbessern.

### Vorteile

- deutlich geringerer Umbau als Debayer-First-AQMH;
- keine dreifache AQMH-Reconstruction;
- kann isoliert gegen die aktuelle CFA-Registration getestet werden.

### Einschränkung

Der eigentliche Warp und die AQMH-Reconstruction bleiben im CFA-Gitter. Der Hauptverlust durch kanalweise Halbabtastung bleibt deshalb bestehen.

### Erwartung

Wahrscheinlich kleinerer, aber sauber messbarer Gewinn. Diese Strategie ist vor Strategie 1 sinnvoll, wenn zunächst ein risikoarmer Proxy-Test gewünscht ist.

---

## 6. Strategie 3: CFA-aware Drizzle/Dither-Reconstruction

### Grundidee

Nicht mehr jeden Bayer-Subplane als reguläres Bild interpolieren, sondern jedes einzelne CFA-Sample mit seiner echten registrierten Subpixelposition in ein Farbkanal-Gitter akkumulieren.

```text
CFA-Sample
→ Bayer-Farbzuordnung
→ registrierte Subpixelposition
→ AQMH-/SNR-/Support-Gewicht
→ R/G/B-Akkumulation
→ Normalisierung durch Gewichts-/Supportkarte
```

Das ist eher ein CFA-aware Drizzle als ein klassischer `warpAffine`-Schritt.

### Warum das die höchste Schärfe-Decke hat

Die 645 Frames enthalten durch Dithering verschiedene Subpixel-Positionen. Wenn diese Positionen präzise bekannt sind, können Informationen zusammengeführt werden, die bei einem einzelnen CFA-Prewarp auf ein fixes 2×2-Gitter verloren gehen.

Strategie 3 kann damit theoretisch:

- die effektive Kanalauflösung erhöhen;
- das Bayer-Raster reduzieren;
- Subpixel-Dithering direkt ausnutzen;
- den Siril-Vorteil teilweise oder sogar vollständig erreichen.

### Warum es keine Garantie gibt

Dafür müssen gleichzeitig stimmen:

- Subpixel-Warp-Genauigkeit;
- Dither-Abdeckung in X und Y;
- korrekte Bayer-Phase pro Sample;
- Photometrie und Kanalnormierung;
- gültige Support-/Coverage-Karten;
- Outlier- und Hotpixel-Behandlung;
- keine systematischen Registrierungsdrifts.

Die gemessene Dither-Quote von 97,98 % sagt nur, dass Verschiebungen vorhanden sind. Sie sagt noch nicht, dass die Subpixel-Positionen gleichmäßig genug verteilt sind.

### Risiken

- höchste Implementierungskomplexität;
- großes Risiko für Löcher und ungleichmäßige Coverage;
- schwierige GPU-/CPU-Konsistenz;
- höhere Anforderungen an Speicher und I/O;
- neu zu definierende AQMH-Semantik.

### Empfehlung

Strategie 3 ist der **beste Kandidat für die maximale Schärfe**, aber nicht für den ersten Produktionsversuch. Sie sollte nach einem erfolgreichen Debayer-First-Prototyp als gezieltes Forschungsprojekt umgesetzt werden.

Die beste langfristige Architektur wäre wahrscheinlich eine Kombination:

```text
Debayer-/RGB- oder CFA-aware Sample-Registration
+ Dither-/Drizzle-Akkumulation
+ gemeinsame AQMH-Qualitätssteuerung
+ per Kanal gültige Supportkarten
```

---

## 7. Strategie 4: Maskierte RGB-Deconvolution

Deconvolution kann nach dem finalen Edge-Aware-Debayering die Kernspitzen verbessern:

```text
lineares RGB
→ positionsabhängige PSF-Schätzung
→ Sternmaske
→ konservative Wiener/Richardson-Lucy-Deconvolution
→ Background unverändert lassen
```

Sie ist aber nicht die primäre Lösung für das Bayer-Raster. Die globale Unsharp-Mask zeigte bereits:

- FWHM-Verbesserung;
- aber starke Background-RMS-Verschlechterung;
- zusätzliche Tail-/Seam-Probleme.

Deshalb nur maskiert und mit striktem Raw-/Background-Gate einsetzen.

---

## 8. Strategie 5: Besseres Demosaicing

Edge-Aware, VNG oder ein hochwertiger Raw-Demosaicer sind sinnvoll. Allein werden sie aber die Siril-Lücke wahrscheinlich nicht schließen.

Demosaicing kommt aktuell zu spät:

```text
CFA-Prewarp und AQMH
→ bereits geprägte CFA-Rasterstruktur
→ Debayering
```

Ein besserer Demosaicer kann diese Struktur reduzieren, aber nicht rückgängig machen.

---

## 9. Empfohlene Implementierungsreihenfolge

### Phase 1: Speicherbegrenzter Prototyp

- 32–64 Frames;
- maximal 2 Worker;
- keine BGE/PCC/HypMetric-Phasen;
- gemeinsame Luminanz-Q-Maps;
- ein Kanal nach dem anderen;
- RSS und Coverage protokollieren.

### Phase 2: Debayer-First-AQMH

Zunächst Strategie 1 umsetzen. Dabei nicht gleichzeitig Drizzle, neue Schärfemetriken und neue Interpolation einführen.

Abnahmekriterien:

- keine Bayer-/2×2-Blockstrukturen;
- keine relevante Background-Regressionssteigerung;
- positionsgematchte radiale FWHM besser als CFA-AQMH;
- Peak/Flux besser als CFA-AQMH;
- kein Raw-Baseline-Verstoß.

### Phase 3: Dither-Verteilung analysieren

Vor Strategie 3 die tatsächliche Subpixel-Verteilung prüfen:

- X-/Y-Shift-Histogramm;
- Verteilung modulo 2 Pixel;
- Coverage-Karte pro Bayer-Phase;
- lokale Dither-Abdeckung an Sternpositionen;
- Registrierungsrestfehler.

### Phase 4: CFA-aware Drizzle

Erst wenn die Dither-/Coverage-Analyse positiv ist:

- Sample-Akkumulation statt CFA-Subplane-Warp;
- zunächst CPU-Referenzpfad;
- danach GPU-Optimierung;
- identische CPU/GPU-Ergebnisse innerhalb Toleranz.

---

## 10. Klare Entscheidung

Wenn die Frage lautet:

> Welche Strategie hat die höchste Chance, mindestens Siril-Schärfe zu erreichen?

Dann lautet die Antwort:

- **praktisch wahrscheinlichster Weg:** echter Debayer-First-AQMH-Pfad;
- **höchstes theoretisches Potenzial:** CFA-aware Drizzle/Dither-AQMH;
- **beste langfristige Lösung:** Kombination aus debayerter bzw. sample-basierter Registration, Dither-Akkumulation und Per-Pixel-AQMH-Gewichten.

Strategie 3 ist somit der beste Kandidat für die maximale Schärfe, aber nicht automatisch der beste erste Implementierungsschritt. Ein sauberer Debayer-First-AQMH-Prototyp ist die notwendige Referenz, um zu entscheiden, ob der zusätzliche Drizzle-Aufwand gerechtfertigt ist.

Weitere reine Interpolations- oder globale Sharpening-Versuche sollten nicht priorisiert werden, weil die bisherigen Runs gezeigt haben, dass sie die CFA-Rasterursache nicht beseitigen.
