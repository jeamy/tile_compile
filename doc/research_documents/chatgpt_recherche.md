# Vergleichbare Methoden zur Qualitätsverbesserung astronomischer CCD-Aufnahmen

## 1. Ausgangspunkt und Referenz

Das Referenzdokument beschreibt eine **streng lineare, tile-basierte Rekonstruktionsmethodik** zur Optimierung von Deep-Sky-CCD-Aufnahmen.  
Kernmerkmale sind:

- vollständige Nutzung aller Frames (kein Frame Rejection)
- explizite Trennung von **globaler atmosphärischer Qualität** und **lokaler, seeing-getriebener Qualität**
- adaptive, seeing-abhängige Tile-Geometrie
- gewichtete Rekonstruktion auf Tile-Ebene
- optionale Zustands-Clustering- und synthetische Qualitätsframes

Im Folgenden werden bekannte Verfahren zusammengefasst, die **funktional oder konzeptionell vergleichbare Ziele** verfolgen.

---

## 2. Klassische CCD-Bildverbesserung (Baseline)

### 2.1 Kalibrierung (Bias, Dark, Flat)
**Ziel:** Entfernung instrumenteller und sensorischer Artefakte  
**Charakteristik:**
- global
- deterministisch
- strikt linear

**Abgrenzung:**  
Diese Verfahren sind notwendige Vorbedingungen, liefern jedoch **keine qualitätsadaptive Rekonstruktion**.

---

## 3. Klassisches Stacking mit Qualitätsgewichtung

### 3.1 Frame-basierte Gewichtung
**Beispiele:**  
- FWHM-basierte Gewichtung  
- SNR- oder Hintergrund-basierte Gewichtung  

**Eigenschaften:**
- ein Gewicht pro Frame
- implizite Annahme homogener Bildqualität

**Relation zur Referenzmethode:**  
✔ Vergleichbar auf globaler Ebene  
✘ Keine lokale Differenzierung  
✘ Seeing- und Transparenz-Effekte nicht orthogonal getrennt  

---

## 4. Lucky Imaging / Frame Selection

### 4.1 Prinzip
- Auswahl eines kleinen Prozentsatzes der besten Frames
- Maximierung der Auflösung bei hellem Signal

**Relation zur Referenzmethode:**  
✘ Fundamentaler Gegensatz  
Die Referenzmethodik ersetzt Frame Selection vollständig durch **kontinuierliche Gewichtung**.

---

## 5. Lokale, segmentierte Rekonstruktionsansätze

### 5.1 Raumvariable PSF-Modelle
**Beispiele:**
- Feldabhängige Deconvolution
- AO-nahe Rekonstruktion

**Eigenschaften:**
- explizite Modellierung lokaler Bildqualität
- oft nichtlinear oder iterativ

**Relation zur Referenzmethode:**  
✔ Lokale Qualitätsbetrachtung  
✘ Häufig nicht strikt linear  
✘ Modell- statt datengetrieben

---

## 6. Multi-Resolution- und Wavelet-Methoden

### 6.1 Wavelet-Denoising
**Ziel:** Trennung von Signal und Rauschen über Skalen  
**Eigenschaften:**
- lokal
- skalenabhängig
- häufig nichtlinear

**Relation:**  
✔ Lokale Signalbewertung  
✘ Keine physikalische Qualitätsgewichtung  
✘ Post-Processing, keine Rekonstruktion

---

## 7. Deconvolution-basierte Rekonstruktion

### 7.1 Richardson–Lucy / Lucy–Hook
**Ziel:** Rückrechnung der PSF  
**Eigenschaften:**
- iterativ
- nichtlinear
- PSF-abhängig

**Relation:**  
✔ Auflösungssteigerung  
✘ Keine Qualitätsmodellierung über Zeit  
✘ Gefahr von Artefakten

---

## 8. KI- und ML-basierte Verfahren

### 8.1 Neuronale Denoiser / Super-Resolution
**Eigenschaften:**
- datengetrieben
- oft visuell beeindruckend
- schwer validierbar

**Relation:**  
✘ Keine physikalische Nachvollziehbarkeit  
✘ Verletzung linearer Invarianz  
✘ Wissenschaftlich riskant

---

## 9. Methodischer Vergleich (Kurzüberblick)

| Ansatz | Lokal | Global | Linear | Frame-Selection-frei | Physikalisch interpretierbar |
|------|-------|--------|--------|----------------------|-----------------------------|
| Klassisches Stacking | ✘ | ✔ | ✔ | ✘ | ✔ |
| Lucky Imaging | ✘ | ✔ | ✔ | ✘ | ✔ |
| PSF-Deconvolution | ✔ | ✘ | ✘ | ✔ | ⚠ |
| Wavelets | ✔ | ✘ | ✘ | ✔ | ⚠ |
| KI-Methoden | ✔ | ✘ | ✘ | ✔ | ✘ |
| **Tile-basierte Rekonstruktion (Referenz)** | ✔ | ✔ | ✔ | ✔ | ✔ |

---

## 10. Kernaussage

Die im Referenzdokument beschriebene Methodik stellt **keine inkrementelle Verbesserung bestehender Verfahren**, sondern eine **strukturell neue Klasse** dar:

- sie ersetzt Frame Selection durch **spatio-temporale Qualitätsfelder**
- sie kombiniert globale und lokale Qualitätsachsen explizit
- sie bleibt strikt linear, deterministisch und validierbar

In der zugänglichen Fachliteratur existieren **Teilparallelen**, jedoch **kein direkt äquivalenter Ansatz**, der diese Eigenschaften vollständig vereint.

---

## 11. Fazit

Die Referenzmethodik kann als:

> **physikalisch konsistente, lokal adaptive Generalisierung klassischen Stackings**

eingeordnet werden – mit klarer Abgrenzung zu Lucky Imaging, Deconvolution und KI-basierter Bildmanipulation.
