# Methodik v4 – Limitations & Future Work

Dieses Dokument ergänzt die **Referenzimplementierung von Methodik v4** um eine **explizite, reviewer‑taugliche Analyse der Grenzen** sowie einen **strukturierten Ausblick auf zukünftige Arbeiten**.

Es dient ausdrücklich dazu, **methodische Ehrlichkeit** zu zeigen und Angriffsflächen in Reviews vorwegzunehmen.

---

## 1. Aktuelle Limitierungen (bewusst akzeptiert)

### 1.1 Lokales Bewegungsmodell ist translationsbasiert

**Status:** bewusst minimal

- aktuelles Modell:
  \[
  p' = p + (\Delta x, \Delta y)
  \]
- lokal korrekt für kleine Tiles
- approximiert Rotation und Verzerrung nur indirekt

**Auswirkung:**
- bei sehr großen Tiles oder extremen Feldrotationen steigt die Warp‑Varianz
- wird aktuell durch **adaptive Tile‑Verfeinerung** kompensiert

**Bewertung:**
- methodisch zulässig
- kein Korrektheitsfehler

---

### 1.2 Zeitliche Warp‑Glättung ist medianbasiert

**Status:** minimalistische Referenz

- robuster Medianfilter
- keine explizite Bewegungsdynamik

**Auswirkung:**
- suboptimale Glättung bei kontinuierlicher Beschleunigung

**Bewertung:**
- korrekt, aber konservativ

---

### 1.3 Konvergenzkriterium ist rein photometrisch

**Status:** absichtlich einfach

- Abbruch basiert auf:
  \[
  \|R_{k} - R_{k-1}\| / \|R_{k}\|
  \]

**Nicht berücksichtigt:**
- PSF‑Stabilität
- Warp‑Varianz‑Plateaus

---

### 1.4 Overlap‑Add verwendet feste Fensterfunktionen

**Status:** deterministisch

- Hann/Hanning‑Fenster
- keine adaptive Randgewichtung

**Auswirkung:**
- lokale Artefakte bei abrupt wechselnder Warp‑Stabilität möglich

---

### 1.5 Keine explizite Modellierung von PSF‑Anisotropie

**Status:** out of scope

- PSF wird nur implizit über lokale Qualitätsmetriken berücksichtigt

---

## 2. Explizit nicht adressierte Problemklassen

Diese Punkte sind **bewusst ausgeschlossen**:

- absolute astrometrische Genauigkeit
- photometrische Kalibrierung auf Katalogniveau
- globale Verzerrungsmodelle
- Echtzeit‑Verarbeitung

Methodik v4 ist eine **Rekonstruktions‑, keine Astrometrie‑Pipeline**.

---

## 3. Future Work – kurz‑ bis mittelfristig

### 3.1 Erweiterte lokale Bewegungsmodelle

- affine Modelle mit starker Regularisierung
- lokale Jacobian‑Schätzung
- adaptive Modellwahl pro Tile

---

### 3.2 Physikalisch motivierte Warp‑Glättung

- Savitzky–Golay Filter
- Kalman‑Filter mit glatter Feldrotationsannahme

---

### 3.3 Adaptive Overlap‑Fenster

- Fenstergewicht abhängig von Warp‑Varianz
- Reduktion von Randartefakten

---

### 3.4 Tile‑Failure‑Taxonomie

- explizite Fehlercodes pro Tile
- bessere Diagnose und statistische Auswertung

---

### 3.5 GPU‑beschleunigte Tile‑Verarbeitung

- GPU‑basierte lokale Registrierung
- Tile‑Batching
- **ohne Aufgabe des Streaming‑Modells**

---

## 4. Langfristige Perspektive

Methodik v4 bildet eine Brücke zwischen:

- klassischem Lucky Imaging
- Multi‑Frame‑Super‑Resolution
- software‑basierter adaptiver Optik

Langfristig eröffnet sie die Möglichkeit einer **rein softwarebasierten Feld‑AO‑Approximation** für Amateur‑ und Semi‑Professional‑Astronomie.

---

## 5. Reviewer‑Positionierung (explizit)

> *The presented framework deliberately trades global geometric consistency for local physical validity. This choice is not a limitation of the implementation, but a methodological decision aligned with the underlying observational conditions.*

---

**Ende: Limitations & Future Work (Methodik v4)**

