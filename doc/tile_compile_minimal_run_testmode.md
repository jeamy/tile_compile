# Tile‑basierte Qualitätsrekonstruktion – Minimal‑Run (Testmodus)

**Status:** Normative Ableitung  
**Bezug:** `tile_basierte_qualitatsrekonstruktion_methodik_v3.md`  
**Ziel:** Schnell überprüfbarer, deterministischer Testlauf zur Validierung der Pipeline‑Semantik

---

## 1. Zweck des Minimal‑Runs

Der **Minimal‑Run (Testmodus)** dient **nicht** der Bildqualität, sondern ausschließlich der **Validierung von Methodik, Pipeline‑Abläufen und Implementierungskorrektheit**.

Er beantwortet die Fragen:

* Läuft die Pipeline **vollständig und deterministisch** durch?
* Sind alle Phasen korrekt verkettet?
* Sind Gewichtungen, Tiles und Rekonstruktion **numerisch stabil**?
* Entstehen **keine methodisch verbotenen Effekte** (Frame‑Selektion, nichtlineare Schritte, Tile‑Artefakte)?

Der Minimal‑Run ist **verbindlicher Bestandteil** der Entwicklung und CI, **kein Produktionsmodus**.

---

## 2. Grundprinzip des Testmodus

Der Testmodus ist eine **strikt reduzierte Ausprägung** der Methodik v3:

* gleiche Phasen
* gleiche mathematische Definitionen
* reduzierte Datenmenge
* reduzierte Auflösung
* reduzierte Tile‑Komplexität

> **Der Testmodus darf nichts enthalten, was im Produktionsmodus verboten ist.**

---

## 3. Reduktionen gegenüber Produktionslauf

### 3.1 Datenumfang

| Parameter | Produktion | Testmodus |
|---------|------------|-----------|
| Anzahl Frames | ≥ 800 | 32 – 64 |
| Kanäle | R/G/B | **nur G** |
| Auflösung | volle Sensorauflösung | optional 2× oder 4× Binning |

Begründung:
* G‑Kanal ist repräsentativ für Geometrie und SNR
* Farbprobleme sind **nicht** Gegenstand des Minimal‑Runs

---

### 3.2 Registrierungspfad

Im Testmodus **MUSS** der stabilste Pfad verwendet werden:

* **Siril‑basierte Registrierung**
* kein CFA‑Pfad

Ziel:
* Reduktion der Variablen
* Fokus auf Methodik‑Kern

---

### 3.3 Tile‑Geometrie

| Parameter | Produktion | Testmodus |
|---------|------------|-----------|
| Tile‑Größe | seeing‑adaptiv | **fix: 64 px** |
| Overlap | 25 % | 25 % |
| Anzahl Tiles | 200–400 | 16–64 |

---

### 3.4 Metriken

**Aktiv:**

* globale Hintergrundschätzung B_f
* globales Rauschen σ_f
* lokale FWHM‑Messung (Stern‑Tiles)

**Deaktiviert:**

* Gradientenergie E
* struktur‑basierte Tile‑Metriken

Ziel:
* Minimierung der algorithmischen Vielfalt
* klare Nachvollziehbarkeit

---

### 3.5 Zustandsbasierte Clusterung

| Modus | Produktion | Testmodus |
|-----|------------|-----------|
| Clusterung | k = 15–30 | **deaktiviert** |
| synthetische Frames | ja | **nein** |

Im Testmodus erfolgt **direkt das finale Stacking**.

---

## 4. Minimal‑Pipeline (Testmodus, normativ)

Die Pipeline **muss exakt diese Phasen in dieser Reihenfolge durchlaufen**:

1. **SCAN_INPUT**  
   * Einlesen von 32–64 Frames
   * Header‑Validierung

2. **REGISTRATION**  
   * Siril‑basierte Debayer + Registrierung

3. **CHANNEL_SPLIT**  
   * RGB → G

4. **NORMALIZATION**  
   * globale lineare Normalisierung (Pflicht)

5. **GLOBAL_METRICS**  
   * B_f, σ_f

6. **TILE_GRID**  
   * fixes 64‑px‑Grid

7. **LOCAL_METRICS**  
   * FWHM pro Tile

8. **TILE_RECONSTRUCTION**  
   * gewichtete Rekonstruktion

9. **STACKING**  
   * lineares Mittel

10. **DONE**

Keine Phase darf übersprungen werden.

---

## 5. Test‑Konfiguration (tile_compile.yaml – Minimal‑Run)

```yaml
pipeline:
  mode: test
  abort_on_fail: true

runtime_limits:
  max_runtime_minutes: 30

registration:
  enabled: true
  estimator:
    backend: siril
  application:
    mode: siril

normalization:
  enabled: true

global_metrics:
  enabled: true

local_metrics:
  enabled: true

synthetic:
  enabled: false

stacking:
  method: average

validation:
  enabled: true
  min_fwhm_improvement_percent: 0
```

---

## 6. Erfolgskriterien (Testmodus)

Der Minimal‑Run gilt als **erfolgreich**, wenn:

* alle Phasen **ohne Abbruch** durchlaufen
* keine NaN / Inf‑Werte auftreten
* ΣW_f,t > 0 für ≥ 95 % der Tiles
* keine harten Tile‑Übergänge sichtbar sind
* Re‑Run mit identischen Daten **bitgleich identisches Ergebnis** liefert

**Bildqualität ist kein Kriterium.**

---

## 7. Abbruchkriterien (Testmodus)

Sofortiger Abbruch bei:

* impliziter Frame‑Selektion
* nichtlinearer Operation vor Phase STACKING
* leeren Tile‑Gewichten
* Änderung der Tile‑Geometrie während des Laufs

---

## 8. Rolle des Minimal‑Runs im Projekt

Der Minimal‑Run ist:

* Pflicht für jede neue Backend‑Änderung
* Pflicht für jede neue Metrik
* Grundlage für CI / Regressionstests
* Referenz für GUI‑Fortschrittslogik

> **Wenn der Minimal‑Run nicht stabil ist, ist der Produktionslauf bedeutungslos.**

---

## 9. Abgrenzung

Der Testmodus:

* ersetzt **keine** Produktionsläufe
* dient **nicht** der Ergebnisoptimierung
* ist **kein Shortcut** für schwache Hardware

Er ist ein **methodischer Integritätstest**.

---

**Ende der Spezifikation**

