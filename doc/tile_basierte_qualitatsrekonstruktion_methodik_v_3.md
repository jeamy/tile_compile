# Tile‑basierte Qualitätsrekonstruktion für DSO – Methodik v3

**Status:** Normative Spezifikation (Single Source of Truth)  
**Version:** v3.1 (2026-01-09)  
**Ersetzt:** Methodik v2  
**Ziel:** Klare, trennscharfe Abläufe für zwei zulässige Registrierungs‑ und Vorverarbeitungspfade

---

## 0. Motivation für v3

Methodik v2 definierte die Qualitäts‑ und Rekonstruktionslogik präzise, ließ jedoch den **Vorverarbeitungspfad (OSC, Registrierung, Kanalbehandlung)** implizit.

Methodik v3 macht dies **explizit** und trennt zwei **gleichwertige, aber unterschiedliche** Pfade:

* **A – Siril‑basierter Pfad** (bewährt, geringes Risiko)
* **B – CFA‑basierter Pfad** (methodisch maximal sauber, höherer Aufwand)

Ab **Phase 2 (Tile‑Erzeugung)** sind beide Pfade **identisch**.

---

## 1. Invariante Grundannahmen (verbindlich)

### 1.1 Harte Annahmen (Verstoß → Abbruch)

* Daten sind **linear** (kein Stretch, keine nichtlinearen Operatoren)
* **keine Frame‑Selektion** (Artefakt‑Rejection auf Pixelebene erlaubt)
* Verarbeitung **kanalgetrennt**
* Pipeline ist **streng linear**, ohne Rückkopplungen
* Einheitliche Belichtungszeit (Toleranz: ±5%)

### 1.2 Weiche Annahmen (mit Toleranzen)

| Annahme | Optimal | Minimum | Reduced Mode |
|---------|---------|---------|---------------|
| Frame‑Anzahl | ≥ 800 | ≥ 50 | 50–199 |
| Registrierungsresiduum | < 0.3 px | < 1.0 px | Warnung bei > 0.5 px |
| Stern‑Elongation | < 0.2 | < 0.4 | Warnung bei > 0.3 |

### 1.3 Implizite Annahmen (neu explizit)

* Stabile optische Konfiguration (Fokus, Feldkrümmung)
* Tracking‑Fehler < 1 Pixel pro Belichtung
* Keine systematischen Drifts während der Session

### 1.4 Reduced Mode (50–199 Frames)

Bei Frame‑Anzahl unterhalb des Optimums aber oberhalb des Minimums:

* Zustandsbasierte Clusterung (§3.7) wird **übersprungen**
* Keine synthetischen Frames (§3.8 wird übersprungen)
* Validierungswarnung im Report

**Workflow im Reduced Mode (verbindlich):**

1. Schritte 1–7 werden normal ausgeführt (inkl. Tile-basierte Rekonstruktion)
2. Schritte 8–9 werden übersprungen
3. Schritt 10 stackt die **rekonstruierten Frames aus §3.6** direkt:

```
R_c = rekonstruiertes Bild aus §3.6
```

Alternativ (falls Clusterung dennoch aktiviert):
* Cluster‑Anzahl wird auf 5–10 reduziert
* Synthetische Frames werden mit reduzierter Cluster-Anzahl erzeugt

**Graduelles Degradieren (statt hartem Abbruch):**

Bei Verletzung von Annahmen wird nicht sofort abgebrochen, sondern stufenweise degradiert:

| Schweregrad | Aktion | Beispiel |
|-------------|--------|----------|
| **Warnung** | Fortfahren mit Hinweis | Registrierungsresiduum 0.5-1.0 px |
| **Degradiert** | Fallback-Modus aktivieren | < 50 Frames → Reduced Mode ohne Clusterung |
| **Kritisch** | Abbruch mit Erklärung | Keine Sterne gefunden, Daten nicht linear |

Nur bei **kritischen** Fehlern (Datenintegrität verletzt) erfolgt ein Abbruch.

---

## 2. Gesamtpipeline (v3, normativ)

Gemeinsame abstrakte Pipeline:

1. Registrierung & geometrische Vereinheitlichung
2. Kanaltrennung
3. Globale lineare Normalisierung (Pflicht)
4. Globale Frame‑Metriken
5. Tile‑Geometrie
6. Lokale Tile‑Metriken
7. Tile‑basierte Rekonstruktion (kanalweise)
8. Zustandsbasierte Clusterung
9. Rekonstruktion synthetischer Frames
10. Finales lineares Stacking (kanalweise)
11. **Kombination (RGB / LRGB) – außerhalb der Methodik**

**Unterschiede A/B betreffen ausschließlich Schritt 1–2.**

---

# A. Siril‑basierter Pfad (Referenz, empfohlen)

## A.1 Zweck und Einordnung

Der Siril‑Pfad nutzt die **jahrelang erprobte Registrierungs‑ und Debayer‑Logik** von Siril. Die Methodik greift **erst danach**.

Dieser Pfad ist:

* stabil
* reproduzierbar
* risikoarm
* für Produktion empfohlen

---

## A.2 Schritte A.1–A.2 (Siril)

### A.2.1 Debayer + Registrierung (Siril)

* Input: rohe OSC‑Frames
* Siril übernimmt:
  * Debayer (Interpolation)
  * Sternfindung
  * Transformationsschätzung
  * Rotation / Translation

**Ergebnis:**

* registrierte, debayerte RGB‑Frames
* exakt eine Geometrie pro Frame

---

### A.2.2 Kanaltrennung (nach Registrierung!)

* RGB → R / G / B
* ab hier **keine kanalübergreifenden Operationen mehr**

Begründung:

> Kanalübergreifendes Stacken führt zu kohärenter Addition farbabhängiger Resampling‑Residuen.

---

## A.3 Übergabepunkt an die Methodik

Ab hier gelten **alle Regeln aus Methodik v2 unverändert**, jedoch **kanalweise**.

Eingangsdaten:

```
R_frames[f][x,y]
G_frames[f][x,y]
B_frames[f][x,y]
```

---

# B. CFA‑basierter Pfad (optional, experimentell)

## B.1 Zweck und Einordnung

Der CFA‑Pfad vermeidet **jede farbabhängige Interpolation vor der Tile‑Analyse**.

Er ist methodisch ideal, aber:

* komplexer
* implementierungsintensiv
* aktuell experimentell

---

## B.2 Schritte B.1–B.2 (CFA)

### B.2.1 Registrierung auf CFA‑Luminanz

* CFA‑Luminanz aus realen Samples (z. B. G‑dominant oder Summe)
* Schätzung **einer einzigen** Transformation pro Frame
* robuste Verfahren (RANSAC / ECC)

**Wichtig:**

> Die Transformation ist farbunabhängig, die Interpolation jedoch **CFA‑aware**.

---

### B.2.2 CFA‑aware Transformation

* CFA‑Mosaik wird in 4 Subplanes zerlegt (R, G1, G2, B)
* identische Transformation auf jeden Subplane
* **keine Interpolation zwischen Bayer‑Phasen**
* Re‑Interleaving zum CFA

Ergebnis:

* registrierte CFA‑Frames ohne Farbphasen‑Mischung

---

### B.2.3 Kanaltrennung

* CFA → R / G / B (oder G‑only)
* ab hier identisch zu Pfad A

---

# 3. Gemeinsamer Kern ab Phase 3 (A == B)

## 3.1 Globale lineare Normalisierung (Pflicht)

* exakt einmal
* vor **jeder** Metrik (außer B_f, das für die Normalisierung benötigt wird)
* getrennt pro Kanal

**Reihenfolge (verbindlich):**

1. B_f,c (Hintergrundniveau) wird auf **Rohdaten** berechnet
2. Normalisierung: `I'_f = I_f / B_f`
3. σ_f,c und E_f,c werden auf **normalisierten** Daten berechnet

Formal:

```
B_f,c = median(I_f,c)           # VOR Normalisierung
I'_f,c = I_f,c / B_f,c          # Normalisierung
σ_f,c = std(I'_f,c)             # NACH Normalisierung
E_f,c = gradient_energy(I'_f,c) # NACH Normalisierung
```

---

## 3.2 Globale Frame‑Metriken

Pro Frame *f* und Kanal *c*:

* B_f,c – Hintergrundniveau (auf Rohdaten, siehe §3.1)
* σ_f,c – Rauschen (auf normalisierten Daten)
* E_f,c – Gradientenergie (auf normalisierten Daten)

Globaler Qualitätsindex:

```
Q_f,c = α(-B̃) + β(-σ̃) + γẼ
G_f,c = exp(Q_f,c)
```

wobei B̃, σ̃, Ẽ die MAD-normalisierten Werte sind:

```
x̃ = (x - median(x)) / (1.4826 · MAD(x))
```

Nebenbedingung (verbindlich):

* α + β + γ = 1
* Default: α = 0.4, β = 0.3, γ = 0.3

**Adaptive Gewichtung:**

Falls die Datencharakteristik stark von typischen Bedingungen abweicht, können die Gewichte adaptiv angepasst werden.

**Algorithmus (varianzbasiert):**

```
1. Berechne Varianz der Metriken:
   Var(B), Var(σ), Var(E)

2. Gewichte basierend auf Varianz:
   α' = Var(B) / (Var(B) + Var(σ) + Var(E))
   β' = Var(σ) / (Var(B) + Var(σ) + Var(E))
   γ' = Var(E) / (Var(B) + Var(σ) + Var(E))

3. Constraints anwenden:
   α', β', γ' = clip(α', β', γ', 0.1, 0.7)

4. Renormalisieren:
   α', β', γ' = normalize(α', β', γ') so dass Σ = 1
```

**Eigenschaften:**
* Je höher die Varianz einer Metrik, desto mehr Gewicht erhält sie
* Minimum 0.1, Maximum 0.7 pro Gewicht (verhindert Extreme)
* Summe garantiert = 1.0
* Fallback auf Default-Gewichte bei Var = 0

**Konfiguration:**

```yaml
global_metrics:
  adaptive_weights: true  # Aktiviert dynamische Gewichtung
  weights:                # Fallback-Werte
    background: 0.4
    noise: 0.3
    gradient: 0.3
```

Stabilitätsregel:

* `Q_f,c` wird auf **[−3, +3]** geklemmt, bevor `exp(·)` angewendet wird.

---

## 3.3 Tile‑Erzeugung (entscheidender Punkt)

> **Tiles werden NACH Registrierung und Kanaltrennung erzeugt, aber VOR jeder Kombination.**

* identisches Tile‑Grid für alle Frames und Kanäle
* seeing‑adaptiv

Definitionen:

* `W`, `H` – Bildbreite/-höhe in Pixel
* `F` – robuste FWHM‑Schätzung in Pixel (z. B. Median über viele Sterne und Frames)
* `s = tile.size_factor` – dimensionsloser Skalierungsfaktor
* `T_min = tile.min_size`
* `D = tile.max_divisor`
* `o = tile.overlap_fraction` mit `0 ≤ o ≤ 0.5`

Herleitung (kompakt, normativ):

1. Ein seeing‑limitierter Stern besitzt eine charakteristische Skala `F` (FWHM). Um lokale Seeing‑/Fokus‑ und Strukturvariationen stabil zu messen, muss ein Tile **mehrere PSF‑Skalen** abdecken.
2. Wir setzen daher die Tile‑Kantenlänge proportional zur PSF‑Skala:

```
T_0 = s · F
```

3. Zusätzlich erzwingen wir **untere** und **obere** Schranken:
    - Untere Schranke: zu kleine Tiles sind instabil (zu wenig Sterne/Struktur, hohe Varianz)
    - Obere Schranke: Tiles dürfen nicht zu groß werden, sonst verschwindet Lokalität

 Normative Tile‑Size‑Formel:
 
 ```
 T = floor(clip(T_0, T_min, floor(min(W, H) / D)))
 O = floor(o · T)
 S = T − O
 ```
 
 wobei `clip(x, a, b) = min(max(x, a), b)`.

**Grenzwertprüfungen (verbindlich):**

Vor der Tile-Berechnung müssen folgende Bedingungen geprüft werden:

1. `F > 0`: Falls FWHM nicht messbar, verwende Default `F = 3.0`
2. `T_min ≥ 16`: Absolute Untergrenze für Tile-Größe
3. `T ≥ T_min`: Falls `T < T_min` nach Berechnung, setze `T = T_min`
4. `S > 0`: Falls `S ≤ 0` (bei extremem Overlap), setze `o = 0.25` und berechne neu
5. `min(W, H) ≥ T`: Falls Bild kleiner als Tile, verwende `T = min(W, H)` und `O = 0`

---

## 3.4 Lokale Tile‑Metriken

Für jedes Tile *t*, Frame *f*, Kanal *c*:

**Stern‑Tiles** (Tiles mit ≥ `tile.star_min_count` Sternen):

* FWHM_t,f,c – mittlere Halbwertsbreite der Sterne im Tile
* R_t,f,c – Rundheit (1 = perfekt rund, 0 = elongiert)
* C_t,f,c – Kontrast (Stern-Signal / Hintergrund)

**Struktur‑Tiles** (Tiles ohne ausreichend Sterne):

* (E/σ)_t,f,c – Signal-zu-Rausch-Verhältnis der Struktur
* B_t,f,c – lokaler Hintergrund

**Lokaler Qualitätsindex (verbindlich):**

Für Stern-Tiles:
```
Q_local[f,t,c] = w_fwhm · (-FWHM̃) + w_round · R̃ + w_con · C̃
```

Für Struktur-Tiles:
```
Q_local[f,t,c] = w_struct · (Ẽ/σ̃) + w_bg · (-B̃)
```

Default-Gewichte:
* Stern-Modus: w_fwhm = 0.6, w_round = 0.2, w_con = 0.2
* Struktur-Modus: w_struct = 0.7, w_bg = 0.3

**Lokales Gewicht:**

```
Q_local wird auf [-3, +3] geklemmt
L_f,t,c = exp(Q_local)
```

---

## 3.5 Effektives Gewicht

```
W_f,t,c = G_f,c · L_f,t,c
```

---

## 3.6 Tile‑basierte Rekonstruktion (kanalweise)

```
I_t,c(p) = Σ_f W_f,t,c · I_f,c(p) / Σ_f W_f,t,c
```

**Overlap‑Add mit Fensterfunktion (verbindlich):**

* Fensterfunktion: **Hanning** (2D, separabel)
* Formel: `w(x,y) = hann(x) · hann(y)` mit `hann(t) = 0.5 · (1 - cos(2πt))`

**Tile‑Normalisierung (verbindlich):**

Vor dem Overlap-Add wird jedes Tile normalisiert:

1. Hintergrund subtrahieren: `T'_t = T_t - median(T_t)`
2. Normalisieren: `T''_t = T'_t / median(|T'_t|)` (falls median > ε)

**Fallbacks für degenerierte / low‑weight Tiles (verbindlich):**

Definiere den Nenner

```
D_t,c = Σ_f W_f,t,c
```

und die Stabilitätskonstante `ε = 1e-6`.

* Wenn `D_t,c ≥ ε`: normale gewichtete Rekonstruktion.
* Wenn `D_t,c < ε` (z. B. alle Gewichte numerisch ~0):
  1. Rekonstruiere das Tile als **ungewichtetes Mittel** über **alle** Frames (kein Frame‑Selection):

```
I_t,c(p) = (1/N) · Σ_f I_f,c(p)
```

  2. Markiere das Tile als `fallback_used=true` (für Validation/Abort‑Entscheidung).

Anmerkung: Diese Fallbacks sind streng linear und verletzen nicht das „keine Frame‑Selektion“‑Prinzip.

---

## 3.7 Zustandsbasierte Clusterung

* Clusterung der Frames (nicht Tiles)
* Zustandsvektor:

```
v_f,c = (G_f,c, ⟨Q_local⟩, Var(Q_local), B_f,c, σ_f,c)
```

**Dynamische Cluster-Anzahl (verbindlich):**

Die Cluster-Anzahl K wird adaptiv basierend auf der Frame-Anzahl N berechnet:

```
K = clip(floor(N / 10), K_min, K_max)
```

wobei:
* K_min = 5 (Minimum für stabile Statistik)
* K_max = 30 (Maximum für Effizienz)
* N = Anzahl der Frames

Beispiele:
* N = 50 → K = 5
* N = 200 → K = 20
* N = 500 → K = 30
* N = 800 → K = 30 (gedeckelt)

---

## 3.8 Synthetische Frames & finales Stacking

**Synthetische Frames (verbindlich):**

Für jeden Cluster *k* (aus §3.7) wird ein synthetischer Frame erzeugt:

```
S_k,c = Σ_{f∈Cluster_k} G_f,c · I_f,c / Σ_{f∈Cluster_k} G_f,c
```

wobei I_f,c die **Original-Frames** (nicht rekonstruiert) sind.

Ergebnis: 15–30 synthetische Frames pro Kanal (entsprechend der Cluster-Anzahl).

**Finales Stacking (verbindlich):**

Die synthetischen Frames werden linear gestackt:

```
R_c = (1/K) · Σ_k S_k,c
```

* lineares Stacking (ungewichtet)
* **kein Drizzle**
* **keine zusätzliche Gewichtung**

Ergebnis:

```
Rekonstruktion_R.fit
Rekonstruktion_G.fit
Rekonstruktion_B.fit
```

---

## 4. Kombination (explizit außerhalb der Methodik)

RGB‑ oder LRGB‑Kombination ist:

* **kein Teil** der Qualitätsrekonstruktion
* ein separater, nachgelagerter Schritt
* frei austauschbar

---

## 4.1 Testfälle (normativ)

Die folgenden Testfälle sind verbindlich. Ein Run gilt nur dann als methodikkonform, wenn diese Tests (automatisiert oder nachvollziehbar manuell) erfüllt sind.

1. **Gewichtsnormierung global**
    - **Given**: α, β, γ aus Konfiguration
    - **Then**: α + β + γ = 1 (harte Fehlermeldung sonst)

2. **Clamping vor Exponentialfunktion**
    - **Given**: beliebige Metrik‑Werte, auch Ausreißer
    - **Then**: `Q_f,c` und `Q_local` werden vor `exp(·)` auf [−3, +3] geklemmt

3. **Tile‑Size‑Monotonie**
    - **Given**: zwei seeing‑Schätzungen `F1 < F2`
    - **Then**: `T(F1) ≤ T(F2)` (unter Berücksichtigung der Clamps)

 4. **Overlap‑Konsistenz**
    - **Then**: `0 ≤ overlap_fraction ≤ 0.5` und `O = floor(o·T)`, `S = T−O` sind ganzzahlig und deterministisch

5. **Low‑weight Tile Fallback**
    - **Given**: Tile mit `D_t,c < ε`
    - **Then**: Rekonstruktion nutzt ungewichtetes Mittel über **alle** Frames; Ergebnis enthält keine NaNs/Infs

6. **Kanaltrennung / keine Kanal‑Kopplung**
    - **Then**: Kein Metrik‑, Gewicht‑ oder Rekonstruktionsschritt mischt Informationen zwischen R/G/B

7. **Keine Frame‑Selektion (invariante Regel)**
    - **Then**: Jede Rekonstruktionsformel verwendet alle Frames; Abweichungen führen zum Abbruch

8. **Determinismus**
    - **Given**: gleiche Inputs (Frames + Config)
    - **Then**: bit‑stabile oder numerisch stabile (Toleranz definiert) Outputs und identische Tile‑Geometrie

---

## 5. Kernaussage v3

* Registrierungspfad ist **austauschbar** (Siril oder CFA)
* Qualitäts‑ und Rekonstruktionslogik ist **einheitlich**
* Tiles sind **kanalrein und lokal bewertet**
* Kombination erfolgt **erst nach Abschluss der Methodik**

Diese Spezifikation ist **normativ**. Abweichungen erfordern explizite Versionierung.

---

## 6. Änderungshistorie

| Datum | Version | Änderungen |
|-------|---------|------------|
| 2026-01-09 | v3.1 | Grenzwertprüfungen für Tile-Geometrie (§3.3) |
| 2026-01-09 | v3.1 | Dynamische Cluster-Anzahl K = clip(N/10, 5, 30) (§3.7) |
| 2026-01-09 | v3.1 | Adaptive Gewichtung als optionale Erweiterung (§3.2) |
| 2026-01-09 | v3.1 | Graduelles Degradieren statt hartem Abbruch (§1.4) |
| 2026-01-09 | v3.1 | Explizite Q_local Formel mit MAD-Normalisierung (§3.4) |
| 2026-01-09 | v3.1 | Hanning-Fensterfunktion und ε=1e-6 spezifiziert (§3.6) |
| 2026-01-09 | v3.1 | Synthetische Frame-Formel explizit dokumentiert (§3.8) |
| 2026-01-09 | v3.0 | Initiale v3 Spezifikation mit Pfad A/B |
