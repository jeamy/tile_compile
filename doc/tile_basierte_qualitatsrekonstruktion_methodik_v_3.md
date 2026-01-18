# Tile‑basierte Qualitätsrekonstruktion für DSO – Methodik v3

**Status:** Referenzspezifikation (Single Source of Truth)
**Version:** v3.1 (2026-01-09)
**Ersetzt:** Methodik v2
**Ziel:** Klare, eindeutige Workflows für zwei zulässige Registrierungs‑ und Vorverarbeitungspfade
**Gilt für:** `tile_compile.proc` (Clean Break) + `tile_compile.yaml`

---

## 0. Motivation für v3

Methodik v2 definierte die Qualitäts‑ und Rekonstruktionslogik präzise, ließ jedoch den **Vorverarbeitungspfad (OSC, Registrierung, Kanalbehandlung)** implizit.

Methodik v3 macht dies **explizit** und trennt zwei **gleichwertige, aber unterschiedliche** Pfade:

* **A – Siril‑basierter Pfad** (bewährt, geringes Risiko)
* **B – CFA‑basierter Pfad** (methodisch maximal sauber, höherer Aufwand)

Ab **Phase 2 (Tile‑Erzeugung)** sind beide Pfade **identisch**.

---

## 1. Ziel

Ziel ist es, aus vollständig registrierten, linearen Kurzzeit‑Frames astronomischer Deep‑Sky‑Objekte ein **räumlich und zeitlich optimal gewichtetes Signal** zu rekonstruieren.

Die Methode modelliert explizit zwei orthogonale Einflüsse:

* **globale atmosphärische Qualität** (Transparenz, Dunst, Hintergrunddrift)
* **lokale seeing‑ und strukturgetriebene Qualität** (Schärfe, Detailtragfähigkeit)

Es gibt **keine Frame‑Selektion**. Jeder Frame trägt gemäß seinem physikalischen Informationsgehalt bei.

---

## 2. Annahmen (verbindlich)

### 2.1 Harte Annahmen (Verstoß → Abbruch)

* Daten sind **linear** (kein Stretch, keine nichtlinearen Operatoren)
* **keine Frame‑Selektion** (Artefakt‑Rejection auf Pixelebene erlaubt)
* Verarbeitung **kanalgetrennt**
* Pipeline ist **streng linear**, ohne Rückkopplungen
* Einheitliche Belichtungszeit (Toleranz: ±5%)

### 2.2 Weiche Annahmen (mit Toleranzen)

| Annahme | Optimal | Minimum | Reduced Mode |
|---------|---------|---------|---------------|
| Frame‑Anzahl | ≥ 800 | ≥ 50 | 50–199 |
| Registrierungsresiduum | < 0.3 px | < 1.0 px | Warnung bei > 0.5 px |
| Stern‑Elongation | < 0.2 | < 0.4 | Warnung bei > 0.3 |

### 2.3 Implizite Annahmen (neu explizit)

* Stabile optische Konfiguration (Fokus, Feldkrümmung)
* Tracking‑Fehler < 1 Pixel pro Belichtung
* Keine systematischen Drifts während der Session

### 2.4 Reduced Mode (50–199 Frames)

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

## 3. Gesamtpipeline (normativ)

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

# A. Siril-basierter Pfad (Referenz, empfohlen)

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

## A.3 Übergabe an den gemeinsamen Kern

Ab hier gelten **alle Regeln aus Methodik v2 unverändert**, jedoch **kanalweise**.

Eingangsdaten:

```
R_frames[f][x,y]
G_frames[f][x,y]
B_frames[f][x,y]
```

---

# B. CFA-basierter Pfad (optional, experimentell)

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

### Zweck

Entkopplung photometrischer Transparenzschwankungen von Qualitätsmetriken.

### Anforderungen

* global
* linear
* exakt einmal
* vor **jeder** Metrik (außer B_f, das für die Normalisierung benötigt wird)
* getrennt pro Kanal

### Zulässige Methoden

* hintergrundbasierte Skalierung (maskiert, robust)
* Fallback: Skalierung über globalen Median

### Verboten

* Histogramm‑Stretch
* asinh / log

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

### Normalisierung

Alle Metriken werden robust über **Median + MAD** skaliert.

Formal (für einen Metrikwert `x`):

[
\tilde x = \frac{x - \mathrm{median}(x)}{1.4826 \cdot \mathrm{MAD}(x)}
]

### Globaler Qualitätsindex

[
Q_{f,c} = \alpha(-\tilde B_{f,c}) + \beta(-\tilde\sigma_{f,c}) + \gamma\tilde E_{f,c}
]

mit:

* α + β + γ = 1 (verbindlich)
* Default: α = 0.4, β = 0.3, γ = 0.3

`Q_f,c` wird auf **[−3, +3]** geklemmt, bevor `exp(·)` angewendet wird.

### Globales Gewicht

[
G_{f,c} = \exp(Q_{f,c})
]

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
  adaptive_weights: true
  weights:
    background: 0.4
    noise: 0.3
    gradient: 0.3
```

**Semantik:** `G_f` kodiert ausschließlich globale atmosphärische Qualität.

---

## 3.3 Tile‑Geometrie (seeing‑adaptiv)

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

## 3.3.1 Tile‑basierte Rauschunterdrückung (optional)

**Zweck:** Vor der Berechnung lokaler Metriken kann eine adaptive Rauschunterdrückung auf Tile‑Ebene angewendet werden. Diese reduziert Hintergrundrauschen, während Sterne und Strukturen erhalten bleiben.

**Algorithmus: Highpass + Soft‑Threshold**

Für jedes Tile *t* im Frame *f*:

1. **Background‑Schätzung:** Box‑Blur mit Kernelgröße *k*
   ```
   B_t = box_blur(T_t, k)
   ```

2. **Residuum (Highpass):**
   ```
   R_t = T_t − B_t
   ```

3. **Robuste Rauschschätzung (MAD):**
   ```
   σ_t = 1.4826 · median(|R_t − median(R_t)|)
   ```

4. **Soft‑Threshold:**
   ```
   τ = α · σ_t
   R'_t = sign(R_t) · max(|R_t| − τ, 0)
   ```

5. **Rekonstruktion:**
   ```
   T'_t = B_t + R'_t
   ```

**Parameter:**

| Parameter | Beschreibung | Default | Empfohlen |
|-----------|--------------|---------|-----------|
| `tile_denoising.enabled` | Aktivierung | false | true |
| `tile_denoising.kernel_size` | Box‑Blur Kernelgröße *k* (ungerade) | 15 | 31 |
| `tile_denoising.alpha` | Threshold‑Multiplikator *α* | 2.0 | 1.5 |

**Overlap‑Blending:**

Da Tiles überlappen, werden die denoisten Tiles mit linearen Gewichten geblendet:
```
w(x, y) = ramp(x) · ramp(y)
```
wobei `ramp` linear von 0 (Rand) nach 1 (Mitte) verläuft.

**Typische Ergebnisse (empirisch):**

| kernel | alpha | Noise‑Reduktion | Stern‑Erhalt |
|--------|-------|-----------------|--------------|
| 15 | 2.0 | ~75% | ~91% |
| 31 | 1.5 | ~89% | ~93% |
| 31 | 2.0 | ~89% | ~91% |

**Empfehlung:** `kernel_size=31, alpha=1.5` bietet die beste Balance zwischen Rauschunterdrückung und Signalerhalt.

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

[
Q_{star} = 0.6,(−\widetilde{\mathrm{FWHM}}) + 0.2,\tilde R + 0.2,\tilde C
]

Dabei ist `FWHM̃` die per MAD normalisierte FWHM (Median+MAD), **ohne** `log(FWHM)`.

Für Struktur-Tiles:

[
Q_{struct} = 0.7,\widetilde{(E/\sigma)} − 0.3,\tilde B
]

Default-Gewichte:
* Stern-Modus: w_fwhm = 0.6, w_round = 0.2, w_con = 0.2
* Struktur-Modus: w_struct = 0.7, w_bg = 0.3

**Lokales Gewicht:**

Alle lokalen Qualitätswerte werden auf **[−3, +3]** geklemmt.

[
L_{f,t} = \exp(Q_{local})
]

---

## 3.5 Effektives Gewicht

[
W_{f,t} = G_f \cdot L_{f,t}
]

`G_f` und `L_f,t` repräsentieren orthogonale Informationsachsen.

---

## 3.6 Tile‑Rekonstruktion

Für jedes Pixel *p* im Tile *t*:

[
I_t(p) = \frac{\sum_f W_{f,t} I_f(p)}{\sum_f W_{f,t}}
]

### Stabilitätsregeln

Definiere den Nenner:

[
D_t = \sum_f W_{f,t}
]

mit einer kleinen Konstante `ε > 0`.

* Wenn `D_t ≥ ε`: normale gewichtete Rekonstruktion.
* Wenn `D_t < ε` (z. B. alle Gewichte numerisch ~0):
  1. Rekonstruiere das Tile als **ungewichtetes Mittel** über **alle** Frames (keine Frame‑Selektion):

     [
     I_t(p) = \frac{1}{N}\sum_f I_f(p)
     ]

  2. Markiere das Tile als `fallback_used=true` (für Validierungs-/Abbruch‑Entscheidungen).

**Overlap‑Add mit Fensterfunktion (verbindlich):**

* Fensterfunktion: **Hanning** (2D, separabel)
* Formel: `w(x,y) = hann(x) · hann(y)` mit `hann(t) = 0.5 · (1 - cos(2πt))`

**Tile‑Normalisierung (verbindlich):**

Vor dem Overlap-Add wird jedes Tile normalisiert:

1. Hintergrund subtrahieren: `T'_t = T_t - median(T_t)`
2. Normalisieren: `T''_t = T'_t / median(|T'_t|)` (falls median > ε)

**Fallbacks für degenerierte / low‑weight Tiles (verbindlich):**

Definiere den Nenner

und die Stabilitätskonstante `ε = 1e-6`.

* Wenn `D_t,c ≥ ε`: normale gewichtete Rekonstruktion.
* Wenn `D_t,c < ε` (z. B. alle Gewichte numerisch ~0):
  1. Rekonstruiere das Tile als **ungewichtetes Mittel** über **alle** Frames (keine Frame‑Selektion):
  2. Markiere das Tile als `fallback_used=true` (für Validation/Abort‑Entscheidung).

Anmerkung: Diese Fallbacks sind streng linear und verletzen nicht das „keine Frame‑Selektion“‑Prinzip.

---

## 3.7 Zustandsbasierte Clusterung

### Prinzip

Ein synthetischer Frame repräsentiert einen **physikalisch kohärenten Beobachtungszustand**, nicht ein Zeitintervall.

### Zustandsvektor

Für jedes Frame *f*:

[
v_f = (G_f, \langle Q_{tile} \rangle, \mathrm{Var}(Q_{tile}), B_f, \sigma_f)
]

### Clusterung

* Clusterung der **Frames**, nicht Tiles

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

## 3.8 Synthetische Frames und finales Stacking

**Synthetische Frames (verbindlich):**

Für jeden Cluster *k* (aus §3.7) wird ein synthetischer Frame erzeugt:

[
S_{k,c} = \frac{\sum_{f\in Cluster_k} G_{f,c} \cdot I_{f,c}}{\sum_{f\in Cluster_k} G_{f,c}}
]

wobei I_f,c die **Original-Frames** (nicht rekonstruiert) sind.

**Optional (tile‑basiert, zur Qualitäts‑Propagation):**

Wenn tile‑basierte Qualitätsverbesserungen (z. B. durch Tile‑Noise‑Filtering in §3.3.1) bis in den finalen Stack getragen werden sollen, kann die Erzeugung der synthetischen Frames alternativ tile‑basiert erfolgen. Dabei wird pro Tile *t* im Cluster der effektive Gewichtsvektor

[
W_{f,t,c} = G_{f,c} \cdot L_{f,t,c}
]

verwendet und das synthetische Frame analog zur Rekonstruktion (§3.6) per Overlap‑Add aus den Tile‑Ergebnissen zusammengesetzt. Aktivierung über:

`synthetic.weighting: tile_weighted` (Default: `global`).

Ergebnis: 15–30 synthetische Frames pro Kanal (entsprechend der Cluster-Anzahl).

**Finales Stacking (verbindlich, Python-only):**

Die synthetischen Frames werden im Backend rein linear gestackt. Optional kann
vor dem Mittelwert ein **Sigma-Clipping auf Pixelebene** durchgeführt werden,
um Ausreißer (z. B. kosmische Strahlung) zu unterdrücken. Die Normativität
bezieht sich auf das Endergebnis:

[
R_c = \operatorname{mean}(\mathcal{S}_c) = \frac{1}{K} \cdot \sum_k S_{k,c}
]

mit:

* linearem Stacking (ungewichtet im Sinne des Zustandsraums – alle Gewichte
  sind bereits in S_{k,c} enthalten)
* **kein Drizzle**
* **keine zusätzliche gewichtete Umverteilung** im Stacking-Schritt

Sigma-Clipping (falls aktiviert) ist dabei als rein pixelebene Ausreißer-
Rejektion mit anschließendem Fallback auf den **unveränderten Mittelwert**
zu verstehen und verletzt weder Linearität noch das „keine Frame-Selektion“-
Prinzip.

---

## 3.9 Kombination (explizit außerhalb der Methodik)

RGB- oder LRGB-Kombination ist:

* **kein Teil** der Qualitätsrekonstruktion
* ein separater, nachgelagerter Schritt
* frei austauschbar

---

## 4. Validierung und Abbruch

### Erfolgskriterien

* mediane FWHM ↓ ≥ 5–10%
* Feldhomogenität ↑
* Hintergrund‑RMS ≤ klassisches Stacking
* keine systematischen Tile‑Artefakte

### Abbruchkriterien

* < 30% der **signaltragenden Tiles** verwendbar
* sehr geringe Streuung der Tile‑Gewichte
* sichtbare Tiling‑/Seam‑Artefakte
* Verletzung der Normalisierungsregeln

---

## 5. Kernsatz

Die Methode ersetzt die Suche nach „besten Frames“ durch eine **räumlich‑zeitliche Qualitätskarte** und nutzt jedes Informationsstück genau dort, wo es physikalisch gültig ist.

Diese Spezifikation ist **normativ**. Abweichungen erfordern explizite Versionierung.

---

## 6. Testfälle (normativ)

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

## 7. Änderungshistorie

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

---

## Anhang A – Implementierungsnotizen (nicht normativ, aber dringend empfohlen)

Dieser Anhang präzisiert rechnerische und algorithmische Details, um **reproduzierbare, robuste Implementierungen** sicherzustellen. Er erweitert die Methodik ohne ihre Semantik zu ändern.

### A.1 Hintergrundschätzung (global und lokal)

**Ziel:** robuste Trennung von Signal und atmosphärischem Dunst.

Empfohlenes Vorgehen:

* grobe Objektmaske (z. B. Sigma‑Clip + Dilatation)
* Hintergrund aus verbleibenden Pixeln berechnen
* robuste Statistik (Median oder Biweight‑Location)

Hinweis:

> Der Hintergrund darf **keine strukturellen Gradienten** enthalten, die später in E oder E/σ einfließen.

---

### A.2 Rauschschätzung σ

**Global:**

* robuste Standardabweichung aus hintergrundmaskierten Pixeln
* kein Smoothing vor der Schätzung

**Lokal (Tile):**

* gleiche Methode, aber auf Tile beschränkt
* σ wird explizit als **Normalisierung** für Strukturmetriken verwendet

---

### A.3 Gradientenergie E

**Empfohlene Definition:**

E = mean(|∇I|²)

Robustere Alternative:

E = median(|∇I|²)

Implementierungsnotizen:

* Sobel‑ oder Scharr‑Operator
* optional leichtes Pre‑Smoothing (σ ≤ 1 px), aber konsistent global & lokal
* Randpixel verwerfen

Wichtig:

> Unterschiedliche Gradientdefinitionen ändern die Skala, **nicht** die Methodik – Skalierung wird durch MAD‑Normalisierung absorbiert.

---

### A.4 Sternauswahl für FWHM

Empfohlene Kriterien:

* SNR > Schwellwert
* Elliptizität < 0.4
* keine Sättigung

FWHM:

* Messung via PSF‑Fit oder Radialprofil
* **kein** Log‑Transform; Verwendung der **MAD‑normalisierten FWHM** direkt als \widetilde{\mathrm{FWHM}}

---

### A.5 Normalisierung (Median + MAD)

Für jede Metrik x:

x̃ = (x − median(x)) / (1.4826 · MAD(x))

Hinweise:

* getrennt pro Metrik
* getrennt für global vs lokal
* Skalen nicht mischen

---

### A.6 Tile‑Normalisierung vor Overlap‑Add

Vorgehen:

1. lokalen Hintergrund schätzen und subtrahieren
2. Tile auf gemeinsamen Median skalieren
3. Fensterfunktion anwenden
4. Overlap‑Add

Guard:

* wenn |median(tile_bgfree)| < ε_median, **nicht** skalieren (scale = 1.0)

Ziel:

* Patchwork‑Helligkeit vermeiden
* Qualitätsmetriken nicht beeinflussen

---

### A.7 Clusterung

Empfehlungen:

* Standard: k‑means oder GMM
* Feature‑Vektor standardisieren
* mehrere Initialisierungen; beste Inertia/LLH wählen

Warnung:

> Zeitbasierte Clusterung ist **kein** Ersatz für Zustands‑Clusterung.

---

### A.8 Numerische Stabilität

* ε im Nenner der Tile‑Rekonstruktion explizit setzen
* exp(Q) clampen (z. B. Q ∈ [−3, 3])
* Double‑Precision bevorzugen

Empfohlene Defaults:

* ε = 1e−6
* ε_median = 1e−6

---

### A.9 Debug‑ und Diagnose‑Artefakte (empfohlen)

Während der Entwicklung speichern:

* Histogramme von Q_f und Q_local
* 2D‑Maps der Tile‑Gewichte
* Differenzbild rekonstruiert − klassisch

Diese Artefakte sind nicht Teil der Produktion, aber essenziell für Verifikation.

---

## Anhang B – Validierungsplots (formal spezifiziert)

Dieser Anhang definiert **verbindliche Validierungsartefakte**, um einen Run als **erfolgreich**, **grenzwertig** oder **fehlgeschlagen** einzustufen. Alle Plots müssen aus **produktionsrelevanten Daten** generiert werden.

### B.1 FWHM‑Verteilung (vor / nach)

**Typ:** Histogramm + Boxplot

**Inputs:**

* klassischer Referenzstack (oder Single Frames)
* synthetische Qualitätsframes

**Metriken:**

* mediane FWHM
* Interquartilsabstand

**Akzeptanz:**

* mediane FWHM‑Reduktion ≥ `validation.min_fwhm_improvement_percent`

---

### B.2 FWHM‑Feldkarte (2D)

**Typ:** Heatmap über Bildkoordinaten

**Inputs:**

* lokale FWHM‑Messungen aus Stern‑Tiles

**Ziel:**

* Feldhomogenisierung
* Reduktion von Rand‑Seeing/Rotationsartefakten

**Warnsignal:**

* harte Übergänge entlang Tile‑Grenzen

---

### B.3 Globaler Hintergrund vs Zeit

**Typ:** Linienplot

**Inputs:**

* B_f (raw) = vor globaler Normalisierung (registrierte, aber noch nicht skalierten Frames)
* effektiver Beitrag nach Gewichtung

**Ziel:**

* korrektes Down‑Weighting wolkiger Phasen

---

### B.4 Globale und lokale Gewichte über Zeit

**Typ:** Scatter/Linie

**Inputs:**

* G_f
* ⟨L_f,t⟩ pro Frame

**Ziel:**

* klare Trennung von Beobachtungszuständen

---

### B.5 Tile‑Gewichtsverteilung

**Typ:** Histogramm

**Inputs:**

* W_f,t für alle Tiles

**Akzeptanz:**

* Varianz ≥ `validation.min_tile_weight_variance`

---

### B.6 Differenzbild

**Typ:** Bild + Histogramm

**Definition:**

difference = Rekonstruktion − klassisches Stacking

**Ziel:**

* sichtbarer Detailgewinn
* keine großskaligen systematischen Muster

**Abbruch:**

* periodische Tile‑Muster

---

### B.7 SNR vs Auflösung

**Typ:** Scatter

**Inputs:**

* lokales SNR
* lokale FWHM

**Ziel:**

* physikalisch plausibler Trade‑off
* kein künstliches Überschärfen

---

## Anhang C – Komplexität und Performance‑Budget

Dieser Anhang unterstützt Planung und Skalierung von Produktionsruns.

### C.1 Rechenkomplexität (grobe Ordnung)

Sei:

* F = Anzahl Frames
* T = Anzahl Tiles
* P = Pixel pro Tile

**Globale Metriken:** O(F · N_pixels)

**Tile‑Analyse:** O(F · T · P)

**Rekonstruktion:** O(T · F · P)

Tile‑Analyse dominiert die Laufzeit.

---

### C.2 Speicherbedarf

* ein Frame im RAM (float32): ~4 · W · H Bytes
* Tile‑Puffer: ~T · P · sizeof(float)

**Empfehlung:**

* per Tile streamen
* keine vollständige Frame‑Matrix im RAM halten

---

### C.3 I/O‑Strategie

* Registrierung: ein Read/Write‑Durchlauf
* Tile‑Analyse: sequentiellen Zugriff bevorzugen
* synthetische Frames: explizit persistieren

Vermeiden:

* zufälligen Tile‑Zugriff auf rotierenden Platten

---

### C.4 Parallelisierung

Geeignete Ebenen:

* Tiles (embarrassingly parallel)
* Frames innerhalb eines Tiles (optional)

Hinweise:

* globale Normalisierung ist pro Frame unabhängig und parallelisierbar (I/O kann limitieren)
* Zustands‑Clusterung ist typischerweise nicht der Bottleneck; parallel optional

Option: RabbitMQ‑basierte Parallelisierung

Diese Option ist für spätere Implementierung vorgesehen und ermöglicht horizontale Skalierung über mehrere Worker.

* Task‑Queue: RabbitMQ
* Granularität:
  * bevorzugt: **Tile‑Tasks** (ein Task = Tile t über alle Frames f)
  * optional: Frame‑Tasks innerhalb eines Tiles (nur wenn lokale I/O schnell ist)
* Ergebnisse:
  * rekonstruiertes Tile‑Block + Summary‑Stats (z. B. ΣW, Tile‑Median nach Background‑Subtraktion)
  * separater Kanal/Queue für Diagnose‑Artefakte
* Aggregation:
  * Master sammelt Tile‑Ergebnisse und führt deterministisches Overlap‑Add aus
  * deterministische Seeds/Sortierung für Reproduzierbarkeit
* Fehlertoleranz:
  * idempotente Tasks (Tile kann neu berechnet werden)
  * Dead‑Letter‑Queue für fehlgeschlagene Tiles

---

### C.5 Laufzeitabschätzung

Für typische Werte:

* F ≈ 1000
* T ≈ 200–400
* P ≈ (64–256)²

Erwartung:

* CPU (8–16 Cores): Stunden
* GPU‑Beschleunigung: optional

---

### C.6 Abbruch bei Laufzeitlimit

Die folgenden Limits sind verbindlich:

* `runtime_limits.tile_analysis_max_factor_vs_stack`
* `runtime_limits.hard_abort_hours`

Bei Überschreitung: kontrollierter Abbruch.
