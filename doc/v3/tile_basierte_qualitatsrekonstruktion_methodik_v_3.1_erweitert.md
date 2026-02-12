# Tile‑basierte Qualitätsrekonstruktion für DSO – Methodik v3.1 (Erweitert)

**Status:** Erweiterte Referenzspezifikation
**Version:** v3.1E (2026-02-12)
**Basiert auf:** Methodik v3.1 (2026-01-09)
**Ziel:** Integration fortschrittlicher Methoden der C++ Implementierung in die Spezifikation
**Gilt für:** `tile_compile.proc` (Clean Break) + `tile_compile.yaml`

---

## 0. Motivation für v3.1E

Methodik v3.1 definierte die Qualitäts‑ und Rekonstruktionslogik präzise, aber die aktuelle C++ Implementierung hat einige methodische Verbesserungen eingeführt, die noch nicht in der Spezifikation enthalten sind. Diese erweiterte Version integriert diese fortschrittlichen Methoden in die formale Spezifikation, insbesondere:

* **Erweiterte Registrierungskaskade** mit mehreren spezialisierten Algorithmen
* **Fortschrittliche CFA‑aware Transformation** für präzisere Bayer‑Muster‑Erhaltung
* **Qualitätsbasierte Referenzauswahl** für optimale Registrierung
* **Adaptive Rauschunterdrückung** auf Tile‑Ebene
* **Robustere statistische Methoden** für Metriken und Validierung

Die Kernkonzepte und Ziele der Methodik v3 bleiben unverändert.

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
11. **Post-Processing (optional):**
   - RGB/LRGB-Kombination (außerhalb der Methodik)
   - Astrometrische Kalibrierung (WCS)
   - Photometrische Farbkalibrierung (PCC)

**Unterschiede A/B betreffen ausschließlich Schritt 1–2.**

---

# A. Registrierung + Debayer Pfad (Referenz)

## A.1 Zweck und Einordnung

Dieser Pfad verwendet eine **eigenständige mehrstufige Registrierung** mit anschließendem Debayer. Die Registrierung erfolgt auf Basis eines extrahierten Luminanzkanals (bei OSC-Daten) oder direkt auf Mono-Daten.

Dieser Pfad ist:

* stabil
* robust
* für Produktion empfohlen

---

## A.2 Schritte A.1–A.2

### A.2.1 Mehrstufige Registrierung (NEU)

#### A.2.1.1 Vorbereitung und Referenzauswahl

1. **Proxy-Erzeugung**:
   - OSC-Daten: Extraktion eines Luminanz-Proxys via `cfa_green_proxy_downsample2x2`
   - Mono-Daten: Downsampling via `downsample2x2_mean` für Geschwindigkeitsoptimierung

2. **Qualitätsbasierte Referenzauswahl** (NEU):
   - Globale Qualitätsmetriken für alle Frames berechnen
   - Auswahl des Referenzframes nach einem der folgenden Kriterien:
     a) Höchstes globales Qualitätsgewicht (`global_weight`) 
     b) Niedrigster Hintergrundwert (`background`)
     c) Niedrigstes Rauschen (`noise`)
     d) Höchste Gradientenergie (`gradient_energy`)
     e) Median-Frame in der zeitlichen Sequenz (Fallback)

#### A.2.1.2 Registrierungskaskade (NEU)

Für jedes Frame wird eine Kaskade von Registrierungsalgorithmen ausgeführt, bis eine erfolgreiche Registrierung erreicht ist:

```
Für jedes Frame:
  1. Versuche hybrid_phase_ecc
  2. Falls fehlgeschlagen: robust_phase_ecc
  3. Falls fehlgeschlagen: triangle_star_matching
  4. Falls fehlgeschlagen: star_registration_similarity
  5. Falls fehlgeschlagen: feature_registration_similarity
  6. Falls alle fehlschlagen: Fallback auf Identitätstransformation mit Warnung
```

Jeder erfolgreiche Registrierungsschritt wird durch **NCC (Normalized Cross Correlation)** validiert:
- Berechnung der NCC zwischen registriertem Frame und Referenz
- Vergleich mit unregistrierter Baseline-NCC
- Akzeptanz nur bei signifikanter Verbesserung

#### A.2.1.3 Registrierungsalgorithmen (NEU)

1. **hybrid_phase_ecc**:
   - Kombination aus Phasenkorrelation und ECC
   - Schätzung der groben Translation via Phasenkorrelation
   - Verfeinung durch Enhanced Correlation Coefficient

2. **robust_phase_ecc**:
   - Robuste Version mit Ausreißererkennung
   - Multi-Resolution-Ansatz für bessere Konvergenz
   - Adaptive Schrittweite

3. **triangle_star_matching**:
   - Geometrische Übereinstimmung von Sterndreiecken
   - Rotation und Translation werden geschätzt
   - RANSAC für robuste Estimation

4. **star_registration_similarity**:
   - RANSAC-basierte robuste Registrierung
   - Direkte Sternpaarzuordnung
   - Similarity-Transformation (Rotation, Translation, Skalierung)

5. **feature_registration_similarity**:
   - Feature-basierte Registrierung (z.B. AKAZE, SIFT)
   - Deskriptor-Matching
   - Homographie-Schätzung

#### A.2.1.4 Anwendung der Transformationen

1. **OSC-Daten**:
   - Transformation via `apply_global_warp`, was intern zu CFA-aware Transformation führt
   - Parameter: Vollauflösungs-Frame, berechnete Warp-Matrix

2. **Mono-Daten**:
   - Standard-Warp via OpenCV's `warpAffine`
   - Parameter: Vollauflösungs-Frame, berechnete Warp-Matrix, Interpolation=INTER_LINEAR, WARP_INVERSE_MAP

---

### A.2.2 Kanaltrennung (nach Registrierung!)

* OSC: Debayer der registrierten Frames via `debayer_nearest_neighbor`
  * R → R_frames[f][x,y]
  * G → G_frames[f][x,y]
  * B → B_frames[f][x,y]
* Mono: Direkte Weiterverarbeitung
* Ab hier **keine kanalübergreifenden Operationen mehr**

Begründung:

> Kanalübergreifendes Stacken führt zu kohärenter Addition farbabhängiger Resampling‑Residuen.

---

## A.3 Übergabe an den gemeinsamen Kern

Ab hier gelten **alle Regeln aus dem gemeinsamen Kern unverändert**, kanalweise.

Eingangsdaten:

```
R_frames[f][x,y]
G_frames[f][x,y]
B_frames[f][x,y]
```

---

# B. CFA-basierter Pfad

## B.1 Zweck und Einordnung

Der CFA‑Pfad vermeidet **jede farbabhängige Interpolation vor der Tile‑Analyse**.

Er ist methodisch ideal für OSC-Daten, da:
* keine Farbinterpolation vor kritischen Analysen erfolgt
* Bayer-Muster-Integrität während der Transformation erhalten bleibt
* maximale Auflösung erhalten wird

---

## B.2 Schritte B.1–B.2 (CFA)

### B.2.1 Registrierung auf CFA‑Luminanz

**CFA-Luminanz-Extraktion**:
1. Erzeugung eines Luminanz-Proxys via `cfa_green_proxy`
2. Optional: Downsampling via `cfa_green_proxy_downsample2x2`

**Registrierung**:
* Gleiche mehrstufige Registrierungskaskade wie in A.2.1.2
* Registrierung erfolgt auf dem Luminanz-Proxy
* Schätzung **einer einzigen** Transformation pro Frame

**Wichtig:**

> Die Transformation ist farbunabhängig, die Anwendung jedoch **CFA‑aware**.

---

### B.2.2 CFA‑aware Transformation (NEU)

**Implementation**: `warp_cfa_mosaic_via_subplanes`

**Algorithmus**:
1. CFA‑Mosaik wird in 4 Subplanes zerlegt:
   * R-Subplane: Pixel an (r_row, r_col) Positionen
   * G1-Subplane: Erste grüne Position
   * G2-Subplane: Zweite grüne Position
   * B-Subplane: Pixel an (b_row, b_col) Positionen

2. Separate Transformation jeder Subplane:
   * Gleiche Warp-Matrix für alle Subplanes
   * Interpolation nur innerhalb der jeweiligen Subplane
   * **Keine Interpolation zwischen Bayer‑Phasen**

3. Re‑Interleaving zum CFA:
   * Rekonstruktion des transformierten CFA-Mosaiks
   * Erhaltung der Bayer-Phasenpositionen

**Ergebnis:**
* Registrierte CFA‑Frames ohne Farbphasen‑Mischung
* Erhaltung der ursprünglichen Auflösung und Farbintegrität

---

### B.2.3 Kanaltrennung

* CFA → R / G / B via `split_cfa_channels`
* Optional: Zusammenführung der G1/G2-Kanäle bei Bedarf
* Ab hier identisch zu Pfad A

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

## 3.3.1 Tile‑basierte Rauschunterdrückung (erweitert)

**Zweck:** Vor der Berechnung lokaler Metriken kann eine adaptive Rauschunterdrückung auf Tile‑Ebene angewendet werden. Diese reduziert Hintergrundrauschen, während Sterne und Strukturen erhalten bleiben.

### Highpass + Soft‑Threshold (Basis)

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

### Wiener-Filter (NEU)

Als alternative fortschrittliche Methode kann ein Wiener-Filter im Frequenzraum angewendet werden:

**Implementation**: `wiener_tile_filter`

**Algorithmus**:
1. **Padding**: Das Tile wird mit einer Reflection-Padding-Strategie erweitert
2. **FFT**: Transformation ins Frequenzspektrum
3. **Power-Spektrum**: Berechnung des Leistungsspektrums |F|²
4. **Wiener-Filter**: Anwendung des Wiener-Filters H = (|F|² - σ²) / |F|²
5. **Clamp**: Begrenzung der Filterwerte auf [0,1]
6. **Inverse FFT**: Rücktransformation in den Ortsraum
7. **Crop**: Ausschneiden des originalen Tile-Bereichs

**Parameter**:
* `sigma`: Rauschschätzung aus dem Tile
* `snr_threshold`: Minimaler SNR-Wert für Filteranwendung
* `q_min`: Minimaler Qualitätswert für Filteranwendung

**Selektive Anwendung**:
* Nur auf Struktur-Tiles, nicht auf Stern-Tiles
* Nur bei ausreichender Struktur (q_struct_tile > q_min)
* Nur bei niedrigem SNR (snr_tile < snr_threshold)

**Overlap‑Blending:**

Da Tiles überlappen, werden die entstörrten Tiles mit linearen Gewichten geblendet:
```
w(x, y) = ramp(x) · ramp(y)
```
wobei `ramp` linear von 0 (Rand) nach 1 (Mitte) verläuft.

**Konfiguration:**

```yaml
tile_denoising:
  enabled: true
  method: "wiener"  # oder "threshold"
  wiener:
    snr_threshold: 10.0
    q_min: 0.2
  threshold:
    kernel_size: 31
    alpha: 1.5
```

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

### FWHM-Messung (erweitert)

**Standard-Methode (PSF-Fit)**:
* Sterne werden via PSF-Fitting vermessen
* FWHM wird direkt aus dem Fit-Parameter extrahiert

**Alternative Proxy-Methode** (für Felder mit wenigen Sternen):
* Für jedes Stern-Tile wird die Anzahl der Pixel oberhalb der halben Maximalhöhe gezählt
* FWHM-Proxy = sqrt(Pixelanzahl / π)

**Sternfindung**:
* OpenCV's `goodFeaturesToTrack` für schnelle Eckpunktdetektion
* Filterung nach Helligkeit, Größe und Isolation
* Minimaler Abstand zwischen Sternen konfigurierbar

**Rundheitsberechnung**:
* Standardmethode: Verhältnis der Hauptachsen einer elliptischen Anpassung
* Proxy-Methode: Verhältnis der Standardabweichungen in x- und y-Richtung um den Peak

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

Anmerkung: Diese Fallbacks sind streng linear und verletzen nicht das „keine Frame‑Selektion"‑Prinzip.

### Gewichtetes Sigma-Clipping auf Tile-Ebene (NEU)

Als Erweiterung kann eine Ausreißerrejektion auf Pixel-Ebene durchgeführt werden, die die Gewichte berücksichtigt:

**Implementation**: `sigma_clip_weighted_tile`

**Algorithmus**:
1. Für jedes Pixel p im Tile:
   a. Sammle die Werte dieses Pixels aus allen Frames und die zugehörigen Gewichte
   b. Berechne den gewichteten Mittelwert und die gewichtete Standardabweichung
   c. Markiere Werte außerhalb von [μ-σ_low, μ+σ_high] als Ausreißer
   d. Wiederhole a-c bis Konvergenz oder max_iter erreicht
   e. Berechne den gewichteten Mittelwert der nicht-ausgeschlossenen Werte

2. Fallback bei zu vielen verworfenen Werten:
   - Falls weniger als min_fraction der Werte übrig bleiben, verwende alle Werte

**Parameter**:
* `sigma_low`: Sigma-Faktor für untere Schranke (Default: 3.0)
* `sigma_high`: Sigma-Faktor für obere Schranke (Default: 3.0)
* `max_iters`: Maximale Anzahl an Iterationen (Default: 3)
* `min_fraction`: Minimaler Anteil der Werte, die beibehalten werden müssen (Default: 0.5)

**Konfiguration**:
```yaml
tile_reconstruction:
  sigma_clip:
    enabled: true
    sigma_low: 3.0
    sigma_high: 3.0
    max_iters: 3
    min_fraction: 0.5
```

**Wichtig**: Die gewichtete Sigma-Clipping-Methode respektiert die Gewichte der Frames und stellt sicher, dass keine systematische Frame-Selektion erfolgt.

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

Wenn tile‑basierte Qualitätsverbesserungen (z. B. durch Tile‑Noise‑Filtering in §3.3.1) bis in den finalen Stack getragen werden sollen, kann die Erzeugung der synthetischen Frames alternativ tile‑basiert erfolgen. Dabei wird pro Tile *t* im Cluster der effektive Gewichtsvektor

[
W_{f,t,c} = G_{f,c} \cdot L_{f,t,c}
]

verwendet und das synthetische Frame analog zur Rekonstruktion (§3.6) per Overlap‑Add aus den Tile‑Ergebnissen zusammengesetzt. Aktivierung über:

`synthetic.weighting: tile_weighted` (Default: `global`).

Ergebnis: 15–30 synthetische Frames pro Kanal (entsprechend der Cluster-Anzahl).

**Finales Stacking (verbindlich):**

Die synthetischen Frames werden im Backend rein linear gestackt. Optional kann
vor dem Mittelwert ein **Sigma-Clipping auf Pixelebene** durchgeführt werden,
um Ausreißer (z. B. kosmische Strahlung) zu unterdrücken.

**Implementation**: `sigma_clip_stack`

**Algorithmus**:
1. Für jedes Pixel p:
   a. Sammle die Werte dieses Pixels aus allen synthetischen Frames
   b. Berechne Mittelwert und Standardabweichung
   c. Markiere Werte außerhalb von [μ-σ_low, μ+σ_high] als Ausreißer
   d. Wiederhole a-c bis Konvergenz oder max_iter erreicht
   e. Berechne den Mittelwert der nicht-ausgeschlossenen Werte

2. Fallback bei zu vielen verworfenen Werten:
   - Falls weniger als min_fraction der Werte übrig bleiben, verwende alle Werte

Die Normativität bezieht sich auf das Endergebnis:

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
zu verstehen und verletzt weder Linearität noch das „keine Frame-Selektion"-
Prinzip.

---

## 3.9 Nachbearbeitung (Post-Processing)

### 3.9.1 RGB-Kombination

RGB- oder LRGB-Kombination ist:

* **kein Teil** der Qualitätsrekonstruktion
* ein separater, nachgelagerter Schritt
* frei austauschbar

### 3.9.2 Astrometrische Kalibrierung (NEU)

Für die Erzeugung einer astrometrischen Lösung (WCS - World Coordinate System) wird folgende Methode implementiert:

**Implementation**: `wcs_from_cdelt_crota` und `parse_wcs_file`

**Datenstruktur**:
- Die WCS-Struktur enthält eine vollständige TAN-Projektion mit CD-Matrix
- Referenzpixel (CRPIX1, CRPIX2)
- Referenz-Himmelskoordinaten (CRVAL1, CRVAL2 in Grad)
- CD-Matrix (CD1_1, CD1_2, CD2_1, CD2_2 in Grad/Pixel)
- Bilddimensionen (NAXIS1, NAXIS2)

**Algorithmen**:
1. **WCS-Parsing**:
   - Einlesen von ASTAP .wcs-Dateien (FITS-ähnliches Format)
   - Konvertierung zwischen CDELT/CROTA und CD-Matrix-Formaten
   - Unterstützung für ältere und neuere FITS-WCS-Standards

2. **Koordinatenumwandlung**:
   - Pixel zu Himmel (RA/Dec): TAN (gnomonic) Projektion
   - Himmel zu Pixel: Inverse TAN-Projektion
   - Suchradius-Berechnung für Katalogabfragen

3. **Convenience-Funktionen**:
   - Berechnung von Pixelskala (Bogensekunden/Pixel)
   - Berechnung der Bildrotation (Grad)
   - Sichtfeld-Abmessungen (Grad)

**Eigenschaften**:
- Validierung der WCS-Parameter auf Konsistenz
- Berechnung der Transformationen mit Doppelpräzision
- Unterstützung für die gnomische (TAN) Projektion

### 3.9.3 Photometrische Farbkalibrierung (NEU)

Die Photometrische Farbkalibrierung (PCC) passt die Farbbalance des Bildes an reale Sternfarben an. Dies erfolgt durch die Anpassung an Sterndaten aus dem Gaia-Katalog.

**Implementation**: `run_pcc`, `measure_stars`, `fit_color_matrix`, `apply_color_matrix`

**Datenquellen**:
1. **Gaia-Katalog (Siril-Format)**:
   - XP-gesampelte Spektren (336-1020 nm in 2 nm Schritten)
   - HEALPix-basierte Partitionierung für effiziente Suche
   - Kegelförmige Suche (Cone Search) innerhalb eines Radius

2. **Fallback-Kataloge über VizieR**:
   - Gaia DR3: Direkter API-Zugriff für Teff-Werte
   - APASS DR9: Fallback für schwächere Sterne, mit B-V-Indizes

**Algorithmischer Ablauf**:
1. **Sternmessung**:
   - Apertur-Photometrie auf katalogidentifizierten Sternen
   - Annulus-basierte Hintergrundsubtraktion
   - Filtermessung in R-, G- und B-Kanälen

2. **Synthetische Flusswerte**:
   - Für Gaia-XP-Spektren: Integration der Spektren über Filterkurven
   - Für Teff-basierte Sterne: Schwarzkörperspektrum-Modellierung
   - B-V zu Teff Konvertierung für APASS-Sterne

3. **Farbmatrix-Berechnung**:
   - Robuste 3x3 Farbkorrekturmatrix-Anpassung
   - Sigma-Clipping für Ausreißererkennung
   - RMS-Residuen-Validierung

4. **Anwendung der Farbkorrektur**:
   - Lineartransformation der RGB-Kanäle
   - In-Place-Anwendung auf lineare Bilddaten

**Parameter**:
```yaml
pcc:
  enabled: true
  aperture_radius_px: 8.0
  annulus_inner_px: 12.0
  annulus_outer_px: 18.0
  mag_limit: 14.0
  mag_bright_limit: 6.0
  min_stars: 10
  sigma_clip: 2.5
  catalog:
    type: "siril_gaia"  # oder "vizier_gaia", "vizier_apass"
    path: "$HOME/.local/share/siril/siril_cat1_healpix8_xpsamp"
    download_missing: true
```

**Validierung**:
- Minimale Sternanzahl für zuverlässige Anpassung (standardmäßig 10)
- RMS-Residuen unter einem konfigurierbaren Schwellenwert
- Deterministische Katalogabfragen für Reproduzierbarkeit

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

Die Methode ersetzt die Suche nach „besten Frames" durch eine **räumlich‑zeitliche Qualitätskarte** und nutzt jedes Informationsstück genau dort, wo es physikalisch gültig ist.

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

9. **Registrierungskaskade Fallback (NEU)**
    - **Given**: Ein Frame, das mit keinem primären Algorithmus registriert werden kann
    - **Then**: Kaskade durchläuft alle verfügbaren Methoden bis eine erfolgreich ist oder Fallback auf Identity

10. **CFA-Transformation Phasenerhalt (NEU)**
    - **Given**: Ein CFA-Mosaikbild mit Bayer-Pattern
    - **Then**: Nach Transformation via `warp_cfa_mosaic_via_subplanes` bleibt das Bayer-Pattern korrekt erhalten

11. **WCS-Konsistenz (NEU)**
    - **Given**: Ein gültiger WCS mit Referenzpixel und -koordinaten
    - **Then**: Pixel-zu-Himmel-zu-Pixel-Konvertierung muss für Bildpunkte innerhalb des Sichtfelds einen Rundtrip-Fehler < 0.01 Pixel haben

12. **PCC-Matrix-Eigenschaft (NEU)**
    - **Given**: Eine erfolgreich angepasste PCC-Farbmatrix
    - **Then**: Die Matrix-Determinante muss größer als 0 sein, und kein Element darf negativ sein

---

## 7. Änderungshistorie

| Datum | Version | Änderungen |
|-------|---------|------------|
| 2026-02-12 | v3.1E | Integration fortschrittlicher Methoden aus C++ Implementierung |
| 2026-02-12 | v3.1E | Erweiterte Registrierungskaskade (A.2.1.2) |
| 2026-02-12 | v3.1E | CFA-aware Transformation detailliert (B.2.2) |
| 2026-02-12 | v3.1E | Wiener-Filter für Tiles hinzugefügt (3.3.1) |
| 2026-02-12 | v3.1E | Gewichtetes Sigma-Clipping spezifiziert (3.6) |
| 2026-02-12 | v3.1E | Astrometrische Kalibrierung (WCS) hinzugefügt (3.9.2) |
| 2026-02-12 | v3.1E | Photometrische Farbkalibrierung (PCC) hinzugefügt (3.9.3) |
| 2026-02-12 | v3.1E | Zusätzliche Testfälle für neue Funktionalität (6.9-10) |
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

### A.10 Registrierungskaskade (NEU)

Die Registrierungskaskade ist ein kritischer Erfolgsfaktor. Empfohlene Implementierung:

1. **Vorverarbeitung für Registrierung**:
   - Rauschunterdrückung (Gauß-Filter mit σ=1.5)
   - Histogrammnormalisierung auf [0,1]
   - Optional: Downsampling für Geschwindigkeit

2. **Kaskade mit frühem Abbruch**:
   ```python
   # Pseudocode
   def register_with_cascade(moving, reference):
       # 1. Start mit schnellstem Verfahren
       result = hybrid_phase_ecc(moving, reference)
       if result.success and validate_ncc(result):
           return result
       
       # 2. Robusteres Phasen-ECC
       result = robust_phase_ecc(moving, reference)
       if result.success and validate_ncc(result):
           return result
           
       # 3. Geometrische Verfahren für schwierige Fälle
       result = triangle_star_matching(moving, reference)
       if result.success and validate_ncc(result):
           return result
           
       # 4. Direkte Sternpaarzuordnung
       result = star_registration_similarity(moving, reference)
       if result.success and validate_ncc(result):
           return result
           
       # 5. Feature-basierte Methoden
       result = feature_registration_similarity(moving, reference)
       if result.success and validate_ncc(result):
           return result
           
       # 6. Fallback auf Identität mit Warnung
       return identity_with_warning()
   ```

3. **Validierung jeder Registrierung**:
   - NCC vor und nach der Transformation berechnen
   - Mindestverbesserung von 10% fordern
   - Visuelle Kontrolle (optional): Differenzbild vorher/nachher

4. **Parameter-Tuning**:
   - Für jede Methode eigene Parameter optimal wählen
   - Hybrid-Methode: schnell, aber weniger robust
   - Geometrische Methoden: langsamer, aber robust bei schwierigen Bildern

---

### A.11 CFA-aware Transformation (NEU)

Die `warp_cfa_mosaic_via_subplanes`-Methode ist wie folgt implementiert:

```python
# Pseudocode
def warp_cfa_mosaic_via_subplanes(mosaic, warp_matrix, bayer_pattern):
    # 1. Bestimme Bayer-Positionen
    r_row, r_col, b_row, b_col = bayer_offsets(bayer_pattern)
    
    # 2. Extrahiere Subplanes
    r_plane = extract_subplane(mosaic, r_row, r_col)
    g1_plane = extract_subplane(mosaic, (r_row+1)%2, r_col)
    g2_plane = extract_subplane(mosaic, r_row, (r_col+1)%2)
    b_plane = extract_subplane(mosaic, b_row, b_col)
    
    # 3. Warp jede Subplane separat
    r_warped = warp_affine(r_plane, warp_matrix)
    g1_warped = warp_affine(g1_plane, warp_matrix)
    g2_warped = warp_affine(g2_plane, warp_matrix)
    b_warped = warp_affine(b_plane, warp_matrix)
    
    # 4. Reassemble CFA mosaic
    result = zeros_like(mosaic)
    
    for y in range(height):
        for x in range(width):
            py = y % 2
            px = x % 2
            
            if py == r_row and px == r_col:
                result[y,x] = r_warped[y//2, x//2]
            elif py == b_row and px == b_col:
                result[y,x] = b_warped[y//2, x//2]
            elif py == g1_row and px == g1_col:
                result[y,x] = g1_warped[y//2, x//2]
            else:
                result[y,x] = g2_warped[y//2, x//2]
    
    return result
```

Diese Methode stellt sicher, dass Farbinformationen während der Transformation nicht vermischt werden und Bayer-Artefakte vermieden werden.

---

### A.12 Wiener-Filterung für Tiles (NEU)

Die Frequenzraum-basierte Wiener-Filterung bietet eine optimale Balance zwischen Rauschunterdrückung und Detailerhalt:

```python
# Pseudocode
def wiener_tile_filter(tile, sigma, snr_tile, q_struct_tile, is_star_tile, config):
    if not config.enabled or is_star_tile:
        return tile  # Keine Filterung für Stern-Tiles
        
    if snr_tile >= config.snr_threshold:
        return tile  # SNR bereits ausreichend gut
        
    if q_struct_tile <= config.q_min:
        return tile  # Qualität zu niedrig für sinnvolle Filterung
    
    # 1. Padding
    h, w = tile.shape
    pad_h, pad_w = h//4, w//4
    padded = cv.copyMakeBorder(tile, pad_h, pad_h, pad_w, pad_w, cv.BORDER_REFLECT_101)
    
    # 2. FFT
    F = cv.dft(padded, flags=cv.DFT_COMPLEX_OUTPUT)
    planes = cv.split(F)
    power = planes[0]*planes[0] + planes[1]*planes[1]
    
    # 3. Wiener-Filter
    sigma_sq = sigma * sigma
    H = cv.max(0, power - sigma_sq)  # max für threshold, nicht negativ
    cv.divide(H, power + 1e-12, H)   # Wiener-Filter: (S-N)/S
    cv.min(H, 1.0, H)                # Nicht verstärken, nur dämpfen
    
    # 4. Anwenden des Filters
    planes[0] = planes[0] * H
    planes[1] = planes[1] * H
    cv.merge(planes, F)
    
    # 5. Inverse FFT
    filtered = cv.idft(F, flags=cv.DFT_SCALE | cv.DFT_REAL_OUTPUT)
    
    # 6. Crop auf ursprüngliche Größe
    result = filtered[pad_h:pad_h+h, pad_w:pad_w+w]
    
    return result
```

Diese Implementierung ist besonders effektiv bei niedrig-SNR Strukturen, während sie scharfe Details (insbesondere Sterne) gut erhält.

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

### B.8 Registrierungsqualität (NEU)

**Typ:** Multi-Panel-Plot

**Inputs:**
* NCC-Werte vor und nach Registrierung
* Verwendete Registrierungsmethode pro Frame
* Residuen (in Pixel)

**Ziel:**
* Validierung der Registrierungskaskade
* Identifikation problematischer Frames

**Akzeptanz:**
* > 95% der Frames erfolgreich registriert
* Mittleres NCC nach Registrierung > 0.8

---

### B.9 CFA-Transformation-Validierung (NEU)

**Typ:** Farbanalyse

**Inputs:**
* Originales CFA-Muster
* Transformiertes CFA-Muster

**Validierung:**
* Bayer-Phasenerhaltung (keine Vermischung von R/G/B-Positionen)
* Farbsättigung und -tonalität (keine systematischen Farbverschiebungen)

**Akzeptanz:**
* Keine sichtbaren Bayer-Artefakte im Endresultat

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