# Unterschiede zwischen Methodology v3.3.8 und v3.3.9

**Status:** Vergleichsdokument  
**Verglichene Dateien:**  
- `doc/v3/tile_basierte_qualitatsrekonstruktion_methodik_v_3.3.8_en.md`
- `doc/v3/tile_basierte_qualitatsrekonstruktion_methodik_v_3.3.9_en.md`

---

## 1. Kurzfazit

`v3.3.9` ist keine bloße redaktionelle Revision von `v3.3.8`, sondern eine fachliche Schärfung des Kerns. Die wichtigsten Änderungen sind:

1. saubere Trennung von additivem Hintergrund `B_{f,c}` und multiplikativer photometrischer Skalierung `P_{f,c}`
2. Entfernung der tileweisen nichtlinearen Renormalisierung vor OLA aus dem verpflichtenden Kern
3. weiches lokales STAR/STRUCTURE-Blending statt harter Umschaltung
4. support-aware OLA-Semantik mit explizitem Ausschluss canvas-ungültiger bzw. nicht gestützter Pixel
5. massenerhaltende Cluster-Aggregation in Full Mode
6. stärkere normative Definition von BGE, PCC, Validierung und ML-Masken-Integration

---

## 2. Strukturelle Änderungen

### 2.1 Versions- und Zieldefinition

- `v3.3.8` wurde zu `v3.3.9` aktualisiert.
- Der Zielabschnitt wurde inhaltlich neu ausgerichtet.
- Neu hervorgehoben in `v3.3.9`:
  - Trennung von additivem Hintergrund und photometrischer Skalierung
  - Verbot tileweiser nichtlinearer Renormalisierung im verpflichtenden Kern
  - weiches lokales Qualitätsblending
  - massenerhaltende Cluster-Aggregation
  - support-aware OLA mit deterministischer Randabdeckung

### 2.2 Neue bzw. verschobene Unterabschnitte

Neu in `v3.3.9`:

- `4.1.1 Optional Pre-Warp CFA Defect-Pixel Suppression (Feature-Gated)`
- `6.3.10 RBF Surface Mathematical Specification`
- `6.4.4 Optional Post-PCC Isolated Chroma-Speckle Suppression`

Entfernt bzw. ersetzt in `v3.3.9`:

- `5.7.1 Tile Normalization before OLA`
- `5.7.1a Robust Tile-Normalization Guard`
- `5.7.1b Photometric Preservation after OLA`
- der alte RBF-Anhang in `9.2.7` wurde entfernt und nach `6.3.10` verschoben

---

## 3. Änderungen im Rekonstruktionskern

### 3.1 Linearity Semantics

`v3.3.9` ergänzt explizit:

- tileweise datenabhängige Renormalisierung rekonstruierter Pixelwerte vor OLA gehört **nicht** zum verpflichtenden linearen Kern

Das ist eine zentrale methodische Änderung gegenüber `v3.3.8`, wo diese Normalisierung in `5.7.1` noch normativ enthalten war.

### 3.2 Globale Normalisierung

`v3.3.8`:

- leitete die lineare Normierung im Kern aus dem globalen Hintergrundniveau ab
- `I = I_raw / max(B, eps_bg)`

`v3.3.9`:

- führt einen zweistufigen normativen Pfad ein:
  - additive Hintergrundsubtraktion `J = I_raw - B`
  - photometrische Skalierung `I = J / max(P, eps_scale)`
- definiert `photometric_scale()` verbindlich:
  - Ensemble-Sternfluss-Skalierung
  - Belichtungszeit-Verhältnis
  - deterministischer Fallback `P = 1`
- verbietet ausdrücklich, `P` allein aus dem Sky-Background `B` abzuleiten

### 3.3 Adaptive Global Weights

`v3.3.8`:

- optionales Adaptive Weighting über Varianzen der z-normalisierten Metriken

`v3.3.9`:

- erklärt die statischen Gewichte zum verpflichtenden Kern
- erlaubt adaptive Gewichte nur noch als optionale Erweiterung
- verlangt dafür dokumentierte Utility-/Tie-Break-Semantik statt bloßer `Var(z(.))`-Heuristik

### 3.4 Tile Geometry

`v3.3.9` verschärft die Tile-Geometrie:

- explizites `T_hi = floor(min(W,H)/D)`
- deterministischer Compact-Tile-Modus, wenn `T_hi < T_min`
- zusätzliche Guards für `S <= 0`, `O >= T`, `T <= 0`
- Hinweis, dass bestimmte Guards unter gültigen Preconditions nur defensive Sicherungen sind

### 3.5 Lokale Tile-Metriken

`v3.3.8`:

- harte STAR/STRUCTURE-Klassifikation

`v3.3.9`:

- ersetzt die Klassifikation durch `eta_t` als weichen STAR/STRUCTURE-Blending-Faktor
- ergänzt Canvas-Exklusion verbindlich für alle lokalen Metriken
- erweitert die Regularisierung:
  - lokale Konfidenz `U_{f,t,c}`
  - nachbarschaftsgewichtete Affinität `A_{t,u}`
  - `eps_affinity`-Guard
- führt `k_local` als expliziten lokalen Gewichtsskalenfaktor ein

### 3.6 Tile Reconstruction und OLA

Das ist einer der größten Unterschiede zwischen beiden Versionen.

`v3.3.8`:

- definierte eine tileweise Median/MAD-Normalisierung vor OLA
- ergänzte globale photometrische Wiederherstellung nach OLA

`v3.3.9`:

- entfernt diesen gesamten Normierungspfad aus dem verpflichtenden Kern
- definiert stattdessen support-aware OLA:
  - gültige Samples über Finitheit plus validen Canvas-Support
  - `M_{t,c}(p)` als explizite Supportmaske
  - canvas-ungültige Pixel und `|V| = 0` tragen nicht zum OLA-Nenner bei
  - verbindliche Partition-of-Unity-/Boundary-Regeln
- ergänzt Guards für Sigma-Clipping:
  - `N_eff`
  - `D_eff`
  - `min_fraction`-Keep-Floor

### 3.7 State-Based Clustering und Synthetic Frames

`v3.3.8`:

- aktiv nur bei `N >= 200`

`v3.3.9`:

- ersetzt den festen Schwellenwert durch das Modus-Framework:
  - aktiv genau dann, wenn `N >= max(N_red, 50)`
- ergänzt für `synthetic.weighting=tile_weighted`:
  - Nutzung derselben support-aware OLA-Semantik
  - deterministischen Fallback auf `global`, wenn Boundary-Regression plus Cross-Tile-Weight-Disagreement erkannt wird

### 3.8 Final Stacking

`v3.3.8`:

- Cluster wurden rein qualitätsgewichtet aggregiert

`v3.3.9`:

- führt `M_{k,c}` als Cluster-Masse ein
- macht die Aggregation massenerhaltend:
  - `w_{k,c}^{raw} = M_{k,c} * exp(kappa_cluster * Q_{k,c}^{rel})`
- ergänzt Zero-Denominator-Fallbacks sowohl auf Cluster- als auch Endaggregations-Ebene

---

## 4. Änderungen in BGE und PCC

### 4.1 BGE Sampling und Tile Reliability

`v3.3.9` verschärft BGE deutlich:

- canvas-ungültige Pixel müssen explizit ausgeschlossen werden
- `structure_score_t` wird verbindlich als dimensionslose relative Hochpassenergie definiert
- `w_t` wird normativ über
  - `exp(-lambda_structure * structure_score_t) * (1 - masked_fraction_t)`
  definiert
- Stabilität unter globaler Intensitätsreskalierung wird explizit gefordert

### 4.2 Grobes Grid und Surface Fit

`v3.3.9`:

- macht `w_cell = median({w_t})` zum normativen Default
- präzisiert die Coarse-Grid-Semantik
- bindet die in `6.3.2(c)` definierte Gewichtsformel direkt an `6.3.4`

### 4.3 BGE Autotuning

`v3.3.8`:

- Zielgröße `J = E_cv + alpha * E_flat + beta * E_rough`
- Konfigurationsnamen `alpha_flatness`, `beta_roughness`

`v3.3.9`:

- helligkeitsnormalisierte Zielgröße
  - `B_ref = max(abs(median_i b_i), eps_bg)`
  - `J = E_cv / B_ref + alpha_f * E_flat / B_ref^2 + beta_r * E_rough / B_ref`
- neue normative Konfigurationsnamen:
  - `bge.autotune.alpha_f`
  - `bge.autotune.beta_r`
- `autotune.best.objective` ist nun ausdrücklich die normalisierte Zielgröße
- `autotune.best.objective_raw` kommt als zusätzliche Diagnose hinzu

### 4.4 Adaptive Grid und RBF

`v3.3.9`:

- ergänzt die Compact-Tile-Mode-Ausnahme für `G >= 2*T`
- verschiebt die RBF-Spezifikation aus dem ML-Anhang an die fachlich richtige Stelle `6.3.10`
- macht Regularisierung und Canvas-Exklusion im RBF-Fit explizit

### 4.5 PCC

Neu in `v3.3.9`:

- `6.4.4 Optional Post-PCC Isolated Chroma-Speckle Suppression`

Damit ist erstmals normativ beschrieben:

- RGB-only nach PCC
- nur innerhalb gültigen Canvas-Supports
- nur für isolierte Einzelkanal-Chroma-Ausreißer
- mit Struktur-/Luma-Guards

---

## 5. Änderungen in Validierung und Defaults

### 5.1 Validierung

`v3.3.8`:

- 13 minimale normative Tests

`v3.3.9`:

- 24 minimale normative Tests
- neu hinzugekommen sind unter anderem:
  - Trennung von additiver und multiplikativer Normierung
  - Zulässigkeit negativer bzw. null Pixel
  - Sigma-Clipping-Guards
  - support-aware OLA
  - massenerhaltende Cluster-Aggregation
  - STAR-/STRUCTURE-Koeffizientensummen
  - Canvas-Invalid-Exclusion
  - Clustering-Gate gemäß Modus-Framework
  - BGE-Stabilität unter Intensitätsreskalierung
  - normierte BGE-Autotune-Zielgröße

### 5.2 Empfohlene numerische Defaults

`v3.3.9` ergänzt gegenüber `v3.3.8` insbesondere:

- `eps_scale`
- `eps_neff`
- `eps_var`
- `tol_ola`
- `eps_affinity`
- `k_global`
- `k_local`
- `min_fraction`
- `lambda_bge`
- `bge.structure_blur_px`
- `validation.wcs_roundtrip_max_arcsec`
- `validation.pcc_max_condition_number`
- `validation.pcc_max_residual_mag`

Entfernt aus der Default-Liste:

- `eps_median`

---

## 6. Änderungen in Scope und optionalen Erweiterungen

### 6.1 Optionale Erweiterungen

Neu in `v3.3.9`:

- deterministische CFA-Defektpixel-Unterdrückung / Cosmetic Correction
- post-PCC isolated chroma-speckle suppression

### 6.2 ML-Erweiterung

`v3.3.9` ergänzt in `9.2.3` verbindlich:

- wie ML-Masken mit der gültigen Sample-Menge interagieren
- dass Soft-Masken die Mitgliedschaft in `V_{t,c}(p)` nicht ändern
- dass Sigma-Clipping auf Pixelebene mit `Ŵ_{f,t,c}(p)` laufen muss
- dass Canvas-Ungültigkeit Vorrang vor ML-Masken hat

Außerdem:

- der alte RBF-Abschnitt wurde aus dem ML-Teil entfernt
- stattdessen bleibt dort nur noch ein Verweis auf `§6.3.10`

---

## 7. Wichtigste fachliche Nettoeffekte

Im Ergebnis verschiebt `v3.3.9` die Methodik in folgende Richtung:

- weniger implizite nichtlineare Tile-Heuristiken im Rekonstruktionskern
- klarere Photometrie-Semantik
- robustere Rand- und Support-Behandlung
- stärker normierte BGE-/PCC-Integration
- präzisere Validierungs- und Diagnoseanforderungen

Die größten fachlichen Unterschiede sind daher:

1. `v3.3.8` erlaubt bzw. fordert noch eine tileweise Vor-OLA-Normalisierung, `v3.3.9` entfernt sie aus dem verpflichtenden Kern.
2. `v3.3.8` hat eine harte STAR/STRUCTURE-Umschaltung, `v3.3.9` ersetzt sie durch weiches Blending plus confidence-aware Regularisierung.
3. `v3.3.8` aggregiert Cluster qualitätsgewichtet, `v3.3.9` zusätzlich massenerhaltend.
4. `v3.3.9` macht Canvas-Support und OLA-Semantik ausdrücklich zu bindenden mathematischen Bedingungen.

---

## 8. Hinweis zum Charakter dieses Dokuments

Dieses Dokument ist **kein roher Zeilen-Diff**, sondern eine inhaltlich strukturierte Zusammenfassung der Unterschiede zwischen `v3.3.8` und `v3.3.9`.


