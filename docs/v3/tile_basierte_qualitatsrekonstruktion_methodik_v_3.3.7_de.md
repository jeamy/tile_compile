# Tile-basierte Qualitätsrekonstruktion für DSO - Methodik v3.3.7

**Status:** Normative Referenzspezifikation  
**Version:** v3.3.7 (2026-03-15)  
**Gilt für:** `tile_compile.yaml`

---

## 0. Ziel von v3.3.7

Kernziele:

1. mathematische Konsistenz (Notation, Formeln, Randfälle)
2. klare Trennung zwischen **verbindlichem Kern** und **optionalen Erweiterungen**
3. präzise Semantik für
   - Linearität,
   - keine Frame-Selektion,
   - robuste Pixel-Outlier-Behandlung

---

## 1. Prinzipien und Definitionen

### 1.1 Physikalisches Ziel

Aus vollständig registrierten, linearen Kurzbelichtungs-Frames wird ein räumlich und zeitlich optimal gewichtetes Signal rekonstruiert.

Die Methode modelliert zwei orthogonale Qualitätsachsen:

- **global** (Atmosphäre): Transparenz, Himmelshelligkeit, Rauschen
- **lokal** (Tile): Schärfe, Strukturtragfähigkeit, lokales Hintergrundniveau

### 1.2 Keine Frame-Selektion (Invariante)

**Verboten:** Entfernen ganzer Frames auf Basis von Qualitätskriterien.  
**Erlaubt:** Pixelweise Outlier-Verwerfung (Sigma-Clipping), sofern

- sie nur auf Pixelebene wirkt,
- deterministische Parameter verwendet,
- und einen dokumentierten Fallback auf das unveränderte Mittel einschließt.

### 1.3 Linearitätssemantik (präzisiert)

"Strikt linear" bedeutet in v3.3.7:

1. **Die photometrische Signalabbildung** bleibt linear (keine globalen nichtlinearen Tonkurven wie Stretch, Asinh, Log).
2. Lineare Rekonstruktionsschritte (Skalierung, gewichtetes Mittel, Overlap-Add) sind verpflichtend.
3. Robuste/statistische Nichtlinearitäten (MAD, Clipping, Sigma-Clipping, adaptive Gate-Entscheidungen) sind als **Hilfsschritte** erlaubt.

---

## 2. Annahmen und Betriebsmodi

### 2.1 Harte Annahmen (Verletzung -> Abbruch)

- Eingangsdaten sind linear (kein Stretch, keine Tonkurven)
- Einheitliche Belichtungszeit (Toleranz +-5%)
- Kanalweise Rekonstruktionssemantik ab Phase 3 (explizite Trennung in `strict`, CFA-Proxy-äquivalenter Core in `practical` erlaubt)
- Keine qualitätsbasierte Frame-Selektion
- Registrierte Geometrie wird im selben Pixelreferenzsystem ausgedrückt

### 2.2 Weiche Annahmen

| Annahme | Optimal | Minimum | Aktion bei Verletzung |
|---|---:|---:|---|
| Anzahl Frames N | >= 800 | >= 50 | Reduced Mode für 50..199 |
| Registrierungsresiduum | < 0.3 px | < 1.0 px | Warnung bei > 0.5 px |
| Sternelongation | < 0.2 | < 0.4 | Warnung bei > 0.3 |

### 2.3 Reduced Mode (eindeutig)

- **Gültig nur für:** `50 <= N <= 199`
- Schritte 8-9 (Clustering + Synthetic Frames) werden übersprungen
- Das finale Ergebnis ist die Rekonstruktion aus Phase 7

### 2.4 Unterhalb des Minimums

- **N < 50:** kein Reduced Mode
- Standardaktion: kontrollierter Abbruch mit Diagnostik
- Optional nur über explizites `runtime.allow_emergency_mode: true`: Emergency Mode mit Warnstatus

### 2.5 Profilabhängige Kanal-Semantik (verbindlich)

- `strict`: explizite Kanaltrennung ist bis Phase 2 abgeschlossen; Phasen 3-10 laufen kanalweise.
- `practical`: ein CFA-Proxy-Core-Pfad ist erlaubt; explizite RGB-Trennung kann bis zur Channel-Stack-Stufe verschoben werden.
- Für den `practical`-CFA-Proxy-Core-Pfad bleiben alle folgenden Punkte zwingend:
  1. lineares und deterministisches Rekonstruktionsverhalten im gemeinsamen Core,
  2. kanaläquivalente Gewichtungs-/Schätzsemantik (keine versteckte kanalübergreifende Kopplung im Core-Schätzer),
  3. Erhalt der CFA-Phase bei geometrischen Operationen,
  4. explizite RGB-Domäne vor Farbkalibrierungs-Erweiterungen (BGE/PCC), bei unveränderter Canvas-Masken-Exklusionspolitik.

---

## 3. Pipeline-Überblick (normativ)

1. Registrierung und geometrische Harmonisierung
2. Kanaltrennung (explizit oder profilbedingt verzögert über CFA-Proxy-Core)
3. Globale lineare Normalisierung
4. Globale Frame-Metriken und globale Gewichte
5. Tile-Geometrie
6. Lokale Tile-Metriken und lokale Gewichte
7. Tile-Rekonstruktion (Overlap-Add)
8. Zustandsbasiertes Clustering (nur Full Mode)
9. Synthetic Frames (nur Full Mode)
10. Finales lineares Stacking
11. Post-Processing (optional, nicht Teil des Qualitätskerns)

Verbindlicher Kern: 1-10.  
Optional/feature-gated: lokale Denoiser, Sigma-Clipping-Varianten, WCS/PCC.

---

## 4. Registrierung und Kanaltrennung bis Phase 2 (normativ)

Bis einschließlich Phase 2 gilt der CFA-basierte Registrierungs- und Kanaltrennungspfad.  
Ab Phase 3 gilt der gemeinsame Core in profilabhängiger Form:

- `strict`: expliziter per-channel Core,
- `practical`: CFA-Proxy-äquivalenter Core.

### 4.1 CFA-basierter Registrierungspfad

- Registrierung auf einer CFA-Luminanz-Proxy-Darstellung
- CFA-bewusstes Warping über Subplanes (`warp_cfa_mosaic_via_subplanes`)
- anschließende Kanaltrennung (`strict`) oder verzögerte Trennung in der Channel-Stack-Stufe (`practical`, CFA-Proxy-Core-Pfad)

### 4.2 Registrierungskaskade

Pro Frame:

1. konfigurierbare Primärmethode (`triangle_star_matching` als Default)
2. feste Fallback-Reihenfolge:
   - `trail_endpoint_registration`
   - `feature_registration_similarity` (AKAZE)
   - `robust_phase_ecc`
   - `hybrid_phase_ecc`
   - Identity-Fallback mit Warnung

Akzeptanzkriterium pro Versuch:

- `NCC(warped, ref) > NCC(identity, ref) + delta_ncc`
- Default `delta_ncc = 0.01`

### 4.3 CFA-Proxy-Core-Pfad (verbindlich)

- Globale/lokale Metriken und Tile-Rekonstruktion dürfen auf CFA-Proxy-Eingängen statt auf frühen expliziten RGB-Ebenen arbeiten.
- Das ist nur dann konform, wenn die Kanal-Semantik- und Linearitätsbedingungen aus §2.5 erhalten bleiben.
- Explizite RGB-Daten sind weiterhin vor BGE/PCC und für finale RGB-Ausgaben erforderlich.

---

## 5. Gemeinsamer Core ab Phase 3

## 5.1 Notation (verbindlich)

- `f` Frame-Index, `t` Tile-Index, `c` Kanalindex, `p` Pixel
- `I_{f,c}(p)` normalisiertes Eingangsbild pro Frame/Kanal
- `B_{f,c}` globaler Hintergrund (vor der Normalisierung)
- `sigma_{f,c}` globales Rauschen (nach der Normalisierung)
- `E_{f,c}` globale Gradientenenergie (nach der Normalisierung)
- `Q_{f,c}` globaler Qualitätsindex
- `G_{f,c}` globales Gewicht
- `Q_{f,t,c}^{local}` lokaler Qualitätsindex
- `L_{f,t,c}` lokales Gewicht
- `W_{f,t,c}` effektives Gewicht

**Ab hier wird der Kanalindex `c` konsistent verwendet.**

---

## 5.2 Globale lineare Normalisierung (verpflichtend)

Reihenfolge:

1. Hintergrund aus Rohdaten:
   - `B_{f,c} = median(I_{f,c}^{raw})`
2. Lineare Skalierung:
   - `I_{f,c} = I_{f,c}^{raw} / max(B_{f,c}, eps_bg)`
3. Metriken auf normalisierten Daten:
   - `sigma_{f,c}`, `E_{f,c}`

Verboten: globale nichtlineare Tonkurven.

Empfohlener Default:

- `eps_bg = 1e-6`

---

## 5.3 Globale Metriken und Gewichte

### 5.3.1 Robuste Metrik-Normalisierung

Für eine Metrikfolge `x`:

`z(x_i) = (x_i - median(x)) / max(1.4826 * MAD(x), eps_mad)`

mit `eps_mad = 1e-6`.

### 5.3.2 Globaler Qualitätsindex

`Q_{f,c} = alpha*(-z(B_{f,c})) + beta*(-z(sigma_{f,c})) + gamma*z(E_{f,c})`

Constraint: `alpha + beta + gamma = 1`

Defaults:

- `alpha=0.4, beta=0.3, gamma=0.3`

Clamping vor der Exponentialfunktion:

`Q_{f,c}^{clamped} = clip(Q_{f,c}, -3, +3)`

Globales Gewicht:

`G_{f,c} = exp(k_global * Q_{f,c}^{clamped})`

mit `k_global > 0`, Default `k_global=1.0`.

### 5.3.3 Optionale adaptive Gewichtung

Wenn `global_metrics.adaptive_weights=true`:

- Varianzen werden auf robust normalisierten Metriken berechnet:
  - `Var(z(B))`, `Var(z(sigma))`, `Var(z(E))`
- Rohgewichte:
  - `alpha' ~ Var(z(B))`, `beta' ~ Var(z(sigma))`, `gamma' ~ Var(z(E))`
- Jedes Gewicht wird auf [0.1, 0.7] geclippt und anschließend auf Summe 1 renormalisiert
- Fallback auf statische Defaults bei degenerierter Gesamtvarianz

---

## 5.4 Tile-Geometrie

Parameter:

- Bildgröße `W,H`
- Robuste Seeing-Schätzung `F` (FWHM in Pixeln)
- `s = tile.size_factor`
- `T_min = tile.min_size`
- `D = tile.max_divisor`
- `o = tile.overlap_fraction`, `0 <= o <= 0.5`

Formeln:

`T0 = s * F`

**Overlap-Erzwingung (verbindlich):**  
`o_clipped = clip(o, 0, 0.5)`

`T = floor(clip(T0, T_min, floor(min(W,H)/D)))`

`O = floor(o_clipped * T)`

`S = T - O`

Schutzregeln (verbindlich):

1. wenn `F <= 0` -> `F = 3.0`
2. `T_min >= 16`
3. wenn `S <= 0` -> setze `o_clipped=0.25`, berechne `O,S` neu (und halte `o_clipped` in [0,0.5])
4. wenn `min(W,H) < T` -> `T=min(W,H)`, `O=0`

---

## 5.5 Lokale Tile-Metriken

### 5.5.1 Klassifikation

- **STAR-Tile:** `star_count >= tile.star_min_count`
- **STRUCTURE-Tile:** sonst

### 5.5.2 STAR-Tile-Metriken

- `FWHM_{f,t,c}`
- `R_{f,t,c}` (Rundheit)
- `C_{f,t,c}` (Kontrast)

Lokaler Index:

`Q_{f,t,c}^{star} = 0.6*(-z(FWHM)) + 0.2*z(R) + 0.2*z(C)`

### 5.5.3 STRUCTURE-Tile-Metriken

- `(E/sigma)_{f,t,c}`
- `B_{f,t,c}`

Lokaler Index:

`Q_{f,t,c}^{struct} = 0.7*z(E/sigma) - 0.3*z(B)`

### 5.5.4 Räumliche Regularisierung lokaler Scores (verbindlich in v3.3.7)

Zuerst wird der unregularisierte lokale Score berechnet:

`Q_{f,t,c}^{raw} = Q_{f,t,c}^{star|struct}`

Um zu verhindern, dass benachbarte Tiles in inkompatible lokale Regime kippen, wird das lokale Score-Feld vor der Exponential-Gewichtung auf dem Tile-Nachbarschaftsgraphen regularisiert.

Sei `N(t)` die 4-Nachbarschaft des Tiles `t` im Tile-Raster.

Für jeden Frame `f`, jedes Tile `t` und jeden Pass `k`:

`Q_{f,t,c}^{(k+1)} = (1 - lambda_local) * Q_{f,t,c}^{(k)} + lambda_local * mean_{u in N(t)} Q_{f,u,c}^{(k)}`

mit Initialisierung:

`Q_{f,t,c}^{(0)} = Q_{f,t,c}^{raw}`

und finalem regularisiertem Score nach `P` Pässen:

`Q_{f,t,c}^{reg} = Q_{f,t,c}^{(P)}`

Normative Default-Parameter:

- `local_metrics.spatial_regularization.enabled = true`
- `local_metrics.spatial_regularization.lambda = 0.35`
- `local_metrics.spatial_regularization.passes = 1`

Verbindliche Randbedingungen:

1. Nur gültige/common Tiles dürfen teilnehmen.
2. Die Regularisierung ist frame-lokal und darf keine verschiedenen Frames koppeln.
3. Tiles ohne gültige Nachbarn behalten `Q_{f,t,c}^{reg} = Q_{f,t,c}^{raw}`.
4. Die Regularisierung wirkt nur auf lokale Qualitätsscores, niemals direkt auf Pixelwerte.

### 5.5.5 Lokales Gewicht

`Q_{f,t,c}^{local} = clip(Q_{f,t,c}^{reg}, -3, +3)`

`L_{f,t,c} = exp(Q_{f,t,c}^{local})`

---

## 5.6 Effektives Gewicht

`W_{f,t,c} = G_{f,c} * L_{f,t,c}`

Semantik:

- `G`: globale atmosphärische Qualität
- `L`: lokale Struktur-/Schärfequalität

---

## 5.7 Tile-Rekonstruktion (konsolidiert)

Für Pixel `p` im Tile `t`:

`D_{t,c} = sum_f W_{f,t,c}`

Wenn `D_{t,c} >= eps_weight`:

`R_{t,c}(p) = sum_f W_{f,t,c} * I_{f,c}(p) / D_{t,c}`

Wenn `D_{t,c} < eps_weight`:

`R_{t,c}(p) = (1/N) * sum_f I_{f,c}(p)`

und `fallback_used=true` für dieses Tile.

Default `eps_weight = 1e-6`.

### 5.7.1 Tile-Normalisierung vor OLA (verbindlich)

Für ein rekonstruiertes Tile `R_{t,c}`:

1. `bg_t = median(R_{t,c})`
2. `X_t = R_{t,c} - bg_t`
3. `m_t = median(abs(X_t))`
4. wenn `m_t >= eps_median`: `Y_t = X_t / m_t`, sonst `Y_t = X_t`

Default `eps_median = 1e-6`.

#### 5.7.1a Robuster Guard für die Tile-Normalisierung (verbindlich)

Implementierungen müssen pathologische Verstärkung verhindern, wenn `m_t` aus zu wenigen gültigen Pixeln geschätzt wird oder weit unter die datensatzweite Tile-Skala kollabiert.

Erforderlicher deterministischer Guard:

1. `bg_t` und `m_t` nur aus endlichen, strikt positiven Tile-Samples innerhalb des gültigen Rekonstruktionssupports schätzen
2. minimale Anzahl gültiger Samples pro Tile verlangen:
   - `n_min = max(64, ceil(0.05 * N_t))`
3. robuste globale Referenzen über gültige Tiles berechnen:
   - `bg_global = median_t(bg_t)`
   - `m_global = median_t(m_t)`
4. wenn ein Tile `n_min` nicht erfüllt, werden seine lokalen Normalisierungsmetadaten durch die globalen Referenzen ersetzt
5. gültige lokale Skalen clampen auf
   - `m_t in [0.5 * m_global, 2.0 * m_global]`

Dieser Guard ist Teil des linearen affinen Normalisierungspfads. Er führt keine nichtlineare Tonkurve ein, sondern verhindert nur, dass instabile tileweise Gain-Explosionen den OLA-Eingang dominieren.

#### 5.7.1b Photometrische Erhaltung nach OLA (empfohlen)

Die Normalisierung `Y_t = (R_{t,c} - bg_t)/m_t` egalisiert lokale Struktur, kann aber die absolute photometrische Skala verändern, wenn sie unkorrigiert bleibt.  
Um eine konsistente globale affine Flussskala zu erhalten, sollen pro-Tile-Metadaten während der Rekonstruktion gesammelt und nach OLA eine globale Skala/ein globaler Offset restauriert werden:

- Pro Tile (bereits berechnet): `bg_t`, `m_t`
- Globale Restaurationsfaktoren (robust):
  - `m_global = median_t(m_t)`
  - `bg_global = median_t(bg_t)`

Nachdem OLA `I_rec` erzeugt hat, restauriere:

`I_final = I_rec * m_global + bg_global`

Dadurch bleibt der Core linear in den Pixelwerten (die Restaurierung ist eine globale affine Transformation), während systematischer tileweiser photometrischer Drift vermieden wird.

### 5.7.2 Fensterung und Overlap-Add

2D-Fenster separabel mit diskreter Hann-Funktion:

`hann(i,N) = 0.5*(1 - cos(2*pi*i/(N-1)))`, `i=0..N-1`

Sonderfall: `N=1 -> hann=1`.

`w(x,y) = hann(x,W_t) * hann(y,H_t)`

Rekonstruktionsbild:

- Zählerakkumulator: `A`
- Fenstersummen-Akkumulator: `S`

`A += w * Y_t`, `S += w`, Ergebnis `I_rec = A / max(S, eps_weight)`

Optional kann nach OLA ein globaler robuster Tile-Hintergrund-Offset restauriert werden (Median über `bg_t`).

### 5.7.3 Boundary-Diagnostik (empfohlen, nicht invasiv)

Um sichtbare Tile-Grenzen zu diagnostizieren, ohne das Rekonstruktionsergebnis zu verändern, dürfen Implementierungen benachbarte Tiles auf dem tatsächlichen OLA-Eingang `Y_t` auswerten.

Empfohlene Praxis ist, diese Diagnostik zweimal zu emittieren:

- einmal auf den rohen rekonstruierten Tiles vor der optionalen per-Tile-Normalisierung
- einmal auf dem normalisierten OLA-Eingang `Y_t`

Für jedes benachbarte Tile-Paar `(a,b)` mit Overlap-Domäne `Omega_ab`, definiere:

`Delta_ab(p) = Y_b(p) - Y_a(p)`, für `p in Omega_ab`

Es dürfen nur Samples innerhalb der gemeinsamen canvas-gültigen Domäne beitragen. Maskierte Canvas-Zonen müssen ausgeschlossen und dürfen nicht als gültige Nullwerte behandelt werden.

Empfohlene Paar-Diagnostik:

- `mean_abs_diff_ab = mean_p |Delta_ab(p)|`
- `p95_abs_diff_ab = p95_p |Delta_ab(p)|`
- `mean_signed_diff_ab = mean_p Delta_ab(p)`
- `n_ab = |Omega_ab|` gültige endliche Overlap-Samples

Zusätzlich dürfen Implementierungen per-Pair-Differenzen zusammenfassen in:

- gültigem Frame-Support,
- Post-Rekonstruktions-Hintergrundmetriken,
- Post-Rekonstruktions-SNR-Proxys,
- Post-Rekonstruktions-Mittelkorrelations-Proxys,
- und Fallback-Mismatch-Flags.

Verbindliche Semantik:

1. Diese Diagnostik muss **read-only** sein und darf `Y_t`, `A`, `S` oder das finale OLA-Ergebnis nicht verändern.
2. Sie darf als Laufzeit-Artefakt für Analyse und Regressionstests emittiert werden.
3. Weil sie nicht in den Schätzer zurückwirkt, verändert sie **nicht** die Linearitätssemantik des Rekonstruktionskerns.

---

## 5.8 Optionale lokale Denoiser (explizit optional)

Diese Schritte sind **nicht Teil des verpflichtenden mathematischen Kerns**, aber zulässige Erweiterungen.

### 5.8.1 Soft-Threshold-High-Pass

- Hintergrund per Box-Blur
- Residuum
- `tau = alpha_d * sigma_tile`
- Soft-Shrinkage
- Rekonstruktion

### 5.8.2 Wiener im Frequenzbereich

- Reflection Padding
- FFT
- Wiener-Transferfunktion
- IFFT und Crop

Nur anwenden, wenn die Gate-Bedingungen erfüllt sind (SNR/Qualität/Tile-Typ).

---

## 5.9 Zustandsbasiertes Clustering (Full Mode)

Aktiv nur für `N >= 200`.

Zustandsvektor pro Frame/Kanal (kanalweise oder kanalaggregiert, konfigurierbar):

`v_f = (G_{f,*}, mean_t(Q_{f,t,*}^{local}), var_t(Q_{f,t,*}^{local}), B_{f,*}, sigma_{f,*})`

Anzahl der Cluster:

`K = clip(floor(N/10), K_min, K_max)`

Defaults: `K_min=5`, `K_max=30`.

---

## 5.10 Synthetic Frames

### 5.10.1 Default (global)

Für Cluster `k`:

`S_{k,c} = sum_{f in k} G_{f,c} * I_{f,c} / sum_{f in k} G_{f,c}`

### 5.10.2 Optional (`tile_weighted`)

Wenn `synthetic.weighting=tile_weighted`:

- rekonstruiere pro Tile/Kanal mit `W_{f,t,c}`
- füge via OLA zu `S_{k,c}` zusammen

### 5.10.3 Semantik von Phase 7 vs. 9

- Full Mode mit `global`: Phase 7 liefert primär lokales Qualitätsmodell/Diagnostik; das finale Produkt entsteht aus Phase 9+10.
- Full Mode mit `tile_weighted`: lokale Tile-Qualität wird explizit in Synthetic Frames propagiert.
- Reduced Mode: die Ausgabe aus Phase 7 ist das direkte Endprodukt.

---

## 5.11 Finales lineares Stacking

### 5.11.1 Definition der Cluster-Qualität (verbindlich)

Für jeden Cluster `k` definiere einen robusten Cluster-Qualitätsindex:

`Q_k = median_{f in k}(Q_{f,c}^{clamped})`

wobei `Q_{f,c}^{clamped}` der globale Frame-Qualitätsindex ist, der bereits auf `[-3,+3]` begrenzt wurde.

### 5.11.2 Qualitätsgewichtete Cluster-Aggregation (verbindlich)

Cluster werden mittels exponentieller Qualitätsgewichtung aggregiert:

`w_k = exp(kappa_cluster * Q_k)`

mit:

- `kappa_cluster > 0` (empfohlener Default: `kappa_cluster = 1.0`)
- `Q_k` bereits geclippt auf `[-3,+3]`

Optionaler Stabilitäts-Cap (empfohlen):

`w_k = min(w_k, r_cap * median_j(w_j))`

mit empfohlenem `r_cap` in `[10, 50]`.

Finales Ergebnis pro Kanal:

`R_c = sum_k (w_k * S_{k,c}) / sum_k w_k`

### 5.11.3 Semantik

- Bessere atmosphärische Zustände (höheres `Q_k`) erhalten stärkeren Einfluss.
- Alle Cluster bleiben enthalten (keine harte Zustandsselektion).
- Der Schätzer bleibt linear in den Synthetic Frames.
- Dominanz wird über optionales Weight-Capping begrenzt.

---

## 6. Post-Processing (nicht Teil des verpflichtenden Kerns)

### 6.1 RGB/LRGB-Kombination

Austauschbar, außerhalb des Rekonstruktionskerns.

### 6.2 Astrometrie (WCS)

Zulässiger Downstream-Schritt, ohne Rückkopplung in die Core-Gewichte.

### 6.3 Hintergrundgradienten-Extraktion vor PCC (BGE) (optional, empfohlen)

Hintergrundgradienten (z.B. künstliche Lichtverschmutzung, Mondlicht, Airglow) können die Photometric Color Calibration (PCC) verzerren, besonders wenn Gradienten spektral nicht gleichmäßig über die Kanäle verteilt sind.  
Zur Abmilderung darf **vor PCC** ein additiver Background Gradient Extraction (BGE)-Schritt angewendet werden.

#### 6.3.1 Prinzip

Für jeden Kanal `c` wird ein glattes großskaliges Hintergrundmodell `B_c(x,y)` geschätzt und subtrahiert:

`I'_c(x,y) = I_c(x,y) - B_c(x,y)`

BGE muss:

- strikt additiv,
- kanalweise,
- unabhängig von der Frame-Gewichtungslogik
- und frei von nichtlinearen Tontransformationen sein.

#### 6.3.2 Tile-getriebenes Sampling-Gitter (verbindlich)

Die Rekonstruktions-Tiles werden als Hintergrund-Sampling-Einheiten wiederverwendet. Ziel ist, **objektfreie** Hintergrundsamples pro Tile zu erhalten.

##### (a) Definition der Hintergrundmaske (verbindlich)

Für jedes Tile `t` und jeden Kanal `c` definiere eine Binärmaske `M_bg`, die Pixel markiert, die als Hintergrundsamples zugelassen sind. `M_bg` muss ausschließen:

1. **Sterne:** Pixel in `M_star` (aus Sternerkennung oder Segmentierung), optional dilatiert um `mask.star_dilate_px` (empfohlener Default: 2–6 px).
2. **Hochstruktur-Pixel:** Pixel mit `structure_metric(p) > structure_thresh`, wobei `structure_metric` aus lokalen Gradienten abgeleitet wird (z.B. High-Pass-Energie) und `structure_thresh` konfigurierbar ist.
3. **Gesättigte Pixel:** Pixel mit `I >= sat_level` und optionaler Dilatationsmarge `mask.sat_dilate_px`.
4. **Optionale Objektmaske:** falls verfügbar (Nebel-/Galaxienmaske), ausschließen, um Bias in Feldern mit ausgedehnten Objekten zu verhindern.

Falls keine Sternerkennung verfügbar ist, darf `M_star` deterministisch über Thresholding einer Bandpass-/DoG-Antwort plus Dilatation approximiert werden.

##### (b) Robustes Tile-Hintergrundsample (verbindlich, konfigurierbar)

Berechne pro Tile ein robustes Hintergrundsample über ein konfigurierbares Quantil:

`b_{t,c} = quantile_q(R_{t,c}[M_bg])`

mit:

- `q = bge.sample_quantile` in `(0, 0.5]`
- **Default:** `q = 0.20` (20%-Quantil)
- der Median ergibt sich durch `q = 0.50`

Begründung: das niedrigere Quantil ist resistenter gegen schwache Objektkontamination und unvollständige Masken, während der Median in spärlichen Feldern mit starker Maskierung akzeptabel ist.

Das Sample wird mit dem Tile-Zentrum `(x_t, y_t)` verknüpft.

##### (c) Tile-Zuverlässigkeitsgewicht (optional, empfohlen)

Tiles dürfen für das spätere Fitting ein Zuverlässigkeitsgewicht erhalten:

`w_t = exp(-lambda * structure_score_t) * (1 - masked_fraction_t)`

wobei `structure_score_t` aus `E/sigma` oder ähnlichen lokalen Strukturmetriken berechnet wird und `masked_fraction_t` der ausgeschlossene Pixelanteil im Tile ist.

#### 6.3.3 Aggregation auf ein grobes Gitter (verbindlich)

Um Overfitting kleiner Strukturen zu vermeiden, werden Tile-Samples vor dem Surface-Fit auf ein **gröberes** Gitter aggregiert.

##### (a) Gitterdefinition

Gegeben ein Gitterabstand `G` (siehe 6.3.9), definiere achsenparallele Gitterzellen über der Bildebene. Jede Zelle ist ein Rechteck der Größe `G x G`.

##### (b) Zuordnung von Tiles zu Gitterzellen (verbindlich)

Jedes Tile-Sample `(x_t, y_t, b_{t,c}, w_t)` wird über Integer-Binning seines Zentrums genau einer Gitterzelle zugeordnet:

`cell_x = floor(x_t / G)`  
`cell_y = floor(y_t / G)`

(Alle Tiles, deren Zentren in dieselbe `G x G`-Zelle fallen, gehören zu dieser Zelle.)

##### (c) Aggregation pro Zelle (verbindlich)

Für jede Zelle und jeden Kanal `c` aggregiere alle zugeordneten Tile-Samples:

- Wert: `b_cell = median({b_{t,c}})` (robust)
- Gewicht: `w_cell = median({w_t})` (oder Summe; Implementierungswahl, muss dokumentiert werden)

##### (d) Unzureichende Samples (verbindlich)

Eine Gitterzelle gilt als **unzureichend**, wenn sie weniger als

`n_cell < bge.min_tiles_per_cell`

enthält.

Empfohlener Default: `bge.min_tiles_per_cell = 3`.

Unzureichende Zellen müssen deterministisch behandelt werden durch:

1. **Discard (Default):** Zelle vom Fit ausschließen, oder
2. **Nearest-neighbor fill:** `(b_cell, w_cell)` durch die nächste ausreichende Zelle ersetzen, oder
3. **Radius expansion:** deterministisch Tiles aus Nachbarzellen innerhalb eines Radius `r = k*G` einbeziehen, bis `n_cell >= min_tiles_per_cell`.

Die gewählte Strategie muss konfigurierbar und in der Diagnostik festgehalten sein.

#### 6.3.4 Surface Fitting

Fitte pro Kanal eine glatte Hintergrundfläche über:

- robustes 2D-Polynom (Ordnung 2–3 empfohlen), oder
- Thin-plate Spline, oder
- bikubischen Spline mit robustem Loss, oder
- Radial Basis Function (RBF) mit Glättung (nur mit expliziter Regularisierung empfohlen), oder
- foreground-aware modeled-mask mesh sky surface (`modeled_mask_mesh`) für Szenen mit großen diffusen Vordergrundstrukturen.

Optionale Gewichte:

`w_t = exp(-lambda * structure_score_t)`

Robusten Loss (Huber/Tukey) verwenden.

#### 6.3.5 Subtraktion

`I'_c(x,y) = I_c(x,y) - B_c(x,y)`

Keine multiplikative Korrektur erlaubt.

#### 6.3.6 Validierungsanforderungen

Wenn BGE aktiviert ist:

1. Hintergrund-RMS muss sinken oder stabil bleiben.
2. Keine künstliche Krümmung über Tile-Grenzen.
3. Sternflux-Verhältnisse müssen innerhalb der Toleranz stabil bleiben.
4. PCC-Residuen müssen sich gegenüber der No-BGE-Baseline verbessern oder stabil bleiben.

BGE darf die Core-Gewichte (`G`, `L`, `W`) nicht verändern.

#### 6.3.7 Auto-getuntes BGE (optional, konservativ) (verbindlich wenn aktiviert)

Diese optionale Erweiterung erlaubt deterministisches **test–adjust–test**-Tuning von BGE-Parametern, um die Robustheit unter wechselnden Gradientenbedingungen (Lichtverschmutzung, Mondgradienten, Airglow) zu verbessern. Der Rekonstruktionskern bleibt unverändert; BGE bleibt strikt additiv und downstream.

##### 6.3.7.1 Zielgröße (verbindlich)

Für einen gegebenen Kanal definiere eine deterministische Zielgröße:

`J = E_cv + alpha * E_flat + beta * E_rough`

mit:

- `E_cv`: Holdout-RMS der Hintergrundsample-Residuen auf einem deterministischen Validierungssplit,
- `E_flat`: großskalige Gradientenenergie des gefitteten Hintergrundmodells,
- `E_rough`: Energie der zweiten Ableitung des Modells (bestraft overfitte Welligkeit).

Alle Terme müssen deterministisch aus demselben Grid-Cell-Set berechnet werden.

##### 6.3.7.2 Deterministischer Holdout-Split (verbindlich)

Grid-Zellen müssen nach `(cell_y, cell_x)` sortiert und deterministisch aufgeteilt werden, indem jede k-te Zelle als Validierung gewählt wird, wobei `k = round(1/holdout_fraction)`.

`holdout_fraction` muss vor der Split-Generierung auf `[0.05, 0.50]` geclippt werden.

##### 6.3.7.3 Kandidatensuche (konservative Defaults)

Wenn aktiviert, müssen Implementierungen eine begrenzte Menge an Parameterkandidaten auswerten (harte Obergrenze `max_evals`) und den Kandidaten mit minimalem `J` nach deterministischen Tie-Break-Regeln auswählen (bevorzuge geringere Rauigkeit, dann gröberes effektives Modell).

Konservative Kandidatenfamilien (implementierungsabhängig, aber zu dokumentieren):

- `sample_quantile`: `{q0, 0.35, 0.50}`
- `structure_thresh_percentile`: `{p0, 0.85}`
- `rbf_mu_factor`: `{m0, 1.4}`
- `rbf_lambda`: darf intern dennoch eine Glättungspräferenz anwenden (glättestes akzeptiertes `lambda` wählen)

Der Gitterabstand `G` soll im konservativen Modus unverändert bleiben, sofern keine explizit nicht-konservative Strategie aktiviert ist.

`max_evals` ist eine harte Obergrenze ausgewerteter Kandidaten und muss `>= 1` sein.

##### 6.3.7.4 Konfigurations-Hooks (normative Namen)

- `bge.autotune.enabled: true|false`
- `bge.autotune.max_evals`
- `bge.autotune.holdout_fraction`
- `bge.autotune.alpha_flatness`
- `bge.autotune.beta_roughness`
- `bge.autotune.strategy: conservative|extended`

Wenn `enabled=true`, müssen der gewählte Parametersatz und die Zielgrößenkomponenten in der Diagnostik enthalten sein.

Minimale Diagnosefelder (verbindlich):

- `autotune.enabled`
- `autotune.strategy`
- `autotune.max_evals`
- `autotune.evals_performed`
- `autotune.best.sample_quantile`
- `autotune.best.structure_thresh_percentile`
- `autotune.best.rbf_mu_factor`
- `autotune.best.objective`
- `autotune.best.cv_rms`
- `autotune.best.flatness`
- `autotune.best.roughness`
- `autotune.fallback_used`

##### 6.3.7.5 Robustheits- und Fallback-Semantik (verbindlich)

Auto-Tuning muss fail-safe und deterministisch sein:

1. Wenn ein Kandidaten-Fit nicht genügend gültige Zellen/Metriken erzeugen kann, wird dieser Kandidat als fehlgeschlagen markiert.
2. Wenn kein Kandidat erfolgreich ist, muss die Implementierung unverändert auf die Nutzer-/Basis-BGE-Konfiguration zurückfallen.
3. Tie-Break bei gleichen Zielwerten muss deterministisch sein (bevorzuge geringere Rauigkeit, dann gröberes effektives Modell).
4. Auto-Tuning darf die Core-Rekonstruktionsgewichte (`G`, `L`, `W`) nicht verändern und bleibt strikt additiv.

### 6.3.8 Mathematisches Surface-Modell (verbindlich)

Seien die Hintergrundsamples definiert als:

`(x_i, y_i, b_i, w_i)`  für i = 1..M

wobei:

- `(x_i, y_i)` die Mittelpunkte der Gitterzellen sind,
- `b_i` die robuste Hintergrundschätzung ist,
- `w_i` ein optionales Zuverlässigkeitsgewicht ist.

Eine robuste Polynomfläche der Ordnung d (empfohlen d = 2 oder 3) ist definiert als:

`B_c(x,y) = sum_{m+n <= d} a_{mn} x^m y^n`

Die Koeffizienten `a_{mn}` werden bestimmt durch Minimierung von:

`argmin_a sum_i w_i * rho( b_i - B_c(x_i,y_i) )`

wobei `rho` eine robuste Loss-Funktion ist, z.B.:

Huber-Loss:

`rho(r) = 0.5 r^2           if |r| <= delta`
`rho(r) = delta(|r| - 0.5 delta)  otherwise`

oder Tukey-Biweight-Loss.

Der Fit muss über Iteratively Reweighted Least Squares (IRLS) oder eine äquivalente deterministische robuste Optimierung gelöst werden.

Thin-plate-Spline-Alternative:

`B_c = argmin_B sum_i w_i (b_i - B(x_i,y_i))^2 + lambda * integral |D^2 B|^2 dx dy`

mit Regularisierungsparameter `lambda`, der die Glätte steuert.

Nur großskalige (niederfrequente) Komponenten sind erlaubt; Overfitting ist verboten.

#### 6.3.9 Adaptive Gitterdefinition (verbindlich)

Der Gitterabstand `G` muss mit den Bilddimensionen skalieren, um Under- oder Overfitting zu vermeiden.

Definiere:

`G = clip( max(2*T, min(W,H)/N_g), G_min, G_max )`

Empfohlene Defaults:

- `N_g = 32` (Ziel-Gitterauflösung über die kleinste Bildachse)
- `G_min = 64 px`
- `G_max = min(W,H)/4`

Damit wird sichergestellt:

- das Hintergrundmodell erfasst nur großskalige Gradienten,
- die Gitterdichte passt sich an die Sensorauflösung an,
- kleine Bilder werden nicht überparametrisiert,
- große Mosaike behalten ausreichende räumliche Abtastung.

Implementierungen müssen garantieren, dass die Gitterauflösung gröber als die Tile-Auflösung ist (`G >= 2*T`).

### 6.4 PCC

Diese Spezifikation empfiehlt, bei vorhandenen räumlichen Hintergrundgradienten BGE vor PCC anzuwenden.

#### 6.4.1 Lokales Hintergrundmodell im Sky-Annulus (verbindlich)

Die PCC-Sternphotometrie muss einen lokalen Hintergrund subtrahieren, der im Sky-Annulus geschätzt wird. Um Gradienten-Bias zu reduzieren, darf das Hintergrundmodell sein:

- `median`: konstanter Median über den Annulus (Legacy), oder
- `plane`: robuster Ebenenfit `bg(dx,dy)=a + b*dx + c*dy` über die Annulus-Pixel (empfohlen bei Gradienten).

Wenn der `plane`-Fit fehlschlägt, muss deterministisch auf `median` zurückgefallen werden.

#### 6.4.2 FWHM-adaptive Radien (optional, empfohlen)

Wenn eine globale Seeing-Schätzung `FWHM` verfügbar ist, dürfen PCC-Radien automatisch gesetzt werden:

- `r_ap = max(min_aperture_px, aperture_fwhm_mult * FWHM)`
- `r_in = max(r_ap + 1, annulus_inner_fwhm_mult * FWHM)`
- `r_out = max(r_in + 2, annulus_outer_fwhm_mult * FWHM)`

Wenn `FWHM <= 0` oder nicht verfügbar ist, muss deterministisch auf `FWHM = 0` zurückgefallen werden, was ergibt:

- `r_ap = min_aperture_px`
- `r_in = max(r_ap + 1, annulus_inner_fwhm_mult * 0) = r_ap + 1`
- `r_out = max(r_in + 2, annulus_outer_fwhm_mult * 0) = r_in + 2`

Empfohlene konservative Defaults:

- `aperture_fwhm_mult = 1.8`
- `annulus_inner_fwhm_mult = 3.0`
- `annulus_outer_fwhm_mult = 5.0`

Diese Änderungen müssen deterministisch bleiben.

#### 6.4.3 Konfigurations-Hooks (normative Namen)

- `pcc.background_model: median|plane`
- `pcc.radii_mode: fixed|auto_fwhm`
- `pcc.aperture_fwhm_mult`
- `pcc.annulus_inner_fwhm_mult`
- `pcc.annulus_outer_fwhm_mult`
- `pcc.min_aperture_px`

Zulässiger Downstream-Schritt, angewendet auf lineare Daten.

---

## 7. Validierung und Abbruch

## 7.1 Erfolgskriterien

- FWHM-Verbesserung gegenüber dem Referenz-Stack gemäß `validation.min_fwhm_improvement_percent`
- Hintergrund-RMS nicht schlechter als der Referenzwert
- Keine systematischen Tile-Seams
- Stabile Gewichtsverteilungen

## 7.2 Abbruchkriterien

- Datenintegrität verletzt (nichtlinear, unlesbar, inkonsistent)
- Registrierungsfehler über große Teile des Datensatzes
- Numerische Instabilität trotz Fallbacks

## 7.3 Minimale Tests (normativ)

1. `alpha+beta+gamma=1`
2. Clamping vor `exp`
3. Tile-Monotonie in `F`
4. Overlap-Konsistenz (`0<=o<=0.5`, explizit `o_clipped=clip(o,0,0.5)`, ganzzahlige O,S)
5. Low-Weight-Fallback ohne NaN/Inf
6. keine Kanal-Kopplung
7. keine qualitätsbasierte Frame-Selektion
8. deterministische Reproduzierbarkeit
9. Registrierungskaskade inklusive Identity-Fallback
10. Erhalt der CFA-Phase
11. qualitätsgewichtete Cluster-Aggregation (`exp(kappa_cluster * Q_k)`) mit optionalem Dominance-Cap
12. WCS-Round-Trip-Fehler unterhalb des Schwellenwerts
13. PCC-Stabilität: positive Determinante, begrenzte Konditionszahl, Residuen unterhalb des Schwellenwerts

Hinweis: Der alte PCC-Test "kein negatives Matrixelement" ist seit v3.3+ **nicht mehr** als hartes Kriterium erforderlich.

---

## 8. Empfohlene numerische Defaults

- `eps_bg = 1e-6`
- `eps_mad = 1e-6`
- `eps_weight = 1e-6`
- `eps_median = 1e-6`
- `delta_ncc = 0.01`
- `Q`-Clamp global/lokal: `[-3, +3]`

---

## 9. Geltungsbereich: verpflichtender Kern vs. Erweiterung

### Verpflichtender Kern

- CFA-basierter Registrierungspfad bis zur expliziten oder verzögerten (profilabhängigen) Kanaltrennung
- globale Normalisierung
- globale/lokale Metriken und Gewichte
- Tile-Rekonstruktion einschließlich konsolidierter Fallbacks
- Clustering/Synthese/Final-Stack abhängig vom Betriebsmodus

### Optionale Erweiterung

- Soft-Threshold / Wiener
- alternative Sigma-Clipping-Strategien
- WCS/PCC
- spezialisierte Performance-Backends (GPU, Queue-Worker)

### 9.1 Praktische Konfigurationsprofile (`tile_compile_cpp`)

Für den operativen Einsatz stehen vollständige Referenzkonfigurationen bereit:

- `tile_compile_cpp/examples/tile_compile.full_mode.example.yaml`
- `tile_compile_cpp/examples/tile_compile.reduced_mode.example.yaml`
- `tile_compile_cpp/examples/tile_compile.emergency_mode.example.yaml`
- `tile_compile_cpp/examples/tile_compile.smart_telescope_dwarf_seestar.example.yaml`

Alle Profile enthalten **alle verfügbaren Konfigurationsoptionen** mit Inline-Kommentaren.  
Hinweis zur Profil-Semantik:

- `strict`: verlangt explizite Kanaltrennung bis Phase 2 und einen per-channel Shared Core.
- `practical`: erlaubt den CFA-Proxy-Core-Pfad aus §2.5/§4.3.

Vorgehen:

1. passendes Profil kopieren,
2. `run_dir`, `input.pattern` und Sensordaten (`image_width/height`, `bayer_pattern`) anpassen,
3. Runner mit dieser Datei starten.

---

## 9.2 Strikte ML-Optimierungserweiterung (optional, nicht Core)

Diese Erweiterung führt Machine-Learning-(ML)-Module **nur** ein, um die Schätzung von Gewichten und Zustandsdeskriptoren zu verbessern, während die verpflichtenden Core-Invarianten erhalten bleiben.

### 9.2.1 Verbindliche Invarianten (harte Constraints)

1. **Keine Frame-Selektion:** Ganze Frames dürfen nicht auf Basis von Qualität entfernt werden (unverändert zu v3.2.x).
2. **Strikte photometrische Linearität des Rekonstruktionskerns:** Die finale Rekonstruktion muss ein gewichteter linearer Schätzer über Eingangsframes (und/oder Synthetic Frames) bleiben, also von der Form

   `R(p) = sum_i w_i(p) * X_i(p) / sum_i w_i(p)`

   mit `w_i(p) >= 0` und deterministischen Fallbacks.
3. **Determinismus:** ML-Inferenz muss deterministisch sein (feste Modellgewichte, festes Preprocessing, feste Seeds, wo anwendbar).
4. **Keine halluzinierten Inhalte:** ML-Module dürfen keine neuen räumlichen Strukturen erzeugen. ML-Ausgaben sind auf **Gewichte, Masken, Metriken und Zustandslabels** beschränkt.
5. **Kanaltrennung bleibt erhalten:** ML-Module müssen per Kanal oder auf explizit definierten kanalaggregierten Features arbeiten; keine implizite kanalübergreifende Kopplung im Core-Schätzer.

### 9.2.2 Erlaubte ML-Ausgaben (streng)

ML darf folgende Größen ausgeben, sofern sie deterministisch und begrenzt sind:

- globaler Qualitätswert pro Frame/Kanal: `Q̂_{f,c}` (dimensionslos, auf `[-3,+3]` gemappt/geclippt)
- globales Gewicht pro Frame/Kanal: `Ĝ_{f,c} = exp(k_global * Q̂_{f,c})`
- lokaler Tile-Qualitätswert: `q̂_{f,t,c}` (dimensionslos, geclippt auf `[-3,+3]`)
- lokales Tile-Gewicht: `L̂_{f,t,c} = exp(q̂_{f,t,c})`
- Pixel-Zuverlässigkeitsmaske (soft, keine harte Verwerfung): `M̂_{f,t,c}(p) in [m_min, 1]` mit empfohlenem `m_min = 0.05`
- Zustandsdeskriptor für Clustering (pro Frame): `v̂_f` (Feature-Vektor)
- Zustandslabels (Cluster) und/oder Übergangswahrscheinlichkeiten (HMM), ausschließlich zur Bildung von Synthetic Frames

Verbotene ML-Ausgaben:

- direkte Vorhersage rekonstruierter Pixelintensitäten (End-to-End-Bilderzeugung)
- Super-Resolution oder Inpainting, die räumliche Details erzeugen, die nicht durch die Eingabe gestützt sind
- jegliches stochastisches Sampling zur Inferenzzeit

### 9.2.3 ML-getriebenes effektives Gewicht (verbindlich)

Wenn ML-Module aktiviert sind, darf das effektive Gewicht auf Pixelebene erweitert werden:

`Ŵ_{f,t,c}(p) = Ĝ_{f,c} * L̂_{f,t,c} * M̂_{f,t,c}(p)`

Die Tile-Rekonstruktion bleibt ein gewichtetes Mittel:

`R_{t,c}(p) = sum_f Ŵ_{f,t,c}(p) * I_{f,c}(p) / sum_f Ŵ_{f,t,c}(p)`

Die Fallback-Regel bleibt unverändert: wenn der Nenner < `eps_weight`, fällt die Implementierung auf das ungewichtete Mittel zurück.

### 9.2.4 Empfohlene Lernparadigmen (nicht verbindliche Leitlinien)

Weil Ground Truth typischerweise nicht verfügbar ist, sollte priorisiert werden:

- **Self-supervised Learning:** Konsistenz über zufällige Frame-Subsets, Noise2Self/Noise2Void-artige Ziele für Masken/Denoise-Proxys (Hinweis: Denoising darf im strikten Modus weiterhin nur Masken/Gewichte, nicht Pixel, ausgeben).
- **Weak Supervision über Proxys:** Gewichte so optimieren, dass deterministische Metriken (FWHM, Elliptizität, Hintergrund-RMS, Seam-Score) auf Validierungssätzen verbessert werden.
- **Uncertainty-aware Models:** Unsicherheit ausgeben, um überkonfidentes Downweighting zu vermeiden; Unsicherheit muss in begrenzte Masken/Gewichte gemappt werden.

### 9.2.5 Modelle, die zur strikten Output-Constraint passen (nicht verbindlich)

- globale Gewichte: Gradient-Boosted Trees (GBM), kleine MLPs auf Frame-Metriken
- Tile-Qualität: kleine CNN-Encoder / leichte ViT-tiny (nur bei ausreichendem Datenvolumen)
- Pixel-Zuverlässigkeitsmasken: U-Net-lite mit Ausgabe `M̂(p)` in `[m_min,1]`

LLMs sind nur für **Konfigurationssynthese, Interpretation von Validierungsreports und Testgenerierung** zulässig, nicht für pixelweise Rekonstruktion.

### 9.2.6 Validierungsanforderungen für die ML-Erweiterung (verbindlich)

Wenn ML aktiviert ist, gelten weiterhin alle verpflichtenden Core-Validierungstests, plus:

1. **Begrenzte Ausgaben:** erzwinge `Q̂ in [-3,+3]`, `M̂ in [m_min,1]`
2. **Deterministische Inferenz:** identische Eingaben liefern identische Gewichte/Masken
3. **Keine Struktursynthese:** Residuen-Korrelation darf keine unphysikalische Hochfrequenz-Injektion zeigen; Seams und Ringing dürfen gegenüber der Non-ML-Baseline nicht zunehmen
4. **Photometrische Konsistenz:** Flux-Verhältnisse von Kalibrationssternen bleiben gegenüber der Baseline innerhalb der Toleranz (konfigurierbar)
5. **Ablation:** Baseline (ohne ML) vs. ML-aktivierte Verbesserungen auf demselben Datensatz berichten

### 9.2.7 Konfigurations-Hooks (normative Namen)

Vorgeschlagene (nicht erschöpfende) Konfigurationsschlüssel:

- `ml.enable: true|false`
- `ml.global_model.path`
- `ml.tile_model.path`
- `ml.mask_model.path`
- `ml.mask.m_min`
- `ml.inference.device: cpu|gpu`
- `ml.inference.deterministic: true`

Fehlende ML-Modelle müssen von Implementierungen als kontrollierter Fallback auf den Non-ML-Core behandelt werden.

#### RBF-Oberfläche (verbindlich, wenn `bge.fit.method = rbf`)

Seien die Grid-Cell-Samples `(r_j, b_j, ω_j)` für `j = 1..J`, wobei:

- `r_j = (x_j, y_j)` die Mittelpunkte der Gitterzellen sind
- `b_j` der aggregierte Hintergrundwert ist
- `ω_j ≥ 0` das Zuverlässigkeitsgewicht der Zelle ist

Definiere die RBF-Oberfläche mit affinem Trendterm:

`B_c(r) = sum_{i=1..J} u_i * φ(||r - r_i||; μ) + a0 + a1*x + a2*y`

wobei:

- `u_i` die RBF-Koeffizienten (unbekannt) sind
- `(a0, a1, a2)` ein affiner Term ist (empfohlen; verbessert die Extrapolationsstabilität)
- `μ > 0` der Kernel-Shape-/Scale-Parameter ist
- `φ` der gewählte radiale Kernel ist

##### Unterstützte Kernel (verbindlich)

1. Multiquadric:

   `φ(d; μ) = sqrt(d^2 + μ^2)`

2. Thin-plate spline:

   `φ(d) = d^2 * log(d + ε)`

   mit kleinem `ε > 0` für numerische Stabilität (empfohlen: `ε = 1e-6 * G`).

3. Gaussian:

   `φ(d; μ) = exp(-d^2 / (2 * μ^2))`

   Für Gaussian wirkt `μ` als Bandbreite (`σ`).

##### Robuster regularisierter Fit (verbindlich)

Löse die Parameter `θ = (u, a)` durch Minimierung von:

`argmin_θ sum_{j=1..J} ω_j * ρ(b_j - B_c(r_j)) + λ * ||u||_2^2`

wobei:

- `ρ` der konfigurierte robuste Loss ist (Huber oder Tukey)
- `λ > 0` bei RBF zwingende Regularisierung ist
- die Optimierung deterministisch sein muss (IRLS oder äquivalent).

RBF ohne Regularisierung (`λ = 0`) ist verboten.

##### Empfohlene Defaults

- `μ = G` (Grid-Abstand)
- `λ = 1e-4` (innerhalb `[1e-6, 1e-2]` je nach Gradientenstärke abstimmbar)
- affinen Term standardmäßig einschließen

---

## 10. Kernaussage

Die Methode ersetzt die starre Suche nach "besten Frames" durch robuste raum-zeitliche Qualitätsmodellierung, verwendet alle Frames ohne qualitätsbasierte Selektion und rekonstruiert Signal dort, wo es physikalisch und statistisch am zuverlässigsten ist.
