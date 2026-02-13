# Tile-basierte Qualitaetsrekonstruktion fuer DSO - Methodik v3.2

**Status:** Normative Referenzspezifikation (konsolidiert)  
**Version:** v3.2 (2026-02-13)  
**Basiert auf:** v3.1E (2026-02-12)  
**Gilt fuer:** `tile_compile.proc` + `tile_compile.yaml`

---

## 0. Ziel von v3.2

v3.2 konsolidiert v3.1E auf eine **widerspruchsfreie, implementierungsstabile Basis**.

Kernziele:

1. mathematische Konsistenz (Notation, Formeln, Randfaelle)
2. klare Trennung von **Pflichtkern** vs. **optionalen Erweiterungen**
3. saubere Semantik fuer
   - Linearitaet,
   - keine Frame-Selektion,
   - robuste Pixel-Ausreisserbehandlung

---

## 1. Prinzipien und Begriffe

### 1.1 Physikalisches Ziel

Aus vollstaendig registrierten, linearen Kurzzeit-Frames wird ein raeumlich/zeitlich optimal gewichtetes Signal rekonstruiert.

Die Methode modelliert zwei orthogonale Qualitaetsachsen:

- **global** (Atmosphaere): Transparenz, Himmelshelligkeit, Rauschen
- **lokal** (Tile): Schaerfe, Strukturtragfaehigkeit, lokale Hintergrundlage

### 1.2 Keine Frame-Selektion (invariant)

**Verboten:** Entfernen ganzer Frames aufgrund von Qualitaet.  
**Erlaubt:** Pixelweise Ausreisserrejektion (Sigma-Clipping), sofern

- sie nur pixelweise wirkt,
- deterministische Parameter nutzt,
- und einen dokumentierten Fallback auf den unveraenderten Mittelwert besitzt.

### 1.3 Linearitaets-Semantik (praezisiert)

"Streng linear" bedeutet in v3.2:

1. **Photometrische Signalabbildung** bleibt linear (keine globalen nichtlinearen Tonkurven wie Stretch, asinh, log).
2. Lineare Rekonstruktionsschritte (Skalierung, gewichtetes Mittel, Overlap-Add) sind Pflicht.
3. Robuste/statistische Nichtlinearitaeten (MAD, Clipping, Sigma-Clipping, adaptive Gating-Entscheidungen) sind als **Hilfsschritte** erlaubt.

---

## 2. Annahmen und Betriebsmodi

### 2.1 Harte Annahmen (Verletzung -> Abbruch)

- Eingangsdaten sind linear (kein Stretch, keine Tonkurven)
- einheitliche Belichtungszeit (Toleranz +-5%)
- kanalgetrennte Verarbeitung ab Kanaltrennung
- keine qualitaetsbasierte Frame-Selektion
- registrierte Geometrie liegt in derselben Pixelreferenz

### 2.2 Weiche Annahmen

| Annahme | Optimal | Minimum | Aktion bei Unterschreitung |
|---|---:|---:|---|
| Frame-Anzahl N | >= 800 | >= 50 | Reduced Mode bei 50..199 |
| Registrierungsresiduum | < 0.3 px | < 1.0 px | Warnung > 0.5 px |
| Stern-Elongation | < 0.2 | < 0.4 | Warnung > 0.3 |

### 2.3 Reduced Mode (eindeutig)

- **Gueltig nur fuer:** `50 <= N <= 199`
- Schritte 8-9 (Clusterung + synthetische Frames) werden uebersprungen
- Finales Ergebnis ist die Rekonstruktion aus Phase 7

### 2.4 Unterminimum

- **N < 50:** kein Reduced Mode
- Standardaktion: kontrollierter Abbruch mit Diagnose
- Optional nur per explizitem `runtime.allow_emergency_mode: true`: Emergency-Mode mit Warnstatus

---

## 3. Pipelineuebersicht (normativ)

1. Registrierung und geometrische Vereinheitlichung
2. Kanaltrennung
3. Globale lineare Normalisierung
4. Globale Frame-Metriken und globale Gewichte
5. Tile-Geometrie
6. Lokale Tile-Metriken und lokale Gewichte
7. Tile-Rekonstruktion (Overlap-Add)
8. Zustandsbasierte Clusterung (nur Full Mode)
9. Synthetische Frames (nur Full Mode)
10. Finales lineares Stacking
11. Nachbearbeitung (optional, nicht Teil des Qualitaetskerns)

Pflichtkern: 1-10.  
Optional/Freischaltbar: lokale Denoiser, Sigma-Clipping-Varianten, WCS/PCC.

---

## 4. Registrierung und Kanaltrennung bis Phase 2 (normativ)

Bis einschliesslich Phase 2 gilt der CFA-basierte Registrierungs- und Kanaltrennungspfad.
Ab Phase 3 gilt gemeinsamer Kern.

### 4.1 CFA-basierter Registrierungspfad

- Registrierung auf CFA-Luminanzproxy
- CFA-aware Warp per Subplanes (`warp_cfa_mosaic_via_subplanes`)
- danach Kanaltrennung

### 4.2 Registrierungskaskade

Pro Frame:

1. konfigurierbare Primaermethode (`triangle_star_matching` default)
2. feste Fallback-Reihenfolge:
   - `trail_endpoint_registration`
   - `feature_registration_similarity` (AKAZE)
   - `robust_phase_ecc`
   - `hybrid_phase_ecc`
   - Identity-Fallback mit Warnung

Akzeptanzkriterium je Versuch:

- `NCC(warped, ref) > NCC(identity, ref) + delta_ncc`
- Default `delta_ncc = 0.01`

---

## 5. Gemeinsamer Kern ab Phase 3

## 5.1 Notation (verbindlich)

- `f` Frame-Index, `t` Tile-Index, `c` Kanal-Index, `p` Pixel
- `I_{f,c}(p)` normalisiertes Eingangsbild pro Frame/Kanal
- `B_{f,c}` globaler Hintergrund (vor Normalisierung)
- `sigma_{f,c}` globales Rauschen (nach Normalisierung)
- `E_{f,c}` globale Gradientenergie (nach Normalisierung)
- `Q_{f,c}` globaler Qualitaetsindex
- `G_{f,c}` globales Gewicht
- `Q_{f,t,c}^{local}` lokaler Qualitaetsindex
- `L_{f,t,c}` lokales Gewicht
- `W_{f,t,c}` effektives Gewicht

**Ab hier durchgaengig mit Kanalindex `c`.**

---

## 5.2 Globale lineare Normalisierung (Pflicht)

Reihenfolge:

1. Hintergrund aus Rohdaten:
   - `B_{f,c} = median(I_{f,c}^{raw})`
2. Lineare Skalierung:
   - `I_{f,c} = I_{f,c}^{raw} / max(B_{f,c}, eps_bg)`
3. Metriken auf normalisierten Daten:
   - `sigma_{f,c}`, `E_{f,c}`

Verboten: globale nichtlineare Tonkurven.

Empfohlene Defaults:

- `eps_bg = 1e-6`

---

## 5.3 Globale Metriken und Gewichte

### 5.3.1 Robuste Metrik-Normalisierung

Fuer Metrikfolge `x`:

`z(x_i) = (x_i - median(x)) / max(1.4826 * MAD(x), eps_mad)`

mit `eps_mad = 1e-6`.

### 5.3.2 Globaler Qualitaetsindex

`Q_{f,c} = alpha*(-z(B_{f,c})) + beta*(-z(sigma_{f,c})) + gamma*z(E_{f,c})`

Nebenbedingung: `alpha + beta + gamma = 1`

Defaults:

- `alpha=0.4, beta=0.3, gamma=0.3`

Clamping vor Exponentialfunktion:

`Q_{f,c}^{clamped} = clip(Q_{f,c}, -3, +3)`

Globales Gewicht:

`G_{f,c} = exp(k_global * Q_{f,c}^{clamped})`

mit `k_global > 0`, Default `k_global=1.0`.

### 5.3.3 Optionale adaptive Gewichtung

Falls `global_metrics.adaptive_weights=true`:

- Varianzen werden auf robust normalisierten Metriken berechnet:
  - `Var(z(B))`, `Var(z(sigma))`, `Var(z(E))`
- Rohgewichte:
  - `alpha' ~ Var(z(B))`, `beta' ~ Var(z(sigma))`, `gamma' ~ Var(z(E))`
- Clip je Gewicht auf [0.1, 0.7], dann Renormalisierung auf Summe 1
- Fallback auf statische Defaults bei degenerierter Gesamtvarianz

---

## 5.4 Tile-Geometrie

Parameter:

- Bildgroesse `W,H`
- robuste Seeing-Schaetzung `F` (FWHM in Pixel)
- `s = tile.size_factor`
- `T_min = tile.min_size`
- `D = tile.max_divisor`
- `o = tile.overlap_fraction`, `0 <= o <= 0.5`

Formeln:

`T0 = s * F`

`T = floor(clip(T0, T_min, floor(min(W,H)/D)))`

`O = floor(o * T)`

`S = T - O`

Guards (verbindlich):

1. wenn `F <= 0` -> `F = 3.0`
2. `T_min >= 16`
3. wenn `S <= 0` -> `o=0.25`, neu berechnen
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

### 5.5.4 Lokales Gewicht

`Q_{f,t,c}^{local} = clip(Q_{f,t,c}^{star|struct}, -3, +3)`

`L_{f,t,c} = exp(Q_{f,t,c}^{local})`

---

## 5.6 Effektives Gewicht

`W_{f,t,c} = G_{f,c} * L_{f,t,c}`

Semantik:

- `G`: globale Atmosphaerenqualitaet
- `L`: lokale Struktur-/Schaerfequalitaet

---

## 5.7 Tile-Rekonstruktion (konsolidiert)

Fuer Pixel `p` in Tile `t`:

`D_{t,c} = sum_f W_{f,t,c}`

Wenn `D_{t,c} >= eps_weight`:

`R_{t,c}(p) = sum_f W_{f,t,c} * I_{f,c}(p) / D_{t,c}`

Wenn `D_{t,c} < eps_weight`:

`R_{t,c}(p) = (1/N) * sum_f I_{f,c}(p)`

und `fallback_used=true` fuer dieses Tile.

Default `eps_weight = 1e-6`.

### 5.7.1 Tile-Normalisierung vor OLA (verbindlich)

Fuer rekonstruiertes Tile `R_{t,c}`:

1. `bg_t = median(R_{t,c})`
2. `X_t = R_{t,c} - bg_t`
3. `m_t = median(abs(X_t))`
4. wenn `m_t >= eps_median`: `Y_t = X_t / m_t`, sonst `Y_t = X_t`

Default `eps_median = 1e-6`.

### 5.7.2 Fensterung und Overlap-Add

2D-Fenster separabel mit diskreter Hann-Funktion:

`hann(i,N) = 0.5*(1 - cos(2*pi*i/(N-1)))`, `i=0..N-1`

Sonderfall: `N=1 -> hann=1`.

`w(x,y) = hann(x,W_t) * hann(y,H_t)`

Rekonstruktionsbild:

- Zaehlerakkumulator: `A`
- Fenstersummenakkumulator: `S`

`A += w * Y_t`, `S += w`, Ergebnis `I_rec = A / max(S, eps_weight)`

Optional kann nach OLA ein global robuster Tile-Hintergrundoffset restauriert werden (Median ueber `bg_t`).

---

## 5.8 Optionale lokale Denoiser (klar optional)

Diese Schritte sind **nicht Teil des zwingenden mathematischen Kerns**, aber zulaessige Erweiterungen.

### 5.8.1 Soft-Threshold Highpass

- Background via Box-Blur
- Residuum
- `tau = alpha_d * sigma_tile`
- Soft-Shrinkage
- Rekonstruktion

### 5.8.2 Wiener im Frequenzraum

- Reflection padding
- FFT
- Wiener-Transferfunktion
- IFFT und Crop

Anwendung nur bei erfuellten Gating-Bedingungen (SNR/Qualitaet/Tile-Typ).

---

## 5.9 Zustandsbasierte Clusterung (Full Mode)

Nur aktiv bei `N >= 200`.

Zustandsvektor je Frame/Kanal (kanalweise oder kanalaggregiert, konfigurierbar):

`v_f = (G_{f,*}, mean_t(Q_{f,t,*}^{local}), var_t(Q_{f,t,*}^{local}), B_{f,*}, sigma_{f,*})`

Clusteranzahl:

`K = clip(floor(N/10), K_min, K_max)`

Defaults: `K_min=5`, `K_max=30`.

---

## 5.10 Synthetische Frames

### 5.10.1 Default (global)

Fuer Cluster `k`:

`S_{k,c} = sum_{f in k} G_{f,c} * I_{f,c} / sum_{f in k} G_{f,c}`

### 5.10.2 Optional (tile_weighted)

Wenn `synthetic.weighting=tile_weighted`:

- pro Tile/Kanal mit `W_{f,t,c}` rekonstruieren
- per OLA zu `S_{k,c}` zusammensetzen

### 5.10.3 Semantik von Phase 7 vs 9

- Full Mode mit `global`: Phase 7 liefert vor allem lokale Qualitaetsmodellierung/Diagnostik; Endprodukt entsteht aus Phase 9+10.
- Full Mode mit `tile_weighted`: lokale Tile-Qualitaet propagiert explizit in synthetische Frames.
- Reduced Mode: Ergebnis aus Phase 7 ist direktes Endprodukt.

---

## 5.11 Finales lineares Stacking

Endergebnis pro Kanal:

`R_c = (1/K) * sum_k S_{k,c}`

Optional davor: pixelweises Sigma-Clipping ueber `S_{k,c}` mit Fallback auf unveraenderten Mittelwert bei zu starker Rejektion.

---

## 6. Nachbearbeitung (nicht Teil des Pflichtkerns)

### 6.1 RGB/LRGB-Kombination

Austauschbar, ausserhalb des Rekonstruktionskerns.

### 6.2 Astrometrie (WCS)

Zulaessiger nachgelagerter Schritt, ohne Rueckkopplung in Kerngewichte.

### 6.3 PCC

Zulaessiger nachgelagerter Schritt, auf linearen Daten.

---

## 7. Validierung und Abbruch

## 7.1 Erfolgskriterien

- FWHM-Verbesserung gegen Referenzstack gemaess `validation.min_fwhm_improvement_percent`
- Hintergrund-RMS nicht schlechter als Referenz
- keine systematischen Tile-Seams
- stabile Gewichtverteilungen

## 7.2 Abbruchkriterien

- Datenintegritaet verletzt (nichtlinear, unlesbar, inkonsistent)
- Registrierung grossflachig fehlgeschlagen
- numerische Instabilitaet trotz Fallbacks

## 7.3 Mindesttests (normativ)

1. `alpha+beta+gamma=1`
2. Clamping vor `exp`
3. Tile-Monotonie in `F`
4. Overlap-Konsistenz (`0<=o<=0.5`, integer O,S)
5. Low-weight-Fallback ohne NaN/Inf
6. keine Kanalkopplung
7. keine qualitaetsbasierte Frame-Selektion
8. deterministische Reproduzierbarkeit
9. Registrierungskaskade inkl. Identity-Fallback
10. CFA-Phasenerhalt
11. WCS-Roundtrip-Fehler unter Schwellwert
12. PCC-Stabilitaet: positive Determinante, begrenzte Konditionszahl, Residuen unter Schwellwert

Hinweis: Der alte Test "kein negatives Matrixelement" fuer PCC wird in v3.2 **nicht** mehr als harter Test gefordert.

---

## 8. Empfohlene numerische Defaults

- `eps_bg = 1e-6`
- `eps_mad = 1e-6`
- `eps_weight = 1e-6`
- `eps_median = 1e-6`
- `delta_ncc = 0.01`
- `Q`-Clamp global/lokal: `[-3, +3]`

---

## 9. Abgrenzung Pflichtkern vs Erweiterung

### Pflichtkern

- CFA-basierter Registrierungspfad bis Kanaltrennung
- globale Normalisierung
- globale/lokale Metriken und Gewichte
- Tile-Rekonstruktion inkl. konsolidierter Fallbacks
- Clusterung/Synthese/Finalstack je nach Betriebsmodus

### Optionale Erweiterung

- Soft-Threshold / Wiener
- alternative Sigma-Clipping-Strategien
- WCS/PCC
- spezielle Performance-Backends (GPU, Queue-Worker)

---

## 10. Aenderungshistorie

| Datum | Version | Aenderung |
|---|---|---|
| 2026-02-13 | v3.2 | Pfad A entfernt; CFA-basierter Registrierungs- und Kanaltrennungspfad als einzig normativer Pfad bis Phase 2 festgelegt |
| 2026-02-13 | v3.2 | Konsolidierung nach mathematischer Diagnose |
| 2026-02-13 | v3.2 | Linearitaets-Semantik praezisiert |
| 2026-02-13 | v3.2 | Reduced-Mode-Grenzen eindeutig gemacht |
| 2026-02-13 | v3.2 | Notation vereinheitlicht auf `f,t,c` |
| 2026-02-13 | v3.2 | Tile-Rekonstruktion/Fallbacks in einen konsistenten Block ueberfuehrt |
| 2026-02-13 | v3.2 | Diskrete Hann-Definition normativ festgelegt |
| 2026-02-13 | v3.2 | PCC-Testkriterium fachlich robust ersetzt |

---

## 11. Kernsatz

Die Methode ersetzt die starre Suche nach "besten Frames" durch eine robuste raeumlich-zeitliche Qualitaetsmodellierung, nutzt alle Frames ohne qualitaetsbasierte Selektion und rekonstruiert Signal dort, wo es physikalisch und statistisch am verlaesslichsten ist.
