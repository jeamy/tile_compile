# Diagnose: C++ Implementierung vs. Methodik v3.1E

**Datum:** 2026-02-12
**Basis-Dokumente:**
- `doc/v3/tile_basierte_qualitatsrekonstruktion_methodik_v_3.1_erweitert.md` (v3.1E)
- `doc/v3/tile_compile_cpp_code_fixes_2026-02-11.txt` (Fixes)
- C++ Quellcode: `tile_compile_cpp/` (runner_main.cpp, lib modules)

---

## Teil 1: Abweichungen C++ Source vs. Methodik 3.1E

### 1.1 ~~Registrierungskaskade — Reihenfolge weicht ab~~ — ✅ SPEZIFIKATION AKTUALISIERT (2026-02-12)

~~**Methodik 3.1E hatte ECC-Methoden vor geometrischen Methoden.**~~

→ ✅ Methodik 3.1E (§A.2.1.2, §A.2.1.3, §A.10) an C++ Implementierung angepasst:
- Zweistufige Kaskade: Primärmethode (konfigurierbar) + feste Fallback-Kaskade
- Reihenfolge: triangle → star_pair → trail_endpoint → AKAZE → robust_phase_ecc → hybrid_phase_ecc → Identity
- `trail_endpoint_registration` als neuer Algorithmus dokumentiert (§A.2.1.3 Punkt 3)
- NCC-Validierung mit Schwelle ncc_identity + 0.01 spezifiziert
- Pseudocode in §A.10 entspricht jetzt `register_single_frame()` in global_registration.cpp

---

### 1.2 ~~Adaptive Gewichtung~~ — ✅ IMPLEMENTIERT (2026-02-12)

~~**Kein Code implementierte die varianzbasierte Anpassung.**~~

→ ✅ Implementiert in `calculate_global_weights()` (metrics.cpp) exakt nach Methodik 3.1E §3.2:
1. Varianz der normalisierten Metriken berechnen: `Var(B̃)`, `Var(σ̃)`, `Var(Ẽ)`
2. Gewichte proportional zur Varianz: `α' = Var(B̃) / (Var(B̃) + Var(σ̃) + Var(Ẽ))`
3. Clip auf [0.1, 0.7] pro Gewicht
4. Renormalisierung auf Σ = 1
5. Fallback auf statische Config-Gewichte bei Var = 0

Aktivierung via `adaptive_weights: true` in Config (metrics.hpp Signatur + runner_main.cpp Aufruf erweitert).

---

### 1.3 ~~Wiener-Filter für Tiles~~ — ✅ AKTIVIERT (2026-02-12)

~~**`wiener_tile_filter()` wurde nirgends im Runner aufgerufen — Dead Code.**~~

→ ✅ Aktiviert in `runner_main.cpp` Phase 6 nach Sigma-Clip-Stacking:
- Rauschschätzung per Tile via `meanStdDev` auf Residualbild (box-blur subtrahiert)
- Gating: `is_star_tile`, `snr_threshold`, `q_min`/`q_max` — wie in Methodik 3.1E §3.3.1 spezifiziert
- Angewendet auf mono `tile_rec` und OSC `tile_rec_R/G/B`
- Config: `tile_denoise.wiener` (neue Struktur, legacy `wiener_denoise` weiterhin unterstützt)

---

### 1.4 ~~Tile-Normalisierung vor Overlap-Add~~ — ✅ IMPLEMENTIERT (2026-02-12)

~~**Keine Hintergrund-Subtraktion, keine Median-Normalisierung pro Tile vor dem Overlap-Add.**~~

→ ✅ Implementiert in `runner_main.cpp` Phase 6 exakt nach Methodik 3.1E §3.6:
1. **Pro Tile** (vor Hanning-Fenster): `normalize_tile_for_ola()` Lambda:
   - `bg = median(T)` via `nth_element` (O(n) partial sort)
   - `T' = T - bg` (Hintergrund subtrahieren)
   - `med_abs = median(|T'|)` → `T'' = T' / med_abs` (Skalennormalisierung)
   - Gibt `bg` zurück für spätere Restaurierung
   - Angewendet auf mono `tile_rec` und OSC `tile_rec_R/G/B`
2. **Nach Overlap-Add**: Globaler Hintergrund restauriert:
   - Median aller validen Tile-Hintergründe berechnet
   - Auf gesamte Rekonstruktion addiert (mono und OSC R/G/B)

---

### 1.5 Synthetische Frames: nur globale Gewichtung, tile_weighted nicht implementiert

**Methodik 3.1E (§3.8):** Definiert optionale tile-basierte Gewichtung:
```
W_f,t,c = G_f,c · L_f,t,c
```
Aktivierung über `synthetic.weighting: tile_weighted`.

**C++ (runner_main.cpp:2024–2051):** `reconstruct_subset()` verwendet ausschließlich `global_weights[fi]`:
```cpp
float w = global_weights[fi];  // nur G_f, kein L_f,t
out += src * w;
```
Das Config-Feld `synthetic.weighting` existiert (configuration.hpp:137), wird aber **nicht ausgewertet**.

**Bewertung:** Konsistent mit dem Default (`global`), aber die Option `tile_weighted` ist toter Code. Für fortgeschrittene Anwendungsfälle wäre tile-basierte synthetische Frame-Erzeugung wertvoll.

---

### 1.6 ~~Highpass + Soft-Threshold Rauschunterdrückung~~ — ✅ IMPLEMENTIERT (2026-02-12)

~~**Weder die Basis-Methode (Soft-Threshold) noch die Config dafür existierte.**~~

→ ✅ Vollständig implementiert exakt nach Methodik 3.1E §3.3.1:
- `reconstruction::soft_threshold_tile_filter()` in reconstruction.cpp:
  1. `B = box_blur(T, k)` — Hintergrund-Schätzung (k=31 default)
  2. `R = T - B` — Highpass-Residuum
  3. `σ = 1.4826 · median(|R - median(R)|)` — robuste MAD-Rauschschätzung
  4. `τ = α · σ` — Schwellwert (α=1.5 default)
  5. `R' = sign(R) · max(|R| - τ, 0)` — Soft-Threshold Shrinkage
  6. `T' = B + R'` — Rekonstruktion
- Config: `tile_denoise.soft_threshold` mit `enabled`, `blur_kernel`, `alpha`, `skip_star_tiles`
- Aktiviert in Phase 6 **vor** Wiener-Filter (spatial → frequency domain Reihenfolge)
- Star-Tile Gating: `skip_star_tiles: true` (default) schützt Sterndetails
- Schema: `tile_compile.schema.json` + inline Schema aktualisiert

---

### 1.7 Gradientenergie: median statt mean

**Methodik 3.1E (Anhang A.3):**
```
E = mean(|∇I|²)    [empfohlen]
E = median(|∇I|²)  [robustere Alternative]
```

**C++ (metrics.cpp:159 + tile_metrics.cpp:175):** Beide verwenden `median_of(grad_vals)`.

**Bewertung:** Konform mit der "robusteren Alternative". Konsistent global und lokal. **OK.**

---

### 1.8 ~~FWHM-Messung~~ — ✅ BEHOBEN (2026-02-12)

~~**C++ (tile_metrics.cpp:23–35):** `compute_fwhm_proxy()` verwendete Half-Maximum Pixel-Counting.~~

→ ✅ Ersetzt durch denselben `fit_1d_fwhm()` Algorithmus wie `metrics.cpp`: Gewichtetes 2. Moment → σ → FWHM = 2.3548·σ. Sternfindung via `goodFeaturesToTrack` auf Residualbild, 11×11 Patches pro Stern, 1D Gauss-Fit in X und Y, geometrisches Mittel. Median über alle Sterne im Tile. `star_count` reflektiert jetzt die Anzahl der erfolgreich gemessenen Sterne (nicht Corner-Count).

---

### 1.9 ~~Lokale Tile-Metriken: Rundheit und Kontrast~~ — ✅ BEHOBEN (2026-02-12)

~~**C++ (tile_metrics.cpp:37–81):** `compute_roundness_proxy()` und `compute_contrast_proxy()` waren grobe Approximationen.~~

→ ✅ Beide Proxies durch stern-basierte Messungen ersetzt:
- **Rundheit:** `min(FWHM_x, FWHM_y) / max(FWHM_x, FWHM_y)` pro Stern → Median. Echtes Achsenverhältnis wie in `metrics.cpp` (estimate_fwhm_xy).
- **Kontrast:** `peak_flux / background` pro Stern → Median. Signal/Hintergrund wie in 3.1E spezifiziert.
- Alle drei Metriken (FWHM, Rundheit, Kontrast) werden aus derselben Sternmessung berechnet — ein Durchlauf statt drei separater Proxies.

---

## Teil 2: Tile-Größen, Gewichtung und Balance

### 2.1 Tile-Größen-Berechnung — KORREKT

**Methodik 3.1E (§3.3):**
```
T_0 = s · F
T = floor(clip(T_0, T_min, floor(min(W,H) / D)))
O = floor(o · T)
S = T - O
```

**C++ (runner_main.cpp:756–779):**
```cpp
float t0 = cfg.tile.size_factor * F;
float tc = min(max(t0, tmin), tmax);
seeing_tile_size = floor(tc);
if (seeing_tile_size < tmin) seeing_tile_size = tmin;
overlap_px = floor(overlap_fraction * seeing_tile_size);
stride_px = seeing_tile_size - overlap_px;
```

**Bewertung:** ✅ Konform. Grenzwertprüfungen (F>0 → Default 3.0, T_min≥16, S>0 → o=0.25) alle korrekt implementiert.

**Default-Config:** `size_factor=32, min_size=64, max_divisor=6, overlap_fraction=0.25`

Bei typischem FWHM=3.5px ergibt sich T_0 = 32 × 3.5 = 112px. Das ist vernünftig (~32 PSF-Durchmesser pro Tile).

### 2.2 FWHM-Probing für Tile-Größe — suboptimal

**C++ (runner_main.cpp:729–749):** Es werden bis zu 5 Frames geprobt, aber der **erste erfolgreiche Messwert wird verwendet** (`break` bei fwhm > 0):
```cpp
if (fwhm > 0.0f) {
    seeing_fwhm_med = fwhm;
    break;  // ← Nur 1 Frame!
}
```

**Bewertung:** Einzelmessung ist fragil. Methodik sagt "robuste FWHM-Schätzung (z.B. Median über viele Sterne **und Frames**)".

**Empfehlung:** Alle 5 Probes sammeln, Median nehmen:
```cpp
std::vector<float> fwhm_probes;
for (...) {
    float fwhm = measure_fwhm_from_image(img);
    if (fwhm > 0.0f) fwhm_probes.push_back(fwhm);
}
seeing_fwhm_med = fwhm_probes.empty() ? 3.0f : core::median_of(fwhm_probes);
```

### 2.3 Gewichtung: Effektives Gewicht W_f,t = G_f · L_f,t — KORREKT

**C++ (runner_main.cpp:1525–1532 + 1594–1601):**
```cpp
float G_f = global_weights[fi];
float L_ft = local_weights[fi][ti];
weights_valid.push_back(G_f * L_ft);
```

**Bewertung:** ✅ Konform mit §3.5: `W_f,t = G_f · L_f,t`.

### 2.4 Balance gute/schlechte Tiles — VERBESSERUNGSWÜRDIG

**Aktuelles Verhalten:**
- Q_f und Q_local werden auf [-3, +3] geklemmt
- Gewicht = exp(Q), also Bereich [exp(-3), exp(+3)] = [0.050, 20.09]
- **Verhältnis bestes:schlechtestes = 403:1**

Das klingt stark, aber in der Praxis:
- Die meisten Frames/Tiles haben Q nahe 0 (z-Score-Normalisierung → median=0)
- Die Clamping-Grenzen [-3, +3] bedeuten 3σ-Ausreißer = Maximum
- Typische Spread: exp(-1) bis exp(+1) = [0.37, 2.72] → **Verhältnis 7.4:1**

**Problem:** Sehr gute Tiles (Q > +1) bekommen nur moderat mehr Gewicht. Die exponentielle Funktion mit Clamping bei ±3 ist konservativ.

**Vorschläge für stärkere Differenzierung:**

1. **Steilerer Exponent:** `G_f = exp(k · Q_f)` mit k > 1 (z.B. k=1.5 oder k=2)
   - k=2: Verhältnis exp(-6):exp(+6) = 162755:1, typisch exp(-2):exp(+2) = [0.14, 7.39] → 54:1
   - Konfigurierbar als `global_metrics.exponent_scale`

2. **Asymmetrisches Clamping:** [-4, +3] statt [-3, +3]
   - Schlechte Tiles stärker heruntergewichten als gute hochgewichten
   - Physikalisch sinnvoll: ein schlechtes Tile (Wolke, Tracking-Fehler) hat keinen Informationsgehalt

3. **Power-Law statt Exponential:** `G_f = max(0, Q_f + shift)^p` mit p > 1
   - Natürlichere Spread-Kontrolle
   - Aber: negative Gewichte müssen vermieden werden

**Empfehlung:** Option 1 (skalierbarer Exponent) ist am einfachsten und effektivsten. Neuer Config-Parameter `weight_exponent_scale` (Default 1.0 für Rückwärtskompatibilität).

### 2.5 Lokale Gewichtung: STAR vs. STRUCTURE — Schwäche bei FWHM-dominierter Formel

**Methodik 3.1E (§3.4):**
```
Q_star  = 0.6·(-FWHM̃) + 0.2·R̃ + 0.2·C̃
Q_struct = 0.7·(Ẽ/σ̃) - 0.3·B̃
```

**C++ (runner_main.cpp:1271–1278):** Konform implementiert.

**Problem:** Bei der STAR-Formel dominiert FWHM mit 60% Gewicht. Da die Tile-FWHM aber über den ungenauen Pixel-Counting-Proxy gemessen wird (§1.8), ist die treibende Metrik die schwächste. Das untergräbt die Diskriminierungsfähigkeit der STAR-Tile-Gewichtung.

**Zusätzliches Problem:** `star_count` in tile_metrics.cpp (Zeile 177–187) verwendet `goodFeaturesToTrack` auf einem normalisierten uint8-Bild — das detektiert auch Rauschpeaks und Nebelknoten, nicht nur Sterne. Die `star_min_count`-Schwelle (Default 10) kann dadurch fälschlicherweise STRUCTURE-Tiles als STAR klassifizieren.

---

## Teil 3: Bewertung der Fixes (code_fixes_2026-02-11.txt)

### Fix 1: Debayer nearest-neighbor correctness — ✅ 3.1E-KONFORM
Korrekte Bayer-Parity für alle Muster. Methodisch neutral (Implementierungsbug).

### Fix 2: Global weights — kein normalize exp(Q) — ✅ 3.1E-KONFORM
3.1E definiert G_f = exp(clip(Q_f)). Keine Normalisierung auf sum=1. **Korrekt umgesetzt.**

### Fix 3: ECC inversion convention fix — ✅ 3.1E-KONFORM (verifiziert)

**Verifizierung (2026-02-12):** Kein Widerspruch. Beide Konventionen sind korrekt:

- **ECC-Methoden** (`hybrid_phase_ecc`, `robust_phase_ecc`): `findTransformECC(ref, moving, ...)` liefert
  eine M→R Warp-Matrix. `apply_warp()` nutzt `WARP_INVERSE_MAP`, was die Matrix als dst→src interpretiert.
  M→R als inverse map = korrekt. **Keine Inversion nötig.**
- **Geometrische Methoden** (triangle, AKAZE, star_pair): Berechnen Forward-Warp M→R, invertieren dann
  explizit zu R→M (`invert_warp_2x3()`), damit `WARP_INVERSE_MAP` korrekt funktioniert.

Beide Wege erzeugen die gleiche korrekte Semantik für `warpAffine`. Die früheren Memory-Einträge
über "ECC-Inversion hinzugefügt" beziehen sich auf eine Zwischenversion, die inzwischen korrekt
aufgelöst wurde.

### Fix 4: No frame selection — ✅ 3.1E-KONFORM
Methodik v3 verbietet Frame-Selektion explizit. `frame_usable` entfernt, alle Frames behalten. **Korrekt.**

Allerdings gibt es `frame_has_data` (Zeile 1107), das nur leere/unlesbare Frames ausschließt — das ist kein quality-based selection und somit konform.

### Fix 5: LOCAL_METRICS math — ✅ 3.1E-KONFORM
- log(FWHM) entfernt → **konform** (3.1E: "ohne log(FWHM)")
- STRUCTURE: `zscore(E/σ)` statt `z(E)/z(σ)` → **konform** (3.1E: "(E/σ)_t,f,c")
- Tile STAR/STRUCTURE Klassifikation per Tile gespeichert → **konform**

**Gut umgesetzt.** Einer der wichtigsten Fixes.

### Fix 6: Synthetic frames use prewarped data — ✅ 3.1E-KONFORM
Vermeidet doppeltes Warping. Konsistent mit Registrierung. **Gut.**

### Fix 7: ASTAP auf linearen Daten — ✅ 3.1E-KONFORM
Plate-Solving auf ungestretchten Daten. Shell-Quoting. WCS-Injektion. **Gut implementiert.**

### Fix 8: Background metric aus Raw-Daten — ✅ 3.1E-KONFORM
3.1E §3.1: "B_f,c = median(I_f,c) # VOR Normalisierung". Fix setzt genau das um. **Exakt konform.**

### Fix 9: OSC debayer-before-stack — ✅ METHODISCH SINNVOLL
Nicht explizit in 3.1E spezifiziert, aber physikalisch korrekt: Debayer nach Sigma-Clipping erzeugt Farbartefakte an steilen Gradienten. **Gute Erweiterung.**

### Fix 10: OSC RGB stacking memory hardening — ✅ GUT
Sequentielles R→G→B Stacking pro Tile + Memory-Budget-Cap. Rein technischer Fix, methodisch neutral.

### Fix 11: PCC neutraler Hintergrund — ✅ 3.1E-KONFORM
3.1E §3.9.3 spezifiziert PCC als Post-Processing. Neutraler Hintergrund nach PCC ist physikalisch korrekt.

### Fix 12: Resume-Kommando — ✅ 3.1E-NEUTRAL
Convenience-Feature, kein Einfluss auf Methodik. **OK.**

---

## Teil 4: Zusammenfassung

### Kritische Abweichungen (sollten behoben werden)

| # | Bereich | Schwere | Beschreibung |
|---|---------|---------|--------------|
| 1 | Tile-Normalisierung | **HOCH** | Fehlende Hintergrund-Subtraktion + Median-Normalisierung vor Overlap-Add (§3.6/A.6). Kann Patchwork-Artefakte verursachen. |
| 2 | Lokale FWHM | **HOCH** | tile_metrics.cpp verwendet Pixel-Counting statt fit_1d_fwhm. Q_star (60% FWHM) wird dadurch unzuverlässig. |
| 3 | Wiener-Filter | MITTEL | Implementiert aber nicht aufgerufen. Dead code. |
| 4 | Adaptive Gewichtung | MITTEL | Config-Feld existiert, Logik fehlt. |

### Verbesserungswürdige Bereiche

| # | Bereich | Beschreibung |
|---|---------|--------------|
| 5 | FWHM-Probing | Nur 1 Frame statt Median über mehrere → instabil |
| 6 | Weight Exponent | Fester exp(Q), kein skalierbarer Exponent → moderate Differenzierung gut/schlecht |
| 7 | Tile Proxies | Rundheit + Kontrast in tile_metrics sind grob |
| 8 | Synthetische tile_weighted | Config-Option vorhanden, nicht implementiert |
| 9 | Threshold-Denoising | Basis-Methode (Soft-Threshold) aus §3.3.1 nicht implementiert |

### Gut umgesetzte Bereiche

| # | Bereich | Bewertung |
|---|---------|-----------|
| ✅ | Tile-Größen-Berechnung | Exakt konform mit §3.3 inkl. Grenzwertprüfungen |
| ✅ | Globale Metriken (B, σ, E) | Korrekt: B auf Raw-Daten, σ/E auf normalisierten |
| ✅ | Globale Gewichte exp(clip(Q)) | Konform, ohne Normalisierung auf sum=1 |
| ✅ | Effektives Gewicht W=G·L | Korrekt multiplikativ |
| ✅ | Lokale Metriken-Formeln | Q_star und Q_struct exakt wie spezifiziert |
| ✅ | Registrierungskaskade | Robust mit NCC-Validierung, besser als Spec |
| ✅ | CFA-aware Transformation | warp_cfa_mosaic_via_subplanes korrekt |
| ✅ | Sigma-Clip Stacking | Gewichtete + ungewichtete Variante, Fallbacks |
| ✅ | K-Means Clusterung | K=clip(N/10, K_min, K_max) exakt konform §3.7 |
| ✅ | PCC Pipeline | Vollständig mit Siril/VizieR Fallback |
| ✅ | Memory Management | Stufenweise Freigabe nach Phasen |
| ✅ | Keine Frame-Selektion | Invariante korrekt eingehalten |

### Fixes-Bewertung

**Alle 12 Fixes sind 3.1E-konform oder methodisch sinnvoll.**

---

## Teil 5: Optimierungsmöglichkeiten

### 5.1 Performance — ✅ ERLEDIGT (2026-02-12)

1. ~~**tile_metrics.cpp: Doppelte Pixel-Sammlung.**~~ → ✅ Single-Pass: Tile- und Residual-Pixel werden in einem Durchlauf gesammelt. `collect_pixels()` entfernt. Temporäre Kopien für destruktive median/MAD-Operationen, Threshold-Split auf bereits gesammelten Vektoren statt erneuter cv::Mat-Iteration.

2. ~~**runner_main.cpp Phase 6 OSC: `load_tile_normalized()` 4× pro Frame.**~~ → ✅ Mosaics werden im Validity-Pass gecacht (`valid_mosaics` Vektor) und direkt im Channel-Stack wiederverwendet. Tile-Extraktion von 4× auf 1× pro Frame reduziert.

3. **metrics.cpp: `calculate_frame_metrics()`** — ✅ Bereits optimiert: Funktion nutzt seit längerem Downsampling auf max 1024px (`kMaxDim`), kein Full-Frame-Vektor nötig.

4. ~~**Hanning-Fenster pro Tile neu berechnet.**~~ → ✅ Zweifach optimiert: (a) `reconstruction::reconstruct_tiles()`: Cache mit `cached_w`/`cached_h`, nur bei Größenänderung neu berechnet. (b) Runner `process_tile`: `shared_hann_x`/`shared_hann_y` einmal vor der Tile-Schleife berechnet, per const-Referenz in allen Tile-Workern geteilt (mit Fallback für nicht-uniforme Rand-Tiles).

### 5.2 Memory

1. **Phase 3 GLOBAL_METRICS: Jeder Frame wird komplett geladen** (`read_fits_float`), normalisiert, und dann wieder verworfen. Das ist notwendig, aber: In Phase 4/5 wird derselbe Frame **erneut geladen**. Wenn genug RAM vorhanden wäre, könnten die normalisierten Frames gecacht werden.

2. **`prewarped_frames` Vektor:** ~N×W×H×4 Bytes. Bei 477 Frames × 1920×1080 = 3.9 GB. Das ist der größte Speicherverbraucher. Alternativ: On-Demand aus Disk laden (registrierte Frames als temporäre Dateien schreiben).

### 5.3 Berechnungs-Genauigkeit

1. ~~**Sigma-Clip Bessel-Korrektur.**~~ → ✅ Beide Sigma-Clip-Funktionen in `reconstruction.cpp` korrigiert: (a) `sigma_clip_stack`: `var *= kept/(kept-1)` für unbiased sample variance. (b) `sigma_clip_weighted_tile`: Reliability-Weights Bessel-Korrektur `var_unbiased = Σ(wi·d²) / (V1 - V2/V1)` mit `V1=Σwi`, `V2=Σwi²`.

2. ~~**K-Means Initialisierung.**~~ → ✅ Ersetzt durch echtes k-means++ (Arthur & Vassilvitskii 2007): Erster Center = mittlerer Frame, weitere Center mit Wahrscheinlichkeit proportional zu D(x)² gesampelt. Fixed seed (42) für Reproduzierbarkeit. `<random>` Header hinzugefügt.

3. **FWHM als Tile-Größen-Treiber:** `seeing_fwhm_med` wird auf einem 1024×1024 Center-Crop gemessen. Bei Montierungen mit starker Feldkrümmung (EQ ohne Korrektor) kann die Rand-FWHM deutlich schlechter sein. Das Tile-Grid berücksichtigt das nicht → Rand-Tiles können zu klein für die lokale PSF sein.
