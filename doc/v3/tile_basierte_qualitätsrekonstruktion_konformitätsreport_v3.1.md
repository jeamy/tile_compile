# Tile‑Compile – Methodik‑v3.1‑Konformitätsreport

**Ziel:** Abgleich der Implementierung (Stand Branch `sigma-clipping`) mit
„Tile‑basierte Qualitätsrekonstruktion für DSO – Methodik v3.1 (2026‑01‑09)“.

Fokus: Pipeline `tile_compile_runner.py` → `runner/phases_impl.py` + `tile_compile_backend/*`.
Legacy‑Pfad in `tile_compile_runner_backup.py` wird als nicht produktiv betrachtet.

---

## 1. Pipeline‑Mapping

**Spezifikation (§3.1–3.9):**

1. Registrierung & geometrische Vereinheitlichung
2. Kanaltrennung
3. Globale lineare Normalisierung
4. Globale Frame‑Metriken
5. Tile‑Geometrie
6. Lokale Tile‑Metriken
7. Tile‑basierte Rekonstruktion
8. Zustandsbasierte Clusterung
9. Rekonstruktion synthetischer Frames
10. Finales lineares Stacking

**Implementierung:**

- Entry‑Point: `tile_compile_runner.py::main()` ruft
  - `runner/phases.run_phases()` → `runner/phases_impl.run_phases_impl(...)`

**Phasen in `phases_impl.py` (§3+4 der Spezifikation):**

| Phase‑ID | Name                  | Entspricht Spec‑Schritt                  |
|---------:|----------------------|------------------------------------------|
| 0        | `SCAN_INPUT`         | Vorbereitende Checks, Kalibration, §2   |
| 1        | `REGISTRATION`       | 1. Registrierung (Pfad A/B)             |
| 2        | `CHANNEL_SPLIT`      | 2. Kanaltrennung                         |
| 3        | `NORMALIZATION`      | 3. Globale Normalisierung                |
| 4        | `GLOBAL_METRICS`     | 4. Globale Metriken, §3.2               |
| 5        | `TILE_GRID`          | 5. Tile‑Geometrie, §3.3                 |
| 6        | `LOCAL_METRICS`      | 6. Lokale Tile‑Metriken, §3.4           |
| 7        | `TILE_RECONSTRUCTION`| 7. Tile‑Rekonstruktion, §3.5–3.6        |
| 8        | `STATE_CLUSTERING`   | 8. Zustandsbasierte Clusterung, §3.7    |
| 9        | `SYNTHETIC_FRAMES`   | 9. Synthetische Frames, §3.8            |
| 10       | `STACKING`           | 10. Finales Stacking, §3.8              |
| 11       | `DONE`               | Abschluss + Validierungsartefakte       |

**Bewertung:** Struktur passt exakt zu §3. Die eigentliche Konformität entscheidet sich in den Details der einzelnen Phasen.

---

## 2. Assumptions & globale Invarianten (§2)

### 2.1 Harte Annahmen

- **Linearität der Daten (§2.1)**
  - Es gibt aktuell **keine explizite Linearitäts‑Validierung** in `phases_impl.py`.
  - Ein (nicht aktiver) Validierungspfad existiert im Legacy‑Runner (`tile_compile_runner_backup.py::process_pipeline`, Aufruf `validate_frames_linearity(...)`), wird aber in der aktuellen Pipeline nicht verwendet.
  - **Abweichung:** Linearität wird vorausgesetzt, aber nicht überprüft; kein harter Abbruch bei Nichtlinearität wie in §2.1/§4 gefordert.

- **Keine Frame‑Selektion (§2.1, §3.6, Test 7)**
  - In der produktiven Pipeline werden:
    - alle `frames` nach Kalibration weitergereicht (`frames` wird bei Fehlern komplett verworfen, aber nie „best‑of“).
    - Globale Gewichte `G_f,c` werden für alle Frames berechnet (`GLOBAL_METRICS`).
    - Tile‑Rekonstruktion (`TILE_RECONSTRUCTION`) verwendet in der Normallogik alle Frames, Gewichte können jedoch numerisch praktisch gegen 0 gehen.
  - Kein Code, der z.B. die „besten 10 % Frames“ explizit auswählt.
  - **Konform im Geist der Methodik.** Abweichungen entstehen nur indirekt durch numerisch sehr kleine Gewichte (kein harter Cut, sondern kontinuierliche Abwertung).

- **Kanalgetrennte Verarbeitung (§2.1, §3.9, Test 6)**
  - Kanaltrennung in `CHANNEL_SPLIT` (`split_cfa_channels`, `split_rgb_frame`) und anschließende Verarbeitung in Schleifen über `"R","G","B"`.
  - Globale Metriken, Tile‑Metriken, Rekonstruktion, Clusterung und synthetische Frames werden pro Kanal berechnet; es gibt keine Stelle, an der R/G/B gemischt werden (außer beim Erzeugen von RGB‑Visualisierungen bzw. `syn_XXXXX.fits` als 3‑Plane‑Stack).
  - **Konform.**

### 2.2 Weiche Annahmen & Reduced Mode (§2.2–2.4)

- Implementiert in `runner/assumptions.py` + `phases_impl.py`:
  - Defaults:
    - `frames_min=50`, `frames_optimal=800`, `frames_reduced_threshold=200`
  - `is_reduced_mode(frame_count, assumptions)` aktiviert Reduced Mode, wenn `frame_count < frames_reduced_threshold`.
  - In `STATE_CLUSTERING`:
    - `reduced_mode` wird bestimmt aus `len(registered_files)` und `assumptions_cfg`.
    - Wenn `reduced_mode` **und** `reduced_mode_skip_clustering=True`, wird Clusterung übersprungen.
    - Sonst werden in Reduced Mode Cluster‑Range angepasst (`reduced_mode_cluster_range`), aber:
      - diese Range wird nur im **Quantil‑Fallback** genutzt, nicht im regulären KMeans‑Clustering (siehe §3.7 unten).

- **Graduelles Degradieren (§1.4, §2.4)**
  - Teilweise umgesetzt:
    - Warnungen/Abbrüche in Kalibration, Registration, etc. liefern detaillierte Events.
    - Reduced Mode führt zu:
      - optionalem Überspringen von Clusterung,
      - Fallback in `STACKING` auf rekonstruierte Kanäle, wenn keine synthetischen Frames verfügbar sind.
  - Aber: Es gibt **kein zentrales „Degradations‑Policy‑System“**, das systematisch zwischen WARN/DEGRADED/CRITICAL unterscheidet; stattdessen verteilt über viele `phase_end(..., "error"/"ok")`.

**Fazit §2:**
Weiche Annahmen & Reduced Mode sind erkennbar implementiert, aber Linearitäts‑Check und ein explizites graduelles Degradationsmodell fehlen bzw. sind nur rudimentär umgesetzt.

---

## 3. Phasen‑Detailanalyse vs. Methodik v3.1

### 3.1 Registrierung & Pfad A/B (§A, §B)

**Siril‑Pfad (Pfad A, empfohlen):**

- `REGISTRATION` Standardpfad (`reg_engine == "siril"`):
  - Siril‑Aufruf mit konfigurierbarem Script (`registration.siril_script`), Default `siril_register_osc.ssf`.
  - Validierung des Scripts gegen `siril_utils.validate_siril_script` (Policy‑Layer).
  - Output wird in `outputs/<reg_out_name>` als `reg_*.fit` o.Ä. kopiert.
- Methodische Anforderungen:
  - Debayer + Registrierung in Siril vor der Methodik – **erfüllt**, da der gemeinsame Kern erst ab `CHANNEL_SPLIT` greift.
  - `CHANNEL_SPLIT` erfolgt **nach** der Registrierung, wie gefordert.

**CFA‑Pfad (Pfad B, experimentell):**

- `REGISTRATION` mit `reg_engine == "opencv_cfa"`:
  - Luminanzberechnung aus CFA (`cfa_downsample_sum2x2`).
  - Referenzframe mit maximaler Sternanzahl.
  - ECC‑basierte Registrierung mit optionaler Rotation; initialer Translation‑Guess (`opencv_best_translation_init`).
  - CFA‑aware Warp (`warp_cfa_mosaic_via_subplanes`), der Subplanes getrennt transformiert und wieder in ein CFA‑Mosaik interleavt.
- Vergleich zu §B.2:
  - Eine einzige geometrische Transformation pro Frame (Translation/ECC), Interpolation CFA‑aware auf Subplanes – **in guter Übereinstimmung** mit B.2.1/B.2.2.
  - Registrierung basiert auf Luminanz aus realen CFA‑Samples → erfüllt §B.2.1.

**Abweichungen / TODO:**

- Die Siril‑Skripte selbst liegen außerhalb des Python‑Codes; deren inhaltliche Konformität (kein Drizzle, keine Zusatzgewichtung, etc.) wird nur indirekt via `validate_siril_script` geprüft (Policy nicht hier dokumentiert).
- Pfad B ist im Code vorhanden, aber in der Praxis vermutlich als „advanced/experimental“ anzusehen; robuste Tests/Validierung für diesen Pfad sind im Repo eher dünn.

---

### 3.2 Kanaltrennung (§3 Schritt 2, §A.2)

- Implementierung: `CHANNEL_SPLIT` in `phases_impl.py`:
  - Für CFA: `split_cfa_channels(mosaic, bayer_pattern)` in `runner/image_processing.py`, doppelte G‑Samples werden gemittelt.
  - Für RGB‑FITS: `split_rgb_frame`.
  - Für „exotische“ Layouts: Fallback auf Graustufen‑Tripel (R=G=B).

**Konformität:**

- Kanaltrennung **nach** Registrierung (Siril oder opencv_cfa) – entspricht §A.2.2 und §B.2.3.
- Keine kanalübergreifenden Operationen danach – **konform**.

---

### 3.3 Globale Normalisierung (§3.1)

Der kritische Teil ist, dass:

1. `B_f,c` vor Normalisierung auf **Rohdaten** berechnet wird,
2. Normalisierung linear und global ist,
3. `σ_f,c` und `E_f,c` auf normalisierten Daten berechnet werden.

**Implementierung:**

- `NORMALIZATION`:
  - Pro Kanal (`per_channel=True` Default):
    - Pass 1:
      - Pro Frame: `median(frame)` → `medians`.
      - Speicherung in `pre_norm_backgrounds[ch]` (das entspricht `B_f,c` auf Rohdaten).
    - Pass 2:
      - Für jeden Frame: divisive Normalisierung via `normalize_frame(...)` im Modus `"background"` (I/B), oder additive Normalisierung im Legacy‑Modus.
  - Globaler Modus (`per_channel=False`) unterstützt, aber ebenfalls 2‑Pass‑Schema mit `pre_norm_backgrounds` beibehalten.

- `GLOBAL_METRICS`:
  - Verwendet `pre_norm_backgrounds[ch]` direkt als `B_f,c`.
  - Liest anschließend die (bereits normalisierten) Kanal‑Frames neu ein und berechnet:
    - `σ_f,c` via `np.std(f)`,
    - `E_f,c` via `np.mean(np.hypot(*np.gradient(f)))`.

**Bewertung:**

- Reihenfolge (B vor Normalisierung, σ/E nach Normalisierung) ist **korrekt** umgesetzt.
- Normalisierung ist global, linear, einmalig und kanalgetrennt – **konform**.
- Fallback, falls `pre_norm_backgrounds` fehlen (z.B. Phase 3 übersprungen), berechnet B_f,c aus den aktuell gespeicherten Daten; hier wäre formal eine Warnung/Abbruch bzgl. Methodik‑Verletzung wünschenswert.

---

### 3.4 Globale Metriken & Gewichte (§3.2)

**Spezifikation:**

- MAD‑Normalisierung pro Metrik:
  \(\tilde{x} = (x - \mathrm{median}(x)) / (1.4826 \cdot \mathrm{MAD}(x))\)
- Qualitätsindex:
  \(Q_{f,c} = \alpha(-\tilde B_{f,c}) + \beta(-\tilde\sigma_{f,c}) + \gamma \tilde E_{f,c}\),
  mit α+β+γ=1 und Default (0.4,0.3,0.3).
- Clamping \(Q_{f,c} \in [-3,3]\), dann \(G_{f,c} = \exp(Q_{f,c})\).
- Optionale adaptive Gewichte basierend auf Varianzen der Metriken.

**Implementierung:**

- MAD‑Normalisierung:
  - `_norm_mad(vals)` in `GLOBAL_METRICS` implementiert exakt die MAD‑Formel (inkl. 1.4826‑Faktor).
- Default‑Gewichte:
  - `w_bg=0.4`, `w_noise=0.3`, `w_grad=0.3`.
- Normierungssumme:
  - Prüft explizit `abs(weight_sum - 1.0) > 1e-6` → harter `phase_end(..., "error")` bei Verletzung.
  - Entspricht Test 1 aus §6.
- Adaptive Gewichtung:
  - Bei `global_metrics.adaptive_weights: true`:
    - Varianzen für B, σ, E werden berechnet.
    - α’,β’,γ’ ∝ Var(B),Var(σ),Var(E).
    - Clipping auf [0.1,0.7], Renormalisierung auf Summe 1.0.
  - Exakt wie in §3.2 beschrieben.
- Clamping & Exponential:
  - `q_f_clamped = np.clip(q, -3.0, 3.0)`
  - `G_f_c = exp(q_f_clamped)`
  - Erfüllt Test 2 aus §6.

**Fazit:** Umsetzung der globalen Metriklogik ist **sehr eng am Text** der Methodik v3.1 und methodisch konform.

---

### 3.5 Tile‑Geometrie (§3.3)

**Spezifikation:**

- `T_0 = s·F`
- `T = floor(clip(T_0, T_min, floor(min(W,H)/D)))`
- `O = floor(o·T)`, `S = T - O`
- Grenzprüfungen: F>0 (F=3 Default), T_min≥16, S>0 mit o‑Fallback, min(W,H) ≥ T etc.

**Implementierung:**

- In `TILE_GRID` (phases_impl.py):
  - Liest `min_size`, `max_divisor`, `overlap_fraction`, `size_factor` aus `cfg["tile"]`, mit Defaults (64, 6, 0.25, 32).
  - Validiert `0 ≤ overlap ≤ 0.5` (Test 4 teilweise).
  - Schätzt FWHM (robust, Multi‑Frame, ECC‑unabhängig) via Gradienten‑Heuristik; Fallback F=3.0.
  - Erzwingt `min_tile_size = max(16, min_tile_size)` ⇒ T_min‑Untergrenze.
  - Berechnet T_0, T_max, prüft Kleinstbildfälle (`min(W,H) < T_min`), berechnet O, S und korrigiert S≤0 via Reset `overlap=0.25`.
  - Übergibt anschließend `fwhm`, `size_factor`, `min_size`, `max_divisor`, `overlap_fraction` an `tile_compile_backend.tile_grid.generate_multi_channel_grid`, welches für alle Kanäle denselben Konfig‑Satz benutzt und somit dieselbe Gittergeometrie (Koordinaten) erzeugt.

**Abweichungen / Hinweise:**

- Defaultwerte (`size_factor=32`, `min_size=64`, `max_divisor=6`) weichen von den in `tile_grid.py` hinterlegten Defaults (s=8, T_min=32, D=4) ab, aber die Methodik schreibt keine konkreten Defaultzahlen fest, sondern nur die **Formeln und Schranken**.
  → **Methodisch zulässig**, aber evtl. zu dokumentieren.
- Es gibt **keinen expliziten Testfall Tile‑Size‑Monotonie (T(F1) ≤ T(F2))**, aber die implementierte Formel ist monoton in F bei fixen Parametern.

---

### 3.6 Lokale Tile‑Metriken (§3.4)

**Spezifikation (Stern‑Tiles):**

- Metriken: FWHM_t,f,c, R_t,f,c, C_t,f,c.
- MAD‑Normalisierung pro Metrik.
- Q\_star = 0.6·(−FWHM̃) + 0.2·R̃ + 0.2·C̃
- Clamping auf [−3,3], dann L_f,t,c = exp(Q_local).

**Implementierung:**

- Lokale Metriken in `LOCAL_METRICS`:
  - Verwendet `TileMetricsCalculator` aus `tile_compile_backend.metrics` nur als Datenlieferant (`fwhm`, `roundness`, `contrast`).
  - Normierung:
    - `_mad_norm` für FWHM, roundness, contrast.
  - Q_local:
    - `q_raw = w_fwhm*(-FWHM̃) + w_round*R̃ + w_con*C̃` mit Defaults (0.6, 0.2, 0.2).
    - `q = clip(q_raw, -3, 3)`.
    - `L = exp(q)`.
- Tile‑Aggregatstatistik (mean/var pro Tile, L‑Mean/Var) wird berechnet und für Heatmaps verwendet.

**Konformität:**

- Formel, Normalisierung und Clamping decken sich mit §3.4 und Anhang A.5.
- Strikte Trennung zwischen globalen und lokalen Skalen ist gewahrt.
- Keine explizite Unterscheidung „Stern‑Tiles vs. Struktur‑Tiles“ im Code (einheitliches Schema); das weicht von der Detaillierung in §3.4 (unterschiedliche Formeln) ab.

**Abweichung:**
Die Spezifikation unterscheidet formal zwischen Stern‑ und Struktur‑Tiles mit zwei unterschiedlichen Q‑Formeln. Die Implementation verwendet nur den **Stern‑Modus** (FWHM/R/Contrast) und derzeit kein explizites Struktur‑Tile‑Modell.

---

### 3.7 Effektive Gewichte & Tile‑Rekonstruktion (§3.5–3.6)

**Spezifikation:**

- \(W_{f,t,c} = G_{f,c} \cdot L_{f,t,c}\).
- \(I_{t,c}(p) = \frac{\sum_f W_{f,t,c} I_{f,c}(p)}{\sum_f W_{f,t,c}}\).
- Fallback bei D_t,c < ε: ungewichtetes Mittel über alle Frames, Tile als Ganzes, `fallback_used=true`.
- Overlap‑Add mit 2D‑Hanning‑Fenster, vorherige Tile‑Normalisierung (Hintergrund subtrahieren & normieren).

**Implementierung:**

Es existieren **zwei** Implementierungsstränge:

1. Allgemeiner Backend‑Rekonstruktor (`tile_compile_backend.reconstruction.TileReconstructor`), der sehr nah an der Spezifikation sitzt, aber im aktuellen `phases_impl` **nicht verwendet** wird.
2. Pipeline‑eigene Rekonstruktion in `TILE_RECONSTRUCTION`:

   - Pro Kanal:
     - `gfc` = Liste der G_f,c (global weights).
     - `l_local` = L_f,t,c aus LOCAL_METRICS (oder von Disk nachgeladen).
     - Tile‑Gitter wird aus `grid_cfg` übernommen (tile_size, step).
   - Für jedes Tile t und Frame f:
     - `l_f_t` = lokales Gewicht aus L.
     - `w_f_t = g_f * l_f_t`.
     - `tile` = Ausschnitt aus Frame (y0:y1,x0:x1).
     - Hintergrundsubtraktion und Normierung wie in §3.6 (Median/Median(|·|)).
     - `hann_2d`‑Fenster (Hanning) angewandt.
     - Akkumulation von `out += tile * w_f_t * window` und `weight_sum += w_f_t * window`.
   - Nach Durchlauf aller Frames:
     - `low_weight_mask = weight_sum < epsilon`.
     - Fallback:
       - `fallback_mean = ungewichtetes Mittel aller Frames` (pro Pixel).
       - Für low‑weight‑Pixel: `out = fallback_mean`; sonst `out/weight_sum`.

**Konformität & Abweichung:**

- Formell:
  - W_f,t,c = G_f,c · L_f,t,c → **konform**.
  - Hanning‑Fenster + Tile‑Normalisierung wie im Anhang – **konform**.
- Fallback:
  - Spezifikation formuliert Fallback pro Tile D_t,c.
  - Implementierung verwendet **pixelbasierten** Fallback (`weight_sum < ε` pro Pixel), was in der Praxis ähnliche, aber nicht identische Entscheidungen produziert; zusätzlich wird ein **globaler** ungewichteter Mittelwert über alle Frames für den Fallback genutzt, statt Tile‑weisen Mitteln.
  - Das Prinzip („Fallback = ungewichtetes Mittel, keine NaNs/Infs, keine Frame‑Selektion“) bleibt erhalten; die Lokalität des Fallbacks ist aber **gröber** als spezifiziert.

**Bewertung:**
Kernidee und Stabilitätsanforderungen sind erfüllt, mit einer (methodisch eher kleinen) Abweichung bei der Granularität der Fallback‑Entscheidung (Pixel statt Tile).

---

### 3.8 Zustandsbasierte Clusterung (§3.7)

**Spezifikation:**

- Zustandsvektor \(v_f = (G_f, \langle Q_{tile}\rangle, Var(Q_{tile}), B_f, \sigma_f)\).
- Clusterung der Frames (nicht Tiles).
- Dynamische Clusterzahl \(K = clip(\lfloor N/10 \rfloor, K_{min}=5, K_{max}=30)\).
- Reduced Mode: bei 50–199 Frames ggf. K reduziert (5–10) oder Clusterung übersprungen.

**Implementierung:**

- Backend `tile_compile_backend.clustering`:
  - `StateClustering._compute_state_vectors(...)` baut Vektoren aus:
    - G_f,c,
    - mean/var von Q_local pro Frame,
    - background_level, noise_level.
  - `_find_optimal_clustering` sucht bestes K über Silhouette‑Score im Bereich `[min_clusters, max_clusters]` (Default 15–30).
  - `cluster_channels(...)` cluster’t **pro Kanal** separat, nicht global.

- Pipeline `STATE_CLUSTERING` (`phases_impl.py`):
  - Baut ein Dummy‑`channels_for_clustering`‐Dict mit Platzhalter‑Arrays (RAM‑freundlich).
  - Ruft `cluster_channels(...)` auf.
  - Bei Exception:
    - **Quantil‑Fallback**:
      - Setzt `n_quantiles = clip(N//10, K_min=5, K_max=30)` (genau wie Spec).
      - In Reduced Mode wird `n_quantiles` auf `reduced_mode_cluster_range[1]` gedeckelt (z.B. 10).
      - Clusterung pro Kanal anhand von G_f,c‑Quantilen.

**Abweichungen:**

1. **Primär‑Clusterung** via Silhouette‑optimiertem KMeans:
   - Spezifikation schreibt **normativ** K=clip(N/10,5,30) vor.
   - Implementierung wählt K datengetrieben **und** kanalweise:
     - Für R,G,B unterschiedliche K möglich.
   - Nur Fallback folgt exakt der Spezifikation für K.
2. Reduced Mode:
   - `reduced_mode_cluster_range` wird im Quantil‑Fallback korrekt verwendet.
   - Für den regulären KMeans‑Pfad gibt es aktuell **keine direkte Verkopplung** zu `reduced_mode_cluster_range`.
3. Pro‑Kanal‑Clusterung:
   - Die Spec ist hier mehrdeutig; sie spricht von „Frames“, nicht „Frames pro Kanal“.
   - Implementierung clustert unabhängig für R,G,B. Dies kann sinnvoll sein, entspricht aber nicht exakt dem beschriebenen globalen Zustandsvektor, wenn man diesen streng pro Frame (über alle Kanäle) definiert.

**Fazit:** Clusterung ist **funktional sehr ähnlich**, aber die Norm „K=N/10 (geclippt)“ wird nur im Fallback eingehalten, nicht im Hauptpfad.

---

### 3.9 Synthetische Frames & finales Stacking (§3.8)

**Spezifikation:**

- Pro Cluster k:
  - \(S_{k,c} = \frac{\sum_{f \in Cluster_k} G_{f,c} I_{f,c}}{\sum_{f \in Cluster_k} G_{f,c}}\) mit I = original Frames.
- Final:
  - \(R_c = \frac{1}{K} \sum_k S_{k,c}\), linear, **ungewichtet**, kein Drizzle.

**Implementierung `SYNTHETIC_FRAMES`:**

- `_synthetic_from_files(ch_name)` in `phases_impl.py`:
  - Holt `labels` für Kanal ch aus `clustering_results` (oder baut sie via Quantil‑Fallback).
  - Holt G_f,c als Gewichte, normalisiert auf Summe 1 pro Cluster.
  - Für jeden Cluster:
    - Summiert `frame * weight` über alle Frames mit dem Clusterlabel.
    - Teilt durch Σw – das ist exakt die Formel für S_{k,c}.
  - Schreibt pro Kanal `syn{ch}_{cluster_id}.fits`.
- Danach:
  - Baut kombinierte RGB‑Frames `syn_{i:05d}.fits` aus passenden R/G/B‑synthetischen Frames.
- Finales Stacking (`STACKING`):
  - Stacking‑Input:
    - Normalfall: synthetische RGB‑Frames `syn_*.fits` (durch Pattern‑Matching).
    - Fallback: rekonstruierte Kanäle (`reconstructed_*.fits` / `reconstructed_rgb.fits`) im Reduced Mode mit übersprungener Clusterung oder ohne synthetische Frames.
  - Stacking erfolgt in Siril (Script `stacking.siril_script`, Default `siril_stack_average.ssf`).
  - Wenn Default‑Script und Methode `average`:
    - Nach dem Siril‑Run wird das Stackergebnis noch einmal durch `n_stack` geteilt (`data_f = data / n_stack`), um ein garantiertes Mittel zu erzwingen (abhängig von der Siril‑Script‑Semantik).

**Konformität & Abweichungen:**

- Synthetische Frames:
  - Per‑Cluster‑Formel exakt wie in §3.8, Gewichte = G_f,c, keine weiteren Faktoren → **konform**.
- Finales Stacking:
  - Formal entspricht das Ergebnis (Summe über Cluster / K) dem beschriebenen linearen Stack, vorausgesetzt, dass:
    - Siril‑Script tatsächlich eine reine Summe oder ein ähnliches lineares Aggregat liefert.
    - Die nachträgliche Division durch `n_stack` korrekt skaliert.
  - Dass RGB‑Stacks `syn_XXXXX.fits` als Eingabe dienen, ist akzeptabel, weil die Methodik Kombination (RGB/LRGB) explizit **außerhalb** der Kernmethodik sieht. Hier wird das Stacking jedoch in einem Schritt für alle Kanäle durchgeführt; per‑Kanal‑Stacking wird an Siril delegiert.

**Fazit:**
Die Generierung synthetischer Frames ist methodik‑konform; das finale Stacking ist wahrscheinlich konform, hängt aber in Details von den Siril‑Skripten ab, auf die der Python‑Code keinen vollständigen Zwang ausübt.

---

## 4. Validierung & Abbruch (§4, §6)

Die Spec definiert explizite Testfälle (§6) und Validierungsplots (Anhang B).

**Umgesetzte Tests / Regeln:**

1. **Gewichtsnormierung global (Test 1):**
   - α+β+γ=1 wird hart geprüft, sonst Fehler in `GLOBAL_METRICS`.

2. **Clamping vor Exponentialfunktion (Test 2):**
   - Global: Q_f,c wird auf [−3,3] geclippt, bevor `exp`.
   - Lokal: Q_local wird ebenfalls geclippt (sowohl im Pipelinecode als auch im Backend‑Rekonstruktor).
   - Tile‑Rekonstruktor‑Fallback nutzt ebenfalls Clamping.

3. **Overlap‑Konsistenz (Test 4):**
   - `overlap_fraction` wird in [0,0.5] validiert; O=floor(o·T), S=T−O, beide ganzzahlig.
   - Kein expliziter Test, aber implizit durch Formeln.

4. **Low‑weight Tile Fallback (Test 5):**
   - Backend‑Rekonstruktor implementiert exact D_t,c<ε→Unweighted Mean pro Tile.
   - Pipeline‑Rekonstruktion implementiert Pixel‑basierten Fallback auf globales Mittel.
   - In beiden Fällen werden NaN/Inf vermieden.

5. **Kanaltrennung / keine Kanal‑Kopplung (Test 6):**
   - Wie in §2.1 diskutiert: erfüllt.

6. **Keine Frame‑Selektion (Test 7):**
   - Keine explizite Testfunktion; Implementation nutzt aber stets alle Frames (Gewichte können sehr klein werden, aber kein harter Cut).

7. **Determinismus (Test 8):**
   - Kein expliziter Test; deterministisches Verhalten hängt von externen Faktoren (OpenCV, Siril) ab.
   - Event‑Logging + Artefakt‑Erzeugung ist umfangreich, aber es gibt keinen formalen Vergleichslauf im Code.

**Validierungsartefakte (Anhang B):**

- Es werden zahlreiche Plots erzeugt:
  - `global_weight_timeseries.png`, `global_weight_hist.png`.
  - `normalization_background_timeseries.png`.
  - Tile‑Heatmaps (`tile_quality_heatmap_*.png`, `tile_weight_heatmap_*.png`, etc.).
  - Rekonstruktions‑Artefakte (`reconstruction_preview_*.png`, `reconstruction_weight_sum_*.png`, `reconstruction_absdiff_vs_mean_*.png`).
  - Registrierungs‑Plots, Clustering‑Zusammenfassungen, etc.
- Diese decken große Teile von Anhang B ab (FWHM‑Feldkarte ist eher nah an den Tile‑Heatmaps; Differenzbilder vs. klassischem Stack gibt es ansatzweise als „recon vs. mean(first n)“).

**Abweichungen / Lücken:**

- Es gibt keine zentral kodierte Prüfung der **Erfolgskriterien** (§4.1) wie „median FWHM ↓ ≥ 5–10 %“; aktuell wird das primär als Diagnose‑Artefakt bereitgestellt, nicht als harte Gate‑Condition.
- Kein automatisiertes Testskript, das alle §6‑Testfälle durchspielt; `test_methodik_v3_conformance.py` deckt nur:
  - Clamping,
  - quantile‑basierten Fallback für Clustering (Backend),
  - Normalisierung der Gewichte (Default 1/3,1/3,1/3 im Test – nicht identisch mit den Pipeline‑Defaults 0.4/0.3/0.3).

---

## 5. Zusammenfassung der wichtigsten Abweichungen

**Größere / methodisch relevante Abweichungen:**

1. **Linearitätsprüfung (§2.1, §4):**
   - Kein aktiver Check im produktiven Pfad; Non‑Linearität führt nicht automatisch zu Abbruch/Degradierung.

2. **Clusteranzahl K (§3.7):**
   - Normative Formel K=clip(N/10,5,30) wird nur im Quantil‑Fallback verwendet.
   - Hauptpfad (KMeans) nutzt Silhouette‑basiertes K im Bereich [15,30] und pro Kanal – nicht exakt der Spezifikation entsprechend.

3. **Unterscheidung Stern‑ vs. Struktur‑Tiles (§3.4):**
   - Implementation verwendet nur den Stern‑Tile‑Ansatz (FWHM/R/Contrast) für alle Tiles, kein eigener Struktur‑Modus mit (E/σ,B).

4. **Tile‑Fallback‑Granularität (§3.6, Test 5):**
   - Formal geforderter Fallback pro Tile (D_t,c<ε) vs. implementiertem Pixel‑weisen Fallback auf globales Mittelbild.
   - Prinzip bleibt erhalten, Lokalität weicht ab.

5. **Explizite Erfolgskriterien / Abbruchregeln (§4):**
   - Es existiert kein Automaton, der z.B. FWHM‑Verbesserung, Varianz der Tile‑Gewichte etc. als harten Pass/Fail‑Kriterium evaluiert; die Methodik‑Erfolgskriterien sind daher eher „diagnostisch“ als normativ in Code gegossen.

**Kleinere / eher implementierungstechnische Abweichungen:**

- Default‑Parameter für Tile‑Geometrie (s,T_min,D) sind sinnvolle, aber nicht in der Methodik festgelegte Werte.
- Reduced Mode:
  - Teilweise korrekt (Skip von Clusterung & synthetischen Frames), aber nicht alle Varianten aus §2.4 sind abgebildet (z.B. „Clusterung trotzdem aktiv mit 5–10 Clustern“ nur im Fallbackpfad berücksichtigt).
- Einige Backend‑Module (`tile_compile_backend.metrics.MetricsCalculator`, `normalization.LinearNormalizer`, `reconstruction.TileReconstructor`, `synthetic.SyntheticFrameGenerator`) implementieren frühere bzw. alternative Varianten der Methodik und werden im aktuellen Pipeline‑Pfad nur partiell oder gar nicht genutzt. Sie sind **nicht** maßgeblich für die Konformität der produktiven Pipeline.

---

## 6. Empfehlungen

Für eine „strikt normative“ v3.1‑Konformität wären die folgenden Anpassungen sinnvoll:

1. **Linearity‑Gate aktivieren:**
   - Einen Minimal‑Check auf Nichtlinearität (z.B. via Referenzkurven, Testframes oder Heuristiken) in `SCAN_INPUT` einbauen; bei klarer Nichtlinearität → Abbruch mit erklärender Meldung.

2. **K‑Logik vereinheitlichen:**
   - Silhouette‑basiertes KMeans nur als Option; Standard: K = clip(N/10,5,30) wie in §3.7, einheitlich zwischen Backend und Pipeline.
   - `reduced_mode_cluster_range` sollte im **Hauptpfad** der Clusterung direkt verwendet werden.

3. **Struktur‑Tile‑Modus ergänzen:**
   - Zusätzlich zu FWHM/R/Contrast einen Pfad implementieren, der Tiles ohne ausreichende Sterne mit (E/σ,B) bewertet und die alternative Q\_struct‑Formel nutzt.

4. **Tile‑Fallback lokaler gestalten:**
   - Fallback‑Entscheidung auf Ebene einzelner Tiles (Summe W_f,t,c) anstatt pixelweise, und pro‑Tile Fallback mit lokalem statt globalem Mittelbild.

5. **Erfolgskriterien automatisieren:**
   - Post‑Run‑Validierung, die die in §4 und Anhang B beschriebenen Kriterien formell prüft und den Run‑Status (`OK`/`WARN`/`FAIL`) daran festmacht.

---

**Kurzantwort auf die ursprüngliche Frage:**
Die aktuelle Pipeline‑Implementierung in `tile_compile_runner.py` → `runner/phases_impl.py` plus den genutzten Backend‑Modulen bildet die Methodik v3.1 **weitgehend korrekt** ab, insbesondere bei globalen & lokalen Metriken, Tile‑Geometrie, Gewichten, synthetischen Frames und Fallbacks.
Es gibt jedoch einige **klar benennbare Abweichungen** (Linearity‑Check, Cluster‑K‑Logik, fehlender expliziter Struktur‑Tile‑Modus, Tile‑Fallback‑Granularität), die für eine streng normative Konformität nachgezogen werden sollten.
