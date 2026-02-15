# REGISTRATION — Kaskadierte globale Registrierung + Pre-Warping

> **C++ Implementierung:** `global_registration.cpp`, `registration.cpp`, `runner_main.cpp`
> **Phase-Enum:** `Phase::REGISTRATION` (gefolgt von eigener `Phase::PREWARP`)

## Übersicht

Die Registrierung richtet alle Frames geometrisch auf einen Referenz-Frame aus. Die C++ Implementierung verwendet eine **6-stufige Registrierungskaskade** mit robusten Fallbacks, die auch bei schwierigen Bedingungen (wenige Sterne, Star Trails durch Feldrotation, Nebel/Wolken) funktioniert. Anschließend werden alle Frames in der **eigenen Pipeline-Phase `PREWARP`** vollständig auf Bildauflösung vorgewrapt, bevor lokale Tile-Metriken berechnet werden.

**Kernprinzip:** Keine Frame-Selektion. Jeder Frame wird behalten, auch bei fehlgeschlagener Registrierung (Identity-Warp mit CC=0).

```
┌──────────────────────────────────────────────────────────┐
│  Für jeden Frame f ≠ ref:                                │
│                                                          │
│  1. Lade + normalisiere Frame                            │
│  2. Downsample 2× (CFA: Green-Proxy, Mono: Mean)        │
│  3. Kaskade (6 Stufen):                                  │
│     ├─ Triangle Star Matching    → OK? → accept          │
│     ├─ Trail Endpoint Matching   → OK? → accept  [NEU]   │
│     ├─ AKAZE Feature Match       → OK? → accept          │
│     ├─ Robust Phase+ECC (Multi)  → OK? → accept  [NEU]   │
│     ├─ Hybrid Phase+ECC          → OK? → accept          │
│     └─ Fallback: Identity Warp (CC=0)                    │
│  4. Skaliere Translation auf Vollauflösung               │
│  5. Pre-warp ganzes Bild (CFA-aware bei OSC)             │
└──────────────────────────────────────────────────────────┘
```

## 1. Referenz-Frame-Auswahl

```cpp
// Wähle Frame mit höchstem globalem Gewicht G_f
float best_w = -1.0f;
for (int i = 0; i < frame_metrics.size(); ++i) {
    float w = global_weights[i];
    if (w > best_w) { best_w = w; global_ref_idx = i; }
}
// Fallback: höchster quality_score, dann Mitte
```

- **Primär:** Frame mit höchstem `G_f` (globales Gewicht aus Phase 4)
- **Fallback 1:** Frame mit höchstem `quality_score`
- **Fallback 2:** Mittlerer Frame (`frames.size() / 2`)
- Der Referenz-Frame erhält Identity-Warp und CC=1.0

## 2. Downsample für Registrierung

```cpp
Matrix2Df ref_reg = (detected_mode == ColorMode::OSC)
    ? image::cfa_green_proxy_downsample2x2(ref_full, detected_bayer_str)
    : registration::downsample2x2_mean(ref_full);
```

| Modus | Downsample-Methode | Faktor | Begründung |
|-------|-------------------|--------|------------|
| **OSC** | CFA Green-Proxy 2×2 | 2× | Nutzt nur G-Pixel, kein Farb-Crosstalk |
| **MONO** | Mean 2×2 | 2× | Einfaches Downsampling |

- **Speedup:** ~4× weniger Pixel für Registrierung
- `global_reg_scale` speichert den Skalierungsfaktor (full_height / reg_height)
- Translationen (tx, ty) werden nach Registrierung mit `global_reg_scale` auf Vollauflösung skaliert

## 3. Registrierungskaskade (6 Stufen)

Für jeden Frame wird die Kaskade sequentiell durchlaufen, bis eine Methode erfolgreich ist:

### Stufe 1: Triangle Star Matching (Primär)

```cpp
RegistrationResult rr = registration::triangle_star_matching(
    mov_p, ref_p, allow_rotation, star_topk, star_min_inliers, star_inlier_tol_px);
```

- **Rotationsinvariant**: Verwendet Dreiecke aus Sternpositionen (Astroalign-Stil)
- Bildet Dreiecke aus Top-30 Sternen, vergleicht Seitenverhältnisse (invariant gegenüber Rotation/Skalierung)
- RANSAC-Konsens über alle Sterne mit `star_min_inliers` und `star_inlier_tol_px`
- **Konfiguration:** `star_topk` (Top-K Sterne), `star_min_inliers`, `star_inlier_tol_px`
- **Ideal für:** Alt/Az-Montierungen mit Feldrotation, ≥6 Sterne
- **Keine Rotationslimits** (entfernt wegen Alt/Az nahe Pol mit >20° Rotation)

### Stufe 2: Trail Endpoint Matching (Star Trails)

```cpp
rr = registration::trail_endpoint_registration(
    mov_p, ref_p, allow_rotation, star_topk, star_min_inliers,
    star_inlier_tol_px, star_dist_bin_px);
```

- **Für Star Trails** bei Feldrotation (Alt/Az) oder langen Belichtungen
- **Morphologischer Top-Hat** (15×15 Ellipse) extrahiert helle dünne Strukturen (Trails)
- **Contour-Analyse** findet die am weitesten entfernten Punkte jedes Trails (= Endpunkte)
- **Brightness-weighted Centroid** verfeinert die Endpunkt-Position sub-pixel-genau
- Kombiniert Trail-Endpunkte + reguläre Sterne für robusteres Matching
- Pair-Distance Similarity Matching mit relaxierten Schwellenwerten:
  - `inlier_tol_px × 2` (doppelte Toleranz, da Endpunkte ungenauer als Sternzentren)
  - `min_inliers / 2` (weniger Inlier nötig, da weniger Punkte verfügbar)
- **Ideal für:** Wolken/Nebel mit wenigen Sternen (<20), Feldrotation

### Stufe 3: AKAZE Feature Matching

```cpp
rr = registration::feature_registration_similarity(mov_p, ref_p, allow_rotation);
```

- **Feature-basiert**: AKAZE Keypoints + Descriptor Matching + RANSAC
- **Rotationsinvariant**: Keine Rotationslimits
- Braucht mind. 8 Feature-Matches, behält Top 30%
- `estimateAffinePartial2D` mit RANSAC → Similarity-Transform
- **Fallback** wenn zu wenige Sterne (z.B. dichte Nebel, Wolken)

### Stufe 4: Robust Multi-Scale Phase+ECC (Gradient-robust)

```cpp
rr = registration::robust_phase_ecc(mov_p, ref_p, allow_rotation);
```

- **Laplacian-of-Gaussian Vorverarbeitung**: Entfernt niederfrequente Gradienten (Nebel, Wolken, Lichtverschmutzung) und bewahrt hochfrequente Strukturen (Sterne, Kanten)
- **3-Level Coarse-to-Fine Pyramide**: 4× → 2× → 1× Auflösung
  - Gröbstes Level: Phase-Correlation (Translation) + Log-Polar DFT (Rotation)
  - Jedes Level: ECC-Refinement (100–200 Iterationen)
- **Ideal für:** Starke Nebel/Wolken-Gradienten + große Rotation

### Stufe 5: Hybrid Phase-Correlation + ECC (Original)

```cpp
rr = registration::hybrid_phase_ecc(mov_p, ref_p, allow_rotation);
```

- **Phase-Correlation** für grobe Translation, dann **ECC-Refinement**
- Arbeitet auf Rohpixeln (ohne Gradient-Preprocessing)
- Akzeptiert nur wenn `correlation >= 0.15`
- Einfacher als Stufe 4, kann bei klaren Bildern ohne Gradienten besser konvergieren

### Stufe 6: Identity-Fallback

```cpp
out.warps_fullres[i] = identity_warp();
out.scores[i] = 0.0f;
out.success[i] = false;
```

- **v3-Regel:** Kein Frame wird ausgeschlossen
- CC=0 → niedrigstes effektives Gewicht, aber Frame bleibt in Pipeline
- In der Praxis: Frame mit Identity-Warp hat minimalen Einfluss auf Rekonstruktion
- Nur aktiv wenn `registration.fallback_to_identity: true` (Default)

## Kaskade — Entscheidungslogik

```
Sterne erkannt (≥6)?
├─ JA → Stufe 1 (Triangle) → Erfolg? → FERTIG
│       └─ NEIN → Stufe 2 (Trail Endpoints) → Erfolg? → FERTIG
│                 └─ NEIN → Stufe 3 (AKAZE) → Erfolg? → FERTIG
│                           └─ NEIN → Stufe 4 (Robust Phase+ECC) → Erfolg? → FERTIG
│                                     └─ NEIN → Stufe 5 (Hybrid Phase+ECC) → Erfolg? → FERTIG
│                                               └─ NEIN → Stufe 6 (Identity)
├─ NEIN (Star Trails) → Stufe 1 scheitert → Stufe 2 (Trail Endpoints) → ...
├─ NEIN (Nebel/Wolken) → Stufe 1–3 scheitern → Stufe 4 (Robust Phase+ECC) → ...
└─ NEIN (komplett leer) → Stufe 1–5 scheitern → Stufe 6 (Identity)
```

## 4. Warp-Skalierung

```cpp
WarpMatrix w_full = rr.warp;
w_full(0, 2) *= global_reg_scale;  // tx skalieren
w_full(1, 2) *= global_reg_scale;  // ty skalieren
```

Die Warp-Matrix wird auf halber Auflösung berechnet. Die Translationskomponenten (tx, ty) werden mit dem Skalierungsfaktor auf Vollauflösung hochskaliert. Die Rotations-/Affin-Komponenten (a00, a01, a10, a11) bleiben unverändert.

## 5. Pre-Warping (CFA-aware)

> Laufzeit-Sichtbarkeit: `phase_start(PREWARP)` / `phase_progress(PREWARP)` / `phase_end(PREWARP)`

**Kritischer Schritt** nach der Registrierung, vor der Tile-Extraktion:

```cpp
for (size_t fi = 0; fi < frames.size(); ++fi) {
    auto pair = load_frame_normalized(fi);
    Matrix2Df img = std::move(pair.first);
    const auto &w = global_frame_warps[fi];

    if (is_identity) {
        prewarped_frames[fi] = std::move(img);
    } else if (detected_mode == ColorMode::OSC) {
        prewarped_frames[fi] = image::warp_cfa_mosaic_via_subplanes(
            img, w, img.rows(), img.cols());
    } else {
        prewarped_frames[fi] = registration::apply_warp(img, w);
    }
}
```

### Warum Pre-Warping?

**Problem:** Rotation-Warps auf kleine Tile-ROIs (z.B. 64×64) verursachen **CFA-Pattern-Korruption** — `warpAffine` braucht Quell-Pixel außerhalb des Tile-Rands, die nicht existieren. Das Ergebnis sind sichtbare farbige Rechtecke.

**Lösung:** Alle Frames werden **vor** der Tile-Extraktion auf **voller Bildauflösung** gewarpt.

| Modus | Warp-Methode | CFA-Sicherheit |
|-------|-------------|----------------|
| **MONO** | `apply_warp()` (OpenCV warpAffine) | N/A |
| **OSC** | `warp_cfa_mosaic_via_subplanes()` | ✓ Keine Bayer-Phasen-Mischung |

`warp_cfa_mosaic_via_subplanes` zerlegt das CFA-Mosaik in 4 Subplanes (R, G1, G2, B), warpt jede separat und interleaved sie zurück.

## Konfigurationsparameter

| Parameter | Beschreibung | C++ Default |
|-----------|-------------|-------------|
| `registration.enabled` | Registrierung aktivieren | `true` |
| `registration.engine` | Primäre Engine | `triangle_star_matching` |
| `registration.allow_rotation` | Rotation erlauben (Alt/Az) | `true` |
| `registration.min_score` | Min. Correlation-Score | `0.05` |
| `registration.fallback_to_identity` | Identity-Fallback erlauben | `true` |
| `registration.star_topk` | Top-K Sterne für Matching | `120` |
| `registration.star_min_inliers` | Mindest-Inlier für Akzeptanz | `6` |
| `registration.star_inlier_tol_px` | Inlier-Toleranz in Pixel | `2.5` |
| `registration.star_dist_bin_px` | Distanz-Bin für Star Pairs | `2.5` |
| `output.write_registered_frames` | Registrierte Frames speichern | `false` |
| `output.write_global_registration` | Registration-Artifact schreiben | `true` |

## Artifact: `global_registration.json`

```json
{
  "num_frames": 100,
  "scale": 2.0,
  "ref_frame": 42,
  "cc": [1.0, 0.95, 0.87, ...],
  "warps": [
    {"a00": 1.0, "a01": 0.0, "tx": 0.0, "a10": 0.0, "a11": 1.0, "ty": 0.0},
    {"a00": 0.999, "a01": -0.012, "tx": 3.5, "a10": 0.012, "a11": 0.999, "ty": -1.2},
    ...
  ]
}
```

## Fehlerbehandlung

| Situation | Verhalten |
|-----------|-----------|
| Ref-Frame leer | `global_reg_status = "error"` |
| Frame leer | Identity-Warp, CC=0 |
| Alle 5 Methoden fehlgeschlagen | Identity-Warp, CC=0 (Frame bleibt!) |
| Exception in Registrierung | Gesamte Phase → "error", Pipeline weiter |
| Star Trails (Feldrotation) | Stufe 2 (Trail Endpoints) übernimmt |
| Nebel/Wolken (keine Sterne) | Stufe 4 (Robust Phase+ECC) übernimmt |

## Nächste Phasen

→ **Phase 2: CHANNEL_SPLIT** (Metadaten)  
→ **Phase 3: NORMALIZATION**  
→ **Phase 4: GLOBAL_METRICS**