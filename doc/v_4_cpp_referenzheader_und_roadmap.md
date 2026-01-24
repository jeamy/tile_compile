# v4‑konforme C++‑Referenz‑Header‑Definition und Implementierungs‑Roadmap

---

## Teil A – C++‑Referenz‑Header (v4‑konform, normativ)

Die folgenden Header definieren **verbindliche Schnittstellen**. Abweichungen verändern die Semantik der Methodik v4.

### A.1 Zentrale Datentypen

```cpp
#pragma once
#include <vector>
#include <cstddef>

using FrameIndex = std::size_t;
using TileIndex  = std::size_t;

struct ImageView {
    const float* data;
    std::size_t width;
    std::size_t height;
    std::size_t stride;
};
```

---

### A.2 Globale Normalisierung (Pflicht)

```cpp
struct GlobalNormalizationResult {
    double background_level;   // B_f
    double scale_factor;       // 1 / B_f
};

GlobalNormalizationResult
normalize_global_frame(
    ImageView& frame,
    const ImageView& background_mask
);
```

**Semantik:**
* exakt einmal pro Frame
* linear
* vor jeder Metrikberechnung

---

### A.3 Globale Frame‑Metriken

```cpp
struct GlobalMetrics {
    double background;   // B_f
    double noise;        // σ_f
    double gradient;     // E_f
};

GlobalMetrics compute_global_metrics(
    const ImageView& normalized_frame,
    const ImageView& background_mask
);
```

---

### A.4 Globaler Qualitätsindex

```cpp
struct GlobalQuality {
    double Q_f;   // clipped to [-3, +3]
    double G_f;   // exp(Q_f)
};

GlobalQuality compute_global_quality(
    const std::vector<GlobalMetrics>& metrics_all_frames,
    FrameIndex f
);
```

**Hinweis:**
MAD‑Normalisierung erfolgt **über alle Frames**.

---

### A.5 Seeing‑adaptive Tile‑Geometrie

```cpp
struct TileGeometry {
    std::size_t tile_size;
    std::size_t overlap;
    std::size_t step;
};

TileGeometry derive_tile_geometry(
    double median_fwhm,
    std::size_t image_width,
    std::size_t image_height
);
```

---

### A.6 Lokale Tile‑Metriken

```cpp
struct LocalStarMetrics {
    double fwhm;
    double roundness;
    double contrast;
};

struct LocalStructureMetrics {
    double gradient_energy;
    double noise;
    double background;
};

struct LocalQuality {
    double Q_local;   // clipped [-3, +3]
    double L_ft;      // exp(Q_local)
};
```

```cpp
LocalQuality compute_local_star_quality(
    const std::vector<LocalStarMetrics>& tile_metrics_all_frames,
    FrameIndex f
);

LocalQuality compute_local_structure_quality(
    const std::vector<LocalStructureMetrics>& tile_metrics_all_frames,
    FrameIndex f
);
```

---

### A.7 Effektives Gewicht

```cpp
double compute_effective_weight(
    const GlobalQuality& global,
    const LocalQuality& local
);
```

**Definition:**
W_f,t = G_f · L_f,t

---

### A.8 Tile‑Rekonstruktion

```cpp
struct TileReconstructionResult {
    std::vector<float> pixels;
    double weight_sum;
};

TileReconstructionResult reconstruct_tile(
    TileIndex tile,
    const std::vector<ImageView>& frames,
    const std::vector<double>& effective_weights,
    double epsilon
);
```

---

### A.9 Zustandsvektor & Clusterung

```cpp
struct FrameStateVector {
    double G_f;
    double mean_Q_tile;
    double var_Q_tile;
    double background;
    double noise;
};
```

```cpp
std::vector<int> cluster_frames_by_state(
    const std::vector<FrameStateVector>& states,
    std::size_t k
);
```

---

### A.10 Synthetische Qualitätsframes

```cpp
ImageView reconstruct_synthetic_frame(
    int cluster_id,
    const std::vector<ImageView>& frames,
    const std::vector<TileReconstructionResult>& tiles
);
```

---

### A.11 Validierung & Abbruch

```cpp
struct ValidationResult {
    bool success;
    double fwhm_improvement_percent;
    double tile_weight_variance;
};
```

```cpp
ValidationResult validate_run(
    const ImageView& reference_stack,
    const ImageView& reconstructed_image
);
```

---

## Teil B – Detaillierte Implementierungs‑Roadmap (ohne Aufwandsschätzung)

### Phase 0 – Absicherung der Semantik
* Trennung: roh vs. global normalisiert
* Durchsetzung: „keine Metrik vor Normalisierung“
* numerische Guards (ε, Clipping)

---

### Phase 1 – Globale Ebene
1. robuste Hintergrundmaskierung
2. globale Normalisierung
3. Berechnung B_f, σ_f, E_f
4. MAD‑Normalisierung
5. Berechnung Q_f und G_f

**Exit‑Kriterium:** sinnvolle Streuung von G_f

---

### Phase 2 – Seeing & Tile‑Geometrie
6. Sternselektion
7. FWHM‑Verteilung
8. adaptive Tile‑Größe + Overlap
9. deterministische Tile‑Indizierung

---

### Phase 3 – Lokale Qualität
10. Stern‑Tiles: FWHM / Rundheit / Kontrast
11. Struktur‑Tiles: E/σ / Hintergrund
12. getrennte Q_local‑Berechnung
13. Clipping und Stabilisierung

---

### Phase 4 – Rekonstruktion
14. effektive Gewichte W_f,t
15. gewichtete Tile‑Rekonstruktion
16. Fensterfunktion + Overlap‑Add
17. Fallback‑Regeln

---

### Phase 5 – Zustandsmodell
18. Aufbau der Frame‑Zustandsvektoren
19. Feature‑Standardisierung
20. Clusterung (k‑means / GMM)
21. Validierung der Cluster

---

### Phase 6 – Synthetische Frames
22. Rekonstruktion pro Cluster
23. Speicherung synthetischer Frames
24. finales lineares Stacking

---

### Phase 7 – Validierung & Abbruch
25. FWHM‑Vergleich
26. Tile‑Artefakt‑Analyse
27. Hintergrund‑RMS‑Vergleich
28. kontrollierter Abbruch oder Erfolg

---

## Abschluss

Diese Header‑Definitionen und die Roadmap bilden zusammen eine **vollständige, überprüfbare und v4‑konforme Referenz**.
Sie sind geeignet für:

* Refactoring des bestehenden C++‑Ports
* Reviewer‑kritische Diskussionen
* parallele Implementierung mehrerer Module

---

