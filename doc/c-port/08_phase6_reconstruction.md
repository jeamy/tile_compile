# Phase 6: Rekonstruktion und Synthetic Frames

## Ziel

Portierung der Tile-Rekonstruktion und Synthetic-Frame-Generierung.

**Geschätzte Dauer**: 2-3 Wochen

---

## 6.1 Rekonstruktion (reconstruction/reconstruction.hpp)

### Header

```cpp
#pragma once

#include "tile_compile/core/types.hpp"
#include "tile_compile/metrics/metrics.hpp"
#include <map>
#include <optional>
#include <vector>

namespace tile_compile::reconstruction {

constexpr float DEFAULT_EPSILON = 1e-10f;

struct ReconstructionResult {
    Matrix2Df reconstructed;
    std::vector<std::pair<int, int>> fallback_tiles;  // (y, x) Positionen
    int n_fallback;
    bool fallback_used;
};

class TileReconstructor {
public:
    TileReconstructor(
        int tile_size = 64,
        float overlap = 0.25f,
        float epsilon = DEFAULT_EPSILON
    );
    
    // Rekonstruiere einen Kanal (Methodik v3 §3.6)
    // I_t,c(p) = Σ_f W_f,t,c · I_f,c(p) / Σ_f W_f,t,c
    ReconstructionResult reconstruct_channel(
        const std::vector<Matrix2Df>& frames,
        const metrics::ChannelMetrics& metrics
    );

private:
    int tile_size_;
    float overlap_;
    float epsilon_;
    int step_;
    
    // Lokale Qualität L_f,t,c = exp(Q_local[f,t,c])
    VectorXf get_local_quality(
        const metrics::ChannelMetrics& metrics,
        int tile_idx,
        int n_frames
    );
    
    // Hann-Fenster für Tile-Blending
    Matrix2Df create_blending_window(int h, int w);
};

// Multi-Kanal-Rekonstruktion
struct ChannelReconstructionResult {
    std::map<std::string, ReconstructionResult> channels;
    int total_fallback_tiles;
    bool any_fallback;
};

ChannelReconstructionResult reconstruct_channels(
    const std::map<std::string, std::vector<Matrix2Df>>& channels,
    const std::map<std::string, metrics::ChannelMetrics>& metrics,
    int tile_size = 64,
    float overlap = 0.25f,
    float epsilon = DEFAULT_EPSILON
);

} // namespace tile_compile::reconstruction
```

### Implementierung

```cpp
#include "tile_compile/reconstruction/reconstruction.hpp"
#include <cmath>

namespace tile_compile::reconstruction {

TileReconstructor::TileReconstructor(int tile_size, float overlap, float epsilon)
    : tile_size_(tile_size)
    , overlap_(overlap)
    , epsilon_(epsilon)
    , step_(static_cast<int>(tile_size * (1.0f - overlap)))
{}

ReconstructionResult TileReconstructor::reconstruct_channel(
    const std::vector<Matrix2Df>& frames,
    const metrics::ChannelMetrics& metrics
) {
    ReconstructionResult result;
    result.n_fallback = 0;
    result.fallback_used = false;
    
    if (frames.empty()) {
        return result;
    }
    
    int h = frames[0].rows();
    int w = frames[0].cols();
    int n_frames = frames.size();
    
    // Output und Gewichtssumme initialisieren
    Matrix2Df output = Matrix2Df::Zero(h, w);
    Matrix2Df weight_sum = Matrix2Df::Zero(h, w);
    
    // Globale Qualitätsindizes G_f,c
    VectorXf G_f_c(n_frames);
    for (int f = 0; f < n_frames; ++f) {
        G_f_c(f) = (f < metrics.global.G_f_c.size()) 
                   ? metrics.global.G_f_c[f] 
                   : 1.0f;
    }
    
    // Blending-Fenster
    Matrix2Df window = create_blending_window(tile_size_, tile_size_);
    
    // Tiles verarbeiten
    int tile_idx = 0;
    for (int y = 0; y <= h - tile_size_; y += step_) {
        for (int x = 0; x <= w - tile_size_; x += step_) {
            // Tile-Stack aus allen Frames extrahieren
            std::vector<Matrix2Df> tile_stack(n_frames);
            for (int f = 0; f < n_frames; ++f) {
                tile_stack[f] = frames[f].block(y, x, tile_size_, tile_size_);
            }
            
            // Effektive Gewichte W_f,t,c = G_f,c · L_f,t,c
            VectorXf L_f_t_c = get_local_quality(metrics, tile_idx, n_frames);
            VectorXf W_f_t_c = G_f_c.array() * L_f_t_c.array();
            
            // Summe der Gewichte D_t,c
            float D_t_c = W_f_t_c.sum();
            
            // Tile rekonstruieren
            Matrix2Df tile_output = Matrix2Df::Zero(tile_size_, tile_size_);
            
            if (D_t_c >= epsilon_) {
                // Normale gewichtete Rekonstruktion
                for (int f = 0; f < n_frames; ++f) {
                    tile_output += tile_stack[f] * W_f_t_c(f);
                }
                tile_output /= D_t_c;
            } else {
                // Fallback: ungewichteter Mittelwert (Methodik v3 §3.6)
                for (int f = 0; f < n_frames; ++f) {
                    tile_output += tile_stack[f];
                }
                tile_output /= static_cast<float>(n_frames);
                result.fallback_tiles.push_back({y, x});
            }
            
            // Tile in Output blenden
            output.block(y, x, tile_size_, tile_size_) += 
                tile_output.array() * window.array();
            weight_sum.block(y, x, tile_size_, tile_size_) += window;
            
            ++tile_idx;
        }
    }
    
    // Normalisieren
    for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j) {
            if (weight_sum(i, j) > 0) {
                output(i, j) /= weight_sum(i, j);
            }
        }
    }
    
    result.reconstructed = output;
    result.n_fallback = result.fallback_tiles.size();
    result.fallback_used = result.n_fallback > 0;
    
    return result;
}

VectorXf TileReconstructor::get_local_quality(
    const metrics::ChannelMetrics& metrics,
    int tile_idx,
    int n_frames
) {
    VectorXf L_f_t_c = VectorXf::Ones(n_frames);
    
    // Q_local aus Tile-Metriken extrahieren
    // L_f,t,c = exp(clamp(Q_local[f,t,c], -3, 3))
    
    // Vereinfachte Implementierung: uniform wenn keine Daten
    // In vollständiger Implementierung: aus metrics.tiles extrahieren
    
    return L_f_t_c;
}

Matrix2Df TileReconstructor::create_blending_window(int h, int w) {
    // Hann-Fenster
    VectorXf y_window(h);
    VectorXf x_window(w);
    
    for (int i = 0; i < h; ++i) {
        y_window(i) = 0.5f * (1.0f - std::cos(2.0f * M_PI * i / (h - 1)));
    }
    for (int j = 0; j < w; ++j) {
        x_window(j) = 0.5f * (1.0f - std::cos(2.0f * M_PI * j / (w - 1)));
    }
    
    // Outer product und sqrt
    Matrix2Df window = y_window * x_window.transpose();
    return window.array().sqrt().matrix();
}

ChannelReconstructionResult reconstruct_channels(
    const std::map<std::string, std::vector<Matrix2Df>>& channels,
    const std::map<std::string, metrics::ChannelMetrics>& metrics,
    int tile_size,
    float overlap,
    float epsilon
) {
    ChannelReconstructionResult result;
    result.total_fallback_tiles = 0;
    result.any_fallback = false;
    
    TileReconstructor reconstructor(tile_size, overlap, epsilon);
    
    for (const auto& [name, frames] : channels) {
        auto it = metrics.find(name);
        if (it == metrics.end()) continue;
        
        auto channel_result = reconstructor.reconstruct_channel(frames, it->second);
        result.channels[name] = channel_result;
        result.total_fallback_tiles += channel_result.n_fallback;
        result.any_fallback = result.any_fallback || channel_result.fallback_used;
    }
    
    return result;
}

} // namespace tile_compile::reconstruction
```

---

## 6.2 Synthetic Frames (synthetic/synthetic.hpp)

### Header

```cpp
#pragma once

#include "tile_compile/core/types.hpp"
#include "tile_compile/metrics/metrics.hpp"
#include "tile_compile/clustering/clustering.hpp"
#include <map>
#include <optional>
#include <vector>

namespace tile_compile::synthetic {

struct SyntheticConfig {
    int frames_min = 15;
    int frames_max = 30;
    std::string weighting = "global";  // "global" | "tile_weighted"
};

class SyntheticFrameGenerator {
public:
    // Generiere synthetische Frames (Methodik v3 §3.8)
    // Ein Frame pro Cluster, gewichteter linearer Stack
    static std::vector<Matrix2Df> generate_synthetic_frames(
        const std::vector<Matrix2Df>& input_frames,
        const metrics::ChannelMetrics& metrics,
        const SyntheticConfig& config = {},
        const clustering::ClusteringResult* clustering_results = nullptr
    );

private:
    // Cluster-basierte Generierung
    static std::vector<Matrix2Df> generate_from_clusters(
        const std::vector<Matrix2Df>& frames,
        const metrics::ChannelMetrics& metrics,
        const clustering::ClusteringResult& clustering
    );
    
    // Quantil-basierte Fallback-Generierung
    static std::vector<Matrix2Df> generate_quantile_based(
        const std::vector<Matrix2Df>& frames,
        const metrics::ChannelMetrics& metrics,
        int n_synthetic
    );
    
    // Gewichteter Durchschnitt
    static Matrix2Df weighted_average(
        const std::vector<Matrix2Df>& frames,
        const VectorXf& weights
    );
};

// Multi-Kanal Synthetic Frame Generierung
std::map<std::string, std::vector<Matrix2Df>> generate_channel_synthetic_frames(
    const std::map<std::string, std::vector<Matrix2Df>>& channels,
    const std::map<std::string, metrics::ChannelMetrics>& metrics,
    const SyntheticConfig& config = {},
    const std::map<std::string, clustering::ClusteringResult>* clustering_results = nullptr
);

} // namespace tile_compile::synthetic
```

### Implementierung

```cpp
#include "tile_compile/synthetic/synthetic.hpp"
#include <algorithm>
#include <numeric>

namespace tile_compile::synthetic {

std::vector<Matrix2Df> SyntheticFrameGenerator::generate_synthetic_frames(
    const std::vector<Matrix2Df>& input_frames,
    const metrics::ChannelMetrics& metrics,
    const SyntheticConfig& config,
    const clustering::ClusteringResult* clustering_results
) {
    if (input_frames.empty()) {
        return {};
    }
    
    // Wenn Clustering-Ergebnisse vorhanden, cluster-basiert generieren
    if (clustering_results && !clustering_results->cluster_labels.empty()) {
        return generate_from_clusters(input_frames, metrics, *clustering_results);
    }
    
    // Fallback: Quantil-basiert
    int n_frames = input_frames.size();
    int n_synthetic = std::clamp(
        n_frames / 10,
        config.frames_min,
        config.frames_max
    );
    n_synthetic = std::max(n_synthetic, config.frames_min);
    
    if (n_frames < config.frames_min) {
        // Zu wenige Frames, einfachen Durchschnitt zurückgeben
        VectorXf uniform_weights = VectorXf::Ones(n_frames) / n_frames;
        return {weighted_average(input_frames, uniform_weights)};
    }
    
    return generate_quantile_based(input_frames, metrics, n_synthetic);
}

std::vector<Matrix2Df> SyntheticFrameGenerator::generate_from_clusters(
    const std::vector<Matrix2Df>& frames,
    const metrics::ChannelMetrics& metrics,
    const clustering::ClusteringResult& clustering
) {
    const auto& labels = clustering.cluster_labels;
    int n_frames = frames.size();
    
    if (labels.size() != n_frames) {
        return {};
    }
    
    // Globale Gewichte G_f,c
    VectorXf global_weights(n_frames);
    for (int f = 0; f < n_frames; ++f) {
        global_weights(f) = (f < metrics.global.G_f_c.size())
                            ? metrics.global.G_f_c[f]
                            : 1.0f;
    }
    
    // Unique Cluster-IDs
    std::set<int> unique_clusters(labels.begin(), labels.end());
    
    std::vector<Matrix2Df> synthetic_frames;
    
    for (int cluster_id : unique_clusters) {
        // Frames in diesem Cluster
        std::vector<int> cluster_indices;
        for (int i = 0; i < n_frames; ++i) {
            if (labels[i] == cluster_id) {
                cluster_indices.push_back(i);
            }
        }
        
        if (cluster_indices.empty()) continue;
        
        // Cluster-Frames und -Gewichte
        std::vector<Matrix2Df> cluster_frames;
        VectorXf cluster_weights(cluster_indices.size());
        
        for (size_t i = 0; i < cluster_indices.size(); ++i) {
            int idx = cluster_indices[i];
            cluster_frames.push_back(frames[idx]);
            cluster_weights(i) = global_weights(idx);
        }
        
        // Gewichte normalisieren
        float weight_sum = cluster_weights.sum();
        if (weight_sum > 1e-10f) {
            cluster_weights /= weight_sum;
        } else {
            cluster_weights.setConstant(1.0f / cluster_weights.size());
        }
        
        // Gewichteter Stack
        synthetic_frames.push_back(weighted_average(cluster_frames, cluster_weights));
    }
    
    return synthetic_frames;
}

std::vector<Matrix2Df> SyntheticFrameGenerator::generate_quantile_based(
    const std::vector<Matrix2Df>& frames,
    const metrics::ChannelMetrics& metrics,
    int n_synthetic
) {
    int n_frames = frames.size();
    
    // Globale Gewichte
    VectorXf global_weights(n_frames);
    for (int f = 0; f < n_frames; ++f) {
        global_weights(f) = (f < metrics.global.G_f_c.size())
                            ? metrics.global.G_f_c[f]
                            : 1.0f;
    }
    
    // Nach Gewicht sortieren
    std::vector<int> sorted_indices(n_frames);
    std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
    std::sort(sorted_indices.begin(), sorted_indices.end(),
              [&](int a, int b) { return global_weights(a) < global_weights(b); });
    
    // In n_synthetic Gruppen aufteilen
    std::vector<Matrix2Df> synthetic_frames;
    float group_size = static_cast<float>(n_frames) / n_synthetic;
    
    for (int i = 0; i < n_synthetic; ++i) {
        int start_idx = static_cast<int>(i * group_size);
        int end_idx = (i < n_synthetic - 1) 
                      ? static_cast<int>((i + 1) * group_size)
                      : n_frames;
        
        if (start_idx >= end_idx) continue;
        
        // Gruppe extrahieren
        std::vector<Matrix2Df> group_frames;
        VectorXf group_weights(end_idx - start_idx);
        
        for (int j = start_idx; j < end_idx; ++j) {
            int idx = sorted_indices[j];
            group_frames.push_back(frames[idx]);
            group_weights(j - start_idx) = global_weights(idx);
        }
        
        // Normalisieren
        float weight_sum = group_weights.sum();
        if (weight_sum > 1e-10f) {
            group_weights /= weight_sum;
        } else {
            group_weights.setConstant(1.0f / group_weights.size());
        }
        
        synthetic_frames.push_back(weighted_average(group_frames, group_weights));
    }
    
    return synthetic_frames;
}

Matrix2Df SyntheticFrameGenerator::weighted_average(
    const std::vector<Matrix2Df>& frames,
    const VectorXf& weights
) {
    if (frames.empty()) {
        return Matrix2Df();
    }
    
    int h = frames[0].rows();
    int w = frames[0].cols();
    Matrix2Df result = Matrix2Df::Zero(h, w);
    
    for (size_t i = 0; i < frames.size(); ++i) {
        result += frames[i] * weights(i);
    }
    
    return result;
}

std::map<std::string, std::vector<Matrix2Df>> generate_channel_synthetic_frames(
    const std::map<std::string, std::vector<Matrix2Df>>& channels,
    const std::map<std::string, metrics::ChannelMetrics>& metrics,
    const SyntheticConfig& config,
    const std::map<std::string, clustering::ClusteringResult>* clustering_results
) {
    std::map<std::string, std::vector<Matrix2Df>> result;
    
    for (const auto& [name, frames] : channels) {
        auto metrics_it = metrics.find(name);
        if (metrics_it == metrics.end()) continue;
        
        const clustering::ClusteringResult* channel_clustering = nullptr;
        if (clustering_results) {
            auto cluster_it = clustering_results->find(name);
            if (cluster_it != clustering_results->end()) {
                channel_clustering = &cluster_it->second;
            }
        }
        
        result[name] = SyntheticFrameGenerator::generate_synthetic_frames(
            frames,
            metrics_it->second,
            config,
            channel_clustering
        );
    }
    
    return result;
}

} // namespace tile_compile::synthetic
```

---

## 6.3 Sigma-Clipping (stacking/sigma_clipping.hpp)

```cpp
#pragma once

#include "tile_compile/core/types.hpp"
#include <optional>
#include <map>
#include <variant>

namespace tile_compile::stacking {

struct SigmaClipConfig {
    float sigma_low = 3.0f;
    float sigma_high = 3.0f;
    int max_iters = 3;
    float min_fraction = 0.5f;
    
    SigmaClipConfig clamp() const;
};

struct SigmaClipResult {
    Matrix2Df clipped_mean;
    Matrix2Db mask;  // true = kept, false = rejected
    
    struct Stats {
        int frames;
        int iterations;
        float kept_fraction;
        float rejected_fraction;
        float min_fraction;
        float sigma_low;
        float sigma_high;
        std::optional<std::string> error;
    } stats;
};

// Sigma-Clipping Stack entlang Achse 0
SigmaClipResult sigma_clip_stack(
    const std::vector<Matrix2Df>& frames,
    const SigmaClipConfig& cfg = {},
    const std::vector<Matrix2Db>* valid_masks = nullptr
);

// Einfacher Mittelwert-Stack
Matrix2Df simple_mean_stack(
    const std::vector<Matrix2Df>& frames,
    const std::vector<Matrix2Db>* valid_masks = nullptr
);

} // namespace tile_compile::stacking
```

---

## 6.4 Tests

```cpp
// tests/reconstruction/test_reconstruction.cpp

#include <catch2/catch_test_macros.hpp>
#include "tile_compile/reconstruction/reconstruction.hpp"

using namespace tile_compile::reconstruction;

TEST_CASE("TileReconstructor basic", "[reconstruction]") {
    // 3 identische Frames
    std::vector<Matrix2Df> frames(3);
    for (auto& f : frames) {
        f = Matrix2Df::Constant(128, 128, 100.0f);
    }
    
    metrics::ChannelMetrics metrics;
    metrics.global.G_f_c = {1.0f, 1.0f, 1.0f};
    
    TileReconstructor reconstructor(32, 0.25f);
    auto result = reconstructor.reconstruct_channel(frames, metrics);
    
    REQUIRE(result.reconstructed.rows() == 128);
    REQUIRE(result.reconstructed.cols() == 128);
    
    // Ergebnis sollte ~100 sein
    float mean = result.reconstructed.mean();
    REQUIRE(std::abs(mean - 100.0f) < 1.0f);
}

TEST_CASE("create_blending_window", "[reconstruction]") {
    TileReconstructor reconstructor(64, 0.25f);
    
    // Hann-Fenster sollte in der Mitte maximal sein
    // und an den Rändern gegen 0 gehen
}
```

---

## Checkliste Phase 6

- [ ] reconstruction.hpp Header erstellt
- [ ] `TileReconstructor` implementiert
- [ ] `reconstruct_channel()` implementiert
- [ ] `create_blending_window()` implementiert
- [ ] `reconstruct_channels()` implementiert
- [ ] synthetic.hpp Header erstellt
- [ ] `SyntheticFrameGenerator` implementiert
- [ ] `generate_from_clusters()` implementiert
- [ ] `generate_quantile_based()` implementiert
- [ ] `generate_channel_synthetic_frames()` implementiert
- [ ] sigma_clipping.hpp implementiert
- [ ] Unit-Tests geschrieben
- [ ] Integration mit Phase 5 getestet
