# Phase 5: Metriken und Clustering

## Ziel

Portierung der Metriken-Berechnung und Clustering-Algorithmen.

**Geschätzte Dauer**: 2-3 Wochen

---

## 5.1 Metriken (metrics/metrics.hpp)

### Header

```cpp
#pragma once

#include "tile_compile/core/types.hpp"
#include <map>
#include <optional>
#include <vector>

namespace tile_compile::metrics {

// Wiener-Filter für Tile-Rauschunterdrückung
Matrix2Df wiener_tile_filter(
    const Matrix2Df& tile,
    float sigma,
    float snr_tile,
    float q_struct_tile,
    bool is_star_tile,
    float snr_threshold = 5.0f,
    float q_min = -0.5f,
    float eps = 1e-12f
);

// Globale Metriken pro Frame (Methodik v3 §3.2)
struct GlobalMetrics {
    std::vector<float> background_level;  // B_f
    std::vector<float> noise_level;       // σ_f
    std::vector<float> gradient_energy;   // E_f
    std::vector<float> Q_f;               // Qualitätsindex vor Clamp
    std::vector<float> Q_f_clamped;       // Nach Clamp
    std::vector<float> G_f_c;             // exp(Q_f_clamped)
    
    struct Weights {
        float alpha;  // background
        float beta;   // noise
        float gamma;  // gradient
    } weights;
    
    int n_frames;
};

struct GlobalMetricsConfig {
    float alpha = 0.4f;   // background weight
    float beta = 0.3f;    // noise weight
    float gamma = 0.3f;   // gradient weight
    std::pair<float, float> clamp_range = {-3.0f, 3.0f};
};

GlobalMetrics calculate_global_metrics(
    const std::vector<Matrix2Df>& frames,
    const GlobalMetricsConfig& config = {}
);

// Tile-Metriken pro Frame (Methodik v3 §3.4)
struct TileMetrics {
    std::vector<float> fwhm;
    std::vector<float> roundness;
    std::vector<float> contrast;
    std::vector<float> background_level;
    std::vector<float> noise_level;
    std::vector<float> gradient_energy;
};

class TileMetricsCalculator {
public:
    TileMetricsCalculator(int tile_size = 64, float overlap = 0.25f);
    
    TileMetrics calculate(
        const Matrix2Df& frame,
        const Matrix2Db* valid_mask = nullptr
    );
    
    int tile_count(int height, int width) const;

private:
    int tile_size_;
    float overlap_;
    int step_;
    
    float calculate_fwhm(const Matrix2Df& tile, const Matrix2Db* mask);
    float calculate_roundness(const Matrix2Df& tile, const Matrix2Db* mask);
    float calculate_contrast(const Matrix2Df& tile, const Matrix2Db* mask);
    float calculate_gradient_energy(const Matrix2Df& tile, const Matrix2Db* mask);
    std::pair<float, float> calculate_background_noise(
        const Matrix2Df& tile, const Matrix2Db* mask
    );
    
    Matrix2Df highpass_filter(const Matrix2Df& tile);
    Matrix2Df box_blur(const Matrix2Df& tile, int kernel_size);
};

// Kanal-Metriken
struct ChannelMetrics {
    GlobalMetrics global;
    TileMetrics tiles;
};

std::map<std::string, ChannelMetrics> compute_channel_metrics(
    const std::map<std::string, std::vector<Matrix2Df>>& channels,
    const GlobalMetricsConfig& config = {}
);

} // namespace tile_compile::metrics
```

### Implementierung (Auszug)

```cpp
#include "tile_compile/metrics/metrics.hpp"
#include "tile_compile/core/utils.hpp"
#include <cmath>
#include <algorithm>
#include <complex>

namespace tile_compile::metrics {

Matrix2Df wiener_tile_filter(
    const Matrix2Df& tile,
    float sigma,
    float snr_tile,
    float q_struct_tile,
    bool is_star_tile,
    float snr_threshold,
    float q_min,
    float eps
) {
    // Nicht filtern wenn Stern-Tile oder hoher SNR
    if (is_star_tile) return tile;
    if (snr_tile >= snr_threshold) return tile;
    if (q_struct_tile <= q_min) return tile;
    
    int h = tile.rows();
    int w = tile.cols();
    
    // Symmetrisches Padding
    int pad_h = h / 4;
    int pad_w = w / 4;
    
    Matrix2Df padded(h + 2 * pad_h, w + 2 * pad_w);
    // Padding mit Spiegelung
    for (int y = 0; y < padded.rows(); ++y) {
        for (int x = 0; x < padded.cols(); ++x) {
            int sy = y - pad_h;
            int sx = x - pad_w;
            
            // Spiegelung an Rändern
            if (sy < 0) sy = -sy;
            if (sy >= h) sy = 2 * h - sy - 2;
            if (sx < 0) sx = -sx;
            if (sx >= w) sx = 2 * w - sx - 2;
            
            sy = std::clamp(sy, 0, h - 1);
            sx = std::clamp(sx, 0, w - 1);
            
            padded(y, x) = tile(sy, sx);
        }
    }
    
    // FFT (vereinfachte Implementierung mit Eigen)
    // Für Produktion: FFTW oder OpenCV DFT verwenden
    Eigen::FFT<float> fft;
    // ... FFT-Implementierung ...
    
    // Wiener-Filter: H(k) = max(|F(k)|² - σ², 0) / |F(k)|²
    float sigma_sq = sigma * sigma;
    
    // ... Filter anwenden ...
    
    // Zurück-Croppen
    Matrix2Df result = filtered.block(pad_h, pad_w, h, w);
    
    return result;
}

GlobalMetrics calculate_global_metrics(
    const std::vector<Matrix2Df>& frames,
    const GlobalMetricsConfig& config
) {
    GlobalMetrics result;
    int n = frames.size();
    result.n_frames = n;
    
    result.background_level.resize(n);
    result.noise_level.resize(n);
    result.gradient_energy.resize(n);
    result.Q_f.resize(n);
    result.Q_f_clamped.resize(n);
    result.G_f_c.resize(n);
    
    // Rohe Metriken berechnen
    for (int i = 0; i < n; ++i) {
        const auto& frame = frames[i];
        
        result.background_level[i] = core::compute_median(frame);
        result.noise_level[i] = frame.array().abs().mean();  // Vereinfacht
        result.gradient_energy[i] = calculate_gradient_energy(frame);
    }
    
    // Z-Score-Normalisierung
    auto normalize = [](std::vector<float>& v) {
        float mean = std::accumulate(v.begin(), v.end(), 0.0f) / v.size();
        float sq_sum = std::inner_product(v.begin(), v.end(), v.begin(), 0.0f);
        float std = std::sqrt(sq_sum / v.size() - mean * mean);
        
        if (std < 1e-12f) {
            std::fill(v.begin(), v.end(), 0.0f);
            return;
        }
        
        for (auto& x : v) {
            x = (x - mean) / std;
        }
    };
    
    std::vector<float> B_tilde = result.background_level;
    std::vector<float> sigma_tilde = result.noise_level;
    std::vector<float> E_tilde = result.gradient_energy;
    
    normalize(B_tilde);
    normalize(sigma_tilde);
    normalize(E_tilde);
    
    // Qualitätsindex: Q_f = α(-B̃) + β(-σ̃) + γẼ
    float alpha = config.alpha;
    float beta = config.beta;
    float gamma = config.gamma;
    
    for (int i = 0; i < n; ++i) {
        result.Q_f[i] = alpha * (-B_tilde[i]) + 
                        beta * (-sigma_tilde[i]) + 
                        gamma * E_tilde[i];
        
        // Clamp
        result.Q_f_clamped[i] = std::clamp(
            result.Q_f[i], 
            config.clamp_range.first, 
            config.clamp_range.second
        );
        
        // G_f = exp(Q_f_clamped)
        result.G_f_c[i] = std::exp(result.Q_f_clamped[i]);
    }
    
    result.weights = {alpha, beta, gamma};
    
    return result;
}

float calculate_gradient_energy(const Matrix2Df& frame) {
    int h = frame.rows();
    int w = frame.cols();
    
    float sum = 0.0f;
    int count = 0;
    
    for (int y = 1; y < h - 1; ++y) {
        for (int x = 1; x < w - 1; ++x) {
            float gx = frame(y, x + 1) - frame(y, x - 1);
            float gy = frame(y + 1, x) - frame(y - 1, x);
            sum += std::sqrt(gx * gx + gy * gy);
            ++count;
        }
    }
    
    return count > 0 ? sum / count : 0.0f;
}

} // namespace tile_compile::metrics
```

---

## 5.2 Clustering (clustering/clustering.hpp)

### Header

```cpp
#pragma once

#include "tile_compile/core/types.hpp"
#include "tile_compile/metrics/metrics.hpp"
#include <map>
#include <optional>
#include <variant>
#include <vector>

namespace tile_compile::clustering {

struct ClusteringResult {
    std::vector<int> cluster_labels;
    std::vector<std::vector<float>> cluster_centers;
    int n_clusters;
    float silhouette_score;
    std::string method;  // "kmeans" | "quantile_fallback"
    
    struct ClusterStats {
        int size;
        std::vector<float> mean_vector;
        std::vector<float> std_vector;
        std::vector<float> min_vector;
        std::vector<float> max_vector;
    };
    std::map<int, ClusterStats> cluster_stats;
};

struct ClusteringConfig {
    std::pair<int, int> cluster_count_range = {5, 30};
    bool use_silhouette = false;
    int fallback_quantiles = 15;
};

// K-Means Clustering
class KMeans {
public:
    KMeans(int n_clusters, int max_iters = 100, int n_init = 10, 
           unsigned int random_state = 42);
    
    std::pair<std::vector<int>, Eigen::MatrixXf> fit_predict(
        const Eigen::MatrixXf& data
    );

private:
    int n_clusters_;
    int max_iters_;
    int n_init_;
    unsigned int random_state_;
    
    std::pair<std::vector<int>, Eigen::MatrixXf> fit_once(
        const Eigen::MatrixXf& data,
        std::mt19937& rng
    );
    
    float compute_inertia(
        const Eigen::MatrixXf& data,
        const std::vector<int>& labels,
        const Eigen::MatrixXf& centers
    );
};

// Standard-Scaler
class StandardScaler {
public:
    Eigen::MatrixXf fit_transform(const Eigen::MatrixXf& data);
    Eigen::MatrixXf transform(const Eigen::MatrixXf& data) const;
    Eigen::MatrixXf inverse_transform(const Eigen::MatrixXf& data) const;

private:
    Eigen::VectorXf mean_;
    Eigen::VectorXf std_;
};

// State-Clustering (Methodik v3 §3.7)
class StateClustering {
public:
    static ClusteringResult cluster_frames(
        const std::vector<Matrix2Df>& frames,
        const metrics::ChannelMetrics& metrics,
        const ClusteringConfig& config = {}
    );
    
    static ClusteringResult cluster_frames_quantile_fallback(
        const std::vector<Matrix2Df>& frames,
        const metrics::ChannelMetrics& metrics,
        const ClusteringConfig& config = {}
    );

private:
    static Eigen::MatrixXf compute_state_vectors(
        const std::vector<Matrix2Df>& frames,
        const metrics::ChannelMetrics& metrics
    );
    
    static std::pair<int, float> find_optimal_k(
        const Eigen::MatrixXf& data,
        int k_min,
        int k_max
    );
    
    static std::map<int, ClusteringResult::ClusterStats> compute_cluster_stats(
        const Eigen::MatrixXf& state_vectors,
        const std::vector<int>& labels
    );
};

// Silhouette-Score
float silhouette_score(
    const Eigen::MatrixXf& data,
    const std::vector<int>& labels
);

// Kanal-Clustering
std::map<std::string, ClusteringResult> cluster_channels(
    const std::map<std::string, std::vector<Matrix2Df>>& channels,
    const std::map<std::string, metrics::ChannelMetrics>& metrics,
    const ClusteringConfig& config = {}
);

} // namespace tile_compile::clustering
```

### K-Means Implementierung

```cpp
#include "tile_compile/clustering/clustering.hpp"
#include <random>
#include <limits>

namespace tile_compile::clustering {

KMeans::KMeans(int n_clusters, int max_iters, int n_init, unsigned int random_state)
    : n_clusters_(n_clusters)
    , max_iters_(max_iters)
    , n_init_(n_init)
    , random_state_(random_state)
{}

std::pair<std::vector<int>, Eigen::MatrixXf> KMeans::fit_predict(
    const Eigen::MatrixXf& data
) {
    std::mt19937 rng(random_state_);
    
    std::vector<int> best_labels;
    Eigen::MatrixXf best_centers;
    float best_inertia = std::numeric_limits<float>::max();
    
    for (int init = 0; init < n_init_; ++init) {
        auto [labels, centers] = fit_once(data, rng);
        float inertia = compute_inertia(data, labels, centers);
        
        if (inertia < best_inertia) {
            best_inertia = inertia;
            best_labels = labels;
            best_centers = centers;
        }
    }
    
    return {best_labels, best_centers};
}

std::pair<std::vector<int>, Eigen::MatrixXf> KMeans::fit_once(
    const Eigen::MatrixXf& data,
    std::mt19937& rng
) {
    int n_samples = data.rows();
    int n_features = data.cols();
    
    // K-Means++ Initialisierung
    Eigen::MatrixXf centers(n_clusters_, n_features);
    std::vector<int> chosen_indices;
    
    std::uniform_int_distribution<int> first_dist(0, n_samples - 1);
    chosen_indices.push_back(first_dist(rng));
    centers.row(0) = data.row(chosen_indices[0]);
    
    for (int k = 1; k < n_clusters_; ++k) {
        // Distanzen zu nächstem Center berechnen
        Eigen::VectorXf min_dists(n_samples);
        for (int i = 0; i < n_samples; ++i) {
            float min_d = std::numeric_limits<float>::max();
            for (int j = 0; j < k; ++j) {
                float d = (data.row(i) - centers.row(j)).squaredNorm();
                min_d = std::min(min_d, d);
            }
            min_dists(i) = min_d;
        }
        
        // Gewichtete Auswahl
        std::discrete_distribution<int> dist(
            min_dists.data(), 
            min_dists.data() + n_samples
        );
        int idx = dist(rng);
        chosen_indices.push_back(idx);
        centers.row(k) = data.row(idx);
    }
    
    // Lloyd's Algorithmus
    std::vector<int> labels(n_samples);
    
    for (int iter = 0; iter < max_iters_; ++iter) {
        // Zuweisungsschritt
        bool changed = false;
        for (int i = 0; i < n_samples; ++i) {
            int best_k = 0;
            float best_d = std::numeric_limits<float>::max();
            
            for (int k = 0; k < n_clusters_; ++k) {
                float d = (data.row(i) - centers.row(k)).squaredNorm();
                if (d < best_d) {
                    best_d = d;
                    best_k = k;
                }
            }
            
            if (labels[i] != best_k) {
                labels[i] = best_k;
                changed = true;
            }
        }
        
        if (!changed) break;
        
        // Update-Schritt
        centers.setZero();
        std::vector<int> counts(n_clusters_, 0);
        
        for (int i = 0; i < n_samples; ++i) {
            centers.row(labels[i]) += data.row(i);
            counts[labels[i]]++;
        }
        
        for (int k = 0; k < n_clusters_; ++k) {
            if (counts[k] > 0) {
                centers.row(k) /= counts[k];
            }
        }
    }
    
    return {labels, centers};
}

float KMeans::compute_inertia(
    const Eigen::MatrixXf& data,
    const std::vector<int>& labels,
    const Eigen::MatrixXf& centers
) {
    float inertia = 0.0f;
    for (int i = 0; i < data.rows(); ++i) {
        inertia += (data.row(i) - centers.row(labels[i])).squaredNorm();
    }
    return inertia;
}

// State-Clustering
ClusteringResult StateClustering::cluster_frames(
    const std::vector<Matrix2Df>& frames,
    const metrics::ChannelMetrics& metrics,
    const ClusteringConfig& config
) {
    // State-Vektoren berechnen
    Eigen::MatrixXf state_vectors = compute_state_vectors(frames, metrics);
    int n_frames = state_vectors.rows();
    
    if (n_frames <= 1) {
        return {
            std::vector<int>(n_frames, 0),
            {},
            1,
            -1.0f,
            "kmeans",
            {}
        };
    }
    
    // Skalieren
    StandardScaler scaler;
    Eigen::MatrixXf scaled = scaler.fit_transform(state_vectors);
    
    // K bestimmen (Methodik v3: K = clip(floor(N/10), K_min, K_max))
    int k_min = config.cluster_count_range.first;
    int k_max = config.cluster_count_range.second;
    int k_default = std::clamp(n_frames / 10, k_min, k_max);
    
    int final_k;
    float sil_score = -1.0f;
    
    if (config.use_silhouette) {
        auto [best_k, best_sil] = find_optimal_k(scaled, k_min, k_max);
        final_k = best_k;
        sil_score = best_sil;
    } else {
        final_k = k_default;
    }
    
    // Clustering
    KMeans kmeans(final_k);
    auto [labels, centers] = kmeans.fit_predict(scaled);
    
    // Statistiken
    auto stats = compute_cluster_stats(state_vectors, labels);
    
    // Centers zurück-transformieren
    std::vector<std::vector<float>> centers_list;
    for (int k = 0; k < centers.rows(); ++k) {
        std::vector<float> row(centers.cols());
        for (int j = 0; j < centers.cols(); ++j) {
            row[j] = centers(k, j);
        }
        centers_list.push_back(row);
    }
    
    return {
        labels,
        centers_list,
        final_k,
        sil_score,
        "kmeans",
        stats
    };
}

Eigen::MatrixXf StateClustering::compute_state_vectors(
    const std::vector<Matrix2Df>& frames,
    const metrics::ChannelMetrics& metrics
) {
    int n = frames.size();
    Eigen::MatrixXf state_vectors(n, 5);
    
    const auto& gm = metrics.global;
    const auto& tm = metrics.tiles;
    
    for (int i = 0; i < n; ++i) {
        // Global quality
        float g_f_c = (i < gm.G_f_c.size()) ? gm.G_f_c[i] : 0.0f;
        
        // Local tile metrics (Mittelwert und Varianz)
        float q_local_mean = 0.0f;
        float q_local_var = 0.0f;
        // ... berechnen aus tm ...
        
        // Frame-Level
        float bg = (i < gm.background_level.size()) ? gm.background_level[i] : 0.0f;
        float noise = (i < gm.noise_level.size()) ? gm.noise_level[i] : 0.0f;
        
        state_vectors(i, 0) = g_f_c;
        state_vectors(i, 1) = q_local_mean;
        state_vectors(i, 2) = q_local_var;
        state_vectors(i, 3) = bg;
        state_vectors(i, 4) = noise;
    }
    
    return state_vectors;
}

} // namespace tile_compile::clustering
```

---

## 5.3 Tests

```cpp
// tests/clustering/test_clustering.cpp

#include <catch2/catch_test_macros.hpp>
#include "tile_compile/clustering/clustering.hpp"

using namespace tile_compile::clustering;

TEST_CASE("KMeans basic", "[clustering]") {
    // Einfache 2D-Daten mit 3 Clustern
    Eigen::MatrixXf data(9, 2);
    data << 0, 0,
            0.1, 0.1,
            -0.1, 0.1,
            5, 5,
            5.1, 5.1,
            4.9, 5.1,
            10, 0,
            10.1, 0.1,
            9.9, -0.1;
    
    KMeans kmeans(3);
    auto [labels, centers] = kmeans.fit_predict(data);
    
    REQUIRE(labels.size() == 9);
    REQUIRE(centers.rows() == 3);
    
    // Punkte im gleichen Cluster sollten gleiches Label haben
    REQUIRE(labels[0] == labels[1]);
    REQUIRE(labels[0] == labels[2]);
    REQUIRE(labels[3] == labels[4]);
    REQUIRE(labels[3] == labels[5]);
    REQUIRE(labels[6] == labels[7]);
    REQUIRE(labels[6] == labels[8]);
    
    // Verschiedene Cluster sollten verschiedene Labels haben
    REQUIRE(labels[0] != labels[3]);
    REQUIRE(labels[0] != labels[6]);
    REQUIRE(labels[3] != labels[6]);
}

TEST_CASE("StandardScaler", "[clustering]") {
    Eigen::MatrixXf data(4, 2);
    data << 1, 10,
            2, 20,
            3, 30,
            4, 40;
    
    StandardScaler scaler;
    auto scaled = scaler.fit_transform(data);
    
    // Mittelwert sollte ~0 sein
    REQUIRE(std::abs(scaled.col(0).mean()) < 1e-5f);
    REQUIRE(std::abs(scaled.col(1).mean()) < 1e-5f);
    
    // Standardabweichung sollte ~1 sein
    float std0 = std::sqrt((scaled.col(0).array() - scaled.col(0).mean()).square().mean());
    float std1 = std::sqrt((scaled.col(1).array() - scaled.col(1).mean()).square().mean());
    REQUIRE(std::abs(std0 - 1.0f) < 0.1f);
    REQUIRE(std::abs(std1 - 1.0f) < 0.1f);
}
```

---

## Checkliste Phase 5

- [ ] metrics.hpp Header erstellt
- [ ] `wiener_tile_filter()` implementiert
- [ ] `calculate_global_metrics()` implementiert
- [ ] `TileMetricsCalculator` implementiert
- [ ] clustering.hpp Header erstellt
- [ ] `KMeans` Klasse implementiert
- [ ] `StandardScaler` Klasse implementiert
- [ ] `StateClustering` implementiert
- [ ] `silhouette_score()` implementiert
- [ ] `cluster_channels()` implementiert
- [ ] Unit-Tests geschrieben
- [ ] Integration mit Phase 4 getestet
