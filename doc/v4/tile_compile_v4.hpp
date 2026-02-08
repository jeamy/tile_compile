#pragma once
/**
 * tile_compile_v4.hpp
 *
 * v4-konformer Referenz-Header
 * Single Source of Truth für die tile-basierte Qualitätsrekonstruktion
 *
 * Jede Implementierung, die diesem Header folgt, ist strukturell v4-konform.
 */

#include <vector>
#include <cstddef>
#include <stdexcept>

/* =========================
 * Basisdatentypen
 * ========================= */

struct ImageView {
    float* data;
    int width;
    int height;
    int stride;
};

struct Tile {
    int x;
    int y;
    int width;
    int height;
};

struct TileResult {
    ImageView image;
    double weight_sum;
};

struct FrameIndex {
    std::size_t idx;
};

/* =========================
 * STAGE 1: Globale Normalisierung
 * ========================= */

struct GlobalNormalizationResult {
    ImageView normalized;
    double background_level;
};

class GlobalNormalizer {
public:
    virtual ~GlobalNormalizer() = default;

    virtual GlobalNormalizationResult
    normalize(const ImageView& registered_frame) const = 0;
};

/* =========================
 * STAGE 2: Globale Frame-Metriken
 * ========================= */

struct GlobalMetrics {
    double B;
    double sigma;
    double E;
    double Q;
    double G;
};

class GlobalMetricsComputer {
public:
    virtual ~GlobalMetricsComputer() = default;

    virtual GlobalMetrics
    compute(const ImageView& normalized_frame) const = 0;
};

/* =========================
 * STAGE 3: Seeing-adaptive Tile-Geometrie
 * ========================= */

class TileGeometry {
public:
    virtual ~TileGeometry() = default;

    virtual std::vector<Tile>
    generate(int image_width,
             int image_height,
             double median_fwhm_pixels) const = 0;
};

/* =========================
 * STAGE 4: Lokale Tile-Metriken
 * ========================= */

struct LocalTileMetrics {
    double Q_local;
    double L;
};

class LocalTileMetricsComputer {
public:
    virtual ~LocalTileMetricsComputer() = default;

    virtual LocalTileMetrics
    compute(const ImageView& normalized_frame,
            const Tile& tile) const = 0;
};

/* =========================
 * STAGE 5: Tile-Rekonstruktion
 * ========================= */

class TileReconstructor {
public:
    virtual ~TileReconstructor() = default;

    virtual TileResult
    reconstruct(const Tile& tile,
                const std::vector<ImageView>& frames,
                const std::vector<double>& weights) const = 0;
};

/* =========================
 * STAGE 6: Zustandsvektor & Clusterung
 * ========================= */

struct FrameStateVector {
    double G;
    double mean_Q_tile;
    double var_Q_tile;
    double B;
    double sigma;
};

class FrameStateClusterer {
public:
    virtual ~FrameStateClusterer() = default;

    virtual std::vector<std::vector<FrameIndex>>
    cluster(const std::vector<FrameStateVector>& states,
            std::size_t k_min,
            std::size_t k_max) const = 0;
};

/* =========================
 * STAGE 7: Synthetische Qualitätsframes
 * ========================= */

class SyntheticFrameBuilder {
public:
    virtual ~SyntheticFrameBuilder() = default;

    virtual ImageView
    build(const std::vector<ImageView>& frames,
          const std::vector<double>& global_weights) const = 0;
};

/* =========================
 * STAGE 8: Finales Stacking
 * ========================= */

class FinalStacker {
public:
    virtual ~FinalStacker() = default;

    virtual ImageView
    stack(const std::vector<ImageView>& synthetic_frames) const = 0;
};

/* =========================
 * Pipeline-Hardstops
 * ========================= */

inline void require(bool condition, const char* msg) {
    if (!condition) {
        throw std::runtime_error(msg);
    }
}
