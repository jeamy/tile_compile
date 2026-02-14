#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <cctype>
#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace tile_compile {

namespace fs = std::filesystem;

// Matrix types (NumPy equivalents)
using Matrix2Df = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using Matrix2Dd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using VectorXf = Eigen::VectorXf;
using VectorXd = Eigen::VectorXd;
using VectorXi = Eigen::VectorXi;

// Bayer pattern enumeration
enum class BayerPattern {
    UNKNOWN,
    RGGB,
    BGGR,
    GRBG,
    GBRG
};

inline std::string bayer_pattern_to_string(BayerPattern pattern) {
    switch (pattern) {
        case BayerPattern::RGGB: return "RGGB";
        case BayerPattern::BGGR: return "BGGR";
        case BayerPattern::GRBG: return "GRBG";
        case BayerPattern::GBRG: return "GBRG";
        default: return "UNKNOWN";
    }
}

inline BayerPattern string_to_bayer_pattern(const std::string& s) {
    std::string norm = s;
    auto not_space = [](unsigned char c) { return !std::isspace(c); };
    norm.erase(norm.begin(),
               std::find_if(norm.begin(), norm.end(), not_space));
    norm.erase(std::find_if(norm.rbegin(), norm.rend(), not_space).base(),
               norm.end());
    std::transform(norm.begin(), norm.end(), norm.begin(),
                   [](unsigned char c) { return static_cast<char>(std::toupper(c)); });

    if (norm == "RGGB") return BayerPattern::RGGB;
    if (norm == "BGGR") return BayerPattern::BGGR;
    if (norm == "GRBG") return BayerPattern::GRBG;
    if (norm == "GBRG") return BayerPattern::GBRG;
    return BayerPattern::UNKNOWN;
}

// Color mode enumeration
enum class ColorMode {
    MONO,
    OSC,  // One-Shot Color (Bayer)
    RGB   // Already debayered RGB
};

inline std::string color_mode_to_string(ColorMode mode) {
    switch (mode) {
        case ColorMode::MONO: return "MONO";
        case ColorMode::OSC: return "OSC";
        case ColorMode::RGB: return "RGB";
        default: return "UNKNOWN";
    }
}

// Normalization mode
enum class NormalizationMode {
    BACKGROUND,
    ADDITIVE
};

// Tile definition
struct Tile {
    int x;       // Top-left x coordinate
    int y;       // Top-left y coordinate
    int width;
    int height;
    int row;     // Grid row index
    int col;     // Grid column index
};

// Tile grid
struct TileGrid {
    int tile_size;
    float overlap_fraction;
    int rows;
    int cols;
    std::vector<Tile> tiles;
};

// Frame metrics (global)
struct FrameMetrics {
    float background;      // B - median background level
    float noise;           // σ - robust noise estimate
    float gradient_energy; // E - gradient energy
    float quality_score;   // Combined quality score
};

// Tile type
enum class TileType {
    STAR,
    STRUCTURE
};

// Tile metrics (local)
struct TileMetrics {
    float fwhm;            // STAR: FWHM estimate
    float roundness;       // STAR: roundness proxy
    float contrast;        // STAR: contrast proxy
    float sharpness;       // reserved
    float background;      // STRUCTURE: background proxy
    float noise;           // STRUCTURE: noise proxy
    float gradient_energy; // STRUCTURE: gradient energy proxy
    int star_count;        // STAR-vs-STRUCTURE classifier
    TileType type;         // STAR or STRUCTURE
    float quality_score;   // Q_local (clipped)
};

// Channel metrics container
struct ChannelMetrics {
    std::string channel_name;  // "R", "G", "B", or "L"
    std::vector<FrameMetrics> frame_metrics;
    std::vector<std::vector<TileMetrics>> tile_metrics;  // [frame][tile]
    VectorXf global_weights;   // G_f,c
};

// Clustering result
struct ClusteringResult {
    int n_clusters;
    VectorXi labels;           // Cluster assignment per frame
    Matrix2Df centers;         // Cluster centers
    float silhouette_score;
    std::string method;        // "kmeans" or "quantile"
};

// Warp matrix (2x3 affine)
using WarpMatrix = Eigen::Matrix<float, 2, 3>;

// Registration result
struct RegistrationResult {
    WarpMatrix warp;
    float correlation;
    bool success;
    std::string error_message;
};

// Pipeline phase enumeration
enum class Phase {
    SCAN_INPUT = 0,
    REGISTRATION = 1,
    CHANNEL_SPLIT = 2,
    NORMALIZATION = 3,
    GLOBAL_METRICS = 4,
    TILE_GRID = 5,
    LOCAL_METRICS = 6,
    TILE_RECONSTRUCTION = 7,
    STATE_CLUSTERING = 8,
    SYNTHETIC_FRAMES = 9,
    STACKING = 10,
    DEBAYER = 11,
    ASTROMETRY = 12,
    PCC = 13,
    DONE = 14
};

inline std::string phase_to_string(Phase phase) {
    switch (phase) {
        case Phase::SCAN_INPUT: return "SCAN_INPUT";
        case Phase::REGISTRATION: return "REGISTRATION";
        case Phase::CHANNEL_SPLIT: return "CHANNEL_SPLIT";
        case Phase::NORMALIZATION: return "NORMALIZATION";
        case Phase::GLOBAL_METRICS: return "GLOBAL_METRICS";
        case Phase::TILE_GRID: return "TILE_GRID";
        case Phase::LOCAL_METRICS: return "LOCAL_METRICS";
        case Phase::TILE_RECONSTRUCTION: return "TILE_RECONSTRUCTION";
        case Phase::STATE_CLUSTERING: return "STATE_CLUSTERING";
        case Phase::SYNTHETIC_FRAMES: return "SYNTHETIC_FRAMES";
        case Phase::STACKING: return "STACKING";
        case Phase::DEBAYER: return "DEBAYER";
        case Phase::ASTROMETRY: return "ASTROMETRY";
        case Phase::PCC: return "PCC";
        case Phase::DONE: return "DONE";
        default: return "UNKNOWN";
    }
}

inline int phase_to_int(Phase phase) {
    return static_cast<int>(phase);
}

inline Phase int_to_phase(int i) {
    if (i >= 0 && i <= 14) {
        return static_cast<Phase>(i);
    }
    return Phase::SCAN_INPUT;
}

} // namespace tile_compile
