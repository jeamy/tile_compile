#pragma once

#include "tile_compile/core/types.hpp"
#include <yaml-cpp/yaml.h>
#include <filesystem>
#include <optional>
#include <string>

namespace tile_compile::config {

namespace fs = std::filesystem;

struct GlobalMetricsConfig {
    struct Weights {
        float background = 0.4f;
        float noise = 0.3f;
        float gradient = 0.3f;
    } weights;
};

struct TileConfig {
    float size_factor = 8.0f;
    int min_size = 32;
    float overlap_fraction = 0.25f;
};

struct SyntheticConfig {
    int frames_min = 3;
    int frames_max = 10;
    std::string weighting = "global";
};

struct RegistrationConfig {
    bool allow_rotation = false;
    int max_iterations = 5000;
    float epsilon = 1e-6f;
    bool rotation_sweep = true;
    float rotation_range_deg = 5.0f;
    int rotation_steps = 11;
};

struct CalibrationConfig {
    bool enabled = false;
    bool use_bias = false;
    bool use_dark = false;
    bool use_flat = false;
    std::optional<fs::path> bias_master;
    std::optional<fs::path> dark_master;
    std::optional<fs::path> flat_master;
    std::optional<fs::path> bias_dir;
    std::optional<fs::path> darks_dir;
    std::optional<fs::path> flats_dir;
};

struct ClusteringConfig {
    bool enabled = true;
    int min_clusters = 3;
    int max_clusters = 10;
    std::string method = "kmeans";
};

struct StackingConfig {
    std::string method = "sigma_clip";
    float sigma_low = 2.5f;
    float sigma_high = 2.5f;
    int max_iters = 5;
};

struct Config {
    GlobalMetricsConfig global_metrics;
    TileConfig tile;
    SyntheticConfig synthetic;
    RegistrationConfig registration;
    CalibrationConfig calibration;
    ClusteringConfig clustering;
    StackingConfig stacking;
    
    std::string color_mode = "auto";
    std::string bayer_pattern = "auto";
    
    static Config load(const fs::path& path);
    static Config from_yaml(const YAML::Node& node);
    
    void save(const fs::path& path) const;
    YAML::Node to_yaml() const;
    
    void validate() const;
};

std::string get_schema_json();

} // namespace tile_compile::config
