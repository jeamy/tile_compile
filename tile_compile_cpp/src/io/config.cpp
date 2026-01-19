#include "tile_compile/config/configuration.hpp"
#include "tile_compile/core/errors.hpp"

#include <fstream>
#include <sstream>

namespace tile_compile::config {

Config Config::load(const fs::path& path) {
    if (!fs::exists(path)) {
        throw ConfigError("Config file not found: " + path.string());
    }
    
    YAML::Node node = YAML::LoadFile(path.string());
    return from_yaml(node);
}

Config Config::from_yaml(const YAML::Node& node) {
    Config cfg;
    
    if (node["global_metrics"]) {
        auto gm = node["global_metrics"];
        if (gm["weights"]) {
            auto w = gm["weights"];
            if (w["background"]) cfg.global_metrics.weights.background = w["background"].as<float>();
            if (w["noise"]) cfg.global_metrics.weights.noise = w["noise"].as<float>();
            if (w["gradient"]) cfg.global_metrics.weights.gradient = w["gradient"].as<float>();
        }
    }
    
    if (node["tile"]) {
        auto t = node["tile"];
        if (t["size_factor"]) cfg.tile.size_factor = t["size_factor"].as<float>();
        if (t["min_size"]) cfg.tile.min_size = t["min_size"].as<int>();
        if (t["overlap_fraction"]) cfg.tile.overlap_fraction = t["overlap_fraction"].as<float>();
    }
    
    if (node["synthetic"]) {
        auto s = node["synthetic"];
        if (s["frames_min"]) cfg.synthetic.frames_min = s["frames_min"].as<int>();
        if (s["frames_max"]) cfg.synthetic.frames_max = s["frames_max"].as<int>();
        if (s["weighting"]) cfg.synthetic.weighting = s["weighting"].as<std::string>();
    }
    
    if (node["registration"]) {
        auto r = node["registration"];
        if (r["allow_rotation"]) cfg.registration.allow_rotation = r["allow_rotation"].as<bool>();
        if (r["max_iterations"]) cfg.registration.max_iterations = r["max_iterations"].as<int>();
        if (r["epsilon"]) cfg.registration.epsilon = r["epsilon"].as<float>();
        if (r["rotation_sweep"]) cfg.registration.rotation_sweep = r["rotation_sweep"].as<bool>();
        if (r["rotation_range_deg"]) cfg.registration.rotation_range_deg = r["rotation_range_deg"].as<float>();
        if (r["rotation_steps"]) cfg.registration.rotation_steps = r["rotation_steps"].as<int>();
    }
    
    if (node["calibration"]) {
        auto c = node["calibration"];
        if (c["enabled"]) cfg.calibration.enabled = c["enabled"].as<bool>();
        if (c["use_bias"]) cfg.calibration.use_bias = c["use_bias"].as<bool>();
        if (c["use_dark"]) cfg.calibration.use_dark = c["use_dark"].as<bool>();
        if (c["use_flat"]) cfg.calibration.use_flat = c["use_flat"].as<bool>();
        if (c["bias_master"]) cfg.calibration.bias_master = fs::path(c["bias_master"].as<std::string>());
        if (c["dark_master"]) cfg.calibration.dark_master = fs::path(c["dark_master"].as<std::string>());
        if (c["flat_master"]) cfg.calibration.flat_master = fs::path(c["flat_master"].as<std::string>());
        if (c["bias_dir"]) cfg.calibration.bias_dir = fs::path(c["bias_dir"].as<std::string>());
        if (c["darks_dir"]) cfg.calibration.darks_dir = fs::path(c["darks_dir"].as<std::string>());
        if (c["flats_dir"]) cfg.calibration.flats_dir = fs::path(c["flats_dir"].as<std::string>());
    }
    
    if (node["clustering"]) {
        auto cl = node["clustering"];
        if (cl["enabled"]) cfg.clustering.enabled = cl["enabled"].as<bool>();
        if (cl["min_clusters"]) cfg.clustering.min_clusters = cl["min_clusters"].as<int>();
        if (cl["max_clusters"]) cfg.clustering.max_clusters = cl["max_clusters"].as<int>();
        if (cl["method"]) cfg.clustering.method = cl["method"].as<std::string>();
    }
    
    if (node["stacking"]) {
        auto st = node["stacking"];
        if (st["method"]) cfg.stacking.method = st["method"].as<std::string>();
        if (st["sigma_low"]) cfg.stacking.sigma_low = st["sigma_low"].as<float>();
        if (st["sigma_high"]) cfg.stacking.sigma_high = st["sigma_high"].as<float>();
        if (st["max_iters"]) cfg.stacking.max_iters = st["max_iters"].as<int>();
    }
    
    if (node["color_mode"]) cfg.color_mode = node["color_mode"].as<std::string>();
    if (node["bayer_pattern"]) cfg.bayer_pattern = node["bayer_pattern"].as<std::string>();
    
    return cfg;
}

void Config::save(const fs::path& path) const {
    YAML::Node node = to_yaml();
    std::ofstream out(path);
    if (!out) {
        throw ConfigError("Cannot write config file: " + path.string());
    }
    out << node;
}

YAML::Node Config::to_yaml() const {
    YAML::Node node;
    
    node["global_metrics"]["weights"]["background"] = global_metrics.weights.background;
    node["global_metrics"]["weights"]["noise"] = global_metrics.weights.noise;
    node["global_metrics"]["weights"]["gradient"] = global_metrics.weights.gradient;
    
    node["tile"]["size_factor"] = tile.size_factor;
    node["tile"]["min_size"] = tile.min_size;
    node["tile"]["overlap_fraction"] = tile.overlap_fraction;
    
    node["synthetic"]["frames_min"] = synthetic.frames_min;
    node["synthetic"]["frames_max"] = synthetic.frames_max;
    node["synthetic"]["weighting"] = synthetic.weighting;
    
    node["registration"]["allow_rotation"] = registration.allow_rotation;
    node["registration"]["max_iterations"] = registration.max_iterations;
    node["registration"]["epsilon"] = registration.epsilon;
    node["registration"]["rotation_sweep"] = registration.rotation_sweep;
    node["registration"]["rotation_range_deg"] = registration.rotation_range_deg;
    node["registration"]["rotation_steps"] = registration.rotation_steps;
    
    node["calibration"]["enabled"] = calibration.enabled;
    node["calibration"]["use_bias"] = calibration.use_bias;
    node["calibration"]["use_dark"] = calibration.use_dark;
    node["calibration"]["use_flat"] = calibration.use_flat;
    if (calibration.bias_master) node["calibration"]["bias_master"] = calibration.bias_master->string();
    if (calibration.dark_master) node["calibration"]["dark_master"] = calibration.dark_master->string();
    if (calibration.flat_master) node["calibration"]["flat_master"] = calibration.flat_master->string();
    if (calibration.bias_dir) node["calibration"]["bias_dir"] = calibration.bias_dir->string();
    if (calibration.darks_dir) node["calibration"]["darks_dir"] = calibration.darks_dir->string();
    if (calibration.flats_dir) node["calibration"]["flats_dir"] = calibration.flats_dir->string();
    
    node["clustering"]["enabled"] = clustering.enabled;
    node["clustering"]["min_clusters"] = clustering.min_clusters;
    node["clustering"]["max_clusters"] = clustering.max_clusters;
    node["clustering"]["method"] = clustering.method;
    
    node["stacking"]["method"] = stacking.method;
    node["stacking"]["sigma_low"] = stacking.sigma_low;
    node["stacking"]["sigma_high"] = stacking.sigma_high;
    node["stacking"]["max_iters"] = stacking.max_iters;
    
    node["color_mode"] = color_mode;
    node["bayer_pattern"] = bayer_pattern;
    
    return node;
}

void Config::validate() const {
    if (global_metrics.weights.background < 0 || global_metrics.weights.background > 1) {
        throw ValidationError("global_metrics.weights.background must be between 0 and 1");
    }
    if (global_metrics.weights.noise < 0 || global_metrics.weights.noise > 1) {
        throw ValidationError("global_metrics.weights.noise must be between 0 and 1");
    }
    if (global_metrics.weights.gradient < 0 || global_metrics.weights.gradient > 1) {
        throw ValidationError("global_metrics.weights.gradient must be between 0 and 1");
    }
    
    if (tile.size_factor <= 0) {
        throw ValidationError("tile.size_factor must be positive");
    }
    if (tile.min_size <= 0) {
        throw ValidationError("tile.min_size must be positive");
    }
    if (tile.overlap_fraction < 0 || tile.overlap_fraction >= 1) {
        throw ValidationError("tile.overlap_fraction must be between 0 and 1");
    }
    
    if (synthetic.frames_min < 1) {
        throw ValidationError("synthetic.frames_min must be at least 1");
    }
    if (synthetic.frames_max < synthetic.frames_min) {
        throw ValidationError("synthetic.frames_max must be >= frames_min");
    }
    
    if (clustering.min_clusters < 2) {
        throw ValidationError("clustering.min_clusters must be at least 2");
    }
    if (clustering.max_clusters < clustering.min_clusters) {
        throw ValidationError("clustering.max_clusters must be >= min_clusters");
    }
}

std::string get_schema_json() {
    return R"({
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "global_metrics": {
      "type": "object",
      "properties": {
        "weights": {
          "type": "object",
          "properties": {
            "background": {"type": "number", "minimum": 0, "maximum": 1},
            "noise": {"type": "number", "minimum": 0, "maximum": 1},
            "gradient": {"type": "number", "minimum": 0, "maximum": 1}
          }
        }
      }
    },
    "tile": {
      "type": "object",
      "properties": {
        "size_factor": {"type": "number", "minimum": 0},
        "min_size": {"type": "integer", "minimum": 1},
        "overlap_fraction": {"type": "number", "minimum": 0, "maximum": 1}
      }
    }
  }
})";
}

} // namespace tile_compile::config
