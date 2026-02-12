#pragma once

#include <array>
#include <filesystem>
#include <string>
#include <yaml-cpp/yaml.h>

namespace tile_compile::config {

namespace fs = std::filesystem;

struct PipelineConfig {
  std::string mode = "production";
  bool abort_on_fail = true;
};

struct OutputConfig {
  std::string registered_dir = "registered";
  std::string artifacts_dir = "artifacts";
  bool write_registered_frames = false;
  bool write_global_metrics = true;
  bool write_global_registration = true;
};

struct DataConfig {
  int image_width = 0;
  int image_height = 0;
  int frames_min = 0;
  int frames_target = 0;
  std::string color_mode = "OSC";
  std::string bayer_pattern = "GBRG";
  bool linear_required = true;
};

struct LinearityConfig {
  bool enabled = true;
  int max_frames = 8;
  float min_overall_linearity = 0.9f;
  std::string strictness = "strict"; // strict | moderate | permissive
};

struct CalibrationConfig {
  bool use_bias = false;
  bool use_dark = false;
  bool use_flat = false;
  bool bias_use_master = false;
  bool dark_use_master = false;
  bool flat_use_master = false;
  bool dark_auto_select = true;
  float dark_match_exposure_tolerance_percent = 5.0f;
  bool dark_match_use_temp = false;
  float dark_match_temp_tolerance_c = 2.0f;
  std::string bias_dir;
  std::string darks_dir;
  std::string flats_dir;
  std::string bias_master;
  std::string dark_master;
  std::string flat_master;
  std::string pattern = "*.fit*";
};

struct AssumptionsConfig {
  int frames_min = 50;
  int frames_optimal = 800;
  int frames_reduced_threshold = 200;
  float exposure_time_tolerance_percent = 5.0f;
  bool reduced_mode_skip_clustering = true;
  std::array<int, 2> reduced_mode_cluster_range{5, 10};
};

struct NormalizationConfig {
  bool enabled = true;
  std::string mode = "background";
  bool per_channel = true;
};

struct RegistrationConfig {
  std::string engine = "triangle_star_matching"; // triangle_star_matching |
                                                 // star_similarity | hybrid_phase_ecc
  bool allow_rotation = true;
  int star_topk = 120;
  int star_min_inliers = 6;
  float star_inlier_tol_px = 2.5f;
  float star_dist_bin_px = 2.5f;
};

struct WienerDenoiseConfig {
  bool enabled = false;
  float snr_threshold = 5.0f;
  float q_min = -0.5f;
  float q_max = 1.0f;
  float q_step = 0.1f;
  float min_snr = 2.0f;
  int max_iterations = 10;
};

struct GlobalMetricsConfig {
  struct Weights {
    float background = 0.4f;
    float noise = 0.3f;
    float gradient = 0.3f;
  } weights;
  std::array<float, 2> clamp{-3.0f, 3.0f};
  bool adaptive_weights = false;
};

struct TileConfig {
  int size_factor = 32;
  int min_size = 64;
  int max_divisor = 6;
  float overlap_fraction = 0.25f;
  int star_min_count = 10;
};

struct LocalMetricsConfig {
  struct StarModeConfig {
    struct Weights {
      float fwhm = 0.6f;
      float roundness = 0.2f;
      float contrast = 0.2f;
    } weights;
  } star_mode;

  struct StructureModeConfig {
    float background_weight = 0.3f;
    float metric_weight = 0.7f;
  } structure_mode;

  std::array<float, 2> clamp{-3.0f, 3.0f};
};

struct SyntheticConfig {
  struct ClusteringConfig {
    std::string mode = "kmeans";
    std::array<int, 2> cluster_count_range{5, 30};
  } clustering;
  std::string weighting = "global";
  int frames_min = 5;
  int frames_max = 30;
};

struct ReconstructionConfig {
  std::string weighting_function = "linear";
  std::string window_function = "hanning";
};

struct StackingConfig {
  struct SigmaClipConfig {
    float sigma_low = 2.0f;
    float sigma_high = 2.0f;
    int max_iters = 3;
    float min_fraction = 0.5f;
  } sigma_clip;

  std::string method = "rej";
  bool output_stretch = true;
};

struct AstrometryConfig {
  bool enabled = false;
  std::string astap_bin;          // empty = default path
  std::string astap_data_dir;     // empty = default path
  int search_radius = 180;        // degrees (180 = blind solve)
};

struct PCCConfig {
  bool enabled = false;
  std::string source = "auto";    // auto | siril | vizier_gaia | vizier_apass
  float mag_limit = 14.0f;
  float mag_bright_limit = 6.0f;
  float aperture_radius_px = 8.0f;
  float annulus_inner_px = 12.0f;
  float annulus_outer_px = 18.0f;
  int min_stars = 10;
  float sigma_clip = 2.5f;
  std::string siril_catalog_dir;  // empty = default path
};

struct ValidationConfig {
  float min_fwhm_improvement_percent = 0.0f;
  float max_background_rms_increase_percent = 0.0f;
  float min_tile_weight_variance = 0.1f;
  bool require_no_tile_pattern = true;
};

struct RuntimeLimitsConfig {
  float tile_analysis_max_factor_vs_stack = 3.0f;
  float hard_abort_hours = 6.0f;
};

struct Config {
  PipelineConfig pipeline;
  OutputConfig output;
  DataConfig data;
  LinearityConfig linearity;
  CalibrationConfig calibration;
  AssumptionsConfig assumptions;
  NormalizationConfig normalization;
  RegistrationConfig registration;
  WienerDenoiseConfig wiener_denoise;
  GlobalMetricsConfig global_metrics;
  TileConfig tile;
  LocalMetricsConfig local_metrics;
  SyntheticConfig synthetic;
  ReconstructionConfig reconstruction;
  bool debayer = true;
  AstrometryConfig astrometry;
  PCCConfig pcc;
  StackingConfig stacking;
  ValidationConfig validation;
  RuntimeLimitsConfig runtime_limits;

  static Config load(const fs::path &path);
  static Config from_yaml(const YAML::Node &node);

  void save(const fs::path &path) const;
  YAML::Node to_yaml() const;

  void validate() const;
};

std::string get_schema_json();

} // namespace tile_compile::config
