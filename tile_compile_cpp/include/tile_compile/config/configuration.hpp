#pragma once

#include <algorithm>
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
  bool write_registered_frames = false;
  bool crop_to_nonzero_bbox = true;
};

struct DataConfig {
  int image_width = 0;
  int image_height = 0;
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
  bool dark_already_bias_corrected = false;
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
  std::string pattern = "*.fit;*.fits;*.fts;*.fit.fz;*.fits.fz;*.fts.fz";
};

struct AssumptionsConfig {
  int frames_min = 50;
  int frames_reduced_threshold = 200;
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
                                                 // star_similarity | hybrid_phase_ecc |
                                                 // robust_phase_ecc
  std::string transform_model = "similarity";    // similarity | affine
  bool enable_star_pair_fallback = true;
  bool allow_rotation = true;
  // When true, the pipeline detects strong field rotation (Alt/Az mount) from
  // the first few frames and automatically overrides engine to
  // triangle_star_matching + transform_model=affine if the configured engine
  // would likely fail (e.g. robust_phase_ecc with allow_rotation=true).
  bool auto_engine = true;
  // Rotation threshold in degrees per frame above which auto_engine triggers.
  float auto_engine_rotation_threshold_deg = 0.05f;
  int star_topk = 150;
  int star_min_inliers = 4;
  float star_inlier_tol_px = 4.0f;
  float star_dist_bin_px = 5.0f;
  bool reject_outliers = true;
  float reject_cc_min_abs = 0.25f;
  float reject_shift_px_min = 100.0f;
  float reject_shift_median_multiplier = 5.0f;
  float reject_scale_min = 0.92f;
  float reject_scale_max = 1.08f;
  // Neue Parameter für Blind-Chain Rescue (§4.1, §8.B)
  int max_blind_chain_depth = 0;               // 0 = auto (N/10), >0 = manual
  float blind_chain_strong_anchor_cc = 0.08f;  // CC threshold for strong anchors
  float blind_chain_drift_threshold_px = 2.0f; // Max drift per frame in chain
  // Astrometric rescue (§4.13)
  bool use_astrometry = true;                  // Enable astrometric rescue
  // Local background subtraction for star detection (§4.4, §8.D)
  bool enable_local_background_subtraction = false;
  // Shift-consistency filter radius for triangle_star_matching (px on proxy).
  // Must cover the maximum expected inter-frame shift.  For equatorial mounts
  // 60 px is sufficient; for Alt/Az sessions (e.g. DWARF II) use 200-400 px.
  float star_shift_radius_px = 200.0f;
};

// §4.1, §8.B — Berechnung effektiver Chain-Tiefe
// 0 = auto (N/10, min 12, max 50), >0 = manuelle Überschreibung
inline int get_effective_chain_depth(int num_frames, const RegistrationConfig& cfg) {
  if (cfg.max_blind_chain_depth > 0) {
    return cfg.max_blind_chain_depth;  // Manuelle Überschreibung
  }
  // Auto: N/10, mindestens 12, maximal 50
  return std::clamp(num_frames / 10, 12, 50);
}

struct WienerDenoiseConfig {
  bool enabled = false;
  float snr_threshold = 5.0f;
  float q_min = -0.5f;
  float q_max = 1.0f;
  float q_step = 0.1f;
  float min_snr = 2.0f;
  int max_iterations = 10;
};

struct SoftThresholdConfig {
  bool enabled = true;
  int blur_kernel = 31;       // box-blur kernel size for background estimation
  float alpha = 1.5f;         // threshold multiplier: τ = α · σ_t
  bool skip_star_tiles = true; // skip denoising for star-dominated tiles
};

struct TileDenoiseConfig {
  SoftThresholdConfig soft_threshold;
  WienerDenoiseConfig wiener;
};

struct ChromaDenoiseConfig {
  struct StarProtectionConfig {
    bool enabled = true;
    float threshold_sigma = 2.2f;
    int dilate_px = 2;
  } star_protection;

  struct StructureProtectionConfig {
    bool enabled = true;
    float gradient_percentile = 85.0f;
  } structure_protection;

  struct ChromaWaveletConfig {
    bool enabled = true;
    int levels = 3;
    float threshold_scale = 1.25f;
    float soft_k = 1.0f;
  } chroma_wavelet;

  struct ChromaBilateralConfig {
    bool enabled = true;
    float sigma_spatial = 1.2f;
    float sigma_range = 0.035f;
  } chroma_bilateral;

  struct BlendConfig {
    std::string mode = "chroma_only"; // chroma_only
    float amount = 0.85f;
  } blend;

  bool enabled = false;
  std::string color_space = "ycbcr_linear";      // ycbcr_linear | opponent_linear
  std::string apply_stage = "post_stack_linear"; // pre_stack_tiles | post_stack_linear
  bool protect_luma = true;
  float luma_guard_strength = 0.75f;
};

struct DitheringConfig {
  bool enabled = false;
  float min_shift_px = 0.5f;
};

struct GlobalMetricsConfig {
  struct Weights {
    float background = 0.4f;
    float noise = 0.3f;
    float gradient = 0.3f;
  } weights;
  std::array<float, 2> clamp{-3.0f, 3.0f};
  bool adaptive_weights = false;
  float weight_exponent_scale = 1.0f; // G_f = exp(k · Q_f), k>1 → stronger differentiation
};

struct TileConfig {
  int size_factor = 32;
  int min_size = 64;
  int max_divisor = 6;
  float overlap_fraction = 0.25f;
  int star_min_count = 10;
  int star_soft_count = 10;
};

struct LocalMetricsConfig {
  struct NeighborhoodNormalizationConfig {
    bool enabled = true;
    int radius = 1;
    float blend = 0.5f;
  } neighborhood_normalization;

  struct SpatialRegularizationConfig {
    bool enabled = true;
    float lambda = 0.35f;
    int passes = 1;
    float tau_local = 1.0f;
  } spatial_regularization;

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
  float k_local = 1.0f; // §5.5.6: L_{f,t} = exp(k_local * Q^local), symmetric with k_global
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

struct StackingConfig {
  struct SigmaClipConfig {
    float sigma_low = 2.0f;
    float sigma_high = 2.0f;
    int max_iters = 3;
    float min_fraction = 0.5f;
  } sigma_clip;

  struct ClusterQualityWeightingConfig {
    bool enabled = true;
    float kappa_cluster = 1.0f;
    bool cap_enabled = false;
    float cap_ratio = 20.0f;
  } cluster_quality_weighting;

  std::string method = "rej";
  float common_overlap_required_fraction = 1.0f;
  float tile_common_valid_min_fraction = 1.0f;
  bool output_stretch = false;
  bool cosmetic_correction = false;
  float cosmetic_correction_sigma = 5.0f;
  bool per_frame_cosmetic_correction = false;
  float per_frame_cosmetic_correction_sigma = 5.0f;
};

struct AstrometryConfig {
  bool enabled = false;
  std::string astap_bin;          // empty = default path
  std::string astap_data_dir;     // empty = default path
  int search_radius = 180;        // degrees (180 = blind solve)
};

struct BGEConfig {
  bool enabled = false;
  
  // Tile sampling (v3.3 §6.3.2)
  float sample_quantile = 0.20f;
  std::string sample_estimator = "quantile";
  float min_sample_bg_value = 1.0f;
  float structure_thresh_percentile = 0.90f;
  int min_tiles_per_cell = 3;
  // Minimum robust tile sample coverage required before attempting per-channel
  // BGE fitting/application. Keeps BGE deterministic but tunable for dense
  // fields where masking leaves fewer valid tiles.
  float min_valid_sample_fraction_for_apply = 0.30f;
  int min_valid_samples_for_apply = 96;
  
  // Masks (v3.3 §6.3.2a)
  struct {
    int star_dilate_px = 4;
    int sat_dilate_px = 4;
  } mask;
  
  // Grid (v3.3 §6.3.3, §6.3.8)
  struct {
    int N_g = 32;
    int G_min_px = 64;
    float G_max_fraction = 0.25f;
    std::string insufficient_cell_strategy = "discard";
  } grid;
  
  // Surface fitting (v3.3 §6.3.4, §6.3.7)
  struct {
    std::string method = "rbf"; // poly | spline | bicubic | rbf | modeled_mask_mesh
    std::string robust_loss = "huber"; // huber | tukey
    float huber_delta = 1.5f;
    int irls_max_iterations = 10;
    float irls_tolerance = 1e-4f;
    int polynomial_order = 2;
    std::string rbf_phi = "multiquadric";
    float rbf_mu_factor = 1.0f;
    float rbf_lambda = 1e-6f;
    float rbf_epsilon = 1e-10f;
  } fit;

  // Autotuning (v3.3.6)
  struct {
    bool enabled = false;
    int max_evals = 24;
    float holdout_fraction = 0.25f;
    float alpha_flatness = 0.25f;  // renamed alpha_f in §6.3.7.1
    float beta_roughness = 0.10f;  // renamed beta_r in §6.3.7.1
    std::string strategy = "conservative"; // conservative | extended
  } autotune;

  // Tile reliability weight (§6.3.2c): w_t = exp(-lambda_structure *
  // structure_score_t) * (1 - masked_fraction_t), with dimensionless
  // structure_score_t after local noise normalization.
  float tile_weight_lambda_structure = 1.0f;
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

  // Local annulus background model (v3.3.6 §6.4)
  std::string background_model = "plane"; // median | plane
  float max_condition_number = 3.0f; // >= 1, rejects unstable PCC matrices
  float max_residual_rms = 0.35f;    // > 0, rejects noisy PCC fits

  // Adaptive radii (v3.3.6 §6.4)
  std::string radii_mode = "auto_fwhm"; // fixed | auto_fwhm
  float aperture_fwhm_mult = 1.8f;
  float annulus_inner_fwhm_mult = 3.0f;
  float annulus_outer_fwhm_mult = 5.0f;
  float min_aperture_px = 4.0f;

  std::string siril_catalog_dir;  // empty = default path
  
  bool apply_attenuation = false;
  float chroma_strength = 1.00f;
  float k_max = 3.20f;
  std::string background_neutralization_mode = "auto"; // always | auto | off
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
  bool allow_emergency_mode = false;
  int parallel_workers = 4;
  int memory_budget = 512;
  std::string acceleration_backend = "auto";
  std::string tile_reconstruction_diagnostics = "full";
  bool tile_boundary_diagnostics_enabled = false; // opt-in, default off
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
  DitheringConfig dithering;
  TileDenoiseConfig tile_denoise;
  ChromaDenoiseConfig chroma_denoise;
  GlobalMetricsConfig global_metrics;
  TileConfig tile;
  LocalMetricsConfig local_metrics;
  SyntheticConfig synthetic;
  AstrometryConfig astrometry;
  BGEConfig bge;
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
