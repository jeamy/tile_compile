#pragma once

#include "tile_compile/config/configuration.hpp"
#include "tile_compile/core/types.hpp"
#include <Eigen/Dense>
#include <array>
#include <random>
#include <string>
#include <vector>

namespace tile_compile::image {

struct SamplePoint {
    int x = 0;
    int y = 0;
};

// BGE configuration: the public config::BGEConfig is the single source of
// truth for every YAML-exposed field, its default and its schema entry. The
// image-side struct only adds runner-internal, non-YAML fields.
struct BGEConfig : config::BGEConfig {
    // Derived by the runner from method != "none" (see to_image_bge_config);
    // load-bearing gate inside apply_background_extraction.
    bool enabled = false;
    // Internal safety knob (not YAML-exposed): relax channel acceptance
    // guards for controlled fallback retries on difficult fields.
    bool internal_relaxed_channel_guards = false;
    // Internal common-overlap support mask (not YAML-exposed). When set,
    // BGE excludes canvas pixels from all computations.
    std::vector<uint8_t> common_valid_mask;
    int common_mask_rows = 0;
    int common_mask_cols = 0;
    std::vector<uint8_t> sampling_valid_mask;
    int sampling_mask_rows = 0;
    int sampling_mask_cols = 0;
    // Internal: cap OpenMP threads used inside BGE (0 = use omp_get_max_threads).
    int max_workers = 0;
};

// Tile background sample
struct TileBGSample {
    float x = 0.0f, y = 0.0f; // Tile center position
    float bg_value = 0.0f;    // Background estimate
    float weight = 0.0f;      // Reliability weight
    bool valid = false;       // Sample is valid
};

// Grid cell for coarse aggregation
struct GridCell {
    int cell_x = 0, cell_y = 0;     // Grid cell indices
    float center_x = 0.0f, center_y = 0.0f; // Cell center position
    float bg_value = 0.0f;          // Aggregated background
    float weight = 0.0f;            // Aggregated weight
    int n_samples = 0;              // Number of tile samples in cell
    bool valid = false;             // Cell has sufficient samples
};

// Background model result
struct BackgroundModel {
    Matrix2Df model;      // Interpolated background surface
    std::vector<GridCell> grid_cells; // Grid cells used for fitting
    int n_valid_cells;    // Number of valid cells
    float rms_residual;   // RMS of fit residuals
    double fit_select_seconds = 0.0;
    double render_seconds = 0.0;
    double total_seconds = 0.0;
    bool success;         // Model was successfully computed
    std::string error_message;
};

struct BGEProfileTiming {
    double total_seconds = 0.0;
    double modeled_prepass_seconds = 0.0;
    double autotune_total_seconds = 0.0;
    double autotune_prep_seconds = 0.0;
    double autotune_eval_seconds = 0.0;
    double autotune_eval_model_select_seconds = 0.0;
    double autotune_eval_surface_sample_seconds = 0.0;
    double autotune_eval_metric_seconds = 0.0;
    double tile_sampling_seconds = 0.0;
    double coarse_grid_seconds = 0.0;
    double final_fit_total_seconds = 0.0;
    double final_fit_select_seconds = 0.0;
    double final_fit_render_seconds = 0.0;
    double apply_correction_seconds = 0.0;
    double guard_seconds = 0.0;
    int autotune_prep_builds = 0;
    int autotune_candidate_jobs = 0;
};

struct BGEValueStats {
    int n = 0;
    float min = 0.0f;
    float max = 0.0f;
    float median = 0.0f;
    float mean = 0.0f;
    float std = 0.0f;
};

struct BGEChannelDiagnostics {
    std::string channel_name;
    bool applied = false;
    bool fit_success = false;
    bool autotune_enabled = false;
    int autotune_evals = 0;
    bool autotune_fallback_used = false;
    std::string autotune_selected_fit_method;
    float autotune_best_objective = 0.0f; // Binding autotune objective
    float autotune_best_objective_raw = 0.0f;
    float autotune_best_objective_normalized = 0.0f;
    float autotune_best_cv_rms = 0.0f;
    float autotune_best_flatness = 0.0f;
    float autotune_best_roughness = 0.0f;
    std::string autotune_selected_sample_estimator;
    float autotune_selected_sample_quantile = 0.0f;
    float autotune_selected_structure_thresh_percentile = 0.0f;
    float autotune_selected_rbf_mu_factor = 0.0f;
    int autotune_selected_grid_spacing = 0;
    int tile_samples_total = 0;
    int tile_samples_valid = 0;
    int grid_cells_valid = 0;
    float fit_rms_residual = 0.0f;
    float mean_shift = 0.0f;
    float guard_flat_pre = 0.0f;
    float guard_flat_post = 0.0f;
    float guard_slope_pre = 0.0f;
    float guard_slope_post = 0.0f;
    bool guard_rejected = false;
    std::string guard_reason;
    BGEProfileTiming profile;
    BGEValueStats input_stats;
    BGEValueStats output_stats;
    BGEValueStats model_stats;
    BGEValueStats sample_bg_stats;
    BGEValueStats sample_weight_stats;
    BGEValueStats residual_stats;
    std::vector<float> sample_bg_values;
    std::vector<float> sample_weight_values;
    std::vector<float> residual_values;
    std::vector<GridCell> grid_cells;
};

struct BGEDiagnostics {
    bool attempted = false;
    bool success = false;
    std::string failure_reason;
    int image_width = 0;
    int image_height = 0;
    int grid_spacing = 0;
    std::string bge_method = "none";
    float auto_gradient_strength = 0.0f;
    float auto_gradient_threshold = 0.0f;
    float auto_sky_median = 0.0f;
    float auto_sky_sigma = 0.0f;
    float auto_extended_source_threshold = 0.0f;
    int auto_extended_source_blocks = 0;
    float auto_extended_source_excluded_fraction = 0.0f;
    std::string method;
    std::string robust_loss;
    std::string insufficient_cell_strategy;
    bool autotune_enabled = false;
    std::string autotune_strategy;
    int autotune_max_evals = 0;
    int autotune_evals = 0;
    std::string autotune_selected_fit_method;
    float autotune_best_objective = 0.0f; // Binding autotune objective
    float autotune_best_objective_raw = 0.0f;
    float autotune_best_objective_normalized = 0.0f;
    float autotune_best_cv_rms = 0.0f;
    float autotune_best_flatness = 0.0f;
    float autotune_best_roughness = 0.0f;
    std::string autotune_selected_sample_estimator;
    float autotune_selected_sample_quantile = 0.0f;
    float autotune_selected_structure_thresh_percentile = 0.0f;
    float autotune_selected_rbf_mu_factor = 0.0f;
    bool autotune_fallback_used = false;
    // Set when a per-channel slope_worsened veto was overridden because the
    // correction still improved the inter-channel background level spread.
    std::string guard_override;
    bool safety_fallback_triggered = false;
    std::string safety_fallback_method;
    std::string safety_fallback_reason;
    BGEProfileTiming profile;
    std::vector<BGEChannelDiagnostics> channels;
};

// Shared background/chroma utilities used by BGE diagnostics and PCC damping.
// Uses external canvas validity mask: pixels where valid_mask[i]==0 are
// treated as canvas (excluded before any gradient/luma thresholding).
// If valid_mask is nullptr, no external canvas mask is applied.
std::vector<uint8_t> build_chroma_background_mask_from_rgb(
    const Matrix2Df& R, const Matrix2Df& G, const Matrix2Df& B,
    const std::vector<uint8_t>* valid_mask);
float log_chroma_std_background(const Matrix2Df& A, const Matrix2Df& G,
                                const std::vector<uint8_t>& bg_mask);

// Canvas-mask utilities shared by BGE/PCC.
bool canvas_mask_matches_image(const std::vector<uint8_t>& mask, int rows, int cols);
void enforce_canvas_mask_on_rgb(Matrix2Df& R, Matrix2Df& G, Matrix2Df& B,
                                const std::vector<uint8_t>& mask);

// Main BGE function (v3.3 §6.3)
// Extracts and subtracts large-scale background gradients from RGB channels
// Returns true if BGE was applied successfully
bool apply_background_extraction(
    Matrix2Df& R, Matrix2Df& G, Matrix2Df& B,
    const std::vector<TileMetrics>& tile_metrics,
    const TileGrid& tile_grid,
    const BGEConfig& config,
    BGEDiagnostics* diagnostics = nullptr);

// Extract tile background samples (v3.3 §6.3.2)
std::vector<TileBGSample> extract_tile_background_samples(
    const Matrix2Df& channel,
    const std::vector<TileMetrics>& tile_metrics,
    const TileGrid& tile_grid,
    const BGEConfig& config);

// Aggregate tiles to coarse grid (v3.3 §6.3.3)
std::vector<GridCell> aggregate_to_coarse_grid(
    const std::vector<TileBGSample>& tile_samples,
    int image_width, int image_height,
    int grid_spacing,
    const BGEConfig& config);

// Fit background surface (v3.3 §6.3.7)
BackgroundModel fit_background_surface(
    const std::vector<GridCell>& grid_cells,
    int image_width, int image_height,
    int grid_spacing,
    const BGEConfig& config);

// Compute adaptive grid spacing (v3.3 §6.3.8)
int compute_grid_spacing(
    int image_width, int image_height,
    int tile_size,
    const BGEConfig& config);

// RBF kernel functions (v3.3 §6.3.7)
float rbf_kernel_multiquadric(float d, float mu);
float rbf_kernel_thinplate(float d, float epsilon);
float rbf_kernel_gaussian(float d, float mu);

// Robust loss functions (v3.3 §6.3.7)
float huber_weight(float r, float delta);
float tukey_weight(float r, float c);

// BGE guard metrics — accept optional canvas validity mask.
// When provided, pixels where valid_mask[y*W+x]==0 are excluded
// (canvas border pixels from bilinear warp interpolation).
float spatial_background_spread(const Matrix2Df& img,
    const std::vector<uint8_t>* valid_mask = nullptr);
float coarse_background_plane_slope(const Matrix2Df& img,
    const std::vector<uint8_t>* valid_mask = nullptr);

// ===== AutoBGE (two-stage poly+RBF background extraction) =====

struct StretchParams {
    std::vector<float> original_mins;
    std::vector<float> original_medians;
    std::vector<float> linear_offsets;
    std::vector<float> linear_scales;
    std::vector<float> mtf_targets;
    std::vector<float> mtf_scales;  // p99-vmin scale used to normalize MTF input
    std::string mode;
    bool was_single_channel = false;
};

struct AutoBGEResult {
    bool success = false;
    bool mono_input = false;
    std::array<BackgroundModel, 3> channel_models;
    std::vector<BGEChannelDiagnostics> channel_diagnostics;
};

Matrix2Df transform_to_autobge_working_space(
    const Matrix2Df& channel, const BGEConfig::AutoBGEConfig& config,
    StretchParams* params, int channel_index,
    const std::vector<uint8_t>* valid_mask = nullptr);

Matrix2Df transform_from_autobge_working_space(
    const Matrix2Df& channel, const StretchParams& params, int channel_index);

Matrix2Df downsample_area(const Matrix2Df& image, int scale);

Matrix2Df upscale_lanczos4(const Matrix2Df& background, int target_rows, int target_cols);

std::vector<SamplePoint> generate_autobge_sample_points(
    const Matrix2Df& image_downsampled,
    const BGEConfig::AutoBGEConfig& config,
    const std::vector<uint8_t>* valid_mask_downsampled = nullptr,
    std::mt19937* rng = nullptr,
    bool random_downselection = true);

Matrix2Df fit_polynomial_autobge(
    const Matrix2Df& image_downsampled,
    const std::vector<SamplePoint>& points,
    const BGEConfig::AutoBGEConfig& config,
    int target_rows, int target_cols);

Matrix2Df fit_rbf_autobge(
    const Matrix2Df& image_downsampled,
    const std::vector<SamplePoint>& points,
    const BGEConfig::AutoBGEConfig& config,
    int target_rows, int target_cols);

AutoBGEResult build_autobge_models(
    const Matrix2Df& R, const Matrix2Df& G, const Matrix2Df& B,
    const BGEConfig& config);

bool finalize_bge_from_channel_models(
    Matrix2Df& R, Matrix2Df& G, Matrix2Df& B,
    const std::array<BackgroundModel, 3>& channel_models,
    const std::vector<BGEChannelDiagnostics>& channel_diagnostics,
    const BGEConfig& config,
    BGEDiagnostics* diagnostics);

} // namespace tile_compile::image
