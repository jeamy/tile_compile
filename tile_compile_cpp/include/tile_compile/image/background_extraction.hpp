#pragma once

#include "tile_compile/core/types.hpp"
#include <Eigen/Dense>
#include <array>
#include <random>
#include <string>
#include <vector>

namespace tile_compile::image {

// BGE Configuration (matches YAML structure from v3.3 §6.3)
struct BGEConfig {
    bool enabled = false;
    std::string method = "none"; // none | classic | autobge
    struct AutoBGEConfig {
        int num_sample_points = 0;
        int poly_degree = 2;
        float rbf_smooth = 0.1f;
        int downsample_scale = 4;
        int patch_size = 15;
        std::string patch_estimator = "median";
        std::string stretch_mode = "linear";
        float stretch_target_median = 0.25f;
        int border_margin = 10;
        float bright_exclusion_fraction = 0.5f;
        int gradient_descent_max_iters = 100;
        int random_seed = 42;
        bool normalize_between_stages = true;
        bool apply_guards = true;
        std::string mono_mode = "rgb_duplicate";
    } autobge;
    // Internal safety knob (not YAML-exposed): relax channel acceptance
    // guards for controlled fallback retries on difficult fields.
    bool internal_relaxed_channel_guards = false;
    // Internal common-overlap support mask (not YAML-exposed). When set,
    // BGE excludes canvas pixels from all computations.
    std::vector<uint8_t> common_valid_mask;
    int common_mask_rows = 0;
    int common_mask_cols = 0;
    
    // Tile sampling (§6.3.2)
    float sample_quantile = 0.20f;
    // Per-tile background statistic. `quantile` preserves the historical
    // behavior; the other modes are more robust in crowded/contaminated meshes.
    std::string sample_estimator = "quantile"; // quantile | sigma_clipped_median | sextractor_mode | biweight
    float min_sample_bg_value = 1.0f;
    float structure_thresh_percentile = 0.90f;
    int min_tiles_per_cell = 3;
    float min_valid_sample_fraction_for_apply = 0.30f;
    int min_valid_samples_for_apply = 96;
    
    // Masks (§6.3.2a)
    struct {
        int star_dilate_px = 4;
        int sat_dilate_px = 4;
    } mask;
    
    // Grid (§6.3.3, §6.3.8)
    struct {
        int N_g = 32;
        int G_min_px = 64;
        float G_max_fraction = 0.25f;
        std::string insufficient_cell_strategy = "discard"; // discard | nearest | radius_expand
    } grid;
    
    // Surface fitting (§6.3.4, §6.3.7)
    struct {
        std::string method = "rbf"; // poly | spline | bicubic | rbf | modeled_mask_mesh
        std::string robust_loss = "huber"; // huber | tukey
        float huber_delta = 1.5f;
        int irls_max_iterations = 10;
        float irls_tolerance = 1e-4f;
        
        // Polynomial
        int polynomial_order = 2;
        
        // RBF
        std::string rbf_phi = "multiquadric"; // thinplate | multiquadric | gaussian
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
    // structure_score_t) * (1 - masked_fraction_t), where structure_score_t is
    // dimensionless via local noise normalization.
    float tile_weight_lambda_structure = 1.0f;
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

// RBF interpolation (v3.3 §6.3.7)
Matrix2Df fit_rbf_surface(
    const std::vector<GridCell>& grid_cells,
    int image_width, int image_height,
    int grid_spacing,
    const BGEConfig& config);

// Polynomial surface fitting (v3.3 §6.3.7)
Matrix2Df fit_polynomial_surface(
    const std::vector<GridCell>& grid_cells,
    int image_width, int image_height,
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
    std::string mode;
    bool was_single_channel = false;
};

struct SamplePoint {
    int x = 0;
    int y = 0;
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
