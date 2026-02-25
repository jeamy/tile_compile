#pragma once

#include "tile_compile/core/types.hpp"
#include <Eigen/Dense>
#include <string>
#include <vector>

namespace tile_compile::image {

// BGE Configuration (matches YAML structure from v3.3 §6.3)
struct BGEConfig {
    bool enabled = false;
    
    // Tile sampling (§6.3.2)
    float sample_quantile = 0.20f;
    float structure_thresh_percentile = 0.90f;
    int min_tiles_per_cell = 3;
    
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
        std::string method = "rbf"; // poly | spline | bicubic | rbf
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
};

// Tile background sample
struct TileBGSample {
    float x, y;           // Tile center position
    float bg_value;       // Background estimate
    float weight;         // Reliability weight
    bool valid;           // Sample is valid
};

// Grid cell for coarse aggregation
struct GridCell {
    int cell_x, cell_y;   // Grid cell indices
    float center_x, center_y; // Cell center position
    float bg_value;       // Aggregated background
    float weight;         // Aggregated weight
    int n_samples;        // Number of tile samples in cell
    bool valid;           // Cell has sufficient samples
};

// Background model result
struct BackgroundModel {
    Matrix2Df model;      // Interpolated background surface
    std::vector<GridCell> grid_cells; // Grid cells used for fitting
    int n_valid_cells;    // Number of valid cells
    float rms_residual;   // RMS of fit residuals
    bool success;         // Model was successfully computed
    std::string error_message;
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
    int tile_samples_total = 0;
    int tile_samples_valid = 0;
    int grid_cells_valid = 0;
    float fit_rms_residual = 0.0f;
    float mean_shift = 0.0f;
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
    int image_width = 0;
    int image_height = 0;
    int grid_spacing = 0;
    std::string method;
    std::string robust_loss;
    std::string insufficient_cell_strategy;
    std::vector<BGEChannelDiagnostics> channels;
};

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
float huber_loss(float r, float delta);
float tukey_loss(float r, float c);
float huber_weight(float r, float delta);
float tukey_weight(float r, float c);

} // namespace tile_compile::image
