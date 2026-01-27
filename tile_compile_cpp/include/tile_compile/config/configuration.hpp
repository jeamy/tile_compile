#pragma once

#include <yaml-cpp/yaml.h>
#include <array>
#include <filesystem>
#include <string>
#include <vector>

namespace tile_compile::config {

namespace fs = std::filesystem;

struct PipelineConfig {
    std::string mode = "production";
    bool abort_on_fail = true;
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
    float warp_variance_warn = 2.0f;
    float warp_variance_max = 8.0f;
    float elongation_warn = 0.3f;
    float elongation_max = 0.4f;
    float tracking_error_max_px = 5.0f;
    bool reduced_mode_skip_clustering = false;
    std::array<int, 2> reduced_mode_cluster_range{2, 6};
};

struct V4Config {
    struct Phase6IoConfig {
        std::string mode = "roi"; // roi | lru | full
        int lru_capacity = 16;     // frames per thread (only for mode=lru)
    } phase6_io;

    struct AdaptiveTilesConfig {
        bool enabled = false;
        int max_refine_passes = 2;
        float refine_variance_threshold = 0.25f;
        int min_tile_size_px = 64;
        bool use_warp_probe = true;
        bool use_hierarchical = true;
        int initial_tile_size = 256;
        int probe_window = 256;
        int num_probe_frames = 5;
        float gradient_sensitivity = 2.0f;
        float split_gradient_threshold = 0.3f;
        int hierarchical_max_depth = 3;
    } adaptive_tiles;

    struct ConvergenceConfig {
        bool enabled = false;
        float epsilon_rel = 1.0e-3f;
    } convergence;

    struct MemoryLimitsConfig {
        int rss_warn_mb = 4096;
        int rss_abort_mb = 8192;
    } memory_limits;

    struct DiagnosticsConfig {
        bool enabled = true;
        bool warp_field = true;
        bool tile_invalid_map = true;
        bool warp_variance_hist = true;
    } diagnostics;

    int iterations = 3;
    float beta = 5.0f;
    float min_valid_tile_fraction = 0.3f;
    int parallel_tiles = 8;
    bool debug_tile_registration = true;
};

struct NormalizationConfig {
    bool enabled = true;
    std::string mode = "background";
    bool per_channel = true;
};

struct RegistrationConfig {
    struct LocalTilesConfig {
        float max_warp_delta_px = 0.3f;
        float ecc_cc_min = 0.2f;
        int min_valid_frames = 10;
        int temporal_smoothing_window = 11;
        float variance_window_sigma = 2.0f;
    } local_tiles;

    std::string mode = "local_tiles";
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

struct ClusteringConfig {
    std::string mode = "state_vector";
    std::vector<std::string> vector;
    std::array<int, 2> cluster_count_range{5, 30};
    std::string k_selection = "auto";
    bool use_silhouette = false;
    int fallback_quantiles = 15;
};

struct SyntheticConfig {
    std::string weighting = "global";
    int frames_min = 15;
    int frames_max = 30;
};

struct ReconstructionConfig {
    std::string weighting_function = "exponential";
    std::string window_function = "hanning";
    std::string tile_rescale = "median_after_background_subtraction";
};

struct StackingConfig {
    struct SigmaClipConfig {
        float sigma_low = 2.0f;
        float sigma_high = 2.0f;
        int max_iters = 3;
        float min_fraction = 0.5f;
    } sigma_clip;

    std::string method = "rej";
    std::string input_dir = "synthetic";
    std::string input_pattern = "syn_*.fits";
    std::string output_file = "stacked.fit";
};

struct ValidationConfig {
    float min_fwhm_improvement_percent = 5.0f;
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
    DataConfig data;
    LinearityConfig linearity;
    CalibrationConfig calibration;
    AssumptionsConfig assumptions;
    V4Config v4;
    NormalizationConfig normalization;
    RegistrationConfig registration;
    WienerDenoiseConfig wiener_denoise;
    GlobalMetricsConfig global_metrics;
    TileConfig tile;
    LocalMetricsConfig local_metrics;
    ClusteringConfig clustering;
    SyntheticConfig synthetic;
    ReconstructionConfig reconstruction;
    bool debayer = true;
    StackingConfig stacking;
    ValidationConfig validation;
    RuntimeLimitsConfig runtime_limits;
    
    static Config load(const fs::path& path);
    static Config from_yaml(const YAML::Node& node);
    
    void save(const fs::path& path) const;
    YAML::Node to_yaml() const;
    
    void validate() const;
};

std::string get_schema_json();

} // namespace tile_compile::config
