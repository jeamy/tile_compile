#pragma once

#include <algorithm>
#include <array>
#include <filesystem>
#include <string>
#include <yaml-cpp/yaml.h>

namespace tile_compile::config {

namespace fs = std::filesystem;

struct OutputConfig {
  std::string registered_dir = "registered";
  bool write_registered_frames = false;
  bool crop_to_nonzero_bbox = true;
};

struct DataConfig {
  int image_width = 0;
  int image_height = 0;
  std::string color_mode = "OSC";
  std::string bayer_pattern = "auto";
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
  // Conservative affine fine registration on already aligned proxy stars.
  // Enabled by default; rejected candidates leave the original warp unchanged
  // and are reported in global_registration.json.
  bool affine_refinement_enabled = true;
  // Smooth local inverse displacement field fitted after the affine/global
  // warp. Held-out, coverage, Jacobian, NCC, and overlap gates must all pass;
  // otherwise prewarp uses the unchanged affine/global warp.
  bool smooth_local_refinement_enabled = true;
  // Prewarp stage: interpolation kernel applied when warping frames onto the
  // common canvas ("bilinear" | "cubic").
  std::string prewarp_interpolation = "cubic";
  // OSC: demosaic before prewarp, then reconstruct the RGB channels directly
  // (instead of CFA prewarp + post-stack debayer).
  bool debayer_first = true;
  // Demosaic method used when debayer_first is active:
  // "bilinear" | "nearest" | "vng" | "edge_aware".
  std::string pre_debayer_method = "edge_aware";
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

  // Protects extended smooth-emission regions (galaxy disks, nebula halos) from
  // chroma denoising.  The protection mask is derived from a heavily-smoothed
  // luma image: any pixel whose smoothed luma exceeds
  //   sky_median + luma_sigma × sky_sigma
  // is treated as extended-source foreground and is blended back to the
  // original chroma (same mechanism as star_protection / luma_guard).
  struct ExtendedSourceProtectionConfig {
    bool enabled = false;
    float luma_sigma = 2.5f; // detection threshold above sky background (in σ)
    int   dilate_px  = 30;   // dilation applied after detection; covers PSF halos
  } extended_source_protection;

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

  // Removes smooth, large-scale (block-sized-and-up) chroma bias/variation
  // from the background -- the wavelet/bilateral stages below only shrink
  // *detail* relative to a progressively-blurred low-pass version and by
  // construction never touch that low-pass (DC/large-scale) remainder, so a
  // background color cast/blotch wider than roughly
  // 2^(chroma_wavelet.levels-1) * 0.75 px is architecturally invisible to
  // them. This stage estimates a coarse block-median surface of each chroma
  // plane using ONLY unprotected (background) pixels -- so real extended-
  // source/star color never contributes to the estimate -- smooths it into a
  // continuous field, and subtracts `strength` of its deviation from the
  // surface's own global background median (so the overall background
  // chroma level, e.g. real sky-glow color, is preserved; only spatial
  // *variation* across it is flattened). Default off: opt-in, since it
  // changes pixel values in the background beyond what the previous
  // filters did.
  struct LargeScaleBiasConfig {
    bool enabled = false;
    int block_size = 32;     // px, grid cell size for the block-median estimate
    float blur_sigma = 24.0f; // px, Gaussian smoothing of the block grid -> surface
    float strength = 1.0f;    // 0..1, fraction of the estimated bias removed
  } large_scale_bias;

  bool enabled = false;
  std::string color_space = "ycbcr_linear";      // ycbcr_linear | opponent_linear
  std::string apply_stage = "post_stack_linear"; // pre_stack_tiles | post_stack_linear
  bool protect_luma = true;
  float luma_guard_strength = 0.75f;
  // Reference chroma-to-luma noise-sigma ratio used to scale the wavelet/
  // bilateral strength to the dataset's actual noise level (see `adapt` in
  // chroma_denoise.cpp). Replaces a previous fixed ABSOLUTE reference sigma
  // (0.02f) that was calibrated for [0,1]-normalized data: on real,
  // unnormalized (ADU-scale) data the absolute chroma sigma is routinely in
  // the tens, so chroma_sigma/0.02f saturated the [0.8,1.4] clamp on every
  // real run, making the adaptation a no-op. Expressing the reference
  // relative to the image's own measured luma-noise sigma keeps it
  // scale-invariant across normalized, ADU, and stretched data.
  float adaptation_reference_ratio = 1.0f;
};

// Luminance noise reduction on the post-stack linear RGB, applied before
// BGE/PCC/HMS. The CFA-forward-drizzle pipeline has no luma denoise stage
// of its own -- the single-method cutover retired the old Classic
// pipeline's per-tile soft-threshold + Wiener denoise (tile_denoise) and it
// was never reimplemented for forward-drizzle output; chroma_denoise only
// ever touches chroma (protect_luma is unconditional there). This closes
// that gap: a multi-level wavelet soft-threshold denoise on the derived
// luma, reconstructed by adding the same smooth per-pixel brightness delta
// to R/G/B (not by rescaling each channel's RATIO to luma -- that divides
// two independently-noisy, correlated quantities and amplifies noise
// exactly where it is proportionally largest: faint background and the
// partially-protected PSF wings around stars). The additive delta leaves
// every channel DIFFERENCE, hence all perceived color, exactly unchanged.
struct LumaDenoiseConfig {
  struct StarProtectionConfig {
    bool enabled = true;
    float threshold_sigma = 6.0f;
    int dilate_px = 8;
  } star_protection;

  struct StructureProtectionConfig {
    bool enabled = true;
    float gradient_percentile = 90.0f;
  } structure_protection;

  struct WaveletConfig {
    bool enabled = true;
    int levels = 3;
    float threshold_scale = 1.5f;
    float soft_k = 1.0f;
  } wavelet;

  bool enabled = false;
  float luma_guard_strength = 0.85f;
  float blend_amount = 0.85f;
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
    float fwhm = 0.0f;
    float roundness = 0.0f;
    float star_count = 0.0f;
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
};

// ---------------------------------------------------------------------------
// Single-method reconstruction contract (CFA forward drizzle + multiband).
// Plan sections 6.1-6.3 / 11.2. This is the public `reconstruction:` config
// root; it is the only reconstruction configuration the pipeline consumes.
// ---------------------------------------------------------------------------

struct ReconstructionDiagnosticsConfig {
  std::string level = "summary";  // "summary" | "full"
  // M2 (plan section 11) is not yet a real pipeline phase: no clipping, no
  // quality weights, no multiband, no transactional store/chunking, no
  // resume contract. When true, run_phase_registration_prewarp additionally
  // computes the CFA-forward-drizzle Uniform-Control profile as a
  // best-effort, non-fatal diagnostic (artifacts/forward_drizzle_uniform_diagnostic.json)
  // after SAMPLING_GEOMETRY, using the same normalized-cache source the
  // final M2+ pipeline will use (never prewarped_frames). Off by default:
  // the single-threaded reference kernel's exact polygon clip is
  // meaningfully slower than SAMPLING_GEOMETRY's coverage-only touch test,
  // and this is diagnostic-only work that must never impose that cost on a
  // normal run.
  bool preview_forward_drizzle_uniform = false;
  // Independent opt-in diagnostic. Streams unclipped Uniform planes into
  // immutable generations under artifacts/forward_drizzle_uniform_store/.
  // current.json commits the complete, checked generation atomically.
  // Includes FITS IO reserve in the drizzle budget. No pipeline resume claim.
  bool persist_forward_drizzle_uniform_store = false;
};

struct ReconstructionDrizzleConfig {
  int internal_scale = 2;             // {1, 2}
  int output_scale = 1;              // {1, 2}, <= internal_scale
  std::string kernel = "square";     // MVP: only "square"
  float pixfrac = 0.8f;              // (0, 1]
  int robust_passes = 2;            // [1, 6]
  int min_clip_contributors = 5;    // >= 2; below this no sigma/MAD clipping
  int chunk_rows = 0;               // 0 = budgeted stripes, <=256 rows; >0 budget-checked
  int chunk_halo_rows = -1;         // compatibility; exact footprint enumeration needs no output halo
  size_t memory_budget_mb = 0;      // MiB; 0 inherits runtime_limits (library: 512 MiB)
};

struct ReconstructionClippingConfig {
  float clip_sigma_low = 3.0f;      // > 0
  float clip_sigma_high = 3.0f;     // > 0
  float min_fraction = 0.4f;        // (0, 1]
  float min_n_eff = 3.0f;           // >= 1
  // P0.1 (redundant_data_reload_analysis §"schwarze Artefakte"): default
  // false preserves the exact plan-11.8 8-step procedure (step 8's
  // min_fraction/min_n_eff veto rejects the pixel/channel outright, no
  // fallback value). When true, a pixel/channel that fails step 8 is NOT
  // erased: the sigma-clip-survivor set is used anyway (or, if the clip
  // rejected every candidate, every original candidate is used unclipped) --
  // trading the veto's noise guarantee for never leaving a geometrically
  // covered pixel black. See robust_clip_core in forward_drizzle.cpp.
  bool guard_fallback = false;
  // Off by default (opt-in): the CFA forward-drizzle kernel runs the sigma/
  // MAD clip independently per (output pixel, color channel), because R/G/B
  // are drizzled straight from the raw, still-mosaiced Bayer samples --
  // each channel has its own native sub-pixel sampling grid (R and B each
  // ~1/4 of raw pixels, G ~1/2), by design (see
  // docs/forward_drizzle_v2_zielarchitektur_2026-09-12_de.md). In faint
  // regions, where per-channel candidate counts sit near
  // min_clip_contributors, that independence means the surviving frame set
  // is effectively an independent coin-flip per channel: verified on a
  // real M42 run, corr(R,B) detail = -0.41 in faint sky (negative --
  // anti-correlated -- vs. the positive correlation a real, achromatic
  // source would show), present already in forward_drizzle_multiband_*.fit
  // before any downstream chroma/luma denoise runs.
  //
  // When enabled, a FRAME is rejected consistently across every channel it
  // contributes to at a given output pixel, instead of each channel
  // re-deriving its own independent reject decision: each channel still
  // runs its own weighted-median/MAD clip pass unchanged (per-channel
  // VALUES are never compared against each other -- R and B sample
  // physically different sub-pixel positions, so there is no shared value
  // to test), but a frame is only actually dropped if
  // shared_frame_rejection_consensus or more of the channels that had a
  // candidate from it flagged it as an outlier. This targets exactly the
  // failure mode above (independent per-channel noise in the survivor
  // *set*) without touching per-channel weights, footprints, sample
  // counts, or geometry -- so point-source (star) sharpness is
  // structurally unaffected, and a consensus (not union) rule means a
  // frame is kept unless most channels that saw it agree it's bad, so a
  // real per-channel faint-detail difference (not every source is equally
  // bright in R/G/B) does not itself get treated as an outlier vote.
  //
  // CPU-only for now: enabling this forces the forward-drizzle stage to
  // run on CPU even when acceleration_backend is "cuda" (no CUDA
  // implementation yet), to avoid silently breaking CPU/CUDA bit-exact
  // parity on a run that enables it.
  bool shared_frame_rejection = false;
  // Fraction (0,1] of the channels that had a candidate from a given frame
  // at a given pixel that must flag it as an outlier before it is rejected
  // for ALL of those channels. 0.5 = majority. Lower values reject more
  // aggressively (closer to "any channel objects" = union); higher values
  // reject more conservatively (closer to "every channel must agree").
  float shared_frame_rejection_consensus = 0.5f;
};

struct ReconstructionCoverageGateConfig {
  int min_frames = 2;                       // >= 2
  float min_supported_fraction = 0.995f;    // (0, 1]
  float min_channel_n_eff_floor = 3.0f;     // >= 1
  float min_channel_n_eff_fraction = 0.15f; // (0, 1]
  int min_analysis_pixels = 1024;           // >= 1
  long long max_internal_hole_area_px = 0;  // >= 0
};

struct ReconstructionQualityPyramidConfig {
  int scales = 4;                   // [1, 8]
  int base_window_px = 4;           // >= 1
  float sharpness_weight = 0.6f;    // >= 0; sharpness_weight+snr_weight > 0
  float snr_weight = 0.4f;          // >= 0
  float score_scale = 1.8f;         // > 0
  float artifact_sigma = 3.0f;      // > 0
  float max_artifact_fraction = 0.25f;  // (0, 1]
};

struct ReconstructionQualityConfig {
  ReconstructionQualityPyramidConfig pyramid;
};

struct ReconstructionMultibandConfig {
  bool enabled = true;
  int levels = 3;                         // [1, 4]; >=2 needs pyramid.scales>=2
  float alpha_cap = 1.0f;                 // [0, 1]
  float fine_quality_exponent = 4.0f;     // >= 0
  float medium_quality_exponent = 2.0f;   // >= 0
  float min_quality_separation = 0.05f;   // 0 <= min < full <= 1
  float full_quality_separation = 0.20f;
  float min_effective_samples = 8.0f;     // 1 <= min < full
  float full_effective_samples = 24.0f;
};

// Candidate-selection gates of plan 15.3.4/15.3.5. Mirrors
// reconstruction::MultibandValidationConfig; kept in the config layer so the
// runner can populate the validation struct from the effective YAML.
struct ReconstructionMultibandValidationConfig {
  double fwhm_ratio_max = 0.95;               // > 0; multiband median FWHM <= x*raw
  double p90_fwhm_ratio_max = 1.00;           // > 0; vs raw
  double tail_ratio_max = 1.10;               // > 0; vs raw
  double elongation_ratio_max = 1.08;         // > 0; vs raw
  double background_rms_ratio_max = 1.05;     // > 0; vs UNIFORM (raw veto + promotion)
  double seam_ratio_max = 1.05;               // > 0; vs UNIFORM at support boundary
  int min_stars_fwhm = 20;                    // >= 0; stars needed for FWHM metric
  int min_stars_p90_tail_elongation = 30;     // >= 0; stars for p90/tail/elongation
  double max_fwhm_ci_relative_width = 0.10;   // > 0; bootstrap 95% CI width veto
};

struct ReconstructionConfig {
  bool delete_source_cache_after_run = false;
  bool keep_profile_cache_after_run = false;
  float common_overlap_required_fraction = 1.0f;  // (0,1], dense frame footprints
  ReconstructionDiagnosticsConfig diagnostics;
  ReconstructionDrizzleConfig drizzle;
  ReconstructionClippingConfig clipping;
  ReconstructionCoverageGateConfig coverage_gate;
  ReconstructionQualityConfig quality;
  ReconstructionMultibandConfig multiband;
  ReconstructionMultibandValidationConfig multiband_validation;

  // Throws tile_compile::ValidationError on a contract violation (plan 6.3).
  void validate() const;
};

struct StackingConfig {
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
  // On/off is expressed solely through `method` ("none" == disabled); there
  // is no separate `enabled` flag. A prior `bge.enabled` legacy field was
  // removed because it could disagree with `method` (whichever was set
  // last silently won depending on write order) -- see
  // docs/configuration_reference.md "bge.method" for the migration note.
  std::string method = "none"; // none | classic | autobge | auto

  // Programmatic BGE selection used when method == "auto".
  // The runner measures the background gradient strength of the stacked image
  // before deciding whether AutoBGE is needed.
  struct AutoDetectConfig {
    // Minimum gradient amplitude relative to sky median to trigger AutoBGE.
    // Gradient amplitude = (max − min of linear-plane fit) / median_sky.
    // Values below this threshold skip BGE entirely.
    float gradient_threshold = 0.05f;
    // When an extended source (galaxy, large nebula) is detected in the
    // background grid it is excluded from the AutoBGE sample points so the
    // polynomial/RBF fit uses only real sky pixels.
    // Detection: block-median grid residual > extended_source_sigma × sky_sigma.
    float extended_source_sigma = 3.0f;
    // Morphological dilation (px) applied to the detected extended-source mask
    // before it is used as a sampling exclusion region.
    int extended_source_dilate_px = 50;
  } auto_detect;

  struct AutoBGEConfig {
    int num_sample_points = 0;
    int poly_degree = 2;
    float rbf_smooth = 2.0f;
    int downsample_scale = 4;
    int patch_size = 35;
    std::string patch_estimator = "sigma_clipped_median";
    std::string stretch_mode = "linear";
    float stretch_target_median = 0.25f;
    int border_margin = 10;
    float bright_exclusion_fraction = 0.2f;
    int gradient_descent_max_iters = 100;
    int random_seed = 42;
    bool normalize_between_stages = true;
    bool apply_guards = true;
    std::string mono_mode = "rgb_duplicate";
    std::vector<std::vector<std::array<float, 2>>> exclusion_polygons;
    // Optional manually placed AutoBGE sample points. Normalized [0..1] in
    // the original image space.
    std::vector<std::array<float, 2>> user_sample_points;
  } autobge;
  
  // Tile sampling (v3.3 §6.3.2)
  float sample_quantile = 0.20f;
  std::string sample_estimator = "quantile";
  float min_sample_bg_value = 1.0f;
  float structure_thresh_percentile = 0.65f;
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

struct HyperMetricStretchConfig {
  bool enabled = true;
  bool require_successful_pcc = true;
  std::string mode = "ready_to_use";       // ready_to_use | scientific
  std::string sensor_profile = "rec709";   // rec709 | profile name | auto
  std::string fallback_profile = "rec709";
  bool adaptive_anchor = true;
  float target_bg = 0.15f;
  float protect_b = 6.0f;
  float convergence_power = 3.5f;
  std::string log_d_mode = "auto";         // auto | fixed
  float fixed_log_d = 2.0f;
  std::string color_strategy = "fixed";    // auto | fixed
  float fixed_color_strategy = 0.0f;       // [-1,1]
  float color_grip = 1.0f;
  float shadow_convergence = 0.0f;
  float linear_expansion = 0.0f;
  // Percentile used as the "must not clip" highlight reference in
  // ready_to_use's adaptive output scaling; 100 = true brightest pixel
  // (never clips, previous behaviour). Lowering it (e.g. 99.9) allows a
  // small, bounded top fraction of pixels to clip in exchange for more
  // contrast in stars/highlights; it does not move the mid-tone/background
  // level, which stays pinned to target_bg regardless.
  float highlight_ceiling_percentile = 100.0f;
  bool write_channels = false;
  std::string output_rgb = "stacked_rgb_hms.fits";
};

struct RuntimeLimitsConfig {
  float hard_abort_hours = 6.0f;
  int parallel_workers = 4;
  int memory_budget = 512;
  std::string acceleration_backend = "auto";
};

struct Config {
  OutputConfig output;
  DataConfig data;
  LinearityConfig linearity;
  CalibrationConfig calibration;
  NormalizationConfig normalization;
  RegistrationConfig registration;
  DitheringConfig dithering;
  ChromaDenoiseConfig chroma_denoise;
  LumaDenoiseConfig luma_denoise;
  GlobalMetricsConfig global_metrics;
  TileConfig tile;
  ReconstructionConfig reconstruction;
  AstrometryConfig astrometry;
  BGEConfig bge;
  PCCConfig pcc;
  HyperMetricStretchConfig hypermetric_stretch;
  StackingConfig stacking;
  RuntimeLimitsConfig runtime_limits;

  static Config load(const fs::path &path);
  static Config from_yaml(const YAML::Node &node);
  static Config from_yaml_text(const std::string &yaml_text);

  // Like from_yaml_text but first applies the single-method legacy config
  // migration (plan section 6.5): rejects `method`/engine keys fail-closed
  // (throws tile_compile::ConfigError) and strips removed structural blocks,
  // recording every change in `report` (see legacy_config_migration.hpp). Used
  // by the production run path.
  static Config from_yaml_text_migrated(const std::string &yaml_text,
                                        struct ConfigMigrationReport &report);

  void save(const fs::path &path) const;
  YAML::Node to_yaml() const;

  void validate() const;
};

std::string get_schema_json();

} // namespace tile_compile::config
