#include "runner_phase_quality_analysis.hpp"
#include "runner_shared.hpp"

#include "tile_compile/core/events.hpp"
#include "tile_compile/core/utils.hpp"
#include "tile_compile/preprocessing/contract.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <numeric>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace tile_compile::runner {

namespace {
using tile_compile::runner::parallel_for_indices;
using tile_compile::runner::default_parallel_workers;

inline std::string pname(preprocessing::Phase p) {
    return preprocessing::phase_to_string(p);
}
}


namespace fs = std::filesystem;
namespace core = tile_compile::core;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

std::string frame_exclusion_reason_to_string(FrameExclusionReason r) {
    switch (r) {
        case FrameExclusionReason::not_excluded:         return "not_excluded";
        case FrameExclusionReason::fwhm_too_high:        return "fwhm_too_high";
        case FrameExclusionReason::eccentricity_too_high: return "eccentricity_too_high";
        case FrameExclusionReason::star_count_too_low:   return "star_count_too_low";
        case FrameExclusionReason::registration_cc_too_low: return "registration_cc_too_low";
        case FrameExclusionReason::manual_exclude:       return "manual_exclude";
    }
    return "unknown";
}

namespace {

/// Compute eccentricity from FWHM_x and FWHM_y.
/// eccentricity = 1 – min(fwhm_x, fwhm_y) / max(fwhm_x, fwhm_y)
/// Returns 0 for perfectly round stars; 1 for infinitely elongated.
float compute_eccentricity(float fwhm_x, float fwhm_y) {
    if (fwhm_x <= 0.0f || fwhm_y <= 0.0f) return 0.0f;
    const float mn = std::min(fwhm_x, fwhm_y);
    const float mx = std::max(fwhm_x, fwhm_y);
    return 1.0f - mn / mx;
}

float measure_clip_fraction_ptr(const float* data, int rows, int cols) {
    if (data == nullptr || rows <= 0 || cols <= 0) return 0.0f;
    float min_v = std::numeric_limits<float>::infinity();
    float max_v = -std::numeric_limits<float>::infinity();
    size_t finite = 0;
    const size_t total = static_cast<size_t>(rows) * static_cast<size_t>(cols);
    for (size_t i = 0; i < total; ++i) {
        const float v = data[i];
        if (!std::isfinite(v)) continue;
        min_v = std::min(min_v, v);
        max_v = std::max(max_v, v);
        ++finite;
    }
    if (finite == 0) return 0.0f;

    const float hi_threshold = max_v <= 1.25f ? 0.999f : max_v - std::max(1e-6f, std::fabs(max_v) * 1e-5f);
    const float lo_threshold = min_v >= -0.01f ? 0.0f : min_v + std::max(1e-6f, std::fabs(min_v) * 1e-5f);
    size_t clipped = 0;
    for (size_t i = 0; i < total; ++i) {
        const float v = data[i];
        if (!std::isfinite(v)) continue;
        if (v <= lo_threshold || v >= hi_threshold) ++clipped;
    }
    return static_cast<float>(clipped) / static_cast<float>(finite);
}

/// Compute per-frame S/N estimate: (gradient_energy proxy of signal) / noise.
float estimate_snr(float gradient_energy, float noise) {
    if (noise <= 0.0f || !std::isfinite(noise)) return 0.0f;
    return std::sqrt(std::max(0.0f, gradient_energy)) / noise;
}

/// Sigma-clip threshold computation on a sorted sample.
/// Returns (median, sigma) of the trimmed distribution.
std::pair<float, float> robust_stats(std::vector<float> values) {
    if (values.empty()) return {0.0f, 0.0f};
    std::sort(values.begin(), values.end());
    const size_t n = values.size();
    const float med = values[n / 2];
    std::vector<float> abs_dev;
    abs_dev.reserve(n);
    for (float v : values) abs_dev.push_back(std::fabs(v - med));
    std::sort(abs_dev.begin(), abs_dev.end());
    const float mad = abs_dev[n / 2];
    const float sigma = mad * 1.4826f; // consistent estimator
    return {med, sigma};
}

/// Auto-rejection in "sigma" mode: exclude frames where `values[i]` exceeds
/// `median + k * sigma` (for "higher is worse" metrics like FWHM and
/// eccentricity) or falls below `median - k * sigma` (for star_count).
/// Operates in-place on `excluded`; already-excluded frames are skipped.
void apply_sigma_rejection_high(const std::vector<float>& values,
                                 float k,
                                 std::vector<bool>& excluded) {
    std::vector<float> active;
    active.reserve(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        if (!excluded[i] && std::isfinite(values[i])) active.push_back(values[i]);
    }
    auto [med, sig] = robust_stats(active);
    if (sig <= 0.0f) return;
    const float threshold = med + k * sig;
    for (size_t i = 0; i < values.size(); ++i) {
        if (!excluded[i] && values[i] > threshold) excluded[i] = true;
    }
}

void apply_sigma_rejection_low(const std::vector<float>& values,
                                float k,
                                std::vector<bool>& excluded) {
    std::vector<float> active;
    active.reserve(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        if (!excluded[i] && std::isfinite(values[i])) active.push_back(values[i]);
    }
    auto [med, sig] = robust_stats(active);
    if (sig <= 0.0f) return;
    const float threshold = med - k * sig;
    for (size_t i = 0; i < values.size(); ++i) {
        if (!excluded[i] && values[i] < threshold) excluded[i] = true;
    }
}

/// Write frame_quality.csv.
void write_frame_quality_csv(const fs::path& path,
                              const std::vector<FrameQualityRecord>& records) {
    std::ofstream f(path);
    f << "index,filename,included,exclusion_reason,star_count,fwhm,fwhm_x,"
         "fwhm_y,eccentricity,background_median,background_rms,snr_estimate,"
         "clip_fraction,registration_cc,quality_score,manual_override\n";
    for (const auto& r : records) {
        f << r.index << ","
          << r.filename << ","
          << (r.included ? "1" : "0") << ","
          << frame_exclusion_reason_to_string(r.exclusion_reason) << ","
          << r.star_count << ","
          << std::fixed << std::setprecision(4)
          << r.fwhm << ","
          << r.fwhm_x << ","
          << r.fwhm_y << ","
          << r.eccentricity << ","
          << r.background_median << ","
          << r.background_rms << ","
          << r.snr_estimate << ","
          << r.clip_fraction << ","
          << r.registration_cc << ","
          << r.quality_score << ","
          << (r.manual_override ? "1" : "0") << "\n";
    }
}

/// Build quality_analysis.json artifact.
core::json build_quality_json(const QualityAnalysisContext& ctx,
                               const preprocessing::QualityFilterConfig& filter_cfg,
                               const std::string& rejection_mode) {
    core::json j;
    j["rejection_mode"]   = rejection_mode;
    j["min_stars"]        = filter_cfg.min_stars;
    j["max_fwhm_sigma"]   = filter_cfg.max_fwhm_sigma;
    j["max_eccentricity"] = filter_cfg.max_eccentricity;
    j["min_correlation"]  = filter_cfg.min_correlation;
    j["n_total"]    = static_cast<int>(ctx.records.size());
    j["n_accepted"] = static_cast<int>(ctx.accepted_indices.size());
    j["n_rejected"] = static_cast<int>(ctx.rejected_indices.size());
    j["frames"] = core::json::array();
    for (const auto& r : ctx.records) {
        j["frames"].push_back({
            {"index",            r.index},
            {"filename",         r.filename},
            {"included",         r.included},
            {"exclusion_reason", frame_exclusion_reason_to_string(r.exclusion_reason)},
            {"exclusion_detail", r.exclusion_detail},
            {"star_count",       r.star_count},
            {"fwhm",             r.fwhm},
            {"fwhm_x",           r.fwhm_x},
            {"fwhm_y",           r.fwhm_y},
            {"eccentricity",     r.eccentricity},
            {"background_median",r.background_median},
            {"background_rms",   r.background_rms},
            {"snr_estimate",     r.snr_estimate},
            {"clip_fraction",    r.clip_fraction},
            {"registration_cc",  r.registration_cc},
            {"quality_score",    r.quality_score},
            {"manual_override",  r.manual_override},
        });
    }
    return j;
}

} // namespace

// ---------------------------------------------------------------------------
// run_quality_analysis
// ---------------------------------------------------------------------------

bool run_quality_analysis(
    const std::string& run_id,
    const preprocessing::Config& cfg,
    const PreprocessPipelineContext& pipeline_ctx,
    const std::vector<metrics::FrameStarMetrics>& star_metrics,
    const fs::path& run_dir,
    core::EventEmitter& emitter,
    std::ostream& log_file,
    QualityAnalysisContext& out)
{
    // -----------------------------------------------------------------------
    // Phase: QUALITY_ANALYSIS
    // -----------------------------------------------------------------------
    emitter.phase_start(run_id, pname(preprocessing::Phase::QUALITY_ANALYSIS), log_file);

    const size_t n = pipeline_ctx.effective_frames.size();
    out.records.resize(n);

    std::atomic<size_t> measured{0};
    std::mutex progress_mutex;
    parallel_for_indices(n, default_parallel_workers(n), [&](size_t i) {
        FrameQualityRecord& r = out.records[i];
        r.index    = static_cast<int>(i);
        r.filename = pipeline_ctx.effective_frames[i].filename().string();

        // --- Star metrics ---
        if (i < star_metrics.size()) {
            r.star_count  = star_metrics[i].star_count;
            r.fwhm        = star_metrics[i].fwhm;
            r.fwhm_x      = star_metrics[i].fwhm_x;
            r.fwhm_y      = star_metrics[i].fwhm_y;
            r.eccentricity = compute_eccentricity(r.fwhm_x, r.fwhm_y);
        }

        // --- Background / noise from FrameMetrics ---
        if (i < pipeline_ctx.frame_metrics.size()) {
            const auto& fm     = pipeline_ctx.frame_metrics[i];
            r.background_median = fm.background;
            r.background_rms    = fm.noise;
            r.snr_estimate      = estimate_snr(fm.gradient_energy, fm.noise);
            r.quality_score     = fm.quality_score;
            if (pipeline_ctx.prewarped_frames.has_data(i)) {
                r.clip_fraction = measure_clip_fraction_ptr(
                    pipeline_ctx.prewarped_frames.frame_data(i),
                    pipeline_ctx.prewarped_frames.rows(),
                    pipeline_ctx.prewarped_frames.cols());
            }
        }

        // --- Registration correlation ---
        if (i < pipeline_ctx.frame_cc.size()) {
            r.registration_cc = pipeline_ctx.frame_cc[i];
        }

        r.included = pipeline_ctx.frame_has_data[i] != 0;
        if (!r.included) {
            r.exclusion_reason = FrameExclusionReason::registration_cc_too_low;
            r.exclusion_detail = "frame not prewarped (registration failed)";
        }

        const size_t done = measured.fetch_add(1) + 1;
        std::lock_guard<std::mutex> lock(progress_mutex);
        emitter.phase_progress(run_id, pname(preprocessing::Phase::QUALITY_ANALYSIS),
                               static_cast<float>(done) / static_cast<float>(n),
                               "measured frame " + std::to_string(done) + "/" +
                               std::to_string(n), log_file);
    });

    for (const auto& override_item : cfg.manual_frame_overrides) {
        for (auto& record : out.records) {
            const bool index_match = override_item.index >= 0 && record.index == override_item.index;
            const bool name_match = !override_item.filename.empty() &&
                record.filename == fs::path(override_item.filename).filename().string();
            if (index_match || name_match) {
                record.manual_override = true;
                record.manual_included = override_item.include;
            }
        }
    }

    // Write artifacts now – even before filtering
    const fs::path artifact_dir = run_dir / "artifacts" / "preprocess";
    fs::create_directories(artifact_dir);
    out.csv_path = artifact_dir / "frame_quality.csv";
    write_frame_quality_csv(out.csv_path, out.records);

    emitter.phase_end(run_id, pname(preprocessing::Phase::QUALITY_ANALYSIS), "ok",
                      {
                          {"n_frames",  static_cast<int>(n)},
                          {"csv_path",  out.csv_path.string()},
                      }, log_file);

    // -----------------------------------------------------------------------
    // Phase: FRAME_FILTERING
    // -----------------------------------------------------------------------
    emitter.phase_start(run_id, pname(preprocessing::Phase::FRAME_FILTERING), log_file);

    const auto& qf   = cfg.quality_filter;
    const std::string mode = qf.mode; // "auto", "strict", "relaxed", "off"

    std::vector<bool> excluded(n, false);
    // Propagate frames that failed registration
    for (size_t i = 0; i < n; ++i) {
        if (!out.records[i].included) excluded[i] = true;
    }

    if (mode != "off") {
        // --- Hard threshold: registration correlation ---
        for (size_t i = 0; i < n; ++i) {
            if (!excluded[i] && out.records[i].registration_cc < qf.min_correlation &&
                out.records[i].registration_cc > 0.0f) {
                excluded[i] = true;
                out.records[i].exclusion_reason =
                    FrameExclusionReason::registration_cc_too_low;
                out.records[i].exclusion_detail =
                    "cc=" + std::to_string(out.records[i].registration_cc) +
                    " < min_correlation=" + std::to_string(qf.min_correlation);
            }
        }

        // --- Hard threshold: minimum star count ---
        for (size_t i = 0; i < n; ++i) {
            if (!excluded[i] && out.records[i].star_count > 0 &&
                out.records[i].star_count < qf.min_stars) {
                excluded[i] = true;
                out.records[i].exclusion_reason =
                    FrameExclusionReason::star_count_too_low;
                out.records[i].exclusion_detail =
                    "stars=" + std::to_string(out.records[i].star_count) +
                    " < min_stars=" + std::to_string(qf.min_stars);
            }
        }

        // --- Hard threshold: max eccentricity ---
        for (size_t i = 0; i < n; ++i) {
            if (!excluded[i] && out.records[i].eccentricity > qf.max_eccentricity) {
                excluded[i] = true;
                out.records[i].exclusion_reason =
                    FrameExclusionReason::eccentricity_too_high;
                out.records[i].exclusion_detail =
                    "eccentricity=" + std::to_string(out.records[i].eccentricity) +
                    " > max_eccentricity=" + std::to_string(qf.max_eccentricity);
            }
        }

        // --- Sigma-based FWHM rejection ---
        if (mode == "auto" || mode == "strict" || mode == "relaxed") {
            const float k_fwhm = (mode == "strict")  ? 1.5f :
                                  (mode == "relaxed") ? 3.0f :
                                  qf.max_fwhm_sigma;  // "auto" uses configured value
            std::vector<float> fwhm_vals(n);
            for (size_t i = 0; i < n; ++i) fwhm_vals[i] = out.records[i].fwhm;
            apply_sigma_rejection_high(fwhm_vals, k_fwhm, excluded);
            for (size_t i = 0; i < n; ++i) {
                if (excluded[i] &&
                    out.records[i].exclusion_reason == FrameExclusionReason::not_excluded) {
                    out.records[i].exclusion_reason =
                        FrameExclusionReason::fwhm_too_high;
                    out.records[i].exclusion_detail =
                        "fwhm=" + std::to_string(out.records[i].fwhm) +
                        " sigma-rejected (k=" + std::to_string(k_fwhm) + ")";
                }
            }
        }
    }

    // Apply manual overrides (overrides auto-rejection in both directions)
    for (size_t i = 0; i < n; ++i) {
        if (out.records[i].manual_override) {
            excluded[i] = !out.records[i].manual_included;
            if (excluded[i]) {
                out.records[i].exclusion_reason = FrameExclusionReason::manual_exclude;
                out.records[i].exclusion_detail = "manually excluded";
            }
        }
    }

    // Build accepted / rejected index lists
    for (size_t i = 0; i < n; ++i) {
        out.records[i].included = !excluded[i];
        if (!excluded[i]) {
            out.accepted_indices.push_back(static_cast<int>(i));
        } else {
            out.rejected_indices.push_back(static_cast<int>(i));
        }
    }

    // Rewrite CSV with final filter decision
    write_frame_quality_csv(out.csv_path, out.records);
    pipeline_ctx.prewarped_frames.clear_mappings();

    // Write quality_analysis.json
    const core::json qa_json = build_quality_json(out, qf, mode);
    core::write_text(artifact_dir / "quality_analysis.json", qa_json.dump(2));

    // Emit warning if many frames rejected
    if (!out.records.empty()) {
        const float reject_fraction =
            static_cast<float>(out.rejected_indices.size()) /
            static_cast<float>(out.records.size());
        if (reject_fraction >= 0.5f) {
            emitter.warning(run_id,
                "FRAME_FILTERING: " + std::to_string(out.rejected_indices.size()) +
                "/" + std::to_string(out.records.size()) +
                " frames rejected (" +
                std::to_string(static_cast<int>(reject_fraction * 100.0f)) + "%)",
                log_file);
        }
    }

    if (out.accepted_indices.empty()) {
        emitter.phase_end(run_id, pname(preprocessing::Phase::FRAME_FILTERING), "error",
                          {{"error", "all frames were rejected by quality filter"}},
                          log_file);
        emitter.run_end(run_id, false, "error", log_file);
        return false;
    }

    emitter.phase_end(run_id, pname(preprocessing::Phase::FRAME_FILTERING), "ok",
                      {
                          {"rejection_mode",   mode},
                          {"n_total",    static_cast<int>(n)},
                          {"n_accepted", static_cast<int>(out.accepted_indices.size())},
                          {"n_rejected", static_cast<int>(out.rejected_indices.size())},
                          {"csv_path",   out.csv_path.string()},
                      }, log_file);

    return true;
}

} // namespace tile_compile::runner
