#include "runner_phase_aqmh_global_quality.hpp"

#include "tile_compile/metrics/aqmh_global_quality.hpp"

namespace tile_compile::runner {

bool run_phase_aqmh_global_quality(
    const std::string &run_id, const config::AqmhGlobalQualityConfig &cfg,
    const std::vector<float> &sharpness_summaries,
    const std::vector<float> &snr_summaries,
    const std::vector<float> &background_penalty_summaries,
    const std::vector<uint8_t> &frame_has_data,
    VectorXf &out_weights,
    std::vector<uint8_t> &out_input_invalid, core::EventEmitter &emitter,
    std::ostream &log_file) {
  emitter.phase_start(run_id, Phase::AQMH_GLOBAL_QUALITY,
                      "AQMH_GLOBAL_QUALITY", log_file);
  try {
    const auto result = metrics::compute_aqmh_global_quality(
        sharpness_summaries, snr_summaries, background_penalty_summaries, cfg);
    out_weights.resize(static_cast<Eigen::Index>(result.weights.size()));
    size_t masked_invalid_frames = 0;
    for (size_t i = 0; i < result.weights.size(); ++i) {
      const bool has_data =
          i < frame_has_data.size() && frame_has_data[i] != 0u;
      if (!has_data) ++masked_invalid_frames;
      out_weights[static_cast<Eigen::Index>(i)] =
          has_data ? result.weights[i] : 0.0f;
    }
    out_input_invalid = result.input_invalid;
    emitter.phase_end(run_id, Phase::AQMH_GLOBAL_QUALITY, "ok",
                      {{"weights", static_cast<uint64_t>(result.weights.size())},
                       {"masked_invalid_frames",
                        static_cast<uint64_t>(masked_invalid_frames)}},
                      log_file);
    return true;
  } catch (const std::exception &e) {
    emitter.phase_end(run_id, Phase::AQMH_GLOBAL_QUALITY, "error",
                      {{"error", e.what()}}, log_file);
    return false;
  }
}

} // namespace tile_compile::runner
