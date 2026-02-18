#include "runner_phase_registration.hpp"

#include "tile_compile/core/utils.hpp"
#include "tile_compile/image/cfa_processing.hpp"
#include "tile_compile/io/fits_io.hpp"
#include "tile_compile/registration/global_registration.hpp"
#include "tile_compile/registration/registration.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <thread>
#include <vector>

namespace tile_compile::runner {

namespace fs = std::filesystem;
namespace core = tile_compile::core;
namespace image = tile_compile::image;
namespace io = tile_compile::io;
namespace registration = tile_compile::registration;

bool run_phase_registration_prewarp(
    const std::string &run_id, const config::Config &cfg,
    const std::vector<std::filesystem::path> &frames,
    const std::filesystem::path &run_dir, int height, int width,
    ColorMode detected_mode, const std::string &detected_bayer_str,
    const std::vector<image::NormalizationScales> &norm_scales,
    const std::vector<FrameMetrics> &frame_metrics,
    const VectorXf &global_weights, const io::FitsHeader &first_header,
    core::EventEmitter &emitter, std::ostream &log_file,
    PhaseRegistrationContext &out) {
  auto load_frame_normalized =
      [&](size_t frame_index) -> std::pair<Matrix2Df, io::FitsHeader> {
    auto frame_pair = io::read_fits_float(frames[frame_index]);
    Matrix2Df img = frame_pair.first;
    image::apply_normalization_inplace(img, norm_scales[frame_index],
                                       detected_mode, detected_bayer_str, 0, 0);
    return {img, frame_pair.second};
  };

  // Config parameters (v3: single global registration, no tile-ECC)
  auto compute_worker_count = [&](size_t task_count) -> int {
    int workers = cfg.runtime_limits.parallel_workers;
    if (workers < 1) {
      workers = 1;
    }
    int cpu_cores = static_cast<int>(std::thread::hardware_concurrency());
    if (cpu_cores > 0) {
      workers = std::min(workers, cpu_cores);
    }
    if (task_count > 0) {
      workers =
          std::min(workers, static_cast<int>(std::max<size_t>(1, task_count)));
    }
    return std::max(1, workers);
  };

  emitter.phase_start(run_id, Phase::REGISTRATION, "REGISTRATION", log_file);

  std::vector<WarpMatrix> global_frame_warps(frames.size(),
                                             registration::identity_warp());
  std::vector<float> global_frame_cc(frames.size(), 0.0f);
  std::string global_reg_status = "skipped";
  core::json global_reg_extra;
  int global_ref_idx = 0;
  // pick best by global_weight, fallback quality_score, fallback mid-frame
  if (!frame_metrics.empty()) {
    float best_w = -1.0f;
    for (int i = 0; i < static_cast<int>(frame_metrics.size()); ++i) {
      float w = (i < global_weights.size()) ? global_weights[i]
                                            : frame_metrics[i].quality_score;
      if (w > best_w) {
        best_w = w;
        global_ref_idx = i;
      }
    }
    if (best_w < 0.05f) {
      // fallback to highest quality_score
      float best_q = frame_metrics[global_ref_idx].quality_score;
      for (int i = 0; i < static_cast<int>(frame_metrics.size()); ++i) {
        if (frame_metrics[i].quality_score > best_q) {
          best_q = frame_metrics[i].quality_score;
          global_ref_idx = i;
        }
      }
    }
  }
  if (global_ref_idx < 0 || global_ref_idx >= static_cast<int>(frames.size())) {
    global_ref_idx = static_cast<int>(frames.size() / 2);
  }
  global_reg_extra["ref_frame"] = global_ref_idx;

  if (!frames.empty()) {
    try {
      Matrix2Df ref_full;
      auto pair = load_frame_normalized(static_cast<size_t>(global_ref_idx));
      ref_full = std::move(pair.first);
      if (ref_full.size() <= 0) {
        global_reg_status = "error";
        global_reg_extra["error"] = "ref_frame_empty";
      } else {
        Matrix2Df ref_reg = (detected_mode == ColorMode::OSC)
                                ? image::cfa_green_proxy_downsample2x2(
                                      ref_full, detected_bayer_str)
                                : registration::downsample2x2_mean(ref_full);
        float global_reg_scale = 1.0f;
        if (ref_reg.rows() > 0) {
          int full_h2 = ref_full.rows() - (ref_full.rows() % 2);
          global_reg_scale =
              static_cast<float>(full_h2) / static_cast<float>(ref_reg.rows());
        }
        Matrix2Df ref_ecc = registration::prepare_ecc_image(ref_reg);

        // Diagnostic: proxy image stats
        {
          float rmin = ref_reg.minCoeff();
          float rmax = ref_reg.maxCoeff();
          float rmean = ref_reg.mean();
          std::cerr << "[REG-DIAG] ref_reg " << ref_reg.rows() << "x" << ref_reg.cols()
                    << " min=" << rmin << " max=" << rmax << " mean=" << rmean
                    << std::endl;
        }

        const int reg_workers = compute_worker_count(frames.size());
        std::cout << "[REGISTRATION] Using " << reg_workers
                  << " parallel workers for " << frames.size() << " frames"
                  << std::endl;
        std::mutex reg_log_mutex;
        std::mutex reg_progress_mutex;
        std::atomic<size_t> reg_next{0};
        std::atomic<size_t> reg_done{0};
        std::atomic<bool> reg_failed{false};
        std::string reg_error;

        auto reg_worker = [&]() {
          while (true) {
            const size_t fi = reg_next.fetch_add(1);
            if (fi >= frames.size()) {
              break;
            }
            try {
              if (static_cast<int>(fi) == global_ref_idx) {
                global_frame_warps[fi] = registration::identity_warp();
                global_frame_cc[fi] = 1.0f;
              } else {
                Matrix2Df mov_full;
                {
                  auto pair = load_frame_normalized(fi);
                  mov_full = std::move(pair.first);
                }
                if (mov_full.size() <= 0) {
                  global_frame_warps[fi] = registration::identity_warp();
                  global_frame_cc[fi] = 0.0f;
                } else {
                  Matrix2Df mov_reg = (detected_mode == ColorMode::OSC)
                                          ? image::cfa_green_proxy_downsample2x2(
                                                mov_full, detected_bayer_str)
                                          : registration::downsample2x2_mean(
                                                mov_full);
                  // Diagnostic: first few moving frames
                  if (fi < 3) {
                    float mmin = mov_reg.minCoeff();
                    float mmax = mov_reg.maxCoeff();
                    float mmean = mov_reg.mean();
                    std::lock_guard<std::mutex> lock(reg_log_mutex);
                    std::cerr << "[REG-DIAG] mov_reg[" << fi << "] "
                              << mov_reg.rows() << "x" << mov_reg.cols()
                              << " min=" << mmin << " max=" << mmax
                              << " mean=" << mmean << std::endl;
                  }

                  if (mov_reg.rows() != ref_reg.rows() ||
                      mov_reg.cols() != ref_reg.cols()) {
                    global_frame_warps[fi] = registration::identity_warp();
                    global_frame_cc[fi] = 0.0f;
                  } else {
                    // Delegate to canonical cascade in module
                    auto sfr = registration::register_single_frame(
                        mov_reg, ref_reg, cfg.registration);

                    if (sfr.reg.success) {
                      global_frame_cc[fi] = sfr.reg.correlation;
                      WarpMatrix w_full = sfr.reg.warp;
                      w_full(0, 2) *= global_reg_scale;
                      w_full(1, 2) *= global_reg_scale;
                      global_frame_warps[fi] = w_full;
                    } else {
                      global_frame_warps[fi] = registration::identity_warp();
                      global_frame_cc[fi] = 0.0f;
                    }

                    // Per-frame logging
                    if (fi < 5 || fi == frames.size() - 1 || (fi % 50 == 0)) {
                      std::lock_guard<std::mutex> lock(reg_log_mutex);
                      std::cerr << "[REG] frame " << fi << "/" << frames.size()
                                << " method=" << sfr.method_used
                                << " ncc_id=" << sfr.ncc_identity
                                << " cc=" << global_frame_cc[fi] << std::endl;
                    }
                  }
                }
              }
            } catch (const std::exception &e) {
              reg_failed.store(true, std::memory_order_relaxed);
              std::lock_guard<std::mutex> lock(reg_log_mutex);
              if (reg_error.empty()) {
                reg_error = e.what();
              }
            } catch (...) {
              reg_failed.store(true, std::memory_order_relaxed);
              std::lock_guard<std::mutex> lock(reg_log_mutex);
              if (reg_error.empty()) {
                reg_error = "unknown_error";
              }
            }

            const size_t done = reg_done.fetch_add(1) + 1;
            if (done % 5 == 0 || done == frames.size()) {
              const float p = frames.empty()
                                  ? 1.0f
                                  : static_cast<float>(done) /
                                        static_cast<float>(frames.size());
              std::lock_guard<std::mutex> lock(reg_progress_mutex);
              emitter.phase_progress(
                  run_id, Phase::REGISTRATION, p,
                  "global_reg " + std::to_string(done) + "/" +
                      std::to_string(frames.size()) + " workers=" +
                      std::to_string(reg_workers),
                  log_file);
            }
          }
        };

        if (reg_workers > 1) {
          std::vector<std::thread> workers;
          workers.reserve(static_cast<size_t>(reg_workers));
          for (int w = 0; w < reg_workers; ++w) {
            workers.emplace_back(reg_worker);
          }
          for (auto &worker : workers) {
            if (worker.joinable()) {
              worker.join();
            }
          }
        } else {
          reg_worker();
        }

        if (reg_failed.load(std::memory_order_relaxed)) {
          throw std::runtime_error(reg_error.empty() ? "registration_failed"
                                                     : reg_error);
        }

        global_reg_status = "ok";
        try {
          core::json j;
          j["num_frames"] = static_cast<int>(frames.size());
          j["scale"] = global_reg_scale;
          j["ref_frame"] = global_ref_idx;
          j["cc"] = core::json::array();
          j["warps"] = core::json::array();
          j["dithering"] = {
              {"enabled", cfg.dithering.enabled},
              {"min_shift_px", cfg.dithering.min_shift_px},
              {"detected_fraction", 0.0},
          };
          int shifts_detected = 0;
          for (size_t fi = 0; fi < frames.size(); ++fi) {
            const auto &w = global_frame_warps[fi];
            j["cc"].push_back(global_frame_cc[fi]);
            const float shift_mag =
                std::sqrt(w(0, 2) * w(0, 2) + w(1, 2) * w(1, 2));
            if (cfg.dithering.enabled && shift_mag >= cfg.dithering.min_shift_px) {
              shifts_detected++;
            }
            j["warps"].push_back({
                {"a00", w(0, 0)},
                {"a01", w(0, 1)},
                {"tx", w(0, 2)},
                {"a10", w(1, 0)},
                {"a11", w(1, 1)},
                {"ty", w(1, 2)},
                {"shift_px", shift_mag},
            });
          }
          if (!frames.empty()) {
            j["dithering"]["detected_fraction"] =
                static_cast<double>(shifts_detected) /
                static_cast<double>(frames.size());
            j["dithering"]["detected_count"] = shifts_detected;
            j["dithering"]["total_frames"] = static_cast<int>(frames.size());
          }
          core::write_text(run_dir / "artifacts" / "global_registration.json",
                           j.dump(2));

          if (cfg.output.write_registered_frames) {
            fs::create_directories(run_dir / cfg.output.registered_dir);
            // first_header is available from outer scope

            for (size_t fi = 0; fi < frames.size(); ++fi) {
              if (static_cast<size_t>(fi) >= global_frame_warps.size())
                continue;
              const auto &w = global_frame_warps[fi];

              auto pair = load_frame_normalized(fi);
              Matrix2Df img = std::move(pair.first);
              if (img.size() <= 0)
                continue;

              Matrix2Df out_img = image::apply_global_warp(img, w,
                                                          detected_mode);

              std::ostringstream name;
              name << "frame_" << std::setw(4) << std::setfill('0') << fi
                   << ".fits";
              io::write_fits_float(run_dir / cfg.output.registered_dir /
                                       name.str(),
                                   out_img, first_header);
            }
          }

        } catch (...) {
        }
      }
    } catch (const std::exception &e) {
      global_reg_status = "error";
      global_reg_extra["error"] = e.what();
    } catch (...) {
      global_reg_status = "error";
      global_reg_extra["error"] = "unknown_error";
    }
  } else {
    global_reg_extra["reason"] = "no_frames";
  }

  // Reject implausible global registration outliers before downstream phases.
  // These outliers can pass NCC but still produce heavy tile/grid artifacts.
  int reg_reject_orientation_outliers = 0;
  int reg_reject_reflection_outliers = 0;
  int reg_reject_scale_outliers = 0;
  int reg_reject_cc_outliers = 0;
  int reg_reject_shift_outliers = 0;
  core::json reg_rejected_frames = core::json::array();
  if (cfg.registration.reject_outliers) {
    std::vector<float> cc_positive;
    cc_positive.reserve(frames.size());
    std::vector<float> shift_mags_positive;
    shift_mags_positive.reserve(frames.size());
    for (size_t fi = 0; fi < frames.size(); ++fi) {
      if (global_frame_cc[fi] <= 0.0f) {
        continue;
      }
      cc_positive.push_back(global_frame_cc[fi]);
      const auto &w = global_frame_warps[fi];
      const float shift_mag =
          std::sqrt(w(0, 2) * w(0, 2) + w(1, 2) * w(1, 2));
      shift_mags_positive.push_back(shift_mag);
    }

    auto robust_median = [](std::vector<float> values) -> float {
      if (values.empty()) {
        return 0.0f;
      }
      const size_t mid = values.size() / 2;
      std::nth_element(values.begin(), values.begin() + static_cast<long>(mid),
                       values.end());
      float med = values[mid];
      if (values.size() % 2 == 0 && mid > 0) {
        std::nth_element(values.begin(), values.begin() + static_cast<long>(mid - 1),
                         values.end());
        med = 0.5f * (med + values[mid - 1]);
      }
      return med;
    };

    const float cc_median = robust_median(cc_positive);
    float cc_mad = 0.0f;
    if (!cc_positive.empty()) {
      std::vector<float> abs_dev;
      abs_dev.reserve(cc_positive.size());
      for (float v : cc_positive) {
        abs_dev.push_back(std::fabs(v - cc_median));
      }
      cc_mad = robust_median(abs_dev);
    }
    const float cc_threshold_abs = cfg.registration.reject_cc_min_abs;
    const float cc_threshold_robust =
        cc_median - cfg.registration.reject_cc_mad_multiplier * cc_mad;
    const float cc_min_keep = std::max(cc_threshold_abs, cc_threshold_robust);

    const float shift_median = robust_median(shift_mags_positive);
    const float shift_limit =
        std::max(cfg.registration.reject_shift_px_min,
                 cfg.registration.reject_shift_median_multiplier * shift_median);

    for (size_t fi = 0; fi < frames.size(); ++fi) {
      if (global_frame_cc[fi] <= 0.0f)
        continue;
      const auto &w = global_frame_warps[fi];

      bool reject = false;
      std::vector<std::string> reject_reasons;
      // Accept both 0° and ~180° rotations (trace can be positive or negative).
      // But reject mirror/reflection solutions (det < 0), which cause
      // characteristic mirrored ghost artifacts in the final stack.
      const float det = w(0, 0) * w(1, 1) - w(0, 1) * w(1, 0);
      if (det < 0.0f) {
        reject = true;
        ++reg_reject_reflection_outliers;
        reject_reasons.push_back("reflection");
      }

      if (!reject) {
        const float scale = std::sqrt(std::fabs(det));
        if (scale < cfg.registration.reject_scale_min ||
            scale > cfg.registration.reject_scale_max) {
          reject = true;
          ++reg_reject_scale_outliers;
          reject_reasons.push_back("scale");
        }
      }

      if (!reject) {
        const float cc = global_frame_cc[fi];
        if (cc < cc_min_keep) {
          reject = true;
          ++reg_reject_cc_outliers;
          reject_reasons.push_back("low_cc");
        }
      }

      if (!reject) {
        const float shift_mag =
            std::sqrt(w(0, 2) * w(0, 2) + w(1, 2) * w(1, 2));
        if (shift_mag > shift_limit) {
          reject = true;
          ++reg_reject_shift_outliers;
          reject_reasons.push_back("shift_outlier");
        }
      }

      if (reject) {
        core::json rej = {
            {"frame_index", static_cast<int>(fi)},
            {"frame_name", frames[fi].filename().string()},
            {"cc", global_frame_cc[fi]},
            {"reasons", reject_reasons},
            {"a00", w(0, 0)},
            {"a01", w(0, 1)},
            {"tx", w(0, 2)},
            {"a10", w(1, 0)},
            {"a11", w(1, 1)},
            {"ty", w(1, 2)},
        };
        reg_rejected_frames.push_back(rej);
        std::ostringstream msg;
        msg << "REGISTRATION outlier rejected: frame=" << fi << " ("
            << frames[fi].filename().string() << ") cc=" << global_frame_cc[fi]
            << " reasons=" << core::join(reject_reasons, ",")
            << " tx=" << w(0, 2) << " ty=" << w(1, 2);
        emitter.warning(run_id, msg.str(), log_file);
        std::cerr << "[REG-FILTER] " << msg.str() << std::endl;

        global_frame_warps[fi] = registration::identity_warp();
        global_frame_cc[fi] = 0.0f;
      }
    }
  }
  if (reg_reject_orientation_outliers > 0 ||
      reg_reject_reflection_outliers > 0 ||
      reg_reject_scale_outliers > 0 ||
      reg_reject_cc_outliers > 0 ||
      reg_reject_shift_outliers > 0) {
    std::cerr << "[REG-FILTER] rejected outlier warps: orientation="
              << reg_reject_orientation_outliers
              << " reflection=" << reg_reject_reflection_outliers
              << " scale=" << reg_reject_scale_outliers
              << " cc=" << reg_reject_cc_outliers
              << " shift=" << reg_reject_shift_outliers << std::endl;
  }
  global_reg_extra["reg_reject_orientation_outliers"] =
      reg_reject_orientation_outliers;
  global_reg_extra["reg_reject_reflection_outliers"] =
      reg_reject_reflection_outliers;
  global_reg_extra["reg_reject_scale_outliers"] = reg_reject_scale_outliers;
  global_reg_extra["reg_reject_cc_outliers"] = reg_reject_cc_outliers;
  global_reg_extra["reg_reject_shift_outliers"] = reg_reject_shift_outliers;
  global_reg_extra["reg_rejected_frames"] = reg_rejected_frames;

  // Methodik v3: no frame selection. Registration may fall back to identity,
  // but all frames are still kept for subsequent weighting/stacking.
  int n_cc_positive = 0;
  int n_cc_zero = 0;
  int n_cc_negative = 0;
  for (size_t fi = 0; fi < frames.size(); ++fi) {
    if (global_frame_cc[fi] > 0.0f) {
      ++n_cc_positive;
    } else if (global_frame_cc[fi] < 0.0f) {
      ++n_cc_negative;
    } else {
      ++n_cc_zero;
    }
  }
  std::cerr << "[REG] cc>0: " << n_cc_positive << ", cc==0: " << n_cc_zero
            << ", cc<0: " << n_cc_negative << std::endl;
  global_reg_extra["frames_cc_positive"] = n_cc_positive;
  global_reg_extra["frames_cc_zero"] = n_cc_zero;
  global_reg_extra["frames_cc_negative"] = n_cc_negative;

  emitter.phase_end(run_id, Phase::REGISTRATION, global_reg_status,
                    global_reg_extra, log_file);

  emitter.phase_start(run_id, Phase::PREWARP, "PREWARP", log_file);

  // Compute bounding box for field rotation: output canvas must be large enough
  // to contain all rotated frames (Alt/Az mounts near pole).
  registration::BoundingBox bbox = 
      registration::compute_warps_bounding_box(width, height, global_frame_warps);
  
  // Round canvas to even dimensions for CFA (Bayer) compatibility.
  // warp_cfa_mosaic_via_subplanes works on half-resolution subplanes, so
  // canvas must be even in both dimensions to avoid size mismatch in store().
  int canvas_width = (bbox.width() + 1) & ~1;   // round up to even
  int canvas_height = (bbox.height() + 1) & ~1; // round up to even
  
  // Offset to shift all frames into positive coordinate space
  int offset_x = -bbox.min_x;
  int offset_y = -bbox.min_y;
  
  // Apply offset correction to all warps
  if (offset_x != 0 || offset_y != 0) {
    for (auto& w : global_frame_warps) {
      w(0, 2) += static_cast<float>(offset_x);
      w(1, 2) += static_cast<float>(offset_y);
    }
  }
  
  // Log canvas expansion for field rotation
  if (canvas_width > width || canvas_height > height) {
    std::ostringstream msg;
    msg << "Field rotation detected: expanding canvas from " << width << "x" << height
        << " to " << canvas_width << "x" << canvas_height
        << " (bbox: [" << bbox.min_x << "," << bbox.min_y << "] to ["
        << bbox.max_x << "," << bbox.max_y << "], offset: ["
        << offset_x << "," << offset_y << "])";
    emitter.warning(run_id, msg.str(), log_file);
    std::cout << "[PREWARP] " << msg.str() << std::endl;
  }

  // Pre-warp all frames at full resolution before tile extraction.
  // Applying rotation warps to small tile ROIs is fundamentally broken:
  // warpAffine needs source pixels outside the tile boundary that don't
  // exist, causing CFA pattern corruption (colored tile rectangles).
  //
  // Disk-backed: frames are written as raw float binaries and mmap'd on
  // demand, so RAM usage is bounded by OS page cache rather than N*W*H*4.
  DiskCacheFrameStore prewarped_frames(
      run_dir / ".prewarped_cache", frames.size(), canvas_height, canvas_width);
  std::vector<uint8_t> frame_has_data(frames.size(), 0);
  const int prewarp_workers = compute_worker_count(frames.size());
  std::cout << "[PREWARP] Using " << prewarp_workers
            << " parallel workers for " << frames.size() << " frames"
            << std::endl;
  std::mutex prewarp_store_mutex;
  std::mutex prewarp_log_mutex;
  std::mutex prewarp_progress_mutex;
  std::atomic<size_t> prewarp_next{0};
  std::atomic<size_t> prewarp_done{0};
  std::atomic<int> n_frames_with_data{0};
  std::atomic<bool> prewarp_failed{false};
  std::string prewarp_error;

  auto prewarp_worker = [&]() {
    while (true) {
      const size_t fi = prewarp_next.fetch_add(1);
      if (fi >= frames.size()) {
        break;
      }
      try {
        auto pair = load_frame_normalized(fi);
        Matrix2Df img = std::move(pair.first);
        if (img.size() <= 0) {
          continue;
        }
        const auto &w = global_frame_warps[fi];
        const float eps = 1.0e-6f;
        const bool is_identity =
            std::fabs(w(0, 0) - 1.0f) < eps && std::fabs(w(0, 1)) < eps &&
            std::fabs(w(1, 0)) < eps && std::fabs(w(1, 1) - 1.0f) < eps &&
            std::fabs(w(0, 2)) < eps && std::fabs(w(1, 2)) < eps;
        Matrix2Df warped;
        if (is_identity) {
          // For identity warp, we still need to expand canvas if bbox requires it
          if (canvas_width > width || canvas_height > height) {
            // Create expanded canvas and place image at offset position
            warped = Matrix2Df::Zero(canvas_height, canvas_width);
            int src_h = img.rows();
            int src_w = img.cols();
            int dst_y = offset_y;
            int dst_x = offset_x;
            // Ensure we don't write outside canvas bounds
            int copy_h = std::min(src_h, canvas_height - dst_y);
            int copy_w = std::min(src_w, canvas_width - dst_x);
            if (copy_h > 0 && copy_w > 0) {
              warped.block(dst_y, dst_x, copy_h, copy_w) = img.block(0, 0, copy_h, copy_w);
            }
          } else {
            warped = std::move(img);
          }
        } else {
          warped = image::apply_global_warp(img, w, detected_mode, canvas_height, canvas_width);
        }
        if (warped.size() > 0) {
          bool has_data = false;
          {
            std::lock_guard<std::mutex> lock(prewarp_store_mutex);
            prewarped_frames.store(fi, warped);
            has_data = prewarped_frames.has_data(fi);
          }
          if (has_data) {
            frame_has_data[fi] = 1;
            n_frames_with_data.fetch_add(1, std::memory_order_relaxed);
          }
        }
      } catch (const std::exception &e) {
        prewarp_failed.store(true, std::memory_order_relaxed);
        std::lock_guard<std::mutex> lock(prewarp_log_mutex);
        if (prewarp_error.empty()) {
          prewarp_error = e.what();
        }
      } catch (...) {
        prewarp_failed.store(true, std::memory_order_relaxed);
        std::lock_guard<std::mutex> lock(prewarp_log_mutex);
        if (prewarp_error.empty()) {
          prewarp_error = "unknown_error";
        }
      }

      const size_t done = prewarp_done.fetch_add(1) + 1;
      if (done % 5 == 0 || done == frames.size()) {
        std::lock_guard<std::mutex> lock(prewarp_progress_mutex);
        emitter.phase_progress_counts(
            run_id, Phase::PREWARP, static_cast<int>(done),
            static_cast<int>(frames.size()), "prewarp workers=" +
                                              std::to_string(prewarp_workers),
            "frames", log_file);
      }
    }
  };

  if (prewarp_workers > 1) {
    std::vector<std::thread> workers;
    workers.reserve(static_cast<size_t>(prewarp_workers));
    for (int w = 0; w < prewarp_workers; ++w) {
      workers.emplace_back(prewarp_worker);
    }
    for (auto &worker : workers) {
      if (worker.joinable()) {
        worker.join();
      }
    }
  } else {
    prewarp_worker();
  }

  if (prewarp_failed.load(std::memory_order_relaxed)) {
    emitter.phase_end(run_id, Phase::PREWARP, "error",
                      {{"error", prewarp_error.empty() ? "unknown_error"
                                                         : prewarp_error}},
                      log_file);
    std::cerr << "Error during PREWARP: "
              << (prewarp_error.empty() ? "unknown_error" : prewarp_error)
              << std::endl;
    emitter.run_end(run_id, false, "error", log_file);
    return false;
  }

  out.n_usable_frames = n_frames_with_data.load(std::memory_order_relaxed);
  out.canvas_width = canvas_width;
  out.canvas_height = canvas_height;
  out.tile_offset_x = offset_x;
  out.tile_offset_y = offset_y;
  
  emitter.phase_end(run_id, Phase::PREWARP, "ok",
                    {{"num_frames", static_cast<int>(frames.size())},
                     {"num_frames_with_data", out.n_usable_frames},
                     {"canvas_width", canvas_width},
                     {"canvas_height", canvas_height},
                     {"tile_offset_x", offset_x},
                     {"tile_offset_y", offset_y},
                     {"workers", prewarp_workers}},
                    log_file);

  out.frame_has_data = std::move(frame_has_data);
  out.prewarped_frames = std::move(prewarped_frames);
  out.min_valid_frames = 1;
  return true;
}

} // namespace tile_compile::runner
