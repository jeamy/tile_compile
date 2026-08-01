#include "runner_phase_post_stack_output.hpp"

#include "tile_compile/core/utils.hpp"
#include "tile_compile/image/cfa_processing.hpp"
#include "tile_compile/image/processing.hpp"
#include "tile_compile/image/normalization.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>

namespace tile_compile::runner {
namespace {

bool write_canvas_mask_fits_impl(const std::filesystem::path &mask_path,
                                 const std::vector<uint8_t> &mask,
                                 int height, int width,
                                 const io::FitsHeader &hdr,
                                 std::string &error_out) {
  if (mask.size() != static_cast<size_t>(height) * width) {
    error_out = "mask size mismatch";
    return false;
  }
  Matrix2Df mask_float(height, width);
  for (size_t i = 0; i < mask.size(); ++i)
    mask_float.data()[i] = mask[i] != 0 ? 1.0f : 0.0f;
  try {
    io::write_fits_float(mask_path, mask_float, hdr);
  } catch (const std::exception &e) {
    error_out = e.what();
    return false;
  }
  return true;
}

} // namespace

bool write_post_stack_outputs(
    Matrix2Df &recon,
    Matrix2Df &recon_R,
    Matrix2Df &recon_G,
    Matrix2Df &recon_B,
    std::vector<uint8_t> &common_valid_mask,
    std::vector<uint8_t> &analysis_valid_mask,
    const OutputScaling &scaling,
    ColorMode detected_mode,
    const std::string &detected_bayer_str,
    int debayer_tile_offset_x,
    int debayer_tile_offset_y,
    const io::FitsHeader &first_hdr,
    const PostStackOutputConfig &cfg,
    const std::filesystem::path &run_dir,
    const std::string &run_id,
    core::EventEmitter &emitter,
    std::ostream &log_file,
    PostStackOutputResult &out,
    const BackgroundModelGrid *background_map_canvas_grid,
    const std::vector<uint8_t> *df_valid_mask_R,
    const std::vector<uint8_t> *df_valid_mask_G,
    const std::vector<uint8_t> *df_valid_mask_B) {
  namespace fs = std::filesystem;

  out.success = false;
  out.crop_box = {0, 0, static_cast<int>(recon.cols()), static_cast<int>(recon.rows())};
  out.crop_applied = false;
  io::FitsHeader output_hdr = first_hdr;

  // --- Crop to non-zero bounding box ---
  if (cfg.crop_to_nonzero_bbox && recon.size() > 0) {
    const int full_rows = recon.rows();
    const int full_cols = recon.cols();
    const bool have_rgb_full =
        (recon_R.rows() == full_rows && recon_R.cols() == full_cols &&
         recon_G.rows() == full_rows && recon_G.cols() == full_cols &&
         recon_B.rows() == full_rows && recon_B.cols() == full_cols);
    const size_t full_mask_px =
        static_cast<size_t>(full_rows) * static_cast<size_t>(full_cols);

    if (common_valid_mask.size() != full_mask_px) {
      out.error = "canvas mask size mismatch during crop";
      return false;
    }

    CropBox crop = cfg.aqmh_enabled
        ? compute_support_mask_bbox(common_valid_mask, full_rows, full_cols)
        : compute_nonzero_data_bbox(
              recon, have_rgb_full ? &recon_R : nullptr,
              have_rgb_full ? &recon_G : nullptr,
              have_rgb_full ? &recon_B : nullptr);

    if (!crop.valid()) {
      out.error = "crop produced empty valid canvas";
      return false;
    }

    const bool needs_crop =
        (crop.x != 0 || crop.y != 0 || crop.width != full_cols || crop.height != full_rows);

    if (needs_crop) {
      recon = recon.block(crop.y, crop.x, crop.height, crop.width).eval();
      if (have_rgb_full) {
        recon_R = recon_R.block(crop.y, crop.x, crop.height, crop.width).eval();
        recon_G = recon_G.block(crop.y, crop.x, crop.height, crop.width).eval();
        recon_B = recon_B.block(crop.y, crop.x, crop.height, crop.width).eval();
      }
      debayer_tile_offset_x -= crop.x;
      debayer_tile_offset_y -= crop.y;

      // CRPIX uses one-based FITS pixel coordinates while crop offsets are
      // zero-based image indices. Subtracting the crop origin preserves the
      // world-coordinate reference for the cropped image.
      if (const auto crpix1 = output_hdr.get_double("CRPIX1"))
        output_hdr.set("CRPIX1", *crpix1 - static_cast<double>(crop.x));
      else if (const auto crpix1_int = output_hdr.get_int("CRPIX1"))
        output_hdr.set("CRPIX1", *crpix1_int - crop.x);
      if (const auto crpix2 = output_hdr.get_double("CRPIX2"))
        output_hdr.set("CRPIX2", *crpix2 - static_cast<double>(crop.y));
      else if (const auto crpix2_int = output_hdr.get_int("CRPIX2"))
        output_hdr.set("CRPIX2", *crpix2_int - crop.y);

      // Crop masks
      std::vector<uint8_t> cropped_mask(
          static_cast<size_t>(crop.height * crop.width), 0u);
      std::vector<uint8_t> cropped_analysis_mask(
          static_cast<size_t>(crop.height * crop.width), 0u);
      for (int y = 0; y < crop.height; ++y) {
        const int sy = crop.y + y;
        const size_t src_row = static_cast<size_t>(sy) * static_cast<size_t>(full_cols);
        const size_t dst_row = static_cast<size_t>(y) * static_cast<size_t>(crop.width);
        for (int x = 0; x < crop.width; ++x) {
          const int sx = crop.x + x;
          cropped_mask[dst_row + x] = common_valid_mask[src_row + sx];
          if (analysis_valid_mask.size() == full_mask_px)
            cropped_analysis_mask[dst_row + x] = analysis_valid_mask[src_row + sx];
        }
      }
      common_valid_mask.swap(cropped_mask);
      analysis_valid_mask.swap(cropped_analysis_mask);

      // Write updated canvas mask
      std::string mask_error;
      write_canvas_mask_fits_impl(
          run_dir / "outputs" / "canvas_mask.fits",
          common_valid_mask, crop.height, crop.width, output_hdr, mask_error);
      write_canvas_mask_fits_impl(
          run_dir / "outputs" / "common_overlap_mask.fits",
          analysis_valid_mask, crop.height, crop.width, output_hdr, mask_error);

      out.crop_applied = true;
    }
    out.crop_box = crop;
  }

  // --- Apply output scaling ---
  // Stufe D: linear outputs (reconstructed_*) are always restored with
  // background, photometric scale and a small pedestal. The stretched
  // presentation files (stacked.fits, stacked_rgb.fits) are derived from those
  // linear images and never replace the linear originals.
  bool have_rgb = recon.size() > 0 &&
                  recon_R.size() == recon.size() &&
                  recon_G.size() == recon.size() &&
                  recon_B.size() == recon.size();
  if (have_rgb) out.debayer_method = "precomputed_rgb";
  if (detected_mode == ColorMode::OSC && !have_rgb) {
    // Same Bayer-parity rule as the primary DEBAYER phase: the mosaic lives on
    // the registration canvas lattice, so the canvas tile offset defines the
    // CFA origin. OpenCV edge-aware demosaicing preserves star-core detail.
    auto debayer = image::debayer_opencv(
        recon, string_to_bayer_pattern(detected_bayer_str),
        -debayer_tile_offset_x, -debayer_tile_offset_y, /*ahd=*/true);
    recon_R = std::move(debayer.R);
    recon_G = std::move(debayer.G);
    recon_B = std::move(debayer.B);
    out.debayer_method = "edge_aware";
    have_rgb = recon.size() > 0 &&
               recon_R.size() == recon.size() &&
               recon_G.size() == recon.size() &&
               recon_B.size() == recon.size();
  }
  io::FitsHeader rgb_output_hdr = output_hdr;
  if (detected_mode == ColorMode::OSC && have_rgb) {
    rgb_output_hdr.set("DEBAYER", "PRE_STACK");

    // Stufe D: linear RGB always gets background + photometric scale + pedestal.
    recon_R.array() *= scaling.scale_r;
    recon_G.array() *= scaling.scale_g;
    recon_B.array() *= scaling.scale_b;
    recon_R.array() += scaling.bg_r;
    recon_G.array() += scaling.bg_g;
    recon_B.array() += scaling.bg_b;
    recon_R.array() += scaling.pedestal;
    recon_G.array() += scaling.pedestal;
    recon_B.array() += scaling.pedestal;

    // Luma for stacked.fits / reconstructed_L.fit
    const float scale_luma = 0.25f * scaling.scale_r + 0.5f * scaling.scale_g +
                             0.25f * scaling.scale_b;
    Matrix2Df recon_luma = recon;
    recon_luma *= scale_luma;
    const float bg_luma = 0.25f * scaling.bg_r + 0.5f * scaling.bg_g +
                          0.25f * scaling.bg_b;
    recon_luma.array() += (bg_luma + scaling.pedestal);

    // Enforce canvas mask on luma
    for (Eigen::Index k = 0; k < recon_luma.size(); ++k) {
      if (static_cast<size_t>(k) >= common_valid_mask.size() ||
          common_valid_mask[static_cast<size_t>(k)] == 0)
        recon_luma.data()[k] = 0.0f;
    }

    // Linear luma file is always written unchanged.
    try {
      io::write_fits_float(run_dir / "outputs" / "reconstructed_L.fit",
                           recon_luma, output_hdr);
    } catch (const std::exception &e) {
      out.error = std::string("reconstructed_L.fit write failed: ") + e.what();
      return false;
    }

    // Stretched luma presentation file is derived from the linear one.
    Matrix2Df luma_presentation = recon_luma;
    if (cfg.output_stretch) {
      const auto stretch =
          core::stretch_to_u16_linear_from_zero_inplace(luma_presentation);
      if (stretch.applied) {
        std::cout << "[STACKING] Output luma linear stretch ["
                  << stretch.low << ".." << stretch.high
                  << "] -> [0..65535] samples=" << stretch.sample_count
                  << std::endl;
      }
    }

    try {
      io::write_fits_float(run_dir / "outputs" / "stacked.fits",
                           luma_presentation, output_hdr);
    } catch (const std::exception &e) {
      out.error = std::string("stacked.fits write failed: ") + e.what();
      return false;
    }

    // Enforce canvas mask on RGB
    image::enforce_canvas_mask_on_rgb(recon_R, recon_G, recon_B, common_valid_mask);

    // Apply per-channel valid masks. Pixels with no contributing frame are
    // marked as NaN instead of a physical 0.
    if (df_valid_mask_R && df_valid_mask_R->size() ==
                              static_cast<size_t>(recon_R.size()) &&
        recon_R.size() > 0) {
      for (size_t i = 0; i < static_cast<size_t>(recon_R.size()); ++i) {
        if ((*df_valid_mask_R)[i] == 0u)
          recon_R.data()[i] = std::numeric_limits<float>::quiet_NaN();
        if ((*df_valid_mask_G)[i] == 0u)
          recon_G.data()[i] = std::numeric_limits<float>::quiet_NaN();
        if ((*df_valid_mask_B)[i] == 0u)
          recon_B.data()[i] = std::numeric_limits<float>::quiet_NaN();
      }
    }

    // Linear per-channel files are always restored.
    try {
      io::write_fits_float(run_dir / "outputs" / "reconstructed_R.fit", recon_R,
                           rgb_output_hdr);
      io::write_fits_float(run_dir / "outputs" / "reconstructed_G.fit", recon_G,
                           rgb_output_hdr);
      io::write_fits_float(run_dir / "outputs" / "reconstructed_B.fit", recon_B,
                           rgb_output_hdr);
    } catch (const std::exception &e) {
      out.error = std::string("reconstructed channel write failed: ") + e.what();
      return false;
    }

    // Linear solve reference matches reconstructed_R/G/B.
    try {
      io::write_fits_rgb(run_dir / "outputs" / "stacked_rgb_solve.fits",
                         recon_R, recon_G, recon_B, rgb_output_hdr);
    } catch (const std::exception &e) {
      out.error = std::string("stacked_rgb_solve.fits write failed: ") + e.what();
      return false;
    }

    // Stretched RGB presentation file (or float RGB if no stretch).
    Matrix2Df R_disk = recon_R;
    Matrix2Df G_disk = recon_G;
    Matrix2Df B_disk = recon_B;
    if (cfg.output_stretch) {
      const std::vector<uint8_t>& statistics_mask =
          analysis_valid_mask.size() == static_cast<size_t>(R_disk.size())
              ? analysis_valid_mask
              : common_valid_mask;
      const auto stretch = core::stretch_rgb_to_u32_linear_from_zero_inplace(
          R_disk, G_disk, B_disk, statistics_mask);
      if (stretch.applied) {
        std::cout << "[STACKING] RGB output per-channel stretch "
                  << "R[" << stretch.low_r << ".." << stretch.high_r << "] "
                  << "G[" << stretch.low_g << ".." << stretch.high_g << "] "
                  << "B[" << stretch.low_b << ".." << stretch.high_b
                  << "] -> [0..4294967295] (robust p99.9)"
                  << " samples=" << stretch.sample_count << std::endl;
      }
      try {
        io::write_fits_rgb_u32(run_dir / "outputs" / "stacked_rgb.fits",
                               R_disk, G_disk, B_disk, rgb_output_hdr);
      } catch (const std::exception &e) {
        out.error = std::string("stacked_rgb.fits write failed: ") + e.what();
        return false;
      }
    } else {
      try {
        io::write_fits_rgb(run_dir / "outputs" / "stacked_rgb.fits",
                           recon_R, recon_G, recon_B, rgb_output_hdr);
      } catch (const std::exception &e) {
        out.error = std::string("stacked_rgb.fits write failed: ") + e.what();
        return false;
      }
    }

  } else {
    // MONO path
    image::apply_output_scaling_inplace(
        recon, -debayer_tile_offset_x, -debayer_tile_offset_y,
        detected_mode, detected_bayer_str,
        scaling.scale_mono, scaling.scale_r, scaling.scale_g, scaling.scale_b,
        scaling.bg_mono, scaling.bg_r, scaling.bg_g, scaling.bg_b,
        scaling.pedestal);

    // Enforce canvas mask
    for (Eigen::Index k = 0; k < recon.size(); ++k) {
      if (static_cast<size_t>(k) >= common_valid_mask.size() ||
          common_valid_mask[static_cast<size_t>(k)] == 0)
        recon.data()[k] = 0.0f;
    }

    // Linear mono file is always restored.
    try {
      io::write_fits_float(run_dir / "outputs" / "reconstructed_L.fit", recon,
                           output_hdr);
    } catch (const std::exception &e) {
      out.error = std::string("reconstructed_L.fit write failed: ") + e.what();
      return false;
    }

    // Stretched mono presentation file is derived from the linear one.
    Matrix2Df luma_presentation = recon;
    if (cfg.output_stretch) {
      const auto stretch =
          core::stretch_to_u16_linear_from_zero_inplace(luma_presentation);
      if (stretch.applied) {
        std::cout << "[STACKING] Output luma linear stretch ["
                  << stretch.low << ".." << stretch.high
                  << "] -> [0..65535] samples=" << stretch.sample_count
                  << std::endl;
      }
    }

    try {
      io::write_fits_float(run_dir / "outputs" / "stacked.fits",
                           luma_presentation, output_hdr);
    } catch (const std::exception &e) {
      out.error = std::string("stacked.fits write failed: ") + e.what();
      return false;
    }
  }

  out.success = true;
  return true;
}

void write_stretched_rgb_snapshot(
    const std::filesystem::path &path,
    const Matrix2Df &R_src,
    const Matrix2Df &G_src,
    const Matrix2Df &B_src,
    const std::vector<uint8_t> &canvas_mask,
    const std::vector<uint8_t> &statistics_mask,
    int canvas_rows,
    int canvas_cols,
    const io::FitsHeader &hdr,
    bool apply_stretch,
    const char *stage_tag) {
  Matrix2Df R_disk = R_src;
  Matrix2Df G_disk = G_src;
  Matrix2Df B_disk = B_src;

  if (!canvas_mask.empty() &&
      canvas_mask.size() == static_cast<size_t>(canvas_rows) * canvas_cols) {
    image::enforce_canvas_mask_on_rgb(R_disk, G_disk, B_disk, canvas_mask);
  }

  if (apply_stretch) {
    const auto stretch = core::stretch_rgb_to_u32_linear_from_zero_inplace(
        R_disk, G_disk, B_disk, statistics_mask);
    if (stretch.applied) {
      std::cout << "[" << stage_tag
                << "] RGB output per-channel stretch "
                << "R[" << stretch.low_r << ".." << stretch.high_r << "] "
                << "G[" << stretch.low_g << ".." << stretch.high_g << "] "
                << "B[" << stretch.low_b << ".." << stretch.high_b
                << "] -> [0..4294967295] (robust p99.9)"
                << " samples=" << stretch.sample_count << std::endl;
    }
  }

  std::error_code ec;
  std::filesystem::remove(path, ec);
  if (apply_stretch) {
    io::write_fits_rgb_u32(path, R_disk, G_disk, B_disk, hdr);
  } else {
    io::write_fits_rgb(path, R_disk, G_disk, B_disk, hdr);
  }
}

} // namespace tile_compile::runner
