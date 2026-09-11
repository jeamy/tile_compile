#include "runner_downstream.hpp"
#include "tile_compile/core/atomic_output.hpp"
#include "tile_compile/core/utils.hpp"
#include "tile_compile/io/fits_io.hpp"
#include "tile_compile/reconstruction/forward_drizzle.hpp"
#include <cmath>
#include <fstream>

namespace tile_compile::runner {
void write_forward_downstream_inputs(const fs::path &dir,
    const registration::RegistrationSamplingPlan &sampling,
    const config::ReconstructionDrizzleConfig &drizzle) {
  const bool mono = sampling.color_mode == ColorMode::MONO;
  const auto normalization_path = dir / "artifacts/normalization.json";
  const auto normalization = core::json::parse(core::read_text(normalization_path));
  auto median = [&](const std::string &key, bool positive) {
    const auto &values = normalization.at(key);
    if (!values.is_array() || values.empty())
      throw std::runtime_error("FORWARD_OUTPUT_NORMALIZATION_MISSING: " + key);
    std::vector<float> finite;
    for (const auto &j : values) {
      if (!j.is_number()) continue;
      const float v = j.get<float>();
      if (std::isfinite(v) && (!positive || v > 0)) finite.push_back(v);
    }
    if (finite.empty()) throw std::runtime_error("FORWARD_OUTPUT_NORMALIZATION_INVALID: " + key);
    return core::median_of(finite);
  };
  const int w = sampling.canvas_width_native * drizzle.output_scale;
  const int h = sampling.canvas_height_native * drizzle.output_scale;
  const size_t pixels = static_cast<size_t>(w) * h;
  auto output_budget = drizzle;
  output_budget.internal_scale = drizzle.output_scale;
  output_budget.chunk_rows = h;
  // Full output RGB/L, mask conversion, and downstream input serialization.
  // Fail before reading image planes if this working set does not fit.
  reconstruction::plan_drizzle_memory(sampling, output_budget, 128);
  const auto outputs = dir / "outputs";
  const std::vector<std::string> channels = mono
      ? std::vector<std::string>{"L"} : std::vector<std::string>{"R", "G", "B"};
  std::vector<Matrix2Df> planes;
  std::vector<uint8_t> support(pixels, 1), analysis(pixels, 0);
  core::json inputs = core::json::object();
  for (const auto &c : channels) {
    const auto path = outputs / ("reconstructed_" + c + ".fit");
    auto plane = io::read_fits_pixels_float(path);
    if (plane.rows() != h || plane.cols() != w)
      throw std::runtime_error("FORWARD_OUTPUT_SHAPE");
    std::string key = mono ? "mono" : core::to_lower(c);
    const float scale = median("P_" + key, true);
    const float background = median("B_" + key, false);
    for (int y = 0; y < h; ++y) for (int x = 0; x < w; ++x) {
      const size_t i = static_cast<size_t>(y) * w + x;
      if (!std::isfinite(plane(y,x))) support[i] = 0;
      else {
        plane(y,x) = plane(y,x) * scale + background;
        if (!std::isfinite(plane(y,x)))
          throw std::runtime_error("FORWARD_OUTPUT_PHOTOMETRY_OVERFLOW");
      }
    }
    inputs[c] = {{"bytes",fs::file_size(path)}, {"scale",scale}, {"background",background}};
    planes.push_back(std::move(plane));
  }
  const auto mask_path = dir / "artifacts/sampling_geometry_analysis_common_mask.fits";
  auto mask = io::read_fits_pixels_float(mask_path);
  const int ratio = drizzle.internal_scale / drizzle.output_scale;
  if (ratio < 1 || mask.rows() != h * ratio || mask.cols() != w * ratio)
    throw std::runtime_error("FORWARD_OUTPUT_MASK_SHAPE");
  for (int y = 0; y < h; ++y) for (int x = 0; x < w; ++x) {
    const size_t i = static_cast<size_t>(y) * w + x;
    bool ok = support[i];
    for (int dy = 0; dy < ratio; ++dy) for (int dx = 0; dx < ratio; ++dx)
      ok = ok && std::isfinite(mask(y*ratio+dy,x*ratio+dx)) && mask(y*ratio+dy,x*ratio+dx) > 0;
    analysis[i] = ok;
    if (!support[i]) for (auto &plane : planes) plane(y,x) = 0;
  }
  io::FitsHeader header;
  // Copy only pointing/instrument metadata, never a source-frame WCS onto a
  // transformed canvas. Astrometry solves the actual output geometry below.
  const auto provenance = core::json::parse(core::read_text(dir / "artifacts/run_provenance.json"));
  if (provenance.at("input_manifest").contains("entries") &&
      !provenance.at("input_manifest").at("entries").empty()) {
    const auto source = io::read_fits_header(provenance.at("input_manifest").at("entries").at(0).at("path").get<std::string>());
    for (const auto *key : {"RA", "DEC", "OBJCTRA", "OBJCTDEC", "OBJECT", "INSTRUME", "TELESCOP"}) {
      if (auto v = source.get_string(key)) header.set(key,*v);
      else if (auto v = source.get_double(key)) header.set(key,*v);
    }
    for (const auto *key : {"FOCALLEN", "XPIXSZ", "YPIXSZ"})
      if (auto v = source.get_double(key)) header.set(key,*v);
  }
  header.set("FDSPACE",std::string("RESTORED_LINEAR"));
  header.set("FDSCALE",drizzle.output_scale);
  header.set("FDNORM",std::to_string(fs::file_size(normalization_path)));  // T1: size, no SHA
  core::AtomicOutput generation(dir / "artifacts/forward_downstream_inputs");
  fs::create_directories(generation.path());
  const auto stage = generation.path();
  Matrix2Df luma = mono ? planes[0] :
      (0.25f * planes[0] + 0.5f * planes[1] + 0.25f * planes[2]).eval();
  io::write_fits_float(stage / "stacked.fits",luma,header);
  if (!mono) {
    io::write_fits_rgb(stage / "stacked_rgb.fits",planes[0],planes[1],planes[2],header);
    io::write_fits_rgb(stage / "stacked_rgb_solve.fits",planes[0],planes[1],planes[2],header);
  }
  io::write_fits_mask_rows(stage / "canvas_mask.fits",support,h,w,header);
  io::write_fits_mask_rows(stage / "common_overlap_mask.fits",analysis,h,w,header);
  core::json files=core::json::object();
  for (const auto &entry : fs::directory_iterator(stage)) {
    files[entry.path().filename().string()]=fs::file_size(entry.path());
    core::AtomicOutput target(outputs / entry.path().filename());
    fs::copy_file(entry.path(),target.path());
    target.commit();
  }
  core::write_text_atomic(dir / "artifacts/forward_downstream_inputs.json",
      core::json({{"version",2},{"normalization_bytes",fs::file_size(normalization_path)},
        {"source_profiles",inputs},{"files",files},{"width",w},{"height",h},
        {"crop_x",0},{"crop_y",0},{"photometry_applied_once",true}}).dump(2));
}
}
