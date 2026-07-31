#if __has_include(<catch2/catch_test_macros.hpp>)
#include "../apps/runner_phase_post_stack_output.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <sstream>
#include <vector>

TEST_CASE("post_stack_crop_updates_fits_wcs_reference_pixel") {
  const auto run_dir = std::filesystem::temp_directory_path() /
                       "tile_compile_post_stack_crop_wcs";
  std::filesystem::remove_all(run_dir);
  std::filesystem::create_directories(run_dir / "outputs");

  tile_compile::Matrix2Df recon = tile_compile::Matrix2Df::Zero(4, 5);
  recon.block(1, 2, 2, 3).setConstant(1.0f);
  auto recon_R = recon;
  auto recon_G = recon;
  auto recon_B = recon;
  std::vector<uint8_t> common_mask(4 * 5, 1u);
  std::vector<uint8_t> analysis_mask(4 * 5, 1u);

  tile_compile::io::FitsHeader header;
  header.set("CRPIX1", 100.0);
  header.set("CRPIX2", 200.0);

  tile_compile::runner::PostStackOutputConfig config;
  config.crop_to_nonzero_bbox = true;
  config.aqmh_enabled = false;
  tile_compile::runner::OutputScaling scaling;
  tile_compile::core::EventEmitter emitter;
  std::ostringstream log;
  tile_compile::runner::PostStackOutputResult result;

  REQUIRE(tile_compile::runner::write_post_stack_outputs(
      recon, recon_R, recon_G, recon_B, common_mask, analysis_mask, scaling,
      tile_compile::ColorMode::OSC, "GBRG", 0, 0, header, config, run_dir,
      "test", emitter, log, result));
  REQUIRE(result.crop_applied);
  REQUIRE(result.crop_box.x == 2);
  REQUIRE(result.crop_box.y == 1);
  REQUIRE(result.crop_box.width == 3);
  REQUIRE(result.crop_box.height == 2);

  const auto written_header =
      tile_compile::io::read_fits_header(run_dir / "outputs" / "stacked_rgb.fits");
  REQUIRE(written_header.get_double("CRPIX1") == Catch::Approx(98.0));
  REQUIRE(written_header.get_double("CRPIX2") == Catch::Approx(199.0));

  const auto written =
      tile_compile::io::read_fits_rgb(run_dir / "outputs" / "stacked_rgb.fits");
  REQUIRE(written.width == 3);
  REQUIRE(written.height == 2);

  std::filesystem::remove_all(run_dir);
}
#endif
