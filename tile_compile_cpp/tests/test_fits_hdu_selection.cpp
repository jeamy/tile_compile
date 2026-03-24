#if __has_include(<catch2/catch_test_macros.hpp>)
#include "tile_compile/io/fits_io.hpp"

#include <fitsio.h>

#include <filesystem>
#include <string>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

namespace {

void require_fits_ok(int status, const std::string& context) {
  if (status == 0) return;
  char msg[FLEN_STATUS] = {0};
  fits_get_errstatus(status, msg);
  FAIL(context << " failed with CFITSIO status " << status << " (" << msg
               << ")");
}

}  // namespace

TEST_CASE("fits loader skips 0x0 primary HDU and reads later image HDU") {
  namespace fs = std::filesystem;

  const fs::path path = fs::temp_directory_path() /
                        "tile_compile_test_zero_primary_then_image.fits";
  std::error_code ec;
  fs::remove(path, ec);

  fitsfile* fptr = nullptr;
  int status = 0;
  const std::string create_path = "!" + path.string();
  fits_create_file(&fptr, create_path.c_str(), &status);
  require_fits_ok(status, "fits_create_file");

  long primary_axes[2] = {0, 0};
  fits_create_img(fptr, FLOAT_IMG, 2, primary_axes, &status);
  require_fits_ok(status, "fits_create_img(primary)");

  long image_axes[2] = {3, 2};
  fits_create_img(fptr, FLOAT_IMG, 2, image_axes, &status);
  require_fits_ok(status, "fits_create_img(extension)");

  std::vector<float> pixels = {
      1.0f, 2.0f, 3.0f,
      4.0f, 5.0f, 6.0f,
  };
  long fpixel[2] = {1, 1};
  fits_write_pix(fptr, TFLOAT, fpixel, static_cast<long>(pixels.size()),
                 pixels.data(), &status);
  require_fits_ok(status, "fits_write_pix");

  fits_close_file(fptr, &status);
  require_fits_ok(status, "fits_close_file");

  const auto [width, height, naxis] = tile_compile::io::get_fits_dimensions(path);
  REQUIRE(width == 3);
  REQUIRE(height == 2);
  REQUIRE(naxis == 2);

  const auto img = tile_compile::io::read_fits_pixels_float(path);
  REQUIRE(img.rows() == 2);
  REQUIRE(img.cols() == 3);
  REQUIRE(img(0, 0) == Catch::Approx(1.0f));
  REQUIRE(img(0, 2) == Catch::Approx(3.0f));
  REQUIRE(img(1, 0) == Catch::Approx(4.0f));
  REQUIRE(img(1, 2) == Catch::Approx(6.0f));

  fs::remove(path, ec);
}
#else
int tile_compile_tests_fits_hdu_selection_stub() { return 0; }
#endif
