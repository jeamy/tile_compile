#if __has_include(<catch2/catch_test_macros.hpp>)
#include "../apps/runner_shared.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <filesystem>
#include <string>

namespace {

std::filesystem::path unique_bg_grid_dir(const std::string &name) {
  static int counter = 0;
  return std::filesystem::temp_directory_path() /
         ("tile_compile_bg_grid_" + name + "_" + std::to_string(++counter));
}

} // namespace

TEST_CASE("background_model_grid_store_roundtrip") {
  const auto dir = unique_bg_grid_dir("roundtrip");
  const int rows = 4, cols = 6, channels = 3;
  tile_compile::runner::BackgroundModelGridStore store(dir, 2, rows, cols,
                                                       {"R", "G", "B"});

  tile_compile::runner::BackgroundModelGrid grid(rows, cols, channels);
  for (int ch = 0; ch < channels; ++ch) {
    for (int r = 0; r < rows; ++r) {
      for (int c = 0; c < cols; ++c) {
        grid.value(r, c, ch) = static_cast<float>(r * cols + c + ch * 100);
        grid.support(r, c, ch) =
            tile_compile::runner::BackgroundModelGrid::kMeasured;
      }
    }
  }
  store.store(0, grid);

  REQUIRE(store.has_data(0));
  REQUIRE_FALSE(store.has_data(1));

  auto back = store.load(0);
  REQUIRE(back.rows() == rows);
  REQUIRE(back.cols() == cols);
  REQUIRE(back.channels() == channels);
  for (int ch = 0; ch < channels; ++ch) {
    for (int r = 0; r < rows; ++r) {
      for (int c = 0; c < cols; ++c) {
        REQUIRE(back.value(r, c, ch) == Catch::Approx(grid.value(r, c, ch)));
        REQUIRE(back.support(r, c, ch) == grid.support(r, c, ch));
      }
    }
  }
}

TEST_CASE("background_model_grid_interpolate_and_upsample") {
  const int rows = 4, cols = 4, channels = 1;
  tile_compile::runner::BackgroundModelGrid grid(rows, cols, channels);
  // Set a gradient on the border, leave the center empty.
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      if (r == 0 || r == rows - 1 || c == 0 || c == cols - 1) {
        grid.value(r, c, 0) = static_cast<float>(r + c);
        grid.support(r, c, 0) =
            tile_compile::runner::BackgroundModelGrid::kMeasured;
      }
    }
  }
  grid.interpolate_empty_cells();

  // The center should now be valid and roughly the average of neighbors.
  REQUIRE(grid.valid(1, 1, 0));
  REQUIRE(grid.value(1, 1, 0) > 0.0f);

  auto upsampled = grid.upsample_channel(0, 8, 8);
  REQUIRE(upsampled.rows() == 8);
  REQUIRE(upsampled.cols() == 8);
  // Check a corner and an edge of the upsampled output.
  REQUIRE(upsampled(0, 0) == Catch::Approx(0.0f));
  REQUIRE(upsampled(7, 0) == Catch::Approx(static_cast<float>(rows - 1)));
}

TEST_CASE("background_model_grid_store_attach_existing") {
  const auto dir = unique_bg_grid_dir("attach");
  const int rows = 3, cols = 3, channels = 2;
  {
    tile_compile::runner::BackgroundModelGridStore store(
        dir, 1, rows, cols, {"R", "G1"});
    tile_compile::runner::BackgroundModelGrid grid(rows, cols, channels);
    grid.value(0, 0, 0) = 1.0f;
    grid.value(0, 0, 1) = 2.0f;
    grid.support(0, 0, 0) =
        tile_compile::runner::BackgroundModelGrid::kMeasured;
    grid.support(0, 0, 1) =
        tile_compile::runner::BackgroundModelGrid::kMeasured;
    store.store(0, grid);
    store.set_preserve_files(true);
  }
  tile_compile::runner::BackgroundModelGridStore store2(
      dir, 1, rows, cols, {"R", "G1"}, true);
  REQUIRE(store2.has_data(0));
  auto back = store2.load(0);
  REQUIRE(back.value(0, 0, 0) == Catch::Approx(1.0f));
  REQUIRE(back.value(0, 0, 1) == Catch::Approx(2.0f));
}

TEST_CASE("background_model_grid_from_image_mono_gradient") {
  const int img_rows = 16, img_cols = 16;
  tile_compile::Matrix2Df img(img_rows, img_cols);
  cv::Mat1b mask(img_rows, img_cols, 255);
  for (int y = 0; y < img_rows; ++y) {
    for (int x = 0; x < img_cols; ++x) {
      img(y, x) = 0.1f * static_cast<float>(x) + 0.2f * static_cast<float>(y);
    }
  }
  // Mask out a bright star in the center cell; the background estimate there
  // must still follow the local gradient, not the star.
  img(8, 8) = 100.0f;
  mask(8, 8) = 0;

  auto grid = tile_compile::runner::BackgroundModelGrid::from_image(
      img, mask, tile_compile::ColorMode::MONO, "RGGB", 4, 4);
  REQUIRE(grid.channels() == 1);
  REQUIRE(grid.rows() == 4);
  REQUIRE(grid.cols() == 4);
  REQUIRE(grid.value(2, 2, 0) == Catch::Approx(2.9f).margin(0.2f));
  REQUIRE(grid.support(2, 2, 0) ==
          tile_compile::runner::BackgroundModelGrid::kMeasured);
}

TEST_CASE("background_model_grid_from_image_osc_planes") {
  const int img_rows = 16, img_cols = 16;
  tile_compile::Matrix2Df img(img_rows, img_cols);
  cv::Mat1b mask(img_rows, img_cols, 255);
  for (int y = 0; y < img_rows; ++y) {
    for (int x = 0; x < img_cols; ++x) {
      const int py = y & 1;
      const int px = x & 1;
      if (py == 0 && px == 0) {
        img(y, x) = 1.0f; // R
      } else if (py == 1 && px == 1) {
        img(y, x) = 3.0f; // B
      } else {
        img(y, x) = 2.0f; // G
      }
    }
  }

  auto grid = tile_compile::runner::BackgroundModelGrid::from_image(
      img, mask, tile_compile::ColorMode::OSC, "RGGB", 4, 4);
  REQUIRE(grid.channels() == 4);
  REQUIRE(grid.rows() == 4);
  REQUIRE(grid.cols() == 4);
  REQUIRE(grid.value(0, 0, 0) == Catch::Approx(1.0f));
  REQUIRE(grid.value(0, 0, 1) == Catch::Approx(2.0f));
  REQUIRE(grid.value(0, 0, 2) == Catch::Approx(2.0f));
  REQUIRE(grid.value(0, 0, 3) == Catch::Approx(3.0f));
}

TEST_CASE("background_map_canvas_identity_and_two_frames") {
  // 4x4 frame grid, 2x2 canvas grid, 1 channel.
  tile_compile::runner::BackgroundModelGrid frame_grid(4, 4, 1);
  for (int r = 0; r < 4; ++r) {
    for (int c = 0; c < 4; ++c) {
      frame_grid.value(r, c, 0) = static_cast<float>(r + c);
      frame_grid.support(r, c, 0) =
          tile_compile::runner::BackgroundModelGrid::kMeasured;
    }
  }

  tile_compile::runner::BackgroundMapCanvas canvas(2, 2, 1);
  tile_compile::WarpMatrix warp;
  warp << 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f;
  canvas.accumulate(frame_grid, 16, 16, 16, 16, warp);
  canvas.accumulate(frame_grid, 16, 16, 16, 16, warp);

  auto out = canvas.finalize();
  REQUIRE(out.rows() == 2);
  REQUIRE(out.cols() == 2);
  // Top-left 8x8 covers frame-grid cells (0..1,0..1) with values 0,1,1,2.
  // Bottom-right 8x8 covers (2..3,2..3) with values 4,5,5,6.
  REQUIRE(out.value(0, 0, 0) == Catch::Approx(1.0f).margin(0.3f));
  REQUIRE(out.value(1, 1, 0) == Catch::Approx(5.0f).margin(0.3f));
  REQUIRE(out.support(0, 0, 0) ==
          tile_compile::runner::BackgroundModelGrid::kMeasured);
}

TEST_CASE("background_map_canvas_translation") {
  // 2x2 frame grid, 4x4 canvas grid, 1 channel.
  tile_compile::runner::BackgroundModelGrid frame_grid(2, 2, 1);
  frame_grid.value(0, 0, 0) = 1.0f;
  frame_grid.value(0, 1, 0) = 2.0f;
  frame_grid.value(1, 0, 0) = 3.0f;
  frame_grid.value(1, 1, 0) = 4.0f;
  for (int r = 0; r < 2; ++r) {
    for (int c = 0; c < 2; ++c) {
      frame_grid.support(r, c, 0) =
          tile_compile::runner::BackgroundModelGrid::kMeasured;
    }
  }

  // Translate the frame by (+1, +2) pixels on the canvas. With
  // WARP_INVERSE_MAP the matrix maps canvas to source, so we sample
  // source at (x-1, y-2).
  tile_compile::WarpMatrix warp;
  warp << 1.0f, 0.0f, -1.0f, 0.0f, 1.0f, -2.0f;

  tile_compile::runner::BackgroundMapCanvas canvas(4, 4, 1);
  canvas.accumulate(frame_grid, 4, 4, 6, 6, warp);

  auto out = canvas.finalize();
  REQUIRE(out.rows() == 4);
  REQUIRE(out.cols() == 4);
  // Background pixels should be present on the canvas.
  int valid_cells = 0;
  for (int r = 0; r < 4; ++r) {
    for (int c = 0; c < 4; ++c) {
      if (out.valid(r, c, 0))
        ++valid_cells;
    }
  }
  REQUIRE(valid_cells >= 2);
}

TEST_CASE("background_map_canvas_realistic_shift_coverage") {
  // Regression for the M31 run 20260802: a small shift/rotation on a
  // real-world frame/canvas geometry must produce a broad canvas grid, not a
  // thin edge strip.  This fails if BackgroundMapCanvas uses floor cell division
  // (last column absorbs remaining pixels and raises the coverage floor) or if
  // the warp orientation is wrong.
  const int frame_rows = 2160;
  const int frame_cols = 3840;
  const int canvas_rows = 2312;
  const int canvas_cols = 3924;
  const int grid_rows = 78;
  const int grid_cols = 131;

  // 72x128 per-frame background grid, fully measured and flat.
  tile_compile::runner::BackgroundModelGrid frame_grid(72, 128, 1);
  for (int r = 0; r < 72; ++r) {
    for (int c = 0; c < 128; ++c) {
      frame_grid.value(r, c, 0) = 200.0f;
      frame_grid.support(r, c, 0) =
          tile_compile::runner::BackgroundModelGrid::kMeasured;
    }
  }

  // Frame 0 warp from global_registration.json (dest->src for WARP_INVERSE_MAP).
  tile_compile::WarpMatrix warp;
  warp << 0.9999178051948547f, 0.013368082232773304f, -84.76670837402344f,
      -0.013215672224760056f, 0.9997730255126953f, -98.33377838134766f;

  tile_compile::runner::BackgroundMapCanvas canvas(grid_rows, grid_cols, 1, 1);
  canvas.accumulate(frame_grid, frame_rows, frame_cols, canvas_rows,
                    canvas_cols, warp);

  auto out = canvas.finalize();
  int valid_cells = 0;
  for (int r = 0; r < grid_rows; ++r) {
    for (int c = 0; c < grid_cols; ++c) {
      if (out.valid(r, c, 0))
        ++valid_cells;
    }
  }
  const float valid_fraction =
      static_cast<float>(valid_cells) / static_cast<float>(grid_rows * grid_cols);
  // A single frame covers ~90 % of the canvas; the canvas grid should reflect
  // that.  We require well over half to reject the broken edge-strip case.
  REQUIRE(valid_fraction > 0.5f);

  // All valid cells should carry the original flat 200 ADU background.
  float sum = 0.0f;
  int n = 0;
  for (int r = 0; r < grid_rows; ++r) {
    for (int c = 0; c < grid_cols; ++c) {
      if (out.valid(r, c, 0)) {
        sum += out.value(r, c, 0);
        ++n;
      }
    }
  }
  REQUIRE(n > 0);
  REQUIRE(sum / n == Catch::Approx(200.0f).margin(0.1f));
}

TEST_CASE("background_map_upsample_then_crop_preserves_background") {
  // Regression for the M31 DF-AQMH runs: the accumulated background grid spans
  // the full canvas, but the reconstruction output is cropped to its valid
  // bounding box.  The output pipeline upsamples the grid to the full canvas and
  // then extracts the crop block.  This must (a) match the cropped output
  // dimensions exactly (so the size-equality guard adds instead of silently
  // dropping the background) and (b) preserve the background values.
  const int grid_rows = 78, grid_cols = 131;
  const int canvas_rows = 2312, canvas_cols = 3924;
  // Cropped output box (a few px trimmed on each side, as crop_to_nonzero_bbox
  // does in practice: full canvas 2312x3924 -> 2309x3922).
  const int crop_y = 1, crop_x = 1, crop_h = 2309, crop_w = 3922;

  tile_compile::runner::BackgroundModelGrid grid(grid_rows, grid_cols, 1);
  for (int r = 0; r < grid_rows; ++r) {
    for (int c = 0; c < grid_cols; ++c) {
      grid.value(r, c, 0) = 205.0f;
      grid.support(r, c, 0) =
          tile_compile::runner::BackgroundModelGrid::kMeasured;
    }
  }

  const tile_compile::Matrix2Df full =
      grid.upsample_channel(0, canvas_rows, canvas_cols);
  REQUIRE(full.rows() == canvas_rows);
  REQUIRE(full.cols() == canvas_cols);

  const tile_compile::Matrix2Df cropped =
      full.block(crop_y, crop_x, crop_h, crop_w).eval();
  // The cropped background map must match the cropped output dimensions exactly.
  REQUIRE(cropped.rows() == crop_h);
  REQUIRE(cropped.cols() == crop_w);
  // A flat background must survive upsample + crop unchanged.
  REQUIRE(cropped(0, 0) == Catch::Approx(205.0f));
  REQUIRE(cropped(crop_h - 1, crop_w - 1) == Catch::Approx(205.0f));
  double sum = 0.0;
  for (int y = 0; y < crop_h; ++y)
    for (int x = 0; x < crop_w; ++x)
      sum += cropped(y, x);
  REQUIRE(sum / (static_cast<double>(crop_h) * crop_w) ==
          Catch::Approx(205.0f).margin(0.01));
}

TEST_CASE("background_model_gradient_offset_acceptance") {
  // Synthetischer Gradient + additive Frame-Offsets: Akzeptanzkriterium aus
  // Stufe A: Gradient nach Normalisierung und Wiederaddition innerhalb von
  // 2% relativer Fehler auf gueltigen Gridzellen erhalten.
  //
  // Wir erzeugen 5 Frames mit einem raeumlichen Gradienten (0.1*x + 0.2*y)
  // und pro Frame einem additiven Offset von i*10. Die Background-Map pro
  // Frame sollte den Gradienten + Offset erfassen; die akkumulierte Map
  // sollte den Gradienten (ohne Offset, da dieser ueber Frames mittelt)
  // innerhalb 2% relativer Fehler rekonstruieren.
  const int img_rows = 64, img_cols = 64;
  const int grid_rows = 8, grid_cols = 8;
  const int n_frames = 5;
  const float gx = 0.1f, gy = 0.2f;

  // Referenz-Gradient an Grid-Zellmittelpunkten.
  auto gradient_at_cell = [&](int r, int c) -> float {
    const float cell_h = static_cast<float>(img_rows) / grid_rows;
    const float cell_w = static_cast<float>(img_cols) / grid_cols;
    const float y = (r + 0.5f) * cell_h;
    const float x = (c + 0.5f) * cell_w;
    return gx * x + gy * y;
  };

  tile_compile::WarpMatrix identity_warp;
  identity_warp << 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f;

  tile_compile::runner::BackgroundMapCanvas canvas(grid_rows, grid_cols, 1,
                                                   n_frames);
  for (int fi = 0; fi < n_frames; ++fi) {
    tile_compile::Matrix2Df img(img_rows, img_cols);
    cv::Mat1b mask(img_rows, img_cols, 255);
    const float offset = static_cast<float>(fi) * 10.0f;
    for (int y = 0; y < img_rows; ++y) {
      for (int x = 0; x < img_cols; ++x) {
        img(y, x) = gx * static_cast<float>(x) +
                    gy * static_cast<float>(y) + offset;
      }
    }
    auto grid = tile_compile::runner::BackgroundModelGrid::from_image(
        img, mask, tile_compile::ColorMode::MONO, "RGGB", grid_rows, grid_cols);
    REQUIRE(grid.channels() == 1);
    canvas.accumulate(grid, img_rows, img_cols, img_rows, img_cols,
                      identity_warp);
  }

  auto accumulated = canvas.finalize();
  REQUIRE(accumulated.rows() == grid_rows);
  REQUIRE(accumulated.cols() == grid_cols);

  // Akzeptanzkriterium: <=2% relativer Fehler auf gueltigen Gridzellen.
  // Der mittlere Offset ueber 5 Frames ist (0+10+20+30+40)/5 = 20.
  const float expected_offset = 20.0f;
  int checked = 0;
  for (int r = 0; r < grid_rows; ++r) {
    for (int c = 0; c < grid_cols; ++c) {
      if (!accumulated.valid(r, c, 0))
        continue;
      const float expected = gradient_at_cell(r, c) + expected_offset;
      const float actual = accumulated.value(r, c, 0);
      const float rel_err = std::fabs(actual - expected) / std::fabs(expected);
      INFO("cell r=" << r << " c=" << c << " expected=" << expected
           << " actual=" << actual << " rel_err=" << rel_err);
      REQUIRE(rel_err <= 0.02f);
      ++checked;
    }
  }
  // Mindestens 80% der Zellen muessen gueltig und geprueft sein.
  REQUIRE(checked >= (grid_rows * grid_cols) * 4 / 5);
}

TEST_CASE("background_model_sigma_clip_rejects_outlier_frame") {
  // Two-pass sigma-Clip in der Cross-Frame-Akkumulation: ein Frame mit
  // stark abweichendem Offset (Ausreisser) sollte verworfen werden und das
  // Ergebnis dem der verbleibenden Frames folgen.
  const int grid_rows = 4, grid_cols = 4;
  const int n_normal = 4;
  const float normal_value = 10.0f;
  const float outlier_value = 100.0f; // klar ausserhalb 3*MAD

  tile_compile::runner::BackgroundModelGridStore store(
      unique_bg_grid_dir("sigma_clip"), n_normal + 1, grid_rows, grid_cols,
      {"L"});

  for (int fi = 0; fi < n_normal; ++fi) {
    tile_compile::runner::BackgroundModelGrid g(grid_rows, grid_cols, 1);
    for (int r = 0; r < grid_rows; ++r)
      for (int c = 0; c < grid_cols; ++c) {
        g.value(r, c, 0) = normal_value;
        g.support(r, c, 0) =
            tile_compile::runner::BackgroundModelGrid::kMeasured;
      }
    store.store(fi, g);
  }
  // Outlier-Frame
  {
    tile_compile::runner::BackgroundModelGrid g(grid_rows, grid_cols, 1);
    for (int r = 0; r < grid_rows; ++r)
      for (int c = 0; c < grid_cols; ++c) {
        g.value(r, c, 0) = outlier_value;
        g.support(r, c, 0) =
            tile_compile::runner::BackgroundModelGrid::kMeasured;
      }
    store.store(n_normal, g);
  }

  std::vector<uint8_t> frame_has_data(n_normal + 1, 1u);
  auto out = tile_compile::runner::accumulate_prewarped_background_maps(
      store, frame_has_data);
  // Mit 5 Frames und 50% Coverage-Floor (ceil(0.5*5)=3) sind alle Zellen
  // gueltig. Der Ausreisser sollte vom Sigma-Clip verworfen werden.
  for (int r = 0; r < grid_rows; ++r) {
    for (int c = 0; c < grid_cols; ++c) {
      REQUIRE(out.valid(r, c, 0));
      // Ergebnis sollte nahe normal_value liegen, nicht beim Mittelwert
      // (10*4 + 100)/5 = 28.
      REQUIRE(out.value(r, c, 0) == Catch::Approx(normal_value).margin(1.0f));
    }
  }
}

TEST_CASE("background_model_coverage_floor_rejects_low_coverage") {
  // 50%-Coverage-Schwelle: Eine Zelle mit deutlich weniger Beitraegen als
  // die am besten abgedeckte Zelle wird als invalid markiert.
  // max_count=4 (Zelle 0,0), coverage_floor=ceil(0.5*4)=2.
  // Zelle (1,1) hat nur 1 Beitrag -> invalid.
  const int grid_rows = 2, grid_cols = 2;
  tile_compile::runner::BackgroundModelGridStore store(
      unique_bg_grid_dir("coverage"), 4, grid_rows, grid_cols, {"L"});

  // Frames 0-3: Zelle (0,0) in allen 4 Frames gueltig, Zelle (1,1) nur in Frame 0.
  for (int fi = 0; fi < 4; ++fi) {
    tile_compile::runner::BackgroundModelGrid g(grid_rows, grid_cols, 1);
    g.value(0, 0, 0) = 5.0f + fi;
    g.support(0, 0, 0) = tile_compile::runner::BackgroundModelGrid::kMeasured;
    if (fi == 0) {
      g.value(1, 1, 0) = 9.0f;
      g.support(1, 1, 0) = tile_compile::runner::BackgroundModelGrid::kMeasured;
    }
    store.store(fi, g);
  }

  std::vector<uint8_t> frame_has_data{1u, 1u, 1u, 1u};
  auto out = tile_compile::runner::accumulate_prewarped_background_maps(
      store, frame_has_data);
  // Zelle (0,0): 4 Beitraege >= coverage_floor=2 -> valid
  REQUIRE(out.valid(0, 0, 0));
  // Zelle (1,1): 1 Beitrag < coverage_floor=2 -> invalid
  REQUIRE_FALSE(out.valid(1, 1, 0));
}

TEST_CASE("background_map_canvas_ceil_division_no_last_column_bloat") {
  // Regression: mit floor-Division cell_w = canvas_cols / cols_ bekommt die
  // letzte Spalte alle uebrigen Pixel und bläht max_count auf, so dass alle
  // anderen Zellen unter die coverage_floor fallen.  Ceil-Division verteilt
  // die Pixel gleichmaessiger.
  //
  // canvas 100x100, grid 3x3 -> floor: cell=33, letzte Spalte = 34 px.
  // ceil: cell=34, letzte Spalte = 32 px.
  const int canvas_h = 100, canvas_w = 100;
  const int grid_rows = 3, grid_cols = 3;
  tile_compile::WarpMatrix identity_warp;
  identity_warp << 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f;

  // Frame-Gitter 3x3, alle Zellen gemessen mit Wert 200.
  tile_compile::runner::BackgroundModelGrid frame_grid(grid_rows, grid_cols, 1);
  for (int r = 0; r < grid_rows; ++r)
    for (int c = 0; c < grid_cols; ++c) {
      frame_grid.value(r, c, 0) = 200.0f;
      frame_grid.support(r, c, 0) =
          tile_compile::runner::BackgroundModelGrid::kMeasured;
    }

  tile_compile::runner::BackgroundMapCanvas canvas(grid_rows, grid_cols, 1, 1);
  canvas.accumulate(frame_grid, canvas_h, canvas_w, canvas_h, canvas_w,
                    identity_warp);
  auto out = canvas.finalize();

  // Mit der alten floor-Division hatte die letzte Spalte ~34*33 = 1122 samples
  // und die mittleren Spalten ~33*33 = 1089.  max_count = 1122,
  // coverage_floor = 561.  Alle Zellen mit 1089 > 561 -> sollten bestehen.
  // Aber bei groesseren Dimensionen (z.B. 3924/131) wird der Effekt extremer.
  //
  // Hier pruefen wir, dass alle 9 Zellen gueltig sind (bei ceil-Division).
  int valid_count = 0;
  for (int r = 0; r < grid_rows; ++r)
    for (int c = 0; c < grid_cols; ++c)
      if (out.valid(r, c, 0))
        ++valid_count;
  // Alle Zellen muessen gueltig sein, da das gesamte Canvas abgedeckt ist.
  REQUIRE(valid_count == grid_rows * grid_cols);

  // Werte sollten nahe bei 200 liegen (kein Clipping durch zu hohe coverage_floor).
  for (int r = 0; r < grid_rows; ++r)
    for (int c = 0; c < grid_cols; ++c) {
      INFO("cell r=" << r << " c=" << c);
      REQUIRE(out.valid(r, c, 0));
      REQUIRE(out.value(r, c, 0) ==
              Catch::Approx(200.0f).margin(1.0f));
    }
}

TEST_CASE("background_map_canvas_large_canvas_last_column_not_dominant") {
  // Regression fuer den echten Production-Bug: canvas 2312x3924, grid 78x131.
  // Mit floor-Division cell_w = 29, letzte Spalte bekommt 154 px (statt 29).
  // Das bläht max_count auf ~4466 und coverage_floor auf ~2233, so dass alle
  // normalen Zellen (841 samples) abgelehnt werden.
  // Mit ceil-Division cell_w = 30, letzte Spante bekommt 24 px (<= 30).
  // max_count ~900, coverage_floor ~450, normale Zellen bestehen.
  const int canvas_h = 2312, canvas_w = 3924;
  const int grid_rows = 78, grid_cols = 131;
  tile_compile::WarpMatrix identity_warp;
  identity_warp << 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f;

  // Frame-Gitter 72x128 (original Background-Grid), alle Zellen gemessen.
  // Es wird auf canvas_h x canvas_w upgesampled.
  const int fg_rows = 72, fg_cols = 128;
  tile_compile::runner::BackgroundModelGrid frame_grid(fg_rows, fg_cols, 1);
  for (int r = 0; r < fg_rows; ++r)
    for (int c = 0; c < fg_cols; ++c) {
      frame_grid.value(r, c, 0) = 200.0f;
      frame_grid.support(r, c, 0) =
          tile_compile::runner::BackgroundModelGrid::kMeasured;
    }

  tile_compile::runner::BackgroundMapCanvas canvas(grid_rows, grid_cols, 1, 1);
  canvas.accumulate(frame_grid, canvas_h, canvas_w, canvas_h, canvas_w,
                    identity_warp);
  auto out = canvas.finalize();

  int valid_count = 0;
  for (int r = 0; r < grid_rows; ++r)
    for (int c = 0; c < grid_cols; ++c)
      if (out.valid(r, c, 0))
        ++valid_count;
  // Mit der alten floor-Division wuerden nur ~74 Zellen gueltig sein.
  // Mit ceil-Division sollten fast alle Zellen gueltig sein.
  // Da das Frame (72x128) auf 2312x3924 upgesampelt und mit Identity-Warp
  // auf das gleiche Canvas gelegt wird, sind alle Zellen abgedeckt.
  INFO("valid_count=" << valid_count << " of " << grid_rows * grid_cols);
  REQUIRE(valid_count > (grid_rows * grid_cols) * 9 / 10);
}
#endif
