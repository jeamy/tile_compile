#include "tile_compile/pipeline/adaptive_tile_grid.hpp"

#include "tile_compile/io/fits_io.hpp"

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cmath>

namespace tile_compile::pipeline {

static cv::Mat make_norm_gradient_fullres(const WarpGradientField& gf,
                                         int image_width,
                                         int image_height);

static Matrix2Df load_frame_region_safe(const fs::path& path,
                                       int x0, int y0, int w, int h,
                                       int image_width, int image_height) {
    int rx0 = std::max(0, x0);
    int ry0 = std::max(0, y0);
    int rx1 = std::min(image_width, x0 + w);
    int ry1 = std::min(image_height, y0 + h);

    int rw = std::max(0, rx1 - rx0);
    int rh = std::max(0, ry1 - ry0);
    if (rw <= 0 || rh <= 0) {
        return Matrix2Df();
    }

    try {
        return io::read_fits_region_float(path, rx0, ry0, rw, rh);
    } catch (...) {
        return Matrix2Df();
    }
}

static std::vector<int> linspace_indices(int n, int k) {
    std::vector<int> out;
    if (n <= 0 || k <= 0) return out;
    if (k == 1) {
        out.push_back(0);
        return out;
    }
    out.reserve(static_cast<size_t>(k));
    for (int i = 0; i < k; ++i) {
        float t = static_cast<float>(i) / static_cast<float>(k - 1);
        int idx = static_cast<int>(std::round(t * static_cast<float>(n - 1)));
        idx = std::min(std::max(idx, 0), n - 1);
        out.push_back(idx);
    }
    return out;
}

WarpGradientField compute_warp_gradient_field(const std::vector<fs::path>& frame_paths,
                                             int image_width,
                                             int image_height,
                                             int probe_window,
                                             int num_probe_frames,
                                             std::ostream* /*progress_out*/,
                                             std::function<void(float)> progress_cb) {
    WarpGradientField out;
    out.probe_window = probe_window;

    const int prev_threads = cv::getNumThreads();
    cv::setNumThreads(1);

    if (frame_paths.size() < 2 || image_width <= 0 || image_height <= 0) {
        out.grid = Matrix2Df::Zero(0, 0);
        cv::setNumThreads(prev_threads);
        return out;
    }

    int n_frames = static_cast<int>(frame_paths.size());
    out.probe_indices = linspace_indices(n_frames, num_probe_frames);

    if (out.probe_indices.size() < 2) {
        out.grid = Matrix2Df::Zero(0, 0);
        cv::setNumThreads(prev_threads);
        return out;
    }

    const int step = probe_window / 2;
    out.step = step;

    int grid_h = (image_height - probe_window) / step + 1;
    int grid_w = (image_width - probe_window) / step + 1;

    if (grid_h < 1 || grid_w < 1) {
        out.grid = Matrix2Df::Zero(0, 0);
        cv::setNumThreads(prev_threads);
        return out;
    }

    out.grid_h = grid_h;
    out.grid_w = grid_w;

    const int n_probe_pairs = static_cast<int>(out.probe_indices.size()) - 1;

    std::vector<float> dx_vals(static_cast<size_t>(grid_h * grid_w * n_probe_pairs), 0.0f);
    std::vector<float> dy_vals(static_cast<size_t>(grid_h * grid_w * n_probe_pairs), 0.0f);

    const int total_cells = grid_h * grid_w;
    int processed = 0;

    for (int gi = 0; gi < grid_h; ++gi) {
        for (int gj = 0; gj < grid_w; ++gj) {
            int y0 = gi * step;
            int x0 = gj * step;

            for (int pi = 0; pi < n_probe_pairs; ++pi) {
                const fs::path ref_path = frame_paths[out.probe_indices[pi]];
                const fs::path mov_path = frame_paths[out.probe_indices[pi + 1]];
                Matrix2Df ref = load_frame_region_safe(ref_path, x0, y0, probe_window, probe_window,
                                                      image_width, image_height);
                Matrix2Df mov = load_frame_region_safe(mov_path, x0, y0, probe_window, probe_window,
                                                      image_width, image_height);
                if (ref.size() <= 0 || mov.size() <= 0 || mov.rows() != ref.rows() || mov.cols() != ref.cols()) {
                    continue;
                }

                cv::Mat ref_cv(ref.rows(), ref.cols(), CV_32F, const_cast<float*>(ref.data()));

                cv::Mat mov_cv(mov.rows(), mov.cols(), CV_32F, const_cast<float*>(mov.data()));

                cv::Point2d shift(0.0, 0.0);
                try {
                    shift = cv::phaseCorrelate(mov_cv, ref_cv);
                } catch (...) {
                    shift = cv::Point2d(0.0, 0.0);
                }

                size_t base = static_cast<size_t>((gi * grid_w + gj) * n_probe_pairs + pi);
                dx_vals[base] = static_cast<float>(shift.x);
                dy_vals[base] = static_cast<float>(shift.y);
            }

            ++processed;
            if (progress_cb) progress_cb(static_cast<float>(processed) / static_cast<float>(total_cells));
        }
    }

    Matrix2Df gradient_mag(grid_h, grid_w);
    for (int gi = 0; gi < grid_h; ++gi) {
        for (int gj = 0; gj < grid_w; ++gj) {
            // variance across probe pairs
            float mean_dx = 0.0f;
            float mean_dy = 0.0f;
            for (int pi = 0; pi < n_probe_pairs; ++pi) {
                size_t base = static_cast<size_t>((gi * grid_w + gj) * n_probe_pairs + pi);
                mean_dx += dx_vals[base];
                mean_dy += dy_vals[base];
            }
            mean_dx /= std::max(1, n_probe_pairs);
            mean_dy /= std::max(1, n_probe_pairs);

            float var_dx = 0.0f;
            float var_dy = 0.0f;
            for (int pi = 0; pi < n_probe_pairs; ++pi) {
                size_t base = static_cast<size_t>((gi * grid_w + gj) * n_probe_pairs + pi);
                float ddx = dx_vals[base] - mean_dx;
                float ddy = dy_vals[base] - mean_dy;
                var_dx += ddx * ddx;
                var_dy += ddy * ddy;
            }
            var_dx /= std::max(1, n_probe_pairs);
            var_dy /= std::max(1, n_probe_pairs);

            gradient_mag(gi, gj) = std::sqrt(std::max(0.0f, var_dx + var_dy));
        }
    }

    out.grid = gradient_mag;

    // Stats
    if (out.grid.size() > 0) {
        float minv = out.grid.minCoeff();
        float maxv = out.grid.maxCoeff();
        float meanv = out.grid.mean();
        out.min_val = minv;
        out.max_val = maxv;
        out.mean_val = meanv;
    }

    cv::setNumThreads(prev_threads);
    return out;
}

std::vector<Tile> build_initial_tile_grid(int image_width,
                                         int image_height,
                                         int tile_size,
                                         float overlap_fraction) {
    std::vector<Tile> tiles;
    if (image_width <= 0 || image_height <= 0 || tile_size <= 0) return tiles;

    if (tile_size > image_width || tile_size > image_height) {
        tiles.push_back(Tile{0, 0, image_width, image_height, 0, 0});
        return tiles;
    }

    overlap_fraction = std::min(std::max(overlap_fraction, 0.0f), 0.5f);
    int overlap_px = static_cast<int>(std::floor(overlap_fraction * static_cast<float>(tile_size)));
    int step = std::max(1, tile_size - overlap_px);

    std::vector<int> xs;
    std::vector<int> ys;
    for (int x = 0; x <= image_width - tile_size; x += step) xs.push_back(x);
    for (int y = 0; y <= image_height - tile_size; y += step) ys.push_back(y);

    if (!xs.empty() && xs.back() + tile_size < image_width) xs.push_back(image_width - tile_size);
    if (!ys.empty() && ys.back() + tile_size < image_height) ys.push_back(image_height - tile_size);

    int row = 0;
    for (int y0 : ys) {
        int col = 0;
        for (int x0 : xs) {
            tiles.push_back(Tile{x0, y0, tile_size, tile_size, row, col});
            ++col;
        }
        ++row;
    }

    return tiles;
}

std::vector<Tile> build_adaptive_tile_grid(int image_width,
                                          int image_height,
                                          const config::Config& cfg,
                                          const WarpGradientField* gradient_field) {
    (void)gradient_field;
    int tile_size = cfg.tile.min_size;
    return build_initial_tile_grid(image_width, image_height, tile_size, cfg.tile.overlap_fraction);
}

static cv::Mat make_integral_from_gradient_grid(const WarpGradientField& gf,
                                               int image_width,
                                               int image_height) {
    if (gf.grid.size() <= 0) {
        return cv::Mat();
    }

    cv::Mat coarse(gf.grid.rows(), gf.grid.cols(), CV_32F, const_cast<float*>(gf.grid.data()));

    cv::Mat up;
    cv::resize(coarse, up, cv::Size(image_width, image_height), 0.0, 0.0, cv::INTER_LINEAR);

    // Smooth similar to python gaussian_filter(sigma=probe_window/4)
    double sigma = std::max(0.1, static_cast<double>(gf.probe_window) / 4.0);
    cv::GaussianBlur(up, up, cv::Size(0, 0), sigma, sigma);

    double minv = 0.0, maxv = 0.0;
    cv::minMaxLoc(up, &minv, &maxv);
    if (maxv > 1.0e-12) {
        up.convertTo(up, CV_32F, 1.0 / maxv);
    } else {
        up = cv::Mat::zeros(up.size(), CV_32F);
    }

    cv::Mat integral;
    cv::integral(up, integral, CV_64F);
    return integral;
}

static cv::Mat make_norm_gradient_fullres(const WarpGradientField& gf,
                                         int image_width,
                                         int image_height) {
    if (gf.grid.size() <= 0) {
        return cv::Mat::zeros(image_height, image_width, CV_32F);
    }

    cv::Mat coarse(gf.grid.rows(), gf.grid.cols(), CV_32F, const_cast<float*>(gf.grid.data()));

    cv::Mat up;
    cv::resize(coarse, up, cv::Size(image_width, image_height), 0.0, 0.0, cv::INTER_LINEAR);

    double sigma = std::max(0.1, static_cast<double>(gf.probe_window) / 4.0);
    cv::GaussianBlur(up, up, cv::Size(0, 0), sigma, sigma);

    double minv = 0.0, maxv = 0.0;
    cv::minMaxLoc(up, &minv, &maxv);
    if (maxv > 1.0e-12) {
        up.convertTo(up, CV_32F, 1.0 / maxv);
    } else {
        up = cv::Mat::zeros(up.size(), CV_32F);
    }

    return up;
}

static double integral_mean(const cv::Mat& integral, int x0, int y0, int w, int h) {
    if (integral.empty()) return 0.0;
    int x1 = x0 + w;
    int y1 = y0 + h;

    // integral is (h+1, w+1)
    x0 = std::max(0, x0);
    y0 = std::max(0, y0);
    x1 = std::max(0, x1);
    y1 = std::max(0, y1);

    x0 = std::min(x0, integral.cols - 1);
    x1 = std::min(x1, integral.cols - 1);
    y0 = std::min(y0, integral.rows - 1);
    y1 = std::min(y1, integral.rows - 1);

    if (x1 <= x0 || y1 <= y0) return 0.0;

    double A = integral.at<double>(y0, x0);
    double B = integral.at<double>(y0, x1);
    double C = integral.at<double>(y1, x0);
    double D = integral.at<double>(y1, x1);

    double sum = D - B - C + A;
    double area = static_cast<double>((x1 - x0) * (y1 - y0));
    if (area <= 0.0) return 0.0;
    return sum / area;
}

static std::vector<Tile> generate_subtiles_with_overlap(int x0, int y0, int tw, int th,
                                                        int tile_size, float overlap_frac) {
    std::vector<Tile> out;

    overlap_frac = std::min(std::max(overlap_frac, 0.0f), 0.5f);
    int overlap_px = static_cast<int>(std::floor(overlap_frac * static_cast<float>(tile_size)));
    int step = std::max(1, tile_size - overlap_px);

    int row = -1;
    for (int sy = y0; sy <= (y0 + th - tile_size); sy += step) {
        int col = -1;
        for (int sx = x0; sx <= (x0 + tw - tile_size); sx += step) {
            out.push_back(Tile{sx, sy, tile_size, tile_size, row, col});
        }
    }

    if (out.empty()) {
        out.push_back(Tile{x0, y0, std::min(tw, tile_size), std::min(th, tile_size), -1, -1});
    }

    return out;
}

static void split_region(std::vector<Tile>& out_tiles,
                         const cv::Mat& grad_integral,
                         int image_width, int image_height,
                         int x0, int y0, int tw, int th,
                         int depth,
                         int max_depth,
                         int min_tile_size,
                         int initial_tile_size,
                         float split_threshold,
                         float overlap_frac) {
    int tile_size = std::max(min_tile_size, initial_tile_size >> depth);

    if (depth >= max_depth || tw <= min_tile_size * 2 || th <= min_tile_size * 2) {
        auto sub = generate_subtiles_with_overlap(x0, y0, tw, th, tile_size, overlap_frac);
        out_tiles.insert(out_tiles.end(), sub.begin(), sub.end());
        return;
    }

    double region_grad = integral_mean(grad_integral, x0, y0, tw, th);

    if (region_grad < static_cast<double>(split_threshold)) {
        auto sub = generate_subtiles_with_overlap(x0, y0, tw, th, tile_size, overlap_frac);
        out_tiles.insert(out_tiles.end(), sub.begin(), sub.end());
        return;
    }

    int hw = tw / 2;
    int hh = th / 2;

    int ovw = static_cast<int>(std::round(static_cast<float>(hw) * overlap_frac));
    int ovh = static_cast<int>(std::round(static_cast<float>(hh) * overlap_frac));

    split_region(out_tiles, grad_integral, image_width, image_height,
                 x0, y0, hw + ovw, hh + ovh, depth + 1, max_depth,
                 min_tile_size, initial_tile_size, split_threshold, overlap_frac);

    split_region(out_tiles, grad_integral, image_width, image_height,
                 x0 + hw - ovw, y0, (tw - hw) + ovw, hh + ovh, depth + 1, max_depth,
                 min_tile_size, initial_tile_size, split_threshold, overlap_frac);

    split_region(out_tiles, grad_integral, image_width, image_height,
                 x0, y0 + hh - ovh, hw + ovw, (th - hh) + ovh, depth + 1, max_depth,
                 min_tile_size, initial_tile_size, split_threshold, overlap_frac);

    split_region(out_tiles, grad_integral, image_width, image_height,
                 x0 + hw - ovw, y0 + hh - ovh, (tw - hw) + ovw, (th - hh) + ovh, depth + 1, max_depth,
                 min_tile_size, initial_tile_size, split_threshold, overlap_frac);
}

std::vector<Tile> build_hierarchical_tile_grid(int image_width,
                                              int image_height,
                                              const config::Config& cfg,
                                              const WarpGradientField* gradient_field) {
    (void)gradient_field;
    return build_initial_tile_grid(image_width, image_height, cfg.tile.min_size, cfg.tile.overlap_fraction);
}

} // namespace tile_compile::pipeline
