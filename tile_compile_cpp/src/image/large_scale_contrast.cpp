#include "tile_compile/image/large_scale_contrast.hpp"

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cmath>
#include <limits>

namespace tile_compile::image {
namespace {

// Large-scale map on a coarse grid: NaN-free values in `map`, `ok` marks cells
// that carry a trustworthy value (enough valid pixels, enough Gaussian support).
struct CoarseMap {
  cv::Mat map;  // CV_32F
  cv::Mat ok;   // CV_8U, 1 = usable
  int rows = 0, cols = 0;
};

double median_of(std::vector<float> v) {
  if (v.empty()) return 0.0;
  const std::size_t mid = v.size() / 2;
  std::nth_element(v.begin(), v.begin() + mid, v.end());
  double m = v[mid];
  if (v.size() % 2 == 0) {
    const double lo = *std::max_element(v.begin(), v.begin() + mid);
    m = 0.5 * (m + lo);
  }
  return m;
}

double percentile_of(std::vector<float> v, double p) {
  if (v.empty()) return 0.0;
  const std::size_t k = std::min(
      v.size() - 1, static_cast<std::size_t>(p * static_cast<double>(v.size() - 1)));
  std::nth_element(v.begin(), v.begin() + k, v.end());
  return v[k];
}

// Fills the cells NOT marked ok in-place, from nearby ok cells only (a few rounds of a small masked
// blur, i.e. a short diffusion/push-outward from the known values), so the median filter right after
// this never sees a value pulled from the WHOLE image. A single global constant is wrong here: for a
// large excluded zone (e.g. a bright star's full sigma_px-scaled margin, which can span several whole
// coarse cells with no real coverage left at all) it pulls those cells toward the whole image's median
// regardless of the true local brightness nearby -- genuinely brighter nebula, or the excluded star's
// own immediate surroundings -- and that then leaks into the 5x5 median filter and shows up as a dark
// disc right where the star was (2026-09-28/29, real IC434 data). A handful of iterations is enough to
// reach across any real star's margin; a cell still unreached after that (e.g. deep inside a much
// larger masked-out region such as the frame border) keeps the global median as a last resort.
// `grounded` (if given) marks every cell this diffusion actually reached with real signal --
// the original cell_ok cells plus every cell the 8 rounds below pulled a value into -- as opposed
// to a cell that only got the last-resort `global_fill` because nothing real was reachable at all
// (e.g. deep inside the image border). Callers that need to know which cells now carry a
// trustworthy value, not just a filled-in placeholder, should use `grounded`, not `cell_ok`: a
// dense cluster of overlapping star-exclusion margins (e.g. a bright star with several fainter
// companions nearby) can leave a combined excluded area wider than the fixed-scale second-stage
// blur below can reach on its own, even though THIS diffusion -- which iterates outward rather
// than using one single fixed-radius blur -- already reached across it (2026-09-29, real IC434
// data: Alnitak plus nearby companions left a dark, wrong-shaped hole in the boost right where the
// combined margin was too wide for the old single-pass reach test, even after the per-star margin
// and the cell_ok-drop fixes).
void fill_from_neighbours(cv::Mat &small, const cv::Mat &cell_ok, float global_fill, cv::Mat *grounded = nullptr) {
  cv::Mat weight;
  cell_ok.convertTo(weight, CV_32F, 1.0 / 255.0);
  cv::Mat value = small.mul(weight);
  for (int iter = 0; iter < 8; ++iter) {
    cv::Mat value_blur, weight_blur;
    cv::GaussianBlur(value, value_blur, cv::Size(0, 0), 2.0, 2.0, cv::BORDER_REPLICATE);
    cv::GaussianBlur(weight, weight_blur, cv::Size(0, 0), 2.0, 2.0, cv::BORDER_REPLICATE);
    for (int y = 0; y < small.rows; ++y)
      for (int x = 0; x < small.cols; ++x) {
        if (weight.at<float>(y, x) >= 0.999f) continue;  // already fully known, do not dilute it
        const float w = weight_blur.at<float>(y, x);
        if (w > 1e-6f) {
          value.at<float>(y, x) = value_blur.at<float>(y, x);
          weight.at<float>(y, x) = std::min(1.0f, w);
        }
      }
  }
  if (grounded) *grounded = cv::Mat(small.rows, small.cols, CV_8U, cv::Scalar(0));
  for (int y = 0; y < small.rows; ++y)
    for (int x = 0; x < small.cols; ++x)
      if (!cell_ok.at<std::uint8_t>(y, x)) {
        const float w = weight.at<float>(y, x);
        const bool reached = w > 1e-3f;
        small.at<float>(y, x) = reached ? value.at<float>(y, x) / w : global_fill;
        if (grounded) grounded->at<std::uint8_t>(y, x) = reached ? 255 : 0;
      } else if (grounded) {
        grounded->at<std::uint8_t>(y, x) = 255;
      }
}

// True for every cell whose not-cell_ok connected component (4-connectivity) never touches the
// grid's outer border. A bright-star exclusion (including several overlapping ones merged into one
// blob) is always an ISLAND fully surrounded by real coverage; a genuinely invalid area -- the
// image's own edge, or an external mask cutting into the frame -- always touches the grid border
// (it has no "far side" to be surrounded by). This tells the diffusion fallback below which excluded
// cells it may fill in (islands) and which it must leave alone (border-touching), so a real border
// still gets a hard, unextrapolated edge instead of a spurious boost bleeding in from extrapolated
// values (caught by a unit test, 2026-09-29).
cv::Mat interior_islands(const cv::Mat &cell_ok) {
  cv::Mat not_ok = cell_ok == 0;
  cv::Mat labels;
  const int n = cv::connectedComponents(not_ok, labels, 4, CV_32S);
  std::vector<std::uint8_t> touches_border(static_cast<std::size_t>(n), 0);
  const int rows = labels.rows, cols = labels.cols;
  for (int x = 0; x < cols; ++x) {
    touches_border[static_cast<std::size_t>(labels.at<std::int32_t>(0, x))] = 1;
    touches_border[static_cast<std::size_t>(labels.at<std::int32_t>(rows - 1, x))] = 1;
  }
  for (int y = 0; y < rows; ++y) {
    touches_border[static_cast<std::size_t>(labels.at<std::int32_t>(y, 0))] = 1;
    touches_border[static_cast<std::size_t>(labels.at<std::int32_t>(y, cols - 1))] = 1;
  }
  cv::Mat island(rows, cols, CV_8U, cv::Scalar(0));
  for (int y = 0; y < rows; ++y)
    for (int x = 0; x < cols; ++x) {
      const std::int32_t lbl = labels.at<std::int32_t>(y, x);
      if (lbl != 0 && !touches_border[static_cast<std::size_t>(lbl)]) island.at<std::uint8_t>(y, x) = 1;
    }
  return island;
}

CoarseMap build_map(const cv::Mat &plane, const cv::Mat &valid, int factor, float sigma_px) {
  CoarseMap out;
  const int rows = std::max(1, plane.rows / factor), cols = std::max(1, plane.cols / factor);
  out.rows = rows;
  out.cols = cols;
  cv::Mat num, den, valid_f;
  valid.convertTo(valid_f, CV_32F);
  cv::multiply(plane, valid_f, num);
  cv::Mat num_s, den_s;
  cv::resize(num, num_s, cv::Size(cols, rows), 0, 0, cv::INTER_AREA);
  cv::resize(valid_f, den_s, cv::Size(cols, rows), 0, 0, cv::INTER_AREA);
  // A LOW bar on purpose: a cell only needs a little real coverage to compute its own local average
  // from it. Originally 0.9 ("mostly inside the valid area"), which was fine when the only excluded
  // region was one contiguous border strip -- every cell was either fully in or fully out. With
  // bright_source_mask's small, scattered per-star exclusions, most affected coarse cells keep the
  // bulk of their area from ordinary sky; forcing them to the >90 % bar instead marked them "not ok"
  // and pulled them onto the GLOBAL median fill below, which then leaked into the 5x5 median filter's
  // neighbourhood at nearby REAL cells and showed up as small dark boxes around faint stars near a
  // brighter area (2026-09-28, real IC434 data). A cell is only truly unusable when it has next to no
  // real coverage left, which for a small dilated star exclusion is rare.
  cv::Mat cell_ok = den_s > 0.15f;
  cv::Mat small(rows, cols, CV_32F, cv::Scalar(0));
  for (int y = 0; y < rows; ++y)
    for (int x = 0; x < cols; ++x)
      if (cell_ok.at<std::uint8_t>(y, x)) small.at<float>(y, x) = num_s.at<float>(y, x) / den_s.at<float>(y, x);
  std::vector<float> usable;
  for (int y = 0; y < rows; ++y)
    for (int x = 0; x < cols; ++x)
      if (cell_ok.at<std::uint8_t>(y, x)) usable.push_back(small.at<float>(y, x));
  const float global_fill = static_cast<float>(median_of(usable));
  cv::Mat grounded;
  fill_from_neighbours(small, cell_ok, global_fill, &grounded);
  // Fallback eligibility (see interior_islands): only an interior excluded island may use the
  // diffusion fallback below; a border-touching excluded area keeps the strict cell_ok-weighted
  // Gaussian result even where that leaves it unmarked.
  cv::Mat fallback_ok;
  cv::bitwise_and(grounded, interior_islands(cell_ok), fallback_ok);
  cv::Mat med;
  if (rows >= 5 && cols >= 5) cv::medianBlur(small, med, 5);  // float median: ksize <= 5
  else med = small;
  // Masked (normalised) Gaussian: border cells are averaged only over usable neighbours. The weight
  // here is `cell_ok` (real, directly-measured coverage only), deliberately NOT `grounded`: a
  // diffusion-filled cell (e.g. deep inside a genuinely invalid image border) carries an
  // extrapolated, not measured, value, and letting the smoothing Gaussian pull that value back
  // into a real, fully-covered cell nearby is a small but measurable leak into otherwise flat sky
  // right next to the border (caught by a unit test, 2026-09-29). `grounded` is used below only as
  // a fallback for cells this Gaussian itself cannot reach.
  const double sigma = std::max(0.5, static_cast<double>(sigma_px) / factor);
  cv::Mat ok_f, med_ok, blur_num, blur_den;
  cell_ok.convertTo(ok_f, CV_32F, 1.0 / 255.0);
  cv::multiply(med, ok_f, med_ok);
  cv::GaussianBlur(med_ok, blur_num, cv::Size(0, 0), sigma, sigma, cv::BORDER_CONSTANT);
  cv::GaussianBlur(ok_f, blur_den, cv::Size(0, 0), sigma, sigma, cv::BORDER_CONSTANT);
  out.map = cv::Mat(rows, cols, CV_32F, cv::Scalar(0));
  out.ok = cv::Mat(rows, cols, CV_8U, cv::Scalar(0));
  // Primary source: the cell_ok-weighted Gaussian above, wherever it has enough real coverage in
  // reach (blur_den > 0.5) -- this is the properly smoothed estimate and is preferred whenever it's
  // available. Fallback: `fallback_ok` (grounded AND an interior island, see interior_islands and
  // fill_from_neighbours) for a cell this Gaussian's fixed reach cannot cover on its own -- typically
  // a coarse cell deep inside a WIDE excluded area made of several overlapping bright_source_mask
  // margins (e.g. a bright star with nearby fainter companions), which can be wider than this
  // Gaussian's sigma even though the diffusion in fill_from_neighbours, which iterates outward rather
  // than using one fixed-radius blur, already reached across it and left a real value in `med`.
  // Without this fallback, such a cell got no value at all (out.map/out.ok stayed at their initial
  // zero), producing a sharp, correctly-shaped but visibly dark hole exactly the size of the combined
  // exclusion area (2026-09-29, real IC434 data, Alnitak plus nearby companions) -- the opposite
  // failure from a global-median leak: not a wrong value, but no value at all. The `interior_islands`
  // restriction keeps this from also applying to a genuinely invalid, border-touching area (e.g. the
  // image edge): the diffusion in fill_from_neighbours has no "far side" to be surrounded by there and
  // still marks it `grounded` after enough iterations, which would otherwise leak an extrapolated
  // value into real sky right next to the border (caught by a unit test, 2026-09-29).
  for (int y = 0; y < rows; ++y)
    for (int x = 0; x < cols; ++x) {
      const float d = blur_den.at<float>(y, x);
      if (d > 0.5f) {
        out.map.at<float>(y, x) = blur_num.at<float>(y, x) / d;
        out.ok.at<std::uint8_t>(y, x) = 1;
      } else if (fallback_ok.at<std::uint8_t>(y, x)) {
        out.map.at<float>(y, x) = med.at<float>(y, x);
        out.ok.at<std::uint8_t>(y, x) = 1;
      }
    }
  return out;
}

// Bright-source (star) mask on the FULL-resolution luminance: POINT-LIKE bright pixels, and a margin
// around them, are excluded from the coarse-map construction below (never from the final boost itself
// -- a star still receives whatever value its surrounding sky gets). A bright, extended star's wings
// can span several coarse cells at typical sigma_px (e.g. a factor-8 grid), so they survive the coarse
// 5x5 median unless removed earlier; without this, the star's own bump is smoothed by the Gaussian and
// added back as a soft halo -- exactly the "large-scale structure" this stage boosts, applied to a star
// instead of a nebula. Detection is deliberately a LOCAL-contrast test (residual against a small-sigma
// blur), not a plain "brighter than the image's global sky level" one: a real nebula core can be
// brighter than the sky by far more than any fixed multiple of the noise, but it varies smoothly across
// many multiples of the local blur scale, so it leaves almost no residual here, while a star -- much
// sharper than the local blur -- does. The margin AROUND a detected core, though, must scale with
// sigma_px: real-data measurement on a very bright star (Alnitak, IC434, 2026-09-28) found its
// above-sky excess still present at 1 % of sky at r=40 px (sigma_px 48 there) and gone by r=60-80,
// i.e. comparable to sigma_px itself, not a small fixed handful of pixels -- a star's smooth WING is
// exactly the kind of gently-varying brightness this stage's own detector is designed to leave alone
// (that is the point, for nebula), so the wing is not caught by the point-source test and needs this
// separate, sigma_px-scaled margin instead. The local sigma, threshold multiple and sigma floor stay
// fixed constants; only the margin depends on sigma_px, and even that is not user-facing -- both exist
// to make the exclusion this header already promises ("hot pixels/stars ... removed") actually hold for
// real stars, not to be a user-tunable star-detector.
cv::Mat bright_source_mask(const cv::Mat &lum, const cv::Mat &valid, float sigma_px) {
  const int rows = lum.rows, cols = lum.cols;
  cv::Mat smooth;
  cv::GaussianBlur(lum, smooth, cv::Size(0, 0), 4.0, 4.0, cv::BORDER_REPLICATE);
  cv::Mat residual = lum - smooth;
  const std::size_t total = static_cast<std::size_t>(rows) * cols;
  const std::size_t stride = std::max<std::size_t>(1, total / 500000u);
  std::vector<float> sample;
  sample.reserve(std::min<std::size_t>(total, 500000u));
  std::size_t linear = 0;
  for (int y = 0; y < rows; ++y)
    for (int x = 0; x < cols; ++x, ++linear)
      if (linear % stride == 0 && valid.at<std::uint8_t>(y, x)) sample.push_back(residual.at<float>(y, x));
  cv::Mat mask(rows, cols, CV_8U, cv::Scalar(0));
  if (sample.empty()) return mask;
  const double med = median_of(sample);
  std::vector<float> dev;
  dev.reserve(sample.size());
  for (float v : sample) dev.push_back(std::fabs(v - static_cast<float>(med)));
  // A floor well below any real per-pixel noise, so a noise-free (or near flat) region does not make
  // the threshold pathologically sensitive to numerical-precision-level residual.
  const double sigma = std::max(1.4826 * median_of(dev), 1e-4);
  const double threshold = med + 8.0 * sigma;
  for (int y = 0; y < rows; ++y)
    for (int x = 0; x < cols; ++x)
      if (valid.at<std::uint8_t>(y, x) && residual.at<float>(y, x) > threshold) mask.at<std::uint8_t>(y, x) = 255;
  cv::Mat dilated;
  const int margin_radius = std::clamp(static_cast<int>(std::lround(sigma_px * 0.75)), 5, 200);
  const int kernel = 2 * margin_radius + 1;
  cv::dilate(mask, dilated, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(kernel, kernel)));
  return dilated;
}

// Robust radial fit in r^2 (degree 3) over the usable cells; returns the fitted
// value at every cell. IRLS with a 2-sigma soft clip keeps a bright nebula
// from driving the vignette estimate.
cv::Mat radial_fit(const CoarseMap &m) {
  const int rows = m.rows, cols = m.cols;
  double cy = 0, cx = 0;
  std::size_t n = 0;
  for (int y = 0; y < rows; ++y)
    for (int x = 0; x < cols; ++x)
      if (m.ok.at<std::uint8_t>(y, x)) { cy += y; cx += x; ++n; }
  cv::Mat fit(rows, cols, CV_32F, cv::Scalar(0));
  if (n < 16) return fit;
  cy /= n; cx /= n;
  const double scale = std::max(cy, cx) * std::max(cy, cx) + 1e-9;
  constexpr int kDeg = 3;
  auto basis = [&](int y, int x, double *b) {
    const double r2 = ((y - cy) * (y - cy) + (x - cx) * (x - cx)) / scale;
    double p = 1.0;
    for (int k = 0; k <= kDeg; ++k) { b[k] = p; p *= r2; }
  };
  std::vector<double> w(n, 1.0);
  double coef[kDeg + 1] = {0, 0, 0, 0};
  for (int iter = 0; iter < 5; ++iter) {
    double ata[kDeg + 1][kDeg + 1] = {}, atb[kDeg + 1] = {};
    std::size_t i = 0;
    for (int y = 0; y < rows; ++y)
      for (int x = 0; x < cols; ++x) {
        if (!m.ok.at<std::uint8_t>(y, x)) continue;
        double b[kDeg + 1];
        basis(y, x, b);
        const double v = m.map.at<float>(y, x);
        for (int r = 0; r <= kDeg; ++r) {
          atb[r] += w[i] * b[r] * v;
          for (int c = 0; c <= kDeg; ++c) ata[r][c] += w[i] * b[r] * b[c];
        }
        ++i;
      }
    // Gaussian elimination on the 4x4 normal equations.
    double a[kDeg + 1][kDeg + 2];
    for (int r = 0; r <= kDeg; ++r) {
      for (int c = 0; c <= kDeg; ++c) a[r][c] = ata[r][c] + (r == c ? 1e-12 : 0.0);
      a[r][kDeg + 1] = atb[r];
    }
    for (int col = 0; col <= kDeg; ++col) {
      int piv = col;
      for (int r = col + 1; r <= kDeg; ++r)
        if (std::fabs(a[r][col]) > std::fabs(a[piv][col])) piv = r;
      if (std::fabs(a[piv][col]) < 1e-14) return fit;
      for (int c = 0; c <= kDeg + 1; ++c) std::swap(a[col][c], a[piv][c]);
      for (int r = col + 1; r <= kDeg; ++r) {
        const double f = a[r][col] / a[col][col];
        for (int c = col; c <= kDeg + 1; ++c) a[r][c] -= f * a[col][c];
      }
    }
    for (int r = kDeg; r >= 0; --r) {
      double s = a[r][kDeg + 1];
      for (int c = r + 1; c <= kDeg; ++c) s -= a[r][c] * coef[c];
      coef[r] = s / a[r][r];
    }
    std::vector<float> res;
    res.reserve(n);
    i = 0;
    std::vector<double> resid(n);
    for (int y = 0; y < rows; ++y)
      for (int x = 0; x < cols; ++x) {
        if (!m.ok.at<std::uint8_t>(y, x)) continue;
        double b[kDeg + 1];
        basis(y, x, b);
        double f = 0;
        for (int k = 0; k <= kDeg; ++k) f += coef[k] * b[k];
        resid[i] = m.map.at<float>(y, x) - f;
        res.push_back(static_cast<float>(resid[i]));
        ++i;
      }
    const double med = median_of(res);
    std::vector<float> dev;
    dev.reserve(n);
    for (float r : res) dev.push_back(std::fabs(r - static_cast<float>(med)));
    const double s = 1.4826 * median_of(dev) + 1e-9;
    for (std::size_t j = 0; j < n; ++j) w[j] = 1.0 / std::max(1.0, std::fabs(resid[j]) / (2.0 * s));
  }
  for (int y = 0; y < rows; ++y)
    for (int x = 0; x < cols; ++x) {
      double b[kDeg + 1];
      basis(y, x, b);
      double f = 0;
      for (int k = 0; k <= kDeg; ++k) f += coef[k] * b[k];
      fit.at<float>(y, x) = static_cast<float>(f);
    }
  return fit;
}

// Deviation map of a plane from its own sky level (median over usable cells),
// vignette component removed if asked; zero outside the usable cells.
cv::Mat deviation_map(const cv::Mat &plane, const cv::Mat &valid, int factor, float sigma_px,
                      bool remove_vignette, CoarseMap *map_out, double *reference_out) {
  CoarseMap m = build_map(plane, valid, factor, sigma_px);
  std::vector<float> usable;
  for (int y = 0; y < m.rows; ++y)
    for (int x = 0; x < m.cols; ++x)
      if (m.ok.at<std::uint8_t>(y, x)) usable.push_back(m.map.at<float>(y, x));
  const double ref = median_of(usable);
  cv::Mat dev(m.rows, m.cols, CV_32F, cv::Scalar(0));
  cv::Mat radial;
  if (remove_vignette) radial = radial_fit(m);
  for (int y = 0; y < m.rows; ++y)
    for (int x = 0; x < m.cols; ++x) {
      if (!m.ok.at<std::uint8_t>(y, x)) continue;
      float d = m.map.at<float>(y, x) - static_cast<float>(ref);
      if (remove_vignette) d -= radial.at<float>(y, x) - static_cast<float>(ref);
      dev.at<float>(y, x) = d;
    }
  if (map_out) *map_out = m;
  if (reference_out) *reference_out = ref;
  return dev;
}

double span_p5_p95(const cv::Mat &dev, const cv::Mat &ok) {
  std::vector<float> v;
  for (int y = 0; y < dev.rows; ++y)
    for (int x = 0; x < dev.cols; ++x)
      if (ok.at<std::uint8_t>(y, x)) v.push_back(dev.at<float>(y, x));
  return percentile_of(v, 0.95) - percentile_of(v, 0.05);
}

cv::Mat wrap(Matrix2Df &m) { return cv::Mat(static_cast<int>(m.rows()), static_cast<int>(m.cols()), CV_32F, m.data()); }
cv::Mat wrap_const(const Matrix2Df &m) {
  return cv::Mat(static_cast<int>(m.rows()), static_cast<int>(m.cols()), CV_32F, const_cast<float *>(m.data()));
}

}  // namespace

LargeScaleContrastResult apply_large_scale_contrast(
    Matrix2Df &R, Matrix2Df &G, Matrix2Df &B, const LargeScaleContrastConfig &cfg,
    const std::vector<std::uint8_t> *valid_mask) {
  LargeScaleContrastResult res;
  if (!cfg.enabled) return res;
  const int rows = static_cast<int>(R.rows()), cols = static_cast<int>(R.cols());
  if (rows < 32 || cols < 32 || G.rows() != R.rows() || G.cols() != R.cols() || B.rows() != R.rows() || B.cols() != R.cols()) {
    res.status = "error";
    res.error_message = "channels must have the same size and at least 32x32 pixels";
    return res;
  }
  if (!(cfg.amount >= 0.0f && cfg.chroma_amount >= 0.0f && cfg.sigma_px > 0.0f)) {
    res.status = "error";
    res.error_message = "amount and chroma_amount must be >= 0 and sigma_px > 0";
    return res;
  }
  if (cfg.amount == 0.0f && cfg.chroma_amount == 0.0f) {
    res.status = "not_needed";
    return res;
  }
  if (valid_mask && valid_mask->size() != static_cast<std::size_t>(rows) * cols) {
    res.status = "error";
    res.error_message = "valid mask size does not match the image";
    return res;
  }
  cv::Mat valid(rows, cols, CV_8U, cv::Scalar(0));
  std::size_t n_valid = 0;
  for (int y = 0; y < rows; ++y)
    for (int x = 0; x < cols; ++x) {
      const std::size_t i = static_cast<std::size_t>(y) * cols + x;
      bool v;
      if (valid_mask) {
        v = (*valid_mask)[i] != 0;
      } else {
        const float r = R(y, x), g = G(y, x), b = B(y, x);
        v = std::isfinite(r) && std::isfinite(g) && std::isfinite(b) && (r != 0.0f || g != 0.0f || b != 0.0f);
      }
      if (v) { valid.at<std::uint8_t>(y, x) = 1; ++n_valid; }
    }
  if (n_valid < static_cast<std::size_t>(rows) * cols / 20) {  // less than 5 % of the frame
    res.status = "too_few_valid_pixels";
    return res;
  }
  // Coarse grid of about sigma/6 pixels per cell (at least 1, at most 16).
  const int factor = std::clamp(static_cast<int>(std::floor(cfg.sigma_px / 6.0f)), 1, 16);
  res.downsample_factor = factor;

  Matrix2Df lum(rows, cols);
  for (int y = 0; y < rows; ++y)
    for (int x = 0; x < cols; ++x)
      lum(y, x) = valid.at<std::uint8_t>(y, x) ? (R(y, x) + G(y, x) + B(y, x)) / 3.0f : 0.0f;

  // Per-channel ceilings: the boost never pushes a channel beyond its own input maximum.
  float max_r = 0, max_g = 0, max_b = 0;
  for (int y = 0; y < rows; ++y)
    for (int x = 0; x < cols; ++x)
      if (valid.at<std::uint8_t>(y, x)) {
        max_r = std::max(max_r, R(y, x));
        max_g = std::max(max_g, G(y, x));
        max_b = std::max(max_b, B(y, x));
      }

  // Excluded only from the coarse-map construction (deviation_map calls below); the boost is still
  // applied to every valid pixel, stars included, via `valid` in the final loop further down.
  cv::Mat clean_valid = valid.clone();
  clean_valid.setTo(0, bright_source_mask(wrap_const(lum), valid, cfg.sigma_px));

  cv::Mat delta_l, delta_rg, delta_bg;
  CoarseMap lum_map;
  double ref = 0.0;
  cv::Mat dev_l = deviation_map(wrap_const(lum), clean_valid, factor, cfg.sigma_px, cfg.remove_vignette, &lum_map, &ref);
  res.sky_reference = ref;
  res.vignette_removed = cfg.remove_vignette;
  res.span_before = span_p5_p95(dev_l, lum_map.ok);
  if (cfg.amount > 0.0f) {
    cv::Mat scaled = dev_l * cfg.amount;
    cv::resize(scaled, delta_l, cv::Size(cols, rows), 0, 0, cv::INTER_LINEAR);
    res.span_after = span_p5_p95(scaled, lum_map.ok);
  } else {
    res.span_after = res.span_before;
  }
  if (cfg.chroma_amount > 0.0f) {
    Matrix2Df rg(rows, cols), bg(rows, cols);
    for (int y = 0; y < rows; ++y)
      for (int x = 0; x < cols; ++x) {
        rg(y, x) = valid.at<std::uint8_t>(y, x) ? R(y, x) - G(y, x) : 0.0f;
        bg(y, x) = valid.at<std::uint8_t>(y, x) ? B(y, x) - G(y, x) : 0.0f;
      }
    cv::Mat d_rg = deviation_map(wrap_const(rg), clean_valid, factor, cfg.sigma_px, cfg.remove_vignette, nullptr, nullptr) * cfg.chroma_amount;
    cv::Mat d_bg = deviation_map(wrap_const(bg), clean_valid, factor, cfg.sigma_px, cfg.remove_vignette, nullptr, nullptr) * cfg.chroma_amount;
    cv::resize(d_rg, delta_rg, cv::Size(cols, rows), 0, 0, cv::INTER_LINEAR);
    cv::resize(d_bg, delta_bg, cv::Size(cols, rows), 0, 0, cv::INTER_LINEAR);
  }
  for (int y = 0; y < rows; ++y)
    for (int x = 0; x < cols; ++x) {
      if (!valid.at<std::uint8_t>(y, x)) continue;
      const float dl = delta_l.empty() ? 0.0f : delta_l.at<float>(y, x);
      const float dr = delta_rg.empty() ? 0.0f : delta_rg.at<float>(y, x);
      const float db = delta_bg.empty() ? 0.0f : delta_bg.at<float>(y, x);
      R(y, x) = std::clamp(R(y, x) + dl + dr, 0.0f, std::max(max_r, R(y, x)));
      G(y, x) = std::clamp(G(y, x) + dl, 0.0f, std::max(max_g, G(y, x)));
      B(y, x) = std::clamp(B(y, x) + dl + db, 0.0f, std::max(max_b, B(y, x)));
    }
  res.applied = true;
  res.status = "applied";
  return res;
}

}  // namespace tile_compile::image
