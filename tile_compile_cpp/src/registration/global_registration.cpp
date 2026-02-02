#include "tile_compile/registration/global_registration.hpp"

#include "tile_compile/image/cfa_processing.hpp"
#include "tile_compile/registration/registration.hpp"

#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <unordered_map>

namespace tile_compile::registration {

namespace {

float median_of(std::vector<float>& v) {
    if (v.empty()) return 0.0f;
    const size_t n = v.size();
    const size_t mid = n / 2;
    std::nth_element(v.begin(), v.begin() + mid, v.end());
    const float hi = v[mid];
    if ((n % 2) == 1) {
        return hi;
    }
    std::nth_element(v.begin(), v.begin() + (mid - 1), v.end());
    const float lo = v[mid - 1];
    return 0.5f * (lo + hi);
}

float robust_sigma_mad(std::vector<float> pixels) {
    if (pixels.empty()) return 0.0f;
    float med = median_of(pixels);
    for (float& x : pixels) x = std::fabs(x - med);
    float mad = median_of(pixels);
    return 1.4826f * mad;
}

Matrix2Df downsample2x2_mean(const Matrix2Df& in) {
    const int h = in.rows();
    const int w = in.cols();
    const int h2 = h - (h % 2);
    const int w2 = w - (w % 2);
    const int out_h = std::max(1, h2 / 2);
    const int out_w = std::max(1, w2 / 2);
    Matrix2Df out(out_h, out_w);
    for (int y = 0; y < out_h; ++y) {
        for (int x = 0; x < out_w; ++x) {
            const int sy = y * 2;
            const int sx = x * 2;
            const float a = in(sy, sx);
            const float b = in(sy, sx + 1);
            const float c = in(sy + 1, sx);
            const float d = in(sy + 1, sx + 1);
            out(y, x) = 0.25f * (a + b + c + d);
        }
    }
    return out;
}

// Estimate rotation (deg) between ref and mov using log-polar phase correlation on magnitude spectrum.
float estimate_rotation_logpolar(const cv::Mat& ref, const cv::Mat& mov) {
    cv::Mat ref_dft, mov_dft;
    cv::dft(ref, ref_dft, cv::DFT_COMPLEX_OUTPUT);
    cv::dft(mov, mov_dft, cv::DFT_COMPLEX_OUTPUT);

    std::vector<cv::Mat> planes_ref, planes_mov;
    cv::split(ref_dft, planes_ref);
    cv::split(mov_dft, planes_mov);
    cv::Mat mag_ref, mag_mov;
    cv::magnitude(planes_ref[0], planes_ref[1], mag_ref);
    cv::magnitude(planes_mov[0], planes_mov[1], mag_mov);

    mag_ref += 1.0e-9f;
    mag_mov += 1.0e-9f;

    cv::Mat lp_ref, lp_mov;
    const cv::Point2f center(static_cast<float>(ref.cols) / 2.0f, static_cast<float>(ref.rows) / 2.0f);
    const double M = ref.cols;
    cv::logPolar(mag_ref, lp_ref, center, M, cv::WARP_FILL_OUTLIERS);
    cv::logPolar(mag_mov, lp_mov, center, M, cv::WARP_FILL_OUTLIERS);

    cv::Point2d shift = cv::phaseCorrelate(lp_mov, lp_ref);
    double rotation_deg = -shift.y * 360.0 / static_cast<double>(lp_ref.rows);
    return static_cast<float>(rotation_deg);
}

cv::Mat to_uint8_stretch(const Matrix2Df& src) {
    cv::Mat f(src.rows(), src.cols(), CV_32F, const_cast<float*>(src.data()));
    std::vector<float> vals;
    vals.reserve(static_cast<size_t>(src.size()));
    for (int r = 0; r < f.rows; ++r) {
        const float* p = f.ptr<float>(r);
        vals.insert(vals.end(), p, p + f.cols);
    }
    if (vals.empty()) return cv::Mat();
    const size_t n = vals.size();
    auto nth = [&](size_t k) {
        std::nth_element(vals.begin(), vals.begin() + k, vals.end());
        return vals[k];
    };
    float lo = nth(static_cast<size_t>(0.01 * n));
    float hi = nth(static_cast<size_t>(0.99 * n));
    if (hi <= lo) hi = lo + 1.0f;
    cv::Mat out;
    f.convertTo(out, CV_8U, 255.0 / (hi - lo), -255.0 * lo / (hi - lo));
    return out;
}

RegistrationResult feature_registration_similarity(const Matrix2Df& mov, const Matrix2Df& ref, bool allow_rotation, float max_rot_deg) {
    cv::Mat ref_cv = to_uint8_stretch(ref);
    cv::Mat mov_cv = to_uint8_stretch(mov);

    RegistrationResult res;
    res.warp = identity_warp();
    res.correlation = 0.0f;
    res.success = false;

    if (ref_cv.empty() || mov_cv.empty()) {
        res.error_message = "empty image";
        return res;
    }

    cv::Ptr<cv::AKAZE> akaze = cv::AKAZE::create();
    std::vector<cv::KeyPoint> kps_ref, kps_mov;
    cv::Mat desc_ref, desc_mov;
    akaze->detectAndCompute(ref_cv, cv::noArray(), kps_ref, desc_ref);
    akaze->detectAndCompute(mov_cv, cv::noArray(), kps_mov, desc_mov);

    if (desc_ref.empty() || desc_mov.empty()) {
        res.error_message = "no features";
        return res;
    }

    cv::BFMatcher matcher(cv::NORM_HAMMING, true);
    std::vector<cv::DMatch> matches;
    matcher.match(desc_mov, desc_ref, matches);
    if (matches.size() < 8) {
        res.error_message = "few matches";
        return res;
    }

    std::sort(matches.begin(), matches.end(), [](const cv::DMatch& a, const cv::DMatch& b){ return a.distance < b.distance; });
    const int keep = std::max<size_t>(15, matches.size() * 0.3);
    matches.resize(std::min<size_t>(static_cast<size_t>(keep), matches.size()));

    std::vector<cv::Point2f> pts_mov, pts_ref;
    pts_mov.reserve(matches.size());
    pts_ref.reserve(matches.size());
    for (const auto& m : matches) {
        pts_mov.push_back(kps_mov[m.queryIdx].pt);
        pts_ref.push_back(kps_ref[m.trainIdx].pt);
    }

    cv::Mat inliers;
    cv::Mat A = cv::estimateAffinePartial2D(pts_mov, pts_ref, inliers, cv::RANSAC, 3.0, 2000, 0.99);
    if (A.empty()) {
        res.error_message = "affine fail";
        return res;
    }

    float a00 = static_cast<float>(A.at<double>(0,0));
    float a01 = static_cast<float>(A.at<double>(0,1));
    float a10 = static_cast<float>(A.at<double>(1,0));
    float a11 = static_cast<float>(A.at<double>(1,1));
    float tx  = static_cast<float>(A.at<double>(0,2));
    float ty  = static_cast<float>(A.at<double>(1,2));

    float theta_deg = std::atan2(-a01, a00) * 180.0f / 3.14159265f;
    if (!allow_rotation) {
        a00 = 1.0f; a01 = 0.0f; a10 = 0.0f; a11 = 1.0f;
    } else if (std::fabs(theta_deg) > max_rot_deg) {
        res.error_message = "rotation clamp";
        return res;
    }

    res.warp << a00, a01, tx,
                a10, a11, ty;
    int inl = inliers.empty() ? 0 : cv::countNonZero(inliers);
    res.correlation = matches.empty() ? 0.0f : static_cast<float>(inl) / static_cast<float>(matches.size());
    res.success = res.correlation > 0.1f;
    return res;
}

struct StarPoint {
    float x = 0.0f;
    float y = 0.0f;
    float flux = 0.0f;
};

std::vector<StarPoint> detect_stars_simple(const Matrix2Df& img, int topk) {
    const int h = img.rows();
    const int w = img.cols();
    if (h < 5 || w < 5) return {};

    std::vector<float> pixels;
    pixels.reserve(static_cast<size_t>(img.size()));
    for (int y = 0; y < h; ++y) {
        const float* row = img.data() + static_cast<size_t>(y) * w;
        pixels.insert(pixels.end(), row, row + w);
    }
    float med = median_of(pixels);
    float sigma = robust_sigma_mad(pixels);
    if (sigma < 1.0e-6f) sigma = 1.0f;
    const float thresh = med + 3.5f * sigma;

    std::vector<StarPoint> stars;
    stars.reserve(static_cast<size_t>(topk) * 2);
    for (int y = 1; y < h - 1; ++y) {
        const float* row = img.data() + static_cast<size_t>(y) * w;
        for (int x = 1; x < w - 1; ++x) {
            const float v = row[x];
            if (v < thresh) continue;
            bool is_max = true;
            for (int dy = -1; dy <= 1 && is_max; ++dy) {
                const float* r2 = img.data() + static_cast<size_t>(y + dy) * w;
                for (int dx = -1; dx <= 1; ++dx) {
                    if (dx == 0 && dy == 0) continue;
                    if (r2[x + dx] >= v) { is_max = false; break; }
                }
            }
            if (!is_max) continue;

            float wsum = 0.0f;
            float xs = 0.0f;
            float ys = 0.0f;
            for (int dy = -1; dy <= 1; ++dy) {
                const float* r2 = img.data() + static_cast<size_t>(y + dy) * w;
                for (int dx = -1; dx <= 1; ++dx) {
                    const float val = r2[x + dx] - med;
                    if (val <= 0.0f) continue;
                    wsum += val;
                    xs += (static_cast<float>(x + dx) * val);
                    ys += (static_cast<float>(y + dy) * val);
                }
            }
            if (wsum <= 0.0f) continue;
            StarPoint s;
            s.x = xs / wsum;
            s.y = ys / wsum;
            s.flux = wsum;
            stars.push_back(s);
        }
    }
    std::sort(stars.begin(), stars.end(), [](const StarPoint& a, const StarPoint& b){ return a.flux > b.flux; });
    if (static_cast<int>(stars.size()) > topk) {
        stars.resize(static_cast<size_t>(topk));
    }
    return stars;
}

struct SimilarityResult {
    bool ok = false;
    float scale = 1.0f;
    float theta = 0.0f; // radians
    Eigen::Vector2f t{0.0f, 0.0f};
    int inliers = 0;
    float mean_err = 1.0e9f;
};

SimilarityResult score_similarity(const std::vector<StarPoint>& mov,
                                  const std::vector<StarPoint>& ref,
                                  float scale, float theta,
                                  const Eigen::Vector2f& t,
                                  float inlier_tol_px) {
    if (mov.empty() || ref.empty()) return {};
    const float ct = std::cos(theta);
    const float st = std::sin(theta);
    int inl = 0;
    float err_sum = 0.0f;
    for (const auto& m : mov) {
        const float xr = scale * (ct * m.x - st * m.y) + t.x();
        const float yr = scale * (st * m.x + ct * m.y) + t.y();
        float best = std::numeric_limits<float>::max();
        for (const auto& r : ref) {
            const float dx = xr - r.x;
            const float dy = yr - r.y;
            const float d = std::sqrt(dx*dx + dy*dy);
            if (d < best) best = d;
        }
        if (best < inlier_tol_px) {
            ++inl;
            err_sum += best;
        }
    }
    SimilarityResult res;
    res.ok = inl > 0;
    res.inliers = inl;
    res.mean_err = (inl > 0) ? (err_sum / static_cast<float>(inl)) : res.mean_err;
    res.scale = scale;
    res.theta = theta;
    res.t = t;
    return res;
}

bool similarity_from_pairs(const Eigen::Vector2f& m1, const Eigen::Vector2f& m2,
                           const Eigen::Vector2f& r1, const Eigen::Vector2f& r2,
                           bool allow_rotation,
                           float& scale, float& theta, Eigen::Vector2f& t) {
    Eigen::Vector2f v_m = m2 - m1;
    Eigen::Vector2f v_r = r2 - r1;
    const float len_m = v_m.norm();
    const float len_r = v_r.norm();
    if (len_m < 1.0e-3f || len_r < 1.0e-3f) return false;
    scale = len_r / len_m;
    if (!std::isfinite(scale) || scale <= 0.0f) return false;
    theta = allow_rotation ? (std::atan2(v_r.y(), v_r.x()) - std::atan2(v_m.y(), v_m.x())) : 0.0f;
    const float ct = std::cos(theta);
    const float st = std::sin(theta);
    Eigen::Matrix2f R;
    R << ct, -st,
         st,  ct;
    t = r1 - scale * (R * m1);
    return true;
}

RegistrationResult star_registration_similarity(const Matrix2Df& mov, const Matrix2Df& ref,
                                                bool allow_rotation, float max_rot_deg,
                                                int topk_stars, int min_inliers,
                                                float inlier_tol_px, float dist_bin_px) {
    RegistrationResult res;
    res.warp = identity_warp();
    res.success = false;
    res.correlation = 0.0f;

    auto mov_stars = detect_stars_simple(mov, topk_stars);
    auto ref_stars = detect_stars_simple(ref, topk_stars);
    if (mov_stars.size() < 3 || ref_stars.size() < 3) {
        res.error_message = "too_few_stars";
        return res;
    }

    struct Pair { int i; int j; float dist; };
    std::vector<Pair> ref_pairs;
    std::vector<Pair> mov_pairs;

    auto build_pairs = [&](const std::vector<StarPoint>& stars, std::vector<Pair>& out){
        const int n = static_cast<int>(stars.size());
        out.reserve(static_cast<size_t>(n) * static_cast<size_t>(n) / 2);
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                const float dx = stars[j].x - stars[i].x;
                const float dy = stars[j].y - stars[i].y;
                const float d = std::sqrt(dx*dx + dy*dy);
                if (d > 1.0f) out.push_back({i, j, d});
            }
        }
        std::sort(out.begin(), out.end(), [](const Pair& a, const Pair& b){ return a.dist > b.dist; });
        const size_t limit = std::min<size_t>(out.size(), 800);
        out.resize(limit);
    };

    build_pairs(ref_stars, ref_pairs);
    build_pairs(mov_stars, mov_pairs);
    if (ref_pairs.empty() || mov_pairs.empty()) {
        res.error_message = "no_pairs";
        return res;
    }

    std::unordered_map<int, std::vector<Pair>> ref_bucket;
    ref_bucket.reserve(ref_pairs.size() * 2);
    for (const auto& p : ref_pairs) {
        int key = static_cast<int>(std::round(p.dist / dist_bin_px));
        ref_bucket[key].push_back(p);
    }

    SimilarityResult best;
    int attempts = 0;
    for (const auto& pm : mov_pairs) {
        int key = static_cast<int>(std::round(pm.dist / dist_bin_px));
        for (int dk = -1; dk <= 1; ++dk) {
            auto it = ref_bucket.find(key + dk);
            if (it == ref_bucket.end()) continue;
            for (const auto& pr : it->second) {
                ++attempts;
                float scale = 1.0f;
                float theta = 0.0f;
                Eigen::Vector2f t;
                const Eigen::Vector2f m1(mov_stars[pm.i].x, mov_stars[pm.i].y);
                const Eigen::Vector2f m2(mov_stars[pm.j].x, mov_stars[pm.j].y);
                const Eigen::Vector2f r1(ref_stars[pr.i].x, ref_stars[pr.i].y);
                const Eigen::Vector2f r2(ref_stars[pr.j].x, ref_stars[pr.j].y);
                if (!similarity_from_pairs(m1, m2, r1, r2, allow_rotation, scale, theta, t)) continue;
                const float theta_deg = theta * 180.0f / 3.14159265f;
                if (std::fabs(theta_deg) > max_rot_deg) continue;
                if (!std::isfinite(scale) || scale < 0.5f || scale > 2.0f) continue;

                SimilarityResult sr = score_similarity(mov_stars, ref_stars, scale, theta, t, inlier_tol_px);
                if (!sr.ok) continue;
                if (sr.inliers > best.inliers || (sr.inliers == best.inliers && sr.mean_err < best.mean_err)) {
                    best = sr;
                }
            }
        }
        if (attempts > 4000 && best.inliers >= min_inliers) break;
    }

    if (best.inliers >= min_inliers && best.mean_err < inlier_tol_px * 1.2f) {
        res.success = true;
        res.correlation = static_cast<float>(best.inliers) / static_cast<float>(std::max<size_t>(1, mov_stars.size()));
        const float ct = std::cos(best.theta);
        const float st = std::sin(best.theta);
        res.warp << best.scale * ct, -best.scale * st, best.t.x(),
                    best.scale * st,  best.scale * ct, best.t.y();
    } else {
        res.error_message = "no_consensus";
    }
    return res;
}

WarpMatrix scale_translation_warp(const WarpMatrix& w, float scale) {
    WarpMatrix out = w;
    out(0,2) *= scale;
    out(1,2) *= scale;
    return out;
}

} // namespace

GlobalRegistrationOutput register_frames_to_reference(
    const std::vector<Matrix2Df>& frames_fullres,
    ColorMode mode,
    BayerPattern bayer,
    const config::RegistrationConfig& rcfg,
    const std::vector<FrameMetrics>* frame_metrics_opt,
    const VectorXf* global_weights_opt) {

    GlobalRegistrationOutput out;
    const int n = static_cast<int>(frames_fullres.size());
    out.warps_fullres.assign(static_cast<size_t>(n), identity_warp());
    out.scores.assign(static_cast<size_t>(n), 0.0f);
    out.success.assign(static_cast<size_t>(n), false);
    out.errors.assign(static_cast<size_t>(n), "");
    out.engine_used = rcfg.engine;

    if (n == 0) return out;

    // Reference selection: prefer global weights (if available), then quality score, else middle frame.
    out.ref_idx = n / 2;
    out.ref_selection_method = "middle";
    out.ref_selection_value = 0.0f;

    if (global_weights_opt && global_weights_opt->size() == n) {
        int best = 0;
        float best_v = (*global_weights_opt)[0];
        for (int i = 1; i < n; ++i) {
            float v = (*global_weights_opt)[i];
            if (v > best_v) { best_v = v; best = i; }
        }
        out.ref_idx = best;
        out.ref_selection_method = "global_weight";
        out.ref_selection_value = best_v;
    } else if (frame_metrics_opt && static_cast<int>(frame_metrics_opt->size()) == n) {
        int best = 0;
        float best_v = (*frame_metrics_opt)[0].quality_score;
        for (int i = 1; i < n; ++i) {
            float v = (*frame_metrics_opt)[i].quality_score;
            if (v > best_v) { best_v = v; best = i; }
        }
        out.ref_idx = best;
        out.ref_selection_method = "quality_score";
        out.ref_selection_value = best_v;
    }

    // Build proxy images
    std::vector<Matrix2Df> proxy;
    proxy.reserve(static_cast<size_t>(n));
    const bool is_osc = (mode == ColorMode::OSC);
    for (int i = 0; i < n; ++i) {
        if (is_osc) {
            proxy.push_back(tile_compile::image::cfa_green_proxy_downsample2x2(
                frames_fullres[i], tile_compile::bayer_pattern_to_string(bayer)));
        } else {
            proxy.push_back(downsample2x2_mean(frames_fullres[i]));
        }
    }
    out.downsample_scale = 2.0f;

    const Matrix2Df ref_p = proxy[static_cast<size_t>(out.ref_idx)];

    // Prepare ECC images once for ref
    const Matrix2Df ref_ecc = prepare_ecc_image(ref_p);
    cv::Mat ref_cv(ref_ecc.rows(), ref_ecc.cols(), CV_32F, const_cast<float*>(ref_ecc.data()));

    for (int i = 0; i < n; ++i) {
        if (i == out.ref_idx) {
            out.success[static_cast<size_t>(i)] = true;
            out.scores[static_cast<size_t>(i)] = 1.0f;
            out.warps_fullres[static_cast<size_t>(i)] = identity_warp();
            continue;
        }

        const Matrix2Df mov_p = proxy[static_cast<size_t>(i)];

        RegistrationResult rr;

        if (rcfg.engine == "star_similarity") {
            rr = star_registration_similarity(mov_p, ref_p,
                                              rcfg.allow_rotation, rcfg.max_rotation_deg,
                                              rcfg.star_topk, rcfg.star_min_inliers,
                                              rcfg.star_inlier_tol_px, rcfg.star_dist_bin_px);
        } else if (rcfg.engine == "opencv_feature") {
            rr = feature_registration_similarity(mov_p, ref_p, rcfg.allow_rotation, rcfg.max_rotation_deg);
        } else if (rcfg.engine == "opencv_logpolar") {
            // rotation estimate + ECC refine
            Matrix2Df mov_ecc = prepare_ecc_image(mov_p);
            cv::Mat mov_cv(mov_ecc.rows(), mov_ecc.cols(), CV_32F, const_cast<float*>(mov_ecc.data()));
            float rot = 0.0f;
            if (rcfg.allow_rotation) {
                rot = estimate_rotation_logpolar(ref_cv, mov_cv);
                if (std::fabs(rot) > rcfg.max_rotation_deg) {
                    rot = (rot > 0.0f) ? rcfg.max_rotation_deg : -rcfg.max_rotation_deg;
                }
            }
            const float th = rot * 3.14159265f / 180.0f;
            const float ct = std::cos(th);
            const float st = std::sin(th);
            WarpMatrix init;
            init << ct, -st, 0.0f,
                    st,  ct, 0.0f;
            rr = ecc_warp(mov_ecc, ref_ecc, rcfg.allow_rotation, init, 200, 1e-6f);
        } else {
            // default: phase correlation for init translation + ECC translation refine
            Matrix2Df mov_ecc = prepare_ecc_image(mov_p);
            auto [dx, dy] = phasecorr_translation(mov_ecc, ref_ecc);
            WarpMatrix init = identity_warp();
            init(0,2) = dx;
            init(1,2) = dy;
            rr = ecc_warp(mov_ecc, ref_ecc, false, init, 200, 1e-6f);
        }

        if (!rr.success) {
            out.warps_fullres[static_cast<size_t>(i)] = identity_warp();
            out.scores[static_cast<size_t>(i)] = 0.0f;
            out.success[static_cast<size_t>(i)] = false;
            out.errors[static_cast<size_t>(i)] = rr.error_message;
            continue;
        }

        out.success[static_cast<size_t>(i)] = true;
        out.scores[static_cast<size_t>(i)] = rr.correlation;
        out.warps_fullres[static_cast<size_t>(i)] = scale_translation_warp(rr.warp, out.downsample_scale);
    }

    return out;
}

} // namespace tile_compile::registration
