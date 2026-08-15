#include "services/pi/pi_image_ops.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/photo.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace tile_compile::pi {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

double clamp_param(double val, double lo, double hi) {
    if (val < lo) return lo;
    if (val > hi) return hi;
    return val;
}

nlohmann::json validate_op(const nlohmann::json& op) {
    if (!op.is_object() || !op.contains("type") || !op["type"].is_string()) {
        return {{"error", "operation must be an object with string \"type\""}};
    }
    const std::string type = op["type"].get<std::string>();
    const auto& p = op.value("params", nlohmann::json::object());

    auto require = [&](const std::string& key, double lo, double hi) -> std::string {
        if (!p.contains(key) || !p[key].is_number())
            return "missing or non-numeric param: " + key;
        const double v = p[key].get<double>();
        if (v < lo || v > hi)
            return "param " + key + " out of range [" + std::to_string(lo) + "," + std::to_string(hi) + "]";
        return {};
    };

    auto require_int = [&](const std::string& key, int lo, int hi) -> std::string {
        if (!p.contains(key) || !p[key].is_number_integer())
            return "missing or non-integer param: " + key;
        const int v = p[key].get<int>();
        if (v < lo || v > hi)
            return "param " + key + " out of range [" + std::to_string(lo) + "," + std::to_string(hi) + "]";
        return {};
    };

    std::string err;
    auto check = [&](const std::string& e) -> bool {
        if (!e.empty()) { err = e; return true; }
        return false;
    };

    if (type == "brightness") {
        if (check(require("midtones", -1.0, 1.0))) return {{"error", err}};
        if (check(require("shadows", -1.0, 1.0))) return {{"error", err}};
        if (check(require("highlights", -1.0, 1.0))) return {{"error", err}};
    } else if (type == "contrast") {
        if (check(require("amount", -1.0, 1.0))) return {{"error", err}};
    } else if (type == "saturation") {
        if (check(require("amount", -1.0, 1.0))) return {{"error", err}};
    } else if (type == "sharpen") {
        if (check(require("amount", 0.0, 1.0))) return {{"error", err}};
        if (check(require("radius", 0.5, 5.0))) return {{"error", err}};
    } else if (type == "denoise") {
        if (check(require("strength", 0.0, 1.0))) return {{"error", err}};
        if (!p.contains("luminance") || !p["luminance"].is_boolean())
            return {{"error", "missing or non-boolean param: luminance"}};
    } else if (type == "rmgreen") {
        if (check(require("strength", 0.0, 1.0))) return {{"error", err}};
    } else if (type == "clahe") {
        if (check(require("cliplimit", 1.0, 10.0))) return {{"error", err}};
        if (check(require_int("tilesize", 8, 64))) return {{"error", err}};
    } else if (type == "bilateral") {
        if (check(require_int("d", 3, 15))) return {{"error", err}};
        if (check(require("sigma_color", 10.0, 150.0))) return {{"error", err}};
        if (check(require("sigma_space", 10.0, 150.0))) return {{"error", err}};
    } else if (type == "threshold") {
        if (check(require("black_point", 0.0, 1.0))) return {{"error", err}};
        if (check(require("white_point", 0.0, 1.0))) return {{"error", err}};
        if (p["black_point"].get<double>() >= p["white_point"].get<double>())
            return {{"error", "black_point must be < white_point"}};
    } else if (type == "invert") {
        // no params
    } else if (type == "crop") {
        if (check(require_int("x", 0, 100000))) return {{"error", err}};
        if (check(require_int("y", 0, 100000))) return {{"error", err}};
        if (check(require_int("w", 1, 100000))) return {{"error", err}};
        if (check(require_int("h", 1, 100000))) return {{"error", err}};
    } else if (type == "crop_rotated") {
        if (check(require_int("cx", 0, 100000))) return {{"error", err}};
        if (check(require_int("cy", 0, 100000))) return {{"error", err}};
        if (check(require_int("w", 1, 100000))) return {{"error", err}};
        if (check(require_int("h", 1, 100000))) return {{"error", err}};
        if (check(require("angle", -180.0, 180.0))) return {{"error", err}};
    } else if (type == "reset") {
        // no params
    } else if (type == "vibrance" || type == "color_temperature") {
        if (check(require("amount", -1.0, 1.0))) return {{"error", err}};
    } else if (type == "levels") {
        if (check(require("black", 0.0, 1.0))) return {{"error", err}};
        if (check(require("white", 0.0, 1.0))) return {{"error", err}};
        if (check(require("gamma", 0.1, 5.0))) return {{"error", err}};
        if (p["black"].get<double>() >= p["white"].get<double>()) return {{"error", "black must be < white"}};
    } else if (type == "curves") {
        const bool has_master = p.contains("points") && p["points"].is_array() && p["points"].size() >= 2 && p["points"].size() <= 32;
        const auto ch_ok = [&](const char* k) { return p.contains(k) && p[k].is_array() && p[k].size() >= 2 && p[k].size() <= 32; };
        const bool has_any_channel = ch_ok("points_r") || ch_ok("points_g") || ch_ok("points_b");
        if (!has_master && !has_any_channel)
            return {{"error", "curves requires points (2-32) or per-channel points_r/g/b"}};
        if (p.contains("points_r") && !ch_ok("points_r"))
            return {{"error", "points_r requires 2-32 points"}};
        if (p.contains("points_g") && !ch_ok("points_g"))
            return {{"error", "points_g requires 2-32 points"}};
        if (p.contains("points_b") && !ch_ok("points_b"))
            return {{"error", "points_b requires 2-32 points"}};
    } else if (type == "shadow_recovery" || type == "highlight_recovery") {
        if (check(require("strength", 0.0, 1.0))) return {{"error", err}};
    } else if (type == "local_contrast") {
        if (check(require("strength", 0.0, 1.0))) return {{"error", err}};
        if (check(require("radius", 0.5, 10.0))) return {{"error", err}};
    } else if (type == "color_balance") {
        if (check(require("red", -1.0, 1.0))) return {{"error", err}};
        if (check(require("green", -1.0, 1.0))) return {{"error", err}};
        if (check(require("blue", -1.0, 1.0))) return {{"error", err}};
    } else if (type == "chroma_denoise") {
        if (check(require("strength", 0.0, 1.0))) return {{"error", err}};
        if (check(require("protect", 0.0, 1.0))) return {{"error", err}};
        if (p.contains("mode") && !p["mode"].is_string()) return {{"error", "mode must be a string"}};
    } else if (type == "unpurple" || type == "star_desaturation" || type == "dehaze") {
        if (check(require("amount", 0.0, 1.0))) return {{"error", err}};
    } else if (type == "fixbanding") {
        if (check(require("amount", 0.0, 1.0))) return {{"error", err}};
        if (check(require("sigma", 0.5, 5.0))) return {{"error", err}};
    } else {
        return {{"error", "unknown operation type: " + type}};
    }
    return {};
}

// ---------------------------------------------------------------------------
// Operations
// ---------------------------------------------------------------------------

cv::Mat apply_brightness(const cv::Mat& img, double midtones,
                         double shadows, double highlights) {
    midtones = clamp_param(midtones, -1.0, 1.0);
    shadows = clamp_param(shadows, -1.0, 1.0);
    highlights = clamp_param(highlights, -1.0, 1.0);

    // Build a 256-entry LUT with three zones:
    //  shadows (0-63), midtones (64-191), highlights (192-255)
    cv::Mat lut(1, 256, CV_8U);
    for (int i = 0; i < 256; ++i) {
        double v = i / 255.0;
        double s_weight = 0.0, m_weight = 0.0, h_weight = 0.0;
        if (i < 64) {
            s_weight = 1.0 - (i / 64.0);
            m_weight = i / 64.0;
        } else if (i < 192) {
            m_weight = 1.0;
        } else {
            m_weight = 1.0 - ((i - 192) / 63.0);
            h_weight = (i - 192) / 63.0;
        }
        // Gamma-style adjustment: shift = param * weight * 0.5
        double shift = shadows * s_weight * 0.5 + midtones * m_weight * 0.5 + highlights * h_weight * 0.5;
        v = clamp_param(v + shift, 0.0, 1.0);
        lut.at<uchar>(0, i) = static_cast<uchar>(std::round(v * 255.0));
    }

    std::vector<cv::Mat> channels;
    cv::split(img, channels);
    for (auto& ch : channels) {
        cv::LUT(ch, lut, ch);
    }
    cv::Mat out;
    cv::merge(channels, out);
    return out;
}

cv::Mat apply_contrast(const cv::Mat& img, double amount) {
    amount = clamp_param(amount, -1.0, 1.0);
    // Linear stretch from 0.5 center: v' = 0.5 + (v - 0.5) * (1 + amount)
    // amount > 0: pushes values away from center (more contrast)
    // amount < 0: pulls values toward center (less contrast)
    const double scale = 1.0 + amount;

    cv::Mat lut(1, 256, CV_8U);
    for (int i = 0; i < 256; ++i) {
        double v = i / 255.0;
        double vprime = 0.5 + (v - 0.5) * scale;
        lut.at<uchar>(0, i) = static_cast<uchar>(std::round(clamp_param(vprime, 0.0, 1.0) * 255.0));
    }

    std::vector<cv::Mat> channels;
    cv::split(img, channels);
    for (auto& ch : channels) {
        cv::LUT(ch, lut, ch);
    }
    cv::Mat out;
    cv::merge(channels, out);
    return out;
}

cv::Mat apply_levels_float(const cv::Mat& img, double black, double white, double gamma) {
    cv::Mat out = img.clone();
    const float scale = static_cast<float>(1.0 / (white - black));
    for (int y = 0; y < out.rows; ++y) {
        auto* row = out.ptr<cv::Vec3f>(y);
        for (int x = 0; x < out.cols; ++x) for (int c = 0; c < 3; ++c) {
            const double n = clamp_param((row[x][c] - black) * scale, 0.0, 1.0);
            row[x][c] = static_cast<float>(std::pow(n, 1.0 / gamma));
        }
    }
    return out;
}

cv::Mat build_curves_lut(const nlohmann::json& points) {
    std::vector<std::pair<double, double>> p;
    for (const auto& point : points) {
        double x = 0.0, y = 0.0;
        if (point.is_array() && point.size() >= 2) { x = point[0].get<double>(); y = point[1].get<double>(); }
        else if (point.is_object() && point.contains("x") && point.contains("y")) { x = point["x"].get<double>(); y = point["y"].get<double>(); }
        else continue;
        p.emplace_back(clamp_param(x, 0.0, 1.0), clamp_param(y, 0.0, 1.0));
    }
    cv::Mat lut(1, 256, CV_32F);
    lut.setTo(0.0f);
    if (p.size() < 2) {
        for (int i = 0; i < 256; ++i) lut.at<float>(0, i) = static_cast<float>(i) / 255.0f;
        return lut;
    }
    std::sort(p.begin(), p.end());
    for (int i = 0; i < 256; ++i) {
        const double x = i / 255.0;
        size_t j = 1; while (j < p.size() && x > p[j].first) ++j;
        if (j >= p.size()) j = p.size() - 1;
        const auto [x0, y0] = p[j - 1]; const auto [x1, y1] = p[j];
        const double t = x1 > x0 ? (x - x0) / (x1 - x0) : 0.0;
        const double yprev = j > 1 ? p[j - 2].second : y0;
        const double ynext = j + 1 < p.size() ? p[j + 1].second : y1;
        const double t2 = t * t, t3 = t2 * t;
        const double spline = 0.5 * ((2.0 * y0) + (-yprev + y1) * t + (2.0 * yprev - 5.0 * y0 + 4.0 * y1 - ynext) * t2 + (-yprev + 3.0 * y0 - 3.0 * y1 + ynext) * t3);
        lut.at<float>(0, i) = static_cast<float>(clamp_param(spline, 0.0, 1.0));
    }
    return lut;
}

bool valid_curves_points(const nlohmann::json& points) {
    return points.is_array() && points.size() >= 2 && points.size() <= 32;
}

cv::Mat apply_curves_float(const cv::Mat& img, const nlohmann::json& points) {
    cv::Mat lut = build_curves_lut(points);
    cv::Mat out = img.clone();
    for (int y = 0; y < out.rows; ++y) {
        auto* row = out.ptr<cv::Vec3f>(y);
        for (int x = 0; x < out.cols; ++x) for (int c = 0; c < 3; ++c) {
            const int idx = std::clamp(static_cast<int>(row[x][c] * 255.0f), 0, 255);
            row[x][c] = lut.at<float>(0, idx);
        }
    }
    return out;
}

// Per-channel curves: OpenCV uses BGR order (ch0=B, ch1=G, ch2=R).
// points_r → ch2, points_g → ch1, points_b → ch0.
// points (master) is used as fallback for any channel without its own curve.
cv::Mat apply_curves_per_channel_float(const cv::Mat& img,
                                        const nlohmann::json& points,
                                        const nlohmann::json& points_r,
                                        const nlohmann::json& points_g,
                                        const nlohmann::json& points_b) {
    cv::Mat lut_master = build_curves_lut(points);
    // OpenCV BGR: index 0=B, 1=G, 2=R
    cv::Mat luts[3];
    luts[0] = valid_curves_points(points_b) ? build_curves_lut(points_b) : lut_master;
    luts[1] = valid_curves_points(points_g) ? build_curves_lut(points_g) : lut_master;
    luts[2] = valid_curves_points(points_r) ? build_curves_lut(points_r) : lut_master;
    cv::Mat out = img.clone();
    for (int y = 0; y < out.rows; ++y) {
        auto* row = out.ptr<cv::Vec3f>(y);
        for (int x = 0; x < out.cols; ++x) for (int c = 0; c < 3; ++c) {
            const int idx = std::clamp(static_cast<int>(row[x][c] * 255.0f), 0, 255);
            row[x][c] = luts[c].at<float>(0, idx);
        }
    }
    return out;
}

cv::Mat apply_shadow_highlight_float(const cv::Mat& img, double strength, bool shadows) {
    cv::Mat out = img.clone();
    for (int y = 0; y < out.rows; ++y) {
        auto* row = out.ptr<cv::Vec3f>(y);
        for (int x = 0; x < out.cols; ++x) {
            float lum = (row[x][0] + row[x][1] + row[x][2]) / 3.0f;
            float weight = shadows ? (1.0f - lum) * (1.0f - lum) : lum * lum;
            float delta = static_cast<float>(strength) * weight * (shadows ? (1.0f - lum) : -lum);
            for (int c = 0; c < 3; ++c) row[x][c] = static_cast<float>(clamp_param(row[x][c] + delta, 0.0, 1.0));
        }
    }
    return out;
}

cv::Mat apply_color_balance_float(const cv::Mat& img, const nlohmann::json& p) {
    cv::Mat out = img.clone();
    auto value = [&](const std::string& key) { return p.value(key, 0.0); };
    const double base[3] = {value("blue"), value("green"), value("red")};
    const double shadow[3] = {value("shadow_blue"), value("shadow_green"), value("shadow_red")};
    const double highlight[3] = {value("highlight_blue"), value("highlight_green"), value("highlight_red")};
    const double mid[3] = {value("mid_blue"), value("mid_green"), value("mid_red")};
    for (int y = 0; y < out.rows; ++y) { auto* row = out.ptr<cv::Vec3f>(y); for (int x = 0; x < out.cols; ++x) {
        const double lum = (row[x][0] + row[x][1] + row[x][2]) / 3.0;
        const double sw = std::max(0.0, 1.0 - lum * 2.0), hw = std::max(0.0, (lum - 0.5) * 2.0), mw = 1.0 - std::min(1.0, sw + hw);
        for (int c = 0; c < 3; ++c) row[x][c] = static_cast<float>(clamp_param(row[x][c] * (1.0 + base[c] + sw * shadow[c] + mw * mid[c] + hw * highlight[c]), 0.0, 1.0));
    }}
    return out;
}

cv::Mat apply_local_contrast_float(const cv::Mat& img, double strength, double radius) {
    cv::Mat blur, out; cv::GaussianBlur(img, blur, cv::Size(0, 0), radius);
    out = img + static_cast<float>(strength) * (img - blur); cv::min(out, 1.0, out); cv::max(out, 0.0, out); return out;
}

cv::Mat apply_chroma_denoise_float(const cv::Mat& img, double strength, double protect, const std::string& mode) {
    cv::Mat blur; cv::GaussianBlur(img, blur, cv::Size(0, 0), mode == "strong" ? 2.0 : 1.2);
    cv::Mat out = img.clone();
    for (int y = 0; y < out.rows; ++y) { auto* row = out.ptr<cv::Vec3f>(y); const auto* b = blur.ptr<cv::Vec3f>(y); for (int x = 0; x < out.cols; ++x) {
        float lum = (row[x][0] + row[x][1] + row[x][2]) / 3.0f; float factor = static_cast<float>(strength * (1.0 - protect));
        for (int c = 0; c < 3; ++c) row[x][c] = static_cast<float>(clamp_param(lum + (1.0 - factor) * (row[x][c] - lum) + factor * (b[x][c] - lum), 0.0, 1.0));
    }} return out;
}

cv::Mat apply_saturation(const cv::Mat& img, double amount) {
    amount = clamp_param(amount, -1.0, 1.0);
    const double scale = 1.0 + amount;

    cv::Mat hsv;
    cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);

    std::vector<cv::Mat> channels;
    cv::split(hsv, channels);
    // S channel is index 1
    channels[1].convertTo(channels[1], CV_32F, scale);
    // Clamp to [0, 255]
    cv::threshold(channels[1], channels[1], 255.0, 255.0, cv::THRESH_TRUNC);
    cv::threshold(channels[1], channels[1], 0.0, 0.0, cv::THRESH_TOZERO);
    channels[1].convertTo(channels[1], CV_8U);

    cv::merge(channels, hsv);
    cv::Mat out;
    cv::cvtColor(hsv, out, cv::COLOR_HSV2BGR);
    return out;
}

cv::Mat apply_sharpen(const cv::Mat& img, double amount, double radius) {
    amount = clamp_param(amount, 0.0, 1.0);
    radius = clamp_param(radius, 0.5, 5.0);

    cv::Mat blurred;
    cv::GaussianBlur(img, blurred, cv::Size(0, 0), radius);

    cv::Mat out;
    // out = img + amount * (img - blur) = img * (1+amount) + blur * (-amount)
    cv::addWeighted(img, 1.0 + amount, blurred, -amount, 0, out);
    return out;
}

cv::Mat apply_denoise(const cv::Mat& img, double strength, bool luminance) {
    strength = clamp_param(strength, 0.0, 1.0);
    const float h = static_cast<float>(strength * 10.0);

    if (luminance) {
        // Convert to YCrCb, denoise only Y, convert back
        cv::Mat ycrcb;
        cv::cvtColor(img, ycrcb, cv::COLOR_BGR2YCrCb);
        std::vector<cv::Mat> channels;
        cv::split(ycrcb, channels);
        cv::Mat y_denoised;
        cv::fastNlMeansDenoising(channels[0], y_denoised, h, 7, 21);
        channels[0] = y_denoised;
        cv::merge(channels, ycrcb);
        cv::Mat out;
        cv::cvtColor(ycrcb, out, cv::COLOR_YCrCb2BGR);
        return out;
    } else {
        cv::Mat out;
        cv::fastNlMeansDenoisingColored(img, out, h, h, 7, 21);
        return out;
    }
}

cv::Mat apply_rmgreen(const cv::Mat& img, double strength) {
    strength = clamp_param(strength, 0.0, 1.0);

    std::vector<cv::Mat> channels;
    cv::split(img, channels);
    // BGR: channels[0]=B, channels[1]=G, channels[2]=R
    cv::Mat minBR;
    cv::min(channels[0], channels[2], minBR);
    // G_new = G - min(B, R) * strength
    cv::Mat subtracted;
    minBR.convertTo(subtracted, CV_32F, strength);
    cv::Mat g_float;
    channels[1].convertTo(g_float, CV_32F);
    cv::subtract(g_float, subtracted, g_float);
    // Clamp to [0, 255]
    cv::threshold(g_float, g_float, 255.0, 255.0, cv::THRESH_TRUNC);
    cv::threshold(g_float, g_float, 0.0, 0.0, cv::THRESH_TOZERO);
    g_float.convertTo(channels[1], CV_8U);

    cv::Mat out;
    cv::merge(channels, out);
    return out;
}

cv::Mat apply_clahe(const cv::Mat& img, double cliplimit, int tilesize) {
    cliplimit = clamp_param(cliplimit, 1.0, 10.0);
    tilesize = std::clamp(tilesize, 8, 64);

    auto clahe = cv::createCLAHE(cliplimit, cv::Size(tilesize, tilesize));

    std::vector<cv::Mat> channels;
    cv::split(img, channels);
    for (auto& ch : channels) {
        cv::Mat eq;
        clahe->apply(ch, eq);
        ch = eq;
    }
    cv::Mat out;
    cv::merge(channels, out);
    return out;
}

cv::Mat apply_bilateral(const cv::Mat& img, int d,
                        double sigma_color, double sigma_space) {
    d = std::clamp(d, 3, 15);
    sigma_color = clamp_param(sigma_color, 10.0, 150.0);
    sigma_space = clamp_param(sigma_space, 10.0, 150.0);

    cv::Mat out;
    cv::bilateralFilter(img, out, d, sigma_color, sigma_space);
    return out;
}

cv::Mat apply_threshold(const cv::Mat& img, double black_point, double white_point) {
    black_point = clamp_param(black_point, 0.0, 1.0);
    white_point = clamp_param(white_point, 0.0, 1.0);
    if (black_point >= white_point) white_point = black_point + 0.001;

    const int bp = static_cast<int>(black_point * 255.0);
    const int wp = static_cast<int>(white_point * 255.0);

    cv::Mat lut(1, 256, CV_8U);
    for (int i = 0; i < 256; ++i) {
        if (i <= bp) {
            lut.at<uchar>(0, i) = 0;
        } else if (i >= wp) {
            lut.at<uchar>(0, i) = 255;
        } else {
            const double v = static_cast<double>(i - bp) / static_cast<double>(wp - bp);
            lut.at<uchar>(0, i) = static_cast<uchar>(std::round(v * 255.0));
        }
    }

    std::vector<cv::Mat> channels;
    cv::split(img, channels);
    for (auto& ch : channels) {
        cv::LUT(ch, lut, ch);
    }
    cv::Mat out;
    cv::merge(channels, out);
    return out;
}

cv::Mat apply_invert(const cv::Mat& img) {
    cv::Mat out;
    cv::bitwise_not(img, out);
    return out;
}

cv::Mat apply_crop(const cv::Mat& img, int x, int y, int w, int h) {
    x = std::max(0, x);
    y = std::max(0, y);
    w = std::min(w, img.cols - x);
    h = std::min(h, img.rows - y);
    if (w <= 0 || h <= 0) {
        return img.clone(); // invalid crop returns original
    }
    return img(cv::Rect(x, y, w, h)).clone();
}

cv::Mat apply_crop_rotated(const cv::Mat& img, int cx, int cy,
                           int w, int h, double angle_deg) {
    cx = std::clamp(cx, 0, img.cols);
    cy = std::clamp(cy, 0, img.rows);
    w = std::clamp(w, 1, img.cols);
    h = std::clamp(h, 1, img.rows);
    if (w <= 0 || h <= 0) return img.clone();

    // Build the same 2x3 affine matrix as getRotationMatrix2D directly.
    // This keeps the rotated-crop operation portable across OpenCV versions
    // where the imgproc convenience declaration is not exposed by the headers.
    const double radians = angle_deg * CV_PI / 180.0;
    const double alpha = std::cos(radians);
    const double beta = std::sin(radians);
    const cv::Mat rot = (cv::Mat_<double>(2, 3) <<
        alpha, beta, (1.0 - alpha) * static_cast<double>(cx) - beta * static_cast<double>(cy),
        -beta, alpha, beta * static_cast<double>(cx) + (1.0 - alpha) * static_cast<double>(cy));

    // Warp the full image so the crop region becomes axis-aligned.
    cv::Mat warped;
    cv::warpAffine(img, warped, rot, img.size(), cv::INTER_LINEAR,
                   cv::BORDER_REFLECT_101);

    // After rotation around (cx, cy), the crop center maps to (cx, cy).
    // Extract the axis-aligned w×h region centered there.
    int x0 = cx - w / 2;
    int y0 = cy - h / 2;
    x0 = std::max(0, std::min(x0, warped.cols - w));
    y0 = std::max(0, std::min(y0, warped.rows - h));
    w = std::min(w, warped.cols - x0);
    h = std::min(h, warped.rows - y0);
    if (w <= 0 || h <= 0) return img.clone();
    return warped(cv::Rect(x0, y0, w, h)).clone();
}

cv::Mat apply_vibrance(const cv::Mat& img, double amount) {
    amount = clamp_param(amount, -1.0, 1.0);
    cv::Mat hsv;
    cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);
    std::vector<cv::Mat> c;
    cv::split(hsv, c);
    c[1].convertTo(c[1], CV_32F, 1.0 / 255.0);
    for (int y = 0; y < c[1].rows; ++y) {
        float* row = c[1].ptr<float>(y);
        for (int x = 0; x < c[1].cols; ++x) {
            const double s = row[x];
            row[x] = static_cast<float>(clamp_param(s + (amount >= 0 ? amount * (1.0 - s) : amount * s), 0.0, 1.0));
        }
    }
    c[1].convertTo(c[1], CV_8U, 255.0);
    cv::merge(c, hsv);
    cv::Mat out;
    cv::cvtColor(hsv, out, cv::COLOR_HSV2BGR);
    return out;
}

cv::Mat apply_color_temperature(const cv::Mat& img, double amount) {
    amount = clamp_param(amount, -1.0, 1.0);
    cv::Mat f;
    img.convertTo(f, CV_32F, 1.0 / 255.0);
    std::vector<cv::Mat> c;
    cv::split(f, c);
    c[0] -= static_cast<float>(0.12 * amount);
    c[2] += static_cast<float>(0.12 * amount);
    for (auto& ch : c) cv::min(cv::max(ch, 0.0f), 1.0f, ch);
    cv::merge(c, f);
    cv::Mat out;
    f.convertTo(out, CV_8U, 255.0);
    return out;
}

cv::Mat apply_unpurple(const cv::Mat& img, double amount) {
    amount = clamp_param(amount, 0.0, 1.0);
    cv::Mat hsv;
    cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);
    std::vector<cv::Mat> c;
    cv::split(hsv, c);
    for (int y = 0; y < hsv.rows; ++y) {
        for (int x = 0; x < hsv.cols; ++x) {
            const int h = c[0].at<uchar>(y, x);
            if (h >= 125 && h <= 165) {
                c[1].at<uchar>(y, x) = static_cast<uchar>(c[1].at<uchar>(y, x) * (1.0 - amount));
            }
        }
    }
    cv::merge(c, hsv);
    cv::Mat out;
    cv::cvtColor(hsv, out, cv::COLOR_HSV2BGR);
    return out;
}

cv::Mat apply_fixbanding(const cv::Mat& img, double amount, double sigma) {
    amount = clamp_param(amount, 0.0, 1.0);
    sigma = clamp_param(sigma, 0.5, 5.0);
    cv::Mat f;
    img.convertTo(f, CV_32F, 1.0 / 255.0);
    cv::Mat smooth;
    cv::GaussianBlur(f, smooth, cv::Size(0, 0), sigma);
    cv::Mat out = f + (f - smooth) * static_cast<float>(-amount);
    cv::min(cv::max(out, 0.0f), 1.0f, out);
    cv::Mat u8;
    out.convertTo(u8, CV_8U, 255.0);
    return u8;
}

cv::Mat apply_star_desaturation(const cv::Mat& img, double amount) {
    amount = clamp_param(amount, 0.0, 1.0);
    cv::Mat hsv;
    cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);
    std::vector<cv::Mat> c;
    cv::split(hsv, c);
    for (int y = 0; y < hsv.rows; ++y) {
        for (int x = 0; x < hsv.cols; ++x) {
            const double v = c[2].at<uchar>(y, x) / 255.0;
            const double factor = 1.0 - amount * std::max(0.0, (v - 0.7) / 0.3);
            c[1].at<uchar>(y, x) = static_cast<uchar>(c[1].at<uchar>(y, x) * factor);
        }
    }
    cv::merge(c, hsv);
    cv::Mat out;
    cv::cvtColor(hsv, out, cv::COLOR_HSV2BGR);
    return out;
}

cv::Mat apply_dehaze(const cv::Mat& img, double amount) {
    amount = clamp_param(amount, 0.0, 1.0);
    cv::Mat f;
    img.convertTo(f, CV_32F, 1.0 / 255.0);
    cv::Mat base;
    cv::GaussianBlur(f, base, cv::Size(0, 0), 15.0);
    cv::Mat out = f + (f - base) * static_cast<float>(amount);
    cv::min(cv::max(out, 0.0f), 1.0f, out);
    cv::Mat u8;
    out.convertTo(u8, CV_8U, 255.0);
    return u8;
}

static cv::Mat apply_vibrance_fits(const cv::Mat& img, double amount) {
    if (img.channels() == 1) return img.clone();
    cv::Mat hsv;
    cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);
    std::vector<cv::Mat> c;
    cv::split(hsv, c);
    for (int y = 0; y < c[1].rows; ++y) {
        float* row = c[1].ptr<float>(y);
        for (int x = 0; x < c[1].cols; ++x) {
            const double s = row[x];
            row[x] = static_cast<float>(clamp_param(s + (amount >= 0 ? amount * (1.0 - s) : amount * s), 0.0, 1.0));
        }
    }
    cv::merge(c, hsv);
    cv::Mat out;
    cv::cvtColor(hsv, out, cv::COLOR_HSV2BGR);
    return out;
}

static cv::Mat apply_color_temperature_fits(const cv::Mat& img, double amount) {
    std::vector<cv::Mat> c;
    cv::split(img, c);
    c[0] -= static_cast<float>(0.12 * amount);
    c[2] += static_cast<float>(0.12 * amount);
    for (auto& ch : c) cv::min(cv::max(ch, 0.0f), 1.0f, ch);
    cv::Mat out;
    cv::merge(c, out);
    return out;
}

static cv::Mat apply_unpurple_fits(const cv::Mat& img, double amount) {
    if (img.channels() == 1) return img.clone();
    cv::Mat hsv;
    cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);
    std::vector<cv::Mat> c;
    cv::split(hsv, c);
    for (int y = 0; y < hsv.rows; ++y) {
        for (int x = 0; x < hsv.cols; ++x) {
            const float h = c[0].at<float>(y, x);
            if (h >= 250.0f && h <= 330.0f)
                c[1].at<float>(y, x) = static_cast<float>(c[1].at<float>(y, x) * (1.0 - amount));
        }
    }
    cv::merge(c, hsv);
    cv::Mat out;
    cv::cvtColor(hsv, out, cv::COLOR_HSV2BGR);
    return out;
}

static cv::Mat apply_fixbanding_fits(const cv::Mat& img, double amount, double sigma) {
    cv::Mat smooth;
    cv::GaussianBlur(img, smooth, cv::Size(0, 0), sigma);
    cv::Mat out = img + (smooth - img) * static_cast<float>(amount);
    cv::min(cv::max(out, 0.0f), 1.0f, out);
    return out;
}

static cv::Mat apply_star_desaturation_fits(const cv::Mat& img, double amount) {
    if (img.channels() == 1) return img.clone();
    cv::Mat hsv;
    cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);
    std::vector<cv::Mat> c;
    cv::split(hsv, c);
    for (int y = 0; y < hsv.rows; ++y) {
        for (int x = 0; x < hsv.cols; ++x) {
            const double v = c[2].at<float>(y, x);
            const double factor = 1.0 - amount * std::max(0.0, (v - 0.7) / 0.3);
            c[1].at<float>(y, x) = static_cast<float>(c[1].at<float>(y, x) * factor);
        }
    }
    cv::merge(c, hsv);
    cv::Mat out;
    cv::cvtColor(hsv, out, cv::COLOR_HSV2BGR);
    return out;
}

static cv::Mat apply_dehaze_fits(const cv::Mat& img, double amount) {
    cv::Mat base;
    cv::GaussianBlur(img, base, cv::Size(0, 0), 15.0);
    cv::Mat out = img + (img - base) * static_cast<float>(amount);
    cv::min(cv::max(out, 0.0f), 1.0f, out);
    return out;
}

// ---------------------------------------------------------------------------
// Float operations for linear [0,1] data (CV_32F, BGR or 1-channel)
// ---------------------------------------------------------------------------

cv::Mat apply_brightness_fits(const cv::Mat& img, double midtones,
                              double shadows, double highlights) {
    midtones = clamp_param(midtones, -1.0, 1.0);
    shadows = clamp_param(shadows, -1.0, 1.0);
    highlights = clamp_param(highlights, -1.0, 1.0);

    cv::Mat out = img.clone();
    const int channels = out.channels();
    const size_t total = out.total();
    float* data = reinterpret_cast<float*>(out.data);
    for (size_t i = 0; i < total * channels; ++i) {
        float v = data[i];
        double s_weight = 0.0, m_weight = 0.0, h_weight = 0.0;
        if (v < 0.25f) {
            s_weight = 1.0 - (v / 0.25);
            m_weight = v / 0.25;
        } else if (v < 0.75f) {
            m_weight = 1.0;
        } else {
            m_weight = 1.0 - ((v - 0.75) / 0.25);
            h_weight = (v - 0.75) / 0.25;
        }
        double shift = shadows * s_weight * 0.5 + midtones * m_weight * 0.5 + highlights * h_weight * 0.5;
        data[i] = static_cast<float>(clamp_param(v + shift, 0.0, 1.0));
    }
    return out;
}

cv::Mat apply_contrast_fits(const cv::Mat& img, double amount) {
    amount = clamp_param(amount, -1.0, 1.0);
    const double scale = 1.0 + amount;

    cv::Mat out = img.clone();
    float* data = reinterpret_cast<float*>(out.data);
    const size_t total = out.total() * out.channels();
    for (size_t i = 0; i < total; ++i) {
        double v = data[i];
        double vprime = 0.5 + (v - 0.5) * scale;
        data[i] = static_cast<float>(clamp_param(vprime, 0.0, 1.0));
    }
    return out;
}

cv::Mat apply_saturation_fits(const cv::Mat& img, double amount) {
    amount = clamp_param(amount, -1.0, 1.0);
    const double scale = 1.0 + amount;

    if (img.channels() == 1) return img.clone();

    cv::Mat hsv;
    cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);
    std::vector<cv::Mat> channels;
    cv::split(hsv, channels);
    channels[1].convertTo(channels[1], CV_32F, scale);
    cv::threshold(channels[1], channels[1], 1.0, 1.0, cv::THRESH_TRUNC);
    cv::threshold(channels[1], channels[1], 0.0, 0.0, cv::THRESH_TOZERO);
    cv::merge(channels, hsv);
    cv::Mat out;
    cv::cvtColor(hsv, out, cv::COLOR_HSV2BGR);
    return out;
}

cv::Mat apply_sharpen_fits(const cv::Mat& img, double amount, double radius) {
    amount = clamp_param(amount, 0.0, 1.0);
    radius = clamp_param(radius, 0.5, 5.0);

    cv::Mat blurred;
    cv::GaussianBlur(img, blurred, cv::Size(0, 0), radius);

    cv::Mat out;
    cv::addWeighted(img, 1.0 + amount, blurred, -amount, 0, out);
    return out;
}

cv::Mat apply_denoise_fits(const cv::Mat& img, double strength, bool luminance) {
    strength = clamp_param(strength, 0.0, 1.0);
    const float h = static_cast<float>(strength * 10.0);

    // fastNlMeansDenoising operates on 8-bit; convert float [0,1] to 8-bit and back.
    cv::Mat u8;
    img.convertTo(u8, CV_8U, 255.0);

    cv::Mat denoised_u8;
    if (luminance) {
        cv::Mat ycrcb;
        cv::cvtColor(u8, ycrcb, cv::COLOR_BGR2YCrCb);
        std::vector<cv::Mat> channels;
        cv::split(ycrcb, channels);
        cv::Mat y_denoised;
        cv::fastNlMeansDenoising(channels[0], y_denoised, h, 7, 21);
        channels[0] = y_denoised;
        cv::merge(channels, ycrcb);
        cv::cvtColor(ycrcb, denoised_u8, cv::COLOR_YCrCb2BGR);
    } else {
        if (img.channels() == 1) {
            cv::fastNlMeansDenoising(u8, denoised_u8, h, 7, 21);
        } else {
            cv::fastNlMeansDenoisingColored(u8, denoised_u8, h, h, 7, 21);
        }
    }

    cv::Mat out;
    denoised_u8.convertTo(out, CV_32F, 1.0 / 255.0);
    return out;
}

cv::Mat apply_rmgreen_fits(const cv::Mat& img, double strength) {
    strength = clamp_param(strength, 0.0, 1.0);

    std::vector<cv::Mat> channels;
    cv::split(img, channels);
    if (channels.size() >= 3) {
        cv::Mat minBR;
        cv::min(channels[0], channels[2], minBR);
        cv::Mat subtracted;
        minBR.convertTo(subtracted, CV_32F, strength);
        cv::Mat g_float;
        channels[1].convertTo(g_float, CV_32F);
        cv::subtract(g_float, subtracted, g_float);
        cv::threshold(g_float, g_float, 1.0, 1.0, cv::THRESH_TRUNC);
        cv::threshold(g_float, g_float, 0.0, 0.0, cv::THRESH_TOZERO);
        g_float.convertTo(channels[1], CV_32F);
    }
    cv::Mat out;
    cv::merge(channels, out);
    return out;
}

cv::Mat apply_clahe_fits(const cv::Mat& img, double cliplimit, int tilesize) {
    cliplimit = clamp_param(cliplimit, 1.0, 10.0);
    tilesize = std::clamp(tilesize, 8, 64);

    // CLAHE supports 8U and 16U. Use 16U for better precision.
    cv::Mat u16;
    img.convertTo(u16, CV_16U, 65535.0);

    auto clahe = cv::createCLAHE(cliplimit, cv::Size(tilesize, tilesize));
    std::vector<cv::Mat> channels;
    cv::split(u16, channels);
    for (auto& ch : channels) {
        cv::Mat eq;
        clahe->apply(ch, eq);
        ch = eq;
    }
    cv::Mat eq_u16;
    cv::merge(channels, eq_u16);

    cv::Mat out;
    eq_u16.convertTo(out, CV_32F, 1.0 / 65535.0);
    return out;
}

cv::Mat apply_bilateral_fits(const cv::Mat& img, int d,
                             double sigma_color, double sigma_space) {
    d = std::clamp(d, 3, 15);
    sigma_color = clamp_param(sigma_color, 10.0, 150.0);
    sigma_space = clamp_param(sigma_space, 10.0, 150.0);

    cv::Mat out;
    cv::bilateralFilter(img, out, d, sigma_color, sigma_space);
    return out;
}

cv::Mat apply_threshold_fits(const cv::Mat& img, double black_point, double white_point) {
    black_point = clamp_param(black_point, 0.0, 1.0);
    white_point = clamp_param(white_point, 0.0, 1.0);
    if (black_point >= white_point) white_point = black_point + 0.001;

    cv::Mat out = img.clone();
    float* data = reinterpret_cast<float*>(out.data);
    const size_t total = out.total() * out.channels();
    for (size_t i = 0; i < total; ++i) {
        float v = data[i];
        if (v <= black_point) {
            data[i] = 0.0f;
        } else if (v >= white_point) {
            data[i] = 1.0f;
        } else {
            data[i] = static_cast<float>((v - black_point) / (white_point - black_point));
        }
    }
    return out;
}

cv::Mat apply_invert_fits(const cv::Mat& img) {
    cv::Mat out = img.clone();
    float* data = reinterpret_cast<float*>(out.data);
    const size_t total = out.total() * out.channels();
    for (size_t i = 0; i < total; ++i) {
        data[i] = 1.0f - data[i];
    }
    return out;
}

cv::Mat apply_crop_fits(const cv::Mat& img, int x, int y, int w, int h) {
    return apply_crop(img, x, y, w, h);
}

ImageOpResult apply_image_op_fits(const cv::Mat& input, const nlohmann::json& op) {
    ImageOpResult result;

    const auto validation = validate_op(op);
    if (!validation.empty()) {
        result.error = validation["error"].get<std::string>();
        return result;
    }

    const std::string type = op["type"].get<std::string>();
    const auto& p = op.value("params", nlohmann::json::object());

    try {
        if (type == "brightness") {
            result.image = apply_brightness_fits(input,
                p["midtones"].get<double>(),
                p["shadows"].get<double>(),
                p["highlights"].get<double>());
        } else if (type == "contrast") {
            result.image = apply_contrast_fits(input, p["amount"].get<double>());
        } else if (type == "saturation") {
            result.image = apply_saturation_fits(input, p["amount"].get<double>());
        } else if (type == "sharpen") {
            result.image = apply_sharpen_fits(input,
                p["amount"].get<double>(),
                p["radius"].get<double>());
        } else if (type == "denoise") {
            result.image = apply_denoise_fits(input,
                p["strength"].get<double>(),
                p["luminance"].get<bool>());
        } else if (type == "rmgreen") {
            result.image = apply_rmgreen_fits(input, p["strength"].get<double>());
        } else if (type == "clahe") {
            result.image = apply_clahe_fits(input,
                p["cliplimit"].get<double>(),
                p["tilesize"].get<int>());
        } else if (type == "bilateral") {
            result.image = apply_bilateral_fits(input,
                p["d"].get<int>(),
                p["sigma_color"].get<double>(),
                p["sigma_space"].get<double>());
        } else if (type == "threshold") {
            result.image = apply_threshold_fits(input,
                p["black_point"].get<double>(),
                p["white_point"].get<double>());
        } else if (type == "invert") {
            result.image = apply_invert_fits(input);
        } else if (type == "crop") {
            result.image = apply_crop_fits(input,
                p["x"].get<int>(), p["y"].get<int>(),
                p["w"].get<int>(), p["h"].get<int>());
        } else if (type == "crop_rotated") {
            result.image = apply_crop_rotated(input,
                p["cx"].get<int>(), p["cy"].get<int>(),
                p["w"].get<int>(), p["h"].get<int>(),
                p["angle"].get<double>());
        } else if (type == "reset") {
            result.image = input.clone();
        } else if (type == "vibrance") {
            result.image = apply_vibrance_fits(input, p["amount"].get<double>());
        } else if (type == "color_temperature") {
            result.image = apply_color_temperature_fits(input, p["amount"].get<double>());
        } else if (type == "unpurple") {
            result.image = apply_unpurple_fits(input, p["amount"].get<double>());
        } else if (type == "fixbanding") {
            result.image = apply_fixbanding_fits(input, p["amount"].get<double>(), p["sigma"].get<double>());
        } else if (type == "star_desaturation") {
            result.image = apply_star_desaturation_fits(input, p["amount"].get<double>());
        } else if (type == "dehaze") {
            result.image = apply_dehaze_fits(input, p["amount"].get<double>());
        } else if (type == "levels") {
            result.image = apply_levels_float(input, p["black"].get<double>(), p["white"].get<double>(), p["gamma"].get<double>());
        } else if (type == "curves") {
            const auto& pm = p.value("points", nlohmann::json::array());
            const auto& pr = p.value("points_r", nlohmann::json::array());
            const auto& pg = p.value("points_g", nlohmann::json::array());
            const auto& pb = p.value("points_b", nlohmann::json::array());
            if (valid_curves_points(pr) || valid_curves_points(pg) || valid_curves_points(pb))
                result.image = apply_curves_per_channel_float(input, pm, pr, pg, pb);
            else
                result.image = apply_curves_float(input, pm);
        } else if (type == "shadow_recovery") {
            result.image = apply_shadow_highlight_float(input, p["strength"].get<double>(), true);
        } else if (type == "highlight_recovery") {
            result.image = apply_shadow_highlight_float(input, p["strength"].get<double>(), false);
        } else if (type == "color_balance") {
            result.image = apply_color_balance_float(input, p);
        } else if (type == "local_contrast") {
            result.image = apply_local_contrast_float(input, p["strength"].get<double>(), p["radius"].get<double>());
        } else if (type == "chroma_denoise") {
            result.image = apply_chroma_denoise_float(input, p["strength"].get<double>(), p["protect"].get<double>(), p.value("mode", "soft"));
        } else {
            result.error = "unknown operation type: " + type;
            return result;
        }

        result.success = !result.image.empty();
        if (!result.success) {
            result.error = "operation produced empty image";
        }
    } catch (const std::exception& e) {
        result.error = e.what();
    }
    return result;
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

ImageOpResult apply_image_op(const cv::Mat& input, const nlohmann::json& op) {
    ImageOpResult result;

    const auto validation = validate_op(op);
    if (!validation.empty()) {
        result.error = validation["error"].get<std::string>();
        return result;
    }

    const std::string type = op["type"].get<std::string>();
    const auto& p = op.value("params", nlohmann::json::object());

    try {
        if (type == "brightness") {
            result.image = apply_brightness(input,
                p["midtones"].get<double>(),
                p["shadows"].get<double>(),
                p["highlights"].get<double>());
        } else if (type == "contrast") {
            result.image = apply_contrast(input, p["amount"].get<double>());
        } else if (type == "saturation") {
            result.image = apply_saturation(input, p["amount"].get<double>());
        } else if (type == "sharpen") {
            result.image = apply_sharpen(input,
                p["amount"].get<double>(),
                p["radius"].get<double>());
        } else if (type == "denoise") {
            result.image = apply_denoise(input,
                p["strength"].get<double>(),
                p["luminance"].get<bool>());
        } else if (type == "rmgreen") {
            result.image = apply_rmgreen(input, p["strength"].get<double>());
        } else if (type == "clahe") {
            result.image = apply_clahe(input,
                p["cliplimit"].get<double>(),
                p["tilesize"].get<int>());
        } else if (type == "bilateral") {
            result.image = apply_bilateral(input,
                p["d"].get<int>(),
                p["sigma_color"].get<double>(),
                p["sigma_space"].get<double>());
        } else if (type == "threshold") {
            result.image = apply_threshold(input,
                p["black_point"].get<double>(),
                p["white_point"].get<double>());
        } else if (type == "invert") {
            result.image = apply_invert(input);
        } else if (type == "crop") {
            result.image = apply_crop(input,
                p["x"].get<int>(), p["y"].get<int>(),
                p["w"].get<int>(), p["h"].get<int>());
        } else if (type == "crop_rotated") {
            result.image = apply_crop_rotated(input,
                p["cx"].get<int>(), p["cy"].get<int>(),
                p["w"].get<int>(), p["h"].get<int>(),
                p["angle"].get<double>());
        } else if (type == "reset") {
            result.image = input.clone();
        } else if (type == "vibrance") {
            result.image = apply_vibrance(input, p["amount"].get<double>());
        } else if (type == "color_temperature") {
            result.image = apply_color_temperature(input, p["amount"].get<double>());
        } else if (type == "unpurple") {
            result.image = apply_unpurple(input, p["amount"].get<double>());
        } else if (type == "fixbanding") {
            result.image = apply_fixbanding(input, p["amount"].get<double>(), p["sigma"].get<double>());
        } else if (type == "star_desaturation") {
            result.image = apply_star_desaturation(input, p["amount"].get<double>());
        } else if (type == "dehaze") {
            result.image = apply_dehaze(input, p["amount"].get<double>());
        } else if (type == "levels") {
            result.image = apply_levels_float(input, p["black"].get<double>(), p["white"].get<double>(), p["gamma"].get<double>());
        } else if (type == "curves") {
            const auto& pm = p.value("points", nlohmann::json::array());
            const auto& pr = p.value("points_r", nlohmann::json::array());
            const auto& pg = p.value("points_g", nlohmann::json::array());
            const auto& pb = p.value("points_b", nlohmann::json::array());
            if (valid_curves_points(pr) || valid_curves_points(pg) || valid_curves_points(pb))
                result.image = apply_curves_per_channel_float(input, pm, pr, pg, pb);
            else
                result.image = apply_curves_float(input, pm);
        } else if (type == "shadow_recovery") {
            result.image = apply_shadow_highlight_float(input, p["strength"].get<double>(), true);
        } else if (type == "highlight_recovery") {
            result.image = apply_shadow_highlight_float(input, p["strength"].get<double>(), false);
        } else if (type == "color_balance") {
            result.image = apply_color_balance_float(input, p);
        } else if (type == "local_contrast") {
            result.image = apply_local_contrast_float(input, p["strength"].get<double>(), p["radius"].get<double>());
        } else if (type == "chroma_denoise") {
            result.image = apply_chroma_denoise_float(input, p["strength"].get<double>(), p["protect"].get<double>(), p.value("mode", "soft"));
        } else {
            result.error = "unknown operation type: " + type;
            return result;
        }

        result.success = !result.image.empty();
        if (!result.success) {
            result.error = "operation produced empty image";
        }
    } catch (const std::exception& e) {
        result.error = e.what();
    }
    return result;
}

// ---------------------------------------------------------------------------
// Inverse operation for undo
// ---------------------------------------------------------------------------

nlohmann::json invert_op(const nlohmann::json& op) {
    if (!op.is_object() || !op.contains("type")) return {{"type", "noop"}};
    const std::string type = op["type"].get<std::string>();

    if (type == "brightness") {
        auto inv = op;
        auto& p = inv["params"];
        p["midtones"] = -p["midtones"].get<double>();
        p["shadows"] = -p["shadows"].get<double>();
        p["highlights"] = -p["highlights"].get<double>();
        return inv;
    }
    if (type == "contrast" || type == "saturation" ||
        type == "vibrance" || type == "color_temperature") {
        auto inv = op;
        inv["params"]["amount"] = -op["params"]["amount"].get<double>();
        return inv;
    }
    // clahe, bilateral, denoise, threshold, crop, invert, reset:
    // not invertible by parameter negation -> snapshot-based undo
    return {{"type", "noop"}};
}

} // namespace tile_compile::pi
