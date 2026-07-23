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
    } else if (type == "reset") {
        // no params
    } else if (type == "vibrance" || type == "color_temperature") {
        if (check(require("amount", -1.0, 1.0))) return {{"error", err}};
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
