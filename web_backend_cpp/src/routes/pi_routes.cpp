#include "routes/pi_routes.hpp"

#include "app_state.hpp"
#include "routes/route_utils.hpp"
#include "services/ai_service.hpp"
#include "services/pi/pi_assistant.hpp"
#include "services/pi/pi_ai_request_builder.hpp"
#include "services/pi/pi_context_builder.hpp"
#include "services/pi/pi_action_validator.hpp"
#include "services/pi/pi_memory_store.hpp"
#include "services/pi/pi_storage_paths.hpp"
#include "services/pi/pi_tool_registry.hpp"
#include "services/pi/pi_image_ops.hpp"
#include "services/pi/pi_live_image_session.hpp"
#include "services/run_inspector.hpp"
#include "subprocess_manager.hpp"
#include "time_utils.hpp"

#include <fitsio.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <regex>
#include <functional>
#include <iomanip>
#include <limits>
#include <set>
#include <sstream>
#include <yaml-cpp/yaml.h>

using namespace tile_compile::routes;
namespace fs = std::filesystem;

namespace {

std::optional<nlohmann::json> parse_body(const crow::request& req) {
    if (req.body.empty()) return nlohmann::json::object();
    auto parsed = nlohmann::json::parse(req.body, nullptr, false);
    if (parsed.is_discarded()) return std::nullopt;
    if (!parsed.is_object()) return nlohmann::json::object();
    return parsed;
}

int int_query_param(const crow::request& req, const char* name, int fallback) {
    const char* raw = req.url_params.get(name);
    if (!raw) return fallback;
    try {
        return std::stoi(raw);
    } catch (...) {
    }
    return fallback;
}

long count_jsonl_records(const fs::path& path) {
    std::ifstream in(path);
    if (!in) return 0;
    long count = 0;
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty()) ++count;
    }
    return count;
}

std::string base64_encode(const std::vector<unsigned char>& bytes) {
    static constexpr char alphabet[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve(((bytes.size() + 2) / 3) * 4);
    int val = 0;
    int valb = -6;
    for (unsigned char c : bytes) {
        val = (val << 8) + c;
        valb += 8;
        while (valb >= 0) {
            out.push_back(alphabet[(val >> valb) & 0x3F]);
            valb -= 6;
        }
    }
    if (valb > -6) out.push_back(alphabet[((val << 8) >> (valb + 8)) & 0x3F]);
    while (out.size() % 4) out.push_back('=');
    return out;
}

std::string fits_status_text(int status) {
    char text[FLEN_STATUS]{};
    fits_get_errstatus(status, text);
    return text;
}

nlohmann::json evaluate_memory_outcome_payload(const nlohmann::json& body) {
    nlohmann::json outcome = body.contains("outcome") && body["outcome"].is_object()
        ? body["outcome"]
        : nlohmann::json::object();
    const std::string user_result = body.value("result", std::string());
    const std::string feedback = body.value("feedback", std::string());
    const nlohmann::json before = body.contains("before") && body["before"].is_object()
        ? body["before"]
        : nlohmann::json::object();
    const nlohmann::json after = body.contains("after") && body["after"].is_object()
        ? body["after"]
        : nlohmann::json::object();

    auto number_or_nan = [](const nlohmann::json& object, const char* key) {
        if (!object.is_object() || !object.contains(key) || !object[key].is_number()) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        return object[key].get<double>();
    };

    // --- Outcome-Delta: Vorher/Nachher fuer alle relevanten Metriken ---
    // quality_score (globaler Composite-Score)
    const double before_score = number_or_nan(before, "quality_score");
    const double after_score = number_or_nan(after, "quality_score");
    double delta = std::numeric_limits<double>::quiet_NaN();
    if (std::isfinite(before_score) && std::isfinite(after_score)) {
        delta = after_score - before_score;
        outcome["quality_score_before"] = before_score;
        outcome["quality_score_after"] = after_score;
        outcome["quality_score_delta"] = delta;
    }

    // fwhm_median: niedrigerer Wert = bessere Sternschärfe
    const double before_fwhm = number_or_nan(before, "fwhm_median");
    const double after_fwhm = number_or_nan(after, "fwhm_median");
    if (std::isfinite(before_fwhm) && std::isfinite(after_fwhm)) {
        outcome["fwhm_before"] = before_fwhm;
        outcome["fwhm_after"] = after_fwhm;
        outcome["fwhm_delta"] = after_fwhm - before_fwhm;
    }

    // background_gradient: niedrigerer Wert = flacherer Hintergrund
    const double before_grad = number_or_nan(before, "background_gradient");
    const double after_grad = number_or_nan(after, "background_gradient");
    if (std::isfinite(before_grad) && std::isfinite(after_grad)) {
        outcome["background_gradient_before"] = before_grad;
        outcome["background_gradient_after"] = after_grad;
        outcome["background_gradient_delta"] = after_grad - before_grad;
    }

    // star_count: zur Plausibilitaetsprüfung (grosse Aenderung = Analyse-Artefakt)
    const double before_stars = number_or_nan(before, "star_count");
    const double after_stars = number_or_nan(after, "star_count");
    if (std::isfinite(before_stars) && std::isfinite(after_stars)) {
        outcome["star_count_before"] = before_stars;
        outcome["star_count_after"] = after_stars;
        outcome["star_count_delta"] = after_stars - before_stars;
    }

    // report_warnings: Array-Vergleich — wurden Warnungen weniger?
    const bool has_before_warnings = before.contains("report_warnings") && before["report_warnings"].is_array();
    const bool has_after_warnings = after.contains("report_warnings") && after["report_warnings"].is_array();
    if (has_before_warnings || has_after_warnings) {
        const int before_warn_count = has_before_warnings
            ? static_cast<int>(before["report_warnings"].size()) : 0;
        const int after_warn_count = has_after_warnings
            ? static_cast<int>(after["report_warnings"].size()) : 0;
        outcome["report_warnings_before_count"] = before_warn_count;
        outcome["report_warnings_after_count"] = after_warn_count;
        outcome["report_warnings_delta"] = after_warn_count - before_warn_count;
        if (has_before_warnings) outcome["report_warnings_before"] = before["report_warnings"];
        if (has_after_warnings) outcome["report_warnings_after"] = after["report_warnings"];
    }

    // resume_phase: welche Phase wurde als Startpunkt gewaehlt?
    if (after.contains("resume_phase") && after["resume_phase"].is_string()) {
        outcome["resume_phase"] = after["resume_phase"];
    }

    // artifact_status: ok/error/missing fuer kritische Artefakte
    if (after.contains("artifact_status") && after["artifact_status"].is_object()) {
        outcome["artifact_status"] = after["artifact_status"];
    }

    // Nutzer-Bewertung (1..5 oder bool)
    if (body.contains("user_rating")) {
        outcome["user_rating"] = body["user_rating"];
    }

    // Verdict ableiten: zuerst aus user_result, dann aus quantitativen Deltas
    std::string verdict = "unknown";
    if (user_result == "improved" || user_result == "better") verdict = "improved";
    else if (user_result == "worse" || user_result == "regression") verdict = "worse";
    else if (user_result == "same" || user_result == "unchanged" || user_result == "no_improvement") verdict = "unchanged";
    else if (std::isfinite(delta)) {
        if (delta > 0.02) verdict = "improved";
        else if (delta < -0.02) verdict = "worse";
        else verdict = "unchanged";
    } else if (std::isfinite(before_fwhm) && std::isfinite(after_fwhm) && before_fwhm > 0) {
        // FWHM-Verbesserung als sekundaeres Signal (kleiner = besser)
        const double fwhm_rel = (before_fwhm - after_fwhm) / before_fwhm;
        if (fwhm_rel > 0.05) verdict = "improved";
        else if (fwhm_rel < -0.05) verdict = "worse";
        else verdict = "unchanged";
    } else if (has_before_warnings && has_after_warnings) {
        // Weniger Warnungen nach dem Run = Verbesserung
        const int dw = static_cast<int>(after["report_warnings"].size())
                     - static_cast<int>(before["report_warnings"].size());
        if (dw < 0) verdict = "improved";
        else if (dw > 0) verdict = "worse";
        else verdict = "unchanged";
    }

    // User-Rating ueberschreibt quantitative Ableitung
    if (body.contains("user_rating")) {
        const auto& ur = body["user_rating"];
        if (ur.is_boolean() && ur.get<bool>()) verdict = "improved";
        else if (ur.is_boolean() && !ur.get<bool>()) verdict = "worse";
        else if (ur.is_number()) {
            const double v = ur.get<double>();
            if (v >= 4.0) verdict = "improved";
            else if (v <= 2.0) verdict = "worse";
            else verdict = "unchanged";
        }
    }

    outcome["schema_version"] = "pi.memory-outcome.v1";
    outcome["verdict"] = verdict;
    outcome["verified"] = verdict != "unknown";
    outcome["human_feedback"] = feedback.empty() ? nlohmann::json(nullptr) : nlohmann::json(feedback);
    outcome["review_recommendation"] = verdict == "improved"
        ? "promotable"
        : (verdict == "worse" || verdict == "unchanged" ? "rejected" : "promotable");
    return outcome;
}

// Extrahiert Outcome-Metriken direkt aus Run-Artefakten fuer den Post-Run-Trigger.
// Liest stats.json und bge.json um FWHM, Warnungen und Artefaktstatus zu ermitteln.
nlohmann::json extract_run_outcome_metrics(const fs::path& run_dir) {
    nlohmann::json metrics = nlohmann::json::object();
    const fs::path artifacts_dir = run_dir / "artifacts";
    std::error_code ec;
    if (!fs::is_directory(artifacts_dir, ec)) return metrics;

    // stats.json: phase_issues, summary.status, quality metrics
    const fs::path stats_path = artifacts_dir / "stats.json";
    if (fs::is_regular_file(stats_path, ec)) {
        std::ifstream in(stats_path);
        if (in) {
            auto parsed = nlohmann::json::parse(in, nullptr, false);
            if (!parsed.is_discarded() && parsed.is_object()) {
                nlohmann::json warnings = nlohmann::json::array();
                if (parsed.contains("phase_issues") && parsed["phase_issues"].is_array()) {
                    for (const auto& issue : parsed["phase_issues"]) {
                        if (issue.is_string()) warnings.push_back(issue);
                        else if (issue.is_object() && issue.contains("message")) warnings.push_back(issue["message"]);
                    }
                }
                metrics["report_warnings"] = warnings;
                if (parsed.contains("summary") && parsed["summary"].is_object()) {
                    metrics["report_status"] = parsed["summary"].value("status", std::string("unknown"));
                }
                // Qualitaetsmetriken aus stats
                for (const char* key : {"fwhm_median", "background_gradient", "star_count", "quality_score"}) {
                    if (parsed.contains(key) && parsed[key].is_number()) metrics[key] = parsed[key];
                }
                // Tiefere Metriken aus aggregate-Sektion
                if (parsed.contains("aggregate") && parsed["aggregate"].is_object()) {
                    const auto& agg = parsed["aggregate"];
                    if (agg.contains("fwhm") && agg["fwhm"].is_object() && agg["fwhm"].contains("median")) {
                        if (!metrics.contains("fwhm_median")) metrics["fwhm_median"] = agg["fwhm"]["median"];
                    }
                }
            }
        }
    }

    // Artefaktstatus: welche bekannten Ausgabedateien existieren?
    nlohmann::json artifact_status = nlohmann::json::object();
    for (const char* artifact : {"bge.json", "stacked.fits", "stacked_rgb.fits",
                                   "pcc.json", "hms_preview.jpg", "hms_preview.png"}) {
        artifact_status[artifact] = fs::is_regular_file(artifacts_dir / artifact, ec) ? "ok" : "missing";
    }
    metrics["artifact_status"] = artifact_status;

    return metrics;
}

std::vector<float> read_fits_plane_preview(const fs::path& path, long plane,
                                           long& width, long& height, long& planes) {
    fitsfile* file = nullptr;
    int status = 0;
    if (fits_open_file(&file, path.string().c_str(), READONLY, &status)) {
        throw std::runtime_error("Cannot open FITS: " + fits_status_text(status));
    }
    int naxis = 0;
    long axes[3]{1, 1, 1};
    if (fits_get_img_dim(file, &naxis, &status) ||
        fits_get_img_size(file, 3, axes, &status) || naxis < 2) {
        fits_close_file(file, &status);
        throw std::runtime_error("Invalid FITS image");
    }
    width = axes[0];
    height = axes[1];
    planes = naxis >= 3 ? axes[2] : 1;
    if (plane < 1 || plane > planes) {
        fits_close_file(file, &status);
        throw std::runtime_error("Missing FITS plane");
    }
    std::vector<float> pixels(static_cast<size_t>(width * height));
    long first[3]{1, 1, plane};
    int any_null = 0;
    if (fits_read_pix(file, TFLOAT, first, width * height, nullptr, pixels.data(), &any_null, &status)) {
        const auto message = fits_status_text(status);
        fits_close_file(file, &status);
        throw std::runtime_error("Cannot read FITS: " + message);
    }
    fits_close_file(file, &status);
    return pixels;
}

cv::Mat plane_to_mat(const std::vector<float>& values, long width, long height) {
    cv::Mat mat(static_cast<int>(height), static_cast<int>(width), CV_32F);
    for (long y = 0; y < height; ++y)
        for (long x = 0; x < width; ++x)
            mat.at<float>(static_cast<int>(y), static_cast<int>(x)) = values[static_cast<size_t>(y * width + x)];
    return mat;
}

// FITS live-edit data is stored as linear float values in the [0, 1] range.
// Preview rendering must not apply an implicit stretch or gamma curve: those
// are display operations, not live-image edits, and make the preview disagree
// with the canonical live_edit.fits values.
unsigned char linear_float_to_u8(float value) {
    if (!std::isfinite(value)) value = 0.0f;
    return cv::saturate_cast<unsigned char>(std::clamp(value, 0.0f, 1.0f) * 255.0f);
}

std::vector<unsigned char> render_fits_preview_png_for_pi(const fs::path& path, int max_edge = 1024) {
    long w = 0, h = 0, p = 0;
    auto rv = read_fits_plane_preview(path, 1, w, h, p);
    cv::Mat r = plane_to_mat(rv, w, h);
    cv::Mat g = r;
    cv::Mat b = r;
    if (p >= 3) {
        long wg = 0, hg = 0, pg = 0;
        auto gv = read_fits_plane_preview(path, 2, wg, hg, pg);
        g = plane_to_mat(gv, wg, hg);
        long wb = 0, hb = 0, pb = 0;
        auto bv = read_fits_plane_preview(path, 3, wb, hb, pb);
        b = plane_to_mat(bv, wb, hb);
    }
    const int edge = std::max(r.rows, r.cols);
    if (edge > max_edge) {
        const double scale = static_cast<double>(max_edge) / static_cast<double>(edge);
        const cv::Size size(std::max(1, static_cast<int>(std::lround(r.cols * scale))),
                            std::max(1, static_cast<int>(std::lround(r.rows * scale))));
        cv::resize(r, r, size, 0, 0, cv::INTER_AREA);
        cv::resize(g, g, size, 0, 0, cv::INTER_AREA);
        cv::resize(b, b, size, 0, 0, cv::INTER_AREA);
    }
    cv::Mat out(r.rows, r.cols, CV_8UC3);
    for (int y = 0; y < r.rows; ++y) {
        for (int x = 0; x < r.cols; ++x) {
            auto& px = out.at<cv::Vec3b>(y, x);
            px[2] = linear_float_to_u8(r.at<float>(y, x));
            px[1] = linear_float_to_u8(g.at<float>(y, x));
            px[0] = linear_float_to_u8(b.at<float>(y, x));
        }
    }
    std::vector<unsigned char> png;
    if (!cv::imencode(".png", out, png)) throw std::runtime_error("PNG encoding failed");
    return png;
}

std::vector<unsigned char> render_fits_full_jpeg(const fs::path& path, int quality = 92) {
    long w = 0, h = 0, p = 0;
    auto rv = read_fits_plane_preview(path, 1, w, h, p);
    cv::Mat r = plane_to_mat(rv, w, h);
    cv::Mat g = r;
    cv::Mat b = r;
    if (p >= 3) {
        long wg = 0, hg = 0, pg = 0;
        auto gv = read_fits_plane_preview(path, 2, wg, hg, pg);
        g = plane_to_mat(gv, wg, hg);
        long wb = 0, hb = 0, pb = 0;
        auto bv = read_fits_plane_preview(path, 3, wb, hb, pb);
        b = plane_to_mat(bv, wb, hb);
    }
    // No downscale — 1:1 resolution
    cv::Mat out(r.rows, r.cols, CV_8UC3);
    for (int y = 0; y < r.rows; ++y) {
        for (int x = 0; x < r.cols; ++x) {
            auto& px = out.at<cv::Vec3b>(y, x);
            px[2] = linear_float_to_u8(r.at<float>(y, x));
            px[1] = linear_float_to_u8(g.at<float>(y, x));
            px[0] = linear_float_to_u8(b.at<float>(y, x));
        }
    }
    int effective_quality = quality;
    if (std::max(r.rows, r.cols) > 4000) effective_quality = 85;
    std::vector<unsigned char> jpeg;
    std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, effective_quality};
    if (!cv::imencode(".jpg", out, jpeg, params))
        throw std::runtime_error("JPEG encoding failed");
    return jpeg;
}

cv::Mat render_float_planes_to_bgr8(const cv::Mat& r, const cv::Mat& g, const cv::Mat& b) {
    cv::Mat out(r.rows, r.cols, CV_8UC3);
    for (int y = 0; y < r.rows; ++y) {
        for (int x = 0; x < r.cols; ++x) {
            auto& px = out.at<cv::Vec3b>(y, x);
            px[2] = linear_float_to_u8(r.at<float>(y, x));
            px[1] = linear_float_to_u8(g.at<float>(y, x));
            px[0] = linear_float_to_u8(b.at<float>(y, x));
        }
    }
    return out;
}

cv::Mat render_float_bgr_to_bgr8(const cv::Mat& img) {
    if (img.channels() == 1) {
        std::vector<cv::Mat> planes;
        cv::split(img, planes);
        return render_float_planes_to_bgr8(planes[0], planes[0], planes[0]);
    }
    std::vector<cv::Mat> planes;
    cv::split(img, planes); // B, G, R
    return render_float_planes_to_bgr8(planes[2], planes[1], planes[0]);
}

cv::Mat read_fits_to_float_bgr(const fs::path& path) {
    long w = 0, h = 0, p = 0;
    auto rv = read_fits_plane_preview(path, 1, w, h, p);
    cv::Mat r = plane_to_mat(rv, w, h);
    cv::Mat g = r;
    cv::Mat b = r;
    if (p >= 3) {
        long wg = 0, hg = 0, pg = 0;
        auto gv = read_fits_plane_preview(path, 2, wg, hg, pg);
        g = plane_to_mat(gv, wg, hg);
        long wb = 0, hb = 0, pb = 0;
        auto bv = read_fits_plane_preview(path, 3, wb, hb, pb);
        b = plane_to_mat(bv, wb, hb);
    }

    cv::Mat bgr(h, w, CV_32FC3);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            auto value = [](float v) -> float {
                return std::isfinite(v) ? v : 0.0f;
            };
            cv::Vec3f& px = bgr.at<cv::Vec3f>(y, x);
            px[0] = value(b.at<float>(y, x)); // B
            px[1] = value(g.at<float>(y, x)); // G
            px[2] = value(r.at<float>(y, x)); // R
        }
    }
    return bgr;
}

void write_float_bgr_to_fits(const cv::Mat& img, const fs::path& path) {
    fitsfile* fptr = nullptr;
    int status = 0;

    // Create parent directory if it doesn't exist
    std::error_code ec;
    fs::create_directories(path.parent_path(), ec);

    // cfitsio requires '!' prefix to overwrite an existing file
    std::string path_str = path.string();
    if (path_str[0] != '!')
        path_str = "!" + path_str;

    if (fits_create_file(&fptr, path_str.c_str(), &status)) {
        throw std::runtime_error("Cannot create FITS file: " + fits_status_text(status));
    }

    const int channels = img.channels();
    const int naxis = (channels == 1) ? 2 : 3;
    long naxes[3] = {img.cols, img.rows, channels};

    if (fits_create_img(fptr, FLOAT_IMG, naxis, naxes, &status)) {
        fits_close_file(fptr, &status);
        throw std::runtime_error("Cannot create FITS image: " + fits_status_text(status));
    }

    double display_min = 0.0;
    double display_max = 1.0;
    fits_update_key(fptr, TDOUBLE, const_cast<char*>("DATAMIN"), &display_min,
                    const_cast<char*>("Live editor display minimum"), &status);
    fits_update_key(fptr, TDOUBLE, const_cast<char*>("DATAMAX"), &display_max,
                    const_cast<char*>("Live editor display maximum"), &status);
    if (channels == 3) {
        char rgb[] = "RGB";
        fits_update_key(fptr, TSTRING, const_cast<char*>("CTYPE3"), rgb,
                        const_cast<char*>("Color channel"), &status);
    }
    if (status) {
        fits_close_file(fptr, &status);
        throw std::runtime_error("Cannot write FITS display metadata: " + fits_status_text(status));
    }

    if (!img.isContinuous()) {
        cv::Mat tmp = img.clone();
        write_float_bgr_to_fits(tmp, path);
        return;
    }

    const size_t plane_pixels = static_cast<size_t>(img.rows) * img.cols;
    std::vector<float> pixels(plane_pixels * channels, 0.0f);
    if (channels == 1) {
        const float* src = reinterpret_cast<const float*>(img.data);
        std::memcpy(pixels.data(), src, pixels.size() * sizeof(float));
    } else {
        std::vector<cv::Mat> planes;
        cv::split(img, planes); // B, G, R
        for (int c = 0; c < 3; ++c) {
            const float* src = reinterpret_cast<const float*>(planes[2 - c].data);
            std::memcpy(pixels.data() + c * plane_pixels, src, plane_pixels * sizeof(float));
        }
    }

    long first[3] = {1, 1, 1};
    if (fits_write_pix(fptr, TFLOAT, first, pixels.size(), pixels.data(), &status)) {
        fits_close_file(fptr, &status);
        throw std::runtime_error("Cannot write FITS data: " + fits_status_text(status));
    }
    fits_close_file(fptr, &status);
}

std::string mat_to_jpeg_base64(const cv::Mat& img, int quality = 85) {
    cv::Mat display;
    if (img.depth() == CV_32F) {
        display = render_float_bgr_to_bgr8(img);
    } else {
        display = img;
    }
    std::vector<unsigned char> jpeg;
    std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, quality};
    if (!cv::imencode(".jpg", display, jpeg, params))
        throw std::runtime_error("JPEG encoding failed");
    return base64_encode(jpeg);
}

std::string mat_to_vision_jpeg_base64(const cv::Mat& img, int max_dim = 1568, int quality = 85) {
    cv::Mat display = (img.depth() == CV_32F) ? render_float_bgr_to_bgr8(img) : img;
    cv::Mat resized = display;
    int longest = std::max(display.rows, display.cols);
    if (longest > max_dim) {
        double scale = static_cast<double>(max_dim) / longest;
        cv::resize(display, resized, cv::Size(), scale, scale, cv::INTER_AREA);
    }
    std::vector<unsigned char> jpeg;
    std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, quality};
    if (!cv::imencode(".jpg", resized, jpeg, params))
        throw std::runtime_error("JPEG encoding failed (vision)");
    return base64_encode(jpeg);
}

nlohmann::json fallback_parse_message(const std::string& msg, const cv::Mat& image) {
    std::string lower = msg;
    std::transform(lower.begin(), lower.end(), lower.begin(),
                   [](unsigned char c) { return std::tolower(c); });

    auto has = [&](const std::string& kw) {
        return lower.find(kw) != std::string::npos;
    };

    nlohmann::json operations = nlohmann::json::array();
    std::string summary;
    bool adjustable = false;
    bool repeatable = false;
    nlohmann::json adjust_step;

    // Derive conservative local parameters from the current image when the
    // AI sidecar is unavailable. The command text still selects the
    // operation; these statistics only determine its strength.
    double median_luma = 0.25;
    double p10_luma = 0.05;
    double p90_luma = 0.60;
    double mean_saturation = 0.35;
    if (!image.empty() && image.depth() == CV_32F) {
        cv::Mat luminance;
        if (image.channels() >= 3) cv::cvtColor(image, luminance, cv::COLOR_BGR2GRAY);
        else luminance = image;
        std::vector<float> samples;
        samples.reserve(static_cast<size_t>(luminance.total() / 16 + 1));
        for (int y = 0; y < luminance.rows; y += 4) {
            for (int x = 0; x < luminance.cols; x += 4) {
                const float v = luminance.at<float>(y, x);
                if (std::isfinite(v)) samples.push_back(std::clamp(v, 0.0f, 1.0f));
            }
        }
        if (!samples.empty()) {
            std::sort(samples.begin(), samples.end());
            auto quantile = [&](double q) {
                const size_t idx = std::min(samples.size() - 1,
                    static_cast<size_t>(q * static_cast<double>(samples.size() - 1)));
                return static_cast<double>(samples[idx]);
            };
            p10_luma = quantile(0.10);
            median_luma = quantile(0.50);
            p90_luma = quantile(0.90);
        }
        if (image.channels() >= 3) {
            cv::Mat hsv;
            cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);
            std::vector<cv::Mat> hsv_channels;
            cv::split(hsv, hsv_channels);
            mean_saturation = cv::mean(hsv_channels[1])[0];
            mean_saturation = std::clamp(mean_saturation, 0.0, 1.0);
        }
    }
    const double brighten_amount = std::clamp((0.32 - median_luma) * 0.75, 0.04, 0.25);
    const double darken_amount = std::clamp((median_luma - 0.10) * 0.75, 0.04, 0.25);
    const double contrast_amount = std::clamp((0.45 - (p90_luma - p10_luma)) * 0.7, 0.04, 0.25);
    const double saturation_amount = std::clamp((0.55 - mean_saturation) * 0.7, 0.04, 0.25);

    if (has("heller") || has("aufhellen") || has("brighter")) {
        operations.push_back({{"type", "brightness"}, {"params", {{"midtones", brighten_amount}, {"shadows", 0.0}, {"highlights", 0.0}}}});
        summary = "Helligkeit erhoeht.";
        adjustable = true;
        adjust_step = {{"type", "brightness"}, {"params", {{"midtones", brighten_amount * 0.35}, {"shadows", 0.0}, {"highlights", 0.0}}}};
    } else if (has("dunkler") || has("darker")) {
        operations.push_back({{"type", "brightness"}, {"params", {{"midtones", -darken_amount}, {"shadows", 0.0}, {"highlights", 0.0}}}});
        summary = "Helligkeit reduziert.";
        adjustable = true;
        adjust_step = {{"type", "brightness"}, {"params", {{"midtones", -darken_amount * 0.35}, {"shadows", 0.0}, {"highlights", 0.0}}}};
    } else if (has("kontrast") && !has("lokal") && !has("local") &&
               (has("mehr") || has("erhoeh"))) {
        operations.push_back({{"type", "contrast"}, {"params", {{"amount", contrast_amount}}}});
        summary = "Kontrast erhoeht.";
        adjustable = true;
        adjust_step = {{"type", "contrast"}, {"params", {{"amount", contrast_amount * 0.5}}}};
    } else if (has("kontrast") && has("weniger")) {
        operations.push_back({{"type", "contrast"}, {"params", {{"amount", -contrast_amount}}}});
        summary = "Kontrast reduziert.";
        adjustable = true;
        adjust_step = {{"type", "contrast"}, {"params", {{"amount", -contrast_amount * 0.5}}}};
    } else if ((has("saettigung") || has("farbe")) && (has("mehr") || has("erhoeh"))) {
        operations.push_back({{"type", "saturation"}, {"params", {{"amount", saturation_amount}}}});
        summary = "Saettigung erhoeht.";
        adjustable = true;
        adjust_step = {{"type", "saturation"}, {"params", {{"amount", saturation_amount * 0.5}}}};
    } else if (has("crop_rotated") || (has("crop") && lower.find("angle") != std::string::npos)) {
        int cx = 0, cy = 0, cw = 0, ch = 0;
        double angle = 0.0;
        std::smatch rmatch;
        if (std::regex_search(lower, rmatch,
                std::regex(R"(crop_rotated\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(-?\d+(?:[\.,]\d+)?))"))) {
            cx = std::stoi(rmatch[1].str());
            cy = std::stoi(rmatch[2].str());
            cw = std::stoi(rmatch[3].str());
            ch = std::stoi(rmatch[4].str());
            std::string a = rmatch[5].str();
            std::replace(a.begin(), a.end(), ',', '.');
            angle = std::stod(a);
        }
        cx = std::clamp(cx, 0, std::max(0, image.cols - 1));
        cy = std::clamp(cy, 0, std::max(0, image.rows - 1));
        cw = std::clamp(cw, 1, image.cols);
        ch = std::clamp(ch, 1, image.rows);
        angle = std::clamp(angle, -180.0, 180.0);
        operations.push_back({{"type", "crop_rotated"}, {"params", {{"cx", cx}, {"cy", cy}, {"w", cw}, {"h", ch}, {"angle", angle}}}});
        summary = "Bild gedreht zugeschnitten.";
    } else if (has("crop") || has("zuschneid") || has("beschneid") || has("rand abschneiden")) {
        int cx = 0, cy = 0, cw = 0, ch = 0;
        std::smatch pxmatch;
        if (std::regex_search(lower, pxmatch,
                std::regex(R"(crop\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+))"))) {
            cx = std::stoi(pxmatch[1].str());
            cy = std::stoi(pxmatch[2].str());
            cw = std::stoi(pxmatch[3].str());
            ch = std::stoi(pxmatch[4].str());
        } else {
            double border_fraction = 0.05;
            std::smatch match;
            if (std::regex_search(lower, match, std::regex(R"((\d+(?:[\.,]\d+)?)\s*%)"))) {
                try { border_fraction = std::stod(match[1].str()); } catch (...) {}
                if (match[1].str().find(',') != std::string::npos) {
                    std::string value = match[1].str();
                    std::replace(value.begin(), value.end(), ',', '.');
                    try { border_fraction = std::stod(value); } catch (...) {}
                }
                border_fraction /= 100.0;
            }
            border_fraction = std::clamp(border_fraction, 0.0, 0.40);
            cx = static_cast<int>(std::lround(image.cols * border_fraction));
            cy = static_cast<int>(std::lround(image.rows * border_fraction));
            cw = std::max(1, image.cols - 2 * cx);
            ch = std::max(1, image.rows - 2 * cy);
        }
        cx = std::clamp(cx, 0, std::max(0, image.cols - 1));
        cy = std::clamp(cy, 0, std::max(0, image.rows - 1));
        cw = std::clamp(cw, 1, image.cols - cx);
        ch = std::clamp(ch, 1, image.rows - cy);
        operations.push_back({{"type", "crop"}, {"params", {{"x", cx}, {"y", cy}, {"w", cw}, {"h", ch}}}});
        summary = "Bild zugeschnitten.";
    } else if (has("schae") || has("sharpen")) {
        operations.push_back({{"type", "sharpen"}, {"params", {{"amount", 0.3}, {"radius", 2.0}}}});
        summary = "Schaerfe erhoeht.";
        repeatable = true;
    } else if ((has("rausch") || has("denoise") || has("noise")) &&
               !has("farbrausch") && !has("chroma")) {
        operations.push_back({{"type", "denoise"}, {"params", {{"strength", 0.5}, {"luminance", false}}}});
        summary = "Rauschen reduziert.";
    } else if (has("gruen")) {
        operations.push_back({{"type", "rmgreen"}, {"params", {{"strength", 0.5}}}});
        summary = "Gruenanteil reduziert.";
    } else if (has("levels") || has("tonwerte") || has("tonwert")) {
        const double black = std::clamp(p10_luma * 0.5, 0.0, 0.4);
        const double white = std::clamp(p90_luma + 0.15, 0.6, 1.0);
        operations.push_back({{"type", "levels"}, {"params", {{"black", black}, {"white", white}, {"gamma", 1.0}}}});
        summary = "Tonwerte angepasst.";
    } else if (has("schatten") || has("shadow")) {
        operations.push_back({{"type", "shadow_recovery"}, {"params", {{"strength", 0.45}}}});
        summary = "Schatten wiederhergestellt.";
    } else if (has("spitzlich") || has("highlight")) {
        operations.push_back({{"type", "highlight_recovery"}, {"params", {{"strength", 0.45}}}});
        summary = "Spitzlichter wiederhergestellt.";
    } else if (has("farbbalance") || has("color balance") || has("farben waermer") || has("farben kuehler")) {
        const double red = has("waermer") ? 0.10 : (has("kuehler") ? -0.10 : 0.0);
        operations.push_back({{"type", "color_balance"}, {"params", {{"red", red}, {"green", 0.0}, {"blue", -red}}}});
        summary = "Farbbalance angepasst.";
    } else if (has("lokal kontrast") || has("local contrast")) {
        operations.push_back({{"type", "local_contrast"}, {"params", {{"strength", 0.4}, {"radius", 3.0}}}});
        summary = "Lokaler Kontrast erhoeht.";
    } else if (has("chrom") || has("farbrausch")) {
        operations.push_back({{"type", "chroma_denoise"}, {"params", {{"strength", 0.5}, {"protect", 0.5}, {"mode", "soft"}}}});
        summary = "Farbrauschen reduziert.";
    } else if (has("lebendig") || has("vibrance")) {
        operations.push_back({{"type", "vibrance"}, {"params", {{"amount", 0.1}}}});
        summary = "Lebendigkeit erhoeht.";
        adjustable = true;
        adjust_step = {{"type", "vibrance"}, {"params", {{"amount", 0.05}}}};
    } else if (has("warm") || has("waerm") || has("temperatur")) {
        operations.push_back({{"type", "color_temperature"}, {"params", {{"amount", 0.1}}}});
        summary = "Farben erwaermt.";
        adjustable = true;
        adjust_step = {{"type", "color_temperature"}, {"params", {{"amount", 0.05}}}};
    } else if (has("lila") || has("purple")) {
        operations.push_back({{"type", "unpurple"}, {"params", {{"amount", 0.6}}}});
        summary = "Lila Farbsaueme reduziert.";
    } else if (has("streifen") || has("banding")) {
        operations.push_back({{"type", "fixbanding"}, {"params", {{"amount", 0.5}, {"sigma", 2.0}}}});
        summary = "Streifenartefakte reduziert.";
    } else if (has("stern") && (has("saett") || has("entsaett"))) {
        operations.push_back({{"type", "star_desaturation"}, {"params", {{"amount", 0.5}}}});
        summary = "Uebersteuerte Sterne entsaettigt.";
    } else if (has("dunst") || has("dehaze")) {
        operations.push_back({{"type", "dehaze"}, {"params", {{"amount", 0.4}}}});
        summary = "Dunst reduziert.";
    } else if (has("detail") || has("lokal") || has("clahe")) {
        operations.push_back({{"type", "clahe"}, {"params", {{"cliplimit", 3.0}, {"tilesize", 8}}}});
        summary = "Lokaler Kontrast erhoeht (CLAHE).";
    } else if (has("invert")) {
        operations.push_back({{"type", "invert"}, {"params", nlohmann::json::object()}});
        summary = "Bild invertiert.";
    } else if (has("zurueck") || has("reset")) {
        operations.push_back({{"type", "reset"}, {"params", nlohmann::json::object()}});
        summary = "Bild zurueckgesetzt.";
    } else {
        summary = "Keine passende Operation gefunden. Versuche: heller, dunkler, Kontrast, Saettigung, SchaeRfe, Rauschen, Gruen, Details.";
    }

    return {
        {"operations", operations},
        {"summary", summary},
        {"adjustable", adjustable},
        {"repeatable", repeatable},
        {"adjust_step", adjust_step},
        {"warnings", nlohmann::json::array()},
        {"mode", "local_fallback"}
    };
}

void trim_json_array_to_latest(nlohmann::json& items, int limit) {
    if (!items.is_array() || limit <= 0) return;
    while (static_cast<int>(items.size()) > limit) items.erase(items.begin());
}

nlohmann::json read_pi_ai_config_file(const std::shared_ptr<AppState>& state) {
    const fs::path path = state->runtime.runtime_dir / "ai_scan_config.json";
    std::ifstream in(path);
    if (!in) return nlohmann::json::object();
    auto parsed = nlohmann::json::parse(in, nullptr, false);
    if (parsed.is_discarded() || !parsed.is_object()) return nlohmann::json::object();
    return parsed;
}

tile_compile::ai::AiConfig current_pi_ai_config(const std::shared_ptr<AppState>& state) {
    nlohmann::json memory_config = nlohmann::json::object();
    {
        std::lock_guard<std::mutex> lk(state->state_mutex);
        if (state->ui_state.contains("ai") && state->ui_state["ai"].contains("scan_analysis") &&
            state->ui_state["ai"]["scan_analysis"].is_object()) {
            memory_config = state->ui_state["ai"]["scan_analysis"];
        }
    }
    nlohmann::json merged = tile_compile::ai::merge_ai_config_json(
        tile_compile::ai::ai_config_to_json(tile_compile::ai::default_ai_config(state->runtime)),
        read_pi_ai_config_file(state),
        state->runtime);
    return tile_compile::ai::ai_config_from_json(
        tile_compile::ai::merge_ai_config_json(merged, memory_config, state->runtime),
        state->runtime);
}

std::filesystem::path pi_run_chat_history_path(const std::shared_ptr<AppState>& state,
                                               const std::string& run_id) {
    std::string safe;
    safe.reserve(run_id.size());
    for (unsigned char ch : run_id) {
        if (std::isalnum(ch) || ch == '-' || ch == '_' || ch == '.') safe.push_back(static_cast<char>(ch));
        else safe.push_back('_');
    }
    if (safe.empty()) safe = "run";
    const auto hash = std::hash<std::string>{}(run_id);
    return tile_compile::pi::pi_storage_dir(state) / "run_chat" /
        (safe + "_" + std::to_string(static_cast<unsigned long long>(hash)) + ".json");
}

std::filesystem::path legacy_pi_run_chat_history_path(const std::shared_ptr<AppState>& state,
                                                      const std::string& run_id) {
    const auto run_dir = state->runtime.resolve_run_dir(run_id);
    return run_dir / "artifacts" / "pi_run_chat_history.json";
}

nlohmann::json read_pi_run_chat_history(const std::shared_ptr<AppState>& state,
                                        const std::string& run_id) {
    auto path = pi_run_chat_history_path(state, run_id);
    if (!std::filesystem::exists(path)) {
        const auto legacy = legacy_pi_run_chat_history_path(state, run_id);
        if (std::filesystem::exists(legacy)) path = legacy;
    }
    if (!std::filesystem::exists(path)) {
        return {
            {"schema_version", "pi.run-chat-history.v1"},
            {"run_id", run_id},
            {"messages", nlohmann::json::array()},
            {"turns", nlohmann::json::array()}
        };
    }
    std::ifstream in(path);
    nlohmann::json parsed = nlohmann::json::parse(in, nullptr, false);
    if (parsed.is_discarded() || !parsed.is_object()) {
        throw std::runtime_error("invalid run chat history");
    }
    parsed["schema_version"] = "pi.run-chat-history.v1";
    parsed["run_id"] = run_id;
    if (!parsed.contains("messages") || !parsed["messages"].is_array()) parsed["messages"] = nlohmann::json::array();
    if (!parsed.contains("turns") || !parsed["turns"].is_array()) parsed["turns"] = nlohmann::json::array();
    return parsed;
}

void write_pi_run_chat_history(const std::shared_ptr<AppState>& state,
                               const std::string& run_id,
                               nlohmann::json history) {
    auto path = pi_run_chat_history_path(state, run_id);
    std::filesystem::create_directories(path.parent_path());
    history["schema_version"] = "pi.run-chat-history.v1";
    history["run_id"] = run_id;
    if (!history.contains("messages") || !history["messages"].is_array()) history["messages"] = nlohmann::json::array();
    if (!history.contains("turns") || !history["turns"].is_array()) history["turns"] = nlohmann::json::array();
    trim_json_array_to_latest(history["messages"], 24);
    trim_json_array_to_latest(history["turns"], 24);
    std::ofstream out(path, std::ios::out | std::ios::trunc);
    if (!out) throw std::runtime_error("failed to open run chat history for writing");
    out << history.dump(2);
}

std::filesystem::path pi_live_image_chat_history_path(const std::shared_ptr<AppState>& state,
                                                      const std::string& run_id) {
    std::string safe;
    safe.reserve(run_id.size());
    for (char c : run_id) {
        if (std::isalnum(static_cast<unsigned char>(c)) || c == '-' || c == '_' || c == '.')
            safe += c;
    }
    if (safe.empty()) safe = "run";
    const auto hash = std::hash<std::string>{}(run_id);
    return tile_compile::pi::pi_storage_dir(state) / "live_image_chat" /
        (safe + "_" + std::to_string(static_cast<unsigned long long>(hash)) + ".json");
}

nlohmann::json read_pi_live_image_chat_history(const std::shared_ptr<AppState>& state,
                                               const std::string& run_id) {
    auto path = pi_live_image_chat_history_path(state, run_id);
    if (!std::filesystem::exists(path))
        return {{"schema_version", "pi.live-image-chat-history.v1"},
                {"run_id", run_id},
                {"chat_history", nlohmann::json::array()},
                {"operation_history", nlohmann::json::array()},
                {"edit_history", nlohmann::json::array()}};
    std::ifstream in(path);
    auto data = nlohmann::json::parse(in, nullptr, false);
    if (data.is_discarded() || !data.is_object())
        return {{"schema_version", "pi.live-image-chat-history.v1"},
                {"run_id", run_id},
                {"chat_history", nlohmann::json::array()},
                {"operation_history", nlohmann::json::array()},
                {"edit_history", nlohmann::json::array()}};
    return data;
}

void write_pi_live_image_chat_history(const std::shared_ptr<AppState>& state,
                                      const std::string& run_id,
                                      nlohmann::json history) {
    auto path = pi_live_image_chat_history_path(state, run_id);
    std::filesystem::create_directories(path.parent_path());
    history["schema_version"] = "pi.live-image-chat-history.v1";
    history["run_id"] = run_id;
    if (!history.contains("chat_history") || !history["chat_history"].is_array())
        history["chat_history"] = nlohmann::json::array();
    if (!history.contains("operation_history") || !history["operation_history"].is_array())
        history["operation_history"] = nlohmann::json::array();
    if (!history.contains("edit_history") || !history["edit_history"].is_array())
        history["edit_history"] = nlohmann::json::array();
    history["last_updated"] = utc_now_iso();
    // Atomic write: temp + rename
    const auto tmp = path.string() + ".tmp";
    {
        std::ofstream out(tmp, std::ios::out | std::ios::trunc);
        if (!out) throw std::runtime_error("failed to open live image chat history for writing");
        out << history.dump(2);
    }
    std::filesystem::rename(tmp, path);
}

void persist_live_session(const std::shared_ptr<AppState>& state,
                          const std::shared_ptr<tile_compile::pi::LiveImageSessionStore>& store,
                          const std::string& session_id) {
    std::string run_id;
    store->with_session(session_id, [&](tile_compile::pi::LiveImageSession& s) {
        run_id = s.run_id;
    });
    if (run_id.empty()) return;
    const auto chat_history = store->get_chat_history(session_id);
    const auto operation_history = store->get_operation_history(session_id);
    nlohmann::json edit_history = nlohmann::json::array();
    store->with_session(session_id, [&](tile_compile::pi::LiveImageSession& s) {
        edit_history = s.edit_history;
    });
    if (chat_history.empty() && operation_history.empty() && edit_history.empty()) {
        std::error_code ec;
        std::filesystem::remove(pi_live_image_chat_history_path(state, run_id), ec);
        return;
    }
    write_pi_live_image_chat_history(state, run_id, {
        {"chat_history", chat_history},
        {"operation_history", operation_history},
        {"edit_history", edit_history},
        {"created_at", utc_now_iso()}
    });
}

std::filesystem::path pi_live_image_presets_dir(const std::shared_ptr<AppState>& state) {
    return tile_compile::pi::pi_storage_dir(state) / "presets";
}

std::string sanitize_live_preset_id(const std::string& name) {
    std::string safe;
    for (unsigned char ch : name) {
        if (std::isalnum(ch) || ch == '-' || ch == '_' || ch == '.') safe.push_back(static_cast<char>(ch));
        else if (std::isspace(ch)) safe.push_back('_');
    }
    while (!safe.empty() && safe.front() == '.') safe.erase(safe.begin());
    if (safe.size() > 80) safe.resize(80);
    return safe;
}

nlohmann::json live_preset_summary(const nlohmann::json& preset) {
    return {
        {"id", preset.value("id", std::string())},
        {"name", preset.value("name", preset.value("id", std::string()))},
        {"updated_at", preset.value("updated_at", std::string())},
        {"operation_count", preset.value("operations", nlohmann::json::array()).size()}
    };
}

// Extract a short label from an operation stack entry for undo/redo tooltips.
// Returns null if the entry is empty or has no type.
nlohmann::json stack_top_label(const std::vector<nlohmann::json>& stack) {
    if (stack.empty()) return nullptr;
    const auto& top = stack.back();
    if (!top.is_object() || !top.contains("type")) return nullptr;
    return top;
}

nlohmann::json read_live_preset(const std::shared_ptr<AppState>& state,
                                const std::string& id) {
    const auto safe = sanitize_live_preset_id(id);
    if (safe.empty() || safe != id) return nlohmann::json();
    const auto path = pi_live_image_presets_dir(state) / (safe + ".json");
    if (!std::filesystem::exists(path)) return nlohmann::json();
    std::ifstream in(path);
    auto parsed = nlohmann::json::parse(in, nullptr, false);
    if (parsed.is_discarded() || !parsed.is_object()) return nlohmann::json();
    if (!parsed.value("operations", nlohmann::json::array()).is_array()) return nlohmann::json();
    return parsed;
}

void write_live_preset(const std::shared_ptr<AppState>& state, nlohmann::json preset) {
    const auto id = preset.value("id", std::string());
    const auto safe = sanitize_live_preset_id(id);
    if (safe.empty() || safe != id) throw std::runtime_error("invalid preset id");
    auto dir = pi_live_image_presets_dir(state);
    std::filesystem::create_directories(dir);
    const auto path = dir / (safe + ".json");
    const auto tmp = path.string() + ".tmp";
    std::ofstream out(tmp, std::ios::out | std::ios::trunc);
    if (!out) throw std::runtime_error("failed to open preset for writing");
    out << preset.dump(2);
    out.close();
    std::error_code ec;
    std::filesystem::rename(tmp, path, ec);
    if (ec) throw std::runtime_error("failed to finalize preset: " + ec.message());
}

void persist_live_edit_fits(const std::shared_ptr<AppState>& state,
                            const std::shared_ptr<tile_compile::pi::LiveImageSessionStore>& store,
                            const std::string& session_id) {
    cv::Mat current_fits;
    std::string run_id;
    store->with_session(session_id, [&](tile_compile::pi::LiveImageSession& s) {
        current_fits = s.current_fits.clone();
        run_id = s.run_id;
    });
    if (run_id.empty() || current_fits.empty()) return;
    try {
        const auto run_dir = state->runtime.resolve_run_dir(run_id);
        const fs::path edit_path = run_dir / "outputs" / "live_edit.fits";
        write_float_bgr_to_fits(current_fits, edit_path);
    } catch (...) {
        // Don't fail API calls if the copy can't be written
    }
}

std::string chat_message_key(const nlohmann::json& item) {
    if (!item.is_object()) return "";
    return item.value("role", std::string()) + "\n" + item.value("content", std::string());
}

void append_chat_message_unique(nlohmann::json& messages,
                                const std::string& role,
                                const std::string& content) {
    if (content.empty()) return;
    nlohmann::json item = {{"role", role}, {"content", content}};
    const std::string key = chat_message_key(item);
    for (const auto& existing : messages) {
        if (chat_message_key(existing) == key) return;
    }
    messages.push_back(std::move(item));
}

nlohmann::json compact_run_chat_history_messages(const nlohmann::json& server_history,
                                                 const nlohmann::json& body,
                                                 const std::string& message) {
    nlohmann::json messages = nlohmann::json::array();

    if (server_history.contains("messages") && server_history["messages"].is_array()) {
        for (const auto& item : server_history["messages"]) {
            if (!item.is_object()) continue;
            append_chat_message_unique(messages,
                                       item.value("role", std::string()),
                                       item.value("content", std::string()));
        }
    }

    if (body.contains("messages") && body["messages"].is_array()) {
        for (const auto& item : body["messages"]) {
            if (!item.is_object()) continue;
            append_chat_message_unique(messages,
                                       item.value("role", std::string()),
                                       item.value("content", std::string()));
        }
    }

    append_chat_message_unique(messages, "user", message);
    trim_json_array_to_latest(messages, 24);
    return messages;
}

std::string run_chat_analysis_message_from_messages(const nlohmann::json& messages,
                                                    const std::string& message) {
    std::string combined;
    int count = 0;
    for (const auto& item : messages) {
        if (!item.is_object()) continue;
        const std::string content = item.value("content", std::string());
        if (content.empty()) continue;
        if (!combined.empty()) combined += "\n";
        combined += content;
        if (++count >= 12) break;
    }
    if (combined.find(message) == std::string::npos) {
        if (!combined.empty()) combined += "\n";
        combined += message;
    }
    return combined;
}

nlohmann::json run_chat_previous_turns_context(const nlohmann::json& history, int limit = 4) {
    nlohmann::json out = nlohmann::json::array();
    if (!history.contains("turns") || !history["turns"].is_array()) return out;
    const auto& turns = history["turns"];
    const size_t start = turns.size() > static_cast<size_t>(limit)
        ? turns.size() - static_cast<size_t>(limit)
        : 0;
    for (size_t i = start; i < turns.size(); ++i) {
        const auto& turn = turns[i];
        if (!turn.is_object()) continue;
        out.push_back({
            {"message", turn.value("message", std::string())},
            {"summary", turn.contains("result") && turn["result"].is_object()
                ? turn["result"].value("summary", std::string())
                : std::string()},
            {"resume_recommendation", turn.contains("result") && turn["result"].is_object() && turn["result"].contains("resume_recommendation")
                ? turn["result"]["resume_recommendation"]
                : nlohmann::json(nullptr)},
            {"action_plan", turn.contains("result") && turn["result"].is_object() && turn["result"].contains("action_plan")
                ? turn["result"]["action_plan"]
                : nlohmann::json(nullptr)}
        });
    }
    return out;
}

std::string chat_turn_key(const nlohmann::json& turn) {
    if (!turn.is_object()) return "";
    std::string summary;
    if (turn.contains("result") && turn["result"].is_object()) {
        summary = turn["result"].value("summary", std::string());
    }
    return turn.value("message", std::string()) + "\n" + summary;
}

nlohmann::json merge_run_chat_history(nlohmann::json existing, const nlohmann::json& incoming) {
    if (!existing.is_object()) existing = nlohmann::json::object();
    if (!existing.contains("messages") || !existing["messages"].is_array()) existing["messages"] = nlohmann::json::array();
    if (!existing.contains("turns") || !existing["turns"].is_array()) existing["turns"] = nlohmann::json::array();

    if (incoming.contains("messages") && incoming["messages"].is_array()) {
        for (const auto& item : incoming["messages"]) {
            if (!item.is_object()) continue;
            append_chat_message_unique(existing["messages"],
                                       item.value("role", std::string()),
                                       item.value("content", std::string()));
        }
    }

    if (incoming.contains("turns") && incoming["turns"].is_array()) {
        for (const auto& turn : incoming["turns"]) {
            const std::string key = chat_turn_key(turn);
            if (key.empty()) continue;
            bool exists = false;
            for (const auto& existing_turn : existing["turns"]) {
                if (chat_turn_key(existing_turn) == key) {
                    exists = true;
                    break;
                }
            }
            if (!exists) existing["turns"].push_back(turn);
        }
    }

    trim_json_array_to_latest(existing["messages"], 24);
    trim_json_array_to_latest(existing["turns"], 24);
    return existing;
}

std::string lower_ext(const fs::path& path) {
    std::string ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return ext;
}

int pi_preview_artifact_score(const std::string& rel) {
    std::string p = rel;
    std::transform(p.begin(), p.end(), p.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    int score = 0;
    if (p.rfind("outputs/", 0) == 0 || p.find("/outputs/") != std::string::npos) score += 20;
    if (p.find("stacked_rgb_hms") != std::string::npos || p.find("hms_") != std::string::npos) score += 100;
    else if (p.find("stacked_rgb_pcc") != std::string::npos || p.find("pcc_") != std::string::npos) score += 90;
    else if (p.find("stacked_rgb_bge") != std::string::npos) score += 80;
    else if (p.find("stacked_rgb") != std::string::npos) score += 60;
    if (p.size() >= 4 && p.substr(p.size() - 4) == ".png") score += 10;
    return score;
}

std::optional<fs::path> resolve_run_relative_artifact(const fs::path& run_dir, const std::string& rel) {
    if (rel.empty()) return std::nullopt;
    fs::path candidate = fs::weakly_canonical(run_dir / fs::path(rel));
    fs::path root = fs::weakly_canonical(run_dir);
    const fs::path relative = candidate.lexically_relative(root);
    if (relative.empty()) return std::nullopt;
    for (const auto& part : relative) {
        if (part == "..") return std::nullopt;
    }
    if (!fs::exists(candidate) || !fs::is_regular_file(candidate)) return std::nullopt;
    return candidate;
}

std::optional<fs::path> find_output_fits(const fs::path& run_dir) {
    const auto artifacts = list_run_artifacts(run_dir);
    std::string best_rel;
    int best_score = -1;
    for (const auto& item : artifacts) {
        if (!item.is_object()) continue;
        const std::string rel = item.value("path", item.value("relative_path", std::string()));
        const std::string ext = lower_ext(rel);
        if (ext != ".fits" && ext != ".fit" && ext != ".fts") continue;
        const int score = pi_preview_artifact_score(rel);
        if (score > best_score) {
            best_score = score;
            best_rel = rel;
        }
    }
    if (best_rel.empty()) return std::nullopt;
    return resolve_run_relative_artifact(run_dir, best_rel);
}

nlohmann::json build_run_chat_preview_image(const fs::path& run_dir) {
    const auto artifacts = list_run_artifacts(run_dir);
    std::string best_rel;
    int best_score = -1;
    for (const auto& item : artifacts) {
        if (!item.is_object()) continue;
        const std::string rel = item.value("path", item.value("relative_path", std::string()));
        const std::string ext = lower_ext(rel);
        if (ext != ".png" && ext != ".fits" && ext != ".fit" && ext != ".fts") continue;
        const int score = pi_preview_artifact_score(rel);
        if (score > best_score) {
            best_score = score;
            best_rel = rel;
        }
    }
    if (best_rel.empty()) return nlohmann::json{{"available", false}};
    auto full = resolve_run_relative_artifact(run_dir, best_rel);
    if (!full) return nlohmann::json{{"available", false}, {"path", best_rel}};
    try {
        std::vector<unsigned char> png;
        const std::string ext = lower_ext(*full);
        if (ext == ".png") {
            std::ifstream in(*full, std::ios::binary);
            png.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
        } else {
            png = render_fits_preview_png_for_pi(*full, 1024);
        }
        return {
            {"available", true},
            {"path", best_rel},
            {"mime", "image/png"},
            {"base64", base64_encode(png)},
            {"bytes", png.size()}
        };
    } catch (const std::exception& e) {
        return {{"available", false}, {"path", best_rel}, {"error", e.what()}};
    }
}

std::string build_provider_run_chat_prompt(const std::string& run_id,
                                           const std::string& message,
                                           const nlohmann::json& local_answer,
                                           const nlohmann::json& conversation_messages,
                                           const nlohmann::json& previous_turns,
                                           const nlohmann::json& status,
                                           const nlohmann::json& artifacts,
                                           const nlohmann::json& image_info,
                                           const nlohmann::json& target_context = nlohmann::json::object()) {
    std::ostringstream prompt;
    prompt
        << "You are PI for tile_compile, an astrophotography stacking/configuration assistant.\n"
        << "Answer the user's run-specific question using the attached preview image when available, plus the structured run context.\n"
        << "Do not invent visual facts. If the image is unavailable or insufficient, say that explicitly.\n"
        << "Do not repeat identical parameter changes that previous turns already tried or suggested; propose a new diagnostic or a different parameter group instead.\n"
        << "Return exactly one JSON object, no markdown.\n\n"
        << "Schema:\n"
        << "{\n"
        << "  \"schema_version\":\"pi.run-chat-answer.v1\",\n"
        << "  \"summary\": string,\n"
        << "  \"likely_causes\": [{\"text\":string,\"evidence_ref\":string}],\n"
        << "  \"checks\": [{\"text\":string,\"evidence_ref\":string}],\n"
        << "  \"recommendations\": [{\"text\":string,\"evidence_ref\":string}],\n"
        << "  \"resume_recommendation\": {\"from_phase\":string,\"confidence\":string,\"reason\":string},\n"
        << "  \"action_plan\": {\"schema_version\":\"pi.action-plan.v1\",\"source\":\"pi.run-chat.provider\",\"run_id\":string,\"mutation_free\":true,\"actions\":[{\"id\":string,\"type\":\"config.set\",\"path\":string,\"value\":any,\"rationale\":string}]},\n"
        << "  \"image_observations\": string[],\n"
        << "  \"warnings\": string[]\n"
        << "}\n\n"
        << "ACTION PLAN RULES:\n"
        << "- Put every concrete parameter recommendation with an exact config path and exact value into action_plan.actions.\n"
        << "- Do not put vague changes into action_plan. Keep those only in recommendations.\n"
        << "- If you recommend disabling/enabling a config path, encode it as false/true in action_plan.\n"
        << "- If you recommend a numeric range or alternatives, choose one conservative primary value for action_plan and mention alternatives in rationale.\n"
        << "- Use only config.set actions unless multiple values are inseparable.\n"
        << "- Do not repeat identical parameter/value pairs already present in PREVIOUS TURNS.\n\n"
        << "USER QUESTION:\n" << message << "\n\n"
        << "RUN ID:\n" << run_id << "\n\n"
        << "TARGET CONTEXT:\n" << target_context.dump(2) << "\n\n"
        << "IMAGE CONTEXT:\n" << image_info.dump(2) << "\n\n"
        << "CONVERSATION MESSAGES:\n" << conversation_messages.dump(2) << "\n\n"
        << "PREVIOUS TURNS:\n" << previous_turns.dump(2) << "\n\n"
        << "RUN STATUS:\n" << status.dump(2).substr(0, 20000) << "\n\n"
        << "ARTIFACT SUMMARY:\n" << artifacts.dump(2).substr(0, 20000) << "\n\n"
        << "LOCAL PI STRUCTURED CONTEXT (use as hints only, verify against image/context):\n"
        << local_answer.dump(2).substr(0, 30000) << "\n";
    return prompt.str();
}

nlohmann::json merge_provider_run_chat_answer(nlohmann::json local_answer, const nlohmann::json& provider_answer) {
    if (!provider_answer.is_object()) return local_answer;
    for (const auto* key : {"summary", "likely_causes", "checks", "recommendations", "resume_recommendation"}) {
        if (provider_answer.contains(key)) local_answer[key] = provider_answer[key];
    }
    local_answer["mode"] = "provider";
    local_answer["provider_meta"] = provider_answer.value("_meta", nlohmann::json::object());
    if (provider_answer.contains("action_plan") && provider_answer["action_plan"].is_object()) {
        nlohmann::json provider_plan = provider_answer["action_plan"];
        provider_plan["schema_version"] = provider_plan.value("schema_version", "pi.action-plan.v1");
        provider_plan["source"] = provider_plan.value("source", "pi.run-chat.provider");
        provider_plan["mutation_free"] = provider_plan.value("mutation_free", true);
        if (!provider_plan.contains("run_id") && local_answer.contains("run_id")) provider_plan["run_id"] = local_answer["run_id"];
        const auto validation = tile_compile::pi::validate_action_plan_shape(provider_plan);
        if (validation.value("valid", false)) {
            nlohmann::json merged_plan = local_answer.value("action_plan", nlohmann::json::object());
            if (!merged_plan.is_object()) merged_plan = nlohmann::json::object();
            merged_plan["schema_version"] = merged_plan.value("schema_version", "pi.action-plan.v1");
            merged_plan["source"] = "pi.run-chat.merged";
            merged_plan["mutation_free"] = merged_plan.value("mutation_free", true);
            if (!merged_plan.contains("run_id") && local_answer.contains("run_id")) merged_plan["run_id"] = local_answer["run_id"];
            if (!merged_plan.contains("actions") || !merged_plan["actions"].is_array()) merged_plan["actions"] = nlohmann::json::array();

            std::set<std::string> seen;
            auto update_key = [](const nlohmann::json& action) {
                if (action.value("type", std::string()) != "config.set" ||
                    !action.contains("path") || !action["path"].is_string() ||
                    !action.contains("value")) {
                    return std::string();
                }
                return action["path"].get<std::string>() + "=" + action["value"].dump();
            };
            for (const auto& action : merged_plan["actions"]) {
                const auto key = update_key(action);
                if (!key.empty()) seen.insert(key);
            }
            for (const auto& action : provider_plan.value("actions", nlohmann::json::array())) {
                const auto key = update_key(action);
                if (!key.empty() && seen.find(key) == seen.end()) {
                    merged_plan["actions"].push_back(action);
                    seen.insert(key);
                }
            }
            const auto merged_validation = tile_compile::pi::validate_action_plan_shape(merged_plan);
            local_answer["action_plan"] = merged_plan;
            local_answer["action_plan_validation"] = merged_validation;
            local_answer["provider_action_plan_validation"] = validation;
        } else {
            local_answer["provider_action_plan_rejected"] = validation;
        }
    }
    if (provider_answer.contains("image_observations")) local_answer["image_observations"] = provider_answer["image_observations"];
    if (provider_answer.contains("warnings")) local_answer["warnings"] = provider_answer["warnings"];
    return local_answer;
}

std::string lower_copy(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

bool contains_any(const std::string& haystack, std::initializer_list<const char*> needles) {
    for (const char* needle : needles) {
        if (haystack.find(needle) != std::string::npos) return true;
    }
    return false;
}

nlohmann::json detect_run_chat_problem_hints(const std::string& message) {
    const std::string text = lower_copy(message);
    nlohmann::json hints = nlohmann::json::array();
    auto add = [&](const std::string& id, const std::string& label, const std::string& confidence) {
        hints.push_back({{"id", id}, {"label", label}, {"confidence", confidence}});
    };
    if (contains_any(text, {
            "schwarzen kern", "schwarzer kern", "schwarze kerne", "schwarzen kernen",
            "schwarzes zentrum", "schwarzen zentrum", "schwarze zentren", "schwarzen zentren",
            "dunkles zentrum", "dunkle zentren", "dunklen zentren",
            "black core", "black cores", "black center", "black centers", "black centre", "black centres",
            "donut", "sternkern", "sternkerne"
        })) {
        add("black_star_cores", "Sterne mit dunklem/schwarzem Kern", "high");
    }
    if (contains_any(text, {"beschnitten", "abgeschnitten", "cropped", "crop", "nicht einbezogen", "outside", "rand"})) {
        add("cropped_nebula", "Nebel oder Randstruktur wirkt beschnitten", "high");
    }
    if (contains_any(text, {"kaum sichtbar", "zu dunkel", "dunkel", "faint", "too dark", "nebula not visible", "nebel"})) {
        add("faint_nebula", "Nebelanteile sind zu schwach sichtbar", "medium");
    }
    if (contains_any(text, {"gradient", "hintergrund", "background", "vignette", "vignett"})) {
        add("background_gradient", "Hintergrundgradient oder Vignettierung", "medium");
    }
    if (contains_any(text, {"farbstich", "gruen", "magenta", "color cast", "colour cast", "farbe"})) {
        add("color_cast", "Farbstich oder unausgewogene Farbe", "medium");
    }
    if (contains_any(text, {"tile", "kachel", "muster", "pattern", "seam", "naht"})) {
        add("tile_pattern", "Tile-/Kachelmuster sichtbar", "medium");
    }
    if (contains_any(text, {"unscharf", "blur", "soft", "fwhm", "elongated", "eier", "verzogen"})) {
        add("soft_or_elongated_stars", "Sterne unscharf oder verzogen", "medium");
    }
    if (hints.empty()) {
        add("general_quality_issue", "Allgemeines sichtbares Qualitaetsproblem", "low");
    }
    return hints;
}

nlohmann::json append_text_item(const std::string& text, const std::string& evidence = "") {
    nlohmann::json item = {{"text", text}};
    if (!evidence.empty()) item["evidence_ref"] = evidence;
    return item;
}

bool hints_contain(const nlohmann::json& hints, const std::string& id) {
    if (!hints.is_array()) return false;
    for (const auto& hint : hints) {
        if (hint.is_object() && hint.value("id", std::string()) == id) return true;
    }
    return false;
}

void add_hint_once(nlohmann::json& hints,
                   const std::string& id,
                   const std::string& label,
                   const std::string& confidence,
                   const std::string& source = "") {
    if (!hints.is_array() || hints_contain(hints, id)) return;
    nlohmann::json hint = {{"id", id}, {"label", label}, {"confidence", confidence}};
    if (!source.empty()) hint["source"] = source;
    hints.push_back(hint);
}

nlohmann::json augment_run_chat_hints_from_context(nlohmann::json hints,
                                                   const nlohmann::json& report,
                                                   const nlohmann::json& artifacts,
                                                   const std::string& run_log_tail = "") {
    const std::string context = lower_copy(report.dump() + "\n" + artifacts.dump() + "\n" + run_log_tail);
    if (context.find("empty_valid_crop") != std::string::npos ||
        context.find("crop_to_nonzero_bbox produced empty valid canvas") != std::string::npos) {
        add_hint_once(hints, "empty_valid_crop",
                      "Stacking bricht wegen leerem Crop-Fenster ab",
                      "high", "artifacts");
        add_hint_once(hints, "cropped_nebula",
                      "Nebel oder Randstruktur wirkt beschnitten",
                      "high", "artifacts");
    }
    return hints;
}

std::string run_chat_log_tail(const std::shared_ptr<AppState>& state, const std::string& run_id, int tail = 120) {
    try {
        const fs::path run_dir = state->runtime.resolve_run_dir(run_id);
        return read_run_logs(run_dir, tail);
    } catch (...) {
        return {};
    }
}

std::string action_update_key(const std::string& path, const nlohmann::json& value) {
    return path + "\n" + value.dump();
}

std::set<std::string> previous_run_chat_action_updates(const nlohmann::json& previous_turns) {
    std::set<std::string> out;
    if (!previous_turns.is_array()) return out;
    for (const auto& turn : previous_turns) {
        if (!turn.is_object()) continue;
        const nlohmann::json* plan_ptr = nullptr;
        if (turn.contains("action_plan") && turn["action_plan"].is_object()) {
            plan_ptr = &turn["action_plan"];
        } else if (turn.contains("result") && turn["result"].is_object() &&
                   turn["result"].contains("action_plan") && turn["result"]["action_plan"].is_object()) {
            plan_ptr = &turn["result"]["action_plan"];
        }
        if (!plan_ptr) continue;
        const auto& plan = *plan_ptr;
        if (!plan.contains("actions") || !plan["actions"].is_array()) continue;
        for (const auto& action : plan["actions"]) {
            if (!action.is_object()) continue;
            const std::string type = action.value("type", std::string());
            if (type == "config.set" && action.contains("path") && action["path"].is_string() && action.contains("value")) {
                out.insert(action_update_key(action["path"].get<std::string>(), action["value"]));
            } else if (type == "config.patch" && action.contains("updates") && action["updates"].is_array()) {
                for (const auto& update : action["updates"]) {
                    if (!update.is_object() || !update.contains("path") || !update["path"].is_string() || !update.contains("value")) continue;
                    out.insert(action_update_key(update["path"].get<std::string>(), update["value"]));
                }
            }
        }
    }
    return out;
}

bool message_reports_ineffective_result(const std::string& message) {
    std::string text = message;
    std::transform(text.begin(), text.end(), text.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    for (const std::string marker : {
             "gleich", "keine verbesserung", "nicht besser", "unveraendert",
             "unverändert", "same result", "no improvement", "unchanged", "not better"
         }) {
        if (text.find(marker) != std::string::npos) return true;
    }
    return false;
}

nlohmann::json suppressed_actions_as_updates(const nlohmann::json& action_plan) {
    nlohmann::json updates = nlohmann::json::array();
    if (!action_plan.is_object()) return updates;
    const nlohmann::json actions = action_plan.contains("suppressed_repeated_actions") &&
        action_plan["suppressed_repeated_actions"].is_array()
        ? action_plan["suppressed_repeated_actions"]
        : (action_plan.contains("actions") && action_plan["actions"].is_array()
            ? action_plan["actions"]
            : nlohmann::json::array());
    if (!actions.is_array()) {
        return updates;
    }
    for (const auto& action : actions) {
        if (!action.is_object() || !action.contains("path") || !action["path"].is_string() || !action.contains("value")) continue;
        updates.push_back({
            {"path", action["path"]},
            {"value", action["value"]},
            {"reason", action.value("reason", std::string("ineffective_repeated_suggestion"))}
        });
    }
    return updates;
}

nlohmann::json maybe_record_negative_run_chat_memory(const std::shared_ptr<AppState>& state,
                                                     const std::string& run_id,
                                                     const std::string& message,
                                                     const nlohmann::json& answer) {
    if (!message_reports_ineffective_result(message)) return nlohmann::json(nullptr);
    nlohmann::json updates = suppressed_actions_as_updates(answer.value("action_plan", nlohmann::json::object()));
    if (updates.empty() && answer.contains("context") && answer["context"].is_object() &&
        answer["context"].contains("conversation") && answer["context"]["conversation"].is_object() &&
        answer["context"]["conversation"].contains("previous_turns") &&
        answer["context"]["conversation"]["previous_turns"].is_array()) {
        for (auto it = answer["context"]["conversation"]["previous_turns"].rbegin();
             it != answer["context"]["conversation"]["previous_turns"].rend(); ++it) {
            if (!it->is_object() || !it->contains("result") || !(*it)["result"].is_object()) continue;
            updates = suppressed_actions_as_updates((*it)["result"].value("action_plan", nlohmann::json::object()));
            if (updates.is_array() && !updates.empty()) break;
        }
    }
    if (!updates.is_array() || updates.empty()) return nlohmann::json(nullptr);

    nlohmann::json affected_paths = nlohmann::json::array();
    for (const auto& update : updates) {
        if (update.is_object() && update.contains("path") && update["path"].is_string()) {
            affected_paths.push_back(update["path"]);
        }
    }
    nlohmann::json problem_ids = nlohmann::json::array();
    if (answer.contains("context") && answer["context"].is_object() &&
        answer["context"].contains("problem_hints") && answer["context"]["problem_hints"].is_array()) {
        for (const auto& hint : answer["context"]["problem_hints"]) {
            if (hint.is_object() && hint.contains("id") && hint["id"].is_string()) problem_ids.push_back(hint["id"]);
        }
    }

    tile_compile::pi::PiMemoryStore store(tile_compile::pi::pi_storage_dir(state));
    nlohmann::json memory = store.append_candidate({
        {"type", "config_optimization"},
        {"status", "candidate"},
        {"privacy_class", "metadata_only"},
        {"source", "pi.run-chat.negative-learning"},
        {"summary", "Run-Chat feedback reported no improvement for repeated parameter suggestions."},
        {"context_signature", {
            {"schema_version", "pi.context_signature.v1"},
            {"target", nlohmann::json::object()},
            {"acquisition", nlohmann::json::object()},
            {"pipeline", {{"affected_paths", affected_paths}}},
            {"problem", {{"classes", problem_ids}, {"hints", problem_ids}}}
        }},
        {"scope", {
            {"applies_when", nlohmann::json::array({"same_run_chat_problem_context_and_same_parameter_values"})},
            {"does_not_apply_when", nlohmann::json::array({"new evidence or materially different acquisition context"})},
            {"confidence", 0.7}
        }},
        {"config_updates", updates},
        {"recommendation", {
            {"avoid_repeating", updates},
            {"explanation", "User feedback indicated the same recommendation did not improve the result."}
        }},
        {"evidence", {
            {"run_id", run_id},
            {"human_feedback", message},
            {"source", "run_chat_followup"}
        }},
        {"outcome", {
            {"schema_version", "pi.memory-outcome.v1"},
            {"verdict", "unchanged"},
            {"verified", true},
            {"human_feedback", message},
            {"applied_count", updates.size()},
            {"applied_paths", affected_paths}
        }},
        {"retrieval", {
            {"keywords", affected_paths},
            {"negative", true}
        }}
    });
    const std::string memory_id = memory.value("memory_id", std::string());
    if (!memory_id.empty() && memory.value("created", true)) {
        store.review(memory_id, "rejected", "pi_negative_learning",
                     "User reported no improvement for repeated run-chat suggestion.",
                     memory["outcome"], memory["scope"]);
    }
    return memory;
}

nlohmann::json build_run_chat_action_plan(const std::string& run_id,
                                          const nlohmann::json& hints,
                                          const nlohmann::json& previous_turns = nlohmann::json::array()) {
    nlohmann::json actions = nlohmann::json::array();
    nlohmann::json suppressed = nlohmann::json::array();
    const auto previous_updates = previous_run_chat_action_updates(previous_turns);
    int index = 1;
    auto add_set = [&](const std::string& path, const nlohmann::json& value, const std::string& rationale, bool force_repeat = false) {
        const std::string key = action_update_key(path, value);
        if (!force_repeat && previous_updates.find(key) != previous_updates.end()) {
            suppressed.push_back({
                {"type", "config.set"},
                {"path", path},
                {"value", value},
                {"reason", "same_parameter_value_was_already_suggested_in_this_run_chat"}
            });
            return;
        }
        actions.push_back({
            {"id", "run_chat_" + std::to_string(index++)},
            {"type", "config.set"},
            {"path", path},
            {"value", value},
            {"rationale", rationale}
        });
    };

    for (const auto& hint : hints) {
        const std::string id = hint.value("id", std::string());
        if (id == "empty_valid_crop") {
            add_set("output.crop_to_nonzero_bbox", false,
                    "Der letzte Resume scheiterte in STACKING mit empty_valid_crop; automatisches Crop muss fuer den naechsten Resume deaktiviert werden.",
                    true);
        } else if (id == "cropped_nebula") {
            add_set("output.crop_to_nonzero_bbox", false,
                    "Wenn Nebel am Rand abgeschnitten wirkt, zuerst ohne automatisches Crop testen.");
        } else if (id == "faint_nebula") {
            add_set("bge.enabled", false,
                    "Bei ausgedehntem Nebel kann Hintergrundextraktion echte schwache Nebelanteile abschwaechen.");
            add_set("normalization.mode", "median",
                    "Median-Normalisierung ist fuer ausgedehnte Nebel oft konservativer als Hintergrund-Normalisierung.");
        } else if (id == "black_star_cores") {
            add_set("stacking.cosmetic_correction", false,
                    "Dunkle Sternkerne koennen durch zu aggressive kosmetische Korrektur/Rejection entstehen; als A/B-Test deaktivieren.");
        } else if (id == "tile_pattern") {
            add_set("tile.overlap_fraction", 0.35,
                    "Mehr Tile-Overlap kann sichtbare Kacheluebergaenge reduzieren.");
        }
    }

    return {
        {"schema_version", "pi.action-plan.v1"},
        {"source", "pi.run-chat"},
        {"run_id", run_id},
        {"mutation_free", true},
        {"actions", actions},
        {"suppressed_repeated_actions", suppressed}
    };
}

nlohmann::json build_resume_recommendation(const nlohmann::json& hints) {
    std::string phase = "HYPERMETRIC_STRETCH";
    std::string reason = "Nur Darstellung/Stretch neu bewerten, wenn keine fruehere Pipeline-Ursache klar ist.";
    std::string execution_note;
    int priority = 90;
    auto choose = [&](int candidate_priority, const std::string& candidate_phase,
                      const std::string& candidate_reason,
                      const std::string& candidate_execution_note = "") {
        if (candidate_priority >= priority) return;
        priority = candidate_priority;
        phase = candidate_phase;
        reason = candidate_reason;
        execution_note = candidate_execution_note;
    };
    for (const auto& hint : hints) {
        const std::string id = hint.value("id", std::string());
        if (id == "empty_valid_crop") {
            choose(1, "STACKING",
                   "Der letzte Resume brach in STACKING mit empty_valid_crop ab; nach Deaktivieren von output.crop_to_nonzero_bbox reicht Resume ab STACKING.");
        } else if (id == "soft_or_elongated_stars") {
            choose(5, "REGISTRATION", "Sternform und Schaerfe haengen oft an Registrierung, Frame-Gewichtung oder fruehen Geometrieschritten.");
        }
        if (id == "cropped_nebula") {
            choose(10, "COMMON_OVERLAP",
                   "Beschnitt und gueltiger Bildbereich entstehen vor Stack/Stretch; daher ab Common-Overlap neu testen.",
                   "COMMON_OVERLAP ist im Runner ein In-place-Full-Rerun; im Log koennen deshalb fruehere Pipeline-Phasen auftauchen.");
        } else if (id == "tile_pattern") {
            choose(20, "TILE_RECONSTRUCTION", "Tile-Muster entstehen in lokaler Rekonstruktion; ab Tile-Reconstruction neu rechnen.");
        } else if (id == "black_star_cores") {
            choose(30, "STACKING", "Dunkle Sternkerne koennen durch Rejection/Kosmetik/Stacking entstehen; ab Stacking ist der kleinste sinnvolle A/B-Test.");
        } else if (id == "background_gradient" || id == "faint_nebula") {
            choose(50, "BGE", "Hintergrundextraktion und Nebelsichtbarkeit werden ab BGE/Stretch entschieden; ab BGE neu testen.");
        } else if (id == "color_cast") {
            choose(60, "PCC", "Farbstich sollte zuerst ab PCC/Farbkalibrierung neu bewertet werden.");
        }
    }
    nlohmann::json out = {{"from_phase", phase}, {"confidence", "medium"}, {"reason", reason}};
    if (!execution_note.empty()) out["execution_note"] = execution_note;
    return out;
}

nlohmann::json build_run_chat_answer(const std::shared_ptr<AppState>& state,
                                     const std::string& run_id,
                                     const std::string& message,
                                     const std::string& analysis_message,
                                     const nlohmann::json& conversation_messages = nlohmann::json::array(),
                                     const nlohmann::json& previous_turns = nlohmann::json::array(),
                                     const nlohmann::json& target_context = nlohmann::json::object()) {
    tile_compile::pi::PiToolRegistry tools(state);
    nlohmann::json report = tools.call_tool("run.report.summary", {{"run_id", run_id}});
    nlohmann::json artifacts = tools.call_tool("run.artifacts.summary", {{"run_id", run_id}});
    const std::string log_tail = run_chat_log_tail(state, run_id);
    const nlohmann::json hints = augment_run_chat_hints_from_context(
        detect_run_chat_problem_hints(analysis_message.empty() ? message : analysis_message),
        report,
        artifacts,
        log_tail);

    tile_compile::pi::PiMemoryStore store(tile_compile::pi::pi_storage_dir(state));
    nlohmann::json memories = store.retrieve({{"type", "config_optimization"}}, 5);

    nlohmann::json evidence = nlohmann::json::array({
        {{"id", "report"}, {"tool", "run.report.summary"}, {"available", report.value("ok", false)}, {"result", report.value("result", nlohmann::json::object())}},
        {{"id", "artifacts"}, {"tool", "run.artifacts.summary"}, {"available", artifacts.value("ok", false)}, {"result", artifacts.value("result", nlohmann::json::object())}},
        {{"id", "run_log_tail"}, {"tool", "run.logs.tail"}, {"available", !log_tail.empty()}, {"result", log_tail.substr(0, 20000)}},
        {{"id", "memories"}, {"tool", "pi.memory.retrieve"}, {"available", !memories.empty()}, {"result", memories}}
    });
    if (previous_turns.is_array() && !previous_turns.empty()) {
        evidence.push_back({
            {"id", "conversation"},
            {"tool", "pi.run-chat.history"},
            {"available", true},
            {"result", previous_turns}
        });
    }

    nlohmann::json likely_causes = nlohmann::json::array();
    nlohmann::json checks = nlohmann::json::array();
    nlohmann::json recommendations = nlohmann::json::array();

    for (const auto& hint : hints) {
        const std::string id = hint.value("id", std::string());
        if (id == "empty_valid_crop") {
            likely_causes.push_back(append_text_item(
                "Der letzte Resume scheiterte in STACKING, weil crop_to_nonzero_bbox nach AQMH-Reconstruction kein gueltiges Nicht-Null-Crop-Fenster mehr fand.",
                "artifacts"));
            checks.push_back(append_text_item(
                "Pruefe STACKING phase_end und AQMH-Reconstruction-Warnungen: empty_valid_crop, unsupported_pixels und post-clipping numerical guard zeigen, dass automatisches Crop hier nicht belastbar ist.",
                "artifacts"));
            recommendations.push_back(append_text_item(
                "Vor dem naechsten Resume `output.crop_to_nonzero_bbox=false` setzen und ab `STACKING` neu rechnen.",
                "artifacts"));
        } else if (id == "black_star_cores") {
            likely_causes.push_back(append_text_item(
                "Dunkle Sternkerne passen zu zu aggressiver kosmetischer Korrektur, Sigma-Rejection, lokaler Hintergrundbehandlung oder Stretch/Star-Protect-Artefakten.",
                "report"));
            checks.push_back(append_text_item(
                "Vergleiche lineares Stack, gestretchtes Ergebnis und ggf. Zwischenergebnisse vor/nach kosmetischer Korrektur und Rejection.",
                "artifacts"));
            recommendations.push_back(append_text_item(
                "Als A/B-Test kosmetische Korrektur oder Rejection weniger aggressiv setzen und nur ab betroffener Phase neu rechnen.",
                "report"));
        } else if (id == "cropped_nebula") {
            likely_causes.push_back(append_text_item(
                "Beschnittener Nebel deutet auf Crop-to-nonzero-BBox, Common-Overlap nach Registrierung oder ein zu enges gueltiges Rekonstruktionsfenster hin.",
                "artifacts"));
            checks.push_back(append_text_item(
                "Pruefe common_overlap, Registration-Artefakte und ob das finale Output kleiner als die registrierten Frames ist.",
                "artifacts"));
            recommendations.push_back(append_text_item(
                "Testweise `output.crop_to_nonzero_bbox=false` und Common-Overlap/Registrierungsdiagnostik pruefen.",
                "report"));
        } else if (id == "faint_nebula") {
            likely_causes.push_back(append_text_item(
                "Schwacher Nebel kann durch Hintergrundextraktion, Hintergrund-Normalisierung oder zu dunklen Stretch-Zielhintergrund entstehen.",
                "report"));
            checks.push_back(append_text_item(
                "Pruefe BGE-Report, Hintergrundkarten, Histogramm/Stretch-Parameter und ob ausgedehnte Emission als Hintergrund behandelt wurde.",
                "artifacts"));
            recommendations.push_back(append_text_item(
                "Bei M42/ausgedehnten Nebeln konservativ testen: BGE aus, Median-Normalisierung, danach Stretch neu bewerten.",
                "memories"));
        } else if (id == "background_gradient") {
            likely_causes.push_back(append_text_item(
                "Gradienten koennen aus Vignettierung, Mond/Light-Pollution, fehlenden Flats oder BGE-Unter-/Ueberfit stammen.",
                "report"));
            checks.push_back(append_text_item(
                "BGE-Diagnostik und Flat-/Kalibrierstatus pruefen; nicht automatisch Nebel als Gradient wegfitten.",
                "artifacts"));
            recommendations.push_back(append_text_item(
                "BGE nur mit konservativen Masken/Validierung verwenden und Ergebnis gegen BGE-off vergleichen.",
                "report"));
        } else if (id == "color_cast") {
            likely_causes.push_back(append_text_item(
                "Farbstich passt zu Bayer-Pattern, PCC-Sternauswahl, Hintergrundneutralisierung oder starker Gradientenbehandlung.",
                "report"));
            checks.push_back(append_text_item(
                "PCC-Report, Bayer-Pattern und Background-Neutralization-Status pruefen.",
                "artifacts"));
            recommendations.push_back(append_text_item(
                "PCC-Parameter und Bayer-Pattern erst anhand Report/Headers bestaetigen, dann gezielt neu rechnen.",
                "report"));
        } else if (id == "tile_pattern") {
            likely_causes.push_back(append_text_item(
                "Tile-Muster spricht fuer zu wenig Overlap, zu starke lokale Gewichtung oder inkonsistente lokale Rekonstruktion.",
                "report"));
            checks.push_back(append_text_item(
                "AQMH-/Tile-Artefakte, lokale Metrikkarten und Rekonstruktionsdiagnostik pruefen.",
                "artifacts"));
            recommendations.push_back(append_text_item(
                "Tile-Overlap erhoehen und lokale Regularisierung/Tile-Groesse gegenpruefen.",
                "report"));
        } else if (id == "soft_or_elongated_stars") {
            likely_causes.push_back(append_text_item(
                "Weiche oder verzogene Sterne passen zu Registrierungsfehlern, Seeing-Streuung, Fokusdrift oder falscher Frame-Gewichtung.",
                "report"));
            checks.push_back(append_text_item(
                "Registration-Report, FWHM-Verlauf und verworfene/gewichtete Frames pruefen.",
                "artifacts"));
            recommendations.push_back(append_text_item(
                "Registration und Qualitätsgewichtung vor Stretch/Color-Fixes validieren.",
                "report"));
        }
    }

    if (recommendations.empty()) {
        recommendations.push_back(append_text_item(
            "Zuerst Report und Artefakte pruefen, dann nur eine Parametergruppe als A/B-Test aendern.",
            "report"));
    }

    const nlohmann::json action_plan = build_run_chat_action_plan(run_id, hints, previous_turns);
    if (action_plan.contains("suppressed_repeated_actions") &&
        action_plan["suppressed_repeated_actions"].is_array() &&
        !action_plan["suppressed_repeated_actions"].empty()) {
        recommendations.push_back(append_text_item(
            "Ich schlage dieselben Parameterwerte nicht erneut vor, weil sie in diesem Run-Chat bereits empfohlen wurden. Wenn das Ergebnis gleich geblieben ist, sollten stattdessen Report/Artefakte verglichen und eine andere Ursache oder Parametergruppe getestet werden.",
            "conversation"));
    }
    const long previous_turn_count = previous_turns.is_array() ? static_cast<long>(previous_turns.size()) : 0L;
    const std::string summary = previous_turn_count > 0
        ? "Ich behandle das als Folgefrage im bisherigen Run-Chat und beziehe die vorherigen Hinweise, Empfehlungen und Resume-Phase mit ein."
        : "Ich behandle die Beschreibung als Hinweis, nicht als bewiesene Ursache. Die naechsten Schritte sollten Report, Artefakte und gezielte A/B-Tests verbinden.";
    return {
        {"schema_version", "pi.run-chat-answer.v1"},
        {"mode", "local_read_only"},
        {"question", message},
        {"run_id", run_id},
        {"context", {
            {"schema_version", "pi.run-chat-context.v1"},
            {"run_id", run_id},
            {"target", target_context},
            {"problem_hints", hints},
            {"report_available", report.value("ok", false)},
            {"artifacts_available", artifacts.value("ok", false)},
            {"memory_count", memories.size()},
            {"conversation", {
                {"message_count", conversation_messages.is_array() ? conversation_messages.size() : 0},
                {"previous_turn_count", previous_turn_count},
                {"previous_turns", previous_turns}
            }}
        }},
        {"summary", summary},
        {"likely_causes", likely_causes},
        {"checks", checks},
        {"recommendations", recommendations},
        {"evidence", evidence},
        {"resume_recommendation", build_resume_recommendation(hints)},
        {"action_plan", action_plan},
        {"action_plan_validation", tile_compile::pi::validate_action_plan_shape(action_plan)}
    };
}

std::string run_chat_analysis_message(const nlohmann::json& body, const std::string& message) {
    std::string combined;
    if (body.contains("messages") && body["messages"].is_array()) {
        int count = 0;
        for (const auto& item : body["messages"]) {
            if (!item.is_object()) continue;
            const std::string content = item.value("content", std::string());
            if (content.empty()) continue;
            if (!combined.empty()) combined += "\n";
            combined += content;
            if (++count >= 12) break;
        }
    }
    if (combined.find(message) == std::string::npos) {
        if (!combined.empty()) combined += "\n";
        combined += message;
    }
    return combined;
}

nlohmann::json pi_audit_log(const std::shared_ptr<AppState>& state, int limit) {
    nlohmann::json items = nlohmann::json::array();
    const int event_scan_limit = std::max(limit, std::min(10000, limit * 10));
    for (const auto& event : state->ui_event_store.list(0, std::max(1, event_scan_limit))) {
        if (event.event.rfind("pi.", 0) != 0 && event.event.rfind("config.ai.", 0) != 0) continue;
        nlohmann::json item = ui_event_to_json(event);
        item["audit_type"] = event.event.rfind("pi.", 0) == 0 ? "pi_event" : "config_ai_event";
        items.push_back(std::move(item));
    }

    tile_compile::pi::PiMemoryStore store(tile_compile::pi::pi_storage_dir(state));
    for (const auto& memory : store.list(100000)) {
        items.push_back({
            {"audit_type", "memory_candidate"},
            {"memory_id", memory.value("memory_id", std::string())},
            {"type", memory.value("type", std::string())},
            {"status", memory.value("status", std::string("candidate"))},
            {"created_at", memory.value("created_at", std::string())},
            {"source", memory.value("source", std::string())},
            {"analysis_id", memory.value("analysis_id", std::string())},
            {"summary", memory.value("summary", std::string())}
        });
        if (!memory.contains("review") || !memory["review"].is_object()) continue;
        nlohmann::json item = {
            {"audit_type", "memory_review"},
            {"memory_id", memory.value("memory_id", std::string())},
            {"type", memory.value("type", std::string())},
            {"status", memory.value("status", std::string())},
            {"created_at", memory.value("created_at", std::string())},
            {"review", memory["review"]},
            {"summary", memory.value("summary", std::string())}
        };
        items.push_back(std::move(item));
    }
    trim_json_array_to_latest(items, limit);

    return {
        {"schema_version", "pi.audit.v1"},
        {"privacy_class", "metadata_only"},
        {"items", items},
        {"count", items.size()},
        {"latest_event_seq", state->ui_event_store.latest_seq()}
    };
}

} // namespace

namespace {

nlohmann::json preview_action_plan(const nlohmann::json& plan) {
    nlohmann::json updates = nlohmann::json::array();
    nlohmann::json actions = plan.contains("actions") && plan["actions"].is_array()
        ? plan["actions"]
        : nlohmann::json::array();
    for (const auto& action : actions) {
        if (!action.is_object()) continue;
        const std::string type = action.value("type", std::string());
        if (type == "config.set" && action.contains("path") && action["path"].is_string() && action.contains("value")) {
            updates.push_back({
                {"path", action["path"]},
                {"value", action["value"]},
                {"action_id", action.value("id", std::string())},
                {"rationale", action.value("rationale", std::string())}
            });
        } else if (type == "config.patch" && action.contains("updates") && action["updates"].is_array()) {
            for (const auto& update : action["updates"]) {
                if (!update.is_object()) continue;
                nlohmann::json item = update;
                item["action_id"] = action.value("id", std::string());
                updates.push_back(std::move(item));
            }
        }
    }
    return {
        {"schema_version", "pi.action-preview.v1"},
        {"mutation_free", true},
        {"action_count", actions.size()},
        {"config_updates", updates},
        {"config_update_count", updates.size()}
    };
}

nlohmann::json yaml_to_json(const YAML::Node& node) {
    if (!node || node.IsNull()) return nullptr;
    if (node.IsMap()) {
        nlohmann::json out = nlohmann::json::object();
        for (auto it = node.begin(); it != node.end(); ++it) out[it->first.as<std::string>()] = yaml_to_json(it->second);
        return out;
    }
    if (node.IsSequence()) {
        nlohmann::json out = nlohmann::json::array();
        for (auto it = node.begin(); it != node.end(); ++it) out.push_back(yaml_to_json(*it));
        return out;
    }
    try { return node.as<bool>(); } catch (...) {}
    try { return node.as<long long>(); } catch (...) {}
    try { return node.as<double>(); } catch (...) {}
    try { return node.as<std::string>(); } catch (...) {}
    return nullptr;
}

YAML::Node json_to_yaml_node(const nlohmann::json& value) {
    if (value.is_object()) {
        YAML::Node node(YAML::NodeType::Map);
        for (auto it = value.begin(); it != value.end(); ++it) {
            if (it.value().is_null()) continue;
            node[it.key()] = json_to_yaml_node(it.value());
        }
        return node;
    }
    if (value.is_array()) {
        YAML::Node node(YAML::NodeType::Sequence);
        for (const auto& item : value) node.push_back(json_to_yaml_node(item));
        return node;
    }
    if (value.is_boolean()) return YAML::Node(value.get<bool>());
    if (value.is_number_integer()) return YAML::Node(value.get<long long>());
    if (value.is_number_unsigned()) return YAML::Node(value.get<unsigned long long>());
    if (value.is_number_float()) return YAML::Node(value.get<double>());
    if (value.is_null()) return YAML::Node();
    return YAML::Node(value.get<std::string>());
}

std::string yaml_dump(const nlohmann::json& value) {
    YAML::Node node = json_to_yaml_node(value);
    std::ostringstream out;
    out << node;
    return out.str();
}

void set_dotted(nlohmann::json& root, const std::string& dotted_path, const nlohmann::json& value) {
    std::vector<std::string> parts;
    std::istringstream iss(dotted_path);
    std::string part;
    while (std::getline(iss, part, '.')) {
        if (!part.empty()) parts.push_back(part);
    }
    if (parts.empty()) return;
    nlohmann::json* node = &root;
    for (size_t i = 0; i + 1 < parts.size(); ++i) {
        if (!node->contains(parts[i]) || !(*node)[parts[i]].is_object()) (*node)[parts[i]] = nlohmann::json::object();
        node = &(*node)[parts[i]];
    }
    (*node)[parts.back()] = value;
}

nlohmann::json load_preview_base_config(const nlohmann::json& body, const std::shared_ptr<AppState>& state) {
    if (body.contains("base_config") && body["base_config"].is_object()) return body["base_config"];
    if (body.contains("config") && body["config"].is_object()) return body["config"];
    if (body.contains("yaml") && body["yaml"].is_string()) {
        return yaml_to_json(YAML::Load(body["yaml"].get<std::string>()));
    }
    std::ifstream in(state->runtime.default_config_path);
    if (!in) return nlohmann::json::object();
    std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    if (text.empty()) return nlohmann::json::object();
    return yaml_to_json(YAML::Load(text));
}

nlohmann::json build_validated_preview(const nlohmann::json& plan,
                                       const nlohmann::json& body,
                                       const std::shared_ptr<AppState>& state) {
    nlohmann::json preview = preview_action_plan(plan);
    nlohmann::json base = load_preview_base_config(body, state);
    if (!base.is_object()) base = nlohmann::json::object();
    nlohmann::json patched = base;
    for (const auto& update : preview["config_updates"]) {
        if (!update.is_object() || !update.contains("path") || !update["path"].is_string() || !update.contains("value")) continue;
        set_dotted(patched, update["path"].get<std::string>(), update["value"]);
    }
    const std::string base_yaml = yaml_dump(base);
    const std::string patched_yaml = yaml_dump(patched);
    SubprocessResult validate_res = run_subprocess({state->runtime.cli_exe, "validate-config", "--stdin"},
                                                   state->runtime.project_root.string(),
                                                   patched_yaml);
    auto validation = nlohmann::json::parse(validate_res.stdout_str, nullptr, false);
    if (validation.is_discarded() || !validation.is_object()) {
        validation = {
            {"valid", false},
            {"errors", nlohmann::json::array({"validate-config returned non-json output"})},
            {"warnings", nlohmann::json::array()}
        };
    }
    preview["base_config"] = base;
    preview["patched_config"] = patched;
    preview["base_yaml"] = base_yaml;
    preview["patched_yaml"] = patched_yaml;
    preview["yaml_changed"] = base_yaml != patched_yaml;
    preview["validation"] = validation;
    preview["config_valid"] = validate_res.exit_code == 0 && validation.value("valid", false);
    return preview;
}

nlohmann::json apply_validated_preview(const nlohmann::json& preview,
                                       const std::shared_ptr<AppState>& state) {
    const std::string patched_yaml = preview.value("patched_yaml", std::string());
    fs::path target = state->runtime.default_config_path;
    SubprocessResult save_res = run_subprocess({state->runtime.cli_exe, "save-config", target.string(), "--stdin"},
                                               state->runtime.project_root.string(),
                                               patched_yaml);
    auto save_payload = nlohmann::json::parse(save_res.stdout_str, nullptr, false);
    if (save_payload.is_discarded() || !save_payload.is_object()) {
        save_payload = nlohmann::json::object();
    }
    if (save_res.exit_code != 0) {
        return {
            {"ok", false},
            {"error", {
                {"code", "BACKEND_COMMAND_FAILED"},
                {"message", "save-config failed"},
                {"exit_code", save_res.exit_code},
                {"stderr", save_res.stderr_str},
                {"stdout", save_res.stdout_str}
            }}
        };
    }
    fs::path saved_path = save_payload.contains("path") && save_payload["path"].is_string()
        ? fs::path(save_payload["path"].get<std::string>())
        : target;
    std::string rev_id = state->revision_store.add(saved_path, patched_yaml, "pi_action_plan");
    {
        std::lock_guard<std::mutex> lk(state->state_mutex);
        state->active_config_revision_id = rev_id;
    }
    state->ui_event_store.push("pi.action_plan.apply", "pi.action_plan_apply", {
        {"path", saved_path.string()},
        {"revision_id", rev_id},
        {"config_update_count", preview.value("config_update_count", 0)}
    });
    return {
        {"ok", true},
        {"saved", save_payload.value("saved", true)},
        {"path", saved_path.string()},
        {"revision_id", rev_id}
    };
}

} // namespace

void tile_compile::routes::register_pi_routes(CrowApp& app, std::shared_ptr<AppState> state) {
    CROW_ROUTE(app, "/api/pi/context").methods("GET"_method)
    ([state]() {
        tile_compile::pi::PiContextBuilder builder(state);
        return json_resp(builder.build_overview_context());
    });

    CROW_ROUTE(app, "/api/pi/tools").methods("GET"_method)
    ([state]() {
        tile_compile::pi::PiToolRegistry registry(state);
        return json_resp(registry.list_tools());
    });

    CROW_ROUTE(app, "/api/pi/tools/call").methods("POST"_method)
    ([state](const crow::request& req) {
        auto body = parse_body(req);
        if (!body) return err_resp("BAD_REQUEST", "Invalid JSON", 400);
        const std::string name = body->value("name", std::string());
        if (name.empty()) return err_resp("BAD_REQUEST", "tool name is required", 400);
        const nlohmann::json input = body->contains("input") && (*body)["input"].is_object()
            ? (*body)["input"]
            : nlohmann::json::object();
        tile_compile::pi::PiToolRegistry registry(state);
        const auto result = registry.call_tool(name, input);
        return json_resp(result, result.value("ok", false) ? 200 : 404);
    });

    CROW_ROUTE(app, "/api/pi/assistant/ask").methods("POST"_method)
    ([state](const crow::request& req) {
        auto body = parse_body(req);
        if (!body) return err_resp("BAD_REQUEST", "Invalid JSON", 400);
        const std::string question = body->value("question", std::string());
        if (question.empty()) return err_resp("BAD_REQUEST", "question is required", 400);
        tile_compile::pi::PiAssistant assistant(state);
        return json_resp(assistant.answer(question));
    });

    CROW_ROUTE(app, "/api/pi/storage").methods("GET"_method)
    ([state](const crow::request&) {
        return json_resp(tile_compile::pi::pi_storage_status(state));
    });

    CROW_ROUTE(app, "/api/pi/storage").methods("POST"_method)
    ([state](const crow::request& req) {
        auto body = parse_body(req);
        if (!body) return err_resp("BAD_REQUEST", "Invalid JSON", 400);
        const std::string storage_dir = body->value("storage_dir", std::string());
        std::filesystem::path resolved;
        std::string error_code;
        std::string error_message;
        if (!tile_compile::pi::set_pi_storage_dir(state, storage_dir, resolved, error_code, error_message)) {
            return err_resp(error_code.empty() ? "BAD_REQUEST" : error_code,
                            error_message.empty() ? "failed to save PI storage directory" : error_message,
                            error_code == "PATH_NOT_ALLOWED" ? 403 : 400);
        }
        state->ui_event_store.push("pi.storage.save", "pi.storage", {{"storage_dir", resolved.string()}});
        return json_resp(tile_compile::pi::pi_storage_status(state));
    });

    CROW_ROUTE(app, "/api/pi/run-chat").methods("POST"_method)
    ([state](const crow::request& req) {
        auto body = parse_body(req);
        if (!body) return err_resp("BAD_REQUEST", "Invalid JSON", 400);
        const std::string run_id = body->value("run_id", std::string());
        const std::string message = body->value("message", std::string());
        const std::string object_name = body->value("object_name", body->value("target", std::string()));
        if (run_id.empty()) return err_resp("BAD_REQUEST", "run_id is required", 400);
        if (message.empty()) return err_resp("BAD_REQUEST", "message is required", 400);
        try {
            nlohmann::json target_context = {
                {"object_name", object_name.empty() ? nlohmann::json(nullptr) : nlohmann::json(object_name)}
            };
            nlohmann::json history = read_pi_run_chat_history(state, run_id);
            history["target"] = target_context;
            nlohmann::json messages = compact_run_chat_history_messages(history, *body, message);
            const nlohmann::json previous_turns = run_chat_previous_turns_context(history);
            nlohmann::json local_answer = build_run_chat_answer(
                state,
                run_id,
                message,
                run_chat_analysis_message_from_messages(messages, message),
                messages,
                previous_turns,
                target_context);

            nlohmann::json answer = local_answer;
            auto ai_config = current_pi_ai_config(state);
            if (!ai_config.model.empty()) {
                const fs::path run_dir = state->runtime.resolve_run_dir(run_id);
                const nlohmann::json status = read_run_status(run_dir);
                const nlohmann::json artifacts = list_run_artifacts(run_dir);
                nlohmann::json image = build_run_chat_preview_image(run_dir);
                nlohmann::json image_info = image;
                image_info.erase("base64");
                nlohmann::json problem_ids = nlohmann::json::array();
                if (local_answer.contains("context") && local_answer["context"].is_object() &&
                    local_answer["context"].contains("problem_hints") &&
                    local_answer["context"]["problem_hints"].is_array()) {
                    for (const auto& hint : local_answer["context"]["problem_hints"]) {
                        if (hint.is_object() && hint.contains("id") && hint["id"].is_string()) {
                            problem_ids.push_back(hint["id"]);
                        }
                    }
                }
                nlohmann::json resume_phases = nlohmann::json::array();
                if (local_answer.contains("resume_recommendation") &&
                    local_answer["resume_recommendation"].is_object() &&
                    local_answer["resume_recommendation"].contains("from_phase")) {
                    resume_phases.push_back(local_answer["resume_recommendation"]["from_phase"]);
                }
                nlohmann::json positive_memories = nlohmann::json::array();
                if (local_answer.contains("evidence") && local_answer["evidence"].is_array()) {
                    for (const auto& item : local_answer["evidence"]) {
                        if (item.is_object() && item.value("id", std::string()) == "memories" &&
                            item.contains("result") && item["result"].is_array()) {
                            positive_memories = item["result"];
                            break;
                        }
                    }
                }
                const nlohmann::json ai_request = tile_compile::pi::build_ai_request_v2({
                    {"task", "run_chat"},
                    {"user_message", message},
                    {"context_signature", {
                        {"schema_version", "pi.context_signature.v1"},
                        {"problem", {
                            {"classes", problem_ids},
                            {"hints", problem_ids}
                        }},
                        {"pipeline", {
                            {"affected_paths", nlohmann::json::array()},
                            {"phases", resume_phases}
                        }}
                    }},
                    {"run_context", {
                        {"run_id", run_id},
                        {"target", target_context},
                        {"status", status},
                        {"local_answer_context", local_answer.value("context", nlohmann::json::object())},
                        {"previous_turns", previous_turns}
                    }},
                    {"artifacts", artifacts},
                    {"image_context", image_info},
                    {"positive_memories", positive_memories},
                    {"conversation", messages},
                    {"expected_response", "pi.run-chat-answer.v1 with diagnosis, checks, recommendations, resume phase and action plan candidates"},
                    {"provider", ai_config.provider},
                    {"model", ai_config.model},
                    {"source_request_schema", "pi.run-chat.request.v1"}
                });
                const std::string prompt = build_provider_run_chat_prompt(
                    run_id, message, local_answer, messages, previous_turns, status, artifacts, image_info, target_context);
                tile_compile::ai::AiSidecarClient client(ai_config);
                nlohmann::json payload = {
                    {"model", ai_config.model},
                    {"prompt", prompt},
                    {"ai_request", ai_request},
                    {"run_id", run_id},
                    {"object_name", object_name},
                    {"image_available", image.value("available", false)},
                    {"image_path", image.value("path", std::string())}
                };
                if (image.value("available", false) && image.contains("base64") && image["base64"].is_string()) {
                    payload["image_base64"] = image["base64"];
                    payload["image_mime"] = image.value("mime", std::string("image/png"));
                }
                const nlohmann::json provider_answer = client.post("/run-chat", payload);
                answer = merge_provider_run_chat_answer(local_answer, provider_answer);
                answer["context"]["image"] = image_info;
            } else {
                answer["mode"] = "local_fallback_no_model_configured";
                answer["warnings"] = nlohmann::json::array({
                    "No AI model is configured; this is a local structured fallback without visual/provider reasoning."
                });
            }
            nlohmann::json negative_learning = maybe_record_negative_run_chat_memory(state, run_id, message, answer);
            if (!negative_learning.is_null()) {
                answer["negative_learning"] = {
                    {"created", negative_learning.value("created", false)},
                    {"duplicate", negative_learning.value("duplicate", false)},
                    {"memory_id", negative_learning.value("memory_id", std::string())}
                };
            }

            append_chat_message_unique(messages, "assistant", answer.value("summary", std::string()));
            history["messages"] = messages;
            if (!history.contains("turns") || !history["turns"].is_array()) history["turns"] = nlohmann::json::array();
            history["turns"].push_back({
                {"message", message},
                {"result", answer},
                {"target", target_context},
                {"created_at", utc_now_iso()}
            });
            write_pi_run_chat_history(state, run_id, history);

            return json_resp(answer);
        } catch (const std::exception& e) {
            return err_resp("RUN_CONTEXT_UNAVAILABLE", e.what(), 400);
        }
    });

    CROW_ROUTE(app, "/api/pi/run-chat/history").methods("GET"_method)
    ([state](const crow::request& req) {
        const std::string run_id = req.url_params.get("run_id") ? std::string(req.url_params.get("run_id")) : "";
        if (run_id.empty()) return err_resp("BAD_REQUEST", "run_id is required", 400);
        try {
            return json_resp(read_pi_run_chat_history(state, run_id));
        } catch (const std::exception& e) {
            return err_resp("RUN_CHAT_HISTORY_UNAVAILABLE", e.what(), 400);
        }
    });

    CROW_ROUTE(app, "/api/pi/run-chat/history").methods("POST"_method)
    ([state](const crow::request& req) {
        auto body = parse_body(req);
        if (!body) return err_resp("BAD_REQUEST", "Invalid JSON", 400);
        const std::string run_id = body->value("run_id", std::string());
        if (run_id.empty()) return err_resp("BAD_REQUEST", "run_id is required", 400);
        try {
            nlohmann::json incoming = body->contains("history") && (*body)["history"].is_object()
                ? (*body)["history"]
                : nlohmann::json::object();
            nlohmann::json history = merge_run_chat_history(read_pi_run_chat_history(state, run_id), incoming);
            write_pi_run_chat_history(state, run_id, history);
            return json_resp(read_pi_run_chat_history(state, run_id));
        } catch (const std::exception& e) {
            return err_resp("RUN_CHAT_HISTORY_SAVE_FAILED", e.what(), 400);
        }
    });

    CROW_ROUTE(app, "/api/pi/memories").methods("GET"_method)
    ([state](const crow::request& req) {
        const int limit = std::max(1, std::min(500, int_query_param(req, "limit", 100)));
        const std::string status_filter = req.url_params.get("status") ? std::string(req.url_params.get("status")) : "";
        tile_compile::pi::PiMemoryStore store(tile_compile::pi::pi_storage_dir(state));
        const bool filtered = !status_filter.empty() && status_filter != "all";
        nlohmann::json items = store.list(filtered ? 100000 : limit);
        if (filtered) {
            nlohmann::json filtered_items = nlohmann::json::array();
            for (const auto& item : items) {
                if (item.value("status", std::string()) == status_filter) filtered_items.push_back(item);
            }
            items = std::move(filtered_items);
            trim_json_array_to_latest(items, limit);
        }
        const fs::path memory_file = store.memories_path();
        const fs::path legacy_memory_file = store.legacy_memories_path();
        const long legacy_count = count_jsonl_records(legacy_memory_file);
        return json_resp({
            {"schema_version", "pi.memories-list.v1"},
            {"memory_dir", store.memory_dir().string()},
            {"memory_file", memory_file.string()},
            {"memory_file_exists", fs::exists(memory_file)},
            {"legacy_memory_file", legacy_memory_file.string()},
            {"legacy_memory_file_exists", fs::exists(legacy_memory_file)},
            {"legacy_ignored_count", legacy_count},
            {"items", items},
            {"count", items.size()}
        });
    });

    CROW_ROUTE(app, "/api/pi/memories/export").methods("GET"_method)
    ([state](const crow::request& req) {
        const std::string privacy = req.url_params.get("privacy")
            ? std::string(req.url_params.get("privacy"))
            : std::string("metadata_only");
        const bool include_reviews = !req.url_params.get("include_reviews") ||
            std::string(req.url_params.get("include_reviews")) != "0";
        tile_compile::pi::PiMemoryStore store(tile_compile::pi::pi_storage_dir(state));
        return json_resp(store.export_bundle(privacy, include_reviews));
    });

    CROW_ROUTE(app, "/api/pi/memories/index").methods("GET"_method)
    ([state](const crow::request&) {
        try {
            tile_compile::pi::PiMemoryStore store(tile_compile::pi::pi_storage_dir(state));
            return json_resp(store.indices());
        } catch (const std::exception& e) {
            return err_resp("MEMORY_INDEX_FAILED", e.what(), 502);
        }
    });

    CROW_ROUTE(app, "/api/pi/memories/import").methods("POST"_method)
    ([state](const crow::request& req) {
        auto body = parse_body(req);
        if (!body) return err_resp("BAD_REQUEST", "Invalid JSON", 400);
        const bool dry_run = body->value("dry_run", false);
        const nlohmann::json bundle = body->contains("bundle") && (*body)["bundle"].is_object()
            ? (*body)["bundle"]
            : *body;
        try {
            tile_compile::pi::PiMemoryStore store(tile_compile::pi::pi_storage_dir(state));
            return json_resp(store.import_bundle(bundle, dry_run));
        } catch (const std::invalid_argument& e) {
            return err_resp("BAD_REQUEST", e.what(), 400);
        } catch (const std::exception& e) {
            return err_resp("BACKEND_COMMAND_FAILED", e.what(), 502);
        }
    });

    CROW_ROUTE(app, "/api/pi/memories/dedupe").methods("POST"_method)
    ([state](const crow::request& req) {
        auto body = parse_body(req);
        if (!body) return err_resp("BAD_REQUEST", "Invalid JSON", 400);
        const bool dry_run = body->value("dry_run", false);
        try {
            tile_compile::pi::PiMemoryStore store(tile_compile::pi::pi_storage_dir(state));
            return json_resp(store.dedupe(dry_run));
        } catch (const std::exception& e) {
            return err_resp("BACKEND_COMMAND_FAILED", e.what(), 502);
        }
    });

    CROW_ROUTE(app, "/api/pi/memories/<string>/review").methods("POST"_method)
    ([state](const crow::request& req, const std::string& memory_id) {
        auto body = parse_body(req);
        if (!body) return err_resp("BAD_REQUEST", "Invalid JSON", 400);
        const std::string status = body->value("status", std::string());
        const std::string reviewer = body->value("reviewer", std::string("user"));
        const std::string note = body->value("note", std::string());
        const nlohmann::json outcome = body->contains("outcome") && (*body)["outcome"].is_object()
            ? (*body)["outcome"]
            : nlohmann::json::object();
        const nlohmann::json scope = body->contains("scope") && (*body)["scope"].is_object()
            ? (*body)["scope"]
            : nlohmann::json::object();
        if (status.empty()) return err_resp("BAD_REQUEST", "status is required", 400);
        try {
            tile_compile::pi::PiMemoryStore store(tile_compile::pi::pi_storage_dir(state));
            const auto review = store.review(memory_id, status, reviewer, note, outcome, scope);
            return json_resp({{"ok", true}, {"review", review}});
        } catch (const std::invalid_argument& e) {
            return err_resp("BAD_REQUEST", e.what(), 400);
        } catch (const std::exception& e) {
            return err_resp("BACKEND_COMMAND_FAILED", e.what(), 502);
        }
    });

    CROW_ROUTE(app, "/api/pi/memories/<string>/outcome").methods("POST"_method)
    ([state](const crow::request& req, const std::string& memory_id) {
        auto body = parse_body(req);
        if (!body) return err_resp("BAD_REQUEST", "Invalid JSON", 400);
        try {
            const nlohmann::json outcome = evaluate_memory_outcome_payload(*body);
            const std::string status = body->value(
                "status",
                outcome.value("review_recommendation", std::string("promotable")));
            const std::string reviewer = body->value("reviewer", std::string("pi_outcome_evaluator"));
            const std::string note = body->value("note", std::string("outcome evaluator"));
            const nlohmann::json scope = body->contains("scope") && (*body)["scope"].is_object()
                ? (*body)["scope"]
                : nlohmann::json::object();
            tile_compile::pi::PiMemoryStore store(tile_compile::pi::pi_storage_dir(state));
            const auto review = store.review(memory_id, status, reviewer, note, outcome, scope);

            // Negative Learning: bei verschlechtertem oder unveraendertem Outcome
            // automatisch einen counterexample-Kandidaten erzeugen, damit die KI
            // dieselbe erfolglose Strategie nicht wiederholt.
            const std::string verdict = outcome.value("verdict", std::string("unknown"));
            nlohmann::json counterexample = nlohmann::json(nullptr);
            const bool create_counterexample = body->value("negative_learning", true)
                && (verdict == "worse" || verdict == "unchanged");
            if (create_counterexample) {
                // Original-Memory finden um Kontext zu uebernehmen
                nlohmann::json original_ctx = nlohmann::json::object({
                    {"schema_version", "pi.context_signature.v1"},
                    {"target", nlohmann::json::object()},
                    {"acquisition", nlohmann::json::object()},
                    {"pipeline", nlohmann::json::object()}
                });
                nlohmann::json original_config_updates = nlohmann::json::array();
                nlohmann::json original_scope = {
                    {"applies_when", nlohmann::json::array({"same_context_and_config_paths"})},
                    {"does_not_apply_when", nlohmann::json::array({"materially_different_acquisition_or_outcome"})},
                    {"confidence", 0.6}
                };
                for (const auto& mem : store.list(10000)) {
                    if (mem.value("memory_id", std::string()) == memory_id) {
                        if (mem.contains("context_signature") && mem["context_signature"].is_object())
                            original_ctx = mem["context_signature"];
                        if (mem.contains("config_updates") && mem["config_updates"].is_array())
                            original_config_updates = mem["config_updates"];
                        if (mem.contains("scope") && mem["scope"].is_object())
                            original_scope = mem["scope"];
                        break;
                    }
                }
                try {
                    counterexample = store.append_candidate({
                        {"type", "counterexample"},
                        {"source", "outcome_evaluator"},
                        {"privacy_class", "metadata_only"},
                        {"summary", "Outcome verdict '" + verdict + "' for memory " + memory_id + ": config change did not improve result."},
                        {"context_signature", original_ctx},
                        {"scope", original_scope},
                        {"config_updates", original_config_updates},
                        {"recommendation", {
                            {"avoid_repeating", original_config_updates},
                            {"explanation", "Outcome evaluator recorded verdict=" + verdict + ". This config change did not produce a measurable improvement."}
                        }},
                        {"evidence", {
                            {"original_memory_id", memory_id},
                            {"outcome_verdict", verdict},
                            {"human_feedback", outcome.value("human_feedback", nlohmann::json(nullptr))}
                        }},
                        {"outcome", outcome},
                        {"retrieval", {
                            {"keywords", nlohmann::json::array({verdict, "counterexample"})},
                            {"negative", true}
                        }}
                    });
                    const std::string ce_id = counterexample.value("memory_id", std::string());
                    if (!ce_id.empty() && counterexample.value("created", true)) {
                        store.review(ce_id, "rejected", "pi_outcome_evaluator",
                                     "auto-rejected counterexample: verdict=" + verdict,
                                     outcome, nlohmann::json::object());
                    }
                } catch (const std::exception&) {
                    // Counterexample-Erzeugung ist best-effort; Fehler nicht nach oben werfen
                }
            }
            nlohmann::json resp = {{"ok", true}, {"outcome", outcome}, {"review", review}};
            if (!counterexample.is_null()) resp["counterexample"] = counterexample;
            return json_resp(resp);
        } catch (const std::invalid_argument& e) {
            return err_resp("BAD_REQUEST", e.what(), 400);
        } catch (const std::exception& e) {
            return err_resp("BACKEND_COMMAND_FAILED", e.what(), 502);
        }
    });

    // Promotable → accepted: wenn ein Memory als promotable markiert ist und der Nutzer
    // oder ein automatischer Prozess es bestaetigt, wird es zu accepted.
    CROW_ROUTE(app, "/api/pi/memories/<string>/promote").methods("POST"_method)
    ([state](const crow::request& req, const std::string& memory_id) {
        auto body = parse_body(req);
        if (!body) return err_resp("BAD_REQUEST", "Invalid JSON", 400);
        try {
            tile_compile::pi::PiMemoryStore store(tile_compile::pi::pi_storage_dir(state));
            // Pruefen ob Memory tatsaechlich promotable ist
            bool is_promotable = false;
            for (const auto& mem : store.list(10000)) {
                if (mem.value("memory_id", std::string()) == memory_id) {
                    is_promotable = (mem.value("status", std::string()) == "promotable");
                    break;
                }
            }
            if (!is_promotable) {
                return err_resp("BAD_REQUEST",
                    "memory is not in promotable status; only promotable memories can be accepted via promote", 400);
            }
            const std::string reviewer = body->value("reviewer", std::string("user"));
            const std::string note = body->value("note", std::string("promoted to accepted"));
            const nlohmann::json outcome = body->contains("outcome") && (*body)["outcome"].is_object()
                ? (*body)["outcome"] : nlohmann::json::object();
            const nlohmann::json scope = body->contains("scope") && (*body)["scope"].is_object()
                ? (*body)["scope"] : nlohmann::json::object();
            const auto review = store.review(memory_id, "accepted", reviewer, note, outcome, scope);
            return json_resp({{"ok", true}, {"memory_id", memory_id}, {"review", review}});
        } catch (const std::invalid_argument& e) {
            return err_resp("BAD_REQUEST", e.what(), 400);
        } catch (const std::exception& e) {
            return err_resp("BACKEND_COMMAND_FAILED", e.what(), 502);
        }
    });

    // Post-Run-Trigger: liest Run-Artefakte und schreibt Outcome-Daten in alle offenen
    // Memory-Kandidaten, die zu diesem Run passen (source = scan_ai_apply).
    // Kann nach Run-Ende manuell oder automatisch vom GUI3-Client aufgerufen werden.
    CROW_ROUTE(app, "/api/pi/memories/evaluate-run").methods("POST"_method)
    ([state](const crow::request& req) {
        auto body = parse_body(req);
        if (!body) return err_resp("BAD_REQUEST", "Invalid JSON", 400);
        const std::string run_id = body->value("run_id", std::string());
        if (run_id.empty()) return err_resp("BAD_REQUEST", "run_id is required", 400);
        fs::path run_dir;
        try {
            run_dir = state->runtime.resolve_run_dir(run_id);
        } catch (const std::runtime_error&) {
            return err_resp("NOT_FOUND", "run directory not found", 404);
        }
        try {
            if (!fs::is_directory(run_dir)) {
                return err_resp("NOT_FOUND", "run directory not found", 404);
            }
            // Metriken aus Run-Artefakten extrahieren
            const nlohmann::json after_metrics = extract_run_outcome_metrics(run_dir);

            // Optionale "before"-Metriken aus Request (vom Caller mitgegeben)
            const nlohmann::json before_metrics = body->contains("before") && (*body)["before"].is_object()
                ? (*body)["before"]
                : nlohmann::json::object();

            // Alle candidate-Memories pruefen, die source=scan_ai_apply haben
            tile_compile::pi::PiMemoryStore store(tile_compile::pi::pi_storage_dir(state));
            int updated = 0;
            int skipped = 0;
            nlohmann::json updated_ids = nlohmann::json::array();

            for (const auto& memory : store.list(10000)) {
                const std::string status = memory.value("status", std::string());
                // Nur Kandidaten ohne abgeschlossenes Outcome
                if (status != "candidate" && status != "promotable") { ++skipped; continue; }
                const std::string source = memory.value("source", std::string());
                if (source != "scan_ai_apply" && source != "run_chat" && source != "resume_feedback") {
                    ++skipped; continue;
                }
                // Pruefe ob Provenance auf diesen Run zeigt
                bool matches_run = false;
                if (memory.contains("evidence") && memory["evidence"].is_object()) {
                    const auto& ev = memory["evidence"];
                    if (ev.value("run_id", std::string()) == run_id) matches_run = true;
                    if (ev.contains("run_id_hash")) matches_run = true; // gehashte Referenz akzeptiert
                }
                if (memory.contains("provenance") && memory["provenance"].is_object()) {
                    if (memory["provenance"].value("run_id", std::string()) == run_id) matches_run = true;
                }
                // Ohne explizite Run-Verknuepfung: neueste Kandidaten einschliessen
                // wenn kein run_id angegeben ist (Fallback fuer aeltere Memories)
                if (!matches_run && body->value("all_candidates", false)) matches_run = true;
                if (!matches_run) { ++skipped; continue; }

                const std::string memory_id = memory.value("memory_id", std::string());
                if (memory_id.empty()) { ++skipped; continue; }

                // Outcome-Payload bauen
                nlohmann::json eval_body = {
                    {"before", before_metrics},
                    {"after", after_metrics},
                    {"feedback", body->value("feedback", std::string())},
                    {"result", body->value("result", std::string())}
                };
                if (body->contains("user_rating")) eval_body["user_rating"] = (*body)["user_rating"];

                const nlohmann::json outcome = evaluate_memory_outcome_payload(eval_body);
                const std::string rec_status = outcome.value("review_recommendation", std::string("promotable"));
                try {
                    store.review(memory_id, rec_status, "pi_run_outcome_evaluator",
                                 "post-run outcome evaluation for run " + run_id,
                                 outcome, nlohmann::json::object());
                    updated_ids.push_back(memory_id);
                    ++updated;
                } catch (const std::exception&) {
                    ++skipped;
                }
            }
            return json_resp({
                {"ok", true},
                {"run_id", run_id},
                {"updated", updated},
                {"skipped", skipped},
                {"updated_memory_ids", updated_ids},
                {"run_metrics", after_metrics}
            });
        } catch (const std::exception& e) {
            return err_resp("BACKEND_COMMAND_FAILED", e.what(), 502);
        }
    });

    // Dedizierter Resume-Feedback-Endpoint: setzt source=resume_feedback explizit.
    // Erzeugt einen Memory-Kandidaten aus dem Resume-Ergebnis und markiert ihn
    // als resume_strategy mit dem gegebenen Feedback.
    CROW_ROUTE(app, "/api/pi/memories/resume-feedback").methods("POST"_method)
    ([state](const crow::request& req) {
        auto body = parse_body(req);
        if (!body) return err_resp("BAD_REQUEST", "Invalid JSON", 400);
        const std::string run_id = body->value("run_id", std::string());
        const std::string from_phase = body->value("from_phase", std::string());
        const std::string feedback = body->value("feedback", std::string());
        const std::string result = body->value("result", std::string());
        if (run_id.empty()) return err_resp("BAD_REQUEST", "run_id is required", 400);
        if (from_phase.empty()) return err_resp("BAD_REQUEST", "from_phase is required", 400);
        if (feedback.empty() && result.empty()) {
            return err_resp("BAD_REQUEST", "feedback or result is required", 400);
        }
        try {
            // Kontext aus dem Run lesen
            const fs::path run_dir = state->runtime.resolve_run_dir(run_id);
            const nlohmann::json context_signature = body->contains("context_signature") &&
                                                      (*body)["context_signature"].is_object()
                ? (*body)["context_signature"]
                : nlohmann::json::object({
                    {"schema_version", "pi.context_signature.v1"},
                    {"target", nlohmann::json::object()},
                    {"acquisition", nlohmann::json::object()},
                    {"pipeline", {{"resume_phase", from_phase}, {"phases", nlohmann::json::array({from_phase})}}}
                });

            nlohmann::json after_metrics = nlohmann::json::object();
            if (fs::is_directory(run_dir)) {
                after_metrics = extract_run_outcome_metrics(run_dir);
            }
            after_metrics["resume_phase"] = from_phase;

            nlohmann::json eval_body = {
                {"before", body->contains("before") ? (*body)["before"] : nlohmann::json::object()},
                {"after", after_metrics},
                {"result", result},
                {"feedback", feedback}
            };
            if (body->contains("user_rating")) eval_body["user_rating"] = (*body)["user_rating"];
            const nlohmann::json outcome = evaluate_memory_outcome_payload(eval_body);

            const std::string verdict = outcome.value("verdict", std::string("unknown"));
            const std::string memory_type = verdict == "improved" ? "resume_strategy" : "counterexample";
            nlohmann::json summary_parts = nlohmann::json::array();
            summary_parts.push_back("Resume from " + from_phase);
            if (!result.empty()) summary_parts.push_back(result);
            if (!feedback.empty()) summary_parts.push_back(feedback);
            std::string summary;
            for (size_t i = 0; i < summary_parts.size(); ++i) {
                if (i > 0) summary += ": ";
                summary += summary_parts[i].get<std::string>();
            }

            tile_compile::pi::PiMemoryStore store(tile_compile::pi::pi_storage_dir(state));
            nlohmann::json memory = store.append_candidate({
                {"type", memory_type},
                {"source", "resume_feedback"},
                {"privacy_class", "metadata_only"},
                {"summary", summary},
                {"context_signature", context_signature},
                {"scope", {
                    {"applies_when", nlohmann::json::array({"resume from " + from_phase + " with similar acquisition context"})},
                    {"does_not_apply_when", nlohmann::json::array({"materially different pipeline or acquisition setup"})},
                    {"confidence", outcome.value("verified", false) ? 0.7 : 0.4}
                }},
                {"recommendation", {
                    {"resume_phase", from_phase},
                    {"verdict", verdict},
                    {"explanation", feedback.empty() ? result : feedback}
                }},
                {"evidence", {
                    {"run_id", run_id},
                    {"human_feedback", feedback.empty() ? nlohmann::json(nullptr) : nlohmann::json(feedback)},
                    {"source", "resume_feedback"}
                }},
                {"outcome", outcome},
                {"retrieval", {
                    {"keywords", nlohmann::json::array({from_phase, verdict})},
                    {"negative", verdict != "improved"}
                }}
            });
            const std::string memory_id = memory.value("memory_id", std::string());
            if (!memory_id.empty() && memory.value("created", true)) {
                const std::string rec_status = outcome.value("review_recommendation", std::string("promotable"));
                store.review(memory_id, rec_status, "pi_resume_feedback",
                             "resume feedback: " + (feedback.empty() ? result : feedback),
                             outcome, nlohmann::json::object());
            }
            return json_resp({
                {"ok", true},
                {"memory", memory},
                {"outcome", outcome},
                {"run_id", run_id},
                {"from_phase", from_phase}
            });
        } catch (const std::invalid_argument& e) {
            return err_resp("BAD_REQUEST", e.what(), 400);
        } catch (const std::exception& e) {
            return err_resp("BACKEND_COMMAND_FAILED", e.what(), 502);
        }
    });

    CROW_ROUTE(app, "/api/pi/memories/retrieve").methods("POST"_method)
    ([state](const crow::request& req) {
        auto body = parse_body(req);
        if (!body) return err_resp("BAD_REQUEST", "Invalid JSON", 400);
        const int limit = std::max(1, std::min(100, body->value("limit", 10)));
        const nlohmann::json query = body->contains("query") && (*body)["query"].is_object()
            ? (*body)["query"]
            : *body;
        tile_compile::pi::PiMemoryStore store(tile_compile::pi::pi_storage_dir(state));
        const auto matches = store.retrieve(query, limit);
        const auto warnings = store.retrieve_negative(query, limit);
        return json_resp({
            {"schema_version", tile_compile::pi::kMemoryRetrievalSchemaVersion},
            {"matches", matches},
            {"warnings", warnings},
            {"count", matches.size()},
            {"warning_count", warnings.size()}
        });
    });

    CROW_ROUTE(app, "/api/pi/audit").methods("GET"_method)
    ([state](const crow::request& req) {
        const int limit = std::max(1, std::min(1000, int_query_param(req, "limit", 200)));
        return json_resp(pi_audit_log(state, limit));
    });

    CROW_ROUTE(app, "/api/pi/action-plans/validate").methods("POST"_method)
    ([](const crow::request& req) {
        auto body = parse_body(req);
        if (!body) return err_resp("BAD_REQUEST", "Invalid JSON", 400);
        const nlohmann::json plan = body->contains("plan") && (*body)["plan"].is_object()
            ? (*body)["plan"]
            : *body;
        return json_resp(tile_compile::pi::validate_action_plan_shape(plan));
    });

    CROW_ROUTE(app, "/api/pi/action-plans/preview").methods("POST"_method)
    ([state](const crow::request& req) {
        auto body = parse_body(req);
        if (!body) return err_resp("BAD_REQUEST", "Invalid JSON", 400);
        const nlohmann::json plan = body->contains("plan") && (*body)["plan"].is_object()
            ? (*body)["plan"]
            : *body;
        const auto validation = tile_compile::pi::validate_action_plan_shape(plan);
        if (!validation.value("valid", false)) {
            return json_resp({
                {"ok", false},
                {"validation", validation}
            }, 400);
        }
        return json_resp({
            {"ok", true},
            {"validation", validation},
            {"preview", build_validated_preview(plan, *body, state)}
        });
    });

    CROW_ROUTE(app, "/api/pi/action-plans/apply").methods("POST"_method)
    ([state](const crow::request& req) {
        auto body = parse_body(req);
        if (!body) return err_resp("BAD_REQUEST", "Invalid JSON", 400);
        const bool confirmed = body->value("confirmed", false) || body->value("reviewed", false);
        if (!confirmed) {
            return err_resp("REVIEW_REQUIRED", "confirmed=true is required before applying a PI action plan", 409);
        }
        const nlohmann::json plan = body->contains("plan") && (*body)["plan"].is_object()
            ? (*body)["plan"]
            : *body;
        const auto validation = tile_compile::pi::validate_action_plan_shape(plan);
        if (!validation.value("valid", false)) {
            return json_resp({
                {"ok", false},
                {"validation", validation}
            }, 400);
        }
        const auto preview = build_validated_preview(plan, *body, state);
        if (!preview.value("config_valid", false)) {
            return json_resp({
                {"ok", false},
                {"validation", validation},
                {"preview", preview},
                {"error", {{"code", "CONFIG_INVALID"}, {"message", "preview config validation failed"}}}
            }, 400);
        }
        if (body->contains("expected_patched_yaml") && (*body)["expected_patched_yaml"].is_string()
            && (*body)["expected_patched_yaml"].get<std::string>() != preview.value("patched_yaml", std::string())) {
            return json_resp({
                {"ok", false},
                {"validation", validation},
                {"preview", preview},
                {"error", {{"code", "STALE_PREVIEW"}, {"message", "expected_patched_yaml does not match current preview"}}}
            }, 409);
        }
        auto applied = apply_validated_preview(preview, state);
        if (!applied.value("ok", false)) {
            return json_resp(applied, 502);
        }
        return json_resp({
            {"ok", true},
            {"validation", validation},
            {"preview", preview},
            {"revision_id", applied["revision_id"]},
            {"path", applied["path"]},
            {"saved", applied["saved"]}
        });
    });

    // ===== Live Image Chat Routes =====

    auto live_store = std::make_shared<tile_compile::pi::LiveImageSessionStore>();

    // 1. POST /api/pi/live-image-chat/create
    CROW_ROUTE(app, "/api/pi/live-image-chat/create").methods("POST"_method)
    ([state, live_store](const crow::request& req) {
        auto body = parse_body(req);
        if (!body) return err_resp("BAD_REQUEST", "Invalid JSON", 400);
        const std::string run_id = body->value("run_id", std::string());
        if (run_id.empty()) return err_resp("BAD_REQUEST", "run_id is required", 400);

        fs::path run_dir;
        try {
            run_dir = state->runtime.resolve_run_dir(run_id);
        } catch (const std::runtime_error&) {
            return err_resp("NOT_FOUND", "run directory not found", 404);
        }
        if (!fs::is_directory(run_dir))
            return err_resp("NOT_FOUND", "run directory not found", 404);

        live_store->evict_expired(1800, 5);

        auto fits_path = find_output_fits(run_dir);
        if (!fits_path)
            return err_resp("NO_FITS", "No FITS output found in run directory", 404);

        try {
            const fs::path live_edit_path = run_dir / "outputs" / "live_edit.fits";

            // The source FITS is immutable. live_edit.fits, when present, is
            // the canonical current working image from a previous session.
            auto saved = read_pi_live_image_chat_history(state, run_id);
            nlohmann::json op_history = saved.value("operation_history", nlohmann::json::array());
            const bool has_live_edit = fs::exists(live_edit_path);
            const bool resumed = has_live_edit || !op_history.empty() ||
                !saved.value("chat_history", nlohmann::json::array()).empty();

            cv::Mat original = read_fits_to_float_bgr(*fits_path);
            cv::Mat current = has_live_edit ? read_fits_to_float_bgr(live_edit_path) : original.clone();
            if (original.empty() || current.empty())
                return err_resp("RENDER_FAILED", "Failed to read FITS data", 500);

            std::string session_id = live_store->create(run_id, original, current);
            live_store->evict_expired(1800, 5);

            // If history exists without a working FITS, reconstruct the current
            // image once and materialize the canonical working file.
            if (!has_live_edit && !op_history.empty()) {
                live_store->with_session(session_id, [&](tile_compile::pi::LiveImageSession& s) {
                    for (const auto& op : op_history) {
                        auto result = tile_compile::pi::apply_image_op_fits(s.current_fits, op);
                        if (result.success) s.current_fits = std::move(result.image);
                    }
                });
                persist_live_edit_fits(state, live_store, session_id);
            }

            live_store->with_session(session_id, [&](tile_compile::pi::LiveImageSession& s) {
                s.chat_history = saved.value("chat_history", nlohmann::json::array());
                s.operation_history = op_history;
                s.edit_history = saved.value("edit_history", nlohmann::json::array());
                // Reconstruct exact pre-operation snapshots for undo after a
                // browser/session reload. The canonical live_edit.fits is
                // still used as the current image; snapshots are derived from
                // the immutable source and persisted operation sequence.
                cv::Mat replay = s.original_fits.clone();
                s.undo_stack.clear();
                s.undo_snapshots.clear();
                s.redo_stack.clear();
                s.redo_snapshots.clear();
                if (op_history.is_array()) {
                    for (const auto& history_op : op_history) {
                        const cv::Mat before = replay.clone();
                        auto replay_result = tile_compile::pi::apply_image_op_fits(replay, history_op);
                        if (!replay_result.success) break;
                        s.undo_snapshots.push_back(before);
                        s.undo_stack.push_back(history_op);
                        replay = std::move(replay_result.image);
                    }
                }
            });

            std::string preview_b64;
            int img_w = 0, img_h = 0;
            bool can_undo = false, can_redo = false;
            nlohmann::json next_undo_op, next_redo_op;
            live_store->with_session(session_id, [&](tile_compile::pi::LiveImageSession& s) {
                preview_b64 = mat_to_jpeg_base64(s.current_fits, 85);
                img_w = s.current_fits.cols;
                img_h = s.current_fits.rows;
                can_undo = !s.undo_stack.empty();
                can_redo = !s.redo_stack.empty();
                next_undo_op = stack_top_label(s.undo_stack);
                next_redo_op = stack_top_label(s.redo_stack);
            });

            nlohmann::json resp = {
                {"session_id", session_id},
                {"run_id", run_id},
                {"image_base64", preview_b64},
                {"image_mime", "image/jpeg"},
                {"image_width", img_w},
                {"image_height", img_h},
                {"resumed", resumed},
                {"can_undo", can_undo},
                {"can_redo", can_redo},
                {"next_undo", next_undo_op},
                {"next_redo", next_redo_op}
            };
            nlohmann::json chat_history = saved.value("chat_history", nlohmann::json::array());
            if (!chat_history.empty()) {
                resp["chat_history"] = chat_history;
                resp["operation_history"] = op_history;
            }

            return json_resp(resp);
        } catch (const std::exception& e) {
            return err_resp("RENDER_FAILED", e.what(), 500);
        }
    });

    // 2. POST /api/pi/live-image-chat
    CROW_ROUTE(app, "/api/pi/live-image-chat").methods("POST"_method)
    ([state, live_store](const crow::request& req) {
        auto body = parse_body(req);
        if (!body) return err_resp("BAD_REQUEST", "Invalid JSON", 400);
        const std::string session_id = body->value("session_id", std::string());
        const std::string message = body->value("message", std::string());
        if (session_id.empty()) return err_resp("BAD_REQUEST", "session_id is required", 400);
        if (message.empty()) return err_resp("BAD_REQUEST", "message is required", 400);

        // Count user messages to decide whether to send image
        int user_msg_count = 0;
        std::string vision_b64;
        nlohmann::json op_history;
        cv::Mat analysis_image;
        bool found = live_store->with_session(session_id, [&](tile_compile::pi::LiveImageSession& s) {
            analysis_image = s.current_fits.clone();
            for (const auto& entry : s.chat_history) {
                if (entry.value("role", "") == "user") user_msg_count++;
            }
            op_history = s.operation_history;
            // Send image every 3rd user message (1st, 4th, 7th, ...)
            // to limit vision API costs while keeping analysis fresh
            if (user_msg_count % 3 == 0) {
                vision_b64 = mat_to_vision_jpeg_base64(s.current_fits, 1568, 85);
            }
        });
        if (!found) return err_resp("NOT_FOUND", "session not found", 404);

        // Append user message to chat history
        live_store->append_chat(session_id, "user", message);

        // Crop is a deterministic geometry operation and must never require
        // or invoke the AI sidecar. Handle it through the local parser.
        const std::string lower_message = [&]() {
            std::string value = message;
            std::transform(value.begin(), value.end(), value.begin(),
                           [](unsigned char c) { return std::tolower(c); });
            return value;
        }();
        const bool local_crop_request = lower_message.find("crop") != std::string::npos ||
            lower_message.find("zuschneid") != std::string::npos ||
            lower_message.find("beschneid") != std::string::npos ||
            lower_message.find("rand abschneiden") != std::string::npos;

        // Try AI sidecar unless this is a local-only crop request.
        nlohmann::json ai_result;
        bool sidecar_ok = false;
        bool ai_model_configured = false;
        std::string sidecar_error;
        nlohmann::json prompt_history = nlohmann::json::array();
        if (op_history.is_array()) {
            const size_t begin = op_history.size() > 10 ? op_history.size() - 10 : 0;
            for (size_t i = begin; i < op_history.size(); ++i) prompt_history.push_back(op_history[i]);
        }
        try {
            if (local_crop_request) throw std::runtime_error("crop is local-only");
            auto ai_config = current_pi_ai_config(state);
            if (!ai_config.model.empty()) {
                ai_model_configured = true;
                tile_compile::ai::AiSidecarClient client(ai_config);
                nlohmann::json payload = {
                    {"prompt", message},
                    {"image_base64", vision_b64},
                    {"image_mime", "image/jpeg"},
                    {"image_width", analysis_image.cols},
                    {"image_height", analysis_image.rows},
                    {"operation_history", prompt_history}
                };
        auto response = client.post("/live-image-chat", payload);
                if (response.contains("operations")) {
                    ai_result = response;
                    sidecar_ok = true;
                }
            }
        } catch (const std::exception& e) {
            sidecar_error = e.what();
            // Sidecar not available, fall through to local
        }

        if (!sidecar_ok) {
            ai_result = fallback_parse_message(message, analysis_image);
            const auto fallback_operations = ai_result.value("operations", nlohmann::json::array());
            if (fallback_operations.is_array() && fallback_operations.empty() && !local_crop_request) {
                if (!ai_model_configured) {
                    ai_result["summary"] = "PI: Kein KI-Provider/Modell ausgewählt. Bitte unter Tools > KI & API einen Provider und ein Modell auswählen.";
                } else {
                    std::string error_lower = sidecar_error;
                    std::transform(error_lower.begin(), error_lower.end(), error_lower.begin(),
                                   [](unsigned char c) { return std::tolower(c); });
                    const bool missing_api_key = error_lower.find("api key") != std::string::npos ||
                        error_lower.find("api_key") != std::string::npos ||
                        error_lower.find("authentication") != std::string::npos ||
                        error_lower.find("unauthorized") != std::string::npos ||
                        error_lower.find("401") != std::string::npos;
                    ai_result["summary"] = missing_api_key
                        ? "PI: Für den ausgewählten Provider wird ein API-Key benötigt. Bitte unter Tools > KI & API eingeben und speichern."
                        : "PI: Der ausgewählte KI-Provider/Modell ist nicht verbunden. Bitte die Verbindung unter Tools > KI & API prüfen.";
                }
            }
        }

        // Commands backed by a parameter dialog must resolve to their
        // dedicated operation type. Older model prompts may still answer
        // with generic contrast/denoise/brightness operations, which would
        // otherwise be applied immediately and bypass confirmation.
        auto contains_message = [&](const std::string& value) {
            return lower_message.find(value) != std::string::npos;
        };
        std::string requested_dialog_type;
        if (contains_message("tonwert") || contains_message("levels"))
            requested_dialog_type = "levels";
        else if (contains_message("schatten") || contains_message("shadow"))
            requested_dialog_type = "shadow_recovery";
        else if (contains_message("spitzlicht") || contains_message("highlight"))
            requested_dialog_type = "highlight_recovery";
        else if (contains_message("farbbalance") || contains_message("color balance"))
            requested_dialog_type = "color_balance";
        else if ((contains_message("lokal") || contains_message("local")) &&
                 contains_message("kontrast"))
            requested_dialog_type = "local_contrast";
        else if (contains_message("farbrausch") || contains_message("chroma"))
            requested_dialog_type = "chroma_denoise";

        if (!requested_dialog_type.empty()) {
            const auto returned_ops = ai_result.value("operations", nlohmann::json::array());
            const bool correct_type = returned_ops.is_array() && returned_ops.size() == 1 &&
                returned_ops.front().is_object() &&
                returned_ops.front().value("type", std::string()) == requested_dialog_type &&
                tile_compile::pi::validate_op(returned_ops.front()).empty();
            if (!correct_type) {
                auto deterministic_suggestion = fallback_parse_message(message, analysis_image);
                const auto fallback_ops = deterministic_suggestion.value("operations", nlohmann::json::array());
                if (fallback_ops.is_array() && fallback_ops.size() == 1 &&
                    fallback_ops.front().value("type", std::string()) == requested_dialog_type) {
                    ai_result["operations"] = fallback_ops;
                    ai_result["summary"] = deterministic_suggestion.value(
                        "summary", std::string("Parameter vorgeschlagen."));
                    ai_result["adjustable"] = false;
                    ai_result["repeatable"] = false;
                }
            }
        }

        // Apply operations
        nlohmann::json operations = ai_result.value("operations", nlohmann::json::array());
        nlohmann::json applied_ops = nlohmann::json::array();
        nlohmann::json proposed_ops = nlohmann::json::array();
        nlohmann::json warnings = ai_result.value("warnings", nlohmann::json::array());
        if (!warnings.is_array()) warnings = nlohmann::json::array();
        std::string last_error;

        const auto requires_parameter_dialog = [](const std::string& type) {
            return type == "levels" || type == "shadow_recovery" ||
                type == "highlight_recovery" || type == "color_balance" ||
                type == "local_contrast" || type == "chroma_denoise";
        };
        const bool requires_confirmation = operations.is_array() && operations.size() == 1 &&
            operations.front().is_object() &&
            requires_parameter_dialog(operations.front().value("type", std::string()));

        for (const auto& op : operations) {
            if (!op.is_object() || !op.contains("type")) continue;
            if (requires_confirmation) {
                const auto validation = tile_compile::pi::validate_op(op);
                if (validation.empty()) proposed_ops.push_back(op);
                else {
                    last_error = validation.value("error", std::string("invalid operation"));
                    warnings.push_back("invalid_operation: " + last_error);
                }
                continue;
            }
            auto res = live_store->apply_operation(session_id, op);
            if (res.success) {
                applied_ops.push_back(op);
            } else {
                last_error = res.error;
                warnings.push_back("invalid_operation: " + res.error);
            }
        }

        const bool sharpen_operation = !applied_ops.empty() &&
            applied_ops.front().value("type", std::string()) == "sharpen";
        const bool adjustable = !requires_confirmation &&
            ai_result.value("adjustable", false) && !sharpen_operation;

        // Set adjust step if provided, otherwise derive from the first applied operation
        nlohmann::json effective_adjust_step = nullptr;
        if (adjustable) {
            nlohmann::json adjust_step = ai_result.value("adjust_step", nlohmann::json());
            if (adjust_step.is_null() && !applied_ops.empty()) {
                adjust_step = applied_ops[0];
            }
            if (!adjust_step.is_null()) {
                effective_adjust_step = adjust_step;
                live_store->set_adjust_step(session_id, adjust_step);
            }
        } else {
            live_store->set_adjust_step(session_id, nullptr);
        }
        const bool has_repeatable_operation = !applied_ops.empty() &&
            applied_ops.front().value("type", std::string()) != "reset";
        const bool repeatable = has_repeatable_operation &&
            (sharpen_operation || ai_result.value("repeatable", false) || !adjustable);

        // Append assistant response to chat history
        std::string summary = ai_result.value("summary", std::string());
        live_store->append_chat(session_id, "assistant", summary,
                                applied_ops.empty() ? nlohmann::json(nullptr) : applied_ops);

        // Persist the float working copy and render preview
        if (!applied_ops.empty()) persist_live_edit_fits(state, live_store, session_id);

        std::string updated_b64;
        int img_w = 0, img_h = 0;
        bool can_undo = false, can_redo = false;
        nlohmann::json next_undo_op, next_redo_op;
        live_store->with_session(session_id, [&](tile_compile::pi::LiveImageSession& s) {
            updated_b64 = mat_to_jpeg_base64(s.current_fits, 85);
            img_w = s.current_fits.cols;
            img_h = s.current_fits.rows;
            can_undo = !s.undo_stack.empty();
            can_redo = !s.redo_stack.empty();
            next_undo_op = stack_top_label(s.undo_stack);
            next_redo_op = stack_top_label(s.redo_stack);
        });

        nlohmann::json resp = {
            {"schema_version", "pi.live-image-chat.v1"},
            {"session_id", session_id},
            {"summary", summary},
            {"operations", requires_confirmation ? proposed_ops : applied_ops},
            {"requires_confirmation", requires_confirmation && !proposed_ops.empty()},
            {"image_base64", updated_b64},
            {"image_mime", "image/jpeg"},
            {"image_width", img_w},
            {"image_height", img_h},
            {"adjustable", adjustable},
            {"repeatable", repeatable},
            {"can_undo", can_undo},
            {"can_redo", can_redo},
            {"next_undo", next_undo_op},
            {"next_redo", next_redo_op},
            {"mode", ai_result.value("mode", sidecar_ok ? "sidecar" : "local_fallback")}
        };
        resp["warnings"] = warnings;
        if (!effective_adjust_step.is_null())
            resp["adjust_step"] = effective_adjust_step;
        if (!last_error.empty())
            resp["last_error"] = last_error;

        try { persist_live_session(state, live_store, session_id); } catch (...) {}
        return json_resp(resp);
    });

    // Repeat the last non-adjustable operation with exactly the same params.
    CROW_ROUTE(app, "/api/pi/live-image-chat/repeat").methods("POST"_method)
    ([state, live_store](const crow::request& req) {
        auto body = parse_body(req);
        if (!body) return err_resp("BAD_REQUEST", "Invalid JSON", 400);
        const std::string session_id = body->value("session_id", std::string());
        if (session_id.empty()) return err_resp("BAD_REQUEST", "session_id is required", 400);
        auto result = live_store->repeat_operation(session_id);
        if (!result.success)
            return err_resp("REPEAT_FAILED", result.error, 400);
        nlohmann::json repeated_op = nlohmann::json(nullptr);
        live_store->with_session(session_id, [&](tile_compile::pi::LiveImageSession& s) {
            if (!s.undo_stack.empty()) repeated_op = s.undo_stack.back();
        });
        live_store->append_chat(session_id, "assistant", "Operation erneut angewendet.", repeated_op);
        persist_live_edit_fits(state, live_store, session_id);
        try { persist_live_session(state, live_store, session_id); } catch (...) {}
        bool can_undo = false, can_redo = false;
        nlohmann::json next_undo_op, next_redo_op;
        live_store->with_session(session_id, [&](tile_compile::pi::LiveImageSession& s) {
            can_undo = !s.undo_stack.empty();
            can_redo = !s.redo_stack.empty();
            next_undo_op = stack_top_label(s.undo_stack);
            next_redo_op = stack_top_label(s.redo_stack);
        });
        return json_resp({
            {"session_id", session_id},
            {"summary", "Operation erneut angewendet."},
            {"operations", repeated_op.is_null() ? nlohmann::json::array() : nlohmann::json::array({repeated_op})},
            {"image_base64", mat_to_jpeg_base64(result.image, 85)},
            {"image_mime", "image/jpeg"},
            {"can_undo", can_undo},
            {"can_redo", can_redo},
            {"next_undo", next_undo_op},
            {"next_redo", next_redo_op},
            {"repeatable", true}
        });
    });

    // Reapply an operation recorded in a chat message without invoking AI.
    CROW_ROUTE(app, "/api/pi/live-image-chat/reapply").methods("POST"_method)
    ([state, live_store](const crow::request& req) {
        auto body = parse_body(req);
        if (!body) return err_resp("BAD_REQUEST", "Invalid JSON", 400);
        const auto session_id = body->value("session_id", std::string());
        nlohmann::json operations = body->value("operations", nlohmann::json::array());
        if (operations.is_object()) operations = nlohmann::json::array({operations});
        if (session_id.empty() || !operations.is_array() || operations.empty())
            return err_resp("BAD_REQUEST", "session_id and operations are required", 400);
        auto result = live_store->apply_preset(session_id, operations);
        if (!result.success) return err_resp("REAPPLY_FAILED", result.error, 400);
        live_store->append_chat(session_id, "assistant", "Operation erneut angewendet.", operations);
        persist_live_edit_fits(state, live_store, session_id);
        try { persist_live_session(state, live_store, session_id); } catch (...) {}
        bool can_undo = false;
        nlohmann::json next_undo_op;
        live_store->with_session(session_id, [&](tile_compile::pi::LiveImageSession& s) {
            can_undo = !s.undo_stack.empty();
            next_undo_op = stack_top_label(s.undo_stack);
        });
        return json_resp({{"ok", true}, {"summary", "Operation erneut angewendet."},
                          {"operations", operations}, {"image_base64", mat_to_jpeg_base64(result.image, 85)},
                          {"image_mime", "image/jpeg"}, {"can_undo", can_undo}, {"can_redo", false},
                          {"next_undo", next_undo_op}, {"next_redo", nullptr}});
    });

    CROW_ROUTE(app, "/api/pi/live-image-chat/preview-operation").methods("POST"_method)
    ([live_store](const crow::request& req) {
        auto body = parse_body(req);
        if (!body) return err_resp("BAD_REQUEST", "Invalid JSON", 400);
        const auto session_id = body->value("session_id", std::string());
        auto operation = body->value("operation", nlohmann::json::object());
        cv::Mat current;
        if (session_id.empty() || !live_store->with_session(session_id, [&](tile_compile::pi::LiveImageSession& s) { current = s.current_fits.clone(); }))
            return err_resp("NOT_FOUND", "session not found", 404);
        auto result = tile_compile::pi::apply_image_op_fits(current, operation);
        if (!result.success) return err_resp("PREVIEW_FAILED", result.error, 400);
        return json_resp({{"ok", true}, {"image_base64", mat_to_jpeg_base64(result.image, 85)},
                          {"image_mime", "image/jpeg"}});
    });

    // 3. POST /api/pi/live-image-chat/adjust
    CROW_ROUTE(app, "/api/pi/live-image-chat/adjust").methods("POST"_method)
    ([state, live_store](const crow::request& req) {
        auto body = parse_body(req);
        if (!body) return err_resp("BAD_REQUEST", "Invalid JSON", 400);
        const std::string session_id = body->value("session_id", std::string());
        const std::string direction = body->value("direction", std::string());
        if (session_id.empty()) return err_resp("BAD_REQUEST", "session_id is required", 400);
        if (direction != "increase" && direction != "decrease")
            return err_resp("BAD_REQUEST", "direction must be 'increase' or 'decrease'", 400);

        auto res = live_store->apply_adjust(session_id, direction);
        if (!res.success && !res.error.empty())
            return err_resp("ADJUST_FAILED", res.error, 400);

        persist_live_edit_fits(state, live_store, session_id);

        std::string b64;
        int adjust_count = 0;
        bool can_undo = false, can_redo = false;
        nlohmann::json next_undo_op, next_redo_op;
        live_store->with_session(session_id, [&](tile_compile::pi::LiveImageSession& s) {
            b64 = mat_to_jpeg_base64(s.current_fits, 85);
            adjust_count = s.adjust_count;
            can_undo = !s.undo_stack.empty();
            can_redo = !s.redo_stack.empty();
            next_undo_op = stack_top_label(s.undo_stack);
            next_redo_op = stack_top_label(s.redo_stack);
        });

        try { persist_live_session(state, live_store, session_id); } catch (...) {}
        return json_resp({
            {"image_base64", b64},
            {"image_mime", "image/jpeg"},
            {"adjust_count", adjust_count},
            {"direction", direction},
            {"can_undo", can_undo},
            {"can_redo", can_redo},
            {"next_undo", next_undo_op},
            {"next_redo", next_redo_op}
        });
    });

    // 4. POST /api/pi/live-image-chat/undo
    CROW_ROUTE(app, "/api/pi/live-image-chat/undo").methods("POST"_method)
    ([state, live_store](const crow::request& req) {
        auto body = parse_body(req);
        if (!body) return err_resp("BAD_REQUEST", "Invalid JSON", 400);
        const std::string session_id = body->value("session_id", std::string());
        if (session_id.empty()) return err_resp("BAD_REQUEST", "session_id is required", 400);

        auto res = live_store->undo(session_id);
        if (res.image.empty())
            return err_resp("NOT_FOUND", "session not found or nothing to undo", 404);

        persist_live_edit_fits(state, live_store, session_id);

        try { persist_live_session(state, live_store, session_id); } catch (...) {}
        nlohmann::json next_undo_op, next_redo_op;
        live_store->with_session(session_id, [&](tile_compile::pi::LiveImageSession& s) {
            next_undo_op = stack_top_label(s.undo_stack);
            next_redo_op = stack_top_label(s.redo_stack);
        });
        return json_resp({
            {"image_base64", mat_to_jpeg_base64(res.image, 85)},
            {"image_mime", "image/jpeg"},
            {"summary", res.summary},
            {"can_undo", res.can_undo},
            {"can_redo", res.can_redo},
            {"next_undo", next_undo_op},
            {"next_redo", next_redo_op}
        });
    });

    // 5. POST /api/pi/live-image-chat/redo
    CROW_ROUTE(app, "/api/pi/live-image-chat/redo").methods("POST"_method)
    ([state, live_store](const crow::request& req) {
        auto body = parse_body(req);
        if (!body) return err_resp("BAD_REQUEST", "Invalid JSON", 400);
        const std::string session_id = body->value("session_id", std::string());
        if (session_id.empty()) return err_resp("BAD_REQUEST", "session_id is required", 400);

        auto res = live_store->redo(session_id);
        if (res.image.empty())
            return err_resp("NOT_FOUND", "session not found or nothing to redo", 404);

        persist_live_edit_fits(state, live_store, session_id);

        try { persist_live_session(state, live_store, session_id); } catch (...) {}
        nlohmann::json next_undo_op_r, next_redo_op_r;
        live_store->with_session(session_id, [&](tile_compile::pi::LiveImageSession& s) {
            next_undo_op_r = stack_top_label(s.undo_stack);
            next_redo_op_r = stack_top_label(s.redo_stack);
        });
        return json_resp({
            {"image_base64", mat_to_jpeg_base64(res.image, 85)},
            {"image_mime", "image/jpeg"},
            {"summary", res.summary},
            {"can_undo", res.can_undo},
            {"can_redo", res.can_redo},
            {"next_undo", next_undo_op_r},
            {"next_redo", next_redo_op_r}
        });
    });

    // 6. POST /api/pi/live-image-chat/reset
    CROW_ROUTE(app, "/api/pi/live-image-chat/reset").methods("POST"_method)
    ([state, live_store](const crow::request& req) {
        auto body = parse_body(req);
        if (!body) return err_resp("BAD_REQUEST", "Invalid JSON", 400);
        const std::string session_id = body->value("session_id", std::string());
        if (session_id.empty()) return err_resp("BAD_REQUEST", "session_id is required", 400);

        auto img = live_store->reset(session_id);
        if (img.empty())
            return err_resp("NOT_FOUND", "session not found", 404);

        // Reset is a new editing session boundary: replace the old canonical
        // working image with a fresh copy of the immutable source and remove
        // persisted history.
        try {
            std::string rid;
            live_store->with_session(session_id, [&](tile_compile::pi::LiveImageSession& s) {
                rid = s.run_id;
            });
            if (!rid.empty()) {
                auto run_dir = state->runtime.resolve_run_dir(rid);
                const auto outputs_dir = run_dir / "outputs";
                std::filesystem::remove(outputs_dir / "live_edit.fits");
                // Remove derived previews as well; live_edit.fits is the only
                // canonical reset result and stale PNG/JPEG files would make
                // the run preview disagree with it.
                for (const char* name : {"live_edit.png", "live_edit.jpg", "live_edit.jpeg"})
                    std::filesystem::remove(outputs_dir / name);
            }
        } catch (...) {}

        // Materialize the reset state again as the new canonical working FITS
        // so the run preview immediately reflects the reset image. History
        // remains deleted; this file contains the untouched source pixels.
        try { persist_live_edit_fits(state, live_store, session_id); } catch (...) {}

        // Delete chat history file entirely
        try {
            std::string rid;
            live_store->with_session(session_id, [&](tile_compile::pi::LiveImageSession& s) {
                rid = s.run_id;
            });
            if (!rid.empty()) {
                auto history_path = pi_live_image_chat_history_path(state, rid);
                std::filesystem::remove(history_path);
            }
        } catch (...) {}

        return json_resp({
            {"image_base64", mat_to_jpeg_base64(img, 85)},
            {"image_mime", "image/jpeg"},
            {"can_undo", false},
            {"can_redo", false}
        });
    });

    // 7. POST /api/pi/live-image-chat/export
    CROW_ROUTE(app, "/api/pi/live-image-chat/export").methods("POST"_method)
    ([state, live_store](const crow::request& req) {
        auto body = parse_body(req);
        if (!body) return err_resp("BAD_REQUEST", "Invalid JSON", 400);
        const std::string session_id = body->value("session_id", std::string());
        const std::string format = body->value("format", std::string("png"));
        if (session_id.empty()) return err_resp("BAD_REQUEST", "session_id is required", 400);

        cv::Mat img;
        std::string run_id;
        bool found = live_store->with_session(session_id, [&](tile_compile::pi::LiveImageSession& s) {
            img = s.current_fits.clone();
            run_id = s.run_id;
        });
        if (!found) return err_resp("NOT_FOUND", "session not found", 404);
        if (img.empty()) return err_resp("INTERNAL", "current image is empty", 500);

        fs::path run_dir;
        try {
            run_dir = state->runtime.resolve_run_dir(run_id);
        } catch (const std::runtime_error&) {
            return err_resp("NOT_FOUND", "run directory not found", 404);
        }

        const fs::path output_dir = run_dir / "outputs";
        std::error_code ec;
        fs::create_directories(output_dir, ec);

        std::string filename = "live_image_export_" + session_id.substr(0, 8);
        std::string ext = (format == "fits") ? ".fits" : ".png";
        fs::path output_path = output_dir / (filename + ext);

        if (format == "fits") {
            // Export the linear float FITS working copy as a proper FITS file.
            write_float_bgr_to_fits(img, output_path);
        } else {
            // Render float data to an 8-bit sRGB PNG preview.
            cv::Mat display = render_float_bgr_to_bgr8(img);
            if (!cv::imwrite(output_path.string(), display))
                return err_resp("EXPORT_FAILED", "Failed to write PNG file", 500);
        }

        return json_resp({
            {"ok", true},
            {"path", output_path.string()},
            {"format", format}
        });
    });

    // Persistent timeline presets shared by all runs.
    CROW_ROUTE(app, "/api/pi/live-image-chat/presets").methods("GET"_method)
    ([state](const crow::request&) {
        nlohmann::json items = nlohmann::json::array();
        const auto dir = pi_live_image_presets_dir(state);
        std::error_code ec;
        if (std::filesystem::is_directory(dir, ec)) {
            for (const auto& entry : std::filesystem::directory_iterator(dir, ec)) {
                if (ec || !entry.is_regular_file() || entry.path().extension() != ".json") continue;
                std::ifstream in(entry.path());
                auto preset = nlohmann::json::parse(in, nullptr, false);
                if (!preset.is_discarded() && preset.is_object()) items.push_back(live_preset_summary(preset));
            }
        }
        std::sort(items.begin(), items.end(), [](const auto& a, const auto& b) {
            return a.value("name", std::string()) < b.value("name", std::string());
        });
        return json_resp({{"items", items}});
    });

    CROW_ROUTE(app, "/api/pi/live-image-chat/presets/save-as").methods("POST"_method)
    ([state, live_store](const crow::request& req) {
        auto body = parse_body(req);
        if (!body) return err_resp("BAD_REQUEST", "Invalid JSON", 400);
        const auto session_id = body->value("session_id", std::string());
        const auto name = body->value("name", std::string());
        const auto id = sanitize_live_preset_id(name);
        if (session_id.empty() || name.empty() || id.empty()) return err_resp("BAD_REQUEST", "session_id and a valid name are required", 400);
        if (read_live_preset(state, id).is_object()) return err_resp("CONFLICT", "Preset already exists", 409);
        nlohmann::json operations, timeline;
        bool found = live_store->with_session(session_id, [&](tile_compile::pi::LiveImageSession& s) {
            operations = s.operation_history;
            timeline = s.edit_history;
        });
        if (!found) return err_resp("NOT_FOUND", "session not found", 404);
        auto now = utc_now_iso();
        nlohmann::json preset = {{"schema_version", "pi.live-image-preset.v1"}, {"id", id}, {"name", name},
                                 {"operations", operations.is_array() ? operations : nlohmann::json::array()},
                                 {"edit_history", timeline.is_array() ? timeline : nlohmann::json::array()},
                                 {"created_at", now}, {"updated_at", now}};
        try { write_live_preset(state, preset); }
        catch (const std::exception& e) { return err_resp("SAVE_FAILED", e.what(), 500); }
        return json_resp({{"ok", true}, {"preset", live_preset_summary(preset)}});
    });

    CROW_ROUTE(app, "/api/pi/live-image-chat/presets/save").methods("POST"_method)
    ([state, live_store](const crow::request& req) {
        auto body = parse_body(req);
        if (!body) return err_resp("BAD_REQUEST", "Invalid JSON", 400);
        const auto session_id = body->value("session_id", std::string());
        const auto id = body->value("preset_id", std::string());
        auto preset = read_live_preset(state, id);
        if (session_id.empty() || id.empty()) return err_resp("BAD_REQUEST", "session_id and preset_id are required", 400);
        if (!preset.is_object()) return err_resp("NOT_FOUND", "preset not found", 404);
        nlohmann::json operations, timeline;
        if (!live_store->with_session(session_id, [&](tile_compile::pi::LiveImageSession& s) {
            operations = s.operation_history; timeline = s.edit_history;
        })) return err_resp("NOT_FOUND", "session not found", 404);
        preset["operations"] = operations.is_array() ? operations : nlohmann::json::array();
        preset["edit_history"] = timeline.is_array() ? timeline : nlohmann::json::array();
        preset["updated_at"] = utc_now_iso();
        try { write_live_preset(state, preset); }
        catch (const std::exception& e) { return err_resp("SAVE_FAILED", e.what(), 500); }
        return json_resp({{"ok", true}, {"preset", live_preset_summary(preset)}});
    });

    CROW_ROUTE(app, "/api/pi/live-image-chat/presets/apply").methods("POST"_method)
    ([state, live_store](const crow::request& req) {
        auto body = parse_body(req);
        if (!body) return err_resp("BAD_REQUEST", "Invalid JSON", 400);
        const auto session_id = body->value("session_id", std::string());
        const auto id = body->value("preset_id", std::string());
        if (session_id.empty() || id.empty()) return err_resp("BAD_REQUEST", "session_id and preset_id are required", 400);
        auto preset = read_live_preset(state, id);
        if (!preset.is_object()) return err_resp("NOT_FOUND", "preset not found", 404);
        auto result = live_store->apply_preset(session_id, preset.value("operations", nlohmann::json::array()));
        if (!result.success) return err_resp("PRESET_FAILED", result.error, 400);
        live_store->append_chat(session_id, "assistant", "Preset angewendet: " + preset.value("name", id),
                                preset.value("operations", nlohmann::json::array()));
        persist_live_edit_fits(state, live_store, session_id);
        try { persist_live_session(state, live_store, session_id); } catch (...) {}
        bool can_undo = false;
        live_store->with_session(session_id, [&](tile_compile::pi::LiveImageSession& s) { can_undo = !s.undo_stack.empty(); });
        return json_resp({{"ok", true}, {"preset", live_preset_summary(preset)},
                          {"image_base64", mat_to_jpeg_base64(result.image, 85)}, {"image_mime", "image/jpeg"},
                          {"can_undo", can_undo}, {"can_redo", false}});
    });

    // 8. GET /api/pi/live-image-chat/history
    CROW_ROUTE(app, "/api/pi/live-image-chat/history").methods("GET"_method)
    ([state](const crow::request& req) {
        const std::string run_id = req.url_params.get("run_id") ? std::string(req.url_params.get("run_id")) : "";
        if (run_id.empty()) return err_resp("BAD_REQUEST", "run_id is required", 400);

        return json_resp(read_pi_live_image_chat_history(state, run_id));
    });

    // 9. POST /api/pi/live-image-chat/close
    CROW_ROUTE(app, "/api/pi/live-image-chat/close").methods("POST"_method)
    ([state, live_store](const crow::request& req) {
        auto body = parse_body(req);
        if (!body) return err_resp("BAD_REQUEST", "Invalid JSON", 400);
        const std::string session_id = body->value("session_id", std::string());
        if (session_id.empty()) return err_resp("BAD_REQUEST", "session_id is required", 400);

        // Persist chat and operation history before closing
        try { persist_live_session(state, live_store, session_id); } catch (...) {}

        live_store->close(session_id);
        return json_resp({{"ok", true}});
    });
}
