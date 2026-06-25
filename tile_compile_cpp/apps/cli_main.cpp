#include "tile_compile/core/types.hpp"
#include "tile_compile/config/configuration.hpp"
#include "tile_compile/io/fits_io.hpp"
#include "tile_compile/astrometry/photometric_color_cal.hpp"
#include "tile_compile/metrics/metrics.hpp"

#include <nlohmann/json.hpp>
#include <yaml-cpp/yaml.h>
#include <fitsio.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <regex>
#include <sstream>
#include <string>
#include <vector>
#include <openssl/evp.h>

#include <atomic>
#include <cctype>
#include <mutex>
#include <thread>

namespace fs = std::filesystem;
using json = nlohmann::json;

static std::string to_lower_copy(std::string value);
static bool parse_boolish(const std::string& value);

/// @brief Implements get executable dir.
/// @details Part of the GUI/CLI adapter that exposes configuration, FITS inspection, run listing, and artifact commands; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
static std::string get_executable_dir() {
    return fs::current_path().string();
}

/// @brief Returns the default gui state path.
/// @details Part of the GUI/CLI adapter that exposes configuration, FITS inspection, run listing, and artifact commands; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
static std::string default_gui_state_path() {
    return (fs::path(get_executable_dir()) / "tile_compile_state.json").string();
}

/// @brief Implements print json.
/// @details Part of the GUI/CLI adapter that exposes configuration, FITS inspection, run listing, and artifact commands; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
static void print_json(const json& j) {
    std::cout << j.dump(2) << std::endl;
}

/// @brief Reads file text.
/// @details Part of the GUI/CLI adapter that exposes configuration, FITS inspection, run listing, and artifact commands; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
static std::string read_file_text(const fs::path& p) {
    std::ifstream ifs(p);
    if (!ifs) return "";
    std::ostringstream ss;
    ss << ifs.rdbuf();
    return ss.str();
}

/// @brief Writes file text.
/// @details Part of the GUI/CLI adapter that exposes configuration, FITS inspection, run listing, and artifact commands; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
static bool write_file_text(const fs::path& p, const std::string& content) {
    std::ofstream ofs(p);
    if (!ofs) return false;
    ofs << content;
    return true;
}

/// @brief Reads stdin.
/// @details Part of the GUI/CLI adapter that exposes configuration, FITS inspection, run listing, and artifact commands; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
static std::string read_stdin() {
    std::ostringstream ss;
    ss << std::cin.rdbuf();
    return ss.str();
}

/// @brief Implements yaml node to json.
/// @details Part of the GUI/CLI adapter that exposes configuration, FITS inspection, run listing, and artifact commands; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
static json yaml_node_to_json(const YAML::Node& node) {
    if (node.IsMap()) {
        json out = json::object();
        for (auto it = node.begin(); it != node.end(); ++it) {
            out[it->first.as<std::string>()] = yaml_node_to_json(it->second);
        }
        return out;
    }
    if (node.IsSequence()) {
        json out = json::array();
        for (auto it = node.begin(); it != node.end(); ++it) out.push_back(yaml_node_to_json(*it));
        return out;
    }
    if (node.IsScalar()) {
        const std::string raw = node.Scalar();
        if (raw == "true") return true;
        if (raw == "false") return false;
        char* end = nullptr;
        errno = 0;
        long long i = std::strtoll(raw.c_str(), &end, 10);
        if (end && end != raw.c_str() && *end == '\0' && errno == 0) return i;
        errno = 0;
        double d = std::strtod(raw.c_str(), &end);
        if (end && end != raw.c_str() && *end == '\0' && errno == 0) return d;
        return raw;
    }
    return nullptr;
}

/// @brief Implements scalar looks like float.
/// @details Part of the GUI/CLI adapter that exposes configuration, FITS inspection, run listing, and artifact commands; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
static bool scalar_looks_like_float(const std::string& raw) {
    return raw.find('.') != std::string::npos ||
           raw.find('e') != std::string::npos ||
           raw.find('E') != std::string::npos;
}

/// @brief Implements trim trailing zeros.
/// @details Part of the GUI/CLI adapter that exposes configuration, FITS inspection, run listing, and artifact commands; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
static std::string trim_trailing_zeros(std::string text) {
    const auto dot = text.find('.');
    if (dot == std::string::npos) return text;
    while (!text.empty() && text.back() == '0') text.pop_back();
    if (!text.empty() && text.back() == '.') text.pop_back();
    if (text == "-0") return "0";
    return text.empty() ? "0" : text;
}

/// @brief Formats config float scalar.
/// @details Part of the GUI/CLI adapter that exposes configuration, FITS inspection, run listing, and artifact commands; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
static std::string format_config_float_scalar(double value) {
    if (!std::isfinite(value)) return "0";

    const double rounded = std::round(value * 100.0) / 100.0;
    if (rounded == 0.0 && value != 0.0 && std::fabs(value) < 0.01) {
        std::ostringstream oss;
        oss << std::scientific << std::setprecision(2) << value;
        return oss.str();
    }

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << rounded;
    return trim_trailing_zeros(oss.str());
}

/// @brief Implements round yaml numeric scalars inplace.
/// @details Part of the GUI/CLI adapter that exposes configuration, FITS inspection, run listing, and artifact commands; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
static void round_yaml_numeric_scalars_inplace(YAML::Node node) {
    if (!node || node.IsNull()) return;
    if (node.IsMap()) {
        for (auto it = node.begin(); it != node.end(); ++it) round_yaml_numeric_scalars_inplace(it->second);
        return;
    }
    if (node.IsSequence()) {
        for (auto it = node.begin(); it != node.end(); ++it) round_yaml_numeric_scalars_inplace(*it);
        return;
    }
    if (!node.IsScalar()) return;

    const std::string raw = node.Scalar();
    if (!scalar_looks_like_float(raw)) return;

    char* end = nullptr;
    errno = 0;
    const double value = std::strtod(raw.c_str(), &end);
    if (errno != 0 || end == raw.c_str() || (end && *end != '\0') || !std::isfinite(value)) return;
    node = format_config_float_scalar(value);
}

/// @brief Computes sha256 file.
/// @details Part of the GUI/CLI adapter that exposes configuration, FITS inspection, run listing, and artifact commands; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
static std::string compute_sha256_file(const fs::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) return "";

    EVP_MD_CTX* ctx = EVP_MD_CTX_new();
    if (!ctx) return "";
    if (EVP_DigestInit_ex(ctx, EVP_sha256(), nullptr) != 1) {
        EVP_MD_CTX_free(ctx);
        return "";
    }

    char buffer[8192];
    while (file.read(buffer, sizeof(buffer)) || file.gcount() > 0) {
        if (EVP_DigestUpdate(ctx, buffer, static_cast<size_t>(file.gcount())) != 1) {
            EVP_MD_CTX_free(ctx);
            return "";
        }
    }

    unsigned char hash[EVP_MAX_MD_SIZE];
    unsigned int hash_len = 0;
    if (EVP_DigestFinal_ex(ctx, hash, &hash_len) != 1) {
        EVP_MD_CTX_free(ctx);
        return "";
    }
    EVP_MD_CTX_free(ctx);

    std::ostringstream oss;
    for (unsigned int i = 0; i < hash_len; ++i) {
        oss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(hash[i]);
    }
    return oss.str();
}

struct FitsHeaderInfo {
    int naxis = 0;
    int naxis1 = 0;
    int naxis2 = 0;
    std::string bayerpat;
    bool has_bayerpat = false;
    double exptime = -1.0;      // EXPTIME / EXPOSURE
    double gain = -1.0;         // GAIN / EGAIN
    double ccd_temp = -999.0;   // CCD-TEMP / SET-TEMP
    std::string object;         // OBJECT (target name)
    std::string instrume;       // INSTRUME (camera)
    std::string telescop;       // TELESCOP
    std::string filter;         // FILTER
    std::string date_obs;       // DATE-OBS
    double ra_deg = -999.0;      // RA in degrees (from RA, OBJCTRA, or SIT-RA)
    double dec_deg = -999.0;     // DEC in degrees (from DEC, OBJCTDEC, or SIT-DEC)
    bool read_error = false;
    std::string error_msg;
};

/// @brief Reads fits header info.
/// @details Part of the GUI/CLI adapter that exposes configuration, FITS inspection, run listing, and artifact commands; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
static FitsHeaderInfo read_fits_header_info(const fs::path& path) {
    FitsHeaderInfo info;
    fitsfile* fptr = nullptr;
    int status = 0;
    
    if (fits_open_file(&fptr, path.string().c_str(), READONLY, &status)) {
        info.read_error = true;
        info.error_msg = "Cannot open FITS file";
        return info;
    }
    
    int naxis = 0;
    long naxes[3] = {0, 0, 0};
    int bitpix = 0;

    auto try_read_bayerpat = [&](int &bp_status) {
        char bayerpat[FLEN_VALUE];
        char comment[FLEN_COMMENT];
        fits_read_key(fptr, TSTRING, const_cast<char*>("BAYERPAT"), bayerpat, comment, &bp_status);
        if (bp_status == 0) {
            std::string bp_str(bayerpat);
            // Trim whitespace and quotes
            bp_str.erase(0, bp_str.find_first_not_of(" \t\n\r'\""));
            bp_str.erase(bp_str.find_last_not_of(" \t\n\r'\"") + 1);
            if (!bp_str.empty()) {
                info.has_bayerpat = true;
                info.bayerpat = bp_str;
                std::transform(info.bayerpat.begin(), info.bayerpat.end(), info.bayerpat.begin(), ::toupper);
            }
        }
    };

    // Primary HDU often carries BAYERPAT even if image data is in extension.
    int bp_status = 0;
    try_read_bayerpat(bp_status);

    bool found_image_hdu = false;
    status = 0;
    fits_get_img_param(fptr, 3, &bitpix, &naxis, naxes, &status);
    if (status == 0 && naxis >= 2) {
        found_image_hdu = true;
    }

    if (!found_image_hdu) {
        status = 0;
        int nhdus = 0;
        fits_get_num_hdus(fptr, &nhdus, &status);
        if (!status) {
            for (int hdu = 2; hdu <= nhdus; ++hdu) {
                int hdu_type = 0;
                status = 0;
                fits_movabs_hdu(fptr, hdu, &hdu_type, &status);
                if (status) continue;
                status = 0;
                fits_get_img_param(fptr, 3, &bitpix, &naxis, naxes, &status);
                if (status == 0 && naxis >= 2) {
                    found_image_hdu = true;
                    break;
                }
            }
        }
    }

    if (!found_image_hdu) {
        info.read_error = true;
        info.error_msg = "Cannot read FITS image parameters";
        fits_close_file(fptr, &status);
        return info;
    }
    
    info.naxis = naxis;
    info.naxis1 = static_cast<int>(naxes[0]);
    info.naxis2 = static_cast<int>(naxes[1]);
    
    // If not in primary, try current image HDU as fallback.
    if (!info.has_bayerpat) {
        int bp2_status = 0;
        try_read_bayerpat(bp2_status);
    }

    // Read additional metadata keywords (best-effort, ignore errors)
    auto read_double_key = [&](const char* key) -> double {
        double val = 0.0;
        int ks = 0;
        fits_read_key(fptr, TDOUBLE, const_cast<char*>(key), &val, nullptr, &ks);
        return (ks == 0) ? val : -1.0;
    };
    auto read_string_key = [&](const char* key) -> std::string {
        char val[FLEN_VALUE];
        int ks = 0;
        fits_read_key(fptr, TSTRING, const_cast<char*>(key), val, nullptr, &ks);
        if (ks != 0) return "";
        std::string s(val);
        s.erase(0, s.find_first_not_of(" \t\n\r'\""));
        s.erase(s.find_last_not_of(" \t\n\r'\"") + 1);
        return s;
    };

    // Move back to primary HDU for header keywords
    {
        int ks = 0;
        fits_movabs_hdu(fptr, 1, nullptr, &ks);
    }
    double exp = read_double_key("EXPTIME");
    if (exp < 0) exp = read_double_key("EXPOSURE");
    info.exptime = exp;

    double gain = read_double_key("GAIN");
    if (gain < 0) gain = read_double_key("EGAIN");
    info.gain = gain;

    double temp = read_double_key("CCD-TEMP");
    if (temp < -900) temp = read_double_key("SET-TEMP");
    info.ccd_temp = temp;

    info.object = read_string_key("OBJECT");
    info.instrume = read_string_key("INSTRUME");
    info.telescop = read_string_key("TELESCOP");
    info.filter = read_string_key("FILTER");
    info.date_obs = read_string_key("DATE-OBS");

    // Read RA/DEC — try multiple keyword variants used by different capture software.
    // Values may be in degrees (float) or in sexagesimal hours/degrees (string like "05 35 17.3").
    auto read_ra_dec = [&]() {
        // Try numeric RA/DEC (degrees) first
        double ra_val = read_double_key("RA");
        double dec_val = read_double_key("DEC");
        if (ra_val >= 0 && dec_val > -900) {
            info.ra_deg = ra_val;
            info.dec_deg = dec_val;
            return;
        }
        // Try OBJCTRA/OBJCTDEC (sexagesimal, "HH MM SS.S" / "DD MM SS.S")
        std::string objctra = read_string_key("OBJCTRA");
        std::string objctdec = read_string_key("OBJCTDEC");
        if (!objctra.empty() && !objctdec.empty()) {
            // Parse sexagesimal: split on whitespace or colons
            auto parse_sex = [](const std::string& s, bool is_ra) -> double {
                // Replace colons with spaces for uniform splitting
                std::string cleaned = s;
                std::replace(cleaned.begin(), cleaned.end(), ':', ' ');
                std::istringstream iss(cleaned);
                double h = 0, m = 0, sec = 0;
                iss >> h >> m >> sec;
                if (iss.fail()) return -999.0;
                double deg = is_ra ? (h + m / 60.0 + sec / 3600.0) * 15.0
                                   : (std::abs(h) + m / 60.0 + sec / 3600.0);
                if (!is_ra && !s.empty() && s[0] == '-') deg = -deg;
                return deg;
            };
            double ra = parse_sex(objctra, true);
            double dec = parse_sex(objctdec, false);
            if (ra >= 0 && ra < 360 && dec > -90 && dec < 90) {
                info.ra_deg = ra;
                info.dec_deg = dec;
                return;
            }
        }
        // Try SIT-RA/SIT-DEC (some mounts write telescope position)
        double sit_ra = read_double_key("SIT-RA");
        double sit_dec = read_double_key("SIT-DEC");
        if (sit_ra >= 0 && sit_dec > -900) {
            info.ra_deg = sit_ra;
            info.dec_deg = sit_dec;
        }
    };
    read_ra_dec();

    fits_close_file(fptr, &status);
    return info;
}

/// @brief Computes fits stats buffer.
/// @details Part of the GUI/CLI adapter that exposes configuration, FITS inspection, run listing, and artifact commands; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
static json compute_fits_stats_buffer(const std::vector<float>& buf) {
    double mean = 0.0;
    double m2 = 0.0;
    int64_t n = 0;

    float min_v = std::numeric_limits<float>::infinity();
    float max_v = -std::numeric_limits<float>::infinity();
    int64_t n_nan = 0;
    int64_t n_inf = 0;
    int64_t n_zero = 0;

    for (float v : buf) {
        if (std::isnan(v)) {
            n_nan++;
            continue;
        }
        if (!std::isfinite(v)) {
            n_inf++;
            continue;
        }

        if (v == 0.0f) n_zero++;
        if (v < min_v) min_v = v;
        if (v > max_v) max_v = v;

        n++;
        const double x = static_cast<double>(v);
        const double delta = x - mean;
        mean += delta / static_cast<double>(n);
        const double delta2 = x - mean;
        m2 += delta * delta2;
    }

    json j;
    j["count"] = n;
    j["nan"] = n_nan;
    j["inf"] = n_inf;
    j["zero"] = n_zero;
    j["min"] = (n > 0) ? min_v : 0.0f;
    j["max"] = (n > 0) ? max_v : 0.0f;
    j["mean"] = (n > 0) ? mean : 0.0;
    j["stddev"] = (n > 1) ? std::sqrt(m2 / static_cast<double>(n - 1)) : 0.0;
    return j;
}

/// @brief Implements fits stats file.
/// @details Part of the GUI/CLI adapter that exposes configuration, FITS inspection, run listing, and artifact commands; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
static json fits_stats_file(const fs::path& path) {
    json result;
    result["path"] = path.string();

    fitsfile* fptr = nullptr;
    int status = 0;
    if (fits_open_file(&fptr, path.string().c_str(), READONLY, &status)) {
        result["ok"] = false;
        result["error"] = "Cannot open FITS file";
        return result;
    }

    int naxis = 0;
    long naxes[3] = {0, 0, 0};
    int bitpix = 0;
    fits_get_img_param(fptr, 3, &bitpix, &naxis, naxes, &status);
    if (status || naxis < 2) {
        fits_close_file(fptr, &status);
        result["ok"] = false;
        result["error"] = "Cannot read FITS image parameters";
        return result;
    }

    result["ok"] = true;
    result["bitpix"] = bitpix;
    result["naxis"] = naxis;
    result["naxes"] = json::array();
    for (int i = 0; i < naxis; ++i) result["naxes"].push_back(static_cast<int64_t>(naxes[i]));

    long width = naxes[0];
    long height = naxes[1];
    long depth = (naxis >= 3) ? naxes[2] : 1;
    const int64_t plane_pixels = static_cast<int64_t>(width) * static_cast<int64_t>(height);
    const int64_t total_pixels = plane_pixels * static_cast<int64_t>(depth);
    result["pixels"] = total_pixels;

    std::vector<float> buffer(static_cast<size_t>(total_pixels));
    long fpixel[3] = {1, 1, 1};
    if (fits_read_pix(fptr, TFLOAT, fpixel, static_cast<long>(total_pixels), nullptr, buffer.data(), nullptr, &status)) {
        fits_close_file(fptr, &status);
        result["ok"] = false;
        result["error"] = "Cannot read FITS pixel data";
        return result;
    }
    fits_close_file(fptr, &status);

    result["stats"] = compute_fits_stats_buffer(buffer);

    if (depth > 1) {
        result["per_plane_stats"] = json::array();
        for (long z = 0; z < depth; ++z) {
            const int64_t start = static_cast<int64_t>(z) * plane_pixels;
            const int64_t end = start + plane_pixels;
            std::vector<float> plane;
            plane.reserve(static_cast<size_t>(plane_pixels));
            for (int64_t i = start; i < end; ++i) plane.push_back(buffer[static_cast<size_t>(i)]);
            json pj = compute_fits_stats_buffer(plane);
            pj["plane"] = z;
            result["per_plane_stats"].push_back(pj);
        }
    }

    return result;
}

// ============================================================================
// get-schema
// ============================================================================
/// @brief Handles CLI command get schema.
/// @details Part of the GUI/CLI adapter that exposes configuration, FITS inspection, run listing, and artifact commands; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
int cmd_get_schema() {
    std::cout << tile_compile::config::get_schema_json() << std::endl;
    return 0;
}

/// @brief Handles CLI command dump default config.
/// @details Part of the GUI/CLI adapter that exposes configuration, FITS inspection, run listing, and artifact commands; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
int cmd_dump_default_config() {
    tile_compile::config::Config cfg;
    YAML::Node node = cfg.to_yaml();
    round_yaml_numeric_scalars_inplace(node);
    YAML::Emitter emitter;
    emitter << node;

    json result;
    result["ok"] = true;
    result["config"] = yaml_node_to_json(node);
    result["yaml"] = std::string(emitter.c_str());
    print_json(result);
    return 0;
}

// ============================================================================
// load-gui-state [--path <path>]
// ============================================================================
/// @brief Handles CLI command load gui state.
/// @details Part of the GUI/CLI adapter that exposes configuration, FITS inspection, run listing, and artifact commands; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
int cmd_load_gui_state(const std::string& path_arg) {
    fs::path p = path_arg.empty() ? default_gui_state_path() : path_arg;
    
    json state = json::object();
    if (fs::exists(p) && fs::is_regular_file(p)) {
        try {
            std::string raw = read_file_text(p);
            state = json::parse(raw);
            if (!state.is_object()) state = json::object();
        } catch (...) {
            state = json::object();
        }
    }
    
    json result;
    result["ok"] = true;
    result["path"] = p.string();
    result["state"] = state;
    print_json(result);
    return 0;
}

/// @brief Handles CLI command fits stats.
/// @details Part of the GUI/CLI adapter that exposes configuration, FITS inspection, run listing, and artifact commands; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
int cmd_fits_stats(const std::string& path) {
    print_json(fits_stats_file(fs::path(path)));
    return 0;
}

// ============================================================================
// pcc-apply <input_rgb_fits> <output_rgb_fits> [--r X] [--g Y] [--b Z]
//
// Applies a diagonal color matrix to an RGB FITS cube using the same
// implementation as the pipeline PCC step. This is mainly useful for
// debugging / reproducing PCC output changes without rerunning the pipeline.
// ============================================================================
/// @brief Handles CLI command pcc apply.
/// @details Part of the GUI/CLI adapter that exposes configuration, FITS inspection, run listing, and artifact commands; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
int cmd_pcc_apply(const std::string& input_path, const std::string& output_path,
                  double r_scale, double g_scale, double b_scale) {
    using tile_compile::Matrix2Df;
    namespace io = tile_compile::io;
    namespace astro = tile_compile::astrometry;

    io::RGBImage rgb = io::read_fits_rgb(fs::path(input_path));

    astro::ColorMatrix m = {{{r_scale, 0.0, 0.0},
                             {0.0, g_scale, 0.0},
                             {0.0, 0.0, b_scale}}};

    astro::apply_color_matrix(rgb.R, rgb.G, rgb.B, m);

    io::write_fits_rgb(fs::path(output_path), rgb.R, rgb.G, rgb.B, rgb.header);

    json result;
    result["ok"] = true;
    result["input"] = input_path;
    result["output"] = output_path;
    result["matrix"] = json::array({
        json::array({r_scale, 0.0, 0.0}),
        json::array({0.0, g_scale, 0.0}),
        json::array({0.0, 0.0, b_scale}),
    });
    print_json(result);
    return 0;
}

// ============================================================================
// save-gui-state [--path <path>] [--stdin | <json>]
// ============================================================================
/// @brief Handles CLI command save gui state.
/// @details Part of the GUI/CLI adapter that exposes configuration, FITS inspection, run listing, and artifact commands; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
int cmd_save_gui_state(const std::string& path_arg, const std::string& json_text, bool use_stdin) {
    fs::path p = path_arg.empty() ? default_gui_state_path() : path_arg;
    
    std::string raw = use_stdin ? read_stdin() : json_text;
    if (raw.empty()) {
        std::cerr << "save-gui-state requires JSON text either as argument or via --stdin\n";
        return 2;
    }
    
    json obj;
    try {
        obj = json::parse(raw);
    } catch (const std::exception& e) {
        std::cerr << "save-gui-state: failed to parse JSON: " << e.what() << "\n";
        return 2;
    }
    
    if (!obj.is_object()) {
        std::cerr << "save-gui-state: state must be a JSON object\n";
        return 2;
    }
    
    if (!write_file_text(p, obj.dump(2) + "\n")) {
        std::cerr << "save-gui-state: failed to write file: " << p.string() << "\n";
        return 1;
    }
    
    json result;
    result["ok"] = true;
    result["path"] = p.string();
    result["saved"] = true;
    print_json(result);
    return 0;
}

// ============================================================================
// load-config <path>
// ============================================================================
/// @brief Handles CLI command load config.
/// @details Part of the GUI/CLI adapter that exposes configuration, FITS inspection, run listing, and artifact commands; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
int cmd_load_config(const std::string& path) {
    fs::path p(path);
    if (!fs::exists(p)) {
        json result;
        result["ok"] = false;
        result["error"] = "File not found: " + path;
        print_json(result);
        return 1;
    }
    
    std::string yaml_text = read_file_text(p);
    json result;
    result["path"] = path;
    result["yaml"] = yaml_text;
    print_json(result);
    return 0;
}

// ============================================================================
// save-config <path> [--stdin | <yaml>]
// ============================================================================
/// @brief Handles CLI command save config.
/// @details Part of the GUI/CLI adapter that exposes configuration, FITS inspection, run listing, and artifact commands; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
int cmd_save_config(const std::string& path, const std::string& yaml_text, bool use_stdin) {
    std::string content = use_stdin ? read_stdin() : yaml_text;
    if (content.empty()) {
        std::cerr << "save-config requires YAML text either as argument or via --stdin\n";
        return 2;
    }
    
    if (!write_file_text(path, content)) {
        std::cerr << "save-config: failed to write file: " << path << "\n";
        return 1;
    }
    
    json result;
    result["path"] = path;
    result["saved"] = true;
    print_json(result);
    return 0;
}

// ============================================================================
// validate-config --path <path> | --yaml <yaml> | --stdin
// ============================================================================
/// @brief Handles CLI command validate config.
/// @details Part of the GUI/CLI adapter that exposes configuration, FITS inspection, run listing, and artifact commands; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
int cmd_validate_config(const std::string& path, const std::string& yaml_arg, bool use_stdin, bool strict_exit) {
    std::string yaml_text;
    if (!path.empty()) {
        yaml_text = read_file_text(path);
    } else if (use_stdin) {
        yaml_text = read_stdin();
    } else {
        yaml_text = yaml_arg;
    }
    
    json result;
    result["valid"] = false;
    result["errors"] = json::array();
    result["warnings"] = json::array();
    if (!path.empty()) result["path"] = path;
    
    try {
        tile_compile::config::Config cfg = tile_compile::config::Config::from_yaml_text(yaml_text);
        cfg.validate();
        result["valid"] = true;
    } catch (const std::exception& e) {
        result["errors"].push_back(e.what());
    }
    
    print_json(result);
    if (strict_exit) {
        return result["valid"].get<bool>() ? 0 : 1;
    }
    return 0;
}

// ============================================================================
// scan <input_path> [--frames-min N] [--with-checksums]
// ============================================================================
/// @brief Implements find fits files.
/// @details Part of the GUI/CLI adapter that exposes configuration, FITS inspection, run listing, and artifact commands; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
static std::vector<fs::path> find_fits_files(const fs::path& dir) {
    std::vector<fs::path> files;
    if (!fs::exists(dir) || !fs::is_directory(dir)) return files;

    for (const auto& entry : fs::directory_iterator(dir)) {
        if (entry.is_regular_file()) {
            if (tile_compile::io::is_fits_image_path(entry.path())) {
                files.push_back(entry.path());
            }
        }
    }
    std::sort(files.begin(), files.end());
    return files;
}

/// @brief Handles CLI command scan.
/// @details Part of the GUI/CLI adapter that exposes configuration, FITS inspection, run listing, and artifact commands; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
int cmd_scan(const std::string& input_path, int frames_min, bool with_checksums) {
    fs::path p(input_path);
    
    json result;
    result["ok"] = false;
    result["input_path"] = input_path;
    result["frames_detected"] = 0;
    result["frames"] = json::array();
    result["errors"] = json::array();
    result["warnings"] = json::array();
    result["color_mode"] = "UNKNOWN";
    result["bayer_pattern"] = nullptr;
    result["color_mode_candidates"] = json::array();
    result["requires_user_confirmation"] = false;
    result["image_width"] = 0;
    result["image_height"] = 0;
    
    if (!fs::exists(p)) {
        json err;
        err["severity"] = "error";
        err["code"] = "input_path_not_found";
        err["message"] = "Input path does not exist: " + input_path;
        result["errors"].push_back(err);
        print_json(result);
        return 0;
    }
    
    if (!fs::is_directory(p)) {
        json err;
        err["severity"] = "error";
        err["code"] = "input_path_not_directory";
        err["message"] = "Input path is not a directory: " + input_path;
        result["errors"].push_back(err);
        print_json(result);
        return 0;
    }
    
    auto files = find_fits_files(p);
    result["frames_detected"] = static_cast<int>(files.size());
    
    if (static_cast<int>(files.size()) < frames_min) {
        json err;
        err["severity"] = "error";
        err["code"] = "too_few_frames";
        err["message"] = "frames_detected (" + std::to_string(files.size()) + 
                        ") < frames_min (" + std::to_string(frames_min) + ")";
        result["errors"].push_back(err);
    }
    
    // Scan FITS headers
    int image_width = 0;
    int image_height = 0;
    bool has_bayerpat = false;
    bool has_rgb_cube = false;
    std::string bayer_pattern;
    bool bayer_pattern_inconsistent = false;
    std::map<std::string, int> bayerpat_counts;
    
    for (const auto& f : files) {
        json frame;
        frame["file_name"] = f.filename().string();
        frame["abs_path"] = f.string();
        
        if (with_checksums) {
            frame["sha256"] = compute_sha256_file(f);
        }
        
        // Read FITS header
        FitsHeaderInfo info = read_fits_header_info(f);
        
        if (info.read_error) {
            json err;
            err["severity"] = "error";
            err["code"] = "fits_read_error";
            err["message"] = "Failed to read FITS header for " + f.filename().string() + ": " + info.error_msg;
            result["errors"].push_back(err);
            continue;
        }
        
        if (info.naxis1 <= 0 || info.naxis2 <= 0) {
            json err;
            err["severity"] = "error";
            err["code"] = "fits_missing_axis";
            err["message"] = "Missing or invalid NAXIS1/NAXIS2 in FITS header for " + f.filename().string();
            result["errors"].push_back(err);
            continue;
        }
        
        // Check dimension consistency
        if (image_width == 0) {
            image_width = info.naxis1;
            image_height = info.naxis2;
        } else {
            if (info.naxis1 != image_width || info.naxis2 != image_height) {
                json err;
                err["severity"] = "error";
                err["code"] = "inconsistent_image_dimensions";
                err["message"] = "Inconsistent image size: expected " + std::to_string(image_width) + "x" + 
                                std::to_string(image_height) + ", got " + std::to_string(info.naxis1) + "x" + 
                                std::to_string(info.naxis2) + " in " + f.filename().string();
                result["errors"].push_back(err);
            }
        }

        if (info.naxis >= 3) {
            has_rgb_cube = true;
        }
        
        // Track BAYERPAT
        if (info.has_bayerpat) {
            has_bayerpat = true;
            const std::string& bp = info.bayerpat;
            
            // Check if it's a valid Bayer pattern
            if (bp == "RGGB" || bp == "BGGR" || bp == "GBRG" || bp == "GRBG") {
                bayerpat_counts[bp]++;
                
                if (bayer_pattern.empty()) {
                    bayer_pattern = bp;
                } else if (bayer_pattern != bp) {
                    bayer_pattern_inconsistent = true;
                }
            }
        }
        
        if (info.exptime >= 0) frame["exposure_seconds"] = info.exptime;
        if (info.gain >= 0) frame["gain"] = info.gain;
        if (info.ccd_temp > -900) frame["temperature_c"] = info.ccd_temp;
        if (!info.object.empty()) frame["target"] = info.object;
        if (!info.instrume.empty()) frame["camera"] = info.instrume;
        if (!info.telescop.empty()) frame["telescope"] = info.telescop;

        result["frames"].push_back(frame);
    }
    
    result["image_width"] = image_width;
    result["image_height"] = image_height;
    
    // Determine color mode
    bool requires_user_confirmation = false;
    std::string color_mode = "UNKNOWN";
    std::vector<std::string> candidates;

    const bool has_readable_frames = (image_width > 0 && image_height > 0);

    if (has_readable_frames) {
        if (has_rgb_cube) {
            candidates.push_back("RGB");
        }
        if (has_bayerpat) {
            candidates.push_back("OSC");
        }
        if (!has_rgb_cube && !has_bayerpat) {
            // No RGB cube and no CFA hint: default to MONO (same fallback philosophy
            // as pipeline SCAN_INPUT for hint-less FITS).
            candidates.push_back("MONO");
        }

        if (candidates.empty()) {
            candidates.push_back("MONO");
        }

        color_mode = candidates.front();
    }

    if (has_readable_frames && candidates.size() > 1) {
        requires_user_confirmation = true;
        color_mode = "UNKNOWN";
        json warn;
        warn["severity"] = "warning";
        warn["code"] = "color_mode_ambiguous";
        warn["message"] = "Mixed color-mode hints across FITS headers (RGB/CFA); color_mode requires user confirmation";
        result["warnings"].push_back(warn);
    }

    if (has_readable_frames && has_bayerpat && bayer_pattern_inconsistent) {
        requires_user_confirmation = true;
        json warn;
        warn["severity"] = "warning";
        warn["code"] = "bayer_pattern_inconsistent";
        warn["message"] = "BAYERPAT differs across frames; bayer_pattern requires user confirmation";
        result["warnings"].push_back(warn);
    }
    
    result["color_mode"] = color_mode;
    if (bayer_pattern.empty()) {
        result["bayer_pattern"] = nullptr;
    } else {
        result["bayer_pattern"] = bayer_pattern;
    }
    result["color_mode_candidates"] = json(candidates);
    result["requires_user_confirmation"] = requires_user_confirmation;
    
    // Check if we have any readable frames
    if (image_width == 0 || image_height == 0) {
        json err;
        err["severity"] = "error";
        err["code"] = "no_readable_frames";
        err["message"] = "No readable FITS frames found";
        result["errors"].push_back(err);
    }
    
    if (result["errors"].empty()) {
        result["ok"] = true;
    }
    
    print_json(result);
    return 0;
}

// ============================================================================
// scan-metrics <input_path>
// ============================================================================

/// @brief Computes image-quality metrics for all frames in the input directory.
/// @details Reads all FITS headers and computes background, noise, gradient energy,
/// FWHM, roundness, and star count for every frame, then outputs per-frame and
/// aggregate statistics as JSON.
int cmd_scan_metrics(const std::string& input_path) {
    namespace io = tile_compile::io;
    namespace metrics = tile_compile::metrics;

    fs::path p(input_path);
    json result;
    result["ok"] = false;
    result["input_path"] = input_path;
    result["frames"] = json::array();
    result["aggregate"] = json::object();

    if (!fs::exists(p) || !fs::is_directory(p)) {
        result["error"] = "Input path does not exist or is not a directory";
        print_json(result);
        return 1;
    }

    auto all_files = find_fits_files(p);
    const int total = static_cast<int>(all_files.size());
    result["frames_total"] = total;

    if (total == 0) {
        result["error"] = "No FITS files found";
        print_json(result);
        return 1;
    }

    std::vector<FitsHeaderInfo> header_infos;
    header_infos.reserve(all_files.size());
    for (const auto& file : all_files) header_infos.push_back(read_fits_header_info(file));

    // All frames are processed — no sampling.
    std::vector<size_t> indices;
    indices.reserve(static_cast<size_t>(total));
    for (int i = 0; i < total; ++i) indices.push_back(static_cast<size_t>(i));
    const int sample_count = total;
    result["sample_count"] = sample_count;

    // Per-frame results storage
    struct FrameResult {
        int frame_index = -1;
        std::string file_name;
        float background = 0.0f;
        float noise = 0.0f;
        float gradient_energy = 0.0f;
        float fwhm = 0.0f;
        float fwhm_x = 0.0f;
        float fwhm_y = 0.0f;
        float roundness = 0.0f;
        int star_count = 0;
        bool ok = false;
        std::string error;
    };
    std::vector<FrameResult> results(indices.size());

    // Process frames in parallel
    const int hw_threads = std::max(1, static_cast<int>(std::thread::hardware_concurrency()));
    const int workers = std::min(hw_threads, static_cast<int>(indices.size()));
    std::atomic<size_t> next_idx{0};
    std::atomic<size_t> done_count{0};

    auto worker_fn = [&]() {
        while (true) {
            const size_t wi = next_idx.fetch_add(1);
            if (wi >= indices.size()) break;
            const size_t frame_idx = indices[wi];
            FrameResult& fr = results[wi];
            fr.frame_index = static_cast<int>(frame_idx);
            fr.file_name = all_files[frame_idx].filename().string();
            try {
                tile_compile::Matrix2Df img = io::read_fits_pixels_float(all_files[frame_idx]);
                if (img.size() <= 0) {
                    fr.error = "empty_image";
                    done_count.fetch_add(1);
                    continue;
                }
                auto fm = metrics::calculate_frame_metrics(img);
                fr.background = fm.background;
                fr.noise = fm.noise;
                fr.gradient_energy = fm.gradient_energy;

                auto sm = metrics::measure_frame_stars(img, 0);
                fr.fwhm = sm.fwhm;
                fr.fwhm_x = sm.fwhm_x;
                fr.fwhm_y = sm.fwhm_y;
                fr.roundness = sm.roundness;
                fr.star_count = sm.star_count;
                fr.ok = true;
            } catch (const std::exception& e) {
                fr.error = e.what();
            }
            done_count.fetch_add(1);
        }
    };

    std::cerr << "[scan-metrics] Analyzing all " << total
              << " frames using " << workers << " threads..." << std::endl;

    std::atomic<bool> progress_done{false};
    std::thread progress_thread([&]() {
        while (!progress_done.load()) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
            const size_t done = done_count.load();
            std::cerr << "[scan-metrics] " << done << " / " << total << " frames analyzed"
                      << (done == static_cast<size_t>(total) ? " - done." : "...") << std::endl;
        }
    });

    if (workers > 1) {
        std::vector<std::thread> threads;
        threads.reserve(static_cast<size_t>(workers));
        for (int w = 0; w < workers; ++w) threads.emplace_back(worker_fn);
        for (auto& t : threads) t.join();
    } else {
        worker_fn();
    }
    progress_done.store(true);
    progress_thread.join();
    std::cerr << "[scan-metrics] " << done_count.load() << " / " << total
              << " frames completed." << std::endl;

    // Build output
    std::vector<float> all_bg, all_noise, all_grad, all_fwhm, all_round;
    std::vector<int> all_stars;
    int ok_count = 0;

    for (const auto& fr : results) {
        json fj;
        fj["frame_index"] = fr.frame_index;
        fj["file_name"] = fr.file_name;
        fj["ok"] = fr.ok;
        if (fr.frame_index >= 0 && static_cast<size_t>(fr.frame_index) < header_infos.size()) {
            const auto& hi = header_infos[static_cast<size_t>(fr.frame_index)];
            json hj = json::object();
            if (!hi.read_error) {
                if (hi.exptime >= 0) hj["exposure_seconds"] = hi.exptime;
                if (hi.gain >= 0) hj["gain"] = hi.gain;
                if (hi.ccd_temp > -900) hj["temperature_c"] = hi.ccd_temp;
                if (!hi.filter.empty()) hj["filter"] = hi.filter;
                if (!hi.object.empty()) hj["target"] = hi.object;
                if (!hi.instrume.empty()) hj["camera"] = hi.instrume;
                if (!hi.date_obs.empty()) hj["date_obs"] = hi.date_obs;
                if (hi.ra_deg > -900) hj["ra_deg"] = hi.ra_deg;
                if (hi.dec_deg > -900) hj["dec_deg"] = hi.dec_deg;
            } else {
                hj["error"] = hi.error_msg;
            }
            if (!hj.empty()) fj["header"] = std::move(hj);
        }
        if (fr.ok) {
            fj["background"] = fr.background;
            fj["noise"] = fr.noise;
            fj["fwhm"] = fr.fwhm;
            fj["roundness"] = fr.roundness;
            fj["star_count"] = fr.star_count;
            all_bg.push_back(fr.background);
            all_noise.push_back(fr.noise);
            all_grad.push_back(fr.gradient_energy);
            if (fr.fwhm > 0) all_fwhm.push_back(fr.fwhm);
            if (fr.roundness > 0) all_round.push_back(fr.roundness);
            all_stars.push_back(fr.star_count);
            ++ok_count;
        } else {
            fj["error"] = fr.error;
        }
        result["frames"].push_back(fj);
    }

    // Aggregate statistics
    auto agg_stats = [](std::vector<float>& vals) -> json {
        if (vals.empty()) return nullptr;
        std::sort(vals.begin(), vals.end());
        float sum = 0;
        for (float v : vals) sum += v;
        float mean = sum / static_cast<float>(vals.size());
        float median = vals[vals.size() / 2];
        float p10 = vals[std::min(vals.size()-1, static_cast<size_t>(vals.size() * 0.1))];
        float p90 = vals[std::min(vals.size()-1, static_cast<size_t>(vals.size() * 0.9))];
        return json{{"min", vals.front()}, {"max", vals.back()},
                    {"mean", mean}, {"median", median},
                    {"p10", p10}, {"p90", p90}, {"count", static_cast<int>(vals.size())}};
    };

    json agg;
    agg["background"] = agg_stats(all_bg);
    agg["noise"] = agg_stats(all_noise);
    agg["gradient_energy"] = agg_stats(all_grad);
    agg["fwhm"] = agg_stats(all_fwhm);
    agg["roundness"] = agg_stats(all_round);
    if (!all_stars.empty()) {
        std::sort(all_stars.begin(), all_stars.end());
        int sum = 0;
        for (int v : all_stars) sum += v;
        agg["star_count"] = json{{"min", all_stars.front()}, {"max", all_stars.back()},
                                  {"mean", static_cast<float>(sum) / static_cast<float>(all_stars.size())},
                                  {"median", all_stars[all_stars.size() / 2]},
                                  {"count", static_cast<int>(all_stars.size())}};
    }
    agg["frames_ok"] = ok_count;
    agg["frames_failed"] = static_cast<int>(results.size()) - ok_count;

    // Session geometry: time span, RA/DEC, estimated field rotation
    {
        json geom = json::object();
        // Collect DATE-OBS timestamps
        std::string first_date, last_date;
        for (const auto& hi : header_infos) {
            if (!hi.date_obs.empty()) {
                if (first_date.empty() || hi.date_obs < first_date) first_date = hi.date_obs;
                if (last_date.empty() || hi.date_obs > last_date) last_date = hi.date_obs;
            }
        }
        if (!first_date.empty()) geom["first_date_obs"] = first_date;
        if (!last_date.empty()) geom["last_date_obs"] = last_date;

        // Median RA/DEC (target coordinates)
        std::vector<double> ra_vals, dec_vals;
        for (const auto& hi : header_infos) {
            if (hi.ra_deg > -900 && hi.dec_deg > -900) {
                ra_vals.push_back(hi.ra_deg);
                dec_vals.push_back(hi.dec_deg);
            }
        }
        if (!ra_vals.empty()) {
            std::sort(ra_vals.begin(), ra_vals.end());
            std::sort(dec_vals.begin(), dec_vals.end());
            double med_ra = ra_vals[ra_vals.size() / 2];
            double med_dec = dec_vals[dec_vals.size() / 2];
            geom["target_ra_deg"] = med_ra;
            geom["target_dec_deg"] = med_dec;
            geom["ra_dec_available"] = true;

            // Estimate field rotation if we have time span
            // Field rotation rate for Alt/Az mount: dθ/dt = |cos(lat) * cos(declination) * cos(H) / sin(altitude)|
            // where H is hour angle. For a rough estimate we use the parallactic angle formula.
            // Without latitude we assume mid-latitude (45°N) as default.
            // The total rotation over the session is approximately:
            //   Δθ ≈ 15°/hr * cos(declination) * session_duration_hours * sin(parallactic_angle_factor)
            // For a conservative upper bound we use: Δθ ≈ 15°/hr * cos(dec) * duration_hours
            if (!first_date.empty() && !last_date.empty()) {
                // Parse ISO 8601 timestamps and compute duration
                auto parse_iso = [](const std::string& s) -> double {
                    // Expected format: YYYY-MM-DDThh:mm:ss[.sss]Z or with timezone
                    struct tm tm_val{};
                    double frac = 0.0;
                    char zone[16] = {0};
                    int matched = sscanf(s.c_str(), "%d-%d-%dT%d:%d:%lf%15s",
                                         &tm_val.tm_year, &tm_val.tm_mon, &tm_val.tm_mday,
                                         &tm_val.tm_hour, &tm_val.tm_min, &frac, zone);
                    if (matched < 6) return -1.0;
                    tm_val.tm_year -= 1900;
                    tm_val.tm_mon -= 1;
#ifdef _WIN32
                    time_t t = _mkgmtime(&tm_val);
#else
                    time_t t = timegm(&tm_val);
#endif
                    return static_cast<double>(t) + frac;
                };
                double t0 = parse_iso(first_date);
                double t1 = parse_iso(last_date);
                if (t0 >= 0 && t1 > t0) {
                    double duration_hours = (t1 - t0) / 3600.0;
                    geom["session_duration_hours"] = duration_hours;

                    // Earth's rotation rate: 15°/hr
                    // For Alt/Az mount, field rotation rate depends on parallactic angle.
                    // Maximum rotation rate (worst case): 15°/hr * cos(dec)
                    // Typical rate at 45°N latitude for dec=0: ~10.6°/hr
                    // We provide both the theoretical max and a mid-latitude estimate.
                    double dec_rad = med_dec * M_PI / 180.0;
                    double max_rotation_rate = 15.0 * std::cos(std::abs(dec_rad));  // °/hr
                    double max_rotation = max_rotation_rate * duration_hours;
                    geom["estimated_max_field_rotation_deg"] = max_rotation;
                    geom["estimated_field_rotation_note"] =
                        "Maximum theoretical field rotation for Alt/Az mount. "
                        "Actual rotation depends on latitude and parallactic angle. "
                        "For equatorial mounts, field rotation is negligible.";
                }
            }
        } else {
            geom["ra_dec_available"] = false;
        }
        result["session_geometry"] = geom;
    }

    result["aggregate"] = agg;
    result["ok"] = (ok_count > 0);

    print_json(result);
    return 0;
}

// ============================================================================
// list-runs <runs_dir>
// ============================================================================
/// @brief Handles CLI command list runs.
/// @details Part of the GUI/CLI adapter that exposes configuration, FITS inspection, run listing, and artifact commands; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
int cmd_list_runs(const std::string& runs_dir) {
    fs::path p(runs_dir);
    
    json result;
    result["runs_dir"] = runs_dir;
    result["runs"] = json::array();
    
    if (!fs::exists(p) || !fs::is_directory(p)) {
        print_json(result);
        return 0;
    }
    
    for (const auto& entry : fs::directory_iterator(p)) {
        if (entry.is_directory()) {
            std::string name = entry.path().filename().string();
            // Check if it looks like a run directory (has run.log or events.jsonl)
            if (fs::exists(entry.path() / "run.log") || fs::exists(entry.path() / "events.jsonl")) {
                json run;
                run["name"] = name;
                run["path"] = entry.path().string();
                
                // Try to get modification time
                auto ftime = fs::last_write_time(entry.path());
                auto sctp = std::chrono::time_point_cast<std::chrono::system_clock::duration>(
                    ftime - fs::file_time_type::clock::now() + std::chrono::system_clock::now());
                auto time_t_val = std::chrono::system_clock::to_time_t(sctp);
                std::ostringstream oss;
                oss << std::put_time(std::localtime(&time_t_val), "%Y-%m-%d %H:%M:%S");
                run["modified"] = oss.str();
                
                result["runs"].push_back(run);
            }
        }
    }
    
    print_json(result);
    return 0;
}

// ============================================================================
// get-run-status <run_dir>
// ============================================================================
/// @brief Handles CLI command get run status.
/// @details Part of the GUI/CLI adapter that exposes configuration, FITS inspection, run listing, and artifact commands; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
int cmd_get_run_status(const std::string& run_dir) {
    fs::path p(run_dir);
    
    json result;
    result["run_dir"] = run_dir;
    result["exists"] = fs::exists(p);
    result["status"] = "unknown";
    result["current_phase"] = nullptr;
    result["progress"] = 0;
    result["events"] = json::array();
    
    if (!fs::exists(p)) {
        print_json(result);
        return 0;
    }
    
    // Read events.jsonl if exists
    fs::path events_file = p / "events.jsonl";
    if (fs::exists(events_file)) {
        std::ifstream ifs(events_file);
        std::string line;
        std::string last_phase;
        std::string last_status;
        
        while (std::getline(ifs, line)) {
            if (line.empty()) continue;
            try {
                json ev = json::parse(line);
                result["events"].push_back(ev);
                
                if (ev.contains("type")) {
                    std::string type = ev["type"].get<std::string>();
                    if (type == "phase_start" && ev.contains("phase")) {
                        last_phase = ev["phase"].get<std::string>();
                        last_status = "running";
                    } else if (type == "phase_end" && ev.contains("status")) {
                        last_status = ev["status"].get<std::string>();
                    } else if (type == "run_end") {
                        if (ev.contains("success") && ev["success"].get<bool>()) {
                            last_status = "completed";
                        } else {
                            last_status = "failed";
                        }
                    }
                }
            } catch (...) {
                // Skip malformed lines
            }
        }
        
        if (last_phase.empty()) {
            result["current_phase"] = nullptr;
        } else {
            result["current_phase"] = last_phase;
        }
        result["status"] = last_status.empty() ? "unknown" : last_status;
    }
    
    print_json(result);
    return 0;
}

// ============================================================================
// get-run-logs <run_dir> [--tail N]
// ============================================================================
/// @brief Handles CLI command get run logs.
/// @details Part of the GUI/CLI adapter that exposes configuration, FITS inspection, run listing, and artifact commands; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
int cmd_get_run_logs(const std::string& run_dir, int tail) {
    fs::path p(run_dir);
    
    json result;
    result["run_dir"] = run_dir;
    result["log_lines"] = json::array();
    
    fs::path log_file = p / "run.log";
    if (!fs::exists(log_file)) {
        print_json(result);
        return 0;
    }
    
    std::vector<std::string> lines;
    std::ifstream ifs(log_file);
    std::string line;
    while (std::getline(ifs, line)) {
        lines.push_back(line);
    }
    
    if (tail > 0 && static_cast<int>(lines.size()) > tail) {
        lines.erase(lines.begin(), lines.end() - tail);
    }
    
    for (const auto& l : lines) {
        result["log_lines"].push_back(l);
    }
    
    print_json(result);
    return 0;
}

// ============================================================================
// list-artifacts <run_dir>
// ============================================================================
/// @brief Handles CLI command list artifacts.
/// @details Part of the GUI/CLI adapter that exposes configuration, FITS inspection, run listing, and artifact commands; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
int cmd_list_artifacts(const std::string& run_dir) {
    fs::path p(run_dir);
    
    json result;
    result["run_dir"] = run_dir;
    result["artifacts"] = json::array();
    
    if (!fs::exists(p) || !fs::is_directory(p)) {
        print_json(result);
        return 0;
    }
    
    std::regex artifact_regex(R"(.*\.(fit|fits|fts|png|jpg|jpeg|tif|tiff|json|yaml|yml)$)", std::regex::icase);
    
    std::function<void(const fs::path&)> scan_dir = [&](const fs::path& dir) {
        for (const auto& entry : fs::directory_iterator(dir)) {
            if (entry.is_regular_file()) {
                std::string name = entry.path().filename().string();
                if (std::regex_match(name, artifact_regex)) {
                    json art;
                    art["path"] = entry.path().string();
                    art["filename"] = name;
                    art["size_bytes"] = static_cast<int64_t>(fs::file_size(entry.path()));
                    art["relative_path"] = fs::relative(entry.path(), p).string();
                    result["artifacts"].push_back(art);
                }
            } else if (entry.is_directory()) {
                scan_dir(entry.path());
            }
        }
    };
    
    scan_dir(p);
    print_json(result);
    return 0;
}

/// @brief Handles CLI command pcc run.
/// @details Part of the GUI/CLI adapter that exposes configuration, FITS inspection, run listing, and artifact commands; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
int cmd_pcc_run(const std::string& input_path,
                const std::string& output_path,
                const std::string& wcs_path,
                const std::string& source,
                const std::string& siril_catalog_dir,
                const tile_compile::astrometry::PCCConfig& config) {
    namespace io = tile_compile::io;
    namespace astro = tile_compile::astrometry;

    json result;
    result["input_rgb"] = input_path;
    result["output_rgb"] = output_path;
    result["wcs_file"] = wcs_path;

    try {
        const std::string source_requested = source.empty() ? "auto" : to_lower_copy(source);
        result["source_requested"] = source_requested;
        result["catalog_dir"] = siril_catalog_dir.empty() ? astro::default_siril_gaia_catalog_dir() : siril_catalog_dir;

        astro::WCS wcs = astro::parse_wcs_file(wcs_path);
        if (!wcs.valid()) {
            result["ok"] = false;
            result["error"] = "Invalid WCS solution";
            print_json(result);
            return 1;
        }

        result["wcs"] = {
            {"ra_deg", wcs.crval1},
            {"dec_deg", wcs.crval2},
            {"pixel_scale_arcsec", wcs.pixel_scale_arcsec()},
            {"image_width", wcs.naxis1},
            {"image_height", wcs.naxis2},
        };

        const double search_r = wcs.search_radius_deg();
        std::vector<astro::GaiaStar> stars;
        std::string used_source;

        auto try_siril = [&]() -> bool {
            const std::string cat_dir = siril_catalog_dir.empty() ? astro::default_siril_gaia_catalog_dir() : siril_catalog_dir;
            result["catalog_dir"] = cat_dir;
            if (!astro::is_siril_gaia_catalog_available(cat_dir)) {
                return false;
            }
            stars = astro::siril_gaia_cone_search(cat_dir, wcs.crval1, wcs.crval2, search_r, config.mag_limit);
            if (!stars.empty()) {
                used_source = "siril";
                return true;
            }
            return false;
        };

        auto try_vizier_gaia = [&]() -> bool {
            stars = astro::vizier_gaia_cone_search(wcs.crval1, wcs.crval2, search_r, config.mag_limit);
            if (!stars.empty()) {
                used_source = "vizier_gaia";
                return true;
            }
            return false;
        };

        auto try_vizier_apass = [&]() -> bool {
            stars = astro::vizier_apass_cone_search(wcs.crval1, wcs.crval2, search_r, config.mag_limit);
            if (!stars.empty()) {
                used_source = "vizier_apass";
                return true;
            }
            return false;
        };

        bool query_ok = false;
        if (source_requested == "siril") {
            query_ok = try_siril();
        } else if (source_requested == "vizier_gaia") {
            query_ok = try_vizier_gaia();
        } else if (source_requested == "vizier_apass") {
            query_ok = try_vizier_apass();
        } else {
            query_ok = try_siril() || try_vizier_gaia() || try_vizier_apass();
        }

        result["search_radius_deg"] = search_r;
        result["stars_catalog"] = static_cast<int>(stars.size());
        result["source"] = used_source;

        if (!query_ok) {
            result["ok"] = false;
            result["error"] = "No catalog stars found for requested PCC source";
            print_json(result);
            return 1;
        }

        io::RGBImage rgb = io::read_fits_rgb(fs::path(input_path));
        astro::PCCConfig resolved_config = config;
        double estimated_fwhm_px = 0.0;
        if (resolved_config.radii_mode == "auto_fwhm") {
            estimated_fwhm_px = static_cast<double>(tile_compile::metrics::measure_fwhm_from_image(rgb.G));
            if (!(estimated_fwhm_px > 0.0) || !std::isfinite(estimated_fwhm_px)) {
                estimated_fwhm_px = static_cast<double>(tile_compile::metrics::measure_fwhm_from_image(rgb.R));
            }
            if (!(estimated_fwhm_px > 0.0) || !std::isfinite(estimated_fwhm_px)) {
                estimated_fwhm_px = static_cast<double>(tile_compile::metrics::measure_fwhm_from_image(rgb.B));
            }
            if (estimated_fwhm_px > 0.0 && std::isfinite(estimated_fwhm_px)) {
                const double r_ap = std::max(static_cast<double>(resolved_config.min_aperture_px),
                                             resolved_config.aperture_fwhm_mult * estimated_fwhm_px);
                const double r_in = std::max(r_ap + 1.0,
                                             resolved_config.annulus_inner_fwhm_mult * estimated_fwhm_px);
                const double r_out = std::max(r_in + 2.0,
                                              resolved_config.annulus_outer_fwhm_mult * estimated_fwhm_px);
                resolved_config.aperture_radius_px = r_ap;
                resolved_config.annulus_inner_px = r_in;
                resolved_config.annulus_outer_px = r_out;
            }
        }
        result["radii_mode"] = resolved_config.radii_mode;
        result["aperture_radius_px"] = resolved_config.aperture_radius_px;
        result["annulus_inner_px"] = resolved_config.annulus_inner_px;
        result["annulus_outer_px"] = resolved_config.annulus_outer_px;
        if (estimated_fwhm_px > 0.0 && std::isfinite(estimated_fwhm_px)) {
            result["estimated_fwhm_px"] = estimated_fwhm_px;
        }
        auto pcc_result = astro::run_pcc(rgb.R, rgb.G, rgb.B, wcs, stars, resolved_config);

        result["stars_matched"] = pcc_result.n_stars_matched;
        result["stars_used"] = pcc_result.n_stars_used;
        result["residual_rms"] = pcc_result.residual_rms;
        result["determinant"] = pcc_result.determinant;
        result["condition_number"] = pcc_result.condition_number;
        result["apply_mode"] = pcc_result.apply_mode;
        result["apply_attenuation"] = resolved_config.apply_attenuation;
        result["chroma_strength"] = resolved_config.chroma_strength;
        result["k_max"] = resolved_config.k_max;
        result["background_model"] = resolved_config.background_model;
        result["max_condition_number"] = resolved_config.max_condition_number;
        result["max_residual_rms"] = resolved_config.max_residual_rms;

        json matrix_json = json::array();
        for (int r = 0; r < 3; ++r) {
            matrix_json.push_back(json::array({
                pcc_result.matrix[r][0],
                pcc_result.matrix[r][1],
                pcc_result.matrix[r][2],
            }));
        }
        result["matrix"] = matrix_json;

        if (!pcc_result.success) {
            result["ok"] = false;
            result["error"] = pcc_result.error_message;
            print_json(result);
            return 1;
        }

        const fs::path out_path(output_path);
        if (out_path.has_parent_path()) {
            fs::create_directories(out_path.parent_path());
        }
        io::write_fits_rgb(out_path, rgb.R, rgb.G, rgb.B, rgb.header);

        const fs::path out_dir = out_path.has_parent_path() ? out_path.parent_path() : fs::current_path();
        const std::string stem = out_path.stem().string().empty() ? "pcc" : out_path.stem().string();
        const fs::path out_r = out_dir / (stem + "_R.fit");
        const fs::path out_g = out_dir / (stem + "_G.fit");
        const fs::path out_b = out_dir / (stem + "_B.fit");
        io::write_fits_float(out_r, rgb.R, rgb.header);
        io::write_fits_float(out_g, rgb.G, rgb.header);
        io::write_fits_float(out_b, rgb.B, rgb.header);

        result["ok"] = true;
        result["output_rgb"] = out_path.string();
        result["output_channels"] = json::array({out_r.string(), out_g.string(), out_b.string()});
        print_json(result);
        return 0;
    } catch (const std::exception& e) {
        result["ok"] = false;
        result["error"] = e.what();
        print_json(result);
        return 1;
    }
}

// ============================================================================
// Main
// ============================================================================
/// @brief Converts lower copy.
/// @details Part of the GUI/CLI adapter that exposes configuration, FITS inspection, run listing, and artifact commands; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
static std::string to_lower_copy(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

/// @brief Parses boolish.
/// @details Part of the GUI/CLI adapter that exposes configuration, FITS inspection, run listing, and artifact commands; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
static bool parse_boolish(const std::string& value) {
    const std::string lower = to_lower_copy(value);
    return lower == "1" || lower == "true" || lower == "yes" || lower == "on";
}

/// @brief Implements print usage.
/// @details Part of the GUI/CLI adapter that exposes configuration, FITS inspection, run listing, and artifact commands; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
void print_usage() {
    std::cout << "Usage: tile_compile_cli <command> [options]\n"
              << "\nCommands:\n"
              << "  get-schema                      Print JSON schema for config\n"
              << "  dump-default-config             Print default config as JSON/YAML\n"
              << "  load-gui-state [--path P]       Load GUI state from file\n"
              << "  save-gui-state [--path P] [--stdin | JSON]  Save GUI state\n"
              << "  load-config <path>              Load config YAML file\n"
              << "  save-config <path> [--stdin | YAML]  Save config YAML file\n"
              << "  validate-config (--path P | --yaml Y | --stdin)  Validate config\n"
              << "  scan <input_path> [--frames-min N]  Scan input directory for frames\n"
              << "  list-runs <runs_dir>            List pipeline runs\n"
              << "  get-run-status <run_dir>        Get status of a run\n"
              << "  get-run-logs <run_dir> [--tail N]  Get run logs\n"
              << "  list-artifacts <run_dir>        List artifacts in run directory\n"
              << "  fits-stats <path>               Print basic statistics for a FITS image\n"
              << "  pcc-run <in> <out> --wcs <wcs> [--source S]  Run full PCC and write corrected RGB FITS\n"
              << "  pcc-apply <in> <out> [--r X] [--g Y] [--b Z]  Apply diagonal PCC matrix to RGB FITS cube\n";
}

/// @brief Implements main.
/// @details Part of the GUI/CLI adapter that exposes configuration, FITS inspection, run listing, and artifact commands; this helper keeps the implementation
/// localized in this translation unit and preserves the surrounding phase,
/// artifact, and error-handling semantics expected by callers.
int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_usage();
        return 1;
    }
    
    std::string command = argv[1];
    
    // Helper to find argument value
    auto get_arg = [&](const char* name, const char* short_name = nullptr) -> std::string {
        for (int i = 2; i < argc - 1; ++i) {
            if (std::strcmp(argv[i], name) == 0 || (short_name && std::strcmp(argv[i], short_name) == 0)) {
                return argv[i + 1];
            }
        }
        return "";
    };
    
    auto has_flag = [&](const char* name) -> bool {
        for (int i = 2; i < argc; ++i) {
            if (std::strcmp(argv[i], name) == 0) return true;
        }
        return false;
    };
    
    auto get_positional = [&](int pos) -> std::string {
        int count = 0;
        for (int i = 2; i < argc; ++i) {
            if (argv[i][0] != '-') {
                if (count == pos) return argv[i];
                ++count;
            } else if (i + 1 < argc && argv[i + 1][0] != '-') {
                ++i; // Skip argument value
            }
        }
        return "";
    };
    
    if (command == "get-schema") {
        return cmd_get_schema();
    }

    if (command == "dump-default-config") {
        return cmd_dump_default_config();
    }
    
    if (command == "load-gui-state") {
        return cmd_load_gui_state(get_arg("--path"));
    }
    
    if (command == "save-gui-state") {
        return cmd_save_gui_state(get_arg("--path"), get_positional(0), has_flag("--stdin"));
    }
    
    if (command == "load-config") {
        std::string path = get_positional(0);
        if (path.empty()) {
            std::cerr << "load-config requires a path argument\n";
            return 1;
        }
        return cmd_load_config(path);
    }
    
    if (command == "save-config") {
        std::string path = get_positional(0);
        if (path.empty()) {
            std::cerr << "save-config requires a path argument\n";
            return 1;
        }
        return cmd_save_config(path, get_positional(1), has_flag("--stdin"));
    }
    
    if (command == "validate-config") {
        std::string path = get_arg("--path");
        std::string yaml = get_arg("--yaml");
        bool use_stdin = has_flag("--stdin");
        bool strict = has_flag("--strict-exit-codes");
        
        if (path.empty() && yaml.empty() && !use_stdin) {
            std::cerr << "validate-config requires --path, --yaml, or --stdin\n";
            return 1;
        }
        return cmd_validate_config(path, yaml, use_stdin, strict);
    }
    
    if (command == "scan") {
        std::string input_path = get_positional(0);
        if (input_path.empty()) {
            std::cerr << "scan requires an input_path argument\n";
            return 1;
        }
        std::string frames_min_str = get_arg("--frames-min");
        int frames_min = frames_min_str.empty() ? 1 : std::stoi(frames_min_str);
        bool with_checksums = has_flag("--with-checksums");
        return cmd_scan(input_path, frames_min, with_checksums);
    }

    if (command == "scan-metrics") {
        std::string input_path = get_positional(0);
        if (input_path.empty()) {
            std::cerr << "scan-metrics requires an input_path argument\n";
            return 1;
        }
        return cmd_scan_metrics(input_path);
    }
    
    if (command == "list-runs") {
        std::string runs_dir = get_positional(0);
        if (runs_dir.empty()) {
            std::cerr << "list-runs requires a runs_dir argument\n";
            return 1;
        }
        return cmd_list_runs(runs_dir);
    }
    
    if (command == "get-run-status") {
        std::string run_dir = get_positional(0);
        if (run_dir.empty()) {
            std::cerr << "get-run-status requires a run_dir argument\n";
            return 1;
        }
        return cmd_get_run_status(run_dir);
    }
    
    if (command == "get-run-logs") {
        std::string run_dir = get_positional(0);
        if (run_dir.empty()) {
            std::cerr << "get-run-logs requires a run_dir argument\n";
            return 1;
        }
        std::string tail_str = get_arg("--tail");
        int tail = tail_str.empty() ? 0 : std::stoi(tail_str);
        return cmd_get_run_logs(run_dir, tail);
    }
    
    if (command == "list-artifacts") {
        std::string run_dir = get_positional(0);
        if (run_dir.empty()) {
            std::cerr << "list-artifacts requires a run_dir argument\n";
            return 1;
        }
        return cmd_list_artifacts(run_dir);
    }

    if (command == "fits-stats") {
        std::string path = get_positional(0);
        if (path.empty()) {
            std::cerr << "fits-stats requires a path argument\n";
            return 1;
        }
        return cmd_fits_stats(path);
    }

    if (command == "pcc-run") {
        std::string in_path = get_positional(0);
        std::string out_path = get_positional(1);
        std::string wcs_path = get_arg("--wcs");
        if (in_path.empty() || out_path.empty() || wcs_path.empty()) {
            std::cerr << "pcc-run requires <input_rgb_fits> <output_rgb_fits> and --wcs <wcs_file>\n";
            return 1;
        }
        tile_compile::astrometry::PCCConfig cfg;
        std::string source = get_arg("--source");
        std::string siril_catalog_dir = get_arg("--siril-catalog-dir");
        std::string mag_limit_str = get_arg("--mag-limit");
        std::string mag_bright_limit_str = get_arg("--mag-bright-limit");
        std::string min_stars_str = get_arg("--min-stars");
        std::string sigma_clip_str = get_arg("--sigma-clip");
        std::string aperture_radius_str = get_arg("--aperture-radius-px");
        std::string annulus_inner_str = get_arg("--annulus-inner-px");
        std::string annulus_outer_str = get_arg("--annulus-outer-px");
        std::string apply_attenuation_str = get_arg("--apply-attenuation");
        std::string chroma_strength_str = get_arg("--chroma-strength");
        std::string k_max_str = get_arg("--k-max");
        std::string radii_mode_str = get_arg("--radii-mode");
        std::string aperture_fwhm_mult_str = get_arg("--aperture-fwhm-mult");
        std::string annulus_inner_fwhm_mult_str = get_arg("--annulus-inner-fwhm-mult");
        std::string annulus_outer_fwhm_mult_str = get_arg("--annulus-outer-fwhm-mult");
        std::string min_aperture_px_str = get_arg("--min-aperture-px");
        std::string background_model_str = get_arg("--background-model");
        std::string max_condition_number_str = get_arg("--max-condition-number");
        std::string max_residual_rms_str = get_arg("--max-residual-rms");
        if (!mag_limit_str.empty()) cfg.mag_limit = std::stod(mag_limit_str);
        if (!mag_bright_limit_str.empty()) cfg.mag_bright_limit = std::stod(mag_bright_limit_str);
        if (!min_stars_str.empty()) cfg.min_stars = std::stoi(min_stars_str);
        if (!sigma_clip_str.empty()) cfg.sigma_clip = std::stod(sigma_clip_str);
        if (!aperture_radius_str.empty()) cfg.aperture_radius_px = std::stod(aperture_radius_str);
        if (!annulus_inner_str.empty()) cfg.annulus_inner_px = std::stod(annulus_inner_str);
        if (!annulus_outer_str.empty()) cfg.annulus_outer_px = std::stod(annulus_outer_str);
        if (!apply_attenuation_str.empty()) cfg.apply_attenuation = parse_boolish(apply_attenuation_str);
        if (!chroma_strength_str.empty()) cfg.chroma_strength = std::stod(chroma_strength_str);
        if (!k_max_str.empty()) cfg.k_max = std::stod(k_max_str);
        if (!radii_mode_str.empty()) cfg.radii_mode = radii_mode_str;
        if (!aperture_fwhm_mult_str.empty()) cfg.aperture_fwhm_mult = std::stod(aperture_fwhm_mult_str);
        if (!annulus_inner_fwhm_mult_str.empty()) cfg.annulus_inner_fwhm_mult = std::stod(annulus_inner_fwhm_mult_str);
        if (!annulus_outer_fwhm_mult_str.empty()) cfg.annulus_outer_fwhm_mult = std::stod(annulus_outer_fwhm_mult_str);
        if (!min_aperture_px_str.empty()) cfg.min_aperture_px = std::stod(min_aperture_px_str);
        if (!background_model_str.empty()) cfg.background_model = background_model_str;
        if (!max_condition_number_str.empty()) cfg.max_condition_number = std::stod(max_condition_number_str);
        if (!max_residual_rms_str.empty()) cfg.max_residual_rms = std::stod(max_residual_rms_str);
        return cmd_pcc_run(in_path, out_path, wcs_path, source, siril_catalog_dir, cfg);
    }

    if (command == "pcc-apply") {
        std::string in_path = get_positional(0);
        std::string out_path = get_positional(1);
        if (in_path.empty() || out_path.empty()) {
            std::cerr << "pcc-apply requires <input_rgb_fits> and <output_rgb_fits> arguments\n";
            return 1;
        }
        double r = 1.0;
        double g = 1.0;
        double b = 1.0;
        std::string r_str = get_arg("--r");
        std::string g_str = get_arg("--g");
        std::string b_str = get_arg("--b");
        if (!r_str.empty()) r = std::stod(r_str);
        if (!g_str.empty()) g = std::stod(g_str);
        if (!b_str.empty()) b = std::stod(b_str);
        return cmd_pcc_apply(in_path, out_path, r, g, b);
    }
    
    std::cerr << "Unknown command: " << command << std::endl;
    print_usage();
    return 1;
}
