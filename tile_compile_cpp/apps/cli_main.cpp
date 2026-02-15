#include "tile_compile/core/types.hpp"
#include "tile_compile/config/configuration.hpp"
#include "tile_compile/io/fits_io.hpp"
#include "tile_compile/astrometry/photometric_color_cal.hpp"

#include <nlohmann/json.hpp>
#include <yaml-cpp/yaml.h>
#include <fitsio.h>

#include <algorithm>
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

namespace fs = std::filesystem;
using json = nlohmann::json;

static std::string get_executable_dir() {
    return fs::current_path().string();
}

static std::string default_gui_state_path() {
    return (fs::path(get_executable_dir()) / "tile_compile_gui_state.json").string();
}

static void print_json(const json& j) {
    std::cout << j.dump(2) << std::endl;
}

static std::string read_file_text(const fs::path& p) {
    std::ifstream ifs(p);
    if (!ifs) return "";
    std::ostringstream ss;
    ss << ifs.rdbuf();
    return ss.str();
}

static bool write_file_text(const fs::path& p, const std::string& content) {
    std::ofstream ofs(p);
    if (!ofs) return false;
    ofs << content;
    return true;
}

static std::string read_stdin() {
    std::ostringstream ss;
    ss << std::cin.rdbuf();
    return ss.str();
}

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
    bool read_error = false;
    std::string error_msg;
};

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
    
    fits_close_file(fptr, &status);
    return info;
}

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
int cmd_get_schema() {
    std::cout << tile_compile::config::get_schema_json() << std::endl;
    return 0;
}

// ============================================================================
// load-gui-state [--path <path>]
// ============================================================================
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
        YAML::Node node = YAML::Load(yaml_text);
        tile_compile::config::Config cfg = tile_compile::config::Config::from_yaml(node);
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
    result["bayer_pattern"] = bayer_pattern.empty() ? nullptr : json(bayer_pattern);
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
// list-runs <runs_dir>
// ============================================================================
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
        
        result["current_phase"] = last_phase.empty() ? nullptr : json(last_phase);
        result["status"] = last_status.empty() ? "unknown" : last_status;
    }
    
    print_json(result);
    return 0;
}

// ============================================================================
// get-run-logs <run_dir> [--tail N]
// ============================================================================
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

// ============================================================================
// Main
// ============================================================================
void print_usage() {
    std::cout << "Usage: tile_compile_cli <command> [options]\n"
              << "\nCommands:\n"
              << "  get-schema                      Print JSON schema for config\n"
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
              << "  pcc-apply <in> <out> [--r X] [--g Y] [--b Z]  Apply diagonal PCC matrix to RGB FITS cube\n";
}

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
