#include "services/pi/pi_decision_state.hpp"

#include <openssl/sha.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <map>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>

namespace tile_compile::pi {

using nlohmann::json;

namespace {

constexpr const char* kMetricNames[] = {"background", "noise", "gradient_energy", "sky_gradient",
                                        "fwhm", "roundness", "star_count", "axis_asymmetry"};

// Config paths whose CURRENT value (and lock state) the external provider may see. M2 replaces
// this constant with the candidate catalog's paths; M1 only needs the first candidate's.
constexpr const char* kProjectedConfigPaths[] = {"global_metrics.adaptive_weights", "reconstruction.drizzle.pixfrac",
                                                 "reconstruction.clipping.clip_sigma_low", "reconstruction.clipping.clip_sigma_high"};

const char* unit_for(const std::string& metric) {
    if (metric == "background" || metric == "noise") return "adu";
    if (metric == "gradient_energy") return "dimensionless";
    if (metric == "sky_gradient") return "ratio";
    if (metric == "fwhm") return "px";
    if (metric == "roundness") return "ratio";
    if (metric == "star_count") return "count";
    if (metric == "axis_asymmetry") return "log_ratio";
    return "unknown";
}

bool as_finite(const json& v, double& out) {
    if (!v.is_number()) return false;
    const double d = v.get<double>();
    if (!std::isfinite(d)) return false;
    out = d;
    return true;
}

// Returns a copy with every non-finite number replaced by null; counts replacements.
json sanitize_non_finite(const json& v, int& replaced) {
    if (v.is_number_float() && !std::isfinite(v.get<double>())) {
        ++replaced;
        return nullptr;
    }
    if (v.is_object()) {
        json out = json::object();
        for (auto it = v.begin(); it != v.end(); ++it) out[it.key()] = sanitize_non_finite(it.value(), replaced);
        return out;
    }
    if (v.is_array()) {
        json out = json::array();
        for (const auto& item : v) out.push_back(sanitize_non_finite(item, replaced));
        return out;
    }
    return v;
}

// Integral floats become integers so 3 and 3.0 hash identically; NaN/Inf throw.
json normalize_for_hash(const json& v) {
    if (v.is_number_float()) {
        const double d = v.get<double>();
        if (!std::isfinite(d)) throw std::invalid_argument("canonical_json_dump: non-finite number");
        if (d == std::floor(d) && std::fabs(d) < 9007199254740992.0) return static_cast<long long>(d);
        return v;
    }
    if (v.is_object()) {
        json out = json::object();
        for (auto it = v.begin(); it != v.end(); ++it) out[it.key()] = normalize_for_hash(it.value());
        return out;
    }
    if (v.is_array()) {
        json out = json::array();
        for (const auto& item : v) out.push_back(normalize_for_hash(item));
        return out;
    }
    return v;
}

json measurement(const json& value, const char* status, const std::string& unit, const std::string& source_ref,
                 int valid_count, int total_count, const char* reason = nullptr) {
    json m = {{"value", std::string(status) == "valid" ? value : json(nullptr)},
              {"status", status},
              {"unit", unit},
              {"source_ref", source_ref},
              {"method_version", kDecisionStateMethodVersion},
              {"valid_count", valid_count},
              {"total_count", total_count}};
    if (reason) m["reason"] = reason;
    return m;
}

double median_of(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

double percentile_of(const std::vector<double>& sorted, double p) {
    const size_t idx = std::min(sorted.size() - 1, static_cast<size_t>(static_cast<double>(sorted.size()) * p));
    return sorted[idx];
}

struct MetricAcc {
    std::vector<double> values;
    int missing = 0;
    int invalid = 0;
};

struct GroupAcc {
    json camera = nullptr, filter = nullptr, exposure = nullptr, gain = nullptr;
    int frames = 0;
    int read_failed = 0;
    std::map<std::string, MetricAcc> metrics;
    std::vector<std::array<double, 3>> agreement_rows;  // background, noise, fwhm of frames valid in all three
};

json nonempty_string_or_null(const json& v) {
    if (v.is_string() && !v.get<std::string>().empty()) return v;
    return nullptr;
}

json rounded_number_or_null(const json& v) {
    double d;
    if (!as_finite(v, d)) return nullptr;
    return std::round(d * 1e6) / 1e6;
}

// Feeds one metric of one frame into the accumulator, keeping the CAUSE of every exclusion.
void feed_metric(MetricAcc& acc, const json& frame, const std::string& name) {
    if (name == "axis_asymmetry") {
        if (!frame.contains("roundness")) { ++acc.missing; return; }
        double r, a;
        if (!as_finite(frame["roundness"], r) || !axis_asymmetry(r, a)) { ++acc.invalid; return; }
        acc.values.push_back(a);
        return;
    }
    if (!frame.contains(name)) { ++acc.missing; return; }
    double d;
    if (!as_finite(frame[name], d)) { ++acc.invalid; return; }
    // The scan writes -1 (and cli_main's aggregate drops) when no reliable star fit exists.
    if ((name == "fwhm" || name == "roundness") && d <= 0.0) { ++acc.invalid; return; }
    if (name == "star_count" && d < 0.0) { ++acc.invalid; return; }
    acc.values.push_back(d);
}

json summarize_metric(const std::string& name, const MetricAcc& acc, int total, int min_valid_for_spread) {
    const std::string unit = unit_for(name);
    const std::string ref = name == "axis_asymmetry"
                                ? "derived:abs(ln(scan-metrics.frames[].roundness))"
                                : "scan-metrics.frames[]." + name;
    const int valid = static_cast<int>(acc.values.size());
    json s = json::object();
    if (valid == 0) {
        for (const char* k : {"median", "p10", "p90", "mad", "relative_spread"})
            s[k] = measurement(nullptr, "missing", unit, ref, 0, total);
        return s;
    }
    std::vector<double> sorted = acc.values;
    std::sort(sorted.begin(), sorted.end());
    const double med = median_of(sorted);
    s["median"] = measurement(med, "valid", unit, ref, valid, total);
    s["p10"] = measurement(percentile_of(sorted, 0.1), "valid", unit, ref, valid, total);
    s["p90"] = measurement(percentile_of(sorted, 0.9), "valid", unit, ref, valid, total);
    if (valid < min_valid_for_spread) {
        s["mad"] = measurement(nullptr, "not_applicable", unit, ref, valid, total, "insufficient_sample");
        s["relative_spread"] = measurement(nullptr, "not_applicable", "ratio", ref, valid, total, "insufficient_sample");
        return s;
    }
    std::vector<double> dev;
    dev.reserve(sorted.size());
    for (double x : sorted) dev.push_back(std::fabs(x - med));
    const double mad = median_of(dev);
    s["mad"] = measurement(mad, "valid", unit, ref, valid, total);
    if (std::fabs(med) > 0.0) {
        s["relative_spread"] = measurement(mad / std::fabs(med), "valid", "ratio", ref, valid, total);
    } else {
        s["relative_spread"] = measurement(nullptr, "invalid", "ratio", ref, valid, total, "zero_denominator");
    }
    return s;
}

// Average ranks (ties share the mean rank) for Spearman's rho.
std::vector<double> average_ranks(const std::vector<double>& v) {
    std::vector<size_t> idx(v.size());
    for (size_t i = 0; i < idx.size(); ++i) idx[i] = i;
    std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b) { return v[a] < v[b]; });
    std::vector<double> ranks(v.size());
    for (size_t i = 0; i < idx.size();) {
        size_t j = i;
        while (j + 1 < idx.size() && v[idx[j + 1]] == v[idx[i]]) ++j;
        const double r = 0.5 * static_cast<double>(i + j) + 1.0;
        for (size_t k = i; k <= j; ++k) ranks[idx[k]] = r;
        i = j + 1;
    }
    return ranks;
}

// Spearman rank correlation; false when either side is constant (rho undefined).
bool spearman_rho(const std::vector<double>& x, const std::vector<double>& y, double& out) {
    if (x.size() != y.size() || x.size() < 2) return false;
    const std::vector<double> rx = average_ranks(x), ry = average_ranks(y);
    const double n = static_cast<double>(x.size());
    double mx = 0.0, my = 0.0;
    for (size_t i = 0; i < rx.size(); ++i) { mx += rx[i]; my += ry[i]; }
    mx /= n; my /= n;
    double sxy = 0.0, sxx = 0.0, syy = 0.0;
    for (size_t i = 0; i < rx.size(); ++i) {
        sxy += (rx[i] - mx) * (ry[i] - my);
        sxx += (rx[i] - mx) * (rx[i] - mx);
        syy += (ry[i] - my) * (ry[i] - my);
    }
    if (sxx <= 0.0 || syy <= 0.0) return false;
    out = sxy / std::sqrt(sxx * syy);
    return true;
}

// Per-group agreement between the frame-quality metrics that drive adaptive weighting. Uses only
// frames where BOTH metrics of a pair are valid; the median needs all three pairs.
json summarize_metric_agreement(const std::vector<std::array<double, 3>>& rows, int total, int min_valid) {
    static const char* kNames[3] = {"background", "noise", "fwhm"};
    static const int kPairs[3][2] = {{0, 1}, {0, 2}, {1, 2}};
    json out = json::object();
    std::vector<double> rhos;
    for (const auto& pr : kPairs) {
        const std::string key = std::string(kNames[pr[0]]) + "~" + kNames[pr[1]];
        const std::string ref = "derived:spearman(scan-metrics.frames[]." + std::string(kNames[pr[0]]) + ",." + kNames[pr[1]] + ")";
        std::vector<double> x, y;
        for (const auto& r : rows) { x.push_back(r[pr[0]]); y.push_back(r[pr[1]]); }
        const int valid = static_cast<int>(rows.size());
        double rho = 0.0;
        if (valid < min_valid) {
            out[key] = measurement(nullptr, "not_applicable", "ratio", ref, valid, total, "insufficient_sample");
        } else if (!spearman_rho(x, y, rho)) {
            out[key] = measurement(nullptr, "invalid", "ratio", ref, valid, total, "constant_metric");
        } else {
            out[key] = measurement(rho, "valid", "ratio", ref, valid, total);
            rhos.push_back(rho);
        }
    }
    const std::string mref = "derived:median(pairwise spearman of background,noise,fwhm)";
    if (rhos.size() == 3)
        out["median"] = measurement(median_of(rhos), "valid", "ratio", mref, static_cast<int>(rows.size()), total);
    else
        out["median"] = measurement(nullptr, "not_applicable", "ratio", mref, static_cast<int>(rows.size()), total,
                                    "needs_all_three_pairs");
    return out;
}

json finding(const std::string& code, const std::string& severity, const std::string& detail) {
    return {{"code", code}, {"severity", severity}, {"detail", detail}};
}

bool looks_like_path_or_secret(const std::string& s) {
    static const std::regex kAbs(R"(^(/|[A-Za-z]:[\\/]|\\\\))");
    if (std::regex_search(s, kAbs)) return true;
    return s.find("sk-or-") != std::string::npos || s.find("://") != std::string::npos ||
           s.find("api_key") != std::string::npos;
}

void assert_projection_clean(const json& v, const std::string& where) {
    if (v.is_string() && looks_like_path_or_secret(v.get<std::string>()))
        throw std::logic_error("provider_projection_leak at " + where);
    if (v.is_object())
        for (auto it = v.begin(); it != v.end(); ++it) {
            if (looks_like_path_or_secret(it.key())) throw std::logic_error("provider_projection_leak (key) at " + where);
            assert_projection_clean(it.value(), where + "." + it.key());
        }
    if (v.is_array())
        for (size_t i = 0; i < v.size(); ++i) assert_projection_clean(v[i], where + "[" + std::to_string(i) + "]");
}

json config_value_at(const json& cfg, const std::string& dotted) {
    const json* cur = &cfg;
    std::stringstream ss(dotted);
    std::string part;
    while (std::getline(ss, part, '.')) {
        if (!cur->is_object() || !cur->contains(part)) return nullptr;
        cur = &(*cur)[part];
    }
    return *cur;
}

std::string basename_of(const std::string& p) {
    const size_t pos = p.find_last_of("/\\");
    return pos == std::string::npos ? p : p.substr(pos + 1);
}

bool is_sha256_hex(const json& v) {
    if (!v.is_string()) return false;
    const std::string s = v.get<std::string>();
    return s.size() == 64 && std::all_of(s.begin(), s.end(), [](unsigned char c) { return std::isxdigit(c); });
}

std::string short_hash(const std::string& s) { return sha256_prefixed(s).substr(7, 12); }

const std::set<std::string>& volatile_scan_keys() {
    static const std::set<std::string> k = {"generated_at", "timestamp", "created_at", "elapsed_seconds",
                                            "duration_seconds", "started_at", "finished_at"};
    return k;
}

json strip_volatile(const json& v) {
    if (v.is_object()) {
        json out = json::object();
        for (auto it = v.begin(); it != v.end(); ++it)
            if (!volatile_scan_keys().count(it.key())) out[it.key()] = strip_volatile(it.value());
        return out;
    }
    if (v.is_array()) {
        json out = json::array();
        for (const auto& item : v) out.push_back(strip_volatile(item));
        return out;
    }
    return v;
}

} // namespace

bool axis_asymmetry(double roundness, double& out) {
    if (!std::isfinite(roundness) || roundness <= 0.0) return false;
    out = std::fabs(std::log(roundness));
    return true;
}

std::string canonical_json_dump(const json& value) { return normalize_for_hash(value).dump(); }

std::string sha256_prefixed(const std::string& bytes) {
    unsigned char digest[SHA256_DIGEST_LENGTH];
    SHA256(reinterpret_cast<const unsigned char*>(bytes.data()), bytes.size(), digest);
    std::ostringstream hex;
    hex << "sha256:" << std::hex << std::setfill('0');
    for (unsigned char b : digest) hex << std::setw(2) << static_cast<int>(b);
    return hex.str();
}

PreRunDecisionResult build_pre_run_decision_state(const PreRunDecisionInputs& in) {
    PreRunDecisionResult result;
    json findings = json::array();
    std::vector<std::string> blocking;
    int non_finite_replaced = 0;

    const json scan = sanitize_non_finite(in.scan.is_object() ? in.scan : json::object(), non_finite_replaced);
    const json metrics = sanitize_non_finite(in.metrics.is_object() ? in.metrics : json::object(), non_finite_replaced);
    const json base_config = sanitize_non_finite(in.base_config, non_finite_replaced);
    if (non_finite_replaced > 0)
        findings.push_back(finding("non_finite_input", "warning",
                                   std::to_string(non_finite_replaced) + " non-finite number(s) treated as missing"));

    // ---- identity of the scan: color mode / bayer (session-level facts) ----
    std::string color_mode = scan.value("color_mode", std::string("UNKNOWN"));
    if (color_mode != "OSC" && color_mode != "MONO") color_mode = "UNKNOWN";
    json bayer = nonempty_string_or_null(scan.contains("bayer_pattern") ? scan["bayer_pattern"] : json(nullptr));
    if (color_mode == "UNKNOWN") blocking.push_back("color_mode_unresolved");
    if (color_mode == "OSC" && bayer.is_null()) blocking.push_back("bayer_pattern_missing");
    if (scan.contains("errors") && scan["errors"].is_array() && !scan["errors"].empty())
        blocking.push_back("scan_errors_present");
    if (!base_config.is_object()) blocking.push_back("base_config_invalid");
    if (scan.contains("warnings") && scan["warnings"].is_array())
        for (const auto& w : scan["warnings"])
            findings.push_back(finding(w.is_object() ? w.value("code", std::string("scan_warning")) : "scan_warning",
                                       "warning", w.is_object() ? w.value("message", std::string()) : w.dump()));

    // ---- frames -> groups ----
    const bool have_frames = metrics.contains("frames") && metrics["frames"].is_array();
    if (!have_frames) blocking.push_back("metrics_missing");
    const json frames = have_frames ? metrics["frames"] : json::array();

    std::map<std::string, GroupAcc> groups;
    int read_ok = 0;
    for (const auto& frame : frames) {
        if (!frame.is_object()) continue;
        const json hdr = frame.contains("header") && frame["header"].is_object() ? frame["header"] : json::object();
        GroupAcc probe;
        probe.camera = nonempty_string_or_null(hdr.contains("camera") ? hdr["camera"] : json(nullptr));
        probe.filter = nonempty_string_or_null(hdr.contains("filter") ? hdr["filter"] : json(nullptr));
        probe.exposure = rounded_number_or_null(hdr.contains("exposure_seconds") ? hdr["exposure_seconds"] : json(nullptr));
        probe.gain = rounded_number_or_null(hdr.contains("gain") ? hdr["gain"] : json(nullptr));
        // Unknown header values are their own explicit bucket: null never equals a known value.
        const std::string key = canonical_json_dump(json::array({probe.camera, probe.filter, probe.exposure, probe.gain}));
        GroupAcc& g = groups.try_emplace(key, probe).first->second;
        ++g.frames;
        const bool ok = frame.value("ok", false);
        if (!ok) {
            ++g.read_failed;
            continue;
        }
        ++read_ok;
        for (const char* name : kMetricNames) feed_metric(g.metrics[name], frame, name);
        double bg, nz, fw;
        if (frame.contains("background") && frame.contains("noise") && frame.contains("fwhm") &&
            as_finite(frame["background"], bg) && as_finite(frame["noise"], nz) && as_finite(frame["fwhm"], fw) && fw > 0.0)
            g.agreement_rows.push_back({bg, nz, fw});
    }
    if (read_ok == 0) blocking.push_back("no_readable_frames");

    json groups_json = json::array();
    for (const auto& [key, g] : groups) {
        const int readable = g.frames - g.read_failed;
        json metrics_json = json::object();
        json coverage_json = json::object();
        for (const char* name : kMetricNames) {
            const auto it = g.metrics.find(name);
            static const MetricAcc kEmpty;
            const MetricAcc& acc = it == g.metrics.end() ? kEmpty : it->second;
            // Unread frames contribute no value for any metric; they are counted in total_count via
            // the group's frame_count, and in frames_read_failed -- never as a measured 0.
            MetricAcc effective = acc;
            if (it == g.metrics.end()) effective.missing = readable;
            metrics_json[name] = summarize_metric(name, effective, g.frames, in.options.min_valid_for_spread);
            coverage_json[name] = {{"missing", effective.missing}, {"invalid", effective.invalid}};
        }
        groups_json.push_back({{"group_id", "g" + short_hash(key)},
                               {"frame_count", g.frames},
                               {"camera", g.camera},
                               {"color_mode", color_mode},
                               {"bayer_pattern", bayer},
                               {"filter", g.filter},
                               {"exposure_seconds", g.exposure},
                               {"gain", g.gain},
                               {"frames_read_failed", g.read_failed},
                               {"metrics", metrics_json},
                               {"metric_coverage", coverage_json},
                               {"metric_agreement", summarize_metric_agreement(g.agreement_rows, g.frames, in.options.min_valid_for_spread)}});
    }
    if (groups_json.size() > 1)
        findings.push_back(finding("mixed_acquisition_groups", "info",
                                   std::to_string(groups_json.size()) + " acquisition groups; statistics are per group, never pooled"));
    if (read_ok > 0 && read_ok < static_cast<int>(frames.size()))
        findings.push_back(finding("coverage_partial", "warning",
                                   std::to_string(frames.size() - static_cast<size_t>(read_ok)) + " frame(s) unreadable"));

    // ---- coverage / session facts ----
    int detected = static_cast<int>(frames.size());
    if (scan.contains("frames_detected") && scan["frames_detected"].is_number_integer())
        detected = scan["frames_detected"].get<int>();
    json session_facts = {{"measured", json::object()}, {"user_stated", json::object()}};
    if (metrics.contains("session_geometry") && metrics["session_geometry"].is_object()) {
        const json& sg = metrics["session_geometry"];
        double rot;
        if (sg.contains("estimated_max_field_rotation_deg") && as_finite(sg["estimated_max_field_rotation_deg"], rot))
            session_facts["measured"]["field_rotation_deg"] = {
                {"value", rot}, {"unit", "deg"}, {"kind", "estimate"},
                {"source_ref", "scan-metrics.session_geometry.estimated_max_field_rotation_deg"}};
        double dur;
        if (sg.contains("session_duration_hours") && as_finite(sg["session_duration_hours"], dur))
            session_facts["measured"]["session_duration_hours"] = {
                {"value", dur}, {"unit", "h"}, {"kind", "measured"},
                {"source_ref", "scan-metrics.session_geometry.session_duration_hours"}};
    }
    if (in.session_context.is_object())
        for (auto it = in.session_context.begin(); it != in.session_context.end(); ++it) {
            const json& e = it.value();
            if (e.is_object() && e.contains("value") && e.value("source", std::string()) == "user") {
                // The object class is a closed enum: anything else would give candidates a free-text "fact".
                if (it.key() == "object_class") {
                    const bool known = e["value"].is_string() && (e["value"] == "compact" || e["value"] == "diffuse" || e["value"] == "star_field");
                    if (!known) {
                        findings.push_back(finding("object_class_invalid", "warning", "object_class must be compact, diffuse or star_field; treated as not stated"));
                        continue;
                    }
                }
                session_facts["user_stated"][it.key()] = {{"value", e["value"]}, {"kind", "user_stated"}, {"source", "user"}};
            } else {
                findings.push_back(finding("session_context_ignored", "warning",
                                           "entry '" + it.key() + "' lacks value/source=user; not used"));
            }
        }

    json capabilities = json::object();
    if (in.capabilities.is_object())
        for (auto it = in.capabilities.begin(); it != in.capabilities.end(); ++it) {
            if (it.value().is_boolean() && std::regex_match(it.key(), std::regex("^[a-z0-9_]{1,64}$"))) {
                capabilities[it.key()] = it.value();
            } else {
                findings.push_back(finding("capability_ignored", "warning", "entry '" + it.key() + "' is not a boolean fact"));
            }
        }

    // ---- identity ----
    json manifest = json::array();
    bool all_digest = false;
    if (in.dataset_manifest.is_array() && !in.dataset_manifest.empty()) {
        all_digest = true;
        for (const auto& e : in.dataset_manifest) {
            if (!e.is_object()) continue;
            json n = {{"id", basename_of(e.value("id", std::string()))},
                      {"size", e.contains("size") ? e["size"] : json(nullptr)},
                      {"mtime", e.contains("mtime") ? e["mtime"] : json(nullptr)}};
            if (e.contains("sha256") && is_sha256_hex(e["sha256"])) n["sha256"] = e["sha256"];
            else all_digest = false;
            manifest.push_back(n);
        }
    } else {
        for (const auto& f : frames)
            if (f.is_object()) manifest.push_back({{"id", basename_of(f.value("file_name", std::string()))}});
    }
    std::sort(manifest.begin(), manifest.end(),
              [](const json& a, const json& b) { return a.value("id", std::string()) < b.value("id", std::string()); });
    const std::string identity_strength = all_digest ? "content_digest" : "metadata";
    if (identity_strength == "metadata")
        findings.push_back(finding("identity_strength_metadata", "info",
                                   "dataset fingerprint is a metadata manifest; content identity is not established"));
    std::string scan_id = in.scan_id;
    if (scan_id.empty()) {
        scan_id = "unspecified";
        findings.push_back(finding("scan_id_unspecified", "warning", "no scan id supplied"));
    }
    std::vector<std::string> locks = in.locked_paths;
    std::sort(locks.begin(), locks.end());
    locks.erase(std::unique(locks.begin(), locks.end()), locks.end());

    // Frame order is not semantic: hash a frame list sorted by its own canonical form.
    json metrics_semantic = strip_volatile(metrics);
    if (metrics_semantic.contains("frames") && metrics_semantic["frames"].is_array()) {
        std::vector<std::pair<std::string, json>> keyed;
        for (const auto& f : metrics_semantic["frames"]) keyed.emplace_back(canonical_json_dump(f), f);
        std::sort(keyed.begin(), keyed.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
        json sorted_frames = json::array();
        for (auto& kv : keyed) sorted_frames.push_back(std::move(kv.second));
        metrics_semantic["frames"] = std::move(sorted_frames);
    }
    json scan_semantic = {{"scan", strip_volatile(scan)}, {"metrics", metrics_semantic}};
    const std::string calibration_state = [&] {
        const std::string c = scan.value("calibration_state", std::string("unknown"));
        return (c == "uncalibrated" || c == "calibrated") ? c : std::string("unknown");
    }();

    json state = {
        {"schema_version", kDecisionStateSchemaVersion},
        {"domain", "pre_run"},
        {"identity",
         {{"dataset_fingerprint", sha256_prefixed(canonical_json_dump({{"entries", manifest}, {"strength", identity_strength}}))},
          {"scan_id", scan_id},
          {"scan_hash", sha256_prefixed(canonical_json_dump(scan_semantic))},
          {"config_hash", sha256_prefixed(canonical_json_dump(base_config))},
          {"locks_hash", sha256_prefixed(canonical_json_dump(json(locks)))},
          {"software_version", in.software_version.empty() ? std::string("unknown") : in.software_version},
          {"identity_strength", identity_strength}}},
        {"measurement_contract",
         {{"method_version", kDecisionStateMethodVersion}, {"pixel_grid", "source"}, {"calibration_state", calibration_state}}},
        {"coverage", {{"detected", detected}, {"measured", static_cast<int>(frames.size())}, {"read_ok", read_ok}}},
        {"policy", {{"min_valid_for_spread", in.options.min_valid_for_spread}}},
        {"groups", groups_json},
        {"session_facts", session_facts},
        {"capabilities", capabilities},
        {"base_config", base_config.is_object() ? base_config : json::object()},
        {"locked_paths", json(locks)},
        {"blocking_findings", json(blocking)}};

    result.state_hash = sha256_prefixed(canonical_json_dump(state));

    // ---- provider projection: explicit allowlist, then a defensive leak check ----
    json proj_groups = json::array();
    for (const auto& g : groups_json) {
        json pm = json::object();
        for (auto it = g["metrics"].begin(); it != g["metrics"].end(); ++it) {
            json pmetric = json::object();
            for (auto sit = it.value().begin(); sit != it.value().end(); ++sit) {
                const json& m = sit.value();
                json compact = {{"value", m["value"]}, {"status", m["status"]}, {"unit", m["unit"]},
                                {"valid_count", m["valid_count"]}, {"total_count", m["total_count"]}};
                if (m.contains("reason")) compact["reason"] = m["reason"];
                pmetric[sit.key()] = compact;
            }
            pm[it.key()] = pmetric;
        }
        json pagree = json::object();
        for (auto sit = g["metric_agreement"].begin(); sit != g["metric_agreement"].end(); ++sit) {
            const json& m = sit.value();
            json compact = {{"value", m["value"]}, {"status", m["status"]}, {"unit", m["unit"]},
                            {"valid_count", m["valid_count"]}, {"total_count", m["total_count"]}};
            if (m.contains("reason")) compact["reason"] = m["reason"];
            pagree[sit.key()] = compact;
        }
        json filter = g["filter"];
        if (filter.is_string() && !std::regex_match(filter.get<std::string>(), std::regex("^[A-Za-z0-9 _.+-]{1,32}$"))) {
            filter = nullptr;
            findings.push_back(finding("filter_not_projected", "info", "filter text not a short plain token; withheld from provider"));
        }
        proj_groups.push_back({{"group_id", g["group_id"]}, {"frame_count", g["frame_count"]},
                               {"color_mode", g["color_mode"]}, {"bayer_pattern", g["bayer_pattern"]},
                               {"filter", filter}, {"exposure_seconds", g["exposure_seconds"]},
                               {"gain", g["gain"]}, {"frames_read_failed", g["frames_read_failed"]}, {"metrics", pm},
                               {"metric_agreement", pagree}});
    }
    json config_facts = json::object();
    json locked = json::object();
    for (const char* path : kProjectedConfigPaths) {
        config_facts[path] = config_value_at(base_config, path);
        locked[path] = std::binary_search(locks.begin(), locks.end(), std::string(path));
    }
    // User-stated facts are free-form: only bools, numbers and short plain tokens reach the provider.
    json projected_session = session_facts;
    projected_session["user_stated"] = json::object();
    for (auto it = session_facts["user_stated"].begin(); it != session_facts["user_stated"].end(); ++it) {
        const json& v = it.value()["value"];
        const bool plain = v.is_boolean() || (v.is_number() && std::isfinite(v.get<double>())) ||
                           (v.is_string() && std::regex_match(v.get<std::string>(), std::regex("^[A-Za-z0-9 _.+-]{1,32}$")));
        if (plain) projected_session["user_stated"][it.key()] = it.value();
        else findings.push_back(finding("user_fact_not_projected", "info", "entry '" + it.key() + "' withheld from provider"));
    }
    json projection = {{"schema_version", kDecisionStateSchemaVersion},
                       {"domain", "pre_run"},
                       {"coverage", state["coverage"]},
                       {"measurement_contract", state["measurement_contract"]},
                       {"policy", state["policy"]},
                       {"groups", proj_groups},
                       {"session_facts", projected_session},
                       {"config_facts", config_facts},
                       {"locked", locked},
                       {"capabilities", capabilities},
                       {"blocking_findings", json(blocking)}};
    assert_projection_clean(projection, "provider_projection");

    result.state = std::move(state);
    result.findings = std::move(findings);
    result.provider_projection = std::move(projection);
    return result;
}

} // namespace tile_compile::pi
