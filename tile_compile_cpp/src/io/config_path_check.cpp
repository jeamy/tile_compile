#include "tile_compile/config/path_check.hpp"

#include "tile_compile/astrometry/astap_paths.hpp"
#include "tile_compile/astrometry/gaia_catalog.hpp"
#include "tile_compile/preprocessing/contract.hpp"

#include <algorithm>
#include <cctype>
#include <system_error>

namespace tile_compile::config {

namespace fs = std::filesystem;

namespace {

std::string trim_copy(const std::string &s) {
  auto is_space = [](unsigned char c) { return std::isspace(c) != 0; };
  auto b = std::find_if_not(s.begin(), s.end(), is_space);
  auto e = std::find_if_not(s.rbegin(), s.rend(), is_space).base();
  return b < e ? std::string(b, e) : std::string();
}

// Same resolution as the runner's calibration loader (resolve_config_path).
fs::path resolve_relative(const fs::path &project_root, const std::string &raw) {
  const std::string trimmed = trim_copy(raw);
  if (trimmed.empty()) return {};
  fs::path p(trimmed);
  if (p.is_relative() && !project_root.empty()) p = project_root / p;
  return p.lexically_normal();
}

bool is_file(const fs::path &p) {
  std::error_code ec;
  return !p.empty() && fs::is_regular_file(p, ec);
}

bool is_dir(const fs::path &p) {
  std::error_code ec;
  return !p.empty() && fs::is_directory(p, ec);
}

bool is_executable_file(const fs::path &p) {
  if (is_file(p)) return true;
#ifdef _WIN32
  fs::path with_exe = p;
  with_exe += ".exe";
  if (is_file(with_exe)) return true;
#endif
  return false;
}

// ASTAP star databases: H17/H18/V17 (.290/.291) and D05/D20/D50/D80 (.1476).
// Keep in sync with registration::is_astap_available.
bool has_astap_catalog(const fs::path &dir) {
  std::error_code ec;
  if (!is_dir(dir)) return false;
  for (fs::directory_iterator it(dir, ec), end; !ec && it != end; it.increment(ec)) {
    const std::string ext = it->path().extension().string();
    if (ext == ".290" || ext == ".291" || ext == ".1476") return true;
  }
  return false;
}

void check_calibration_kind(const std::string &kind, bool use_kind,
                            const std::string &master_raw,
                            const std::string &dir_raw,
                            const std::string &dir_key,
                            const fs::path &project_root, bool in_scope,
                            std::vector<PathIssue> &out) {
  const fs::path master = resolve_relative(project_root, master_raw);
  const fs::path dir = resolve_relative(project_root, dir_raw);
  const bool master_bad = !master.empty() && !is_file(master);
  const bool dir_bad = !dir.empty() && !is_dir(dir);
  const bool master_ok = !master.empty() && !master_bad;
  const bool dir_ok = !dir.empty() && !dir_bad;
  const bool needed = in_scope && use_kind;
  const std::string use_key = "calibration.use_" + kind;

  auto report = [&](const std::string &key, const std::string &configured,
                    const fs::path &resolved, bool other_ok,
                    const fs::path &other, const std::string &what) {
    PathIssue issue;
    issue.key = key;
    issue.configured = configured;
    issue.resolved = resolved.string();
    if (!needed) {
      issue.status = "unused";
      issue.message = what + " not found: " + resolved.string() + " (not used by this run: " +
                      use_key + " is off or calibration does not run)";
    } else if (other_ok) {
      issue.status = "fallback_used";
      issue.fallback = other.string();
      issue.message = what + " not found: " + resolved.string() +
                      "; the run uses " + other.string() + " instead";
    } else {
      issue.status = "missing";
      issue.blocking = true;
      issue.message = what + " not found: " + resolved.string() +
                      "; " + use_key + " is on and no other " + kind +
                      " source exists, the run cannot calibrate";
    }
    out.push_back(std::move(issue));
  };

  if (master_bad) {
    report("calibration." + kind + "_master", master_raw, master, dir_ok, dir,
           kind + " master file");
  }
  if (dir_bad) {
    report("calibration." + dir_key, dir_raw, dir, master_ok, master,
           kind + " directory");
  }
}

// ASTAP data dir / binary. `needed_hard`: the ASTROMETRY phase runs (abort
// without a replacement); `needed_soft`: only the optional registration rescue.
void check_astap(const std::string &dir_raw, const std::string &bin_raw,
                 bool needed_hard, bool needed_soft, std::vector<PathIssue> &out) {
  const std::string astap_dir_cfg = trim_copy(dir_raw);
  const std::string astap_bin_cfg = trim_copy(bin_raw);
  const bool astap_dir_bad =
      !astap_dir_cfg.empty() && !is_dir(fs::path(astap_dir_cfg));
  const bool astap_bin_bad =
      !astap_bin_cfg.empty() && !is_executable_file(fs::path(astap_bin_cfg));
  if (!astap_dir_bad && !astap_bin_bad) return;
  const auto resolved = astrometry::resolve_astap_paths(astap_bin_cfg, astap_dir_cfg);
  const bool usable =
      !resolved.binary.empty() && has_astap_catalog(fs::path(resolved.data_dir));

  auto report = [&](const std::string &key, const std::string &configured,
                    const std::string &what, const std::string &fallback) {
    PathIssue issue;
    issue.key = key;
    issue.configured = configured;
    issue.resolved = configured;
    if (usable) {
      issue.status = "default_used";
      issue.fallback = fallback;
      issue.message = what + " not found: " + configured +
                      "; the run uses " + fallback + " instead";
    } else if (needed_hard) {
      issue.status = "missing";
      issue.blocking = true;
      issue.message = what + " not found: " + configured +
                      "; no ASTAP installation with star catalog was found in "
                      "the standard locations, astrometry is enabled";
    } else if (needed_soft) {
      issue.status = "missing";
      issue.message = what + " not found: " + configured +
                      "; no ASTAP fallback found, the astrometric registration "
                      "rescue is skipped";
    } else {
      issue.status = "unused";
      issue.message = what + " not found: " + configured +
                      " (not used by this run)";
    }
    out.push_back(std::move(issue));
  };
  if (astap_dir_bad) {
    report("astrometry.astap_data_dir", astap_dir_cfg, "ASTAP data directory",
           resolved.data_dir);
  }
  if (astap_bin_bad) {
    report("astrometry.astap_bin", astap_bin_cfg, "ASTAP binary",
           resolved.binary.string());
  }
}

// Siril Gaia catalog dir. `pcc_runs`: the PCC phase runs.
void check_siril(const std::string &dir_raw, const std::string &source,
                 bool pcc_runs, std::vector<PathIssue> &out) {
  const std::string siril_cfg = trim_copy(dir_raw);
  if (siril_cfg.empty() || astrometry::is_siril_gaia_catalog_available(siril_cfg))
    return;
  PathIssue issue;
  issue.key = "pcc.siril_catalog_dir";
  issue.configured = siril_cfg;
  issue.resolved = siril_cfg;
  const std::string what = is_dir(fs::path(siril_cfg))
                               ? "Siril Gaia catalog not present in "
                               : "Siril Gaia catalog directory not found: ";
  const std::string def = astrometry::default_siril_gaia_catalog_dir();
  const bool reads_siril = source == "siril" || source == "auto";
  const bool needed = pcc_runs && reads_siril;
  if (!needed) {
    issue.status = "unused";
    issue.message = what + siril_cfg + " (not used by this run)";
  } else if (!def.empty() && astrometry::is_siril_gaia_catalog_available(def)) {
    issue.status = "default_used";
    issue.fallback = def;
    issue.message = what + siril_cfg + "; the run uses " + def + " instead";
  } else if (source == "auto") {
    issue.status = "online_fallback";
    issue.message = what + siril_cfg +
                    "; no local catalog found, PCC queries VizieR online "
                    "(needs internet access)";
  } else {
    issue.status = "missing";
    issue.blocking = true;
    issue.message = what + siril_cfg +
                    "; pcc.source is siril and no catalog was found in the "
                    "standard location";
  }
  out.push_back(std::move(issue));
}

// Preprocess calibration semantics (runner_phase_preprocess_pipeline.cpp):
// a non-empty path switches its calibration type on; the master is used when
// `*_use_master` is set or the master path is non-empty, otherwise the
// directory. There is no master <-> directory fallback.
void check_preprocess_kind(const std::string &kind, bool type_used,
                           bool use_master_flag, const std::string &master_raw,
                           const std::string &dir_raw, const std::string &dir_key,
                           const fs::path &project_root, std::vector<PathIssue> &out) {
  const fs::path master = resolve_relative(project_root, master_raw);
  const fs::path dir = resolve_relative(project_root, dir_raw);
  const bool master_route = use_master_flag || !master.empty();
  auto report = [&](const std::string &key, const std::string &configured,
                    const fs::path &resolved, bool needed, const std::string &what) {
    PathIssue issue;
    issue.key = key;
    issue.configured = configured;
    issue.resolved = resolved.string();
    if (needed) {
      issue.status = "missing";
      issue.blocking = true;
      issue.message = what + " not found: " + resolved.string() +
                      "; preprocessing needs it and has no other " + kind + " source";
    } else {
      issue.status = "unused";
      issue.message = what + " not found: " + resolved.string() +
                      (type_used ? " (not used: the " + kind + " master file is used instead)"
                                 : std::string(" (not used by this run)"));
    }
    out.push_back(std::move(issue));
  };
  if (!master.empty() && !is_file(master)) {
    report("calibration." + kind + "_master", master_raw, master, type_used,
           kind + " master file");
  }
  if (!dir.empty() && !is_dir(dir)) {
    report(dir_key, dir_raw, dir, type_used && !master_route, kind + " directory");
  }
}

} // namespace

std::vector<PathIssue> check_config_paths(const Config &cfg,
                                          const fs::path &project_root,
                                          const PathCheckScope &scope) {
  std::vector<PathIssue> out;

  // --- Calibration: no standard location exists for user calibration data.
  const auto &cal = cfg.calibration;
  check_calibration_kind("bias", cal.use_bias, cal.bias_master, cal.bias_dir,
                         "bias_dir", project_root, scope.calibration, out);
  check_calibration_kind("dark", cal.use_dark, cal.dark_master, cal.darks_dir,
                         "darks_dir", project_root, scope.calibration, out);
  check_calibration_kind("flat", cal.use_flat, cal.flat_master, cal.flats_dir,
                         "flats_dir", project_root, scope.calibration, out);

  // --- ASTAP: falls back to the per-user default directory / PATH.
  check_astap(cfg.astrometry.astap_data_dir, cfg.astrometry.astap_bin,
              scope.astrometry && cfg.astrometry.enabled,
              scope.registration && cfg.registration.use_astrometry, out);

  // --- Siril Gaia catalog: default directory, then (source "auto") VizieR.
  check_siril(cfg.pcc.siril_catalog_dir, cfg.pcc.source,
              scope.pcc && cfg.pcc.enabled, out);

  return out;
}

std::vector<PathIssue> check_preprocessing_paths(const preprocessing::Config &cfg,
                                                 const fs::path &project_root) {
  std::vector<PathIssue> out;
  const auto &cal = cfg.calibration;
  const bool use_bias = cal.use_bias || !trim_copy(cfg.bias_dir).empty() ||
                        !trim_copy(cal.bias_master).empty();
  const bool use_dark = cal.use_dark || !trim_copy(cfg.darks_dir).empty() ||
                        !trim_copy(cal.dark_master).empty();
  const bool use_flat = cal.use_flat || !trim_copy(cfg.flats_dir).empty() ||
                        !trim_copy(cal.flat_master).empty();
  check_preprocess_kind("bias", use_bias, cal.bias_use_master, cal.bias_master,
                        cfg.bias_dir, "bias_dir", project_root, out);
  check_preprocess_kind("dark", use_dark, cal.dark_use_master, cal.dark_master,
                        cfg.darks_dir, "darks_dir", project_root, out);
  check_preprocess_kind("flat", use_flat, cal.flat_use_master, cal.flat_master,
                        cfg.flats_dir, "flats_dir", project_root, out);
  // Dark-flats are subtracted from the flat master, so they matter only when
  // a flat is applied.
  check_preprocess_kind("darkflat", use_flat, cal.darkflat_use_master,
                        cal.darkflat_master, cfg.darkflats_dir, "darkflats_dir",
                        project_root, out);

  if (cfg.has_astrometry_config) {
    check_astap(cfg.astrometry.astap_data_dir, cfg.astrometry.astap_bin,
                cfg.postprocess.astrometry, false, out);
  }
  if (cfg.has_pcc_config) {
    check_siril(cfg.pcc.siril_catalog_dir, cfg.pcc.source, cfg.postprocess.pcc, out);
  }
  return out;
}

bool has_blocking_path_issue(const std::vector<PathIssue> &issues) {
  return std::any_of(issues.begin(), issues.end(),
                     [](const PathIssue &i) { return i.blocking; });
}

nlohmann::json path_issues_to_json(const std::vector<PathIssue> &issues) {
  nlohmann::json arr = nlohmann::json::array();
  for (const auto &i : issues) {
    arr.push_back({{"key", i.key},
                   {"path", i.key},
                   {"configured", i.configured},
                   {"resolved", i.resolved},
                   {"status", i.status},
                   {"fallback", i.fallback},
                   {"blocking", i.blocking},
                   {"message", i.message}});
  }
  return arr;
}

} // namespace tile_compile::config
