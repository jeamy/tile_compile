#pragma once

#include "tile_compile/config/configuration.hpp"

#include <filesystem>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace tile_compile::config {

/// One finding about a configured (non-empty) filesystem path that does not
/// exist on this machine. Empty paths are never reported: they keep their
/// documented default behaviour.
struct PathIssue {
  std::string key;         ///< Config key, e.g. "calibration.dark_master".
  std::string configured;  ///< Value as written in the config.
  std::string resolved;    ///< Path that was checked (relative -> project root).
  /// "default_used": a standard location is used instead.
  /// "fallback_used": another configured source of the same data is used.
  /// "online_fallback": the run falls back to an online catalog query.
  /// "unused": missing, but the run does not read this path.
  /// "missing": missing and no replacement exists.
  std::string status;
  std::string fallback;    ///< Replacement path, if any.
  bool blocking = false;   ///< True when the run needs the data and must abort.
  std::string message;     ///< English one-line explanation.
};

/// Which parts of the pipeline will run, so only paths whose data is actually
/// read can block. A downstream-only resume, for example, never calibrates.
struct PathCheckScope {
  bool calibration = true;     ///< SCAN_INPUT calibration (bias/dark/flat).
  bool registration = true;    ///< Registration (ASTAP astrometric rescue).
  bool astrometry = true;      ///< ASTROMETRY phase.
  bool pcc = true;             ///< PCC phase (Siril Gaia catalog).
};

/// Check every configured path of `cfg`. Calibration paths are resolved like
/// the runner does (trimmed, relative to `project_root`); ASTAP and Siril
/// catalog paths are used verbatim, as the solver phases do.
std::vector<PathIssue> check_config_paths(const Config &cfg,
                                          const std::filesystem::path &project_root,
                                          const PathCheckScope &scope = {});

} // namespace tile_compile::config

namespace tile_compile::preprocessing { struct Config; }

namespace tile_compile::config {

/// Same check for the separate `preprocess` pipeline, with its calibration
/// semantics (a non-empty path enables its type; master or directory, no
/// fallback between them; dark-flats only when a flat is applied). ASTAP and
/// Siril paths are checked only when their config block is present, as the
/// pipeline ignores them otherwise.
std::vector<PathIssue> check_preprocessing_paths(const preprocessing::Config &cfg,
                                                 const std::filesystem::path &project_root);

/// True when any issue blocks the run.
bool has_blocking_path_issue(const std::vector<PathIssue> &issues);

/// JSON array of {key, path, configured, resolved, status, fallback, blocking, message}.
/// `path` repeats `key` so generic warning renderers show the config key.
nlohmann::json path_issues_to_json(const std::vector<PathIssue> &issues);

} // namespace tile_compile::config
