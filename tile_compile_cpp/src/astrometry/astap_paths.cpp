#include "tile_compile/astrometry/astap_paths.hpp"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <system_error>
#include <vector>

namespace tile_compile::astrometry {

namespace fs = std::filesystem;

namespace {

bool astap_binary_exists(const fs::path &candidate) {
  if (candidate.empty()) return false;
  std::error_code ec;
  if (fs::is_regular_file(candidate, ec)) return true;
#ifdef _WIN32
  // Also try the .exe extension on Windows
  fs::path with_exe = candidate;
  std::string ext = with_exe.extension().string();
  std::transform(ext.begin(), ext.end(), ext.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  if (ext != ".exe") {
    with_exe += ".exe";
    if (fs::is_regular_file(with_exe, ec)) return true;
  }
#endif
  return false;
}

std::optional<std::string> popen_first_line(const std::string &cmd) {
  FILE *fp = popen(cmd.c_str(), "r");
  if (!fp) return std::nullopt;
  char buf[512] = {};
  if (fgets(buf, sizeof(buf), fp)) {
    std::string found(buf);
    while (!found.empty() &&
           (found.back() == '\n' || found.back() == '\r' || found.back() == ' '))
      found.pop_back();
    pclose(fp);
    if (!found.empty()) return found;
  } else {
    pclose(fp);
  }
  return std::nullopt;
}

} // namespace

std::string default_astap_data_dir() {
#ifdef _WIN32
  if (const char *la = std::getenv("LOCALAPPDATA"); la && la[0] != '\0') {
    return std::string(la) + "\\tile_compile\\astap";
  }
#else
  if (const char *home = std::getenv("HOME"); home && home[0] != '\0') {
    return std::string(home) + "/.local/share/tile_compile/astap";
  }
#endif
  return {};
}

fs::path resolve_astap_binary_path(const std::string &astap_bin_cfg,
                                   const std::string &astap_data_dir) {
  // 1. Explicitly configured path
  if (!astap_bin_cfg.empty()) {
    fs::path explicit_path(astap_bin_cfg);
    if (astap_binary_exists(explicit_path)) return explicit_path;
  }

  // 2. Inside the configured data directory
  if (!astap_data_dir.empty()) {
    fs::path data_dir_path(astap_data_dir);
    for (const char *name : {"astap_cli", "astap"}) {
      fs::path candidate = data_dir_path / name;
      if (astap_binary_exists(candidate)) return candidate;
    }
  }

  // 3. PATH lookup
  for (const char *name : {"astap_cli", "astap"}) {
#ifdef _WIN32
    std::string cmd = std::string("where ") + name + " 2>nul";
#else
    std::string cmd = std::string("which ") + name + " 2>/dev/null";
#endif
    auto found = popen_first_line(cmd);
    if (found && astap_binary_exists(*found)) return fs::path(*found);
  }

  // 4. Common Windows install locations
#ifdef _WIN32
  auto probe_common_dirs = [](const std::vector<fs::path> &roots) -> fs::path {
    for (const auto &root : roots) {
      if (root.empty()) continue;
      for (const char *name : {"astap_cli.exe", "astap.exe"}) {
        fs::path candidate = root / "astap" / name;
        if (astap_binary_exists(candidate)) return candidate;
        candidate = root / name;
        if (astap_binary_exists(candidate)) return candidate;
      }
    }
    return {};
  };
  std::vector<fs::path> roots;
  if (const char *pf = std::getenv("ProgramFiles")) roots.emplace_back(pf);
  if (const char *pf = std::getenv("ProgramFiles(x86)")) roots.emplace_back(pf);
  if (const char *la = std::getenv("LOCALAPPDATA")) roots.emplace_back(la);
  fs::path common = probe_common_dirs(roots);
  if (!common.empty()) return common;
#endif

  return {};
}

AstapPaths resolve_astap_paths(const std::string &binary_cfg,
                               const std::string &data_dir_cfg) {
  AstapPaths out;
  out.data_dir = data_dir_cfg;
  std::error_code ec;
  // A configured but non-existent directory (e.g. a config copied from
  // another machine) falls back to the per-user default like an empty one.
  if (out.data_dir.empty() || !fs::is_directory(fs::path(out.data_dir), ec)) {
    out.data_dir = default_astap_data_dir();
  }
  out.binary = resolve_astap_binary_path(binary_cfg, out.data_dir);
  if (!out.binary.empty()) {
    const auto relative = fs::relative(out.binary, fs::path(out.data_dir), ec);
    if (ec || relative.empty() || *relative.begin() == "..") {
      out.data_dir = out.binary.parent_path().string();
    }
  }
  return out;
}

} // namespace tile_compile::astrometry
