#pragma once

#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace tile_compile::gui {

class BackendClient {
  public:
    BackendClient(std::string project_root, nlohmann::json constants);

    std::vector<std::string> backend_cmd() const;
    const nlohmann::json& constants() const { return constants_; }

    nlohmann::json run_json(const std::string &cwd,
                            const std::vector<std::string> &args,
                            const std::string &stdin_text = std::string(),
                            int timeout_ms = 30000) const;

  private:
    std::string project_root_;
    nlohmann::json constants_;

    std::vector<std::string> resolve_backend_cmd() const;
};

}
