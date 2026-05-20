#include "routes/route_utils.hpp"
#include "app_state.hpp"
#include <filesystem>

namespace tile_compile::routes {

crow::response json_resp(const nlohmann::json& j, int status) {
    crow::response res(status, j.dump());
    res.set_header("Content-Type", "application/json");
    return res;
}

crow::response json_response(const nlohmann::json& j, int status) {
    return json_resp(j, status);
}

crow::response err_resp(const std::string& msg, int status) {
    nlohmann::json err = nlohmann::json::object();
    err["error"] = msg;
    return json_resp(err, status);
}

crow::response err_resp(const std::string& code, const std::string& msg, int status) {
  nlohmann::json j = {
    {"error", true},
    {"code", code},
    {"message", msg}
  };
  return json_resp(j, status);
}

crow::response err_resp(const std::string& code, const std::string& msg, int status, const nlohmann::json& details) {
  nlohmann::json j = {
    {"error", true},
    {"code", code},
    {"message", msg}
  };
  if (!details.is_null()) {
    j["details"] = details;
  }
  return json_resp(j, status);
}

std::optional<crow::response> validate_path(const std::shared_ptr<AppState>& state,
                                           const std::string& path) {
    if (!state) {
        return err_resp("internal_error", "Application state not available", 500);
    }
    
    if (path.empty()) {
        return err_resp("invalid_path", "Path cannot be empty", 400);
    }
    
    auto resolved = state->runtime.resolve_input_path(fs::path(path), true);
    if (resolved.status == PathStatus::not_allowed) {
        return err_resp("invalid_path", "Path outside allowed directory", 400);
    }
    if (resolved.status == PathStatus::not_found) {
        return err_resp("path_not_found", "Path does not exist", 404);
    }
    
    return std::nullopt;
}

int parse_int_param(const crow::request& req, const std::string& param_name, int default_value) {
    try {
        auto param = req.url_params.get(param_name);
        if (param) {
            return std::stoi(param);
        }
    } catch (...) {
        // Silently fall back to default on parse error
    }
    return default_value;
}

std::string parse_string_param(const crow::request& req, const std::string& param_name, const std::string& default_value) {
    auto param = req.url_params.get(param_name);
    return param ? std::string(param) : default_value;
}

} // namespace tile_compile::routes
