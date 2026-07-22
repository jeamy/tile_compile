#include "routes/route_utils.hpp"
#include <filesystem>
#include <fstream>
#include <yaml-cpp/yaml.h>
#include <cmath>
#include <cstdio>
#include <sstream>

namespace tile_compile::routes {

static nlohmann::json allowed_roots_json(const std::shared_ptr<AppState>& state) {
    nlohmann::json roots = nlohmann::json::array();
    for (const auto& root : state->runtime.allowed_roots()) roots.push_back(root.string());
    return roots;
}

crow::response json_resp(const nlohmann::json& j, int status) {
    crow::response res(status, j.dump());
    res.set_header("Content-Type", "application/json");
    return res;
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
        return err_resp("PATH_INVALID", "Path cannot be empty", 400, {{"path", path}});
    }
    
    auto resolved = state->runtime.resolve_input_path(fs::path(path), true);
    if (resolved.status == PathStatus::not_allowed) {
        return err_resp("PATH_NOT_ALLOWED", "Path is outside allowed roots", 403, {{"path", resolved.path.string()}, {"allowed_roots", allowed_roots_json(state)}});
    }
    if (resolved.status == PathStatus::not_found) {
        return err_resp("PATH_NOT_FOUND", "Path does not exist", 404, {{"path", resolved.path.string()}});
    }
    
    return std::nullopt;
}

std::optional<crow::response> validate_path(const std::shared_ptr<AppState>& state,
                                           fs::path& path,
                                           bool must_exist) {
    if (!state) {
        return err_resp("internal_error", "Application state not available", 500);
    }
    auto resolved = state->runtime.resolve_input_path(path, must_exist);
    path = resolved.path;
    if (resolved.status == PathStatus::not_allowed) {
        return err_resp("PATH_NOT_ALLOWED", "Path is outside allowed roots", 403, {{"path", path.string()}, {"allowed_roots", allowed_roots_json(state)}});
    }
    if (resolved.status == PathStatus::not_found) {
        return err_resp("PATH_NOT_FOUND", "Path does not exist", 400, {{"path", path.string()}});
    }
    return std::nullopt;
}

std::optional<crow::response> validate_path(const std::shared_ptr<AppState>& state,
                                           fs::path& path,
                                           const std::string& label,
                                           bool must_exist) {
    if (!state) {
        return err_resp("internal_error", "Application state not available", 500);
    }
    auto resolved = state->runtime.resolve_input_path(path, must_exist);
    path = resolved.path;
    if (resolved.status == PathStatus::not_allowed) {
        return err_resp("PATH_NOT_ALLOWED", label + " is outside allowed roots", 403, {{"path", path.string()}, {"allowed_roots", allowed_roots_json(state)}});
    }
    if (resolved.status == PathStatus::not_found) {
        return err_resp("PATH_NOT_FOUND", label + " does not exist", 400, {{"path", path.string()}});
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

std::optional<nlohmann::json> parse_json_string(const std::string& raw) {
    auto parsed = nlohmann::json::parse(raw, nullptr, false);
    if (parsed.is_discarded()) return std::nullopt;
    return parsed;
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
    if (value.is_number_float()) {
        double d = value.get<double>();
        if (d != 0.0 && std::isfinite(d)) {
            char buf[64];
            std::snprintf(buf, sizeof(buf), "%.4g", d);
            return YAML::Node(std::string(buf));
        }
        return YAML::Node(d);
    }
    if (value.is_null()) return YAML::Node();
    return YAML::Node(value.get<std::string>());
}

std::string yaml_dump(const nlohmann::json& value) {
    YAML::Node node = json_to_yaml_node(value);
    YAML::Emitter out;
    out.SetFloatPrecision(6);
    out.SetDoublePrecision(6);
    out << node;
    return std::string(out.c_str());
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

std::optional<nlohmann::json> parse_body(const crow::request& req) {
    if (req.body.empty()) return nlohmann::json::object();
    auto parsed = nlohmann::json::parse(req.body, nullptr, false);
    if (parsed.is_discarded()) return std::nullopt;
    if (!parsed.is_object()) return nlohmann::json::object();
    return parsed;
}

crow::response backend_command_failed(const std::string& message, const SubprocessResult& result) {
    return err_resp("BACKEND_COMMAND_FAILED", message, 502, {
        {"exit_code", result.exit_code},
        {"stdout", result.stdout_str},
        {"stderr", result.stderr_str},
    });
}

std::string read_file_str(const std::filesystem::path& path) {
    std::ifstream in(path);
    if (!in) return "";
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

bool write_file_str(const std::filesystem::path& path, const std::string& text) {
    if (!path.parent_path().empty()) {
        std::error_code ec;
        std::filesystem::create_directories(path.parent_path(), ec);
    }
    std::ofstream out(path, std::ios::out | std::ios::trunc);
    if (!out) return false;
    out << text;
    return static_cast<bool>(out);
}

nlohmann::json yaml_to_json(const YAML::Node& node) {
    if (!node || node.IsNull()) return nullptr;
    if (node.IsMap()) {
        nlohmann::json out = nlohmann::json::object();
        for (auto it = node.begin(); it != node.end(); ++it) {
            out[it->first.as<std::string>()] = yaml_to_json(it->second);
        }
        return out;
    }
    if (node.IsSequence()) {
        nlohmann::json out = nlohmann::json::array();
        for (auto it = node.begin(); it != node.end(); ++it) {
            out.push_back(yaml_to_json(*it));
        }
        return out;
    }
    try { return node.as<bool>(); } catch (...) {}
    try { return node.as<int>(); } catch (...) {}
    try { return node.as<double>(); } catch (...) {}
    try { return node.as<std::string>(); } catch (...) {}
    return nullptr;
}

std::optional<nlohmann::json> parse_yaml_text(const std::string& yaml_text) {
    if (yaml_text.empty()) return nlohmann::json::object();
    try {
        return yaml_to_json(YAML::Load(yaml_text));
    } catch (const std::exception&) {
        return std::nullopt;
    }
}

std::optional<nlohmann::json> parse_yaml_file(const std::filesystem::path& path) {
    std::ifstream f(path);
    if (!f) return std::nullopt;
    try {
        return yaml_to_json(YAML::Load(f));
    } catch (const std::exception&) {
        return std::nullopt;
    }
}

} // namespace tile_compile::routes
