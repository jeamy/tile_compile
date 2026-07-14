#include "services/ai_service.hpp"

#include <curl/curl.h>
#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <utility>

namespace tile_compile::ai {
namespace {

std::string env_string(const char* name, const std::string& fallback = "") {
    const char* value = std::getenv(name);
    return value ? std::string(value) : fallback;
}

bool env_bool(const char* name, bool fallback) {
    std::string raw = env_string(name);
    if (raw.empty()) return fallback;
    std::transform(raw.begin(), raw.end(), raw.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return raw == "1" || raw == "true" || raw == "yes" || raw == "on";
}

int env_int(const char* name, int fallback, int min_value, int max_value) {
    const std::string raw = env_string(name);
    if (raw.empty()) return fallback;
    try {
        return std::clamp(std::stoi(raw), min_value, max_value);
    } catch (...) {
        return fallback;
    }
}

double env_double(const char* name, double fallback, double min_value, double max_value) {
    const std::string raw = env_string(name);
    if (raw.empty()) return fallback;
    try {
        return std::clamp(std::stod(raw), min_value, max_value);
    } catch (...) {
        return fallback;
    }
}

size_t curl_write(char* ptr, size_t size, size_t nmemb, void* userdata) {
    auto* out = static_cast<std::string*>(userdata);
    out->append(ptr, size * nmemb);
    return size * nmemb;
}

std::string join_url(const std::string& base, const std::string& endpoint) {
    if (endpoint.empty()) return base;
    if (base.empty()) return endpoint;
    const bool base_slash = base.back() == '/';
    const bool endpoint_slash = endpoint.front() == '/';
    if (base_slash && endpoint_slash) return base + endpoint.substr(1);
    if (!base_slash && !endpoint_slash) return base + "/" + endpoint;
    return base + endpoint;
}

void set_if_present(nlohmann::json& target, const nlohmann::json& source, const char* key) {
    if (!source.contains(key)) return;
    const auto& next = source[key];
    if (!target.contains(key)) {
        target[key] = next;
        return;
    }
    const auto& current = target[key];
    if (current.is_string()) {
        if (next.is_string()) target[key] = next;
        return;
    }
    if (current.is_boolean()) {
        if (next.is_boolean() || next.is_string()) target[key] = next;
        return;
    }
    if (current.is_number()) {
        if (next.is_number() || next.is_string()) target[key] = next;
        return;
    }
    target[key] = next;
}

std::string json_string_field(const nlohmann::json& value,
                              const char* key,
                              const std::string& fallback) {
    if (!value.is_object() || !value.contains(key)) return fallback;
    const auto& field = value[key];
    if (field.is_string()) return field.get<std::string>();
    return fallback;
}

bool json_bool_field(const nlohmann::json& value, const char* key, bool fallback) {
    if (!value.is_object() || !value.contains(key)) return fallback;
    const auto& field = value[key];
    if (field.is_boolean()) return field.get<bool>();
    if (field.is_string()) {
        std::string raw = field.get<std::string>();
        std::transform(raw.begin(), raw.end(), raw.begin(), [](unsigned char ch) {
            return static_cast<char>(std::tolower(ch));
        });
        if (raw == "1" || raw == "true" || raw == "yes" || raw == "on") return true;
        if (raw == "0" || raw == "false" || raw == "no" || raw == "off") return false;
    }
    return fallback;
}

int json_int_field(const nlohmann::json& value, const char* key, int fallback, int min_value, int max_value) {
    if (!value.is_object() || !value.contains(key)) return fallback;
    const auto& field = value[key];
    try {
        if (field.is_number_integer()) return std::clamp(field.get<int>(), min_value, max_value);
        if (field.is_number()) return std::clamp(static_cast<int>(field.get<double>()), min_value, max_value);
        if (field.is_string()) return std::clamp(std::stoi(field.get<std::string>()), min_value, max_value);
    } catch (...) {
        return fallback;
    }
    return fallback;
}

double json_double_field(const nlohmann::json& value,
                         const char* key,
                         double fallback,
                         double min_value,
                         double max_value) {
    if (!value.is_object() || !value.contains(key)) return fallback;
    const auto& field = value[key];
    try {
        if (field.is_number()) return std::clamp(field.get<double>(), min_value, max_value);
        if (field.is_string()) return std::clamp(std::stod(field.get<std::string>()), min_value, max_value);
    } catch (...) {
        return fallback;
    }
    return fallback;
}

} // namespace

AiConfig default_ai_config(const BackendRuntime& runtime) {
    AiConfig config;
    config.enabled = env_bool("AI_SCAN_ENABLED", false);
    config.model = env_string("AI_SCAN_MODEL", env_string("AI_RESEARCH_MODEL", ""));
    const auto slash = config.model.find('/');
    if (slash != std::string::npos) config.provider = config.model.substr(0, slash);
    config.temperature = env_double("AI_SCAN_TEMPERATURE", 0.0, 0.0, 2.0);
    config.max_tokens = env_int("AI_SCAN_MAX_TOKENS", 8000, 1, 200000);
    config.timeout_ms = env_int("AI_SCAN_TIMEOUT_MS", 600000, 1000, 1200000);
    config.sidecar_url = env_string("AI_AGENT_URL", "http://127.0.0.1:3001");
    (void)runtime;
    return config;
}

nlohmann::json ai_config_to_json(const AiConfig& config) {
    return {
        {"enabled", config.enabled},
        {"mode", config.mode},
        {"provider", config.provider},
        {"model", config.model},
        {"temperature", config.temperature},
        {"max_tokens", config.max_tokens},
        {"timeout_ms", config.timeout_ms},
        {"send_paths", config.send_paths},
        {"persist_recommendations", config.persist_recommendations},
        {"sidecar_url", config.sidecar_url},
    };
}

AiConfig ai_config_from_json(const nlohmann::json& value, const BackendRuntime& runtime) {
    AiConfig config = default_ai_config(runtime);
    if (!value.is_object()) return config;
    config.enabled = json_bool_field(value, "enabled", config.enabled);
    config.mode = json_string_field(value, "mode", config.mode);
    config.provider = json_string_field(value, "provider", config.provider);
    config.model = json_string_field(value, "model", config.model);
    config.temperature = json_double_field(value, "temperature", config.temperature, 0.0, 2.0);
    config.max_tokens = json_int_field(value, "max_tokens", config.max_tokens, 1, 200000);
    config.timeout_ms = json_int_field(value, "timeout_ms", config.timeout_ms, 1000, 1200000);
    config.send_paths = json_bool_field(value, "send_paths", config.send_paths);
    config.persist_recommendations = json_bool_field(value, "persist_recommendations", config.persist_recommendations);
    config.sidecar_url = json_string_field(value, "sidecar_url", config.sidecar_url);
    return config;
}

nlohmann::json merge_ai_config_json(const nlohmann::json& base,
                                    const nlohmann::json& patch,
                                    const BackendRuntime& runtime) {
    nlohmann::json merged = ai_config_to_json(ai_config_from_json(base, runtime));
    for (const char* key : {"ui", "vision_overrides"}) {
        if (base.is_object() && base.contains(key)) merged[key] = base[key];
    }
    if (!patch.is_object()) return merged;
    set_if_present(merged, patch, "enabled");
    set_if_present(merged, patch, "mode");
    set_if_present(merged, patch, "provider");
    set_if_present(merged, patch, "model");
    set_if_present(merged, patch, "temperature");
    set_if_present(merged, patch, "max_tokens");
    set_if_present(merged, patch, "timeout_ms");
    set_if_present(merged, patch, "send_paths");
    set_if_present(merged, patch, "persist_recommendations");
    set_if_present(merged, patch, "sidecar_url");
    for (const char* key : {"ui", "vision_overrides"}) {
        if (patch.contains(key)) merged[key] = patch[key];
    }
    nlohmann::json normalized = ai_config_to_json(ai_config_from_json(merged, runtime));
    for (const char* key : {"ui", "vision_overrides"}) {
        if (merged.contains(key)) normalized[key] = merged[key];
    }
    return normalized;
}

AiSidecarClient::AiSidecarClient(AiConfig config) : _config(std::move(config)) {}

nlohmann::json AiSidecarClient::get(const std::string& endpoint) const {
    return request("GET", endpoint, nullptr);
}

nlohmann::json AiSidecarClient::post(const std::string& endpoint, const nlohmann::json& payload) const {
    return request("POST", endpoint, &payload);
}

nlohmann::json AiSidecarClient::del(const std::string& endpoint) const {
    return request("DELETE", endpoint, nullptr);
}

nlohmann::json AiSidecarClient::request(const std::string& method,
                                        const std::string& endpoint,
                                        const nlohmann::json* payload) const {
    CURL* curl = curl_easy_init();
    if (!curl) throw std::runtime_error("curl_easy_init failed");

    std::string response_body;
    const std::string url = join_url(_config.sidecar_url, endpoint);
    const std::string payload_text = payload ? payload->dump() : std::string();

    // Debug: log payload being sent
    std::cerr << "[AI_SIDECAR] Request to " << endpoint << ": " << payload_text.substr(0, 2000) << std::endl;

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, curl_write);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_body);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT_MS, static_cast<long>(_config.timeout_ms));

    struct curl_slist* headers = nullptr;
    if (method == "POST") {
        curl_easy_setopt(curl, CURLOPT_POST, 1L);
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, payload_text.c_str());
        headers = curl_slist_append(headers, "Content-Type: application/json");
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    } else if (method == "DELETE") {
        curl_easy_setopt(curl, CURLOPT_CUSTOMREQUEST, "DELETE");
    }

    CURLcode rc = curl_easy_perform(curl);
    long status = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &status);
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);

    if (rc != CURLE_OK) {
        throw std::runtime_error(std::string("AI sidecar unavailable: ") + curl_easy_strerror(rc));
    }

    std::cerr << "[AI_SIDECAR] Response status " << status << ": " << response_body.substr(0, 2000) << std::endl;

    auto parsed = nlohmann::json::parse(response_body, nullptr, false);
    if (parsed.is_discarded()) {
        throw std::runtime_error("AI sidecar returned invalid JSON");
    }
    parsed["_http_status"] = status;
    if (status >= 400) {
        std::string error_msg = "AI sidecar request failed";
        if (parsed.contains("message") && parsed["message"].is_string()) {
            error_msg = parsed["message"].get<std::string>();
        }
        std::cerr << "[AI_SIDECAR] Error: " << error_msg << std::endl;
        throw std::runtime_error(error_msg);
    }
    return parsed;
}

} // namespace tile_compile::ai
