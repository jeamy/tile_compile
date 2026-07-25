#pragma once

#include "backend_runtime.hpp"
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>

namespace tile_compile::ai {

struct AiConfig {
    bool enabled{false};
    std::string mode{"manual"};
    std::string provider;
    std::string model;
    double temperature{0.0};
    int max_tokens{8000};
    int timeout_ms{600000};
    bool send_paths{false};
    bool persist_recommendations{false};
    std::string sidecar_url{"http://127.0.0.1:3001"};
};

AiConfig default_ai_config();
nlohmann::json ai_config_to_json(const AiConfig& config);
nlohmann::json redact_ai_payload_for_log(const nlohmann::json& payload);
AiConfig ai_config_from_json(const nlohmann::json& value);
nlohmann::json merge_ai_config_json(const nlohmann::json& base,
                                    const nlohmann::json& patch);

class AiSidecarHttpError : public std::runtime_error {
public:
    AiSidecarHttpError(long status, nlohmann::json payload, const std::string& message);

    long status() const noexcept;
    const nlohmann::json& payload() const noexcept;

private:
    long _status;
    nlohmann::json _payload;
};

class AiSidecarClient {
public:
    explicit AiSidecarClient(AiConfig config);

    nlohmann::json get(const std::string& endpoint) const;
    nlohmann::json post(const std::string& endpoint, const nlohmann::json& payload) const;
    nlohmann::json del(const std::string& endpoint) const;

private:
    nlohmann::json request(const std::string& method,
                           const std::string& endpoint,
                           const nlohmann::json* payload) const;

    AiConfig _config;
};

} // namespace tile_compile::ai
