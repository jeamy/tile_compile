#include "services/pi/pi_ai_request_builder.hpp"

namespace tile_compile::pi {
namespace {

nlohmann::json object_or_empty(const nlohmann::json& value) {
    return value.is_object() ? value : nlohmann::json::object();
}

nlohmann::json array_or_empty(const nlohmann::json& value) {
    return value.is_array() ? value : nlohmann::json::array();
}

nlohmann::json value_or(const nlohmann::json& input,
                        const char* key,
                        const nlohmann::json& fallback) {
    if (!input.is_object() || !input.contains(key)) return fallback;
    return input[key];
}

} // namespace

nlohmann::json positive_memories_from_session_context(const nlohmann::json& session_context) {
    if (!session_context.is_object() || !session_context.contains("accepted_pi_memories")) {
        return nlohmann::json::array();
    }
    return array_or_empty(session_context["accepted_pi_memories"]);
}

nlohmann::json negative_memories_from_session_context(const nlohmann::json& session_context) {
    if (!session_context.is_object() || !session_context.contains("negative_pi_memories")) {
        return nlohmann::json::array();
    }
    return array_or_empty(session_context["negative_pi_memories"]);
}

nlohmann::json build_ai_request_v2(const nlohmann::json& input) {
    const nlohmann::json session_context = object_or_empty(value_or(input, "session_context", nlohmann::json::object()));
    nlohmann::json positive_memories = array_or_empty(value_or(input, "positive_memories", nlohmann::json::array()));
    nlohmann::json negative_memories = array_or_empty(value_or(input, "negative_memories", nlohmann::json::array()));
    if (positive_memories.empty()) positive_memories = positive_memories_from_session_context(session_context);
    if (negative_memories.empty()) negative_memories = negative_memories_from_session_context(session_context);

    nlohmann::json request = {
        {"schema_version", kAiRequestSchemaVersion},
        {"task", value_or(input, "task", "analysis")},
        {"user_message", value_or(input, "user_message", "")},
        {"context_signature", object_or_empty(value_or(input, "context_signature", nlohmann::json::object()))},
        {"scan_context", object_or_empty(value_or(input, "scan_context", nlohmann::json::object()))},
        {"run_context", object_or_empty(value_or(input, "run_context", nlohmann::json::object()))},
        {"artifacts", value_or(input, "artifacts", nlohmann::json::object())},
        {"image_context", object_or_empty(value_or(input, "image_context", nlohmann::json::object()))},
        {"config", object_or_empty(value_or(input, "config", nlohmann::json::object()))},
        {"allowed_config_paths", array_or_empty(value_or(input, "allowed_config_paths", nlohmann::json::array()))},
        {"positive_memories", positive_memories},
        {"negative_memories", negative_memories},
        {"conversation", array_or_empty(value_or(input, "conversation", nlohmann::json::array()))},
        {"expected_response", value_or(input, "expected_response", "structured_parameter_recommendations")}
    };

    if (!session_context.empty()) request["session_context"] = session_context;
    if (input.contains("provider")) request["provider"] = input["provider"];
    if (input.contains("model")) request["model"] = input["model"];
    if (input.contains("source_request_schema")) request["source_request_schema"] = input["source_request_schema"];
    return request;
}

} // namespace tile_compile::pi
