#pragma once

#include <nlohmann/json.hpp>

namespace tile_compile::pi {

inline constexpr const char* kAiRequestSchemaVersion = "pi.ai-request.v2";

nlohmann::json build_ai_request_v2(const nlohmann::json& input);
nlohmann::json positive_memories_from_session_context(const nlohmann::json& session_context);
nlohmann::json negative_memories_from_session_context(const nlohmann::json& session_context);

} // namespace tile_compile::pi
