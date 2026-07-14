#pragma once

#include <memory>
#include <nlohmann/json.hpp>

struct AppState;

namespace tile_compile::pi {

class PiToolRegistry {
public:
    explicit PiToolRegistry(std::shared_ptr<AppState> state);

    nlohmann::json list_tools() const;
    nlohmann::json call_tool(const std::string& name, const nlohmann::json& input) const;

private:
    std::shared_ptr<AppState> _state;
};

} // namespace tile_compile::pi
