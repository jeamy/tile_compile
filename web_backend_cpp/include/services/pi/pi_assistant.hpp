#pragma once

#include <memory>
#include <nlohmann/json.hpp>

struct AppState;

namespace tile_compile::pi {

class PiAssistant {
public:
    explicit PiAssistant(std::shared_ptr<AppState> state);

    nlohmann::json answer(const std::string& question) const;

private:
    std::shared_ptr<AppState> _state;
};

} // namespace tile_compile::pi
