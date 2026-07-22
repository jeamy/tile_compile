#pragma once

#include <memory>
#include <nlohmann/json.hpp>

struct AppState;

namespace tile_compile::pi {

class PiContextBuilder {
public:
    explicit PiContextBuilder(std::shared_ptr<AppState> state);

    nlohmann::json build_overview_context() const;

private:
    std::shared_ptr<AppState> _state;
};

} // namespace tile_compile::pi
