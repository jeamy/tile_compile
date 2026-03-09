#pragma once

#include <string>

#include <nlohmann/json.hpp>

namespace tile_compile::gui {

nlohmann::json read_gui_constants(const std::string &project_root);

}
