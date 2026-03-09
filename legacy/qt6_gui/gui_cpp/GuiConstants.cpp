#include "GuiConstants.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>

namespace tile_compile::gui {

namespace {
std::string read_text_file(const std::string &path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("cannot read " + path);
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}
}

nlohmann::json read_gui_constants(const std::string &project_root) {
    const std::string js_path = project_root + "/gui_cpp/constants.js";
    const std::string raw = read_text_file(js_path);
    const std::string key = "GUI_CONSTANTS_JSON";
    const auto key_pos = raw.find(key);
    if (key_pos == std::string::npos) {
        throw std::runtime_error("GUI_CONSTANTS_JSON not found in gui_cpp/constants.js");
    }
    const auto start = raw.find('`', key_pos);
    if (start == std::string::npos) {
        throw std::runtime_error("failed to parse gui_cpp/constants.js (start)");
    }
    const auto end = raw.find('`', start + 1);
    if (end == std::string::npos) {
        throw std::runtime_error("failed to parse gui_cpp/constants.js (end)");
    }
    const std::string payload = raw.substr(start + 1, end - start - 1);
    return nlohmann::json::parse(payload);
}

}
