#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <sstream>
#include <string>
#include <thread>
#include <chrono>

namespace fs = std::filesystem;

namespace {

std::string read_stdin() {
    std::ostringstream buffer;
    buffer << std::cin.rdbuf();
    return buffer.str();
}

}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << nlohmann::json{{"error", "missing command"}}.dump() << std::endl;
        return 2;
    }
    const std::string command = argv[1];

    if (command == "get-schema") {
        std::cout << nlohmann::json{
            {"type", "object"},
            {"properties", {
                {"data", {
                    {"type", "object"},
                    {"properties", {
                        {"color_mode", {
                            {"type", "string"},
                            {"enum", {"OSC", "MONO", "RGB"}}
                        }}
                    }}
                }}
            }}
        }.dump() << std::endl;
        return 0;
    }

    if (command == "load-config" && argc >= 3) {
        const fs::path path = argv[2];
        std::ifstream in(path);
        std::ostringstream buffer;
        buffer << in.rdbuf();
        std::cout << nlohmann::json{{"yaml", buffer.str()}, {"path", path.string()}}.dump() << std::endl;
        return 0;
    }

    if (command == "validate-config") {
        std::string text;
        for (int i = 2; i < argc; ++i) {
            if (std::string(argv[i]) == "--path" && i + 1 < argc) {
                std::ifstream in(argv[++i]);
                std::ostringstream buffer;
                buffer << in.rdbuf();
                text = buffer.str();
            } else if (std::string(argv[i]) == "--stdin") {
                text = read_stdin();
            }
        }
        const bool valid = text.find("invalid: true") == std::string::npos;
        nlohmann::json payload = {
            {"valid", valid},
            {"errors", valid ? nlohmann::json::array() : nlohmann::json::array({"fixture validation error"})},
            {"warnings", text.find("warn: true") != std::string::npos ? nlohmann::json::array({"fixture warning"}) : nlohmann::json::array()}
        };
        std::cout << payload.dump() << std::endl;
        return valid ? 0 : 2;
    }

    if (command == "save-config" && argc >= 3) {
        const fs::path path = argv[2];
        fs::create_directories(path.parent_path());
        std::ofstream out(path);
        out << read_stdin();
        std::cout << nlohmann::json{{"saved", true}, {"path", path.string()}}.dump() << std::endl;
        return 0;
    }

    if (command == "pcc-run" && argc >= 4) {
        const fs::path input_rgb = argv[2];
        const fs::path output_rgb = argv[3];
        fs::create_directories(output_rgb.parent_path());
        if (output_rgb.filename().string().find("slow") != std::string::npos) {
            std::this_thread::sleep_for(std::chrono::seconds(5));
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        std::ofstream out(output_rgb, std::ios::binary);
        out << "fake pcc output\n";
        std::cout << nlohmann::json{
            {"stars_matched", 42},
            {"stars_used", 37},
            {"residual_rms", 0.123},
            {"determinant", 0.98},
            {"condition_number", 1.11},
            {"apply_attenuation", false},
            {"chroma_strength", 1.0},
            {"k_max", 2.5},
            {"matrix", {{1.01, 0.0, 0.0}, {0.0, 0.99, 0.0}, {0.0, 0.0, 1.02}}},
            {"output_rgb", output_rgb.string()},
            {"output_channels", nlohmann::json::array()},
            {"input_rgb", input_rgb.string()}
        }.dump() << std::endl;
        return 0;
    }

    std::cout << nlohmann::json{{"error", std::string("unsupported command: ") + command}}.dump() << std::endl;
    return 2;
}
