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

std::string repeated(char ch, size_t count) {
    return std::string(count, ch);
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
                }},
                {"reconstruction", {
                    {"type", "object"},
                    {"properties", {
                        {"drizzle", {
                            {"type", "object"},
                            {"properties", {
                                {"internal_scale", {
                                    {"type", "integer"},
                                    {"enum", {1, 2}}
                                }}
                            }}
                        }},
                        {"clipping", {
                            {"type", "object"},
                            {"properties", {
                                {"min_fraction", {
                                    {"type", "number"},
                                    {"exclusiveMinimum", 0},
                                    {"maximum", 1}
                                }}
                            }}
                        }},
                        {"quality", {
                            {"type", "object"},
                            {"properties", {
                                {"pyramid", {
                                    {"type", "object"},
                                    {"properties", {
                                        {"base_window_px", {
                                            {"type", "integer"},
                                            {"minimum", 1},
                                            {"default", 4}
                                        }}
                                    }}
                                }}
                            }}
                        }},
                        {"diagnostics", {
                            {"type", "object"},
                            {"properties", {
                                {"preview_forward_drizzle_uniform", {
                                    {"type", "boolean"},
                                    {"default", false}
                                }}
                            }}
                        }}
                    }}
                }},
                {"pcc", {
                    {"type", "object"},
                    {"properties", {
                        {"max_residual_rms", {
                            {"type", "number"},
                            {"exclusiveMinimum", 0}
                        }},
                        {"k_max", {
                            {"type", "number"},
                            {"exclusiveMinimum", 0}
                        }}
                    }}
                }},
                {"registration", {
                    {"type", "object"},
                    {"properties", {
                        {"enable_local_background_subtraction", {
                            {"type", "boolean"}
                        }}
                    }}
                }},
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

    if (command == "scan" && argc >= 3) {
        const fs::path input_dir = argv[2];
        const std::string name = input_dir.filename().string();
        const int frame_count = name.find("many_frames") != std::string::npos ? 600 : 12;
        nlohmann::json frames = nlohmann::json::array();
        for (int i = 0; i < frame_count; ++i) {
            frames.push_back({
                {"path", (input_dir / ("frame_" + std::to_string(i) + ".fits")).string()},
                {"index", i},
                {"quality", 0.95 - (static_cast<double>(i % 10) * 0.01)}
            });
        }
        nlohmann::json payload = {
            {"ok", true},
            {"input_path", input_dir.string()},
            {"frames_detected", frame_count},
            {"image_width", 2048},
            {"image_height", 1536},
            {"color_mode", name.find("mono") != std::string::npos ? "MONO" : "OSC"},
            {"color_mode_candidates", {"OSC", "MONO", "RGB"}},
            {"bayer_pattern", name.find("mono") != std::string::npos ? nlohmann::json(nullptr) : nlohmann::json("RGGB")},
            {"requires_user_confirmation", false},
            {"errors", nlohmann::json::array()},
            {"warnings", name.find("warn") != std::string::npos ? nlohmann::json::array({"fixture warning"}) : nlohmann::json::array()},
            {"frames", frames}
        };
        std::cout << payload.dump() << std::endl;
        return 0;
    }

    if (command == "scan-metrics" && argc >= 3) {
        const fs::path input_dir = argv[2];
        const std::string name = input_dir.filename().string();
        const int frame_count = name.find("many_frames") != std::string::npos ? 600 : 12;
        nlohmann::json frames = nlohmann::json::array();
        for (int i = 0; i < frame_count; ++i) {
            frames.push_back({
                {"frame_index", i},
                {"file_name", "frame_" + std::to_string(i) + ".fits"},
                {"ok", true},
                {"header", {
                    {"exposure_seconds", 60.0},
                    {"gain", 100},
                    {"temperature_c", -10.0},
                    {"filter", "L"},
                    {"target", "FixtureTarget"},
                    {"camera", "DWARF II"},
                    {"date_obs", "2026-09-22T21:00:00"},
                    {"ra_deg", 300.0},
                    {"dec_deg", 44.0}
                }},
                {"background", 500.0 + i * 2.0},
                {"noise", 8.0 + i * 0.05},
                {"gradient_energy", 0.012},
                {"sky_gradient", 0.030},
                {"fwhm", 2.8 + i * 0.03},
                {"roundness", 1.02},
                {"star_count", 200 + i}
            });
        }
        nlohmann::json payload = {
            {"ok", true},
            {"input_path", input_dir.string()},
            {"frames_total", frame_count},
            {"frames_ok", frame_count},
            {"frames_failed", 0},
            {"frames", frames},
            {"aggregate", {
                {"background", {{"min", 500.0}, {"max", 522.0}, {"mean", 511.0}, {"median", 511.0}, {"p10", 502.0}, {"p90", 520.0}, {"count", frame_count}}},
                {"noise", {{"min", 8.0}, {"max", 8.55}, {"mean", 8.275}, {"median", 8.275}, {"p10", 8.05}, {"p90", 8.50}, {"count", frame_count}}},
                {"gradient_energy", {{"min", 0.012}, {"max", 0.012}, {"mean", 0.012}, {"median", 0.012}, {"p10", 0.012}, {"p90", 0.012}, {"count", frame_count}}},
                {"sky_gradient", {{"min", 0.030}, {"max", 0.030}, {"mean", 0.030}, {"median", 0.030}, {"p10", 0.030}, {"p90", 0.030}, {"count", frame_count}}},
                {"fwhm", {{"min", 2.8}, {"max", 3.13}, {"mean", 2.965}, {"median", 2.965}, {"p10", 2.83}, {"p90", 3.10}, {"count", frame_count}}},
                {"roundness", {{"min", 1.02}, {"max", 1.02}, {"mean", 1.02}, {"median", 1.02}, {"p10", 1.02}, {"p90", 1.02}, {"count", frame_count}}},
                {"star_count", {{"min", 200}, {"max", 211}, {"mean", 205.5}, {"median", 205.5}, {"count", frame_count}}},
                {"frames_ok", frame_count},
                {"frames_failed", 0}
            }}
        };
        std::cout << payload.dump() << std::endl;
        return 0;
    }

    if (command == "pcc-run" && argc >= 4) {
        const fs::path input_rgb = argv[2];
        const fs::path output_rgb = argv[3];
        const std::string output_name = output_rgb.filename().string();
        fs::create_directories(output_rgb.parent_path());
        if (output_name.find("slow") != std::string::npos) {
            std::this_thread::sleep_for(std::chrono::seconds(5));
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        if (output_name.find("loud") != std::string::npos) {
            std::cerr << repeated('e', 300000);
        }
        std::ofstream out(output_rgb, std::ios::binary);
        out << "fake pcc output\n";
        nlohmann::json payload = {
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
        };
        if (output_name.find("widejson") != std::string::npos) {
            payload["debug_blob"] = repeated('x', 200000);
        }
        std::cout << payload.dump() << std::endl;
        return 0;
    }

    std::cout << nlohmann::json{{"error", std::string("unsupported command: ") + command}}.dump() << std::endl;
    return 2;
}
