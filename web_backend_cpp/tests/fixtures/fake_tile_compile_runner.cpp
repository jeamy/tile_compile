#include <algorithm>
#include <iostream>
#include <nlohmann/json.hpp>
#include <thread>
#include <cstdlib>

int main(int argc, char** argv) {
    int sleep_ms = 100;
    if (const char* raw = std::getenv("FAKE_TILE_COMPILE_RUNNER_SLEEP_MS")) {
        try {
            sleep_ms = std::max(0, std::stoi(raw));
        } catch (...) {}
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));
    nlohmann::json args = nlohmann::json::array();
    for (int i = 1; i < argc; ++i) args.push_back(argv[i]);
    std::cout << nlohmann::json{{"ok", true}, {"args", args}}.dump() << std::endl;
    return 0;
}
