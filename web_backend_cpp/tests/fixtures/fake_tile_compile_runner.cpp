#include <iostream>
#include <nlohmann/json.hpp>
#include <thread>

int main(int argc, char** argv) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    nlohmann::json args = nlohmann::json::array();
    for (int i = 1; i < argc; ++i) args.push_back(argv[i]);
    std::cout << nlohmann::json{{"ok", true}, {"args", args}}.dump() << std::endl;
    return 0;
}
