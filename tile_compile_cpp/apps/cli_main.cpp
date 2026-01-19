#include "tile_compile/core/types.hpp"
#include "tile_compile/config/configuration.hpp"

#include <iostream>
#include <string>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: tile_compile_cli <command> [options]\n"
                  << "Commands: get-schema, load-config, validate-config, scan, list-runs\n";
        return 1;
    }
    
    std::string command = argv[1];
    
    if (command == "get-schema") {
        std::cout << tile_compile::config::get_schema_json() << std::endl;
        return 0;
    }
    
    std::cerr << "Unknown command: " << command << std::endl;
    return 1;
}
