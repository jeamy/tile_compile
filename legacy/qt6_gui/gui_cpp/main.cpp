#include <QApplication>
#include <QFile>
#include <QTextStream>
#include <iostream>
#include <filesystem>

#include "MainWindow.hpp"

namespace fs = std::filesystem;

std::string get_executable_dir(const char *argv0) {
    fs::path exe_path = fs::absolute(argv0);
    return exe_path.parent_path().string();
}

std::string resolve_project_root(const std::string &exe_dir) {
    // First check if gui_cpp/constants.js exists relative to executable
    fs::path p = fs::path(exe_dir);
    if (fs::exists(p / "gui_cpp" / "constants.js")) {
        return p.string();
    }
    
    // Fallback: search upward from current working directory
    p = fs::current_path();
    while (true) {
        if (fs::exists(p / "gui_cpp" / "constants.js")) {
            return p.string();
        }
        if (fs::exists(p / "tile_compile_cpp" / "gui_cpp" / "constants.js")) {
            return (p / "tile_compile_cpp").string();
        }
        
        if (p.parent_path() == p) {
            // Last resort: use executable directory
            return exe_dir;
        }
        p = p.parent_path();
    }
}

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);
    
    const std::string exe_dir = get_executable_dir(argv[0]);
    const std::string project_root = resolve_project_root(exe_dir);
    
    app.setStyle("Fusion");
    
    tile_compile::gui::MainWindow win(project_root);
    win.show();
    
    return app.exec();
}
