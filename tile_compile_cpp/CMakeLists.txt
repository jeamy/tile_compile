cmake_minimum_required(VERSION 3.16)
project(tile_compile VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra -Wpedantic")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# Find dependencies
find_package(PkgConfig REQUIRED)

# Eigen3 - try find_package first, fallback to pkg-config
find_package(Eigen3 QUIET)
if(NOT Eigen3_FOUND)
    pkg_check_modules(EIGEN3 REQUIRED eigen3)
    set(EIGEN3_INCLUDE_DIR ${EIGEN3_INCLUDE_DIRS})
endif()

# OpenCV
find_package(OpenCV REQUIRED)

pkg_check_modules(CFITSIO REQUIRED cfitsio)
pkg_check_modules(YAML_CPP REQUIRED yaml-cpp)

# nlohmann/json - try find_package first, fallback to pkg-config
find_package(nlohmann_json QUIET)
if(NOT nlohmann_json_FOUND)
    pkg_check_modules(NLOHMANN_JSON REQUIRED nlohmann_json)
endif()

# OpenSSL for SHA256
find_package(OpenSSL REQUIRED)

# Qt6 Core + Network for lib (HTTP queries), Widgets for GUI
find_package(Qt6 REQUIRED COMPONENTS Core Network Widgets)

# Optional: spdlog for logging
find_package(spdlog QUIET)

# Optional: CLI11 for argument parsing
find_package(CLI11 QUIET)

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${EIGEN3_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${CFITSIO_INCLUDE_DIRS}
    ${YAML_CPP_INCLUDE_DIRS}
)

if(NLOHMANN_JSON_FOUND)
    include_directories(${NLOHMANN_JSON_INCLUDE_DIRS})
endif()

# Library sources
set(LIB_SOURCES
    src/core/mode_gating.cpp
    src/core/utils.cpp
    src/core/events.cpp
    src/io/fits_io.cpp
    src/io/config.cpp
    src/image/processing.cpp
    src/image/cfa_processing.cpp
    src/image/normalization.cpp
    src/registration/registration.cpp
    src/registration/global_registration.cpp
    src/metrics/metrics.cpp
    src/metrics/tile_metrics.cpp
    src/metrics/linearity.cpp
    src/reconstruction/reconstruction.cpp
    src/pipeline/adaptive_tile_grid.cpp
    src/pipeline/pipeline.cpp
    src/astrometry/wcs.cpp
    src/astrometry/gaia_catalog.cpp
    src/astrometry/photometric_color_cal.cpp
)

# Main library
add_library(tile_compile_lib STATIC ${LIB_SOURCES})

target_link_libraries(tile_compile_lib PUBLIC
    Eigen3::Eigen
    ${OpenCV_LIBS}
    ${CFITSIO_LIBRARIES}
    ${YAML_CPP_LIBRARIES}
    OpenSSL::Crypto
    Qt6::Core
    Qt6::Network
)

if(nlohmann_json_FOUND)
    target_link_libraries(tile_compile_lib PUBLIC nlohmann_json::nlohmann_json)
else()
    target_link_libraries(tile_compile_lib PUBLIC ${NLOHMANN_JSON_LIBRARIES})
endif()

if(spdlog_FOUND)
    target_link_libraries(tile_compile_lib PUBLIC spdlog::spdlog)
    target_compile_definitions(tile_compile_lib PUBLIC HAVE_SPDLOG)
endif()

# CLI Runner executable
add_executable(tile_compile_runner apps/runner_main.cpp)
target_link_libraries(tile_compile_runner PRIVATE tile_compile_lib)

if(CLI11_FOUND)
    target_link_libraries(tile_compile_runner PRIVATE CLI11::CLI11)
    target_compile_definitions(tile_compile_runner PRIVATE HAVE_CLI11)
endif()

# Backend CLI executable
add_executable(tile_compile_cli apps/cli_main.cpp)
target_link_libraries(tile_compile_cli PRIVATE tile_compile_lib)

if(CLI11_FOUND)
    target_link_libraries(tile_compile_cli PRIVATE CLI11::CLI11)
    target_compile_definitions(tile_compile_cli PRIVATE HAVE_CLI11)
endif()

# Qt6 GUI executable
set(GUI_SOURCES
    gui_cpp/main.cpp
    gui_cpp/MainWindow.cpp
    gui_cpp/PhaseProgressWidget.cpp
    gui_cpp/AssumptionsWidget.cpp
    gui_cpp/BackendClient.cpp
    gui_cpp/RunnerProcess.cpp
    gui_cpp/GuiConstants.cpp
    gui_cpp/tabs/ScanTab.cpp
    gui_cpp/tabs/ConfigTab.cpp
    gui_cpp/tabs/RunTab.cpp
    gui_cpp/tabs/CurrentRunTab.cpp
    gui_cpp/tabs/HistoryTab.cpp
    gui_cpp/tabs/AstrometryTab.cpp
    gui_cpp/tabs/PCCTab.cpp
)

add_executable(tile_compile_gui ${GUI_SOURCES})
target_include_directories(tile_compile_gui PRIVATE gui_cpp)
target_link_libraries(tile_compile_gui PRIVATE tile_compile_lib Qt6::Widgets Qt6::Network)

set_target_properties(tile_compile_gui PROPERTIES
    AUTOMOC_PATH_PREFIX OFF
    AUTOMOC_MOC_OPTIONS "-p;${CMAKE_SOURCE_DIR}/gui_cpp/"
)

# Copy GUI resource files to build directory
configure_file(${CMAKE_SOURCE_DIR}/gui_cpp/constants.js ${CMAKE_BINARY_DIR}/gui_cpp/constants.js COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/gui_cpp/styles.qss ${CMAKE_BINARY_DIR}/gui_cpp/styles.qss COPYONLY)

# Also copy to same directory as executable for release
add_custom_command(TARGET tile_compile_gui POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_SOURCE_DIR}/gui_cpp/constants.js
        $<TARGET_FILE_DIR:tile_compile_gui>/gui_cpp/constants.js
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_SOURCE_DIR}/gui_cpp/styles.qss
        $<TARGET_FILE_DIR:tile_compile_gui>/gui_cpp/styles.qss
)

# Copy config/schema files to build directory on every build
add_custom_command(TARGET tile_compile_runner POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_SOURCE_DIR}/tile_compile.yaml
        ${CMAKE_BINARY_DIR}/tile_compile.yaml
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_SOURCE_DIR}/tile_compile.schema.yaml
        ${CMAKE_BINARY_DIR}/tile_compile.schema.yaml
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_SOURCE_DIR}/tile_compile.schema.json
        ${CMAKE_BINARY_DIR}/tile_compile.schema.json
    COMMENT "Copying config/schema files to build directory"
)

# Tests (optional)
option(BUILD_TESTS "Build unit tests" ON)
if(BUILD_TESTS)
    find_package(Catch2 QUIET)
    if(Catch2_FOUND)
        enable_testing()
        add_executable(tests
            tests/test_main.cpp
            tests/test_utils.cpp
            tests/test_fits.cpp
            tests/test_debayer.cpp
            tests/test_mode_gating.cpp
            tests/test_reconstruction_regression.cpp
        )
        target_link_libraries(tests PRIVATE tile_compile_lib Catch2::Catch2WithMain)
        include(CTest)
        include(Catch)
        catch_discover_tests(tests)
    endif()
endif()

# Install targets
install(TARGETS tile_compile_runner tile_compile_cli tile_compile_gui
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/tile_compile
    DESTINATION include
)

# Print configuration summary
message(STATUS "")
message(STATUS "=== Tile-Compile C++ Configuration ===")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Eigen3: ${EIGEN3_VERSION}")
message(STATUS "OpenCV: ${OpenCV_VERSION}")
message(STATUS "CFITSIO: ${CFITSIO_VERSION}")
message(STATUS "yaml-cpp: found")
message(STATUS "nlohmann/json: ${nlohmann_json_FOUND}")
message(STATUS "spdlog: ${spdlog_FOUND}")
message(STATUS "CLI11: ${CLI11_FOUND}")
message(STATUS "Catch2: ${Catch2_FOUND}")
message(STATUS "")
