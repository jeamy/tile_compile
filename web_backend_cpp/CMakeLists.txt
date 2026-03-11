cmake_minimum_required(VERSION 3.16)
project(tile_compile_web_backend VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

option(TILE_COMPILE_BACKEND_STATIC_STDLIB "Link libgcc/libstdc++ statically for release builds" OFF)
option(BUILD_TESTS "Build backend test executables" ON)

set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG   "-g -O0 -Wall -Wextra")

# -----------------------------------------------------------------------
# Dependencies shared with tile_compile_cpp — reuse parent variables if
# this CMakeLists is included from the parent, otherwise find them here.
# -----------------------------------------------------------------------

if(NOT DEFINED YAML_CPP_LIBRARIES)
    find_package(PkgConfig QUIET)
    find_package(yaml-cpp QUIET)
    if(yaml-cpp_FOUND)
        if(TARGET yaml-cpp::yaml-cpp)
            get_target_property(_yaml_loc yaml-cpp::yaml-cpp LOCATION)
            set(YAML_CPP_LIBRARIES ${_yaml_loc})
        else()
            set(YAML_CPP_LIBRARIES "yaml-cpp")
        endif()
    else()
        if(PkgConfig_FOUND)
            pkg_check_modules(YAML_CPP REQUIRED yaml-cpp)
        else()
            message(FATAL_ERROR "yaml-cpp not found")
        endif()
    endif()
endif()

if(NOT DEFINED NLOHMANN_JSON_INCLUDE_DIRS AND NOT TARGET nlohmann_json::nlohmann_json)
    find_package(nlohmann_json QUIET)
    if(NOT nlohmann_json_FOUND)
        find_path(NLOHMANN_JSON_INCLUDE_DIR nlohmann/json.hpp)
        if(NLOHMANN_JSON_INCLUDE_DIR)
            set(NLOHMANN_JSON_INCLUDE_DIRS ${NLOHMANN_JSON_INCLUDE_DIR})
        else()
            message(FATAL_ERROR "nlohmann/json not found")
        endif()
    endif()
endif()

find_package(CURL REQUIRED)
find_package(OpenSSL REQUIRED)
find_package(Threads REQUIRED)

# -----------------------------------------------------------------------
# Crow + Asio (header-only, fetched at configure time)
# Asio must be fetched FIRST so Crow's find_package(asio) succeeds.
# -----------------------------------------------------------------------
include(FetchContent)

FetchContent_Declare(
    asio
    GIT_REPOSITORY https://github.com/chriskohlhoff/asio.git
    GIT_TAG        asio-1-28-0
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(asio)

# Tell Crow's CMake where Asio headers live
set(ASIO_INCLUDE_DIR ${asio_SOURCE_DIR}/asio/include CACHE PATH "Asio include dir" FORCE)

FetchContent_Declare(
    crow
    GIT_REPOSITORY https://github.com/CrowCpp/Crow.git
    GIT_TAG        v1.2.0
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(crow)

# Asio standalone (no Boost)
add_library(asio_header INTERFACE)
target_include_directories(asio_header INTERFACE
    ${asio_SOURCE_DIR}/asio/include
)
target_compile_definitions(asio_header INTERFACE
    ASIO_STANDALONE
    ASIO_HAS_STD_CHRONO
)

# -----------------------------------------------------------------------
# Source files
# -----------------------------------------------------------------------
set(BACKEND_SOURCES
    src/main.cpp
    src/backend_runtime.cpp
    src/job_store.cpp
    src/subprocess_manager.cpp
    src/ui_event_store.cpp
    src/routes/system_routes.cpp
    src/routes/jobs_routes.cpp
    src/routes/app_state_routes.cpp
    src/routes/scan_routes.cpp
    src/routes/config_routes.cpp
    src/routes/runs_routes.cpp
    src/routes/ws_routes.cpp
    src/routes/tools_routes.cpp
    src/services/run_inspector.cpp
    src/services/report_generator.cpp
    src/services/scan_summary.cpp
    src/services/config_revisions.cpp
    src/services/download_manager.cpp
)

add_executable(tile_compile_web_backend ${BACKEND_SOURCES})

target_include_directories(tile_compile_web_backend PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${crow_SOURCE_DIR}/include
    ${YAML_CPP_INCLUDE_DIRS}
)

if(TARGET nlohmann_json::nlohmann_json)
    target_link_libraries(tile_compile_web_backend PRIVATE nlohmann_json::nlohmann_json)
else()
    target_include_directories(tile_compile_web_backend PRIVATE ${NLOHMANN_JSON_INCLUDE_DIRS})
endif()

target_link_libraries(tile_compile_web_backend PRIVATE
    asio_header
    ${YAML_CPP_LIBRARIES}
    CURL::libcurl
    OpenSSL::SSL
    OpenSSL::Crypto
    Threads::Threads
)

target_compile_definitions(tile_compile_web_backend PRIVATE
    CROW_ENABLE_SSL
    CROW_ENFORCE_WS_SPEC
)

if(WIN32)
    target_compile_definitions(tile_compile_web_backend PRIVATE
        _WIN32_WINNT=0x0601
        WIN32_LEAN_AND_MEAN
        NOMINMAX
    )
    target_link_libraries(tile_compile_web_backend PRIVATE
        ws2_32
        mswsock
    )
endif()

# Linux: statisch libgcc/libstdc++ optional in Release fuer maximale Portabilitaet
if(UNIX AND NOT APPLE AND CMAKE_BUILD_TYPE STREQUAL "Release" AND TILE_COMPILE_BACKEND_STATIC_STDLIB)
    target_link_options(tile_compile_web_backend PRIVATE
        -static-libgcc
        -static-libstdc++
    )
endif()

# -----------------------------------------------------------------------
# Optional: UI-Verzeichnis beim Build in das Output-Dir kopieren
# -----------------------------------------------------------------------
if(EXISTS "${CMAKE_SOURCE_DIR}/../web_frontend/dist")
    add_custom_command(TARGET tile_compile_web_backend POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_SOURCE_DIR}/../web_frontend/dist
            ${CMAKE_BINARY_DIR}/ui
        COMMENT "Copying frontend dist to build/ui"
    )
endif()

message(STATUS "")
message(STATUS "=== tile_compile_web_backend ===")
message(STATUS "Build type : ${CMAKE_BUILD_TYPE}")
message(STATUS "Crow       : ${crow_SOURCE_DIR}")
message(STATUS "Asio       : ${asio_SOURCE_DIR}")
message(STATUS "yaml-cpp   : ${YAML_CPP_LIBRARIES}")
message(STATUS "curl       : ${CURL_LIBRARIES}")
message(STATUS "")

if(BUILD_TESTS)
    enable_testing()

    add_library(web_backend_cpp_test_support STATIC
        tests/backend_test_harness.cpp
    )
    target_include_directories(web_backend_cpp_test_support PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/tests
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    if(TARGET nlohmann_json::nlohmann_json)
        target_link_libraries(web_backend_cpp_test_support PRIVATE nlohmann_json::nlohmann_json)
    else()
        target_include_directories(web_backend_cpp_test_support PRIVATE ${NLOHMANN_JSON_INCLUDE_DIRS})
    endif()
    target_link_libraries(web_backend_cpp_test_support PRIVATE CURL::libcurl Threads::Threads)

    add_executable(fake_tile_compile_cli tests/fixtures/fake_tile_compile_cli.cpp)
    target_include_directories(fake_tile_compile_cli PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
    if(TARGET nlohmann_json::nlohmann_json)
        target_link_libraries(fake_tile_compile_cli PRIVATE nlohmann_json::nlohmann_json)
    else()
        target_include_directories(fake_tile_compile_cli PRIVATE ${NLOHMANN_JSON_INCLUDE_DIRS})
    endif()

    add_executable(fake_tile_compile_runner tests/fixtures/fake_tile_compile_runner.cpp)
    target_include_directories(fake_tile_compile_runner PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
    if(TARGET nlohmann_json::nlohmann_json)
        target_link_libraries(fake_tile_compile_runner PRIVATE nlohmann_json::nlohmann_json)
    else()
        target_include_directories(fake_tile_compile_runner PRIVATE ${NLOHMANN_JSON_INCLUDE_DIRS})
    endif()

    function(add_backend_cpp_test target_name source_file)
        add_executable(${target_name} ${source_file})
        target_include_directories(${target_name} PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/tests
            ${CMAKE_CURRENT_SOURCE_DIR}/include
        )
        if(TARGET nlohmann_json::nlohmann_json)
            target_link_libraries(${target_name} PRIVATE web_backend_cpp_test_support nlohmann_json::nlohmann_json)
        else()
            target_link_libraries(${target_name} PRIVATE web_backend_cpp_test_support)
            target_include_directories(${target_name} PRIVATE ${NLOHMANN_JSON_INCLUDE_DIRS})
        endif()
        add_test(
            NAME ${target_name}
            COMMAND ${target_name}
                    $<TARGET_FILE:tile_compile_web_backend>
                    $<TARGET_FILE:fake_tile_compile_cli>
                    $<TARGET_FILE:fake_tile_compile_runner>
                    ${CMAKE_CURRENT_SOURCE_DIR}/..
        )
    endfunction()

    add_backend_cpp_test(web_backend_cpp_contract tests/test_backend_contract.cpp)
    add_backend_cpp_test(web_backend_cpp_tools_jobs tests/test_tools_api_jobs.cpp)
    add_backend_cpp_test(web_backend_cpp_runs_queue tests/test_runs_queue.cpp)
    add_backend_cpp_test(web_backend_cpp_run_status_resume_progress tests/test_run_status_resume_progress.cpp)
    add_backend_cpp_test(web_backend_cpp_run_start_naming tests/test_run_start_naming.cpp)
    add_backend_cpp_test(web_backend_cpp_run_stop_status_overlay tests/test_run_stop_status_overlay.cpp)
endif()
