cmake_minimum_required(VERSION 3.16)
project(tile_compile VERSION 1.0.0 LANGUAGES CXX)

include(CheckIncludeFileCXX)

if(POLICY CMP0146)
    cmake_policy(SET CMP0146 OLD)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(_tile_compile_opencv_dir_candidates
    "$ENV{TILE_COMPILE_OPENCV_DIR}"
    "$ENV{OpenCV_DIR}"
    "$ENV{HOME}/.local/opencv-cuda-4.11/lib64/cmake/opencv4"
    "$ENV{HOME}/.local/opencv-cuda/lib64/cmake/opencv4")
set(_tile_compile_preferred_opencv_dir "")
foreach(_opencv_dir ${_tile_compile_opencv_dir_candidates})
    if(_opencv_dir AND EXISTS "${_opencv_dir}/OpenCVConfig.cmake")
        set(_tile_compile_preferred_opencv_dir "${_opencv_dir}")
        break()
    endif()
endforeach()
if((NOT DEFINED OpenCV_DIR OR "${OpenCV_DIR}" STREQUAL "") OR
   (DEFINED TILE_COMPILE_OPENCV_DIR_AUTO_LAST AND
    "${OpenCV_DIR}" STREQUAL "${TILE_COMPILE_OPENCV_DIR_AUTO_LAST}"))
    if(_tile_compile_preferred_opencv_dir)
        set(OpenCV_DIR "${_tile_compile_preferred_opencv_dir}" CACHE PATH
            "Preferred OpenCV config directory for tile_compile" FORCE)
    endif()
endif()
set(TILE_COMPILE_OPENCV_DIR_AUTO_LAST "${_tile_compile_preferred_opencv_dir}"
    CACHE INTERNAL "Last auto-detected preferred OpenCV_DIR")

set(_tile_compile_cuda_hints)
if(DEFINED ENV{CUDA_NVCC_EXECUTABLE} AND NOT "$ENV{CUDA_NVCC_EXECUTABLE}" STREQUAL "")
    get_filename_component(_tile_compile_env_nvcc_dir "$ENV{CUDA_NVCC_EXECUTABLE}" DIRECTORY)
    if(_tile_compile_env_nvcc_dir)
        list(APPEND _tile_compile_cuda_hints "${_tile_compile_env_nvcc_dir}")
    endif()
endif()
foreach(_cuda_root "/usr/local/cuda-12.9"
                   "$ENV{CUDA_TOOLKIT_ROOT_DIR}" "$ENV{CUDA_HOME}" "$ENV{CUDA_PATH}"
                   "/usr/local/cuda" "/opt/cuda")
    if(_cuda_root)
        list(APPEND _tile_compile_cuda_hints "${_cuda_root}" "${_cuda_root}/bin")
    endif()
endforeach()
set(_tile_compile_preferred_nvcc "")
foreach(_cuda_hint ${_tile_compile_cuda_hints})
    if(_cuda_hint AND EXISTS "${_cuda_hint}/nvcc")
        set(_tile_compile_preferred_nvcc "${_cuda_hint}/nvcc")
        break()
    endif()
endforeach()
if((NOT DEFINED TILE_COMPILE_NVCC_EXECUTABLE OR
    "${TILE_COMPILE_NVCC_EXECUTABLE}" STREQUAL "") OR
   (DEFINED TILE_COMPILE_NVCC_EXECUTABLE_AUTO_LAST AND
    "${TILE_COMPILE_NVCC_EXECUTABLE}" STREQUAL "${TILE_COMPILE_NVCC_EXECUTABLE_AUTO_LAST}"))
    if(_tile_compile_preferred_nvcc)
        set(TILE_COMPILE_NVCC_EXECUTABLE "${_tile_compile_preferred_nvcc}"
            CACHE FILEPATH "Preferred nvcc executable for tile_compile" FORCE)
    else()
        find_program(TILE_COMPILE_NVCC_EXECUTABLE
                     NAMES nvcc
                     HINTS ${_tile_compile_cuda_hints})
    endif()
endif()
set(TILE_COMPILE_NVCC_EXECUTABLE_AUTO_LAST "${_tile_compile_preferred_nvcc}"
    CACHE INTERNAL "Last auto-detected preferred nvcc executable")
set(_tile_compile_cuda_auto_default OFF)
if(TILE_COMPILE_NVCC_EXECUTABLE)
    set(_tile_compile_cuda_auto_default ON)
endif()
set(_tile_compile_cuda_help
    "Enable experimental custom CUDA backend hooks (auto-defaults to ON when a CUDA compiler is available)")
if(NOT DEFINED TILE_COMPILE_ENABLE_CUDA)
    set(TILE_COMPILE_ENABLE_CUDA ${_tile_compile_cuda_auto_default}
        CACHE BOOL "${_tile_compile_cuda_help}")
elseif(DEFINED TILE_COMPILE_ENABLE_CUDA_AUTO_LAST AND
       "${TILE_COMPILE_ENABLE_CUDA}" STREQUAL "${TILE_COMPILE_ENABLE_CUDA_AUTO_LAST}")
    set(TILE_COMPILE_ENABLE_CUDA ${_tile_compile_cuda_auto_default}
        CACHE BOOL "${_tile_compile_cuda_help}" FORCE)
else()
    set_property(CACHE TILE_COMPILE_ENABLE_CUDA PROPERTY HELPSTRING
                 "${_tile_compile_cuda_help}")
endif()
set(TILE_COMPILE_ENABLE_CUDA_AUTO_LAST "${_tile_compile_cuda_auto_default}"
    CACHE INTERNAL "Last auto-detected CUDA default")
set(TILE_COMPILE_WITH_CUDA OFF)
if(TILE_COMPILE_ENABLE_CUDA)
    # There are currently no .cu translation units in the project. The custom
    # CUDA backend hook is therefore a feature gate, not a separate CUDA build
    # pipeline, and should not depend on CMake successfully enabling the CUDA
    # language/compiler-id checks on this host.
    if(TILE_COMPILE_NVCC_EXECUTABLE)
        set(TILE_COMPILE_WITH_CUDA ON)
    else()
        message(WARNING "TILE_COMPILE_ENABLE_CUDA=ON but nvcc was not found; custom CUDA backend hooks will be compiled out.")
    endif()
endif()

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra -Wpedantic")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# Find dependencies
find_package(PkgConfig QUIET)
find_package(OpenMP)

# Eigen3 - try find_package first, fallback to pkg-config
find_package(Eigen3 QUIET)
if(NOT Eigen3_FOUND)
    if(PkgConfig_FOUND)
        pkg_check_modules(EIGEN3 REQUIRED eigen3)
        set(EIGEN3_INCLUDE_DIR ${EIGEN3_INCLUDE_DIRS})
    else()
        message(FATAL_ERROR "Eigen3 nicht gefunden und pkg-config ist nicht verfuegbar. Bitte Eigen3 installieren oder CMAKE_PREFIX_PATH entsprechend setzen.")
    endif()
endif()

# OpenCV
find_package(OpenCV QUIET)
if(NOT OpenCV_FOUND AND WIN32)
    foreach(_opencv_prefix
        "C:/msys64/mingw64"
        "C:/msys64/ucrt64"
        "C:/msys64/clang64"
        "C:/tools/vcpkg/installed/x64-windows"
        "C:/tools/vcpkg/installed/x64-mingw-dynamic"
        "C:/tools/vcpkg/installed/x64-mingw-static"
    )
        if(EXISTS "${_opencv_prefix}/lib/cmake/opencv4/OpenCVConfig.cmake")
            set(OpenCV_DIR "${_opencv_prefix}/lib/cmake/opencv4")
            find_package(OpenCV QUIET)
            if(OpenCV_FOUND)
                break()
            endif()
        endif()
    endforeach()
endif()

if(NOT OpenCV_FOUND)
    if(PkgConfig_FOUND)
        pkg_check_modules(OPENCV4 QUIET opencv4)
    else()
        set(OPENCV4_FOUND FALSE)
    endif()

    if(OPENCV4_FOUND)
        set(OpenCV_INCLUDE_DIRS ${OPENCV4_INCLUDE_DIRS})
        set(OpenCV_LIBS ${OPENCV4_LIBRARIES})
        set(OpenCV_VERSION ${OPENCV4_VERSION})
    else()
        if(PkgConfig_FOUND)
            pkg_check_modules(OPENCV QUIET opencv)
        else()
            set(OPENCV_FOUND FALSE)
        endif()

        if(OPENCV_FOUND)
            set(OpenCV_INCLUDE_DIRS ${OPENCV_INCLUDE_DIRS})
            set(OpenCV_LIBS ${OPENCV_LIBRARIES})
            set(OpenCV_VERSION ${OPENCV_VERSION})
        else()
            # Fallback for non-pkg-config setups (e.g. custom installs)
            find_path(OPENCV_INCLUDE_DIR opencv2/opencv.hpp
                PATHS
                    /opt/local /usr/local /opt/homebrew
                    C:/msys64/mingw64 C:/msys64/ucrt64 C:/msys64/clang64
                    C:/tools/vcpkg/installed/x64-windows
                    C:/tools/vcpkg/installed/x64-mingw-dynamic
                    C:/tools/vcpkg/installed/x64-mingw-static
                PATH_SUFFIXES include/opencv4 include
            )
            find_library(OPENCV_CORE_LIBRARY NAMES opencv_core opencv4_core
                PATHS
                    /opt/local/lib /usr/local/lib /opt/homebrew/lib
                    C:/msys64/mingw64/lib C:/msys64/ucrt64/lib C:/msys64/clang64/lib
                    C:/tools/vcpkg/installed/x64-windows/lib
                    C:/tools/vcpkg/installed/x64-mingw-dynamic/lib
                    C:/tools/vcpkg/installed/x64-mingw-static/lib
            )
            find_library(OPENCV_IMGPROC_LIBRARY NAMES opencv_imgproc opencv4_imgproc
                PATHS
                    /opt/local/lib /usr/local/lib /opt/homebrew/lib
                    C:/msys64/mingw64/lib C:/msys64/ucrt64/lib C:/msys64/clang64/lib
                    C:/tools/vcpkg/installed/x64-windows/lib
                    C:/tools/vcpkg/installed/x64-mingw-dynamic/lib
                    C:/tools/vcpkg/installed/x64-mingw-static/lib
            )
            find_library(OPENCV_IMGCODECS_LIBRARY NAMES opencv_imgcodecs opencv4_imgcodecs
                PATHS
                    /opt/local/lib /usr/local/lib /opt/homebrew/lib
                    C:/msys64/mingw64/lib C:/msys64/ucrt64/lib C:/msys64/clang64/lib
                    C:/tools/vcpkg/installed/x64-windows/lib
                    C:/tools/vcpkg/installed/x64-mingw-dynamic/lib
                    C:/tools/vcpkg/installed/x64-mingw-static/lib
            )
            find_library(OPENCV_VIDEOIO_LIBRARY NAMES opencv_videoio opencv4_videoio
                PATHS
                    /opt/local/lib /usr/local/lib /opt/homebrew/lib
                    C:/msys64/mingw64/lib C:/msys64/ucrt64/lib C:/msys64/clang64/lib
                    C:/tools/vcpkg/installed/x64-windows/lib
                    C:/tools/vcpkg/installed/x64-mingw-dynamic/lib
                    C:/tools/vcpkg/installed/x64-mingw-static/lib
            )

            if(OPENCV_INCLUDE_DIR AND OPENCV_CORE_LIBRARY)
                set(OpenCV_INCLUDE_DIRS ${OPENCV_INCLUDE_DIR})
                set(OpenCV_LIBS ${OPENCV_CORE_LIBRARY})
                if(OPENCV_IMGPROC_LIBRARY)
                    list(APPEND OpenCV_LIBS ${OPENCV_IMGPROC_LIBRARY})
                endif()
                if(OPENCV_IMGCODECS_LIBRARY)
                    list(APPEND OpenCV_LIBS ${OPENCV_IMGCODECS_LIBRARY})
                endif()
                if(OPENCV_VIDEOIO_LIBRARY)
                    list(APPEND OpenCV_LIBS ${OPENCV_VIDEOIO_LIBRARY})
                endif()
                set(OpenCV_VERSION "manual")
            else()
                message(FATAL_ERROR "OpenCV nicht gefunden. Setze OpenCV_DIR oder CMAKE_PREFIX_PATH auf die OpenCV-Installation (oder installiere OpenCV inkl. Dev-Dateien).")
            endif()
        endif()
    endif()
endif()

find_path(CFITSIO_INCLUDE_DIR fitsio.h)
find_library(CFITSIO_LIBRARY NAMES cfitsio)
if(CFITSIO_INCLUDE_DIR AND CFITSIO_LIBRARY)
    set(CFITSIO_INCLUDE_DIRS ${CFITSIO_INCLUDE_DIR})
    set(CFITSIO_LIBRARIES ${CFITSIO_LIBRARY})
else()
    if(PkgConfig_FOUND)
        pkg_check_modules(CFITSIO QUIET cfitsio)
    endif()
    
    if(NOT CFITSIO_FOUND)
        # Manual fallback for MSYS2/MinGW installations
        set(_cfitsio_search_paths
            "C:/msys64/mingw64"
            "C:/msys64/ucrt64"
            "C:/msys64/clang64"
            "/usr/local"
            "/usr"
        )
        foreach(_prefix ${_cfitsio_search_paths})
            if(EXISTS "${_prefix}/include/fitsio.h")
                find_path(CFITSIO_INCLUDE_DIR fitsio.h PATHS "${_prefix}/include" NO_DEFAULT_PATH)
                find_library(CFITSIO_LIBRARY NAMES cfitsio PATHS "${_prefix}/lib" NO_DEFAULT_PATH)
                if(CFITSIO_INCLUDE_DIR AND CFITSIO_LIBRARY)
                    set(CFITSIO_INCLUDE_DIRS ${CFITSIO_INCLUDE_DIR})
                    set(CFITSIO_LIBRARIES ${CFITSIO_LIBRARY})
                    set(CFITSIO_FOUND TRUE)
                    break()
                endif()
            endif()
        endforeach()
        
        if(NOT CFITSIO_FOUND)
            message(FATAL_ERROR "CFITSIO nicht gefunden. Bitte CFITSIO installieren oder CMAKE_PREFIX_PATH setzen.")
        endif()
    endif()
endif()

find_package(yaml-cpp QUIET)
if(yaml-cpp_FOUND)
    # Nicht yaml-cpp::yaml-cpp verwenden - kann defekt sein in Ubuntu 20.04
    # Stattdessen direkt die Bibliothek verlinken
    if(TARGET yaml-cpp::yaml-cpp)
        get_target_property(_yaml_location yaml-cpp::yaml-cpp LOCATION)
        if(_yaml_location)
            set(YAML_CPP_LIBRARIES ${_yaml_location})
        else()
            set(YAML_CPP_LIBRARIES "yaml-cpp")
        endif()
    else()
        set(YAML_CPP_LIBRARIES "yaml-cpp")
    endif()
else()
    if(PkgConfig_FOUND)
        pkg_check_modules(YAML_CPP QUIET yaml-cpp)
    endif()
    
    if(NOT YAML_CPP_FOUND)
        # Manual fallback for MSYS2/MinGW installations
        set(_yamlcpp_search_paths
            "C:/msys64/mingw64"
            "C:/msys64/ucrt64"
            "C:/msys64/clang64"
            "/usr/local"
            "/usr"
        )
        foreach(_prefix ${_yamlcpp_search_paths})
            if(EXISTS "${_prefix}/include/yaml-cpp/yaml.h")
                find_path(YAML_CPP_INCLUDE_DIR yaml-cpp/yaml.h PATHS "${_prefix}/include" NO_DEFAULT_PATH)
                find_library(YAML_CPP_LIBRARY NAMES yaml-cpp PATHS "${_prefix}/lib" NO_DEFAULT_PATH)
                if(YAML_CPP_INCLUDE_DIR AND YAML_CPP_LIBRARY)
                    set(YAML_CPP_INCLUDE_DIRS ${YAML_CPP_INCLUDE_DIR})
                    set(YAML_CPP_LIBRARIES ${YAML_CPP_LIBRARY})
                    set(YAML_CPP_FOUND TRUE)
                    break()
                endif()
            endif()
        endforeach()
        
        if(NOT YAML_CPP_FOUND)
            message(FATAL_ERROR "yaml-cpp nicht gefunden. Bitte yaml-cpp installieren oder CMAKE_PREFIX_PATH setzen.")
        endif()
    endif()
endif()

# nlohmann/json - try find_package first, fallback to pkg-config
find_package(nlohmann_json QUIET)
if(NOT nlohmann_json_FOUND)
    if(PkgConfig_FOUND)
        pkg_check_modules(NLOHMANN_JSON QUIET nlohmann_json)
    endif()
    
    if(NOT NLOHMANN_JSON_FOUND)
        find_path(NLOHMANN_JSON_INCLUDE_DIR nlohmann/json.hpp)
        if(NLOHMANN_JSON_INCLUDE_DIR)
            set(NLOHMANN_JSON_INCLUDE_DIRS ${NLOHMANN_JSON_INCLUDE_DIR})
            set(NLOHMANN_JSON_FOUND TRUE)
        else()
            # Manual fallback for MSYS2/MinGW installations
            set(_nlohmann_search_paths
                "C:/msys64/mingw64"
                "C:/msys64/ucrt64"
                "C:/msys64/clang64"
                "/usr/local"
                "/usr"
            )
            foreach(_prefix ${_nlohmann_search_paths})
                if(EXISTS "${_prefix}/include/nlohmann/json.hpp")
                    find_path(NLOHMANN_JSON_INCLUDE_DIR nlohmann/json.hpp PATHS "${_prefix}/include" NO_DEFAULT_PATH)
                    if(NLOHMANN_JSON_INCLUDE_DIR)
                        set(NLOHMANN_JSON_INCLUDE_DIRS ${NLOHMANN_JSON_INCLUDE_DIR})
                        set(NLOHMANN_JSON_FOUND TRUE)
                        break()
                    endif()
                endif()
            endforeach()
            
            if(NOT NLOHMANN_JSON_FOUND)
                message(FATAL_ERROR "nlohmann-json nicht gefunden. Bitte nlohmann-json installieren oder CMAKE_PREFIX_PATH setzen.")
            endif()
        endif()
    endif()
endif()

# OpenSSL for SHA256
find_package(OpenSSL QUIET)
if(NOT OpenSSL_FOUND)
    # Manual fallback for MSYS2/MinGW installations
    set(_openssl_search_paths
        "C:/msys64/mingw64"
        "C:/msys64/ucrt64"
        "C:/msys64/clang64"
        "/usr/local"
        "/usr"
    )
    foreach(_prefix ${_openssl_search_paths})
        if(EXISTS "${_prefix}/include/openssl/ssl.h")
            find_path(OPENSSL_INCLUDE_DIR openssl/ssl.h PATHS "${_prefix}/include" NO_DEFAULT_PATH)
            find_library(OPENSSL_CRYPTO_LIBRARY NAMES crypto libcrypto PATHS "${_prefix}/lib" NO_DEFAULT_PATH)
            find_library(OPENSSL_SSL_LIBRARY NAMES ssl libssl PATHS "${_prefix}/lib" NO_DEFAULT_PATH)
            if(OPENSSL_INCLUDE_DIR AND OPENSSL_CRYPTO_LIBRARY AND OPENSSL_SSL_LIBRARY)
                set(OPENSSL_FOUND TRUE)
                set(OPENSSL_INCLUDE_DIRS ${OPENSSL_INCLUDE_DIR})
                set(OPENSSL_LIBRARIES ${OPENSSL_SSL_LIBRARY} ${OPENSSL_CRYPTO_LIBRARY})
                break()
            endif()
        endif()
    endforeach()
    
    if(NOT OPENSSL_FOUND)
        message(FATAL_ERROR "OpenSSL nicht gefunden. Bitte OpenSSL installieren oder CMAKE_PREFIX_PATH setzen.")
    endif()
endif()

find_package(CURL REQUIRED)

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
    src/core/acceleration.cpp
    src/io/fits_io.cpp
    src/io/config.cpp
    src/image/processing.cpp
    src/image/cfa_processing.cpp
    src/image/normalization.cpp
    src/image/background_extraction.cpp
    src/registration/registration.cpp
    src/registration/global_registration.cpp
    src/registration/astrometric_rescue.cpp
    src/metrics/metrics.cpp
    src/metrics/tile_metrics.cpp
    src/metrics/linearity.cpp
    src/reconstruction/reconstruction.cpp
    src/reconstruction/memory_budget.cpp
    src/reconstruction/dead_tile_detector.cpp
    src/reconstruction/progress_reporter.cpp
    src/reconstruction/tile_scheduler.cpp
    src/reconstruction/local_weight_regularization.cpp
    src/reconstruction/progress_reporter.cpp
    src/reconstruction/tile_scheduler.cpp
    src/reconstruction/tile_boundary_diagnostics.cpp
    src/reconstruction/tile_normalization.cpp
    src/reconstruction/tile_weight_profile_diagnostics.cpp
    src/pipeline/adaptive_tile_grid.cpp
    src/astrometry/wcs.cpp
    src/astrometry/gaia_catalog.cpp
    src/astrometry/photometric_color_cal.cpp
)

# Main library
add_library(tile_compile_lib STATIC ${LIB_SOURCES})

target_compile_definitions(tile_compile_lib PUBLIC
    TILE_COMPILE_WITH_CUDA=$<BOOL:${TILE_COMPILE_WITH_CUDA}>
)

target_link_libraries(tile_compile_lib PUBLIC
    Eigen3::Eigen
    ${OpenCV_LIBS}
    ${CFITSIO_LIBRARIES}
    ${YAML_CPP_LIBRARIES}
    OpenSSL::Crypto
    CURL::libcurl
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(tile_compile_lib PUBLIC OpenMP::OpenMP_CXX)
endif()

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
add_executable(tile_compile_runner
    apps/runner_main.cpp
    apps/runner_pipeline.cpp
    apps/runner_phase_metrics.cpp
    apps/runner_phase_local_metrics.cpp
    apps/runner_phase_registration.cpp
    apps/runner_shared.cpp
    apps/runner_resume.cpp
)
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

if(APPLE)
    set(_TC_INSTALL_RPATH "@executable_path/../lib")
elseif(UNIX)
    set(_TC_INSTALL_RPATH "$ORIGIN/../lib")
endif()

if(UNIX)
    set_target_properties(tile_compile_runner tile_compile_cli PROPERTIES
        INSTALL_RPATH "${_TC_INSTALL_RPATH}"
    )
endif()

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
            tests/test_acceleration_backend.cpp
            tests/test_background_extraction.cpp
            tests/test_utils.cpp
            tests/test_fits.cpp
            tests/test_fits_hdu_selection.cpp
            tests/test_debayer.cpp
            tests/test_mode_gating.cpp
            tests/test_registration_cascade.cpp
            tests/test_registration_new_features.cpp
            tests/test_registration_outlier_utils.cpp
            tests/test_reconstruction_regression.cpp
            tests/test_stacking_quality_weighting.cpp
            tests/test_memory_budget.cpp
            tests/test_dead_tile_detector.cpp
            tests/test_coverage_filtered_grid.cpp
            tests/test_tile_scheduler.cpp
            tests/test_reconstruction_parallel.cpp
        )
        target_link_libraries(tests PRIVATE tile_compile_lib Catch2::Catch2WithMain)
        include(CTest)
        include(Catch)
        catch_discover_tests(tests)
    endif()
endif()

# Install targets
if(WIN32)
    set(TILE_COMPILE_RUNTIME_DESTINATION bin)
else()
    set(TILE_COMPILE_RUNTIME_DESTINATION lib)
endif()

# Install executables
install(TARGETS tile_compile_runner tile_compile_cli
    RUNTIME DESTINATION bin
)

# Install runtime dependencies (CMake 3.21+ only)
# For older CMake, runtime dependency copying must be handled externally.
if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.21" AND WIN32)
    cmake_language(EVAL CODE "
        install(CODE \"
            file(GET_RUNTIME_DEPENDENCIES
                EXECUTABLES
                    \\\${CMAKE_INSTALL_PREFIX}/bin/tile_compile_runner.exe
                    \\\${CMAKE_INSTALL_PREFIX}/bin/tile_compile_cli.exe
                RESOLVED_DEPENDENCIES_VAR resolved_deps
                UNRESOLVED_DEPENDENCIES_VAR unresolved_deps
                PRE_EXCLUDE_REGEXES \\\"api-ms-win-.*\\\" \\\"ext-ms-.*\\\"
                POST_EXCLUDE_REGEXES \\\".*[Ss]ystem32/.*\\\"
            )
            foreach(dep IN LISTS resolved_deps)
                file(INSTALL \\\${dep} DESTINATION \\\${CMAKE_INSTALL_PREFIX}/bin)
            endforeach()
        \")
    ")
endif()

install(DIRECTORY include/tile_compile
    DESTINATION include
)

# Print configuration summary
message(STATUS "")
message(STATUS "=== Tile-Compile C++ Configuration ===")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "TILE_COMPILE_ENABLE_CUDA: ${TILE_COMPILE_ENABLE_CUDA}")
message(STATUS "TILE_COMPILE_WITH_CUDA: ${TILE_COMPILE_WITH_CUDA}")
message(STATUS "CUDA nvcc: ${TILE_COMPILE_NVCC_EXECUTABLE}")
message(STATUS "Eigen3: ${EIGEN3_VERSION}")
message(STATUS "OpenCV: ${OpenCV_VERSION}")

# Detect available GPU acceleration backends from the actual OpenCV include
# directories visible to this build. The previous check used
# `check_include_file_cxx(...)`, which can produce false negatives with custom
# OpenCV CUDA installs even when the headers are present and the project builds.
set(_tile_compile_gpu_backends "")
set(TILE_COMPILE_CMAKE_HAS_OPENCV_CUDA_HEADER FALSE)
set(TILE_COMPILE_CMAKE_HAS_OPENCV_CUDAWARPING_HEADER FALSE)
set(TILE_COMPILE_CMAKE_HAS_OPENCV_CUDAARITHM_HEADER FALSE)
set(TILE_COMPILE_CMAKE_HAS_OPENCV_OPENCL_HEADER FALSE)
set(_tile_compile_opencv_include_dirs ${OpenCV_INCLUDE_DIRS})
foreach(_opencv_target opencv_core OpenCV::opencv_core)
    if(TARGET ${_opencv_target})
        get_target_property(_target_includes ${_opencv_target}
                            INTERFACE_INCLUDE_DIRECTORIES)
        if(_target_includes AND NOT _target_includes STREQUAL "_target_includes-NOTFOUND")
            list(APPEND _tile_compile_opencv_include_dirs ${_target_includes})
        endif()
    endif()
endforeach()
list(REMOVE_DUPLICATES _tile_compile_opencv_include_dirs)

foreach(_inc_dir ${_tile_compile_opencv_include_dirs})
    if(NOT _inc_dir OR _inc_dir MATCHES "^\\$<")
        continue()
    endif()
    if(EXISTS "${_inc_dir}/opencv2/core/cuda.hpp")
        set(TILE_COMPILE_CMAKE_HAS_OPENCV_CUDA_HEADER TRUE)
    endif()
    if(EXISTS "${_inc_dir}/opencv2/cudawarping.hpp")
        set(TILE_COMPILE_CMAKE_HAS_OPENCV_CUDAWARPING_HEADER TRUE)
    endif()
    if(EXISTS "${_inc_dir}/opencv2/cudaarithm.hpp")
        set(TILE_COMPILE_CMAKE_HAS_OPENCV_CUDAARITHM_HEADER TRUE)
    endif()
    if(EXISTS "${_inc_dir}/opencv2/core/ocl.hpp")
        set(TILE_COMPILE_CMAKE_HAS_OPENCV_OPENCL_HEADER TRUE)
    endif()
endforeach()

if(TILE_COMPILE_CMAKE_HAS_OPENCV_CUDA_HEADER AND
   (TILE_COMPILE_CMAKE_HAS_OPENCV_CUDAWARPING_HEADER OR
    TILE_COMPILE_CMAKE_HAS_OPENCV_CUDAARITHM_HEADER))
    list(APPEND _tile_compile_gpu_backends "CUDA")
endif()
if(TILE_COMPILE_CMAKE_HAS_OPENCV_OPENCL_HEADER)
    list(APPEND _tile_compile_gpu_backends "OpenCL")
endif()

if(_tile_compile_gpu_backends)
    string(REPLACE ";" ", " _gpu_backends_str "${_tile_compile_gpu_backends}")
    message(STATUS "GPU acceleration: ${_gpu_backends_str} (runtime detection)")
else()
    message(STATUS "GPU acceleration: CPU-only (no GPU backends available)")
endif()

message(STATUS "CFITSIO: ${CFITSIO_VERSION}")
message(STATUS "yaml-cpp: found")
message(STATUS "nlohmann/json: ${nlohmann_json_FOUND}")
message(STATUS "spdlog: ${spdlog_FOUND}")
message(STATUS "CLI11: ${CLI11_FOUND}")
message(STATUS "Catch2: ${Catch2_FOUND}")
message(STATUS "")
