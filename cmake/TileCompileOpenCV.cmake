# Shared OpenCV discovery for tile_compile_cpp and web_backend_cpp.
#
# An OpenCV-CUDA build records the CUDA version it was compiled with
# (OpenCV_CUDA_VERSION in OpenCVConfig.cmake) and its config insists on that
# exact toolkit. CUDA keeps the runtime ABI stable within a major version
# (minor-version compatibility), so an OpenCV built against CUDA 13.0 runs on
# a CUDA 13.5 toolkit and driver. These macros
#
#   1. pick the first OpenCV config whose CUDA major version matches the
#      installed toolkit (or that has no CUDA at all), and
#   2. let OpenCVConfig.cmake accept a newer toolkit of the same major version
#      while every CUDA library still resolves from the installed toolkit.
#
# An older toolkit than OpenCV was built with, or a different major version,
# is not bridged: OpenCV then fails with its own message.
#
# Usage (directory scope, before find_package(OpenCV)):
#   include(".../cmake/TileCompileOpenCV.cmake")
#   tile_compile_select_opencv_dir()
#   tile_compile_find_opencv(QUIET)

# Major.minor of the installed CUDA toolkit, from nvcc (empty if none).
macro(_tile_compile_cuda_toolkit_version out_version out_root)
    set(${out_version} "")
    set(${out_root} "")
    set(_tc_nvcc "")
    foreach(_tc_candidate
            "${TILE_COMPILE_NVCC_EXECUTABLE}"
            "${CUDA_NVCC_EXECUTABLE}"
            "$ENV{CUDA_NVCC_EXECUTABLE}"
            "${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc"
            "$ENV{CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc"
            "$ENV{CUDA_HOME}/bin/nvcc"
            "$ENV{CUDA_PATH}/bin/nvcc"
            "/usr/local/cuda/bin/nvcc"
            "/opt/cuda/bin/nvcc")
        if(_tc_candidate AND NOT _tc_candidate MATCHES "^/bin/nvcc$" AND EXISTS "${_tc_candidate}")
            set(_tc_nvcc "${_tc_candidate}")
            break()
        endif()
    endforeach()
    if(_tc_nvcc)
        execute_process(COMMAND "${_tc_nvcc}" --version
                        OUTPUT_VARIABLE _tc_nvcc_out ERROR_QUIET
                        RESULT_VARIABLE _tc_nvcc_rc)
        if(_tc_nvcc_rc EQUAL 0 AND _tc_nvcc_out MATCHES "release ([0-9]+\\.[0-9]+)")
            set(${out_version} "${CMAKE_MATCH_1}")
            get_filename_component(_tc_nvcc_real "${_tc_nvcc}" REALPATH)
            get_filename_component(_tc_nvcc_bin "${_tc_nvcc_real}" DIRECTORY)
            get_filename_component(${out_root} "${_tc_nvcc_bin}" DIRECTORY)
        endif()
    endif()
endmacro()

# OpenCV_CUDA_VERSION recorded in an OpenCVConfig.cmake ("" for CPU builds).
macro(_tile_compile_opencv_cuda_version config_dir out_version)
    set(${out_version} "")
    if(EXISTS "${config_dir}/OpenCVConfig.cmake")
        file(STRINGS "${config_dir}/OpenCVConfig.cmake" _tc_cuda_line
             REGEX "^set\\(OpenCV_CUDA_VERSION +\"[0-9.]*\"\\)")
        if(_tc_cuda_line MATCHES "\"([0-9.]*)\"")
            set(${out_version} "${CMAKE_MATCH_1}")
        endif()
    endif()
endmacro()

macro(_tile_compile_major version out_major)
    string(REGEX MATCH "^[0-9]+" ${out_major} "${version}")
endmacro()

# Choose OpenCV_DIR unless the caller set one explicitly. An auto-selected
# value from an earlier configure is re-evaluated, so a toolkit upgrade picks
# the matching OpenCV build on the next configure.
macro(tile_compile_select_opencv_dir)
    _tile_compile_cuda_toolkit_version(_tc_toolkit_version _tc_toolkit_root)
    _tile_compile_major("${_tc_toolkit_version}" _tc_toolkit_major)
    set(_tc_candidates
        "$ENV{TILE_COMPILE_OPENCV_DIR}"
        "$ENV{OpenCV_DIR}"
        "/opt/opencv-4.11-cuda${_tc_toolkit_major}/lib64/cmake/opencv4"
        "/opt/opencv-4.11-cuda${_tc_toolkit_major}/lib/cmake/opencv4"
        "$ENV{HOME}/.local/opencv-cuda${_tc_toolkit_major}-4.11/lib64/cmake/opencv4"
        "$ENV{HOME}/.local/opencv-cuda-4.11/lib64/cmake/opencv4"
        "$ENV{HOME}/.local/opencv-cuda/lib64/cmake/opencv4")
    # An explicit environment choice is taken as is. Among the rest, a build
    # for exactly the installed toolkit wins over one that needs the
    # same-major bridge (an older build may also lag behind other system
    # libraries it was linked against); the first compatible one is the
    # fallback.
    set(_tc_preferred "")
    set(_tc_compatible "")
    foreach(_tc_env_dir "$ENV{TILE_COMPILE_OPENCV_DIR}" "$ENV{OpenCV_DIR}")
        if(_tc_env_dir AND EXISTS "${_tc_env_dir}/OpenCVConfig.cmake")
            set(_tc_preferred "${_tc_env_dir}")
            break()
        endif()
    endforeach()
    if(NOT _tc_preferred)
        foreach(_tc_dir ${_tc_candidates})
            if(NOT _tc_dir OR NOT EXISTS "${_tc_dir}/OpenCVConfig.cmake")
                continue()
            endif()
            _tile_compile_opencv_cuda_version("${_tc_dir}" _tc_opencv_cuda)
            _tile_compile_major("${_tc_opencv_cuda}" _tc_opencv_major)
            if(_tc_opencv_cuda AND NOT "${_tc_opencv_major}" STREQUAL "${_tc_toolkit_major}")
                message(STATUS "OpenCV candidate skipped (built for CUDA ${_tc_opencv_cuda}, "
                               "toolkit is ${_tc_toolkit_version}): ${_tc_dir}")
                continue()
            endif()
            if(_tc_opencv_cuda AND _tc_toolkit_version AND
               _tc_opencv_cuda VERSION_EQUAL _tc_toolkit_version)
                set(_tc_preferred "${_tc_dir}")
                break()
            endif()
            if(NOT _tc_compatible)
                set(_tc_compatible "${_tc_dir}")
            endif()
        endforeach()
        if(NOT _tc_preferred)
            set(_tc_preferred "${_tc_compatible}")
        endif()
    endif()
    # OpenCV_DIR is ours to re-evaluate when it is unset, when it is the
    # candidate chosen last time, or when it is what an automatic choice
    # ended up with (find_package's own search writes e.g. the system OpenCV
    # into the cache when no candidate matched; without this that value would
    # stick like an explicit -DOpenCV_DIR and hide a matching build later).
    set(_tc_opencv_dir_auto FALSE)
    if((NOT DEFINED OpenCV_DIR OR "${OpenCV_DIR}" STREQUAL "" OR
        "${OpenCV_DIR}" MATCHES "-NOTFOUND$") OR
       (DEFINED TILE_COMPILE_OPENCV_DIR_AUTO_LAST AND
        "${OpenCV_DIR}" STREQUAL "${TILE_COMPILE_OPENCV_DIR_AUTO_LAST}") OR
       (DEFINED TILE_COMPILE_OPENCV_DIR_AUTO_RESULT AND
        "${OpenCV_DIR}" STREQUAL "${TILE_COMPILE_OPENCV_DIR_AUTO_RESULT}"))
        set(_tc_opencv_dir_auto TRUE)
        if(_tc_preferred)
            set(OpenCV_DIR "${_tc_preferred}" CACHE PATH
                "Preferred OpenCV config directory for tile_compile" FORCE)
        elseif(DEFINED TILE_COMPILE_OPENCV_DIR_AUTO_RESULT AND
               "${OpenCV_DIR}" STREQUAL "${TILE_COMPILE_OPENCV_DIR_AUTO_RESULT}")
            # No candidate now: let find_package search afresh.
            set(OpenCV_DIR "OpenCV_DIR-NOTFOUND" CACHE PATH
                "Preferred OpenCV config directory for tile_compile" FORCE)
        endif()
    endif()
    set(TILE_COMPILE_OPENCV_DIR_AUTO_LAST "${_tc_preferred}"
        CACHE INTERNAL "Last auto-detected preferred OpenCV_DIR")
    if(_tc_toolkit_root AND NOT CUDA_TOOLKIT_ROOT_DIR)
        set(TILE_COMPILE_CUDA_TOOLKIT_ROOT_AUTO "${_tc_toolkit_root}")
    endif()
endmacro()

# find_package(OpenCV ${ARGN}) with the same-major CUDA bridge described above.
macro(tile_compile_find_opencv)
    set(_tc_bridged FALSE)
    if(OpenCV_DIR AND EXISTS "${OpenCV_DIR}/OpenCVConfig.cmake")
        _tile_compile_opencv_cuda_version("${OpenCV_DIR}" _tc_opencv_cuda)
    else()
        set(_tc_opencv_cuda "")
    endif()
    if(_tc_opencv_cuda)
        if(NOT CUDA_TOOLKIT_ROOT_DIR AND TILE_COMPILE_CUDA_TOOLKIT_ROOT_AUTO)
            set(CUDA_TOOLKIT_ROOT_DIR "${TILE_COMPILE_CUDA_TOOLKIT_ROOT_AUTO}"
                CACHE PATH "CUDA toolkit used by OpenCV's FindCUDA" FORCE)
        endif()
        if(POLICY CMP0146)
            cmake_policy(PUSH)
            cmake_policy(SET CMP0146 OLD)
        endif()
        find_package(CUDA QUIET)
        if(POLICY CMP0146)
            cmake_policy(POP)
        endif()
        if(CUDA_FOUND)
            _tile_compile_major("${CUDA_VERSION_STRING}" _tc_found_major)
            _tile_compile_major("${_tc_opencv_cuda}" _tc_opencv_major)
            if(NOT CUDA_VERSION_STRING VERSION_EQUAL _tc_opencv_cuda AND
               "${_tc_found_major}" STREQUAL "${_tc_opencv_major}" AND
               CUDA_VERSION_STRING VERSION_GREATER _tc_opencv_cuda)
                message(STATUS "OpenCV was built against CUDA ${_tc_opencv_cuda}; "
                               "using CUDA ${CUDA_VERSION_STRING} (same major version, "
                               "minor-version compatible)")
                set(_tc_bridged TRUE)
                set(_tc_real_cuda_version_string "${CUDA_VERSION_STRING}")
                set(CUDA_VERSION_STRING "${_tc_opencv_cuda}")
            endif()
        endif()
    endif()
    if(POLICY CMP0146)
        cmake_policy(PUSH)
        cmake_policy(SET CMP0146 OLD)
    endif()
    find_package(OpenCV ${ARGN})
    if(POLICY CMP0146)
        cmake_policy(POP)
    endif()
    if(_tc_bridged)
        set(CUDA_VERSION_STRING "${_tc_real_cuda_version_string}")
    endif()
    # Remember what an automatic choice resolved to, so the next configure
    # may replace it (see tile_compile_select_opencv_dir); an explicit
    # OpenCV_DIR is never recorded and stays as given.
    if(_tc_opencv_dir_auto)
        set(TILE_COMPILE_OPENCV_DIR_AUTO_RESULT "${OpenCV_DIR}"
            CACHE INTERNAL "OpenCV_DIR an automatic choice resolved to")
    endif()
endmacro()
