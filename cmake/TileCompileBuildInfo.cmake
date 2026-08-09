include(CMakeParseArguments)

set(_TILE_COMPILE_BUILD_INFO_MODULE_DIR "${CMAKE_CURRENT_LIST_DIR}")

function(tile_compile_add_build_info target_name)
    set(one_value_args
        COMPONENT PROJECT_VERSION SOURCE_ROOT OPENCV_VERSION CUDA_VERSION
        FEATURE_FLAGS BUILD_INFO_SOURCE)
    cmake_parse_arguments(TCBI "" "${one_value_args}" "" ${ARGN})

    foreach(required_arg COMPONENT PROJECT_VERSION SOURCE_ROOT BUILD_INFO_SOURCE)
        if(NOT TCBI_${required_arg})
            message(FATAL_ERROR "tile_compile_add_build_info(${target_name}): ${required_arg} is required")
        endif()
    endforeach()

    set(generated_dir "${CMAKE_CURRENT_BINARY_DIR}/generated/build_info/${target_name}/$<CONFIG>")
    set(generated_header "${generated_dir}/tile_compile_build_info_generated.hpp")
    set(generated_json "${generated_dir}/${target_name}.build-info.json")
    set(generator_script "${_TILE_COMPILE_BUILD_INFO_MODULE_DIR}/GenerateBuildInfo.cmake")

    add_custom_target(${target_name}_build_info_generation
        COMMAND ${CMAKE_COMMAND} -E make_directory "${generated_dir}"
        COMMAND ${CMAKE_COMMAND}
            "-DOUTPUT_HEADER=${generated_header}"
            "-DOUTPUT_JSON=${generated_json}"
            "-DREPOSITORY_ROOT=${TCBI_SOURCE_ROOT}"
            "-DCOMPONENT=${TCBI_COMPONENT}"
            "-DPROJECT_VERSION=${TCBI_PROJECT_VERSION}"
            "-DBUILD_TYPE=$<CONFIG>"
            "-DCXX_COMPILER_ID=${CMAKE_CXX_COMPILER_ID}"
            "-DCXX_COMPILER_VERSION=${CMAKE_CXX_COMPILER_VERSION}"
            "-DCXX_COMPILER_PATH=${CMAKE_CXX_COMPILER}"
            "-DCXX_STANDARD=${CMAKE_CXX_STANDARD}"
            "-DPOINTER_SIZE=${CMAKE_SIZEOF_VOID_P}"
            "-DOS_NAME=${CMAKE_SYSTEM_NAME}"
            "-DOS_VERSION=${CMAKE_SYSTEM_VERSION}"
            "-DARCHITECTURE=${CMAKE_SYSTEM_PROCESSOR}"
            "-DOPENCV_VERSION=${TCBI_OPENCV_VERSION}"
            "-DCUDA_VERSION=${TCBI_CUDA_VERSION}"
            "-DFEATURE_FLAGS=${TCBI_FEATURE_FLAGS}"
            -P "${generator_script}"
        DEPENDS "${generator_script}"
        COMMENT "Generating reproducible build metadata for ${target_name}"
        VERBATIM)

    add_dependencies(${target_name} ${target_name}_build_info_generation)
    target_sources(${target_name} PRIVATE "${TCBI_BUILD_INFO_SOURCE}")
    target_include_directories(${target_name} PRIVATE "${generated_dir}")
    add_custom_command(TARGET ${target_name} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${generated_json}"
            "$<TARGET_FILE_DIR:${target_name}>/${target_name}.build-info.json"
        COMMENT "Copying ${target_name} build metadata sidecar"
        VERBATIM)
    install(FILES "${generated_json}"
        DESTINATION bin
        RENAME "${target_name}.build-info.json")
endfunction()
