cmake_minimum_required(VERSION 3.16)

foreach(required_var OUTPUT_HEADER OUTPUT_JSON REPOSITORY_ROOT COMPONENT PROJECT_VERSION)
    if(NOT DEFINED ${required_var})
        message(FATAL_ERROR "GenerateBuildInfo.cmake: ${required_var} is required")
    endif()
endforeach()

function(normalize_value variable_name fallback)
    if(NOT DEFINED ${variable_name} OR "${${variable_name}}" STREQUAL "")
        set(${variable_name} "${fallback}" PARENT_SCOPE)
    endif()
endfunction()

function(cpp_escape input output_var)
    set(value "${input}")
    string(REPLACE "\\" "\\\\" value "${value}")
    string(REPLACE "\"" "\\\"" value "${value}")
    string(REPLACE "\n" "\\n" value "${value}")
    string(REPLACE "\r" "" value "${value}")
    set(${output_var} "${value}" PARENT_SCOPE)
endfunction()

function(json_escape input output_var)
    cpp_escape("${input}" value)
    string(REPLACE "\t" "\\t" value "${value}")
    set(${output_var} "${value}" PARENT_SCOPE)
endfunction()

function(write_if_different path content)
    set(write_file TRUE)
    if(EXISTS "${path}")
        file(READ "${path}" previous_content)
        if(previous_content STREQUAL content)
            set(write_file FALSE)
        endif()
    endif()
    if(write_file)
        file(WRITE "${path}" "${content}")
    endif()
endfunction()

normalize_value(BUILD_TYPE "unknown")
normalize_value(CXX_COMPILER_ID "unknown")
normalize_value(CXX_COMPILER_VERSION "unknown")
normalize_value(CXX_COMPILER_PATH "unknown")
normalize_value(CXX_STANDARD "20")
normalize_value(POINTER_SIZE "unknown")
normalize_value(OS_NAME "unknown")
normalize_value(OS_VERSION "unknown")
normalize_value(ARCHITECTURE "unknown")
normalize_value(OPENCV_VERSION "unknown")
normalize_value(CUDA_VERSION "not_enabled")
normalize_value(FEATURE_FLAGS "none")

set(GIT_SHA "unknown")
set(GIT_DESCRIBE "unknown")
set(GIT_DIRTY FALSE)
set(GIT_STATUS_TEXT "")
set(SOURCE_FILES "")
find_program(GIT_EXECUTABLE git)
if(GIT_EXECUTABLE AND EXISTS "${REPOSITORY_ROOT}/.git")
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" -C "${REPOSITORY_ROOT}" rev-parse HEAD
        RESULT_VARIABLE git_sha_result OUTPUT_VARIABLE GIT_SHA
        ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(NOT git_sha_result EQUAL 0)
        set(GIT_SHA "unknown")
    endif()
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" -C "${REPOSITORY_ROOT}" describe --always --tags
        RESULT_VARIABLE git_describe_result OUTPUT_VARIABLE GIT_DESCRIBE
        ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(NOT git_describe_result EQUAL 0)
        set(GIT_DESCRIBE "${GIT_SHA}")
    endif()
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" -C "${REPOSITORY_ROOT}" status --porcelain=v1 --untracked-files=all
        RESULT_VARIABLE git_status_result OUTPUT_VARIABLE GIT_STATUS_TEXT
        ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(git_status_result EQUAL 0 AND NOT "${GIT_STATUS_TEXT}" STREQUAL "")
        set(GIT_DIRTY TRUE)
    endif()
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" -C "${REPOSITORY_ROOT}" ls-files --cached --others --exclude-standard
        RESULT_VARIABLE git_files_result OUTPUT_VARIABLE git_files_text
        ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(git_files_result EQUAL 0)
        string(REPLACE "\r\n" "\n" git_files_text "${git_files_text}")
        string(REPLACE "\n" ";" SOURCE_FILES "${git_files_text}")
    endif()
elseif(DEFINED ENV{GITHUB_SHA} AND NOT "$ENV{GITHUB_SHA}" STREQUAL "")
    set(GIT_SHA "$ENV{GITHUB_SHA}")
    set(GIT_DESCRIBE "$ENV{GITHUB_SHA}")
endif()

if(NOT SOURCE_FILES)
    file(GLOB_RECURSE archive_files LIST_DIRECTORIES FALSE
        RELATIVE "${REPOSITORY_ROOT}" "${REPOSITORY_ROOT}/*")
    foreach(relative_path IN LISTS archive_files)
        if(relative_path MATCHES "(^|/)(\\.git|build[^/]*|runs|node_modules|\\.venv|Testing)(/|$)")
            continue()
        endif()
        list(APPEND SOURCE_FILES "${relative_path}")
    endforeach()
endif()
list(REMOVE_DUPLICATES SOURCE_FILES)
list(SORT SOURCE_FILES)

set(source_manifest "")
set(source_file_count 0)
foreach(relative_path IN LISTS SOURCE_FILES)
    if(relative_path MATCHES "(^|/)(\\.git|\\.ai_analyses|\\.tile_compile|build[^/]*|runs|node_modules|dist|\\.venv|Testing|tmp)(/|$)")
        continue()
    endif()
    set(absolute_path "${REPOSITORY_ROOT}/${relative_path}")
    if(IS_DIRECTORY "${absolute_path}" OR NOT EXISTS "${absolute_path}")
        continue()
    endif()
    file(SHA256 "${absolute_path}" file_sha256)
    file(SIZE "${absolute_path}" file_size)
    string(APPEND source_manifest "${relative_path}\t${file_size}\t${file_sha256}\n")
    math(EXPR source_file_count "${source_file_count} + 1")
endforeach()
string(SHA256 SOURCE_TREE_DIGEST "${source_manifest}")
if(GIT_DIRTY)
    string(SHA256 DIRTY_TREE_DIGEST "${GIT_STATUS_TEXT}\n${SOURCE_TREE_DIGEST}")
else()
    set(DIRTY_TREE_DIGEST "clean")
endif()

set(GITHUB_REPOSITORY "$ENV{GITHUB_REPOSITORY}")
set(GITHUB_RUN_ID "$ENV{GITHUB_RUN_ID}")
set(GITHUB_RUN_ATTEMPT "$ENV{GITHUB_RUN_ATTEMPT}")
set(GITHUB_SERVER_URL "$ENV{GITHUB_SERVER_URL}")
if(GITHUB_SERVER_URL STREQUAL "")
    set(GITHUB_SERVER_URL "https://github.com")
endif()
set(GITHUB_RUN_URL "")
if(NOT GITHUB_REPOSITORY STREQUAL "" AND NOT GITHUB_RUN_ID STREQUAL "")
    set(GITHUB_RUN_URL "${GITHUB_SERVER_URL}/${GITHUB_REPOSITORY}/actions/runs/${GITHUB_RUN_ID}")
endif()
set(SOURCE_DATE_EPOCH "$ENV{SOURCE_DATE_EPOCH}")
set(BUILD_TIMESTAMP_UTC "reproducible-unspecified")
if(NOT SOURCE_DATE_EPOCH STREQUAL "")
    string(TIMESTAMP BUILD_TIMESTAMP_UTC "%Y-%m-%dT%H:%M:%SZ" UTC)
endif()

math(EXPR POINTER_BITS "${POINTER_SIZE} * 8" OUTPUT_FORMAT DECIMAL)
set(COMPILER_ABI "${CXX_COMPILER_ID}-${CXX_COMPILER_VERSION}-cxx${CXX_STANDARD}-${POINTER_BITS}bit")
set(build_id_input
    "component=${COMPONENT}\nversion=${PROJECT_VERSION}\ngit_sha=${GIT_SHA}\ngit_describe=${GIT_DESCRIBE}\ngit_dirty=${GIT_DIRTY}\ndirty_tree_digest=${DIRTY_TREE_DIGEST}\nsource_tree_digest=${SOURCE_TREE_DIGEST}\nbuild_type=${BUILD_TYPE}\ncompiler=${CXX_COMPILER_ID}-${CXX_COMPILER_VERSION}\ncompiler_abi=${COMPILER_ABI}\nos=${OS_NAME}-${OS_VERSION}\narch=${ARCHITECTURE}\nopencv=${OPENCV_VERSION}\ncuda=${CUDA_VERSION}\nfeatures=${FEATURE_FLAGS}\ngithub_repository=${GITHUB_REPOSITORY}\ngithub_run_id=${GITHUB_RUN_ID}\ngithub_run_attempt=${GITHUB_RUN_ATTEMPT}\nsource_date_epoch=${SOURCE_DATE_EPOCH}\n")
string(SHA256 BUILD_ID "${build_id_input}")

foreach(field COMPONENT PROJECT_VERSION GIT_SHA GIT_DESCRIBE DIRTY_TREE_DIGEST
              SOURCE_TREE_DIGEST BUILD_TYPE CXX_COMPILER_ID CXX_COMPILER_VERSION
              CXX_COMPILER_PATH COMPILER_ABI OS_NAME OS_VERSION ARCHITECTURE
              OPENCV_VERSION CUDA_VERSION FEATURE_FLAGS GITHUB_REPOSITORY
              GITHUB_RUN_ID GITHUB_RUN_ATTEMPT GITHUB_RUN_URL SOURCE_DATE_EPOCH
              BUILD_TIMESTAMP_UTC BUILD_ID)
    cpp_escape("${${field}}" ${field}_CPP)
    json_escape("${${field}}" ${field}_JSON)
endforeach()

if(GIT_DIRTY)
    set(GIT_DIRTY_LITERAL true)
else()
    set(GIT_DIRTY_LITERAL false)
endif()

set(header_content "#pragma once\n\nnamespace tile_compile::core::generated_build_info {\ninline constexpr const char* component = \"${COMPONENT_CPP}\";\ninline constexpr const char* project_version = \"${PROJECT_VERSION_CPP}\";\ninline constexpr const char* git_sha = \"${GIT_SHA_CPP}\";\ninline constexpr const char* git_describe = \"${GIT_DESCRIBE_CPP}\";\ninline constexpr bool git_dirty = ${GIT_DIRTY_LITERAL};\ninline constexpr const char* dirty_tree_digest = \"${DIRTY_TREE_DIGEST_CPP}\";\ninline constexpr const char* source_tree_digest = \"${SOURCE_TREE_DIGEST_CPP}\";\ninline constexpr int source_file_count = ${source_file_count};\ninline constexpr const char* build_id = \"${BUILD_ID_CPP}\";\ninline constexpr const char* build_type = \"${BUILD_TYPE_CPP}\";\ninline constexpr const char* compiler_id = \"${CXX_COMPILER_ID_CPP}\";\ninline constexpr const char* compiler_version = \"${CXX_COMPILER_VERSION_CPP}\";\ninline constexpr const char* compiler_path = \"${CXX_COMPILER_PATH_CPP}\";\ninline constexpr const char* compiler_abi = \"${COMPILER_ABI_CPP}\";\ninline constexpr const char* os_name = \"${OS_NAME_CPP}\";\ninline constexpr const char* os_version = \"${OS_VERSION_CPP}\";\ninline constexpr const char* architecture = \"${ARCHITECTURE_CPP}\";\ninline constexpr const char* opencv_version = \"${OPENCV_VERSION_CPP}\";\ninline constexpr const char* cuda_version = \"${CUDA_VERSION_CPP}\";\ninline constexpr const char* feature_flags = \"${FEATURE_FLAGS_CPP}\";\ninline constexpr const char* github_repository = \"${GITHUB_REPOSITORY_CPP}\";\ninline constexpr const char* github_run_id = \"${GITHUB_RUN_ID_CPP}\";\ninline constexpr const char* github_run_attempt = \"${GITHUB_RUN_ATTEMPT_CPP}\";\ninline constexpr const char* github_run_url = \"${GITHUB_RUN_URL_CPP}\";\ninline constexpr const char* source_date_epoch = \"${SOURCE_DATE_EPOCH_CPP}\";\ninline constexpr const char* build_timestamp_utc = \"${BUILD_TIMESTAMP_UTC_CPP}\";\n} // namespace tile_compile::core::generated_build_info\n")

set(json_content "{\n  \"schema_version\": 1,\n  \"component\": \"${COMPONENT_JSON}\",\n  \"project_version\": \"${PROJECT_VERSION_JSON}\",\n  \"build_id\": \"${BUILD_ID_JSON}\",\n  \"source\": {\n    \"git_sha\": \"${GIT_SHA_JSON}\",\n    \"git_describe\": \"${GIT_DESCRIBE_JSON}\",\n    \"git_dirty\": ${GIT_DIRTY_LITERAL},\n    \"dirty_tree_digest\": \"${DIRTY_TREE_DIGEST_JSON}\",\n    \"source_tree_digest\": \"${SOURCE_TREE_DIGEST_JSON}\",\n    \"source_file_count\": ${source_file_count}\n  },\n  \"toolchain\": {\n    \"build_type\": \"${BUILD_TYPE_JSON}\",\n    \"compiler_id\": \"${CXX_COMPILER_ID_JSON}\",\n    \"compiler_version\": \"${CXX_COMPILER_VERSION_JSON}\",\n    \"compiler_path\": \"${CXX_COMPILER_PATH_JSON}\",\n    \"compiler_abi\": \"${COMPILER_ABI_JSON}\",\n    \"os\": \"${OS_NAME_JSON}\",\n    \"os_version\": \"${OS_VERSION_JSON}\",\n    \"architecture\": \"${ARCHITECTURE_JSON}\"\n  },\n  \"dependencies\": {\n    \"opencv\": \"${OPENCV_VERSION_JSON}\",\n    \"cuda\": \"${CUDA_VERSION_JSON}\"\n  },\n  \"feature_flags\": \"${FEATURE_FLAGS_JSON}\",\n  \"ci\": {\n    \"github_repository\": \"${GITHUB_REPOSITORY_JSON}\",\n    \"github_run_id\": \"${GITHUB_RUN_ID_JSON}\",\n    \"github_run_attempt\": \"${GITHUB_RUN_ATTEMPT_JSON}\",\n    \"github_run_url\": \"${GITHUB_RUN_URL_JSON}\"\n  },\n  \"reproducibility\": {\n    \"source_date_epoch\": \"${SOURCE_DATE_EPOCH_JSON}\",\n    \"build_timestamp_utc\": \"${BUILD_TIMESTAMP_UTC_JSON}\"\n  }\n}\n")

get_filename_component(output_header_dir "${OUTPUT_HEADER}" DIRECTORY)
get_filename_component(output_json_dir "${OUTPUT_JSON}" DIRECTORY)
file(MAKE_DIRECTORY "${output_header_dir}" "${output_json_dir}")
write_if_different("${OUTPUT_HEADER}" "${header_content}")
write_if_different("${OUTPUT_JSON}" "${json_content}")
