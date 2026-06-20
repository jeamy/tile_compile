#pragma once

#include <crow.h>
#include <nlohmann/json.hpp>
#include <yaml-cpp/yaml.h>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "app_state.hpp"

namespace tile_compile::routes {

/// @brief Creates a JSON response with the given status code.
/// @param j JSON payload
/// @param status HTTP status code (default: 200)
/// @return Crow response object
crow::response json_resp(const nlohmann::json& j, int status = 200);

/// @brief Creates an error response with a message.
/// @param msg Error message
/// @param status HTTP status code (default: 400)
/// @return Crow response object
crow::response err_resp(const std::string& msg, int status = 400);

/// @brief Creates an error response with code and message.
/// @param code Error code
/// @param msg Error message
/// @param status HTTP status code (default: 400)
/// @return Crow response object
crow::response err_resp(const std::string& code, const std::string& msg, int status = 400);

/// @brief Creates an error response with code, message, and details JSON.
/// @param code Error code
/// @param msg Error message
/// @param status HTTP status code
/// @param details Additional JSON details
/// @return Crow response object
crow::response err_resp(const std::string& code, const std::string& msg, int status, const nlohmann::json& details);

/// @brief Validates that a path exists and is within the allowed directory.
/// @param state Application state
/// @param path Path to validate
/// @return Response if invalid, empty optional if valid
std::optional<crow::response> validate_path(const std::shared_ptr<AppState>& state,
                                           const std::string& path);

/// @brief Validates and resolves a filesystem path, modifying it in-place.
/// @param state Application state
/// @param path Path to validate (modified in-place to resolved path)
/// @param must_exist If true, path must exist on filesystem
/// @return Response if invalid, empty optional if valid
std::optional<crow::response> validate_path(const std::shared_ptr<AppState>& state,
                                           std::filesystem::path& path,
                                           bool must_exist = false);

/// @brief Validates and resolves a filesystem path with a labeled error message.
/// @param state Application state
/// @param path Path to validate (modified in-place to resolved path)
/// @param label Human-readable field name for error messages
/// @param must_exist If true, path must exist on filesystem
/// @return Response if invalid, empty optional if valid
std::optional<crow::response> validate_path(const std::shared_ptr<AppState>& state,
                                           std::filesystem::path& path,
                                           const std::string& label,
                                           bool must_exist = false);

/// @brief Safely parses integer parameter from request.
/// @param req Crow request
/// @param param_name Parameter name
/// @param default_value Default value if parsing fails
/// @return Parsed integer or default
int parse_int_param(const crow::request& req, const std::string& param_name, int default_value = 0);

/// @brief Safely parses string parameter from request.
/// @param req Crow request
/// @param param_name Parameter name
/// @param default_value Default value if parameter is missing
/// @return Parameter value or default
std::string parse_string_param(const crow::request& req, const std::string& param_name, const std::string& default_value = "");

/// @brief Parses a JSON string safely, returning nullopt on parse failure.
/// @param raw JSON text to parse
/// @return Parsed JSON, or nullopt if parsing fails
std::optional<nlohmann::json> parse_json_string(const std::string& raw);

/// @brief Converts a nlohmann::json value to a YAML::Node.
/// @details Floats are rounded to 4 significant digits to remove floating-point noise.
/// @param value JSON value to convert
/// @return Equivalent YAML node
YAML::Node json_to_yaml_node(const nlohmann::json& value);

/// @brief Dumps a JSON value as a YAML string.
/// @details Uses json_to_yaml_node internally, with float precision set to 6.
/// @param value JSON value to dump
/// @return YAML string representation
std::string yaml_dump(const nlohmann::json& value);

/// @brief Sets a nested JSON value at a dotted path (e.g. "a.b.c").
/// @param root JSON object to modify
/// @param dotted_path Dot-separated path
/// @param value Value to set
void set_dotted(nlohmann::json& root, const std::string& dotted_path, const nlohmann::json& value);

/// @brief Safely parses a Crow request body as JSON.
/// @param req Crow request
/// @return Parsed JSON object, or nullopt if parsing fails
std::optional<nlohmann::json> parse_body(const crow::request& req);

/// @brief Creates a standardized error response for backend command failures.
/// @param message Error message
/// @param result Subprocess result with exit code and output
/// @return Crow error response with BACKEND_COMMAND_FAILED code
crow::response backend_command_failed(const std::string& message, const SubprocessResult& result);

/// @brief Reads a file's contents into a string.
/// @param path File path to read
/// @return File contents, or empty string on failure
std::string read_file_str(const std::filesystem::path& path);

/// @brief Writes a string to a file, creating parent directories if needed.
/// @param path File path to write
/// @param text Content to write
/// @return True on success, false on failure
bool write_file_str(const std::filesystem::path& path, const std::string& text);

/// @brief Sorts a vector in-place and removes duplicate elements.
/// @param v Vector to deduplicate
template <typename T>
void sort_unique_inplace(std::vector<T>& v) {
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());
}

/// @brief Converts a YAML::Node to nlohmann::json.
/// @param node YAML node
/// @return Equivalent JSON value
nlohmann::json yaml_to_json(const YAML::Node& node);

/// @brief Parses a YAML string into JSON.
/// @param yaml_text YAML text
/// @return Parsed JSON, or nullopt on parse error
std::optional<nlohmann::json> parse_yaml_text(const std::string& yaml_text);

/// @brief Reads a YAML file and converts it to JSON.
/// @param path Filesystem path
/// @return Parsed JSON, or nullopt if the file cannot be read or parsed
std::optional<nlohmann::json> parse_yaml_file(const std::filesystem::path& path);

/// @brief Creates a job, sets its state to running, and spawns a detached thread running `task(state, job_id)`.
/// @details Centralizes the boilerplate repeated by every route that launches a background job.
/// The task is responsible for all state updates and exception handling.
/// @return The newly created job id.
template <typename F>
std::string spawn_job_thread(std::shared_ptr<AppState> state,
                             const std::string& job_type,
                             const std::string& run_id,
                             const nlohmann::json& initial_data,
                             F&& task) {
    std::string job_id = state->job_store.create(job_type, run_id);
    state->job_store.update_state(job_id, JobState::running, initial_data);
    std::thread([state, job_id, task = std::forward<F>(task)]() mutable {
        task(state, job_id);
    }).detach();
    return job_id;
}

/// @brief Creates a job, sets its state to running, and runs a task that returns a JSON result.
/// @details The helper translates the result into `ok` or `error` state and catches exceptions.
/// The returned JSON should contain an `ok` boolean; on error it may contain an `error` string.
/// @return The newly created job id.
template <typename F>
std::string spawn_job(std::shared_ptr<AppState> state,
                      const std::string& job_type,
                      const std::string& run_id,
                      const nlohmann::json& initial_data,
                      F&& task) {
    std::string job_id = state->job_store.create(job_type, run_id);
    state->job_store.update_state(job_id, JobState::running, initial_data);
    std::thread([state, job_id, task = std::forward<F>(task)]() mutable {
        try {
            nlohmann::json result = task();
            if (result.value("ok", false)) {
                state->job_store.update_state(job_id, JobState::ok, result);
            } else {
                state->job_store.update_state(job_id, JobState::error, result,
                    result.value("error", std::string("job failed")));
            }
        } catch (const std::exception& e) {
            state->job_store.update_state(job_id, JobState::error, {
                {"ok", false},
                {"error", e.what()}
            }, e.what());
        } catch (...) {
            state->job_store.update_state(job_id, JobState::error, {
                {"ok", false},
                {"error", "unknown job error"}
            }, "unknown job error");
        }
    }).detach();
    return job_id;
}

} // namespace tile_compile::routes
