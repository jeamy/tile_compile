#pragma once

#include <crow.h>
#include <nlohmann/json.hpp>
#include <memory>
#include <optional>
#include <string>

struct AppState;

namespace tile_compile::routes {

/// @brief Creates a JSON response with the given status code.
/// @param j JSON payload
/// @param status HTTP status code (default: 200)
/// @return Crow response object
crow::response json_resp(const nlohmann::json& j, int status = 200);

/// @brief Alias for json_resp for backward compatibility.
/// @param j JSON payload
/// @param status HTTP status code (default: 200)
/// @return Crow response object
crow::response json_response(const nlohmann::json& j, int status = 200);

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

} // namespace tile_compile::routes
