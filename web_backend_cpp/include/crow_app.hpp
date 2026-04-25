#pragma once
#include "crow/middlewares/cors.h"
#include <crow.h>

/// @brief Crow application type shared by all route registration modules.
using CrowApp = crow::App<crow::CORSHandler>;
