#pragma once
#include "crow/middlewares/cors.h"
#include "origin_guard.hpp"
#include <crow.h>

/// @brief Rejects cross-site requests before any route runs (origin_guard.hpp).
/// @details Crow does not run middlewares for WebSocket upgrades; ws_routes.cpp
/// checks the same rule in its onaccept handlers.
struct OriginGuardMiddleware {
    struct context {};
    std::string extra_origins = origin_guard::configured_extra_origins();

    void before_handle(crow::request& req, crow::response& res, context&) {
        if (origin_guard::origin_allowed(req.get_header_value("Origin"),
                                         req.get_header_value("Host"), extra_origins))
            return;
        res.code = 403;
        res.set_header("Content-Type", "application/json");
        res.write(R"({"error":{"code":"ORIGIN_NOT_ALLOWED","message":"cross-site request rejected","details":{}}})");
        res.end();
    }
    void after_handle(crow::request&, crow::response&, context&) {}
};

/// @brief Crow application type shared by all route registration modules.
using CrowApp = crow::App<OriginGuardMiddleware, crow::CORSHandler>;
