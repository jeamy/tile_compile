#pragma once

#include <stdexcept>
#include <string>

namespace tile_compile {

class TileCompileError : public std::runtime_error {
public:
    explicit TileCompileError(const std::string& message)
        : std::runtime_error(message) {}
};

class ConfigError : public TileCompileError {
public:
    explicit ConfigError(const std::string& message)
        : TileCompileError("Config error: " + message) {}
};

class ValidationError : public TileCompileError {
public:
    explicit ValidationError(const std::string& message)
        : TileCompileError("Validation error: " + message) {}
};

class IOError : public TileCompileError {
public:
    explicit IOError(const std::string& message)
        : TileCompileError("I/O error: " + message) {}
};

class FitsError : public IOError {
public:
    explicit FitsError(const std::string& message)
        : IOError("FITS error: " + message) {}
};

class RegistrationError : public TileCompileError {
public:
    explicit RegistrationError(const std::string& message)
        : TileCompileError("Registration error: " + message) {}
};

class PipelineError : public TileCompileError {
public:
    explicit PipelineError(const std::string& message)
        : TileCompileError("Pipeline error: " + message) {}
};

class StopRequested : public TileCompileError {
public:
    StopRequested() : TileCompileError("Stop requested by user") {}
};

} // namespace tile_compile
