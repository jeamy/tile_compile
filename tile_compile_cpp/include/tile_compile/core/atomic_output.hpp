#pragma once

#include <atomic>
#include <chrono>
#include <filesystem>
#include <stdexcept>
#ifndef _WIN32
#include <fcntl.h>
#include <unistd.h>
#endif

namespace tile_compile::core {

// Owns only its uniquely created staging directory. A failed writer never
// truncates the published artifact. Commit requires the writer to be closed.
class AtomicOutput {
  std::filesystem::path target_, directory_, staged_;

public:
  explicit AtomicOutput(const std::filesystem::path &target) : target_(target) {
    static std::atomic<unsigned long long> sequence{0};
    for (int retry = 0; retry < 64; ++retry) {
      const auto id =
          std::chrono::steady_clock::now().time_since_epoch().count();
      auto candidate = target.parent_path() /
                       (target.filename().string() + ".stage-" +
                        std::to_string(id) + "-" + std::to_string(sequence++));
      std::error_code ec;
      if (std::filesystem::create_directory(candidate, ec)) {
        directory_ = candidate;
        staged_ = directory_ / "payload";
        return;
      }
      if (ec)
        throw std::runtime_error("Cannot create artifact staging directory: " +
                                 ec.message());
    }
    throw std::runtime_error("Cannot reserve artifact staging directory");
  }
  AtomicOutput(const AtomicOutput &) = delete;
  AtomicOutput &operator=(const AtomicOutput &) = delete;
  ~AtomicOutput() {
    std::error_code ec;
    if (!staged_.empty())
      std::filesystem::remove(staged_, ec);
    if (!directory_.empty())
      std::filesystem::remove(directory_, ec);
  }
  const std::filesystem::path &path() const { return staged_; }
  void commit() {
#ifndef _WIN32
    int fd = ::open(staged_.c_str(), O_RDONLY);
    if (fd < 0)
      throw std::runtime_error("Cannot open staged artifact for sync");
    const int status = ::fsync(fd);
    ::close(fd);
    if (status)
      throw std::runtime_error("Cannot sync staged artifact");
#endif
    std::filesystem::rename(staged_, target_);
#ifndef _WIN32
    const auto parent = target_.parent_path().empty()
                            ? std::filesystem::path(".")
                            : target_.parent_path();
    fd = ::open(parent.c_str(), O_RDONLY | O_DIRECTORY);
    if (fd < 0)
      throw std::runtime_error("Cannot open artifact directory for sync");
    const int dir_status = ::fsync(fd);
    ::close(fd);
    if (dir_status)
      throw std::runtime_error("Cannot sync artifact directory");
#endif
  }
};

} // namespace tile_compile::core
