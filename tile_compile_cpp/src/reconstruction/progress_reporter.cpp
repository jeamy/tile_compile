#include "tile_compile/reconstruction/progress_reporter.hpp"

#include <thread>

namespace tile_compile::reconstruction {

struct ProgressReporter::Impl {
    std::thread thread;
};

ProgressReporter::ProgressReporter(int tiles_total,
                                   ProgressLogFn log_fn,
                                   double interval_s,
                                   double eta_warn_factor)
    : tiles_total_(tiles_total),
      log_fn_(std::move(log_fn)),
      interval_s_(interval_s),
      eta_warn_factor_(eta_warn_factor),
      start_time_(std::chrono::steady_clock::now()),
      impl_(std::make_unique<Impl>())
{
    impl_->thread = std::thread([this] { run_loop(); });
}

ProgressReporter::~ProgressReporter() {
    finish();
}

void ProgressReporter::tick() {
    ++tiles_completed_;
}

void ProgressReporter::set_workers_active(int n) {
    workers_active_.store(n);
}

void ProgressReporter::finish() {
    stop_.store(true);
    if (impl_ && impl_->thread.joinable()) impl_->thread.join();
}

void ProgressReporter::run_loop() {
    using clock = std::chrono::steady_clock;
    const auto interval = std::chrono::duration<double>(interval_s_);
    auto next_tick = start_time_ + interval;

    while (!stop_.load()) {
        std::this_thread::sleep_until(next_tick);
        next_tick += interval;
        if (stop_.load()) break;

        const int completed = tiles_completed_.load();
        const int active    = workers_active_.load();
        const double elapsed =
            std::chrono::duration<double>(clock::now() - start_time_).count();

        double eta = 0.0;
        if (completed > 0 && tiles_total_ > 0) {
            eta = elapsed / static_cast<double>(completed) *
                  static_cast<double>(tiles_total_ - completed);
        }

        // Warn once after 25 % of tiles if ETA looks excessive.
        if (!eta_warned_.load() &&
            tiles_total_ > 0 &&
            completed >= tiles_total_ / 4 &&
            eta > elapsed * eta_warn_factor_)
        {
            eta_warned_.store(true);
            if (log_fn_) {
                log_fn_("warning",
                        "TILE_RECONSTRUCTION ETA exceeds expected duration",
                        completed, tiles_total_, elapsed, eta, active);
            }
        }

        if (log_fn_) {
            log_fn_("phase_progress", "workers=" + std::to_string(active),
                    completed, tiles_total_, elapsed, eta, active);
        }
    }

    // Final entry.
    const int completed = tiles_completed_.load();
    const int active    = workers_active_.load();
    const double elapsed =
        std::chrono::duration<double>(clock::now() - start_time_).count();
    if (log_fn_) {
        log_fn_("phase_progress", "done",
                completed, tiles_total_, elapsed, 0.0, active);
    }
}

} // namespace tile_compile::reconstruction
