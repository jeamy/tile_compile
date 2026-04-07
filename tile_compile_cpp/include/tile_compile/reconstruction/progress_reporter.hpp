#pragma once

#include <atomic>
#include <chrono>
#include <cstddef>
#include <functional>
#include <memory>
#include <string>

namespace tile_compile::reconstruction {

/// Callback type used by ProgressReporter to emit log entries.
/// Arguments: event_type, substep, tiles_completed, tiles_total,
///            elapsed_s, eta_s, workers_active.
using ProgressLogFn = std::function<void(
    const std::string& event_type,
    const std::string& substep,
    int tiles_completed,
    int tiles_total,
    double elapsed_s,
    double eta_s,
    int workers_active)>;

/// Thread-safe progress reporter for TILE_RECONSTRUCTION.
///
/// Call tick() from worker threads after each tile completes.
/// A background thread emits progress log entries every @p interval_s seconds.
class ProgressReporter {
public:
    /// @param tiles_total      Total number of tiles to process (dead tiles excluded).
    /// @param log_fn           Callback invoked on the reporter thread.
    /// @param interval_s       Logging interval in seconds (default 60 s).
    /// @param eta_warn_factor  Warn when ETA after 25 % tiles exceeds this factor
    ///                         times the expected duration (default 3.0).
    explicit ProgressReporter(int tiles_total,
                              ProgressLogFn log_fn,
                              double interval_s    = 60.0,
                              double eta_warn_factor = 3.0);
    ~ProgressReporter();

    // Non-copyable, non-movable.
    ProgressReporter(const ProgressReporter&)            = delete;
    ProgressReporter& operator=(const ProgressReporter&) = delete;

    /// Called by a worker thread when one tile has been processed.
    void tick();

    /// Update the number of currently active workers (called by scheduler).
    void set_workers_active(int n);

    /// Flush a final progress entry and stop the background thread.
    void finish();

private:
    void run_loop();

    int                                    tiles_total_;
    ProgressLogFn                          log_fn_;
    double                                 interval_s_;
    double                                 eta_warn_factor_;
    std::atomic<int>                       tiles_completed_{0};
    std::atomic<int>                       workers_active_{0};
    std::atomic<bool>                      stop_{false};
    std::chrono::steady_clock::time_point  start_time_;
    std::atomic<bool>                      eta_warned_{false};

    // Background thread — declared last so it starts after all members are init.
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace tile_compile::reconstruction
