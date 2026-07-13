#include "tile_compile/reconstruction/aqmh_pipeline_overlap.hpp"
#include "tile_compile/metrics/aqmh_quality_map_cache.hpp"

#include <condition_variable>
#include <exception>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace tile_compile::reconstruction {

namespace {

// RAII wrapper to join threads on destruction
class ThreadJoiner {
public:
  explicit ThreadJoiner(std::vector<std::thread> &threads) : threads_(threads) {}
  ~ThreadJoiner() {
    for (auto &t : threads_) {
      if (t.joinable()) t.join();
    }
  }
  ThreadJoiner(const ThreadJoiner &) = delete;
  ThreadJoiner &operator=(const ThreadJoiner &) = delete;

private:
  std::vector<std::thread> &threads_;
};

} // namespace

struct AqmhPrefetchCoordinator::Impl {
  explicit Impl(size_t frame_count, metrics::QualityMapCache* q_map_cache)
      : q_map_cache_(q_map_cache), published_count_(0), finished_(false),
        shutdown_(false), error_(nullptr), prefetched_count_(0) {
    // Create worker threads
    const unsigned num_workers = std::min(4u, std::thread::hardware_concurrency());
    workers_.reserve(num_workers);
    for (unsigned i = 0; i < num_workers; ++i) {
      workers_.emplace_back(&Impl::worker, this);
    }
    joiner_.reset(new ThreadJoiner(workers_));
  }

  ~Impl() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      shutdown_ = true;
    }
    cv_.notify_all();
    // ThreadJoiner handles the actual joining
  }

  void worker() {
    while (true) {
      size_t fi = static_cast<size_t>(-1);
      {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] {
          return !queue_.empty() || shutdown_ || (finished_ && queue_.empty());
        });
        if (shutdown_ && queue_.empty()) return;
        if (!queue_.empty()) {
          fi = queue_.front();
          queue_.pop();
        } else if (finished_) {
          // No more work, but check if we need to exit
          return;
        }
      }

      if (fi == static_cast<size_t>(-1)) continue;

      try {
        // Load the Q-map into the LRU cache via read_cached()
        // This populates the resident_ cache
        if (q_map_cache_) {
          (void)q_map_cache_->read_cached(fi);
        }
        {
          std::lock_guard<std::mutex> lock(mutex_);
          ++prefetched_count_;
        }
        cv_.notify_all();
      } catch (...) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!error_) {
          error_.reset(new std::exception_ptr(std::current_exception()));
        }
        cv_.notify_all();
      }
    }
  }

  void publish_frame(size_t fi) {
    if (finished_) return; // No more frames accepted after finish()
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (shutdown_) return;
      queue_.push(fi);
      ++published_count_;
    }
    cv_.notify_one();
  }

  void finish() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      finished_ = true;
    }
    cv_.notify_all();
  }

  void wait_all_prefetched() {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this] {
      return (prefetched_count_ >= published_count_ && finished_) ||
             shutdown_ || error_ != nullptr;
    });
    // If there was an error, disable prefetch but don't throw
    // The reconstruction will continue through the normal read path
  }

  bool prefetch_active() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return error_ == nullptr;
  }

  size_t prefetched_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return prefetched_count_;
  }

  std::mutex mutable mutex_;
  std::condition_variable cv_;
  std::queue<size_t> queue_;
  metrics::QualityMapCache* q_map_cache_;
  size_t published_count_;
  size_t prefetched_count_;
  bool finished_;
  bool shutdown_;
  std::unique_ptr<std::exception_ptr> error_;
  std::vector<std::thread> workers_;
  std::unique_ptr<ThreadJoiner> joiner_;
};

AqmhPrefetchCoordinator::AqmhPrefetchCoordinator(
    size_t frame_count, metrics::QualityMapCache* q_map_cache)
    : impl_(std::make_unique<Impl>(frame_count, q_map_cache)) {}

AqmhPrefetchCoordinator::~AqmhPrefetchCoordinator() = default;

void AqmhPrefetchCoordinator::publish_frame(size_t fi) {
  impl_->publish_frame(fi);
}

void AqmhPrefetchCoordinator::finish() {
  impl_->finish();
}

void AqmhPrefetchCoordinator::wait_all_prefetched() {
  impl_->wait_all_prefetched();
}

bool AqmhPrefetchCoordinator::prefetch_active() const {
  return impl_->prefetch_active();
}

size_t AqmhPrefetchCoordinator::prefetched_count() const {
  return impl_->prefetched_count();
}

} // namespace tile_compile::reconstruction
