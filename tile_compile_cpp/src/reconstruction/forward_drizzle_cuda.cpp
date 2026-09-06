#include "tile_compile/reconstruction/forward_drizzle_cuda.hpp"

#include <atomic>
#include <cstdlib>

namespace tile_compile::reconstruction {

namespace {

// -1 = disabled. Initialised once from the environment, then overridable in
// process by set_forward_drizzle_cuda_fault_after_chunks() (tests).
std::atomic<int> g_fault_after_chunks{-2};  // -2 = "not yet read from env"

int read_env_fault() {
  const char *v = std::getenv("TILE_COMPILE_FORWARD_DRIZZLE_CUDA_FAULT_AFTER_CHUNKS");
  if (v == nullptr || *v == '\0') return -1;
  char *end = nullptr;
  const long n = std::strtol(v, &end, 10);
  if (end == v || *end != '\0' || n < 0 || n > 1'000'000) return -1;
  return static_cast<int>(n);
}

}  // namespace

bool forward_drizzle_cuda_runtime_available() {
  // Slice 1: the transactional-restart contract only --- no kernels yet, so
  // there is no usable CUDA path regardless of TILE_COMPILE_WITH_CUDA or a
  // present device. Slice 2 replaces this with a real device probe.
  return false;
}

void set_forward_drizzle_cuda_fault_after_chunks(int n) {
  g_fault_after_chunks.store(n < 0 ? -1 : n, std::memory_order_relaxed);
}

int forward_drizzle_cuda_fault_after_chunks() {
  int cur = g_fault_after_chunks.load(std::memory_order_relaxed);
  if (cur == -2) {
    const int from_env = read_env_fault();
    // Only the first reader installs the env value; a concurrent setter wins.
    int expected = -2;
    if (g_fault_after_chunks.compare_exchange_strong(expected, from_env,
                                                     std::memory_order_relaxed))
      return from_env;
    cur = g_fault_after_chunks.load(std::memory_order_relaxed);
  }
  return cur;
}

}  // namespace tile_compile::reconstruction
