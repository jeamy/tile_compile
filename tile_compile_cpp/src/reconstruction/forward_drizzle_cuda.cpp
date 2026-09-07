#include "tile_compile/reconstruction/forward_drizzle_cuda.hpp"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <string>

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

#if !TILE_COMPILE_WITH_CUDA
// CUDA-free build: no device at all. The CUDA build defines these in
// forward_drizzle_cuda_device.cu against the real runtime.
bool forward_drizzle_cuda_runtime_available() { return false; }
CudaDeviceMemory forward_drizzle_cuda_device_memory() { return {}; }
bool forward_drizzle_cuda_polygon_rect_area_batch(const double *, const double *,
                                                  int, double *) {
  return false;
}
bool forward_drizzle_cuda_affine_leaf_corners_batch(const double *, int, double,
                                                    const double *, int,
                                                    double *) {
  return false;
}
bool forward_drizzle_cuda_affine_frame_contributions(
    const double *, int, double, int, int, int, int, int, int, int,
    const float *, int, int, int, bool, int, CudaDrizzleContribRecord *,
    long long, long long *out_written) {
  if (out_written) *out_written = 0;
  return false;
}
#endif

CudaChunkPlan plan_cuda_chunking(std::size_t free_bytes,
                                 std::size_t bytes_per_row, int image_rows,
                                 int requested_chunk_rows,
                                 double reserve_fraction,
                                 std::size_t reserve_floor_bytes) {
  CudaChunkPlan p;
  p.bytes_per_row = bytes_per_row;
  if (image_rows <= 0 || bytes_per_row == 0) return p;  // infeasible

  if (!(reserve_fraction >= 0.0)) reserve_fraction = 0.0;  // NaN-safe
  if (reserve_fraction > 0.9) reserve_fraction = 0.9;
  const std::size_t frac_reserve =
      static_cast<std::size_t>(static_cast<double>(free_bytes) * reserve_fraction);
  p.reserve_bytes = std::max(frac_reserve, reserve_floor_bytes);
  p.usable_bytes = (free_bytes > p.reserve_bytes) ? free_bytes - p.reserve_bytes : 0;

  const std::size_t rows_that_fit = p.usable_bytes / bytes_per_row;
  if (rows_that_fit == 0) return p;  // not enough for even one row -> infeasible
  p.feasible = true;

  int chunk = static_cast<int>(std::min<std::size_t>(
      rows_that_fit, static_cast<std::size_t>(image_rows)));
  if (requested_chunk_rows > 0) chunk = std::min(chunk, requested_chunk_rows);
  chunk = std::max(chunk, 1);
  p.chunk_rows = chunk;
  p.min_chunk_rows = 1;

  // Retry ladder: how many times can the height be halved and still be >= 1.
  int retries = 0;
  for (int h = chunk; h > 1; h /= 2) ++retries;
  p.max_retries = retries;
  return p;
}

int run_cuda_chunked(const CudaChunkPlan &plan, int image_rows,
                     const CudaChunkProcessor &process) {
  if (!plan.feasible || image_rows <= 0 || !process)
    throw ForwardDrizzleCudaError(
        "forward_drizzle CUDA: chunk plan not feasible");
  const int floor_rows = std::max(1, plan.min_chunk_rows);
  int committed = 0;
  int y0 = 0;
  while (y0 < image_rows) {
    int band = std::max(floor_rows,
                        std::min(plan.chunk_rows, image_rows - y0));
    if (band > image_rows - y0) band = image_rows - y0;
    bool done = false;
    while (!done) {
      try {
        process(y0, band);
        done = true;
      } catch (const CudaAllocFailure &) {
        if (band <= floor_rows)
          throw ForwardDrizzleCudaError(
              "forward_drizzle CUDA: allocation failed at the minimum chunk "
              "height (" + std::to_string(floor_rows) +
              " rows) after retries; restarting the phase on CPU");
        band = std::max(floor_rows, band / 2);
      }
      // Any other exception propagates: a hard CUDA failure -> CPU restart.
    }
    y0 += band;
    ++committed;
  }
  return committed;
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
