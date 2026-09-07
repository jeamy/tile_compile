#pragma once

// CFA-forward-drizzle CUDA path --- milestone M7 (plan section 19).
//
// M7 splits into slices. THIS slice ships only the transactional restart
// contract of plan 19.4:
//
//   * chunks write exclusively to a phase-local, uncommitted profile-store
//     generation (StoreWriter already discards an unpublished generation in
//     its destructor);
//   * if the CUDA path fails in ANY chunk after its allowed retries, the
//     whole FORWARD_DRIZZLE phase restarts on the CPU reference path and only
//     a fully computed, validated, hashed CPU result is committed --- never a
//     mixed CPU/CUDA image or a half-accumulated pixel.
//
// The droplet / clipping / profile-accumulation kernels (plan 19.2 stages
// 3--7) and their parity matrix (plan 19.5) are a LATER slice. Until then
// `forward_drizzle_cuda_runtime_available()` is false and every "attempt CUDA"
// resolves to an immediate ForwardDrizzleCudaError, which the caller turns
// into a clean CPU run.

#include <cstddef>
#include <functional>
#include <stdexcept>

namespace tile_compile::reconstruction {

// Thrown when the CUDA forward-drizzle path cannot complete a chunk (real
// device/allocation failure, or an injected fault). The caller MUST discard
// any uncommitted profile-store generation and restart the ENTIRE
// FORWARD_DRIZZLE phase on the CPU reference path (plan 19.4). It is never a
// partial-commit or a per-chunk fallback.
struct ForwardDrizzleCudaError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

// True iff this binary has a usable custom forward-drizzle CUDA path. Slice 1
// has no kernels, so this is always false; slice 2 makes it probe
// TILE_COMPILE_WITH_CUDA && a present device.
bool forward_drizzle_cuda_runtime_available();

// Test-only fault injection for the plan-19.4 restart contract. When set to
// n >= 0, an attempted CUDA persist throws ForwardDrizzleCudaError after n
// committed stripes (n == 0 => before the first stripe, i.e. an immediate
// failure). -1 (the default) disables injection. Process-global; a test that
// sets it must reset it. Also honoured from the environment variable
// TILE_COMPILE_FORWARD_DRIZZLE_CUDA_FAULT_AFTER_CHUNKS at first read.
void set_forward_drizzle_cuda_fault_after_chunks(int n);
int forward_drizzle_cuda_fault_after_chunks();

// Options threaded into persist_forward_drizzle_multiband to request the CUDA
// path. `attempt == false` is the plain CPU reference path (default).
struct ForwardDrizzleCudaOptions {
  bool attempt = false;
};

// --- Plan 19.4: device memory + auto-chunking ------------------------------

// Free / total bytes of the active CUDA device. {0, 0} when no usable device
// (or the binary was built without CUDA). Slice-2 device probe.
struct CudaDeviceMemory {
  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
};
CudaDeviceMemory forward_drizzle_cuda_device_memory();

// The plan-19.4 chunking decision: an initial chunk height plus a bounded
// retry ladder that halves the height on each CUDA allocation failure, down
// to `min_chunk_rows`. A safety margin (`reserve_bytes`) is held back for the
// driver / OpenCV before any chunk is sized.
struct CudaChunkPlan {
  int chunk_rows = 0;         // initial attempt
  int min_chunk_rows = 1;     // floor; below this the phase restarts on CPU
  int max_retries = 0;        // halvings available from chunk_rows to the floor
  std::size_t reserve_bytes = 0;  // margin withheld for driver / OpenCV
  std::size_t usable_bytes = 0;   // free_bytes - reserve_bytes (0 if underwater)
  std::size_t bytes_per_row = 0;  // the working-set estimate that was used
  bool feasible = false;     // false => not enough memory for even one row
};

// Pure arithmetic (no device calls): pick the chunk plan from a free-memory
// figure and a per-output-row working-set estimate. `requested_chunk_rows > 0`
// caps the initial height (a config ceiling); 0 means "as large as fits".
// `reserve_fraction` of free memory is held back, but never less than
// `reserve_floor_bytes`. Deterministic and unit-tested without a GPU.
CudaChunkPlan plan_cuda_chunking(std::size_t free_bytes,
                                 std::size_t bytes_per_row, int image_rows,
                                 int requested_chunk_rows = 0,
                                 double reserve_fraction = 0.20,
                                 std::size_t reserve_floor_bytes =
                                     static_cast<std::size_t>(256) << 20);

// A per-chunk device allocation failed: the driver of run_cuda_chunked() then
// halves the chunk height and retries. Any OTHER exception from a chunk
// processor is a hard failure --- run_cuda_chunked lets it propagate and the
// caller restarts FORWARD_DRIZZLE on the CPU reference path (plan 19.4).
struct CudaAllocFailure : std::runtime_error {
  using std::runtime_error::runtime_error;
};

// Process one output-row band [y0, y0 + rows). Throw CudaAllocFailure to ask
// for a smaller band; return normally on success.
using CudaChunkProcessor = std::function<void(int y0, int rows)>;

// --- Plan 19.2 stage 3/5 kernel building block ----------------------------

// The exact square-droplet vs. output-cell overlap area (Sutherland-Hodgman
// convex clip + shoelace), evaluated ON THE DEVICE for a batch of
// (convex quad, axis-aligned rectangle) pairs. Ported 1:1 from the CPU
// polygon_rectangle_intersection_area() so the parity matrix (plan 19.5) can
// pin CPU == CUDA on the numerically hardest piece before the full rasterizer
// is wired into persist_forward_drizzle_multiband.
//   quad_xy  : n*8 doubles  --- x0,y0, x1,y1, x2,y2, x3,y3  per quad
//   rect     : n*4 doubles  --- rx0,ry0, rx1,ry1            per cell
//   out_area : n doubles    --- caller-allocated
// Returns false (out_area untouched) on a CUDA-free build, no usable device,
// or any CUDA error; true on success.
bool forward_drizzle_cuda_polygon_rect_area_batch(const double *quad_xy,
                                                  const double *rect, int n,
                                                  double *out_area);

// The exact affine square-droplet corner map, on the device, for a batch of
// native-source samples under ONE 2x3 affine (source->canvas, row-major
// a00,a01,a02, a10,a11,a12). Per sample: the square
// [sx-half, sx+half] x [sy-half, sy+half] is mapped through the affine and then
// scaled by `internal_scale` (native-canvas -> internal-canvas), giving the 4
// leaf corners in CCW order matching build_affine_leaf() + to_internal().
//   sample_xy    : n*2 doubles (sx, sy)
//   out_corners  : n*8 doubles (x0,y0, x1,y1, x2,y2, x3,y3), caller-allocated
// Returns false (out untouched) on a CUDA-free build, no device, or CUDA error.
bool forward_drizzle_cuda_affine_leaf_corners_batch(const double affine6[6],
                                                    int internal_scale,
                                                    double half,
                                                    const double *sample_xy,
                                                    int n, double *out_corners);

// Plan-19.4 chunk driver (host-only, no device calls of its own). Walks the
// image in bands of `plan.chunk_rows`; on CudaAllocFailure it halves the
// CURRENT band height and retries the SAME band, down to `plan.min_chunk_rows`.
// If a band still cannot be processed at the floor height, every temporary
// CUDA store is void and the whole phase must restart on CPU --- signalled by
// throwing ForwardDrizzleCudaError. Returns the number of bands committed.
// `plan.feasible` must be true.
int run_cuda_chunked(const CudaChunkPlan &plan, int image_rows,
                     const CudaChunkProcessor &process);

}  // namespace tile_compile::reconstruction
