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

}  // namespace tile_compile::reconstruction
