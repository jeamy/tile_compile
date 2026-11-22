#pragma once

// Test-only CUDA parity harness (not part of tile_compile_lib).
//
// forward_drizzle_cuda_polygon_rect_area_batch evaluates the exact
// square-droplet vs. output-cell overlap area (Sutherland-Hodgman convex clip
// + shoelace) ON THE DEVICE for a batch of (convex quad, axis-aligned
// rectangle) pairs, through the same device code every production scatter
// kernel uses (src/reconstruction/forward_drizzle_cuda_clip.cuh). The parity
// test holds it bit for bit against the CPU
// polygon_rectangle_intersection_area().
//   quad_xy  : n*8 doubles  --- x0,y0, x1,y1, x2,y2, x3,y3  per quad
//   rect     : n*4 doubles  --- rx0,ry0, rx1,ry1            per cell
//   out_area : n doubles    --- caller-allocated
// Returns false (out_area untouched) on a CUDA-free build, no usable device,
// or any CUDA error; true on success.

namespace tile_compile::reconstruction {

#if TILE_COMPILE_WITH_CUDA
bool forward_drizzle_cuda_polygon_rect_area_batch(const double *quad_xy,
                                                  const double *rect, int n,
                                                  double *out_area);
#else
inline bool forward_drizzle_cuda_polygon_rect_area_batch(const double *,
                                                         const double *, int,
                                                         double *) {
  return false;
}
#endif

}  // namespace tile_compile::reconstruction
