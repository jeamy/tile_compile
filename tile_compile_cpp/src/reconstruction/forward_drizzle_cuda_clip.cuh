// CFA forward drizzle --- device polygon clip shared by the production
// scatter/gather kernels (forward_drizzle_cuda_device.cu) and the parity
// harness TU (forward_drizzle_cuda_test_harness.cu). One copy only: every
// kernel that computes a droplet/cell overlap area includes this header.
//
// 1:1 device port of the CPU polygon_rectangle_intersection_area math
// (src/reconstruction/forward_drizzle.cpp). Convex quad, clipped Sutherland-
// Hodgman against the 4 axis-aligned half-planes of [rx0,rx1] x [ry0,ry1],
// then shoelace. Static bound 8 = 4 quad vertices + <=1 new vertex per edge
// per plane, matching the CPU comment. Must be compiled with --fmad=false
// (see CMakeLists.txt) so every fp64 expression rounds like the
// -ffp-contract=off CPU reference.
#pragma once

#if TILE_COMPILE_WITH_CUDA

#include <cuda_runtime.h>

namespace tile_compile::reconstruction {
namespace fd_cuda_clip {

__device__ __forceinline__ double d_shoelace_area(const double *x,
                                                  const double *y, int n) {
  if (n < 3) return 0.0;
  double s = 0.0;
  for (int i = 0; i < n; ++i) {
    const int j = (i + 1 == n) ? 0 : i + 1;
    s += x[i] * y[j] - x[j] * y[i];
  }
  return fabs(s) * 0.5;
}

// plane: 0 => x>=c, 1 => x<=c, 2 => y>=c, 3 => y<=c
__device__ __forceinline__ int d_clip_plane(const double *ix_,
                                            const double *iy_, int in_n,
                                            int plane, double c, double *ox,
                                            double *oy) {
  if (in_n == 0) return 0;
  int out_n = 0;
  for (int i = 0; i < in_n; ++i) {
    const int j = (i + 1 == in_n) ? 0 : i + 1;
    const double xi = ix_[i], yi = iy_[i], xj = ix_[j], yj = iy_[j];
    bool in_i, in_j;
    switch (plane) {
      case 0:  in_i = xi >= c; in_j = xj >= c; break;
      case 1:  in_i = xi <= c; in_j = xj <= c; break;
      case 2:  in_i = yi >= c; in_j = yj >= c; break;
      default: in_i = yi <= c; in_j = yj <= c; break;
    }
    if (in_i) { ox[out_n] = xi; oy[out_n] = yi; ++out_n; }
    if (in_i != in_j) {
      double px, py;
      if (plane <= 1) {  // vertical plane at x == c
        const double t = (c - xi) / (xj - xi);
        px = c;
        py = yi + t * (yj - yi);
      } else {           // horizontal plane at y == c
        const double t = (c - yi) / (yj - yi);
        px = xi + t * (xj - xi);
        py = c;
      }
      ox[out_n] = px; oy[out_n] = py; ++out_n;
    }
  }
  return out_n;
}

// Overlap area of the convex quad (qx, qy) with [rx0,rx1] x [ry0,ry1].
//
// Two bit-exact early-outs precede the full 4-plane clip:
//  (a) every vertex inside the rectangle (closed): each clip plane is a pure
//      pass-through (in_i && in_j for every edge, no intersection vertex),
//      so the full path runs the shoelace over the original 4 vertices in
//      the original order --- computed here directly. NaN coordinates fail
//      the comparisons and take the full path, as before.
//  (b) every vertex strictly outside one half-plane. For x < rx0 the FIRST
//      clip already yields the empty polygon (0.0); for x > rx1 plane 0 is a
//      pass-through (x > rx1 >= rx0) and plane 1 empties it. For the y
//      planes the x clips run first and may create intersection vertices
//      py = yi + t*(yj - yi); they stay inside [min y, max y] (and thus
//      strictly outside the y half-plane) when yj - yi is exact, i.e. when
//      all y share a sign and lie within a factor of two (Sterbenz), which
//      the guard below requires --- otherwise the full path runs. The
//      bbox-driven scatter kernels never visit a cell for which (b) holds
//      (the cell lies outside the quad's floor/ceil bbox); the coverage
//      gather, which scans a conservative source neighbourhood per cell,
//      hits it often.
__device__ __forceinline__ double d_polygon_rect_area(const double *qx,
                                                      const double *qy,
                                                      double rx0, double ry0,
                                                      double rx1,
                                                      double ry1) {
  const double x0 = qx[0], x1 = qx[1], x2 = qx[2], x3 = qx[3];
  const double y0 = qy[0], y1 = qy[1], y2 = qy[2], y3 = qy[3];
  if (x0 >= rx0 && x1 >= rx0 && x2 >= rx0 && x3 >= rx0 &&
      x0 <= rx1 && x1 <= rx1 && x2 <= rx1 && x3 <= rx1 &&
      y0 >= ry0 && y1 >= ry0 && y2 >= ry0 && y3 >= ry0 &&
      y0 <= ry1 && y1 <= ry1 && y2 <= ry1 && y3 <= ry1)
    return d_shoelace_area(qx, qy, 4);
  // The pass-through arguments for the far planes need a proper rectangle.
  if (rx0 <= rx1 && ry0 <= ry1) {
    if ((x0 < rx0 && x1 < rx0 && x2 < rx0 && x3 < rx0) ||
        (x0 > rx1 && x1 > rx1 && x2 > rx1 && x3 > rx1))
      return 0.0;
    const double ymin = fmin(fmin(y0, y1), fmin(y2, y3));
    const double ymax = fmax(fmax(y0, y1), fmax(y2, y3));
    const bool sterbenz = (ymin > 0.0 && ymax <= 2.0 * ymin) ||
                          (ymax < 0.0 && ymin >= 2.0 * ymax);
    if (sterbenz && (ymax < ry0 || ymin > ry1)) return 0.0;
  }
  double ax[8], ay[8], bx[8], by[8];
  int n = 4;
  for (int i = 0; i < 4; ++i) {
    ax[i] = qx[i];
    ay[i] = qy[i];
  }
  n = d_clip_plane(ax, ay, n, 0, rx0, bx, by);
  n = d_clip_plane(bx, by, n, 1, rx1, ax, ay);
  n = d_clip_plane(ax, ay, n, 2, ry0, bx, by);
  n = d_clip_plane(bx, by, n, 3, ry1, ax, ay);
  return d_shoelace_area(ax, ay, n);
}

}  // namespace fd_cuda_clip
}  // namespace tile_compile::reconstruction

#endif  // TILE_COMPILE_WITH_CUDA
