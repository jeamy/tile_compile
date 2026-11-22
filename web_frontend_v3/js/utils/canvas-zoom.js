// js/utils/canvas-zoom.js – wheel zoom about the pointer for the canvas previews.

/**
 * The view after one wheel step: `scale` grows or shrinks (clamped to
 * [0.1, 20]) and the offsets keep the image point under the pointer in place.
 * `rect` is the canvas's client rect, `width`/`height` its pixel size.
 */
export function wheelZoom({ scale, offsetX, offsetY }, event, rect, width, height) {
  const next = Math.max(0.1, Math.min(20, scale * (event.deltaY < 0 ? 1.15 : 0.87)));
  const x = ((event.clientX - rect.left) * width) / rect.width;
  const y = ((event.clientY - rect.top) * height) / rect.height;
  return {
    scale: next,
    offsetX: x - ((x - offsetX) * next) / scale,
    offsetY: y - ((y - offsetY) * next) / scale,
  };
}
