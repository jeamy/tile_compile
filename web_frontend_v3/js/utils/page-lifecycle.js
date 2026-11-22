// js/utils/page-lifecycle.js – cleanup hooks for the page shown in #content.
//
// main.js replaces the page on every tab or sub-tab change. A page that starts
// timers, sockets or global listeners registers their teardown here; it runs
// once, right before the next page replaces it.

let cleanups = [];

/** Register `fn` to run when the current page is replaced. */
export function onPageUnmount(fn) {
  if (typeof fn === "function") cleanups.push(fn);
}

/** Run and forget every registered cleanup; errors are reported, not thrown. */
export function runPageUnmount() {
  const pending = cleanups;
  cleanups = [];
  for (const fn of pending) {
    try {
      fn();
    } catch (e) {
      console.error("Page cleanup failed:", e);
    }
  }
}
