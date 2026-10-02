// js/state/feature-flags.js – Global on/off switches for AI and Jev.
//
// Both default to enabled (unchanged current behaviour for anyone who never touches the switches).
// Persisted in the browser (per-device); a fresh browser or a cleared site sees the defaults again.
// When a flag is off, the owning page hides every input field of its own settings card and every
// other place in the GUI that shows that source's recommendations or traffic, and stops requesting
// its data. Switching a flag off never deletes stored settings (mode, key, etc.); it only hides them.

import { getStore } from "./store.js";

const DEFAULT = { aiEnabled: true, jevEnabled: true };

const store = getStore("feature-flags", DEFAULT);

export function getFeatureFlags() {
  return store.getState();
}

export function setFeatureFlags(patch) {
  store.setState(patch);
}

export function onFeatureFlagsChange(fn) {
  return store.subscribe(fn);
}
