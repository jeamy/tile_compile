// js/state/scan-state.js – Scan-Results, KI-Analyse

import { getStore } from "./store.js";

const store = getStore("scan-state", {
  scanResult: null,
  loading: false,
  error: null,
});

export function getScanState() { return store.getState(); }
export function setScanState(patch) { store.setState(patch); }
