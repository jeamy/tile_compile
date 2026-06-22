// js/state/store.js – Minimaler Pub/Sub State-Manager

const PERSIST_LOCAL = new Set(["ui-state", "input-scan", "ai-config", "ai-state", "scan-state", "run-state", "config-state", "raw-stack", "astrometry", "pcc", "run-history"]);

const stores = new Map();

function createStore(key, initialState = {}) {
  let state = { ...initialState };
  const listeners = new Set();

  if (PERSIST_LOCAL.has(key)) {
    const saved = localStorage.getItem(`gui3.${key}`);
    if (saved) {
      try {
        state = { ...state, ...JSON.parse(saved) };
      } catch {}
    }
  }

  return {
    key,
    getState() {
      return { ...state };
    },
    setState(patch) {
      state = { ...state, ...patch };
      if (PERSIST_LOCAL.has(key)) {
        localStorage.setItem(`gui3.${key}`, JSON.stringify(state));
      }
      for (const fn of listeners) {
        try { fn(state); } catch (e) { console.error("Store listener error:", e); }
      }
    },
    subscribe(fn) {
      listeners.add(fn);
      return () => listeners.delete(fn);
    },
  };
}

export function getStore(key, initialState) {
  if (!stores.has(key)) {
    stores.set(key, createStore(key, initialState));
  }
  return stores.get(key);
}

export function getState(key) {
  return getStore(key).getState();
}

export function setState(key, patch) {
  getStore(key).setState(patch);
}

export function subscribe(key, fn) {
  return getStore(key).subscribe(fn);
}

// Server-UI-State sync
let _api = null;

export function setApi(api) {
  _api = api;
}

export async function loadServerUiState() {
  if (!_api) return {};
  try {
    return await _api.get("/api/app/ui-state");
  } catch {
    return {};
  }
}

export async function syncServerUiState(patch) {
  if (!_api) return;
  try {
    await _api.patch("/api/app/ui-state", patch);
  } catch (e) {
    console.error("Server UI state sync failed:", e);
  }
}
