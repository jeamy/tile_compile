// js/state/store.js – Minimaler Pub/Sub State-Manager

const PERSIST_LOCAL = new Set(["ui-state", "input-scan", "ai-config", "ai-state", "scan-state", "run-state", "config-state", "raw-stack", "astrometry", "pcc", "run-history", "run-chat", "feature-flags", "assistant-dock"]);

const stores = new Map();

function createStore(key, initialState = {}) {
  let state = { ...initialState };
  const listeners = new Set();

  if (PERSIST_LOCAL.has(key)) {
    try {
      const saved = localStorage.getItem(`gui3.${key}`);
      if (saved) state = { ...state, ...JSON.parse(saved) };
    } catch {
      // Storage blocked or corrupt entry: start from the defaults.
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
        try {
          localStorage.setItem(`gui3.${key}`, JSON.stringify(state));
        } catch (e) {
          // QuotaExceededError or blocked storage: keep the in-memory state.
          console.warn(`Could not persist gui3.${key}:`, e?.name || e);
        }
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

