// js/state/ai-state.js – KI-Config, Modelle, Auth

import { getStore } from "./store.js";

const store = getStore("ai-state", {
  config: null,
  models: [],
  currentAnalysis: null,
  analysisHistory: [],
  trafficLog: [],
  loading: false,
  error: null,
});

export function getAiState() { return store.getState(); }
export function setAiState(patch) { store.setState(patch); }
export function onAiChange(fn) { return store.subscribe(fn); }

export function addTrafficEntry(entry) {
  const { trafficLog } = store.getState();
  store.setState({ trafficLog: [...trafficLog, entry] });
}
