// js/state/run-state.js – Current Run, Phase, Log-Lines

import { getStore } from "./store.js";

const store = getStore("run-state", {
  currentRunId: null,
  currentRunDir: null,
  runStatus: null,
  status: null,
  phases: [],
  logLines: [],
  wsSocket: null,
  loading: false,
  resumeActive: false,
  resumePending: false,
});

export function getRunState() { return store.getState(); }
export function setRunState(patch) { store.setState(patch); }
export function onRunChange(fn) { return store.subscribe(fn); }

export function appendLogLine(line) {
  const { logLines } = store.getState();
  const next = [...logLines, line];
  if (next.length > 10000) next.shift();
  store.setState({ logLines: next });
}

export function clearLog() {
  store.setState({ logLines: [] });
}
