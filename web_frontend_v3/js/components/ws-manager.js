// js/components/ws-manager.js – WebSocket-Manager für Run-Monitor

import { API_ENDPOINTS } from "../api/endpoints.js";
import { api } from "../api/client.js";

let ws = null;
let reconnectTimer = null;
let listeners = new Set();
let currentRunId = null;
let currentRunDir = "";
let wsGeneration = 0;

export function connectWebSocket(runId, force = false, runDir = "") {
  if (ws && currentRunId === runId && currentRunDir === String(runDir || "") && !force) return;
  if (force) disconnectWebSocket();
  currentRunId = runId;
  currentRunDir = String(runDir || "");
  const generation = ++wsGeneration;

  const url = api._toWsUrl(API_ENDPOINTS.ws.run(runId, currentRunDir));
  ws = new WebSocket(url);

  ws.onopen = () => {
    if (generation !== wsGeneration) return;
    notify({ type: "ws:open" });
  };

  ws.onmessage = (event) => {
    if (generation !== wsGeneration) return;
    try {
      const data = JSON.parse(event.data);
      notify({ type: "ws:message", data });
    } catch {
      notify({ type: "ws:raw", data: event.data });
    }
  };

  ws.onerror = (err) => {
    if (generation !== wsGeneration) return;
    notify({ type: "ws:error", error: err });
  };

  ws.onclose = () => {
    if (generation !== wsGeneration) return;
    notify({ type: "ws:close" });
    if (currentRunId) {
      reconnectTimer = setTimeout(() => connectWebSocket(currentRunId, false, currentRunDir), 3000);
    }
  };
}

export function disconnectWebSocket() {
  wsGeneration++;
  currentRunId = null;
  currentRunDir = "";
  if (reconnectTimer) {
    clearTimeout(reconnectTimer);
    reconnectTimer = null;
  }
  if (ws) {
    ws.close();
    ws = null;
  }
}

export function onWebSocketMessage(fn) {
  listeners.add(fn);
  return () => listeners.delete(fn);
}

function notify(event) {
  listeners.forEach(fn => fn(event));
}
