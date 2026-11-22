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
  const sameTarget = currentRunId === runId && currentRunDir === String(runDir || "");
  const live = ws && (ws.readyState === WebSocket.CONNECTING || ws.readyState === WebSocket.OPEN);
  if (live && sameTarget && !force) return;
  // A different run, a forced reconnect or a dead socket: drop the old one
  // (and any pending reconnect) before opening the next.
  disconnectWebSocket();
  currentRunId = runId;
  currentRunDir = String(runDir || "");
  const generation = ++wsGeneration;

  const url = api.wsUrl(API_ENDPOINTS.ws.run(runId, currentRunDir));
  const socket = new WebSocket(url);
  ws = socket;

  socket.onopen = () => {
    if (generation !== wsGeneration) return;
    notify({ type: "ws:open" });
  };

  socket.onmessage = (event) => {
    if (generation !== wsGeneration) return;
    try {
      const data = JSON.parse(event.data);
      notify({ type: "ws:message", data });
    } catch {
      notify({ type: "ws:raw", data: event.data });
    }
  };

  socket.onerror = (err) => {
    if (generation !== wsGeneration) return;
    notify({ type: "ws:error", error: err });
  };

  socket.onclose = () => {
    if (generation !== wsGeneration) return;
    if (ws === socket) ws = null;
    notify({ type: "ws:close" });
    if (currentRunId) {
      const runIdAtClose = currentRunId;
      const runDirAtClose = currentRunDir;
      reconnectTimer = setTimeout(() => {
        reconnectTimer = null;
        if (generation !== wsGeneration) return;
        connectWebSocket(runIdAtClose, false, runDirAtClose);
      }, 3000);
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
