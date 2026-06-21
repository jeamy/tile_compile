// js/components/ws-manager.js – WebSocket-Manager für Run-Monitor

import { API_ENDPOINTS } from "../api/endpoints.js";
import { api } from "../api/client.js";

let ws = null;
let reconnectTimer = null;
let listeners = new Set();
let currentRunId = null;

export function connectWebSocket(runId) {
  if (ws && currentRunId === runId) return;
  disconnectWebSocket();
  currentRunId = runId;

  const url = api._toWsUrl(API_ENDPOINTS.ws.run(runId));
  ws = new WebSocket(url);

  ws.onopen = () => {
    notify({ type: "ws:open" });
  };

  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      notify({ type: "ws:message", data });
    } catch {
      notify({ type: "ws:raw", data: event.data });
    }
  };

  ws.onerror = (err) => {
    notify({ type: "ws:error", error: err });
  };

  ws.onclose = () => {
    notify({ type: "ws:close" });
    if (currentRunId) {
      reconnectTimer = setTimeout(() => connectWebSocket(currentRunId), 3000);
    }
  };
}

export function disconnectWebSocket() {
  currentRunId = null;
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

export function isConnected() {
  return ws && ws.readyState === WebSocket.OPEN;
}
