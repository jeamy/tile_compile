// js/api/client.js – ApiClient (aus GUI2 migriert, 1:1)

export class ApiClient {
  constructor(baseUrl = "/") {
    this.setBase(baseUrl);
  }

  setBase(baseUrl) {
    const raw = String(baseUrl || "/").trim();
    this.baseUrl = raw.endsWith("/") ? raw.slice(0, -1) : raw;
    if (this.baseUrl === "") this.baseUrl = "";
  }

  async get(path, opts) {
    return this._request("GET", path, undefined, opts);
  }

  async post(path, body, opts) {
    return this._request("POST", path, body, opts);
  }

  async patch(path, body) {
    return this._request("PATCH", path, body);
  }

  async delete(path) {
    return this._request("DELETE", path);
  }

  ws(path, onEvent, onError) {
    const wsUrl = this._toWsUrl(path);
    const socket = new WebSocket(wsUrl);
    socket.onmessage = (ev) => {
      try {
        const data = JSON.parse(ev.data);
        onEvent(data);
      } catch (err) {
        onError?.(err);
      }
    };
    socket.onerror = (ev) => onError?.(ev);
    return socket;
  }

  httpUrl(path) {
    return this._toHttpUrl(path);
  }

  async _request(method, path, body, opts = {}) {
    const url = this._toHttpUrl(path);
    const controller = new AbortController();
    let timeoutId = null;
    if (opts.timeoutMs) {
      timeoutId = setTimeout(() => controller.abort(), opts.timeoutMs);
    }
    try {
      const resp = await fetch(url, {
        method,
        headers: { "Content-Type": "application/json" },
        body: body === undefined ? undefined : JSON.stringify(body),
        signal: opts.timeoutMs ? controller.signal : undefined,
      });

      let payload = null;
      const txt = await resp.text();
      if (txt) {
        try {
          payload = JSON.parse(txt);
        } catch {
          payload = { raw: txt };
        }
      }

      if (!resp.ok) {
        const message = payload?.message || payload?.error?.message || payload?.detail?.error?.message || `HTTP ${resp.status}`;
        const e = new Error(message);
        e.status = resp.status;
        e.payload = payload;
        throw e;
      }
      return payload;
    } finally {
      if (timeoutId) clearTimeout(timeoutId);
    }
  }

  _toHttpUrl(path) {
    const p = path.startsWith("/") ? path : `/${path}`;
    if (this.baseUrl === "") return p;
    return `${this.baseUrl}${p}`;
  }

  _toWsUrl(path) {
    const http = this._toHttpUrl(path);
    if (http.startsWith("http://")) return `ws://${http.slice(7)}`;
    if (http.startsWith("https://")) return `wss://${http.slice(8)}`;
    const proto = window.location.protocol === "https:" ? "wss" : "ws";
    return `${proto}://${window.location.host}${http}`;
  }
}

export const api = new ApiClient();

// Auto-detect: if we're not served from the backend (e.g. local dev server),
// point API calls to localhost:8080 where the backend runs
if (window.location.port && window.location.port !== "8080") {
  api.setBase(`http://${window.location.hostname}:8080`);
}
