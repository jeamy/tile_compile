export class ApiClient {
  constructor(baseUrl = "/") {
    this.setBase(baseUrl);
  }

  setBase(baseUrl) {
    const raw = String(baseUrl || "/").trim();
    this.baseUrl = raw.endsWith("/") ? raw.slice(0, -1) : raw;
    if (this.baseUrl === "") this.baseUrl = "";
  }

  async get(path) {
    return this._request("GET", path);
  }

  async post(path, body) {
    return this._request("POST", path, body);
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

  async _request(method, path, body) {
    const url = this._toHttpUrl(path);
    const resp = await fetch(url, {
      method,
      headers: {
        "Content-Type": "application/json",
      },
      body: body === undefined ? undefined : JSON.stringify(body),
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
      const err = payload?.error || payload?.detail?.error || payload || { message: resp.statusText };
      const message = err.message || `HTTP ${resp.status}`;
      const e = new Error(message);
      e.status = resp.status;
      e.payload = payload;
      throw e;
    }
    return payload;
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
