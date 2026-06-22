// js/utils/path.js – Path utilities

export function encodeRunIdPathSegment(runId) {
  const text = String(runId || "");
  const bytes = new TextEncoder().encode(text);
  let binary = "";
  bytes.forEach((value) => {
    binary += String.fromCharCode(value);
  });
  const base64 = btoa(binary).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/g, "");
  return `b64_${base64}`;
}

export function basename(path) {
  return String(path || "").split("/").pop() || "";
}

export function dirname(path) {
  const parts = String(path || "").split("/");
  parts.pop();
  return parts.join("/") || "/";
}

export function joinPath(...parts) {
  return parts.filter(Boolean).join("/").replace(/\/+/g, "/");
}
