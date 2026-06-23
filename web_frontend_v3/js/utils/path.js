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
