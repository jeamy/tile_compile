// js/utils/yaml-scalar.js – scalar read/write for the line-based YAML patches
// of the BGE and HMS preview dialogs.

/** A YAML scalar as the type of `fallback` (comment and quotes stripped). */
export function yamlScalar(raw, fallback) {
  const value = String(raw ?? "").trim().replace(/\s+#.*$/, "");
  if (typeof fallback === "boolean") return value === "true" ? true : value === "false" ? false : fallback;
  if (typeof fallback === "number") { const n = Number(value); return Number.isFinite(n) ? n : fallback; }
  return value.replace(/^['"]|['"]$/g, "") || fallback;
}

/** A value as YAML scalar text. */
export function yamlScalarText(value) {
  return typeof value === "boolean" ? (value ? "true" : "false") : String(value);
}
