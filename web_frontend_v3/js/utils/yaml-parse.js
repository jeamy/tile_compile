// js/utils/yaml-parse.js – Minimal YAML parser for config key-value access
// Supports nested objects, arrays, scalars (string/int/float/bool/null).
// Not a full YAML parser — sufficient for tile_compile config files.

export function parseYaml(text) {
  if (!text || typeof text !== "string") return {};
  const lines = text.split("\n");
  const root = {};
  const stack = [{ indent: -1, obj: root }];
  let i = 0;

  while (i < lines.length) {
    const raw = lines[i];
    i++;

    // Strip comments
    const commentIdx = findComment(raw);
    const line = commentIdx >= 0 ? raw.slice(0, commentIdx) : raw;
    const trimmed = line.trim();
    if (!trimmed) continue;

    const indent = line.length - line.trimStart().length;

    // Pop stack to current indent level
    while (stack.length > 1 && stack[stack.length - 1].indent >= indent) {
      stack.pop();
    }

    const parent = stack[stack.length - 1].obj;

    // Array item
    if (trimmed.startsWith("- ")) {
      const val = trimmed.slice(2).trim();
      if (!Array.isArray(parent)) {
        // Convert to array — but this shouldn't happen at root level
        continue;
      }
      if (val.includes(":")) {
        // Array of objects: - key: value
        const obj = {};
        parent.push(obj);
        const [k, ...rest] = val.split(":");
        const v = rest.join(":").trim();
        if (v) obj[k.trim()] = parseScalar(v);
        else {
          // Multi-line object, push new context
          stack.push({ indent: indent, obj });
        }
      } else {
        parent.push(parseScalar(val));
      }
      continue;
    }

    // Key: value
    const colonIdx = trimmed.indexOf(":");
    if (colonIdx < 0) continue;

    const key = trimmed.slice(0, colonIdx).trim();
    const valStr = trimmed.slice(colonIdx + 1).trim();

    if (!valStr) {
      // Nested object or array — look ahead
      const nextLine = lines[i] || "";
      const nextTrimmed = nextLine.trim();
      const nextIndent = nextLine.length - nextLine.trimStart().length;

      if (nextTrimmed.startsWith("- ") && nextIndent > indent) {
        const arr = [];
        parent[key] = arr;
        stack.push({ indent: indent, obj: arr });
      } else if (nextIndent > indent) {
        const obj = {};
        parent[key] = obj;
        stack.push({ indent: indent, obj });
      } else {
        parent[key] = null;
      }
    } else {
      parent[key] = parseScalar(valStr);
    }
  }

  return root;
}

export function stringifyYaml(obj, indent = 0) {
  const pad = "  ".repeat(indent);
  const lines = [];

  if (Array.isArray(obj)) {
    for (const item of obj) {
      if (item !== null && typeof item === "object" && !Array.isArray(item)) {
        const entries = Object.entries(item);
        if (entries.length === 0) {
          lines.push(`${pad}- {}`);
        } else {
          const [k, v] = entries[0];
          lines.push(`${pad}- ${k}: ${formatScalar(v)}`);
          for (let j = 1; j < entries.length; j++) {
            const [k2, v2] = entries[j];
            if (v2 !== null && typeof v2 === "object") {
              lines.push(`${pad}  ${k2}:`);
              lines.push(stringifyYaml(v2, indent + 2));
            } else {
              lines.push(`${pad}  ${k2}: ${formatScalar(v2)}`);
            }
          }
        }
      } else {
        lines.push(`${pad}- ${formatScalar(item)}`);
      }
    }
  } else if (obj !== null && typeof obj === "object") {
    for (const [key, val] of Object.entries(obj)) {
      if (val === null) continue;
      if (Array.isArray(val)) {
        lines.push(`${pad}${key}: ${formatScalar(val)}`);
      } else if (typeof val === "object") {
        lines.push(`${pad}${key}:`);
        lines.push(stringifyYaml(val, indent + 1));
      } else {
        lines.push(`${pad}${key}: ${formatScalar(val)}`);
      }
    }
  }

  return lines.join("\n");
}

function parseScalar(s) {
  const str = s.trim();
  if (!str) return "";

  // Quoted strings
  if ((str.startsWith('"') && str.endsWith('"')) ||
      (str.startsWith("'") && str.endsWith("'"))) {
    return str.slice(1, -1);
  }

  // Boolean
  if (str === "true" || str === "True" || str === "TRUE") return true;
  if (str === "false" || str === "False" || str === "FALSE") return false;

  // Null
  if (str === "null" || str === "Null" || str === "~" || str === "") return null;

  // Flow sequence [a, b, c]
  if (str.startsWith("[") && str.endsWith("]")) {
    const inner = str.slice(1, -1).trim();
    if (!inner) return [];
    return splitFlow(inner).map(s => parseScalar(s.trim()));
  }

  // Flow mapping {a: 1, b: 2}
  if (str.startsWith("{") && str.endsWith("}")) {
    const obj = {};
    const inner = str.slice(1, -1);
    for (const pair of splitFlow(inner)) {
      const ci = pair.indexOf(":");
      if (ci >= 0) {
        obj[pair.slice(0, ci).trim()] = parseScalar(pair.slice(ci + 1).trim());
      }
    }
    return obj;
  }

  // Number. Keep this after flow collections because Number("[]") is 0.
  const num = Number(str);
  if (str !== "" && !isNaN(num)) return num;

  return str;
}

function formatScalar(v) {
  if (v === null) return "null";
  if (v === true) return "true";
  if (v === false) return "false";
  if (typeof v === "number") return String(v);
  if (Array.isArray(v)) return `[${v.map(formatScalar).join(", ")}]`;
  if (typeof v === "string") {
    if (v.includes(":") || v.includes("#") || v.startsWith(" ") || v.startsWith("*") || v.startsWith("&") || v.startsWith("!") || v.startsWith("{") || v.startsWith("[") || v.startsWith("|") || v.startsWith(">") || v.startsWith("@") || v.startsWith("`")) {
      return `"${v.replace(/"/g, '\\"')}"`;
    }
    return v;
  }
  if (typeof v === "object") return "{}";
  return String(v);
}

function findComment(line) {
  let inStr = false;
  let strChar = "";
  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (inStr) {
      if (ch === strChar && line[i - 1] !== "\\") inStr = false;
    } else {
      if (ch === '"' || ch === "'") { inStr = true; strChar = ch; }
      else if (ch === "#") return i;
    }
  }
  return -1;
}

function splitFlow(s) {
  const parts = [];
  let depth = 0;
  let start = 0;
  let inStr = false;
  let strChar = "";
  for (let i = 0; i < s.length; i++) {
    const ch = s[i];
    if (inStr) {
      if (ch === strChar && s[i - 1] !== "\\") inStr = false;
    } else {
      if (ch === '"' || ch === "'") { inStr = true; strChar = ch; }
      else if (ch === "[" || ch === "{") depth++;
      else if (ch === "]" || ch === "}") depth--;
      else if (ch === "," && depth === 0) {
        parts.push(s.slice(start, i));
        start = i + 1;
      }
    }
  }
  if (start < s.length) parts.push(s.slice(start));
  return parts;
}
