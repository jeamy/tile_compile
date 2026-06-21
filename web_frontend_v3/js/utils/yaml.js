// js/utils/yaml.js – YAML-Diff-Utilities

import { parseYaml } from "./yaml-parse.js";

export function formatYamlDiff(before, after) {
  const beforeObj = typeof before === "string" ? parseYaml(before) : (before || {});
  const afterObj = typeof after === "string" ? parseYaml(after) : (after || {});
  const changes = deepDiff(beforeObj, afterObj);
  const result = [];
  for (const c of changes) {
    if (c.oldValue !== undefined && c.newValue !== undefined) {
      result.push({ type: "removed", text: `${c.path}: ${formatVal(c.oldValue)}` });
      result.push({ type: "added", text: `${c.path}: ${formatVal(c.newValue)}` });
    } else if (c.oldValue !== undefined) {
      result.push({ type: "removed", text: `${c.path}: ${formatVal(c.oldValue)}` });
    } else {
      result.push({ type: "added", text: `${c.path}: ${formatVal(c.newValue)}` });
    }
  }
  return result;
}

function deepDiff(before, after, prefix = "", out = []) {
  const allKeys = new Set([...Object.keys(before || {}), ...Object.keys(after || {})]);
  for (const key of allKeys) {
    const path = prefix ? `${prefix}.${key}` : key;
    const bv = before?.[key];
    const av = after?.[key];
    if (bv === av) continue;
    if (isPlainObject(bv) && isPlainObject(av)) {
      deepDiff(bv, av, path, out);
    } else if (Array.isArray(bv) && Array.isArray(av)) {
      if (JSON.stringify(bv) !== JSON.stringify(av)) {
        out.push({ path, oldValue: bv, newValue: av });
      }
    } else {
      if (bv === undefined) {
        out.push({ path, newValue: av });
      } else if (av === undefined) {
        out.push({ path, oldValue: bv });
      } else {
        out.push({ path, oldValue: bv, newValue: av });
      }
    }
  }
  return out;
}

function isPlainObject(v) {
  return v !== null && typeof v === "object" && !Array.isArray(v);
}

function formatVal(v) {
  if (v === null) return "null";
  if (v === true) return "true";
  if (v === false) return "false";
  if (Array.isArray(v)) return `[${v.map(formatVal).join(", ")}]`;
  if (typeof v === "object") return JSON.stringify(v);
  return String(v);
}
