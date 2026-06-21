// js/utils/yaml.js – YAML-Diff-Utilities

export function formatYamlDiff(before, after) {
  const beforeLines = String(before || "").split("\n");
  const afterLines = String(after || "").split("\n");
  const maxLen = Math.max(beforeLines.length, afterLines.length);
  const result = [];
  for (let i = 0; i < maxLen; i++) {
    const b = beforeLines[i] || "";
    const a = afterLines[i] || "";
    if (b === a) {
      result.push({ type: "unchanged", text: a });
    } else if (b && !a) {
      result.push({ type: "removed", text: b });
    } else if (!b && a) {
      result.push({ type: "added", text: a });
    } else {
      result.push({ type: "removed", text: b });
      result.push({ type: "added", text: a });
    }
  }
  return result;
}
