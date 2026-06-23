// js/utils/dom.js – DOM-Helper

export function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (key === "class") {
      node.className = value;
    } else if (key === "style" && typeof value === "object") {
      Object.assign(node.style, value);
    } else if (key.startsWith("data-")) {
      node.setAttribute(key, value);
    } else if (key === "onclick" || key.startsWith("on")) {
      node.addEventListener(key.slice(2).toLowerCase(), value);
    } else if (key === "checked" || key === "disabled" || key === "readonly" || key === "selected" || key === "hidden") {
      if (value) node.setAttribute(key, "");
      else node.removeAttribute(key);
    } else if (key === "value" && (tag === "input" || tag === "textarea" || tag === "select")) {
      node.value = value ?? "";
    } else if (value !== null && value !== undefined) {
      node.setAttribute(key, value);
    }
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

export function clear(node) {
  while (node.firstChild) node.removeChild(node.firstChild);
  return node;
}

export function setBadge(badgeEl, ok, text) {
  if (!badgeEl) return;
  badgeEl.textContent = text;
  badgeEl.className = `tc-badge ${ok ? "tc-badge-success" : "tc-badge-error"}`;
  badgeEl.style.flexShrink = "0";
  badgeEl.style.whiteSpace = "nowrap";
}

export function statItem(label, value) {
  return el("div", {},
    el("div", { class: "tc-label" }, label),
    el("div", { class: "tc-text-sm tc-mono" }, String(value)),
  );
}
