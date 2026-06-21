// js/components/toast.js – Notification-System

import { el } from "../utils/dom.js";

let container = null;

function getContainer() {
  if (!container) {
    container = document.getElementById("toast-container");
  }
  return container;
}

export function toast(title, body = "", type = "info", duration = 5000) {
  const c = getContainer();
  if (!c) return;

  const t = el("div", { class: `tc-toast tc-toast-${type}` },
    el("span", { class: "tc-toast-close", onclick: () => t.remove() }, "\u00d7"),
    el("div", { class: "tc-toast-title" }, title),
    body ? el("div", { class: "tc-toast-body" }, body) : null,
  );

  c.appendChild(t);

  if (duration > 0) {
    setTimeout(() => {
      if (t.parentNode) t.remove();
    }, duration);
  }
}

export function toastSuccess(title, body, duration) {
  toast(title, body, "success", duration);
}

export function toastError(title, body, duration = 0) {
  toast(title, body, "error", duration);
}

export function toastWarning(title, body, duration = 10000) {
  toast(title, body, "warning", duration);
}

export function toastInfo(title, body, duration) {
  toast(title, body, "info", duration);
}
