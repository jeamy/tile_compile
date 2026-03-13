document.addEventListener("DOMContentLoaded", () => {
  const controls = document.querySelectorAll("a, button, input, select, textarea, [role='button']");
  controls.forEach((el) => {
    const title = (el.getAttribute("title") || "").trim();
    if (title) return;

    const explicitTip = (el.getAttribute("data-tooltip") || "").trim();
    if (explicitTip) {
      el.setAttribute("title", explicitTip);
      return;
    }

    const aria = (el.getAttribute("aria-label") || "").trim();
    if (aria) {
      el.setAttribute("title", aria);
      return;
    }

    const placeholder = (el.getAttribute("placeholder") || "").trim();
    if (placeholder) {
      el.setAttribute("title", "Eingabefeld: " + placeholder);
      return;
    }

    const text = (el.textContent || "").replace(/\s+/g, " ").trim();
    if (text) {
      el.setAttribute("title", text);
      return;
    }

    el.setAttribute("title", "Interaktives Element");
  });
});
