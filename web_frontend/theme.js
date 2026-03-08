(() => {
  const KEY = "gui2_theme";
  const allowed = new Set(["observatory", "slate", "sand"]);
  const params = new URLSearchParams(window.location.search);

  let theme = params.get("theme") || window.localStorage.getItem(KEY) || "observatory";
  if (!allowed.has(theme)) {
    theme = "observatory";
  }

  document.documentElement.setAttribute("data-theme", theme);
  window.localStorage.setItem(KEY, theme);
})();
