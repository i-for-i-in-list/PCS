// Run before the stylesheet so the first painted frame uses the chosen theme.
(() => {
  const STORAGE_KEY = "dashboard-v3-theme";
  const normalize = (value) => ["light", "dark"].includes(value) ? value : "system";
  const systemTheme = window.matchMedia?.("(prefers-color-scheme: dark)");
  let preference = "system";
  try { preference = normalize(localStorage.getItem(STORAGE_KEY)); }
  catch { /* Storage can be unavailable for file URLs or private browsing. */ }

  function applyTheme() {
    const theme = preference === "system" ? (systemTheme?.matches ? "dark" : "light") : preference;
    const root = document.documentElement;
    const changed = root.dataset.theme !== theme;
    root.dataset.theme = theme;
    root.style.colorScheme = theme;
    const select = document.getElementById("themeSelect");
    if (select) select.value = preference;
    if (changed) window.dispatchEvent(new CustomEvent("dashboard-theme-changed"));
  }

  function bindSelector() {
    const select = document.getElementById("themeSelect");
    if (!select) return;
    select.value = preference;
    select.addEventListener("change", () => {
      preference = normalize(select.value);
      try { localStorage.setItem(STORAGE_KEY, preference); }
      catch { /* Keep the in-memory choice even when persistence is blocked. */ }
      applyTheme();
    });
  }

  applyTheme();
  systemTheme?.addEventListener("change", () => {
    if (preference === "system") applyTheme();
  });
  window.addEventListener("storage", (event) => {
    if (event.key === STORAGE_KEY || event.key === null) {
      preference = normalize(event.newValue);
      applyTheme();
    }
  });
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", bindSelector, { once: true });
  } else bindSelector();
})();
