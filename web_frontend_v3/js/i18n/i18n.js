// js/i18n/i18n.js – i18n-Logik (vereinfacht für GUI3)

let messages = {};
let currentLocale = "de";

export async function loadLocale(locale) {
  try {
    const resp = await fetch(`i18n/${locale}.json`);
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    messages = await resp.json();
    currentLocale = locale;
    document.documentElement.setAttribute("lang", locale);
  } catch (e) {
    console.error(`Failed to load locale ${locale}:`, e);
    if (locale !== "de") {
      return loadLocale("de");
    }
  }
}

export function t(key, fallback) {
  if (key in messages && typeof messages[key] === "string") {
    return messages[key];
  }
  return fallback || key;
}

export function getLocale() {
  return currentLocale;
}

export function setLocale(locale) {
  return loadLocale(locale);
}
