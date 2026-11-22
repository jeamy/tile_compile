import { el, clear } from "../utils/dom.js";
import { t } from "../i18n/i18n.js";
import { api } from "../api/client.js";
import { API_ENDPOINTS } from "../api/endpoints.js";
import { createLiveImageViewer } from "./live-image-viewer.js";

function artifactItems(payload) {
  if (!payload) return [];
  return Array.isArray(payload?.items) ? payload.items : Array.isArray(payload) ? payload : [];
}

function artifactPath(item) {
  return String(item?.path || item?.relative_path || item?.filename || item?.name || item || "");
}

function lowerPath(item) {
  return artifactPath(item).toLowerCase();
}

function scoreArtifact(item) {
  const p = lowerPath(item);
  let score = 0;
  if (p.includes("/outputs/") || p.startsWith("outputs/")) score += 20;
  if (p.endsWith(".png")) score += 10;
  if (p.includes("live_edit")) {
    score += 200;
    // The canonical working state is the float FITS. Prefer it over an
    // older derived PNG/JPEG preview that may still exist after reset.
    if (/\.(fits?|fts)$/.test(p)) score += 50;
  }
  else if (p.includes("stacked_rgb_hms") || p.includes("hms_")) score += 100;
  else if (p.includes("stacked_rgb_pcc") || p.includes("pcc_")) score += 90;
  else if (p.includes("stacked_rgb_bge")) score += 80;
  else if (p.includes("stacked_rgb_solve")) score += 70;
  else if (p.includes("stacked_rgb")) score += 60;
  else if (p.endsWith(".png")) score += 30;
  return score;
}

function bestArtifact(items) {
  return artifactItems(items)
    .filter(item => /\.(png|fits?|fts)$/i.test(artifactPath(item)))
    .sort((a, b) => scoreArtifact(b) - scoreArtifact(a))[0] || null;
}

// bestArtifact only returns PNG and FITS files: a PNG is shown as it is, a
// FITS through the backend's image preview.
function previewUrl(runId, runDir, artifact) {
  const path = artifactPath(artifact);
  return lowerPath(artifact).endsWith(".png")
    ? api.httpUrl(API_ENDPOINTS.runs.artifactRaw(runId, path, runDir))
    : api.httpUrl(API_ENDPOINTS.runs.imagePreview(runId, path, runDir));
}

function panelBody(targetId) {
  return document.getElementById(`${targetId}-body`);
}

function setPreviewStatus(targetId, text) {
  const status = document.getElementById(`${targetId}-status`);
  if (status) status.textContent = text;
}

function openLiveEditor(runId, runDir, targetId) {
  const viewer = createLiveImageViewer(runId, runDir, () => {
    refreshRunImagePreviewPanel(targetId);
  });
  viewer.open();
}

function refreshRunImagePreviewPanel(targetId = "run-image-preview") {
  const body = panelBody(targetId);
  const runId = body?._runId || "";
  const runDir = body?._runDir || "";
  if (!runId) {
    setPreviewStatus(targetId, t("ui.state.no_run_selected", "Kein Run ausgewählt"));
    return;
  }
  loadRunImagePreview(runId, runDir, null, targetId, { force: true });
}

export function createRunImagePreviewPanel(targetId = "run-image-preview") {
  return el("div", { class: "tc-card", id: targetId },
    el("div", { class: "tc-card-title tc-flex tc-items-center tc-justify-between" },
      el("span", {}, t("ui.title.latest_image_preview", "Letztes Bild")),
      el("div", { class: "tc-flex tc-items-center tc-gap-2" },
        el("span", { class: "tc-text-muted tc-text-sm tc-mono", id: `${targetId}-status` }, t("ui.state.not_loaded", "nicht geladen")),
        el("button", {
          class: "tc-btn tc-btn-sm",
          title: t("ui.tooltip.image_preview_refresh", "Preview aus dem neuesten Bild neu erzeugen."),
          "aria-label": t("ui.tooltip.image_preview_refresh", "Preview aus dem neuesten Bild neu erzeugen."),
          onclick: () => refreshRunImagePreviewPanel(targetId),
        }, "\u21bb"),
      ),
    ),
    el("div", { id: `${targetId}-body`, class: "tc-flex-col tc-gap-2" },
      el("div", { class: "tc-text-muted tc-text-sm" }, t("ui.state.preview_loading_hint", "Preview wird geladen, sobald ein Run aktiv ist.")),
    ),
  );
}

export async function loadRunImagePreview(runId, runDir = "", artifactsPayload = null, targetId = "run-image-preview", opts = {}) {
  const body = panelBody(targetId);
  if (!body) return;
  body._runId = runId || "";
  body._runDir = runDir || "";
  const previewKey = `${runId || ""}|${runDir || ""}`;
  if (!opts.force && body._previewBuiltKey === previewKey) return;
  body._previewBuiltKey = previewKey;
  // Each load owns the panel until a newer one starts; a stale response must
  // not draw.
  const generation = (body._previewGeneration || 0) + 1;
  body._previewGeneration = generation;
  const stale = () => body._previewGeneration !== generation;
  clear(body);
  if (!runId) {
    setPreviewStatus(targetId, t("ui.state.no_run_selected", "Kein Run ausgewählt"));
    body.appendChild(el("div", { class: "tc-text-muted tc-text-sm" }, t("ui.state.no_run_selected", "Kein Run ausgewählt")));
    return;
  }

  setPreviewStatus(targetId, t("ui.state.loading", "Lädt..."));
  body.appendChild(el("div", { class: "tc-text-muted tc-text-sm" }, t("ui.state.loading", "Lädt...")));
  try {
    const artifacts = artifactsPayload || await api.get(API_ENDPOINTS.runs.artifacts(runId, runDir));
    if (stale()) return;
    const candidate = bestArtifact(artifacts);
    if (!candidate) {
      clear(body);
      setPreviewStatus(targetId, t("ui.state.no_image_preview", "Kein Bildpreview verfügbar"));
      body.appendChild(el("div", { class: "tc-text-muted tc-text-sm" }, t("ui.state.no_image_preview", "Kein Bildpreview verfügbar")));
      return;
    }

    const src = previewUrl(runId, runDir, candidate);
    clear(body);
    const path = artifactPath(candidate);
    setPreviewStatus(targetId, path);
    const pathRow = el("div", {
      class: "tc-flex tc-items-center tc-justify-between tc-gap-2",
    });
    pathRow.appendChild(el("div", { class: "tc-text-sm tc-text-muted tc-mono" }, path));
    pathRow.appendChild(el("button", {
      class: "tc-btn tc-btn-sm",
      title: t("ui.button.live_editor", "Live Editor"),
      "aria-label": t("ui.button.live_editor", "Live Editor"),
      onclick: () => openLiveEditor(runId, runDir, targetId),
    }, t("ui.button.live_editor", "Live Editor")));
    body.appendChild(pathRow);
    body.appendChild(el("img", {
      src,
      alt: t("ui.title.latest_image_preview", "Letztes Bild"),
      style: {
        width: "100%",
        maxHeight: "520px",
        objectFit: "contain",
        background: "var(--bg)",
        borderRadius: "6px",
        cursor: "pointer",
      },
      title: t("liveImage.clickToOpen", "Click to open Live Image Editor"),
      onclick: () => {
        openLiveEditor(runId, runDir, targetId);
      },
    }));
  } catch (e) {
    if (stale()) return;
    clear(body);
    setPreviewStatus(targetId, t("ui.state.preview_failed", "Preview fehlgeschlagen"));
    body.appendChild(el("div", { class: "tc-text-error tc-text-sm" }, e.message));
  }
}
