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
  if (p.includes("stacked_rgb_hms") || p.includes("hms_")) score += 100;
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

function previewKindForArtifact(item) {
  const p = lowerPath(item);
  if (p.endsWith(".png")) return "raw";
  if (/\.(fits?|fts)$/.test(p)) return "fits";
  if (p.includes("stacked_rgb_hms") || p.includes("hms_")) return "hms";
  if (p.includes("stacked_rgb_pcc") || p.includes("pcc_")) return "hms";
  if (p.includes("stacked_rgb_bge")) return "bge";
  return "";
}

async function fetchPngPreview(runId, runDir, artifact, kind) {
  if (kind === "raw") {
    return api.httpUrl(API_ENDPOINTS.runs.artifactRaw(runId, artifactPath(artifact), runDir));
  }
  if (kind === "fits") {
    return api.httpUrl(API_ENDPOINTS.runs.imagePreview(runId, artifactPath(artifact), runDir));
  }
  const endpoint = kind === "bge" ? API_ENDPOINTS.runs.bgePreview(runId) : API_ENDPOINTS.runs.hmePreview(runId);
  const body = kind === "bge"
    ? { run_dir: runDir || "", view: "corrected", params: {} }
    : { run_dir: runDir || "", params: {} };
  const response = await fetch(api.httpUrl(endpoint), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return URL.createObjectURL(await response.blob());
}

function panelBody(targetId) {
  return document.getElementById(`${targetId}-body`);
}

function setPreviewStatus(targetId, text) {
  const status = document.getElementById(`${targetId}-status`);
  if (status) status.textContent = text;
}

export function refreshRunImagePreviewPanel(targetId = "run-image-preview") {
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
  clear(body);
  if (body._objectUrl) {
    URL.revokeObjectURL(body._objectUrl);
    body._objectUrl = "";
  }
  if (!runId) {
    setPreviewStatus(targetId, t("ui.state.no_run_selected", "Kein Run ausgewählt"));
    body.appendChild(el("div", { class: "tc-text-muted tc-text-sm" }, t("ui.state.no_run_selected", "Kein Run ausgewählt")));
    return;
  }

  setPreviewStatus(targetId, t("ui.state.loading", "Lädt..."));
  body.appendChild(el("div", { class: "tc-text-muted tc-text-sm" }, t("ui.state.loading", "Lädt...")));
  try {
    const artifacts = artifactsPayload || await api.get(API_ENDPOINTS.runs.artifacts(runId, runDir));
    const candidate = bestArtifact(artifacts);
    if (!candidate) {
      clear(body);
      setPreviewStatus(targetId, t("ui.state.no_image_preview", "Kein Bildpreview verfügbar"));
      body.appendChild(el("div", { class: "tc-text-muted tc-text-sm" }, t("ui.state.no_image_preview", "Kein Bildpreview verfügbar")));
      return;
    }

    const kind = previewKindForArtifact(candidate);
    if (!kind) throw new Error(t("ui.state.no_image_preview", "Kein Bildpreview verfügbar"));
    const src = await fetchPngPreview(runId, runDir, candidate, kind);
    if (src.startsWith("blob:")) body._objectUrl = src;
    clear(body);
    const path = artifactPath(candidate);
    setPreviewStatus(targetId, path);
    body.appendChild(el("div", { class: "tc-text-sm tc-text-muted tc-mono" }, path));
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
        const viewer = createLiveImageViewer(runId, runDir, null);
        viewer.open();
      },
    }));
  } catch (e) {
    clear(body);
    setPreviewStatus(targetId, t("ui.state.preview_failed", "Preview fehlgeschlagen"));
    body.appendChild(el("div", { class: "tc-text-error tc-text-sm" }, e.message));
  }
}
