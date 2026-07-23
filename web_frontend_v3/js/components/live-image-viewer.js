import { api } from "../api/client.js";
import { API_ENDPOINTS } from "../api/endpoints.js";
import { t } from "../i18n/i18n.js";

const FEATURE_PHASE = 2;

const PRESET_COMMANDS = [
  { category: "brightness", label: "liveImage.cmd.brightnessUp", command: "helle das Bild auf", help: "liveImage.cmd.brightnessUp.help", phase: 1 },
  { category: "brightness", label: "liveImage.cmd.brightnessDown", command: "dunkle das Bild ab", help: "liveImage.cmd.brightnessDown.help", phase: 1 },
  { category: "brightness", label: "liveImage.cmd.shadowsUp", command: "helle die Schatten auf", help: "liveImage.cmd.shadowsUp.help", phase: 1 },
  { category: "brightness", label: "liveImage.cmd.highlightsDown", command: "reduziere die Spitzlichter", help: "liveImage.cmd.highlightsDown.help", phase: 1 },
  { category: "brightness", label: "liveImage.cmd.contrastUp", command: "erhoehe den Kontrast", help: "liveImage.cmd.contrastUp.help", phase: 1 },
  { category: "brightness", label: "liveImage.cmd.contrastDown", command: "verringere den Kontrast", help: "liveImage.cmd.contrastDown.help", phase: 1 },
  { category: "brightness", label: "liveImage.cmd.blackLevel", command: "hebe den Schwarzwert an", help: "liveImage.cmd.blackLevel.help", phase: 1 },
  { category: "color", label: "liveImage.cmd.saturationUp", command: "erhoehe die Farbsaettigung", help: "liveImage.cmd.saturationUp.help", phase: 1 },
  { category: "color", label: "liveImage.cmd.saturationDown", command: "verringere die Farbsaettigung", help: "liveImage.cmd.saturationDown.help", phase: 1 },
  { category: "color", label: "liveImage.cmd.rmgreen", command: "entferne gruene Farbnebel", help: "liveImage.cmd.rmgreen.help", phase: 1 },
  { category: "noise", label: "liveImage.cmd.denoise", command: "unterdruecke das Rauschen", help: "liveImage.cmd.denoise.help", phase: 1 },
  { category: "noise", label: "liveImage.cmd.bilateral", command: "entferne Rauschen ohne Detailverlust", help: "liveImage.cmd.bilateral.help", phase: 1 },
  { category: "noise", label: "liveImage.cmd.sharpen", command: "schaerfe das Bild", help: "liveImage.cmd.sharpen.help", phase: 1 },
  { category: "misc", label: "liveImage.cmd.invert", command: "zeige das Bild als Negativ", help: "liveImage.cmd.invert.help", phase: 1 },
  { category: "misc", label: "liveImage.cmd.reset", command: "setze das Bild zurueck", help: "liveImage.cmd.reset.help", phase: 1 },
  { category: "color", label: "liveImage.cmd.vibrance", command: "erhoehe die Lebendigkeit", help: "liveImage.cmd.vibrance.help", phase: 2 },
  { category: "color", label: "liveImage.cmd.temperatureWarm", command: "mache die Farben waermer", help: "liveImage.cmd.temperatureWarm.help", phase: 2 },
  { category: "color", label: "liveImage.cmd.unpurple", command: "entferne lila Farbsaeume", help: "liveImage.cmd.unpurple.help", phase: 2 },
  { category: "details", label: "liveImage.cmd.fixbanding", command: "reduziere Streifenartefakte", help: "liveImage.cmd.fixbanding.help", phase: 2 },
  { category: "details", label: "liveImage.cmd.starDesaturation", command: "entsaettige uebersteuerte Sterne", help: "liveImage.cmd.starDesaturation.help", phase: 2 },
  { category: "details", label: "liveImage.cmd.dehaze", command: "reduziere den Dunst", help: "liveImage.cmd.dehaze.help", phase: 2 },
];

export function createLiveImageViewer(runId, runDir, onClose) {
  const state = {
    sessionId: null,
    currentImageSrc: null,
    previousImageSrc: null,
    showingPrevious: false,
    pendingBeforeImageSrc: null,
    adjustActive: false,
    adjustLabel: "",
    adjustCount: 0,
    repeatActive: false,
    canUndo: false,
    canRedo: false,
    chatHistory: [],
    isLoading: false,
    scale: 1,
    panX: 0,
    panY: 0,
    isDragging: false,
    hasDragged: false,
    dragStartX: 0,
    dragStartY: 0,
  };

  const overlay = document.createElement("div");
  overlay.className = "live-image-viewer-overlay";

  const viewer = document.createElement("div");
  viewer.className = "live-image-viewer";

  // --- Toolbar ---
  const toolbar = document.createElement("div");
  toolbar.className = "live-image-viewer__toolbar";

  const titleEl = document.createElement("span");
  titleEl.className = "live-image-viewer__title";
  titleEl.textContent = t("liveImage.title", "Live Image Editor");
  toolbar.appendChild(titleEl);

  const toolbarBtns = document.createElement("div");
  toolbarBtns.className = "live-image-viewer__toolbar-btns";

  const undoBtn = document.createElement("button");
  undoBtn.className = "live-image-viewer__btn live-image-viewer__btn--undo";
  undoBtn.textContent = t("liveImage.undo", "Undo");
  undoBtn.disabled = true;
  undoBtn.addEventListener("click", () => doUndo());
  toolbarBtns.appendChild(undoBtn);

  const redoBtn = document.createElement("button");
  redoBtn.className = "live-image-viewer__btn live-image-viewer__btn--redo";
  redoBtn.textContent = t("liveImage.redo", "Redo");
  redoBtn.disabled = true;
  redoBtn.addEventListener("click", () => doRedo());
  toolbarBtns.appendChild(redoBtn);

  const resetBtn = document.createElement("button");
  resetBtn.className = "live-image-viewer__btn live-image-viewer__btn--reset";
  resetBtn.textContent = t("liveImage.reset", "Reset");
  resetBtn.addEventListener("click", () => doReset());
  toolbarBtns.appendChild(resetBtn);

  const closeBtn = document.createElement("button");
  closeBtn.className = "live-image-viewer__btn live-image-viewer__btn--close";
  closeBtn.textContent = "\u00d7";
  closeBtn.title = t("liveImage.close", "Close");
  closeBtn.addEventListener("click", () => doClose());
  toolbarBtns.appendChild(closeBtn);

  toolbar.appendChild(toolbarBtns);
  viewer.appendChild(toolbar);

  // --- Main content area ---
  const content = document.createElement("div");
  content.className = "live-image-viewer__content";

  // --- Image container ---
  const imageWrap = document.createElement("div");
  imageWrap.className = "live-image-viewer__image-wrap";

  const imgEl = document.createElement("img");
  imgEl.className = "live-image-viewer__image";
  imgEl.alt = "Live preview";
  imgEl.draggable = false;
  imageWrap.appendChild(imgEl);

  const compareBtn = document.createElement("button");
  compareBtn.type = "button";
  compareBtn.className = "live-image-viewer__compare-badge";
  compareBtn.style.display = "none";
  compareBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    toggleComparison();
  });
  imageWrap.appendChild(compareBtn);

  imgEl.addEventListener("click", () => {
    if (state.hasDragged) {
      state.hasDragged = false;
      return;
    }
    toggleComparison();
  });

  const loadingOverlay = document.createElement("div");
  loadingOverlay.className = "live-image-viewer__loading";
  loadingOverlay.style.display = "none";
  const spinner = document.createElement("div");
  spinner.className = "live-image-viewer__spinner";
  loadingOverlay.appendChild(spinner);
  const loadingText = document.createElement("span");
  loadingText.textContent = t("liveImage.loading", "Loading...");
  loadingOverlay.appendChild(loadingText);
  imageWrap.appendChild(loadingOverlay);

  // Zoom/pan
  imgEl.addEventListener("wheel", (e) => {
    e.preventDefault();
    const delta = e.deltaY > 0 ? 0.9 : 1.1;
    state.scale = Math.max(0.1, Math.min(10, state.scale * delta));
    applyTransform();
  });

  imgEl.addEventListener("mousedown", (e) => {
    if (e.button !== 0) return;
    state.isDragging = true;
    state.hasDragged = false;
    state.dragStartX = e.clientX - state.panX;
    state.dragStartY = e.clientY - state.panY;
    imgEl.style.cursor = "grabbing";
  });

  window.addEventListener("mousemove", (e) => {
    if (!state.isDragging) return;
    if (Math.abs(e.clientX - state.dragStartX - state.panX) > 3 ||
        Math.abs(e.clientY - state.dragStartY - state.panY) > 3) state.hasDragged = true;
    state.panX = e.clientX - state.dragStartX;
    state.panY = e.clientY - state.dragStartY;
    applyTransform();
  });

  window.addEventListener("mouseup", () => {
    if (state.isDragging) {
      state.isDragging = false;
      imgEl.style.cursor = "grab";
    }
  });

  function applyTransform() {
    imgEl.style.transform = `translate(${state.panX}px, ${state.panY}px) scale(${state.scale})`;
  }

  function renderDisplayedImage() {
    const src = state.showingPrevious ? state.previousImageSrc : state.currentImageSrc;
    if (src) imgEl.src = src;
    compareBtn.style.display = state.previousImageSrc ? "block" : "none";
    compareBtn.textContent = state.showingPrevious
      ? t("liveImage.compareAfter", "AFTER")
      : t("liveImage.compareBefore", "BEFORE");
    compareBtn.title = state.showingPrevious
      ? t("liveImage.compareAfterTitle", "Show current image")
      : t("liveImage.compareBeforeTitle", "Show image before the last operation");
  }

  function toggleComparison() {
    if (!state.previousImageSrc || state.isLoading) return;
    state.showingPrevious = !state.showingPrevious;
    renderDisplayedImage();
  }

  function beginOperation() {
    if (state.showingPrevious) {
      state.showingPrevious = false;
      renderDisplayedImage();
    }
    state.pendingBeforeImageSrc = state.currentImageSrc;
    return state.pendingBeforeImageSrc;
  }

  function commitOperation(base64, beforeSrc) {
    state.previousImageSrc = beforeSrc || null;
    state.pendingBeforeImageSrc = null;
    state.currentImageSrc = base64 ? `data:image/jpeg;base64,${base64}` : state.currentImageSrc;
    state.showingPrevious = false;
    renderDisplayedImage();
  }

  function cancelOperation() {
    state.pendingBeforeImageSrc = null;
  }

  content.appendChild(imageWrap);

  // --- Chat panel ---
  const chatPanel = document.createElement("div");
  chatPanel.className = "live-image-viewer__chat";

  const chatHistoryEl = document.createElement("div");
  chatHistoryEl.className = "live-image-viewer__chat-history";
  chatPanel.appendChild(chatHistoryEl);

  // Adjust controls
  const adjustRow = document.createElement("div");
  adjustRow.className = "live-image-viewer__adjust";
  adjustRow.style.display = "none";

  const adjustDown = document.createElement("button");
  adjustDown.className = "live-image-viewer__adjust-btn";
  adjustDown.textContent = "\u2212";
  adjustDown.addEventListener("click", () => doAdjust("decrease"));

  const adjustLabelEl = document.createElement("span");
  adjustLabelEl.className = "live-image-viewer__adjust-label";

  const adjustUp = document.createElement("button");
  adjustUp.className = "live-image-viewer__adjust-btn";
  adjustUp.textContent = "+";
  adjustUp.addEventListener("click", () => doAdjust("increase"));

  adjustRow.appendChild(adjustDown);
  adjustRow.appendChild(adjustLabelEl);
  adjustRow.appendChild(adjustUp);
  chatPanel.appendChild(adjustRow);

  const repeatRow = document.createElement("div");
  repeatRow.className = "live-image-viewer__repeat";
  repeatRow.style.display = "none";
  const repeatBtn = document.createElement("button");
  repeatBtn.type = "button";
  repeatBtn.className = "live-image-viewer__btn live-image-viewer__btn--repeat";
  repeatBtn.textContent = t("liveImage.repeat", "Apply again");
  repeatBtn.addEventListener("click", () => doRepeat());
  repeatRow.appendChild(repeatBtn);
  chatPanel.appendChild(repeatRow);

  // Help box
  const helpBox = document.createElement("div");
  helpBox.className = "live-image-viewer__help";
  helpBox.style.display = "none";
  chatPanel.appendChild(helpBox);

  // Dropdown
  const dropdown = document.createElement("select");
  dropdown.className = "live-image-viewer__dropdown";
  const placeholder = document.createElement("option");
  placeholder.value = "";
  placeholder.textContent = t("liveImage.dropdownPlaceholder", "Select command...");
  dropdown.appendChild(placeholder);

  const categories = {};
  for (const cmd of PRESET_COMMANDS) {
    if (cmd.phase > FEATURE_PHASE) continue;
    if (!categories[cmd.category]) {
      const group = document.createElement("optgroup");
      group.label = t(`liveImage.category.${cmd.category}`, cmd.category);
      categories[cmd.category] = group;
      dropdown.appendChild(group);
    }
    const opt = document.createElement("option");
    opt.value = cmd.command;
    opt.dataset.help = cmd.help;
    opt.textContent = t(cmd.label, cmd.label);
    categories[cmd.category].appendChild(opt);
  }
  chatPanel.appendChild(dropdown);

  // Chat input
  const inputRow = document.createElement("div");
  inputRow.className = "live-image-viewer__input-row";

  const chatInput = document.createElement("input");
  chatInput.type = "text";
  chatInput.className = "live-image-viewer__input";
  chatInput.placeholder = t("liveImage.chatPlaceholder", "Type a message...");
  chatInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && chatInput.value.trim()) {
      sendChat(chatInput.value.trim());
      chatInput.value = "";
    }
  });

  const sendBtn = document.createElement("button");
  sendBtn.className = "live-image-viewer__send-btn";
  sendBtn.textContent = t("liveImage.send", "Send");
  sendBtn.addEventListener("click", () => {
    if (chatInput.value.trim()) {
      sendChat(chatInput.value.trim());
      chatInput.value = "";
    }
  });

  inputRow.appendChild(chatInput);
  inputRow.appendChild(sendBtn);
  chatPanel.appendChild(inputRow);

  // Export buttons
  const exportRow = document.createElement("div");
  exportRow.className = "live-image-viewer__export-row";

  const exportPngBtn = document.createElement("button");
  exportPngBtn.className = "live-image-viewer__btn live-image-viewer__btn--export";
  exportPngBtn.textContent = t("liveImage.exportPng", "Export PNG");
  exportPngBtn.addEventListener("click", () => doExport("png"));

  const exportFitsBtn = document.createElement("button");
  exportFitsBtn.className = "live-image-viewer__btn live-image-viewer__btn--export";
  exportFitsBtn.textContent = t("liveImage.exportFits", "Export FITS");
  exportFitsBtn.addEventListener("click", () => doExport("fits"));

  exportRow.appendChild(exportPngBtn);
  exportRow.appendChild(exportFitsBtn);
  chatPanel.appendChild(exportRow);

  content.appendChild(chatPanel);
  viewer.appendChild(content);
  overlay.appendChild(viewer);

  // Dropdown handler
  dropdown.addEventListener("change", () => {
    const selected = dropdown.selectedOptions[0];
    if (!selected || !selected.value) return;
    const helpKey = selected.dataset.help;
    if (helpKey) {
      helpBox.textContent = t(helpKey, helpKey);
      helpBox.style.display = "block";
    }
    sendChat(selected.value);
    dropdown.selectedIndex = 0;
  });

  // --- Functions ---

  function setLoading(loading) {
    state.isLoading = loading;
    compareBtn.disabled = loading;
    loadingOverlay.style.display = loading ? "flex" : "none";
  }

  function updateImage(jpegBase64) {
    imgEl.style.opacity = "0";
    setTimeout(() => {
      state.currentImageSrc = `data:image/jpeg;base64,${jpegBase64}`;
      state.showingPrevious = false;
      renderDisplayedImage();
      imgEl.style.opacity = "1";
    }, 150);
  }

  function addChatBubble(role, text) {
    const bubble = document.createElement("div");
    bubble.className = `live-image-viewer__chat-bubble live-image-viewer__chat-bubble--${role}`;
    bubble.textContent = text;
    chatHistoryEl.appendChild(bubble);
    chatHistoryEl.scrollTop = chatHistoryEl.scrollHeight;
  }

  function restoreChatHistory(history) {
    if (!Array.isArray(history)) return;
    chatHistoryEl.innerHTML = "";
    for (const entry of history) {
      if (!entry || !entry.role) continue;
      addChatBubble(entry.role, entry.content || entry.text || "");
    }
  }

  function updateAdjustControls(show, label, count) {
    state.adjustActive = show;
    state.adjustLabel = label || "";
    state.adjustCount = count || 0;
    adjustRow.style.display = show ? "flex" : "none";
    adjustLabelEl.textContent = show ? `${state.adjustLabel} (${state.adjustCount})` : "";
  }

  function updateRepeatControl(show) {
    state.repeatActive = show;
    repeatRow.style.display = show ? "flex" : "none";
  }

  function updateUndoRedo(canUndo, canRedo) {
    state.canUndo = canUndo;
    state.canRedo = canRedo;
    undoBtn.disabled = !canUndo;
    redoBtn.disabled = !canRedo;
  }

  async function open() {
    document.body.appendChild(overlay);
    setLoading(true);
    try {
      const resp = await api.post(API_ENDPOINTS.pi.liveImageChat.create, { run_id: runId });
      state.sessionId = resp.session_id;
      updateImage(resp.image_base64);
      imgEl.style.cursor = "grab";

      if (resp.chat_history) {
        restoreChatHistory(resp.chat_history);
      }

      updateUndoRedo(false, false);
    } catch (err) {
      addChatBubble("assistant", `Error: ${err.message}`);
    } finally {
      setLoading(false);
    }
  }

  async function sendChat(msg) {
    if (!state.sessionId || state.isLoading) return;
    const beforeSrc = beginOperation();
    addChatBubble("user", msg);
    setLoading(true);
    updateAdjustControls(false, "", 0);
    updateRepeatControl(false);
    try {
      const resp = await api.post(API_ENDPOINTS.pi.liveImageChat.chat, {
        session_id: state.sessionId,
        message: msg,
      });
      addChatBubble("assistant", resp.summary || "");
      commitOperation(resp.image_base64, beforeSrc);
      updateUndoRedo(resp.can_undo === true, resp.can_redo === true);

      if (resp.adjustable) {
        const label = resp.adjust_step?.label || resp.adjust_step?.type || t("liveImage.adjustStep", "Adjust");
        updateAdjustControls(true, label, 0);
      }
      updateRepeatControl(resp.repeatable === true && !resp.adjustable);

      if (resp.warnings && resp.warnings.length > 0) {
        for (const w of resp.warnings) {
          addChatBubble("assistant", `\u26a0 ${w}`);
        }
      }
    } catch (err) {
      cancelOperation();
      addChatBubble("assistant", `Error: ${err.message}`);
    } finally {
      setLoading(false);
    }
  }

  async function doAdjust(direction) {
    if (!state.sessionId || state.isLoading) return;
    const beforeSrc = beginOperation();
    setLoading(true);
    try {
      const resp = await api.post(API_ENDPOINTS.pi.liveImageChat.adjust, {
        session_id: state.sessionId,
        direction,
      });
      commitOperation(resp.image_base64, beforeSrc);
      updateAdjustControls(true, state.adjustLabel, resp.adjust_count || 0);
    } catch (err) {
      cancelOperation();
      addChatBubble("assistant", `Error: ${err.message}`);
    } finally {
      setLoading(false);
    }
  }

  async function doRepeat() {
    if (!state.sessionId || state.isLoading || !state.repeatActive) return;
    const beforeSrc = beginOperation();
    setLoading(true);
    try {
      const resp = await api.post(API_ENDPOINTS.pi.liveImageChat.repeat, {
        session_id: state.sessionId,
      });
      addChatBubble("assistant", resp.summary || t("liveImage.repeatDone", "Operation applied again."));
      commitOperation(resp.image_base64, beforeSrc);
      updateUndoRedo(resp.can_undo === true, resp.can_redo === true);
      updateRepeatControl(resp.repeatable !== false);
    } catch (err) {
      cancelOperation();
      addChatBubble("assistant", `Error: ${err.message}`);
    } finally {
      setLoading(false);
    }
  }

  async function doUndo() {
    if (!state.sessionId || state.isLoading) return;
    const beforeSrc = beginOperation();
    setLoading(true);
    try {
      const resp = await api.post(API_ENDPOINTS.pi.liveImageChat.undo, {
        session_id: state.sessionId,
      });
      commitOperation(resp.image_base64, beforeSrc);
      updateUndoRedo(resp.can_undo, resp.can_redo);
    } catch (err) {
      cancelOperation();
      if (err.status === 404) {
        updateUndoRedo(false, state.canRedo);
      } else {
        addChatBubble("assistant", `Error: ${err.message}`);
      }
    } finally {
      setLoading(false);
    }
  }

  async function doRedo() {
    if (!state.sessionId || state.isLoading) return;
    const beforeSrc = beginOperation();
    setLoading(true);
    try {
      const resp = await api.post(API_ENDPOINTS.pi.liveImageChat.redo, {
        session_id: state.sessionId,
      });
      commitOperation(resp.image_base64, beforeSrc);
      updateUndoRedo(resp.can_undo, resp.can_redo);
    } catch (err) {
      cancelOperation();
      if (err.status === 404) {
        updateUndoRedo(state.canUndo, false);
      } else {
        addChatBubble("assistant", `Error: ${err.message}`);
      }
    } finally {
      setLoading(false);
    }
  }

  async function doReset() {
    if (!state.sessionId || state.isLoading) return;
    if (!window.confirm(t("liveImage.confirmReset", "Delete chat history and restore the original image?"))) return;
    setLoading(true);
    try {
      const resp = await api.post(API_ENDPOINTS.pi.liveImageChat.reset, {
        session_id: state.sessionId,
      });
      state.currentImageSrc = `data:image/jpeg;base64,${resp.image_base64}`;
      state.previousImageSrc = null;
      state.showingPrevious = false;
      renderDisplayedImage();
      updateUndoRedo(false, false);
      updateAdjustControls(false, "", 0);
      updateRepeatControl(false);
      chatHistoryEl.innerHTML = "";
      if (typeof onClose === "function") onClose();
    } catch (err) {
      addChatBubble("assistant", `Error: ${err.message}`);
    } finally {
      setLoading(false);
    }
  }

  async function doExport(format) {
    if (!state.sessionId) return;
    setLoading(true);
    try {
      const resp = await api.post(API_ENDPOINTS.pi.liveImageChat.export, {
        session_id: state.sessionId,
        format,
      });
      addChatBubble("assistant", t("liveImage.exportDone", "Exported: {path}", { path: resp.path || "" }));
    } catch (err) {
      addChatBubble("assistant", `Export error: ${err.message}`);
    } finally {
      setLoading(false);
    }
  }

  async function doClose() {
    if (state.sessionId) {
      try {
        await api.post(API_ENDPOINTS.pi.liveImageChat.close, {
          session_id: state.sessionId,
        });
      } catch {
        // ignore close errors
      }
    }
    overlay.remove();
    if (typeof onClose === "function") onClose();
  }

  return { open, sendChat, doUndo, doRedo, doReset, doClose };
}
