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
  { category: "misc", label: "liveImage.cmd.crop", action: "crop", help: "liveImage.cmd.crop.help", phase: 1 },
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
    selectedPresetId: "",
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
    if (cropOverlayEl && cropOverlayEl._refreshGeometry) cropOverlayEl._refreshGeometry();
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

  const presetRow = document.createElement("div");
  presetRow.className = "live-image-viewer__presets";
  const presetSelect = document.createElement("select");
  presetSelect.className = "live-image-viewer__dropdown live-image-viewer__preset-select";
  const presetPlaceholder = document.createElement("option");
  presetPlaceholder.value = "";
  presetPlaceholder.textContent = t("liveImage.presetSelect", "Preset auswählen...");
  presetSelect.appendChild(presetPlaceholder);
  const presetApplyBtn = document.createElement("button");
  presetApplyBtn.type = "button";
  presetApplyBtn.className = "live-image-viewer__btn";
  presetApplyBtn.textContent = t("liveImage.presetApply", "Anwenden");
  const presetSaveBtn = document.createElement("button");
  presetSaveBtn.type = "button";
  presetSaveBtn.className = "live-image-viewer__btn";
  presetSaveBtn.textContent = t("liveImage.presetSave", "Sichern");
  const presetSaveAsBtn = document.createElement("button");
  presetSaveAsBtn.type = "button";
  presetSaveAsBtn.className = "live-image-viewer__btn";
  presetSaveAsBtn.textContent = t("liveImage.presetSaveAs", "Sichern unter");
  presetRow.append(presetSelect, presetApplyBtn, presetSaveBtn, presetSaveAsBtn);
  chatPanel.appendChild(presetRow);

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
    opt.value = cmd.command || cmd.action || "";
    opt.dataset.help = cmd.help;
    if (cmd.action) opt.dataset.action = cmd.action;
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
    const cropAction = selected.dataset.action;
    if (cropAction === "crop") {
      openCropOverlay();
    } else {
      sendChat(selected.value);
    }
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

  function addChatBubble(role, text, operations = null) {
    const bubble = document.createElement("div");
    bubble.className = `live-image-viewer__chat-bubble live-image-viewer__chat-bubble--${role}`;
    bubble.textContent = text;
    if (role === "assistant" && Array.isArray(operations) && operations.length > 0) {
      bubble.classList.add("live-image-viewer__chat-bubble--reapply");
      bubble.title = t("liveImage.reapplyTitle", "Klicken, um diesen Befehl erneut anzuwenden");
      bubble.addEventListener("click", () => reapplyOperations(operations));
    }
    chatHistoryEl.appendChild(bubble);
    chatHistoryEl.scrollTop = chatHistoryEl.scrollHeight;
  }

  function restoreChatHistory(history) {
    if (!Array.isArray(history)) return;
    chatHistoryEl.innerHTML = "";
    for (const entry of history) {
      if (!entry || !entry.role) continue;
      addChatBubble(entry.role, entry.content || entry.text || "", entry.operations);
    }
  }

  async function reapplyOperations(operations) {
    if (!state.sessionId || state.isLoading || !Array.isArray(operations) || !operations.length) return;
    if (!window.confirm(t("liveImage.reapplyConfirm", "Befehl noch einmal anwenden?"))) return;
    const beforeSrc = beginOperation();
    setLoading(true);
    try {
      const resp = await api.post(API_ENDPOINTS.pi.liveImageChat.reapply, {
        session_id: state.sessionId,
        operations,
      });
      addChatBubble("assistant", resp.summary || t("liveImage.repeatDone", "Operation erneut angewendet."), resp.operations || operations);
      commitOperation(resp.image_base64, beforeSrc);
      updateUndoRedo(resp.can_undo === true, resp.can_redo === true);
      updateAdjustControls(false, "", 0);
      updateRepeatControl(false);
    } catch (err) {
      cancelOperation();
      addChatBubble("assistant", `Error: ${err.message}`);
    } finally {
      setLoading(false);
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

  function populatePresets(items) {
    presetSelect.innerHTML = "";
    presetSelect.appendChild(presetPlaceholder);
    for (const item of (Array.isArray(items) ? items : [])) {
      const option = document.createElement("option");
      option.value = item.id || "";
      option.textContent = `${item.name || item.id} (${item.operation_count || 0})`;
      presetSelect.appendChild(option);
    }
    presetSelect.value = state.selectedPresetId;
    if (presetSelect.value !== state.selectedPresetId) state.selectedPresetId = "";
  }

  async function loadPresets() {
    try {
      const resp = await api.get(API_ENDPOINTS.pi.liveImageChat.presets);
      populatePresets(resp.items || []);
    } catch (err) {
      addChatBubble("assistant", `Error: ${err.message}`);
    }
  }

  presetSelect.addEventListener("change", () => { state.selectedPresetId = presetSelect.value; });
  async function doSavePresetAs() {
    if (!state.sessionId || state.isLoading) return;
    const name = window.prompt(t("liveImage.presetNamePrompt", "Name des Presets:"));
    if (!name || !name.trim()) return;
    try {
      const resp = await api.post(API_ENDPOINTS.pi.liveImageChat.presetSaveAs, { session_id: state.sessionId, name: name.trim() });
      state.selectedPresetId = resp.preset?.id || "";
      await loadPresets();
      presetSelect.value = state.selectedPresetId;
    } catch (err) { window.alert(err.message); }
  }
  presetSaveAsBtn.addEventListener("click", () => doSavePresetAs());
  presetSaveBtn.addEventListener("click", async () => {
    if (!state.sessionId) return;
    if (!state.selectedPresetId) { await doSavePresetAs(); return; }
    if (!window.confirm(t("liveImage.presetOverwriteConfirm", "Ausgewähltes Preset überschreiben?"))) return;
    try { await api.post(API_ENDPOINTS.pi.liveImageChat.presetSave, { session_id: state.sessionId, preset_id: state.selectedPresetId }); await loadPresets(); }
    catch (err) { window.alert(err.message); }
  });
  presetApplyBtn.addEventListener("click", () => doApplyPreset());

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
      await loadPresets();
    } catch (err) {
      addChatBubble("assistant", `Error: ${err.message}`);
    } finally {
      setLoading(false);
    }
  }

  async function doApplyPreset() {
    if (!state.sessionId || state.isLoading || !state.selectedPresetId) { window.alert(t("liveImage.presetNoSelection", "Bitte zuerst ein Preset auswählen.")); return; }
    const beforeSrc = beginOperation();
    setLoading(true);
    try {
      const resp = await api.post(API_ENDPOINTS.pi.liveImageChat.presetApply, { session_id: state.sessionId, preset_id: state.selectedPresetId });
      addChatBubble("assistant", t("liveImage.presetApplied", "Preset angewendet."));
      commitOperation(resp.image_base64, beforeSrc);
      updateUndoRedo(resp.can_undo === true, resp.can_redo === true);
      updateAdjustControls(false, "", 0);
      updateRepeatControl(false);
    } catch (err) { cancelOperation(); addChatBubble("assistant", `Error: ${err.message}`); }
    finally { setLoading(false); }
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
      addChatBubble("assistant", resp.summary || "", resp.operations);
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
      addChatBubble("assistant", resp.summary || t("liveImage.repeatDone", "Operation applied again."), resp.operations);
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
    closeCropOverlay();
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

  // --- Crop overlay (with rotation) ---
  let cropOverlayEl = null;

  function openCropOverlay() {
    if (!state.sessionId || !imgEl.naturalWidth) return;
    if (cropOverlayEl) closeCropOverlay();

    const rect = imgEl.getBoundingClientRect();
    const wrapRect = imageWrap.getBoundingClientRect();
    let imgLeft = rect.left - wrapRect.left;
    let imgTop = rect.top - wrapRect.top;
    let imgW = rect.width;
    let imgH = rect.height;

    function visibleImageBounds() {
      const wrapW = imageWrap.clientWidth;
      const wrapH = imageWrap.clientHeight;
      const left = Math.max(0, imgLeft);
      const top = Math.max(0, imgTop);
      const right = Math.min(wrapW, imgLeft + imgW);
      const bottom = Math.min(wrapH, imgTop + imgH);
      return {
        left,
        top,
        right: Math.max(left, right),
        bottom: Math.max(top, bottom),
      };
    }

    const margin = 0.05;
    const initialBounds = visibleImageBounds();
    let cropCx = (initialBounds.left + initialBounds.right) / 2;
    let cropCy = (initialBounds.top + initialBounds.bottom) / 2;
    let cropW = (initialBounds.right - initialBounds.left) * (1 - 2 * margin);
    let cropH = (initialBounds.bottom - initialBounds.top) * (1 - 2 * margin);
    let cropAngle = 0;

    cropOverlayEl = document.createElement("div");
    cropOverlayEl.className = "live-image-viewer__crop-overlay";

    // SVG dark mask with a rotated-rect hole
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("class", "live-image-viewer__crop-svg");
    svg.style.position = "absolute";
    svg.style.left = "0";
    svg.style.top = "0";
    svg.style.width = "100%";
    svg.style.height = "100%";
    svg.style.pointerEvents = "none";
    cropOverlayEl.appendChild(svg);

    const defs = document.createElementNS("http://www.w3.org/2000/svg", "defs");
    svg.appendChild(defs);
    const maskEl = document.createElementNS("http://www.w3.org/2000/svg", "mask");
    maskEl.setAttribute("id", "cropMaskDyn");
    maskEl.setAttribute("maskUnits", "userSpaceOnUse");
    defs.appendChild(maskEl);

    const maskBg = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    maskBg.setAttribute("fill", "white");
    maskEl.appendChild(maskBg);

    let maskHole = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
    maskHole.setAttribute("fill", "black");
    maskEl.appendChild(maskHole);

    const maskRect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    maskRect.setAttribute("fill", "rgba(0,0,0,0.55)");
    maskRect.setAttribute("mask", "url(#cropMaskDyn)");
    svg.appendChild(maskRect);

    // Crop box (rotated via CSS transform)
    const cropBox = document.createElement("div");
    cropBox.className = "live-image-viewer__crop-box";
    cropOverlayEl.appendChild(cropBox);

    const hintLabel = document.createElement("div");
    hintLabel.className = "live-image-viewer__crop-hint";
    hintLabel.textContent = t("liveImage.cropHint", "Drag handles to define the crop region");
    cropOverlayEl.appendChild(hintLabel);

    const btnRow = document.createElement("div");
    btnRow.className = "live-image-viewer__crop-buttons";
    const okBtn = document.createElement("button");
    okBtn.className = "live-image-viewer__btn live-image-viewer__btn--crop-ok";
    okBtn.textContent = t("liveImage.cropOk", "OK");
    okBtn.addEventListener("click", (e) => { e.stopPropagation(); applyCrop(); });
    const cancelBtn = document.createElement("button");
    cancelBtn.className = "live-image-viewer__btn live-image-viewer__btn--crop-cancel";
    cancelBtn.textContent = t("liveImage.cropCancel", "Cancel");
    cancelBtn.addEventListener("click", (e) => { e.stopPropagation(); closeCropOverlay(); });
    btnRow.appendChild(cancelBtn);
    btnRow.appendChild(okBtn);
    cropOverlayEl.appendChild(btnRow);

    // Rotation handle
    const rotHandle = document.createElement("div");
    rotHandle.className = "live-image-viewer__crop-rot-handle";
    rotHandle.textContent = "\u21ba";
    cropOverlayEl.appendChild(rotHandle);

    const rotLine = document.createElement("div");
    rotLine.className = "live-image-viewer__crop-rot-line";
    cropOverlayEl.appendChild(rotLine);

    imageWrap.appendChild(cropOverlayEl);

    // 8 resize handles
    const handles = [];
    const handlePositions = ["nw", "n", "ne", "e", "se", "s", "sw", "w"];
    for (const pos of handlePositions) {
      const h = document.createElement("div");
      h.className = `live-image-viewer__crop-handle live-image-viewer__crop-handle--${pos}`;
      h.dataset.pos = pos;
      cropOverlayEl.appendChild(h);
      handles.push(h);
    }

    function rotPt(x, y, cx, cy, deg) {
      const rad = deg * Math.PI / 180;
      const cos = Math.cos(rad), sin = Math.sin(rad);
      return {
        x: cx + (x - cx) * cos - (y - cy) * sin,
        y: cy + (x - cx) * sin + (y - cy) * cos,
      };
    }

    function updateCropVisuals() {
      const wrapW = imageWrap.clientWidth;
      const wrapH = imageWrap.clientHeight;
      maskBg.setAttribute("x", "0");
      maskBg.setAttribute("y", "0");
      maskBg.setAttribute("width", wrapW);
      maskBg.setAttribute("height", wrapH);

      const hx = cropCx - cropW / 2;
      const hy = cropCy - cropH / 2;
      const corners = [
        rotPt(hx, hy, cropCx, cropCy, cropAngle),
        rotPt(hx + cropW, hy, cropCx, cropCy, cropAngle),
        rotPt(hx + cropW, hy + cropH, cropCx, cropCy, cropAngle),
        rotPt(hx, hy + cropH, cropCx, cropCy, cropAngle),
      ];
      maskHole.setAttribute("points", corners.map(c => `${c.x},${c.y}`).join(" "));

      cropBox.style.left = `${hx}px`;
      cropBox.style.top = `${hy}px`;
      cropBox.style.width = `${cropW}px`;
      cropBox.style.height = `${cropH}px`;
      cropBox.style.transform = `rotate(${cropAngle}deg)`;

      for (const h of handles) {
        const pos = h.dataset.pos;
        let lx, ly;
        if (pos.includes("w")) lx = hx;
        else if (pos.includes("e")) lx = hx + cropW;
        else lx = hx + cropW / 2;
        if (pos.includes("n")) ly = hy;
        else if (pos.includes("s")) ly = hy + cropH;
        else ly = hy + cropH / 2;
        const rp = rotPt(lx, ly, cropCx, cropCy, cropAngle);
        h.style.left = `${rp.x - 6}px`;
        h.style.top = `${rp.y - 6}px`;
      }

      // Rotation handle above top-center
      const topMid = rotPt(cropCx, hy - 30, cropCx, cropCy, cropAngle);
      rotHandle.style.left = `${topMid.x - 12}px`;
      rotHandle.style.top = `${topMid.y - 12}px`;
      const lineEnd = rotPt(cropCx, hy, cropCx, cropCy, cropAngle);
      const lineLen = Math.hypot(lineEnd.x - topMid.x, lineEnd.y - topMid.y);
      const lineAng = Math.atan2(lineEnd.y - topMid.y, lineEnd.x - topMid.x) * 180 / Math.PI;
      rotLine.style.left = `${topMid.x}px`;
      rotLine.style.top = `${topMid.y}px`;
      rotLine.style.width = `${lineLen}px`;
      rotLine.style.height = "1px";
      rotLine.style.transformOrigin = "0 0";
      rotLine.style.transform = `rotate(${lineAng}deg)`;

      // Buttons are fixed by CSS at the viewport's top-right corner. They do
      // not move with the transformed image or the crop rectangle.
      btnRow.style.left = "";
      btnRow.style.top = "";

      // Hint above top-center
      const hintPos = rotPt(cropCx, hy - 4, cropCx, cropCy, cropAngle);
      hintLabel.style.left = `${hintPos.x - 100}px`;
      hintLabel.style.top = `${hintPos.y - 24}px`;
    }

    function clampCrop() {
      const bounds = visibleImageBounds();
      const boundsW = Math.max(20, bounds.right - bounds.left);
      const boundsH = Math.max(20, bounds.bottom - bounds.top);
      cropW = Math.max(20, Math.min(cropW, boundsW));
      cropH = Math.max(20, Math.min(cropH, boundsH));
      cropCx = Math.max(bounds.left + cropW / 2,
        Math.min(cropCx, bounds.right - cropW / 2));
      cropCy = Math.max(bounds.top + cropH / 2,
        Math.min(cropCy, bounds.bottom - cropH / 2));
    }

    function syncImageGeometry() {
      const oldW = imgW;
      const oldH = imgH;
      const oldRelCx = oldW > 0 ? (cropCx - imgLeft) / oldW : 0.5;
      const oldRelCy = oldH > 0 ? (cropCy - imgTop) / oldH : 0.5;
      const oldRelW = oldW > 0 ? cropW / oldW : 0.9;
      const oldRelH = oldH > 0 ? cropH / oldH : 0.9;
      const currentRect = imgEl.getBoundingClientRect();
      const currentWrap = imageWrap.getBoundingClientRect();
      imgLeft = currentRect.left - currentWrap.left;
      imgTop = currentRect.top - currentWrap.top;
      imgW = currentRect.width;
      imgH = currentRect.height;
      cropCx = imgLeft + oldRelCx * imgW;
      cropCy = imgTop + oldRelCy * imgH;
      cropW = oldRelW * imgW;
      cropH = oldRelH * imgH;
    }

    function refreshAll() {
      syncImageGeometry();
      clampCrop();
      updateCropVisuals();
    }

    refreshAll();

    // --- Dragging ---
    let dragMode = null;
    let dragStartX = 0, dragStartY = 0;
    let startCx = 0, startCy = 0, startW = 0, startH = 0;

    for (const h of handles) {
      h.addEventListener("mousedown", (e) => {
        e.preventDefault();
        e.stopPropagation();
        dragMode = h.dataset.pos;
        dragStartX = e.clientX;
        dragStartY = e.clientY;
        startCx = cropCx; startCy = cropCy;
        startW = cropW; startH = cropH;
      });
    }

    cropBox.addEventListener("mousedown", (e) => {
      e.preventDefault();
      e.stopPropagation();
      dragMode = "move";
      dragStartX = e.clientX;
      dragStartY = e.clientY;
      startCx = cropCx; startCy = cropCy;
    });

    rotHandle.addEventListener("mousedown", (e) => {
      e.preventDefault();
      e.stopPropagation();
      dragMode = "rotate";
      dragStartX = e.clientX;
      dragStartY = e.clientY;
    });

    function onMouseMove(e) {
      if (!dragMode) return;
      const dx = e.clientX - dragStartX;
      const dy = e.clientY - dragStartY;

      if (dragMode === "move") {
        cropCx = startCx + dx;
        cropCy = startCy + dy;
      } else if (dragMode === "rotate") {
        const wr = imageWrap.getBoundingClientRect();
        const mx = e.clientX - wr.left;
        const my = e.clientY - wr.top;
        cropAngle = Math.atan2(my - cropCy, mx - cropCx) * 180 / Math.PI + 90;
      } else {
        const rad = -cropAngle * Math.PI / 180;
        const cos = Math.cos(rad), sin = Math.sin(rad);
        const ldx = dx * cos - dy * sin;
        const ldy = dx * sin + dy * cos;
        if (dragMode.includes("w")) cropW = Math.max(20, startW - ldx);
        if (dragMode.includes("e")) cropW = Math.max(20, startW + ldx);
        if (dragMode.includes("n")) cropH = Math.max(20, startH - ldy);
        if (dragMode.includes("s")) cropH = Math.max(20, startH + ldy);
      }
      refreshAll();
    }

    function onMouseUp() { dragMode = null; }

    window.addEventListener("mousemove", onMouseMove);
    window.addEventListener("mouseup", onMouseUp);

    cropOverlayEl._cleanup = () => {
      window.removeEventListener("mousemove", onMouseMove);
      window.removeEventListener("mouseup", onMouseUp);
    };

    function closeCropOverlay() {
      if (!cropOverlayEl) return;
      if (cropOverlayEl._cleanup) cropOverlayEl._cleanup();
      cropOverlayEl.remove();
      cropOverlayEl = null;
    }

    function applyCrop() {
      if (!state.sessionId) { closeCropOverlay(); return; }
      const scaleX = imgEl.naturalWidth / imgW;
      const scaleY = imgEl.naturalHeight / imgH;
      const px = Math.round((cropCx - imgLeft) * scaleX);
      const py = Math.round((cropCy - imgTop) * scaleY);
      const pw = Math.round(cropW * scaleX);
      const ph = Math.round(cropH * scaleY);

      closeCropOverlay();

      const beforeSrc = beginOperation();
      addChatBubble("user", t("liveImage.cmd.crop", "Crop"));
      setLoading(true);
      updateAdjustControls(false, "", 0);
      updateRepeatControl(false);

      const msg = (Math.abs(cropAngle) < 0.5)
        ? `crop ${px - Math.round(pw / 2)} ${py - Math.round(ph / 2)} ${pw} ${ph}`
        : `crop_rotated ${px} ${py} ${pw} ${ph} ${cropAngle.toFixed(1)}`;

      api.post(API_ENDPOINTS.pi.liveImageChat.chat, {
        session_id: state.sessionId,
        message: msg,
      }).then((resp) => {
        addChatBubble("assistant", resp.summary || "");
        commitOperation(resp.image_base64, beforeSrc);
        updateUndoRedo(resp.can_undo === true, resp.can_redo === true);
        updateRepeatControl(resp.repeatable === true && !resp.adjustable);
      }).catch((err) => {
        cancelOperation();
        addChatBubble("assistant", `Error: ${err.message}`);
      }).finally(() => {
        setLoading(false);
      });
    }

    cropOverlayEl._closeCropOverlay = closeCropOverlay;
    cropOverlayEl._applyCrop = applyCrop;
    cropOverlayEl._refreshGeometry = () => {
      syncImageGeometry();
      clampCrop();
      updateCropVisuals();
    };
  }


  function closeCropOverlay() {
    if (cropOverlayEl && cropOverlayEl._closeCropOverlay) cropOverlayEl._closeCropOverlay();
  }

  function applyCrop() {
    if (cropOverlayEl && cropOverlayEl._applyCrop) cropOverlayEl._applyCrop();
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
