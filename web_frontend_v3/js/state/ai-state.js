// js/state/ai-state.js – KI-Config, Modelle, Auth

import { getStore } from "./store.js";

const store = getStore("ai-state", {
  config: null,
  models: [],
  currentAnalysis: null,
  analysisHistory: [],
  trafficLog: [],
  loading: false,
  error: null,
  aiFormData: {
    mount: "EQ",
    object_type: "Galaxie",
    camera: "Consumer OSC",
    calibration_darks: false,
    calibration_flats: false,
    calibration_bias: false,
    notes: "",
    provider: "anthropic",
    model: "claude-sonnet-4-20250514",
    apiKey: "",
  },
});

export function getAiState() { return store.getState(); }
export function setAiState(patch) { store.setState(patch); }
export function onAiChange(fn) { return store.subscribe(fn); }

export function getAiFormData() { return store.getState().aiFormData; }
export function setAiFormData(patch) { store.setState({ aiFormData: { ...getAiFormData(), ...patch } }); }

