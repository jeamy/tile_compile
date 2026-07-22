import { appendTrafficLog } from "./trafficLog.js";

export function modelCapabilitySummary(model: any) {
  return {
    reasoning: Boolean(model?.reasoning),
    input: Array.isArray(model?.input) ? model.input : [],
    context_window: Number.isFinite(Number(model?.contextWindow)) ? Number(model.contextWindow) : null,
    max_tokens: Number.isFinite(Number(model?.maxTokens)) ? Number(model.maxTokens) : null,
    thinking_level_map: model?.thinkingLevelMap || null,
  };
}

export function sanitizeProviderPayloadForModel(payload: Record<string, unknown>, model: any, label: string) {
  const provider = String(model?.provider || "").toLowerCase();
  const id = String(model?.id || "");
  const supportsReasoning = Boolean(model?.reasoning);
  const removed: string[] = [];
  const clamped: string[] = [];
  const remove = (key: string) => {
    if (Object.prototype.hasOwnProperty.call(payload, key)) {
      delete payload[key];
      removed.push(key);
    }
  };

  if (!supportsReasoning) {
    remove("thinking");
    remove("reasoning");
    remove("reasoning_effort");
    remove("reasoningEffort");
    remove("prompt_mode");
    remove("promptMode");
  }

  if (provider === "mistral") {
    // PI 0.80.x marks some Mistral models as reasoning-capable, but Mistral's
    // API rejects promptMode=reasoning for several of them. Prefer a successful
    // non-prompt-mode request; models that support effort-based reasoning keep
    // reasoningEffort below.
    const promptMode = payload.promptMode ?? payload.prompt_mode;
    if (promptMode === "reasoning") {
      remove("promptMode");
      remove("prompt_mode");
    }
    if (!supportsReasoning) {
      remove("reasoningEffort");
      remove("reasoning_effort");
    }
  }

  // Clamp max_tokens / max_output_tokens to the model's declared maximum to
  // avoid provider-side rejection or silent truncation.
  const modelMaxTokens = Number.isFinite(Number(model?.maxTokens)) ? Number(model.maxTokens) : 0;
  if (modelMaxTokens > 0) {
    for (const key of ["max_tokens", "max_output_tokens"] as const) {
      if (Object.prototype.hasOwnProperty.call(payload, key)) {
        const requested = Number(payload[key]);
        if (Number.isFinite(requested) && requested > modelMaxTokens) {
          payload[key] = modelMaxTokens;
          clamped.push(`${key}:${requested}→${modelMaxTokens}`);
        }
      }
    }
  }

  if (removed.length > 0 || clamped.length > 0) {
    appendTrafficLog(`${label} provider_payload_sanitized model=${provider}/${id} removed=${removed.join(",") || "none"} clamped=${clamped.join(",") || "none"} reasoning=${supportsReasoning}`);
  }
  return payload;
}
