import {
  createAgentSession,
  DefaultResourceLoader,
  getAgentDir,
  SessionManager,
  type ExtensionAPI,
} from "@earendil-works/pi-coding-agent";
import path from "node:path";
import type { AgentConfig } from "../types.js";
import type { ModelService } from "./modelService.js";
import { promptWithTimeout } from "./agentPrompt.js";
import { sidecarExtensionPaths } from "./piExtensions.js";
import { sanitizeProviderPayloadForModel } from "./providerPayload.js";
import { recordProviderResponseHeaders, rememberProviderRateLimitError, waitForProviderSlot } from "./providerRateLimit.js";
import { appendTrafficLog } from "./trafficLog.js";

const SYSTEM_PROMPT = `You are PI Live Image Editor for tile_compile astrophotography software.
The user sees a 1:1 preview of their stacked astronomical image and gives
natural language instructions for image adjustments.

You must return a JSON object with image operations. Do NOT describe what
you would do — return concrete operations with exact parameters.

Available operations:
- brightness: { midtones: -1..1, shadows: -1..1, highlights: -1..1 }
- contrast: { amount: -1..1 }
- saturation: { amount: -1..1 }
- sharpen: { amount: 0..1, radius: 0.5..5 }
- denoise: { strength: 0..1, luminance: bool }
- rmgreen: { strength: 0..1 }
- clahe: { cliplimit: 1..10, tilesize: 8..64 }
- bilateral: { d: 3..15, sigma_color: 10..150, sigma_space: 10..150 }
- threshold: { black_point: 0..1, white_point: 0..1 }
- invert: {}
- reset: {}
- vibrance: { amount: -1..1 }
- color_temperature: { amount: -1..1 }
- unpurple: { amount: 0..1 }
- fixbanding: { amount: 0..1, sigma: 0.5..5 }
- star_desaturation: { amount: 0..1 }
- dehaze: { amount: 0..1 }
- levels: { black: 0..1, white: 0..1, gamma: 0.1..5 } (AI suggests initial values; GUI fine-tunes)
- shadow_recovery: { strength: 0..1 }
- highlight_recovery: { strength: 0..1 }
- color_balance: { red: -1..1, green: -1..1, blue: -1..1, optional shadow_red/green/blue, mid_red/green/blue, highlight_red/green/blue: -1..1 }
- local_contrast: { strength: 0..1, radius: 0.5..10 }
- chroma_denoise: { strength: 0..1, protect: 0..1, mode: "soft"|"strong" }
- curves: GUI-only; never generate curves operations here

Note: clahe, bilateral, denoise and threshold cannot be exactly undone by
negating a parameter (they are non-linear or lossy). Do NOT set
adjustable=true with a negatable adjust_step for these — if finer control
makes sense, ask the user to issue a new command instead of relying on the
+/- buttons for these types.

For signed operations such as brightness, contrast, saturation, vibrance and
color_temperature, commands like "mehr X", "weniger X", "helle X auf" and
"erhoehe X" are adjustable. Set adjustable=true and provide an adjust_step
with a moderate signed increment (e.g. +5%). Do not mark one-sided operations
(unpurple, fixbanding, star_desaturation, dehaze or rmgreen) as adjustable.
Sharpen is repeatable with the same positive parameters, but is not a +/-
operation. You do not need to loop or detect "stop".
For levels, shadow_recovery, highlight_recovery, color_balance,
local_contrast and chroma_denoise, choose conservative initial parameters;
the GUI exposes controls for subsequent fine adjustment. Never return a
curves operation: curves are edited only by the user in the GUI.

Return exactly:
{
  "schema_version": "pi.live-image-chat.v1",
  "summary": "string — was wurde gemacht (auf Deutsch)",
  "operations": [...],
  "adjustable": false,
  "adjust_step": { "type": "string", "params": {...}, "label": "string" },
  "repeatable": false,
  "warnings": []
}`;

function createProviderOptionsExtension(config: AgentConfig, model: any) {
  return (pi: ExtensionAPI) => {
    pi.on("before_provider_request", (event) => {
      if (!event.payload || typeof event.payload !== "object" || Array.isArray(event.payload)) return undefined;
      const payload = { ...(event.payload as Record<string, unknown>) };
      const hasThinking = payload.thinking !== undefined && payload.thinking !== null && payload.thinking !== false;
      if (Number.isFinite(config.temperature) && (!hasThinking || config.temperature === 1)) payload.temperature = config.temperature;
      let effectiveMaxTokens = config.maxTokens;
      if (Number.isFinite(config.maxTokens) && config.maxTokens > 0) {
        const isKiro = String(model?.provider || "") === "kiro-api-key" || String(model?.api || "") === "kiro-api";
        effectiveMaxTokens = hasThinking && !isKiro ? Math.max(config.maxTokens, 32000) : config.maxTokens;
        if (Object.prototype.hasOwnProperty.call(payload, "max_tokens")) payload.max_tokens = effectiveMaxTokens;
        if (Object.prototype.hasOwnProperty.call(payload, "max_output_tokens")) payload.max_output_tokens = effectiveMaxTokens;
      }
      const sanitized = sanitizeProviderPayloadForModel(payload, model, "live_image_chat");
      appendTrafficLog(`live_image_chat provider_options thinking=${hasThinking ? "yes" : "no"} temperature=${String(sanitized.temperature ?? "")} max_tokens=${String(sanitized.max_tokens ?? sanitized.max_output_tokens ?? "")} effective_max_tokens=${String(effectiveMaxTokens)}`);
      return sanitized;
    });
    pi.on("after_provider_response", (event) => {
      recordProviderResponseHeaders(model, event.headers || {}, "live_image_chat");
    });
  };
}

function extractTextContent(value: unknown): string {
  if (typeof value === "string") return value;
  if (Array.isArray(value)) return value.map(extractTextContent).join("");
  if (!value || typeof value !== "object") return "";
  const item = value as Record<string, unknown>;
  if (item.type === "thinking" || item.type === "redacted_thinking") return "";
  if (typeof item.text === "string") return item.text;
  if (typeof item.delta === "string") return item.delta;
  if (typeof item.content === "string") return item.content;
  if (Array.isArray(item.content)) return extractTextContent(item.content);
  return "";
}

function extractAssistantTextFromEvent(event: any): string {
  if (event?.assistantMessageEvent?.type === "text_delta") {
    return typeof event.assistantMessageEvent.delta === "string" ? event.assistantMessageEvent.delta : "";
  }
  if (event?.message?.role === "assistant") return extractTextContent(event.message.content);
  if (Array.isArray(event?.messages)) {
    const assistantMessages = event.messages.filter((message: any) => message?.role === "assistant");
    return extractTextContent(assistantMessages[assistantMessages.length - 1]?.content);
  }
  return "";
}

function parseJsonObject(text: string): Record<string, unknown> {
  const match = text.match(/\{[\s\S]*\}/);
  if (!match) throw new Error("No JSON object found in PI live-image-chat response");
  const parsed = JSON.parse(match[0]);
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) throw new Error("PI live-image-chat response is not an object");
  return parsed as Record<string, unknown>;
}

export class LiveImageChatService {
  constructor(
    private readonly config: AgentConfig,
    private readonly modelService: ModelService,
  ) {}

  async ask(body: any): Promise<Record<string, unknown>> {
    const modelRef = String(body.model || this.config.model || "");
    const model = await this.modelService.findModel(modelRef);
    if (!model) throw new Error(`Model ${modelRef || "(empty)"} not found in PI registry`);

    const agentCwd = process.env.TILE_COMPILE_PROJECT_ROOT
      ? path.resolve(process.env.TILE_COMPILE_PROJECT_ROOT)
      : process.cwd();
    const resourceLoader = new DefaultResourceLoader({
      cwd: agentCwd,
      agentDir: getAgentDir(),
      additionalExtensionPaths: sidecarExtensionPaths(),
      extensionFactories: [createProviderOptionsExtension(this.config, model)],
    });
    await resourceLoader.reload();

    const { session } = await createAgentSession({
      cwd: agentCwd,
      model,
      modelRuntime: await this.modelService.getModelRuntime(),
      sessionManager: SessionManager.inMemory(),
      resourceLoader,
      tools: [],
    });

    let responseText = "";
    let assistantError = "";
    const startedAt = Date.now();
    const unsubscribe = session.subscribe((event: any) => {
      if (["agent_start", "message_start", "message_update", "message_end", "agent_end"].includes(String(event.type))) {
        appendTrafficLog(`live_image_chat agent_event ${String(event.type)} role=${String(event.message?.role || "")} stop=${String(event.message?.stopReason || "")}`);
      }
      if (event.type === "message_end" && event.message?.role === "assistant" && event.message?.stopReason === "error") {
        assistantError = String(event.message?.errorMessage || "");
        appendTrafficLog(`live_image_chat assistant_error ${assistantError}`);
      }
      const text = extractAssistantTextFromEvent(event);
      if (text && text.length >= responseText.length) responseText = text;
    });

    try {
      const hasImage = Boolean(body.image_base64);
      const images = hasImage
        ? [{ type: "image" as const, data: String(body.image_base64), mimeType: String(body.image_mime || "image/jpeg") }]
        : undefined;
      const userMessage = String(body.message ?? body.prompt ?? "");
      const operationHistory = body.operation_history
        ? `\nPREVIOUS OPERATIONS: ${JSON.stringify(body.operation_history)}`
        : "";
      const imageContext = hasImage
        ? "CURRENT IMAGE ANALYSIS: <analyze the provided image>"
        : "CURRENT IMAGE ANALYSIS: <no image provided this turn — rely on previous analysis and operation history>";
      const prompt = `${SYSTEM_PROMPT}\n\nUSER MESSAGE: ${userMessage}\n${imageContext}${operationHistory}`;
      appendTrafficLog(`live_image_chat prompt model=${model.id} has_image=${hasImage ? "yes" : "no"} prompt_length=${prompt.length}`);
      appendTrafficLog(`live_image_chat prompt_text ${prompt.substring(0, 50000)}`);
      await waitForProviderSlot(model, prompt, this.config, "live_image_chat");
      await promptWithTimeout(
        session,
        prompt,
        this.config.timeoutMs,
        `PI live-image-chat request (model=${model.id}, provider=${model.provider || "unknown"})`,
        images ? { images } : undefined,
      );
    } finally {
      unsubscribe();
      session.dispose();
    }

    appendTrafficLog(`live_image_chat raw_response ${responseText.substring(0, 20000)}`);
    if (!responseText.trim() && assistantError) {
      rememberProviderRateLimitError(model, assistantError, "live_image_chat");
      throw new Error(`PI live-image-chat provider error: ${assistantError}`);
    }
    const parsed = parseJsonObject(responseText);
    parsed._meta = {
      model: model.id,
      provider: model.provider,
      has_image: Boolean(body.image_base64),
      duration_ms: Date.now() - startedAt,
    };
    return parsed;
  }
}
