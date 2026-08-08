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
      const sanitized = sanitizeProviderPayloadForModel(payload, model, "run_chat");
      appendTrafficLog(`run_chat provider_options thinking=${hasThinking ? "yes" : "no"} temperature=${String(sanitized.temperature ?? "")} max_tokens=${String(sanitized.max_tokens ?? sanitized.max_output_tokens ?? "")} effective_max_tokens=${String(effectiveMaxTokens)}`);
      return sanitized;
    });
    pi.on("after_provider_response", (event) => {
      recordProviderResponseHeaders(model, event.headers || {}, "run_chat");
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
  if (!match) throw new Error("No JSON object found in PI run-chat response");
  const parsed = JSON.parse(match[0]);
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) throw new Error("PI run-chat response is not an object");
  return parsed as Record<string, unknown>;
}

export class RunChatService {
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
        appendTrafficLog(`run_chat agent_event ${String(event.type)} role=${String(event.message?.role || "")} stop=${String(event.message?.stopReason || "")}`);
      }
      if (event.type === "message_end" && event.message?.role === "assistant" && event.message?.stopReason === "error") {
        assistantError = String(event.message?.errorMessage || "");
        appendTrafficLog(`run_chat assistant_error ${assistantError}`);
      }
      if (event.type === "message_update" && event.assistantMessageEvent?.type === "text_delta") {
        const delta = extractAssistantTextFromEvent(event);
        if (delta) responseText += delta;
        return;
      }
      if (event.type === "message_end" && event.message?.role === "assistant") {
        const fullText = extractAssistantTextFromEvent(event);
        if (fullText) responseText = fullText;
        return;
      }
      const text = extractAssistantTextFromEvent(event);
      if (text && text.length >= responseText.length) responseText = text;
    });

    try {
      const images = body.image_base64
        ? [{ type: "image" as const, data: String(body.image_base64), mimeType: String(body.image_mime || "image/png") }]
        : undefined;
      const aiRequestSection = body.ai_request
        ? [
            "=== AI REQUEST V2 (canonical task/context/memory container) ===",
            "Prefer this structured container for task intent, run context, image status, memory evidence and conversation state. The legacy prompt below is compatibility detail.",
            JSON.stringify(body.ai_request, null, 2),
            "",
          ].join("\n")
        : "";
      const prompt = `${aiRequestSection}${String(body.prompt || "")}`;
      appendTrafficLog(`run_chat prompt model=${model.id} has_image=${images ? "yes" : "no"} ai_request=${body.ai_request ? "yes" : "no"} prompt_length=${prompt.length}`);
      appendTrafficLog(`run_chat prompt_text ${prompt.substring(0, 50000)}`);
      await waitForProviderSlot(model, prompt, this.config, "run_chat");
      await promptWithTimeout(
        session,
        prompt,
        this.config.timeoutMs,
        `PI run-chat request (model=${model.id}, provider=${model.provider || "unknown"})`,
        images ? { images } : undefined,
      );
    } finally {
      unsubscribe();
      session.dispose();
    }

    appendTrafficLog(`run_chat raw_response ${responseText.substring(0, 20000)}`);
    if (!responseText.trim() && assistantError) {
      rememberProviderRateLimitError(model, assistantError, "run_chat");
      throw new Error(`PI run-chat provider error: ${assistantError}`);
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
