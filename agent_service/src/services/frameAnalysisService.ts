import {
  createAgentSession,
  DefaultResourceLoader,
  getAgentDir,
  SessionManager,
  type ExtensionAPI,
} from "@earendil-works/pi-coding-agent";
import crypto from "node:crypto";
import path from "node:path";
import type { AgentConfig, ScanAnalysisRequest, ScanAnalysisResponse, ProgressCallback, AnalysisProgressEvent, SessionContext } from "../types.js";
import type { ModelService } from "./modelService.js";
import { promptWithTimeout } from "./agentPrompt.js";
import { sidecarExtensionPaths } from "./piExtensions.js";
import { sanitizeProviderPayloadForModel } from "./providerPayload.js";
import { recordProviderResponseHeaders, rememberProviderRateLimitError, waitForProviderSlot } from "./providerRateLimit.js";
import { appendTrafficLog } from "./trafficLog.js";

const MAX_PROMPT_CHARS = 120_000;
const MAX_AI_REQUEST_JSON_CHARS = 18_000;
const MAX_SCAN_METRICS_JSON_CHARS = 35_000;
const MAX_MEMORY_ITEMS = 4;

function stableNormalize(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(stableNormalize);
  if (!value || typeof value !== "object") return value;
  const normalized: Record<string, unknown> = {};
  for (const key of Object.keys(value as Record<string, unknown>).sort()) {
    normalized[key] = stableNormalize((value as Record<string, unknown>)[key]);
  }
  return normalized;
}

function sha256Text(value: string): string {
  return crypto.createHash("sha256").update(value, "utf8").digest("hex");
}

function sha256Json(value: unknown): string {
  return sha256Text(JSON.stringify(stableNormalize(value)));
}

function boundedJson(value: unknown, maxChars: number): string {
  const text = JSON.stringify(value, null, 2);
  if (text.length <= maxChars) return text;
  return `${text.slice(0, maxChars)}\n... [truncated ${text.length - maxChars} chars]`;
}

function compactLargeJson(value: unknown, depth = 0): unknown {
  if (Array.isArray(value)) {
    if (value.length === 0) return [];
    if (depth >= 4 || value.length > 24) {
      return {
        _truncated_array: true,
        original_length: value.length,
        sample: value.slice(0, 5).map((item) => compactLargeJson(item, depth + 1)),
      };
    }
    return value.map((item) => compactLargeJson(item, depth + 1));
  }
  if (!value || typeof value !== "object") return value;
  const out: Record<string, unknown> = {};
  for (const [key, child] of Object.entries(value as Record<string, unknown>)) {
    if (key === "frames" && Array.isArray(child)) {
      out.frames = { _omitted_array: true, original_length: child.length };
      continue;
    }
    if ((key === "scan_result" || key === "scan_metrics" || key === "base_config" || key === "config_schema" || key === "pi_context") && depth <= 2) {
      out[key] = { _omitted_duplicate_section: true };
      continue;
    }
    out[key] = compactLargeJson(child, depth + 1);
  }
  return out;
}

function compactAiRequestForPrompt(aiRequest: Record<string, any>): unknown {
  const sessionContext = aiRequest.session_context && typeof aiRequest.session_context === "object"
    ? compactLargeJson(aiRequest.session_context)
    : undefined;
  const compact: Record<string, unknown> = {
    schema_version: aiRequest.schema_version,
    task: aiRequest.task,
    user_message: aiRequest.user_message,
    context_signature: compactLargeJson(aiRequest.context_signature),
    run_context: compactLargeJson(aiRequest.run_context),
    image_context: compactLargeJson(aiRequest.image_context),
    artifacts: compactLargeJson(aiRequest.artifacts),
    session_context: sessionContext,
    allowed_config_paths_count: Array.isArray(aiRequest.allowed_config_paths) ? aiRequest.allowed_config_paths.length : undefined,
    expected_response: aiRequest.expected_response,
  };
  if (Array.isArray(aiRequest.conversation)) {
    compact.conversation = aiRequest.conversation.slice(-6).map((item: unknown) => compactLargeJson(item));
    compact.conversation_truncated_from = aiRequest.conversation.length;
  }
  if (Array.isArray(aiRequest.positive_memories)) {
    compact.positive_memories = aiRequest.positive_memories.slice(0, MAX_MEMORY_ITEMS).map((item: unknown) => compactLargeJson(item));
    compact.positive_memories_truncated_from = aiRequest.positive_memories.length;
  }
  if (Array.isArray(aiRequest.negative_memories)) {
    compact.negative_memories = aiRequest.negative_memories.slice(0, MAX_MEMORY_ITEMS).map((item: unknown) => compactLargeJson(item));
    compact.negative_memories_truncated_from = aiRequest.negative_memories.length;
  }
  return compact;
}

function compactScanMetricsForPrompt(scanMetrics: any): unknown {
  if (!scanMetrics || typeof scanMetrics !== "object" || Array.isArray(scanMetrics)) return scanMetrics;
  const out: Record<string, unknown> = {};
  for (const key of [
    "schema_version",
    "frames_total",
    "frame_count",
    "sample_count",
    "aggregate",
    "session_geometry",
    "diagnostics",
    "warnings",
    "quality_spread",
    "registration",
  ]) {
    if (scanMetrics[key] !== undefined) out[key] = compactLargeJson(scanMetrics[key]);
  }
  for (const [key, value] of Object.entries(scanMetrics)) {
    if (out[key] !== undefined || key === "frames") continue;
    if (Array.isArray(value) && value.length > 24) {
      out[key] = { _omitted_array: true, original_length: value.length };
    } else if (typeof value !== "object" || value === null) {
      out[key] = value;
    }
  }
  if (Array.isArray(scanMetrics.frames)) {
    out.frames = { _omitted_array: true, original_length: scanMetrics.frames.length };
  }
  return out;
}

function createProviderOptionsExtension(config: AgentConfig, model: any) {
  return (pi: ExtensionAPI) => {
    pi.on("before_provider_request", (event) => {
      if (!event.payload || typeof event.payload !== "object" || Array.isArray(event.payload)) {
        return undefined;
      }
      const payload = { ...(event.payload as Record<string, unknown>) };
      const hasThinking = payload.thinking !== undefined
        && payload.thinking !== null
        && payload.thinking !== false;
      if (Number.isFinite(config.temperature) && (!hasThinking || config.temperature === 1)) {
        payload.temperature = config.temperature;
      }
      let effectiveMaxTokens = config.maxTokens;
      if (Number.isFinite(config.maxTokens) && config.maxTokens > 0) {
        // When extended thinking is active, non-Kiro models can consume a large
        // portion of max_tokens before emitting text. Kiro's API-key bridge is
        // more sensitive to large output budgets, so use the configured value.
        // (Claude Sonnet 4.x thinking traces alone can use 8-12k tokens.)
        const isKiro = String(model?.provider || "") === "kiro-api-key" || String(model?.api || "") === "kiro-api";
        effectiveMaxTokens = hasThinking && !isKiro ? Math.max(config.maxTokens, 32000) : config.maxTokens;
        if (Object.prototype.hasOwnProperty.call(payload, "max_tokens")) {
          payload.max_tokens = effectiveMaxTokens;
        }
        if (Object.prototype.hasOwnProperty.call(payload, "max_output_tokens")) {
          payload.max_output_tokens = effectiveMaxTokens;
        }
      }
      const sanitized = sanitizeProviderPayloadForModel(payload, model, "scan_analysis");
      appendTrafficLog(`provider_options thinking=${hasThinking ? "yes" : "no"} temperature=${String(sanitized.temperature ?? "")} max_tokens=${String(sanitized.max_tokens ?? sanitized.max_output_tokens ?? "")} effective_max_tokens=${String(effectiveMaxTokens)}`);
      return sanitized;
    });
    pi.on("after_provider_response", (event) => {
      recordProviderResponseHeaders(model, event.headers || {}, "scan_analysis");
    });
  };
}

function extractTextContent(value: unknown): string {
  if (typeof value === "string") return value;
  if (Array.isArray(value)) return value.map(extractTextContent).join("");
  if (!value || typeof value !== "object") return "";
  const item = value as Record<string, unknown>;
  // Skip thinking blocks — they are internal reasoning, not the text output
  if (item.type === "thinking" || item.type === "redacted_thinking") return "";
  if (typeof item.text === "string") return item.text;
  if (typeof item.delta === "string") return item.delta;
  if (typeof item.content === "string") return item.content;
  if (Array.isArray(item.content)) return extractTextContent(item.content);
  return "";
}

function extractAssistantTextFromEvent(event: any): string {
  if (event?.assistantMessageEvent?.type === "text_delta") {
    return typeof event.assistantMessageEvent.delta === "string"
      ? event.assistantMessageEvent.delta
      : "";
  }
  if (event?.message?.role === "assistant") {
    return extractTextContent(event.message.content);
  }
  if (Array.isArray(event?.messages)) {
    const assistantMessages = event.messages.filter((message: any) => message?.role === "assistant");
    const lastAssistant = assistantMessages[assistantMessages.length - 1];
    return extractTextContent(lastAssistant?.content);
  }
  return "";
}

export class FrameAnalysisService {
  constructor(
    private readonly config: AgentConfig,
    private readonly modelService: ModelService,
  ) {}

  async analyze(
    request: ScanAnalysisRequest,
    onProgress?: ProgressCallback
  ): Promise<ScanAnalysisResponse> {
    if (!this.config.enabled && !request.force) {
      throw new Error("AI scan analysis is disabled");
    }

    const modelRef = request.model || this.config.model;
    const model = await this.modelService.findModel(modelRef);
    if (!model) throw new Error(`Model ${modelRef || "(empty)"} not found in PI registry`);

    this.emitProgress(onProgress, { phase: "initializing", message: "Creating AI session...", progress: 5 });

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

    this.emitProgress(onProgress, { phase: "building_prompt", message: "Building analysis prompt...", progress: 10 });
    const prompt = this.buildPrompt(request);
    const requestAuditPayload = {
      schema_version: request.schema_version,
      ai_request: request.ai_request,
      scan_result: request.scan_result,
      base_config: request.base_config,
      config_schema: request.config_schema,
      pi_context: request.pi_context,
      scan_metrics: request.scan_metrics,
      session_context: request.session_context,
      allowed_config_paths: request.allowed_config_paths,
      model: model.id,
      force: request.force,
    };
    appendTrafficLog(`prompt_length ${prompt.length} sections=${[
      'AI REQUEST V2','PI CONTEXT V2','PARAMETER CATALOG','IMAGE QUALITY METRICS','FRAME STATISTICS','CURRENT CONFIG','CONFIG SCHEMA','SCAN RESULT','scan_metrics'
    ].map(s => s + ':' + (prompt.includes(s) ? 'YES' : 'NO')).join(' ')}`);
    appendTrafficLog(`prompt ${prompt.substring(0, 50000)}`);

    let responseText = "";
    let assistantError = "";
    let textDeltaCount = 0;
    const startTime = Date.now();
    let lastProgressEmit = startTime;

    this.emitProgress(onProgress, { phase: "ai_thinking", message: "Waiting for AI response...", progress: 15 });
    await waitForProviderSlot(model, prompt, this.config, "scan_analysis");

    const unsubscribe = session.subscribe((event: any) => {
      if (["agent_start", "message_start", "message_update", "message_end", "agent_end"].includes(String(event.type))) {
        appendTrafficLog(`agent_event ${String(event.type)} role=${String(event.message?.role || "")} stop=${String(event.message?.stopReason || "")}`);
      }
      if (
        event.type === "message_update" &&
        event.message?.role === "assistant" &&
        event.assistantMessageEvent?.type !== "text_delta"
      ) {
        const now = Date.now();
        if (now - lastProgressEmit > 1000) {
          this.emitProgress(onProgress, {
            phase: "ai_thinking",
            message: `AI is still thinking... (${Math.max(1, Math.round((now - startTime) / 1000))}s)`,
            progress: 15,
          });
          lastProgressEmit = now;
        }
      }
      if (event.type === "message_end" && event.message?.role === "assistant" && event.message?.stopReason === "error") {
        assistantError = String(event.message?.errorMessage || "");
        appendTrafficLog(`assistant_error ${assistantError}`);
      }
      if (event.type === "message_update" && event.assistantMessageEvent?.type === "text_delta") {
        const delta = extractAssistantTextFromEvent(event);
        if (!delta) return;
        responseText += delta;
        textDeltaCount++;

        // Emit progress every 100ms to avoid flooding
        const now = Date.now();
        if (now - lastProgressEmit > 100) {
          this.emitProgress(onProgress, {
            phase: "receiving_tokens",
            message: `Receiving response... (${responseText.length} chars)`,
            delta,
            charsReceived: responseText.length,
            progress: Math.min(15 + (responseText.length / 40), 90),
          });
          lastProgressEmit = now;
        }
        return;
      }
      // message_end carries the authoritative complete content from the provider.
      // Replace responseText unconditionally so streaming deltas (which may be
      // partial when stop=length) are superseded by the final message content.
      if (event.type === "message_end" && event.message?.role === "assistant") {
        const fullText = extractAssistantTextFromEvent(event);
        if (fullText) {
          responseText = fullText;
          textDeltaCount++;
          appendTrafficLog(`message_end replaced responseText length=${fullText.length}`);
        }
        return;
      }
      if (event.type === "message_update" && event.message?.role === "assistant") {
        const fullText = extractAssistantTextFromEvent(event);
        if (fullText && fullText.length > responseText.length) {
          const delta = fullText.slice(responseText.length);
          responseText = fullText;
          textDeltaCount++;
          const now = Date.now();
          if (now - lastProgressEmit > 100) {
            this.emitProgress(onProgress, {
              phase: "receiving_tokens",
              message: `Receiving response... (${responseText.length} chars)`,
              delta,
              charsReceived: responseText.length,
              progress: Math.min(15 + (responseText.length / 40), 90),
            });
            lastProgressEmit = now;
          }
        }
      }
    });

    try {
      await promptWithTimeout(
        session,
        prompt,
        this.config.timeoutMs,
        `PI scan-analysis request (model=${model.id}, provider=${model.provider || "unknown"})`,
        undefined,
        {
          maxDurationMs: this.config.maxDurationMs,
          onDiagnostic: (message) => appendTrafficLog(`scan_analysis ${message}`),
        },
      );
    } finally {
      unsubscribe();
      session.dispose();
    }

    this.emitProgress(onProgress, { phase: "parsing_response", message: "Parsing AI response...", progress: 90 });

    const duration = Date.now() - startTime;
    console.log(`[AI Analysis] Completed: ${textDeltaCount} deltas, ${responseText.length} chars, ${duration}ms`);
    appendTrafficLog(`raw_response ${responseText.substring(0, 20000)}`);
    if (!responseText.trim() && assistantError) {
      rememberProviderRateLimitError(model, assistantError, "scan_analysis");
      throw new Error(`PI agent provider error: ${assistantError}`);
    }
    if (!responseText.trim()) {
      const timeoutNote = duration >= (this.config.timeoutMs || 180000) * 0.9
        ? " (likely timeout — provider did not respond within the configured timeout)"
        : "";
      throw new Error(
        `PI agent returned empty response after ${Math.round(duration / 1000)}s` +
        ` (model=${model.id}, provider=${model.provider || "unknown"})${timeoutNote}.` +
        ` Check that the API key is valid and the provider is reachable.`
      );
    }

    const allowedPaths = new Set<string>(
      Array.isArray(request.allowed_config_paths) ? request.allowed_config_paths : []
    );
    const configSchema = request.config_schema || {};
    if (allowedPaths.size === 0) {
      for (const key of Object.keys(configSchema)) allowedPaths.add(key);
    }
    const result = this.parseResponse(responseText, allowedPaths);
    result._meta = {
      streaming_duration_ms: duration,
      response_chars: responseText.length,
      model: model.id,
      provider: model.provider,
      temperature: this.config.temperature,
      max_tokens: this.config.maxTokens,
      prompt,
      prompt_sha256: sha256Text(prompt),
      request_sha256: sha256Json(requestAuditPayload),
      base_config_sha256: request.base_config === undefined ? undefined : sha256Json(request.base_config),
      config_schema_sha256: request.config_schema === undefined ? undefined : sha256Json(request.config_schema),
      pi_context_sha256: request.pi_context === undefined ? undefined : sha256Json(request.pi_context),
      scan_result_sha256: request.scan_result === undefined ? undefined : sha256Json(request.scan_result),
      scan_metrics_sha256: request.scan_metrics === undefined ? undefined : sha256Json(request.scan_metrics),
    };

    this.emitProgress(onProgress, { phase: "complete", message: "Analysis complete", progress: 100 });
    return result;
  }

  private emitProgress(callback: ProgressCallback | undefined, event: AnalysisProgressEvent): void {
    if (callback) {
      try {
        callback(event);
      } catch (e) {
        // Ignore progress callback errors
      }
    }
  }

  private buildPrompt(request: ScanAnalysisRequest): string {
    const aiRequest = request.ai_request && typeof request.ai_request === "object" && !Array.isArray(request.ai_request)
      ? request.ai_request as Record<string, any>
      : {};
    const scanContext = aiRequest.scan_context && typeof aiRequest.scan_context === "object" && !Array.isArray(aiRequest.scan_context)
      ? aiRequest.scan_context as Record<string, any>
      : {};
    const configContext = aiRequest.config && typeof aiRequest.config === "object" && !Array.isArray(aiRequest.config)
      ? aiRequest.config as Record<string, any>
      : {};
    const piContext = request.pi_context && typeof request.pi_context === "object" && !Array.isArray(request.pi_context)
      ? request.pi_context as Record<string, any>
      : aiRequest.pi_context && typeof aiRequest.pi_context === "object" && !Array.isArray(aiRequest.pi_context)
        ? aiRequest.pi_context as Record<string, any>
        : {};
    const parameterCatalog = piContext.parameter_catalog && typeof piContext.parameter_catalog === "object" && !Array.isArray(piContext.parameter_catalog)
      ? piContext.parameter_catalog as Record<string, any>
      : configContext.parameter_catalog && typeof configContext.parameter_catalog === "object" && !Array.isArray(configContext.parameter_catalog)
        ? configContext.parameter_catalog as Record<string, any>
        : {};
    const baseConfig = (request.base_config as any) || configContext.base_config || {};
    const currentMethod = typeof baseConfig?.method === "string"
      ? baseConfig.method
      : baseConfig?.aqmh?.enabled === true
        ? "aqmh"
        : "classic_tile_compile";
    const isAqmhMethod = currentMethod === "aqmh";
    const isClassicOnlyPath = (path: string): boolean =>
      path.startsWith("global_metrics.") ||
      path.startsWith("local_metrics.") ||
      path.startsWith("synthetic.");
    const sessionContextFromAiRequest = aiRequest.session_context && typeof aiRequest.session_context === "object" && !Array.isArray(aiRequest.session_context)
      ? aiRequest.session_context as SessionContext
      : undefined;
    // Build compact schema reference: only leaf paths (non-object types)
    const configSchema = request.config_schema || configContext.config_schema || {};
    const schemaLines: string[] = [];
    for (const [path, info] of Object.entries<any>(configSchema)) {
      if (isAqmhMethod && isClassicOnlyPath(path)) continue;
      if (path.startsWith("aqmh.cherry_pick.")) continue;
      if (path.startsWith("global_metrics.weights.")) continue;
      if (path === "global_metrics.weight_exponent_scale") continue;
      if (path === "aqmh.storage.dtype") continue;
      if (path === "aqmh.storage.max_resident_maps") continue;
      if (info?.type === "object") continue; // skip parent objects
      const parts = [path, `type:${info?.type || "unknown"}`];
      if (info?.enum) parts.push(`enum:${JSON.stringify(info.enum)}`);
      if (info?.minimum !== undefined) parts.push(`min:${info.minimum}`);
      if (info?.exclusiveMinimum !== undefined) parts.push(`exclusive_min:${info.exclusiveMinimum}`);
      if (info?.maximum !== undefined) parts.push(`max:${info.maximum}`);
      if (info?.desc) parts.push(info.desc);
      schemaLines.push(parts.join("  "));
    }
    const catalogLines: string[] = [];
    for (const [path, meta] of Object.entries<any>(parameterCatalog)) {
      if (isAqmhMethod && isClassicOnlyPath(path)) continue;
      const parts = [path];
      if (meta?.current_value !== undefined) parts.push(`current:${JSON.stringify(meta.current_value)}`);
      if (meta?.cpp_default !== undefined) parts.push(`cpp_default:${JSON.stringify(meta.cpp_default)}`);
      if (meta?.schema_default !== undefined) parts.push(`schema_default:${JSON.stringify(meta.schema_default)}`);
      if (meta?.schema_enum !== undefined) parts.push(`schema_enum:${JSON.stringify(meta.schema_enum)}`);
      if (meta?.schema_min !== undefined) parts.push(`schema_min:${JSON.stringify(meta.schema_min)}`);
      if (meta?.schema_exclusive_min !== undefined) parts.push(`schema_exclusive_min:${JSON.stringify(meta.schema_exclusive_min)}`);
      if (Object.prototype.hasOwnProperty.call(meta || {}, "schema_max")) parts.push(`schema_max:${JSON.stringify(meta.schema_max)}`);
      if (Object.prototype.hasOwnProperty.call(meta || {}, "recommended_value")) parts.push(`recommended_value:${JSON.stringify(meta.recommended_value)}`);
      if (meta?.diagnostic_only !== undefined) parts.push(`diagnostic_only:${Boolean(meta.diagnostic_only)}`);
      if (meta?.semantic) parts.push(`semantic:${String(meta.semantic)}`);
      if (Array.isArray(meta?.requires_evidence)) parts.push(`requires_evidence:${JSON.stringify(meta.requires_evidence)}`);
      if (Array.isArray(meta?.hard_rules)) parts.push(`hard_rules:${JSON.stringify(meta.hard_rules)}`);
      catalogLines.push(parts.join("  "));
    }

    // Build compact scan summary (no frame list, just counts and metadata)
    const scan = (request as any).scan_result || scanContext.scan_result || {};
    const frames = Array.isArray(scan.frames) ? scan.frames : [];
    const frameCount = scan.frames_detected ?? scan.frames_total ?? frames.length;

    const numericAgg = (key: string) => {
      const vals = frames.map((f: any) => f[key]).filter((v: any) => typeof v === "number" && isFinite(v));
      if (vals.length === 0) return null;
      const sorted = [...vals].sort((a, b) => a - b);
      const min = sorted[0];
      const max = sorted[sorted.length - 1];
      const mean = vals.reduce((a: number, b: number) => a + b, 0) / vals.length;
      const median = sorted[Math.floor(sorted.length / 2)];
      return { min, max, mean: Math.round(mean * 100) / 100, median, count: vals.length };
    };
    const stringAgg = (key: string) => {
      const vals = frames.map((f: any) => f[key]).filter((v: any) => typeof v === "string" && v);
      if (vals.length === 0) return null;
      const counts: Record<string, number> = {};
      for (const v of vals) counts[v] = (counts[v] || 0) + 1;
      const top = Object.entries(counts).sort((a, b) => b[1] - a[1])[0];
      return { value: top[0], count: top[1], sampleCount: vals.length, uniformInSample: top[1] === vals.length };
    };

    const scanCompact: Record<string, unknown> = {
      color_mode: scan.color_mode,
      bayer_pattern: scan.bayer_pattern,
      frame_count: frameCount,
      color_mode_candidates: scan.color_mode_candidates,
      errors_total: scan.errors_total ?? 0,
    };
    // Top-level metadata from scan summary
    if (scan.image_width != null) scanCompact.image_width = scan.image_width;
    if (scan.image_height != null) scanCompact.image_height = scan.image_height;
    if (scan.input_path != null) scanCompact.input_path = scan.input_path;
    for (const key of ["exposure_seconds", "gain", "temperature_c"]) {
      const agg = numericAgg(key);
      if (!agg) continue;
      scanCompact[key] = agg.min === agg.max
        ? { value: agg.min, uniform_in_sample: true, sample_count: agg.count }
        : { min: agg.min, max: agg.max, median: agg.median, mean: agg.mean, sample_count: agg.count };
    }
    for (const key of ["target", "camera", "telescope"]) {
      const agg = stringAgg(key);
      if (!agg) continue;
      scanCompact[key] = {
        most_common: agg.value,
        count: agg.count,
        sample_count: agg.sampleCount,
        uniform_in_sample: agg.uniformInSample,
      };
    }
    if (frames[0]) {
      const f = frames[0];
      if (f.image_width != null && !scanCompact.image_width) scanCompact.image_width = f.image_width;
      if (f.image_height != null && !scanCompact.image_height) scanCompact.image_height = f.image_height;
    }

    // Build compact current config (only leaf values, flatten dotted paths)
    const configLines: string[] = [];
    const flattenConfig = (obj: any, prefix: string) => {
      if (obj == null) return;
      if (typeof obj !== "object" || Array.isArray(obj)) {
        configLines.push(`${prefix} = ${JSON.stringify(obj)}`);
        return;
      }
      for (const [k, v] of Object.entries(obj)) {
        flattenConfig(v, prefix ? `${prefix}.${k}` : k);
      }
    };
    flattenConfig(baseConfig, "");

    // Use scan_metrics (from scan-metrics CLI) if provided — authoritative quality data
    const scanMetrics = (request.scan_metrics as any) || scanContext.scan_metrics || null;
    const metricsLines: string[] = [];
    if (scanMetrics && scanMetrics.aggregate) {
      const agg = scanMetrics.aggregate;
      metricsLines.push(`Sampled ${scanMetrics.sample_count ?? "?"} of ${scanMetrics.frames_total ?? frameCount} frames:`);
      for (const key of ["fwhm", "background", "noise", "gradient_energy", "sky_gradient", "roundness", "star_count"]) {
        const s = agg[key];
        if (!s) continue;
        if (s.min === s.max) {
          metricsLines.push(`  ${key}: ${s.min} (uniform)`);
        } else {
          const parts = [`  ${key}: median=${s.median}`];
          if (s.mean != null) parts.push(`mean=${typeof s.mean === "number" ? s.mean.toFixed(2) : s.mean}`);
          parts.push(`min=${s.min} max=${s.max}`);
          if (s.p10 != null && s.p90 != null) parts.push(`p10=${s.p10} p90=${s.p90}`);
          parts.push(`(n=${s.count})`);
          metricsLines.push(parts.join(" "));
        }
      }
      metricsLines.push("");
      metricsLines.push("Compact measured scan_metrics JSON (large per-frame arrays omitted):");
      metricsLines.push(boundedJson(compactScanMetricsForPrompt(scanMetrics), MAX_SCAN_METRICS_JSON_CHARS));
    }

    // Aggregate frame statistics if available (frames[] may be truncated by backend)
    const frameStats: string[] = [];
    if (frames.length > 0 && frames.length < frameCount) {
      frameStats.push(`(Note: statistics sampled from ${frames.length} of ${frameCount} total frames)`);
    }
    if (frames.length > 0) {
      for (const key of ["exposure_seconds", "gain", "temperature_c", "fwhm", "snr", "sky_background"]) {
        const agg = numericAgg(key);
        if (!agg) continue;
        if (agg.min === agg.max) {
          const scope = frames.length === frameCount ? `all ${frameCount} frames` : `${agg.count} sampled frames`;
          frameStats.push(`${key}: ${agg.min} (uniform in ${scope})`);
        } else {
          frameStats.push(`${key}: min=${agg.min} max=${agg.max} mean=${agg.mean} (sampled ${agg.count} frames)`);
        }
      }
      // String fields: pick most common value
      for (const key of ["target", "camera", "telescope"]) {
        const agg = stringAgg(key);
        if (agg) {
          const scope = frames.length === frameCount ? `all ${frameCount} frames` : `${agg.sampleCount} sampled frames`;
          frameStats.push(`${key}: "${agg.value}" (${agg.count}/${scope})`);
        }
      }
    }

    const sections = [
      "You are an expert in astronomical image stacking and tile_compile configuration.",
      "Analyze the scan result, measured image-quality statistics, and current configuration below. Create a suitable tile_compile configuration by returning a precise patch of recommended config changes.",
      "Separate measured facts, configured values, and astrophotography assumptions rigorously.",
      "",
      "RESPONSE FORMAT: Respond with exactly one JSON object. No markdown, no prose, no code fences.",
      "The root JSON object must have these fields:",
      '  "schema_version": "pi.scan-analysis.v1"  (exactly this string)',
      '  "summary": string (one paragraph describing the dataset and key observations)',
      '  "confidence": number 0..1 (overall confidence)',
      '  "detected_scenarios": string[] (e.g. ["osc_short_exposure","large_frame_count"])',
      '  "recommendations": array of recommendation objects',
      '  "warnings": string[] (notable issues, missing data, caveats)',
      '  "review_required": boolean (true if any recommendation needs manual review)',
      "",
      "Each recommendation object must have exactly these fields:",
      '  "id": string (e.g. "rec_stacking_method")',
      '  "path": string (MUST be one of the paths listed in CONFIG SCHEMA below)',
      '  "value": the recommended value (MUST match the type and enum constraints from CONFIG SCHEMA)',
      '  "current_value": the current value of this path from CURRENT CONFIG (or null if not set)',
      '  "confidence": number 0..1',
      '  "review_required": boolean',
      '  "reason": string (brief justification, mention why the change from current value is beneficial)',
      '  "risk": "low" | "medium" | "high"',
      '  "evidence": string[] (specific measured/configured facts used; do not include assumptions as evidence)',
      "",
      "STRICT RULES:",
      "- If CONFIG SCHEMA is empty, return no recommendations and explain this in warnings.",
      "- The path field MUST be an exact match from the CONFIG SCHEMA below. Do NOT invent paths.",
      "- The value MUST match the type constraint: boolean for boolean, number for number/integer, string for string.",
      "- If the schema lists enum values, the value MUST be one of those enum values.",
      "- If the schema lists min/max constraints, the value MUST be within that range. Never recommend a value outside [min, max].",
      "- If PI CONTEXT V2 / PARAMETER CATALOG is provided, treat it as the authoritative source for defaults, schema bounds, recommended values, disabled sentinels, semantic meaning, diagnostic-only status and hard rules.",
      "- Never claim a schema maximum, schema default, schema recommendation or recommended value unless that exact field is present and non-null in PARAMETER CATALOG or CONFIG SCHEMA.",
      "- If a metadata field is null or absent, say it is unknown; do not infer it from general astrophotography knowledge.",
      "- Current values equal to cpp_default or schema_default are not misconfigurations.",
      "- Do not recommend a threshold stricter than an observed successful phase metric. Example: do not set pcc.max_residual_rms below pcc.residual_rms when pcc.status is ok.",
      "- Diagnostic-only parameters cannot be claimed to improve reconstruction quality.",
      "- Evidence should cite fact IDs from PI CONTEXT V2 when available, not free-form invented metric names.",
      "- Do NOT recommend paths of type 'object' or 'array'.",
      "- Do NOT recommend file/directory paths (e.g. calibration.darks_dir, calibration.flat_master).",
      "- Do NOT recommend aqmh.cherry_pick.* paths. Cherry-pick is excluded from AI recommendations because it has produced unreliable quality decisions.",
      ...(isAqmhMethod
        ? ["- Do NOT recommend classic_tile_compile-only paths for method=aqmh: global_metrics.*, local_metrics.*, synthetic.*. AQMH reconstruction uses aqmh.global_quality.* and per-pixel quality maps; clustering/synthetic-frame generation and classic local/tile metrics are skipped."]
        : ["- global_metrics.weights.* and global_metrics.weight_exponent_scale are ALLOWED but RESTRICTED: set review_required=true, confidence <= 0.6, and include a warning explaining the downstream PCC/color-calibration impact. Prefer balanced weights (background, noise, gradient) over sharpness-dominated weights (fwhm, roundness). Never set fwhm weight > 0.2. Always recommend ALL weights in GROUP A together (must sum to 1.0)."]),
      "- Do NOT recommend aqmh.storage.dtype or aqmh.storage.max_resident_maps. Storage/cache settings are performance/I/O concerns, not image-quality recommendations.",
      "- Only recommend changes where the new value differs from the current value, or where the current value is missing/default.",
      "- Use IMAGE QUALITY METRICS as the primary evidence for strategy decisions; use FITS header metadata only as secondary context.",
      "- Prefer a complete coherent configuration strategy over isolated tweaks: include every relevant config change needed for the detected dataset, but do not include unchanged values.",
      "- Treat scan_result fields and scan_metrics values as measured facts.",
      "- Treat CURRENT CONFIG values as configured facts, not measured facts.",
      "- Treat target-specific astrophotography knowledge as assumptions unless directly measured in scan_metrics or present in FITS/config data.",
      "- Do not use CRITICAL warnings for assumptions or generic astrophotography advice.",
      "- If scenario_profile metadata conflicts with scan metadata, describe it as a configured metadata mismatch only; do not claim the processing parameters are wrong unless specific current config values support that claim.",
      "- For every warning, explicitly state whether it is based on measured data, configured data, or an assumption.",
      "- Do not assert saturation, clipping, calibration defects, vignetting, hot pixels, or temperature impact as facts unless measured or configured evidence is present.",
      "- If evidence is insufficient for a parameter, set review_required=true and lower confidence.",
      "- Do NOT invent measurements not present in the scan data.",
      "- Every recommendation must cite at least one evidence[] item from IMAGE QUALITY METRICS, FRAME STATISTICS, SCAN RESULT, CURRENT CONFIG, or SESSION CONTEXT.",
      "- Use assumptions only in reason/warnings, never as evidence. Recommendations based primarily on assumptions must use review_required=true and confidence <= 0.55.",
      "- If SESSION CONTEXT does not include mount_type: use registration.star_shift_radius_px=200 as safe default and set review_required=true for all registration parameters.",
      "- If SESSION CONTEXT does not include target_angular_size: do NOT recommend normalization.mode=background or bge.enabled=true — omit those paths or set review_required=true.",
      "",
      "AQMH HARD RULES (mandatory, override intuitive frame-retention reasoning):",
      "- aqmh.cherry_pick.enabled, aqmh.cherry_pick.k_frac, and all other aqmh.cherry_pick.* paths are not allowed recommendation targets. Leave existing cherry-pick settings unchanged.",
      "- aqmh.cherry_pick.k_frac is the fraction of frames retained per pixel/tile, not the fraction of bad frames removed.",
      "- For frame_count > 300, NEVER recommend aqmh.cherry_pick.k_frac > 0.5. Arguments such as 'discarding too many usable frames' are explicitly invalid; 0.3-0.5 still retains enough frames.",
      "- For frame_count > 300, treat aqmh.cherry_pick.k_frac > 0.4 as aggressive. Recommend it only when measured evidence shows stable registration, no transient line artifacts, and a large clean quality spread.",
      "- For frame_count 100-300, NEVER recommend aqmh.cherry_pick.k_frac > 0.7.",
      "- If you recommend aqmh.cherry_pick.enabled=true and the effective aqmh.storage.resolution_divisor is not already 1, you MUST recommend aqmh.storage.resolution_divisor=1 in the same response.",
      "- NEVER recommend aqmh.cherry_pick.enabled=true together with aqmh.storage.resolution_divisor=2 or 4. That pair is internally contradictory.",
      "- AQMH cherry-pick is a local quality selector, not a substitute for satellite/airplane trail rejection, cosmetic masking, sigma clipping, or rejecting badly registered frames.",
      "- NEVER justify aqmh.cherry_pick.enabled or increasing aqmh.cherry_pick.k_frac as a way to remove airplane trails, satellite trails, line artifacts, hot pixels, or bad registration unless explicit artifact-mask or registration diagnostics are present in the provided data.",
      "- If SESSION CONTEXT, warnings, or measured diagnostics indicate visible trails, suspected transient line artifacts, registration failures, or many rescue/interpolated registrations, keep aqmh.cherry_pick.k_frac at or below the current value and set review_required=true for any AQMH/cherry-pick recommendation.",
      "- If transient artifacts or registration quality are unknown because no detector metrics are provided, do not claim AQMH will solve them. Mention the missing evidence in warnings when recommending cherry-pick.",
      "- aqmh.storage.dtype affects cache size and I/O only. Do not present dtype changes as an image-quality improvement, and do not couple dtype changes to cherry-pick quality claims.",
      "",
      "RECOMMENDATION STRATEGY:",
      "- First classify the dataset: OSC/mono, frame count regime, exposure/gain consistency, target scale, mount/tracking context, calibration availability, and measured quality spread.",
      "- Then recommend coherent parameter groups only when the provided evidence supports them.",
      "- Prefer fewer high-confidence changes over many weak tweaks. Do not tune cosmetic or path parameters.",
      "- For registration parameters, require mount/shift evidence. Without it, only use conservative defaults and mark review_required.",
      "- If registration diagnostics are absent, do not infer registration stability from FWHM, star_count, background, noise, or gradient metrics alone.",
      "- For normalization/background extraction, require target_angular_size and measured background/gradient evidence.",
      "- For rejection/local quality weighting, require measured FWHM/noise/background/roundness/star_count spread.",
      "",
      "BGE (BACKGROUND GRADIENT EXTRACTION) RULES:",
      "- sky_gradient is the BGE-relevant metric: it measures large-scale background variation (quadrant median diff / overall median). gradient_energy measures local pixel-scale structure (Sobel) and is NOT a reliable proxy for sky gradient strength. Always use sky_gradient (not gradient_energy) to decide whether BGE is needed.",
      "- If sky_gradient >= 0.05 (median across frames): recommend bge.enabled=true and bge.method=classic with confidence >= 0.7. The gradient is strong enough to benefit from correction.",
      "- If sky_gradient is 0.02–0.05: recommend bge.enabled=true with review_required=true and confidence <= 0.65. The gradient is moderate; BGE may help but could also introduce artifacts.",
      "- If sky_gradient < 0.02: only recommend BGE with review_required=true and confidence <= 0.6 — the correction may be negligible.",
      "- Prefer bge.method=classic over bge.method=autobge. AutoBGE uses a different internal workflow (downsampling, stretch, patch-based sampling) that can trigger flatness_worsened guard rejections, especially for weak gradients (<5% variation). Classic BGE with autotune is more robust.",
      "- Prefer bge.fit.method=poly over rbf. Run analysis shows autotune consistently selects poly (polynomial) over rbf for weak-to-moderate gradients. RBF tends to overfit diffuse nebulosity.",
      "- For nebulosity targets (IC434, Flame, Horsehead, etc.): use bge.autotune.strategy=extended to sweep additional estimators (sigma_clipped_median, biweight) and higher quantiles.",
      "- For weak gradients (sky_gradient < 0.05): set bge.autotune.alpha_flatness=0.40 (more weight on flatness) and bge.autotune.beta_roughness=0.08 (less roughness penalty).",
      "- For strong gradients (sky_gradient >= 0.05): keep bge.autotune.alpha_flatness=0.25 (default) and bge.autotune.beta_roughness=0.10 (default).",
      "- Use bge.grid.insufficient_cell_strategy=radius_expand instead of discard to avoid losing grid cells at image borders.",
      "- Use bge.sample_quantile=0.15 for conservative background sampling in nebula fields.",
      "- Use bge.mask.star_dilate_px=6 and bge.mask.sat_dilate_px=6 for dense star fields.",
      "",
      "PIPELINE CAUSALITY (critical — your recommendations have downstream effects):",
      ...(isAqmhMethod
        ? [
            "- For method=aqmh, do not use global_metrics.weights.* as a color/PCC or reconstruction recommendation. AQMH uses aqmh.global_quality.* and quality maps for reconstruction weights.",
          ]
        : [
            "- global_metrics.weights.* → determines per-frame stacking influence → affects PCC (Photometric Color Calibration) star-color measurement → affects final color balance.",
            "  UNEVEN WEIGHTS (max/median ratio > 3) can cause color cast because PCC measures star colors from a weighted-averaged stack. A few heavily-weighted frames skew the color balance, and PCC cannot correct it because it measures the already-skewed stack.",
            "  If you recommend weight changes, ensure the resulting distribution stays balanced (max/median < 3). Avoid concentrating weight on a single metric like fwhm — this over-weights sharp frames regardless of their overall quality.",
          ]),
      "- registration.* → determines frame alignment quality → affects overlap area → affects stacking coverage and PCC star detection.",
      "  Poor registration reduces the number of stars PCC can use, degrading color calibration accuracy.",
      "- normalization.* → determines per-frame background/signal scaling → directly affects PCC color matrix.",
      "  Incorrect normalization can introduce color casts that PCC may not fully correct.",
      "- RECOMMENDATIONS THAT CHANGE METHOD-APPLICABLE WEIGHTS, NORMALIZATION, OR REGISTRATION CAN CAUSE COLOR CAST IN THE FINAL IMAGE. Always consider the downstream impact.",
      "",
      "SENSOR COLOR CONTEXT (critical for OSC sensors):",
      "- OSC (One-Shot Color) sensors with a Bayer matrix have 2x green pixels vs 1x red and 1x blue. This makes raw images green-dominant.",
      "- The bayer_pattern field tells you the exact arrangement (RGGB, BGGR, GBRG, GRBG). All patterns have 2 green pixels.",
      "- PCC must correct this green dominance. A correct PCC matrix for an OSC sensor typically shows B>1.2 and R>1.05 (boosting red and blue to compensate for green dominance).",
      "- If the PCC matrix is near-identity (all values ≈1.0), color correction is INSUFFICIENT and the final image will have a green cast.",
      "- A green cast in the final image is almost always caused by either: (1) uneven frame weights skewing the stack color balance, (2) insufficient PCC correction due to poor star detection from bad registration, or (3) normalization not properly equalizing per-channel backgrounds.",
      ...(isAqmhMethod
        ? ["- For method=aqmh, color-bias reasoning must refer to aqmh.global_quality.*, PCC, registration, or normalization evidence; do not recommend global_metrics.*."]
        : ["- Do NOT recommend weight configurations that could amplify the sensor's natural green bias. Prefer balanced weights (background, noise, gradient) over sharpness-dominated weights (fwhm, roundness)."]),
      "",
      "NUMERIC PRECISION RULES (mandatory):",
      "- All recommended numeric values must be EXACT and precise — never approximate, never 'around X', never rounded to single decimal unless the schema minimum step is 0.1.",
      "- Weight groups: the config contains several groups of weights that MUST each sum to exactly 1.0. If you recommend any weight within a group, you MUST recommend ALL weights in that group so they sum to exactly 1.0.",
      "  Known weight groups (always recommend all paths in the same group together):",
      ...(isAqmhMethod ? [] : ["  GROUP A: global_metrics.weights.background + global_metrics.weights.gradient + global_metrics.weights.noise + global_metrics.weights.fwhm + global_metrics.weights.roundness + global_metrics.weights.star_count = 1.0"]),
      "  GROUP B: local_metrics.star_mode.weights.fwhm + local_metrics.star_mode.weights.roundness + local_metrics.star_mode.weights.contrast = 1.0",
      "- Never recommend a single weight from a group without recommending all others in the same group.",
      "- Double-check that all weights in a group sum to exactly 1.0 before including them.",
      "",
      ...((() => {
        if (!request.ai_request) return [];
        return [
          "=== AI REQUEST V2 SUMMARY (large duplicate scan/config payload omitted) ===",
          "Prefer this structured container for task intent, context signature, memory evidence and conversation state. Detailed scan/config facts are in the validation sections below.",
          boundedJson(compactAiRequestForPrompt(aiRequest), MAX_AI_REQUEST_JSON_CHARS),
          "",
        ];
      })()),
      ...((() => {
        if (!piContext || Object.keys(piContext).length === 0) return [];
        return [
          "=== PI CONTEXT V2 (authoritative facts, parameter catalog and evidence rules) ===",
          "Use this context before any generic knowledge. Every schema/default/recommended claim must be traceable to it.",
          boundedJson(piContext, MAX_AI_REQUEST_JSON_CHARS),
          "",
        ];
      })()),
      ...(catalogLines.length > 0 ? [
        "=== PARAMETER CATALOG (authoritative semantic metadata) ===",
        ...catalogLines,
        "",
      ] : []),
      "=== CONFIG SCHEMA (path  type  [enum]  [description]) ===",
      ...schemaLines,
      "",
      ...(configLines.length > 0 ? [
        "=== CURRENT CONFIG (path = value) ===",
        ...configLines,
        "",
      ] : []),
      ...(metricsLines.length > 0 ? [
        "=== IMAGE QUALITY METRICS (measured from actual frames) ===",
        ...metricsLines,
        "",
      ] : []),
      ...(frameStats.length > 0 ? [
        "=== FRAME STATISTICS (from FITS headers) ===",
        ...frameStats,
        "",
      ] : []),
      ...((() => {
        const ctx = request.session_context || sessionContextFromAiRequest || {};
        const lines: string[] = [];
        if (ctx.mount_type) lines.push(`mount_type: ${ctx.mount_type}  (eq=equatorial tracker, altaz=alt/az unguided)`);
        if (ctx.target_name) lines.push(`target_name: ${ctx.target_name}`);
        if (ctx.target_angular_size) lines.push(`target_angular_size: ${ctx.target_angular_size}  (compact=<5% frame, extended=>10% frame, full_frame=fills frame)`);
        if (ctx.camera_type) lines.push(`camera_type: ${ctx.camera_type}`);
        if (ctx.calibration_darks !== undefined) lines.push(`calibration_darks_available: ${ctx.calibration_darks}`);
        if (ctx.calibration_flats !== undefined) lines.push(`calibration_flats_available: ${ctx.calibration_flats}`);
        if (ctx.calibration_bias !== undefined) lines.push(`calibration_bias_available: ${ctx.calibration_bias}`);
        if (ctx.system_ram_mb) lines.push(`system_ram_mb: ${ctx.system_ram_mb}`);
        if (ctx.cpu_cores) lines.push(`cpu_cores: ${ctx.cpu_cores}`);
        if ((scan as any).registration_success_rate != null) lines.push(`registration_success_rate: ${(scan as any).registration_success_rate}`);
        if ((scan as any).max_shift_px != null) lines.push(`max_registration_shift_px: ${(scan as any).max_shift_px}`);
        // Session geometry from scan_metrics (field rotation estimate)
        const sg = scanMetrics?.session_geometry;
        if (sg) {
          if (sg.target_ra_deg != null) lines.push(`target_ra_deg: ${sg.target_ra_deg}`);
          if (sg.target_dec_deg != null) lines.push(`target_dec_deg: ${sg.target_dec_deg}`);
          if (sg.session_duration_hours != null) lines.push(`session_duration_hours: ${sg.session_duration_hours}`);
          if (sg.estimated_max_field_rotation_deg != null) lines.push(`estimated_max_field_rotation_deg: ${sg.estimated_max_field_rotation_deg}  (max theoretical rotation for Alt/Az; negligible for equatorial)`);
          if (sg.first_date_obs) lines.push(`first_date_obs: ${sg.first_date_obs}`);
          if (sg.last_date_obs) lines.push(`last_date_obs: ${sg.last_date_obs}`);
        }
        if (ctx.notes) lines.push(`notes: ${ctx.notes}`);
        return lines.length > 0 ? ["=== SESSION CONTEXT (mount, target, system, geometry) ===", ...lines, ""] : [];
      })()),
      ...((() => {
        const memories = Array.isArray(aiRequest.positive_memories)
          ? aiRequest.positive_memories
          : Array.isArray(request.session_context?.accepted_pi_memories)
          ? request.session_context.accepted_pi_memories
          : [];
        if (memories.length === 0) return [];
        const lines = [
          "Use these as reviewed historical experience only. Do not copy values blindly; every recommendation still must match CONFIG SCHEMA and current evidence.",
          JSON.stringify(memories.slice(0, 8), null, 2),
        ];
        return ["=== ACCEPTED PI MEMORIES (reviewed historical optimizations) ===", ...lines, ""];
      })()),
      ...((() => {
        const memories = Array.isArray(aiRequest.negative_memories)
          ? aiRequest.negative_memories
          : Array.isArray(request.session_context?.negative_pi_memories)
          ? request.session_context.negative_pi_memories
          : [];
        if (memories.length === 0) return [];
        const lines = [
          "Avoid repeating these reviewed rejected or deprecated optimizations unless current evidence is materially different and review_required=true.",
          JSON.stringify(memories.slice(0, 8), null, 2),
        ];
        return ["=== NEGATIVE PI MEMORIES (reviewed rejected/deprecated optimizations) ===", ...lines, ""];
      })()),
      "=== SCAN RESULT ===",
      JSON.stringify(scanCompact, null, 2),
    ];
    const prompt = sections.join("\n");
    if (prompt.length <= MAX_PROMPT_CHARS) return prompt;
    appendTrafficLog(`prompt_budget_truncate original_chars=${prompt.length} max_chars=${MAX_PROMPT_CHARS}`);
    return [
      prompt.slice(0, MAX_PROMPT_CHARS),
      "",
      `... [PROMPT TRUNCATED from ${prompt.length} to ${MAX_PROMPT_CHARS} chars by tile_compile budget guard]`,
      "Return the required JSON object using only the complete facts visible above. Do not invent missing measurements.",
    ].join("\n");
  }

  private parseResponse(responseText: string, allowedPaths: Set<string> = new Set()): ScanAnalysisResponse {
    console.log(`[AI Analysis] Raw response (${responseText.length} chars):`, responseText.substring(0, 500));
    appendTrafficLog(`parse_raw_preview ${responseText.substring(0, 2000)}`);

    const match = responseText.match(/\{[\s\S]*\}/);
    if (!match) {
      const preview = responseText.substring(0, 300).replace(/\n/g, "\\n");
      console.error("[AI Analysis] No JSON object found in response. Preview:", preview);
      throw new Error(
        `No JSON object found in PI agent response (${responseText.length} chars received).` +
        ` Response starts with: "${preview.substring(0, 120)}..."` +
        ` — the model may not have produced structured JSON output.`
      );
    }

    let parsed: any;
    try {
      parsed = JSON.parse(match[0]);
    } catch (e) {
      console.error("[AI Analysis] JSON parse failed:", e);
      throw new Error(`Invalid JSON in PI agent response: ${e instanceof Error ? e.message : String(e)}`);
    }

    console.log("[AI Analysis] Parsed schema_version:", parsed?.schema_version);
    appendTrafficLog(`parsed_schema_version ${String(parsed?.schema_version)}`);

    const schemaVersion = parsed?.schema_version || parsed?.schema;
    if (schemaVersion !== "pi.scan-analysis.v1") {
      throw new Error(`Invalid schema_version: expected "pi.scan-analysis.v1", got "${schemaVersion}"`);
    }

    const warnings: string[] = Array.isArray(parsed.warnings)
      ? parsed.warnings.map(String)
      : Array.isArray(parsed.flags)
        ? parsed.flags.map((flag: any) => String(flag?.message || flag?.id || JSON.stringify(flag)))
        : [];
    const summary = typeof parsed.summary === "string"
      ? parsed.summary
      : parsed.summary
        ? JSON.stringify(parsed.summary)
        : "";
    const confidence = Number(parsed.confidence ?? parsed.overall_confidence ?? 0);

    // Validate recommendation paths against allowed schema paths
    const rawRecs = Array.isArray(parsed.recommendations) ? parsed.recommendations : [];
    const validRecs: unknown[] = [];
    const hasAllowedPaths = allowedPaths.size > 0;

    for (const rec of rawRecs) {
      if (!rec || typeof rec !== "object") continue;
      const item = rec as Record<string, unknown>;
      const path = typeof item.path === "string" ? item.path : "";
      if (!path) {
        warnings.push(`Recommendation without path skipped: ${JSON.stringify(rec).substring(0, 100)}`);
        continue;
      }
      if (hasAllowedPaths && !allowedPaths.has(path)) {
        warnings.push(`AI recommended unknown path "${path}" (not in config schema) — skipped`);
        appendTrafficLog(`REJECTED recommendation: unknown path "${path}"`);
        continue;
      }
      const reason = typeof item.reason === "string"
        ? item.reason
        : typeof item.rationale === "string"
          ? item.rationale
          : "";
      const risk = ["low", "medium", "high"].includes(String(item.risk))
        ? String(item.risk)
        : "unknown";
      const evidence = Array.isArray(item.evidence)
        ? item.evidence.map((entry) => String(entry)).filter(Boolean)
        : [];
      validRecs.push({
        id: typeof item.id === "string" ? item.id : `rec_${path.replace(/[^A-Za-z0-9]+/g, "_")}`,
        path,
        value: item.value,
        current_value: Object.prototype.hasOwnProperty.call(item, "current_value") ? item.current_value : null,
        confidence: Number.isFinite(Number(item.confidence)) ? Number(item.confidence) : 0,
        review_required: Boolean(item.review_required),
        reason,
        rationale: reason,
        risk,
        evidence,
      });
    }

    if (hasAllowedPaths) {
      appendTrafficLog(`path_validation: ${validRecs.length} valid, ${rawRecs.length - validRecs.length} rejected out of ${rawRecs.length} total`);
    }

    const reviewRequired = Boolean(
      parsed.review_required ||
      parsed?.review_summary?.review_required_count > 0
    );

    return {
      schema_version: "pi.scan-analysis.v1",
      summary,
      confidence,
      detected_scenarios: Array.isArray(parsed.detected_scenarios) ? parsed.detected_scenarios : [],
      recommendations: validRecs,
      warnings,
      review_required: reviewRequired,
    };
  }
}
