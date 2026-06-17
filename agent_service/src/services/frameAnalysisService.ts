import {
  createAgentSession,
  SessionManager,
} from "@earendil-works/pi-coding-agent";
import fs from "node:fs";
import path from "node:path";
import type { AgentConfig, ScanAnalysisRequest, ScanAnalysisResponse, ProgressCallback, AnalysisProgressEvent } from "../types.js";
import type { ModelService } from "./modelService.js";

const trafficLogPath = path.resolve(process.env.TILE_COMPILE_PROJECT_ROOT || path.resolve(process.cwd(), ".."), "runs", "pi_agent_traffic.log");

function appendTrafficLog(message: string) {
  try {
    fs.mkdirSync(path.dirname(trafficLogPath), { recursive: true });
    fs.appendFileSync(trafficLogPath, `[${new Date().toISOString()}] ${message}\n`);
  } catch {
    // Ignore logging errors.
  }
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
    const model = this.modelService.findModel(modelRef);
    if (!model) throw new Error(`Model ${modelRef || "(empty)"} not found in PI registry`);

    this.emitProgress(onProgress, { phase: "initializing", message: "Creating AI session...", progress: 5 });

    const { session } = await createAgentSession({
      model,
      authStorage: this.modelService.getAuthStorage(),
      modelRegistry: this.modelService.getModelRegistry(),
      sessionManager: SessionManager.inMemory(),
      tools: [],
    });

    this.emitProgress(onProgress, { phase: "building_prompt", message: "Building analysis prompt...", progress: 10 });
    const prompt = this.buildPrompt(request);
    appendTrafficLog(`prompt_length ${prompt.length} sections=${[
      'IMAGE QUALITY METRICS','FRAME STATISTICS','CURRENT CONFIG','CONFIG SCHEMA','SCAN RESULT','scan_metrics'
    ].map(s => s + ':' + (prompt.includes(s) ? 'YES' : 'NO')).join(' ')}`);
    appendTrafficLog(`prompt ${prompt.substring(0, 50000)}`);

    let responseText = "";
    let textDeltaCount = 0;
    const startTime = Date.now();
    let lastProgressEmit = startTime;

    this.emitProgress(onProgress, { phase: "ai_thinking", message: "Waiting for AI response...", progress: 15 });

    const unsubscribe = session.subscribe((event: any) => {
      if (event.type === "message_update" && event.assistantMessageEvent?.type === "text_delta") {
        const delta = event.assistantMessageEvent.delta as string;
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
            progress: Math.min(15 + (responseText.length / 50), 85), // Rough estimate: ~5000 chars = 100%
          });
          lastProgressEmit = now;
        }
      }
    });

    try {
      await session.prompt(prompt);
    } finally {
      unsubscribe();
      session.dispose();
    }

    this.emitProgress(onProgress, { phase: "parsing_response", message: "Parsing AI response...", progress: 90 });

    const duration = Date.now() - startTime;
    console.log(`[AI Analysis] Completed: ${textDeltaCount} deltas, ${responseText.length} chars, ${duration}ms`);
    appendTrafficLog(`raw_response ${responseText.substring(0, 20000)}`);

    const allowedPaths = new Set<string>(
      Array.isArray(request.allowed_config_paths) ? request.allowed_config_paths : []
    );
    const configSchema = (request as any).config_schema || {};
    if (allowedPaths.size === 0) {
      for (const key of Object.keys(configSchema)) allowedPaths.add(key);
    }
    const result = this.parseResponse(responseText, allowedPaths);
    result._meta = {
      streaming_duration_ms: duration,
      response_chars: responseText.length,
      model: model.id,
      provider: model.provider,
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
    // Build compact schema reference: only leaf paths (non-object types)
    const configSchema = (request as any).config_schema || {};
    const schemaLines: string[] = [];
    for (const [path, info] of Object.entries<any>(configSchema)) {
      if (info?.type === "object") continue; // skip parent objects
      const parts = [path, `type:${info?.type || "unknown"}`];
      if (info?.enum) parts.push(`enum:${JSON.stringify(info.enum)}`);
      if (info?.minimum !== undefined) parts.push(`min:${info.minimum}`);
      if (info?.maximum !== undefined) parts.push(`max:${info.maximum}`);
      if (info?.desc) parts.push(info.desc);
      schemaLines.push(parts.join("  "));
    }

    // Build compact scan summary (no frame list, just counts and metadata)
    const scan = (request as any).scan_result || {};
    const frames = Array.isArray(scan.frames) ? scan.frames : [];
    const frameCount = scan.frames_detected ?? scan.frames_total ?? frames.length;
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
    // Extract per-frame metadata from first frame if available
    if (frames[0]) {
      const f = frames[0];
      if (f.exposure_seconds != null) scanCompact.exposure_seconds = f.exposure_seconds;
      if (f.gain != null) scanCompact.gain = f.gain;
      if (f.image_width != null && !scanCompact.image_width) scanCompact.image_width = f.image_width;
      if (f.image_height != null && !scanCompact.image_height) scanCompact.image_height = f.image_height;
      if (f.target != null) scanCompact.target = f.target;
      if (f.camera != null) scanCompact.camera = f.camera;
      if (f.telescope != null) scanCompact.telescope = f.telescope;
      if (f.temperature_c != null) scanCompact.temperature_c = f.temperature_c;
      if (f.fwhm != null) scanCompact.fwhm = f.fwhm;
      if (f.snr != null) scanCompact.snr = f.snr;
    }

    // Build compact current config (only leaf values, flatten dotted paths)
    const baseConfig = (request as any).base_config || {};
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
    const scanMetrics = (request as any).scan_metrics || null;
    const metricsLines: string[] = [];
    if (scanMetrics && scanMetrics.aggregate) {
      const agg = scanMetrics.aggregate;
      metricsLines.push(`Sampled ${scanMetrics.sample_count ?? "?"} of ${scanMetrics.frames_total ?? frameCount} frames:`);
      for (const key of ["fwhm", "background", "noise", "gradient_energy", "roundness", "star_count"]) {
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
      metricsLines.push("Full measured scan_metrics JSON:");
      metricsLines.push(JSON.stringify(scanMetrics, null, 2));
    }

    // Aggregate frame statistics if available (frames[] may be truncated by backend)
    const frameStats: string[] = [];
    if (frames.length > 0 && frames.length < frameCount) {
      frameStats.push(`(Note: statistics sampled from ${frames.length} of ${frameCount} total frames)`);
    }
    if (frames.length > 0) {
      const numericAgg = (key: string) => {
        const vals = frames.map((f: any) => f[key]).filter((v: any) => typeof v === "number" && isFinite(v));
        if (vals.length === 0) return null;
        const min = Math.min(...vals);
        const max = Math.max(...vals);
        const mean = vals.reduce((a: number, b: number) => a + b, 0) / vals.length;
        return { min, max, mean: Math.round(mean * 100) / 100, count: vals.length };
      };
      for (const key of ["exposure_seconds", "gain", "temperature_c", "fwhm", "snr", "sky_background"]) {
        const agg = numericAgg(key);
        if (!agg) continue;
        if (agg.min === agg.max) {
          frameStats.push(`${key}: ${agg.min} (uniform across all ${frameCount} frames)`);
        } else {
          frameStats.push(`${key}: min=${agg.min} max=${agg.max} mean=${agg.mean} (sampled ${agg.count} frames)`);
        }
      }
      // String fields: pick most common value
      for (const key of ["target", "camera", "telescope"]) {
        const vals = frames.map((f: any) => f[key]).filter((v: any) => typeof v === "string" && v);
        if (vals.length > 0) {
          const counts: Record<string, number> = {};
          for (const v of vals) counts[v] = (counts[v] || 0) + 1;
          const top = Object.entries(counts).sort((a, b) => b[1] - a[1])[0];
          if (top[1] === vals.length) {
            frameStats.push(`${key}: "${top[0]}" (all ${frameCount} frames)`);
          } else {
            frameStats.push(`${key}: "${top[0]}" (${top[1]}/${vals.length} sampled frames)`);
          }
        }
      }
    }

    return [
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
      '  "rationale": string (brief justification, mention why the change from current value is beneficial)',
      "",
      "STRICT RULES:",
      "- The path field MUST be an exact match from the CONFIG SCHEMA below. Do NOT invent paths.",
      "- The value MUST match the type constraint: boolean for boolean, number for number/integer, string for string.",
      "- If the schema lists enum values, the value MUST be one of those enum values.",
      "- If the schema lists min/max constraints, the value MUST be within that range. Never recommend a value outside [min, max].",
      "- Do NOT recommend paths of type 'object' or 'array'.",
      "- Do NOT recommend file/directory paths (e.g. calibration.darks_dir, calibration.flat_master).",
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
      "",
      "NUMERIC PRECISION RULES (mandatory):",
      "- All recommended numeric values must be EXACT and precise — never approximate, never 'around X', never rounded to single decimal unless the schema minimum step is 0.1.",
      "- Weight groups: the config contains several groups of weights that MUST each sum to exactly 1.0. If you recommend any weight within a group, you MUST recommend ALL weights in that group so they sum to exactly 1.0.",
      "  Known weight groups (always recommend all paths in the same group together):",
      "  GROUP A: global_metrics.weights.background + global_metrics.weights.gradient + global_metrics.weights.noise = 1.0",
      "  GROUP B: quality_filter.weights.contrast + quality_filter.weights.fwhm + quality_filter.weights.roundness = 1.0",
      "- Never recommend a single weight from a group without recommending all others in the same group.",
      "- Double-check that all weights in a group sum to exactly 1.0 before including them.",
      "",
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
      "=== SCAN RESULT ===",
      JSON.stringify(scanCompact, null, 2),
    ].join("\n");
  }

  private parseResponse(responseText: string, allowedPaths: Set<string> = new Set()): ScanAnalysisResponse {
    console.log(`[AI Analysis] Raw response (${responseText.length} chars):`, responseText.substring(0, 500));
    appendTrafficLog(`parse_raw_preview ${responseText.substring(0, 2000)}`);

    const match = responseText.match(/\{[\s\S]*\}/);
    if (!match) {
      console.error("[AI Analysis] No JSON object found in response");
      throw new Error("No JSON object found in PI agent response");
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
      const path = String(rec.path || "");
      if (!path) {
        warnings.push(`Recommendation without path skipped: ${JSON.stringify(rec).substring(0, 100)}`);
        continue;
      }
      if (hasAllowedPaths && !allowedPaths.has(path)) {
        warnings.push(`AI recommended unknown path "${path}" (not in config schema) — skipped`);
        appendTrafficLog(`REJECTED recommendation: unknown path "${path}"`);
        continue;
      }
      validRecs.push(rec);
    }

    if (hasAllowedPaths) {
      appendTrafficLog(`path_validation: ${validRecs.length} valid, ${rawRecs.length - validRecs.length} rejected out of ${rawRecs.length} total`);
    }

    const reviewRequired = Boolean(
      parsed.review_required ||
      parsed?.review_summary?.review_required_count > 0 ||
      warnings.length > 0
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
