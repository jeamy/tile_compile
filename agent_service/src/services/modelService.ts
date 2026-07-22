import {
  createAgentSession,
  DefaultResourceLoader,
  getAgentDir,
  ModelRuntime,
  SessionManager,
} from "@earendil-works/pi-coding-agent";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { redactedEnvSources } from "../config.js";
import { loadSidecarExtensionProviderRegistrations, sidecarExtensionPaths } from "./piExtensions.js";
import { modelCapabilitySummary } from "./providerPayload.js";
import { appendTrafficLog } from "./trafficLog.js";

const PI_PACKAGE_NAME = "@earendil-works/pi-coding-agent";

const PROVIDER_ACCOUNT_META: Record<string, { billingUrl: string; notes: string }> = {
  anthropic: {
    billingUrl: "https://console.anthropic.com/settings/billing",
    notes: "Anthropic exposes billing and plan details in the Console; this sidecar does not query billing endpoints.",
  },
  openai: {
    billingUrl: "https://platform.openai.com/settings/organization/billing/overview",
    notes: "OpenAI billing and usage are account/organization scoped; this sidecar does not query billing endpoints.",
  },
  google: {
    billingUrl: "https://aistudio.google.com/app/apikey",
    notes: "Google/Gemini quota and billing depend on AI Studio or Cloud project settings.",
  },
  "azure-openai-responses": {
    billingUrl: "https://portal.azure.com/#view/Microsoft_Azure_Billing/ModernBillingMenuBlade/~/Overview",
    notes: "Azure OpenAI costs and quota are managed through Azure subscription/resource settings.",
  },
  "amazon-bedrock": {
    billingUrl: "https://console.aws.amazon.com/billing/home",
    notes: "Bedrock usage and spend are AWS account scoped.",
  },
  openrouter: {
    billingUrl: "https://openrouter.ai/settings/credits",
    notes: "OpenRouter credit balance is available in the OpenRouter dashboard.",
  },
  "vercel-ai-gateway": {
    billingUrl: "https://vercel.com/dashboard",
    notes: "Vercel AI Gateway billing is managed in the Vercel dashboard.",
  },
  xai: {
    billingUrl: "https://console.x.ai/",
    notes: "xAI account and billing details are managed in the xAI console.",
  },
  mistral: {
    billingUrl: "https://console.mistral.ai/billing/",
    notes: "Mistral billing details are managed in the Mistral console.",
  },
  groq: {
    billingUrl: "https://console.groq.com/settings/billing",
    notes: "Groq billing and limits are managed in the Groq console.",
  },
  deepseek: {
    billingUrl: "https://platform.deepseek.com/usage",
    notes: "DeepSeek usage and balance are managed in the DeepSeek platform dashboard.",
  },
  together: {
    billingUrl: "https://api.together.ai/settings/billing",
    notes: "Together billing and usage are managed in the Together dashboard.",
  },
  fireworks: {
    billingUrl: "https://fireworks.ai/account/billing",
    notes: "Fireworks billing is managed in the Fireworks account dashboard.",
  },
  huggingface: {
    billingUrl: "https://huggingface.co/settings/billing",
    notes: "Hugging Face billing and quotas are managed in account settings.",
  },
};

function defaultAccountMeta(provider: string) {
  return {
    billingUrl: "",
    notes: provider
      ? "This provider has no built-in billing query in tile_compile; check the provider console."
      : "Select a provider to see API-key and account metadata.",
  };
}

export class ModelService {
  private modelRuntimePromise = this.createModelRuntime();
  private storedAuthProviders = new Set<string>();
  private capabilityCachePath: string;
  private piVersionCache: { checkedAt: number; payload: any } | null = null;

  constructor(projectRoot = process.env.TILE_COMPILE_PROJECT_ROOT || path.resolve(process.cwd(), "..")) {
    this.capabilityCachePath = path.join(path.resolve(projectRoot), ".tile_compile", "pi_model_capabilities.json");
  }

  getModelRuntime() {
    return this.modelRuntimePromise;
  }

  markStoredAuthProvider(provider: string) {
    const normalized = String(provider || "").trim();
    if (normalized) this.storedAuthProviders.add(normalized);
  }

  unmarkStoredAuthProvider(provider: string) {
    const normalized = String(provider || "").trim();
    if (normalized) this.storedAuthProviders.delete(normalized);
  }

  async modelsJson() {
    const envSources = redactedEnvSources();
    let available: any[] = [];
    try {
      const modelRuntime = await this.getModelRuntime();
      available = Array.from(await modelRuntime.getAvailable());
    } catch {
      available = [];
    }

    const providerMap = new Map<string, any[]>();
    for (const model of available) {
      const provider = String(model.provider || "");
      if (!provider) continue;
      if (!providerMap.has(provider)) providerMap.set(provider, []);
      providerMap.get(provider)!.push({
        id: model.id,
        label: model.name || model.label || model.id,
        available: true,
        capabilities: this.capabilityJson(`${provider}/${model.id}`, model),
        auth_source: "auth_storage",
      });
    }

    for (const [provider, source] of Object.entries(envSources)) {
      if (!source) continue;
      if (!providerMap.has(provider)) providerMap.set(provider, []);
    }

    return {
      pi: await this.piVersionJson(),
      providers: Array.from(providerMap.entries()).map(([provider, models]) => ({
        provider,
        auth_source: envSources[provider] || (models.length > 0 ? "auth_storage" : ""),
        models,
      })),
    };
  }

  async accountJson(provider = "") {
    const normalizedProvider = String(provider || "").trim();
    const envSources = redactedEnvSources();
    const models = await this.modelsJson();
    const knownProviders = new Set(Object.keys(envSources));
    for (const providerName of this.storedAuthProviders) knownProviders.add(providerName);
    for (const item of Array.isArray(models.providers) ? models.providers : []) {
      if (item?.provider) knownProviders.add(String(item.provider));
    }

    const buildStatus = (name: string) => {
      const modelProvider = Array.isArray(models.providers)
        ? models.providers.find((item: any) => String(item?.provider || "") === name)
        : null;
      const authSource = envSources[name] || (this.storedAuthProviders.has(name) ? "auth_storage" : "") || modelProvider?.auth_source || "";
      const meta = PROVIDER_ACCOUNT_META[name] || defaultAccountMeta(name);
      return {
        provider: name,
        key_configured: Boolean(authSource),
        auth_source: authSource,
        credit_query_supported: false,
        subscription_query_supported: false,
        credit_status: "not_supported",
        subscription_status: "not_supported",
        billing_url: meta.billingUrl,
        message: meta.notes,
      };
    };

    const providers = Array.from(knownProviders).sort().map(buildStatus);
    const selected = normalizedProvider ? buildStatus(normalizedProvider) : null;
    return {
      schema_version: "pi.account-status.v1",
      privacy_class: "metadata_only",
      provider: normalizedProvider,
      selected,
      providers,
    };
  }

  async findModel(modelRef: string) {
    const [provider, ...modelParts] = String(modelRef || "").split("/");
    const modelId = modelParts.join("/");
    if (!provider || !modelId) return null;
    const modelRuntime = await this.getModelRuntime();
    const available = await modelRuntime.getAvailable(provider);
    return available.find((model: any) => String(model.id || "") === modelId) || null;
  }

  capabilityJson(modelRef: string, model: any = null) {
    const found = model;
    const cached = this.readCapabilities().models?.[modelRef] || {};
    const registrySupportsImages = Array.isArray(found?.input) && found.input.includes("image");
    const heuristicSupportsImages = inferVisionSupport(modelRef);
    const override = typeof cached.vision_override === "boolean" ? cached.vision_override : null;
    const live = cached.vision_live && typeof cached.vision_live === "object" ? cached.vision_live : null;
    const liveStatus = live && isProbePayloadError(live.error) ? "error" : live?.status;
    const supportsImages = override !== null
      ? override
      : liveStatus === "tested"
        ? Boolean(live.supports_images)
        : registrySupportsImages || heuristicSupportsImages;
    const source = override !== null
      ? "override"
      : liveStatus === "tested"
        ? "live_probe"
        : registrySupportsImages
          ? "registry"
          : heuristicSupportsImages
            ? "heuristic"
            : "unknown";
    return {
      schema_version: "pi.model-capability.v1",
      model: modelRef,
      ...modelCapabilitySummary(found),
      supports_images: supportsImages,
      source,
      registry_supports_images: registrySupportsImages,
      heuristic_supports_images: heuristicSupportsImages,
      override,
      live,
    };
  }

  async testModel(modelRef: string, opts: { visionProbe?: boolean; visionOverride?: boolean | null } = {}) {
    const model = await this.findModel(modelRef);
    if (opts.visionOverride !== undefined) {
      this.setVisionOverride(modelRef, opts.visionOverride);
    }
    let capabilities = this.capabilityJson(modelRef, model);
    if (model && opts.visionProbe) {
      const live = await this.probeVision(modelRef);
      this.updateCapability(modelRef, { vision_live: live });
      capabilities = this.capabilityJson(modelRef, model);
    }
    return {
      ok: Boolean(model),
      model: modelRef,
      error: model ? undefined : "model_not_found",
      capabilities,
    };
  }

  setVisionOverride(modelRef: string, value: boolean | null) {
    const normalized = String(modelRef || "").trim();
    if (!normalized) return;
    const cache = this.readCapabilities();
    cache.models = cache.models || {};
    cache.models[normalized] = cache.models[normalized] || {};
    if (value === null) delete cache.models[normalized].vision_override;
    else cache.models[normalized].vision_override = Boolean(value);
    cache.models[normalized].updated_at = new Date().toISOString();
    this.writeCapabilities(cache);
  }

  private async probeVision(modelRef: string) {
    const startedAt = new Date().toISOString();
    const model = await this.findModel(modelRef);
    if (!model) {
      return { status: "error", supports_images: false, tested_at: startedAt, error: "model_not_found" };
    }
    const agentCwd = process.env.TILE_COMPILE_PROJECT_ROOT
      ? path.resolve(process.env.TILE_COMPILE_PROJECT_ROOT)
      : process.cwd();
    const tinyPng = "iVBORw0KGgoAAAANSUhEUgAAAAQAAAAECAIAAAAmkwkpAAAAEElEQVR4nGP4z8AARwzEcQCukw/x0F8jngAAAABJRU5ErkJggg==";
    let assistantError = "";
    try {
      appendTrafficLog(`vision_probe start model=${modelRef}`);
      const resourceLoader = new DefaultResourceLoader({
        cwd: agentCwd,
        agentDir: getAgentDir(),
        additionalExtensionPaths: sidecarExtensionPaths(),
        extensionFactories: [],
      });
      await resourceLoader.reload();
      const { session } = await createAgentSession({
        cwd: agentCwd,
        model,
        modelRuntime: await this.getModelRuntime(),
        sessionManager: SessionManager.inMemory(),
        resourceLoader,
        tools: [],
      });
      const unsubscribe = session.subscribe((event: any) => {
        if (event.type === "message_end" && event.message?.role === "assistant" && event.message?.stopReason === "error") {
          assistantError = String(event.message?.errorMessage || "");
        }
      });
      try {
        await Promise.race([
          session.prompt("Vision capability probe. Reply with exactly: vision-ok", {
            images: [{ type: "image", data: tinyPng, mimeType: "image/png" }],
          }),
          new Promise((_, reject) => setTimeout(() => reject(new Error("vision_probe_timeout")), 60000)),
        ]);
      } finally {
        unsubscribe();
        session.dispose();
      }
      if (assistantError) throw new Error(assistantError);
      appendTrafficLog(`vision_probe result model=${modelRef} supports_images=true`);
      return { status: "tested", supports_images: true, tested_at: new Date().toISOString() };
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      const unsupported = isUnsupportedImageInputError(message);
      appendTrafficLog(`vision_probe result model=${modelRef} supports_images=${unsupported ? "false" : "unknown"} error=${message.substring(0, 500)}`);
      return {
        status: unsupported ? "tested" : "error",
        supports_images: unsupported ? false : null,
        tested_at: new Date().toISOString(),
        error: message,
      };
    }
  }

  private updateCapability(modelRef: string, patch: Record<string, unknown>) {
    const cache = this.readCapabilities();
    cache.models = cache.models || {};
    cache.models[modelRef] = { ...(cache.models[modelRef] || {}), ...patch, updated_at: new Date().toISOString() };
    this.writeCapabilities(cache);
  }

  private readCapabilities(): any {
    try {
      if (!fs.existsSync(this.capabilityCachePath)) return { schema_version: "pi.model-capabilities.v1", models: {} };
      const parsed = JSON.parse(fs.readFileSync(this.capabilityCachePath, "utf8"));
      if (!parsed || typeof parsed !== "object") return { schema_version: "pi.model-capabilities.v1", models: {} };
      parsed.models = parsed.models && typeof parsed.models === "object" ? parsed.models : {};
      return parsed;
    } catch {
      return { schema_version: "pi.model-capabilities.v1", models: {} };
    }
  }

  private writeCapabilities(cache: any) {
    fs.mkdirSync(path.dirname(this.capabilityCachePath), { recursive: true });
    fs.writeFileSync(this.capabilityCachePath, JSON.stringify({
      schema_version: "pi.model-capabilities.v1",
      ...cache,
      updated_at: new Date().toISOString(),
    }, null, 2));
  }

  async storeApiKey(provider: string, apiKey: string) {
    this.modifyAuthJson(provider, apiKey ? { type: "api_key", key: apiKey } : null);
    const modelRuntime = await this.reloadModelRuntime();
    if (apiKey) await modelRuntime.setRuntimeApiKey(provider, apiKey);
    this.markStoredAuthProvider(provider);
  }

  async removeApiKey(provider: string) {
    this.modifyAuthJson(provider, null);
    const modelRuntime = await this.reloadModelRuntime();
    await modelRuntime.removeRuntimeApiKey(provider);
    this.unmarkStoredAuthProvider(provider);
  }

  private async reloadModelRuntime() {
    this.modelRuntimePromise = this.createModelRuntime();
    return this.modelRuntimePromise;
  }

  private async createModelRuntime() {
    const modelRuntime = await ModelRuntime.create();
    const agentCwd = process.env.TILE_COMPILE_PROJECT_ROOT
      ? path.resolve(process.env.TILE_COMPILE_PROJECT_ROOT)
      : process.cwd();
    for (const registration of await loadSidecarExtensionProviderRegistrations(agentCwd, getAgentDir())) {
      if (!registration?.name) continue;
      modelRuntime.registerProvider(registration.name, registration.config);
    }
    return modelRuntime;
  }

  private modifyAuthJson(provider: string, credential: { type: "api_key"; key: string } | null) {
    const normalized = String(provider || "").trim();
    if (!normalized) throw new Error("provider is required");
    const authPath = path.join(getAgentDir(), "auth.json");
    fs.mkdirSync(path.dirname(authPath), { recursive: true });
    let parsed: Record<string, unknown> = {};
    try {
      if (fs.existsSync(authPath)) {
        const loaded = JSON.parse(fs.readFileSync(authPath, "utf8"));
        if (loaded && typeof loaded === "object" && !Array.isArray(loaded)) {
          parsed = loaded as Record<string, unknown>;
        }
      }
    } catch {
      parsed = {};
    }
    if (credential) parsed[normalized] = credential;
    else delete parsed[normalized];
    fs.writeFileSync(authPath, JSON.stringify(parsed, null, 2), { mode: 0o600 });
    try {
      fs.chmodSync(authPath, 0o600);
    } catch {
      // Best effort: Windows and some mounted filesystems may not support chmod.
    }
  }

  private async piVersionJson() {
    const now = Date.now();
    if (this.piVersionCache && now - this.piVersionCache.checkedAt < 10 * 60 * 1000) {
      return this.piVersionCache.payload;
    }
    const current = installedPiVersion();
    const payload: any = {
      package: PI_PACKAGE_NAME,
      current,
      latest: null,
      update_available: null,
      status: current ? "unknown" : "not_installed",
      checked_at: new Date().toISOString(),
      check_supported: true,
    };
    if (!current) {
      this.piVersionCache = { checkedAt: now, payload };
      return payload;
    }
    try {
      const latest = await fetchLatestPiVersion();
      payload.latest = latest;
      payload.update_available = compareSemver(current, latest) < 0;
      payload.status = payload.update_available ? "update_available" : "current";
    } catch (error) {
      payload.error = error instanceof Error ? error.message : String(error);
    }
    this.piVersionCache = { checkedAt: now, payload };
    return payload;
  }
}

function installedPiVersion(): string {
  try {
    let packagePath = "";
    let currentPath = path.dirname(fileURLToPath(import.meta.url));
    while (currentPath && currentPath !== path.dirname(currentPath)) {
      const candidate = path.join(currentPath, "node_modules", ...PI_PACKAGE_NAME.split("/"), "package.json");
      if (fs.existsSync(candidate)) {
        packagePath = candidate;
        break;
      }
      currentPath = path.dirname(currentPath);
    }
    if (!packagePath) return "";
    const parsed = JSON.parse(fs.readFileSync(packagePath, "utf8"));
    return parsed?.name === PI_PACKAGE_NAME && typeof parsed?.version === "string" ? parsed.version : "";
  } catch {
    return "";
  }
}

async function fetchLatestPiVersion(): Promise<string> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 2000);
  try {
    const response = await fetch(`https://registry.npmjs.org/${encodeURIComponent(PI_PACKAGE_NAME)}/latest`, {
      signal: controller.signal,
      headers: { "Accept": "application/json" },
    });
    if (!response.ok) throw new Error(`npm_registry_http_${response.status}`);
    const payload = await response.json();
    const version = typeof payload?.version === "string" ? payload.version : "";
    if (!version) throw new Error("npm_registry_missing_version");
    return version;
  } finally {
    clearTimeout(timeout);
  }
}

function compareSemver(left: string, right: string): number {
  const parse = (value: string) => String(value || "")
    .split(/[.+-]/)
    .slice(0, 3)
    .map(part => Number.parseInt(part, 10))
    .map(part => Number.isFinite(part) ? part : 0);
  const a = parse(left);
  const b = parse(right);
  for (let i = 0; i < 3; ++i) {
    if ((a[i] || 0) < (b[i] || 0)) return -1;
    if ((a[i] || 0) > (b[i] || 0)) return 1;
  }
  return 0;
}

function isProbePayloadError(message: unknown): boolean {
  const text = String(message || "");
  return /invalid image|image data.*valid image|could not process image|failed to process image|decode image|unsupported image format/i.test(text);
}

function isUnsupportedImageInputError(message: unknown): boolean {
  const text = String(message || "");
  if (isProbePayloadError(text)) return false;
  return /vision.*not supported|image.*not supported|does not support.*image|multimodal.*not supported|content.*type.*unsupported|unsupported.*image input/i.test(text);
}

function inferVisionSupport(modelRef: string): boolean {
  const ref = String(modelRef || "").toLowerCase();
  if (!ref) return false;
  if (/(embedding|moderation|tts|whisper|audio|rerank|text-only)/.test(ref)) return false;
  return [
    /gpt-4o/,
    /gpt-4\.1/,
    /gpt-5/,
    /o[34]/,
    /claude-(3|3\.5|3\.7|4|4\.5)/,
    /sonnet/,
    /opus/,
    /haiku/,
    /gemini/,
    /grok-.*vision/,
    /llava/,
    /qwen.*vl/,
    /pixtral/,
    /mistral.*vision/,
  ].some((pattern) => pattern.test(ref));
}
