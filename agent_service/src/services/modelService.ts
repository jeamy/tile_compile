import {
  AuthStorage,
  createAgentSession,
  DefaultResourceLoader,
  getAgentDir,
  ModelRegistry,
  SessionManager,
} from "@earendil-works/pi-coding-agent";
import fs from "node:fs";
import path from "node:path";
import { redactedEnvSources } from "../config.js";
import { appendTrafficLog } from "./trafficLog.js";

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
  private authStorage = AuthStorage.create();
  private modelRegistry = ModelRegistry.create(this.authStorage);
  private storedAuthProviders = new Set<string>();
  private capabilityCachePath: string;

  constructor(projectRoot = process.env.TILE_COMPILE_PROJECT_ROOT || path.resolve(process.cwd(), "..")) {
    this.capabilityCachePath = path.join(path.resolve(projectRoot), ".tile_compile", "pi_model_capabilities.json");
  }

  getAuthStorage() {
    return this.authStorage;
  }

  getModelRegistry() {
    return this.modelRegistry;
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
      available = await this.modelRegistry.getAvailable();
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

  findModel(modelRef: string) {
    const [provider, ...modelParts] = String(modelRef || "").split("/");
    const modelId = modelParts.join("/");
    if (!provider || !modelId) return null;
    return this.modelRegistry.find(provider, modelId);
  }

  capabilityJson(modelRef: string, model: any = null) {
    const found = model || this.findModel(modelRef);
    const cached = this.readCapabilities().models?.[modelRef] || {};
    const registrySupportsImages = Array.isArray(found?.input) && found.input.includes("image");
    const heuristicSupportsImages = inferVisionSupport(modelRef);
    const override = typeof cached.vision_override === "boolean" ? cached.vision_override : null;
    const live = cached.vision_live && typeof cached.vision_live === "object" ? cached.vision_live : null;
    const supportsImages = override !== null
      ? override
      : live?.status === "tested"
        ? Boolean(live.supports_images)
        : registrySupportsImages || heuristicSupportsImages;
    const source = override !== null
      ? "override"
      : live?.status === "tested"
        ? "live_probe"
        : registrySupportsImages
          ? "registry"
          : heuristicSupportsImages
            ? "heuristic"
            : "unknown";
    return {
      schema_version: "pi.model-capability.v1",
      model: modelRef,
      supports_images: supportsImages,
      source,
      registry_supports_images: registrySupportsImages,
      heuristic_supports_images: heuristicSupportsImages,
      override,
      live,
    };
  }

  async testModel(modelRef: string, opts: { visionProbe?: boolean; visionOverride?: boolean | null } = {}) {
    const model = this.findModel(modelRef);
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
    const model = this.findModel(modelRef);
    if (!model) {
      return { status: "error", supports_images: false, tested_at: startedAt, error: "model_not_found" };
    }
    const agentCwd = process.env.TILE_COMPILE_PROJECT_ROOT
      ? path.resolve(process.env.TILE_COMPILE_PROJECT_ROOT)
      : process.cwd();
    const tinyPng = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII=";
    let assistantError = "";
    try {
      appendTrafficLog(`vision_probe start model=${modelRef}`);
      const resourceLoader = new DefaultResourceLoader({
        cwd: agentCwd,
        agentDir: getAgentDir(),
        extensionFactories: [],
      });
      await resourceLoader.reload();
      const { session } = await createAgentSession({
        cwd: agentCwd,
        model,
        authStorage: this.authStorage,
        modelRegistry: this.modelRegistry,
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
      const unsupported = /image|vision|multimodal|content.*type|unsupported|not support/i.test(message);
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
