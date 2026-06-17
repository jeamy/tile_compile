import { AuthStorage, ModelRegistry } from "@earendil-works/pi-coding-agent";
import { redactedEnvSources } from "../config.js";

export class ModelService {
  private authStorage = AuthStorage.create();
  private modelRegistry = ModelRegistry.create(this.authStorage);

  getAuthStorage() {
    return this.authStorage;
  }

  getModelRegistry() {
    return this.modelRegistry;
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

  findModel(modelRef: string) {
    const [provider, ...modelParts] = String(modelRef || "").split("/");
    const modelId = modelParts.join("/");
    if (!provider || !modelId) return null;
    return this.modelRegistry.find(provider, modelId);
  }
}
