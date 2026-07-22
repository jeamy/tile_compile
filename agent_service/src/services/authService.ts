import type { ModelService } from "./modelService.js";

export class AuthService {
  constructor(private readonly modelService: ModelService) {}

  async storeKey(provider: string, apiKey: string) {
    if (!provider || !apiKey) {
      throw new Error("provider and api_key are required");
    }
    await this.modelService.storeApiKey(provider, apiKey);
    return { provider, stored: true };
  }

  async removeKey(provider: string) {
    if (!provider) throw new Error("provider is required");
    await this.modelService.removeApiKey(provider);
    return { provider, removed: true };
  }
}
