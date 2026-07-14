import type { ModelService } from "./modelService.js";

export class AuthService {
  constructor(private readonly modelService: ModelService) {}

  async storeKey(provider: string, apiKey: string) {
    if (!provider || !apiKey) {
      throw new Error("provider and api_key are required");
    }
    const authStorage: any = this.modelService.getAuthStorage();
    if (typeof authStorage.setApiKey === "function") {
      await authStorage.setApiKey(provider, apiKey);
    } else if (typeof authStorage.set === "function") {
      await authStorage.set(provider, apiKey);
    } else {
      throw new Error("PI AuthStorage does not expose a supported key setter");
    }
    this.modelService.markStoredAuthProvider(provider);
    return { provider, stored: true };
  }

  async removeKey(provider: string) {
    if (!provider) throw new Error("provider is required");
    const authStorage: any = this.modelService.getAuthStorage();
    if (typeof authStorage.deleteApiKey === "function") {
      await authStorage.deleteApiKey(provider);
    } else if (typeof authStorage.delete === "function") {
      await authStorage.delete(provider);
    } else if (typeof authStorage.remove === "function") {
      await authStorage.remove(provider);
    } else {
      throw new Error("PI AuthStorage does not expose a supported key remover");
    }
    this.modelService.unmarkStoredAuthProvider(provider);
    return { provider, removed: true };
  }
}
