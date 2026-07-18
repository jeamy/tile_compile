import {
  DefaultResourceLoader,
  type ExtensionRuntime,
} from "@earendil-works/pi-coding-agent";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);

let cachedExtensionPaths: string[] | null = null;

export function sidecarExtensionPaths(): string[] {
  if (cachedExtensionPaths) return cachedExtensionPaths;

  const paths: string[] = [];
  try {
    paths.push(require.resolve("pi-kiro-api/extension.ts"));
  } catch {
    // Optional extension: installed in the dev sidecar package, but keep startup robust.
  }

  cachedExtensionPaths = paths;
  return paths;
}

export async function loadSidecarExtensionProviderRegistrations(cwd: string, agentDir: string) {
  const paths = sidecarExtensionPaths();
  if (paths.length === 0) return [];

  const resourceLoader = new DefaultResourceLoader({
    cwd,
    agentDir,
    additionalExtensionPaths: paths,
    extensionFactories: [],
    noSkills: true,
    noPromptTemplates: true,
    noThemes: true,
    noContextFiles: true,
  });
  await resourceLoader.reload();

  const runtime = resourceLoader.getExtensions().runtime as ExtensionRuntime & {
    pendingProviderRegistrations?: Array<{ name: string; config: unknown }>;
  };
  return Array.isArray(runtime.pendingProviderRegistrations)
    ? runtime.pendingProviderRegistrations
    : [];
}
