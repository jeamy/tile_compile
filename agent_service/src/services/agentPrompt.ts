type PromptOptions = Record<string, unknown>;

export async function promptWithTimeout(
  session: { prompt: (text: string, options?: PromptOptions) => Promise<void> },
  text: string,
  timeoutMs: number,
  label: string,
  options?: PromptOptions,
) {
  const effectiveTimeoutMs = Number.isFinite(timeoutMs) && timeoutMs > 0 ? timeoutMs : 180000;
  let timeout: ReturnType<typeof setTimeout> | null = null;
  try {
    await Promise.race([
      session.prompt(text, options),
      new Promise<never>((_, reject) => {
        timeout = setTimeout(
          () => reject(new Error(`${label} timed out after ${Math.max(1, Math.ceil(effectiveTimeoutMs / 1000))}s`)),
          effectiveTimeoutMs,
        );
      }),
    ]);
  } finally {
    if (timeout) clearTimeout(timeout);
  }
}
