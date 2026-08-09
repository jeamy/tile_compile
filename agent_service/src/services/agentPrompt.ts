type PromptOptions = Record<string, unknown>;
type PromptEvent = {
  type?: unknown;
  message?: {
    role?: unknown;
    stopReason?: unknown;
    errorMessage?: unknown;
  };
};

type PromptSession = {
  prompt: (text: string, options?: PromptOptions) => Promise<void>;
  abort?: () => Promise<void>;
  subscribe?: (listener: (event: PromptEvent) => void) => () => void;
};

export interface PromptTimeoutOptions {
  maxDurationMs?: number;
  abortGraceMs?: number;
  onDiagnostic?: (message: string) => void;
}

const DEFAULT_INACTIVITY_TIMEOUT_MS = 180000;
const DEFAULT_MAX_DURATION_MS = 900000;
const DEFAULT_ABORT_GRACE_MS = 5000;
const PROGRESS_LOG_INTERVAL_MS = 30000;
const ACTIVITY_EVENT_TYPES = new Set([
  "agent_start",
  "message_start",
  "message_update",
  "message_end",
  "agent_end",
]);

function positiveMs(value: number | undefined, fallback: number): number {
  return Number.isFinite(value) && Number(value) > 0 ? Number(value) : fallback;
}

function seconds(ms: number): number {
  return Math.max(1, Math.ceil(ms / 1000));
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function isAbortError(error: unknown): boolean {
  if (!(error instanceof Error)) return false;
  const errorCode = (error as Error & { code?: unknown }).code;
  return error.name === "AbortError" || errorCode === "ABORT_ERR";
}

function boundedProviderError(value: unknown): string {
  return String(value || "").replace(/\s+/g, " ").trim().slice(0, 1000);
}

export async function promptWithTimeout(
  session: PromptSession,
  text: string,
  inactivityTimeoutMs: number,
  label: string,
  options?: PromptOptions,
  timeoutOptions: PromptTimeoutOptions = {},
): Promise<void> {
  const effectiveInactivityMs = positiveMs(inactivityTimeoutMs, DEFAULT_INACTIVITY_TIMEOUT_MS);
  const configuredMaxDurationMs = positiveMs(timeoutOptions.maxDurationMs, DEFAULT_MAX_DURATION_MS);
  const effectiveMaxDurationMs = configuredMaxDurationMs;
  const abortGraceMs = positiveMs(timeoutOptions.abortGraceMs, DEFAULT_ABORT_GRACE_MS);
  const startedAt = Date.now();
  let lastActivityAt = startedAt;
  let lastActivity = "prompt_start";
  let lastDiagnosticAt = 0;
  let providerError = "";
  let promptError: unknown;
  let inactivityTimer: ReturnType<typeof setTimeout> | undefined;
  let maxDurationTimer: ReturnType<typeof setTimeout> | undefined;
  let unsubscribe: (() => void) | undefined;
  let timeoutTriggered = false;
  let settled = false;

  const diagnostic = (message: string): void => {
    try {
      timeoutOptions.onDiagnostic?.(message);
    } catch {
      // Diagnostics must never affect the provider request.
    }
  };

  const clearTimers = (): void => {
    if (inactivityTimer) clearTimeout(inactivityTimer);
    if (maxDurationTimer) clearTimeout(maxDurationTimer);
    inactivityTimer = undefined;
    maxDurationTimer = undefined;
  };

  return new Promise<void>((resolve, reject) => {
    const finish = (error?: unknown): void => {
      if (settled) return;
      settled = true;
      clearTimers();
      unsubscribe?.();
      if (error !== undefined) reject(error);
      else resolve();
    };

    const timeoutMessage = (kind: "inactivity" | "maximum_duration", now: number): string => {
      const elapsedMs = now - startedAt;
      const idleMs = now - lastActivityAt;
      if (kind === "inactivity") {
        return `${label} timed out after ${seconds(effectiveInactivityMs)}s without provider progress ` +
          `(elapsed=${seconds(elapsedMs)}s, last_event=${lastActivity})`;
      }
      return `${label} exceeded maximum duration of ${seconds(effectiveMaxDurationMs)}s ` +
        `(idle=${seconds(idleMs)}s, last_event=${lastActivity})`;
    };

    const triggerTimeout = async (kind: "inactivity" | "maximum_duration"): Promise<void> => {
      if (settled || timeoutTriggered) return;
      timeoutTriggered = true;
      clearTimers();
      const now = Date.now();
      const timeoutError = new Error(timeoutMessage(kind, now));
      diagnostic(
        `prompt_timeout kind=${kind} elapsed_ms=${now - startedAt} idle_ms=${now - lastActivityAt} ` +
        `last_event=${lastActivity}`,
      );

      let abortStatus = "unsupported";
      if (session.abort) {
        abortStatus = "pending";
        try {
          let graceTimer: ReturnType<typeof setTimeout> | undefined;
          try {
            const result = await Promise.race([
              session.abort().then(() => "completed" as const),
              new Promise<"grace_expired">((resolveAbort) => {
                graceTimer = setTimeout(() => resolveAbort("grace_expired"), abortGraceMs);
              }),
            ]);
            abortStatus = result;
          } finally {
            if (graceTimer) clearTimeout(graceTimer);
          }
        } catch (error) {
          abortStatus = `failed:${boundedProviderError(errorMessage(error))}`;
        }
      }
      diagnostic(`prompt_abort status=${abortStatus} grace_ms=${abortGraceMs}`);

      if (settled) return;
      if (providerError) {
        finish(new Error(`${label} provider error before ${kind} timeout: ${providerError}`));
        return;
      }
      if (promptError !== undefined && !isAbortError(promptError)) {
        finish(promptError);
        return;
      }
      finish(timeoutError);
    };

    const armInactivityTimer = (): void => {
      if (inactivityTimer) clearTimeout(inactivityTimer);
      inactivityTimer = setTimeout(() => {
        void triggerTimeout("inactivity");
      }, effectiveInactivityMs);
    };

    if (session.subscribe) {
      unsubscribe = session.subscribe((event) => {
        const eventType = String(event.type || "unknown");
        if (!ACTIVITY_EVENT_TYPES.has(eventType) || settled) return;

        const stopReason = String(event.message?.stopReason || "pending");
        if (
          eventType === "message_end" &&
          event.message?.role === "assistant" &&
          stopReason === "error"
        ) {
          providerError = boundedProviderError(event.message.errorMessage);
        }
        if (timeoutTriggered) return;

        const now = Date.now();
        lastActivityAt = now;
        lastActivity = `${eventType}:${stopReason}`;
        armInactivityTimer();

        if (eventType !== "message_update" || now - lastDiagnosticAt >= PROGRESS_LOG_INTERVAL_MS) {
          diagnostic(
            `prompt_progress event=${eventType} stop=${stopReason} elapsed_ms=${now - startedAt}`,
          );
          lastDiagnosticAt = now;
        }
      });
    }

    diagnostic(
      `prompt_wait_start inactivity_timeout_ms=${effectiveInactivityMs} ` +
      `max_duration_ms=${effectiveMaxDurationMs} abort_grace_ms=${abortGraceMs}`,
    );
    armInactivityTimer();
    maxDurationTimer = setTimeout(() => {
      void triggerTimeout("maximum_duration");
    }, effectiveMaxDurationMs);

    Promise.resolve()
      .then(() => session.prompt(text, options))
      .then(
        () => {
          if (!timeoutTriggered) finish();
        },
        (error: unknown) => {
          promptError = error;
          if (!timeoutTriggered || !isAbortError(error)) finish(error);
        },
      );
  });
}
