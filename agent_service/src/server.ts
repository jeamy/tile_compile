import http from "node:http";
import { runtimeConfig } from "./config.js";
import { AuthService } from "./services/authService.js";
import { FrameAnalysisService } from "./services/frameAnalysisService.js";
import { ModelService } from "./services/modelService.js";
import { LiveImageChatService } from "./services/liveImageChatService.js";
import { RunChatService } from "./services/runChatService.js";
import { appendTrafficLog, readTrafficLog } from "./services/trafficLog.js";
import { createJevLogger, readJevLog } from "./services/decisionsLog.js";
import { DecisionsRequestError, DecisionsService, decisionsConfigFromEnv } from "./services/decisionsService.js";
import { decisionsSettingsPath, loadDecisionsSettings, saveDecisionsSettings } from "./services/decisionsSettings.js";
import type { AnalysisProgressEvent } from "./types.js";

const config = runtimeConfig();
const modelService = new ModelService(config.projectRoot);
const authService = new AuthService(modelService);

// Jev decisions adapter (independent of PI's provider slot). An invalid PI_DECISIONS_* setting must not
// take the whole sidecar down: the route answers 503 with the reason instead.
let decisionsService: DecisionsService | null = null;
let decisionsConfigError = "";
try {
  decisionsService = new DecisionsService(decisionsConfigFromEnv(), {
    log: (line) => appendTrafficLog(line),
    // Every wire request, raw response and normalized result go to their own file (jevLogPath()).
    trace: createJevLogger("sidecar"),
  });
  // Settings saved through the Jev card override the environment defaults.
  const stored = loadDecisionsSettings(decisionsSettingsPath());
  decisionsService.applySettings({
    mode: stored.mode,
    allowExperimentalSuggestions: stored.allow_experimental_suggestions,
    apiKey: stored.api_key,
  });
} catch (error) {
  decisionsConfigError = error instanceof Error ? error.message : String(error);
  appendTrafficLog(`decisions config invalid: ${decisionsConfigError}`);
}

function sendJson(res: http.ServerResponse, status: number, payload: unknown) {
  const body = JSON.stringify(payload);
  res.writeHead(status, {
    "Content-Type": "application/json",
    "Content-Length": Buffer.byteLength(body),
  });
  res.end(body);
}

async function readJsonLimited(req: http.IncomingMessage, maxBytes: number): Promise<any> {
  const chunks: Buffer[] = [];
  let total = 0;
  for await (const chunk of req) {
    const buf = Buffer.from(chunk);
    total += buf.length;
    if (total > maxBytes) throw new DecisionsRequestError("body_too_large");
    chunks.push(buf);
  }
  const raw = Buffer.concat(chunks).toString("utf8");
  if (!raw.trim()) return {};
  try {
    return JSON.parse(raw);
  } catch {
    throw new DecisionsRequestError("body_not_json");
  }
}

async function readJson(req: http.IncomingMessage): Promise<any> {
  const chunks: Buffer[] = [];
  for await (const chunk of req) chunks.push(Buffer.from(chunk));
  const raw = Buffer.concat(chunks).toString("utf8");
  if (!raw.trim()) return {};
  return JSON.parse(raw);
}

function setCorsHeaders(res: http.ServerResponse) {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");
}

async function handle(req: http.IncomingMessage, res: http.ServerResponse) {
  const url = new URL(req.url || "/", `http://${req.headers.host || "127.0.0.1"}`);

  // Handle CORS preflight
  if (req.method === "OPTIONS") {
    setCorsHeaders(res);
    res.writeHead(200);
    res.end();
    return;
  }

  setCorsHeaders(res);
  try {
    if (req.method === "GET" && url.pathname === "/health") {
      sendJson(res, 200, { ok: true, status: "ok" });
      return;
    }
    if (req.method === "GET" && url.pathname === "/models") {
      sendJson(res, 200, await modelService.modelsJson());
      return;
    }
    if (req.method === "GET" && url.pathname === "/account") {
      sendJson(res, 200, await modelService.accountJson(url.searchParams.get("provider") || ""));
      return;
    }
    if (req.method === "GET" && url.pathname === "/traffic") {
      sendJson(res, 200, {
        schema_version: "pi.ai-traffic.v1",
        privacy_class: "redacted",
        ...readTrafficLog(Number(url.searchParams.get("limit") || 500)),
      });
      return;
    }
    if (req.method === "POST" && url.pathname === "/auth") {
      const body = await readJson(req);
      sendJson(res, 200, await authService.storeKey(String(body.provider || ""), String(body.api_key || "")));
      return;
    }
    if (req.method === "DELETE" && url.pathname.startsWith("/auth/")) {
      const provider = decodeURIComponent(url.pathname.slice("/auth/".length));
      sendJson(res, 200, await authService.removeKey(provider));
      return;
    }
    if (req.method === "POST" && url.pathname === "/test") {
      const body = await readJson(req);
      const modelRef = String(body.model || config.agent.model || "");
      const overrideRaw = body.vision_override;
      const visionOverride = overrideRaw === true ? true : overrideRaw === false ? false : overrideRaw === null ? null : undefined;
      const result = await modelService.testModel(modelRef, {
        visionProbe: Boolean(body.vision_probe),
        visionOverride,
      });
      sendJson(res, result.ok ? 200 : 404, result);
      return;
    }
    if (req.method === "POST" && url.pathname === "/analyze") {
      const body = await readJson(req);
      appendTrafficLog(`POST /analyze request ${JSON.stringify(body).substring(0, 10000)}`);
      const service = new FrameAnalysisService(config.agent, modelService);
      const result = await service.analyze(body);
      appendTrafficLog(`POST /analyze response ${JSON.stringify(result).substring(0, 10000)}`);
      sendJson(res, 200, result);
      return;
    }
    if (url.pathname === "/decisions/status" && req.method === "GET") {
      if (!decisionsService) {
        sendJson(res, 503, { error: true, code: "DECISIONS_CONFIG_INVALID", message: decisionsConfigError });
        return;
      }
      sendJson(res, 200, await decisionsService.status());
      return;
    }
    if (url.pathname === "/decisions/log" && req.method === "GET") {
      // Read-only view of the Jev request/response log (already redacted when written).
      sendJson(res, 200, { schema_version: "pi.jev-traffic.v1", privacy_class: "redacted", ...readJevLog(Number(url.searchParams.get("limit") || 500)) });
      return;
    }
    if (url.pathname === "/decisions/settings" && req.method === "POST") {
      if (!decisionsService) {
        sendJson(res, 503, { error: true, code: "DECISIONS_CONFIG_INVALID", message: decisionsConfigError });
        return;
      }
      // Never logged: the body may contain the API key.
      try {
        const body = await readJsonLimited(req, 16 * 1024);
        const file = decisionsSettingsPath();
        const current = loadDecisionsSettings(file);
        const next = { ...current };
        if (body.mode !== undefined) next.mode = body.mode;
        if (body.allow_experimental_suggestions !== undefined) next.allow_experimental_suggestions = Boolean(body.allow_experimental_suggestions);
        if (body.api_key !== undefined) {
          if (body.api_key === "" || body.api_key === null) delete next.api_key;
          else next.api_key = String(body.api_key);
        }
        decisionsService.applySettings({
          mode: next.mode,
          allowExperimentalSuggestions: next.allow_experimental_suggestions,
          apiKey: next.api_key ?? null,
        });
        saveDecisionsSettings(file, next);
        sendJson(res, 200, await decisionsService.status());
      } catch (error) {
        if (error instanceof DecisionsRequestError) {
          sendJson(res, 400, { error: true, code: "INVALID_REQUEST", message: error.reason });
          return;
        }
        sendJson(res, 400, { error: true, code: "INVALID_SETTINGS", message: error instanceof Error ? error.message : "invalid settings" });
      }
      return;
    }
    if (url.pathname === "/decisions" && req.method === "POST") {
      if (!decisionsService) {
        sendJson(res, 503, { error: true, code: "DECISIONS_CONFIG_INVALID", message: decisionsConfigError });
        return;
      }
      // The body carries the provider state projection: it is never written to the traffic log
      // (DecisionsService logs a one-line summary itself).
      try {
        const abort = new AbortController();
        res.on("close", () => { if (!res.writableEnded) abort.abort(); });
        const result = await decisionsService.decide(await readJsonLimited(req, 256 * 1024), abort.signal);
        sendJson(res, 200, result);
      } catch (error) {
        if (error instanceof DecisionsRequestError) {
          sendJson(res, 400, { error: true, code: "INVALID_REQUEST", message: error.reason });
          return;
        }
        throw error;
      }
      return;
    }
    if (req.method === "POST" && url.pathname === "/analyze/stream") {
      const body = await readJson(req);
      appendTrafficLog(`POST /analyze/stream request ${JSON.stringify(body).substring(0, 10000)}`);
      await handleAnalyzeStream(req, res, body);
      return;
    }
    if (req.method === "POST" && url.pathname === "/run-completion-analysis") {
      const body = await readJson(req);
      if (String(body?.schema_version || "") !== "pi.run-completion-analysis.request.v1") {
        sendJson(res, 400, { error: true, code: "INVALID_SCHEMA", message: "Expected pi.run-completion-analysis.request.v1" });
        return;
      }
      appendTrafficLog(`POST /run-completion-analysis request ${JSON.stringify({ ...body, image_base64: body?.image_base64 ? "<image>" : undefined }).substring(0, 10000)}`);
      const service = new RunChatService(config.agent, modelService);
      const result = await service.ask(body);
      if (String(result.schema_version || "") !== "pi.run-completion-analysis.v1" ||
          !Array.isArray(result.findings) || !Array.isArray(result.updates) ||
          !result.resume_recommendation || typeof result.resume_recommendation !== "object") {
        sendJson(res, 502, { error: true, code: "INVALID_MODEL_RESPONSE", message: "Completion analysis response does not match pi.run-completion-analysis.v1" });
        return;
      }
      appendTrafficLog(`POST /run-completion-analysis response ${JSON.stringify(result).substring(0, 10000)}`);
      sendJson(res, 200, result);
      return;
    }
    if (req.method === "POST" && url.pathname === "/run-chat") {
      const body = await readJson(req);
      appendTrafficLog(`POST /run-chat request ${JSON.stringify({ ...body, image_base64: body?.image_base64 ? "<image>" : undefined }).substring(0, 10000)}`);
      const service = new RunChatService(config.agent, modelService);
      const result = await service.ask(body);
      appendTrafficLog(`POST /run-chat response ${JSON.stringify(result).substring(0, 10000)}`);
      sendJson(res, 200, result);
      return;
    }
    if (req.method === "POST" && url.pathname === "/live-image-chat") {
      const body = await readJson(req);
      appendTrafficLog(`POST /live-image-chat request ${JSON.stringify({ ...body, image_base64: body?.image_base64 ? "<image>" : undefined }).substring(0, 10000)}`);
      const service = new LiveImageChatService(config.agent, modelService);
      const result = await service.ask(body);
      appendTrafficLog(`POST /live-image-chat response ${JSON.stringify(result).substring(0, 10000)}`);
      sendJson(res, 200, result);
      return;
    }

    sendJson(res, 404, { error: "not_found" });
  } catch (error) {
    sendJson(res, 500, {
      error: true,
      message: error instanceof Error ? error.message : "unknown error",
    });
  }
}

async function handleAnalyzeStream(
  req: http.IncomingMessage,
  res: http.ServerResponse,
  body: unknown
) {
  const service = new FrameAnalysisService(config.agent, modelService);

  // Setup SSE headers
  res.writeHead(200, {
    "Content-Type": "text/event-stream",
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
  });

  const sendEvent = (event: AnalysisProgressEvent) => {
    const data = JSON.stringify(event);
    appendTrafficLog(`SSE progress ${data.substring(0, 2000)}`);
    res.write(`event: progress\n`);
    res.write(`data: ${data}\n\n`);
  };

  const sendResult = (result: unknown) => {
    appendTrafficLog(`SSE complete ${JSON.stringify(result).substring(0, 10000)}`);
    res.write(`event: complete\n`);
    res.write(`data: ${JSON.stringify(result)}\n\n`);
    res.end();
  };

  const sendError = (error: Error) => {
    appendTrafficLog(`SSE error ${error.message}`);
    res.write(`event: error\n`);
    res.write(`data: ${JSON.stringify({ message: error.message })}\n\n`);
    res.end();
  };

  try {
    const result = await service.analyze(body as any, sendEvent);
    sendResult(result);
  } catch (error) {
    sendError(error instanceof Error ? error : new Error(String(error)));
  }
}

const server = http.createServer((req, res) => {
  void handle(req, res);
});

server.requestTimeout = 0;
server.headersTimeout = 0;

server.listen(config.port, config.host, () => {
  console.log(`[tile_compile_pi_agent] listening on http://${config.host}:${config.port}`);
});
