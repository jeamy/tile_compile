import http from "node:http";
import { runtimeConfig } from "./config.js";
import { AuthService } from "./services/authService.js";
import { FrameAnalysisService } from "./services/frameAnalysisService.js";
import { ModelService } from "./services/modelService.js";
import { LiveImageChatService } from "./services/liveImageChatService.js";
import { RunChatService } from "./services/runChatService.js";
import { appendTrafficLog, readTrafficLog } from "./services/trafficLog.js";
import type { AnalysisProgressEvent } from "./types.js";

const config = runtimeConfig();
const modelService = new ModelService(config.projectRoot);
const authService = new AuthService(modelService);

function sendJson(res: http.ServerResponse, status: number, payload: unknown) {
  const body = JSON.stringify(payload);
  res.writeHead(status, {
    "Content-Type": "application/json",
    "Content-Length": Buffer.byteLength(body),
  });
  res.end(body);
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
    if (req.method === "POST" && url.pathname === "/analyze/stream") {
      const body = await readJson(req);
      appendTrafficLog(`POST /analyze/stream request ${JSON.stringify(body).substring(0, 10000)}`);
      await handleAnalyzeStream(req, res, body);
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
