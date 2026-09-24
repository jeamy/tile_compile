// Dev/evaluation script (not part of the sidecar build): sends a prepared pi.decisions.request.v1 to the
// real Jev endpoint through the same DecisionsService the sidecar uses, N times, and writes the
// normalized responses. Costs ~1e-5 USD per call. Needs JEV_OPENROUTER_API_KEY in the environment/.env.
//   node --import tsx scripts/jev_eval_call.ts <request.json> <out.jsonl> [repeats=5]
import fs from "node:fs";
import path from "node:path";
import dotenv from "dotenv";
import { DecisionsService } from "../src/services/decisionsService.js";
import { createJevLogger } from "../src/services/decisionsLog.js";

dotenv.config({ path: path.resolve(process.cwd(), "..", ".env") });
const [reqPath, outPath, repeatsArg] = process.argv.slice(2);
if (!reqPath || !outPath) throw new Error("usage: jev_eval_call.ts <request.json> <out.jsonl> [repeats]");
const request = JSON.parse(fs.readFileSync(reqPath, "utf8"));
const svc = new DecisionsService({ mode: "shadow", maxConcurrent: 1 }, { trace: createJevLogger("eval") });
const lines: string[] = [];
for (let i = 0; i < Number(repeatsArg || 5); i++) {
  // a distinct request_id per repeat; identical state_hash is fine because calls are sequential (no dedup overlap)
  const res = await svc.decide({ ...request, request_id: `${request.request_id}-${i}` });
  lines.push(JSON.stringify(res));
}
fs.writeFileSync(outPath, lines.join("\n") + "\n");
console.log(`${lines.length} responses -> ${outPath}`);
