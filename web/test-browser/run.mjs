/**
 * In-repo browser tests for the web GUI, with zero added dependencies:
 * Node's built-in WebSocket drives the system Chromium over CDP, and
 * the API is stubbed at the Fetch layer, so no engine build is needed.
 * Run with `npm run test:browser` (builds first). Requires `chromium`
 * on PATH; exits 2 with a clear message when it is absent.
 */
import { spawn, spawnSync } from "node:child_process";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { stopChild } from "./cleanup.mjs";

const PORT = 4319;
const BASE = `http://127.0.0.1:${PORT}`;
const failures = [];
const check = (name, ok, detail = "") => {
  console.log((ok ? "PASS " : "FAIL ") + name + (detail ? " :: " + detail : ""));
  if (!ok) failures.push(name);
};

if (spawnSync("chromium", ["--version"], { stdio: "ignore" }).status !== 0) {
  console.error("test:browser needs `chromium` on PATH (no browser dependency is added by design).");
  process.exit(2);
}

const now = Date.now();
const ROW = {
  model: "fake", engine: "e1", session: "web-test", started_at_ms: now, stream: true,
  status: 200, outcome: "ok", error_code: null, finish_reason: "stop",
  prompt_tokens: 100, completion_tokens: 20, reused_tokens: 10,
  queue_wait_ms: 3, prefill_ms: 40, time_to_first_token_ms: 120, decode_tokens_per_second: 55.5,
};
const stub = (url) => {
  if (url.endsWith("/health") || url.endsWith("/ready")) return { body: { status: "ok" } };
  if (url.includes("/v1/models")) return { body: { object: "list", data: [{ id: "fake", object: "model", created: 1700000000, owned_by: "fifi", engines: [{ id: "e1", state: "ready", cache: { kv_bytes: 1024, prefix_hits: 3, prefix_misses: 1 } }] }] } };
  if (url.includes("/v1/system")) return { body: { version: "test", backends_compiled: [], host: { cpu_threads: 8, mem_total_bytes: 8589934592, mem_available_bytes: 4294967296 }, engines: [{ id: "e1", model: "fake", state: "ready", queued: 1, context_limit: 4096, settings: [] }] } };
  if (url.includes("/v1/telemetry")) return { body: { gpus: [{ index: 0, name: "Test GPU", util_percent: 42, mem_used_bytes: 1073741824, mem_total_bytes: 8589934592, temp_c: 50, power_w: 100 }], source: "nvidia-smi", error: null } };
  if (url.includes("/v1/recent-requests")) return { body: { requests: [ROW] } };
  return null;
};
const SSE = [
  'data: {"choices":[{"delta":{"content":"```ts\\nconst x"}}]}\n\n',
  'data: {"choices":[{"delta":{"content":" = \\"hi\\";\\n```\\nDone."}}]}\n\n',
  'data: {"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}\n\n',
  "data: [DONE]\n\n",
].join("");

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
async function waitFor(fn, label, timeout = 15000) {
  const start = Date.now();
  for (;;) {
    const v = await fn();
    if (v) return v;
    if (Date.now() - start > timeout) throw new Error("timeout waiting for " + label);
    await sleep(150);
  }
}

// Spawn Vite directly so the process we stop actually owns the listening port.
const vite = spawn(process.execPath, [join(process.cwd(), "node_modules", "vite", "bin", "vite.js"),
  "preview", "--port", String(PORT), "--strictPort", "--host", "127.0.0.1"], { stdio: "ignore" });
const profile = mkdtempSync(join(tmpdir(), "fifi-browser-test-"));
const chrome = spawn("chromium", ["--headless=new", "--disable-gpu", "--no-sandbox", "--remote-debugging-port=9339", `--user-data-dir=${profile}`, "about:blank"],
  { stdio: "ignore", detached: process.platform !== "win32" });

let ws; let nextId = 0; const pending = new Map(); let sessionId = null;
const send = (method, params = {}) => new Promise((resolve, reject) => {
  const id = ++nextId;
  pending.set(id, { resolve, reject });
  ws.send(JSON.stringify({ id, method, params, ...(sessionId ? { sessionId } : {}) }));
});
const evaluate = async (expression) => {
  const r = await send("Runtime.evaluate", { expression, returnByValue: true, awaitPromise: true });
  if (r.exceptionDetails) throw new Error("evaluate failed: " + JSON.stringify(r.exceptionDetails).slice(0, 300));
  return r.result?.value;
};

try {
  await waitFor(async () => {
    try { return (await fetch(BASE + "/")).ok; } catch { return false; }
  }, "vite preview");
  const version = await waitFor(async () => {
    try { return await (await fetch("http://127.0.0.1:9339/json/version")).json(); } catch { return null; }
  }, "chromium CDP");
  ws = new WebSocket(version.webSocketDebuggerUrl);
  await new Promise((r) => ws.addEventListener("open", r, { once: true }));
  ws.addEventListener("message", (ev) => {
    const msg = JSON.parse(String(ev.data));
    if (msg.id && pending.has(msg.id)) {
      const { resolve, reject } = pending.get(msg.id);
      pending.delete(msg.id);
      if (msg.error) reject(new Error(msg.error.message));
      else resolve(msg.result);
    }
    if (msg.method === "Fetch.requestPaused" && (msg.sessionId === sessionId || msg.params)) {
      const { requestId, request } = msg.params;
      const respond = (status, type, body) => ws.send(JSON.stringify({
        id: ++nextId + 100000, method: "Fetch.fulfillRequest", sessionId,
        params: { requestId, responseCode: status, responseHeaders: [{ name: "Content-Type", value: type }], body: Buffer.from(body).toString("base64") },
      }));
      if (request.url.includes("/v1/chat/completions")) {
        respond(200, "text/event-stream", SSE);
      } else {
        const hit = stub(request.url);
        if (hit) respond(200, "application/json", JSON.stringify(hit.body));
        else ws.send(JSON.stringify({ id: ++nextId + 100000, method: "Fetch.continueRequest", sessionId, params: { requestId } }));
      }
    }
  });
  const target = await send("Target.createTarget", { url: "about:blank" });
  const attached = await send("Target.attachToTarget", { targetId: target.targetId, flatten: true });
  sessionId = attached.sessionId;
  await send("Page.enable");
  await send("Runtime.enable");
  await send("Fetch.enable", { patterns: [{ urlPattern: BASE + "/v1/*" }, { urlPattern: BASE + "/health" }, { urlPattern: BASE + "/ready" }] });
  await send("Page.addScriptToEvaluateOnNewDocument", {
    source: "localStorage.setItem('fifi-settings', JSON.stringify({apiKey:'test', theme:'system', model:'fake', drawer:{temperature:0.7,topP:1,topK:0,maxTokens:2048,thinking:'default',systemMessage:'',recordTrace:false}}));",
  });
  await send("Page.navigate", { url: BASE + "/" });
  await waitFor(() => evaluate("!!document.querySelector('#composer-input')"), "composer");

  // Streaming markdown: code box, highlight spans, copy button.
  await waitFor(() => evaluate("!!document.querySelector('.attach-btn')"), "chat view ready");
  await evaluate("(() => { const i = document.querySelector('#composer-input'); i.focus(); i.value = 'hello'; i.dispatchEvent(new Event('input', {bubbles: true})); document.querySelector('#send-button').click(); return true; })()");
  await waitFor(() => evaluate("!!document.querySelector('.code-block')"), "code box");
  check("streaming code box renders", true);
  check("syntax highlight spans", await evaluate("document.querySelectorAll('.code-block .tok-keyword').length >= 1"));
  check("code copy button", await evaluate("!!document.querySelector('.code-block .code-copy')"));
  check("message copy action", await evaluate("[...document.querySelectorAll('.msg-copy')].some(b => b.textContent === 'Copy')"));
  check("regenerate action", await evaluate("[...document.querySelectorAll('.msg-copy')].some(b => b.textContent === 'Regenerate')"));

  // Large paste becomes a chip, not textarea text.
  await evaluate(`(() => { const input = document.querySelector('#composer-input'); const text = Array.from({length: 40}, (_, i) => 'line ' + i).join('\\n'); const e = new ClipboardEvent('paste', {bubbles: true, cancelable: true}); Object.defineProperty(e, 'clipboardData', {value: {getData: (t) => t === 'text' ? text : '', files: []}}); input.dispatchEvent(e); return true; })()`);
  check("large paste chip", await evaluate("document.querySelectorAll('.paste-chip').length === 1"));
  check("textarea not flooded", await evaluate("document.querySelector('#composer-input').value === ''"));

  // Monitor: cards, rates, GPU, detail drawer.
  await evaluate("location.hash = '#monitor'; true");
  await waitFor(() => evaluate("document.querySelectorAll('.stat-card').length === 4"), "stat cards");
  check("monitor stat cards", true);
  const mtext = await waitFor(async () => {
    const t = await evaluate("document.querySelector('#view-monitor').textContent");
    return t.includes("Session · web-test") ? t : null;
  }, "rates card with session group");
  check("rates groups render", mtext.includes("Session · web-test"), mtext.slice(0, 200));
  check("GPU card renders", mtext.includes("Test GPU"));

  await waitFor(() => evaluate("!!document.querySelector('.req-row-clickable')"), "request row");
  await evaluate("document.querySelector('.req-row-clickable').click(); true");
  await waitFor(() => evaluate("document.querySelector('#view-monitor').textContent.includes('Request detail')"), "detail drawer");
  check("request detail drawer", true);
} catch (err) {
  check("browser run completed", false, String(err).slice(0, 300));
} finally {
  try { ws?.close(); } catch {}
  await Promise.all([stopChild(chrome, { processGroup: true }), stopChild(vite)]);
  rmSync(profile, { recursive: true, force: true, maxRetries: 10, retryDelay: 100 });
}
console.log(failures.length === 0 ? "ALL BROWSER CHECKS PASSED" : "FAILURES: " + failures.join(", "));
process.exit(failures.length === 0 ? 0 : 1);
