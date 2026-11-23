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
const requests = new Set();
const check = (name, ok, detail = "") => {
  console.log((ok ? "PASS " : "FAIL ") + name + (detail ? " :: " + detail : ""));
  if (!ok) failures.push(name);
};

if (spawnSync("chromium", ["--version"], { stdio: "ignore" }).status !== 0) {
  console.error("test:browser needs `chromium` on PATH (no browser dependency is added by design).");
  process.exit(2);
}

const now = Date.now();
const ACTIVE = {
  request_id: "r-live", model: "fake", engine: "e1", session: "web-live", phase: "decode",
  prompt_tokens: 50, completion_tokens: 7, started_at_ms: now,
};
const ROW = {
  model: "fake", engine: "e1", session: "web-test", started_at_ms: now, stream: true,
  status: 200, outcome: "ok", error_code: null, finish_reason: "stop",
  prompt_tokens: 100, completion_tokens: 20, reused_tokens: 10,
  queue_wait_ms: 3, prefill_ms: 40, time_to_first_token_ms: 120, decode_tokens_per_second: 55.5,
  prefill_bank_passes: 4, prefill_bank_misses: 2, prefill_bank_upload_bytes: 1024,
};
let sample = 0;
let logPoll = 0;
const stub = (url) => {
  if (url.endsWith("/health") || url.endsWith("/ready")) return { body: { status: "ok" } };
  if (url.includes("/v1/models")) return { body: { object: "list", data: [{ id: "fake", object: "model", created: 1700000000, owned_by: "fifi", engines: [{ id: "e1", state: "ready", cache: { kv_bytes: 1024, prefix_hits: 3, prefix_misses: 1 } }] }] } };
  if (url.includes("/v1/system")) {
    sample += 1;
    return { body: { version: "test", backends_compiled: [], host: {
      cpu_threads: 8, mem_total_bytes: 8589934592, mem_available_bytes: 4294967296,
      cpu_times: [sample * 25, 0, 0, sample * 75],
      proc_io: { read_bytes: sample * 10000, write_bytes: 0 },
    }, engines: [{ id: "e1", model: "fake", state: "ready", queued: 1, context_limit: 4096, settings: [],
      expert_placement: { mode: "disk_streamed", vram_bytes: 1024, ram_bytes: 0,
        disk_bytes: 8 * 1024 ** 3, slots_per_layer: 8 } }] } };
  }
  if (url.includes("/v1/logs")) {
    logPoll += 1;
    return { body: { events: Array.from({ length: 30 + logPoll }, (_, i) => ({
      ts_ms: now + i, line: i === 0 ? "engine e1 registered" : "Download finished " + i,
    })) } };
  }
  if (url.includes("/v1/telemetry")) return { body: { gpus: [{ index: 0, name: "Test GPU", util_percent: 42, mem_used_bytes: 1073741824, mem_total_bytes: 8589934592, temp_c: 50, power_w: 100 }], source: "nvidia-smi", error: null } };
  if (url.includes("/v1/recent-requests")) return { body: { requests: [ROW], active: [ACTIVE] } };
  if (url.includes("/v1/engines")) return { body: { engines: [] } };
  if (url.includes("/v1/hf/cache")) return { body: { hub_dir: "/demo/cache", models: [] } };
  if (url.includes("/v1/hf/downloads")) return { body: { downloads: [
    { id: 1, repo: "org/finished", revision: "main", state: "done" },
    { id: 2, repo: "org/active", revision: "main", state: "downloading" },
  ] } };
  if (url.includes("/v1/hf/search")) return { body: { results: URL.parse(url)?.searchParams.get("gguf") === "true"
    ? [
      { id: "org/model-gguf", likes: 8, downloads: 12, gated: true, pipeline_tag: "text-generation" },
      { id: "org/open-gguf", likes: 4, downloads: 9, gated: false },
    ] : [] } };
  if (url.includes("/v1/hf/tree")) return { body: {
    revision: "main", commit: "a".repeat(40), total_bytes: 10 * 2 ** 30,
    files: [{ path: "model-Q4_K_M.gguf", size: 4 * 2 ** 30 }, { path: "model-Q8_0.gguf", size: 6 * 2 ** 30 }],
  } };
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
      requests.add(request.url);
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
  await waitFor(() => evaluate("document.querySelector('#model-picker').value === 'fake' && !document.querySelector('#model-picker').disabled"), "chat model ready");
  await evaluate("(() => { const i = document.querySelector('#composer-input'); i.focus(); i.value = 'hello'; i.dispatchEvent(new Event('input', {bubbles: true})); return true; })()");
  await waitFor(() => evaluate("!document.querySelector('#send-button').disabled"), "chat send ready");
  await evaluate("document.querySelector('#send-button').click(); true");
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

  // Monitor: hero band, active lanes, rates, GPU, detail drawer.
  await evaluate("location.hash = '#monitor'; true");
  await waitFor(() => evaluate("!!document.querySelector('.monitor-hero')"), "hero band");
  check("hero band shows in-flight count and GPU load", await evaluate(`(() => {
    const hero = document.querySelector(".monitor-hero").textContent;
    return hero.includes("in flight") && hero.includes("waiting") && hero.includes("GPU load");
  })()`));
  const liveLane = await waitFor(async () => {
    const t = await evaluate(`(() => {
      const row = [...document.querySelectorAll("#view-monitor tr")].find(tr => tr.textContent.includes("web-live"));
      return row ? row.textContent : null;
    })()`);
    return t && t.includes("decode") ? t : null;
  }, "active board lane");
  check("active board lane shows phase, session and context use", liveLane.includes("decode") && liveLane.includes("web-live") && liveLane.includes("%"));
  const mtext = await waitFor(async () => {
    const t = await evaluate("document.querySelector('#view-monitor').textContent");
    return t.includes("Session · web-test") ? t : null;
  }, "rates card with session group");
  check("rates groups render", mtext.includes("Session · web-test"), mtext.slice(0, 200));
  check("GPU table renders", mtext.includes("Test GPU"));
  check("engine expert placement renders reported tiers",
    mtext.includes("Disk streaming · VRAM bank 1 KiB · Pinned RAM store 0 B · Disk backing 8.0 GiB · 8 slots/layer"));

  await waitFor(() => evaluate("document.querySelector('.log-tail')?.scrollTop > 0"), "event tail follows newest");
  check("event tail starts at the bottom", await evaluate(`(() => {
    const tail = document.querySelector(".log-tail");
    return tail.scrollTop + tail.clientHeight >= tail.scrollHeight - 4;
  })()`));
  const eventCount = await evaluate(`(() => {
    const tail = document.querySelector(".log-tail");
    tail.scrollTop = 40;
    tail.focus({ preventScroll: true });
    document.querySelector(".monitor-refresh").click();
    return tail.childElementCount;
  })()`);
  await waitFor(() => evaluate(`document.querySelector(".log-tail")?.childElementCount > ${eventCount}`), "new events after refresh");
  check("event tail preserves reading position and keyboard focus", await evaluate(`(() => {
    const tail = document.querySelector(".log-tail");
    return tail.scrollTop === 40 && document.activeElement === tail;
  })()`));
  check("host CPU and process disk rates render after the next sample", await evaluate(`(() => {
    const hero = document.querySelector(".monitor-hero").textContent;
    return hero.includes("25%Host CPU") && hero.includes("↓ ") && hero.includes("/s · ↑ 0 B/sProcess disk");
  })()`));

  await waitFor(() => evaluate("!!document.querySelector('.req-inspect')"), "request row");
  await evaluate("document.querySelector('.req-inspect').click(); true");
  await waitFor(() => evaluate("document.querySelector('#monitor-request-detail').open"), "detail drawer");
  check("request detail drawer shows request facts", await evaluate("document.querySelector('#monitor-request-detail').textContent.includes('Prompt tokens')"));
  check("request detail shows measured prefill expert activity", await evaluate(
    "document.querySelector('#monitor-request-detail').textContent.includes('4 admit passes · 2 experts moved · 1 KiB queued for copy')"
  ));
  await evaluate("document.querySelector('#monitor-request-detail button').click(); true");

  // Discovery: a search with no results can be retried with a format filter.
  await evaluate('location.hash = "#models"; true');
  await waitFor(() => evaluate('!document.querySelector("#view-models").hidden'), "Models view");
  await waitFor(() => evaluate('document.querySelector(".models-capacity").innerText.includes("7.0 GiB")'), "server capacity");
  check("local status is visible without opening a disclosure", await evaluate(`(() => {
    const headings = [...document.querySelectorAll(".models-local h2")];
    return ["Server capacity", "Engines", "Local cache"].every(label =>
      headings.some(heading => heading.textContent === label && heading.getClientRects().length > 0));
  })()`));
  await waitFor(() => evaluate('document.querySelector(".models-download-launcher").textContent.includes("1 active")'), "active download badge");
  await evaluate('document.querySelector(".models-download-launcher").click(); true');
  check("download drawer clears finished entries and keeps active entries", await evaluate(`(() => {
    const drawer = document.querySelector("#models-downloads");
    if (!drawer.open || !drawer.textContent.includes("org/finished")) return false;
    [...drawer.querySelectorAll("button")].find(button => button.textContent === "Clear finished").click();
    const cleared = !drawer.textContent.includes("org/finished") && drawer.textContent.includes("org/active");
    drawer.querySelector('[aria-label="Close downloads"]').click();
    return cleared && !drawer.open;
  })()`));
  await evaluate(`(() => {
    const form = document.querySelector('#view-models form');
    form.querySelector("input").value = "test";
    form.requestSubmit();
    return true;
  })()`);
  await waitFor(() => evaluate('document.querySelector("#view-models").textContent.includes("No models match your search.")'), "empty search");
  await evaluate(`(() => {
    const form = document.querySelector('#view-models form');
    form.querySelector("select").value = "gguf";
    form.requestSubmit();
    return true;
  })()`);
  await waitFor(() => evaluate('document.querySelector("#view-models").textContent.includes("Access required")'), "gated result");
  check("filtered search shows gated repository without a Download button", await evaluate(`(() => {
    const result = document.querySelector('#view-models a[href="https://huggingface.co/org/model-gguf"]');
    return !!result && !result.closest("article").querySelector("button");
  })()`));
  check("download is available before opening comparison", await evaluate(`(() => {
    const download = document.querySelector('#view-models button[aria-label="Download full revision for org/open-gguf"]');
    const details = document.getElementById("models-fit-1");
    return download?.getClientRects().length > 0 && details?.hidden;
  })()`));
  await evaluate(`document.querySelector('button[aria-controls="models-fit-1"]').click(); true`);
  await waitFor(() => evaluate('document.querySelector("#view-models").textContent.includes("Recommended starting point")'), "quant recommendation");
  check("model fit contrasts recommended GGUF with the full download", await evaluate(`(() => {
    const view = document.querySelector("#view-models");
    return view.textContent.includes("VRAM only")
      && view.textContent.includes("SSD streaming candidate")
      && view.textContent.includes("Download full revision · 10.0 GiB")
      && [...view.querySelectorAll("a")].some(link => link.href.endsWith("/model-Q4_K_M.gguf"));
  })()`));
} catch (err) {
  check("browser run completed", false, String(err).slice(0, 300));
  if (sessionId !== null) {
    try {
      console.log("Requests:", [...requests]);
      console.log("Page:", await evaluate("document.body.innerText.slice(-800)"));
    } catch { /* the browser may have already exited */ }
  }
} finally {
  try { ws?.close(); } catch {}
  await Promise.all([stopChild(chrome, { processGroup: true }), stopChild(vite)]);
  rmSync(profile, { recursive: true, force: true, maxRetries: 10, retryDelay: 100 });
}
console.log(failures.length === 0 ? "ALL BROWSER CHECKS PASSED" : "FAILURES: " + failures.join(", "));
process.exit(failures.length === 0 ? 0 : 1);
