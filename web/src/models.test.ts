import { test } from "node:test";
import { strictEqual } from "node:assert";
import { statusOf } from "./models.js";
import { searchHubModels, startHubDownload, type HubDownloadJob } from "./api.js";

function job(overrides: Partial<HubDownloadJob>): HubDownloadJob {
  return { id: 1, repo: "org/name", revision: "main", state: "resolving", ...overrides };
}

test("download status lines carry the counters the GUI polls", () => {
  strictEqual(statusOf(job({})), "resolving");
  strictEqual(statusOf(job({ state: "done" })), "done");
  strictEqual(statusOf(job({ state: "cancelled" })), "cancelled");
  strictEqual(
    statusOf(job({ state: "failed", error: "hub returned 429" })),
    "hub returned 429",
  );
  strictEqual(
    statusOf(job({
      state: "downloading",
      files_done: 3, files_total: 10,
      bytes: 2 * 1024 ** 3, bytes_total: 5 * 1024 ** 3,
    })),
    "3/10 files, 2.0 GiB of 5.0 GiB",
  );
  strictEqual(
    statusOf(job({
      state: "downloading",
      files_done: 1, files_total: 2,
      bytes: 1024, bytes_total: 4096,
      current_file: "model-00001-of-00002.gguf",
    })),
    "1/2 files, 1 KiB of 4 KiB — model-00001-of-00002.gguf",
  );
});

test("hub search builds the hub's own filter query", async () => {
  const original = globalThis.fetch;
  let target = "";
  globalThis.fetch = (async (input: RequestInfo | URL) => {
    target = String(input);
    return new Response(JSON.stringify({ results: [] }), { status: 200 });
  }) as typeof fetch;
  try {
    await searchHubModels("qwen 36", { gguf: true, exl3: false });
    strictEqual(target, "/v1/hf/search?q=qwen+36&gguf=true");
    await searchHubModels("gpt2", {});
    strictEqual(target, "/v1/hf/search?q=gpt2");
  } finally {
    globalThis.fetch = original;
  }
});

test("a download starts as a POST with the repo only", async () => {
  const original = globalThis.fetch;
  let method = "";
  let body = "";
  globalThis.fetch = (async (_input: RequestInfo | URL, init?: RequestInit) => {
    method = (init?.method ?? "GET").toUpperCase();
    body = String(init?.body ?? "");
    return new Response(
      JSON.stringify({ id: 7, repo: "org/name", revision: "main" }),
      { status: 202 },
    );
  }) as typeof fetch;
  try {
    const started = await startHubDownload("org/name");
    strictEqual(method, "POST");
    strictEqual(body, JSON.stringify({ repo: "org/name" }));
    strictEqual(started.id, 7);
  } finally {
    globalThis.fetch = original;
  }
});
