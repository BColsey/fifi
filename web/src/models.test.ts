import { test } from "node:test";
import { strictEqual } from "node:assert";
import { statusOf, stopQuestion, stoppedLine, snapshotStatus } from "./models.js";
import { searchHubModels, startHubDownload, type HubDownloadJob } from "./api.js";
import { assessFit, ggufVariants } from "./model-fit.js";

test("GGUF size counts only the complete shard set for each quant", () => {
  const variants = ggufVariants([
    { path: "Q4/model-Q4_K_M-00001-of-00002.gguf", size: 12 },
    { path: "Q4/model-Q4_K_M-00002-of-00002.gguf", size: 8 },
    { path: "model-Q6_K.gguf", size: 27 },
    { path: "model-Q8_0-00001-of-00003.gguf", size: 13 },
    { path: "mmproj-F16.gguf", size: 2 },
  ]);
  strictEqual(variants.map(({ bytes }) => bytes).join(","), "20,27");
  strictEqual(variants[0]?.files.length, 2);
});

test("fit prefers the fastest tier with headroom without adding VRAM to RAM", () => {
  const hardware = { vram: 24, ram: 32, ssd: 64 };
  strictEqual(assessFit(20, hardware), "gpu");
  strictEqual(assessFit(27, hardware), "ram");
  strictEqual(assessFit(35, hardware), "ssd");
  strictEqual(assessFit(65, hardware), "too-large");
  strictEqual(assessFit(35, { ...hardware, ssd: undefined }), "disk-unknown");
  strictEqual(assessFit(20, { ...hardware, vram: undefined }), "unknown");
  strictEqual(assessFit(20, { ...hardware, ssd: 19 }), "too-large");
  strictEqual(assessFit(0, hardware), "unknown");
});

function job(overrides: Partial<HubDownloadJob>): HubDownloadJob {
  return { id: 1, repo: "org/name", revision: "main", state: "downloading", ...overrides };
}

test("download status lines carry the counters the GUI polls", () => {
  strictEqual(statusOf(job({ files_done: 0, files_total: 2, bytes: 0, bytes_total: 10 })), "0/2 files, 0 B of 10 B");
  strictEqual(statusOf(job({ state: "done" })), "done");
  strictEqual(statusOf(job({ state: "cancelled" })), "cancelled");
  strictEqual(
    statusOf(job({ state: "failed", detail: "hub returned 429" })),
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

test("an unavailable download size does not display a zero-byte total", () => {
  strictEqual(statusOf(job({ bytes: 1024 })), "1 KiB received");
  strictEqual(statusOf(job({ current_file: "config.json" })), "Preparing download — config.json");
});

test("a stop says when the card memory stays held until the server restarts", () => {
  strictEqual(stoppedLine("e1", { memory_released: true }), "Engine e1 stopped.");
  strictEqual(
    stoppedLine("q36", {
      memory_released: false,
      note: "stopped; card memory returns when the server process restarts",
    }),
    "Engine q36 stopped; card memory returns when the server process restarts",
  );
  strictEqual(stopQuestion("e1", false), "Stop engine e1? Running jobs on it are cancelled.");
  strictEqual(stopQuestion("q36", true).includes("not returned until the server process restarts"), true);
});

test("the page offers Serve only for a complete snapshot of a known family", () => {
  strictEqual(
    snapshotStatus({ complete: true, servable: true, families: ["qwen3_6_35b_a3b"] }),
    "serves as qwen3_6_35b_a3b",
  );
  strictEqual(
    snapshotStatus({ complete: false, servable: false, families: [] }).startsWith("incomplete"),
    true,
  );
  strictEqual(
    snapshotStatus({ complete: true, servable: false, families: [] }),
    "no family of this build recognises it",
  );
});
