import { test } from "node:test";
import { deepStrictEqual, ok, strictEqual } from "node:assert";
import {
  applyProbe,
  engineErrorLines,
  engineStageLine,
  pollPlan,
  refreshRecentRows,
  settingsLines,
  systemEngineFor,
} from "./panel.js";
import type { Engine, RecentRequest, SystemInfo } from "./api.js";

function row(tokens: number): RecentRequest {
  return {
    model: "m",
    stream: false,
    status: 200,
    outcome: "ok",
    error_code: null,
    prompt_tokens: 10,
    completion_tokens: tokens,
    reused_tokens: 0,
    time_to_first_token_ms: 100,
    decode_tokens_per_second: tokens,
  };
}

const answers: (signal: AbortSignal) => Promise<{ requests: RecentRequest[] }> = (
  signal: AbortSignal
) => {
  ok(signal.aborted === false, "the fetch sees a live signal first");
  return Promise.resolve({ requests: [row(2)] });
};

const rejects = () => Promise.reject(new TypeError("offline"));

const frozenClock = () => 12345;

test("an engine that fails or is not ready does not freeze the panel: the ring row is still fetched", async () => {
  const next = await refreshRecentRows(4000, { fetch: answers, now: frozenClock });
  strictEqual(next.recentError, null, "the recent route is engine-independent");
  ok(
    next.recent !== null && next.recent.length === 1 && next.recent[0]!.decode_tokens_per_second === 2,
    "the freshest ring row landed"
  );
  strictEqual(next.checkedAt, 12345, "the check is stamped from the injected clock");
});

test("an offline server turns the card unavailable, not stale-but-alive", async () => {
  const next = await refreshRecentRows(4000, { fetch: rejects, now: frozenClock });
  strictEqual(next.recent, null, "no last-good rows are re-shown as current");
  strictEqual(next.recentError, "network");
});

test("recovery: when the route answers again the rows return", async () => {
  const offline = await refreshRecentRows(4000, { fetch: rejects, now: frozenClock });
  strictEqual(offline.recent, null);
  const back = await refreshRecentRows(4000, { fetch: answers, now: frozenClock });
  strictEqual(back.recentError, null);
  ok(back.recent !== null && back.recent[0]!.decode_tokens_per_second === 2);
});

test("a hung route is cut by the abort and reported as unreachable", async () => {
  // No real timer: the test aborts the injected signal itself, which is
  // the only thing that can end the hung fetch.
  const ctrl = new AbortController();
  const hung = (signal: AbortSignal) =>
    new Promise<{ requests: RecentRequest[] }>((_resolve, reject) => {
      signal.addEventListener("abort", () =>
        reject(new DOMException("The operation was aborted.", "AbortError"))
      );
    });
  const pending = refreshRecentRows(50, {
    fetch: hung,
    now: frozenClock,
    timeout: () => ctrl.signal,
  });
  ctrl.abort();
  const next = await pending;
  strictEqual(next.recent, null);
  strictEqual(next.recentError, "network");
});

test("a failed probe fully replaces the panel: no last-good ok survives a fresh timestamp", () => {
  const wasHealthy = applyProbe({
    healthOk: true,
    readyOk: true,
    notReadyCode: null,
    models: null,
    modelsError: null,
    system: null,
    systemError: null,
    recent: [row(1)],
    recentError: null,
  });
  ok(wasHealthy.healthOk && wasHealthy.readyOk && wasHealthy.recent !== null);
  const now = applyProbe({
    healthOk: false,
    readyOk: false,
    notReadyCode: null,
    models: null,
    modelsError: null,
    system: null,
    systemError: null,
    recent: null,
    recentError: null,
  });
  strictEqual(now.healthOk, false, "health is down, not the last good value");
  strictEqual(now.readyOk, false, "readiness is down, not the last good value");
  strictEqual(now.recent, null, "the ring card goes unavailable");
  strictEqual(now.recentError, null);
  ok(now.checkedAt >= wasHealthy.checkedAt, "the timestamp still advances");
  // The full-replacement invariant in one sweep of every display field: nothing that
  // reads as "ok" outlives the probe that found otherwise.
  const okish = [now.healthOk, now.readyOk, now.models, now.recent].every(
    (field) => field === false || field === null
  );
  ok(okish, "no field keeps a stale ok alongside a fresh time");
});

test("the poll refreshes the monitor whenever the view is visible — including offline (the round-1 regression)", () => {
  // An earlier version gated the refresh on the last known server state
  // (serverReady), so the visible-offline row below refreshed nothing and
  // the panel froze on its last good values. Pin every row.
  const table: Array<{
    monitorVisible: boolean;
    lastKnownServerReady: boolean;
    refreshMonitor: boolean;
  }> = [
    { monitorVisible: true, lastKnownServerReady: true, refreshMonitor: true },
    { monitorVisible: true, lastKnownServerReady: false, refreshMonitor: true },
    { monitorVisible: false, lastKnownServerReady: true, refreshMonitor: false },
    { monitorVisible: false, lastKnownServerReady: false, refreshMonitor: false },
  ];
  for (const row of table) {
    strictEqual(
      pollPlan(row).refreshMonitor,
      row.refreshMonitor,
      `visible=${row.monitorVisible} ready=${row.lastKnownServerReady}`
    );
  }
});

function failedEngine(): Engine {
  return {
    id: "e1",
    model: "m",
    state: "error",
    error: {
      code: "checkpoint_failure",
      stage: "load",
      message: "engine load failed: fixture load failure",
      retry: "after_correction",
      suggestion: "verify the model files",
    },
  };
}

test("engine error lines carry the backend structure verbatim", () => {
  const lines = engineErrorLines(failedEngine());
  strictEqual(lines.length, 3);
  const joined = lines.join("\n");
  ok(joined.includes("checkpoint_failure"), joined);
  ok(joined.includes("stage load"), joined);
  ok(joined.includes("fixture load failure"), joined);
  ok(joined.includes("fix the cause"), joined);
  ok(lines.includes("verify the model files"));
});

test("healthy engines and unknown retry render honestly", () => {
  strictEqual(engineErrorLines({ id: "e", model: "m", state: "ready" }).length, 0);
  const backend: Engine = {
    id: "e",
    model: "m",
    state: "native-failed",
    error: { code: "backend_error", stage: "cuda initialize", message: "x", retry: "unknown" },
  };
  const lines = engineErrorLines(backend);
  strictEqual(lines.length, 2);
  ok(lines.some((line) => line.includes("not known to help")));
});

test("loading engines show the loader's own stage, others show none", () => {
  strictEqual(
    engineStageLine({ id: "e", model: "m", state: "loading", load_stage: "loading weights" }),
    "loading: loading weights"
  );
  // A loading engine before its first stage note has no line — the
  // state chip already says loading; no invented stage.
  strictEqual(engineStageLine({ id: "e", model: "m", state: "loading" }), null);
  // A stage left over on a non-loading engine must never render.
  strictEqual(
    engineStageLine({ id: "e", model: "m", state: "ready", load_stage: "ready" }),
    null
  );
});

test("settings lines carry the resolver's facts verbatim", () => {
  const system: SystemInfo = {
    version: "0.1.0",
    backends_compiled: ["cuda"],
    host: {
      cpu_threads: 24,
      mem_total_bytes: 1,
      mem_available_bytes: { unavailable: "why" },
    },
    engines: [
      {
        id: "q36",
        model: "m",
        state: "ready",
        context_limit: 32768,
        settings: [
          { key: "kv_dtype", value: "q8", origin: "config_file" },
          { key: "context_limit", value: "32768", origin: "default" },
        ],
      },
    ],
  };
  strictEqual(systemEngineFor(null, "q36"), null);
  strictEqual(systemEngineFor(system, "other"), null);
  const engine = systemEngineFor(system, "q36");
  if (engine === null) throw new Error("q36 must be in the system view");
  deepStrictEqual(settingsLines(engine), [
    "kv_dtype: q8 (config_file)",
    "context_limit: 32768 (default)",
  ]);
  // Engines of families that compose no rows show an empty list, not
  // an invented one.
  const shallow: SystemInfo = {
    ...system,
    engines: [{ id: "f", model: "m", state: "ready", context_limit: 1, settings: [] }],
  };
  const flint = systemEngineFor(shallow, "f");
  if (flint === null) throw new Error("f must be in the system view");
  deepStrictEqual(settingsLines(flint), []);
});
