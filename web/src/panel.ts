/**
 * The speed panel's DOM-free state. MonitorView renders whatever these
 * return.
 */

import { ApiError, listRecentRequests, type Engine, type Model, type RecentRequest, type SystemInfo } from "./api.js";

/** The full panel state the view renders (re-exported as MonitorData). */
export interface PanelData {
  healthOk: boolean;
  readyOk: boolean;
  notReadyCode: string | null;
  models: Model[] | null;
  modelsError: string | null;
  system: SystemInfo | null;
  systemError: string | null;
  recent: RecentRequest[] | null;
  recentError: string | null;
  checkedAt: number;
}

export interface PanelState {
  recent: RecentRequest[] | null;
  recentError: string | null;
  checkedAt: number;
}

/** Injected seams so the tests need no global fetch mock and no wall
 * clock: the ring fetch, the clock, and the abort timer. */
export interface RecentDeps {
  fetch?: (signal: AbortSignal) => Promise<{ requests: RecentRequest[] }>;
  now?: () => number;
  timeout?: (ms: number) => AbortSignal;
}

/** One refresh of the recent rows. A refused, offline, or hung route
 * leaves `recent` null and names the cause — never the last good rows,
 * which is what made the panel look alive while the engine was down. */
export async function refreshRecentRows(timeoutMs = 4000, deps: RecentDeps = {}): Promise<PanelState> {
  const fetchRing = deps.fetch ?? ((signal) => listRecentRequests(signal));
  const now = deps.now ?? Date.now;
  const timeout = deps.timeout ?? AbortSignal.timeout;
  let recent: RecentRequest[] | null = null;
  let recentError: string | null = null;
  try {
    recent = (await fetchRing(timeout(timeoutMs))).requests;
  } catch (err) {
    recentError = err instanceof ApiError ? err.code : "network";
  }
  return { recent, recentError, checkedAt: now() };
}

/** Null `recent` rows mean the server was unreachable — the sweep
 *  never fetches the ring from a dead process. */
export interface ProbeOutcome {
  healthOk: boolean;
  readyOk: boolean;
  notReadyCode: string | null;
  models: Model[] | null;
  modelsError: string | null;
  system: SystemInfo | null;
  systemError: string | null;
  recent: RecentRequest[] | null;
  recentError: string | null;
}

/** The merge: a probe FULLY REPLACES the panel data and stamps
 * one fresh time. Nothing survives from the previous state — that is the
 * invariant; the poll may only call the full sweep, so a dead engine can
 * never show a last-good "runs"/"ready" next to a current timestamp. */
export function applyProbe(probe: ProbeOutcome): PanelData {
  return { ...probe, checkedAt: Date.now() };
}

/** The poll's per-tick decision: which refresh runs. The last
 * known server state is an input ONLY so the decision owns it — the
 * correct answer consults it for nothing, because the full sweep probes
 * health itself. Gating the monitor refresh on readiness was a
 * bug: the view froze when the engine was down or still starting. */
export function pollPlan(opts: {
  monitorVisible: boolean;
  lastKnownServerReady: boolean;
}): { refreshMonitor: boolean } {
  void opts.lastKnownServerReady;
  return { refreshMonitor: opts.monitorVisible };
}

/** The failure lines for one engine, formatted from the backend's
 * structured error (api `EngineError`). Formatting only: every fact —
 * code, stage, retryability, suggestion — is the server's; nothing is
 * recomputed or reinterpreted here. */
export function engineErrorLines(engine: Engine): string[] {
  const error = engine.error;
  if (error === undefined) return [];
  const retry =
    error.retry === "after_correction"
      ? "fix the cause, then restart"
      : "retrying is not known to help";
  const lines = [
    "failed (" + error.code + ", stage " + error.stage + "): " + error.message,
    retry,
  ];
  if (error.suggestion !== undefined && error.suggestion !== "") {
    lines.push(error.suggestion);
  }
  return lines;
}

/** The loading line for one engine: the loader's own stage name, shown
 * verbatim while the engine loads. A stage that stops changing reads
 * as stalled — that is the point of naming stages instead of guessing
 * percentages. Null for every non-loading engine. */
export function engineStageLine(engine: Engine): string | null {
  if (engine.state !== "loading" || engine.load_stage === undefined) return null;
  return "loading: " + engine.load_stage;
}

/** The system view's engine entry for one model-list engine id, or null
 * when the sweep has no system answer. Matching by id is the whole
 * join: the two routes describe the same engines. */
export function systemEngineFor(system: SystemInfo | null, id: string): SystemEngineFor | null {
  if (system === null) return null;
  return system.engines.find((engine) => engine.id === id) ?? null;
}

export type SystemEngineFor = SystemInfo["engines"][number];

/** One text line per retained setting row: key, value exactly as the
 * resolver rendered it, and the origin that decided it. Formatting
 * only — every fact is the server's. */
export function settingsLines(engine: SystemEngineFor): string[] {
  return engine.settings.map(
    (row) => row.key + ": " + row.value + " (" + row.origin + ")"
  );
}

export interface RequestTotals {
  total: number;
  ok: number;
  errors: number;
  cancelled: number;
  promptTokens: number;
  completionTokens: number;
  reusedTokens: number;
  /** Reused share of prompt tokens, 0..1; null with no prompt tokens. */
  reuseShare: number | null;
}

/** Sums over the retained ring rows. Counts only; nothing is invented. */
export function requestTotals(rows: readonly RecentRequest[]): RequestTotals {
  let ok = 0;
  let errors = 0;
  let cancelled = 0;
  let promptTokens = 0;
  let completionTokens = 0;
  let reusedTokens = 0;
  for (const row of rows) {
    if (row.outcome === "ok") ok++;
    else if (row.outcome === "error") errors++;
    else cancelled++;
    promptTokens += row.prompt_tokens;
    completionTokens += row.completion_tokens;
    reusedTokens += row.reused_tokens;
  }
  return {
    total: rows.length,
    ok,
    errors,
    cancelled,
    promptTokens,
    completionTokens,
    reusedTokens,
    reuseShare: promptTokens > 0 ? reusedTokens / promptTokens : null,
  };
}

/** Prefix-cache hit rate 0..1, or null before any lookup is recorded. */
export function prefixHitRate(cache: { prefix_hits: number; prefix_misses: number }): number | null {
  const total = cache.prefix_hits + cache.prefix_misses;
  return total > 0 ? cache.prefix_hits / total : null;
}

/** An SVG line path for a sparkline in a 100x32 viewBox, Strata-style.
 * Fewer than two points draws nothing. Nulls count as zero. */
export function sparkPath(values: readonly (number | null)[], max?: number): string {
  if (values.length < 2) return "";
  const nums = values.map((v) => (v === null || !Number.isFinite(v) ? 0 : Math.max(0, v)));
  const top = Math.max(max ?? 0, ...nums, 1e-9);
  return nums
    .map((v, i) => {
      const x = (i / (nums.length - 1)) * 100;
      const y = 30 - (v / top) * 26;
      return (i === 0 ? "M" : "L") + x.toFixed(2) + "," + y.toFixed(2);
    })
    .join("");
}

/** The filled area under sparkPath, closed along the bottom edge. */
export function sparkArea(values: readonly (number | null)[], max?: number): string {
  const line = sparkPath(values, max);
  return line === "" ? "" : line + "L100,32L0,32Z";
}
