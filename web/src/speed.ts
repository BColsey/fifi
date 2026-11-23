/**
 * Medians over finished-ok rows only: the panel shows what a healthy
 * request delivered. DOM-free so the node:test suite runs it.
 */

export interface SpeedRow {
  outcome: "ok" | "error" | "cancelled";
  decode_tokens_per_second: number | null;
  time_to_first_token_ms: number | null;
  prompt_tokens?: number;
  reused_tokens?: number;
  prefill_ms?: number | null;
}

export interface SpeedSummary {
  decodePerSecond: number | null;
  firstTokenMs: number | null;
  /** Median measured prefill rate: (prompt − reused) over prefill_ms. */
  prefillPerSecond: number | null;
  okCount: number;
}

/** CPU utilisation in percent between two /proc/stat aggregate
 * samples: busy jiffies over total jiffies, with idle (field 3) and
 * iowait (field 4) counted idle. Null when either sample is missing
 * or too short, or when the counter went backwards. */
export function cpuUtilization(
  before: number[] | null | undefined,
  after: number[] | null | undefined
): number | null {
  if (!before || !after || before.length < 4 || before.length !== after.length) return null;
  let total = 0;
  let idle = 0;
  for (const [i, time] of after.slice(0, 8).entries()) {
    const delta = time - (before[i] ?? NaN);
    // Linux iowait can decrease even when the aggregate idle time increases.
    if (!Number.isFinite(delta) || delta < 0) return null;
    total += delta;
    if (i === 3 || i === 4) idle += delta;
  }
  return total > 0 ? (total - idle) * 100 / total : null;
}

/** Bytes per second between two cumulative counters over `dtMs`.
 * Null when the elapsed time or the delta is not a usable rate. */
export function byteRate(before: number, after: number, dtMs: number): number | null {
  if (!(dtMs > 0 && after >= before)) return null;
  const rate = (after - before) / (dtMs / 1000);
  return Number.isFinite(rate) ? rate : null;
}

/** The lower of the two middle values on even counts: dumb and stable. */
function median(values: number[]): number | null {
  if (values.length === 0) return null;
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor((sorted.length - 1) / 2)] ?? null;
}

/** The measured prefill rate of one row: the tokens the engine actually
 * read (prompt minus the prefix it reused) over the prefill span. */
function prefillRate(row: SpeedRow): number | null {
  const prefill = row.prefill_ms;
  const prompt = row.prompt_tokens;
  const reused = row.reused_tokens ?? 0;
  if (prefill === null || prefill === undefined || prefill <= 0 || prompt === undefined) {
    return null;
  }
  return (prompt - reused) / (prefill / 1000);
}

export function speedSummary(rows: SpeedRow[]): SpeedSummary {
  const ok = rows.filter((row) => row.outcome === "ok");
  return {
    decodePerSecond: median(
      ok
        .map((row) => row.decode_tokens_per_second)
        .filter((value): value is number => value !== null)
    ),
    firstTokenMs: median(
      ok
        .map((row) => row.time_to_first_token_ms)
        .filter((value): value is number => value !== null)
    ),
    prefillPerSecond: median(
      ok
        .map(prefillRate)
        .filter((value): value is number => value !== null && value > 0)
    ),
    okCount: ok.length,
  };
}

export interface GroupRow extends SpeedRow {
  model?: string;
  engine?: string | null;
  session?: string | null;
  prompt_tokens: number;
  reused_tokens: number;
  queue_wait_ms?: number | null;
  prefill_ms?: number | null;
}

export interface GroupSummary extends SpeedSummary {
  key: string;
  total: number;
  /** Median measured prefill rate: (prompt − reused) over prefill_ms. */
  prefillPerSecond: number | null;
  /** Median queue wait over finished rows that reported one. */
  queueWaitMs: number | null;
}

/** Per-key medians over the ring: one group per model or per engine. */
export function groupSpeed(rows: readonly GroupRow[], keyOf: (row: GroupRow) => string): GroupSummary[] {
  const groups = new Map<string, GroupRow[]>();
  for (const row of rows) {
    const key = keyOf(row);
    const list = groups.get(key);
    if (list === undefined) groups.set(key, [row]);
    else list.push(row);
  }
  return [...groups.entries()]
    .map(([key, list]) => {
      const base = speedSummary(list);
      return {
        key,
        total: list.length,
        ...base,
        queueWaitMs: median(
          list
            .map((row) => row.queue_wait_ms ?? null)
            .filter((value): value is number => value !== null)
        ),
      };
    })
    .sort((a, b) => b.total - a.total || a.key.localeCompare(b.key));
}
