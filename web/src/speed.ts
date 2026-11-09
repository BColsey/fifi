/**
 * Medians over finished-ok rows only: the panel shows what a healthy
 * request delivered. DOM-free so the node:test suite runs it.
 */

export interface SpeedRow {
  outcome: "ok" | "error" | "cancelled";
  decode_tokens_per_second: number | null;
  time_to_first_token_ms: number | null;
}

export interface SpeedSummary {
  decodePerSecond: number | null;
  firstTokenMs: number | null;
  okCount: number;
}

/** The lower of the two middle values on even counts: dumb and stable. */
function median(values: number[]): number | null {
  if (values.length === 0) return null;
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor((sorted.length - 1) / 2)] ?? null;
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
      const ok = list.filter((row) => row.outcome === "ok");
      const prefillRates = ok
        .map((row) =>
          row.prefill_ms !== null && row.prefill_ms !== undefined && row.prefill_ms > 0
            ? (row.prompt_tokens - row.reused_tokens) / (row.prefill_ms / 1000)
            : null
        )
        .filter((value): value is number => value !== null && value > 0);
      return {
        key,
        total: list.length,
        ...base,
        prefillPerSecond: median(prefillRates),
        queueWaitMs: median(
          list
            .map((row) => row.queue_wait_ms ?? null)
            .filter((value): value is number => value !== null)
        ),
      };
    })
    .sort((a, b) => b.total - a.total || a.key.localeCompare(b.key));
}
