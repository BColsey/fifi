/**
 * Persistence for the monitor's sparkline history: the last hour of
 * decode and first-token medians, so a reload does not start the trend
 * empty. Pure functions over the stored JSON; the view owns storage.
 * DOM-free so the node:test suite runs it.
 */

export interface SparkPoint {
  t: number;
  v: number | null;
}

export interface SparkHistory {
  decode: SparkPoint[];
  ttft: SparkPoint[];
}

export const HISTORY_WINDOW_MS = 60 * 60 * 1000;
export const HISTORY_CAP = 360;
export const HISTORY_KEY = "fifi-monitor-history-v1";

/** Drop points older than the window and beyond the cap (newest kept). */
export function prune(points: readonly SparkPoint[], now: number): SparkPoint[] {
  return points.filter((p) => now - p.t <= HISTORY_WINDOW_MS).slice(-HISTORY_CAP);
}

export function appendPoint(points: readonly SparkPoint[], t: number, v: number | null): SparkPoint[] {
  return prune([...points, { t, v }], t);
}

export function serialize(history: SparkHistory): string {
  return JSON.stringify(history);
}

/** Parse stored history; any corrupt shape yields empty history. */
export function parseHistory(raw: string | null, now: number): SparkHistory {
  if (raw === null) return { decode: [], ttft: [] };
  try {
    const data: unknown = JSON.parse(raw);
    if (typeof data !== "object" || data === null) return { decode: [], ttft: [] };
    const read = (value: unknown): SparkPoint[] => {
      if (!Array.isArray(value)) return [];
      const points: SparkPoint[] = [];
      for (const item of value) {
        if (typeof item !== "object" || item === null) continue;
        const t = (item as { t?: unknown }).t;
        const v = (item as { v?: unknown }).v;
        if (typeof t !== "number") continue;
        points.push({ t, v: typeof v === "number" ? v : null });
      }
      return prune(points, now);
    };
    const obj = data as { decode?: unknown; ttft?: unknown };
    return { decode: read(obj.decode), ttft: read(obj.ttft) };
  } catch {
    return { decode: [], ttft: [] };
  }
}

export function values(points: readonly SparkPoint[]): (number | null)[] {
  return points.map((p) => p.v);
}
