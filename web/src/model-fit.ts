import type { HubTree } from "./api.js";

type HubFile = HubTree["files"][number];

export interface Variant {
  name: string;
  bytes: number;
  files: string[];
}

export function ggufVariants(files: HubFile[]): Variant[] {
  const groups = new Map<string, { name: string; count: number; parts: Map<number, HubFile> }>();
  for (const file of files) {
    if (!/\.gguf$/i.test(file.path) || /(^|\/)mmproj/i.test(file.path) || !(file.size > 0)) continue;
    const stem = file.path.slice(0, -5);
    const split = /^(.*)-(\d{5})-of-(\d{5})$/.exec(stem);
    const name = split?.[1] ?? stem;
    const part = split ? Number(split[2]) : 1;
    const count = split ? Number(split[3]) : 1;
    if (part < 1 || part > count) continue;
    const key = `${name}.gguf:${count}`;
    const group = groups.get(key) ?? { name, count, parts: new Map<number, HubFile>() };
    group.parts.set(part, file);
    groups.set(key, group);
  }
  const variants: Variant[] = [];
  for (const group of groups.values()) {
    if (group.parts.size !== group.count) continue;
    const parts = [...group.parts.values()].sort((a, b) => a.path.localeCompare(b.path));
    variants.push({
      name: group.name, bytes: parts.reduce((total, file) => total + file.size, 0),
      files: parts.map((file) => file.path),
    });
  }
  return variants.sort((a, b) => a.bytes - b.bytes);
}

export interface FitBudget {
  vram?: number;
  ram?: number;
  ssd?: number;
}

export type Fit = "gpu" | "ram" | "ssd" | "disk-unknown" | "too-large" | "unknown";

export function assessFit(bytes: number, budget: FitBudget): Fit {
  if (!(bytes > 0)) return "unknown";
  if (budget.ssd !== undefined && bytes > budget.ssd) return "too-large";
  if (!budget.vram) return "unknown";
  // shortcut: reserve 15% as a screening margin; only the engine can admit the working set and KV cache.
  if (bytes <= budget.vram * 0.85) return "gpu";
  if (budget.ram === undefined) return "unknown";
  if (bytes <= budget.ram * 0.85) return "ram";
  return budget.ssd === undefined ? "disk-unknown" : "ssd";
}
