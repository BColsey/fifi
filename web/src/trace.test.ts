import { test } from "node:test";
import { deepStrictEqual, strictEqual } from "node:assert";
import { fmtProb, mapOffset, type FieldAt } from "./trace.js";

const segs = [
  { field: "reasoning", bytes: 4 },
  { field: "content", bytes: 6 },
  { field: "tool", bytes: 3 },
] as const;

test("mapOffset walks segments in order", () => {
  deepStrictEqual(mapOffset(segs, 0), { field: "reasoning", offsetInField: 0 });
  deepStrictEqual(mapOffset(segs, 3), { field: "reasoning", offsetInField: 3 });
  deepStrictEqual(mapOffset(segs, 4), { field: "content", offsetInField: 0 });
  deepStrictEqual(mapOffset(segs, 9), { field: "content", offsetInField: 5 });
  deepStrictEqual(mapOffset(segs, 10), { field: "tool", offsetInField: 0 });
});

test("mapOffset is null past the kept text and for bad input", () => {
  strictEqual(mapOffset(segs, 13), null);
  strictEqual(mapOffset(segs, 999), null);
  strictEqual(mapOffset(undefined, 0), null);
  strictEqual(mapOffset([], 0), null);
  strictEqual(mapOffset(segs, -1), null);
});

test("mapOffset maps each field's first byte at its boundary", () => {
  const boundaries: FieldAt[] = [];
  for (const at of [4, 10]) boundaries.push(mapOffset(segs, at) as FieldAt);
  deepStrictEqual(boundaries, [
    { field: "content", offsetInField: 0 },
    { field: "tool", offsetInField: 0 },
  ]);
});

test("fmtProb renders one decimal", () => {
  strictEqual(fmtProb(0), "0.0%");
  strictEqual(fmtProb(0.41234), "41.2%");
  strictEqual(fmtProb(1), "100.0%");
});
