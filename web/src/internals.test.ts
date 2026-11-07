import { test } from "node:test";
import { deepStrictEqual, strictEqual } from "node:assert";
import {
  cellColor,
  heatRow,
  tokenColumn,
  topExperts,
  totalPicks,
  type RoutingTracePacket,
} from "./internals.js";

function packet(rows: { layer: number; ids: number[][]; gates: number[][] }[]): RoutingTracePacket {
  return {
    packet: "router_traces",
    family: "test",
    checkpoint: "/m/test",
    activation_dtype: "f32",
    kv_dtype: "f32",
    routing: "Host",
    step_mode: "Record",
    path_note: "test",
    layer_rows: rows,
    summary: {
      tokens: rows[0]?.ids.length ?? 0,
      layers: rows.length,
      top_k: rows[0]?.ids[0]?.length ?? 0,
      experts: 4,
      experts_seen: 3,
      expert_histogram: {},
    },
  };
}

test("topExperts ranks descending, breaks ties by id, and honors the limit", () => {
  const ranked = topExperts({ "3": 5, "1": 9, "2": 9, "7": 1 }, 3);
  deepStrictEqual(
    ranked.map((e) => e.id),
    [1, 2, 3]
  );
  strictEqual(ranked[0]?.picks, 9);
  strictEqual(topExperts({}, 5).length, 0);
});

test("totalPicks sums every histogram entry", () => {
  strictEqual(totalPicks({ "0": 40, "5": 2 }), 42);
  strictEqual(totalPicks({}), 0);
});

test("heatRow takes each token's first-choice expert and its gate", () => {
  const cells = heatRow({
    layer: 0,
    ids: [
      [7, 1],
      [2, 9],
    ],
    gates: [
      [0.5, 0.2],
      [0.9, 0.1],
    ],
  });
  deepStrictEqual(cells, [
    { topId: 7, topGate: 0.5 },
    { topId: 2, topGate: 0.9 },
  ]);
});

test("cellColor is deterministic and clamps id and gate to the expert count", () => {
  const a = cellColor(10, 0.5, 100);
  strictEqual(a, cellColor(10, 0.5, 100));
  strictEqual(a, "hsl(36.0 62% 58.0%)");
  // gate above 1 and id above experts-1 clamp, they do not throw.
  strictEqual(cellColor(500, 5, 100), cellColor(99, 1, 100));
});

test("tokenColumn returns every layer's full pick list in choice order", () => {
  const col = tokenColumn(
    packet([
      {
        layer: 0,
        ids: [[4, 0]],
        gates: [[0.75, 0.25]],
      },
      {
        layer: 1,
        ids: [[3, 1]],
        gates: [[0.6, 0.4]],
      },
    ]),
    0
  );
  deepStrictEqual(col, [
    { layer: 0, picks: [{ id: 4, gate: 0.75 }, { id: 0, gate: 0.25 }] },
    { layer: 1, picks: [{ id: 3, gate: 0.6 }, { id: 1, gate: 0.4 }] },
  ]);
});

test("an out-of-range token index yields empty pick lists, never a throw", () => {
  const col = tokenColumn(
    packet([{ layer: 0, ids: [[1]], gates: [[1.0]] }]),
    9
  );
  deepStrictEqual(col, [{ layer: 0, picks: [] }]);
});
