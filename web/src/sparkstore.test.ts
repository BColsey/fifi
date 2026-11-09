import { test } from "node:test";
import { deepStrictEqual, strictEqual } from "node:assert";
import { appendPoint, HISTORY_CAP, HISTORY_WINDOW_MS, parseHistory, prune, serialize } from "./sparkstore.js";

test("prune drops points older than the window and caps the count", () => {
  const now = HISTORY_WINDOW_MS + 1000;
  const kept = prune([{ t: 0, v: 1 }, { t: now - 10, v: 2 }], now);
  deepStrictEqual(kept, [{ t: now - 10, v: 2 }]);
  const many = Array.from({ length: HISTORY_CAP + 5 }, (_, i) => ({ t: now, v: i }));
  strictEqual(prune(many, now).length, HISTORY_CAP);
});

test("history round-trips through storage JSON", () => {
  const history = { decode: [{ t: 100, v: 12.5 }], ttft: [{ t: 100, v: null }] };
  deepStrictEqual(parseHistory(serialize(history), 200), history);
});

test("corrupt storage yields empty history, never a throw", () => {
  deepStrictEqual(parseHistory("not json", 0), { decode: [], ttft: [] });
  deepStrictEqual(parseHistory('{"decode": "nope"}', 0), { decode: [], ttft: [] });
  deepStrictEqual(parseHistory(null, 0), { decode: [], ttft: [] });
});

test("append adds and prunes in one step", () => {
  const out = appendPoint([{ t: 0, v: 1 }], HISTORY_WINDOW_MS + 5, 9);
  deepStrictEqual(out, [{ t: HISTORY_WINDOW_MS + 5, v: 9 }]);
});
