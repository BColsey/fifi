import { test } from "node:test";
import { strictEqual } from "node:assert";
import { getSamplerTrace } from "./api.js";

test("trace lookup identifies the reply model without changing its id", async () => {
  const original = globalThis.fetch;
  let target = "";
  let model: string | null = null;
  globalThis.fetch = async (input, init) => {
    target = String(input);
    model = new Headers(init?.headers).get("X-Fifi-Trace-Model");
    return new Response(JSON.stringify({ reply_id: "same id", model }), { status: 200 });
  };
  try {
    await getSamplerTrace("same id", undefined, "fifi-q36-2080ti");
    strictEqual(target, "/v1/sampler-trace?id=same%20id");
    strictEqual(model, "fifi-q36-2080ti");
    await getSamplerTrace("old-reply");
    strictEqual(model, null);
  } finally {
    globalThis.fetch = original;
  }
});
