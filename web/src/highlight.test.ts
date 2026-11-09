import { test } from "node:test";
import { deepStrictEqual, strictEqual } from "node:assert";
import { highlightable, tokenize } from "./highlight.js";

test("keywords, strings, comments, and numbers split from plain code", () => {
  const spans = tokenize('const s = "hi"; // note\nreturn 42;');
  const kinds = spans.map((s) => s.kind + ":" + s.text);
  strictEqual(kinds.some((k) => k === "keyword:const"), true);
  strictEqual(kinds.some((k) => k === 'string:"hi"'), true);
  strictEqual(kinds.some((k) => k === "comment:// note"), true);
  strictEqual(kinds.some((k) => k === "number:42"), true);
  strictEqual(spans.map((s) => s.text).join(""), 'const s = "hi"; // note\nreturn 42;', "spans reassemble the code");
});

test("comment markers inside strings stay strings", () => {
  const spans = tokenize('let u = "http://x";');
  strictEqual(spans.some((s) => s.kind === "comment"), false);
  strictEqual(spans.some((s) => s.kind === "string"), true);
});

test("unterminated strings and block comments run to the end", () => {
  deepStrictEqual(tokenize("/* open").map((s) => s.kind), ["comment"]);
  strictEqual(tokenize('let s = "open').map((s) => s.text).join(""), 'let s = "open');
});

test("plain languages are not highlighted", () => {
  strictEqual(highlightable("ts"), true);
  strictEqual(highlightable("text"), false);
  strictEqual(highlightable(""), false);
});
