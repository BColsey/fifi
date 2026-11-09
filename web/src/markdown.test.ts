import { test } from "node:test";
import { deepStrictEqual, strictEqual } from "node:assert";
import { parseBlocks, parseInline } from "./markdown.js";

test("a fenced block becomes one code block with its language", () => {
  const blocks = parseBlocks("before\n\n```ts\nconst x = 1;\n```\n\nafter");
  strictEqual(blocks.length, 3);
  deepStrictEqual(blocks[1], { kind: "code", lang: "ts", code: "const x = 1;" });
});

test("an unterminated fence is still code while streaming", () => {
  const blocks = parseBlocks("```py\nprint(1)");
  deepStrictEqual(blocks, [{ kind: "code", lang: "py", code: "print(1)" }]);
});

test("markup inside code stays literal text for the renderer", () => {
  const blocks = parseBlocks("```\n<script>alert(1)</script>\n```");
  deepStrictEqual(blocks, [{ kind: "code", lang: "", code: "<script>alert(1)</script>" }]);
});

test("headings, lists, quotes, rules and tables parse as blocks", () => {
  const blocks = parseBlocks(
    "# Title\n\n- one\n- two\n\n> quoted\n\n---\n\n| A | B |\n| --- | --- |\n| 1 | 2 |"
  );
  deepStrictEqual(
    blocks.map((b) => b.kind),
    ["heading", "list", "quote", "rule", "table"]
  );
  const table = blocks[4]!;
  if (table.kind === "table") {
    deepStrictEqual(table.headers, ["A", "B"]);
    deepStrictEqual(table.rows, [["1", "2"]]);
  }
});

test("inline code shields its contents from emphasis", () => {
  deepStrictEqual(parseInline("use `**not bold**` here"), [
    { kind: "text", text: "use " },
    { kind: "code", text: "**not bold**" },
    { kind: "text", text: " here" },
  ]);
});

test("bold, italic and https links parse; other links stay literal", () => {
  const spans = parseInline("**b** *i* [ok](https://example.com) [no](javascript:alert(1))");
  strictEqual(spans.some((s) => s.kind === "bold"), true);
  strictEqual(spans.some((s) => s.kind === "italic"), true);
  strictEqual(spans.some((s) => s.kind === "link"), true);
  strictEqual(
    spans.filter((s) => s.kind === "text").map((s) => (s.kind === "text" ? s.text : "")).join("").includes("javascript:"),
    true,
    "a non-http link never becomes an anchor"
  );
});

test("plain text with no markers is one text span", () => {
  deepStrictEqual(parseInline("hello world"), [{ kind: "text", text: "hello world" }]);
});
