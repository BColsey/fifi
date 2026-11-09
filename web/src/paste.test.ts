import { test } from "node:test";
import { strictEqual } from "node:assert";
import {
  composeOutgoing,
  isLargePaste,
  isLongMessage,
  pasteSummary,
} from "./paste.js";

test("short pastes stay in the textarea; long ones attach", () => {
  strictEqual(isLargePaste("hello"), false);
  strictEqual(isLargePaste("x".repeat(4001)), true, "over the char limit attaches");
  strictEqual(isLargePaste(Array(31).fill("line").join("\n")), true, "over the line limit attaches");
  strictEqual(isLargePaste(Array(30).fill("line").join("\n")), false);
});

test("the outgoing message joins typed text and held pastes in order", () => {
  strictEqual(composeOutgoing("question", ["first paste", "second paste"]), "question\n\nfirst paste\n\nsecond paste");
  strictEqual(composeOutgoing("", ["only paste"]), "only paste");
  strictEqual(composeOutgoing("  ", []), "");
});

test("long transcript messages collapse; short ones do not", () => {
  strictEqual(isLongMessage("short"), false);
  strictEqual(isLongMessage("x".repeat(1201)), true);
  strictEqual(isLongMessage(Array(13).fill("line").join("\n")), true);
});

test("the paste summary counts chars and lines", () => {
  strictEqual(pasteSummary("a\nb"), "3 chars · 2 lines");
  strictEqual(pasteSummary("x".repeat(2500)), "2.5k chars · 1 line");
});

test("textual detection accepts prose and rejects binary noise", async () => {
  const { looksTextual } = await import("./paste.js");
  strictEqual(looksTextual("hello world\nplain text"), true);
  strictEqual(looksTextual(""), true);
  strictEqual(looksTextual(""), false);
  strictEqual(looksTextual("mostly text with one � bad char in a longer readable sentence here"), true);
});

test("incremental rendering keeps a settled prefix", async () => {
  const { blocksSharePrefix, parseBlocks } = await import("./markdown.js");
  const before = parseBlocks("one\n\ntwo\n\nthree par");
  const after = parseBlocks("one\n\ntwo\n\nthree partial reply");
  strictEqual(blocksSharePrefix(before, after), true);
  strictEqual(blocksSharePrefix(after, before), true, "same-count tail change still shares the settled prefix");
  strictEqual(blocksSharePrefix(parseBlocks("one\n\nCHANGED\n\nthree"), after), false);
});
