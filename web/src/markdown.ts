/**
 * Minimal markdown for chat replies. No dependencies and no innerHTML:
 * every string from the model lands in a text node, so markup in a reply
 * can never become live DOM. The parser is DOM-free and unit-tested; only
 * renderMarkdown touches the document.
 */

import { highlightable, tokenize } from "./highlight.js";

export type Inline =
  | { kind: "text"; text: string }
  | { kind: "code"; text: string }
  | { kind: "bold"; children: Inline[] }
  | { kind: "italic"; children: Inline[] }
  | { kind: "link"; href: string; text: string };

export type Block =
  | { kind: "paragraph"; text: string }
  | { kind: "heading"; level: number; text: string }
  | { kind: "code"; lang: string; code: string }
  | { kind: "list"; ordered: boolean; items: string[] }
  | { kind: "quote"; text: string }
  | { kind: "table"; headers: string[]; rows: string[][] }
  | { kind: "rule" };

/** Split raw text into blocks. An unterminated fence is still a code
 * block (the reply is streaming), with the text seen so far. */
export function parseBlocks(text: string): Block[] {
  const blocks: Block[] = [];
  const lines = text.split("\n");
  let para: string[] = [];
  let quote: string[] = [];
  let list: { ordered: boolean; items: string[] } | null = null;

  const flushPara = (): void => {
    if (para.length > 0) {
      blocks.push({ kind: "paragraph", text: para.join("\n") });
      para = [];
    }
  };
  const flushQuote = (): void => {
    if (quote.length > 0) {
      blocks.push({ kind: "quote", text: quote.join("\n") });
      quote = [];
    }
  };
  const flushList = (): void => {
    if (list !== null) {
      blocks.push({ kind: "list", ordered: list.ordered, items: list.items });
      list = null;
    }
  };
  const flushAll = (): void => {
    flushPara();
    flushQuote();
    flushList();
  };

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]!;
    const fence = line.match(/^\s*```([^\n`]*)$/);
    if (fence !== null) {
      flushAll();
      const lang = (fence[1] ?? "").trim();
      const body: string[] = [];
      i++;
      while (i < lines.length && !/^\s*```[ \t]*$/.test(lines[i]!)) {
        body.push(lines[i]!);
        i++;
      }
      blocks.push({ kind: "code", lang, code: body.join("\n") });
      continue;
    }
    if (line.trim() === "") {
      flushAll();
      continue;
    }
    const heading = line.match(/^(#{1,6})\s+(.*)$/);
    if (heading !== null) {
      flushAll();
      blocks.push({ kind: "heading", level: (heading[1] ?? "#").length, text: heading[2] ?? "" });
      continue;
    }
    if (/^\s*([-*_])(\s*\1\s*){2,}$/.test(line)) {
      flushAll();
      blocks.push({ kind: "rule" });
      continue;
    }
    const quoteMatch = line.match(/^>\s?(.*)$/);
    if (quoteMatch !== null) {
      flushPara();
      flushList();
      quote.push(quoteMatch[1] ?? "");
      continue;
    }
    // A table needs a header row and a separator row under it.
    if (
      /^\s*\|.*\|\s*$/.test(line) &&
      i + 1 < lines.length &&
      /^\s*\|?[\s:-]+\|[\s|:-]*$/.test(lines[i + 1]!)
    ) {
      flushAll();
      const cells = (row: string): string[] =>
        row.trim().replace(/^\||\|$/g, "").split("|").map((c) => c.trim());
      const headers = cells(line);
      const rows: string[][] = [];
      i += 2;
      while (i < lines.length && /^\s*\|.*\|\s*$/.test(lines[i]!)) {
        rows.push(cells(lines[i]!));
        i++;
      }
      i--;
      blocks.push({ kind: "table", headers, rows });
      continue;
    }
    const item = line.match(/^\s*(?:[-*+]|(\d+)[.)])\s+(.*)$/);
    if (item !== null) {
      flushPara();
      flushQuote();
      const ordered = item[1] !== undefined;
      if (list === null || list.ordered !== ordered) {
        flushList();
        list = { ordered, items: [] };
      }
      list.items.push(item[2] ?? "");
      continue;
    }
    if (list !== null && /^\s{2,}\S/.test(line)) {
      const items = list.items;
      items[items.length - 1] = (items[items.length - 1] ?? "") + " " + line.trim();
      continue;
    }
    flushQuote();
    flushList();
    para.push(line);
  }
  flushAll();
  return blocks;
}

/** Inline spans: `code`, **bold**, *italic*, [text](https://…). Only
 * http(s) links become anchors; anything else stays literal text. Code
 * spans are extracted first so their contents are never reformatted. */
export function parseInline(text: string): Inline[] {
  const out: Inline[] = [];
  let rest = text;
  const pushText = (t: string): void => {
    if (t !== "") out.push({ kind: "text", text: t });
  };
  while (rest !== "") {
    const codeAt = rest.indexOf("`");
    const boldAt = rest.indexOf("**");
    const linkAt = rest.indexOf("[");
    const starAt = rest.indexOf("*");
    // The earliest opener wins; a single * only counts when it is not **.
    let next = -1;
    let kind: "code" | "bold" | "italic" | "link" = "code";
    const consider = (at: number, k: typeof kind): void => {
      if (at >= 0 && (next === -1 || at < next)) {
        next = at;
        kind = k;
      }
    };
    consider(codeAt, "code");
    consider(boldAt, "bold");
    consider(linkAt, "link");
    if (starAt >= 0 && rest[starAt + 1] !== "*") consider(starAt, "italic");
    if (next === -1) {
      pushText(rest);
      break;
    }
    pushText(rest.slice(0, next));
    rest = rest.slice(next);
    if (kind === "code") {
      const end = rest.indexOf("`", 1);
      if (end === -1) {
        pushText(rest);
        break;
      }
      out.push({ kind: "code", text: rest.slice(1, end) });
      rest = rest.slice(end + 1);
    } else if (kind === "bold") {
      const end = rest.indexOf("**", 2);
      if (end === -1) {
        pushText(rest);
        break;
      }
      out.push({ kind: "bold", children: parseInline(rest.slice(2, end)) });
      rest = rest.slice(end + 2);
    } else if (kind === "italic") {
      const end = rest.indexOf("*", 1);
      if (end === -1 || rest[1] === "*") {
        pushText(rest);
        break;
      }
      out.push({ kind: "italic", children: parseInline(rest.slice(1, end)) });
      rest = rest.slice(end + 1);
    } else {
      const m = rest.match(/^\[([^\]\n]+)\]\((https?:\/\/[^)\s]+)\)/);
      if (m === null) {
        pushText("[");
        rest = rest.slice(1);
      } else {
        out.push({ kind: "link", href: m[2]!, text: m[1]! });
        rest = rest.slice(m[0].length);
      }
    }
  }
  return out;
}

function renderInline(parent: HTMLElement, inlines: Inline[]): void {
  for (const span of inlines) {
    if (span.kind === "text") {
      parent.append(document.createTextNode(span.text));
    } else if (span.kind === "code") {
      const code = document.createElement("code");
      code.className = "inline-code";
      code.textContent = span.text;
      parent.append(code);
    } else if (span.kind === "bold") {
      const strong = document.createElement("strong");
      renderInline(strong, span.children);
      parent.append(strong);
    } else if (span.kind === "italic") {
      const em = document.createElement("em");
      renderInline(em, span.children);
      parent.append(em);
    } else {
      const a = document.createElement("a");
      a.href = span.href;
      a.target = "_blank";
      a.rel = "noopener noreferrer";
      a.textContent = span.text;
      parent.append(a);
    }
  }
}

export interface RenderOptions {
  /** Called with the raw code when a code block's Copy button is hit. */
  onCopyCode?: (code: string, button: HTMLButtonElement) => void;
}

/** Append code as highlighted spans (or one text node when plain). */
function renderCode(codeEl: HTMLElement, lang: string, code: string): void {
  if (!highlightable(lang)) {
    codeEl.textContent = code;
    return;
  }
  for (const span of tokenize(code)) {
    if (span.kind === "plain") {
      codeEl.append(document.createTextNode(span.text));
    } else {
      const s = document.createElement("span");
      s.className = "tok-" + span.kind;
      s.textContent = span.text;
      codeEl.append(s);
    }
  }
}

/** Render one block to a fresh element. */
function renderBlock(block: Block, opts: RenderOptions): HTMLElement {
  const host = document.createElement("div");
  {
    if (block.kind === "code") {
      const wrap = document.createElement("div");
      wrap.className = "code-block";
      const head = document.createElement("div");
      head.className = "code-head";
      const lang = document.createElement("span");
      lang.className = "code-lang";
      lang.textContent = block.lang === "" ? "code" : block.lang;
      const copy = document.createElement("button");
      copy.type = "button";
      copy.className = "code-copy";
      copy.textContent = "Copy";
      const codeText = block.code;
      copy.addEventListener("click", () => opts.onCopyCode?.(codeText, copy));
      head.append(lang, copy);
      const pre = document.createElement("pre");
      const code = document.createElement("code");
      renderCode(code, block.lang, block.code);
      pre.append(code);
      wrap.append(head, pre);
      host.append(wrap);
    } else if (block.kind === "heading") {
      const h = document.createElement(block.level <= 2 ? "h3" : "h4");
      renderInline(h, parseInline(block.text));
      host.append(h);
    } else if (block.kind === "list") {
      const list = document.createElement(block.ordered ? "ol" : "ul");
      for (const item of block.items) {
        const li = document.createElement("li");
        renderInline(li, parseInline(item));
        list.append(li);
      }
      host.append(list);
    } else if (block.kind === "quote") {
      const quote = document.createElement("blockquote");
      renderInline(quote, parseInline(block.text));
      host.append(quote);
    } else if (block.kind === "table") {
      const table = document.createElement("table");
      table.className = "md-table";
      const thead = document.createElement("thead");
      const headRow = document.createElement("tr");
      for (const cell of block.headers) {
        const th = document.createElement("th");
        renderInline(th, parseInline(cell));
        headRow.append(th);
      }
      thead.append(headRow);
      const tbody = document.createElement("tbody");
      for (const row of block.rows) {
        const tr = document.createElement("tr");
        for (const cell of row) {
          const td = document.createElement("td");
          renderInline(td, parseInline(cell));
          tr.append(td);
        }
        tbody.append(tr);
      }
      table.append(thead, tbody);
      const scroll = document.createElement("div");
      scroll.className = "md-table-wrap";
      scroll.append(table);
      host.append(scroll);
    } else if (block.kind === "rule") {
      host.append(document.createElement("hr"));
    } else {
      const p = document.createElement("p");
      const parts = block.text.split("\n");
      parts.forEach((part, idx) => {
        if (idx > 0) p.append(document.createElement("br"));
        renderInline(p, parseInline(part));
      });
      host.append(p);
    }
  }
  return host.firstElementChild as HTMLElement;
}

/** Replace the host's children with the rendered markdown. */
export function renderMarkdown(host: HTMLElement, text: string, opts: RenderOptions = {}): void {
  host.textContent = "";
  for (const block of parseBlocks(text)) {
    host.append(renderBlock(block, opts));
  }
}

/** Whether every block of `a` before its last equals the same-index
 * block of `b`: the condition for re-rendering only the tail. */
export function blocksSharePrefix(a: Block[], b: Block[]): boolean {
  if (b.length < a.length || a.length === 0) return false;
  for (let i = 0; i < a.length - 1; i++) {
    if (JSON.stringify(a[i]) !== JSON.stringify(b[i])) return false;
  }
  return true;
}

/**
 * Streaming render: when the new text only extends the old, keep the
 * settled blocks' DOM and re-render from the last old block on. A long
 * reply then costs one block per frame, not the whole message.
 */
export function renderMarkdownIncremental(
  host: HTMLElement,
  prevText: string,
  nextText: string,
  opts: RenderOptions = {}
): void {
  if (prevText === "" || !nextText.startsWith(prevText)) {
    renderMarkdown(host, nextText, opts);
    return;
  }
  const prevBlocks = parseBlocks(prevText);
  const nextBlocks = parseBlocks(nextText);
  if (!blocksSharePrefix(prevBlocks, nextBlocks) || host.childElementCount !== prevBlocks.length) {
    renderMarkdown(host, nextText, opts);
    return;
  }
  while (host.childElementCount > prevBlocks.length - 1) {
    host.lastElementChild?.remove();
  }
  for (let i = prevBlocks.length - 1; i < nextBlocks.length; i++) {
    host.append(renderBlock(nextBlocks[i]!, opts));
  }
}
