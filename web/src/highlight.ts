/**
 * A small, language-agnostic code tokenizer for chat code boxes:
 * comments, strings, numbers, and a keyword set. It only splits text
 * into spans; rendering builds text nodes, so highlighted code is as
 * safe as plain code. DOM-free so the node:test suite runs it.
 */

export type SpanKind = "plain" | "keyword" | "string" | "comment" | "number";

export interface Span {
  kind: SpanKind;
  text: string;
}

const KEYWORDS = new Set([
  "fn", "let", "mut", "pub", "struct", "enum", "impl", "trait", "use", "mod", "match", "if", "else",
  "for", "while", "loop", "return", "break", "continue", "const", "static", "type", "where", "async",
  "await", "move", "ref", "self", "Self", "true", "false", "null", "None", "Some", "Ok", "Err",
  "function", "var", "class", "extends", "new", "import", "from", "export", "default", "try", "catch",
  "throw", "switch", "case", "typeof", "instanceof", "in", "of", "do", "void", "this",
  "def", "print", "lambda", "with", "as", "pass", "raise", "yield", "global", "nonlocal", "assert",
  "int", "float", "str", "bool", "list", "dict", "set", "tuple",
]);

/** Split code into spans. Adjacent plain runs merge into one span. */
export function tokenize(code: string): Span[] {
  const spans: Span[] = [];
  const push = (kind: SpanKind, text: string): void => {
    if (text === "") return;
    const last = spans[spans.length - 1];
    if (last !== undefined && last.kind === kind) last.text += text;
    else spans.push({ kind, text });
  };
  let i = 0;
  while (i < code.length) {
    const rest = code.slice(i);
    const ch = code[i]!;
    // Comments: // and # to end of line, /* ... */ blocks.
    if (rest.startsWith("//") || (ch === "#" && (i === 0 || /\s/.test(code[i - 1]!)))) {
      const end = code.indexOf("\n", i);
      const stop = end === -1 ? code.length : end;
      push("comment", code.slice(i, stop));
      i = stop;
      continue;
    }
    if (rest.startsWith("/*")) {
      const end = code.indexOf("*/", i + 2);
      const stop = end === -1 ? code.length : end + 2;
      push("comment", code.slice(i, stop));
      i = stop;
      continue;
    }
    // Strings: single, double, backtick; backslash escapes the next char.
    if (ch === '"' || ch === "'" || ch === "`") {
      let j = i + 1;
      while (j < code.length && code[j] !== ch) {
        if (code[j] === "\\") j++;
        j++;
      }
      const stop = Math.min(j + 1, code.length);
      push("string", code.slice(i, stop));
      i = stop;
      continue;
    }
    // Numbers: digits with separators, dots, and suffix letters.
    if (/[0-9]/.test(ch) && (i === 0 || !/[\w]/.test(code[i - 1]!))) {
      const m = rest.match(/^[0-9][0-9_]*(\.[0-9_]+)?([eE][+-]?[0-9]+)?[a-zA-Z]*/);
      if (m !== null) {
        push("number", m[0]);
        i += m[0].length;
        continue;
      }
    }
    // Words: keywords get their own span, the rest stays plain.
    if (/[A-Za-z_$]/.test(ch)) {
      const m = rest.match(/^[A-Za-z_$][\w$]*/);
      if (m !== null) {
        push(KEYWORDS.has(m[0]) ? "keyword" : "plain", m[0]);
        i += m[0].length;
        continue;
      }
    }
    push("plain", ch);
    i++;
  }
  return spans;
}

/** Languages the tokenizer is worth applying to; others stay plain. */
export function highlightable(lang: string): boolean {
  const l = lang.trim().toLowerCase();
  return l !== "" && l !== "text" && l !== "plain" && l !== "plaintext";
}
