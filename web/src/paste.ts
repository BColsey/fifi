/**
 * Large-paste handling for the composer. A paste past the threshold is
 * held as an attachment chip above the textarea. The text is sent
 * verbatim on Send, and the user can remove it. DOM-free so the
 * node:test suite runs it.
 */

export const PASTE_CHAR_LIMIT = 4000;
export const PASTE_LINE_LIMIT = 30;

/** Long user text collapses in the transcript past this size. */
export const LONG_MESSAGE_CHARS = 1200;
export const LONG_MESSAGE_LINES = 12;

export function isLargePaste(text: string): boolean {
  if (text.length > PASTE_CHAR_LIMIT) return true;
  let lines = 1;
  for (const ch of text) if (ch === "\n") lines++;
  return lines > PASTE_LINE_LIMIT;
}

export function isLongMessage(text: string): boolean {
  if (text.length > LONG_MESSAGE_CHARS) return true;
  let lines = 1;
  for (const ch of text) if (ch === "\n") lines++;
  return lines > LONG_MESSAGE_LINES;
}

export function pasteSummary(text: string): string {
  let lines = 1;
  for (const ch of text) if (ch === "\n") lines++;
  const chars =
    text.length >= 1000 ? (text.length / 1000).toFixed(1) + "k chars" : text.length + " chars";
  return chars + " · " + lines + (lines === 1 ? " line" : " lines");
}

/** Whether decoded file bytes read as text: fewer than 1 in 16
 * replacement or control characters in the sampled head. */
export function looksTextual(text: string): boolean {
  const sample = text.slice(0, 4096);
  if (sample === "") return true;
  let bad = 0;
  for (const ch of sample) {
    const code = ch.codePointAt(0) ?? 0;
    if (ch === "\uFFFD" || (code < 32 && ch !== "\n" && ch !== "\t" && ch !== "\r")) bad++;
  }
  return bad / sample.length < 1 / 16;
}

/** What Send actually sends: typed text first, then each held paste. */
export function composeOutgoing(typed: string, pastes: readonly string[]): string {
  const parts = [typed, ...pastes].map((p) => p.trim()).filter((p) => p !== "");
  return parts.join("\n\n");
}
