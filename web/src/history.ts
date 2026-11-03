/**
 * No DOM and no storage access, so Node can unit test this module.
 */

import type { ChatMessage, ChatRequestBody } from "./api.js";
import type { Settings } from "./store.js";

/** History turns as they are stored; only role and content are read. */
export interface HistoryTurn {
  role: "user" | "assistant";
  content: string;
}

/** Contract messages for a request: the system message first when set,
 *  every user turn exactly once, and assistant turns that produced text
 *  (a whitespace-only reply counts as no text). */
export function buildMessages(systemMessage: string, history: readonly HistoryTurn[]): ChatMessage[] {
  const messages: ChatMessage[] = [];
  if (systemMessage.trim() !== "") {
    messages.push({ role: "system", content: systemMessage });
  }
  for (const m of history) {
    if (m.role === "user") messages.push({ role: "user", content: m.content });
    else if (m.content.trim() !== "") messages.push({ role: "assistant", content: m.content });
  }
  return messages;
}

/** Why a send must not go out yet, or null when it may. The server
 * requires a model (or x-engine/x-session, which this UI never sends),
 * but buildChatBody omits the field while no model is selected — for
 * example in the moment between saving an API key and the model list
 * (which auto-selects the first model) landing. Sending then earned
 * an opaque server 400 rendered as "Reply failed". */
export function noModelNotice(model: string): string | null {
  if (model !== "") return null;
  return "No model is selected yet. Pick one in the model menu; if it is empty, the server has not reported a ready model.";
}

export function buildChatBody(settings: Settings, messages: ChatMessage[]): ChatRequestBody {
  return {
    model: settings.model === "" ? undefined : settings.model,
    messages,
    temperature: settings.drawer.temperature,
    top_p: settings.drawer.topP,
    top_k: settings.drawer.topK > 0 ? settings.drawer.topK : undefined,
    max_tokens: settings.drawer.maxTokens,
    "x-thinking": settings.drawer.thinking,
  };
}
