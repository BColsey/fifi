/**
 * Streaming text accumulates on the message and the reply's markdown is
 * re-rendered at most once per animation frame, so code fences settle
 * as they stream without re-rendering per token.
 */

import {
  ApiError,
  chatStream,
  describeError,
  type ChatRequestBody,
  type ToolCallDelta,
  type Usage,
} from "./api";
import { buildChatBody, buildMessages, noModelNotice } from "./history";
import {
  deleteChat,
  getChat,
  loadSettings,
  newChatRecord,
  newUiMessage,
  putChat,
  type ChatRecord,
  type TraceSeg,
  type UiMessage,
} from "./store";
import { formatDuration, formatSpeed, formatTokens } from "./format";
import { DeadChats } from "./deadchats";
import { SendGate } from "./sendgate";
import { renderMarkdown } from "./markdown";
import { copyText, flashCopied } from "./copy";
import { composeOutgoing, isLargePaste, isLongMessage, pasteSummary } from "./paste";

export interface ChatHooks {
  onBusyChange(busy: boolean): void;
  onUnauthorized(): void;
  onChatsChanged(): void;
  onNotice(text: string): void;
  /** One polite announcement per stream lifecycle event, never per token. */
  onAnnounce(text: string): void;
  /** The user asked to inspect a reply's sampler trace. */
  onShowTrace(message: UiMessage): void;
}

const utf8 = new TextEncoder();

/** Appends one piece to the reply's raw→field map, merging consecutive
 * pieces of the same field so the stored array stays small. Byte lengths
 * because the packet's offsets are UTF-8 bytes into the raw text. */
function appendSeg(segs: TraceSeg[], field: TraceSeg["field"], text: string): void {
  if (text === "") return;
  const bytes = utf8.encode(text).length;
  const last = segs[segs.length - 1];
  if (last !== undefined && last.field === field) last.bytes += bytes;
  else segs.push({ field, bytes });
}

interface StreamDom {
  root: HTMLElement;
  waiting: HTMLElement;
  thinking: HTMLDetailsElement | null;
  thinkingText: Text | null;
  /** The markdown host for the reply text; re-rendered per flush. */
  textHost: HTMLElement;
  tools: HTMLElement | null;
  meta: HTMLElement;
  actions: HTMLElement;
}

export class ChatView {
  private readonly column: HTMLElement;
  private readonly scroll: HTMLElement;
  private readonly input: HTMLTextAreaElement;
  private readonly sendButton: HTMLButtonElement;
  private readonly sendIcon: HTMLElement;
  private readonly stopIcon: HTMLElement;
  private readonly hooks: ChatHooks;

  private chat: ChatRecord | null = null;
  private controller: AbortController | null = null;
  private readonly sendGate = new SendGate();
  private offline = false;
  private readonly deadChats = new DeadChats();

  private pendingContent = "";
  private pendingReasoning = "";
  private flushHandle = 0;
  private streamDom: StreamDom | null = null;
  private startedAt = 0;
  private streamChat: ChatRecord | null = null;
  private streamMessage: UiMessage | null = null;
  private toolCalls = new Map<number, { name: string; args: string }>();
  /** Large pastes held as composer attachments until Send. */
  private pendingPastes: string[] = [];
  private pasteTray: HTMLElement | null = null;

  constructor(
    column: HTMLElement,
    scroll: HTMLElement,
    input: HTMLTextAreaElement,
    sendButton: HTMLButtonElement,
    sendIcon: HTMLElement,
    stopIcon: HTMLElement,
    hooks: ChatHooks
  ) {
    this.column = column;
    this.scroll = scroll;
    this.input = input;
    this.sendButton = sendButton;
    this.sendIcon = sendIcon;
    this.stopIcon = stopIcon;
    this.hooks = hooks;

    this.input.addEventListener("input", () => this.autosize());
    this.input.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
        e.preventDefault();
        void this.send();
      }
    });
    this.input.addEventListener("paste", (e) => {
      const pasted = e.clipboardData?.getData("text") ?? "";
      if (pasted === "" || !isLargePaste(pasted)) return; // small paste: normal textarea behaviour
      e.preventDefault();
      this.pendingPastes.push(pasted);
      this.renderPasteTray();
      this.updateSendEnabled();
      this.input.focus();
    });
    this.buildPasteTray();
    this.sendButton.addEventListener("click", () => {
      if (this.controller !== null) this.controller.abort();
      else void this.send();
    });
    this.renderEmpty();
  }

  setOffline(offline: boolean): void {
    this.offline = offline;
    this.updateSendEnabled();
    if (offline) this.input.placeholder = "The server is unreachable. Sending is disabled.";
    else this.input.placeholder = "Message the model";
  }

  private updateSendEnabled(): void {
    const hasText = this.input.value.trim() !== "" || this.pendingPastes.length > 0;
    const streaming = this.controller !== null;
    this.sendButton.disabled = !streaming && (!hasText || this.offline);
    // Exactly one icon at a time: the class drives CSS, `hidden` is the
    // no-CSS fallback. Both together fix the both-icons-visible state.
    this.sendButton.classList.toggle("is-streaming", streaming);
    this.sendIcon.hidden = streaming;
    this.stopIcon.hidden = !streaming;
    this.sendButton.setAttribute("aria-label", streaming ? "Stop" : "Send");
    this.sendButton.title = streaming ? "Stop" : "Send";
  }

  private autosize(): void {
    this.input.style.height = "auto";
    const cap = Math.max(5 * 24 + 18, Math.floor(window.innerHeight * 0.4));
    this.input.style.height = Math.min(this.input.scrollHeight, cap) + "px";
    this.updateSendEnabled();
  }

  /** The tray above the composer holding large pastes until Send. */
  private buildPasteTray(): void {
    const wrap = this.input.closest(".composer-wrap");
    const composer = this.input.closest(".composer");
    if (wrap === null || composer === null) return;
    const tray = document.createElement("div");
    tray.className = "paste-tray";
    tray.hidden = true;
    wrap.insertBefore(tray, composer);
    this.pasteTray = tray;
  }

  private renderPasteTray(): void {
    const tray = this.pasteTray;
    if (tray === null) return;
    tray.textContent = "";
    tray.hidden = this.pendingPastes.length === 0;
    this.pendingPastes.forEach((text, index) => {
      const chip = document.createElement("span");
      chip.className = "paste-chip";
      const label = document.createElement("span");
      label.textContent = "Pasted text · " + pasteSummary(text);
      const remove = document.createElement("button");
      remove.type = "button";
      remove.className = "paste-remove";
      remove.textContent = "Remove";
      remove.setAttribute("aria-label", "Remove pasted text " + (index + 1));
      remove.addEventListener("click", () => {
        this.pendingPastes.splice(index, 1);
        this.renderPasteTray();
        this.updateSendEnabled();
      });
      chip.append(label, remove);
      tray.append(chip);
    });
  }

  private clearPastes(): void {
    this.pendingPastes = [];
    this.renderPasteTray();
  }

  /** The partial reply is kept. */
  abort(): void {
    if (this.controller !== null) this.controller.abort();
  }

  /** Mark a chat id as deleted so a finishing stream cannot write it back. */
  forgetChat(id: string): void {
    this.deadChats.mark(id);
  }

  /** Undo the dead mark: the delete failed, so the chat keeps saving. */
  unforgetChat(id: string): void {
    this.deadChats.unmark(id);
  }

  async openChat(id: string): Promise<void> {
    this.abort();
    let record: ChatRecord | undefined;
    try {
      record = await getChat(id);
    } catch (err) {
      // Leave the current view in place; the read failed, not the chat.
      this.hooks.onNotice("Could not open this chat: " + reasonOf(err));
      return;
    }
    this.chat = record ?? null;
    this.renderAll();
  }

  newChat(): void {
    this.abort();
    this.chat = null;
    this.clearPastes();
    this.renderEmpty();
    this.input.focus();
  }

  currentChatId(): string | null {
    return this.chat?.id ?? null;
  }

  private renderEmpty(): void {
    this.column.textContent = "";
    const hero = document.createElement("div");
    hero.className = "chat-empty-hero";
    const h2 = document.createElement("h2");
    h2.textContent = "Chat with fifi";
    hero.append(h2);
    this.column.append(hero);
  }

  private renderAll(): void {
    this.column.textContent = "";
    const messages = this.chat?.messages ?? [];
    if (messages.length === 0) {
      this.renderEmpty();
      return;
    }
    for (const m of messages) this.column.append(this.renderMessage(m));
    this.scrollToEnd();
  }

  private renderMessage(m: UiMessage): HTMLElement {
    const root = document.createElement("div");
    if (m.role === "user") {
      root.className = "msg msg-user";
      const bubble = document.createElement("div");
      bubble.className = "bubble";
      if (isLongMessage(m.content)) {
        bubble.classList.add("bubble-collapsed");
        const preview = document.createElement("span");
        preview.textContent = m.content;
        const toggle = document.createElement("button");
        toggle.type = "button";
        toggle.className = "msg-expand";
        toggle.textContent = "Show more";
        toggle.setAttribute("aria-expanded", "false");
        toggle.addEventListener("click", () => {
          const expanded = bubble.classList.toggle("bubble-expanded");
          bubble.classList.toggle("bubble-collapsed", !expanded);
          toggle.textContent = expanded ? "Show less" : "Show more";
          toggle.setAttribute("aria-expanded", String(expanded));
        });
        bubble.append(preview, toggle);
      } else {
        bubble.textContent = m.content;
      }
      root.append(bubble);
      root.append(this.buildActions(m));
      return root;
    }
    root.className = "msg msg-assistant";
    if (m.reasoning !== undefined && m.reasoning !== "") {
      root.append(this.buildThinking(m.reasoning, true));
    }
    const text = document.createElement("div");
    text.className = "msg-text markdown";
    renderMarkdown(text, m.content, { onCopyCode: (code, btn) => void this.copyCode(code, btn) });
    root.append(text);
    if (m.toolCalls !== undefined && m.toolCalls.length > 0) {
      root.append(this.buildTools(m.toolCalls));
    }
    if (m.state === "error" && m.errorMessage !== undefined) {
      const err = document.createElement("div");
      err.className = "msg-error";
      err.textContent = m.errorMessage;
      root.append(err);
    }
    root.append(this.buildActions(m));
    root.append(this.buildMeta(m));
    return root;
  }

  private async copyCode(code: string, button: HTMLButtonElement): Promise<void> {
    flashCopied(button, await copyText(code));
  }

  /** Copy for a whole message, plus the trace shortcut on traced replies. */
  private buildActions(m: UiMessage): HTMLElement {
    const row = document.createElement("div");
    row.className = "msg-actions";
    const copy = document.createElement("button");
    copy.type = "button";
    copy.className = "msg-copy";
    copy.textContent = "Copy";
    copy.setAttribute("aria-label", m.role === "user" ? "Copy your message" : "Copy reply");
    const raw = m.content;
    copy.addEventListener("click", () => {
      void copyText(raw).then((ok) => flashCopied(copy, ok));
    });
    row.append(copy);
    if (m.role === "assistant" && m.replyId !== undefined && m.traceSegs !== undefined) {
      row.append(this.buildTraceButton(m));
    }
    return row;
  }

  private buildTraceButton(m: UiMessage): HTMLButtonElement {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "msg-trace-btn";
    btn.textContent = "Token trace";
    btn.addEventListener("click", () => this.hooks.onShowTrace(m));
    return btn;
  }

  private buildThinking(reasoning: string, closed: boolean): HTMLDetailsElement {
    const details = document.createElement("details");
    details.className = "msg-thinking";
    if (!closed) details.open = true;
    const summary = document.createElement("summary");
    summary.textContent = "Thinking";
    const body = document.createElement("div");
    body.className = "thinking-text";
    body.textContent = reasoning;
    details.append(summary, body);
    return details;
  }

  private buildTools(calls: { name: string; arguments: string }[]): HTMLElement {
    const wrap = document.createElement("div");
    wrap.className = "msg-tools";
    for (const call of calls) {
      wrap.append(this.buildToolBlock(call.name, call.arguments));
    }
    return wrap;
  }

  private buildToolBlock(name: string, argsText: string): HTMLElement {
    const block = document.createElement("div");
    block.className = "tool-call";
    const head = document.createElement("div");
    head.className = "tool-call-head";
    const nameEl = document.createElement("div");
    nameEl.className = "tool-call-name";
    nameEl.textContent = name;
    const copy = document.createElement("button");
    copy.type = "button";
    copy.className = "code-copy";
    copy.textContent = "Copy";
    copy.setAttribute("aria-label", "Copy tool arguments");
    copy.addEventListener("click", () => {
      void copyText(argsText).then((ok) => flashCopied(copy, ok));
    });
    head.append(nameEl, copy);
    const args = document.createElement("pre");
    args.className = "tool-call-args";
    args.textContent = argsText;
    block.append(head, args);
    return block;
  }

  private buildMeta(m: UiMessage): HTMLElement {
    const meta = document.createElement("div");
    meta.className = "msg-meta";
    const parts: string[] = [];
    if (m.state === "stopped") parts.push("stopped");
    if (m.finishReason === "length") parts.push("Cut off: length");
    else if (m.finishReason === "tool_calls") parts.push("Tool calls");
    if (m.usage !== undefined) {
      parts.push(
        formatTokens(m.usage.prompt_tokens) + " prompt",
        formatTokens(m.usage.completion_tokens) + " out"
      );
      if (m.usage.reused_tokens !== undefined && m.usage.reused_tokens > 0) {
        parts.push(formatTokens(m.usage.reused_tokens) + " reused");
      }
    }
    if (m.durationMs !== undefined) {
      parts.push(formatDuration(m.durationMs));
      if (m.usage !== undefined) {
        const speed = formatSpeed(m.usage.completion_tokens, m.durationMs);
        if (speed !== "") parts.push(speed);
      }
    }
    meta.textContent = parts.join(" \u00b7 ");
    return meta;
  }

  private scrollToEnd(): void {
    this.scroll.scrollTop = this.scroll.scrollHeight;
  }

  private buildBody(chat: ChatRecord): ChatRequestBody {
    const settings = loadSettings();
    const body = buildChatBody(
      settings,
      buildMessages(settings.drawer.systemMessage, chat.messages)
    );
    // One named session per chat: turns stick to one engine, and the
    // monitor groups this chat's rates under its id.
    body["x-session"] = "web-" + chat.id;
    return body;
  }

  private async send(): Promise<void> {
    const text = composeOutgoing(this.input.value, this.pendingPastes);
    if (text === "" || this.offline) return;
    const noModel = noModelNotice(loadSettings().model);
    if (noModel !== null) {
      this.hooks.onNotice(noModel);
      return;
    }
    // Close the gate and hold the controller before the first await, so a
    // second send, a chat switch, or a delete cannot start a second stream.
    if (!this.sendGate.tryBegin()) return;
    const controller = new AbortController();
    this.controller = controller;
    this.updateSendEnabled();
    try {
      if (this.chat === null) {
        this.chat = newChatRecord(loadSettings().model);
      }
      const chat = this.chat;

      const userMsg = newUiMessage("user", text);
      chat.messages.push(userMsg);
      chat.title = chat.messages.some((m) => m.role === "user")
        ? titleOf(chat)
        : chat.title;
      chat.updatedAt = Date.now();
      try {
        await putChat(chat);
      } catch (err) {
        // The chat lives in memory; the request still goes out.
        this.hooks.onNotice("Could not save this chat: " + reasonOf(err));
      }
      this.hooks.onChatsChanged();
      // A switch, a delete, or a stop click aborted the pending send.
      if (controller.signal.aborted) {
        this.endPendingSend(controller);
        return;
      }

      this.input.value = "";
      this.clearPastes();
      this.autosize();
      if (chat.messages.length <= 2) this.renderAll();
      else this.column.append(this.renderMessage(userMsg));
      this.scrollToEnd();

      const assistant = newUiMessage("assistant", "");
      assistant.state = "streaming";
      // The record joins the chat at stream start, so a reload, a chat switch,
      // and the next request all see it. buildMessages skips its empty content.
      chat.messages.push(assistant);
      this.startStream(chat, assistant, controller);
    } catch (err) {
      // The stream never started; drop the busy state instead of wedging it.
      this.endPendingSend(controller);
      this.hooks.onNotice("Could not send the message: " + reasonOf(err));
    }
  }

  private endPendingSend(controller: AbortController): void {
    if (this.controller === controller) this.controller = null;
    this.sendGate.end();
    this.updateSendEnabled();
  }

  private startStream(chat: ChatRecord, assistant: UiMessage, controller: AbortController): void {
    const body = this.buildBody(chat);
    // Arming is pre-generation: the segment map is recorded only for
    // replies the server was asked to trace.
    if (body["x-sample-trace"] === true) {
      assistant.traceSegs = [];
      assistant.traceModel = body.model;
    }
    this.controller = controller;
    this.streamDom = this.createStreamDom();
    this.streamChat = chat;
    this.streamMessage = assistant;
    this.toolCalls.clear();
    this.startedAt = performance.now();
    this.updateSendEnabled();
    this.hooks.onBusyChange(true);
    this.hooks.onAnnounce("Answering");

    const finish = (state: "complete" | "stopped" | "error", errorMessage?: string, usage?: Usage) => {
      const dom = this.streamDom;
      if (dom !== null) {
        dom.waiting.remove();
        dom.textHost.classList.remove("caret");
        // Final render settles any unterminated fence, then actions appear.
        renderMarkdown(dom.textHost, assistant.content, {
          onCopyCode: (code, btn) => void this.copyCode(code, btn),
        });
        if (state === "error" && errorMessage !== undefined) {
          const error = document.createElement("div");
          error.className = "msg-error";
          error.textContent = errorMessage;
          dom.root.insertBefore(error, dom.meta);
        }
        dom.actions.replaceChildren(...this.buildActions(assistant).childNodes);
        dom.meta.textContent = "";
      }
      // Keep the placeholder; the error line carries the reason.
      assistant.state = state;
      assistant.errorMessage = errorMessage;
      if (usage !== undefined) assistant.usage = usage;
      assistant.durationMs = performance.now() - this.startedAt;
      if (dom !== null) dom.meta.textContent = this.buildMeta(assistant).textContent;
      if (!this.deadChats.has(chat.id)) {
        chat.updatedAt = Date.now();
        putChat(chat)
          .then(() => this.hooks.onChatsChanged())
          .catch((err: unknown) =>
            this.hooks.onNotice("Could not save this chat: " + reasonOf(err))
          );
      }
      this.controller = null;
      this.sendGate.end();
      this.streamDom = null;
      this.streamChat = null;
      this.streamMessage = null;
      this.updateSendEnabled();
      this.hooks.onBusyChange(false);
      this.hooks.onAnnounce(
        state === "complete" ? "Reply finished" : state === "stopped" ? "Stopped" : "Reply failed"
      );
    };

    chatStream(
      body,
      {
        onRole: () => {
          const dom = this.streamDom;
          if (dom !== null) dom.waiting.textContent =
            "Admitted. Prefilling.";
        },
        onReplyId: (id) => {
          const msg = this.streamMessage;
          if (msg !== null) msg.replyId = id;
        },
        onDelta: (delta) => {
          const msg = this.streamMessage;
          if (msg === null) return;
          if (delta.reasoning !== undefined) {
            msg.reasoning = (msg.reasoning ?? "") + delta.reasoning;
            this.pendingReasoning += delta.reasoning;
            if (msg.traceSegs !== undefined) appendSeg(msg.traceSegs, "reasoning", delta.reasoning);
          }
          if (delta.content !== undefined) {
            msg.content += delta.content;
            this.pendingContent += delta.content;
            if (msg.traceSegs !== undefined) appendSeg(msg.traceSegs, "content", delta.content);
          }
          if (delta.toolCall !== undefined) {
            // Tool-call pieces are part of the raw text the packet's
            // offsets run over, so they need a segment of their own —
            // otherwise every later offset would shift. The wire carries
            // re-serialized name + arguments, not the raw markup, so these
            // byte lengths are approximate; the trace view marks its region
            // labels accordingly (think tags shift them the same way).
            if (msg.traceSegs !== undefined) {
              appendSeg(
                msg.traceSegs,
                "tool",
                (delta.toolCall.function?.name ?? "") + (delta.toolCall.function?.arguments ?? "")
              );
            }
            this.mergeToolCall(delta.toolCall);
          }
          this.scheduleFlush();
        },
        onFinish: (reason, usage) => {
          const msg = this.streamMessage;
          if (msg === null) return;
          if (usage !== undefined) msg.usage = usage;
          msg.finishReason = reason;
        },
      },
      controller.signal
    )
      .then((end) => {
        this.flushNow();
        // A stream that ends without [DONE] was cancelled by disconnect.
        finish(end === "done" ? "complete" : "stopped");
      })
      .catch((err: unknown) => {
        this.flushNow();
        if (err instanceof Error && err.name === "AbortError") {
          finish("stopped");
          return;
        }
        if (err instanceof ApiError) {
          if (err.code === "unauthorized") this.hooks.onUnauthorized();
          finish("error", describeError(err), this.streamMessage?.usage);
          return;
        }
        finish("error", "The stream ended unexpectedly.");
      });
  }

  private mergeToolCall(frag: ToolCallDelta): void {
    const name = frag.function?.name;
    const args = frag.function?.arguments ?? "";
    const existing = this.toolCalls.get(frag.index);
    if (existing === undefined) {
      this.toolCalls.set(frag.index, { name: name ?? "", args });
    } else {
      if (name !== undefined) existing.name = name;
      existing.args += args;
    }
    const msg = this.streamMessage;
    if (msg !== null) {
      msg.toolCalls = [...this.toolCalls.values()].map((c) => ({
        name: c.name,
        arguments: c.args,
      }));
    }
    this.renderToolsNow();
  }

  private createStreamDom(): StreamDom {
    const root = document.createElement("div");
    root.className = "msg msg-assistant";
    const waiting = document.createElement("div");
    waiting.className = "waiting";
    waiting.textContent = "Waiting for admission\u2026";
    const textEl = document.createElement("div");
    textEl.className = "msg-text markdown caret";
    const actions = document.createElement("div");
    actions.className = "msg-actions";
    const meta = document.createElement("div");
    meta.className = "msg-meta";
    root.append(waiting, textEl, actions, meta);
    this.column.append(root);
    this.scrollToEnd();
    return { root, waiting, thinking: null, thinkingText: null, textHost: textEl, tools: null, meta, actions };
  }

  private scheduleFlush(): void {
    if (this.flushHandle !== 0) return;
    this.flushHandle = requestAnimationFrame(() => {
      this.flushHandle = 0;
      this.flushNow();
    });
  }

  private flushNow(): void {
    if (this.flushHandle !== 0) {
      cancelAnimationFrame(this.flushHandle);
      this.flushHandle = 0;
    }
    const dom = this.streamDom;
    const msg = this.streamMessage;
    if (dom === null || msg === null) {
      this.pendingContent = "";
      this.pendingReasoning = "";
      return;
    }
    if (this.pendingReasoning !== "") {
      if (dom.thinking === null) {
        const details = this.buildThinking("", false);
        dom.waiting.remove();
        dom.root.prepend(details);
        const body = details.querySelector<HTMLElement>(".thinking-text");
        dom.thinking = details;
        dom.thinkingText = body !== null ? document.createTextNode("") : null;
        body?.prepend(dom.thinkingText ?? "");
      }
      dom.thinkingText?.insertData(dom.thinkingText.length, this.pendingReasoning);
      this.pendingReasoning = "";
    }
    if (this.pendingContent !== "") {
      dom.waiting.remove();
      this.pendingContent = "";
      // Re-render the markdown built so far. This runs at most once per
      // animation frame, so code fences and emphasis settle as they stream.
      renderMarkdown(dom.textHost, msg.content, {
        onCopyCode: (code, btn) => void this.copyCode(code, btn),
      });
      dom.textHost.scrollIntoView({ block: "end" });
    }
  }

  private renderToolsNow(): void {
    const dom = this.streamDom;
    if (dom === null) return;
    if (dom.tools === null) {
      dom.tools = document.createElement("div");
      dom.tools.className = "msg-tools";
      dom.root.insertBefore(dom.tools, dom.actions);
    }
    dom.tools.textContent = "";
    for (const call of this.toolCalls.values()) {
      dom.tools.append(this.buildToolBlock(call.name, call.args));
    }
  }
}

function reasonOf(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

function titleOf(chat: ChatRecord): string {
  const first = chat.messages.find((m) => m.role === "user");
  if (first === undefined) return chat.title;
  const clean = first.content.trim().replace(/\s+/g, " ");
  return clean.length <= 42 ? clean : clean.slice(0, 41) + "\u2026";
}
