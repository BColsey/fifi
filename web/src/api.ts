/**
 * This module owns every fetch; views never build URLs.
 * Contract: api/openapi.yaml.
 */

import { SseParser } from "./sse.js";
import { loadSettings } from "./store.js";

export type EngineState = "loading" | "ready" | "busy" | "error" | "stopped" | "native-failed";

export interface CacheStats {
  kv_bytes: number;
  prefix_hits: number;
  prefix_misses: number;
  snapshot_bytes?: number;
}

export interface EngineErrorInfo {
  code: string;
  stage: string;
  message: string;
  retry: "after_correction" | "unknown" | string;
  suggestion?: string;
}

export interface Engine {
  id: string;
  model: string;
  state: EngineState;
  /** Present only while `state` is `loading`: the loader's own name
   * for the phase it is in. Absent once the engine is ready, failed,
   * or stopped — never a percentage, never invented. */
  load_stage?: string;
  cache?: CacheStats;
  error?: EngineErrorInfo;
}

export interface EffectiveSetting {
  key: string;
  value: string;
  origin: string;
}

export interface SystemEngine {
  id: string;
  model: string;
  state: EngineState;
  context_limit: number;
  /** The rows the engine started with, as `config effective` prints
   * them — not a re-resolution. Empty where a family composes no
   * per-key origins. */
  settings: EffectiveSetting[];
}

export interface SystemInfo {
  version: string;
  backends_compiled: string[];
  host: {
    cpu_threads: number;
    mem_total_bytes: number | { unavailable: string };
    mem_available_bytes: number | { unavailable: string };
  };
  engines: SystemEngine[];
}

export interface Model {
  id: string;
  object: "model";
  created: number;
  owned_by: string;
  engines?: Engine[];
}

export interface ModelList {
  object: "list";
  data: Model[];
}

export type Role = "system" | "user" | "assistant" | "tool";

export interface ToolCall {
  id: string;
  type: "function";
  function: { name: string; arguments: string };
}

export interface ToolCallDelta {
  index: number;
  id?: string;
  type?: "function";
  function?: { name?: string; arguments?: string };
}

export interface ChatMessage {
  role: Role;
  content: string | null;
  tool_call_id?: string;
  tool_calls?: ToolCall[];
}

export type ThinkingMode = "default" | "on" | "off";
export type FinishReason = "stop" | "length" | "tool_calls";

export interface ChatRequestBody {
  model?: string;
  messages: ChatMessage[];
  temperature?: number;
  top_p?: number;
  top_k?: number;
  max_tokens?: number;
  "x-thinking"?: ThinkingMode;
  /** Arms sampler-trace capture for this one reply. The server refuses
   * the request with 400 unless its sample_trace setting permits it. */
  "x-sample-trace"?: boolean;
  stop?: string[];
}

export interface Usage {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  reused_tokens?: number;
}

export interface ResponseMessage {
  role: "assistant";
  content: string | null;
  reasoning_content?: string;
  tool_calls?: ToolCall[];
}

export interface Choice {
  index: number;
  message: ResponseMessage;
  finish_reason: FinishReason;
}

export interface ChatCompletion {
  id: string;
  object: "chat.completion";
  created: number;
  model: string;
  engine?: string;
  session?: string;
  choices: Choice[];
  usage: Usage;
}

export interface ChoiceDelta {
  role?: "assistant";
  content?: string;
  reasoning_content?: string;
  tool_calls?: ToolCallDelta[];
}

export interface ChunkChoice {
  index: number;
  delta: ChoiceDelta;
  finish_reason?: FinishReason;
}

export interface ChatCompletionChunk {
  id: string;
  object: "chat.completion.chunk";
  created: number;
  model: string;
  engine?: string;
  session?: string;
  choices: ChunkChoice[];
  usage?: Usage;
}

export type ErrorCode =
  | "bad_request"
  | "unauthorized"
  | "not_found"
  | "memory_plan"
  | "context_limit"
  | "checkpoint_failure"
  | "overloaded"
  | "not_ready"
  | "backend_error";

/** "network" is a client-side class with no contract code. */
export class ApiError extends Error {
  constructor(
    message: string,
    readonly code: ErrorCode | "network",
    readonly status: number
  ) {
    super(message);
    this.name = "ApiError";
  }
}

/** Plain-language mapping from the brief. The code stays visible for reports. */
export function describeError(err: ApiError): string {
  switch (err.code) {
    case "bad_request":
      return "The request failed a check: " + err.message + " (bad_request)";
    case "unauthorized":
      return "The server rejected the API key. (unauthorized)";
    case "not_found":
      return "The model or route does not exist on this server. (not_found)";
    case "memory_plan":
      return "The memory plan cannot fit this request. (memory_plan)";
    case "context_limit":
      return "The prompt exceeds the position limit of the engine. Shorten it and send again. (context_limit)";
    case "checkpoint_failure":
      return "A checkpoint failed identity or load. (checkpoint_failure)";
    case "overloaded":
      return "The job queue is full. Try again shortly. (overloaded)";
    case "not_ready":
      return "No engine is ready yet. (not_ready)";
    case "backend_error":
      return "Model execution failed on the server. (backend_error)";
    case "network":
      return "The server is unreachable.";
  }
}

function authHeaders(): Record<string, string> {
  const key = loadSettings().apiKey;
  return key === "" ? {} : { Authorization: "Bearer " + key };
}

async function toApiError(res: Response): Promise<ApiError> {
  let message = "HTTP " + res.status;
  let code: ErrorCode | "network" = "backend_error";
  try {
    const body = (await res.json()) as { error?: { message?: string; code?: ErrorCode } };
    if (body.error !== undefined) {
      if (body.error.message !== undefined) message = body.error.message;
      if (body.error.code !== undefined) code = body.error.code;
    }
  } catch {
    // Non-JSON body keeps the HTTP status text.
  }
  if (res.status === 401 && code !== "unauthorized") code = "unauthorized";
  return new ApiError(message, code, res.status);
}

async function getJson<T>(path: string, auth: boolean, signal?: AbortSignal): Promise<T> {
  let res: Response;
  try {
    res = await fetch(path, { signal, headers: auth ? authHeaders() : undefined });
  } catch (err) {
    if (err instanceof Error && err.name === "AbortError") throw err;
    throw new ApiError("Server unreachable.", "network", 0);
  }
  if (!res.ok) throw await toApiError(res);
  return (await res.json()) as T;
}

export interface HealthBody {
  status: "ok";
}

export interface ReadyBody {
  status: "ready";
}

export function getHealth(signal?: AbortSignal): Promise<HealthBody> {
  return getJson<HealthBody>("/health", false, signal);
}

export function getReady(signal?: AbortSignal): Promise<ReadyBody> {
  return getJson<ReadyBody>("/ready", false, signal);
}

export function listModels(signal?: AbortSignal): Promise<ModelList> {
  return getJson<ModelList>("/v1/models", true, signal);
}

export function getSystem(signal?: AbortSignal): Promise<SystemInfo> {
  return getJson<SystemInfo>("/v1/system", true, signal);
}

/** One row of the monitor's recent-request ring: the log line reduced to
 * the server's allowlist, then to the fields this panel reads — the
 * request/metadata keys (request_id, method, endpoint), finish_reason,
 * and the log's permanent-null queue_wait_ms stay off the wire type.
 * reused_tokens is kept on the wire type for future use; the card does
 * not render it today. */
export interface RecentRequest {
  model: string;
  stream: boolean;
  status: number | null;
  outcome: "ok" | "error" | "cancelled";
  error_code: string | null;
  prompt_tokens: number;
  completion_tokens: number;
  reused_tokens: number;
  time_to_first_token_ms: number | null;
  decode_tokens_per_second: number | null;
}

export interface RecentRequestBody {
  requests: RecentRequest[];
}

/** The server extension behind the GUI's live speed panel. */
export function listRecentRequests(signal?: AbortSignal): Promise<RecentRequestBody> {
  return getJson<RecentRequestBody>("/v1/recent-requests", true, signal);
}

export interface SamplerCandidate {
  token_id: number;
  prob: number;
}

/** One sampled decode position, ids and probabilities only. `offset` is
 * the piece's byte offset into the raw generated text; null when the
 * replay never reached this position (early stop or alignment error). */
export interface SamplerPosition {
  token_id: number;
  rank: number;
  kept: number;
  greedy: boolean;
  /** Null on greedy rows: by contract they carry no distribution. */
  selected_prob: number | null;
  candidates: SamplerCandidate[];
  offset: number | null;
  piece: string | null;
}

/** Packet v2 of GET /v1/sampler-trace?id=<reply id> — api/openapi.yaml. */
export interface SamplerTracePacket {
  packet: 2;
  reply_id: string;
  model: string;
  sampling: { temperature: number; top_p: number; top_k: number };
  capture_scope: string;
  truncated: boolean;
  captured_positions: number;
  raw_byte_len: number;
  aligned_positions: number;
  /** True once the replay's token stream disagreed with the recorded
   * positions or a piece failed to decode; offsets below it stay valid. */
  alignment_error: boolean;
  trailer: string;
  positions: SamplerPosition[];
}

export function getSamplerTrace(replyId: string, signal?: AbortSignal): Promise<SamplerTracePacket> {
  return getJson<SamplerTracePacket>(
    "/v1/sampler-trace?id=" + encodeURIComponent(replyId),
    true,
    signal
  );
}

export async function chatCompletion(
  body: ChatRequestBody,
  signal?: AbortSignal
): Promise<ChatCompletion> {
  let res: Response;
  try {
    res = await fetch("/v1/chat/completions", {
      method: "POST",
      signal,
      headers: { ...authHeaders(), "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  } catch (err) {
    if (err instanceof Error && err.name === "AbortError") throw err;
    throw new ApiError("Server unreachable.", "network", 0);
  }
  if (!res.ok) throw await toApiError(res);
  return (await res.json()) as ChatCompletion;
}

export interface StreamCallbacks {
  /** The server admitted the job. Headers and the role chunk arrived. */
  onRole?: () => void;
  /** The reply id, once, from the first chunk that carries it. This is
   * the id that keys /v1/sampler-trace for the finished reply. */
  onReplyId?: (id: string) => void;
  onDelta?: (delta: { content?: string; reasoning?: string; toolCall?: ToolCallDelta }) => void;
  onFinish?: (reason: FinishReason, usage: Usage | undefined) => void;
}

export type StreamEnd = "done" | "closed";

export async function chatStream(
  body: ChatRequestBody,
  callbacks: StreamCallbacks,
  signal?: AbortSignal
): Promise<StreamEnd> {
  let res: Response;
  try {
    res = await fetch("/v1/chat/completions", {
      method: "POST",
      signal,
      headers: { ...authHeaders(), "Content-Type": "application/json" },
      body: JSON.stringify({ ...body, stream: true }),
    });
  } catch (err) {
    if (err instanceof Error && err.name === "AbortError") throw err;
    throw new ApiError("Server unreachable.", "network", 0);
  }
  if (!res.ok) throw await toApiError(res);
  if (res.body === null) throw new ApiError("Empty stream body.", "backend_error", res.status);

  let done = false;
  let streamError: ApiError | null = null;
  let replyId: string | null = null;

  const parser = new SseParser((ev) => {
    if (ev.event === "error") {
      let message = "The stream failed.";
      let code: ErrorCode = "backend_error";
      try {
        const parsed = JSON.parse(ev.data) as { error?: { message?: string; code?: ErrorCode } };
        if (parsed.error !== undefined) {
          if (parsed.error.message !== undefined) message = parsed.error.message;
          if (parsed.error.code !== undefined) code = parsed.error.code;
        }
      } catch {
        // Malformed error data keeps the default text.
      }
      streamError = new ApiError(message, code, res.status);
      return;
    }
    if (ev.data === "[DONE]") {
      done = true;
      return;
    }
    let chunk: ChatCompletionChunk;
    try {
      chunk = JSON.parse(ev.data) as ChatCompletionChunk;
    } catch {
      return;
    }
    if (replyId === null && typeof chunk.id === "string" && chunk.id !== "") {
      replyId = chunk.id;
      callbacks.onReplyId?.(chunk.id);
    }
    for (const choice of chunk.choices ?? []) {
      if (choice.delta.role !== undefined) callbacks.onRole?.();
      if (choice.delta.content !== undefined || choice.delta.reasoning_content !== undefined) {
        callbacks.onDelta?.({
          content: choice.delta.content,
          reasoning: choice.delta.reasoning_content,
        });
      }
      for (const frag of choice.delta.tool_calls ?? []) {
        callbacks.onDelta?.({ toolCall: frag });
      }
      if (choice.finish_reason !== undefined) {
        callbacks.onFinish?.(choice.finish_reason, chunk.usage);
      }
    }
  });

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  try {
    for (;;) {
      const { value, done: readable } = await reader.read();
      if (readable) break;
      parser.push(decoder.decode(value, { stream: true }));
    }
    parser.push(decoder.decode());
    parser.close();
  } finally {
    reader.releaseLock();
  }

  if (streamError !== null) throw streamError;
  return done ? "done" : "closed";
}
