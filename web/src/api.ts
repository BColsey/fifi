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

export interface ExpertPlacement {
  mode: "resident" | "host_streamed" | "disk_streamed";
  vram_bytes?: number;
  ram_bytes?: number;
  disk_bytes?: number;
  slots_per_layer?: number;
}

export interface SystemEngine {
  id: string;
  model: string;
  state: EngineState;
  /** Jobs submitted but not yet dequeued; absent on older builds. */
  queued?: number;
  context_limit: number;
  /** The rows the engine started with, as `config effective` prints
   * them — not a re-resolution. Empty where a family composes no
   * per-key origins. */
  settings: EffectiveSetting[];
  /** Startup capacities from the loaded backend, not live occupancy. */
  expert_placement?: ExpertPlacement;
}

export interface SystemInfo {
  version: string;
  backends_compiled: string[];
  host: {
    cpu_threads: number;
    mem_total_bytes: number | { unavailable: string };
    mem_available_bytes: number | { unavailable: string };
    /** Cumulative /proc/stat jiffies; null off Linux. Diff two polls
     * for the CPU percentage. */
    cpu_times?: number[] | null;
    /** This process's block-level I/O; null off Linux. */
    proc_io?: { read_bytes: number; write_bytes: number } | null;
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
  /** Named session for sticky routing and monitor attribution. */
  "x-session"?: string;
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

async function getJson<T>(
  path: string, auth: boolean, signal?: AbortSignal, extraHeaders?: Record<string, string>
): Promise<T> {
  let res: Response;
  try {
    const headers = extraHeaders === undefined
      ? (auth ? authHeaders() : undefined)
      : { ...(auth ? authHeaders() : {}), ...extraHeaders };
    res = await fetch(path, { signal, headers });
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

/** One server event line (see /v1/logs): engine lifecycle, failures,
 * and download lifecycle. Excludes raw errors and request text. */
export interface LogEvent {
  ts_ms: number;
  line: string;
}

export function listLogs(signal?: AbortSignal): Promise<{ events: LogEvent[] }> {
  return getJson<{ events: LogEvent[] }>("/v1/logs", true, signal);
}

/** One row of the monitor's recent-request ring: the log line reduced
 * to the server's allowlist (see reqlog RING_KEYS), then to the fields
 * this panel reads. The request/metadata keys (request_id, method,
 * endpoint) stay off the wire type. */
export interface RecentRequest {
  model: string;
  /** The engine that took the job; null on rows from older builds. */
  engine?: string | null;
  /** The named session the request carried; null when it carried none. */
  session?: string | null;
  /** Wall-clock start, Unix milliseconds; absent on older builds. */
  started_at_ms?: number;
  stream: boolean;
  status: number | null;
  outcome: "ok" | "error" | "cancelled";
  error_code: string | null;
  /** The wire row carries it (see reqlog RING_KEYS); older builds may not. */
  finish_reason?: string | null;
  prompt_tokens: number;
  completion_tokens: number;
  reused_tokens: number;
  /** Submit to engine start; null when the job never started. */
  queue_wait_ms?: number | null;
  /** Engine start to first batch; null on the same condition. */
  prefill_ms?: number | null;
  time_to_first_token_ms: number | null;
  decode_tokens_per_second: number | null;
  /** Expert-bank admit activity of the prefill (see reqlog), when the
   * executor has a bank: admit passes, experts moved in, bytes queued. */
  prefill_bank_passes?: number | null;
  prefill_bank_misses?: number | null;
  prefill_bank_upload_bytes?: number | null;
}

/** One in-flight request of the active board (see reqlog ACTIVE_KEYS):
 * counts, names, and clocks only — never prompt text. The row lives
 * from admission to the written log line. */
export interface ActiveRequest {
  request_id: string;
  model: string;
  /** The engine that took the job; null before routing. */
  engine?: string | null;
  /** The named chat session the request carried; null when none. */
  session?: string | null;
  /** queued until the engine thread takes the job, prefill until its
   * first batch, decode after. */
  phase: "queued" | "prefill" | "decode";
  prompt_tokens: number;
  /** Tokens the decode batches committed so far. */
  completion_tokens: number;
  /** Epoch milliseconds; the panel derives elapsed time from it. */
  started_at_ms: number;
}

export interface RecentRequestBody {
  requests: RecentRequest[];
  /** In-flight rows; older builds omit the array. */
  active?: ActiveRequest[];
}

/** The server extension behind the GUI's live speed panel. */
export function listRecentRequests(signal?: AbortSignal): Promise<RecentRequestBody> {
  return getJson<RecentRequestBody>("/v1/recent-requests", true, signal);
}

export interface GpuInfo {
  index: number | null;
  name: string | null;
  util_percent: number | null;
  mem_used_bytes: number | null;
  mem_total_bytes: number | null;
  temp_c: number | null;
  power_w: number | null;
}

export interface TelemetryInfo {
  gpus: GpuInfo[];
  source: string;
  error: string | null;
}

/** Per-GPU telemetry (subprocess probe on the server, 2 s cache). */
export function getTelemetry(signal?: AbortSignal): Promise<TelemetryInfo> {
  return getJson<TelemetryInfo>("/v1/telemetry", true, signal);
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

export function getSamplerTrace(
  replyId: string, signal?: AbortSignal, model?: string
): Promise<SamplerTracePacket> {
  return getJson<SamplerTracePacket>(
    "/v1/sampler-trace?id=" + encodeURIComponent(replyId),
    true,
    signal,
    model === undefined ? undefined : { "X-Fifi-Trace-Model": model }
  );
}

/** One MoE layer of a live routing step: the picked expert ids in choice
 * order, their renormalized gates, and the shared-expert gate. */
export interface RouteTraceLayer {
  ids: number[];
  weights: number[];
  gate: number;
}

/** Packet v1 of GET /v1/route-trace — a server extension, not in
 * api/openapi.yaml. Steps are decode steps, oldest first; the shape fields
 * are the engine's build-time facts (null when no stage ever ran). */
export interface RouteTracePacket {
  packet: 1;
  ring_capacity: number;
  layers: number | null;
  top_k: number | null;
  experts: number | null;
  steps: { step: number; layers: RouteTraceLayer[] }[];
}

export function getRouteTrace(signal?: AbortSignal): Promise<RouteTracePacket> {
  return getJson<RouteTracePacket>("/v1/route-trace", true, signal);
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

/** Shared body for the mutating routes of the models view. */
async function sendJson<T>(
  method: string, path: string, body?: unknown, signal?: AbortSignal
): Promise<T> {
  let res: Response;
  try {
    res = await fetch(path, {
      method,
      signal,
      headers: { ...authHeaders(), "Content-Type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
  } catch (err) {
    if (err instanceof Error && err.name === "AbortError") throw err;
    throw new ApiError("Server unreachable.", "network", 0);
  }
  if (!res.ok) throw await toApiError(res);
  return (await res.json()) as T;
}

export interface EngineSummary {
  id: string;
  model: string;
  model_dir: string;
  state: EngineState;
  load_stage?: string;
  /** Present only after a failure: the loader's own refusal. */
  error?: EngineErrorInfo;
  context_limit: number;
  runtime_added: boolean;
  /** Stopping this engine returns no memory until the server restarts. */
  memory_held_until_restart: boolean;
}

export interface EngineList {
  engines: EngineSummary[];
}

export interface AddEngineRequest {
  model_dir: string;
  id?: string;
  family?: string;
  context_limit?: number;
  kv_dtype?: string;
  tokenizer_dir?: string;
}

export interface HubSearchRow {
  id: string;
  likes: number;
  downloads: number;
  last_modified?: string | null;
  pipeline_tag?: string | null;
  gated: boolean;
}

export interface HubCacheRef {
  name: string;
  commit: string;
}

export interface ResourceEstimate {
  vram_bytes: number;
  ram_bytes: number;
  disk_bytes: number;
  /** True when the planner produced it from a snapshot on disk. */
  planned: boolean;
  /** What the figures count, in the server's words. */
  basis: string;
  mode: string;
}

export interface HubCacheSnapshot {
  commit: string;
  path: string;
  gguf: string[];
  /** No download is still filling it and every file resolves. */
  complete: boolean;
  /** Complete, and `serve --model` would start it as a known family. */
  servable: boolean;
  families: string[];
  estimate?: ResourceEstimate;
}

export interface HubCacheModel {
  repo: string;
  bytes: number;
  refs: HubCacheRef[];
  snapshots: HubCacheSnapshot[];
}

export interface HubCache {
  hub_dir: string;
  models: HubCacheModel[];
}

export type HubDownloadState = "downloading" | "done" | "failed" | "cancelled";

export interface HubDownloadJob {
  id: number;
  repo: string;
  revision: string;
  state: HubDownloadState;
  commit?: string;
  files_total?: number;
  files_done?: number;
  bytes?: number;
  bytes_total?: number;
  current_file?: string;
  /** The failure cause of a failed job; the path of a finished one. */
  detail?: string;
  estimate?: ResourceEstimate;
}

export function listEngines(signal?: AbortSignal): Promise<EngineList> {
  return getJson<EngineList>("/v1/engines", true, signal);
}

export function searchHubModels(
  q: string,
  opts: { gguf?: boolean; exl3?: boolean; limit?: number },
  signal?: AbortSignal
): Promise<{ results: HubSearchRow[] }> {
  const params = new URLSearchParams({ q });
  if (opts.gguf === true) params.set("gguf", "true");
  if (opts.exl3 === true) params.set("exl3", "true");
  if (opts.limit !== undefined) params.set("limit", String(opts.limit));
  return getJson<{ results: HubSearchRow[] }>(`/v1/hf/search?${params}`, true, signal);
}

export interface HubTree {
  revision: string;
  commit: string;
  total_bytes: number;
  estimate: ResourceEstimate;
  files: { path: string; size: number }[];
}

export function getHubTree(repo: string, signal?: AbortSignal): Promise<HubTree> {
  return getJson<HubTree>(`/v1/hf/tree?${new URLSearchParams({ repo })}`, true, signal);
}

export function listHubCache(signal?: AbortSignal): Promise<HubCache> {
  return getJson<HubCache>("/v1/hf/cache", true, signal);
}

export function listHubDownloads(
  signal?: AbortSignal
): Promise<{ downloads: HubDownloadJob[] }> {
  return getJson<{ downloads: HubDownloadJob[] }>("/v1/hf/downloads", true, signal);
}

export function startHubDownload(
  repo: string, revision?: string
): Promise<{ id: number; repo: string; revision: string }> {
  return sendJson("POST", "/v1/hf/downloads",
    revision === undefined ? { repo } : { repo, revision });
}

export function cancelHubDownload(
  id: number
): Promise<{ id: number; cancelling: boolean }> {
  return sendJson("DELETE", `/v1/hf/downloads/${id}`);
}

export function addEngine(
  body: AddEngineRequest
): Promise<{ id: string; model: string; model_dir: string; state: string }> {
  return sendJson("POST", "/v1/engines", body);
}

export function removeEngine(
  id: string
): Promise<{
  id: string;
  model: string;
  state: string;
  shutdown_failure?: string;
  /** False when the family keeps the engine's memory until the server restarts. */
  memory_released: boolean;
  memory_held_until_restart?: boolean;
  note?: string;
}> {
  return sendJson("DELETE", `/v1/engines/${encodeURIComponent(id)}`);
}
